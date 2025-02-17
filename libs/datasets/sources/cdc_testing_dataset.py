import dataclasses
from functools import lru_cache

from libs.datasets import AggregationLevel
from libs.datasets import data_source
import pandas as pd
import abc
from datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.timeseries import MultiRegionDataset
from libs.pipeline import Region

DC_COUNTY_LOCATION_ID = Region.from_fips("11001").location_id
DC_STATE_LOCATION_ID = Region.from_state("DC").location_id


def _remove_trailing_zeros(series: pd.Series) -> pd.Series:

    series = pd.Series(series.values.copy(), index=series.index.get_level_values(CommonFields.DATE))

    index = series.loc[series != 0].last_valid_index()

    if index is None:
        # If test positivity is 0% the entire time, considering the data inaccurate, returning
        # none.
        series[:] = None
        return series

    series[index + pd.DateOffset(1) :] = None
    return series


def remove_trailing_zeros(data: pd.DataFrame) -> pd.DataFrame:
    # TODO(tom): See if TailFilter+zeros_filter produce the same data and if so, remove this
    #  function.
    data = data.sort_index()
    test_pos = data.groupby(CommonFields.LOCATION_ID)[CommonFields.TEST_POSITIVITY_7D].apply(
        _remove_trailing_zeros
    )
    data[CommonFields.TEST_POSITIVITY_7D] = test_pos
    return data


class CDCTestingBase(data_source.CanScraperBase, abc.ABC):
    """Default CanScraperBase class with make_dataset() overridden to handle CDC test data."""

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        return modify_dataset(super().make_dataset())


class CDCHistoricalTestingDataset(CDCTestingBase):
    """Data source connecting to the official CDC historical test positivity dataset."""

    SOURCE_TYPE = "CDCHistoricalTesting"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_positive",
            measurement="rolling_average_7_day",
            provider="cdc2",
            unit="percentage",
            common_field=CommonFields.TEST_POSITIVITY_7D,
        ),
    ]


class CDCOriginallyPostedTestingDataset(CDCTestingBase):
    """Data source connecting to the official as-originally-posted CDC test positivity dataset."""

    SOURCE_TYPE = "CDCOriginallyPostedTesting"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_positive",
            measurement="rolling_average_7_day",
            provider="cdc_originally_posted",
            unit="percentage",
            common_field=CommonFields.TEST_POSITIVITY_7D,
        ),
    ]


class CDCCombinedTestingDataset(data_source.DataSource):
    """Data source combining the CDC's historical and as-originally-posted datasets."""

    # This gets overwritten by the tags generated in CDCOriginallyPostedTestingDataset.
    # We use the as-originally-posted source for the tags as this is the dataset with the
    # most up-to-date data points
    SOURCE_TYPE = "CDCTesting"
    EXPECTED_FIELDS = [CommonFields.TEST_POSITIVITY_7D]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        """Use historical data when possible, use as-originally-posted for data that the historical dataset does not have.
        
        The historical data is usually delayed behind the as-originally-posted by ~3 days. 
        To get the combination of the most up-to-date and accurate data we use the historical dataset
        by default, but use as-originally-posted for the most recent days that the historical dataset
        does not yet have.
        """

        historical_ds = CDCHistoricalTestingDataset().make_dataset()
        as_posted_ds = CDCOriginallyPostedTestingDataset().make_dataset()

        historical_ts = historical_ds.timeseries
        as_posted_ts = as_posted_ds.timeseries

        merged_ts = historical_ts.combine_first(as_posted_ts)

        merged_ds = dataclasses.replace(
            as_posted_ds, timeseries=merged_ts, timeseries_bucketed=None,
        )
        return merged_ds


def modify_dataset(ds: MultiRegionDataset) -> MultiRegionDataset:
    ts_copy = ds.timeseries.copy()
    # Test positivity should be a ratio
    ts_copy.loc[:, CommonFields.TEST_POSITIVITY_7D] = (
        ts_copy.loc[:, CommonFields.TEST_POSITIVITY_7D] / 100.0
    )

    levels = set(
        Region.from_location_id(l).level
        for l in ds.timeseries.index.get_level_values(CommonFields.LOCATION_ID)
    )
    # Should only be picking up county all_df for now.  May need additional logic if states
    # are included as well
    assert levels == {AggregationLevel.COUNTY}

    # Duplicating DC County results as state results because of a downstream
    # use of how dc state data is used to override DC county data.
    dc_results = ts_copy.xs(
        DC_COUNTY_LOCATION_ID, axis=0, level=CommonFields.LOCATION_ID, drop_level=False
    )
    dc_results = dc_results.rename(
        index={DC_COUNTY_LOCATION_ID: DC_STATE_LOCATION_ID}, level=CommonFields.LOCATION_ID
    )

    ts_copy = ts_copy.append(dc_results, verify_integrity=True).sort_index()

    return dataclasses.replace(
        ds, timeseries=remove_trailing_zeros(ts_copy), timeseries_bucketed=None
    )
