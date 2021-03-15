import dataclasses
from functools import lru_cache
from libs import pipeline

from covidactnow.datapublic.common_fields import CommonFields
import pandas as pd
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets import data_source
from libs.datasets.timeseries import MultiRegionDataset


# Early data is noisy due to lack of reporting, etc.
DEFAULT_START_DATE = "2020-09-01"
CUSTOM_START_DATES = {
    "02": "2020-10-06",  # Alaska
    "04": "2020-09-02",  # Arizona
    "15": "2020-10-10",  # Hawaii
    "16": "2020-10-17",  # Idaho
    "19": "2020-09-05",  # Iowa
    "21": "2020-10-15",  # Kentucky
    "28": "2020-11-11",  # Mississippi
    "34": "2020-09-01",  # New Jersey
    "38": "2020-11-03",  # North Dakota
    "46": "2020-11-02",  # South Dakota
    "47": "2020-09-20",  # Tennessee
    "53": "2020-10-25",  # Washington
}

FIELD_MAPPING = {
    "adult_icu_beds_capacity": CommonFields.ICU_BEDS,
    "adult_icu_beds_in_use": CommonFields.CURRENT_ICU_TOTAL,
    "adult_icu_beds_in_use_covid": CommonFields.CURRENT_ICU,
    "hospital_beds_capacity": CommonFields.STAFFED_BEDS,
    "hospital_beds_in_use": CommonFields.HOSPITAL_BEDS_IN_USE_ANY,
    "hospital_beds_in_use_covid": CommonFields.CURRENT_HOSPITALIZED,
}


def make_variable(can_scraper_field, common_field, measurement):
    return ccd_helpers.ScraperVariable(
        variable_name=can_scraper_field,
        measurement=measurement,
        provider="hhs",
        unit="beds",
        common_field=common_field,
    )


class HHSHospitalStateDataset(data_source.CanScraperBase):
    SOURCE_TYPE = "HHSHospitalState"

    VARIABLES = [
        make_variable(can_field, common_field, "current")
        for can_field, common_field in FIELD_MAPPING.items()
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        return modify_dataset(super().make_dataset())


class HHSHospitalCountyDataset(data_source.CanScraperBase):
    SOURCE_TYPE = "HHSHospitalCounty"

    VARIABLES = [
        make_variable(can_field, common_field, "rolling_average_7_day")
        for can_field, common_field in FIELD_MAPPING.items()
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        return modify_dataset(super().make_dataset())


def modify_dataset(ds: MultiRegionDataset) -> MultiRegionDataset:
    ts_copy = ds.timeseries.copy()

    return dataclasses.replace(ds, timeseries=filter_early_data(ts_copy), timeseries_bucketed=None)


def filter_early_data(df):
    keep_rows = df.index.get_level_values(CommonFields.DATE.value) >= pd.to_datetime(
        DEFAULT_START_DATE
    )
    df = df.loc[keep_rows]

    for (fips, start_date) in CUSTOM_START_DATES.items():
        location_id = pipeline.location_id_to_fips(fips)
        keep_rows = (df.index.get_level_values(CommonFields.LOCATION_ID.value) != location_id) | (
            df.index.get_level_values(CommonFields.DATE.value) >= pd.to_datetime(start_date)
        )
        df = df.loc[keep_rows]

    return df
