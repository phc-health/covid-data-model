import json
import datetime
import structlog
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, List, Optional, Mapping, Tuple, Dict

try:  # To work with python 3.7 and 3.9 without changes.
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property

from libs import pipeline
from libs.parallel_utils import parallel_map
from libs.datasets.sources import zeros_filter
from libs.us_state_abbrev import US_STATE_ABBREV, ABBREV_US_UNKNOWN_COUNTY_FIPS
from libs.datasets.timeseries import MultiRegionDataset
from datapublic.common_fields import CommonFields, PdFields, FieldName
from libs.datasets.dataset_utils import (
    REGION_OVERRIDES_JSON,
    CUMULATIVE_FIELDS_TO_FILTER,
    DEFAULT_REPORTING_RATIO,
)
from libs.datasets import (
    custom_patches,
    weekly_hospitalizations,
    nytimes_anomalies,
    custom_aggregations,
    statistical_areas,
    timeseries,
    outlier_detection,
    dataset_utils,
    combined_datasets,
    new_cases_and_deaths,
    vaccine_backfills,
    tail_filter,
    custom_patches,
    weekly_hospitalizations,
    manual_filter,
    dataset_utils,
    timeseries,
    combined_datasets,
)

TailFilter = tail_filter.TailFilter
_log = structlog.get_logger()

# Ideally this should probably go in dataset_utils, but that causes
# a circular import with libs.pipeline, so leaving here for now.
KNOWN_LOCATION_ID_WITHOUT_POPULATION = [
    # Territories other than PR
    "iso1:us#iso2:us-vi",
    "iso1:us#iso2:us-as",
    "iso1:us#iso2:us-gu",
    # Subregion of AS
    "iso1:us#iso2:us-vi#fips:78030",
    "iso1:us#iso2:us-vi#fips:78020",
    "iso1:us#iso2:us-vi#fips:78010",
    # Retired FIPS
    "iso1:us#iso2:us-sd#fips:46113",
    "iso1:us#iso2:us-va#fips:51515",
    # All the unknown county FIPS
    *[pipeline.fips_to_location_id(f) for f in ABBREV_US_UNKNOWN_COUNTY_FIPS.values()],
]


@dataclass(frozen=True)
class OneStateDataset:
    """Wrapper class containing a MultiRegionDataset with data for a single state and its subregions. 
    
    As consistent with the Region class, territories are considered states (e.g. GU, PR)
    """

    multiregion_dataset: MultiRegionDataset
    state: pipeline.Region

    @classmethod
    def from_mrds(cls, dataset: MultiRegionDataset) -> "OneStateDataset":
        states = [pipeline.Region.from_location_id(loc).state for loc in dataset.location_ids]
        if len(set(states)) != 1:
            raise ValueError(
                "Can only create a OneStateDataset from datasets with data "
                f"for a single state. Data for {len(set(states))} found."
            )
        return OneStateDataset(
            multiregion_dataset=dataset, state=pipeline.Region.from_state(states[0])
        )


@dataclass(frozen=True)
class DatasetUpdater:
    """Class to orchestrate dataset updates through state-focused parallelization."""

    state_datasets: Iterable[OneStateDataset]
    cbsa_aggregator: statistical_areas.CountyToCBSAAggregator
    region_overrides: Dict
    print_stats: bool

    @cached_property
    def region_overrides_config(self) -> manual_filter.Config:
        return manual_filter.transform_region_overrides(
            self.region_overrides, self.cbsa_aggregator.cbsa_to_counties_region_map
        )

    @classmethod
    def from_bulk_mrds(
        cls,
        states: Optional[List[str]] = None,
        refresh_datasets: Optional[bool] = True,
        print_stats: bool = True,
    ) -> "DatasetUpdater":
        bulk_dataset = load_bulk_dataset(refresh_datasets=refresh_datasets)
        if not states:
            states = list(US_STATE_ABBREV.values())
        state_datasets = [
            OneStateDataset.from_mrds(bulk_dataset.get_subset(state=region)) for region in states
        ]

        cbsa_aggregator = statistical_areas.CountyToCBSAAggregator.from_local_public_data()
        region_overrides = json.load(open(REGION_OVERRIDES_JSON))
        return DatasetUpdater(
            state_datasets=state_datasets,
            region_overrides=region_overrides,
            cbsa_aggregator=cbsa_aggregator,
            print_stats=print_stats,
        )

    def build_and_combine_regions(
        self, aggregate_to_country: bool = True, generate_cbsas: bool = True
    ) -> Tuple[MultiRegionDataset, MultiRegionDataset]:
        """Run pipeline on self.state_datasets, combine and return outputs and manually removed data"""
        processed_states, manual_filter_datasets = zip(
            *parallel_map(self._build_state_dataset, self.state_datasets)
        )
        _log.info("Finished processing individual regions...")

        _log.info("Combining individual regions...")
        manual_filter_touched = _combine_datasets(manual_filter_datasets)
        multiregion_dataset = self._combine_state_datasets(
            list(processed_states),
            aggregate_to_country=aggregate_to_country,
            generate_cbsas=generate_cbsas,
        )
        return multiregion_dataset, manual_filter_touched

    def _build_state_dataset(self, state_dataset: OneStateDataset) -> Tuple[MultiRegionDataset]:
        """Runs pipeline on state_dataset returning the output and the manual filtered dataset"""
        state = state_dataset.state
        multiregion_dataset = state_dataset.multiregion_dataset

        before_manual_filter = multiregion_dataset
        multiregion_dataset = manual_filter.run(multiregion_dataset, self.region_overrides_config)
        manual_filter_touched = manual_filter.touched_subset(
            before_manual_filter, multiregion_dataset
        )
        multiregion_dataset = timeseries.drop_observations(
            multiregion_dataset, after=datetime.datetime.utcnow().date()
        )
        multiregion_dataset = outlier_detection.drop_tail_positivity_outliers(multiregion_dataset)
        # Filter for stalled cumulative values before deriving NEW_CASES from CASES.
        _, multiregion_dataset = TailFilter.run(multiregion_dataset, CUMULATIVE_FIELDS_TO_FILTER)
        multiregion_dataset = zeros_filter.drop_all_zero_timeseries(
            multiregion_dataset,
            [
                CommonFields.VACCINES_DISTRIBUTED,
                CommonFields.VACCINES_ADMINISTERED,
                CommonFields.VACCINATIONS_COMPLETED,
                CommonFields.VACCINATIONS_INITIATED,
                CommonFields.VACCINATIONS_ADDITIONAL_DOSE,
            ],
        )
        multiregion_dataset = vaccine_backfills.estimate_initiated_from_state_ratio(
            multiregion_dataset
        )
        multiregion_dataset = new_cases_and_deaths.add_new_cases(multiregion_dataset)
        multiregion_dataset = new_cases_and_deaths.add_new_deaths(multiregion_dataset)
        multiregion_dataset = weekly_hospitalizations.add_weekly_hospitalizations(
            multiregion_dataset
        )
        if state.state == "MD":
            multiregion_dataset = custom_patches.patch_maryland_missing_case_data(
                multiregion_dataset
            )
        multiregion_dataset = nytimes_anomalies.filter_by_nyt_anomalies(multiregion_dataset)
        multiregion_dataset = outlier_detection.drop_new_case_outliers(multiregion_dataset)
        multiregion_dataset = outlier_detection.drop_new_deaths_outliers(multiregion_dataset)
        multiregion_dataset = timeseries.drop_regions_without_population(
            multiregion_dataset, KNOWN_LOCATION_ID_WITHOUT_POPULATION, _log
        )
        if state.state == "PR":
            multiregion_dataset = custom_aggregations.aggregate_puerto_rico_from_counties(
                multiregion_dataset
            )
        if state.state == "NY":
            multiregion_dataset = custom_aggregations.aggregate_to_new_york_city(
                multiregion_dataset
            )
        if state.state == "DC":
            multiregion_dataset = custom_aggregations.replace_dc_county_with_state_data(
                multiregion_dataset
            )

        if self.print_stats:
            multiregion_dataset.print_stats(f"Finished {state.state}")
        return multiregion_dataset, manual_filter_touched

    def _combine_state_datasets(
        self,
        datasets: List[MultiRegionDataset],
        aggregate_to_country: bool = True,
        generate_cbsas: bool = True,
        generate_hsas: bool = True,
    ) -> MultiRegionDataset:
        multiregion_dataset = _combine_datasets(datasets)

        if generate_hsas:
            _log.info("Starting HSA aggregation...")
            hsa_aggregator = statistical_areas.CountyToHSAAggregator.from_local_data()
            multiregion_dataset = hsa_aggregator.aggregate(multiregion_dataset)

        if generate_cbsas:
            _log.info("Starting CBSA aggregation...")
            cbsa_dataset = self.cbsa_aggregator.aggregate(
                multiregion_dataset, reporting_ratio_required_to_aggregate=DEFAULT_REPORTING_RATIO
            )
            multiregion_dataset = multiregion_dataset.append_regions(cbsa_dataset)
            multiregion_dataset.print_stats("CBSA dataset")

        if aggregate_to_country:
            _log.info("Starting country aggregation...")
            multiregion_dataset = custom_aggregations.aggregate_to_country(
                multiregion_dataset, reporting_ratio_required_to_aggregate=DEFAULT_REPORTING_RATIO
            )
            multiregion_dataset.print_stats("Aggregate to country")
        return multiregion_dataset


def _combine_datasets(datasets: List[MultiRegionDataset]):
    regions = [region for dataset in datasets for region in dataset.location_ids]
    assert len(regions) == len(set(regions)), "Can't combine datasets with duplicate locations"

    _log.info("Starting region combination...")
    timeseries_df = (
        pd.concat([dataset.timeseries_bucketed for dataset in datasets])
        .sort_index()
        .rename_axis(columns=PdFields.VARIABLE)
    )
    static_df = (
        pd.concat(dataset.static for dataset in datasets)
        .sort_index()
        .rename_axis(columns=PdFields.VARIABLE)
    )
    tag = pd.concat([dataset.tag for dataset in datasets]).sort_index()
    return MultiRegionDataset(timeseries_bucketed=timeseries_df, static=static_df, tag=tag)


def load_bulk_dataset(refresh_datasets: Optional[bool] = True) -> MultiRegionDataset:
    if refresh_datasets:
        timeseries_field_datasets = load_datasets_by_field(
            combined_datasets.ALL_TIMESERIES_FEATURE_DEFINITION
        )
        static_field_datasets = load_datasets_by_field(
            combined_datasets.ALL_FIELDS_FEATURE_DEFINITION
        )

        multiregion_dataset = timeseries.combined_datasets(
            timeseries_field_datasets, static_field_datasets
        )
        multiregion_dataset.to_compressed_pickle(dataset_utils.COMBINED_RAW_PICKLE_GZ_PATH)
    else:
        multiregion_dataset = timeseries.MultiRegionDataset.from_compressed_pickle(
            dataset_utils.COMBINED_RAW_PICKLE_GZ_PATH
        )
    return multiregion_dataset


def load_datasets_by_field(
    feature_definition_config: combined_datasets.FeatureDataSourceMap, *, state=False, fips=False
) -> Mapping[FieldName, List[timeseries.MultiRegionDataset]]:
    def _load_dataset(data_source_cls) -> timeseries.MultiRegionDataset:
        try:
            dataset = data_source_cls.make_dataset()
            if state or fips:
                dataset = dataset.get_subset(state=state, fips=fips)
            return dataset
        except Exception:
            raise ValueError(f"Problem with {data_source_cls}")

    feature_definition = {
        # Put the highest priority first, as expected by timeseries.combined_datasets.
        # TODO(tom): reverse the hard-coded FeatureDataSourceMap and remove the reversed call.
        field_name: list(reversed(list(_load_dataset(cls) for cls in classes)))
        for field_name, classes in feature_definition_config.items()
        if classes
    }
    return feature_definition
