from datetime import datetime
from typing import Optional

from api.can_api_v2_definition import (
    Actuals,
    ActualsTimeseriesRow,
    AggregateFlattenedTimeseries,
    AggregateRegionSummary,
    Metrics,
    PredictionTimeseriesRow,
    PredictionTimeseriesRowWithHeader,
    Projections,
    RegionSummary,
    RegionSummaryWithTimeseries,
    ResourceUsageProjection,
)
from covidactnow.datapublic.common_fields import CommonFields
from libs import us_state_abbrev
from libs.datasets import can_model_output_schema as can_schema
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.datasets.timeseries import TimeseriesDataset
from libs.enums import Intervention
from libs.functions import get_can_projection


def _generate_api_for_projections(model_output: CANPyseirLocationOutput):
    _hospital_beds = ResourceUsageProjection(
        peakDate=model_output.peak_hospitalizations_date,
        shortageStartDate=model_output.hospitals_shortfall_date,
        peakShortfall=model_output.peak_hospitalizations_shortfall,
    )
    projections = Projections(
        totalHospitalBeds=_hospital_beds,
        ICUBeds=None,
        Rt=model_output.latest_rt,
        RtCI90=model_output.latest_rt_ci90,
    )
    return projections


def _generate_actuals(actual_data: dict) -> Actuals:
    """Generate actuals entry.

    Args:
        actual_data: Dictionary of data, generally derived one of the combined datasets.
        intervention: Current state level intervention.

    """
    return Actuals(
        population=actual_data.get(CommonFields.POPULATION),
        cumulativeConfirmedCases=actual_data[CommonFields.CASES],
        cumulativeDeaths=actual_data[CommonFields.DEATHS],
        cumulativePositiveTests=actual_data.get(CommonFields.POSITIVE_TESTS),
        cumulativeNegativeTests=actual_data.get(CommonFields.NEGATIVE_TESTS),
        hospitalBeds={
            "Capacity": actual_data.get(CommonFields.MAX_BED_COUNT),
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_HOSPITALIZED),
            "currentUsageTotal": actual_data.get(CommonFields.CURRENT_HOSPITALIZED_TOTAL),
            "typicalUsageRate": actual_data.get(CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE),
        },
        ICUBeds={
            "capacity": actual_data.get(CommonFields.ICU_BEDS),
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_ICU),
            "currentUsageTotal": actual_data.get(CommonFields.CURRENT_ICU_TOTAL),
            "typicalUsageRate": actual_data.get(CommonFields.ICU_TYPICAL_OCCUPANCY_RATE),
        },
        contactTracers=actual_data.get(CommonFields.CONTACT_TRACERS_COUNT),
    )


def _generate_prediction_timeseries_row(json_data_row) -> PredictionTimeseriesRow:

    return PredictionTimeseriesRow(
        date=json_data_row[can_schema.DATE].to_pydatetime(),
        hospitalBedsRequired=json_data_row[can_schema.ALL_HOSPITALIZED],
        hospitalBedCapacity=json_data_row[can_schema.BEDS],
        ICUBedsInUse=json_data_row[can_schema.INFECTED_C],
        ICUBedCapacity=json_data_row[can_schema.ICU_BED_CAPACITY],
        ventilatorsInUse=json_data_row[can_schema.CURRENT_VENTILATED],
        ventilatorCapacity=json_data_row[can_schema.VENTILATOR_CAPACITY],
        RtIndicator=json_data_row[can_schema.RT_INDICATOR],
        RtIndicatorCI90=json_data_row[can_schema.RT_INDICATOR_CI90],
        currentInfected=json_data_row[can_schema.ALL_INFECTED],
        currentSusceptible=json_data_row[can_schema.TOTAL_SUSCEPTIBLE],
        currentExposed=json_data_row[can_schema.EXPOSED],
        cumulativeDeaths=json_data_row[can_schema.DEAD],
        cumulativeInfected=json_data_row[can_schema.CUMULATIVE_INFECTED],
    )


def generate_region_summary(
    latest_values: dict,
    latest_metrics: Optional[Metrics],
    model_output: Optional[CANPyseirLocationOutput],
) -> RegionSummary:
    fips = latest_values[CommonFields.FIPS]
    state = latest_values[CommonFields.STATE]
    actuals = _generate_actuals(latest_values)

    projections = None
    if model_output:
        projections = _generate_api_for_projections(model_output)

    return RegionSummary(
        population=latest_values[CommonFields.POPULATION],
        stateName=us_state_abbrev.ABBREV_US_STATE[state],
        countyName=latest_values.get(CommonFields.COUNTY),
        fips=fips,
        lat=latest_values.get(CommonFields.LATITUDE),
        long=latest_values.get(CommonFields.LONGITUDE),
        actuals=actuals,
        metrics=latest_metrics,
        lastUpdatedDate=datetime.utcnow(),
        projections=projections,
    )


def generate_region_timeseries(
    region_summary: RegionSummary,
    timeseries: TimeseriesDataset,
    metrics_timeseries,
    model_output: Optional[CANPyseirLocationOutput],
) -> RegionSummaryWithTimeseries:
    actuals_timeseries = []

    for row in timeseries.yield_records():
        # Timeseries records don't have population
        row[CommonFields.POPULATION] = region_summary.population
        actual = _generate_actuals(row)
        timeseries_row = ActualsTimeseriesRow(**actual.dict(), date=row[CommonFields.DATE])
        actuals_timeseries.append(timeseries_row)

    model_timeseries = []
    if model_output:
        model_timeseries = [
            _generate_prediction_timeseries_row(row)
            for row in model_output.data.to_dict(orient="records")
        ]

    region_summary_data = {key: getattr(region_summary, key) for (key, _) in region_summary}
    return RegionSummaryWithTimeseries(
        **region_summary_data,
        timeseries=model_timeseries,
        actualsTimeseries=actuals_timeseries,
        metricsTimeseries=metrics_timeseries
    )


def generate_bulk_flattened_timeseries(
    bulk_timeseries: AggregateRegionSummary,
) -> AggregateFlattenedTimeseries:
    rows = []
    for region_timeseries in bulk_timeseries.__root__:
        # Iterate through each state or county in data, adding summary data to each
        # timeseries row.
        summary_data = {
            "countryName": region_timeseries.countryName,
            "countyName": region_timeseries.countyName,
            "stateName": region_timeseries.stateName,
            "fips": region_timeseries.fips,
            "lat": region_timeseries.lat,
            "long": region_timeseries.long,
            # TODO(chris): change this to reflect latest time data updated?
            "lastUpdatedDate": datetime.utcnow(),
        }

        for timeseries_data in region_timeseries.timeseries:
            timeseries_row = PredictionTimeseriesRowWithHeader(
                **summary_data, **timeseries_data.dict()
            )
            rows.append(timeseries_row)

    return AggregateFlattenedTimeseries(__root__=rows)
