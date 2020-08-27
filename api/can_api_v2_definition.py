from typing import List, Optional
from libs.enums import Intervention
from libs.datasets.dataset_utils import AggregationLevel
from libs import us_state_abbrev
from libs import base_model
import pydantic
import datetime


class HospitalResourceUtilization(base_model.APIBaseModel):
    capacity: int
    currentUsageTotal: Optional[int]
    currentUsageCovid: int
    typicalUsageRate: float


class Actuals(base_model.APIBaseModel):
    """Known actuals data."""

    cases: Optional[int] = pydantic.Field(
        ..., description="Cumulative number of confirmed or suspected cases"
    )
    deaths: Optional[int] = pydantic.Field(
        ...,
        description=(
            "Cumulative number of deaths that are suspected or "
            "confirmed to have been caused by COVID-19"
        ),
    )
    positiveTests: Optional[int]
    negativeTests: Optional[int]
    tests: Optional[int]
    hospitalBeds: Optional[HospitalResourceUtilization] = pydantic.Field(...)
    icuBeds: Optional[HospitalResourceUtilization] = pydantic.Field(...)
    contactTracers: Optional[int] = pydantic.Field(default=None, description="# of Contact Tracers")


class ActualsTimeseriesRow(Actuals):
    """Actual data for a specific day."""

    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")


class MetricsTimeseriesRow(Metrics):
    """Metrics data for a specific day."""

    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")


class RegionSummary(base_model.APIBaseModel):
    """Summary of actual and prediction data for a single region."""

    countryName: str = "US"
    fips: str = pydantic.Field(
        ...,
        description="Fips Code.  For state level data, 2 characters, for county level data, 5 characters.",
    )
    lat: Optional[float] = pydantic.Field(
        ..., description="Latitude of point within the state or county"
    )
    long: Optional[float] = pydantic.Field(
        ..., description="Longitude of point within the state or county"
    )
    stateName: str = pydantic.Field(..., description="The state name")
    population: int = pydantic.Field(
        ..., description="Total Population in geographic region.", gt=0
    )
    countyName: Optional[str] = pydantic.Field(default=None, description="The county name")
    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")

    metrics: Optional[Metrics] = pydantic.Field(...)
    actuals: Optional[Actuals] = pydantic.Field(...)

    @property
    def aggregate_level(self) -> AggregationLevel:
        if len(self.fips) == 2:
            return AggregationLevel.STATE

        if len(self.fips) == 5:
            return AggregationLevel.COUNTY

    @property
    def state(self) -> str:
        """State abbreviation."""
        return us_state_abbrev.US_STATE_ABBREV[self.stateName]

    def output_key(self, *args):

        if self.aggregate_level is AggregationLevel.STATE:
            return f"{self.state}"

        if self.aggregate_level is AggregationLevel.COUNTY:
            return f"{self.fips}"


class RegionSummaryWithTimeseries(RegionSummary):
    """Summary data for a region with prediction timeseries data and actual timeseries data."""

    metricsTimeseries: Optional[List[MetricsTimeseriesRow]] = pydantic.Field(...)
    actualsTimeseries: List[ActualsTimeseriesRow] = pydantic.Field(...)

    @property
    def region_summary(self) -> RegionSummary:

        data = {}
        # Iterating through self does not force any conversion
        # https://pydantic-docs.helpmanual.io/usage/exporting_models/#dictmodel-and-iteration
        for field, value in self:
            if field not in RegionSummary.__fields__:
                continue
            data[field] = value

        return RegionSummary(**data)


class AggregateRegionSummary(base_model.APIBaseModel):
    """Summary data for multiple regions."""

    __root__: List[RegionSummary] = pydantic.Field(...)

    def output_key(self, *args):
        aggregate_level = self.__root__[0].aggregate_level
        if aggregate_level is AggregationLevel.COUNTY:
            return f"counties"
        if aggregate_level is AggregationLevel.STATE:
            return f"states"


class AggregateRegionSummaryWithTimeseries(base_model.APIBaseModel):
    """Timeseries and summary data for multiple regions."""

    __root__: List[RegionSummaryWithTimeseries] = pydantic.Field(...)

    def output_key(self, *args):
        aggregate_level = self.__root__[0].aggregate_level
        if aggregate_level is AggregationLevel.COUNTY:
            return f"counties.timeseries"
        if aggregate_level is AggregationLevel.STATE:
            return f"states.timeseries"


class MetricsTimeseriesRowWithHeader(MetricsTimeseriesRow):
    """Prediction timeseries row with location information."""

    countryName: str = "US"
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: Optional[str] = pydantic.Field(..., description="The county name")
    intervention: str = pydantic.Field(..., description="Name of high-level intervention in-place")
    fips: str = pydantic.Field(..., description="Fips for State + County. Five character code")
    lat: Optional[float] = pydantic.Field(
        ..., description="Latitude of point within the state or county"
    )
    long: Optional[float] = pydantic.Field(
        ..., description="Longitude of point within the state or county"
    )
    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")

    @property
    def aggregate_level(self) -> AggregationLevel:
        if len(self.fips) == 2:
            return AggregationLevel.STATE

        if len(self.fips) == 5:
            return AggregationLevel.COUNTY


class AggregateFlattenedTimeseries(base_model.APIBaseModel):
    """Flattened prediction timeseries data for multiple regions."""

    __root__: List[MetricsTimeseriesRowWithHeader] = pydantic.Field(...)

    @property
    def aggregate_level(self) -> AggregationLevel:
        if len(self.fips) == 2:
            return AggregationLevel.STATE

        if len(self.fips) == 5:
            return AggregationLevel.COUNTY

    def output_key(self, *args):
        aggregate_level = self.__root__[0].aggregate_level
        if aggregate_level is AggregationLevel.COUNTY:
            return f"counties.timeseries"
        if aggregate_level is AggregationLevel.STATE:
            return f"states.timeseries"
