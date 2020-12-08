from covidactnow.datapublic import common_df

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.timeseries import TimeseriesDataset


class CovidTrackingDataSource(data_source.DataSource):
    INPUT_PATH = dataset_utils.LOCAL_PUBLIC_DATA_PATH / "data" / "covid-tracking" / "timeseries.csv"
    SOURCE_NAME = "covid_tracking"

    INDEX_FIELD_MAP = {f: f for f in TimeseriesDataset.INDEX_FIELDS}

    @classmethod
    def local(cls) -> "CovidTrackingDataSource":
        data = common_df.read_csv(cls.INPUT_PATH).reset_index()
        # Column names are already CommonFields so don't need to rename
        return cls(data, provenance=None)
