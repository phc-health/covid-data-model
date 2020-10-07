import dataclasses
from typing import Mapping
from typing import Sequence

import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields

from covidactnow.datapublic.common_fields import FieldName

from libs.datasets import timeseries


@dataclasses.dataclass
class Method:
    """A method of calculating test positivity"""

    name: str
    numerator: FieldName
    denominator: FieldName

    def calculate(self, smoothed_data: Mapping[FieldName, pd.Series]) -> pd.Series:
        return smoothed_data[self.numerator] / smoothed_data[self.denominator]


@dataclasses.dataclass
class AllMethods:
    all_results: Mapping[str, pd.Series]
    best_result_name: str

    @staticmethod
    def calculate(
        smoothed_timeseries: Mapping[FieldName, pd.Series], methods: Sequence[Method]
    ) -> "AllMethods":
        all_results = {}
        best_method_name = None
        for method in methods:
            result = method.calculate(smoothed_timeseries)
            if result.notna().any():
                all_results[method.name] = result
                if best_method_name is None and result[-14:].notna().any():
                    best_method_name = method.name

        return AllMethods(all_results, best_method_name)
