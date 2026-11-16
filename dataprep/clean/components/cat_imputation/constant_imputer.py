"""
Implement categorical constant imputer.
"""

# pylint: disable=unused-argument
from typing import Any, List, Optional

import pandas as pd

from ...utils import in_null_values


class ConstantImputer:
    """Constant imputer for imputing categorical values
    Attributes:
        null_values
            Specified null values which should be recognized
        fill_value
            Value used for imputing missing values, the default value is "Missing"
    """

    def __init__(self, null_values: Optional[List[Any]], fill_value: str = "") -> None:
        """
        This function initiate constant imputer.

        Parameters
        ----------
        null_values
            Specified null values which should be recognized
        fill_value
            Value used for imputing missing values.
        """

        self.null_values = null_values
        if len(fill_value) == 0:
            self.fill_value = "Missing"
        else:
            self.fill_value = fill_value

    def fit(self, col_df: pd.Series) -> Any:
        """
        Constant imputer don't need to fit any parameter.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        return self

    def transform(self, col_df: pd.Series) -> pd.Series:
        """
        Impute the provided data column with the fitted parameters.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.fillna)
        return result

    def fit_transform(self, col_df: pd.Series) -> pd.Series:
        """
        Impute the data column with constant value.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def fillna(self, val: str) -> str:
        """
        Check if the value is in the list of null value.
        If yes, impute the data column with constant value.
        If no, just return the value.

        Parameters
        ----------
        val
            Each value in the pandas Series
        """

        if in_null_values(val, self.null_values):
            return self.fill_value
        return val
