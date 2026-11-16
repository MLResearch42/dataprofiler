"""
Implement categorical drop imputer.
"""

from typing import Any, List, Optional

import pandas as pd

from ...utils import in_null_values


class DropImputer:
    """Drop column with missing values
    Attributes:
        null_values
            Specified null values which should be recognized
        fill_value
            Value used for imputing missing values.
    """

    def __init__(self, null_values: Optional[List[Any]], fill_value: str = "") -> None:
        """
        This function initiate drop imputer.

        Parameters
        ----------
        null_values
            Specified null values which should be recognized
        fill_value
            Value used for imputing missing values.
        """

        self.null_values = null_values
        self.fill_value = fill_value
        self.isdrop = False

    def fit(self, col_df: pd.Series) -> Any:
        """
        Check if the provided column need to be dropped.
        If categorical values in null values,
            then the column should be dropped.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.isdrop = any(self.check_isdrop(val) for val in col_df.tolist())
        return self

    def transform(self, col_df: pd.Series) -> pd.Series:
        """
        Check the value of isdrop.
        If yes, then drop this column.
        If no, then return origin column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        if not self.isdrop:
            return col_df
        return pd.Series([], dtype=col_df.dtype, name=col_df.name)

    def fit_transform(self, col_df: pd.Series) -> pd.Series:
        """
        Check if the provided column need to be dropped.
        If yes, then drop this column.
        If no, then return origin df.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def check_isdrop(self, val: str) -> bool:
        """
        Check if the value is missing value.
        If yes, then the whole column should be dropped.
        If no, then return origin df.

        Parameters
        ----------
        val
            Current value needs to be checked.
        """

        return in_null_values(val, self.null_values)
