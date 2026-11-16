"""
Implement numerical minmax scaler.
"""

from typing import Any, Union

import pandas as pd


class MinmaxScaler:
    """Min Value and Max Value Scaler for scaling numerical values
    Attributes:
        name
            Name of scaler
        min
            Min value of provided data column
        max
            Max value of provided data column
    """

    def __init__(self) -> None:
        """
        This function initiate numerical scaler.
        """
        self.name = "minmaxScaler"
        self.min = 0
        self.max = 0
        self.range = 1

    def fit(self, col_df: pd.Series) -> Any:
        """
        Extract min value and max value for Minmax Scaler according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.min = col_df.min()
        self.max = col_df.max()
        self.range = self.max - self.min
        # a constant column has no spread: it is scaled to 0 instead of dividing by zero
        if not self.range > 0:
            self.range = 1
        return self

    def transform(self, col_df: pd.Series) -> pd.Series:
        """
        Transform the provided data column with the extracted min value and max value.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.compute_val)
        return result

    def fit_transform(self, col_df: pd.Series) -> pd.Series:
        """ "
        Extract min value and max value for Minmax Scaler according to the provided column.
        Transform the provided data column with the extracted min value and max value.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def compute_val(self, val: Union[int, float]) -> Union[int, float]:
        """
        Compute scaling value of provided value with fitted min value and max value.

        Parameters
        ----------
        val
            Value should be scaled.
        """

        return (val - self.min) / self.range
