"""
Implement numerical variance thresholder.
"""

from typing import Any, Union

import pandas as pd


class VarThreholder:
    """Drop column if the variance of this column is less than a threshold.
    Attributes:
        variance_thresh
            Specified variance threshold.
        variance
            Variance of provided data column.
    """

    def __init__(self, variance: Union[int, float]) -> None:
        """
        This function initiate variance thresholder.

        Parameters
        ----------
        variance_thresh
            Variance threshold provided by user. The default value is 0.
        """
        self.variance_thresh = variance
        self.variance = 0

    def fit(self, col_df: pd.Series) -> Any:
        """
        Extract the variance of the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.variance = col_df.var()
        return self

    def transform(self, col_df: pd.Series) -> pd.Series:
        """
        Check if variance of provided column is larger than threshold.
        If yes, then keep the provided column
        If no, just drop it.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        if self.variance > self.variance_thresh:
            return col_df
        return pd.Series([], dtype=col_df.dtype, name=col_df.name)

    def fit_transform(self, col_df: pd.Series) -> pd.Series:
        """
        Extract the variance of the provided column.
        Check if variance of provided column is larger than threshold.
        If yes, then keep the provided column
        If no, just drop it.

        Parameters
        ----------
        col_df
            Provided data column.
        """
        return self.fit(col_df).transform(col_df)
