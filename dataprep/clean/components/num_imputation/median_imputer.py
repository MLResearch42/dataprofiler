"""
Implement numerical median imputer.
"""

from typing import Any, List, Optional, Union

import pandas as pd

from ...utils import in_null_values, is_missing


class MedianImputer:
    """Median imputer for imputing numerical values
    Attributes:
        null_values
            Specified null values which should be recognized
        median
            Median value
    """

    def __init__(self, null_values: Optional[List[Any]]) -> None:
        """
        This function initiate median imputer.

        Parameters
        ----------
        null_values
            Specified null values which should be recognized
        """

        self.null_values = null_values
        self.median = 0

    def fit(self, col_df: pd.Series) -> Any:
        """
        Find the median value for median imputer according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """
        self.median = pd.to_numeric(col_df, errors="coerce").median()
        return self

    def transform(self, col_df: pd.Series) -> pd.Series:
        """
        Impute the provided data column with the fitted median value.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.fillna)
        return result

    def fit_transform(self, col_df: pd.Series) -> pd.Series:
        """
        Extract the median value from provided column.
        Impute the data column with extracted median value.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def fillna(self, val: Union[int, float]) -> Union[int, float]:
        """
        Check if the value is in the list of null value.
        If yes, impute the data column with extracted median value.
        If no, just return the value.

        Parameters
        ----------
        val
            Each value in the pandas Series
        """

        if is_missing(val) or in_null_values(val, self.null_values):
            return self.median
        return val
