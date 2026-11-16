"""
Implement one-hot encoder.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ...utils import is_missing


class OneHotEncoder:
    """One-hot encoder for encoding categorical values
    Attributes:
        name
            Name of encoder
        unique_list
            Unique categorical values in provided data columns
        unique_num
            Number of unique categorical values in provided data columns
    """

    def __init__(self) -> None:
        """
        This function initiate numerical scaler.
        """

        self.name = "OneHotEncoder"
        self.unique_list = np.zeros(1)
        self.unique_num = 0
        self._positions: Dict[Any, int] = {}

    def fit(self, col_df: pd.Series) -> Any:
        """
        Extract the unique categories of the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.unique_list = col_df.drop_duplicates().values
        self.unique_num = len(self.unique_list)
        self._positions = {self._key(val): pos for pos, val in enumerate(self.unique_list.tolist())}
        return self

    def transform(self, col_df: pd.Series) -> pd.Series:
        """
        Transform the provided data column with the extracted categories.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.compute_val)
        return result

    def fit_transform(self, col_df: pd.Series) -> pd.Series:
        """ "
        Extract the unique categories.
        Transform the provided data column with the extracted categories.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def compute_val(self, val: Any) -> List[float]:
        """
        Compute the one-hot vector of a value. The vector has one entry for each
        category seen by ``fit``. A category that was not seen by ``fit`` is
        encoded as a vector of zeros.

        Parameters
        ----------
        val
            Value should be transferred to one-hot encoding.
        """
        result = [0.0] * self.unique_num
        position = self._positions.get(self._key(val))
        if position is not None:
            result[position] = 1.0
        return result

    @staticmethod
    def _key(val: Any) -> Any:
        """
        Dictionary key of a category: all missing value markers are one category.
        """
        return None if is_missing(val) else val
