"""
Clean and validate a DataFrame column containing Vietnamese tax numbers (MST).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.vn import mst

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": mst.compact,
    "standard": mst.format,
}


def clean_vn_mst(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Vietnamese tax numbers (MST) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of MST type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of MST, the compact format is the same as the standard one.

        (default: "standard")
    inplace
        If True, the cleaned column replaces the original column.
        Otherwise, keep the original column and append the cleaned column.

        (default: False)
    errors
        How to handle parsing errors.
        - 'coerce': invalid parsing will be set to NaN.
        - 'ignore': invalid parsing will return the input.
        - 'raise': invalid parsing will raise an exception.

        (default: 'coerce')
    progress
        If True, display a progress bar. Only shown for Dask DataFrames.

        (default: True)

    Examples
    --------
    Clean a column of MST data.

    >>> df = pd.DataFrame({'mst': ['0100233488', '12345']})
    >>> clean_vn_mst(df, 'mst')
              mst   mst_clean
    0  0100233488  0100233488
    1       12345         NaN
    """
    return clean_stdnum(
        df,
        column,
        mst.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_vn_mst(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Vietnamese tax numbers (MST). For each cell, return True or False.

    Parameters
    ----------
    df
        A string, a pandas or Dask Series, or a pandas or Dask DataFrame
        containing the data to be validated.
    column
        The name of the column to be validated. Only used when df is a
        DataFrame; if empty, every cell of the DataFrame is validated.

    Examples
    --------
    >>> validate_vn_mst('0100233488')
    True
    >>> df = pd.DataFrame({'mst': ['0100233488', '12345']})
    >>> validate_vn_mst(df['mst'])
    0     True
    1    False
    Name: mst, dtype: bool
    >>> validate_vn_mst(df, 'mst')
    0     True
    1    False
    Name: mst, dtype: bool
    """
    return validate_stdnum(df, column, mst.is_valid)
