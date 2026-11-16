"""
Clean and validate a DataFrame column containing U.S. individual taxpayer identification numbers (ITIN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.us import itin

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": itin.compact,
    "standard": itin.format,
}


def clean_us_itin(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean U.S. individual taxpayer identification numbers (ITIN) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of ITIN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.

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
    Clean a column of ITIN data.

    >>> df = pd.DataFrame({'itin': ['912-90-3456', '9129-03456']})
    >>> clean_us_itin(df, 'itin')
              itin   itin_clean
    0  912-90-3456  912-90-3456
    1   9129-03456          NaN
    """
    return clean_stdnum(
        df,
        column,
        itin.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_us_itin(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate U.S. individual taxpayer identification numbers (ITIN). For each cell, return True or False.

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
    >>> validate_us_itin('912-90-3456')
    True
    >>> df = pd.DataFrame({'itin': ['912-90-3456', '9129-03456']})
    >>> validate_us_itin(df['itin'])
    0     True
    1    False
    Name: itin, dtype: bool
    >>> validate_us_itin(df, 'itin')
    0     True
    1    False
    Name: itin, dtype: bool
    """
    return validate_stdnum(df, column, itin.is_valid)
