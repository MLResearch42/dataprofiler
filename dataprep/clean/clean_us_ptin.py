"""
Clean and validate a DataFrame column containing U.S. preparer tax identification numbers (PTIN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.us import ptin

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": ptin.compact,
    "standard": ptin.compact,
}


def clean_us_ptin(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean U.S. preparer tax identification numbers (PTIN) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of PTIN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of PTIN, the compact format is the same as the standard one.

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
    Clean a column of PTIN data.

    >>> df = pd.DataFrame({'ptin': ['P-00634642', '00634642']})
    >>> clean_us_ptin(df, 'ptin')
             ptin ptin_clean
    0  P-00634642  P00634642
    1    00634642        NaN
    """
    return clean_stdnum(
        df,
        column,
        ptin.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_us_ptin(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate U.S. preparer tax identification numbers (PTIN). For each cell, return True or False.

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
    >>> validate_us_ptin('P-00634642')
    True
    >>> df = pd.DataFrame({'ptin': ['P-00634642', '00634642']})
    >>> validate_us_ptin(df['ptin'])
    0     True
    1    False
    Name: ptin, dtype: bool
    >>> validate_us_ptin(df, 'ptin')
    0     True
    1    False
    Name: ptin, dtype: bool
    """
    return validate_stdnum(df, column, ptin.is_valid)
