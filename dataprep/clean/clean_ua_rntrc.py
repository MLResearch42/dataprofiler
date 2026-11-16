"""
Clean and validate a DataFrame column containing Ukrainian individual taxpayer numbers (RNTRC).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.ua import rntrc

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": rntrc.compact,
    "standard": rntrc.format,
}


def clean_ua_rntrc(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Ukrainian individual taxpayer numbers (RNTRC) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of RNTRC type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of RNTRC, the compact format is the same as the standard one.

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
    Clean a column of RNTRC data.

    >>> df = pd.DataFrame({'rntrc': ['1759013776', '1759013770']})
    >>> clean_ua_rntrc(df, 'rntrc')
            rntrc rntrc_clean
    0  1759013776  1759013776
    1  1759013770         NaN
    """
    return clean_stdnum(
        df,
        column,
        rntrc.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_ua_rntrc(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Ukrainian individual taxpayer numbers (RNTRC). For each cell, return True or False.

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
    >>> validate_ua_rntrc('1759013776')
    True
    >>> df = pd.DataFrame({'rntrc': ['1759013776', '1759013770']})
    >>> validate_ua_rntrc(df['rntrc'])
    0     True
    1    False
    Name: rntrc, dtype: bool
    >>> validate_ua_rntrc(df, 'rntrc')
    0     True
    1    False
    Name: rntrc, dtype: bool
    """
    return validate_stdnum(df, column, rntrc.is_valid)
