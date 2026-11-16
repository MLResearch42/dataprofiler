"""
Clean and validate a DataFrame column containing U.S. employer identification numbers (EIN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.us import ein

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": ein.compact,
    "standard": ein.format,
}


def clean_us_ein(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean U.S. employer identification numbers (EIN) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of EIN type.
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
    Clean a column of EIN data.

    >>> df = pd.DataFrame({'ein': ['91-1144442', '911-14-4442']})
    >>> clean_us_ein(df, 'ein')
               ein   ein_clean
    0   91-1144442  91-1144442
    1  911-14-4442         NaN
    """
    return clean_stdnum(
        df,
        column,
        ein.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_us_ein(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate U.S. employer identification numbers (EIN). For each cell, return True or False.

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
    >>> validate_us_ein('91-1144442')
    True
    >>> df = pd.DataFrame({'ein': ['91-1144442', '911-14-4442']})
    >>> validate_us_ein(df['ein'])
    0     True
    1    False
    Name: ein, dtype: bool
    >>> validate_us_ein(df, 'ein')
    0     True
    1    False
    Name: ein, dtype: bool
    """
    return validate_stdnum(df, column, ein.is_valid)
