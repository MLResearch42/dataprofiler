"""
Clean and validate a DataFrame column containing ISO 6346 shipping container codes (container code).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum import iso6346

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": iso6346.compact,
    "standard": iso6346.format,
}


def clean_iso6346(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean ISO 6346 shipping container codes (container code) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of container code type.
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
    Clean a column of container code data.

    >>> df = pd.DataFrame({'iso6346': ['csqu3054383', 'CSQU3054384']})
    >>> clean_iso6346(df, 'iso6346')
           iso6346  iso6346_clean
    0  csqu3054383  CSQU 305438 3
    1  CSQU3054384            NaN
    """
    return clean_stdnum(
        df,
        column,
        iso6346.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_iso6346(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate ISO 6346 shipping container codes (container code). For each cell, return True or False.

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
    >>> validate_iso6346('csqu3054383')
    True
    >>> df = pd.DataFrame({'iso6346': ['csqu3054383', 'CSQU3054384']})
    >>> validate_iso6346(df['iso6346'])
    0     True
    1    False
    Name: iso6346, dtype: bool
    >>> validate_iso6346(df, 'iso6346')
    0     True
    1    False
    Name: iso6346, dtype: bool
    """
    return validate_stdnum(df, column, iso6346.is_valid)
