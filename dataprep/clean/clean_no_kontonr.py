"""
Clean and validate a DataFrame column containing Norwegian bank account numbers (kontonrs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.no import kontonr

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": kontonr.compact,
    "standard": kontonr.format,
    "iban": kontonr.to_iban,
}


def clean_no_kontonr(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Norwegian bank account numbers (kontonrs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of kontonr type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'iban', convert the number to an IBAN.

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
    Clean a column of kontonr data.

    >>> df = pd.DataFrame({"kontonr": ["8601 11 17947", "8601 11 17949"]})
    >>> clean_no_kontonr(df, "kontonr")
             kontonr  kontonr_clean
    0  8601 11 17947  8601.11.17947
    1  8601 11 17949            NaN
    """
    return clean_stdnum(
        df,
        column,
        kontonr.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_no_kontonr(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is kontonr in a DataFrame column. For each cell, return True or False.

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
    >>> validate_no_kontonr("8601 11 17947")
    True
    >>> df = pd.DataFrame({"kontonr": ["8601 11 17947", "8601 11 17949"]})
    >>> validate_no_kontonr(df["kontonr"])
    0     True
    1    False
    Name: kontonr, dtype: bool
    >>> validate_no_kontonr(df, "kontonr")
    0     True
    1    False
    Name: kontonr, dtype: bool
    """
    return validate_stdnum(df, column, kontonr.is_valid)
