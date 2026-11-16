"""
Clean and validate a DataFrame column containing IBAN numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum import iban

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    compacted = iban.compact(val)
    return [compacted[:2], compacted[2:4], compacted[4:]]


_FORMATTERS = {
    "compact": iban.compact,
    "standard": iban.format,
}

_SPLIT_COLUMNS = ("country_code", "check_digits", "bban_code")


def clean_iban(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean IBAN type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of ISBN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators.
        If output_format = 'standard', return string with proper separators.

        (default: "standard")
    split
        If True, each component derived from the number string is put into its
        own column: country_code, check_digits, bban_code.

        (default: False)
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
    Clean a column of ISBN data.

    >>> df = pd.DataFrame({"iban": ["978-9024538270", "978-9024538271"]})
    >>> clean_iban(df, "iban", inplace=True)
       iban_clean
    0         NaN
    1         NaN
    """
    return clean_stdnum(
        df,
        column,
        iban.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_iban(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is IBAN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_iban("GR16 0110 1050 0000 1054 7023 795")
    True
    >>> df = pd.DataFrame({"iban": ["978-9024538270", "978-9024538271"]})
    >>> validate_iban(df["iban"])
    0    False
    1    False
    Name: iban, dtype: bool
    >>> validate_iban(df, "iban")
    0    False
    1    False
    Name: iban, dtype: bool
    """
    return validate_stdnum(df, column, iban.is_valid)
