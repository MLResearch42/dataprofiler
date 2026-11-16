"""
Clean and validate a DataFrame column containing
International Securities Identification Number (ISIN) numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum import isin

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    compacted = isin.compact(val)
    return [compacted[:2], compacted[2:11], compacted[11:]]


_FORMATTERS = {
    "compact": isin.compact,
    "standard": isin.compact,
}

_SPLIT_COLUMNS = ("country_code", "national_security_identifier", "check_digit")


def clean_isin(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean ISIN type data in a DataFrame column.

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
        own column: country_code, national_security_identifier, check_digit.

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
    Clean a column of ISIN data.

    >>> df = pd.DataFrame({"isin": ["US0378331005", "GB00BYXJL758"]})
    >>> clean_isin(df, "isin", inplace=True)
         isin_clean
    0  US0378331005
    1  GB00BYXJL758
    """
    return clean_stdnum(
        df,
        column,
        isin.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_isin(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is ISIN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_isin("US0378331005")
    True
    >>> df = pd.DataFrame({"isin": ["US0378331005", "GB00BYXJL758"]})
    >>> validate_isin(df["isin"])
    0    True
    1    True
    Name: isin, dtype: bool
    >>> validate_isin(df, "isin")
    0    True
    1    True
    Name: isin, dtype: bool
    """
    return validate_stdnum(df, column, isin.is_valid)
