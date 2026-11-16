"""
Clean and validate a DataFrame column containing ISO 9362 Business identifier codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum import bic

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    compacted = bic.compact(val)
    return [compacted[:4], compacted[4:6], compacted[6:8]]


_FORMATTERS = {
    "compact": bic.compact,
    "standard": bic.format,
}

_SPLIT_COLUMNS = ("institution_code", "country_code", "location_code")


def clean_bic(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean ISO 9362 Business identifier codes type data in a DataFrame column.

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
        own column: institution_code, country_code, location_code.

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
    Clean a column of BIC data.

    >>> df = pd.DataFrame({"bic": ["AGRIFRPP", "ABNA BE 2A"]})
    >>> clean_bic(df, "bic", inplace=True)
      bic_clean
    0  AGRIFRPP
    1  ABNABE2A
    """
    return clean_stdnum(
        df,
        column,
        bic.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_bic(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is BIC in a DataFrame column. For each cell, return True or False.

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
    >>> validate_bic("AGRIFRPP")
    True
    >>> df = pd.DataFrame({"bic": ["AGRIFRPP", "ABNA BE 2A"]})
    >>> validate_bic(df["bic"])
    0    True
    1    True
    Name: bic, dtype: bool
    >>> validate_bic(df, "bic")
    0    True
    1    True
    Name: bic, dtype: bool
    """
    return validate_stdnum(df, column, bic.is_valid)
