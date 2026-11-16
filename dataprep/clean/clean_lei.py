"""
Clean and validate a DataFrame column containing
Legal Entity Identifier (LEI) Numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum import lei

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    compacted = lei.compact(val)
    return [compacted[:4], compacted[6:18], compacted[18:]]


_FORMATTERS = {
    "compact": lei.compact,
    "standard": lei.compact,
}

_SPLIT_COLUMNS = ("issuing_LOU", "organisation_identifier", "check_digits")


def clean_lei(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean LEI Numbers type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of LEI type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators.
        If output_format = 'standard', return string with proper separators.

        (default: "standard")
    split
        If True, each component derived from the number string is put into its
        own column: issuing_LOU, organisation_identifier, check_digits.

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
    Clean a column of LEI Numbers data.

    >>> df = pd.DataFrame({"lei": ["213800KUD8LAJWSQ9D15"]})
    >>> clean_lei(df, "lei", inplace=True)
                  lei_clean
    0  213800KUD8LAJWSQ9D15
    """
    return clean_stdnum(
        df,
        column,
        lei.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_lei(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is LEI Numbers in a DataFrame column.
    For each cell, return True or False.

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
    >>> validate_lei("213800KUD8LAJWSQ9D15")
    True
    >>> df = pd.DataFrame({"lei": ["213800KUD8LAJWSQ9D15"]})
    >>> validate_lei(df["lei"])
    0    True
    Name: lei, dtype: bool
    >>> validate_lei(df, "lei")
    0    True
    Name: lei, dtype: bool
    """
    return validate_stdnum(df, column, lei.is_valid)
