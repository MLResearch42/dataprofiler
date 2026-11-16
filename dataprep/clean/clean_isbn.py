"""
Clean and validate a DataFrame column containing ISBN numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum import isbn

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _to_isbn13(val: str) -> Any:
    """Render a valid number in the 'isbn13' output format."""
    return isbn.format(isbn.to_isbn13(val))


def _to_isbn10(val: str) -> Any:
    """Render a valid number in the 'isbn10' output format."""
    return isbn.format(isbn.to_isbn10(val))


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    return list(isbn.split(isbn.to_isbn13(val)))


_FORMATTERS = {
    "compact": isbn.compact,
    "standard": isbn.format,
    "isbn13": _to_isbn13,
    "isbn10": _to_isbn10,
}

_SPLIT_COLUMNS = ("prefix_code", "group_code", "publisher_code", "item_code", "check_digit")


def clean_isbn(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean ISBN type data in a DataFrame column.

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
        If output_format = 'isbn13', return ISBN string with 13 digits.
        If output_format = 'isbn10', return ISBN string with 10 digits.

        (default: "standard")
    split
        If True, each component derived from the number string is put into its
        own column: prefix_code, group_code, publisher_code, item_code, check_digit.

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

    >>> df = pd.DataFrame({"isbn": ["978-9024538270", "978-9024538271"]})
    >>> clean_isbn(df, "isbn", inplace=True)
              isbn_clean
    0  978-90-245-3827-0
    1                NaN
    """
    return clean_stdnum(
        df,
        column,
        isbn.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_isbn(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is ISBN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_isbn("978-9024538270")
    True
    >>> df = pd.DataFrame({"isbn": ["978-9024538270", "978-9024538271"]})
    >>> validate_isbn(df["isbn"])
    0     True
    1    False
    Name: isbn, dtype: bool
    >>> validate_isbn(df, "isbn")
    0     True
    1    False
    Name: isbn, dtype: bool
    """
    return validate_stdnum(df, column, isbn.is_valid)
