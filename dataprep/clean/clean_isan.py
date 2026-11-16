"""
Clean and validate a DataFrame column containing
International Standard Audiovisual Numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum import isan

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    formatted = isan.format(val, strip_check_digits=True, add_check_digits=False)
    return [formatted[0:14], formatted[15:19], formatted[20:]]


_FORMATTERS = {
    "compact": isan.compact,
    "standard": isan.format,
    "binary": isan.to_binary,
    "urn": isan.to_urn,
    "xml": isan.to_xml,
}

_SPLIT_COLUMNS = ("root_identifier", "episode", "version")


def clean_isan(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean ISAN type data in a DataFrame column.

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
        If output_format = 'binary', return ISAN string with binary format.
        If output_format = 'urn', return ISAN string with URN format.
        If output_format = 'xml', return ISAN string with XML format.

        (default: "standard")
    split
        If True, each component derived from the number string is put into its
        own column: root_identifier, episode, version.

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
    Clean a column of ISAN data.

    >>> df = pd.DataFrame({"isan": ["000000018947000000000000"]})
    >>> clean_isan(df, "isan", inplace=True)
                              isan_clean
    0  0000-0001-8947-0000-8-0000-0000-D
    """
    return clean_stdnum(
        df,
        column,
        isan.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_isan(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is ISAN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_isan("000000018947000000000000")
    True
    >>> df = pd.DataFrame({"isan": ["000000018947000000000000"]})
    >>> validate_isan(df["isan"])
    0    True
    Name: isan, dtype: bool
    >>> validate_isan(df, "isan")
    0    True
    Name: isan, dtype: bool
    """
    return validate_stdnum(df, column, isan.is_valid)
