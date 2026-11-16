"""
Clean and validate a DataFrame column containing
International Mobile Subscriber Identity (IMSI) numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum import imsi

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    return list(imsi.split(val))


_FORMATTERS = {
    "compact": imsi.compact,
    "standard": imsi.compact,
}

_SPLIT_COLUMNS = (
    "mobile_country_code",
    "mobile_network_code",
    "mobile_station_identification_number",
)


def clean_imsi(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean IMSI type data in a DataFrame column.

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
        own column: mobile_country_code, mobile_network_code, mobile_station_identification_number.

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

    >>> df = pd.DataFrame({"imsi": ["429 011 234567890", "310-150-123456789"]})
    >>> clean_imsi(df, "imsi", inplace=True)
            imsi_clean
    0  429011234567890
    1  310150123456789
    """
    return clean_stdnum(
        df,
        column,
        imsi.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_imsi(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is IMSI in a DataFrame column. For each cell, return True or False.

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
    >>> validate_imsi("429 011 234567890")
    True
    >>> df = pd.DataFrame({"imsi": ["429 011 234567890", "310-150-123456789"]})
    >>> validate_imsi(df["imsi"])
    0    True
    1    True
    Name: imsi, dtype: bool
    >>> validate_imsi(df, "imsi")
    0    True
    1    True
    Name: imsi, dtype: bool
    """
    return validate_stdnum(df, column, imsi.is_valid)
