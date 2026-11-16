"""
Clean and validate a DataFrame column containing
CUSIP numbers (financial security identification number).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum import cusip

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cusip.compact,
    "standard": cusip.compact,
    "isin": cusip.to_isin,
}


def clean_cusip(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean CUSIP numbers type data in a DataFrame column.

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
        If output_format = 'isin', return string with ISIN format.

        (default: "standard")
    split
        This identifier has no components to split into separate columns. The
        parameter is accepted for consistency with the other clean functions and
        has no effect.

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
    Clean a column of CUSIP numbers data.

    >>> df = pd.DataFrame({"cusip": ["DUS0421C5"]})
    >>> clean_cusip(df, "cusip", inplace=True)
      cusip_clean
    0   DUS0421C5
    """
    # pylint: disable=unused-argument
    return clean_stdnum(
        df,
        column,
        cusip.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_cusip(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CUSIP numbers in a DataFrame column.
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
    >>> validate_cusip("DUS0421C5")
    True
    >>> df = pd.DataFrame({"cusip": ["DUS0421C5"]})
    >>> validate_cusip(df["cusip"])
    0    True
    Name: cusip, dtype: bool
    >>> validate_cusip(df, "cusip")
    0    True
    Name: cusip, dtype: bool
    """
    return validate_stdnum(df, column, cusip.is_valid)
