"""
Clean and validate a DataFrame column containing Indian virtual identity numbers (VID).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.in_ import vid

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": vid.compact,
    "standard": vid.format,
}


def clean_in_vid(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Indian virtual identity numbers (VID) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of VID type.
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
    Clean a column of VID data.

    >>> df = pd.DataFrame({'vid': ['2341234123412341', '2341234123412342']})
    >>> clean_in_vid(df, 'vid')
                    vid            vid_clean
    0  2341234123412341  2341 2341 2341 2341
    1  2341234123412342                  NaN
    """
    return clean_stdnum(
        df,
        column,
        vid.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_in_vid(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Indian virtual identity numbers (VID). For each cell, return True or False.

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
    >>> validate_in_vid('2341234123412341')
    True
    >>> df = pd.DataFrame({'vid': ['2341234123412341', '2341234123412342']})
    >>> validate_in_vid(df['vid'])
    0     True
    1    False
    Name: vid, dtype: bool
    >>> validate_in_vid(df, 'vid')
    0     True
    1    False
    Name: vid, dtype: bool
    """
    return validate_stdnum(df, column, vid.is_valid)
