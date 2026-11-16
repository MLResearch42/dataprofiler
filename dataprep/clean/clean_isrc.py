"""
Clean and validate a DataFrame column containing International Standard Recording Codes (ISRC).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum import isrc

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": isrc.compact,
    "standard": isrc.format,
}


def clean_isrc(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean International Standard Recording Codes (ISRC) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of ISRC type.
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
    Clean a column of ISRC data.

    >>> df = pd.DataFrame({'isrc': ['US-SKG-19-12345', 'XX-SKG-19-12345']})
    >>> clean_isrc(df, 'isrc')
                  isrc       isrc_clean
    0  US-SKG-19-12345  US-SKG-19-12345
    1  XX-SKG-19-12345              NaN
    """
    return clean_stdnum(
        df,
        column,
        isrc.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_isrc(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate International Standard Recording Codes (ISRC). For each cell, return True or False.

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
    >>> validate_isrc('US-SKG-19-12345')
    True
    >>> df = pd.DataFrame({'isrc': ['US-SKG-19-12345', 'XX-SKG-19-12345']})
    >>> validate_isrc(df['isrc'])
    0     True
    1    False
    Name: isrc, dtype: bool
    >>> validate_isrc(df, 'isrc')
    0     True
    1    False
    Name: isrc, dtype: bool
    """
    return validate_stdnum(df, column, isrc.is_valid)
