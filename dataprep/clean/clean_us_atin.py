"""
Clean and validate a DataFrame column containing U.S. adoption taxpayer identification numbers (ATIN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.us import atin

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": atin.compact,
    "standard": atin.format,
}


def clean_us_atin(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean U.S. adoption taxpayer identification numbers (ATIN) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of ATIN type.
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
    Clean a column of ATIN data.

    >>> df = pd.DataFrame({'atin': ['123-45-6789', '1234-56789']})
    >>> clean_us_atin(df, 'atin')
              atin   atin_clean
    0  123-45-6789  123-45-6789
    1   1234-56789          NaN
    """
    return clean_stdnum(
        df,
        column,
        atin.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_us_atin(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate U.S. adoption taxpayer identification numbers (ATIN). For each cell, return True or False.

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
    >>> validate_us_atin('123-45-6789')
    True
    >>> df = pd.DataFrame({'atin': ['123-45-6789', '1234-56789']})
    >>> validate_us_atin(df['atin'])
    0     True
    1    False
    Name: atin, dtype: bool
    >>> validate_us_atin(df, 'atin')
    0     True
    1    False
    Name: atin, dtype: bool
    """
    return validate_stdnum(df, column, atin.is_valid)
