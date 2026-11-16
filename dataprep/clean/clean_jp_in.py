"""
Clean and validate a DataFrame column containing Japanese individual numbers (IN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.jp import in_ as individual_number

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": individual_number.compact,
    "standard": individual_number.format,
}


def clean_jp_in(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Japanese individual numbers (IN) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of IN type.
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
    Clean a column of IN data.

    >>> df = pd.DataFrame({'in': ['6214 9832 0257', '6214 9832 0258']})
    >>> clean_jp_in(df, 'in')
                   in        in_clean
    0  6214 9832 0257  6214 9832 0257
    1  6214 9832 0258             NaN
    """
    return clean_stdnum(
        df,
        column,
        individual_number.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_jp_in(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Japanese individual numbers (IN). For each cell, return True or False.

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
    >>> validate_jp_in('6214 9832 0257')
    True
    >>> df = pd.DataFrame({'in': ['6214 9832 0257', '6214 9832 0258']})
    >>> validate_jp_in(df['in'])
    0     True
    1    False
    Name: in, dtype: bool
    >>> validate_jp_in(df, 'in')
    0     True
    1    False
    Name: in, dtype: bool
    """
    return validate_stdnum(df, column, individual_number.is_valid)
