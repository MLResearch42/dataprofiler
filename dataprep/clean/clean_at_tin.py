"""
Clean and validate a DataFrame column containing Austrian tax identification numbers (Abgabenkontonummer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.at import tin

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": tin.compact,
    "standard": tin.format,
}


def clean_at_tin(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Austrian tax identification numbers (Abgabenkontonummer) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Abgabenkontonummer type.
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
    Clean a column of Abgabenkontonummer data.

    >>> df = pd.DataFrame({'tin': ['59-119/9013', '59-119/9014']})
    >>> clean_at_tin(df, 'tin')
               tin    tin_clean
    0  59-119/9013  59-119/9013
    1  59-119/9014          NaN
    """
    return clean_stdnum(
        df,
        column,
        tin.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_at_tin(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Austrian tax identification numbers (Abgabenkontonummer). For each cell, return True or False.

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
    >>> validate_at_tin('59-119/9013')
    True
    >>> df = pd.DataFrame({'tin': ['59-119/9013', '59-119/9014']})
    >>> validate_at_tin(df['tin'])
    0     True
    1    False
    Name: tin, dtype: bool
    >>> validate_at_tin(df, 'tin')
    0     True
    1    False
    Name: tin, dtype: bool
    """
    return validate_stdnum(df, column, tin.is_valid)
