"""
Clean and validate a DataFrame column containing Austrian company register numbers (Firmenbuchnummer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.at import businessid

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": businessid.compact,
    "standard": businessid.compact,
}


def clean_at_businessid(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Austrian company register numbers (Firmenbuchnummer) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Firmenbuchnummer type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of Firmenbuchnummer, the compact format is the same as the standard one.

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
    Clean a column of Firmenbuchnummer data.

    >>> df = pd.DataFrame({'businessid': ['FN 122119m', 'm123123']})
    >>> clean_at_businessid(df, 'businessid')
       businessid businessid_clean
    0  FN 122119m          122119m
    1     m123123              NaN
    """
    return clean_stdnum(
        df,
        column,
        businessid.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_at_businessid(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Austrian company register numbers (Firmenbuchnummer). For each cell, return True or False.

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
    >>> validate_at_businessid('FN 122119m')
    True
    >>> df = pd.DataFrame({'businessid': ['FN 122119m', 'm123123']})
    >>> validate_at_businessid(df['businessid'])
    0     True
    1    False
    Name: businessid, dtype: bool
    >>> validate_at_businessid(df, 'businessid')
    0     True
    1    False
    Name: businessid, dtype: bool
    """
    return validate_stdnum(df, column, businessid.is_valid)
