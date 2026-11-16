"""
Clean and validate a DataFrame column containing Swedish company numbers (Orgnr).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.se import orgnr

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": orgnr.compact,
    "standard": orgnr.format,
}


def clean_se_orgnr(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Swedish company numbers (Orgnr) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Orgnr type.
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
    Clean a column of Orgnr data.

    >>> df = pd.DataFrame({'orgnr': ['1234567897', '1234567891']})
    >>> clean_se_orgnr(df, 'orgnr')
            orgnr  orgnr_clean
    0  1234567897  123456-7897
    1  1234567891          NaN
    """
    return clean_stdnum(
        df,
        column,
        orgnr.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_se_orgnr(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Swedish company numbers (Orgnr). For each cell, return True or False.

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
    >>> validate_se_orgnr('1234567897')
    True
    >>> df = pd.DataFrame({'orgnr': ['1234567897', '1234567891']})
    >>> validate_se_orgnr(df['orgnr'])
    0     True
    1    False
    Name: orgnr, dtype: bool
    >>> validate_se_orgnr(df, 'orgnr')
    0     True
    1    False
    Name: orgnr, dtype: bool
    """
    return validate_stdnum(df, column, orgnr.is_valid)
