"""
Clean and validate a DataFrame column containing Tunisian tax numbers (MF).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.tn import mf

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": mf.compact,
    "standard": mf.format,
}


def clean_tn_mf(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Tunisian tax numbers (MF) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of MF type.
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
    Clean a column of MF data.

    >>> df = pd.DataFrame({'mf': ['1234567/M/A/E/001', '1219773U']})
    >>> clean_tn_mf(df, 'mf')
                      mf           mf_clean
    0  1234567/M/A/E/001  1234567/M/A/E/001
    1           1219773U                NaN
    """
    return clean_stdnum(
        df,
        column,
        mf.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_tn_mf(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Tunisian tax numbers (MF). For each cell, return True or False.

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
    >>> validate_tn_mf('1234567/M/A/E/001')
    True
    >>> df = pd.DataFrame({'mf': ['1234567/M/A/E/001', '1219773U']})
    >>> validate_tn_mf(df['mf'])
    0     True
    1    False
    Name: mf, dtype: bool
    >>> validate_tn_mf(df, 'mf')
    0     True
    1    False
    Name: mf, dtype: bool
    """
    return validate_stdnum(df, column, mf.is_valid)
