"""
Clean and validate a DataFrame column containing Guinean tax numbers (NIFp).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.gn import nifp

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": nifp.compact,
    "standard": nifp.format,
}


def clean_gn_nifp(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Guinean tax numbers (NIFp) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of NIFp type.
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
    Clean a column of NIFp data.

    >>> df = pd.DataFrame({'nifp': ['693770885', '12345']})
    >>> clean_gn_nifp(df, 'nifp')
            nifp   nifp_clean
    0  693770885  693-770-885
    1      12345          NaN
    """
    return clean_stdnum(
        df,
        column,
        nifp.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_gn_nifp(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Guinean tax numbers (NIFp). For each cell, return True or False.

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
    >>> validate_gn_nifp('693770885')
    True
    >>> df = pd.DataFrame({'nifp': ['693770885', '12345']})
    >>> validate_gn_nifp(df['nifp'])
    0     True
    1    False
    Name: nifp, dtype: bool
    >>> validate_gn_nifp(df, 'nifp')
    0     True
    1    False
    Name: nifp, dtype: bool
    """
    return validate_stdnum(df, column, nifp.is_valid)
