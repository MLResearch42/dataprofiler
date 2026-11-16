"""
Clean and validate a DataFrame column containing Czech bank account numbers (bank account).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.cz import bankaccount

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": bankaccount.compact,
    "standard": bankaccount.format,
}


def clean_cz_bankaccount(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Czech bank account numbers (bank account) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of bank account type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of bank account, the compact format is the same as the standard one.

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
    Clean a column of bank account data.

    >>> df = pd.DataFrame({'bankaccount': ['34278-0727558021/0100', '4278-727558021/0100']})
    >>> clean_cz_bankaccount(df, 'bankaccount')
                 bankaccount       bankaccount_clean
    0  34278-0727558021/0100  034278-0727558021/0100
    1    4278-727558021/0100                     NaN
    """
    return clean_stdnum(
        df,
        column,
        bankaccount.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_cz_bankaccount(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Czech bank account numbers (bank account). For each cell, return True or False.

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
    >>> validate_cz_bankaccount('34278-0727558021/0100')
    True
    >>> df = pd.DataFrame({'bankaccount': ['34278-0727558021/0100', '4278-727558021/0100']})
    >>> validate_cz_bankaccount(df['bankaccount'])
    0     True
    1    False
    Name: bankaccount, dtype: bool
    >>> validate_cz_bankaccount(df, 'bankaccount')
    0     True
    1    False
    Name: bankaccount, dtype: bool
    """
    return validate_stdnum(df, column, bankaccount.is_valid)
