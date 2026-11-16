"""
Clean and validate a DataFrame column containing New Zealand bank account numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.nz import bankaccount

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": bankaccount.compact,
    "standard": bankaccount.format,
    "info": bankaccount.info,
}


def clean_nz_bankaccount(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean New Zealand bank account numbers type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of bankaccount type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'info', return a dictionary of data about the supplied number.
            This typically returns the name of the bank and branch and a BIC if it is valid.

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
    Clean a column of bankaccount data.

    >>> df = pd.DataFrame({"bankaccount": ["0102420100194000", "01-0242-0100195-00"]})
    >>> clean_nz_bankaccount(df, "bankaccount")
              bankaccount    bankaccount_clean
    0    0102420100194000  01-0242-0100194-000
    1  01-0242-0100195-00                  NaN
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


def validate_nz_bankaccount(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is bankaccount in a DataFrame column.
    For each cell, return True or False.

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
    >>> validate_nz_bankaccount("0102420100194000")
    True
    >>> df = pd.DataFrame({"bankaccount": ["0102420100194000", "01-0242-0100195-00"]})
    >>> validate_nz_bankaccount(df["bankaccount"])
    0     True
    1    False
    Name: bankaccount, dtype: bool
    >>> validate_nz_bankaccount(df, "bankaccount")
    0     True
    1    False
    Name: bankaccount, dtype: bool
    """
    return validate_stdnum(df, column, bankaccount.is_valid)
