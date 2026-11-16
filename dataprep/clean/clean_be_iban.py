"""
Clean and validate a DataFrame column containing Belgian IBANs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.be import iban

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": iban.compact,
    "standard": iban.format,
    "bic": iban.to_bic,
}


def clean_be_iban(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Belgian IBAN (International Bank Account Number) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Belgian IBAN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'bic', return the BIC for the bank that this number refers to.

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
    Clean a column of Belgian IBANs data.

    >>> df = pd.DataFrame({"be_iban": ["BE32 123-4567890-02", "BE41091811735141"]})
    >>> clean_be_iban(df, "be_iban")
                   be_iban        be_iban_clean
    0  BE32 123-4567890-02  BE32 1234 5678 9002
    1     BE41091811735141                  NaN
    """
    return clean_stdnum(
        df,
        column,
        iban.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_be_iban(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is Belgian IBAN in a DataFrame column.
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
    >>> validate_be_iban("BE32 123-4567890-02")
    True
    >>> df = pd.DataFrame({"be_iban": ["BE32 123-4567890-02", "BE41091811735141"]})
    >>> validate_be_iban(df["be_iban"])
    0     True
    1    False
    Name: be_iban, dtype: bool
    >>> validate_be_iban(df, "be_iban")
    0     True
    1    False
    Name: be_iban, dtype: bool
    """
    return validate_stdnum(df, column, iban.is_valid)
