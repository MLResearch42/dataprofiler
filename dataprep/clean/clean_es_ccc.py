"""
Clean and validate a DataFrame column containing Spanish Bank Account Codes (CCCs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.es import ccc

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": ccc.compact,
    "standard": ccc.format,
    "iban": ccc.to_iban,
}


def clean_es_ccc(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Spanish Bank Account Codes (CCCs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CCC type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'iban', convert the number to an IBAN.

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
    Clean a column of CCC data.

    >>> df = pd.DataFrame({"ccc": ["12341234161234567890", "134-1234-16 1234567890"]})
    >>> clean_es_ccc(df, "ccc")
                          ccc                 ccc_clean
    0    12341234161234567890  1234 1234 16 12345 67890
    1  134-1234-16 1234567890                       NaN
    """
    return clean_stdnum(
        df,
        column,
        ccc.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_es_ccc(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CCC in a DataFrame column. For each cell, return True or False.

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
    >>> validate_es_ccc("12341234161234567890")
    True
    >>> df = pd.DataFrame({"ccc": ["12341234161234567890", "134-1234-16 1234567890"]})
    >>> validate_es_ccc(df["ccc"])
    0     True
    1    False
    Name: ccc, dtype: bool
    >>> validate_es_ccc(df, "ccc")
    0     True
    1    False
    Name: ccc, dtype: bool
    """
    return validate_stdnum(df, column, ccc.is_valid)
