"""
Clean and validate a DataFrame column containing ISO 10962 Classification of Financial Instruments codes (CFI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum import cfi

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cfi.compact,
    "standard": cfi.compact,
}


def clean_cfi(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean ISO 10962 Classification of Financial Instruments codes (CFI) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CFI type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of CFI, the compact format is the same as the standard one.

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
    Clean a column of CFI data.

    >>> df = pd.DataFrame({'cfi': ['ELNUFR', 'ELNUFQ']})
    >>> clean_cfi(df, 'cfi')
          cfi cfi_clean
    0  ELNUFR    ELNUFR
    1  ELNUFQ       NaN
    """
    return clean_stdnum(
        df,
        column,
        cfi.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_cfi(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate ISO 10962 Classification of Financial Instruments codes (CFI). For each cell, return True or False.

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
    >>> validate_cfi('ELNUFR')
    True
    >>> df = pd.DataFrame({'cfi': ['ELNUFR', 'ELNUFQ']})
    >>> validate_cfi(df['cfi'])
    0     True
    1    False
    Name: cfi, dtype: bool
    >>> validate_cfi(df, 'cfi')
    0     True
    1    False
    Name: cfi, dtype: bool
    """
    return validate_stdnum(df, column, cfi.is_valid)
