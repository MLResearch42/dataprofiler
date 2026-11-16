"""
Clean and validate a DataFrame column containing Slovenian VAT numbers (DDV).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.si import ddv

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": ddv.compact,
    "standard": ddv.compact,
}


def clean_si_ddv(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Slovenian VAT numbers (DDV) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of DDV type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of DDV, the compact format is the same as the standard one.

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
    Clean a column of DDV data.

    >>> df = pd.DataFrame({'ddv': ['SI 5022 3054', 'SI 50223055']})
    >>> clean_si_ddv(df, 'ddv')
                ddv ddv_clean
    0  SI 5022 3054  50223054
    1   SI 50223055       NaN
    """
    return clean_stdnum(
        df,
        column,
        ddv.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_si_ddv(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Slovenian VAT numbers (DDV). For each cell, return True or False.

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
    >>> validate_si_ddv('SI 5022 3054')
    True
    >>> df = pd.DataFrame({'ddv': ['SI 5022 3054', 'SI 50223055']})
    >>> validate_si_ddv(df['ddv'])
    0     True
    1    False
    Name: ddv, dtype: bool
    >>> validate_si_ddv(df, 'ddv')
    0     True
    1    False
    Name: ddv, dtype: bool
    """
    return validate_stdnum(df, column, ddv.is_valid)
