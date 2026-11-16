"""
Clean and validate a DataFrame column containing Senegalese company and association numbers (NINEA).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.sn import ninea

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": ninea.compact,
    "standard": ninea.format,
}


def clean_sn_ninea(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Senegalese company and association numbers (NINEA) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of NINEA type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of NINEA, the compact format is the same as the standard one.

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
    Clean a column of NINEA data.

    >>> df = pd.DataFrame({'ninea': ['306 7221', '3067222']})
    >>> clean_sn_ninea(df, 'ninea')
          ninea ninea_clean
    0  306 7221     3067221
    1   3067222         NaN
    """
    return clean_stdnum(
        df,
        column,
        ninea.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_sn_ninea(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Senegalese company and association numbers (NINEA). For each cell, return True or False.

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
    >>> validate_sn_ninea('306 7221')
    True
    >>> df = pd.DataFrame({'ninea': ['306 7221', '3067222']})
    >>> validate_sn_ninea(df['ninea'])
    0     True
    1    False
    Name: ninea, dtype: bool
    >>> validate_sn_ninea(df, 'ninea')
    0     True
    1    False
    Name: ninea, dtype: bool
    """
    return validate_stdnum(df, column, ninea.is_valid)
