"""
Clean and validate a DataFrame column containing Venezuelan tax numbers (RIF).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.ve import rif

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": rif.compact,
    "standard": rif.compact,
}


def clean_ve_rif(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Venezuelan tax numbers (RIF) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of RIF type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of RIF, the compact format is the same as the standard one.

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
    Clean a column of RIF data.

    >>> df = pd.DataFrame({'rif': ['V-11470283-4', 'V-11470283-3']})
    >>> clean_ve_rif(df, 'rif')
                rif   rif_clean
    0  V-11470283-4  V114702834
    1  V-11470283-3         NaN
    """
    return clean_stdnum(
        df,
        column,
        rif.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_ve_rif(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Venezuelan tax numbers (RIF). For each cell, return True or False.

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
    >>> validate_ve_rif('V-11470283-4')
    True
    >>> df = pd.DataFrame({'rif': ['V-11470283-4', 'V-11470283-3']})
    >>> validate_ve_rif(df['rif'])
    0     True
    1    False
    Name: rif, dtype: bool
    >>> validate_ve_rif(df, 'rif')
    0     True
    1    False
    Name: rif, dtype: bool
    """
    return validate_stdnum(df, column, rif.is_valid)
