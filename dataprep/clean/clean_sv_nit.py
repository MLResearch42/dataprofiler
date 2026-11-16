"""
Clean and validate a DataFrame column containing Salvadoran tax numbers (NIT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.sv import nit

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": nit.compact,
    "standard": nit.format,
}


def clean_sv_nit(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Salvadoran tax numbers (NIT) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of NIT type.
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
    Clean a column of NIT data.

    >>> df = pd.DataFrame({'nit': ['0614-050707-104-8', '0614-050707-104-0']})
    >>> clean_sv_nit(df, 'nit')
                     nit          nit_clean
    0  0614-050707-104-8  0614-050707-104-8
    1  0614-050707-104-0                NaN
    """
    return clean_stdnum(
        df,
        column,
        nit.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_sv_nit(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Salvadoran tax numbers (NIT). For each cell, return True or False.

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
    >>> validate_sv_nit('0614-050707-104-8')
    True
    >>> df = pd.DataFrame({'nit': ['0614-050707-104-8', '0614-050707-104-0']})
    >>> validate_sv_nit(df['nit'])
    0     True
    1    False
    Name: nit, dtype: bool
    >>> validate_sv_nit(df, 'nit')
    0     True
    1    False
    Name: nit, dtype: bool
    """
    return validate_stdnum(df, column, nit.is_valid)
