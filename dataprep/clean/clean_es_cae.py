"""
Clean and validate a DataFrame column containing Spanish activity establishment codes (CAE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.es import cae

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cae.compact,
    "standard": cae.compact,
}


def clean_es_cae(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Spanish activity establishment codes (CAE) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CAE type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of CAE, the compact format is the same as the standard one.

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
    Clean a column of CAE data.

    >>> df = pd.DataFrame({'cae': ['ES00008V1488Q', '00008V1488']})
    >>> clean_es_cae(df, 'cae')
                 cae      cae_clean
    0  ES00008V1488Q  ES00008V1488Q
    1     00008V1488            NaN
    """
    return clean_stdnum(
        df,
        column,
        cae.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_es_cae(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Spanish activity establishment codes (CAE). For each cell, return True or False.

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
    >>> validate_es_cae('ES00008V1488Q')
    True
    >>> df = pd.DataFrame({'cae': ['ES00008V1488Q', '00008V1488']})
    >>> validate_es_cae(df['cae'])
    0     True
    1    False
    Name: cae, dtype: bool
    >>> validate_es_cae(df, 'cae')
    0     True
    1    False
    Name: cae, dtype: bool
    """
    return validate_stdnum(df, column, cae.is_valid)
