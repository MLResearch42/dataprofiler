"""
Clean and validate a DataFrame column containing Slovak VAT numbers (DPH).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.sk import dph

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": dph.compact,
    "standard": dph.compact,
}


def clean_sk_dph(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Slovak VAT numbers (DPH) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of DPH type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of DPH, the compact format is the same as the standard one.

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
    Clean a column of DPH data.

    >>> df = pd.DataFrame({'dph': ['SK 202 274 96 19', 'SK 202 274 96 18']})
    >>> clean_sk_dph(df, 'dph')
                    dph   dph_clean
    0  SK 202 274 96 19  2022749619
    1  SK 202 274 96 18         NaN
    """
    return clean_stdnum(
        df,
        column,
        dph.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_sk_dph(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Slovak VAT numbers (DPH). For each cell, return True or False.

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
    >>> validate_sk_dph('SK 202 274 96 19')
    True
    >>> df = pd.DataFrame({'dph': ['SK 202 274 96 19', 'SK 202 274 96 18']})
    >>> validate_sk_dph(df['dph'])
    0     True
    1    False
    Name: dph, dtype: bool
    >>> validate_sk_dph(df, 'dph')
    0     True
    1    False
    Name: dph, dtype: bool
    """
    return validate_stdnum(df, column, dph.is_valid)
