"""
Clean and validate a DataFrame column containing Turkish personal identification numbers (T.C. Kimlik).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.tr import tckimlik

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": tckimlik.compact,
    "standard": tckimlik.compact,
}


def clean_tr_tckimlik(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Turkish personal identification numbers (T.C. Kimlik) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of T.C. Kimlik type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of T.C. Kimlik, the compact format is the same as the standard one.

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
    Clean a column of T.C. Kimlik data.

    >>> df = pd.DataFrame({'tckimlik': ['17291716060', '17291716050']})
    >>> clean_tr_tckimlik(df, 'tckimlik')
          tckimlik tckimlik_clean
    0  17291716060    17291716060
    1  17291716050            NaN
    """
    return clean_stdnum(
        df,
        column,
        tckimlik.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_tr_tckimlik(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Turkish personal identification numbers (T.C. Kimlik). For each cell, return True or False.

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
    >>> validate_tr_tckimlik('17291716060')
    True
    >>> df = pd.DataFrame({'tckimlik': ['17291716060', '17291716050']})
    >>> validate_tr_tckimlik(df['tckimlik'])
    0     True
    1    False
    Name: tckimlik, dtype: bool
    >>> validate_tr_tckimlik(df, 'tckimlik')
    0     True
    1    False
    Name: tckimlik, dtype: bool
    """
    return validate_stdnum(df, column, tckimlik.is_valid)
