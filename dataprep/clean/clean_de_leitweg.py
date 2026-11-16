"""
Clean and validate a DataFrame column containing German electronic invoice routing identifiers (Leitweg-ID).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.de import leitweg

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": leitweg.compact,
    "standard": leitweg.compact,
}


def clean_de_leitweg(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean German electronic invoice routing identifiers (Leitweg-ID) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Leitweg-ID type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of Leitweg-ID, the compact format is the same as the standard one.

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
    Clean a column of Leitweg-ID data.

    >>> df = pd.DataFrame({'leitweg': ['991-03730-19', '1-03730-19']})
    >>> clean_de_leitweg(df, 'leitweg')
            leitweg leitweg_clean
    0  991-03730-19  991-03730-19
    1    1-03730-19           NaN
    """
    return clean_stdnum(
        df,
        column,
        leitweg.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_de_leitweg(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate German electronic invoice routing identifiers (Leitweg-ID). For each cell, return True or False.

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
    >>> validate_de_leitweg('991-03730-19')
    True
    >>> df = pd.DataFrame({'leitweg': ['991-03730-19', '1-03730-19']})
    >>> validate_de_leitweg(df['leitweg'])
    0     True
    1    False
    Name: leitweg, dtype: bool
    >>> validate_de_leitweg(df, 'leitweg')
    0     True
    1    False
    Name: leitweg, dtype: bool
    """
    return validate_stdnum(df, column, leitweg.is_valid)
