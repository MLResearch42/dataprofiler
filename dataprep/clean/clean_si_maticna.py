"""
Clean and validate a DataFrame column containing Slovenian corporate registration numbers (Matična).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.si import maticna

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": maticna.compact,
    "standard": maticna.compact,
}


def clean_si_maticna(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Slovenian corporate registration numbers (Matična) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Matična type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of Matična, the compact format is the same as the standard one.

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
    Clean a column of Matična data.

    >>> df = pd.DataFrame({'maticna': ['9331310000', '9331320000']})
    >>> clean_si_maticna(df, 'maticna')
          maticna maticna_clean
    0  9331310000       9331310
    1  9331320000           NaN
    """
    return clean_stdnum(
        df,
        column,
        maticna.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_si_maticna(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Slovenian corporate registration numbers (Matična). For each cell, return True or False.

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
    >>> validate_si_maticna('9331310000')
    True
    >>> df = pd.DataFrame({'maticna': ['9331310000', '9331320000']})
    >>> validate_si_maticna(df['maticna'])
    0     True
    1    False
    Name: maticna, dtype: bool
    >>> validate_si_maticna(df, 'maticna')
    0     True
    1    False
    Name: maticna, dtype: bool
    """
    return validate_stdnum(df, column, maticna.is_valid)
