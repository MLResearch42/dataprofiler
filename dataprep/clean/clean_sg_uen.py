"""
Clean and validate a DataFrame column containing Singaporean unique entity numbers (UEN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.sg import uen

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": uen.compact,
    "standard": uen.format,
}


def clean_sg_uen(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Singaporean unique entity numbers (UEN) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of UEN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of UEN, the compact format is the same as the standard one.

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
    Clean a column of UEN data.

    >>> df = pd.DataFrame({'uen': ['00192200M', '123456']})
    >>> clean_sg_uen(df, 'uen')
             uen  uen_clean
    0  00192200M  00192200M
    1     123456        NaN
    """
    return clean_stdnum(
        df,
        column,
        uen.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_sg_uen(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Singaporean unique entity numbers (UEN). For each cell, return True or False.

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
    >>> validate_sg_uen('00192200M')
    True
    >>> df = pd.DataFrame({'uen': ['00192200M', '123456']})
    >>> validate_sg_uen(df['uen'])
    0     True
    1    False
    Name: uen, dtype: bool
    >>> validate_sg_uen(df, 'uen')
    0     True
    1    False
    Name: uen, dtype: bool
    """
    return validate_stdnum(df, column, uen.is_valid)
