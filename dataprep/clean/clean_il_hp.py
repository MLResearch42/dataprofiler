"""
Clean and validate a DataFrame column containing Israeli company numbers (HPs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.il import hp

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": hp.compact,
    "standard": hp.compact,
}


def clean_il_hp(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Israeli company numbers (HPs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of HP type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of HP, the compact format is the same as the standard one.

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
    Clean a column of HP data.

    >>> df = pd.DataFrame({"hp": [" 5161 79157 ", "516179150"]})
    >>> clean_il_hp(df, "hp")
                 hp   hp_clean
    0   5161 79157   516179157
    1     516179150        NaN
    """
    return clean_stdnum(
        df,
        column,
        hp.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_il_hp(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is HP in a DataFrame column. For each cell, return True or False.

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
    >>> validate_il_hp(" 5161 79157 ")
    True
    >>> df = pd.DataFrame({"hp": [" 5161 79157 ", "516179150"]})
    >>> validate_il_hp(df["hp"])
    0     True
    1    False
    Name: hp, dtype: bool
    >>> validate_il_hp(df, "hp")
    0     True
    1    False
    Name: hp, dtype: bool
    """
    return validate_stdnum(df, column, hp.is_valid)
