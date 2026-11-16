"""
Clean and validate a DataFrame column containing French excise numbers (accise).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.fr import accise

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": accise.compact,
    "standard": accise.compact,
}


def clean_fr_accise(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean French excise numbers (accise) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of accise type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of accise, the compact format is the same as the standard one.

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
    Clean a column of accise data.

    >>> df = pd.DataFrame({'accise': ['FR023004N9448', 'FR0XX907E0820']})
    >>> clean_fr_accise(df, 'accise')
              accise   accise_clean
    0  FR023004N9448  FR023004N9448
    1  FR0XX907E0820            NaN
    """
    return clean_stdnum(
        df,
        column,
        accise.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_fr_accise(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate French excise numbers (accise). For each cell, return True or False.

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
    >>> validate_fr_accise('FR023004N9448')
    True
    >>> df = pd.DataFrame({'accise': ['FR023004N9448', 'FR0XX907E0820']})
    >>> validate_fr_accise(df['accise'])
    0     True
    1    False
    Name: accise, dtype: bool
    >>> validate_fr_accise(df, 'accise')
    0     True
    1    False
    Name: accise, dtype: bool
    """
    return validate_stdnum(df, column, accise.is_valid)
