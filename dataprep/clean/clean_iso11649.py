"""
Clean and validate a DataFrame column containing ISO 11649 structured creditor references (creditor reference).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum import iso11649

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": iso11649.compact,
    "standard": iso11649.format,
}


def clean_iso11649(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean ISO 11649 structured creditor references (creditor reference) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of creditor reference type.
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
    Clean a column of creditor reference data.

    >>> df = pd.DataFrame({'iso11649': ['RF18 5390 0754 7034', 'RF17 5390 0754 7034']})
    >>> clean_iso11649(df, 'iso11649')
                  iso11649       iso11649_clean
    0  RF18 5390 0754 7034  RF18 5390 0754 7034
    1  RF17 5390 0754 7034                  NaN
    """
    return clean_stdnum(
        df,
        column,
        iso11649.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_iso11649(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate ISO 11649 structured creditor references (creditor reference). For each cell, return True or False.

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
    >>> validate_iso11649('RF18 5390 0754 7034')
    True
    >>> df = pd.DataFrame({'iso11649': ['RF18 5390 0754 7034', 'RF17 5390 0754 7034']})
    >>> validate_iso11649(df['iso11649'])
    0     True
    1    False
    Name: iso11649, dtype: bool
    >>> validate_iso11649(df, 'iso11649')
    0     True
    1    False
    Name: iso11649, dtype: bool
    """
    return validate_stdnum(df, column, iso11649.is_valid)
