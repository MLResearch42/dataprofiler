"""
Clean and validate a DataFrame column containing Ghanaian taxpayer identification numbers (TIN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.gh import tin

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": tin.compact,
    "standard": tin.compact,
}


def clean_gh_tin(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Ghanaian taxpayer identification numbers (TIN) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of TIN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of TIN, the compact format is the same as the standard one.

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
    Clean a column of TIN data.

    >>> df = pd.DataFrame({'tin': ['C0000803561', 'C0000803562']})
    >>> clean_gh_tin(df, 'tin')
               tin    tin_clean
    0  C0000803561  C0000803561
    1  C0000803562          NaN
    """
    return clean_stdnum(
        df,
        column,
        tin.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_gh_tin(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Ghanaian taxpayer identification numbers (TIN). For each cell, return True or False.

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
    >>> validate_gh_tin('C0000803561')
    True
    >>> df = pd.DataFrame({'tin': ['C0000803561', 'C0000803562']})
    >>> validate_gh_tin(df['tin'])
    0     True
    1    False
    Name: tin, dtype: bool
    >>> validate_gh_tin(df, 'tin')
    0     True
    1    False
    Name: tin, dtype: bool
    """
    return validate_stdnum(df, column, tin.is_valid)
