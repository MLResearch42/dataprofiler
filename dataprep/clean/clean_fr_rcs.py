"""
Clean and validate a DataFrame column containing French trade registration numbers (RCS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.fr import rcs

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": rcs.compact,
    "standard": rcs.format,
}


def clean_fr_rcs(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean French trade registration numbers (RCS) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of RCS type.
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
    Clean a column of RCS data.

    >>> df = pd.DataFrame({'rcs': ['RCS Nancy B 323 159 715', 'RCS Nancy B 323 159 716']})
    >>> clean_fr_rcs(df, 'rcs')
                           rcs                rcs_clean
    0  RCS Nancy B 323 159 715  RCS Nancy B 323 159 715
    1  RCS Nancy B 323 159 716                      NaN
    """
    return clean_stdnum(
        df,
        column,
        rcs.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_fr_rcs(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate French trade registration numbers (RCS). For each cell, return True or False.

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
    >>> validate_fr_rcs('RCS Nancy B 323 159 715')
    True
    >>> df = pd.DataFrame({'rcs': ['RCS Nancy B 323 159 715', 'RCS Nancy B 323 159 716']})
    >>> validate_fr_rcs(df['rcs'])
    0     True
    1    False
    Name: rcs, dtype: bool
    >>> validate_fr_rcs(df, 'rcs')
    0     True
    1    False
    Name: rcs, dtype: bool
    """
    return validate_stdnum(df, column, rcs.is_valid)
