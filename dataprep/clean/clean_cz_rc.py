"""
Clean and validate a DataFrame column containing Czech birth numbers (RCs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.cz import rc

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": rc.compact,
    "standard": rc.format,
}


def clean_cz_rc(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Czech birth numbers (RCs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of RC type.
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
    Clean a column of RC data.

    >>> df = pd.DataFrame({"rc": ["7103192745", "7103192746"]})
    >>> clean_cz_rc(df, "rc")
               rc     rc_clean
    0  7103192745  710319/2745
    1  7103192746          NaN
    """
    return clean_stdnum(
        df,
        column,
        rc.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_cz_rc(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is RC in a DataFrame column. For each cell, return True or False.

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
    >>> validate_cz_rc("7103192745")
    True
    >>> df = pd.DataFrame({"rc": ["7103192745", "7103192746"]})
    >>> validate_cz_rc(df["rc"])
    0     True
    1    False
    Name: rc, dtype: bool
    >>> validate_cz_rc(df, "rc")
    0     True
    1    False
    Name: rc, dtype: bool
    """
    return validate_stdnum(df, column, rc.is_valid)
