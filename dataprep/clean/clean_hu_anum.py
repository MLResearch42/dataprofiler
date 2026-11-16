"""
Clean and validate a DataFrame column containing Hungarian ANUM numbers (ANUMs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.hu import anum

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": anum.compact,
    "standard": anum.compact,
}


def clean_hu_anum(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Hungarian ANUM numbers (ANUMs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of ANUM type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of ANUM, the compact format is the same as the standard one.

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
    Clean a column of ANUM data.

    >>> df = pd.DataFrame({"anum": ["HU-12892312", "HU-12892313"]})
    >>> clean_hu_anum(df, "anum")
              anum anum_clean
    0  HU-12892312   12892312
    1  HU-12892313        NaN
    """
    return clean_stdnum(
        df,
        column,
        anum.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_hu_anum(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is ANUM in a DataFrame column. For each cell, return True or False.

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
    >>> validate_hu_anum("HU-12892312")
    True
    >>> df = pd.DataFrame({"anum": ["HU-12892312", "HU-12892313"]})
    >>> validate_hu_anum(df["anum"])
    0     True
    1    False
    Name: anum, dtype: bool
    >>> validate_hu_anum(df, "anum")
    0     True
    1    False
    Name: anum, dtype: bool
    """
    return validate_stdnum(df, column, anum.is_valid)
