"""
Clean and validate a DataFrame column containing Japanese Corporate Numbers (CNs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.jp import cn

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cn.compact,
    "standard": cn.format,
}


def clean_jp_cn(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Japanese Corporate Numbers (CNs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CN type.
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
    Clean a column of CN data.

    >>> df = pd.DataFrame({"cn": ["5835678256246", "2-8356-7825-6246"]})
    >>> clean_jp_cn(df, "cn")
                     cn          cn_clean
    0     5835678256246  5-8356-7825-6246
    1  2-8356-7825-6246               NaN
    """
    return clean_stdnum(
        df,
        column,
        cn.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_jp_cn(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_jp_cn("5835678256246")
    True
    >>> df = pd.DataFrame({"cn": ["5835678256246", "2-8356-7825-6246"]})
    >>> validate_jp_cn(df["cn"])
    0     True
    1    False
    Name: cn, dtype: bool
    >>> validate_jp_cn(df, "cn")
    0     True
    1    False
    Name: cn, dtype: bool
    """
    return validate_stdnum(df, column, cn.is_valid)
