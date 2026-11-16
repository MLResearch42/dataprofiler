"""
Clean and validate a DataFrame column containing Irish personal numbers (PPSs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.ie import pps

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": pps.compact,
    "standard": pps.compact,
}


def clean_ie_pps(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Irish personal numbers (PPSs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of PPS type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of PPS, the compact format is the same as the standard one.

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
    Clean a column of PPS data.

    >>> df = pd.DataFrame({"pps": ["6433435OA", "6433435VH"]})
    >>> clean_ie_pps(df, "pps")
             pps  pps_clean
    0  6433435OA  6433435OA
    1  6433435VH        NaN
    """
    return clean_stdnum(
        df,
        column,
        pps.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_ie_pps(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is PPS in a DataFrame column. For each cell, return True or False.

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
    >>> validate_ie_pps("6433435OA")
    True
    >>> df = pd.DataFrame({"pps": ["6433435OA", "6433435VH"]})
    >>> validate_ie_pps(df["pps"])
    0     True
    1    False
    Name: pps, dtype: bool
    >>> validate_ie_pps(df, "pps")
    0     True
    1    False
    Name: pps, dtype: bool
    """
    return validate_stdnum(df, column, pps.is_valid)
