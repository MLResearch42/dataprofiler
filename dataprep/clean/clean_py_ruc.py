"""
Clean and validate a DataFrame column containing Paraguay RUC numbers (RUCs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.py import ruc

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": ruc.compact,
    "standard": ruc.format,
}


def clean_py_ruc(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Paraguay RUC numbers (RUCs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of RUC type.
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
    Clean a column of RUC data.

    >>> df = pd.DataFrame({"ruc": ["800000358", "80123456789"]})
    >>> clean_py_ruc(df, "ruc")
               ruc   ruc_clean
    0    800000358  80000035-8
    1  80123456789         NaN
    """
    return clean_stdnum(
        df,
        column,
        ruc.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_py_ruc(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is RUC in a DataFrame column. For each cell, return True or False.

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
    >>> validate_py_ruc("800000358")
    True
    >>> df = pd.DataFrame({"ruc": ["800000358", "80123456789"]})
    >>> validate_py_ruc(df["ruc"])
    0     True
    1    False
    Name: ruc, dtype: bool
    >>> validate_py_ruc(df, "ruc")
    0     True
    1    False
    Name: ruc, dtype: bool
    """
    return validate_stdnum(df, column, ruc.is_valid)
