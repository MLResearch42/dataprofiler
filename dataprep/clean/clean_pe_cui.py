"""
Clean and validate a DataFrame column containing Peruvian personal numbers (CUIs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.pe import cui

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cui.compact,
    "standard": cui.compact,
    "ruc": cui.to_ruc,
}


def clean_pe_cui(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Peruvian personal numbers (CUIs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CUI type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'ruc', convert the number to a valid RUC.
        Note: in the case of CUI, the compact format is the same as the standard one.

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
    Clean a column of CUI data.

    >>> df = pd.DataFrame({"cui": ["10117410", "10117410-3"]})
    >>> clean_pe_cui(df, "cui")
              cui cui_clean
    0    10117410  10117410
    1  10117410-3       NaN
    """
    return clean_stdnum(
        df,
        column,
        cui.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_pe_cui(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CUI in a DataFrame column. For each cell, return True or False.

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
    >>> validate_pe_cui("10117410")
    True
    >>> df = pd.DataFrame({"cui": ["10117410", "10117410-3"]})
    >>> validate_pe_cui(df["cui"])
    0     True
    1    False
    Name: cui, dtype: bool
    >>> validate_pe_cui(df, "cui")
    0     True
    1    False
    Name: cui, dtype: bool
    """
    return validate_stdnum(df, column, cui.is_valid)
