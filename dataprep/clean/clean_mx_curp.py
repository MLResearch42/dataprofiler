"""
Clean and validate a DataFrame column containing Mexican personal identifiers (CURPs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.mx import curp

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": curp.compact,
    "standard": curp.compact,
    "birthdate": curp.get_birth_date,
    "gender": curp.get_gender,
}


def clean_mx_curp(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Estonian Personcal ID number (CURPs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CURP type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', get the person's birthdate.
        If output_format = 'gender', get the person's birth gender ('M' or 'F').
        Note: in the case of CURP, the compact format is the same as the standard one.

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
    Clean a column of CURP data.

    >>> df = pd.DataFrame({"curp": ["BOXW310820HNERXN09", "BOXW310820HNERXN08"]})
    >>> clean_mx_curp(df, "curp")
                     curp          curp_clean
    0  BOXW310820HNERXN09  BOXW310820HNERXN09
    1  BOXW310820HNERXN08                 NaN
    """
    return clean_stdnum(
        df,
        column,
        curp.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_mx_curp(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CURP in a DataFrame column. For each cell, return True or False.

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
    >>> validate_mx_curp("BOXW310820HNERXN09")
    True
    >>> df = pd.DataFrame({"curp": ["BOXW310820HNERXN09", "BOXW310820HNERXN08"]})
    >>> validate_mx_curp(df["curp"])
    0     True
    1    False
    Name: curp, dtype: bool
    >>> validate_mx_curp(df, "curp")
    0     True
    1    False
    Name: curp, dtype: bool
    """
    return validate_stdnum(df, column, curp.is_valid)
