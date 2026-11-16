"""
Clean and validate a DataFrame column containing South African identity document numbers (ID number).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.za import idnr

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": idnr.compact,
    "standard": idnr.format,
    "birthdate": idnr.get_birth_date,
    "gender": idnr.get_gender,
}


def clean_za_idnr(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean South African identity document numbers (ID number) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of ID number type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', return the birth date encoded in the number.
        If output_format = 'gender', return the gender encoded in the number ('M' or 'F').

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
    Clean a column of ID number data.

    >>> df = pd.DataFrame({'idnr': ['7503305044089', '8503305044089']})
    >>> clean_za_idnr(df, 'idnr')
                idnr        idnr_clean
    0  7503305044089  750330 5044 08 9
    1  8503305044089               NaN
    """
    return clean_stdnum(
        df,
        column,
        idnr.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_za_idnr(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate South African identity document numbers (ID number). For each cell, return True or False.

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
    >>> validate_za_idnr('7503305044089')
    True
    >>> df = pd.DataFrame({'idnr': ['7503305044089', '8503305044089']})
    >>> validate_za_idnr(df['idnr'])
    0     True
    1    False
    Name: idnr, dtype: bool
    >>> validate_za_idnr(df, 'idnr')
    0     True
    1    False
    Name: idnr, dtype: bool
    """
    return validate_stdnum(df, column, idnr.is_valid)
