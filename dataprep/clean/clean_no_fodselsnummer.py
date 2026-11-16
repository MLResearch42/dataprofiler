"""
Clean and validate a DataFrame column containing Norwegian birth numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.no import fodselsnummer

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": fodselsnummer.compact,
    "standard": fodselsnummer.format,
    "birthdate": fodselsnummer.get_birth_date,
    "gender": fodselsnummer.get_gender,
}


def clean_no_fodselsnummer(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Norwegian birth number data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of fodselsnummer type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', get the person's birthdate.
        If output_format = 'gender', get the person's birth gender ('M' or 'F').

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
    Clean a column of fodselsnummer data.

    >>> df = pd.DataFrame({"fodselsnummer": ["15108695088", "15108695077"]})
    >>> clean_no_fodselsnummer(df, "fodselsnummer")
      fodselsnummer fodselsnummer_clean
    0   15108695088        151086 95088
    1   15108695077                 NaN
    """
    return clean_stdnum(
        df,
        column,
        fodselsnummer.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_no_fodselsnummer(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is fodselsnummer in a DataFrame column.
    For each cell, return True or False.

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
    >>> validate_no_fodselsnummer("15108695088")
    True
    >>> df = pd.DataFrame({"fodselsnummer": ["15108695088", "15108695077"]})
    >>> validate_no_fodselsnummer(df["fodselsnummer"])
    0     True
    1    False
    Name: fodselsnummer, dtype: bool
    >>> validate_no_fodselsnummer(df, "fodselsnummer")
    0     True
    1    False
    Name: fodselsnummer, dtype: bool
    """
    return validate_stdnum(df, column, fodselsnummer.is_valid)
