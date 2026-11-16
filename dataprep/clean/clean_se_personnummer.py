"""
Clean and validate a DataFrame column containing Swedish personal identity numbers (Personnummer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.se import personnummer

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": personnummer.compact,
    "standard": personnummer.format,
    "birthdate": personnummer.get_birth_date,
    "gender": personnummer.get_gender,
}


def clean_se_personnummer(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Swedish personal identity numbers (Personnummer) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Personnummer type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', return the birth date encoded in the number.
        If output_format = 'gender', return the gender encoded in the number ('M' or 'F').
        Note: in the case of Personnummer, the compact format is the same as the standard one.

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
    Clean a column of Personnummer data.

    >>> df = pd.DataFrame({'personnummer': ['880320-0016', '880320-0018']})
    >>> clean_se_personnummer(df, 'personnummer')
      personnummer personnummer_clean
    0  880320-0016        880320-0016
    1  880320-0018                NaN
    """
    return clean_stdnum(
        df,
        column,
        personnummer.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_se_personnummer(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Swedish personal identity numbers (Personnummer). For each cell, return True or False.

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
    >>> validate_se_personnummer('880320-0016')
    True
    >>> df = pd.DataFrame({'personnummer': ['880320-0016', '880320-0018']})
    >>> validate_se_personnummer(df['personnummer'])
    0     True
    1    False
    Name: personnummer, dtype: bool
    >>> validate_se_personnummer(df, 'personnummer')
    0     True
    1    False
    Name: personnummer, dtype: bool
    """
    return validate_stdnum(df, column, personnummer.is_valid)
