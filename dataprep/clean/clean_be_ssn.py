"""
Clean and validate a DataFrame column containing Belgian social security numbers (SSN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.be import ssn

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": ssn.compact,
    "standard": ssn.format,
    "birthdate": ssn.get_birth_date,
    "gender": ssn.get_gender,
}


def clean_be_ssn(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Belgian social security numbers (SSN) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of SSN type.
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
    Clean a column of SSN data.

    >>> df = pd.DataFrame({'ssn': ['85.07.30-033 28', '12345678901']})
    >>> clean_be_ssn(df, 'ssn')
                   ssn        ssn_clean
    0  85.07.30-033 28  85.07.30-033.28
    1      12345678901              NaN
    """
    return clean_stdnum(
        df,
        column,
        ssn.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_be_ssn(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Belgian social security numbers (SSN). For each cell, return True or False.

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
    >>> validate_be_ssn('85.07.30-033 28')
    True
    >>> df = pd.DataFrame({'ssn': ['85.07.30-033 28', '12345678901']})
    >>> validate_be_ssn(df['ssn'])
    0     True
    1    False
    Name: ssn, dtype: bool
    >>> validate_be_ssn(df, 'ssn')
    0     True
    1    False
    Name: ssn, dtype: bool
    """
    return validate_stdnum(df, column, ssn.is_valid)
