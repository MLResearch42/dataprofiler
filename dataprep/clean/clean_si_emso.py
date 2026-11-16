"""
Clean and validate a DataFrame column containing Slovenian unique master citizen numbers (EMŠO).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.si import emso

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": emso.compact,
    "standard": emso.format,
    "birthdate": emso.get_birth_date,
    "gender": emso.get_gender,
}


def clean_si_emso(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Slovenian unique master citizen numbers (EMŠO) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of EMŠO type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', return the birth date encoded in the number.
        If output_format = 'gender', return the gender encoded in the number ('M' or 'F').
        Note: in the case of EMŠO, the compact format is the same as the standard one.

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
    Clean a column of EMŠO data.

    >>> df = pd.DataFrame({'emso': ['0101006500006', '0101006500007']})
    >>> clean_si_emso(df, 'emso')
                emso     emso_clean
    0  0101006500006  0101006500006
    1  0101006500007            NaN
    """
    return clean_stdnum(
        df,
        column,
        emso.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_si_emso(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Slovenian unique master citizen numbers (EMŠO). For each cell, return True or False.

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
    >>> validate_si_emso('0101006500006')
    True
    >>> df = pd.DataFrame({'emso': ['0101006500006', '0101006500007']})
    >>> validate_si_emso(df['emso'])
    0     True
    1    False
    Name: emso, dtype: bool
    >>> validate_si_emso(df, 'emso')
    0     True
    1    False
    Name: emso, dtype: bool
    """
    return validate_stdnum(df, column, emso.is_valid)
