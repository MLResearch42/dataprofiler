"""
Clean and validate a DataFrame column containing Greek social security numbers (AMKAs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.gr import amka

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": amka.compact,
    "standard": amka.compact,
    "birthdate": amka.get_birth_date,
    "gender": amka.get_gender,
}


def clean_gr_amka(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Greek social security numbers (AMKAs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of AMKA type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', get the person's birthdate.
        If output_format = 'gender', get the person's birth gender ('M' or 'F').
        Note: in the case of AMKA, the compact format is the same as the standard one.

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
    Clean a column of AMKA data.

    >>> df = pd.DataFrame({"amka": ["01013099997", "01013099999"]})
    >>> clean_gr_amka(df, "amka")
              amka   amka_clean
    0  01013099997  01013099997
    1  01013099999          NaN
    """
    return clean_stdnum(
        df,
        column,
        amka.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_gr_amka(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is AMKA in a DataFrame column. For each cell, return True or False.

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
    >>> validate_gr_amka("01013099997")
    True
    >>> df = pd.DataFrame({"amka": ["01013099997", "01013099999"]})
    >>> validate_gr_amka(df["amka"])
    0     True
    1    False
    Name: amka, dtype: bool
    >>> validate_gr_amka(df, "amka")
    0     True
    1    False
    Name: amka, dtype: bool
    """
    return validate_stdnum(df, column, amka.is_valid)
