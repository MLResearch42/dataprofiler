"""
Clean and validate a DataFrame column containing Danish citizen number (CPRs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.dk import cpr

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cpr.compact,
    "standard": cpr.format,
    "birthdate": cpr.get_birth_date,
}


def clean_dk_cpr(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Danish citizen number (CPRs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CPR type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', split the number and return the birth date.

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
    Clean a column of CPR data.

    >>> df = pd.DataFrame({"cpr": ["2110625629", "511062-5629"]})
    >>> clean_dk_cpr(df, "cpr")
               cpr    cpr_clean
    0   2110625629  211062-5629
    1  511062-5629          NaN
    """
    return clean_stdnum(
        df,
        column,
        cpr.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_dk_cpr(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CPR in a DataFrame column. For each cell, return True or False.

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
    >>> validate_dk_cpr("2110625629")
    True
    >>> df = pd.DataFrame({"cpr": ["2110625629", "511062-5629"]})
    >>> validate_dk_cpr(df["cpr"])
    0     True
    1    False
    Name: cpr, dtype: bool
    >>> validate_dk_cpr(df, "cpr")
    0     True
    1    False
    Name: cpr, dtype: bool
    """
    return validate_stdnum(df, column, cpr.is_valid)
