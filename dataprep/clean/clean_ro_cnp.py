"""
Clean and validate a DataFrame column containing Romanian Numerical Personal Codes (CNPs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.ro import cnp

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cnp.compact,
    "standard": cnp.compact,
    "birthdate": cnp.get_birth_date,
}


def clean_ro_cnp(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Estonian Personcal ID number (CNPs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CNP type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', get the person's birthdate.
        Note: in the case of CNP, the compact format is the same as the standard one.

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
    Clean a column of CNP data.

    >>> df = pd.DataFrame({"cnp": ["1630615123457", "0800101221142"]})
    >>> clean_ro_cnp(df, "cnp")
                 cnp      cnp_clean
    0  1630615123457  1630615123457
    1  0800101221142            NaN
    """
    return clean_stdnum(
        df,
        column,
        cnp.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_ro_cnp(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CNP in a DataFrame column. For each cell, return True or False.

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
    >>> validate_ro_cnp("1630615123457")
    True
    >>> df = pd.DataFrame({"cnp": ["1630615123457", "0800101221142"]})
    >>> validate_ro_cnp(df["cnp"])
    0     True
    1    False
    Name: cnp, dtype: bool
    >>> validate_ro_cnp(df, "cnp")
    0     True
    1    False
    Name: cnp, dtype: bool
    """
    return validate_stdnum(df, column, cnp.is_valid)
