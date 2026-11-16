"""
Clean and validate a DataFrame column containing Cuban identity card numbers (NIs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.cu import ni

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": ni.compact,
    "standard": ni.compact,
    "birthdate": ni.get_birth_date,
    "gender": ni.get_gender,
}


def clean_cu_ni(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Cuban identity card numbers (NIs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of NI type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', return the date of birth.
        If output_format = 'gender', return the gender (M/F) from the person's NI.
        Note: in the case of NI, the compact format is the same as the standard one.

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
    Clean a column of NI data.

    >>> df = pd.DataFrame({"ni": ["91021027775", "9102102777A"]})
    >>> clean_cu_ni(df, "ni")
                ni     ni_clean
    0  91021027775  91021027775
    1  9102102777A          NaN
    """
    return clean_stdnum(
        df,
        column,
        ni.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_cu_ni(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is NI in a DataFrame column. For each cell, return True or False.

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
    >>> validate_cu_ni("91021027775")
    True
    >>> df = pd.DataFrame({"ni": ["91021027775", "9102102777A"]})
    >>> validate_cu_ni(df["ni"])
    0     True
    1    False
    Name: ni, dtype: bool
    >>> validate_cu_ni(df, "ni")
    0     True
    1    False
    Name: ni, dtype: bool
    """
    return validate_stdnum(df, column, ni.is_valid)
