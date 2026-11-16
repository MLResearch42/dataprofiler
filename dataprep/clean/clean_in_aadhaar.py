"""
Clean and validate a DataFrame column containing
Indian digital resident personal identity numbers (Aadhaars).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.in_ import aadhaar

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": aadhaar.compact,
    "standard": aadhaar.format,
    "mask": aadhaar.mask,
}


def clean_in_aadhaar(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Indian digital resident personal identity number (Aadhaars) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Aadhaar type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'mask', mask the first 8 digits as per MeitY guidelines for
            securing identity information and Sensitive personal data.

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
    Clean a column of Aadhaar data.

    >>> df = pd.DataFrame({"aadhaar": ["234123412346", "643343121"]})
    >>> clean_in_aadhaar(df, "aadhaar")
            aadhaar   aadhaar_clean
    0  234123412346  2341 2341 2346
    1     643343121             NaN
    """
    return clean_stdnum(
        df,
        column,
        aadhaar.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_in_aadhaar(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is Aadhaar in a DataFrame column. For each cell, return True or False.

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
    >>> validate_in_aadhaar("234123412346")
    True
    >>> df = pd.DataFrame({"aadhaar": ["234123412346", "643343121"]})
    >>> validate_in_aadhaar(df["aadhaar"])
    0     True
    1    False
    Name: aadhaar, dtype: bool
    >>> validate_in_aadhaar(df, "aadhaar")
    0     True
    1    False
    Name: aadhaar, dtype: bool
    """
    return validate_stdnum(df, column, aadhaar.is_valid)
