"""
Clean and validate a DataFrame column containing
Malaysian National Registration Identity Card Numbers (NRICs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.my import nric

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": nric.compact,
    "standard": nric.format,
    "birthdate": nric.get_birth_date,
    "birthplace": nric.get_birth_place,
}


def clean_my_nric(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Malaysian National Registration Identity Card Numbers (NRICs) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of NRIC type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', return the registration date or the birth date.
        If output_format = 'birthplace', return a dict containing the birthplace of the person.

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
    Clean a column of NRIC data.

    >>> df = pd.DataFrame({"nric": ["770305021234", "771305-02-1234"]})
    >>> clean_my_nric(df, "nric")
                 nric      nric_clean
    0    770305021234  770305-02-1234
    1  771305-02-1234             NaN
    """
    return clean_stdnum(
        df,
        column,
        nric.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_my_nric(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is NRIC in a DataFrame column. For each cell, return True or False.

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
    >>> validate_my_nric("770305021234")
    True
    >>> df = pd.DataFrame({"nric": ["770305021234", "771305-02-1234"]})
    >>> validate_my_nric(df["nric"])
    0     True
    1    False
    Name: nric, dtype: bool
    >>> validate_my_nric(df, "nric")
    0     True
    1    False
    Name: nric, dtype: bool
    """
    return validate_stdnum(df, column, nric.is_valid)
