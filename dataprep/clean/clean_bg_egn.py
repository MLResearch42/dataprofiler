"""
Clean and validate a DataFrame column containing Bulgarian national identification numbers (EGNs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.bg import egn

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": egn.compact,
    "standard": egn.compact,
    "birthdate": egn.get_birth_date,
}


def clean_bg_egn(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Bulgarian national identification numbers (EGNs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of EGN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', return the birth date attained from the number.
        Note: in the case of EGN, the compact format is the same as the standard one.

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
    Clean a column of EGN data.

    >>> df = pd.DataFrame({"egn": ["752316 926 3", "7552A10004"]})
    >>> clean_bg_egn(df, "egn")
                egn   egn_clean
    0  752316 926 3  7523169263
    1    7552A10004         NaN
    """
    return clean_stdnum(
        df,
        column,
        egn.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_bg_egn(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is EGN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_bg_egn("752316 926 3")
    True
    >>> df = pd.DataFrame({"egn": ["752316 926 3", "7552A10004"]})
    >>> validate_bg_egn(df["egn"])
    0     True
    1    False
    Name: egn, dtype: bool
    >>> validate_bg_egn(df, "egn")
    0     True
    1    False
    Name: egn, dtype: bool
    """
    return validate_stdnum(df, column, egn.is_valid)
