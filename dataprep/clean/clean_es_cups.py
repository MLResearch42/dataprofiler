"""
Clean and validate a DataFrame column containing Spanish meter point numbers (CUPSs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.es import cups

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cups.compact,
    "standard": cups.format,
}


def clean_es_cups(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Spanish meter point numbers (CUPSs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CUPS type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.

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
    Clean a column of CUPS data.

    >>> df = pd.DataFrame({"cups": ["ES1234123456789012JY1F", "ES 1234-123456789012-XY 1F"]})
    >>> clean_es_cups(df, "cups")
                             cups                    cups_clean
    0      ES1234123456789012JY1F  ES 1234 1234 5678 9012 JY 1F
    1  ES 1234-123456789012-XY 1F                           NaN
    """
    return clean_stdnum(
        df,
        column,
        cups.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_es_cups(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CUPS in a DataFrame column. For each cell, return True or False.

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
    >>> validate_es_cups("ES1234123456789012JY1F")
    True
    >>> df = pd.DataFrame({"cups": ["ES1234123456789012JY1F", "ES 1234-123456789012-XY 1F"]})
    >>> validate_es_cups(df["cups"])
    0     True
    1    False
    Name: cups, dtype: bool
    >>> validate_es_cups(df, "cups")
    0     True
    1    False
    Name: cups, dtype: bool
    """
    return validate_stdnum(df, column, cups.is_valid)
