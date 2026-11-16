"""
Clean and validate a DataFrame column containing AT-02 (SEPA Creditor identifier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.eu import at_02

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": at_02.compact,
    "standard": at_02.compact,
}


def clean_eu_at_02(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean AT-02 (SEPA Creditor identifier) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of AT-02 type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of AT-02, the compact format is the same as the standard one.

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
    Clean a column of AT-02 data.

    >>> df = pd.DataFrame({"at_02": ["ES++()+23ZZZ4//7690558N", "ES2900047690558N"]})
    >>> clean_eu_at_02(df, "at_02")
                         at_02       at_02_clean
    0  ES++()+23ZZZ4//7690558N  ES23ZZZ47690558N
    1         ES2900047690558N               NaN
    """
    return clean_stdnum(
        df,
        column,
        at_02.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_eu_at_02(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is AT-02 in a DataFrame column. For each cell, return True or False.

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
    >>> validate_eu_at_02("ES++()+23ZZZ4//7690558N")
    True
    >>> df = pd.DataFrame({"at_02": ["ES++()+23ZZZ4//7690558N", "ES2900047690558N"]})
    >>> validate_eu_at_02(df["at_02"])
    0     True
    1    False
    Name: at_02, dtype: bool
    >>> validate_eu_at_02(df, "at_02")
    0     True
    1    False
    Name: at_02, dtype: bool
    """
    return validate_stdnum(df, column, at_02.is_valid)
