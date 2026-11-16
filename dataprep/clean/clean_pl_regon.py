"""
Clean and validate a DataFrame column containing Polish register of economic units (REGONs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.pl import regon

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": regon.compact,
    "standard": regon.compact,
}


def clean_pl_regon(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Polish register of economic units (REGONs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of REGON type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of REGON, the compact format is the same as the standard one.

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
    Clean a column of REGON data.

    >>> df = pd.DataFrame({"regon": ["192598184", "192598183"]})
    >>> clean_pl_regon(df, "regon")
           regon regon_clean
    0  192598184   192598184
    1  192598183         NaN
    """
    return clean_stdnum(
        df,
        column,
        regon.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_pl_regon(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is REGON in a DataFrame column. For each cell, return True or False.

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
    >>> validate_pl_regon("192598184")
    True
    >>> df = pd.DataFrame({"regon": ["192598184", "192598183"]})
    >>> validate_pl_regon(df["regon"])
    0     True
    1    False
    Name: regon, dtype: bool
    >>> validate_pl_regon(df, "regon")
    0     True
    1    False
    Name: regon, dtype: bool
    """
    return validate_stdnum(df, column, regon.is_valid)
