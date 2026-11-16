"""
Clean and validate a DataFrame column containing Polish VAT numbers (NIPs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.pl import nip

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": nip.compact,
    "standard": nip.format,
}


def clean_pl_nip(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Polish VAT numbers (NIPs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of NIP type.
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
    Clean a column of NIP data.

    >>> df = pd.DataFrame({"nip": ["PL 8567346215", "PL 8567346216"]})
    >>> clean_pl_nip(df, "nip")
                 nip      nip_clean
    0  PL 8567346215  856-734-62-15
    1  PL 8567346216            NaN
    """
    return clean_stdnum(
        df,
        column,
        nip.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_pl_nip(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is NIP in a DataFrame column. For each cell, return True or False.

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
    >>> validate_pl_nip("PL 8567346215")
    True
    >>> df = pd.DataFrame({"nip": ["PL 8567346215", "PL 8567346216"]})
    >>> validate_pl_nip(df["nip"])
    0     True
    1    False
    Name: nip, dtype: bool
    >>> validate_pl_nip(df, "nip")
    0     True
    1    False
    Name: nip, dtype: bool
    """
    return validate_stdnum(df, column, nip.is_valid)
