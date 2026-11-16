"""
Clean and validate a DataFrame column containing European Energy Identification Codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.eu import eic

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": eic.compact,
    "standard": eic.compact,
}


def clean_eu_eic(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean European Energy Identification Codes (EICs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of EIC type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of EIC, the compact format is the same as the standard one.

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
    Clean a column of EIC data.

    >>> df = pd.DataFrame({"eic": ["22XWATTPLUS----G", "22XWATTPLUS----X"]})
    >>> clean_eu_eic(df, "eic")
                    eic         eic_clean
    0  22XWATTPLUS----G  22XWATTPLUS----G
    1  22XWATTPLUS----X               NaN
    """
    return clean_stdnum(
        df,
        column,
        eic.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_eu_eic(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is EIC in a DataFrame column. For each cell, return True or False.

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
    >>> validate_eu_eic("22XWATTPLUS----G")
    True
    >>> df = pd.DataFrame({"eic": ["22XWATTPLUS----G", "22XWATTPLUS----X"]})
    >>> validate_eu_eic(df["eic"])
    0     True
    1    False
    Name: eic, dtype: bool
    >>> validate_eu_eic(df, "eic")
    0     True
    1    False
    Name: eic, dtype: bool
    """
    return validate_stdnum(df, column, eic.is_valid)
