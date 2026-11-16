"""
Clean and validate a DataFrame column containing
German Securities Identification Codes (WKNs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.de import wkn

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": wkn.compact,
    "standard": wkn.compact,
    "isin": wkn.to_isin,
}


def clean_de_wkn(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Wertpapierkennnummer (WKNs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of WKN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'isin', convert the number to an ISIN.
        Note: in the case of WKN, the compact format is the same as the standard one.

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
    Clean a column of WKN data.

    >>> df = pd.DataFrame({"wkn": ["A0MNRK", "AOMNRK"]})
    >>> clean_de_wkn(df, "wkn")
          wkn wkn_clean
    0  A0MNRK    A0MNRK
    1  AOMNRK       NaN
    """
    return clean_stdnum(
        df,
        column,
        wkn.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_de_wkn(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is WKN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_de_wkn("A0MNRK")
    True
    >>> df = pd.DataFrame({"wkn": ["A0MNRK", "AOMNRK"]})
    >>> validate_de_wkn(df["wkn"])
    0     True
    1    False
    Name: wkn, dtype: bool
    >>> validate_de_wkn(df, "wkn")
    0     True
    1    False
    Name: wkn, dtype: bool
    """
    return validate_stdnum(df, column, wkn.is_valid)
