"""
Clean and validate a DataFrame column containing Swiss VAT numbers (VATs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.ch import vat

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": vat.compact,
    "standard": vat.format,
}


def clean_ch_vat(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Swiss VAT numbers (VATs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of VAT type.
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
    Clean a column of VAT data.

    >>> df = pd.DataFrame({"vat": ["CHE107787577IVA", "CHE-107.787.578 IVA"]})
    >>> clean_ch_vat(df, "vat")
                       vat            vat_clean
    0      CHE107787577IVA  CHE-107.787.577 IVA
    1  CHE-107.787.578 IVA                  NaN
    """
    return clean_stdnum(
        df,
        column,
        vat.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_ch_vat(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is VAT in a DataFrame column. For each cell, return True or False.

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
    >>> validate_ch_vat("CHE107787577IVA")
    True
    >>> df = pd.DataFrame({"vat": ["CHE107787577IVA", "CHE-107.787.578 IVA"]})
    >>> validate_ch_vat(df["vat"])
    0     True
    1    False
    Name: vat, dtype: bool
    >>> validate_ch_vat(df, "vat")
    0     True
    1    False
    Name: vat, dtype: bool
    """
    return validate_stdnum(df, column, vat.is_valid)
