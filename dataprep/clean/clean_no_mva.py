"""
Clean and validate a DataFrame column containing Norwegian VAT numbers (MVAs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.no import mva

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": mva.compact,
    "standard": mva.format,
}


def clean_no_mva(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Norwegian VAT numbers (MVAs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of MVA type.
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
    Clean a column of MVA data.

    >>> df = pd.DataFrame({"mva": ["995525828MVA", "NO 995 525 829 MVA"]})
    >>> clean_no_mva(df, "mva")
                      mva           mva_clean
    0        995525828MVA  NO 995 525 828 MVA
    1  NO 995 525 829 MVA                 NaN
    """
    return clean_stdnum(
        df,
        column,
        mva.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_no_mva(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is MVA in a DataFrame column. For each cell, return True or False.

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
    >>> validate_no_mva("995525828MVA")
    True
    >>> df = pd.DataFrame({"mva": ["995525828MVA", "NO 995 525 829 MVA"]})
    >>> validate_no_mva(df["mva"])
    0     True
    1    False
    Name: mva, dtype: bool
    >>> validate_no_mva(df, "mva")
    0     True
    1    False
    Name: mva, dtype: bool
    """
    return validate_stdnum(df, column, mva.is_valid)
