"""
Clean and validate a DataFrame column containing Chinese Unified Social Credit Code
(China tax number) (USCCs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.cn import uscc

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": uscc.compact,
    "standard": uscc.format,
}


def clean_cn_uscc(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Chinese Unified Social Credit Code (USCCs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of USCC type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note that in the case of USCC, the compact format is the same as the standard one.

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
    Clean a column of USCC data.

    >>> df = pd.DataFrame({"uscc": ["9 1 110000 600037341L", "A1110000600037341L"]})
    >>> clean_cn_uscc(df, "uscc")
                        uscc          uscc_clean
    0  9 1 110000 600037341L  91110000600037341L
    1     A1110000600037341L                 NaN
    """
    return clean_stdnum(
        df,
        column,
        uscc.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_cn_uscc(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is USCC in a DataFrame column. For each cell, return True or False.

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
    >>> validate_cn_uscc("9 1 110000 600037341L")
    True
    >>> df = pd.DataFrame({"uscc": ["9 1 110000 600037341L", "A1110000600037341L"]})
    >>> validate_cn_uscc(df["uscc"])
    0     True
    1    False
    Name: uscc, dtype: bool
    >>> validate_cn_uscc(df, "uscc")
    0     True
    1    False
    Name: uscc, dtype: bool
    """
    return validate_stdnum(df, column, uscc.is_valid)
