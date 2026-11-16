"""
Clean and validate a DataFrame column containing Dutch BTW numbers (BTWs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.nl import btw

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": btw.compact,
    "standard": btw.compact,
}


def clean_nl_btw(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Dutch BTW numbers (BTWs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of BTW type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of BTW, the compact format is the same as the standard one.

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
    Clean a column of BTW data.

    >>> df = pd.DataFrame({"btw": ["004495445B01", "123456789B90"]})
    >>> clean_nl_btw(df, "btw")
                btw     btw_clean
    0  004495445B01  004495445B01
    1  123456789B90           NaN
    """
    return clean_stdnum(
        df,
        column,
        btw.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_nl_btw(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is BTW in a DataFrame column. For each cell, return True or False.

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
    >>> validate_nl_btw("004495445B01")
    True
    >>> df = pd.DataFrame({"btw": ["004495445B01", "123456789B90"]})
    >>> validate_nl_btw(df["btw"])
    0     True
    1    False
    Name: btw, dtype: bool
    >>> validate_nl_btw(df, "btw")
    0     True
    1    False
    Name: btw, dtype: bool
    """
    return validate_stdnum(df, column, btw.is_valid)
