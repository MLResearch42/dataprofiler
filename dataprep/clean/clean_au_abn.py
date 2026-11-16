"""
Clean and validate a DataFrame column containing Australian Business Numbers (ABNs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.au import abn

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": abn.compact,
    "standard": abn.format,
}


def clean_au_abn(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Australian Business Numbers (ABNs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of ABN type.
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
    Clean a column of ABN data.

    >>> df = pd.DataFrame({"abn": ["51824753556", "99999999999"]})
    >>> clean_au_abn(df, "abn")
               abn       abn_clean
    0  51824753556  51 824 753 556
    1  99999999999             NaN
    """
    return clean_stdnum(
        df,
        column,
        abn.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_au_abn(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is ABN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_au_abn("51824753556")
    True
    >>> df = pd.DataFrame({"abn": ["51824753556", "99999999999"]})
    >>> validate_au_abn(df["abn"])
    0     True
    1    False
    Name: abn, dtype: bool
    >>> validate_au_abn(df, "abn")
    0     True
    1    False
    Name: abn, dtype: bool
    """
    return validate_stdnum(df, column, abn.is_valid)
