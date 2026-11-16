"""
Clean and validate a DataFrame column containing Euro banknote serial numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.eu import banknote

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": banknote.compact,
    "standard": banknote.compact,
}


def clean_eu_banknote(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Euro banknote serial numbers type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Euro banknote type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of Euro banknote, the compact format is the same as the standard one.

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
    Clean a column of Euro banknote data.

    >>> df = pd.DataFrame({"banknote": ["P36007033744", "P36007033743"]})
    >>> clean_eu_banknote(df, "banknote")
           banknote banknote_clean
    0  P36007033744   P36007033744
    1  P36007033743            NaN
    """
    return clean_stdnum(
        df,
        column,
        banknote.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_eu_banknote(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is Euro banknote in a DataFrame column.
    For each cell, return True or False.

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
    >>> validate_eu_banknote("P36007033744")
    True
    >>> df = pd.DataFrame({"banknote": ["P36007033744", "P36007033743"]})
    >>> validate_eu_banknote(df["banknote"])
    0     True
    1    False
    Name: banknote, dtype: bool
    >>> validate_eu_banknote(df, "banknote")
    0     True
    1    False
    Name: banknote, dtype: bool
    """
    return validate_stdnum(df, column, banknote.is_valid)
