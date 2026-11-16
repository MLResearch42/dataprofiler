"""
Clean and validate a DataFrame column containing Bulgarian personal number of a foreigner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.bg import pnf

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": pnf.compact,
    "standard": pnf.compact,
}


def clean_bg_pnf(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Bulgarian personal number of a foreigner type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of PNF type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of PNF, the compact format is the same as the standard one.

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
    Clean a column of PNF data.

    >>> df = pd.DataFrame({"pnf": ["7111 042 925", "7111042922"]})
    >>> clean_bg_pnf(df, "pnf")
                pnf   pnf_clean
    0  7111 042 925  7111042925
    1    7111042922         NaN
    """
    return clean_stdnum(
        df,
        column,
        pnf.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_bg_pnf(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is PNF in a DataFrame column. For each cell, return True or False.

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
    >>> validate_bg_pnf("7111 042 925")
    True
    >>> df = pd.DataFrame({"pnf": ["7111 042 925", "7111042922"]})
    >>> validate_bg_pnf(df["pnf"])
    0     True
    1    False
    Name: pnf, dtype: bool
    >>> validate_bg_pnf(df, "pnf")
    0     True
    1    False
    Name: pnf, dtype: bool
    """
    return validate_stdnum(df, column, pnf.is_valid)
