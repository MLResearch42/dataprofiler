"""
Clean and validate a DataFrame column containing
VNR, SVNR, VSNR (Versicherungsnummer, Austrian social security number).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.at import vnr

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": vnr.compact,
    "standard": vnr.compact,
}


def clean_at_vnr(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Austrian VNR type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Austrian VNR type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators.
        If output_format = 'standard', return string with proper separators.

        (default: "standard")
    split
        This identifier has no components to split into separate columns. The
        parameter is accepted for consistency with the other clean functions and
        has no effect.

        (default: False)
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
    Clean a column of Austrian VNR data.

    >>> df = pd.DataFrame({"at_vnr": ["1237 010180", "2237 010180"]})
    >>> clean_at_vnr(df, "at_vnr", inplace=True)
      at_vnr_clean
    0   1237010180
    1          NaN
    """
    # pylint: disable=unused-argument
    return clean_stdnum(
        df,
        column,
        vnr.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_at_vnr(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is Austrian VNR in a DataFrame column.
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
    >>> validate_at_vnr("1237 010180")
    True
    >>> df = pd.DataFrame({"at_vnr": ["1237 010180", "2237 010180"]})
    >>> validate_at_vnr(df["at_vnr"])
    0     True
    1    False
    Name: at_vnr, dtype: bool
    >>> validate_at_vnr(df, "at_vnr")
    0     True
    1    False
    Name: at_vnr, dtype: bool
    """
    return validate_stdnum(df, column, vnr.is_valid)
