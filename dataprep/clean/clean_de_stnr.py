"""
Clean and validate a DataFrame column containing German tax numbers (STNRs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.de import stnr

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": stnr.compact,
    "standard": stnr.format,
}


def clean_de_stnr(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean German tax numbers (STNRs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of STNR type.
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
    Clean a column of STNR data.

    >>> df = pd.DataFrame({"stnr": ["181/815/0815 5", "136695978"]})
    >>> clean_de_stnr(df, "stnr")
                 stnr     stnr_clean
    0  181/815/0815 5  181/815/08155
    1       136695978            NaN
    """
    return clean_stdnum(
        df,
        column,
        stnr.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_de_stnr(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
    region: str | None = None,
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is STNR in a DataFrame column. For each cell, return True or False.
    The region can be supplied to verify that the number is assigned in that region.

    Parameters
    ----------
    df
        A string, a pandas or Dask Series, or a pandas or Dask DataFrame
        containing the data to be validated.
    column
        The name of the column to be validated. Only used when df is a
        DataFrame; if empty, every cell of the DataFrame is validated.
    region
        Specify the region that the number belongs to.

        (default: None)

    Examples
    --------
    >>> validate_de_stnr("181/815/0815 5")
    True
    >>> df = pd.DataFrame({"stnr": ["181/815/0815 5", "136695978"]})
    >>> validate_de_stnr(df["stnr"])
    0     True
    1    False
    Name: stnr, dtype: bool
    >>> validate_de_stnr(df, "stnr")
    0     True
    1    False
    Name: stnr, dtype: bool
    """
    return validate_stdnum(df, column, lambda val: stnr.is_valid(val, region))
