"""
Clean and validate a DataFrame column containing Indian Permanent Account numbers (PANs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.in_ import pan

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": pan.compact,
    "standard": pan.compact,
    "info": pan.info,
    "mask": pan.mask,
}


def clean_in_pan(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Indian Permanent Account numbers (PANs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of PAN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'info', return a dictionary containing information
            that can be decoded from the PAN.
        If output_format = 'mask', mask the PAN as per CBDT masking standard.
        Note: in the case of PAN, the compact format is the same as the standard one.

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
    Clean a column of PAN data.

    >>> df = pd.DataFrame({"pan": ["ACUPA7085R", "234123412347"]})
    >>> clean_in_pan(df, "pan")
                pan   pan_clean
    0    ACUPA7085R  ACUPA7085R
    1  234123412347         NaN
    """
    return clean_stdnum(
        df,
        column,
        pan.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_in_pan(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is PAN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_in_pan("ACUPA7085R")
    True
    >>> df = pd.DataFrame({"pan": ["ACUPA7085R", "234123412347"]})
    >>> validate_in_pan(df["pan"])
    0     True
    1    False
    Name: pan, dtype: bool
    >>> validate_in_pan(df, "pan")
    0     True
    1    False
    Name: pan, dtype: bool
    """
    return validate_stdnum(df, column, pan.is_valid)
