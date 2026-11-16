"""
Clean and validate a DataFrame column containing
International Mobile Equipment Identity (IMEI) numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from stdnum import imei

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    compacted = imei.compact(val)
    check_digit = compacted[14:] if len(compacted) > 14 else np.nan
    return [compacted[:8], compacted[8:14], check_digit]


_FORMATTERS = {
    "compact": imei.compact,
    "standard": imei.format,
}

_SPLIT_COLUMNS = ("tac_code", "serial_number", "check_digits")


def clean_imei(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean IMEI type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of ISBN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators.
        If output_format = 'standard', return string with proper separators.

        (default: "standard")
    split
        If True, each component derived from the number string is put into its
        own column: tac_code, serial_number, check_digits.

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
    Clean a column of IMEI data.

    >>> df = pd.DataFrame({"imei": ["35686800-004141-20", "354178036859789"]})
    >>> clean_imei(df, "imei", inplace=True)
                imei_clean
    0  35-686800-004141-20
    1   35-417803-685978-9
    """
    return clean_stdnum(
        df,
        column,
        imei.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_imei(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is IMEI in a DataFrame column. For each cell, return True or False.

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
    >>> validate_imei("35686800-004141-20")
    True
    >>> df = pd.DataFrame({"imei": ["35686800-004141-20", "354178036859789"]})
    >>> validate_imei(df["imei"])
    0    True
    1    True
    Name: imei, dtype: bool
    >>> validate_imei(df, "imei")
    0    True
    1    True
    Name: imei, dtype: bool
    """
    return validate_stdnum(df, column, imei.is_valid)
