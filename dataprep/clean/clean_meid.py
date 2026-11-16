"""
Clean and validate a DataFrame column containing
Mobile Equipment Identifiers (MEIDs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum import meid

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _to_standard(val: str) -> Any:
    """Render a valid number in the 'standard' output format."""
    return meid.format(val, add_check_digit=True)


def _to_hex(val: str) -> Any:
    """Render a valid number in the 'hex' output format."""
    return meid.format(val, format="hex", add_check_digit=True)


def _to_dec(val: str) -> Any:
    """Render a valid number in the 'dec' output format."""
    return meid.format(val, format="dec", add_check_digit=True)


_FORMATTERS = {
    "compact": meid.compact,
    "standard": _to_standard,
    "hex": _to_hex,
    "dec": _to_dec,
    "binary": meid.to_binary,
    "pseudo_esn": meid.to_pseudo_esn,
}


def clean_meid(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean MEID type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of MEID type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators.
        If output_format = 'standard', return string with proper separators.
        If output_format = 'hex', return MEID string with hex format.
        If output_format = 'dec', return MEID string with dec format.
        If output_format = 'binary', return MEID string with binary format.
        If output_format = 'pseudo_esn', return MEID string with pseudo_esn format.

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
    Clean a column of MEID data.

    >>> df = pd.DataFrame({"meid": ["af0123450abcDEC"]})
    >>> clean_meid(df, "meid", inplace=True)
                   meid_clean
    0  AF 01 23 45 0A BC DE C
    """
    # pylint: disable=unused-argument
    return clean_stdnum(
        df,
        column,
        meid.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_meid(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is ISBN in a DataFrame column.
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
    >>> validate_meid("af0123450abcDEC")
    True
    >>> df = pd.DataFrame({"meid": ["af0123450abcDEC"]})
    >>> validate_meid(df["meid"])
    0    True
    Name: meid, dtype: bool
    >>> validate_meid(df, "meid")
    0    True
    Name: meid, dtype: bool
    """
    return validate_stdnum(df, column, meid.is_valid)
