"""
Clean and validate a DataFrame column containing
Italian code for identification of drugs (AICs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum.exceptions import ValidationError
from stdnum.it import aic

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _to_base10(val: str) -> Any:
    """Render a valid number in the 'base10' output format."""
    try:
        aic.validate_base10(val)
        return val
    except ValidationError:  # a valid AIC that is not BASE10 can only be BASE32
        return aic.from_base32(val)


def _to_base32(val: str) -> Any:
    """Render a valid number in the 'base32' output format."""
    try:
        aic.validate_base32(val)
        return val
    except ValidationError:  # a valid AIC that is not BASE32 can only be BASE10
        return aic.to_base32(val)


_FORMATTERS = {
    "compact": aic.compact,
    "standard": aic.compact,
    "base10": _to_base10,
    "base32": _to_base32,
}


def clean_it_aic(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Italian code for identification of drugs (AICs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of AIC type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'base10', convert a BASE32 representation to a BASE10 one.
        If output_format = 'base32', convert a BASE10 representation to a BASE32 one.
        Note: in the case of AIC, the compact format is the same as the standard one.
            And 'compact' may contain both BASE10 and BASE32 represatation.

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
    Clean a column of AIC data.

    >>> df = pd.DataFrame({"aic": ["000307052", "999999"]})
    >>> clean_it_aic(df, "aic")
             aic  aic_clean
    0  000307052  000307052
    1     999999        NaN
    """
    return clean_stdnum(
        df,
        column,
        aic.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_it_aic(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is AIC in a DataFrame column. For each cell, return True or False.

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
    >>> validate_it_aic("000307052")
    True
    >>> df = pd.DataFrame({"aic": ["000307052", "999999"]})
    >>> validate_it_aic(df["aic"])
    0     True
    1    False
    Name: aic, dtype: bool
    >>> validate_it_aic(df, "aic")
    0     True
    1    False
    Name: aic, dtype: bool
    """
    return validate_stdnum(df, column, aic.is_valid)
