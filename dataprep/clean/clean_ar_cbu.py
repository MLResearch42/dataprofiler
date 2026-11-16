"""
Clean and validate a DataFrame column containing
CBU (Clave Bancaria Uniforme, Argentine bank account number).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum.ar import cbu

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    compacted = cbu.compact(val)
    return [compacted[:3], compacted[3:7], compacted[8:21]]


_FORMATTERS = {
    "compact": cbu.compact,
    "standard": cbu.format,
}

_SPLIT_COLUMNS = ("bank_identifier", "branch_identifier", "account_identifier")


def clean_ar_cbu(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Argentine CBU type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Argentine CBU type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators.
        If output_format = 'standard', return string with proper separators.

        (default: "standard")
    split
        If True, each component derived from the number string is put into its
        own column: bank_identifier, branch_identifier, account_identifier.

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
    Clean a column of Argentine CBU Numbers data.

    >>> df = pd.DataFrame({"ar_cbu": ["2850590940090418135201"]})
    >>> clean_ar_cbu(df, "ar_cbu", inplace=True)
                  ar_cbu_clean
    0  28505909 40090418135201
    """
    return clean_stdnum(
        df,
        column,
        cbu.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_ar_cbu(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is Argentine CBU in a DataFrame column.
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
    >>> validate_ar_cbu("2850590940090418135201")
    True
    >>> df = pd.DataFrame({"ar_cbu": ["2850590940090418135201"]})
    >>> validate_ar_cbu(df["ar_cbu"])
    0    True
    Name: ar_cbu, dtype: bool
    >>> validate_ar_cbu(df, "ar_cbu")
    0    True
    Name: ar_cbu, dtype: bool
    """
    return validate_stdnum(df, column, cbu.is_valid)
