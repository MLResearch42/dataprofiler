"""
Clean and validate a DataFrame column containing Spanish fiscal numbers (CIFs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from stdnum.es import cif

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _split(val: str) -> list[Any]:
    """Return the components of a valid number, in the order of _SPLIT_COLUMNS."""
    return list(cif.split(val))


_FORMATTERS = {
    "compact": cif.compact,
    "standard": cif.compact,
}

_SPLIT_COLUMNS = ("type_code", "province_code", "within_code", "check_digit")


def clean_es_cif(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Spanish fiscal numbers (CIFs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CIF type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of CIF, the compact format is the same as the standard one.

        (default: "standard")
    split
        If True, each component derived from the number string is put into its
        own column: type_code, province_code, within_code, check_digit.

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
    Clean a column of CIF data.

    >>> df = pd.DataFrame({"cif": ["A13 585 625", "M-1234567-L"]})
    >>> clean_es_cif(df, "cif")
               cif  cif_clean
    0  A13 585 625  A13585625
    1  M-1234567-L        NaN
    """
    return clean_stdnum(
        df,
        column,
        cif.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
        split=split,
        split_columns=_SPLIT_COLUMNS,
        splitter=_split,
    )


def validate_es_cif(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CIF in a DataFrame column. For each cell, return True or False.

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
    >>> validate_es_cif("A13 585 625")
    True
    >>> df = pd.DataFrame({"cif": ["A13 585 625", "M-1234567-L"]})
    >>> validate_es_cif(df["cif"])
    0     True
    1    False
    Name: cif, dtype: bool
    >>> validate_es_cif(df, "cif")
    0     True
    1    False
    Name: cif, dtype: bool
    """
    return validate_stdnum(df, column, cif.is_valid)
