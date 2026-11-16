"""
Clean and validate a DataFrame column containing
Liechtenstein tax code for individuals and entities (PEIDs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.li import peid

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": peid.compact,
    "standard": peid.compact,
}


def clean_li_peid(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Liechtenstein tax code for individuals and entities data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of PEID type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of PEID, the compact format is the same as the standard one.

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
    Clean a column of PEID data.

    >>> df = pd.DataFrame({"peid": ["00001234567", "00001234568913454545"]})
    >>> clean_li_peid(df, "peid")
                       peid peid_clean
    0           00001234567    1234567
    1  00001234568913454545        NaN
    """
    return clean_stdnum(
        df,
        column,
        peid.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_li_peid(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is PEID in a DataFrame column. For each cell, return True or False.

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
    >>> validate_li_peid("00001234567")
    True
    >>> df = pd.DataFrame({"peid": ["00001234567", "00001234568913454545"]})
    >>> validate_li_peid(df["peid"])
    0     True
    1    False
    Name: peid, dtype: bool
    >>> validate_li_peid(df, "peid")
    0     True
    1    False
    Name: peid, dtype: bool
    """
    return validate_stdnum(df, column, peid.is_valid)
