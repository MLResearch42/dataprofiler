"""
Clean and validate a DataFrame column containing
French company identification numbers (SIRENs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.fr import siren

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": siren.compact,
    "standard": siren.compact,
    "tva": siren.to_tva,
}


def clean_fr_siren(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean French company identification numbers (SIRENs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of SIREN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'tva', return a TVA that preposes two extra check digits to the data.
        Note: in the case of SIREN, the compact format is the same as the standard one.

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
    Clean a column of SIREN data.

    >>> df = pd.DataFrame({"siren": ["552 008 443", "404833047"]})
    >>> clean_fr_siren(df, "siren")
             siren siren_clean
    0  552 008 443   552008443
    1    404833047         NaN
    """
    return clean_stdnum(
        df,
        column,
        siren.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_fr_siren(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is SIREN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_fr_siren("552 008 443")
    True
    >>> df = pd.DataFrame({"siren": ["552 008 443", "404833047"]})
    >>> validate_fr_siren(df["siren"])
    0     True
    1    False
    Name: siren, dtype: bool
    >>> validate_fr_siren(df, "siren")
    0     True
    1    False
    Name: siren, dtype: bool
    """
    return validate_stdnum(df, column, siren.is_valid)
