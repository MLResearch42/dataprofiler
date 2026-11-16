"""
Clean and validate a DataFrame column containing
French company establishment identification numbers (SIRETs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.fr import siret

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": siret.compact,
    "standard": siret.format,
    "siren": siret.to_siren,
    "tva": siret.to_tva,
}


def clean_fr_siret(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean French Company Establishment Identification Numbers (SIRETs) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of SIRET type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'siren', convert the SIRET number to a SIREN number.
        If output_format = 'tva', convert the SIRET number to a TVA number.

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
    Clean a column of SIRET data.

    >>> df = pd.DataFrame({"siret": ["73282932000074", "73282932000079"]})
    >>> clean_fr_siret(df, "siret")
                siret        siret_clean
    0  73282932000074  732 829 320 00074
    1  73282932000079                NaN
    """
    return clean_stdnum(
        df,
        column,
        siret.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_fr_siret(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is SIRET in a DataFrame column. For each cell, return True or False.

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
    >>> validate_fr_siret("73282932000074")
    True
    >>> df = pd.DataFrame({"siret": ["73282932000074", "73282932000079"]})
    >>> validate_fr_siret(df["siret"])
    0     True
    1    False
    Name: siret, dtype: bool
    >>> validate_fr_siret(df, "siret")
    0     True
    1    False
    Name: siret, dtype: bool
    """
    return validate_stdnum(df, column, siret.is_valid)
