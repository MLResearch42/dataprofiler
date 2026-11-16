"""
Clean and validate a DataFrame column containing Swiss EinzahlungsSchein mit Referenznummer (ESRs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.ch import esr

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": esr.compact,
    "standard": esr.format,
}


def clean_ch_esr(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Swiss EinzahlungsSchein mit Referenznummer (ESRs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of ESR type.
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
    Clean a column of ESR data.

    >>> df = pd.DataFrame({"esr": ["18 78583", "210000000003139471430009016"]})
    >>> clean_ch_esr(df, "esr")
                               esr                         esr_clean
    0                     18 78583  00 00000 00000 00000 00018 78583
    1  210000000003139471430009016                               NaN
    """
    return clean_stdnum(
        df,
        column,
        esr.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_ch_esr(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is ESR in a DataFrame column. For each cell, return True or False.

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
    >>> validate_ch_esr("18 78583")
    True
    >>> df = pd.DataFrame({"esr": ["18 78583", "210000000003139471430009016"]})
    >>> validate_ch_esr(df["esr"])
    0     True
    1    False
    Name: esr, dtype: bool
    >>> validate_ch_esr(df, "esr")
    0     True
    1    False
    Name: esr, dtype: bool
    """
    return validate_stdnum(df, column, esr.is_valid)
