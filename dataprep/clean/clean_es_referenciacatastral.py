"""
Clean and validate a DataFrame column containing Spanish real state ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.es import referenciacatastral

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": referenciacatastral.compact,
    "standard": referenciacatastral.format,
}


def clean_es_referenciacatastral(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Spanish real state ids (Referencia Catastrals) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Referencia Catastral type.
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
    Clean a column of Referencia Catastral data.

    >>> df = pd.DataFrame(
    ...     {
    ...         "referenciacatastral": [
    ...             "4A08169P03PRAT0001LR",
    ...             "7837301/VG8173B 0001 TT",
    ...         ]
    ...     }
    ... )
    >>> clean_es_referenciacatastral(df, "referenciacatastral")
           referenciacatastral referenciacatastral_clean
    0     4A08169P03PRAT0001LR   4A08169 P03PRAT 0001 LR
    1  7837301/VG8173B 0001 TT                       NaN
    """
    return clean_stdnum(
        df,
        column,
        referenciacatastral.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_es_referenciacatastral(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is Referencia Catastral in a DataFrame column.
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
    >>> validate_es_referenciacatastral("4A08169P03PRAT0001LR")
    True
    >>> df = pd.DataFrame(
    ...     {
    ...         "referenciacatastral": [
    ...             "4A08169P03PRAT0001LR",
    ...             "7837301/VG8173B 0001 TT",
    ...         ]
    ...     }
    ... )
    >>> validate_es_referenciacatastral(df["referenciacatastral"])
    0     True
    1    False
    Name: referenciacatastral, dtype: bool
    >>> validate_es_referenciacatastral(df, "referenciacatastral")
    0     True
    1    False
    Name: referenciacatastral, dtype: bool
    """
    return validate_stdnum(df, column, referenciacatastral.is_valid)
