"""
Clean and validate a DataFrame column containing Monacan TVA numbers (TVAs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.mc import tva

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": tva.compact,
    "standard": tva.compact,
}


def clean_mc_tva(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Monacan TVA numbers (TVAs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of TVA type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of TVA, the compact format is the same as the standard one.

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
    Clean a column of TVA data.

    >>> df = pd.DataFrame({"tva": ["53 0000 04605", "FR 61 954 506 077"]})
    >>> clean_mc_tva(df, "tva")
                     tva      tva_clean
    0      53 0000 04605  FR53000004605
    1  FR 61 954 506 077            NaN
    """
    return clean_stdnum(
        df,
        column,
        tva.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_mc_tva(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is TVA in a DataFrame column. For each cell, return True or False.

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
    >>> validate_mc_tva("53 0000 04605")
    True
    >>> df = pd.DataFrame({"tva": ["53 0000 04605", "FR 61 954 506 077"]})
    >>> validate_mc_tva(df["tva"])
    0     True
    1    False
    Name: tva, dtype: bool
    >>> validate_mc_tva(df, "tva")
    0     True
    1    False
    Name: tva, dtype: bool
    """
    return validate_stdnum(df, column, tva.is_valid)
