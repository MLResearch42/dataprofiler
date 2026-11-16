"""
Clean and validate a DataFrame column containing
United Kingdom National Health Service patient identifier (NHSs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.gb import nhs

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": nhs.compact,
    "standard": nhs.format,
}


def clean_gb_nhs(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean United Kingdom NHS numbers (NHSs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of NHS type.
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
    Clean a column of NHS data.

    >>> df = pd.DataFrame({"nhs": ["9434765870", "9434765871"]})
    >>> clean_gb_nhs(df, "nhs")
              nhs     nhs_clean
    0  9434765870  943 476 5870
    1  9434765871           NaN
    """
    return clean_stdnum(
        df,
        column,
        nhs.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_gb_nhs(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is NHS in a DataFrame column. For each cell, return True or False.

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
    >>> validate_gb_nhs("9434765870")
    True
    >>> df = pd.DataFrame({"nhs": ["9434765870", "9434765871"]})
    >>> validate_gb_nhs(df["nhs"])
    0     True
    1    False
    Name: nhs, dtype: bool
    >>> validate_gb_nhs(df, "nhs")
    0     True
    1    False
    Name: nhs, dtype: bool
    """
    return validate_stdnum(df, column, nhs.is_valid)
