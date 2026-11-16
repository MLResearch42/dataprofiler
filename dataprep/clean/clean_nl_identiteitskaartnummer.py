"""
Clean and validate a DataFrame column containing Dutch passport and identity card numbers (document number).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.nl import identiteitskaartnummer

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": identiteitskaartnummer.compact,
    "standard": identiteitskaartnummer.compact,
}


def clean_nl_identiteitskaartnummer(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Dutch passport and identity card numbers (document number) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of document number type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        Note: in the case of document number, the compact format is the same as the standard one.

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
    Clean a column of document number data.

    >>> df = pd.DataFrame({'identiteitskaartnummer': ['EM0000000', 'XR1001R']})
    >>> clean_nl_identiteitskaartnummer(df, 'identiteitskaartnummer')
      identiteitskaartnummer identiteitskaartnummer_clean
    0              EM0000000                    EM0000000
    1                XR1001R                          NaN
    """
    return clean_stdnum(
        df,
        column,
        identiteitskaartnummer.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_nl_identiteitskaartnummer(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Dutch passport and identity card numbers (document number). For each cell, return True or False.

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
    >>> validate_nl_identiteitskaartnummer('EM0000000')
    True
    >>> df = pd.DataFrame({'identiteitskaartnummer': ['EM0000000', 'XR1001R']})
    >>> validate_nl_identiteitskaartnummer(df['identiteitskaartnummer'])
    0     True
    1    False
    Name: identiteitskaartnummer, dtype: bool
    >>> validate_nl_identiteitskaartnummer(df, 'identiteitskaartnummer')
    0     True
    1    False
    Name: identiteitskaartnummer, dtype: bool
    """
    return validate_stdnum(df, column, identiteitskaartnummer.is_valid)
