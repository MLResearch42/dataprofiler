"""
Clean and validate a DataFrame column containing British Columbia personal health numbers (PHN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.ca import bc_phn

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": bc_phn.compact,
    "standard": bc_phn.format,
}


def clean_ca_bc_phn(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean British Columbia personal health numbers (PHN) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of PHN type.
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
    Clean a column of PHN data.

    >>> df = pd.DataFrame({'bc_phn': ['9698 658 215', '9698648215']})
    >>> clean_ca_bc_phn(df, 'bc_phn')
             bc_phn  bc_phn_clean
    0  9698 658 215  9698 658 215
    1    9698648215           NaN
    """
    return clean_stdnum(
        df,
        column,
        bc_phn.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_ca_bc_phn(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate British Columbia personal health numbers (PHN). For each cell, return True or False.

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
    >>> validate_ca_bc_phn('9698 658 215')
    True
    >>> df = pd.DataFrame({'bc_phn': ['9698 658 215', '9698648215']})
    >>> validate_ca_bc_phn(df['bc_phn'])
    0     True
    1    False
    Name: bc_phn, dtype: bool
    >>> validate_ca_bc_phn(df, 'bc_phn')
    0     True
    1    False
    Name: bc_phn, dtype: bool
    """
    return validate_stdnum(df, column, bc_phn.is_valid)
