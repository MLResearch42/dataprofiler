"""
Clean and validate a DataFrame column containing Belgian structured payment references (OGM-VCS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.be import ogm_vcs

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": ogm_vcs.compact,
    "standard": ogm_vcs.format,
}


def clean_be_ogm_vcs(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Belgian structured payment references (OGM-VCS) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of OGM-VCS type.
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
    Clean a column of OGM-VCS data.

    >>> df = pd.DataFrame({'ogm_vcs': ['+++010/8068/17183+++', '010/8068/17180']})
    >>> clean_be_ogm_vcs(df, 'ogm_vcs')
                    ogm_vcs   ogm_vcs_clean
    0  +++010/8068/17183+++  010/8068/17183
    1        010/8068/17180             NaN
    """
    return clean_stdnum(
        df,
        column,
        ogm_vcs.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_be_ogm_vcs(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Belgian structured payment references (OGM-VCS). For each cell, return True or False.

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
    >>> validate_be_ogm_vcs('+++010/8068/17183+++')
    True
    >>> df = pd.DataFrame({'ogm_vcs': ['+++010/8068/17183+++', '010/8068/17180']})
    >>> validate_be_ogm_vcs(df['ogm_vcs'])
    0     True
    1    False
    Name: ogm_vcs, dtype: bool
    >>> validate_be_ogm_vcs(df, 'ogm_vcs')
    0     True
    1    False
    Name: ogm_vcs, dtype: bool
    """
    return validate_stdnum(df, column, ogm_vcs.is_valid)
