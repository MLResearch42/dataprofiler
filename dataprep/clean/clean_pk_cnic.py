"""
Clean and validate a DataFrame column containing Pakistani national identity card numbers (CNIC).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.pk import cnic

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cnic.compact,
    "standard": cnic.format,
    "gender": cnic.get_gender,
}


def clean_pk_cnic(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Pakistani national identity card numbers (CNIC) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CNIC type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'gender', return the gender encoded in the number ('M' or 'F').

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
    Clean a column of CNIC data.

    >>> df = pd.DataFrame({'cnic': ['34201-0891231-8', '34201-0891231']})
    >>> clean_pk_cnic(df, 'cnic')
                  cnic       cnic_clean
    0  34201-0891231-8  34201-0891231-8
    1    34201-0891231              NaN
    """
    return clean_stdnum(
        df,
        column,
        cnic.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_pk_cnic(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate Pakistani national identity card numbers (CNIC). For each cell, return True or False.

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
    >>> validate_pk_cnic('34201-0891231-8')
    True
    >>> df = pd.DataFrame({'cnic': ['34201-0891231-8', '34201-0891231']})
    >>> validate_pk_cnic(df['cnic'])
    0     True
    1    False
    Name: cnic, dtype: bool
    >>> validate_pk_cnic(df, 'cnic')
    0     True
    1    False
    Name: cnic, dtype: bool
    """
    return validate_stdnum(df, column, cnic.is_valid)
