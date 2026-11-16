"""
Clean and validate a DataFrame column containing Latvian PVN (VAT) numbers (PVNs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from stdnum.exceptions import ValidationError
from stdnum.lv import pvn

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


def _to_birthdate(val: str) -> Any:
    """Render a valid number in the 'birthdate' output format."""
    try:
        return pvn.get_birth_date(val)
    except ValidationError:  # the PVN stands for a legal entity
        return np.nan


_FORMATTERS = {
    "compact": pvn.compact,
    "standard": pvn.compact,
    "birthdate": _to_birthdate,
}


def clean_lv_pvn(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Latvian PVN (VAT) numbers (PVNs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of PVN type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', return the birthdate of the person. Note only when
            PVN refers to a person (but not a legal entity) this format will be available.
        Note: in the case of PVN, the compact format is the same as the standard one.

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
    Clean a column of PVN data.

    >>> df = pd.DataFrame({"pvn": ["161175-19997", "40003521601"]})
    >>> clean_lv_pvn(df, "pvn")
                pvn    pvn_clean
    0  161175-19997  16117519997
    1   40003521601          NaN
    """
    return clean_stdnum(
        df,
        column,
        pvn.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_lv_pvn(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is PVN in a DataFrame column. For each cell, return True or False.

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
    >>> validate_lv_pvn("161175-19997")
    True
    >>> df = pd.DataFrame({"pvn": ["161175-19997", "40003521601"]})
    >>> validate_lv_pvn(df["pvn"])
    0     True
    1    False
    Name: pvn, dtype: bool
    >>> validate_lv_pvn(df, "pvn")
    0     True
    1    False
    Name: pvn, dtype: bool
    """
    return validate_stdnum(df, column, pvn.is_valid)
