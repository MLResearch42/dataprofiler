"""
Clean and validate a DataFrame column containing
classification for businesses in the European Union (NACE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.eu import nace

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": nace.compact,
    "standard": nace.format,
    "label": nace.get_label,
}


def clean_eu_nace(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean classification for businesses in the European Union type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of NACE type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'label', return the category label for the number.

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
    Clean a column of NACE data.

    >>> df = pd.DataFrame({"nace": ["6201", "99999999999"]})
    >>> clean_eu_nace(df, "nace")
              nace  nace_clean
    0         6201         NaN
    1  99999999999         NaN
    """
    return clean_stdnum(
        df,
        column,
        nace.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_eu_nace(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is NACE in a DataFrame column. For each cell, return True or False.

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
    >>> validate_eu_nace("A")
    True
    >>> df = pd.DataFrame({"nace": ["6201", "99999999999"]})
    >>> validate_eu_nace(df["nace"])
    0    False
    1    False
    Name: nace, dtype: bool
    >>> validate_eu_nace(df, "nace")
    0    False
    1    False
    Name: nace, dtype: bool
    """
    return validate_stdnum(df, column, nace.is_valid)
