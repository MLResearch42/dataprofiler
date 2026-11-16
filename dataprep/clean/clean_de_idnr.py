"""
Clean and validate a DataFrame column containing German personal tax number (IDNRs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.de import idnr

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": idnr.compact,
    "standard": idnr.format,
}


def clean_de_idnr(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean German personal tax number (IDNRs) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of IDNR type.
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
    Clean a column of IDNR data.

    >>> df = pd.DataFrame({"idnr": ["36574261809", "36554266806"]})
    >>> clean_de_idnr(df, "idnr")
              idnr      idnr_clean
    0  36574261809  36 574 261 809
    1  36554266806             NaN
    """
    return clean_stdnum(
        df,
        column,
        idnr.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_de_idnr(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is IDNR in a DataFrame column. For each cell, return True or False.

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
    >>> validate_de_idnr("36574261809")
    True
    >>> df = pd.DataFrame({"idnr": ["36574261809", "36554266806"]})
    >>> validate_de_idnr(df["idnr"])
    0     True
    1    False
    Name: idnr, dtype: bool
    >>> validate_de_idnr(df, "idnr")
    0     True
    1    False
    Name: idnr, dtype: bool
    """
    return validate_stdnum(df, column, idnr.is_valid)
