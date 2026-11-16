"""
Clean and validate a DataFrame column containing CPF numbers, Brazilian national identifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.br import cpf

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cpf.compact,
    "standard": cpf.format,
}


def clean_br_cpf(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Brazilian national identifier data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CPF type.
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
    Clean a column of CPF data.

    >>> df = pd.DataFrame({"cpf": ["23100299900", "231.002.999-00", "390.533.447=0"]})
    >>> clean_br_cpf(df, "cpf")
                  cpf  cpf_clean
    0     23100299900        NaN
    1  231.002.999-00        NaN
    2   390.533.447=0        NaN
    """
    return clean_stdnum(
        df,
        column,
        cpf.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_br_cpf(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CPF in a DataFrame column. For each cell, return True or False.

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
    >>> validate_br_cpf("390.533.447-05")
    True
    >>> df = pd.DataFrame({"cpf": ["23100299900", "231.002.999-00", "390.533.447=0"]})
    >>> validate_br_cpf(df["cpf"])
    0    False
    1    False
    2    False
    Name: cpf, dtype: bool
    >>> validate_br_cpf(df, "cpf")
    0    False
    1    False
    2    False
    Name: cpf, dtype: bool
    """
    return validate_stdnum(df, column, cpf.is_valid)
