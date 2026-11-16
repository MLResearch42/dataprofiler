"""
Clean and validate a DataFrame column containing CNPJ numbers, Brazilian company identifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.br import cnpj

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": cnpj.compact,
    "standard": cnpj.format,
}


def clean_br_cnpj(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Brazilian company identifier data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of CNPJ type.
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
    Clean a column of CNPJ data.

    >>> df = pd.DataFrame(
    ...     {
    ...         "cnpj": [
    ...             "16727230000197",
    ...             "16.727.230.0001-98",
    ...             "16.727.230/0001=97",
    ...         ]
    ...     }
    ... )
    >>> clean_br_cnpj(df, "cnpj")
                     cnpj          cnpj_clean
    0      16727230000197  16.727.230/0001-97
    1  16.727.230.0001-98                 NaN
    2  16.727.230/0001=97                 NaN
    """
    return clean_stdnum(
        df,
        column,
        cnpj.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_br_cnpj(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is CNPJ in a DataFrame column. For each cell, return True or False.

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
    >>> validate_br_cnpj("16727230000197")
    True
    >>> df = pd.DataFrame(
    ...     {
    ...         "cnpj": [
    ...             "16727230000197",
    ...             "16.727.230.0001-98",
    ...             "16.727.230/0001=97",
    ...         ]
    ...     }
    ... )
    >>> validate_br_cnpj(df["cnpj"])
    0     True
    1    False
    2    False
    Name: cnpj, dtype: bool
    >>> validate_br_cnpj(df, "cnpj")
    0     True
    1    False
    2    False
    Name: cnpj, dtype: bool
    """
    return validate_stdnum(df, column, cnpj.is_valid)
