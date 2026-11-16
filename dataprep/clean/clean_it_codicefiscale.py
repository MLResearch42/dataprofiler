"""
Clean and validate a DataFrame column containing Italian fiscal codes (Codice Fiscales).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum.it import codicefiscale

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": codicefiscale.compact,
    "standard": codicefiscale.compact,
    "birthdate": codicefiscale.get_birth_date,
    "gender": codicefiscale.get_gender,
}


def clean_it_codicefiscale(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Italian fiscal code (Codice Fiscales) type data in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of Codice Fiscale type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'birthdate', get the person's birthdate.
        If output_format = 'gender', get the person's birth gender ('M' or 'F').
        Note: in the case of Codice Fiscale, the compact format is the same as the standard.

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
    Clean a column of Codice Fiscale data.

    >>> df = pd.DataFrame({"codicefiscale": ["RCCMNL83S18D969H", "RCCMNL83S18D969"]})
    >>> clean_it_codicefiscale(df, "codicefiscale")
          codicefiscale codicefiscale_clean
    0  RCCMNL83S18D969H    RCCMNL83S18D969H
    1   RCCMNL83S18D969                 NaN
    """
    return clean_stdnum(
        df,
        column,
        codicefiscale.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_it_codicefiscale(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate if a data cell is Codice Fiscale in a DataFrame column.
    For each cell, return True or False.

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
    >>> validate_it_codicefiscale("RCCMNL83S18D969H")
    True
    >>> df = pd.DataFrame({"codicefiscale": ["RCCMNL83S18D969H", "RCCMNL83S18D969"]})
    >>> validate_it_codicefiscale(df["codicefiscale"])
    0     True
    1    False
    Name: codicefiscale, dtype: bool
    >>> validate_it_codicefiscale(df, "codicefiscale")
    0     True
    1    False
    Name: codicefiscale, dtype: bool
    """
    return validate_stdnum(df, column, codicefiscale.is_valid)
