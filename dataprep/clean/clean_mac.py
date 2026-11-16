"""
Clean and validate a DataFrame column containing MAC addresses (MAC address).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from stdnum import mac

from .stdnum_utils import clean_stdnum, validate_stdnum

if TYPE_CHECKING:
    import dask.dataframe as dd


_FORMATTERS = {
    "compact": mac.compact,
    "standard": mac.compact,
    "eui48": mac.to_eui48,
    "manufacturer": mac.get_manufacturer,
}


def clean_mac(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean MAC addresses (MAC address) in a DataFrame column.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing data of MAC address type.
    output_format
        The output format of standardized number string.
        If output_format = 'compact', return string without any separators or whitespace.
        If output_format = 'standard', return string with proper separators and whitespace.
        If output_format = 'eui48', return the address in EUI-48 notation (upper case, separated by dashes).
        If output_format = 'manufacturer', return the manufacturer registered for the address.
        Note: in the case of MAC address, the compact format is the same as the standard one.

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
    Clean a column of MAC address data.

    >>> df = pd.DataFrame({'mac': ['D0-50-99-84-A2-A0', 'D0-50-99-84-A2']})
    >>> clean_mac(df, 'mac')
                     mac          mac_clean
    0  D0-50-99-84-A2-A0  d0:50:99:84:a2:a0
    1     D0-50-99-84-A2                NaN
    """
    return clean_stdnum(
        df,
        column,
        mac.is_valid,
        _FORMATTERS,
        output_format,
        inplace,
        errors,
        progress,
    )


def validate_mac(
    df: str | pd.Series | dd.Series | pd.DataFrame | dd.DataFrame,
    column: str = "",
) -> bool | pd.Series | pd.DataFrame:
    """
    Validate MAC addresses (MAC address). For each cell, return True or False.

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
    >>> validate_mac('D0-50-99-84-A2-A0')
    True
    >>> df = pd.DataFrame({'mac': ['D0-50-99-84-A2-A0', 'D0-50-99-84-A2']})
    >>> validate_mac(df['mac'])
    0     True
    1    False
    Name: mac, dtype: bool
    >>> validate_mac(df, 'mac')
    0     True
    1    False
    Name: mac, dtype: bool
    """
    return validate_stdnum(df, column, mac.is_valid)
