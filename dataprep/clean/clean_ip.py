"""
Clean and validate a DataFrame column containing IP addresses.
"""

from __future__ import annotations

from functools import partial
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .utils import (
    check_dataframe,
    check_errors,
    clean_column,
    create_report_new,
    is_null,
    validate_values,
)

if TYPE_CHECKING:
    import dask.dataframe as dd


def clean_ip(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    input_format: str = "auto",
    output_format: str = "compressed",
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize IP addresses.

    Read more in the :ref:`User Guide <ip_userguide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing IP addresses.
    input_format
        The input format of the IP addresses.
            - 'auto': parse both ipv4 and ipv6 addresses.
            - 'ipv4': only parse ipv4 addresses.
            - 'ipv6': only parse ipv6 addresses.

        (default: 'auto')
    output_format
        The desired output format of the IP addresses.
            - 'compressed': compressed representation ('12.3.4.5')
            - 'full': full representation ('0012.0003.0004.0005')
            - 'binary': binary representation ('00001100000000110000010000000101')
            - 'hexa': hexadecimal representation ('0xc030405')
            - 'integer': integer representation (201524229)
            - 'packed': packed binary representation (big-endian, a bytes object)

        (default: 'compressed')
    inplace
        If True, delete the column containing the data that was cleaned. Otherwise,
        keep the original column.

        (default: False)
    errors
        How to handle parsing errors.
            - ‘coerce’: invalid parsing will be set to NaN.
            - ‘ignore’: invalid parsing will return the input.
            - ‘raise’: invalid parsing will raise an exception.

        (default: 'coerce')
    report
        If True, output the summary report. Otherwise, no report is outputted.

        (default: True)
    progress
        If True, display a progress bar.

        (default: True)

    Examples
    --------

    >>> df = pd.DataFrame({'ip': ['2001:0db8:85a3:0000:0000:8a2e:0370:7334', '233.5.6.000']})
    >>> clean_ip(df, 'ip')
    IP Cleaning Report:
        2 values cleaned (100.0%)
    Result contains 2 (100.0%) values in the correct format and 0 null values (0.0%)
                                            ip                      ip_clean
    0  2001:0db8:85a3:0000:0000:8a2e:0370:7334  2001:db8:85a3::8a2e:370:7334
    1                              233.5.6.000                     233.5.6.0
    """
    # pylint: disable=too-many-arguments

    # check if the parameters are of correct processing types and values
    check_dataframe(df)

    if not isinstance(column, str):
        raise ValueError(f"column {column} is invalid")

    if input_format not in {"ipv4", "ipv6", "auto"}:
        raise ValueError(
            f'input_format {input_format} is invalid, it needs to be "ipv4", "ipv6" or "auto"'
        )

    if output_format not in {"compressed", "full", "binary", "hexa", "integer", "packed"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "compressed", "full", '
            '"binary", "hexa", "integer" or "packed"'
        )

    if not isinstance(inplace, bool):
        raise ValueError(f"inplace {inplace} is invalid, it needs to be True or False")

    if not isinstance(report, bool):
        raise ValueError(f"report {report} is invalid, it needs to be True or False")

    check_errors(errors)

    df, stats = clean_column(
        df,
        column,
        partial(_format_ip, input_format=input_format, output_format=output_format, errors=errors),
        [f"{column}_clean"],
        inplace=inplace,
        progress=progress,
    )

    # output a report describing the result of clean_ip
    if report:
        create_report_new("IP", stats, errors)

    return df


def validate_ip(x: str | pd.Series, input_format: str = "auto") -> bool | pd.Series:
    """
    Validate IP addresses.

    Read more in the :ref:`User Guide <ip_userguide>`.

    Parameters
    ----------
    x
        pandas Series of IP addresses or a str ip address value
    input_format
        The IP address format to validate.
            - 'auto': validate both ipv4 and ipv6 addresses.
            - 'ipv4': only validate ipv4 addresses.
            - 'ipv6': only validate ipv6 addresses.

        (default: 'auto')

    Examples
    --------

    >>> validate_ip('fdf8:f53b:82e4::53')
    True
    >>> df = pd.DataFrame({'ip': ['fdf8:f53b:82e4::53', None]})
    >>> validate_ip(df['ip'])
    0     True
    1    False
    Name: ip, dtype: bool
    """
    return validate_values(x, partial(_check_ip, input_format=input_format, clean=False))


def _format_ip(val: Any, input_format: str, output_format: str, errors: str) -> Any:
    """
    This function transforms the value val into the desired ip format if possible

    The last component of the returned tuple contains a code indicating how the
    input value was changed:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value is cleaned and the cleaned value is DIFFERENT than the input value
        3 := the value is cleaned and is THE SAME as the input value (no transformation)
    """
    # pylint: disable=too-many-branches
    address, status = _check_ip(val, input_format, True)

    if status == "null":
        return np.nan, 0
    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"Unable to parse value {val}")
        return val if errors == "ignore" else np.nan, 1

    # compressed version without the leading zeros (for ipv6 double colon for zeros)
    if output_format == "compressed":
        result = address.compressed

    # Converts the integer repesentation of the ip address to its hexadecimal
    # form. Does not contain any dots or colons.
    elif output_format == "hexa":
        result = hex(int(address))

    # converts the ip address to its binary representation
    elif output_format == "binary":
        if address.version == 4:
            result = f"{int(address):032b}"
        else:
            result = f"{int(address):0128b}"

    # converts to integer format
    elif output_format == "integer":
        result = int(address)

    # converts to packed binary format (big-endian)
    elif output_format == "packed":
        result = address.packed

    # convert to full representation
    else:
        dlm = "." if address.version == 4 else ":"  # delimiter
        result = dlm.join(f"{'0' * (4 - len(x))}{x}" for x in address.exploded.split(dlm))

    return result, 2 if result != val else 3


def _check_ip(val: Any, input_format: str, clean: bool) -> Any:
    """
    Function to check whether a value is valid ip address
    """
    if is_null(val):
        return (None, "null") if clean else False
    if isinstance(val, bool):
        return (None, "unknown") if clean else False

    try:
        address = ip_address(val)
        vers = address.version

        if vers == 4 and input_format != "ipv6" or vers == 6 and input_format != "ipv4":
            return (address, "success") if clean else True
        return (None, "unknown") if clean else False

    except (TypeError, ValueError):
        return (None, "unknown") if clean else False
