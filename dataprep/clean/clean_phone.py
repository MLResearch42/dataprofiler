"""
Clean and validate a DataFrame column containing phone numbers.
"""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .utils import check_errors, clean_column, create_report_new, is_null, validate_values

if TYPE_CHECKING:
    import dask.dataframe as dd

PHONE_PATTERN = re.compile(
    r"""
    ^\s*
    (?:[+(]?(?P<country>(9[976]\d|8[987530]\d|6[987]\d|5[90]\d|42\d|3[875]\d|
                        2[98654321]\d|9[8543210]|8[6421]|6[6543210]|5[87654321]|
                        4[987654310]|3[9643210]|2[70]|7|1))[)\/]?)?
    [-. (]*
    (?P<area>\d{3})?
    [-. )\/]*
    (?:(?P<office>\d{3})
    [-. \/]*
    (?P<station>\d{4})|
    (?P<letters>[0-9A-Z-. \/]{7,13}?))
    (?:[ \t]*(?:\#|x[.:]?|[Ee]xt[.:]?|[Ee]xtension)[ \t]*(?P<ext>\d+))?
    \s*$
    """,
    re.VERBOSE,
)

US_PHONE_PATTERN = re.compile(
    r"""
    ^\s*
    (?:[+(]?(?P<country>1)[)\/]?)?
    [-. (]*
    (?P<area>\d{3})?
    [-. )\/]*
    (?:(?P<office>\d{3})
    [-. \/]*
    (?P<station>\d{4})|
    (?P<letters>[0-9A-Z-. \/]{7,13}?))
    (?:[ \t]*(?:\#|x[.:]?|[Ee]xt[.:]?|[Ee]xtension)[ \t]*(?P<ext>\d+))?
    \s*$
    """,
    re.VERBOSE,
)

ALPHA_NUM_MAP = {
    "A": "2",
    "B": "2",
    "C": "2",
    "D": "3",
    "E": "3",
    "F": "3",
    "G": "4",
    "H": "4",
    "I": "4",
    "J": "5",
    "K": "5",
    "L": "5",
    "M": "6",
    "N": "6",
    "O": "6",
    "P": "7",
    "Q": "7",
    "R": "7",
    "S": "7",
    "T": "8",
    "U": "8",
    "V": "8",
    "W": "9",
    "X": "9",
    "Y": "9",
    "Z": "9",
}


def clean_phone(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "nanp",
    fix_missing: str = "empty",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize phone numbers.

    Read more in the :ref:`User Guide <phone_userguide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing phone numbers.
    output_format
        The desired format of the phone numbers.
            - 'nanp': 'NPA-NXX-XXXX'
            - 'e164': '+1NPANXXXXXX'
            - 'national': '(NPA) NXX-XXXX'

        (default: 'nanp')
    fix_missing
        Fix the missing country code of a parsed phone number.
            - 'empty': leave the missing component as is.
            - 'auto': set the country code to a default value (1).

        (default: 'empty')
    split
        If True, split a column containing a phone number into different
        columns containing individual components.

        (default: False)
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
        If True, output the summary report. Else, no report is outputted.

        (default: True)
    progress
        If True, enable the progress bar.

        (default: True)

    Examples
    --------

    >>> df = pd.DataFrame({'phone': ['555-234-5678', '(555) 234-5678', '555.234.5678']})
    >>> clean_phone(df, 'phone')
    Phone Number Cleaning Report:
        2 values cleaned (66.67%)
    Result contains 3 (100.0%) values in the correct format and 0 null values (0.0%)
                phone   phone_clean
    0    555-234-5678  555-234-5678
    1  (555) 234-5678  555-234-5678
    2    555.234.5678  555-234-5678
    """
    # pylint: disable=too-many-arguments

    if output_format not in {"nanp", "e164", "national"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "nanp", "e164" or "national"'
        )

    if fix_missing not in {"auto", "empty"}:
        raise ValueError(
            f'fix_missing {fix_missing} is invalid, it needs to be "auto" or "empty"'
        )

    check_errors(errors)

    new_columns = (
        ["country_code", "area_code", "office_code", "station_code", "ext_num"]
        if split
        else [f"{column}_clean"]
    )
    df, stats = clean_column(
        df,
        column,
        partial(
            _format_phone,
            output_format=output_format,
            fix_missing=fix_missing,
            split=split,
            errors=errors,
        ),
        new_columns,
        inplace=inplace,
        progress=progress,
    )

    # output a report describing the result of clean_phone
    if report:
        create_report_new("Phone Number", stats, errors)

    return df


def validate_phone(x: str | pd.Series) -> bool | pd.Series:
    """
    Validate phone numbers.

    Read more in the :ref:`User Guide <phone_userguide>`.

    Parameters
    ----------
    x
        pandas Series of phone numbers or a string/int containing a phone number.

    Examples
    --------

    >>> validate_phone('1 800 234 6789')
    True
    >>> df = pd.DataFrame({'phone': [1234567, '1234']})
    >>> validate_phone(df['phone'])
    0     True
    1    False
    Name: phone, dtype: bool
    """

    return validate_values(x, partial(_check_phone, clean=False))


def _format_phone(
    phone: Any, output_format: str, fix_missing: str, split: bool, errors: str
) -> Any:
    """
    Function to transform a phone number instance into the desired format.

    The last component of the returned tuple contains a code indicating how the
    input value was changed:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value is cleaned and the cleaned value is DIFFERENT than the input value
        3 := the value is cleaned and is THE SAME as the input value (no transformation)
    """
    country_code, area_code, office_code, station_code, ext_num, status = _check_phone(phone, True)
    if status == "null":
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0) if split else (np.nan, 0)

    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"unable to parse value {phone}")
        result = phone if errors == "ignore" else np.nan
        return (result, np.nan, np.nan, np.nan, np.nan, 1) if split else (result, 1)

    if split:
        missing_code = "1" if fix_missing == "auto" and area_code else np.nan
        country_code = country_code if country_code else missing_code
        area_code = area_code if area_code else np.nan
        ext_num = ext_num if ext_num else np.nan
        return country_code, area_code, office_code, station_code, ext_num, 2

    if output_format == "nanp":  # NPA-NXX-XXXX
        area_code = f"{area_code}-" if area_code else ""
        ext_num = f" ext. {ext_num}" if ext_num else ""
        result = f"{area_code}{office_code}-{station_code}{ext_num}"
    elif output_format == "e164":  # +NPANXXXXXX
        if area_code:
            country_code = "+" + (country_code or "1")
        else:
            country_code = ""
        area_code = area_code if area_code else ""
        ext_num = f" ext. {ext_num}" if ext_num else ""
        result = f"{country_code}{area_code}{office_code}{station_code}{ext_num}"
    elif output_format == "national":  # (NPA) NXX-XXXX
        area_code = f"({area_code}) " if area_code else ""
        ext_num = f" ext. {ext_num}" if ext_num else ""
        result = f"{area_code}{office_code}-{station_code}{ext_num}"

    return result, 2 if phone != result else 3


def split_phone(mch: re.Match[str], clean: bool) -> Any:
    """
    Function to parse a phone number and return the components if the
    parse is successful.

    Parameters
    ----------
    mch
        Phone number matched the regex pattern.
    clean
        If True, return the components of the parse (if successful) and
        the status "null" (if the value is null), "unknown" (if the value
        could not be parsed) or "success" (if the value was successfully
        parsed). Else, return False for an unsuccesful parse and True
        for a successful parse.
    """
    if mch.group("letters"):
        # Check that there are 7 alphanumeric characters present
        letters = re.sub(r"\W+", "", mch.group("letters"))
        if len(letters) != 7:
            return (None,) * 5 + ("unknown",) if clean else False
        # Convert letters to numbers
        numlist = [ALPHA_NUM_MAP[char] if char.isalpha() else char for char in letters]
        numbers = "".join(numlist)
    # Components for phone number
    country_code = mch.group("country")
    area_code = mch.group("area")
    office_code = numbers[:3] if mch.group("letters") else mch.group("office")
    station_code = numbers[3:] if mch.group("letters") else mch.group("station")
    ext_num = mch.group("ext")

    return (
        (country_code, area_code, office_code, station_code, ext_num, "success") if clean else True
    )


def _check_phone(phone: Any, clean: bool) -> Any:
    """
    Function to parse a phone number and return the components if the
    parse is successful.

    Parameters
    ----------
    val
        Phone number to be parsed.
    clean
        If True, return the components of the parse (if successful) and
        the status "null" (if the value is null), "unknown" (if the value
        could not be parsed) or "success" (if the value was successfully
        parsed). Else, return False for an unsuccesful parse and True
        for a successful parse.
    """
    # If the value is null, return None for the components
    # and "null" for the "status"
    if is_null(phone):
        return (None,) * 5 + ("null",) if clean else False

    text = str(phone).replace("''", '"')
    mch = US_PHONE_PATTERN.match(text)
    # Check if the value was able to be parsed

    if not mch:
        mch = PHONE_PATTERN.match(text)
        if not mch:
            return (None,) * 5 + ("unknown",) if clean else False
        else:
            return split_phone(mch, clean)
    if mch.group("country") and not mch.group("area"):
        return (None,) * 5 + ("unknown",) if clean else False

    return split_phone(mch, clean)
