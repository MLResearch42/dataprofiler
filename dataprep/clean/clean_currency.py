"""
Clean and validate a DataFrame column containing currencies
"""

# pylint: disable=too-many-arguments, line-too-long, too-many-locals, too-many-branches, too-many-statements, too-many-return-statements,

from __future__ import annotations

import json
from functools import cache, partial
from os import path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import pandas as pd

from .utils import (
    _get_rate,
    _get_rate_crypto,
    clean_column,
    create_report_new,
    is_dask,
    is_null,
    validate_values,
)

if TYPE_CHECKING:
    import dask.dataframe as dd


CURRENCIES_DATA_FILE = path.join(path.split(path.abspath(__file__))[0], "currencies.json")
CRYPTOCURRENCIES_DATA_FILE = path.join(path.split(path.abspath(__file__))[0], "cryptocurrency.json")

FIAT_BASE_URL = "api.ratesapi.io"
CRYPTO_BASE_URL = "api.coingecko.com"


def clean_currency(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    n_round: int = 2,
    input_currency: str = "usd",
    target_currency: Optional[str] = None,
    target_representation: str = "decimal",
    user_rate: Optional[float] = None,
    ignore_symbol: Optional[Union[List[str], str]] = None,
    errors: str = "coerce",
    split: bool = False,
    inplace: bool = False,
    report: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Clean, standardize and convert currencies.

    Read more in the :ref:`User Guide <currency_userguide>`.

    Cleaning a column never needs internet access. Converting to another currency
    (``target_currency``) needs an exchange rate: pass it with ``user_rate`` to work
    offline, otherwise the latest rate is requested from an online service and a
    ``RatesNotAvailableError`` is raised if the service cannot be reached.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing currencies
    n_round
        Round a float to given number of decimals

        (default: 2)
    input_currency
        Name of the input currency
            - For fiat currencies use the 3 letter abbreviation. Eg. "usd", "inr", etc.
            - For cryptocurrencies use the full name of the coin. Eg "Binance Coin", "ethereum".

        (default: "usd")
    target_currency
        Optional Parameter - Name of the output currency.
            - For fiat currencies use the 3 letter abbreviation. Eg. "usd", "inr", etc.
            - For cryptocurrencies use the full name of the coin. Eg "Binance Coin", "ethereum".

        (default: None)
    target_representation
        The desired format in which the result would be displayed.
            - 'decimal' (floating point number)
            - 'abbr' (string with comma separated thousands)

        (default: 'decimal')
    user_rate
        Optional parameter - user defined exchange rate. If it is given, no exchange
        rate is requested from the internet.

        (default: None)
    ignore_symbol
        Optional parameter - A string or list of strings (e.g. currency symbols) that the
        parser removes from the values before parsing them.

        (default: None)
    errors
        How to handle parsing errors.
            - 'coerce': invalid parsing will be set to NaN.
            - 'ignore': invalid parsing will return the input.
            - 'raise': invalid parsing will raise an exception.

        (default: 'coerce')
    split
        If True, split a column containing a cleaned or converted currency into different
        columns containing individual components.

        (default: False)
    inplace
        If True, delete the column containing the data that was cleaned. Otherwise,
        keep the original column.

        (default: False)
    report
        If True, output the summary report. Else, no report is outputted.

        (default: True)
    progress
        If True, enable the progress bar. Only shown for Dask DataFrames.

        (default: False)

    Examples
    --------

    >>> df = pd.DataFrame({'currencies': [234.56, 18790, '234,456.45']})
    >>> clean_currency(df, 'currencies', input_currency='INR', target_currency='USD',
    ...                user_rate=0.0133)
    Currency Cleaning Report:
    \t3 values cleaned (100.0%)
    Result contains 3 (100.0%) values in the correct format and 0 null values (0.0%)
       currencies  currencies_clean
    0      234.56              3.12
    1       18790            249.91
    2  234,456.45           3118.27
    """

    _check_params(
        df=df,
        column=column,
        n_round=n_round,
        input_currency=input_currency,
        target_currency=target_currency,
        target_representation=target_representation,
        user_rate=user_rate,
        ignore_symbol=ignore_symbol,
        errors=errors,
        split=split,
        inplace=inplace,
        report=report,
        progress=progress,
    )

    target_symbol = ""
    conversion_rate = 1.0
    conversion_type = ""

    input_currency = input_currency.lower()
    if target_currency is not None:
        target_currency = target_currency.lower()
        conversion_type = _detect_conversion_type(input_currency, target_currency)
        input_symbol, target_symbol = _get_symbols(input_currency, target_currency, conversion_type)
        # a rate supplied by the user means that no internet access is needed
        conversion_rate = (
            user_rate
            if user_rate is not None
            else _get_conversion_rate(input_currency, target_currency, conversion_type)
        )
    else:
        input_symbol = _get_input_symbol(input_currency)
        if user_rate is not None:
            conversion_rate = user_rate

    ignore_symbols = [ignore_symbol] if isinstance(ignore_symbol, str) else ignore_symbol
    has_target = target_currency is not None

    if not split:
        new_columns = [f"{column}_clean"]
    elif has_target:
        new_columns = [
            "input_currency_symbol",
            f"{column}_clean",
            "conversion_rate",
            "target_symbol",
            "target_val",
        ]
    else:
        new_columns = ["input_currency_symbol", f"{column}_clean"]

    df, stats = clean_column(
        df,
        column,
        partial(
            _format_currency,
            ignore_symbol=ignore_symbols,
            n_round=n_round,
            errors=errors,
            target_representation=target_representation,
            input_symbol=input_symbol.upper(),
            split=split,
            conversion_rate=conversion_rate,
            conversion_type=conversion_type,
            target_symbol=target_symbol.upper(),
            has_target=has_target,
        ),
        new_columns,
        inplace=inplace,
        progress=progress,
    )

    # output a report describing the result of clean_currency
    if report:
        create_report_new("Currency", stats, errors)

    return df


def _format_currency(
    val: Any,
    ignore_symbol: Optional[List[str]],
    n_round: int,
    errors: str,
    target_representation: str,
    input_symbol: str,
    split: bool,
    conversion_rate: float,
    conversion_type: str,
    target_symbol: str,
    has_target: bool = False,
) -> Any:
    """
    Formats a value and returns it in the specified format, followed by a code
    describing how the value was handled:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value was cleaned
    """
    # pylint: disable=too-many-arguments
    if not split:
        width = 1
    else:
        width = 5 if has_target else 2

    number, status = _check_currency(val, ignore_symbol, True)

    if status == "null":
        return (np.nan,) * width + (0,)

    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"Unable to parse value {val}")
        result = val if errors == "ignore" else np.nan
        if not split:
            return result, 1
        return (np.nan, result) + (np.nan,) * (width - 2) + (1,)

    amount, converted = _get_values_target_representation(
        val=number,
        target_representation=target_representation,
        conversion_type=conversion_type,
        conversion_rate=conversion_rate,
        n_round=n_round,
        split=split,
        input_symbol=input_symbol,
        target_symbol=target_symbol,
    )

    if not split:
        return converted, 2
    if has_target:
        return input_symbol, amount, conversion_rate, target_symbol, converted, 2
    return input_symbol, amount, 2


def validate_currency(
    x: Union[str, pd.Series], symbol: Optional[Union[List[str], str]] = None
) -> Union[bool, pd.Series]:
    """
    Validate currency values: a value is valid if it is a number once the
    thousands separators and the given symbols are removed.

    Parameters
    ----------
    x
        pandas Series of currencies or a single value.
    symbol
        A string or list of strings (e.g. currency symbols) that are removed from
        the values before checking them.

    Examples
    --------
    >>> validate_currency("1,234.50")
    True
    >>> validate_currency("$1,234.50")
    False
    >>> validate_currency("$1,234.50", symbol="$")
    True
    """
    symbols = [symbol] if isinstance(symbol, str) else symbol
    return validate_values(x, partial(_check_currency, symbol=symbols, clean=False))


def _check_currency(val: Union[str, Any], symbol: Optional[List[str]], clean: bool) -> Any:
    """
    Function to check whether a value is a valid currency
    """
    if is_null(val):
        return (None, "null") if clean else False

    if symbol is None:
        symbol = []

    text = str(val).replace(",", "")

    # strip symbols from the left and right of the value
    for s_s in symbol:
        text = text.strip(s_s)

    text = "".join(c for c in text if (c.isdigit() or c == "." or c not in symbol))

    try:
        number = float(text)
    except ValueError:
        return (None, "unknown") if clean else False
    if number != number:  # pylint: disable=comparison-with-itself
        return (None, "unknown") if clean else False
    return (text, "success") if clean else True


def _get_values_target_representation(
    val: Union[str, Any],
    target_representation: str,
    conversion_type: str,
    conversion_rate: float,
    n_round: int,
    split: bool,
    input_symbol: str,
    target_symbol: str,
) -> Any:
    """
    Returns the value of the converted currency in the specified format.
    The two formats specified are "abbr", "decimal".
    """
    # pylint: disable=too-many-arguments
    val = float(val)

    # 1. for fiat-to-fiat and crypto-to-fiat we multiply
    # 2. for fiat-to-crypto we divide

    if conversion_type in ("fiat_to_fiat", "crypto_to_fiat"):
        val_new = val * conversion_rate
    else:
        val_new = val / conversion_rate

    if target_representation == "abbr":
        val_abbr = f"{val:,.{n_round}f}"
        target_val = f"{val_new:,.{n_round}f}"
        if split:
            return val_abbr, target_val
        return input_symbol.upper() + val_abbr, target_symbol.upper() + target_val
    return float(np.round(val, n_round)), float(np.round(val_new, n_round))


def _check_params(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    n_round: int = 2,
    input_currency: str = "usd",
    target_currency: Optional[str] = None,
    target_representation: str = "decimal",
    user_rate: Optional[float] = None,
    ignore_symbol: Optional[Union[List[str], str]] = None,
    errors: str = "coerce",
    split: bool = False,
    inplace: bool = False,
    report: bool = True,
    progress: bool = False,
) -> None:
    """
    Checks whether the params passed by the end user is of the correct data type
    """
    if not isinstance(df, pd.DataFrame) and not (is_dask(df) and hasattr(df, "columns")):
        raise ValueError("df is invalid, it needs to be a pandas or Dask DataFrame")

    if not isinstance(column, str):
        raise ValueError(f"column name {column} is invalid, it name needs to a string")

    if not isinstance(n_round, int):
        raise ValueError(f"n_round {n_round} is invalid, it needs to be an integer")

    if not isinstance(input_currency, str):
        raise ValueError(f"input currency {input_currency} is invalid, it needs to be a string")

    if target_currency is not None and not isinstance(target_currency, str):
        raise ValueError(f"target_currency {target_currency} is invalid, it needs to be a string")

    if target_representation not in {"decimal", "abbr"}:
        raise ValueError(
            f"target_representation {target_representation} is invalid, "
            'it needs to be "decimal" or "abbr"'
        )

    if ignore_symbol is not None and not isinstance(ignore_symbol, (list, str)):
        raise ValueError(
            f"symbol {ignore_symbol} is invalid, it needs to be either a string or list of strings"
        )

    if not isinstance(split, bool):
        raise ValueError(f"split {split} is invalid, it needs to be a boolean")

    if not isinstance(inplace, bool):
        raise ValueError(f"inplace {inplace} is invalid, it needs to be to be a boolean")

    if not isinstance(report, bool):
        raise ValueError(f"report {report} is invalid, it needs to be a boolean")

    if not isinstance(progress, bool):
        raise ValueError(f"progress {progress} is invalid, it needs to be a boolean")

    if user_rate is not None and (
        isinstance(user_rate, bool) or not isinstance(user_rate, (int, float)) or user_rate == 0
    ):
        raise ValueError(f"user_rate {user_rate} is invalid, it needs to be a non-zero number")

    if errors not in {"coerce", "ignore", "raise"}:
        raise ValueError(f'errors {errors} is invalid, it needs to be "coerce", "ignore", "raise"')


@cache
def _get_dictionary_from_json(file_path: str) -> Any:
    """
    Utility function to load a json file and return it as a dictionary
    """
    with open(file_path, encoding="utf-8") as f:
        return json.loads(f.read())


def _fiat_symbols() -> dict[str, str]:
    """
    Returns the mapping of each (lower-case) fiat currency code to its symbol.
    """
    return {
        item["cc"].lower(): item.get("symbol") or item["cc"]
        for item in _get_dictionary_from_json(CURRENCIES_DATA_FILE)
    }


def _crypto_data() -> dict[str, list[str]]:
    """
    Returns the mapping of each cryptocurrency name to its [id, symbol].
    """
    data: dict[str, list[str]] = _get_dictionary_from_json(CRYPTOCURRENCIES_DATA_FILE)
    return data


def _detect_conversion_type(input_currency: str, target_currency: str) -> str:
    """
    This function detects the type of conversion that is to be performed. There are 3 types
    of conversions that are supported

    1. Fiat to Fiat
    2. Fiat to Cryptocurrency
    3. Cryptocurrency to Fiat
    """
    fiat = _fiat_symbols()
    crypto = _crypto_data()

    if input_currency in fiat and target_currency in fiat:
        return "fiat_to_fiat"
    if input_currency in fiat and target_currency in crypto:
        return "fiat_to_crypto"
    if input_currency in crypto and target_currency in fiat:
        return "crypto_to_fiat"
    if input_currency in crypto and target_currency in crypto:
        raise ValueError("Currently we do not support crypto to crypto conversion")
    raise ValueError("Please check your input and target currencies")


def _get_input_symbol(input_currency: str) -> str:
    """
    Returns the symbol of a fiat currency (3 letter code) or cryptocurrency (full name).
    """
    crypto = _crypto_data()
    if input_currency in crypto:
        return crypto[input_currency][1]
    fiat = _fiat_symbols()
    if input_currency in fiat:
        return fiat[input_currency]
    raise ValueError(
        f"input_currency {input_currency} is not a known currency. Use the 3 letter code "
        "of a fiat currency (e.g. 'usd') or the full name of a cryptocurrency (e.g. 'bitcoin')."
    )


def _get_symbols(input_currency: str, target_currency: str, conversion_type: str) -> tuple[str, str]:
    """
    Returns the symbols of the input and target currencies. No internet access is needed.
    """
    fiat = _fiat_symbols()
    crypto = _crypto_data()
    if conversion_type == "fiat_to_fiat":
        return fiat[input_currency], fiat[target_currency]
    if conversion_type == "fiat_to_crypto":
        return fiat[input_currency], crypto[target_currency][1]
    return crypto[input_currency][1], fiat[target_currency]


def _get_conversion_rate(input_currency: str, target_currency: str, conversion_type: str) -> Any:
    """
    Requests the latest conversion rate from the online rate services. Raises a
    RatesNotAvailableError if the rate cannot be obtained, e.g. without internet access.

        - fiat to fiat: the price of 1 unit of the input currency in the target currency
        - fiat to crypto: the price of 1 coin of the target currency in the input currency
        - crypto to fiat: the price of 1 coin of the input currency in the target currency
    """
    if conversion_type == "fiat_to_fiat":
        return _get_rate(input_currency.upper(), target_currency.upper(), FIAT_BASE_URL)
    crypto = _crypto_data()
    if conversion_type == "fiat_to_crypto":
        return _get_rate_crypto(crypto[target_currency][0], input_currency, CRYPTO_BASE_URL)
    return _get_rate_crypto(crypto[input_currency][0], target_currency, CRYPTO_BASE_URL)
