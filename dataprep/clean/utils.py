"""Common functions shared by the ``dataprep.clean`` modules."""

from __future__ import annotations

import http.client
import json
import sys
from collections import Counter
from itertools import chain
from math import ceil
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # dask is only imported at runtime when a dask object is passed in
    import dask.dataframe as dd

NULL_VALUES = {
    np.nan,
    float("NaN"),
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
    "",
    None,
}

# The string members of NULL_VALUES. Membership tests against NULL_VALUES itself are
# unreliable for missing values (``float("nan") in NULL_VALUES`` is False for any NaN
# object other than ``np.nan``, and ``pd.NA`` is not in the set), so the cleaning
# functions use :func:`is_null` instead.
NULL_STRINGS = frozenset(val for val in NULL_VALUES if isinstance(val, str))

# Seconds to wait for the exchange rate services used by clean_currency
RATES_TIMEOUT = 10.0


def is_missing(val: Any) -> bool:
    """
    Return True if ``val`` is a missing value marker: None, NaN (any float NaN
    object), ``pd.NA`` or ``pd.NaT``.

    This is a fast scalar equivalent of ``pd.isna``; strings are never missing.
    """
    if val is None or val is pd.NA or val is pd.NaT:
        return True
    if isinstance(val, (float, np.floating)):
        return bool(val != val)  # pylint: disable=comparison-with-itself
    if isinstance(val, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(val))
    return False


def is_null(val: Any) -> bool:
    """
    Return True if ``val`` is a null value.

    Null values are the missing value markers (see :func:`is_missing`) and the
    strings in NULL_VALUES such as "", "NULL", "nan" and "N/A".
    """
    if isinstance(val, str):
        return val in NULL_STRINGS
    return is_missing(val)


def in_null_values(val: Any, null_values: Iterable[Any] | None) -> bool:
    """
    Return True if ``val`` is one of the user supplied ``null_values``.

    Missing value markers (None, NaN, pd.NA, pd.NaT) are all treated as the same
    value, so a NaN is found no matter which missing value marker is in the list.
    """
    if null_values is None:
        return False
    if is_missing(val):
        return any(is_missing(null) for null in null_values)
    return any(not is_missing(null) and val == null for null in null_values)


NEARBYKEYS = {
    "a": ["q", "w", "s", "x", "z"],
    "b": ["v", "g", "h", "n"],
    "c": ["x", "d", "f", "v"],
    "d": ["s", "e", "r", "f", "c", "x"],
    "e": ["w", "s", "d", "r"],
    "f": ["d", "r", "t", "g", "v", "c"],
    "g": ["f", "t", "y", "h", "b", "v"],
    "h": ["g", "y", "u", "j", "n", "b"],
    "i": ["u", "j", "k", "o"],
    "j": ["h", "u", "i", "k", "n", "m"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "m": ["n", "j", "k", "l"],
    "n": ["b", "h", "j", "m"],
    "o": ["i", "k", "l", "p"],
    "p": ["o", "l"],
    "q": ["w", "a", "s"],
    "r": ["e", "d", "f", "t"],
    "s": ["w", "e", "d", "x", "z", "a"],
    "t": ["r", "f", "g", "y"],
    "u": ["y", "h", "j", "i"],
    "v": ["c", "f", "g", "v", "b"],
    "w": ["q", "a", "s", "e"],
    "x": ["z", "s", "d", "c"],
    "y": ["t", "g", "h", "u"],
    "z": ["a", "s", "x"],
    " ": ["c", "v", "b", "n", "m"],
}


def is_dask(obj: Any) -> bool:
    """
    Return True if ``obj`` is a dask DataFrame or Series.

    dask is not imported by this check: if the caller holds a dask object,
    ``dask.dataframe`` is already in ``sys.modules``.
    """
    ddf = sys.modules.get("dask.dataframe")
    return ddf is not None and isinstance(obj, (ddf.DataFrame, ddf.Series))


def check_dataframe(df: Any) -> None:
    """Raise a ValueError if ``df`` is not a pandas or dask DataFrame."""
    ddf = sys.modules.get("dask.dataframe")
    if isinstance(df, pd.DataFrame) or (ddf is not None and isinstance(df, ddf.DataFrame)):
        return
    raise ValueError("df is invalid, it needs to be a pandas or Dask DataFrame.")


def check_errors(errors: str) -> None:
    """Raise a ValueError if ``errors`` is not one of "coerce", "ignore" or "raise"."""
    if errors not in {"coerce", "ignore", "raise"}:
        raise ValueError(
            f'errors {errors} is invalid, it needs to be "coerce", "ignore" or "raise".'
        )


def to_dask(df: pd.DataFrame | dd.DataFrame) -> dd.DataFrame:
    """Convert a dataframe to a dask dataframe."""
    import dask.dataframe as ddf  # pylint: disable=import-outside-toplevel

    if isinstance(df, ddf.DataFrame):
        return df

    df_size = df.memory_usage(deep=True).sum()
    npartitions = max(1, ceil(df_size / 128 / 1024 / 1024))  # 128 MB partition size
    return ddf.from_pandas(df, npartitions=npartitions)


def apply_partitions(
    df: pd.DataFrame | dd.DataFrame,
    func: Callable[[pd.DataFrame], Any],
    progress: bool = False,
) -> list[Any]:
    """
    Call ``func`` on a pandas DataFrame, or on every partition of a dask DataFrame.

    Returns the list of results, one per partition (a single result for pandas
    input). pandas input is processed directly without going through dask. For
    dask input the partitions are processed by the active dask scheduler and
    ``progress`` controls whether a progress bar is displayed.
    """
    if not is_dask(df):
        return [func(df)]

    import dask  # pylint: disable=import-outside-toplevel

    from ..progress_bar import ProgressBar  # pylint: disable=import-outside-toplevel

    tasks = [dask.delayed(func)(part) for part in df.to_delayed()]
    with ProgressBar(minimum=1, disable=not progress):
        return list(dask.compute(*tasks))


def map_column(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    func: Callable[[Any], tuple[Any, ...]],
    progress: bool = False,
) -> tuple[pd.DataFrame, list[tuple[Any, ...]]]:
    """
    Apply ``func`` to every value of ``df[column]``.

    ``func`` must return a tuple. Returns the (computed) pandas DataFrame and a
    list with one result tuple per row, in row order.
    """

    def _map_partition(pdf: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[Any, ...]]]:
        return pdf, [func(val) for val in pdf[column].tolist()]

    results = apply_partitions(df, _map_partition, progress)
    if len(results) == 1:
        return results[0]
    pdf = pd.concat([res[0] for res in results])
    return pdf, list(chain.from_iterable(res[1] for res in results))


def unzip(rows: Sequence[tuple[Any, ...]], width: int) -> list[tuple[Any, ...]]:
    """
    Transpose a list of equally sized row tuples into ``width`` column tuples.
    """
    if not rows:
        return [()] * width
    return list(zip(*rows))


def add_columns(
    df: pd.DataFrame,
    new_columns: Mapping[str, Iterable[Any]],
    drop: str | None = None,
    replace: str | None = None,
) -> pd.DataFrame:
    """
    Return a new DataFrame with the columns in ``new_columns`` appended to ``df``.

    The input DataFrame is never modified. The dtype of each new column is
    inferred by pandas, so a column of strings has the default string dtype of
    the installed pandas version (``object`` in pandas 2, ``str`` in pandas 3).

    Parameters
    ----------
    df
        A pandas DataFrame.
    new_columns
        Mapping of column name to the values of the column, in row order.
    drop
        Name of a column to drop from the result.
    replace
        Name of a column whose position is taken over by the first new column.
        The replaced column is removed from the result.
    """
    out = df.copy(deep=False)
    for position, (name, values) in enumerate(new_columns.items()):
        series = pd.Series(list(values), index=df.index)
        if position == 0 and replace is not None:
            loc = out.columns.get_loc(replace)
            out = out.drop(columns=replace)
            if name in out.columns:
                out = out.drop(columns=name)
                loc = min(loc, out.shape[1])
            out.insert(loc, name, series)
        else:
            out[name] = series
    if drop is not None:
        out = out.drop(columns=drop)
    return out


def clean_column(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    func: Callable[[Any], tuple[Any, ...]],
    new_columns: Sequence[str],
    inplace: bool = False,
    progress: bool = False,
) -> tuple[pd.DataFrame, Counter[Any]]:
    """
    Clean ``df[column]`` value by value and append the results as new columns.

    ``func`` is called with every value of the column and must return a tuple
    with one entry for each name in ``new_columns`` followed by a status code
    (see :func:`create_report_new`). Returns the resulting pandas DataFrame and
    the count of each status code. If ``inplace`` is True the original column is
    dropped from the result.
    """
    pdf, rows = map_column(df, column, func, progress)
    columns = unzip(rows, len(new_columns) + 1)
    out = add_columns(pdf, dict(zip(new_columns, columns)), drop=column if inplace else None)
    return out, Counter(columns[-1])


def validate_values(
    x: Any, func: Callable[[Any], bool], *, tabular: bool = False
) -> bool | pd.Series | pd.DataFrame | dd.Series | dd.DataFrame:
    """
    Apply the validation function ``func`` to a scalar or to every value of a Series.

    pandas Series give a boolean pandas Series, dask Series give a lazy boolean
    dask Series and everything else is treated as a single value. If ``tabular``
    is True, DataFrames (pandas or dask) are validated cell by cell as well.
    """
    if isinstance(x, pd.Series):
        return _validate_series(x, func)
    if tabular and isinstance(x, pd.DataFrame):
        return _validate_frame(x, func)
    if is_dask(x):
        ddf = sys.modules["dask.dataframe"]
        if isinstance(x, ddf.Series):
            meta = pd.Series([], dtype=bool, name=x.name)
            return x.map_partitions(_validate_series, func, meta=meta)
        if tabular:
            meta = pd.DataFrame({col: pd.Series([], dtype=bool) for col in x.columns})
            return x.map_partitions(_validate_frame, func, meta=meta)
    return bool(func(x))


def _validate_series(srs: pd.Series, func: Callable[[Any], bool]) -> pd.Series:
    """Apply ``func`` to every value of a pandas Series, returning a boolean Series."""
    return pd.Series(
        [bool(func(val)) for val in srs.tolist()], index=srs.index, name=srs.name, dtype=bool
    )


def _validate_frame(df: pd.DataFrame, func: Callable[[Any], bool]) -> pd.DataFrame:
    """Apply ``func`` to every cell of a pandas DataFrame, returning a boolean DataFrame."""
    return pd.DataFrame(
        {col: _validate_series(df[col], func) for col in df.columns}, index=df.index
    )


def create_report(type_cleaned: str, stats: Mapping[str, int], nrows: int) -> None:
    """
    Describe what was done in the cleaning process
    """
    # pylint: disable=line-too-long
    print(f"{type_cleaned} Cleaning Report:")
    if stats["cleaned"] > 0:
        nclnd = stats["cleaned"]
        pclnd = round(nclnd / nrows * 100, 2)
        print(f"\t{nclnd} values cleaned ({pclnd}%)")
    if stats["unknown"] > 0:
        nunknown = stats["unknown"]
        punknown = round(nunknown / nrows * 100, 2)
        print(f"\t{nunknown} values unable to be parsed ({punknown}%), set to NaN")
    nnull = stats["null"] + stats["unknown"]
    pnull = round(nnull / nrows * 100, 2)
    ncorrect = nrows - nnull
    pcorrect = round(100 - pnull, 2)
    print(
        f"""Result contains {ncorrect} ({pcorrect}%) values in the correct format and {nnull} null values ({pnull}%)"""
    )


def _percent(count: int, total: int) -> float:
    """Return ``count`` as a percentage of ``total`` rounded to 2 decimals (0 if total is 0)."""
    return round(count / total * 100, 2) if total else 0.0


def create_report_new(type_cleaned: str, stats: Mapping[int, int] | pd.Series, errors: str) -> None:
    """
    Describe what was done in the cleaning process

    ``stats`` maps the following codes to the number of values with that code
        0 := the number of null values
        1 := the number of values that could not be parsed
        2 := the number of values that were transformed during cleaning
        3 := the number of values that were already in the correct format
    """
    counts = Counter({int(code): int(count) for code, count in dict(stats).items()})
    print(f"{type_cleaned} Cleaning Report:")
    nrows = sum(counts.values())

    nclnd = counts[2]
    if nclnd > 0:
        print(f"\t{nclnd} values cleaned ({_percent(nclnd, nrows)}%)")

    nunknown = counts[1]
    if nunknown > 0:
        expl = "set to NaN" if errors == "coerce" else "left unchanged"
        print(
            f"\t{nunknown} values unable to be parsed ({_percent(nunknown, nrows)}%), {expl}"
        )

    nnull = counts[0]
    if errors == "coerce":
        nnull += nunknown

    ncorrect = nclnd + counts[3]
    print(
        f"Result contains {ncorrect} ({_percent(ncorrect, nrows)}%) values in the correct format "
        f"and {nnull} null values ({_percent(nnull, nrows)}%)"
    )


def _get_data(currency_code: str, file_path: str) -> Any:
    """
    returns the fiat currency's symbol and full name
    """
    with open(file_path, encoding="utf-8") as f:
        currency_data = json.loads(f.read())
    currency_dict = next((item for item in currency_data if item["cc"] == currency_code), None)
    if currency_dict:
        symbol = currency_dict.get("symbol")
        currency_name = currency_dict.get("name")
        return symbol, currency_name
    return None, None


def _fetch_json(host: str, request_path: str, timeout: float | None = None) -> Any:
    """
    GET ``https://{host}{request_path}`` and return the decoded JSON body.

    Raises RatesNotAvailableError with an actionable message if the service cannot
    be reached (no network, DNS failure, timeout) or does not answer with HTTP 200.
    """
    timeout = RATES_TIMEOUT if timeout is None else timeout
    conn = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        conn.request("GET", request_path)
        response = conn.getresponse()
        status = response.status
        body = response.read()
    except (TimeoutError, OSError, http.client.HTTPException) as err:
        raise RatesNotAvailableError(
            f"Could not fetch live exchange rates from https://{host} "
            f"({type(err).__name__}: {err}). clean_currency needs internet access to "
            "look up conversion rates; on an offline or restricted network pass the "
            "conversion rate yourself with the `user_rate` parameter."
        ) from err
    finally:
        conn.close()

    if status != 200:
        raise RatesNotAvailableError(
            f"The exchange rate service https://{host} answered with HTTP status {status}. "
            "Pass the conversion rate yourself with the `user_rate` parameter."
        )
    try:
        return json.loads(body)
    except ValueError as err:
        raise RatesNotAvailableError(
            f"The exchange rate service https://{host} returned an unreadable response. "
            "Pass the conversion rate yourself with the `user_rate` parameter."
        ) from err


def _get_rate(base_cur: str, dest_cur: str, url: str) -> Any:
    """
    returns the conversion rate between 2 fiat currencies
    """
    if base_cur == dest_cur:
        return 1.0
    response_json = _fetch_json(url, f"/api/latest?base={base_cur}&symbols={dest_cur}&rtype=fpy")
    try:
        rate = np.round(response_json["rates"][dest_cur], 4)
    except (KeyError, TypeError) as err:
        raise RatesNotAvailableError(
            f"Currency Rate {base_cur} => {dest_cur} not available latest"
        ) from err
    if not rate:
        raise RatesNotAvailableError(f"Currency Rate {base_cur} => {dest_cur} not available latest")
    return rate


def _get_rate_crypto(base_cur: str, dest_cur: str, url: str) -> Any:
    """
    returns the price of the cryptocurrecy in the specified base currency
    """
    if base_cur == dest_cur:
        return 1.0
    response_json = _fetch_json(url, f"/api/v3/simple/price?ids={base_cur}&vs_currencies={dest_cur}")
    try:
        rate = response_json[base_cur][dest_cur]
    except (KeyError, TypeError) as err:
        raise RatesNotAvailableError(
            f"Currency Rate {base_cur} => {dest_cur} not available latest"
        ) from err
    if not rate:
        raise RatesNotAvailableError(f"Currency Rate {base_cur} => {dest_cur} not available latest")
    return rate


def _get_crypto_symbol_and_id(crypto_name: str, file_path: str) -> Any:
    """
    gets the cryprocurrency ID and symbol from cryptocurries.json file
    """
    cryptocurrencies = {}
    with open(file_path, encoding="utf-8") as file:
        cryptocurrencies = json.load(file)
    try:
        crypto = cryptocurrencies[crypto_name]
        crypto_id = crypto[0]
        crypto_symbol = crypto[1]
        return crypto_id, crypto_symbol
    except Exception as key_error:
        raise KeyError(
            f"The target curency name `{crypto_name}` doesn't sound correct, please recheck your value"
        ) from key_error


class RatesNotAvailableError(Exception):
    """
    Raised when the live exchange rate needed by clean_currency cannot be obtained,
    for example because the machine has no internet access or the rate service is down.
    """
