"""
Clean and validate a DataFrame column containing URLs.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import partial
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from .utils import (
    _percent,
    add_columns,
    check_errors,
    is_null,
    map_column,
    unzip,
    validate_values,
)

if TYPE_CHECKING:
    import dask.dataframe as dd

# to extract queries
QUERY_REGEX = re.compile(r"(\?|\&)([^=]+)\=([^&]+)")

# regex to validate url
VALID_URL_REGEX = re.compile(
    r"^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+$"
)

# removing auth params
AUTH_VALUES = {
    "access_token",
    "auth_key",
    "auth",
    "password",
    "username",
    "login",
    "token",
    "passcode",
    "access-token",
    "auth-key",
    "authentication",
    "authentication-key",
}


def clean_url(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    remove_auth: bool | list[str] = False,
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize URLs.

    Read more in the :ref:`User Guide <url_userguide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing URL addresses.
    remove_auth
        Can be a boolean value or list of strings representing the names of
        Auth queries to be removed. If True, remove default Auth values. If
        False, do not remove Auth values.

        (default: False)
    split
        If True, split the URL into the scheme, hostname, queries, cleaned_url columns.
        If False, return a column of dictionaries with the relavant
        information (e.g., scheme, hostname, etc.) as key-value pairs.

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
        If True, output the summary report. Otherwise, no report is outputted.

        (default: True)
    progress
        If True, display a progress bar.

        (default: True)

    Examples
    --------
    Split a URL into its components.

    >>> df = pd.DataFrame({'url': ['https://github.com/sfu-db/dataprep','https://www.google.com/']})
    >>> clean_url(df, 'url')
    URL Cleaning Report:
        2 values parsed (100.0%)
    Result contains 2 (100.0%) parsed key-value pairs and 0 null values (0.0%)
                                    url                                        url_details
    0  https://github.com/sfu-db/dataprep  {'scheme': 'https', 'host': 'github.com', 'url...
    1             https://www.google.com/  {'scheme': 'https', 'host': 'www.google.com', ...
    """
    # pylint: disable=too-many-arguments
    check_errors(errors)

    # the query names to remove: the default authentication parameters, plus the
    # ones supplied by the user if remove_auth is a list
    auth_values: frozenset[str] | None = None
    if remove_auth:
        extra = () if isinstance(remove_auth, bool) else remove_auth
        auth_values = frozenset({*AUTH_VALUES, *extra})

    df, rows = map_column(
        df,
        column,
        partial(_format_url, col=column, auth_values=auth_values, split=split, errors=errors),
        progress,
    )
    if split:
        schemes, hosts, cleaned, queries, codes, removed = unzip(rows, 6)
        new_columns = {
            "scheme": schemes,
            "host": hosts,
            f"{column}_clean": cleaned,
            "queries": queries,
        }
    else:
        details, codes, removed = unzip(rows, 3)
        new_columns = {f"{column}_details": details}
    df = add_columns(df, new_columns, drop=column if inplace else None)

    # output the report describing the changes to the column
    if report:
        _report_url(Counter(codes), sum(removed), errors)

    return df


def validate_url(x: str | pd.Series) -> bool | pd.Series:
    """
    Validate URLs.

    Read more in the :ref:`User Guide <url_userguide>`.

    Parameters
    ----------
    x
        pandas Series of URLs or string URL.

    Examples
    --------

    >>> validate_url('https://github.com/sfu-db/dataprep')
    True
    >>> df = pd.DataFrame({'url': ['https://www.google.com/', 'NaN']})
    >>> validate_url(df['url'])
    0     True
    1    False
    Name: url, dtype: bool
    """

    return validate_values(x, partial(_check_url, clean=False))


def _format_url(
    url: Any, col: str, auth_values: frozenset[str] | None, split: bool, errors: str
) -> Any:
    """
    This function formats the input value "url"

    The last two components of the returned tuple hold the following codes:
        the first component: code indicating how the value was transformed (see below)
        the second component: the count of auth queries removed, if applicable
    In the first component, there are the following four codes:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value was parsed and DID NOT have authentication queries removed
        3 := the value was parsed and DID have authentication querires removed
    """
    # pylint: disable=too-many-locals

    # check if the url is a valid URL, returns a "status" value "null" (url is null),
    # "unknwon" (url is not a URL), and "success" (url is a URL)
    status = _check_url(url, True)

    if status == "null":
        return (np.nan, np.nan, np.nan, np.nan, 0, 0) if split else (np.nan, 0, 0)
    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"Unable to parse value {url}")
        result = url if errors == "ignore" else np.nan
        return (result, np.nan, np.nan, np.nan, 1, 0) if split else (result, 1, 0)

    # regex for finding the query / params and values
    url = str(url)
    re_queries = re.findall(QUERY_REGEX, url)
    all_queries = dict((y, z) for _, y, z in re_queries)

    # initialize the removed authentication code and count for the stats
    rem_auth_code, rem_auth_cnt = 2, 0
    # removing auth queries
    filtered_queries = all_queries
    if auth_values is not None:
        filtered_queries = {k: v for k, v in all_queries.items() if k not in auth_values}

        # count of removed auth queries
        rem_auth_cnt = len(all_queries) - len(filtered_queries)
        # code to indicate whether queries were removed
        rem_auth_code = 2 if rem_auth_cnt == 0 else 3

    # parse the url using urllib
    parsed = urlparse(url)

    # extracting params
    scheme = parsed.scheme
    host = parsed.hostname if parsed.hostname else ""
    path = parsed.path if parsed.path else ""
    cleaned_url = unquote(f"{scheme}://{host}{path}").replace(" ", "")
    queries = filtered_queries

    # returning the type based upon the split parameter.
    if split:
        return scheme, host, cleaned_url, queries, rem_auth_code, rem_auth_cnt
    return (
        {"scheme": scheme, "host": host, f"{col}_clean": cleaned_url, "queries": queries},
        rem_auth_code,
        rem_auth_cnt,
    )


def _check_url(url: Any, clean: bool) -> Any:
    """
    Function to check whether a value is a valid url
    """
    # check if the url is parsable
    if is_null(url):
        return "null" if clean else False

    url = unquote(str(url)).replace(" ", "")

    if VALID_URL_REGEX.match(url):
        return "success" if clean else True
    return "unknown" if clean else False


def _report_url(stats: Mapping[int, int], removed_auth_cnt: int, errors: str) -> None:
    """
    This function displays the stats report

    In the stats DataFrame, the codes have the following meaning:
        0 := values that are null
        1 := values that could not be parsed
        2 := values that are parsed and DID NOT have authentication queries removed
        3 := values that are parsed and DID have authentication querires removed
    """
    print("URL Cleaning Report:")
    counts = Counter(stats)
    nrows = sum(counts.values())

    nclnd = counts[2] + counts[3]
    pclnd = _percent(nclnd, nrows)
    if nclnd > 0:
        print(f"\t{nclnd} values parsed ({pclnd}%)")

    nunknown = counts[1]
    if nunknown > 0:
        expl = "set to NaN" if errors == "coerce" else "left unchanged"
        print(
            f"\t{nunknown} values unable to be parsed ({_percent(nunknown, nrows)}%), {expl}"
        )

    if removed_auth_cnt > 0:
        print(f"Removed {removed_auth_cnt} auth queries from {counts[3]} rows")

    nnull = counts[0]
    if errors == "coerce":  # add unknown values that were set to NaN
        nnull += nunknown
    print(
        f"Result contains {nclnd} ({pclnd}%) parsed key-value pairs "
        f"and {nnull} null values ({_percent(nnull, nrows)}%)"
    )
