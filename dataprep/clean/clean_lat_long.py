"""
Clean and validate a DataFrame column containing geographic coordinates.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .utils import (
    add_columns,
    apply_partitions,
    check_errors,
    clean_column,
    create_report_new,
    is_null,
    validate_values,
)

if TYPE_CHECKING:
    import dask.dataframe as dd

LAT_LONG_PATTERN = re.compile(
    r"""
    [^/-]*?[(]?
      (?P<dir_front>[NS])?[ ]*
        (?P<deg>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
        (?:(?P<min>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
        (?:(?P<sec>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
      )?(?P<dir_back>[NS])?
      \s*[,;/\s]\s*
      (?P<dir_front2>[EW])?[ ]*
        (?P<deg2>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
        (?:(?P<min2>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
        (?:(?P<sec2>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
      )?(?P<dir_back2>[EW])?
    [)]?\s*$
"""
    % {
        "FLOAT": r"\d+(?:\.\d+)?",
        "DEGREE": chr(176),
        "PRIME": chr(8242),
        "DOUBLE_PRIME": chr(8243),
    },
    re.VERBOSE | re.UNICODE,
)

LAT_PATTERN = re.compile(
    r"""
    [^/-]*?
      (?P<dir_front>[NS])?[ ]*
        (?P<deg>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
        (?:(?P<min>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
        (?:(?P<sec>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
      )?(?P<dir_back>[NS])?
    \s*$
"""
    % {
        "FLOAT": r"\d+(?:\.\d+)?",
        "DEGREE": chr(176),
        "PRIME": chr(8242),
        "DOUBLE_PRIME": chr(8243),
    },
    re.VERBOSE | re.UNICODE,
)

LONG_PATTERN = re.compile(
    r"""
    [^/-]*?
      (?P<dir_front>[EW])?[ ]*
        (?P<deg>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
        (?:(?P<min>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
        (?:(?P<sec>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
      )?(?P<dir_back>[EW])?
    \s*$
"""
    % {
        "FLOAT": r"\d+(?:\.\d+)?",
        "DEGREE": chr(176),
        "PRIME": chr(8242),
        "DOUBLE_PRIME": chr(8243),
    },
    re.VERBOSE | re.UNICODE,
)


def clean_lat_long(
    df: pd.DataFrame | dd.DataFrame,
    lat_long: str | None = None,
    *,
    lat_col: str | None = None,
    long_col: str | None = None,
    output_format: str = "dd",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize latitude and longitude coordinates.

    Read more in the :ref:`User Guide <clean_lat_long_user_guide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    lat_long
        The name of the column containing latitude and longitude coordinates.
    lat_col
        The name of the column containing latitude coordinates.

        If specified, the parameter lat_long must be None.
    long_col
        The name of the column containing longitude coordinates.

        If specified, the parameter lat_long must be None.
    output_format
        The desired format of the coordinates.
            - 'dd': decimal degrees (51.4934, 0.0098)
            - 'ddh': decimal degrees with hemisphere ('51.4934° N, 0.0098° E')
            - 'dm': degrees minutes ('51° 29.604′ N, 0° 0.588′ E')
            - 'dms': degrees minutes seconds ('51° 29′ 36.24″ N, 0° 0′ 35.28″ E')

        (default: 'dd')
    split
        If True, split the latitude and longitude coordinates into one column
        for latitude and a separate column for longitude. Otherwise, merge
        the latitude and longitude coordinates into one column.

        (default: False)
    inplace
        If True, delete the column(s) containing the data that was cleaned. Otherwise,
        keep the original column(s).

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
    Split a column containing latitude and longitude strings into separate
    columns in decimal degrees format.

    >>> df = pd.DataFrame({'coord': ['51° 29′ 36.24″ N, 0° 0′ 35.28″ E', '51.4934° N, 0.0098° E']})
    >>> clean_lat_long(df, 'coord', split=True)
    Latitude and Longitude Cleaning Report:
        2 values cleaned (100.0%)
    Result contains 2 (100.0%) values in the correct format and 0 null values (0.0%)
                            coord  latitude  longitude
    0  51° 29′ 36.24″ N, 0° 0′ 35.28″ E   51.4934     0.0098
    1             51.4934° N, 0.0098° E   51.4934     0.0098
    """
    # pylint: disable=too-many-branches

    if lat_long and (lat_col or long_col):
        raise ValueError("lat_long must be None if either lat_col or long_col is not None")

    if output_format not in {"dd", "ddh", "dm", "dms"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it must be "dd", "ddh", "dm", or "dms"'
        )

    if not (lat_long or lat_col or long_col):
        raise ValueError("one of lat_long, lat_col or long_col must be specified")
    check_errors(errors)

    if lat_long:
        # clean a latitude and longitude column
        new_columns = ["latitude", "longitude"] if split else [f"{lat_long}_clean"]
        df, stats = clean_column(
            df,
            lat_long,
            partial(_format_lat_long, output_format=output_format, split=split, errors=errors),
            new_columns,
            inplace=inplace,
            progress=progress,
        )
    else:
        df, stats = _clean_separate_columns(
            df, lat_col, long_col, output_format, split, inplace, errors, progress
        )

    # output the report describing the changes to the column
    if report:
        create_report_new("Latitude and Longitude", stats, errors)

    return df


def _clean_separate_columns(
    df: pd.DataFrame | dd.DataFrame,
    lat_col: str | None,
    long_col: str | None,
    output_format: str,
    split: bool,
    inplace: bool,
    errors: str,
    progress: bool,
) -> tuple[pd.DataFrame, Counter[int]]:
    """
    Clean a latitude column and/or a longitude column. If both are given and
    split is False, the cleaned values are merged into one "latitude_longitude" column.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    targets = [(col, hor_dir) for col, hor_dir in ((lat_col, "lat"), (long_col, "long")) if col]

    def _clean_partition(pdf: pd.DataFrame) -> tuple[pd.DataFrame, list[list[tuple[Any, int]]]]:
        return pdf, [
            [_format_lat_or_long(val, output_format, errors, hor_dir) for val in pdf[col].tolist()]
            for col, hor_dir in targets
        ]

    parts = apply_partitions(df, _clean_partition, progress)
    pdf = parts[0][0] if len(parts) == 1 else pd.concat([part[0] for part in parts])

    cleaned: list[list[Any]] = []
    codes: list[int] = []
    for position in range(len(targets)):
        rows = list(chain.from_iterable(part[1][position] for part in parts))
        cleaned.append([row[0] for row in rows])
        codes = [row[1] for row in rows]

    if len(targets) == 2 and not split:
        # merge the cleaned latitude and longitude columns
        if output_format == "dd":
            merged: list[Any] = list(zip(cleaned[0], cleaned[1]))
        else:
            merged = [
                np.nan if is_null(lat) or is_null(lon) else f"{lat}, {lon}"
                for lat, lon in zip(cleaned[0], cleaned[1])
            ]
        new_columns: dict[str, list[Any]] = {"latitude_longitude": merged}
        stats = Counter({2: len(merged)})
    else:
        new_columns = {f"{col}_clean": values for (col, _), values in zip(targets, cleaned)}
        stats = Counter(codes)

    pdf = add_columns(pdf, new_columns)
    if inplace:
        pdf = pdf.drop(columns=[col for col, _ in targets])
    return pdf, stats


def validate_lat_long(
    x: pd.Series | str | float | tuple[float, float],
    *,
    lat_long: bool = True,
    lat: bool = False,
    lon: bool = False,
) -> bool | pd.Series | None:
    """
    Validate latitude and longitude coordinates.

    Read more in the :ref:`User Guide <clean_lat_long_user_guide>`.

    Parameters
    ----------
    x
        A pandas Series, string, float, or tuple of floats, containing the latitude
        and/or longitude coordinates to be validated.
    lat_long
        If True, valid values contain latitude and longitude coordinates. Parameters
        lat and lon must be False if lat_long is True.

        (default: True)
    lat
        If True, valid values contain only latitude coordinates. Parameters
        lat_long and lon must be False if lat is True.

       (default: False)
    lon
        If True, valid values contain only longitude coordinates. Parameters
        lat_long and lat must be False if lon is True.

        (default: False)

    Examples
    --------
    Validate a coordinate string or series of coordinates.

    >>> validate_lat_long('51° 29′ 36.24″ N, 0° 0′ 35.28″ E')
    True
    >>> df = pd.DataFrame({'coordinates', ['51° 29′ 36.24″ N, 0° 0′ 35.28″ E', 'NaN']})
    >>> validate_lat_long(df['coordinates'])
    0     True
    1    False
    Name: coordinates, dtype: bool
    """

    if lat or lon:
        hor_dir = "lat" if lat else "long"
        return validate_values(x, partial(_check_lat_or_long, clean=False, hor_dir=hor_dir))
    if lat_long:
        return validate_values(x, partial(_check_lat_long, clean=False))

    return None


def _format_lat_long(val: Any, output_format: str, split: bool, errors: str) -> Any:
    """
    Function to transform a coordinate instance into the desired format

    The last component of the returned tuple contains a code indicating how the
    input value was changed:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value is cleaned and the cleaned value is DIFFERENT than the input value
        3 := the value is cleaned and is THE SAME as the input value (no transformation)
    """
    # pylint: disable=too-many-locals
    # _check_lat_long parses the value val, and will return the components
    # if the parse is succesful. The returned value "status" can be either 0 ie
    # "null" (which means val is a null value), 1 ie ("unkwonw") (in which case
    # val could not be parsed) or 2 ie "success" (a succesful parse of val).
    # dds, mins, secs, hem are the latitude components and # dds2, mins2, secs2, hem2
    # are the longitude components
    dds, mins, secs, hem, dds2, mins2, secs2, hem2, status = _check_lat_long(val, True)

    if status == 0:  # val is a null value
        return (np.nan, np.nan, 0) if split else (np.nan, 0)

    if status == 1:  # val contains an unknown value
        if errors == "raise":
            raise ValueError(f"unable to parse value {val}")
        result = val if errors == "ignore" else np.nan
        return (result, np.nan, 1) if split else (result, 1)

    # derive the hemisphere if not given in the initial coordinate
    if not hem:
        hem = "N" if dds >= 0 else "S"
    if not hem2:
        hem2 = "E" if dds2 >= 0 else "W"
    dds, dds2 = abs(dds), abs(dds2)

    # the following code if/elif blocks converts the
    # coordinate components to the desired output
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Change_of_units_and_format
    if output_format == "dd":
        fctr = -1 if hem == "S" else 1
        fctr2 = -1 if hem2 == "W" else 1
        lat, lon = round(fctr * dds, 4), round(fctr2 * dds2, 4)
    elif output_format == "ddh":
        lat = f"{round(dds, 4)}{chr(176)} {hem}"
        lon = f"{round(dds2, 4)}{chr(176)} {hem2}"
    elif output_format == "dm":
        mins = round(60 * (dds - int(dds)), 4)
        mins = int(mins) if mins.is_integer() else mins
        mins2 = round(60 * (dds2 - int(dds2)), 4)
        mins2 = int(mins2) if mins2.is_integer() else mins2
        lat = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {hem}"
        lon = f"{int(dds2)}{chr(176)} {mins2}{chr(8242)} {hem2}"
    elif output_format == "dms":
        mins = int(60 * (dds - int(dds)))
        secs = round(3600 * (dds - int(dds)) - 60 * mins, 4)
        secs = int(secs) if secs.is_integer() else secs
        mins2 = int(60 * (dds2 - int(dds2)))
        secs2 = round(3600 * (dds2 - int(dds2)) - 60 * mins2, 4)
        secs2 = int(secs2) if secs2.is_integer() else secs2
        lat = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {secs}{chr(8243)} {hem}"
        lon = f"{int(dds2)}{chr(176)} {mins2}{chr(8242)} {secs2}{chr(8243)} {hem2}"

    if split:
        return lat, lon, 2
    result = (lat, lon) if output_format == "dd" else f"{lat}, {lon}"
    return result, 2 if val != result else 3


def _check_lat_long(val: Any, clean: bool) -> Any:
    """
    Function to check if a coordinate instance is valid
    """
    # pylint: disable=too-many-boolean-expressions
    # if the value is null, return empty strings for the components
    # and the code 0 to indicate a null status
    if is_null(val):
        return (None,) * 8 + (0,) if clean else False

    mch = LAT_LONG_PATTERN.match(str(val).replace("''", '"'))
    # check if the value was able to be parsed
    if not mch:
        return (None,) * 8 + (1,) if clean else False
    if not mch.group("deg") or not mch.group("deg2"):
        return (None,) * 8 + (1,) if clean else False

    # coordinates for latitude
    mins = float(mch.group("min")) if mch.group("min") else 0
    secs = float(mch.group("sec")) if mch.group("sec") else 0
    dds = float(mch.group("deg")) + mins / 60 + secs / 3600
    hem = mch.group("dir_back") or mch.group("dir_front")

    # coordinates for longitude
    mins2 = float(mch.group("min2")) if mch.group("min2") else 0
    secs2 = float(mch.group("sec2")) if mch.group("sec2") else 0
    dds2 = float(mch.group("deg2")) + mins2 / 60 + secs2 / 3600
    hem2 = mch.group("dir_back2") or mch.group("dir_front2")

    # minutes and seconds need to be in the interval [0, 60)
    # for degrees:
    #  if hemisphere is given, then 0<=lat<=90 and 0<=long<=180
    #  if hemisphere is not given, then -90<=lat<=90 and -180<=long<=180
    # decimal degrees must be -90<=lat<=90 and -180<=long<=180
    # the first given hemisphere and last hemisphere cannot both be set
    if (
        not 0 <= mins < 60
        or not 0 <= mins2 < 60
        or not 0 <= secs < 60
        or not 0 <= secs2 < 60
        or hem
        and not 0 <= float(mch.group("deg")) <= 90
        or hem2
        and not 0 <= float(mch.group("deg2")) <= 180
        or not hem
        and abs(float(mch.group("deg"))) > 90
        or not hem2
        and abs(float(mch.group("deg2"))) > 180
        or abs(dds) > 90
        or abs(dds2) > 180
        or sum([mch.group("dir_back") is not None, mch.group("dir_front") is not None]) > 1
        or sum([mch.group("dir_back2") is not None, mch.group("dir_front2") is not None]) > 1
    ):
        return (None,) * 8 + (1,) if clean else False

    return (dds, mins, secs, hem, dds2, mins2, secs2, hem2, 2) if clean else True


def _format_lat_or_long(val: Any, output_format: str, errors: str, hor_dir: str) -> Any:
    """
    Function to transform a coordinate instance into the desired format
    """
    dds, mins, secs, hem, status = _check_lat_or_long(val, True, hor_dir)

    if status == 0:  # val contains a null value
        return np.nan, 0

    if status == 1:  # val contains an unknown value
        if errors == "raise":
            raise ValueError(f"unable to parse value {val}")
        return val if errors == "ignore" else np.nan, 1

    if not hem:
        if hor_dir == "lat":
            hem = "N" if dds >= 0 else "S"
        else:
            hem = "E" if dds >= 0 else "W"
    dds = abs(dds)

    if output_format == "dd":
        fctr = 1 if hem in {"N", "E"} else -1
        res = round(fctr * dds, 4)
    elif output_format == "ddh":
        res = f"{round(dds, 4)}{chr(176)} {hem}"
    elif output_format == "dm":
        mins = round(60 * (dds - int(dds)), 4)
        mins = int(mins) if mins.is_integer() else mins
        res = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {hem}"
    elif output_format == "dms":
        mins = int(60 * (dds - int(dds)))
        secs = round(3600 * (dds - int(dds)) - 60 * mins, 4)
        secs = int(secs) if secs.is_integer() else secs
        res = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {secs}{chr(8243)} {hem}"

    return res, 2 if val != res else 3


def _check_lat_or_long(val: Any, clean: bool, hor_dir: str) -> Any:
    """
    Function to check if a coordinate instance is valid
    """
    # pylint: disable=too-many-boolean-expressions
    if is_null(val):
        return (None,) * 4 + (0,) if clean else False

    pat = LAT_PATTERN if hor_dir == "lat" else LONG_PATTERN

    mch = pat.match(str(val).replace("''", '"'))
    if not mch:
        return (None,) * 4 + (1,) if clean else False
    if not mch.group("deg"):
        return (None,) * 4 + (1,) if clean else False

    # coordinates
    mins = float(mch.group("min")) if mch.group("min") else 0
    secs = float(mch.group("sec")) if mch.group("sec") else 0
    dds = float(mch.group("deg")) + mins / 60 + secs / 3600
    hem = mch.group("dir_back") or mch.group("dir_front")

    # range is [-90, 90] for latitude and [-180, 180] for longitude
    bound = 90 if hor_dir == "lat" else 180

    # minutes and seconds need to be in the interval [0, 60]
    # for degrees:
    #  if hemisphere is give, then 0<=deg<=bound
    #  if hemisphere is not given, then -bound<=deg<=bound
    # decimal degrees must be -bound<=lat<=bound
    # the first given hemisphere and last hemisphere cannot both be set
    if (
        not 0 <= mins <= 60
        or not 0 <= secs <= 60
        or hem
        and not 0 <= float(mch.group("deg")) <= bound
        or not hem
        and abs(float(mch.group("deg"))) > bound
        or abs(dds) > bound
        or sum([mch.group("dir_back") is not None, mch.group("dir_front") is not None]) > 1
    ):
        return (None,) * 4 + (1,) if clean else False

    return (dds, mins, secs, hem, 2) if clean else True
