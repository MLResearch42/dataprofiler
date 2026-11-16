"""
Clean and validate a DataFrame column containing dates and times.
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import datetime
from copy import deepcopy
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .clean_date_utils import (
    ALL_TIMEZONES,
    AM,
    JUMP,
    MONTHS,
    PM,
    TARGET_DAY,
    TARGET_HOUR,
    TARGET_MINUTE,
    TARGET_MONTH,
    TARGET_SECOND,
    TARGET_WEEKDAY,
    TARGET_YEAR,
    TEXT_MONTHS,
    TEXT_WEEKDAYS,
    WEEKDAYS,
    ZONE,
    ParsedDate,
    ParsedTargetFormat,
    check_date,
    fix_missing_current,
    fix_missing_minimum,
    split,
)
from .utils import (
    apply_partitions,
    check_errors,
    clean_column,
    create_report_new,
    is_null,
    validate_values,
)

if TYPE_CHECKING:
    import dask.dataframe as dd


def clean_date(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    output_format: str = "YYYY-MM-DD hh:mm:ss",
    input_timezone: str = "UTC",
    output_timezone: str = "",
    fix_missing: str = "minimum",
    infer_day_first: bool = True,
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize dates and times.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing dates.
    output_format
        The desired format of the date.

        (default: 'YYYY-MM-DD hh:mm:ss')
    input_timezone
        Time zone of the input date.

        (default: 'UTC')
    output_timezone
        The desired time zone of the date.

        (default: '')
    fix_missing
        Specify how to fill missing components of a date value.
            - 'minimum': fill hours, minutes, seconds with zeros, and month, day, year with \
            January 1st, 2000.
            - 'current': fill with the current date and time.
            - 'empty': don't fill missing components.

        (default: 'minimum')
    infer_day_first
        If True, the program will infer the ambiguous format '09-10-03' and '25-09-03' according \
        to '25-09-03' (day is the number of first position). The result should be '2003-10-09' and \
        '2003-09-25'.
        If False, do nothing of inferring. The result should be '2003-09-10' and '2003-09-25'.
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

    >>> df = pd.DataFrame({'date': ['Thu Sep 25 2003', 'Thu 10:36:28', '2003 09 25']})
    >>> clean_date(df, 'date')
    Dates Cleaning Report:
        3 values cleaned (100.0%)
    Result contains 3 (100.0%) values in the correct format and 0 null values (0.0%)
                date           date_clean
    0  Thu Sep 25 2003  2003-09-25 00:00:00
    1     Thu 10:36:28  2000-01-01 10:36:28
    2       2003 09 25  2003-09-25 00:00:00
    """
    # pylint: disable=too-many-arguments

    if fix_missing not in {"minimum", "current", "empty"}:
        raise ValueError(
            f"fix_missing {fix_missing} is invalid. "
            'It needs to be "minimum", "current" or "empty".'
        )
    if input_timezone not in ALL_TIMEZONES and input_timezone not in ZONE:
        raise ValueError(f"input_timezone {input_timezone} is invalid.")
    if (
        output_timezone not in ALL_TIMEZONES
        and output_timezone not in ZONE
        and output_timezone != ""
    ):
        raise ValueError(f"output_timezone {output_timezone} is invalid.")

    check_errors(errors)

    # the output format only needs to be parsed once
    parsed_output_format = _parse_output_format(output_format, output_timezone)
    if not parsed_output_format.valid:
        raise ValueError(
            f"output_format {output_format} is invalid. "
            f"Invalid tokens are {parsed_output_format.invalid_tokens}."
        )

    is_day_first: Optional[bool] = False
    if infer_day_first:
        is_day_first = _is_day_first(df, column)

    df, stats = clean_column(
        df,
        column,
        partial(
            _format_date,
            output_format=output_format,
            input_timezone=input_timezone,
            output_timezone=output_timezone,
            fix_missing=fix_missing,
            is_day_first=is_day_first,
            errors=errors,
            parsed_output_format=parsed_output_format,
        ),
        [f"{column}_clean"],
        inplace=inplace,
        progress=progress,
    )

    # output a report describing the result of clean_date
    if report:
        create_report_new("Dates", stats, errors)

    return df


def validate_date(date: str | pd.Series) -> bool | pd.Series:
    """
    Validate dates and times.

    Parameters
    ----------
    date
        pandas Series of dates or a date string

    Examples
    --------

    >>> validate_date('3rd of May 2001')
    True
    >>> df = pd.DataFrame({'date': ['2003/09/25', 'This is Sep.']})
    >>> validate_date(df['date'])
    0     True
    1    False
    Name: date, dtype: bool
    """
    return validate_values(date, partial(check_date, clean=False))


def _is_day_first(df: pd.DataFrame | dd.DataFrame, column: str) -> bool:
    """
    Infer whether the dates in the column are written with the day before the month.

    Returns True if at least one date can only be read as day-first
    (for example "25/09/2003"), otherwise False.
    """

    def _any_day_first(pdf: pd.DataFrame) -> bool:
        return any(_check_is_day_first(val) is True for val in pdf[column].tolist())

    return any(apply_partitions(df, _any_day_first))


def _check_is_day_first(val: Any) -> Optional[bool]:
    """
    Inferring if the first number of ambiguous string is the day.

    Parameters
    ----------
    val
        date string
    """
    if is_null(val):
        return None
    date = str(val)
    is_day_first = None
    status = check_date(date, True)
    if status == "null":
        return is_day_first
    elif status == "unknown":
        return is_day_first
    else:
        tokens = split(date, JUMP)
        _, _, is_day_first = _ensure_ymd(tokens, None)
    return is_day_first


def _format_date(
    val: Any,
    output_format: str,
    input_timezone: str,
    output_timezone: str,
    fix_missing: str,
    is_day_first: Optional[bool],
    errors: str,
    parsed_output_format: Optional[ParsedTargetFormat] = None,
) -> Tuple[Any, int]:
    """
    This function cleans date string.
    Parameters
    ----------
    df, col, target_format, tz_info, fix_missing
        same as explained in clean_date function
    """
    # pylint: disable=too-many-arguments
    if is_null(val):
        return np.nan, 0
    date = str(val)
    status = check_date(date, True)
    if status == "null":
        return np.nan, 0
    elif status == "unknown":
        if errors == "raise":
            raise ValueError(f"unable to parse value {val}")
        return val if errors == "ignore" else np.nan, 1
    else:
        # Handle date data and timezone
        parsed_date_data = _parse(date, fix_missing, is_day_first)
        parsed_date_data.set_tzinfo(timezone=input_timezone)
        parsed_date_data = _set_parseddate_timezone_offset(input_timezone, parsed_date_data)
        # Handle target format and timezone
        parsed_output_format_data = parsed_output_format or _parse_output_format(
            output_format, output_timezone
        )
        if parsed_output_format_data.valid:
            if parsed_date_data.valid == "cleaned":
                transformed_date = _transform(
                    parsed_date_data, parsed_output_format_data, output_format, output_timezone
                )
                return transformed_date, 2 if val != transformed_date else 3
            else:
                if errors == "raise":
                    raise ValueError(f"unable to parse value {val}")
                return val if errors == "ignore" else np.nan, 1
        else:
            raise ValueError(
                f"output_format {output_format} is invalid. "
                f"Invalid tokens are {parsed_output_format_data.invalid_tokens}."
            )


def _parse_output_format(output_format: str, output_timezone: str) -> ParsedTargetFormat:
    """
    Parse the output format string and attach the offset of the output timezone.
    """
    return _set_parsedtargetformat_timezone_offset(
        output_timezone, _check_output_format(output_format)
    )


def _utcoffset(timezone: str, date: datetime.datetime) -> Optional[timedelta]:
    """
    Returns the UTC offset of an IANA timezone at the given (naive, local) date.
    """
    return ZoneInfo(timezone).utcoffset(date)


def _set_parseddate_timezone_offset(timezone: str, parsed_data: ParsedDate) -> ParsedDate:
    """
    This function set timezone information for parsed date or parsed target format
    Parameters
    ----------
    timezone
        string name of timezone
    parsed_data
        parsed date or parsed target format
    """
    example_date = datetime.datetime(2009, 9, 1)
    if timezone in ALL_TIMEZONES:
        days, seconds = 0, 0
        tz_offset = _utcoffset(timezone, example_date)
        if tz_offset is not None:
            days = tz_offset.days
            seconds = tz_offset.seconds
        parsed_data.set_tzinfo(utc_offset_hours=int(abs(days) * 24 + abs(seconds) / 3600))
        parsed_data.set_tzinfo(
            utc_offset_minutes=int((abs(seconds) - (abs(seconds) / 3600) * 3600) / 60)
        )
        if days >= 0 and seconds >= 0:
            parsed_data.set_tzinfo(utc_add="+")
        elif days <= 0 and seconds < 0:
            parsed_data.set_tzinfo(utc_add="-")
    elif timezone in ZONE:
        parsed_data.set_tzinfo(utc_offset_hours=abs(ZONE[timezone]), utc_offset_minutes=0)
        if ZONE[timezone] >= 0:
            parsed_data.set_tzinfo(utc_add="+")
        elif ZONE[timezone] < 0:
            parsed_data.set_tzinfo(utc_add="-")
    return parsed_data


def _set_parsedtargetformat_timezone_offset(
    timezone: str, parsed_data: ParsedTargetFormat
) -> ParsedTargetFormat:
    """
    This function set timezone information for parsed date or parsed target format
    Parameters
    ----------
    timezone
        string name of timezone
    parsed_data
        parsed date or parsed target format
    """
    example_date = datetime.datetime(2009, 9, 1)
    if timezone in ALL_TIMEZONES:
        days, seconds = 0, 0
        tz_offset = _utcoffset(timezone, example_date)
        if tz_offset is not None:
            days = tz_offset.days
            seconds = tz_offset.seconds
        parsed_data.set_tzinfo(utc_offset_hours=int(abs(days) * 24 + abs(seconds) / 3600))
        parsed_data.set_tzinfo(
            utc_offset_minutes=int((abs(seconds) - (abs(seconds) / 3600) * 3600) / 60)
        )
        if days >= 0 and seconds >= 0:
            parsed_data.set_tzinfo(utc_add="+")
        elif days <= 0 and seconds < 0:
            parsed_data.set_tzinfo(utc_add="-")
    elif timezone in ZONE:
        parsed_data.set_tzinfo(utc_offset_hours=abs(ZONE[timezone]))
        parsed_data.set_tzinfo(utc_offset_minutes=0)
        if ZONE[timezone] >= 0:
            parsed_data.set_tzinfo(utc_add="+")
        elif ZONE[timezone] < 0:
            parsed_data.set_tzinfo(utc_add="-")
    return parsed_data


def _check_output_format(output_format: str) -> ParsedTargetFormat:
    """
    This function check validation of output_format.
    Parameters
    ----------
    output_format
        output_format string
    """
    result = ParsedTargetFormat()
    target_tokens = split(output_format, JUMP)
    remain_tokens = list(target_tokens)
    # Handle Timezone
    result, remain_tokens = _figure_output_format_timezone(result, target_tokens, remain_tokens)
    # Handle year, month, day
    result, remain_tokens = _figure_output_format_ymd(result, target_tokens, remain_tokens)
    # Handle AM, PM with JUMP seperators
    result, remain_tokens = _figure_output_format_ampm(result, target_tokens, remain_tokens)
    # Handle hour, minute, second
    result, remain_tokens = _figure_output_format_hms(result, remain_tokens)
    # If len(remain_tokens) = 0, then is valid format
    if len(remain_tokens) > 0:
        result.set_valid(False)
        for token in remain_tokens:
            result.add_invalid_token(token)
    return result


def _figure_output_format_timezone(
    parsed_data: ParsedTargetFormat,
    target_tokens: List[str],
    remain_tokens: List[str],
) -> Tuple[ParsedTargetFormat, List[str]]:
    """
    This function figure timezone token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    target_tokens
        parsed target tokens
    remain_tokens
        remained tokens after figuring tokens
    """
    for token in target_tokens:
        if token in ALL_TIMEZONES:
            parsed_data.set_tzinfo(timezone=token)
            remain_tokens.remove(token)
    for token in target_tokens:
        if token in ("z", "Z"):
            parsed_data.set_timezone_token(token)
            remain_tokens.remove(token)
    return parsed_data, remain_tokens


def _figure_output_format_ymd(
    parsed_data: ParsedTargetFormat,
    target_tokens: List[str],
    remain_tokens: List[str],
) -> Tuple[ParsedTargetFormat, List[str]]:
    """
    This function figure year, month and day token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    target_tokens
        parsed target tokens
    remain_tokens
        remained tokens after figuring tokens
    """
    for token in target_tokens:
        if token in TARGET_YEAR:
            parsed_data.set_year_token(token)
            remain_tokens.remove(token)
        if token in TARGET_MONTH:
            parsed_data.set_month_token(token)
            remain_tokens.remove(token)
        if token in TARGET_DAY:
            parsed_data.set_day_token(token)
            remain_tokens.remove(token)
        if token in TARGET_WEEKDAY:
            parsed_data.set_weekday_token(token)
            remain_tokens.remove(token)
    return parsed_data, remain_tokens


def _figure_output_format_ampm(
    parsed_data: ParsedTargetFormat,
    target_tokens: List[str],
    remain_tokens: List[str],
) -> Tuple[ParsedTargetFormat, List[str]]:
    """
    This function figure AM or PM token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    target_tokens
        parsed target tokens
    remain_tokens
        remained tokens after figuring tokens
    """
    for token in target_tokens:
        if token in AM:
            remain_tokens.remove(token)
        if token in PM:
            parsed_data.set_ispm(True)
            remain_tokens.remove(token)
    return parsed_data, remain_tokens


def _figure_output_format_hms(
    parsed_data: ParsedTargetFormat, remain_tokens: List[str]
) -> Tuple[ParsedTargetFormat, List[str]]:
    """
    This function figure hour, minute and second token in target format
    Parameters
    ----------
    parsed_data
        parsed target format
    remain_tokens
        remained tokens after figuring tokens
    """
    if len(remain_tokens) > 0:
        remain_str = ""
        for token in remain_tokens:
            if (
                token not in TARGET_MONTH
                and token not in TARGET_WEEKDAY
                and token not in AM
                and token not in PM
            ):
                remain_str = token
        parsed_data, hms_tokens = _get_output_format_hms_tokens(parsed_data, remain_str)
        for token in hms_tokens:
            if token in TARGET_HOUR:
                parsed_data.set_hour_token(token)
            if token in TARGET_MINUTE:
                parsed_data.set_minute_token(token)
            if token in TARGET_SECOND:
                parsed_data.set_second_token(token)
        if len(remain_str) > 0:
            remain_tokens.remove(remain_str)
    return parsed_data, remain_tokens


def _get_output_format_hms_tokens(
    parsed_data: ParsedTargetFormat, remain_str: str
) -> Tuple[ParsedTargetFormat, List[str]]:
    """
    This function get hour, minute and second token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    remain_str
        remained string after figuring tokens
    """
    if "z" in remain_str:
        parsed_data.timezone_token = "z"
        hms_tokens = split(remain_str, [":", parsed_data.timezone_token])
    elif "Z" in remain_str:
        parsed_data.timezone_token = "Z"
        hms_tokens = split(remain_str, [":", parsed_data.timezone_token])
    else:
        hms_tokens = split(remain_str, [":"])
    # ensure AM, PM tokens without JUMP seperators
    for token in AM:
        if token in remain_str:
            hms_tokens = split(remain_str, AM)
            break
    for token in PM:
        if token in remain_str:
            hms_tokens = split(remain_str, PM)
            break
    if len(hms_tokens) == 0:
        hms_tokens = split(remain_str, [":"])
    return parsed_data, hms_tokens


def _ensure_ymd(
    tokes: List[str], is_day_first: Optional[bool]
) -> Tuple[ParsedDate, List[str], Optional[bool]]:
    """
    This function extract value of year, month, day
    Parameters
    ----------
    tokes
        generated tokens
    is_day_first
        signal of inferring result.
    """
    result = ParsedDate()
    result, remain_tokens = _ensure_year(result, tokes, list(tokes))
    if len(remain_tokens) == 0:
        return result, remain_tokens, is_day_first is not None
    num_tokens = []
    for token in remain_tokens:
        if token.isnumeric():
            num_tokens.append(token)
    for token in num_tokens:
        remain_tokens.remove(token)
    if result.ymd["year"] != -1:
        result, is_day_first = _ensure_month_day(result, num_tokens, is_day_first)
    else:
        result, is_day_first = _ensure_year_month_day(result, num_tokens, is_day_first)
    return result, remain_tokens, is_day_first


def _ensure_year(
    parsed_data: ParsedDate,
    tokes: List[str],
    remain_tokens: List[str],
) -> Tuple[ParsedDate, List[str]]:
    """
    This function extract year number whose length is 4
    Parameters
    ----------
    parsed_data
        parsed date
    tokes
        parsed tokens
    remain_tokens
        remained tokens
    """
    for token in tokes:
        if token in MONTHS:
            parsed_data.set_month(MONTHS[token])
            remain_tokens.remove(token)
        if token in WEEKDAYS:
            parsed_data.set_weekday(WEEKDAYS[token])
            remain_tokens.remove(token)
    for token in remain_tokens:
        if len(token) == 4 and token.isnumeric():
            parsed_data.set_year(int(token))
            remain_tokens.remove(token)
            break
    return parsed_data, remain_tokens


def _ensure_month_day(
    parsed_data: ParsedDate, num_tokens: List[str], is_day_first: Optional[bool]
) -> Tuple[ParsedDate, Optional[bool]]:
    """
    This function extract month and day when year is not None.
    Parameters
    ----------
    parsed_data
        parsed date
    num_tokens
        remained numerical tokens
    is_day_first
        signal of inferring result.
    """
    if len(num_tokens) == 1:
        if parsed_data.ymd["month"] != -1:
            parsed_data.set_day(int(num_tokens[0]))
        else:
            parsed_data.set_month(int(num_tokens[0]))
        if is_day_first is None:
            is_day_first = False
    else:
        if int(num_tokens[0]) > 12:
            parsed_data.set_month(int(num_tokens[1]))
            parsed_data.set_day(int(num_tokens[0]))
            if is_day_first is None:
                is_day_first = True
        elif int(num_tokens[1]) > 12:
            parsed_data.set_month(int(num_tokens[0]))
            parsed_data.set_day(int(num_tokens[1]))
            if is_day_first is None:
                is_day_first = False
        else:
            if is_day_first is None:
                is_day_first = False
                parsed_data.set_month(int(num_tokens[0]))
                parsed_data.set_day(int(num_tokens[1]))
            elif is_day_first:
                parsed_data.set_month(int(num_tokens[1]))
                parsed_data.set_day(int(num_tokens[0]))
            elif not is_day_first:
                parsed_data.set_month(int(num_tokens[0]))
                parsed_data.set_day(int(num_tokens[1]))
    return parsed_data, is_day_first


def _ensure_year_month_day(
    parsed_data: ParsedDate, num_tokens: List[str], is_day_first: Optional[bool]
) -> Tuple[ParsedDate, Optional[bool]]:
    """
    This function extract month and day when year is None.
    Parameters
    ----------
    parsed_data
        parsed date
    num_tokens
        remained numerical tokens
    is_day_first
        signal of inferring result.
    """
    # pylint: disable=too-many-branches
    if len(num_tokens) == 1:
        parsed_data.set_year(int(num_tokens[-1]) + 2000)
        if is_day_first is None:
            is_day_first = False
    elif len(num_tokens) == 2:
        parsed_data.set_year(int(num_tokens[-1]) + 2000)
        if parsed_data.ymd["month"] == -1:
            parsed_data.set_month(int(num_tokens[0]))
        else:
            parsed_data.set_day(int(num_tokens[0]))
        if is_day_first is None:
            is_day_first = False
    elif len(num_tokens) == 3:
        parsed_data.set_year(int(num_tokens[-1]) + 2000)
        if int(num_tokens[0]) > 12:
            parsed_data.set_month(int(num_tokens[1]))
            parsed_data.set_day(int(num_tokens[0]))
            if is_day_first is None:
                is_day_first = True
        elif int(num_tokens[1]) > 12:
            parsed_data.set_month(int(num_tokens[0]))
            parsed_data.set_day(int(num_tokens[1]))
            if is_day_first is None:
                is_day_first = False
        else:
            if is_day_first is None:
                is_day_first = False
                parsed_data.set_month(int(num_tokens[0]))
                parsed_data.set_day(int(num_tokens[1]))
            elif is_day_first:
                parsed_data.set_month(int(num_tokens[1]))
                parsed_data.set_day(int(num_tokens[0]))
            elif not is_day_first:
                parsed_data.set_month(int(num_tokens[0]))
                parsed_data.set_day(int(num_tokens[1]))
    return parsed_data, is_day_first


def _ensure_hms(inner_result: ParsedDate, remain_tokens: List[str]) -> ParsedDate:
    """
    This function extract value of hour, minute, second
    Parameters
    ----------
    inner_result
        already generated year, month, day value
    remain_tokens
        remained tokens used for generating hour, minute, second
    """
    result = deepcopy(inner_result)
    remain_str = remain_tokens[0]
    hms_tokens = []
    # Judge the expression of am pm
    ispm = False
    for token in AM:
        if token in remain_str:
            hms_tokens = split(remain_str, AM)
            break
    for token in PM:
        if token in remain_str:
            ispm = True
            hms_tokens = split(remain_str, PM)
            break
    if len(hms_tokens) == 0:
        hms_tokens = split(remain_str, [":"])
    else:
        hms_tokens = split(hms_tokens[0], [":"])
    if ispm:
        result = _ensure_pm(result, hms_tokens, 12)
    else:
        result = _ensure_pm(result, hms_tokens, 0)
    return result


def _ensure_pm(parsed_data: ParsedDate, hms_tokens: List[str], offset: int) -> ParsedDate:
    """
    This function extract values which stand for pm time
    Parameters
    ----------
    parsed_data
        already generated parsed value
    hms_tokens
        tokens of hour, minute, second
    offset
        if it is pm time, offset = 12
        otherwise, offset = 0
    """
    if len(hms_tokens) == 1:
        parsed_data.set_hour(int(hms_tokens[0]) + offset)
    elif len(hms_tokens) == 2:
        parsed_data.set_hour(int(hms_tokens[0]) + offset)
        parsed_data.set_minute(int(hms_tokens[1]))
    elif len(hms_tokens) == 3:
        parsed_data.set_hour(int(hms_tokens[0]) + offset)
        parsed_data.set_minute(int(hms_tokens[1]))
        parsed_data.set_second(int(hms_tokens[2]))
    return parsed_data


def _fix_missing_element(parsed_res: ParsedDate, fix_missing: str) -> ParsedDate:
    """
    This function fix empty part of transformed format
    Parameters
    ----------
    parsed_res
        generated year, month, day, hour, minute, second
    fix_missing
        the format of fixing empty part
    """
    if parsed_res.valid == "unknown":
        return parsed_res
    if fix_missing == "current":
        parsed_res = fix_missing_current(parsed_res)
    elif fix_missing == "minimum":
        parsed_res = fix_missing_minimum(parsed_res)
    return parsed_res


def _parse(date: str, fix_missing: str, is_day_first: Optional[bool]) -> ParsedDate:
    """
    This function parse string into tokens
    Parameters
    ----------
    date
        date string
    fix_missing
        format of fixing empty
    is_day_first
        signal of inferring result.
    """
    tokens = split(date, JUMP)
    parsed_date_res, remain_tokens, _ = _ensure_ymd(tokens, is_day_first)
    if len(remain_tokens) > 0:
        parsed_time_res = _ensure_hms(parsed_date_res, remain_tokens)
    else:
        parsed_time_res = parsed_date_res
    parsed_res = _fix_missing_element(parsed_time_res, fix_missing)
    return parsed_res


def _change_timezone(parsed_date_data: ParsedDate, output_timezone: str) -> ParsedDate:
    """
    This function change timezone for already parsed date string
    Parameters
    ----------
    parsed_date_data
        parsed date string
    output_timezone
        target timezone string
    """
    origin_tz_offset = timedelta(days=0, seconds=0)
    target_tz_offset = timedelta(days=0, seconds=0)
    origin_date = datetime.datetime(
        year=parsed_date_data.ymd["year"],
        month=parsed_date_data.ymd["month"],
        day=parsed_date_data.ymd["day"],
        hour=parsed_date_data.hms["hour"],
        minute=parsed_date_data.hms["minute"],
        second=parsed_date_data.hms["second"],
    )
    origin_add, target_add = 0, 0
    if parsed_date_data.tzinfo["timezone"] in ALL_TIMEZONES:
        tz_offset = _utcoffset(str(parsed_date_data.tzinfo["timezone"]), origin_date)
        if tz_offset is not None:
            origin_add = -1 if tz_offset.days > 0 and tz_offset.seconds > 0 else 1
            origin_tz_offset = timedelta(
                days=abs(tz_offset.days), seconds=abs(tz_offset.seconds)
            )
    elif parsed_date_data.tzinfo["timezone"] in ZONE:
        origin_add = -1 if ZONE[str(parsed_date_data.tzinfo["timezone"])] > 0 else 1
        offset_value = abs(ZONE[str(parsed_date_data.tzinfo["timezone"])]) * 3600
        origin_tz_offset = timedelta(days=0, seconds=offset_value)
    if output_timezone in ALL_TIMEZONES:
        tz_offset = _utcoffset(output_timezone, origin_date)
        if tz_offset is not None:
            target_add = 1 if tz_offset.days >= 0 and tz_offset.seconds >= 0 else -1
            target_tz_offset = timedelta(
                days=abs(tz_offset.days), seconds=abs(tz_offset.seconds)
            )
    elif output_timezone in ZONE:
        target_add = 1 if ZONE[output_timezone] >= 0 else -1
        offset_value = abs(ZONE[output_timezone]) * 3600
        target_tz_offset = timedelta(days=0, seconds=offset_value)
    result = deepcopy(parsed_date_data)
    if -1 in [
        parsed_date_data.ymd["year"],
        parsed_date_data.ymd["month"],
        parsed_date_data.ymd["day"],
        parsed_date_data.hms["hour"],
        parsed_date_data.hms["minute"],
        parsed_date_data.hms["second"],
    ]:
        return parsed_date_data
    utc_date = origin_date + origin_tz_offset if origin_add == 1 else origin_date - origin_tz_offset
    target_date = utc_date + target_tz_offset if target_add == 1 else utc_date - target_tz_offset
    result.set_year(target_date.year)
    result.set_month(target_date.month)
    result.set_day(target_date.day)
    result.set_hour(target_date.hour)
    result.set_minute(target_date.minute)
    result.set_second(target_date.second)
    result.set_tzinfo(timezone=output_timezone)
    days = target_tz_offset.days
    seconds = target_tz_offset.seconds
    result.set_tzinfo(utc_offset_hours=int(abs(days) * 24 + abs(seconds) / 3600))
    result.set_tzinfo(utc_offset_minutes=int((abs(seconds) - (abs(seconds) / 3600) * 3600) / 60))
    if target_add >= 0:
        result.set_tzinfo(utc_add="+")
    elif target_add < 0:
        result.set_tzinfo(utc_add="-")
    return result


def _transform_year(result_str: str, year_token: str, year: int) -> str:
    """
    This function transform parsed year into target format
    Parameters
    ----------
    result_str
        result string
    year_token
        token of year
    year
        value of year
    """
    result = result_str
    if year_token != "":
        if year == -1:
            if len(year_token) == 4:
                result = result.replace(year_token, "----")
            elif len(year_token) == 2:
                result = result.replace(year_token, "--")
            elif len(year_token) == 1:
                result = result.replace(year_token, "-")
        else:
            if len(year_token) == 4:
                result = result.replace(year_token, str(year))
            else:
                year = year - 2000
                if year < 10:
                    result = result.replace(year_token, f"{0}{year}")
                else:
                    result = result.replace(year_token, str(year))
    return result


def _transform_month(result_str: str, month_token: str, month: int) -> str:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    month_token
        token of month
    month
        value of month
    """
    result = result_str
    if month_token != "":
        if month == -1:
            if len(month_token) == 3:
                result = result.replace(month_token, "---")
            elif len(month_token) == 5:
                result = result.replace(month_token, "-----")
            elif len(month_token) == 2:
                result = result.replace(month_token, "--")
            elif len(month_token) == 1:
                result = result.replace(month_token, "-")
        else:
            if len(month_token) == 2:
                if month < 10:
                    result = result.replace(month_token, f"{0}{month}", 1)
                else:
                    result = result.replace(month_token, str(month), 1)
            elif len(month_token) == 3:
                result = result.replace(month_token, TEXT_MONTHS[month - 1][0], 1)
            elif len(month_token) == 5:
                result = result.replace(month_token, TEXT_MONTHS[month - 1][1], 1)
            else:
                result = result.replace(month_token, str(month), 1)
    return result


def _transform_day(result_str: str, day_token: str, day: int) -> str:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    day_token
        token of day
    day
        value of day
    """
    result = result_str
    if day_token != "":
        if day == -1:
            if len(day_token) == 2:
                result = result.replace(day_token, "--")
            elif len(day_token) == 1:
                result = result.replace(day_token, "-")
        else:
            if len(day_token) == 2:
                if day < 10:
                    result = result.replace(day_token, f"{0}{day}", 1)
                else:
                    result = result.replace(day_token, str(day), 1)
            else:
                result = result.replace(day_token, str(day))
    return result


def _transform_hms(result_str: str, hms_token: str, ispm: bool, hms_value: int) -> str:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    hms_token, ispm
        token of hour, minute or second, justify whether it is time in P.M.
    hms_value
        value of hour, minute or second
    """
    result = result_str
    if hms_token != "":
        if hms_value == -1:
            if len(hms_token) == 2:
                result = result.replace(hms_token, "--")
            elif len(hms_token) == 1:
                result = result.replace(hms_token, "-")
        else:
            if ispm:
                hms_value = hms_value - 12
            if len(hms_token) == 2:
                if hms_value < 10:
                    result = result.replace(hms_token, f"{0}{hms_value}", 1)
                else:
                    result = result.replace(hms_token, str(hms_value), 1)
            else:
                result = result.replace(hms_token, str(hms_value))
    return result


def _transform_weekday(result_str: str, weekday_token: str, weekday: int) -> str:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    weekday_token
        token of weekday
    weekday
        value of weekday
    """
    result = result_str
    if weekday_token != "":
        if weekday == -1:
            if len(weekday_token) == 3:
                result = result.replace(weekday_token, "---")
            elif len(weekday_token) == 5:
                result = result.replace(weekday_token, "-----")
        else:
            if len(weekday_token) == 3:
                result = result.replace(weekday_token, TEXT_WEEKDAYS[weekday - 1][0])
            elif len(weekday_token) == 5:
                result = result.replace(weekday_token, TEXT_WEEKDAYS[weekday - 1][1])
    return result


def _transform_timezone(
    result_str: str,
    timezone_token: str,
    timezone: str,
    utc_add: str,
    utc_offset_hours: int,
    utc_offset_minutes: int,
) -> str:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    timezone_token
        token of timezone in target format
    timezone
        value of timezone string
    tz_info
        information of timezone, including offset hours and mins comparing to UTC
    """
    # pylint: disable=too-many-arguments
    result = result_str
    if timezone_token != "":
        if timezone_token == "z":
            result = result.replace(timezone_token, timezone)
        elif timezone_token == "Z":
            offset_hours_str = str(int(utc_offset_hours))
            if len(offset_hours_str) == 1:
                offset_hours_str = f"{0}{offset_hours_str}"
            offset_minutes_str = str(int(utc_offset_minutes))
            if len(offset_minutes_str) == 1:
                offset_minutes_str = f"{0}{offset_minutes_str}"
            result = result.replace(
                timezone_token, f"UTC{utc_add}{offset_hours_str}:{offset_minutes_str}"
            )
    return result


def _transform(
    parsed_date_data: ParsedDate,
    parsed_output_format_data: ParsedTargetFormat,
    output_format: str,
    output_timezone: str,
) -> str:
    """
    This function transform parsed result into target format
    Parameters
    ----------
    parsed_date_data
        generated year, month, day, hour, minute, second
    parsed_output_format_data
        generated year token, month token, day token, hour token,
        minute token, second token of target format
    output_format
        target format string
    output_timezone
        target timezone string
    """
    result = output_format
    if output_timezone != "":
        parsed_date_data = _change_timezone(parsed_date_data, output_timezone)
    # Handle year
    result = _transform_year(
        result, parsed_output_format_data.ymd_token["year_token"], parsed_date_data.ymd["year"]
    )
    # Handle day
    result = _transform_day(
        result, parsed_output_format_data.ymd_token["day_token"], parsed_date_data.ymd["day"]
    )
    # Handle hours
    result = _transform_hms(
        result,
        str(parsed_output_format_data.hms_token["hour_token"]),
        bool(parsed_output_format_data.hms_token["ispm"]),
        parsed_date_data.hms["hour"],
    )
    # Handle minutes
    result = _transform_hms(
        result,
        str(parsed_output_format_data.hms_token["minute_token"]),
        False,
        parsed_date_data.hms["minute"],
    )
    # Handle seconds
    result = _transform_hms(
        result,
        str(parsed_output_format_data.hms_token["second_token"]),
        False,
        parsed_date_data.hms["second"],
    )
    # Handle month
    result = _transform_month(
        result, parsed_output_format_data.ymd_token["month_token"], parsed_date_data.ymd["month"]
    )
    # Handle weekday
    result = _transform_weekday(
        result, parsed_output_format_data.weekday_token, parsed_date_data.weekday
    )
    # Handle timezone
    result = _transform_timezone(
        result,
        parsed_output_format_data.timezone_token,
        str(parsed_date_data.tzinfo["timezone"]),
        str(parsed_date_data.tzinfo["utc_add"]),
        int(parsed_date_data.tzinfo["utc_offset_hours"]),
        int(parsed_date_data.tzinfo["utc_offset_minutes"]),
    )
    return result
