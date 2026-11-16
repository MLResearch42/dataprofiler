"""
Shared implementation of the ``clean_<identifier>`` and ``validate_<identifier>``
functions that are built on top of the python-stdnum library.

Every identifier module (for example ``clean_au_abn.py``) only declares what is
specific to the identifier: how a valid number is rendered in each output format
and, where supported, how it is split into components. The work of walking over
the column, handling missing and invalid values and assembling the result is done
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from stdnum.exceptions import ValidationError

from .utils import add_columns, check_errors, is_dask, is_null, map_column, unzip, validate_values

if TYPE_CHECKING:
    import dask.dataframe as dd


def clean_stdnum(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    is_valid: Callable[[str], bool],
    formatters: Mapping[str, Callable[[str], Any]],
    output_format: str,
    inplace: bool,
    errors: str,
    progress: bool,
    split: bool = False,
    split_columns: Sequence[str] = (),
    splitter: Callable[[str], Sequence[Any]] | None = None,
) -> pd.DataFrame:
    """
    Clean a column of identifier numbers.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing the numbers.
    is_valid
        Function returning True if a number string is valid.
    formatters
        Mapping of each supported ``output_format`` to the function that renders
        a valid number string in that format.
    output_format
        The requested output format, a key of ``formatters``.
    inplace
        If True, the cleaned column takes the place of the original column.
    errors
        How to handle invalid values: "coerce" (set to NaN), "ignore" (keep the
        input value) or "raise" (raise a ValueError).
    progress
        If True, display a progress bar (Dask input only).
    split
        If True and the identifier supports it, add one column per component.
    split_columns
        Names of the component columns.
    splitter
        Function returning the components of a valid number string, in the order
        of ``split_columns``.
    """
    if output_format not in formatters:
        raise ValueError(
            f"output_format {output_format} is invalid. It needs to be {_quote_options(formatters)}."
        )
    check_errors(errors)

    render = formatters[output_format]

    def formatter(text: str) -> Any:
        # Some output formats do not exist for every valid number, e.g. the birth date
        # encoded in a personal number when the number belongs to a company
        try:
            return render(text)
        except ValidationError:
            return np.nan

    do_split = bool(split) and splitter is not None
    n_parts = len(split_columns) if do_split else 0
    nulls = (np.nan,) * (1 + n_parts)

    def _format(val: Any) -> tuple[Any, ...]:
        if is_null(val):
            return nulls
        text = val if isinstance(val, str) else str(val)
        if not is_valid(text):
            if errors == "raise":
                raise ValueError(f"Unable to parse value {val}")
            return (val if errors == "ignore" else np.nan,) + nulls[1:]
        if not do_split:
            return (formatter(text),)
        parts = tuple(splitter(text))  # type: ignore[misc]
        if not parts:
            return nulls
        parts = (parts + nulls)[:n_parts]
        return (formatter(text),) + parts

    pdf, rows = map_column(df, column, _format, progress)
    values = unzip(rows, 1 + n_parts)
    new_columns = {f"{column}_clean": values[0]}
    if do_split:
        new_columns.update(zip(split_columns, values[1:]))
    return add_columns(pdf, new_columns, replace=column if inplace else None)


def validate_stdnum(
    df: Any,
    column: str,
    is_valid: Callable[[str], bool],
) -> bool | pd.Series | pd.DataFrame | dd.Series | dd.DataFrame:
    """
    Validate identifier numbers.

    Parameters
    ----------
    df
        A single value, a pandas or Dask Series, or a pandas or Dask DataFrame.
    column
        For DataFrame input, the name of the column to validate. If empty, every
        cell of the DataFrame is validated.
    is_valid
        Function returning True if a number string is valid.
    """

    def _check(val: Any) -> bool:
        if isinstance(val, str):
            return bool(is_valid(val))
        if is_null(val):
            return False
        return bool(is_valid(str(val)))

    is_frame = isinstance(df, pd.DataFrame) or (is_dask(df) and hasattr(df, "columns"))
    if is_frame and column != "":
        return validate_values(df[column], _check)
    return validate_values(df, _check, tabular=True)


def _quote_options(options: Mapping[str, Any]) -> str:
    """Format the option names as: "a", "b" or "c"."""
    names = [f'"{name}"' for name in options]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]
