"""
Clean and validate a DataFrame column containing JSON.
"""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .utils import add_columns, check_errors, is_null, map_column, validate_values

if TYPE_CHECKING:
    import dask.dataframe as dd


def clean_json(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    Parse a column of JSON strings and split the JSON objects into columns.

    The values of the column are replaced by the parsed JSON (for example a
    dictionary for a JSON object), and one column is appended for every key of
    the JSON objects. Nested objects are flattened, with the keys joined by a
    dot ("address.city"). The input DataFrame is not modified.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing JSON.
    errors
        How to handle parsing errors.
            - 'coerce': invalid parsing will be set to NaN.
            - 'ignore': invalid parsing will return the input.
            - 'raise': invalid parsing will raise an exception.

        (default: 'coerce')

    Examples
    --------
    Split a json into its components.

    >>> df = pd.DataFrame({"messy_json": ['{"name": "jane doe", "salary": 9000}', "not json"]})
    >>> clean_json(df, "messy_json")
                               messy_json      name  salary
    0  {'name': 'jane doe', 'salary': 9000}  jane doe  9000.0
    1                                   NaN       NaN     NaN
    """
    check_errors(errors)

    pdf, rows = map_column(df, column, partial(_format_json, errors=errors))
    parsed = [row[0] for row in rows]

    # one column for each (flattened) key of the JSON objects
    records = [val if isinstance(val, dict) else {} for val in parsed]
    flattened = pd.json_normalize(records) if records else pd.DataFrame()

    new_columns: dict[str, Any] = {column: parsed}
    for key in flattened.columns:
        name = str(key)
        # do not overwrite a column of the input DataFrame that has the same name as a key
        if name in pdf.columns:
            name = f"{column}_{name}"
        new_columns[name] = flattened[key].tolist()
    return add_columns(pdf, new_columns)


def validate_json(x: str | pd.Series) -> bool | pd.Series:
    """
    Validate JSON.

    Parameters
    ----------
    x
        pandas Series of JSON strings, or a single JSON string.

    Examples
    --------

    >>> validate_json('{"name": "jane doe"}')
    True
    >>> df = pd.DataFrame(
    ...     {
    ...         "messy_json": [
    ...             '{"name": "jane doe", "salary": 9000, "email": "jane.doe@pynative.com",}',
    ...             '{"name": "jane doe", "salary": 9000, "email": "jane.doe@pynative.com"}',
    ...         ]
    ...     }
    ... )
    >>> validate_json(df["messy_json"])
    0    False
    1     True
    Name: messy_json, dtype: bool
    """
    return validate_values(x, partial(_check_json, clean=False))


def _format_json(val: Any, errors: str) -> tuple[Any]:
    """
    Parse a JSON string. Null values and, if errors is "coerce",
    invalid values are returned as NaN.
    """
    if is_null(val):
        return (np.nan,)
    try:
        return (json.loads(val),)
    except (ValueError, TypeError):
        if errors == "raise":
            raise ValueError(f"Unable to parse value {val}") from None
        return (val if errors == "ignore" else np.nan,)


def _check_json(json_data: Any, clean: bool) -> Any:
    """
    Function to check whether a value is a valid json
    """
    if is_null(json_data):
        return "null" if clean else False
    try:
        json.loads(json_data)
    except (ValueError, TypeError):
        return "unknown" if clean else False
    return "success" if clean else True
