"""
Conduct a set of operations that would be useful for
cleaning and standardizing a full Pandas DataFrame.
"""

# pylint: disable-msg=relative-beyond-top-level
# pylint: disable-msg=cyclic-import

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_float_dtype, is_integer_dtype, is_object_dtype

from .clean_headers import clean_headers
from .utils import apply_partitions, is_dask, is_null


def clean_df(
    df: pd.DataFrame,
    clean_header: bool = True,
    data_type_detection: str = "semantic",
    standardize_missing_values: str = "fill",
    downcast_memory: bool = True,
    remove_duplicate_entries: bool = False,
    report: bool = True,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """
    This function cleans the whole DataFrame with a sequence of useful operations.

    Parameters
    ----------
    df
        A Pandas DataFrame containing the data to be cleaned.
    clean_header
        If True, call the clean_headers() function to clean the column names.

        (default: True)
    data_type_detection
        The desired way to check data types. Must from: {'semantic', 'atomic', 'none'}.
        * If ’semantic’, then perform a column-wise semantic and atomic type detection.
        * If 'atomic', then return the column data types from Python atomic ones.
        * If 'none', then no results will be returned.

        (default: 'semantic')
    standardize_missing_values
        The desired way to standardize missing values. Must from: {‘fill’, ‘remove’, ’ignore’}.
        * If ’fill’, then fill detected missing values using np.NaN or pd.NaT (for "Date" type).
        * If ‘remove’, then rows with any missing values will be removed.
        * If ‘ignore’, then no action will be taken.

        (default: 'fill')
    remove_duplicate_entries
        If True, remove the repetitive data entries (rows) and report how many entries are removed.

        (default: False)
    downcast_memory
        If True, downcast the memory of the DataFrame by using subtypes in numerical columns;
        for categorical types, downcast from `object` to `category`.

        (default: True)
    report
        If True, output the summary report. Otherwise, no report is outputted.

        (default: True)

    Examples
    --------
    >>> df = pd.DataFrame({"Name": ["Abby", None], "Age": [12, np.nan], "wEIGht": [32.5, np.nan],
                   "Email Address": ["abby@gmail.com", None], "Country of Birth": ["CA", None]})
    >>> clean_df(df)
    Data Type Detection Report:
        These data types are supported by DataPrep to clean: ['email', 'country']
    Column Headers Cleaning Report:
        5 values cleaned (100.0%)
    Downcast Memory Report:
        Memory reducted from 475 to 451. New size: (94.95%)
    (                 semantic_data_type atomic_data_type
    Name                         string           string
    Age                        floating         floating
    wEIGht                     floating         floating
    Email Address                 email           string
    Country of Birth            country           string,
        name   age  w_ei_ght   email_address country_of_birth
    0  Abby    12      32.5  abby@gmail.com               CA
    1  <NA>  <NA>       NaN            <NA>             <NA>)
    """
    # pylint: disable=too-many-arguments
    # pylint: disable-msg=too-many-locals
    if is_dask(df):
        # the whole DataFrame is cleaned in memory
        parts = apply_partitions(df, lambda pdf: pdf)
        df = parts[0] if len(parts) == 1 else pd.concat(parts)
    # pylint:disable=too-many-branches
    # type: ignore

    if data_type_detection not in {"semantic", "atomic", "none"}:
        raise ValueError(
            f"data_type_detection {data_type_detection} is invalid, "
            'it needs to be "semantic", "atomic" or "none"'
        )

    if standardize_missing_values not in {"fill", "remove", "ignore"}:
        raise ValueError(
            f"standardize_missing_values {standardize_missing_values} is invalid, "
            'it needs to be "fill", "remove" or "ignore"'
        )

    # Data type detection
    if data_type_detection != "none":
        df_datatype_detection = _infer_data_type_df(df, data_type_detection)

    # Clean headers
    if clean_header:
        orig_columns = df.columns.astype(str).tolist()
        df = clean_headers(df, report=False)

    # Clean missing values and duplicate rows; keep track of how many rows are removed.
    orig_nrows = df.shape[0]

    df = _standardize_missing_values_df(df, standardize_missing_values)
    if remove_duplicate_entries:
        df = df.drop_duplicates()

    cleaned_nrows = df.shape[0]

    # Downcast memory; keep track of how much memory is saved
    orig_size = df.memory_usage(deep=True).sum()
    if downcast_memory:
        df = _downcast_memory(df)

    # Prepare for reports depending on which cleaning steps are performed
    if report:
        # Output which semantic data types are supported by Dataprep to further clean
        if data_type_detection == "semantic":
            semantic_data_type_values_list = df_datatype_detection[
                "semantic_data_type"
            ].values.tolist()
            stats_datatype = {
                "email": "email" in semantic_data_type_values_list,
                "phone": "phone" in semantic_data_type_values_list,
                "country": "country" in semantic_data_type_values_list,
                "coordinate": "coordinate" in semantic_data_type_values_list,
                "ip": "ip" in semantic_data_type_values_list,
                "URL": "URL" in semantic_data_type_values_list,
                "address": "address" in semantic_data_type_values_list,
            }

            _create_report_df(stats_datatype, len(df.columns), "datatype")
        if clean_header:
            new_columns = df.columns.astype(str).tolist()
            cleaned = [
                1 if new_columns[i] != orig_columns[i] else 0 for i in range(len(orig_columns))
            ]
            stats_header = {"cleaned": sum(cleaned)}

            _create_report_df(stats_header, len(df.columns), "header")
        if standardize_missing_values == "remove" or remove_duplicate_entries:
            stats_duplicate = {"cleaned": orig_nrows - cleaned_nrows}

            _create_report_df(stats_duplicate, orig_nrows, "duplicate")
        if downcast_memory:
            cleaned_size = df.memory_usage(deep=True).sum()
            stats_memory = {"cleaned": cleaned_size}

            _create_report_df(stats_memory, orig_size, "memory")

    df = df.reset_index(drop=True)

    if data_type_detection != "none":
        return df_datatype_detection, df
    else:
        return df


def _not_null_mask(column: pd.Series) -> np.ndarray:
    """
    Boolean array that is True where the value of the column is not null
    (not NaN/None/NA and not a null string such as "", "NULL" or "nan").
    """
    return np.fromiter((not is_null(val) for val in column.tolist()), dtype=bool, count=len(column))


def _count_valid(validate: Callable[..., Any], column: pd.Series, **kwargs: Any) -> int:
    """
    Number of values of the column accepted by a validate_* function.
    """
    return int(pd.Series(validate(column, **kwargs)).sum())


def _infer_semantic_data_type(column: pd.Series) -> Any:
    """
    Infer the semantic data type of a column.
    The semantic data type is the logical or business meaning of a column:
    "email", "country", "phone", "coordinate", "ip", "URL" or "address". If the
    column has no such meaning, its atomic data type is returned.

    Parameters
    ----------
    column
        A column (pandas Series) of the DataFrame.
    """
    # pylint: disable=import-outside-toplevel
    from .clean_country import validate_country
    from .clean_email import validate_email
    from .clean_ip import validate_ip
    from .clean_lat_long import validate_lat_long
    from .clean_phone import validate_phone
    from .clean_url import validate_url

    # Remove null values, and select a sample of the column to infer the type from
    column_not_na = column[_not_null_mask(column)]
    sample_size = (
        column_not_na.size if column_not_na.size <= 100 else min(int(0.1 * column_not_na.size), 500)
    )
    column_not_na_subset = column_not_na.sample(n=sample_size, random_state=1)
    half = column_not_na_subset.size / 2

    # Coordinates are recognized by the name of the column
    lat_lon = ["lat", "Lat", "LAT", "lon", "Lon", "LON"]
    if any(x in str(column.name) for x in lat_lon):
        if _count_valid(validate_lat_long, column_not_na_subset, lat_long=False, lat=True) > half:
            return "coordinate"

    # Only columns of strings can have one of the other semantic data types
    default_infer_dtype = infer_dtype(column_not_na_subset)
    if default_infer_dtype != "string":
        return default_infer_dtype

    semantic_data_type_dic = {
        "email": _count_valid(validate_email, column_not_na_subset),
        "country": _count_valid(validate_country, column_not_na_subset),
        "phone": _count_valid(validate_phone, column_not_na_subset),
        "ip": _count_valid(validate_ip, column_not_na_subset),
        "URL": _count_valid(validate_url, column_not_na_subset),
        "address": _count_valid_addresses(column_not_na_subset),
    }

    best_infer, best_infer_count = max(semantic_data_type_dic.items(), key=lambda item: item[1])
    if best_infer_count == 0:
        return default_infer_dtype
    return best_infer if best_infer_count > half else default_infer_dtype


def _count_valid_addresses(column: pd.Series) -> int:
    """
    Number of values of the column that are valid addresses. Address detection
    needs the optional python-crfsuite package and is skipped if it is missing.
    """
    from .clean_address import validate_address  # pylint: disable=import-outside-toplevel

    try:
        return _count_valid(validate_address, column)
    except ImportError:
        return 0


def _infer_atomic_data_type(column: pd.Series) -> Any:
    """
    Infer the atomic data type of a column, ignoring its null values.
    The atomic data types are the names returned by pandas.api.types.infer_dtype,
    for example "string", "integer", "floating" or "date".

    Parameters
    ----------
    column
        A column (pandas Series) of the DataFrame.
    """
    return infer_dtype(column[_not_null_mask(column)])


def _infer_data_type_df(df: pd.DataFrame, data_type_detection: str) -> pd.DataFrame:
    """
    Infer the data type of every column of the DataFrame.

    Parameters
    ----------
    df
        A Pandas DataFrame containing the data to be cleaned.
    data_type_detection
        If "semantic", infer the semantic and the atomic data type of each column.
        If "atomic", only infer the atomic data type of each column.
    """
    atomic = [_infer_atomic_data_type(df.iloc[:, pos]) for pos in range(df.shape[1])]
    if data_type_detection == "semantic":
        semantic = [_infer_semantic_data_type(df.iloc[:, pos]) for pos in range(df.shape[1])]
        return pd.DataFrame(
            {"semantic_data_type": semantic, "atomic_data_type": atomic}, index=df.columns
        )
    return pd.DataFrame({"atomic_data_type": atomic}, index=df.columns)


def _standardize_missing_values_df(
    df: pd.DataFrame, standardize_missing_values: str
) -> pd.DataFrame:
    """
    Standardize the missing values of the DataFrame. A new DataFrame is returned,
    the input DataFrame is not modified.

    Parameters
    ----------
    df
        A Pandas DataFrame containing the data to be cleaned.
    standardize_missing_values
        If "fill", null values (including null strings such as "NULL" or "nan") are
        replaced by NaN, or by NaT in columns of dates.
        If "remove", rows with any null value are removed.
        If "ignore", the DataFrame is returned as is.
    """
    if standardize_missing_values == "ignore":
        return df

    columns = []
    for pos in range(df.shape[1]):
        column = df.iloc[:, pos]
        null_mask = ~_not_null_mask(column)
        if null_mask.any():
            fill_value = pd.NaT if infer_dtype(column) == "date" else np.nan
            column = column.astype(object).mask(null_mask, fill_value).infer_objects()
        columns.append(column)
    result = pd.concat(columns, axis=1) if columns else df.copy()
    result.columns = df.columns
    result = result.convert_dtypes()

    if standardize_missing_values == "remove":
        result = result.dropna(how="any")
    return result


def _create_report_df(stats: Dict[str, Any], old_stat: Any, option: str) -> None:
    """
    Describe what was done in the cleaning process.
    """
    if option not in {"datatype", "header", "duplicate", "memory"}:
        raise ValueError(
            f"clean_df report option {option} is invalid, "
            'it needs to be "datatype", "header", "duplicate", or "memory"'
        )

    if option == "datatype":
        print("Data Type Detection Report:")
        print(
            "\tThese data types are supported by DataPrep to clean:",
            [k for k, v in stats.items() if v],
        )
    if option == "header":
        print("Column Headers Cleaning Report:")
        if stats["cleaned"] > 0:
            nclnd = stats["cleaned"]
            pclnd = round(nclnd / old_stat * 100, 2)
            print(f"\t{nclnd} values cleaned ({pclnd}%)")
        else:
            print("No Headers Cleaned.")
    if option == "duplicate":
        print("Number of Entries Cleaning Report:")
        if stats["cleaned"] > 0:
            nclnd = stats["cleaned"]
            pclnd = round(nclnd / old_stat * 100, 2)
            print(f"\t{nclnd} entries dropped ({pclnd}%)")
        else:
            print("No Duplicated Entries Cleaned.")
    if option == "memory":
        print("Downcast Memory Report:")
        if stats["cleaned"] > 0:
            nclnd = stats["cleaned"]
            pclnd = round(nclnd / old_stat * 100, 2)
            print(f"\tMemory reducted from {old_stat} to {nclnd}. New size: ({pclnd}%)")
        else:
            print("Downcast Memory not Performed.")


def _downcast_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to downcast the memory size of the DataFrame by using subtypes in
    numerical columns; columns of Python objects or of the default string type
    are converted to `category`. A new DataFrame is returned.
    """
    df = df.copy(deep=False)
    for pos, dtype in enumerate(df.dtypes.tolist()):
        column = df.iloc[:, pos]
        new_column = column
        if is_integer_dtype(dtype) or is_float_dtype(dtype):
            if column.notna().any():
                new_column = _downcast_numeric(column)
        elif is_object_dtype(dtype) or str(dtype) == "str":
            try:
                new_column = column.astype("category")
            except TypeError:  # unhashable values such as lists or dictionaries
                continue
        if new_column is not column:
            df.isetitem(pos, new_column)

    return df


def _downcast_numeric(column: pd.Series) -> pd.Series:
    """
    Downcast a numerical column that has at least one value to the smallest subtype.
    """
    col_min, col_max = column.min(), column.max()
    if is_integer_dtype(column.dtype):
        for int_type, nullable_name in ((np.int8, "Int8"), (np.int16, "Int16"), (np.int32, "Int32")):
            if col_min > np.iinfo(int_type).min and col_max < np.iinfo(int_type).max:
                return column.astype(nullable_name)
        return column.astype("Int64")

    if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float32).max:
        return pd.to_numeric(column, downcast="float")
    return column.astype("Float64")
