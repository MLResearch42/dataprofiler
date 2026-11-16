"""
The EDA type system.

Every column handed to ``plot``, ``plot_missing``, ``create_report`` and friends is
given one of the *semantic* types defined here. The semantic type, not the pandas
dtype, decides which statistics are computed and which charts are drawn:

==================  =====================================================
Semantic type       Typical pandas columns
==================  =====================================================
``Continuous``      ints and floats with 10 or more distinct values
``SmallCardNum``    ints and floats with fewer than 10 distinct values
``Nominal``         strings, booleans, ``category``, ``Period``, mixed objects
``DateTime``        ``datetime64`` columns (with or without a timezone)
``GeoGraphy``       strings that are mostly country names or codes
``GeoPoint``        strings/tuples that are mostly ``latitude, longitude`` pairs
==================  =====================================================

You can always override the detected type with the ``dtype`` argument, e.g.
``plot(df, "zip_code", dtype={"zip_code": "nominal"})``.
"""

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

__all__ = [
    "DType",
    "Categorical",
    "Nominal",
    "Ordinal",
    "Numerical",
    "Continuous",
    "SmallCardNum",
    "Discrete",
    "DateTime",
    "Text",
    "GeoGraphy",
    "GeoPoint",
    "LatLong",
    "DTypeDef",
    "detect_dtype",
    "normalize_dtype",
    "is_dtype",
    "is_nominal",
    "is_continuous",
    "is_datetime",
    "drop_null",
    "NULL_VALUES",
]

_DECIMAL_COMMA = r"\s*[-+]?\d+,\d+\s*"

# How many leading rows are inspected when sniffing for country / lat-long columns.
GEO_SAMPLE_SIZE = 100
# A column is "geo" when more than this share of the sampled values validates.
GEO_THRESHOLD = 0.8
# Numerical columns with fewer distinct values than this are plotted as categories.
SMALL_CARDINALITY = 10

# Kept for backwards compatibility with code that introspected these lists.
STRING_PANDAS_DTYPES = [pd.StringDtype]
STRING_DTYPES = STRING_PANDAS_DTYPES
CATEGORICAL_NUMPY_DTYPES = [bool, object]
CATEGORICAL_PANDAS_DTYPES = [pd.CategoricalDtype, pd.PeriodDtype]
CATEGORICAL_DTYPES = CATEGORICAL_NUMPY_DTYPES + CATEGORICAL_PANDAS_DTYPES + STRING_DTYPES
NUMERICAL_NUMPY_DTYPES = [np.number]
NUMERICAL_DTYPES = NUMERICAL_NUMPY_DTYPES
DATETIME_NUMPY_DTYPES = [np.datetime64]
DATETIME_PANDAS_DTYPES = [pd.DatetimeTZDtype]
DATETIME_DTYPES = DATETIME_NUMPY_DTYPES + DATETIME_PANDAS_DTYPES

#: Strings that are treated as missing by :func:`drop_null`.
NULL_VALUES = {
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
}


class DType:
    """Root of the type tree."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((type(self).__name__, tuple(sorted(self.__dict__.items()))))


############## Syntactic DTypes ##############
class Categorical(DType):
    """A column whose values are labels."""


class Nominal(Categorical):
    """Labels without an order, e.g. a product name."""


class Ordinal(Categorical):
    """Labels with an order, e.g. small / medium / large."""


class Numerical(DType):
    """A column whose values are numbers."""


class Continuous(Numerical):
    """Numbers measured on a scale, e.g. a price."""


class SmallCardNum(Numerical):
    """Numbers with only a handful of distinct values, plotted as categories."""


class Discrete(Numerical):
    """Whole-number counts."""


############## Semantic DTypes ##############
class DateTime(Numerical):
    """Timestamps."""


class Text(Nominal):
    """Free text."""


class GeoGraphy(Categorical):
    """Country names or codes."""


class GeoPoint(DType):
    """A latitude / longitude pair stored in one column."""


class LatLong(GeoPoint):
    """A latitude / longitude pair stored in two columns.

    Examples
    --------
    >>> from dataprep.eda import plot
    >>> from dataprep.eda.dtypes import LatLong
    >>> plot(df, LatLong("lat", "lon"), "price")  # doctest: +SKIP
    """

    def __init__(self, lat_col: str, long_col: str) -> None:
        self.lat = lat_col
        self.long = long_col

    def __repr__(self) -> str:
        return f"LatLong({self.lat!r}, {self.long!r})"


############## End of the Type Tree ##############

DTypeOrStr = Union[DType, Type[DType], str, None]
DTypeDict = Union[Dict[str, Union[DType, Type[DType], str]], None]
DTypeDef = Union[Dict[str, Union[DType, Type[DType], str]], DType, Type[DType], str, None]

_NAMED_DTYPES: Dict[str, Type[DType]] = {
    "categorical": Categorical,
    "ordinal": Ordinal,
    "nominal": Nominal,
    "numerical": Numerical,
    "continuous": Continuous,
    "discrete": Discrete,
    "datetime": DateTime,
    "text": Text,
    "geography": GeoGraphy,
    "geopoint": GeoPoint,
    "smallcardnum": SmallCardNum,
}


def normalize_dtype(dtype_repr: Any) -> DType:
    """Turn ``"nominal"``, ``Nominal`` or ``Nominal()`` into a ``Nominal()`` instance.

    Raises
    ------
    ValueError
        If ``dtype_repr`` does not name a known type.
    """
    if isinstance(dtype_repr, DType):
        return dtype_repr
    if isinstance(dtype_repr, type) and issubclass(dtype_repr, DType):
        return dtype_repr()
    if isinstance(dtype_repr, str) and dtype_repr.lower() in _NAMED_DTYPES:
        return _NAMED_DTYPES[dtype_repr.lower()]()

    # Classes from a different copy of this module (e.g. after an autoreload)
    # are matched by name so that user code keeps working.
    name = getattr(dtype_repr, "__name__", type(dtype_repr).__name__).lower()
    if name in _NAMED_DTYPES and not isinstance(dtype_repr, (str, int, float, bool)):
        return _NAMED_DTYPES[name]()

    known = ", ".join(sorted(cls.__name__ for cls in set(_NAMED_DTYPES.values())))
    raise ValueError(f"Unknown dtype {dtype_repr!r}. Use one of: {known}.")


def map_dtype(dtype: DType) -> DType:
    """Flatten the type tree: ``Categorical`` -> ``Nominal``, ``Numerical`` -> ``Continuous``."""
    if isinstance(dtype, Categorical) and not isinstance(dtype, (Ordinal, Nominal, GeoGraphy)):
        return Nominal()
    if isinstance(dtype, Ordinal):
        # Ordinal columns are plotted the same way as nominal ones.
        return Nominal()
    if isinstance(dtype, Numerical) and not isinstance(
        dtype, (Continuous, Discrete, DateTime, SmallCardNum)
    ):
        return Continuous()
    if isinstance(dtype, Discrete):
        return Continuous()
    return dtype


def _known_dtype_for(name: Any, known_dtype: Optional[DTypeDef]) -> Optional[DType]:
    """Look up the user supplied type for column ``name``, if there is one."""
    if known_dtype is None or (isinstance(known_dtype, dict) and not known_dtype):
        return None
    if isinstance(known_dtype, dict):
        if name in known_dtype:
            return map_dtype(normalize_dtype(known_dtype[name]))
        if str(name) in known_dtype:
            return map_dtype(normalize_dtype(known_dtype[str(name)]))
        return None
    return map_dtype(normalize_dtype(known_dtype))


def detect_dtype(
    col: pd.Series,
    known_dtype: Optional[DTypeDef] = None,
    *,
    head: Optional[pd.Series] = None,
    detect_small_distinct: bool = True,
) -> DType:
    """Detect the semantic type of a column, honouring a user supplied override.

    Parameters
    ----------
    col
        The column.
    known_dtype
        A single type applied to every column, or a ``{column name: type}`` dict.
        A type can be given as a string (``"nominal"``), a class (``Nominal``) or an
        instance (``Nominal()``).
    head
        The first rows of ``col``, used to sniff for geographic data. Computed from
        ``col`` when omitted.
    detect_small_distinct
        Report numerical columns with fewer than 10 distinct values as
        :class:`SmallCardNum` instead of :class:`Continuous`.
    """
    col = _as_pandas(col)
    known = _known_dtype_for(col.name, known_dtype)
    if known is not None:
        return known
    return detect_without_known(col, head, detect_small_distinct=detect_small_distinct)


def detect_without_known(
    col: pd.Series,
    head: Optional[pd.Series] = None,
    *,
    detect_small_distinct: bool = True,
) -> DType:
    """Detect the semantic type of a column from its values alone."""
    col = _as_pandas(col)
    if is_continuous(col.dtype):
        if detect_small_distinct and col.nunique(dropna=True) < SMALL_CARDINALITY:
            return SmallCardNum()
        return Continuous()
    if is_datetime(col.dtype):
        return DateTime()

    if head is None:
        head = col.head(GEO_SAMPLE_SIZE)
    if is_geography(head):
        return GeoGraphy()
    if is_geopoint(head):
        return GeoPoint()
    return Nominal()


def is_dtype(dtype1: Any, dtype2: DType) -> bool:
    """Whether ``dtype1`` is the same type as, or a sub-type of, ``dtype2``."""
    return isinstance(dtype1, dtype2.__class__)


def is_continuous(dtype: Any) -> bool:
    """Whether a pandas/numpy dtype holds real numbers (bools and complex excluded)."""
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    return (
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
    ) or pd.api.types.is_timedelta64_dtype(dtype)


def is_datetime(dtype: Any) -> bool:
    """Whether a pandas/numpy dtype holds timestamps."""
    return pd.api.types.is_datetime64_any_dtype(dtype)


def is_nominal(dtype: Any) -> bool:
    """Whether a pandas/numpy dtype holds labels (strings, bools, categories, objects)."""
    return not is_continuous(dtype) and not is_datetime(dtype)


def is_pandas_categorical(dtype: Any) -> bool:
    """Whether a dtype is a pandas ``category`` or ``Period`` dtype."""
    return isinstance(dtype, (pd.CategoricalDtype, pd.PeriodDtype))


def is_geography(head: pd.Series) -> bool:
    """Whether most of the given values are country names or codes."""
    head = _geo_sample(head)
    if head is None:
        return False
    from ..clean import validate_country  # pylint: disable=import-outside-toplevel

    try:
        valid = np.asarray(validate_country(head), dtype=bool)
    except Exception:  # pylint: disable=broad-except
        return False
    return bool(valid.sum() / len(head) > GEO_THRESHOLD)


def is_geopoint(head: pd.Series) -> bool:
    """Whether most of the given values are ``latitude, longitude`` pairs."""
    head = _geo_sample(head)
    if head is None:
        return False
    from ..clean import validate_lat_long  # pylint: disable=import-outside-toplevel

    as_str = head.dropna().astype(object).map(str)
    # "12,5" is far more likely a number written with a decimal comma than a
    # coordinate pair, so columns made of such values are not geo points.
    if len(as_str) and as_str.str.fullmatch(_DECIMAL_COMMA).mean() > 0.5:
        return False

    try:
        valid = np.asarray(validate_lat_long(head.astype(object).map(str)), dtype=bool)
    except Exception:  # pylint: disable=broad-except
        return False
    return bool(valid.sum() / len(head) > GEO_THRESHOLD)


def _geo_sample(head: pd.Series) -> Optional[pd.Series]:
    """Prepare the sniffing sample. Returns None when the column cannot be geographic."""
    head = _as_pandas(head).head(GEO_SAMPLE_SIZE)
    if len(head) == 0 or pd.api.types.is_bool_dtype(head.dtype):
        return None
    if head.isna().all():
        return None
    return head


def _as_pandas(col: Any) -> pd.Series:
    """Materialise a lazy (e.g. dask) series; pandas series are returned untouched."""
    if isinstance(col, pd.Series):
        return col
    if hasattr(col, "compute"):
        return col.compute()
    return pd.Series(col)


def drop_null(var: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Drop missing values, *including* the string spellings in :data:`NULL_VALUES`."""
    if hasattr(var, "compute") and not isinstance(var, (pd.Series, pd.DataFrame)):
        var = var.compute()

    if isinstance(var, pd.Series):
        if is_datetime(var.dtype):
            return var.dropna()
        return var[~(var.isna() | var.isin(NULL_VALUES))]

    if isinstance(var, pd.DataFrame):
        keep = pd.Series(True, index=var.index)
        for name in var.columns:
            col = var[name]
            keep &= ~col.isna() if is_datetime(col.dtype) else ~(col.isna() | col.isin(NULL_VALUES))
        return var[keep]

    raise ValueError("Input should be a pandas DataFrame or Series")


def get_dtype_cnts_and_num_cols(
    df: pd.DataFrame,
    dtype: Optional[DTypeDef] = None,
) -> Tuple[Dict[str, int], List[str]]:
    """Count the columns of each semantic type and list the continuous ones."""
    dtype_cnts: DefaultDict[str, int] = defaultdict(int)
    num_cols: List[str] = []
    for col in df.columns:
        col_dtype = detect_dtype(df[col], dtype)
        if isinstance(col_dtype, (Nominal, SmallCardNum)):
            dtype_cnts["Categorical"] += 1
        elif isinstance(col_dtype, Continuous):
            dtype_cnts["Numerical"] += 1
            num_cols.append(col)
        elif isinstance(col_dtype, DateTime):
            dtype_cnts["DateTime"] += 1
        elif isinstance(col_dtype, GeoGraphy):
            dtype_cnts["GeoGraphy"] += 1
        elif isinstance(col_dtype, GeoPoint):
            dtype_cnts["GeoPoint"] += 1
        else:
            raise NotImplementedError(f"column {col!r} has unsupported type {col_dtype!r}")
    return dict(dtype_cnts), num_cols
