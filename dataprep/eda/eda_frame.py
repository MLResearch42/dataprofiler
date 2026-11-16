"""Defines EDAFrame, the normalised in-memory view of the user's data."""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .dtypes import (
    Continuous,
    DateTime,
    DType,
    DTypeDef,
    GeoGraphy,
    GeoPoint,
    Nominal,
    SmallCardNum,
    _known_dtype_for,
    detect_without_known,
    is_continuous,
    is_datetime,
)

DataFrame = Union[pd.DataFrame, "EDAFrame", Any]

__all__ = ["EDAFrame", "DataFrame", "to_pandas", "to_str_series", "sample_rows"]

# Seed of the random generator used by ``sample=``: the same data and sample
# size always give the same sample, so results can be reproduced.
SAMPLE_RANDOM_STATE = 0


def sample_rows(df: Any, sample: Union[int, float, None]) -> Tuple[Any, Optional[str]]:
    """Take a random sample of the rows, for the ``sample=`` option of the EDA functions.

    Parameters
    ----------
    df
        Any input the EDA functions accept.
    sample
        ``None`` keeps every row. An integer is the number of rows to keep, a
        float between 0 and 1 the fraction of rows to keep. Asking for at least
        as many rows as there are keeps every row.

    Returns
    -------
    The (possibly sampled) data, and a sentence describing the sample for the
    output, or None when every row was kept. pandas input stays in its original
    row order. Dask input is sampled before it is loaded into memory; its sample
    size is approximate, because Dask samples a fraction of each partition.
    """
    if sample is None:
        return df, None
    if isinstance(sample, bool) or not isinstance(sample, (int, float, np.integer, np.floating)):
        raise TypeError(
            "sample must be a number of rows (int) or a fraction of the rows (float), "
            f"got {sample!r}"
        )
    if isinstance(sample, (float, np.floating)):
        if not 0 < sample <= 1:
            raise ValueError(f"A fractional sample must be between 0 and 1, got {sample}")
    elif sample < 1:
        raise ValueError(f"sample must be at least 1 row, got {sample}")

    is_dask = hasattr(df, "compute") and type(df).__module__.startswith("dask")
    nrows = len(df) if is_dask else None
    if not is_dask:
        df = to_pandas(df)
        nrows = len(df)
    assert nrows is not None

    keep = round(sample * nrows) if isinstance(sample, (float, np.floating)) else int(sample)
    keep = max(keep, 1)
    if keep >= nrows:
        return df, None

    if is_dask:
        sampled = df.sample(frac=keep / nrows, random_state=SAMPLE_RANDOM_STATE)
        sampled = to_pandas(sampled)
    else:
        positions = np.sort(
            np.random.default_rng(SAMPLE_RANDOM_STATE).choice(nrows, size=keep, replace=False)
        )
        sampled = df.iloc[positions]
    note = (
        f"Based on a random sample of {len(sampled):,} of {nrows:,} rows; "
        "counts and statistics describe the sample."
    )
    return sampled, note


def to_pandas(df: Any) -> pd.DataFrame:
    """Turn any supported tabular object into a pandas DataFrame.

    Supported inputs are pandas DataFrames and Series, Dask DataFrames and Series
    (they are computed, so the data has to fit in memory) and anything that has a
    ``to_pandas()`` method such as Polars DataFrames and PyArrow Tables.
    """
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pd.Series):
        return df.to_frame()
    if isinstance(df, EDAFrame):
        return df.frame
    # dask's DataFrame lives in dask.dataframe, or in the dask_expr package in
    # dask 2024.x, so match on the module name prefix
    if hasattr(df, "compute") and type(df).__module__.startswith("dask"):
        return to_pandas(df.compute())
    if hasattr(df, "to_pandas"):
        return to_pandas(df.to_pandas())
    if isinstance(df, (dict, list, np.ndarray)):
        return pd.DataFrame(df)
    raise TypeError(
        f"Unsupported data type {type(df).__name__}. Pass a pandas DataFrame, "
        "a Dask DataFrame, or an object with a to_pandas() method."
    )


def to_str_series(srs: pd.Series, na_as_str: bool = False) -> pd.Series:
    """Convert every value of a series to ``str``.

    Missing values stay missing unless ``na_as_str`` is True, in which case they
    become the string ``"nan"``. Cells holding lists, tuples or dicts are
    converted as a whole instead of raising.
    """
    if isinstance(srs.dtype, pd.StringDtype) or (
        srs.dtype == object and pd.api.types.infer_dtype(srs, skipna=True) == "string"
    ):
        out = srs
    else:
        out = srs.astype(object).map(str, na_action="ignore")

    if na_as_str and out.isna().any():
        out = out.astype(object).where(out.notna(), "nan")
    return out


class EDAFrame:
    """A normalised, in-memory view of the data being explored.

    Building an ``EDAFrame`` does the housekeeping every plot function needs:

    * column names become strings and duplicated names get a ``_1``, ``_2`` suffix,
    * every column is given a semantic type (see :mod:`dataprep.eda.dtypes`),
    * categorical columns are converted to strings, nullable/Arrow numerics to
      plain NumPy numbers, timedeltas to seconds and timezone-aware timestamps to
      naive ones,
    * missing-value counts are computed once and cached.

    The user's DataFrame is never modified.

    Parameters
    ----------
    df
        A pandas DataFrame or Series, a Dask DataFrame or Series, another
        ``EDAFrame``, or any object with a ``to_pandas()`` method.
    dtype
        Override the detected types, e.g. ``{"a": "continuous", "b": Nominal()}``
        or a single type for every column.
    repartition
        Ignored. Kept so that old call sites keep working.
    """

    def __init__(
        self,
        df: Optional[DataFrame] = None,
        dtype: Optional[DTypeDef] = None,
        repartition: bool = True,  # pylint: disable=unused-argument
    ) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._columns: pd.Index = pd.Index([], dtype=object)
        self._eda_dtypes: Dict[str, DType] = {}
        self._str_col_cache: Dict[Tuple[str, bool], pd.Series] = {}
        self._nulls_cnt: Dict[str, int] = {}
        self._nulls: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None

        if df is None:
            return

        if isinstance(df, EDAFrame):
            self._df = df._df
            self._columns = df._columns
            self._eda_dtypes = df._eda_dtypes
            self._str_col_cache = df._str_col_cache
            self._nulls_cnt = df._nulls_cnt
            self._nulls = df._nulls
            self._values = df._values
            return

        pdf = to_pandas(df)
        names = _process_column_name(pdf.columns)

        columns: Dict[str, pd.Series] = {}
        for pos, name in enumerate(names):
            # Re-wrap the underlying array: a fresh RangeIndex without copying the data.
            srs = pd.Series(pdf.iloc[:, pos].array, name=name, copy=False)
            srs, eda_dtype = _normalize_column(srs, _known_dtype_for(name, dtype))
            columns[name] = srs
            self._eda_dtypes[name] = eda_dtype
            self._nulls_cnt[name] = int(srs.isna().sum())

        self._df = pd.DataFrame(columns, index=pd.RangeIndex(len(pdf)), copy=False)
        self._columns = self._df.columns

    @property
    def columns(self) -> pd.Index:
        """The (string) column names."""
        return self._columns

    @property
    def dtypes(self) -> pd.Series:
        """The pandas dtypes of the normalised columns."""
        if self._df is None:
            return pd.DataFrame().dtypes
        return self._df.dtypes

    @property
    def nulls(self) -> np.ndarray:
        """A boolean array, True where a cell is missing. Shape: rows x columns."""
        if self._nulls is None:
            self._nulls = self.frame.isna().to_numpy(dtype=bool)
        return self._nulls

    @property
    def shape(self) -> Tuple[int, int]:
        """(number of rows, number of columns)."""
        if self._df is None:
            return (0, 0)
        return self._df.shape

    @property
    def values(self) -> np.ndarray:
        """The data as a 2-D array (floats when every column is numerical)."""
        if self._values is None:
            frame = self.frame
            if all(is_continuous(dtp) for dtp in frame.dtypes):
                self._values = frame.to_numpy(dtype=float, na_value=np.nan)
            else:
                self._values = frame.to_numpy()
        return self._values

    @property
    def frame(self) -> pd.DataFrame:
        """The normalised pandas DataFrame."""
        if self._df is None:
            raise RuntimeError("dataframe is empty!")
        return self._df

    def head(self, n: int = 5) -> pd.DataFrame:
        """The first ``n`` rows."""
        return self.frame.head(n)

    def get_col_as_str(self, col: str, na_as_str: bool = False) -> pd.Series:
        """Return a column with every value converted to a string.

        Missing values stay missing unless ``na_as_str`` is True.
        """
        if self._df is None:
            raise RuntimeError("dataframe is empty!")
        if col not in self._columns:
            raise RuntimeError(f"column is not exists: {col}")

        key = (col, na_as_str)
        if key not in self._str_col_cache:
            self._str_col_cache[key] = to_str_series(self._df[col], na_as_str=na_as_str)
        return self._str_col_cache[key]

    def get_missing_cnt(self, col: str) -> int:
        """The number of missing values in a column."""
        return self._nulls_cnt[col]

    def get_eda_dtype(self, col: str) -> DType:
        """The semantic type of a column."""
        return self._eda_dtypes[col]

    def select_dtypes(self, include: List[Any]) -> "EDAFrame":
        """A new EDAFrame holding only the columns of the given pandas dtypes."""
        return self[list(self.frame.select_dtypes(include).columns)]

    def select_num_columns(self) -> "EDAFrame":
        """A new EDAFrame holding only the numerical columns."""
        dtypes = self.frame.dtypes
        return self[[col for col in self._columns if is_continuous(dtypes[col])]]

    def __getitem__(self, indexer: Union[Sequence[str], str]) -> "EDAFrame":
        """A new EDAFrame restricted to the given column name(s)."""
        if isinstance(indexer, str):
            indexer = [indexer]
        indexer = list(indexer)

        sub = EDAFrame()
        sub._df = self.frame[indexer]
        sub._columns = sub._df.columns
        sub._eda_dtypes = {col: self._eda_dtypes[col] for col in indexer}
        sub._nulls_cnt = {col: self._nulls_cnt[col] for col in indexer}
        sub._str_col_cache = {
            key: val for key, val in self._str_col_cache.items() if key[0] in set(indexer)
        }
        return sub

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        types = ", ".join(f"{col}: {dtp!r}" for col, dtp in self._eda_dtypes.items())
        return f"EDAFrame(shape={self.shape}, dtypes={{{types}}})"


def _process_column_name(df_columns: pd.Index) -> List[str]:
    """Make column names strings and give duplicated names a ``_1``, ``_2`` ... suffix."""
    if isinstance(df_columns, pd.MultiIndex):
        columns = ["_".join(str(level) for level in col) for col in df_columns]
    else:
        columns = [str(col) for col in df_columns]

    column_count = Counter(columns)
    taken = set(columns)
    current_id: Dict[str, int] = {}
    for i, col in enumerate(columns):
        if column_count[col] > 1:
            current_id[col] = current_id.get(col, 0) + 1
            new_name = f"{col}_{current_id[col]}"
            while new_name in taken:
                current_id[col] += 1
                new_name = f"{col}_{current_id[col]}"
            taken.add(new_name)
            columns[i] = new_name
    return columns


def _to_numpy_numeric(srs: pd.Series) -> pd.Series:
    """Give a numerical column a plain NumPy dtype (nullable ints become floats)."""
    if pd.api.types.is_timedelta64_dtype(srs.dtype):
        return srs.dt.total_seconds()
    if isinstance(srs.dtype, np.dtype):
        return srs
    # pandas nullable (Int64, Float64, ...) and Arrow backed dtypes
    if pd.api.types.is_integer_dtype(srs.dtype) and not srs.isna().any():
        try:
            return srs.astype(srs.dtype.numpy_dtype)
        except (AttributeError, TypeError, ValueError):
            pass
    return pd.Series(srs.to_numpy(dtype="float64", na_value=np.nan), index=srs.index, name=srs.name)


def _normalize_column(srs: pd.Series, known: Optional[DType]) -> Tuple[pd.Series, DType]:
    """Detect a column's semantic type and convert it to the matching representation."""
    eda_dtype = known if known is not None else detect_without_known(srs)

    if isinstance(eda_dtype, DateTime):
        if not is_datetime(srs.dtype):
            srs = pd.to_datetime(srs, errors="coerce")
        if getattr(srs.dtype, "tz", None) is not None:
            srs = srs.dt.tz_localize(None)
        elif not isinstance(srs.dtype, np.dtype):
            srs = srs.astype("datetime64[ns]")
        return srs, eda_dtype

    if isinstance(eda_dtype, (Continuous, SmallCardNum)):
        if not is_continuous(srs.dtype):
            srs = pd.to_numeric(srs, errors="coerce")
        return _to_numpy_numeric(srs), eda_dtype

    if isinstance(eda_dtype, (Nominal, GeoGraphy)):
        return to_str_series(srs), eda_dtype

    if isinstance(eda_dtype, GeoPoint):
        # Geo points may be tuples or "lat, long" strings; keep the values as they are.
        if not isinstance(srs.dtype, np.dtype) and not isinstance(srs.dtype, pd.StringDtype):
            srs = srs.astype(object)
        return srs, eda_dtype

    raise TypeError(f"Column {srs.name!r} has unsupported type {eda_dtype!r}")
