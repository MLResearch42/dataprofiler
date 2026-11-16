"""
This module implements summarize(df): the statistics behind plot(df) and
create_report(df), returned as a pandas DataFrame instead of HTML.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .configs import Config
from .distribution.compute.overview import (
    _cont_calcs,
    _count_unique_rows,
    _format_cont_ins,
    _format_nom_ins,
    _nom_calcs,
)
from .dtypes import Continuous, DateTime, DTypeDef, GeoGraphy, GeoPoint, Nominal, is_continuous
from .eda_frame import DataFrame, EDAFrame, sample_rows, to_pandas
from .parallel import Progress, pmap

__all__ = ["summarize", "SUMMARY_COLUMNS"]

# The columns of the table returned by summarize(), in order. The set is the same
# for every input, so code that reads the summary does not depend on the data;
# a statistic that does not apply to a column is NaN.
SUMMARY_COLUMNS = [
    "type",
    "dtype",
    "count",
    "missing",
    "missing_pct",
    "distinct",
    "distinct_pct",
    "top",
    "top_count",
    "mean",
    "std",
    "min",
    "p25",
    "median",
    "p75",
    "max",
    "skewness",
    "kurtosis",
    "zeros",
    "negatives",
    "infinite",
    "min_length",
    "max_length",
    "mean_length",
    "earliest",
    "latest",
    "insights",
]

_INT_COLUMNS = ["count", "missing", "distinct", "top_count"]


def summarize(
    df: DataFrame,
    *,
    target: Optional[str] = None,
    dtype: Optional[DTypeDef] = None,
    config: Optional[Dict[str, Any]] = None,
    sample: Union[int, float, None] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Summary statistics and data-quality insights of every column, as a DataFrame.

    This is the information shown by ``plot(df)`` and ``create_report(df)`` in a
    form you can filter, save, compare between runs or check in a test.

    Parameters
    ----------
    df
        A pandas DataFrame, or any other input ``plot`` accepts.
    target
        The name of a target column. Adds ``target_score`` and ``target_method``:
        how strongly each column is associated with the target (see
        :func:`rank_features`).
    dtype
        Override the detected column types, e.g. ``{"Pclass": "nominal"}``.
    config
        Thresholds of the insights, with the same keys as for ``plot``, e.g.
        ``{"insight.missing.threshold": 5}``.
    sample
        Summarize a random sample of the rows instead of all of them: a number of
        rows (int) or a fraction (float). The sample is the same on every run.
    progress
        Show a progress bar.

    Returns
    -------
    pandas.DataFrame
        One row per column, indexed by column name, with the columns in
        ``SUMMARY_COLUMNS``:

        - ``type``: the DataPrep column type (Continuous, Nominal, DateTime, ...).
        - ``dtype``: the pandas dtype of the input column.
        - ``count``, ``missing``, ``missing_pct``: present and missing values
          (the percentage is out of 100).
        - ``distinct``, ``distinct_pct``: distinct present values, and their share
          of the present values.
        - ``top``, ``top_count``: the most frequent value and how often it occurs.
        - ``mean`` ... ``infinite``: statistics of numerical columns, computed on
          the finite values.
        - ``min_length``, ``max_length``, ``mean_length``: text lengths of
          text-like columns.
        - ``earliest``, ``latest``: range of datetime columns.
        - ``insights``: the data-quality findings of ``plot(df)`` for the column,
          e.g. ``"Missing, Skewed"`` (empty when there are none).

        ``summary.attrs["dataset"]`` holds table-level figures: rows, columns,
        duplicate rows, missing cells and memory use.

    Examples
    --------
    >>> from dataprep.datasets import load_dataset
    >>> from dataprep.eda import summarize
    >>> summary = summarize(load_dataset("titanic"))
    >>> summary.loc["Age", ["type", "missing", "mean"]]  # doctest: +SKIP
    >>> summary[summary["missing_pct"] > 5].index.tolist()
    ['Age', 'Cabin']
    >>> summary[summary["insights"].str.contains("Skewed")].index.tolist()  # doctest: +SKIP
    """
    cfg = Config.from_dict(None, config)
    data, note = sample_rows(df, sample)
    pdf = to_pandas(data)

    with Progress(enabled=progress):
        frame = EDAFrame(pdf, dtype=dtype)
        nrows, ncols = frame.shape
        if ncols == 0:
            raise ValueError("The DataFrame has no columns, there is nothing to summarize.")

        input_dtypes = [str(dtp) for dtp in pdf.dtypes]
        rows = pmap(
            lambda col: _summarize_column(frame, col, cfg, nrows),
            list(frame.columns),
            desc="Columns",
        )

    summary = pd.DataFrame(rows, index=pd.Index(list(frame.columns), name="column"))
    summary.insert(1, "dtype", input_dtypes)
    summary = summary.reindex(columns=SUMMARY_COLUMNS)
    for col in _INT_COLUMNS:
        summary[col] = summary[col].astype("int64")
    for col in ("type", "dtype", "insights"):
        summary[col] = summary[col].astype(object)

    if target is not None:
        from .target import rank_features  # pylint: disable=import-outside-toplevel

        ranking = rank_features(frame, target)
        summary["target_score"] = ranking["score"].reindex(summary.index)
        summary["target_method"] = ranking["method"].reindex(summary.index)

    missing_cells = int(summary["missing"].sum())
    summary.attrs["dataset"] = {
        "rows": nrows,
        "columns": ncols,
        "duplicate_rows": nrows - _count_unique_rows(frame.frame) if nrows else 0,
        "missing_cells": missing_cells,
        "missing_cells_pct": 100 * missing_cells / (nrows * ncols) if nrows else 0.0,
        "memory_bytes": int(pdf.memory_usage(deep=True).sum()),
        "sample": note,
    }
    return summary


def _summarize_column(frame: EDAFrame, col: str, cfg: Config, nrows: int) -> Dict[str, Any]:
    """The statistics of one column (a row of the summary)."""
    srs = frame.frame[col]
    dtp = frame.get_eda_dtype(col)
    nmissing = frame.get_missing_cnt(col)
    present = srs.dropna()
    count = nrows - nmissing

    row: Dict[str, Any] = {
        "type": type(dtp).__name__,
        "count": count,
        "missing": nmissing,
        "missing_pct": 100 * nmissing / nrows if nrows else np.nan,
    }

    textual = isinstance(dtp, (Nominal, GeoGraphy, GeoPoint))
    as_str = frame.get_col_as_str(col).dropna() if textual else None
    values = as_str if as_str is not None else present
    try:
        counts = values.value_counts()
    except TypeError:  # unhashable cells
        counts = frame.get_col_as_str(col).dropna().value_counts()
    row["distinct"] = len(counts)
    row["distinct_pct"] = 100 * len(counts) / count if count else np.nan
    row["top"] = counts.index[0] if len(counts) else np.nan
    row["top_count"] = int(counts.iloc[0]) if len(counts) else 0

    if is_continuous(srs.dtype) and not isinstance(dtp, DateTime):
        row.update(_numeric_stats(present))
    if textual and as_str is not None:
        lengths = as_str.str.len()
        if len(lengths):
            row["min_length"] = int(lengths.min())
            row["max_length"] = int(lengths.max())
            row["mean_length"] = float(lengths.mean())
    if isinstance(dtp, DateTime) and len(present):
        row["earliest"] = present.min()
        row["latest"] = present.max()

    row["insights"] = ", ".join(_column_insights(frame, col, dtp, present, cfg, nrows))
    return row


def _numeric_stats(present: pd.Series) -> Dict[str, Any]:
    """Statistics of a numerical column without missing values."""
    arr = present.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    stats: Dict[str, Any] = {
        "infinite": int(len(arr) - len(finite)),
        "zeros": int((finite == 0).sum()),
        "negatives": int((finite < 0).sum()),
    }
    if len(finite):
        p25, median, p75 = np.quantile(finite, [0.25, 0.5, 0.75])
        fin = pd.Series(finite)
        stats.update(
            mean=float(finite.mean()),
            std=float(fin.std()),
            min=float(finite.min()),
            p25=float(p25),
            median=float(median),
            p75=float(p75),
            max=float(finite.max()),
            skewness=float(fin.skew()),
            kurtosis=float(fin.kurt()),
        )
    return stats


def _column_insights(
    frame: EDAFrame, col: str, dtp: Any, present: pd.Series, cfg: Config, nrows: int
) -> List[str]:
    """The names of the plot(df) insights that apply to a column."""
    if nrows == 0:
        return []
    if isinstance(dtp, Continuous):
        _, found = _format_cont_ins(col, _cont_calcs(present, cfg), nrows, cfg)
    elif isinstance(dtp, DateTime):
        pmiss = round((1 - len(present) / nrows) * 100, 2)
        found = [{"Missing": ""}] if pmiss > cfg.insight.missing__threshold else []
    else:
        strings = frame.get_col_as_str(col).dropna()
        _, found = _format_nom_ins(col, _nom_calcs(strings, cfg), nrows, cfg)
    return [name for insight in found for name in insight]
