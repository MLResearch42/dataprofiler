"""Computations for plot_diff([df1, df2, ..., dfn])."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...configs import Config
from ...dtypes import Continuous, DateTime, DType, DTypeDef, Nominal
from ...eda_frame import EDAFrame
from ...intermediate import Intermediate
from ...utils import (
    DTMAP,
    _get_timeunit,
    _period_labels,
    finite,
    gaussian_kde,
    histogram,
    value_counts,
)

__all__ = ["compare_multiple_df", "calc_stats", "baseline_position"]


def baseline_position(orig: List[str], labels: List[str], baseline: int) -> int:
    """Position of the baseline DataFrame among the DataFrames that have a column.

    ``orig`` lists the labels of the DataFrames containing the column. When the
    baseline DataFrame does not contain it, the first one that does is used.
    """
    base_label = labels[baseline]
    return orig.index(base_label) if base_label in orig else 0


def _plot_dtype(dtype: DType) -> DType:
    """Reduce a semantic type to the three kinds of chart plot_diff draws."""
    if isinstance(dtype, Continuous):
        return Continuous()
    if isinstance(dtype, DateTime):
        return DateTime()
    return Nominal()


def compare_multiple_df(
    df_list: List[pd.DataFrame], cfg: Config, dtype: Optional[DTypeDef]
) -> Intermediate:
    """
    Compute function for plot_diff([df...])

    Parameters
    ----------
    df_list
        DataFrames to be compared.
    cfg
        Config instance
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-branches, too-many-locals

    frames = [EDAFrame(df, dtype=dtype) for df in df_list]
    labels: List[str] = list(cfg.diff.label)  # type: ignore[arg-type]
    baseline: int = cfg.diff.baseline

    # every column name, in order of first appearance
    uniq_cols = list(dict.fromkeys(col for frame in frames for col in frame.columns))

    plot_data: List[Tuple[str, DType, Any, List[str]]] = []
    for col in uniq_cols:
        having = [i for i, frame in enumerate(frames) if col in frame.columns]
        orig = [labels[i] for i in having]
        base = baseline_position(orig, labels, baseline)
        # the baseline DataFrame decides how the column is plotted
        col_dtype = _plot_dtype(frames[having[base]].get_eda_dtype(col))

        if isinstance(col_dtype, Continuous) and cfg.hist.enable:
            srss = [_as_numeric(frames[i].frame[col]) for i in having]
            datum = _cont_calcs(srss, cfg)
            if cfg.diff.density:
                plot_data.append((col, col_dtype, (datum["kde"], datum["dens"]), orig))
            else:
                plot_data.append((col, col_dtype, datum["hist"], orig))
        elif isinstance(col_dtype, Nominal) and cfg.bar.enable:
            srss = [frames[i].get_col_as_str(col).dropna() for i in having]
            datum = _nom_calcs(srss, cfg, base)
            plot_data.append((col, col_dtype, (datum["bar"], datum["nuniq"]), orig))
        elif isinstance(col_dtype, DateTime) and cfg.line.enable:
            srss = [_as_datetime(frames[i].frame[col]) for i in having]
            plot_data.append((col, col_dtype, _calc_line_dt(srss, col, cfg.line.unit), orig))

    stats = calc_stats(frames, cfg)
    return Intermediate(data=plot_data, stats=stats, visual_type="comparison_grid")


def _as_numeric(srs: pd.Series) -> pd.Series:
    """The finite numerical values of a series (non-numbers are dropped)."""
    if not pd.api.types.is_numeric_dtype(srs.dtype) or pd.api.types.is_bool_dtype(srs.dtype):
        srs = pd.to_numeric(srs, errors="coerce")
    return finite(srs.dropna())


def _as_datetime(srs: pd.Series) -> pd.Series:
    """The valid timestamps of a series (other values are dropped)."""
    if not pd.api.types.is_datetime64_any_dtype(srs.dtype):
        srs = pd.to_datetime(srs, errors="coerce")
    return srs.dropna()


def calc_stats(frames: List[EDAFrame], cfg: Config) -> Dict[str, List[Any]]:
    """
    Calculate the statistics for plot_diff([df1, df2, ..., dfn])

    Parameters
    ----------
    frames
        DataFrames to be compared
    """
    # pylint: disable=import-outside-toplevel
    from ...distribution.compute.overview import _count_unique_rows, _get_dtype_cnts_and_num_cols

    stats: Dict[str, List[Any]] = {"nrows": [frame.shape[0] for frame in frames]}
    if cfg.stats.enable:
        stats["ncols"] = [frame.shape[1] for frame in frames]
        stats["npresent_cells"] = [
            int(
                frame.shape[0] * frame.shape[1]
                - sum(frame.get_missing_cnt(col) for col in frame.columns)
            )
            for frame in frames
        ]
        stats["nrows_wo_dups"] = [_count_unique_rows(frame.frame) for frame in frames]
        stats["mem_use"] = [int(frame.frame.memory_usage(deep=True).sum()) for frame in frames]
        stats["dtype_cnts"] = [_get_dtype_cnts_and_num_cols(frame)[0] for frame in frames]

    return stats


def shared_range(srss: List[pd.Series]) -> Optional[Tuple[float, float]]:
    """The (min, max) over several numerical series, or None when all are empty."""
    mins = [srs.min() for srs in srss if len(srs)]
    maxs = [srs.max() for srs in srss if len(srs)]
    if not mins:
        return None
    return float(min(mins)), float(max(maxs))


def density_and_kde(
    srss: List[pd.Series], bins: int, rng: Optional[Tuple[float, float]]
) -> Tuple[List[Any], List[Any]]:
    """A density histogram and a KDE per series; both None where the KDE is undefined."""
    dens: List[Any] = []
    kdes: List[Any] = []
    for srs in srss:
        arr = srs.to_numpy(dtype=float)
        smp = arr if arr.size <= 1000 else np.random.default_rng(0).choice(arr, 1000, False)
        kde = gaussian_kde(smp)
        kdes.append(kde)
        dens.append(histogram(arr, bins, rng, density=True) if kde is not None else None)
    return dens, kdes


def _cont_calcs(srss: List[pd.Series], cfg: Config) -> Dict[str, List[Any]]:
    """
    Computations for a continuous column in plot_diff([df1, df2, ..., dfn]).
    ``srss`` holds the finite values of the column in each DataFrame.
    """
    data: Dict[str, List[Any]] = {}
    rng = shared_range(srss)

    # histograms on shared bins, so that they can be overlaid
    data["hist"] = [histogram(srs.to_numpy(dtype=float), cfg.hist.bins, rng) for srs in srss]

    # density histograms and gaussian kernel density estimates
    if cfg.diff.density:
        data["dens"], data["kde"] = density_and_kde(srss, cfg.kde.bins, rng)

    return data


def _nom_calcs(srss: List[pd.Series], cfg: Config, base: int) -> Dict[str, Any]:
    """
    Computations for a nominal column in plot_diff([df1, df2, ..., dfn]).
    ``srss`` holds the non-null string values of the column in each DataFrame;
    ``base`` is the position of the baseline, whose top groups are the ones shown.
    """
    data: Dict[str, Any] = {}
    if cfg.bar.enable:
        grps = [value_counts(srs) for srs in srss]
        data["nuniq"] = grps[base].shape[0]
        # select the largest or smallest groups of the baseline
        grps[base] = (
            grps[base].nlargest(cfg.bar.bars)
            if cfg.bar.sort_descending
            else grps[base].nsmallest(cfg.bar.bars)
        )
        data["bar"] = grps
    return data


def _calc_line_dt(srss: List[pd.Series], x: str, unit: str) -> Tuple[List[pd.DataFrame], str]:
    """
    Calculate the frequency of timestamps over time, once per DataFrame.

    Parameters
    ----------
    srss
        The datetime column of each DataFrame, without nulls
    x
        The column name
    unit
        The unit of time over which to group the values, or "auto"
    """
    present = [srs for srs in srss if len(srs)]
    if unit == "auto":
        unit = (
            _get_timeunit(
                min(srs.min() for srs in present), max(srs.max() for srs in present), 100
            )
            if present
            else "day"
        )
    if unit not in DTMAP:
        raise ValueError(f"Unknown time unit {unit!r}. Use one of {list(DTMAP)} or 'auto'.")

    dfr: List[pd.DataFrame] = []
    for srs in srss:
        nrows = len(srs)
        df = srs.to_frame(name=x).groupby(pd.Grouper(key=x, freq=DTMAP[unit][0])).size()
        df = df.reset_index()
        df.columns = [x, "freq"]
        df["pct"] = df["freq"] / nrows * 100 if nrows else 0.0
        if unit == "week":
            df[x] = df[x] - pd.to_timedelta(6, unit="D")
        df["lbl"] = _period_labels(df[x], DTMAP[unit][1])
        dfr.append(df)

    return dfr, DTMAP[unit][3]
