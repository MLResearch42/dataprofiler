"""Computations for plot_diff([df1, df2, ..., dfn], x)."""

import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ....errors import DataprepError
from ...configs import Config
from ...correlation.compute.common import CorrelationMethod
from ...correlation.compute.univariate import (
    _corr_filter,
    _kendall_tau_1xn,
    _pearson_1xn,
    _spearman_1xn,
)
from ...distribution.compute.univariate import _calc_box, _quantiles
from ...dtypes import Continuous
from ...eda_frame import EDAFrame
from ...intermediate import Intermediate
from ...utils import finite, histogram, kurtosis, skew, value_counts
from .multiple_df import density_and_kde, shared_range

__all__ = ["compare_multiple_col"]


def compare_multiple_col(
    df_list: List[pd.DataFrame],
    x: str,
    cfg: Config,
) -> Intermediate:
    """
    Compute function for plot_diff([df...], x)

    Parameters
    ----------
    df_list
        DataFrames to be compared.
    x
        Name of the column to be compared
    cfg
        Config instance
    """
    labels: List[str] = list(cfg.diff.label)  # type: ignore[arg-type]
    baseline: int = cfg.diff.baseline

    all_frames = [EDAFrame(df) for df in df_list]
    having = [i for i, frame in enumerate(all_frames) if x in frame.columns]
    frames = [all_frames[i] for i in having]
    used_labels = [labels[i] for i in having]
    base = used_labels.index(labels[baseline]) if labels[baseline] in used_labels else 0

    col_dtype = frames[base].get_eda_dtype(x)
    if not isinstance(col_dtype, Continuous) or not all(
        pd.api.types.is_numeric_dtype(frame.frame[x].dtype) for frame in frames
    ):
        raise DataprepError(
            f"plot_diff(dfs, x) compares a numerical column, but {x!r} was detected as "
            f"{type(col_dtype).__name__}. Use plot_diff(dfs) to compare all columns."
        )

    srss = [frame.frame[x] for frame in frames]
    data = [_cont_calcs([finite(srs.dropna()) for srs in srss], cfg, frames, x)]
    stats = calc_stats_cont(srss, cfg)

    return Intermediate(
        col=x, data=data, stats=stats, labels=used_labels, visual_type="comparison_continuous"
    )


def _cont_calcs(
    srss: List[pd.Series], cfg: Config, frames: List[EDAFrame], x: str
) -> Dict[str, List[Any]]:
    """
    Computations for a continuous column in plot_diff([df...], x).
    ``srss`` holds the finite values of the column in each DataFrame.
    """
    # pylint:disable = too-many-branches, too-many-locals

    data: Dict[str, List[Any]] = {}
    rng = shared_range(srss)

    # histograms on shared bins
    if cfg.hist.enable:
        data["hist"] = [histogram(srs.to_numpy(dtype=float), cfg.hist.bins, rng) for srs in srss]
    # density histograms and gaussian kernel density estimates
    if cfg.kde.enable:
        data["dens"], data["kde"] = density_and_kde(srss, cfg.kde.bins, rng)
    if cfg.box.enable:
        data["box"] = [_calc_box(srs, _quantiles(srs, [0.25, 0.5, 0.75]), cfg) for srs in srss]
    if cfg.correlations.enable:
        data["corr"] = [_correlations(frame, x, cfg) for frame in frames]
    if cfg.value_table.enable:
        data["value_table"] = [
            value_counts(srs).nlargest(cfg.value_table.ngroups) for srs in srss
        ]
    return data


def _correlations(frame: EDAFrame, x: str, cfg: Config) -> Dict[str, pd.DataFrame]:
    """Correlation of column x with every other numerical column of one DataFrame."""
    num_df = frame.select_num_columns()
    columns = num_df.columns[num_df.columns != x]
    xarr = num_df.values[:, num_df.columns == x]
    data_corr = num_df.values[:, num_df.columns != x]

    funcs = []
    if cfg.pearson.enable:
        funcs.append((CorrelationMethod.Pearson, _pearson_1xn))
    if cfg.spearman.enable:
        funcs.append((CorrelationMethod.Spearman, _spearman_1xn))
    if cfg.kendall.enable:
        funcs.append((CorrelationMethod.KendallTau, _kendall_tau_1xn))

    dfs = {}
    for meth, func in funcs:
        indices, corrs = _corr_filter(
            func(xarr, data_corr), cfg.correlations.value_range, cfg.correlations.k
        )
        if len(indices) == 0 and len(columns) > 0:
            print(
                f"Correlation for {meth.name} is empty, try to broaden the value_range.",
                file=sys.stderr,
            )
        dfs[meth.name] = pd.DataFrame(
            {
                "x": np.full(len(indices), x),
                "y": columns[indices],
                "correlation": corrs,
            }
        )
    return dfs


def calc_stats_cont(srss: List[pd.Series], cfg: Config) -> Dict[str, List[Any]]:
    """
    Calculate the statistics for plot_diff([df1, df2, ..., dfn], x)

    Parameters
    ----------
    srss
        The column in each DataFrame (nulls included)
    """
    stats: Dict[str, List[Any]] = {"nrows": [srs.shape[0] for srs in srss]}
    if cfg.stats.enable:
        present = [srs.dropna() for srs in srss]
        stats["npres"] = [srs.shape[0] for srs in present]  # number of present values
        # remove infinite values
        reals = [finite(srs) for srs in present]
        arrs = [srs.to_numpy(dtype=float) for srs in reals]
        stats["qntls"] = [_quantiles(srs, [0.05, 0.25, 0.5, 0.75, 0.95]) for srs in reals]
        stats["skew"] = [skew(arr) for arr in arrs]
        stats["mean"] = [arr.mean() if arr.size else np.nan for arr in arrs]
        stats["std"] = [arr.std(ddof=1) if arr.size > 1 else np.nan for arr in arrs]
        stats["min"] = [arr.min() if arr.size else np.nan for arr in arrs]
        stats["max"] = [arr.max() if arr.size else np.nan for arr in arrs]
        stats["nreals"] = [arr.size for arr in arrs]
        stats["nzero"] = [int((arr == 0).sum()) for arr in arrs]
        stats["nneg"] = [int((arr < 0).sum()) for arr in arrs]
        stats["kurt"] = [kurtosis(arr) for arr in arrs]
        stats["mem_use"] = [int(srs.memory_usage(deep=True)) for srs in reals]
        stats["nuniq"] = [int(srs.nunique()) for srs in reals]
    return stats
