"""This module implements the intermediates computation
for plot_correlation(df) function."""

import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ...configs import Config
from ...eda_frame import EDAFrame
from ...intermediate import Intermediate
from .common import CorrelationMethod, corrcoef, kendalltau, nanrankdata


def _calc_univariate(
    df: EDAFrame,
    x: str,
    cfg: Config,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    """
    Parameters
    ----------
    df
        The dataframe for which plots are calculated.
    x
        A valid column name of the dataframe.
    cfg
        Config instance
    value_range
        If the correlation value is out of the range, don't show it.
    k
        Choose top-k element
    """

    num_df = df.select_num_columns()

    if len(num_df.columns) == 0:
        raise ValueError("plot_correlation needs numerical columns, but the data has none.")

    if x not in num_df.columns:
        raise ValueError(
            f"{x!r} is not a numerical column. Numerical columns: {list(num_df.columns)}"
        )

    columns = num_df.columns[num_df.columns != x]
    xarr = num_df.values[:, num_df.columns == x]
    data = num_df.values[:, num_df.columns != x]

    funcs = []
    if cfg.pearson.enable:
        funcs.append((CorrelationMethod.Pearson, _pearson_1xn))
    if cfg.spearman.enable:
        funcs.append((CorrelationMethod.Spearman, _spearman_1xn))
    if cfg.kendall.enable:
        funcs.append((CorrelationMethod.KendallTau, _kendall_tau_1xn))

    dfs = {}
    computed = {meth: func(xarr, data) for meth, func in funcs}

    for meth, corrs in computed.items():
        indices, corrs = _corr_filter(corrs, value_range, k)
        if len(indices) == 0:
            print(
                f"Correlation for {meth.name} is empty, try to broaden the value_range.",
                file=sys.stderr,
            )
        num_df = pd.DataFrame(
            {
                "x": np.full(len(indices), x),
                "y": columns[indices],
                "correlation": corrs,
            }
        )
        dfs[meth.name] = num_df

    return Intermediate(data=dfs, visual_type="correlation_single_heatmaps")


def _pearson_1xn(x: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Pearson correlation of column x (rows x 1) with every column of data.

    Each pair uses the rows where both values are present.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    data = np.asarray(data, dtype=float)
    xmask = ~np.isnan(x)

    corrs = np.empty(data.shape[1])
    for j in range(data.shape[1]):
        y = data[:, j]
        mask = xmask & ~np.isnan(y)
        corrs[j] = corrcoef(np.column_stack([x[mask], y[mask]]))
    return corrs


def _spearman_1xn(x: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Spearman correlation of column x with every column of data."""
    x = np.asarray(x, dtype=float).reshape(-1)
    data = np.asarray(data, dtype=float)
    xmask = ~np.isnan(x)

    corrs = np.empty(data.shape[1])
    for j in range(data.shape[1]):
        y = data[:, j]
        mask = xmask & ~np.isnan(y)
        # rank within the rows both columns share, so the result matches pandas
        corrs[j] = corrcoef(np.column_stack([nanrankdata(x[mask]), nanrankdata(y[mask])]))
    return corrs


def _kendall_tau_1xn(x: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Kendall's tau of column x with every column of data."""
    x = np.asarray(x, dtype=float).reshape(-1)
    data = np.asarray(data, dtype=float)
    xmask = ~np.isnan(x)

    corrs = np.empty(data.shape[1])
    for j in range(data.shape[1]):
        y = data[:, j]
        mask = xmask & ~np.isnan(y)
        corrs[j] = kendalltau(x[mask], y[mask])
    return corrs


def _corr_filter(
    corrs: np.ndarray,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter correlation values by k and value_range
    """

    if (value_range is not None) and (k is not None):
        raise ValueError("value_range and k cannot be present in both")

    if k is not None:
        sorted_idx = np.argsort(corrs)
        sorted_corrs = corrs[sorted_idx]
        # pylint: disable=invalid-unary-operand-type
        return (sorted_idx[-k:], corrs[sorted_idx[-k:]])
        # pylint: enable=invalid-unary-operand-type

    sorted_idx = np.argsort(corrs)
    sorted_idx = np.roll(sorted_idx, np.count_nonzero(np.isnan(corrs)))
    sorted_corrs = corrs[sorted_idx]

    if value_range is not None:
        start, end = value_range
        istart = np.searchsorted(sorted_corrs, start)
        iend = np.searchsorted(sorted_corrs, end, side="right")
        return sorted_idx[istart:iend], sorted_corrs[istart:iend]
    return sorted_idx, sorted_corrs
