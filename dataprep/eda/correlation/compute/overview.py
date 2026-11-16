"""Implementations of correlations.

Currently this boils down to pandas' implementation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...configs import Config
from ...eda_frame import EDAFrame
from ...intermediate import Intermediate
from ...utils import cut_long_name
from .common import CorrelationMethod


def _calc_overview(
    df: EDAFrame,
    cfg: Config,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    # pylint: disable=too-many-statements,too-many-locals,too-many-branches

    most_show = 6  # the most number of column/row to show in "insight"

    if value_range is not None and k is not None:
        raise ValueError("value_range and k cannot be present in both")

    num_df = df.select_num_columns()
    if len(num_df.columns) < 2:
        raise ValueError(
            "plot_correlation needs at least two numerical columns, "
            f"but the data has {len(num_df.columns)}."
        )

    cordx, cordy, corrs = correlation_nxn(num_df, cfg)

    if cfg.stats.enable or cfg.insight.enable:
        if cfg.stats.enable or cfg.pearson.enable:
            (
                pearson_pos_max,
                pearson_neg_max,
                pearson_mean,
                pearson_pos_cols,
                pearson_neg_cols,
            ) = most_corr(corrs[CorrelationMethod.Pearson])
            pearson_min, pearson_cols = least_corr(corrs[CorrelationMethod.Pearson])
        if cfg.stats.enable or cfg.spearman.enable:
            (
                spearman_pos_max,
                spearman_neg_max,
                spearman_mean,
                spearman_pos_cols,
                spearman_neg_cols,
            ) = most_corr(corrs[CorrelationMethod.Spearman])
            spearman_min, spearman_cols = least_corr(corrs[CorrelationMethod.Spearman])
        if cfg.stats.enable or cfg.kendall.enable:
            (
                kendalltau_pos_max,
                kendalltau_neg_max,
                kendalltau_mean,
                kendalltau_pos_cols,
                kendalltau_neg_cols,
            ) = most_corr(corrs[CorrelationMethod.KendallTau])
            kendalltau_min, kendalltau_cols = least_corr(corrs[CorrelationMethod.KendallTau])

    if cfg.stats.enable:
        tabledata = {
            "Highest Positive Correlation": {
                "Pearson": pearson_pos_max,
                "Spearman": spearman_pos_max,
                "KendallTau": kendalltau_pos_max,
            },
            "Highest Negative Correlation": {
                "Pearson": pearson_neg_max,
                "Spearman": spearman_neg_max,
                "KendallTau": kendalltau_neg_max,
            },
            "Lowest Correlation": {
                "Pearson": pearson_min,
                "Spearman": spearman_min,
                "KendallTau": kendalltau_min,
            },
            "Mean Correlation": {
                "Pearson": pearson_mean,
                "Spearman": spearman_mean,
                "KendallTau": kendalltau_mean,
            },
        }

    if cfg.insight.enable:
        insights: Dict[str, List[Any]] = {}
        if cfg.pearson.enable:
            p_p_corr = create_string("positive", pearson_pos_cols, most_show, num_df)
            p_n_corr = create_string("negative", pearson_neg_cols, most_show, num_df)
            p_corr = create_string("least", pearson_cols, most_show, num_df)
            insights["Pearson"] = [p_p_corr, p_n_corr, p_corr]
        if cfg.spearman.enable:
            s_p_corr = create_string("positive", spearman_pos_cols, most_show, num_df)
            s_n_corr = create_string("negative", spearman_neg_cols, most_show, num_df)
            s_corr = create_string("least", spearman_cols, most_show, num_df)
            insights["Spearman"] = [s_p_corr, s_n_corr, s_corr]
        if cfg.kendall.enable:
            k_p_corr = create_string("positive", kendalltau_pos_cols, most_show, num_df)
            k_n_corr = create_string("negative", kendalltau_neg_cols, most_show, num_df)
            k_corr = create_string("least", kendalltau_cols, most_show, num_df)
            insights["KendallTau"] = [k_p_corr, k_n_corr, k_corr]

    dfs = {}
    for method, corr in corrs.items():
        if (  # pylint: disable=too-many-boolean-expressions
            method == CorrelationMethod.Pearson
            and not cfg.pearson.enable
            or method == CorrelationMethod.Spearman
            and not cfg.spearman.enable
            or method == CorrelationMethod.KendallTau
            and not cfg.kendall.enable
        ):
            continue

        ndf = pd.DataFrame(
            {
                "x": num_df.columns[cordx],
                "y": num_df.columns[cordy],
                "correlation": corr.ravel(),
            }
        )
        ndf = ndf[cordy > cordx]  # Retain only lower triangle (w/o diag)

        if k is not None:
            thresh = ndf["correlation"].abs().nlargest(k).iloc[-1]
            ndf = ndf[(ndf["correlation"] >= thresh) | (ndf["correlation"] <= -thresh)]
        elif value_range is not None:
            mask = (value_range[0] <= ndf["correlation"]) & (ndf["correlation"] <= value_range[1])
            ndf = ndf[mask]

        dfs[method.name] = ndf

    return Intermediate(
        data=dfs,
        axis_range=list(num_df.columns.unique()),
        visual_type="correlation_impact",
        tabledata=tabledata if cfg.stats.enable else {},
        insights=insights if cfg.insight.enable else {},
    )


def correlation_nxn(
    df: EDAFrame, cfg: Config
) -> Tuple[np.ndarray, np.ndarray, Dict[CorrelationMethod, np.ndarray]]:
    """
    Calculation of a n x n correlation matrix for n columns

    Returns
    -------
        The long format of the correlations
    """

    ncols = len(df.columns)
    cordx, cordy = np.meshgrid(range(ncols), range(ncols))
    cordx, cordy = cordy.ravel(), cordx.ravel()

    corrs: Dict[CorrelationMethod, np.ndarray] = {}

    if cfg.pearson.enable or cfg.stats.enable:
        corrs[CorrelationMethod.Pearson] = _pearson_nxn(df)
    if cfg.spearman.enable or cfg.stats.enable:
        corrs[CorrelationMethod.Spearman] = _spearman_nxn(df)
    if cfg.kendall.enable or cfg.stats.enable:
        corrs[CorrelationMethod.KendallTau] = _kendall_tau_nxn(df)

    return cordx, cordy, corrs


# Kendall's tau costs O(n log n) per column pair and has no vectorised form, so
# on long tables it is estimated from a random sample of this many rows.
KENDALL_MAX_ROWS = 20_000


def _corr(df: EDAFrame, method: str) -> np.ndarray:
    if len(df.columns) == 0:
        return np.empty((0, 0))
    frame = df.frame
    if method == "kendall" and frame.shape[0] > KENDALL_MAX_ROWS:
        frame = frame.sample(n=KENDALL_MAX_ROWS, random_state=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # .copy(): the callers overwrite the diagonal in place
        return frame.corr(method=method).to_numpy(dtype=float).copy()


def _pearson_nxn(df: EDAFrame) -> np.ndarray:
    """Calculate column-wise pearson correlation."""
    return _corr(df, "pearson")


def _spearman_nxn(df: EDAFrame) -> np.ndarray:
    """Calculate column-wise spearman correlation."""
    return _corr(df, "spearman")


def _kendall_tau_nxn(df: EDAFrame) -> np.ndarray:
    """Calculate column-wise kendalltau correlation."""
    return _corr(df, "kendall")


def _upper_pairs(corrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices, column indices and values of the pairs above the diagonal."""
    rows, cols = np.triu_indices(corrs.shape[0], k=1)
    return rows, cols, corrs[rows, cols]


def most_corr(corrs: np.ndarray) -> Tuple[float, float, float, List[Any], List[Any]]:
    """Find the most correlated columns.

    Returns the highest positive correlation, the highest negative correlation,
    the mean correlation over all column pairs, and the column index pairs that
    reach the positive and the negative extreme. Undefined (NaN) correlations
    are ignored. The input matrix is not modified.
    """
    rows, cols, vals = _upper_pairs(corrs)
    valid = np.isfinite(vals)
    if not valid.any():
        return 0.0, 0.0, float("nan"), [], []

    p_maximum = max(float(vals[valid].max()), 0.0)
    n_maximum = min(float(vals[valid].min()), 0.0)
    mean = float(vals[valid].mean())

    positive = [(int(i), int(j)) for i, j in zip(rows[vals == p_maximum], cols[vals == p_maximum])]
    negative = [(int(i), int(j)) for i, j in zip(rows[vals == n_maximum], cols[vals == n_maximum])]

    return (
        round(p_maximum, 3),
        round(n_maximum, 3),
        round(mean, 3),
        positive if p_maximum != 0 else [],
        negative if n_maximum != 0 else [],
    )


def least_corr(corrs: np.ndarray) -> Tuple[float, List[Any]]:
    """Find the least correlated columns: the pairs whose correlation is closest to 0.

    Undefined (NaN) correlations are ignored. The input matrix is not modified.
    """
    rows, cols, vals = _upper_pairs(corrs)
    valid = np.isfinite(vals)
    if not valid.any():
        return float("nan"), []

    absvals = np.abs(vals)
    minimum = float(absvals[valid].min())
    sel = absvals == minimum
    return round(minimum, 3), [(int(i), int(j)) for i, j in zip(rows[sel], cols[sel])]


def create_string(flag: str, source: List[Any], most_show: int, df: EDAFrame) -> str:
    """Create the output string"""
    suffix = "" if len(source) <= most_show else ", ..."
    if flag == "positive":
        prefix = "Most positive correlated: "
        temp = "Most positive correlated: None"
    elif flag == "negative":
        prefix = "Most negative correlated: "
        temp = "Most negative correlated: None"
    elif flag == "least":
        prefix = "Least correlated: "
        temp = "Least correlated: None"

    if source != []:
        out = (
            prefix
            + ", ".join(
                "(" + cut_long_name(df.columns[e[0]]) + ", " + cut_long_name(df.columns[e[1]]) + ")"
                for e in source[:most_show]
            )
            + suffix
        )
    else:
        out = temp

    return out
