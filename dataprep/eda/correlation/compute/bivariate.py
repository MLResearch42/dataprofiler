"""This module implements the intermediates computation
for plot_correlation(df, x, y) function."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ...configs import Config
from ...eda_frame import EDAFrame
from ...intermediate import Intermediate
from ...utils import sample_frame


def _calc_bivariate(
    df: EDAFrame,
    cfg: Config,
    x: str,
    y: str,
    *,
    k: Optional[int] = None,
) -> Intermediate:
    num_df: pd.DataFrame = df.select_num_columns().frame

    for col in (x, y):
        if col not in num_df.columns:
            raise ValueError(
                f"{col!r} is not a numerical column. Numerical columns: {list(num_df.columns)}"
            )

    num_df = num_df[[x, y]].dropna()
    coeffs, df_smp, influences = scatter_with_regression(num_df, cfg, k=k)

    result = {"coeffs": coeffs, "data": df_smp}

    if influences is not None and k is not None:
        infidx = np.argsort(influences)
        labels = np.full(len(influences), "=")
        # pylint: disable=invalid-unary-operand-type
        labels[infidx[-k:]] = "-"
        # pylint: enable=invalid-unary-operand-type
        labels[infidx[:k]] = "+"
        result["data"]["influence"] = labels

    return Intermediate(**result, visual_type="correlation_scatter")


def scatter_with_regression(
    df: pd.DataFrame, cfg: Config, k: Optional[int] = None
) -> Tuple[Tuple[float, float], pd.DataFrame, Optional[np.ndarray]]:
    """Fit the regression line y = a * x + b on all rows, then sample rows for the scatter."""
    xarr = df.iloc[:, 0].to_numpy(dtype=float)
    yarr = df.iloc[:, 1].to_numpy(dtype=float)
    finite = np.isfinite(xarr) & np.isfinite(yarr)

    if finite.sum() >= 2 and np.ptp(xarr[finite]) > 0:
        design = np.column_stack([xarr[finite], np.ones(int(finite.sum()))])
        (coeffa, coeffb), *_ = np.linalg.lstsq(design, yarr[finite], rcond=None)
    else:
        coeffa, coeffb = np.nan, np.nan

    df_smp = sample_frame(df, cfg.scatter.sample_size, cfg.scatter.sample_rate).copy()
    # NOTE: influences are computed on the sample that is drawn
    influences = (
        pearson_influence(
            df_smp.iloc[:, 0].to_numpy(dtype=float), df_smp.iloc[:, 1].to_numpy(dtype=float)
        )
        if k
        else None
    )

    return (float(coeffa), float(coeffb)), df_smp, influences


def pearson_influence(xarr: np.ndarray, yarr: np.ndarray) -> np.ndarray:
    """The Pearson correlation that would result from deleting each point in turn."""

    if xarr.shape != yarr.shape:
        raise ValueError(
            f"The shape of xarr and yarr should be same, got {xarr.shape}, {yarr.shape}"
        )

    n = xarr.shape[0]
    x2, y2 = np.square(xarr), np.square(yarr)
    xy = xarr * yarr

    xsum, ysum, xysum = xarr.sum(), yarr.sum(), xy.sum()
    x2sum, y2sum = x2.sum(), y2.sum()

    # (n-1)^2 is multiplied into both numerator and denominator to avoid divisions.
    numerator = (n - 1) * (xysum - xy) - (xsum - xarr) * (ysum - yarr)
    varx = (n - 1) * (x2sum - x2) - np.square(xsum - xarr)
    vary = (n - 1) * (y2sum - y2) - np.square(ysum - yarr)

    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / np.sqrt(varx * vary)
