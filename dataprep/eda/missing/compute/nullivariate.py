"""This module implements the plot_missing(df) function's
calculating intermediate part
"""

import warnings
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

from ...configs import Config
from ...eda_frame import EDAFrame
from ...intermediate import Intermediate
from ...staged import staged
from ...utils import cut_long_name


def _compute_missing_nullivariate(df: EDAFrame, cfg: Config) -> Generator[Any, Any, Intermediate]:
    """Calculate the data for visualizing the plot_missing(df).
    This contains the missing spectrum, missing bar chart and missing heatmap."""
    # pylint: disable=too-many-locals

    most_show = 5  # the most number of column/row to show in "insight"

    nullity = df.nulls
    null_cnts = nullity.sum(axis=0)
    nrows = df.shape[0]
    ncols = df.shape[1]
    if nrows == 0 or ncols == 0:
        raise ValueError("plot_missing needs a DataFrame with at least one row and one column.")
    total_nulls = int(null_cnts.sum())
    null_perc = null_cnts / nrows
    miss_perc = total_nulls / (nrows * ncols)
    avg_row = total_nulls / nrows
    avg_col = total_nulls / ncols

    tasks = (
        missing_spectrum(df, cfg.spectrum.bins) if cfg.spectrum.enable else None,
        null_perc if cfg.spectrum.enable or cfg.heatmap.enable else None,
        missing_bars(null_cnts, df.columns.values, nrows) if cfg.bar.enable else None,
        missing_heatmap(df) if cfg.heatmap.enable else None,
        # dendrogram cannot be computed for single column dataframe
        missing_dendrogram(df) if cfg.dendro.enable and ncols > 1 else None,
        total_nulls if cfg.stats.enable else None,
        missing_col_cnt(df) if cfg.stats.enable else None,
        missing_row_cnt(df) if cfg.stats.enable else None,
        missing_most_col(df) if cfg.insight.enable else None,
        missing_most_row(df) if cfg.insight.enable else None,
        miss_perc if cfg.stats.enable else None,
        avg_row if cfg.stats.enable else None,
        avg_col if cfg.stats.enable else None,
    )

    ### Lazy Region End
    (
        spectrum,
        null_perc,
        bars,
        heatmap,
        dendrogram,
        cnt,
        col_cnt,
        row_cnt,
        most_col,
        most_row,
        miss_perc,
        avg_row,
        avg_col,
    ) = yield tasks
    ### Eager Region Begin
    if cfg.heatmap.enable:
        sel = ~((null_perc == 0) | (null_perc == 1))
        heatmap = pd.DataFrame(
            data=heatmap[:, sel][sel, :], columns=df.columns[sel], index=df.columns[sel]
        )

    if cfg.stats.enable:
        missing_stat = {
            "Missing Cells": cnt,
            "Missing Cells (%)": str(round(miss_perc * 100, 1)) + "%",
            "Missing Columns": col_cnt,
            "Missing Rows": row_cnt,
            "Avg Missing Cells per Column": round(avg_col, 2),
            "Avg Missing Cells per Row": round(avg_row, 2),
        }

    if cfg.insight.enable:
        suffix_col = "" if most_col[0] <= most_show else ", ..."
        suffix_row = "" if most_row[0] <= most_show else ", ..."

        top_miss_col = (
            str(most_col[0])
            + " col(s): "
            + str(
                "("
                + ", ".join(cut_long_name(df.columns[e]) for e in most_col[2][:most_show])
                + suffix_col
                + ")"
            )
        )

        top_miss_row = (
            str(most_row[0])
            + " row(s): "
            + str("(" + ", ".join(str(e) for e in most_row[2][:most_show]) + suffix_row + ")")
        )

        insights = (
            {
                "Bar Chart": [
                    top_miss_col
                    + " contain the most missing values with rate "
                    + str(round(most_col[1] * 100, 1))
                    + "%",
                    top_miss_row
                    + " contain the most missing columns with rate "
                    + str(round(most_row[1] * 100, 1))
                    + "%",
                ]
            },
        )

    data_total_missing = {}
    if cfg.spectrum.enable:
        data_total_missing = {col: null_perc[i] for i, col in enumerate(df.columns)}

    return Intermediate(
        data_total_missing=data_total_missing,
        data_spectrum=pd.DataFrame(spectrum) if spectrum else spectrum,
        data_bars=bars,
        data_heatmap=heatmap,
        data_dendrogram=dendrogram,
        visual_type="missing_impact",
        missing_stat=missing_stat if cfg.stats.enable else {},
        insights=insights if cfg.insight.enable else {},
        ncols=ncols,
    )


# Not using decorator here because jupyter autoreload does not support it.
compute_missing_nullivariate = staged(_compute_missing_nullivariate)  # pylint: disable=invalid-name


# The dendrogram compares every pair of columns over all rows. On long tables it
# is built from a random sample of this many rows.
DENDROGRAM_MAX_ROWS = 100_000


def missing_spectrum(df: EDAFrame, bins: int) -> Dict[str, np.ndarray]:
    """Calculate a missing spectrum for each column.

    The rows are cut into (at most) ``bins`` consecutive blocks and the share of
    missing values is computed per block and column.
    """

    nrows, ncols = df.shape
    data = df.nulls

    num_bins = max(min(bins, nrows - 1), 1)
    bin_size = max(nrows // num_bins, 1)

    locs0 = np.arange(0, nrows, bin_size)
    locs1 = np.minimum(locs0 + bin_size, nrows)
    locs_middle = locs0 + bin_size / 2
    num_bins = len(locs0)

    # share of missing values per block: (num_bins, ncols)
    sums = np.add.reduceat(data.astype(np.int64), locs0, axis=0)
    rates = sums / (locs1 - locs0)[:, None]

    return {
        "column": np.repeat(np.asarray(df.columns.values, dtype=object), num_bins),
        "location": np.tile(locs_middle, ncols),
        "missing_rate": rates.T.ravel(),
        "loc_start": np.tile(locs0, ncols),
        "loc_end": np.tile(locs1, ncols),
    }


def missing_bars(
    null_cnts: np.ndarray, cols: np.ndarray, nrows: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a bar chart visualization of nullity correlation
    in the given DataFrame."""
    return nrows - null_cnts, null_cnts, cols


def missing_heatmap(df: EDAFrame) -> np.ndarray:
    """Calculate a heatmap visualization of nullity correlation
    in the given DataFrame."""
    nulls = df.nulls
    if nulls.shape[1] < 2 or nulls.shape[0] < 2:
        return np.full((nulls.shape[1], nulls.shape[1]), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.atleast_2d(np.corrcoef(nulls.astype(np.float32), rowvar=False))


def missing_dendrogram(df: EDAFrame) -> Any:
    """Calculate a missing values dendrogram."""
    nulls = df.nulls
    if nulls.shape[0] > DENDROGRAM_MAX_ROWS:
        rows = np.random.default_rng(0).choice(nulls.shape[0], DENDROGRAM_MAX_ROWS, replace=False)
        nulls = nulls[np.sort(rows)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", hierarchy.ClusterWarning)
        # Link the hierarchical output matrix, figure out orientation, construct base dendrogram.
        linkage_matrix = hierarchy.linkage(nulls.T, "average")
        dendrogram = hierarchy.dendrogram(
            Z=linkage_matrix,
            orientation="bottom",
            labels=list(df.columns),
            distance_sort="descending",
            no_plot=True,
        )

    return dendrogram


def missing_col_cnt(df: EDAFrame) -> int:
    """Calculate how many columns contain missing values."""
    return int(df.nulls.any(axis=0).sum())


def missing_row_cnt(df: EDAFrame) -> int:
    """Calculate how many rows contain missing values."""
    return int(df.nulls.any(axis=1).sum())


def missing_most_col(df: EDAFrame) -> Tuple[int, float, List[Any]]:
    """Find which column has the most number of missing values.

    Returns
    -------
    cnt
        the count of columns having the most missing values
    rate
        the highest rate of missing values in one column
    rst
        a list of column indices with highest missing rate
    """
    col_sum = df.nulls.sum(axis=0)
    maximum = col_sum.max()
    rate = maximum / df.shape[0]
    rst = np.where(col_sum == maximum)[0]

    return int(len(rst)), float(rate), rst


def missing_most_row(df: EDAFrame) -> Tuple[int, float, List[Any]]:
    """Find which row has the most number of missing values.

    Returns
    -------
    cnt
        the count of rows having the most missing values
    rate
        the highest rate of missing values in one row
    rst
        a list of row indices with highest missing rate
    """
    row_sum = df.nulls.sum(axis=1)
    maximum = row_sum.max()
    rate = maximum / df.shape[1]
    rst = np.where(row_sum == maximum)[0]

    return int(len(rst)), float(rate), rst
