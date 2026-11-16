"""This module implements the plot_missing(df, x) function's
calculating intermediate part
"""

from typing import Any, Generator, List

import numpy as np
import pandas as pd

from ...configs import Config
from ...dtypes import Continuous, DateTime, GeoGraphy, GeoPoint, Nominal, SmallCardNum
from ...eda_frame import EDAFrame
from ...intermediate import ColumnsMetadata, Intermediate
from ...staged import staged
from ...utils import finite
from .common import LABELS, uni_histogram


def _compute_missing_univariate(  # pylint: disable=too-many-locals
    df: EDAFrame,
    x: str,
    cfg: Config,
) -> Generator[Any, Any, Intermediate]:
    """Calculate the distribution change on other columns when
    the missing values in x is dropped."""
    # pylint: disable = too-many-boolean-expressions

    if x not in df.columns:
        raise ValueError(f"Column {x!r} not found. Available columns: {list(df.columns)}")

    # rows where column x is present
    present = ~df.frame[x].isna().to_numpy()

    hists = {}

    for col in df.columns:
        col_dtype = df.get_eda_dtype(col)
        categorical = isinstance(col_dtype, (Nominal, GeoGraphy, GeoPoint, SmallCardNum, DateTime))
        if (
            col == x
            or (categorical and not cfg.bar.enable)
            or (isinstance(col_dtype, Continuous) and not cfg.hist.enable)
        ):
            continue

        if categorical:
            full = df.get_col_as_str(col)
        elif isinstance(col_dtype, Continuous):
            full = df.frame[col]
        else:
            raise ValueError(f"unprocessed type:{col_dtype}")

        srs0 = full.dropna()  # series from original dataframe
        srs1 = full[present].dropna()  # series with null rows from col x removed

        rng = None
        if isinstance(col_dtype, Continuous):
            # both histograms share the bins of the original data
            vals = finite(srs0)
            rng = (vals.min(), vals.max()) if len(vals) else None
        hists[col] = [uni_histogram(srs, col_dtype, cfg, rng) for srs in [srs0, srs1]]

    ### Lazy Region End
    hists = yield hists
    ### Eager Region Begin

    dfs = {}

    meta = ColumnsMetadata()

    for col_name, hists_ in hists.items():
        counts, xs, *edges = zip(*hists_)

        labels = np.repeat(LABELS, [len(x) for x in xs])

        data = {
            "x": np.concatenate(xs),
            "count": np.concatenate(counts),
            "label": labels,
        }

        if edges:
            lower_bound: List[float] = []
            upper_bound: List[float] = []

            for edge in edges[0]:
                lower_bound.extend(edge[:-1])
                upper_bound.extend(edge[1:])

            data["lower_bound"] = lower_bound
            data["upper_bound"] = upper_bound

        ret_df = pd.DataFrame(data)

        # If the cardinality of a categorical column is too large,
        # we show the top `num_bins` values, sorted by their count before drop
        col_dtype = df.get_eda_dtype(col_name)
        if len(counts[0]) > cfg.bar.bars and (
            isinstance(col_dtype, (Nominal, GeoGraphy, GeoPoint, SmallCardNum, DateTime))
        ):
            sortidx = np.argsort(-counts[0])
            selected_xs = xs[0][sortidx[: cfg.bar.bars]]
            ret_df = ret_df[ret_df["x"].isin(selected_xs)]
            meta[col_name, "shown"] = cfg.bar.bars
        else:
            meta[col_name, "shown"] = len(counts[0])
        meta[col_name, "total"] = len(counts[0])
        meta[col_name, "dtype"] = col_dtype
        dfs[col_name] = ret_df

    return Intermediate(data=dfs, x=x, meta=meta, visual_type="missing_impact_1vn")


# Not using decorator here because jupyter autoreload does not support it.
compute_missing_univariate = staged(_compute_missing_univariate)  # pylint: disable=invalid-name
