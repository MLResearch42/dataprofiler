"""
Target-aware analysis: how strongly each column is associated with a target
column. Implements ``rank_features(df, target)`` and ``plot(df, target=...)``.

Every column gets a score between 0 (no association) and 1 (the column
determines the target), chosen by the types of the two columns:

==================  ==================  ============================================
column              target              score
==================  ==================  ============================================
numerical           numerical           absolute Spearman rank correlation
numerical           categorical         correlation ratio (eta) of the column
categorical         numerical           correlation ratio (eta) of the target
categorical         categorical         Cramér's V
==================  ==================  ============================================

Datetime columns count as numerical. The correlation ratio and Cramér's V are
bias-corrected, so a column with many distinct values (such as an ID) does not
look associated with the target just because it splits the rows finely.
"""

import html
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bokeh.models import (
    ColumnDataSource,
    CustomJSTickFormatter,
    FactorRange,
    HoverTool,
    LabelSet,
    Range1d,
)
from bokeh.plotting import figure

from .configs import Config
from .dtypes import Continuous, DateTime, DTypeDef, GeoPoint
from .eda_frame import DataFrame, EDAFrame, sample_rows
from .intermediate import Intermediate
from .parallel import Progress, pmap

__all__ = ["rank_features", "compute_target", "render_target", "RANKING_COLUMNS"]

# Columns of the table returned by rank_features().
RANKING_COLUMNS = ["type", "score", "method", "correlation", "n", "distinct", "note"]

SPEARMAN = "Spearman"
CORRELATION_RATIO = "Correlation ratio"
CRAMERS_V = "Cramér's V"

# Cramér's V: categories beyond the most frequent ones are grouped into one.
MAX_CATEGORIES = 100
# A score this high usually means the column is derived from the target.
LEAKAGE_THRESHOLD = 0.95
# Bars shown in the ranking chart.
MAX_FEATURES_SHOWN = 50
# Classes shown in the target distribution chart.
MAX_CLASSES_SHOWN = 30

METHOD_COLORS = {SPEARMAN: "#1f77b4", CORRELATION_RATIO: "#ff7f0e", CRAMERS_V: "#2ca02c"}


def rank_features(
    df: DataFrame,
    target: str,
    *,
    dtype: Optional[DTypeDef] = None,
    sample: Union[int, float, None] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Rank the columns by how strongly they are associated with a target column.

    Useful for a first look at which columns could predict the target, and for
    spotting columns that leak it.

    Parameters
    ----------
    df
        A pandas DataFrame, or any other input ``plot`` accepts.
    target
        The name of the target column. A numerical target is treated as a
        regression target, any other (including numbers with fewer than 10
        distinct values, such as 0/1 labels) as classes.
    dtype
        Override the detected column types, e.g. ``{"rating": "nominal"}`` to
        treat a numerical target as classes.
    sample
        Use a random sample of the rows: a number of rows (int) or a fraction
        (float). The sample is the same on every run.
    progress
        Show a progress bar.

    Returns
    -------
    pandas.DataFrame
        One row per column other than the target, strongest first, indexed by
        column name, with the columns:

        - ``type``: the DataPrep column type.
        - ``score``: the association, from 0 (none) to 1 (perfect). NaN when it
          cannot be computed (see ``note``).
        - ``method``: ``"Spearman"``, ``"Correlation ratio"`` or ``"Cramér's V"``
          (see the module documentation).
        - ``correlation``: the signed Spearman correlation, for a numerical
          column and a numerical target; NaN otherwise.
        - ``n``: the number of rows where both the column and the target are present.
        - ``distinct``: the number of distinct values of the column in those rows.
        - ``note``: warnings, e.g. a possible target leakage or an ID-like column.

    Examples
    --------
    >>> from dataprep.datasets import load_dataset
    >>> from dataprep.eda import rank_features
    >>> ranking = rank_features(load_dataset("titanic"), "Survived")
    >>> ranking.index[0]
    'Sex'
    """
    if isinstance(df, EDAFrame) and sample is None and dtype is None:
        frame = df
    else:
        data, _ = sample_rows(df, sample)
        frame = EDAFrame(data, dtype=dtype)
    with Progress(enabled=progress):
        return _rank(frame, target)


def compute_target(
    df: DataFrame,
    target: str,
    cfg: Optional[Config] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """The computations behind ``plot(df, target=...)``."""
    cfg = cfg or Config()
    frame = EDAFrame(df, dtype=dtype)
    ranking = _rank(frame, target)
    target = ranking.attrs["target"]
    values = frame.frame[target]
    if ranking.attrs["target_type"] == "categorical":
        counts = frame.get_col_as_str(target).value_counts()
        dist: Any = ("categorical", counts)
    else:
        arr = _numeric_values(values, frame.get_eda_dtype(target))
        arr = arr[~np.isnan(arr)]
        dist = ("numerical", np.histogram(arr, bins=cfg.hist.bins or 50))
    return Intermediate(
        ranking=ranking,
        target=target,
        target_dist=dist,
        nrows=frame.shape[0],
        visual_type="target_ranking",
    )


def _rank(frame: EDAFrame, target: Any) -> pd.DataFrame:
    """rank_features() on an EDAFrame."""
    if target not in frame.columns and str(target) in frame.columns:
        target = str(target)
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} not found in the DataFrame.")
    if frame.shape[1] < 2:
        raise ValueError("The DataFrame has no columns other than the target.")

    tdtype = frame.get_eda_dtype(target)
    if isinstance(tdtype, GeoPoint):
        raise ValueError(
            f"The target column {target!r} holds latitude/longitude pairs, "
            "which cannot be used as a target."
        )
    numeric_target = isinstance(tdtype, (Continuous, DateTime))
    if numeric_target:
        yvals: np.ndarray = _numeric_values(frame.frame[target], tdtype)
        ymask = ~np.isnan(yvals)
    else:
        ystr = frame.get_col_as_str(target)
        ymask = ystr.notna().to_numpy()
        yvals = ystr.to_numpy(dtype=object)
    present = yvals[ymask]
    if len(pd.unique(present)) < 2:
        raise ValueError(f"The target column {target!r} needs at least two distinct values.")

    features = [col for col in frame.columns if col != target]
    rows = pmap(
        lambda col: _associate(frame, col, yvals, ymask, numeric_target),
        features,
        desc="Columns",
    )
    table = pd.DataFrame(rows, index=pd.Index(features, name="column"), columns=RANKING_COLUMNS)
    table["n"] = table["n"].astype("int64")
    table["distinct"] = table["distinct"].astype("int64")
    table = table.sort_values("score", ascending=False, na_position="last", kind="stable")
    table.attrs["target"] = target
    table.attrs["target_type"] = "numerical" if numeric_target else "categorical"
    return table


def _numeric_values(srs: pd.Series, dtp: Any) -> np.ndarray:
    """A column as floats, NaN where missing or infinite. Datetimes become seconds."""
    if isinstance(dtp, DateTime):
        stamps = srs.to_numpy(dtype="datetime64[ns]")
        out = stamps.astype("int64").astype(float) / 1e9
        out[np.isnat(stamps)] = np.nan
        return out
    out = srs.to_numpy(dtype=float, na_value=np.nan).copy()
    out[~np.isfinite(out)] = np.nan
    return out


def _associate(
    frame: EDAFrame, col: str, yvals: np.ndarray, ymask: np.ndarray, numeric_target: bool
) -> Dict[str, Any]:
    """One row of the ranking: the association of a column with the target."""
    dtp = frame.get_eda_dtype(col)
    row: Dict[str, Any] = {
        "type": type(dtp).__name__,
        "score": np.nan,
        "method": None,
        "correlation": np.nan,
        "n": 0,
        "distinct": 0,
        "note": "",
    }
    if isinstance(dtp, GeoPoint):
        row["note"] = "latitude/longitude columns are not ranked"
        return row

    numeric = isinstance(dtp, (Continuous, DateTime))
    if numeric:
        xvals = _numeric_values(frame.frame[col], dtp)
        mask = ~np.isnan(xvals) & ymask
    else:
        xstr = frame.get_col_as_str(col)
        mask = xstr.notna().to_numpy() & ymask
        xvals = xstr.to_numpy(dtype=object)
    x, y = xvals[mask], yvals[mask]
    row["n"] = n = int(mask.sum())
    row["distinct"] = distinct = len(pd.unique(x)) if n else 0

    if numeric and numeric_target:
        row["method"] = SPEARMAN
    elif numeric or numeric_target:
        row["method"] = CORRELATION_RATIO
    else:
        row["method"] = CRAMERS_V

    notes: List[str] = []
    if n < 3:
        row["note"] = "fewer than 3 rows with both values present"
        return row
    if distinct < 2:
        row["score"] = 0.0
        row["note"] = "constant column"
        return row

    if numeric and numeric_target:
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        row["correlation"] = rho
        row["score"] = abs(rho)
    elif numeric:
        row["score"] = correlation_ratio(x, y)
    elif numeric_target:
        row["score"] = correlation_ratio(y, x)
    else:
        row["score"], grouped = cramers_v(x, y)

    if not numeric and distinct > 0.5 * n and distinct > 20:
        notes.append("mostly unique values (an identifier?)")
    elif not numeric and not numeric_target and grouped:
        notes.append(f"{distinct} categories; all but the {MAX_CATEGORIES - 1} most common grouped")
    if row["score"] >= LEAKAGE_THRESHOLD:
        notes.append("very strong association: check for target leakage")
    elif np.isnan(row["score"]) and not notes:
        notes.append("too few rows for the number of distinct values")
    row["note"] = "; ".join(notes)
    return row


def correlation_ratio(values: np.ndarray, groups: np.ndarray) -> float:
    """Bias-corrected correlation ratio of numerical ``values`` grouped by ``groups``.

    0 when the group means are all the same, 1 when the groups determine the
    values. The correction (epsilon squared) subtracts the share of variance
    that the number of groups explains by chance.
    """
    values = np.asarray(values, dtype=float)
    codes, uniques = pd.factorize(groups)
    k, n = len(uniques), len(values)
    if k < 2 or n - k <= 0:
        return float("nan")
    total_ss = float(((values - values.mean()) ** 2).sum())
    if total_ss == 0:
        return 0.0
    counts = np.bincount(codes, minlength=k)
    means = np.bincount(codes, weights=values, minlength=k) / counts
    between_ss = float((counts * (means - values.mean()) ** 2).sum())
    within_ss = max(total_ss - between_ss, 0.0)
    eps2 = (between_ss - (k - 1) * within_ss / (n - k)) / total_ss
    return math.sqrt(min(max(eps2, 0.0), 1.0))


def cramers_v(a: np.ndarray, b: np.ndarray) -> Tuple[float, bool]:
    """Bias-corrected Cramér's V of two categorical arrays (Bergsma, 2013).

    Returns the statistic and whether rare categories were grouped (at most
    ``MAX_CATEGORIES`` per array are kept).
    """
    a, grouped_a = _group_rare(a)
    b, grouped_b = _group_rare(b)
    ca, ua = pd.factorize(a)
    cb, ub = pd.factorize(b)
    r, k = len(ua), len(ub)
    n = len(ca)
    if r < 2 or k < 2 or n < 2:
        return 0.0, grouped_a or grouped_b
    table = np.bincount(ca * k + cb, minlength=r * k).reshape(r, k).astype(float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    phi2 = max(chi2 / n - (k - 1) * (r - 1) / (n - 1), 0.0)
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return float("nan"), grouped_a or grouped_b
    return math.sqrt(min(phi2 / denom, 1.0)), grouped_a or grouped_b


def _group_rare(values: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Replace all but the most frequent categories by one "(other)" category."""
    counts = pd.Series(values).value_counts()
    if len(counts) <= MAX_CATEGORIES:
        return values, False
    keep = counts.index[: MAX_CATEGORIES - 1]
    out = np.where(pd.Series(values).isin(keep).to_numpy(), values, "\x00(other)")
    return out, True


########## Rendering ##########

_TRUNCATE_JS = """
    if (tick.length > 22) return tick.substring(0, 20) + '...';
    else return tick;
"""


def render_target(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """Render the ranking chart and the distribution of the target."""
    width = cfg.plot.width if cfg.plot.width is not None else 600
    height = cfg.plot.height if cfg.plot.height is not None else 400
    ranking: pd.DataFrame = itmdt["ranking"]
    target: str = itmdt["target"]

    figs = [_ranking_figure(ranking, target, width)]
    meta = ["Feature Ranking"]
    dist_fig = _distribution_figure(itmdt["target_dist"], target, width, height)
    if dist_fig is not None:
        figs.append(dist_fig)
        meta.append("Target Distribution")

    return {
        "layout": figs,
        "meta": meta,
        "insights": {"Feature Ranking": _insights(itmdt)},
        "container_width": width + 80,
        "title": f"Columns associated with {target}",
    }


def _ranking_figure(ranking: pd.DataFrame, target: str, width: int) -> figure:
    scored = ranking[ranking["score"].notna()]
    shown = scored.head(MAX_FEATURES_SHOWN)
    names = [str(name) for name in shown.index]
    title = f"Association with {target}"
    if len(scored) > len(shown):
        title += f" (top {len(shown)} of {len(scored)} columns)"

    source = ColumnDataSource(
        {
            "column": names,
            "score": shown["score"].to_numpy(dtype=float),
            "label": [f"{val:.2f}" for val in shown["score"]],
            "method": shown["method"].astype(str).tolist(),
            "correlation": ["" if pd.isna(val) else f"{val:+.3f}" for val in shown["correlation"]],
            "n": shown["n"].tolist(),
            "note": shown["note"].tolist(),
            "color": [METHOD_COLORS.get(m, "#7f7f7f") for m in shown["method"]],
        }
    )
    fig = figure(
        y_range=FactorRange(*reversed(names)) if names else FactorRange(""),
        x_range=Range1d(0, 1.12),
        width=width,
        height=max(160, 24 * len(names) + 90),
        title=title,
        toolbar_location=None,
        tools="",
        background_fill_color="#fafafa",
    )
    bars = fig.hbar(
        y="column",
        right="score",
        height=0.7,
        source=source,
        color="color",
        legend_field="method",
    )
    fig.add_layout(
        LabelSet(
            x="score",
            y="column",
            text="label",
            source=source,
            x_offset=4,
            text_baseline="middle",
            text_font_size="8pt",
        )
    )
    fig.add_tools(
        HoverTool(
            renderers=[bars],
            tooltips=[
                ("column", "@column"),
                ("score", "@score{0.000}"),
                ("method", "@method"),
                ("correlation", "@correlation"),
                ("rows used", "@n"),
                ("note", "@note"),
            ],
        )
    )
    fig.xaxis.axis_label = "association (0 = none, 1 = perfect)"
    fig.yaxis.formatter = CustomJSTickFormatter(code=_TRUNCATE_JS)
    fig.ygrid.grid_line_color = None
    fig.axis.major_label_text_font_size = "9pt"
    fig.legend.location = "bottom_right"
    fig.legend.label_text_font_size = "8pt"
    return fig


def _distribution_figure(dist: Any, target: str, width: int, height: int) -> Optional[figure]:
    kind, data = dist
    if kind == "categorical":
        counts: pd.Series = data.head(MAX_CLASSES_SHOWN)
        if counts.empty:
            return None
        total = int(data.sum())
        labels = [str(val) for val in counts.index]
        source = ColumnDataSource(
            {
                "cls": labels,
                "count": counts.to_numpy(),
                "pct": [f"{100 * c / total:.1f}%" for c in counts],
            }
        )
        title = f"Classes of {target}"
        if len(data) > len(counts):
            title += f" (top {len(counts)} of {len(data)})"
        fig = figure(
            x_range=FactorRange(*labels),
            width=width,
            height=height,
            title=title,
            toolbar_location=None,
            tools="",
            background_fill_color="#fafafa",
        )
        bars = fig.vbar(x="cls", top="count", width=0.8, source=source, color="#1f77b4")
        fig.add_tools(
            HoverTool(
                renderers=[bars],
                tooltips=[("class", "@cls"), ("count", "@count"), ("share", "@pct")],
            )
        )
        fig.xaxis.formatter = CustomJSTickFormatter(code=_TRUNCATE_JS)
        if len(labels) > 6:
            fig.xaxis.major_label_orientation = math.pi / 3
        fig.y_range.start = 0
    else:
        counts_arr, edges = data
        if len(counts_arr) == 0:
            return None
        source = ColumnDataSource({"left": edges[:-1], "right": edges[1:], "count": counts_arr})
        fig = figure(
            width=width,
            height=height,
            title=f"Distribution of {target}",
            toolbar_location=None,
            tools="",
            background_fill_color="#fafafa",
        )
        quads = fig.quad(
            left="left",
            right="right",
            bottom=0,
            top="count",
            source=source,
            fill_color="#1f77b4",
            line_color="white",
        )
        fig.add_tools(
            HoverTool(
                renderers=[quads],
                tooltips=[("range", "[@left{0.[00]}, @right{0.[00]})"), ("count", "@count")],
            )
        )
        fig.y_range.start = 0
    fig.xgrid.grid_line_color = None
    fig.axis.major_label_text_font_size = "9pt"
    return fig


def _insights(itmdt: Intermediate) -> List[str]:
    """Warnings shown in the insight panel of the ranking chart (HTML-escaped)."""
    ranking: pd.DataFrame = itmdt["ranking"]
    out: List[str] = []
    for col, note in ranking["note"].items():
        if note:
            out.append(f"<b>{html.escape(str(col))}</b>: {html.escape(note)}")
    kind, data = itmdt["target_dist"]
    if kind == "categorical" and len(data) >= 2:
        share = data.iloc[-1] / data.sum()
        if share < 0.1:
            smallest = html.escape(str(data.index[-1]))
            out.insert(
                0,
                f"Imbalanced classes: the smallest class, <b>{smallest}</b>, "
                f"has {share:.1%} of the rows",
            )
    return out
