"""
This module implements the visualization for the plot_diff function.
"""  # pylint: disable=too-many-lines

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bokeh.layouts import row
from bokeh.models import (
    FactorRange,
    HoverTool,
    TabPanel,
)
from bokeh.plotting import figure
from bokeh.transform import dodge

from ..configs import Config
from ..correlation.render import create_color_mapper
from ..distribution.render import _format_values as _format_value
from ..distribution.render import _ratio
from ..dtypes import Continuous, DateTime, Nominal, is_dtype
from ..intermediate import Intermediate
from ..palette import CATEGORY10, RDBU
from ..utils import _format_axis, _format_bin_intervals, tweak_figure
from .compute.multiple_df import baseline_position

__all__ = ["render_diff"]


def _format_values(key: str, value: List[Any]) -> List[str]:
    """Format one statistic for every DataFrame being compared."""
    return [_format_value(key, val) for val in value]


def _source_colors(sources: List[str], df_labels: List[str]) -> List[str]:
    """The colour of each source: the one its DataFrame has in the legend."""
    return [
        CATEGORY10[(df_labels.index(lbl) if lbl in df_labels else i) % len(CATEGORY10)]
        for i, lbl in enumerate(sources)
    ]


def bar_viz(
    df: List[pd.DataFrame],
    ttl_grps: int,
    nrows: List[int],
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    orig: List[str],
    df_labels: List[str],
    baseline: int,
) -> figure:
    """
    Render a bar chart
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # every source is shown on the baseline's groups
    base_index = df[baseline].index
    df = [
        (srs if isinstance(srs, pd.Series) else srs[col])
        .reindex(index=base_index, fill_value=0)
        .to_frame(name=col)
        for srs in df
    ]
    colors = _source_colors(orig, df_labels)

    tooltips = [
        (col, "@index"),
        ("Count", f"@{{{col}}}"),
        ("Percent", "@pct{0.2f}%"),
        ("Source", "@orig"),
    ]

    if show_yticks:
        if len(df[baseline]) > 10:
            plot_width = 28 * len(df[baseline])
    fig = figure(
        width=plot_width,
        height=plot_height,
        title=col,
        toolbar_location=None,
        tooltips=tooltips,
        tools="hover",
        x_range=list(df[baseline].index),
        y_axis_type=yscale,
    )

    offset = np.linspace(-0.08 * len(df), 0.08 * len(df), len(df)) if len(df) > 1 else [0]
    for i, (nrow, data) in enumerate(zip(nrows, df)):
        data["pct"] = data[col] / nrow * 100 if nrow else 0.0
        data.index = [str(val) for val in data.index]
        data["orig"] = orig[i]

        fig.vbar(
            x=dodge("index", offset[i], range=fig.x_range),
            width=0.6 / len(df),
            top=col,
            bottom=0.01,
            source=data,
            fill_color=colors[i],
            line_color=colors[i],
        )
    tweak_figure(fig, "bar", show_yticks)

    fig.yaxis.axis_label = "Count"

    x_axis_label = ""
    if ttl_grps > len(df[baseline]):
        x_axis_label += f"Top {len(df[baseline])} of {ttl_grps} {col}"

    if orig != df_labels:
        if x_axis_label:
            x_axis_label += f", this vairable is only in {','.join(orig)}"
        else:
            x_axis_label += f"This vairable is only in {','.join(orig)}"

    fig.xaxis.axis_label = x_axis_label
    fig.xaxis.axis_label_standoff = 0

    if show_yticks and yscale == "linear":
        _format_axis(fig, 0, df[baseline].max(), "y")
    return fig


def hist_viz(
    hist: List[Tuple[np.ndarray, np.ndarray]],
    nrows: List[int],
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    df_labels: List[str],
    orig: Optional[List[str]] = None,
) -> figure:
    """
    Render a histogram
    """
    # pylint: disable=too-many-arguments,too-many-locals

    tooltips = [
        ("Bin", "@intvl"),
        ("Frequency", "@freq"),
        ("Percent", "@pct{0.2f}%"),
        ("Source", "@orig"),
    ]
    fig = figure(
        height=plot_height,
        width=plot_width,
        title=col,
        toolbar_location=None,
        y_axis_type=yscale,
    )

    sources = orig if orig else df_labels
    colors = _source_colors(sources, df_labels)
    df = None
    max_freq = 0
    for i, hst in enumerate(hist):
        counts, bins = hst
        if sum(counts) == 0:
            fig.rect(x=0, y=0, width=0, height=0)
            continue
        intvls = _format_bin_intervals(bins)
        df = pd.DataFrame(
            {
                "intvl": intvls,
                "left": bins[:-1],
                "right": bins[1:],
                "freq": counts,
                "pct": counts / nrows[i] * 100 if nrows[i] else 0.0,
                "orig": sources[i] if i < len(sources) else None,
            }
        )
        max_freq = max(max_freq, int(counts.max()))
        bottom = 0 if yscale == "linear" or df.empty else max(counts.min() / 2, 0.1)
        fig.quad(
            source=df,
            left="left",
            right="right",
            bottom=bottom,
            alpha=0.5,
            top="freq",
            fill_color=colors[i],
            line_color=colors[i],
        )
    hover = HoverTool(tooltips=tooltips, attachment="vertical", mode="vline")
    fig.add_tools(hover)

    tweak_figure(fig, "hist", show_yticks)
    fig.yaxis.axis_label = "Frequency"

    if df is not None:
        _format_axis(fig, df.iloc[0]["left"], df.iloc[-1]["right"], "x")

    x_axis_label = ""
    if show_yticks:
        x_axis_label += col
        if yscale == "linear":
            _format_axis(fig, 0, max_freq, "y")
    if orig:
        if orig != df_labels:
            if x_axis_label:
                x_axis_label += f", this vairable is only in {','.join(orig)}"
            else:
                x_axis_label += f"This vairable is only in {','.join(orig)}"
    fig.xaxis.axis_label = x_axis_label
    fig.xaxis.axis_label_standoff = 0

    return fig


def kde_viz_figure(
    hist: List[Optional[Tuple[np.ndarray, np.ndarray]]],
    kde: List[Any],
    col: str,
    plot_width: int,
    plot_height: int,
    cfg: Config,
    sources: Optional[List[str]] = None,
) -> Optional[figure]:
    """
    Render histograms with overlayed kde, one per DataFrame.

    DataFrames whose values are constant (no density estimate) are skipped.
    Returns None when nothing can be drawn.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    df_labels: List[str] = list(cfg.diff.label or [])
    sources = list(sources) if sources else df_labels[: len(hist)]
    colors = _source_colors(sources, df_labels)
    fig = figure(
        width=plot_width,
        height=plot_height,
        title=col,
        toolbar_location=None,
        y_axis_type=cfg.kde.yscale,
    )
    hist_rends, line_rends = [], []
    left, right, top = np.inf, -np.inf, 0.0
    for i, (data, kde2) in enumerate(zip(hist, kde)):
        if data is None or kde2 is None:
            continue
        dens, bins = data
        intvls = _format_bin_intervals(bins)
        df = pd.DataFrame(
            {
                "intvl": intvls,
                "left": bins[:-1],
                "right": bins[1:],
                "dens": dens,
                "orig": sources[i] if i < len(sources) else "",
            }
        )
        bottom = 0 if cfg.kde.yscale == "linear" or df.empty else df["dens"].min() / 2
        hist_rends.append(
            fig.quad(
                source=df,
                left="left",
                right="right",
                bottom=bottom,
                top="dens",
                alpha=0.5,
                fill_color=colors[i],
                line_color=colors[i],
            )
        )
        pts_rng = np.linspace(bins[0], bins[-1], 1000)
        pdf = kde2(pts_rng)
        line_rends.append(
            fig.line(x=pts_rng, y=pdf, line_color=colors[i], line_width=2, alpha=0.5)
        )
        left, right = min(left, bins[0]), max(right, bins[-1])
        top = max(top, float(np.max(dens)), float(pdf.max()))

    if not hist_rends:
        return None

    fig.add_tools(
        HoverTool(
            renderers=hist_rends,
            tooltips=[("Bin", "@intvl"), ("Density", "@dens"), ("Source", "@orig")],
            mode="vline",
        )
    )
    fig.add_tools(HoverTool(renderers=line_rends, tooltips=[("x", "@x"), ("y", "@y")]))
    tweak_figure(fig, "kde")
    fig.yaxis.axis_label = "Density"
    fig.xaxis.axis_label = col
    _format_axis(fig, left, right, "x")
    if cfg.kde.yscale == "linear":
        _format_axis(fig, 0, top, "y")
    return fig


def kde_viz_panel(
    hist: List[Optional[Tuple[np.ndarray, np.ndarray]]],
    kde: List[Any],
    col: str,
    plot_width: int,
    plot_height: int,
    cfg: Config,
    sources: Optional[List[str]] = None,
) -> Optional[TabPanel]:
    """
    Render histogram with overlayed kde
    """
    # pylint: disable=too-many-arguments, too-many-locals
    fig = kde_viz_figure(hist, kde, col, plot_width, plot_height, cfg, sources)
    if fig is None:
        return None
    return TabPanel(child=row(fig), title="KDE Plot")


def dt_line_viz(
    df: List[pd.DataFrame],
    x: str,
    timeunit: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    orig: List[str],
    df_labels: List[str],
) -> figure:
    """
    Render a line chart
    """
    # pylint: disable=too-many-arguments

    tooltips = [(timeunit, "@lbl"), ("Frequency", "@freq"), ("Source", "@orig")]
    colors = _source_colors(orig, df_labels)
    for i, _ in enumerate(df):
        df[i]["orig"] = orig[i]

    fig = figure(
        width=plot_width,
        height=plot_height,
        toolbar_location=None,
        title=x,
        tools=[],
        y_axis_type=yscale,
        x_axis_type="datetime",
    )

    for i, _ in enumerate(df):
        fig.line(
            source=df[i],
            x=x,
            y="freq",
            line_width=2,
            line_alpha=0.8,
            color=colors[i],
        )
    hover = HoverTool(
        tooltips=tooltips,
        mode="vline",
    )
    fig.add_tools(hover)

    tweak_figure(fig, "line", show_yticks)
    if show_yticks and yscale == "linear":
        _format_axis(fig, 0, max([d["freq"].max() for d in df]), "y")

    if orig != df_labels:
        fig.xaxis.axis_label = f"This variable is only in {','.join(orig)}"
        fig.xaxis.axis_label_standoff = 0
    fig.yaxis.axis_label = "Frequency"
    return fig


def box_viz(
    df_list: List[pd.DataFrame],
    x: str,
    plot_width: int,
    plot_height: int,
    cfg: Config,  # pylint: disable=unused-argument
    group_all: List[str],
    colors: Optional[List[str]] = None,
) -> TabPanel:
    """
    Render one box per DataFrame, side by side.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements

    width, title = 0.7, f"{x}"
    colors = colors or list(CATEGORY10)
    fig = figure(
        width=plot_width,
        height=plot_height,
        title=title,
        toolbar_location=None,
        x_range=group_all,
    )
    lows: List[float] = []
    highs: List[float] = []
    for i, df in enumerate(df_list):
        otlr_lists = [
            list(ols) if isinstance(ols, (list, tuple, np.ndarray)) else [] for ols in df["otlrs"]
        ]
        df = df.drop(columns=["otlrs"])
        df["x0"], df["x1"] = df.index + 0.2, df.index + 0.8
        low = fig.segment(x0="x0", y0="lw", x1="x1", y1="lw", line_color="black", source=df)
        ltail = fig.segment(x0="grp", y0="lw", x1="grp", y1="q1", line_color="black", source=df)
        lbox = fig.vbar(
            x="grp",
            width=width,
            top="q2",
            bottom="q1",
            fill_color=colors[i % len(colors)],
            line_color="black",
            source=df,
        )
        ubox = fig.vbar(
            x="grp",
            width=width,
            top="q3",
            bottom="q2",
            fill_color=colors[i % len(colors)],
            line_color="black",
            source=df,
        )
        utail = fig.segment(x0="grp", y0="uw", x1="grp", y1="q3", line_color="black", source=df)
        upw = fig.segment(x0="x0", y0="uw", x1="x1", y1="uw", line_color="black", source=df)

        otlrs = [float(otl) for otls in otlr_lists for otl in otls]
        if otlrs:
            gps = [grp for grp, ols in zip(df["grp"], otlr_lists) for _ in range(len(ols))]
            circ = fig.scatter(
                x=gps,
                y=otlrs,
                size=3,
                line_color="black",
                color="black",
                fill_alpha=0.6,
            )
            fig.add_tools(
                HoverTool(
                    renderers=[circ],
                    tooltips=[("Outlier", "@y")],
                )
            )

        tooltips = [
            ("Upper Whisker", "@uw"),
            ("Upper Quartile", "@q3"),
            ("Median", "@q2"),
            ("Lower Quartile", "@q1"),
            ("Lower Whisker", "@lw"),
        ]

        fig.add_tools(
            HoverTool(
                renderers=[upw, utail, ubox, lbox, ltail, low],
                tooltips=tooltips,
            )
        )
        lows.extend([df["lw"].min(), *otlrs])
        highs.extend([df["uw"].max(), *otlrs])

    tweak_figure(fig, "box")
    fig.xaxis.major_tick_line_color = None
    fig.xaxis.major_label_text_font_size = "0pt"
    fig.xaxis.axis_label = None
    fig.yaxis.axis_label = x

    if lows:
        _format_axis(fig, np.nanmin(lows), np.nanmax(highs), "y")

    return TabPanel(child=row(fig), title="Box Plot")


# pylint:disable = unused-argument
def render_correlation_single_heatmaps(
    df_list: List[Dict[str, pd.DataFrame]],
    col: str,
    plot_width: int,
    plot_height: int,
    cfg: Config,
    labels: Optional[List[str]] = None,
) -> List[TabPanel]:
    """
    Render correlation heatmaps, but with single column
    """
    # pylint:disable = too-many-locals
    corr: Dict[str, List[Any]] = {}
    labels = labels or [str(i + 1) for i in range(len(df_list))]
    group_all_x = [f"{col} ({label})" for label in labels]
    methods = [meth for meth in ("Pearson", "Spearman", "KendallTau") if meth in df_list[0]]
    # every column that is correlated with `col` in any of the DataFrames
    group_all_y = list(
        dict.fromkeys(val for df in df_list for meth in methods for val in df[meth]["y"])
    )
    for meth in methods:
        corr[meth] = []
        for i, df in enumerate(df_list):
            corr[meth].append(df[meth].assign(x=group_all_x[i]))
    if not group_all_y:
        return []
    tabs: List[TabPanel] = []
    tooltips = [("y", "@y"), ("correlation", "@correlation{1.11}")]
    for method, dfs in corr.items():
        mapper, color_bar = create_color_mapper(RDBU)

        x_range = FactorRange(*group_all_x)
        y_range = FactorRange(*group_all_y)
        fig = figure(
            x_range=x_range,
            y_range=y_range,
            width=plot_width,
            height=plot_height,
            x_axis_location="below",
            tools="hover",
            toolbar_location=None,
            tooltips=tooltips,
            title=" ",
        )

        tweak_figure(fig)
        for df in dfs:
            fig.rect(
                x="x",
                y="y",
                width=0.7,
                height=1,
                source=df,
                fill_color={"field": "correlation", "transform": mapper},
                line_color=None,
            )

        fig.add_layout(color_bar, "right")
        tab = TabPanel(child=fig, title=method)
        tabs.append(tab)
        for panel in tabs:
            panel.child.frame_width = int(plot_width * 0.9)
    return tabs


def format_ov_stats(stats: Dict[str, List[Any]]) -> Tuple[Dict[str, List[str]], List[Dict[str, int]]]:
    """
    Format the dataset level statistics, one value per DataFrame
    """
    nrows, ncols = stats["nrows"], stats["ncols"]
    npresent_cells, nrows_wo_dups = stats["npresent_cells"], stats["nrows_wo_dups"]
    mem_use, dtypes_cnt = stats["mem_use"], stats["dtype_cnts"]
    ncells = [nrow * ncol for nrow, ncol in zip(nrows, ncols)]

    data = {
        "Number of Variables": ncols,
        "Number of Rows": nrows,
        "Missing Cells": [float(ncell - npres) for ncell, npres in zip(ncells, npresent_cells)],
        "Missing Cells (%)": [
            1 - _ratio(npres, ncell) for ncell, npres in zip(ncells, npresent_cells)
        ],
        "Duplicate Rows": [nrow - nuniq for nrow, nuniq in zip(nrows, nrows_wo_dups)],
        "Duplicate Rows (%)": [
            1 - _ratio(nuniq, nrow) for nrow, nuniq in zip(nrows, nrows_wo_dups)
        ],
        "Total Size in Memory": [float(mem) for mem in mem_use],
        "Average Row Size in Memory": [_ratio(mem, nrow) for mem, nrow in zip(mem_use, nrows)],
    }
    return {k: _format_values(k, v) for k, v in data.items()}, dtypes_cnt


def format_num_stats(data: Dict[str, List[Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Format numerical statistics, one value per DataFrame
    """
    nrows, npres, nreals = data["nrows"], data["npres"], data["nreals"]
    overview = {
        "Distinct Count": list(data["nuniq"]),
        "Unique (%)": [_ratio(nuniq, npre) for nuniq, npre in zip(data["nuniq"], npres)],
        "Missing": [nrow - npre for nrow, npre in zip(nrows, npres)],
        "Missing (%)": [1 - _ratio(npre, nrow) for npre, nrow in zip(npres, nrows)],
        "Infinite": [npre - nreal for npre, nreal in zip(npres, nreals)],
        "Infinite (%)": [
            _ratio(npre - nreal, nrow) for npre, nreal, nrow in zip(npres, nreals, nrows)
        ],
        "Memory Size": data["mem_use"],
        "Mean": data["mean"],
        "Minimum": data["min"],
        "Maximum": data["max"],
        "Zeros": data["nzero"],
        "Zeros (%)": [_ratio(nzero, nrow) for nzero, nrow in zip(data["nzero"], nrows)],
        "Negatives": data["nneg"],
        "Negatives (%)": [_ratio(nneg, nrow) for nneg, nrow in zip(data["nneg"], nrows)],
    }
    qntls_list = [
        dict(zip(np.round(np.asarray(qntls.index, dtype=float), 2), qntls.to_numpy()))
        for qntls in data["qntls"]
    ]
    quantile = {
        "Minimum": data["min"],
        "5-th Percentile": [qntls[0.05] for qntls in qntls_list],
        "Q1": [qntls[0.25] for qntls in qntls_list],
        "Median": [qntls[0.5] for qntls in qntls_list],
        "Q3": [qntls[0.75] for qntls in qntls_list],
        "95-th Percentile": [qntls[0.95] for qntls in qntls_list],
        "Maximum": data["max"],
        "Range": [max_value - min_value for max_value, min_value in zip(data["max"], data["min"])],
        "IQR": [qntls[0.75] - qntls[0.25] for qntls in qntls_list],
    }
    descriptive = {
        "Mean": data["mean"],
        "Standard Deviation": data["std"],
        "Variance": [std**2 for std in data["std"]],
        "Sum": [mean * npre for mean, npre in zip(data["mean"], npres)],
        "Skewness": [float(skew) for skew in data["skew"]],
        "Kurtosis": [float(kurt) for kurt in data["kurt"]],
        "Coefficient of Variation": [
            std / mean if mean != 0 else np.nan for mean, std in zip(data["mean"], data["std"])
        ],
    }
    return {
        "Overview": {k: _format_values(k, v) for k, v in overview.items()},
        "Quantile Statistics": {k: _format_values(k, v) for k, v in quantile.items()},
        "Descriptive Statistics": {k: _format_values(k, v) for k, v in descriptive.items()},
    }


def render_comparison_grid(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df)
    """
    # pylint: disable=too-many-locals, line-too-long
    plot_width = cfg.plot.width if cfg.plot.width is not None else 324
    plot_height = cfg.plot.height if cfg.plot.height is not None else 300
    df_labels: List[str] = cfg.diff.label  # type: ignore
    baseline: int = cfg.diff.baseline

    figs: List[figure] = []
    titles: List[str] = []

    all_nrows = itmdt["stats"]["nrows"]
    for col, dtp, data, orig in itmdt["data"]:
        fig = None
        # row counts of the DataFrames that contain this column
        nrows = [all_nrows[df_labels.index(lbl)] for lbl in orig]
        if is_dtype(dtp, Nominal()):
            df, ttl_grps = data
            fig = bar_viz(
                list(df),
                ttl_grps,
                nrows,
                col,
                cfg.bar.yscale,
                plot_width,
                plot_height,
                False,
                orig,
                df_labels,
                baseline_position(orig, df_labels, baseline),
            )
        elif is_dtype(dtp, Continuous()):
            if cfg.diff.density:
                kde, dens = data
                fig = kde_viz_figure(dens, kde, col, plot_width, plot_height, cfg, orig)
            else:
                fig = hist_viz(
                    data,
                    nrows,
                    col,
                    cfg.hist.yscale,
                    plot_width,
                    plot_height,
                    False,
                    df_labels,
                    orig,
                )
        elif is_dtype(dtp, DateTime()):
            df, timeunit = data
            fig = dt_line_viz(
                list(df),
                col,
                timeunit,
                cfg.line.yscale,
                plot_width,
                plot_height,
                False,
                orig,
                df_labels,
            )
        if fig is not None:
            fig.frame_height = plot_height
            titles.append(fig.title.text)
            figs.append(fig)

    if cfg.stats.enable:
        toggle_content = "Stats"
    else:
        toggle_content = None  # type: ignore
    return {
        "layout": figs,
        "meta": titles,
        "comparison_stats": format_ov_stats(itmdt["stats"]) if cfg.stats.enable else None,
        "container_width": plot_width * 3,
        "toggle_content": toggle_content,
        "df_labels": cfg.diff.label,
        "legend_labels": [
            {"label": label, "color": color}
            for label, color in zip(cfg.diff.label, CATEGORY10[: len(cfg.diff.label)])  # type: ignore
        ],
        "baseline": baseline,
    }


def render_comparison_continous(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render for continuous variable comparison
    """
    # pylint:disable = too-many-locals
    plot_width = cfg.plot.width if cfg.plot.width is not None else 450
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400
    all_labels: List[str] = list(cfg.diff.label)  # type: ignore[arg-type]
    # the DataFrames that contain the column, in order
    df_labels: List[str] = list(itmdt.get("labels") or all_labels)
    colors = _source_colors(df_labels, all_labels)
    tabs: List[TabPanel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}
    col, data = itmdt["col"], itmdt["data"][0]
    if cfg.hist.enable:
        nrows = itmdt["stats"]["nrows"]
        fig = hist_viz(
            data["hist"],
            nrows,
            col,
            cfg.hist.yscale,
            plot_width,
            plot_height,
            False,
            all_labels,
            df_labels,
        )
        fig.xaxis.axis_label = ""
        tabs.append(TabPanel(child=row(fig), title="Histogram"))
    if cfg.kde.enable:
        panel = kde_viz_panel(
            data["dens"], data["kde"], col, plot_width, plot_height, cfg, df_labels
        )
        if panel is not None:
            tabs.append(panel)
    if cfg.box.enable:
        df_list = []
        group_all = []
        for i, data_box in enumerate(data["box"]):
            box_data = {
                "grp": f"{col} ({df_labels[i]})",
                "q1": data_box["qrtl1"],
                "q2": data_box["qrtl2"],
                "q3": data_box["qrtl3"],
                "lw": data_box["lw"],
                "uw": data_box["uw"],
                "otlrs": [data_box["otlrs"]],
            }
            df_list.append(pd.DataFrame(box_data, index=[i]))
            group_all.append(box_data["grp"])
        tabs.append(box_viz(df_list, col, plot_width, plot_height, cfg, group_all, colors))
    for panel in tabs:
        panel.child.children[0].frame_width = int(plot_width * 0.9)
    if cfg.correlations.enable:
        tabs = tabs + (
            render_correlation_single_heatmaps(
                data["corr"], col, plot_width, plot_height, cfg, df_labels
            )
        )

    legend_lables = [{"label": label, "color": color} for label, color in zip(df_labels, colors)]
    return {
        "comparison_stats": format_num_stats(itmdt["stats"]) if cfg.stats.enable else [],
        "value_table": [],
        "insights": [],
        "layout": [panel.child for panel in tabs],
        "meta": ["Stats"] + [tab.title for tab in tabs],
        "container_width": plot_width + 110,
        "how_to_guide": htgs,
        "df_labels": df_labels,
        "legend_labels": legend_lables,
    }


def render_diff(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render a basic plot

    Parameters
    ----------
    itmdt
        The Intermediate containing results from the compute function.
    cfg
        Config instance
    """

    if itmdt.visual_type == "comparison_grid":
        return render_comparison_grid(itmdt, cfg)
    if itmdt.visual_type == "comparison_continuous":
        return render_comparison_continous(itmdt, cfg)
    raise NotImplementedError(f"Unknown visual type {itmdt.visual_type!r}")
