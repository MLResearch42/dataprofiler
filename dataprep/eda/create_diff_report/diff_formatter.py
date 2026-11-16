"""This module implements the formatting
for create_diff_report([df1, df2, ...]) function."""  # pylint: disable=line-too-long,

import sys
from typing import Any, Dict, List, Optional, Union
from warnings import catch_warnings, simplefilter

from bokeh.plotting import figure

from ...errors import DataprepError
from ..configs import Config
from ..create_report.formatter import _compute_overview, _compute_variables
from ..diff import render_diff
from ..diff.compute.multiple_df import compare_multiple_df
from ..distribution import render
from ..distribution.compute.overview import (
    _format_cont_ins,
    _format_nom_ins,
    _format_ov_ins,
    _insight_pagination,
)
from ..distribution.render import (
    format_cat_stats,
    format_num_stats,
    format_ov_stats,
    stats_viz_dt,
)
from ..dtypes import (
    Continuous,
    DateTime,
    GeoGraphy,
    GeoPoint,
    Nominal,
    SmallCardNum,
)
from ..eda_frame import DataFrame, EDAFrame
from ..embed import components
from ..intermediate import Intermediate
from ..palette import CATEGORY10
from ..parallel import Progress


def format_diff_report(
    df_list: Union[List[DataFrame], Dict[str, DataFrame]],
    cfg: Config,
    mode: Optional[str],
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Format the data and figures needed by create_diff_report

    Parameters
    ----------
    df_list
        The DataFrames for which data are calculated, as a list or a
        ``{label: DataFrame}`` dict.
    cfg
        The config instance
    mode
        This controls what type of report to be generated.
        Currently only the 'basic' is fully implemented.
    progress
        Whether to show the progress bar.

    Returns
    -------
    Dict[str, Any]
        A dictionary of results
    """
    if isinstance(df_list, dict):
        if not cfg.diff.label:
            cfg.diff.label = [str(key) for key in df_list.keys()]
        df_list = list(df_list.values())
    elif isinstance(df_list, (list, tuple)):
        df_list = list(df_list)
        if not cfg.diff.label:
            cfg.diff.label = [f"df{i+1}" for i in range(len(df_list))]
    else:
        raise TypeError(
            "create_diff_report expects a list or dict of DataFrames, "
            f"got {type(df_list).__name__}."
        )

    if len(df_list) < 2:
        raise DataprepError("create_diff_report needs at least 2 DataFrames.")
    if len(df_list) > 5:
        raise DataprepError("Too many DataFrames, max: 5.")
    if len(cfg.diff.label) != len(df_list):
        raise ValueError("Number of the given label doesn't match the number of DataFrames.")
    if cfg.diff.baseline > len(df_list) - 1:
        raise ValueError("Baseline is out of the boundary of the input.")
    if mode != "basic":
        raise ValueError(f"Unknown mode: {mode!r}. Only 'basic' is supported.")

    with Progress(enabled=progress, desc="Diff report"):
        report = format_basic(df_list, cfg)
    return report


def format_basic(df_list: List[DataFrame], cfg: Config) -> Dict[str, Any]:
    """
    Format basic version.

    Parameters
    ----------
    df_list
        The DataFrames for which data are calculated.
    cfg
        The config dict user passed in. E.g. config =  {"hist.bins": 20}
        Without user's specifications, the default is "auto"

    Returns
    -------
    Dict[str, Any]
        A dictionary in which formatted data is stored.
        This variable acts like an API in passing data to the template engine.
    """
    # pylint: disable=too-many-locals
    final_results: Dict[str, Any] = {"dfs": []}
    figs_var: List[figure] = []
    cfg.plot.report = True

    frames = [EDAFrame(df) for df in df_list]
    with catch_warnings():
        simplefilter("ignore", RuntimeWarning)
        for frame in frames:
            data = basic_computations(frame, cfg)
            res_overview = _format_overview(data, cfg)
            res_variables = _format_variables(frame, cfg, data)
            final_results["dfs"].append({**res_overview, **res_variables})
        res_plots = _format_plots(frames, cfg)

    for tab in res_plots["layout"]:
        try:
            fig = tab.children[0]
        except AttributeError:
            fig = tab
        figs_var.append(fig)
    final_results["graphs"] = components(figs_var)
    final_results["legend_lables"] = [
        {"label": label, "color": color}
        for label, color in zip(cfg.diff.label, CATEGORY10[: len(cfg.diff.label)])  # type: ignore
    ]
    return final_results


def basic_computations(df: EDAFrame, cfg: Config) -> Dict[str, Any]:
    """Computations for the basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    cfg
        The config dict user passed in. E.g. config =  {"hist.bins": 20}
        Without user's specifications, the default is "auto"
    """
    variables_data = _compute_variables(df, cfg)
    overview_data = _compute_overview(df, cfg)
    data: Dict[str, Any] = {**variables_data, **overview_data}
    return data


def compute_plot_data(df_list: List[Any], cfg: Config, dtype: Any = None) -> Intermediate:
    """
    Compute function for create_diff_report's plots: overlaid histograms for
    numerical columns and grouped bar charts for categorical ones.
    """
    density = cfg.diff.density
    cfg.diff.density = False  # the report always shows plain histograms
    try:
        return compare_multiple_df(df_list, cfg, dtype)  # type: ignore[arg-type]
    finally:
        cfg.diff.density = density


def _format_variables(df: EDAFrame, cfg: Config, data: Dict[str, Any]) -> Dict[str, Any]:
    """Formatting of variables section"""
    res: Dict[str, Any] = {}
    # variables
    if not cfg.variables.enable:
        res["has_variables"] = False
        return res

    res["variables"] = {}
    res["has_variables"] = True
    for col in df.columns:
        try:
            stats: Any = None  # needed for pylint
            dtp = df.get_eda_dtype(col)
            tab_names: List[str] = []
            if isinstance(dtp, Continuous):
                itmdt = Intermediate(col=col, data=data[col], visual_type="numerical_column")
                stats = format_num_stats(data[col])
                tab_names = ["Stats", "Histogram", "KDE Plot", "Normal Q-Q Plot"]
            elif isinstance(dtp, (Nominal, SmallCardNum, GeoGraphy, GeoPoint)):
                itmdt = Intermediate(col=col, data=data[col], visual_type="categorical_column")
                stats = format_cat_stats(
                    data[col]["stats"], data[col]["len_stats"], data[col]["letter_stats"]
                )
                tab_names = ["Stats", "Word Length", "Pie Chart", "Word Cloud", "Word Frequency"]
            elif isinstance(dtp, DateTime):
                itmdt = Intermediate(
                    col=col,
                    data=data[col]["stats"],
                    line=data[col]["line"],
                    visual_type="datetime_column",
                )
                stats = stats_viz_dt(data[col]["stats"])
            else:
                raise RuntimeError(f"the type of column {col} is unknown: {type(dtp)}")

            rndrd = render(itmdt, cfg)
            layout = rndrd["layout"]
            figs_var: List[figure] = []
            for tab in layout:
                try:
                    fig = tab.children[0]
                except AttributeError:
                    fig = tab
                # fig.title = Title(text=tab.title, align="center")
                figs_var.append(fig)
            comp = components(figs_var)

            res["variables"][col] = {
                "tabledata": stats,
                "col_type": itmdt.visual_type.replace("_column", ""),
                "tab_names": tab_names,
                "plots": comp,
            }

        except:
            print(f"error happended in column:{col}", file=sys.stderr)
            raise

    return res


def _format_plots(df_list: List[Any], cfg: Config) -> Dict[str, Any]:
    """Formatting of plots section"""
    itmdt = compute_plot_data(df_list=df_list, cfg=cfg, dtype=None)
    return render_diff(itmdt, cfg=cfg)


def _format_overview(data: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Format of Overview section"""
    # results dictionary
    res: Dict[str, Any] = {}
    # overview
    if cfg.overview.enable:
        # insight
        all_ins = _format_ov_ins(data["ov"], cfg)
        for col, dtp, dat in data["insights"]:
            if isinstance(dtp, Continuous):
                ins = _format_cont_ins(col, dat, data["ov"]["nrows"], cfg)[1]
            elif isinstance(dtp, (Nominal, SmallCardNum, GeoGraphy, GeoPoint)):
                ins = _format_nom_ins(col, dat, data["ov"]["nrows"], cfg)[1]
            else:
                continue
            all_ins += ins
        res["overview_insights"] = _insight_pagination(all_ins)
        res["overview"] = format_ov_stats(data["ov"])
        res["has_overview"] = True
    else:
        res["has_overview"] = False
    return res
