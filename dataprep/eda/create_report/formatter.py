"""This module implements the formatting
for create_report(df) function."""  # pylint: disable=line-too-long,

import sys
from typing import Any, Dict, List, Optional, Tuple
from warnings import catch_warnings, simplefilter

import pandas as pd
from bokeh.plotting import figure

from ..configs import Config
from ..correlation import render_correlation
from ..correlation.compute.overview import correlation_nxn
from ..distribution import compute, render
from ..distribution.compute.overview import (
    _cont_calcs,
    _format_cont_ins,
    _format_nom_ins,
    _format_ov_ins,
    _insight_pagination,
    _nom_calcs,
    calc_stats,
)
from ..distribution.compute.univariate import calc_stats_dt, cont_comps, nom_comps
from ..distribution.render import format_cat_stats, format_num_stats, format_ov_stats, stats_viz_dt
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
from ..missing import render_missing
from ..missing.compute.nullivariate import compute_missing_nullivariate
from ..parallel import Progress, pmap
from ..utils import _calc_line_dt, sample_frame

# pylint: disable=E1133


def format_report(
    df: DataFrame,
    cfg: Config,
    mode: Optional[str],
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Format the data and figures needed by report

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
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
        A dictionary in which formatted data will be stored.
        This variable acts like an API in passing data to the template engine.
    """
    if mode != "basic":
        raise ValueError(f"Unknown mode: {mode!r}. Only 'basic' is supported.")
    with Progress(enabled=progress, desc="Report"):
        edaframe = EDAFrame(df)
        if edaframe.shape[0] == 0 or edaframe.shape[1] == 0:
            return {}
        comps = format_basic(edaframe, cfg)
    return comps


def _format_variables(df: EDAFrame, cfg: Config, data: Dict[str, Any]) -> Dict[str, Any]:
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
            if isinstance(dtp, Continuous):
                itmdt = Intermediate(col=col, data=data[col], visual_type="numerical_column")
                stats = format_num_stats(data[col])
            elif isinstance(dtp, (Nominal, SmallCardNum, GeoGraphy, GeoPoint)):
                itmdt = Intermediate(col=col, data=data[col], visual_type="categorical_column")
                stats = format_cat_stats(
                    data[col]["stats"], data[col]["len_stats"], data[col]["letter_stats"]
                )
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
            insight_keys = list(rndrd["insights"].keys())[2:] if rndrd["insights"] else []
            res["variables"][col] = {
                "tabledata": stats,
                "plots": comp,
                "col_type": itmdt.visual_type.replace("_column", ""),
                "tab_name": rndrd["meta"],
                "plots_tab": zip(comp[1][1:], rndrd["meta"][1:], insight_keys),
                "insights_tab": rndrd["insights"],
            }

        except:
            print(f"error happended in column:{col}", file=sys.stderr)
            raise

    return res


def _format_interaction(data: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Format of Interaction section"""
    res: Dict[str, Any] = {}
    # interactions
    if cfg.interactions.enable:
        res["has_interaction"] = True
        itmdt = Intermediate(
            scatter_source=data["interaction.scatter_source"],
            other_plots=data["interaction.other_plots"],
            num_cols=data["num_cols"],
            all_cols=data["all_cols"],
            visual_type="correlation_crossfilter",
        )
        rndrd = render_correlation(itmdt, cfg)
        rndrd.sizing_mode = "stretch_width"
        res["interactions"] = components(rndrd)
    else:
        res["has_interaction"] = False
    return res


def _format_correlation(data: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Format of Correlation section"""
    res: Dict[str, Any] = {}
    if len(data["num_cols"]) > 0:
        # correlations
        if cfg.correlations.enable:
            res["has_correlation"] = True
            dfs: Dict[str, pd.DataFrame] = {}
            for method, corr in data["corrs"].items():
                ndf = pd.DataFrame(
                    {
                        "x": data["num_cols"][data["cordx"]],
                        "y": data["num_cols"][data["cordy"]],
                        "correlation": corr.ravel(),
                    }
                )
                dfs[method.name] = ndf[data["cordy"] > data["cordx"]]
            itmdt = Intermediate(
                data=dfs,
                axis_range=list(data["num_cols"]),
                visual_type="correlation_heatmaps",
            )
            rndrd = render_correlation(itmdt, cfg)
            res["correlation_names"] = []
            figs_corr: List[figure] = []
            # pylint: disable = not-an-iterable
            for tab in rndrd.tabs:
                fig = tab.child
                fig.sizing_mode = "stretch_width"
                figs_corr.append(fig)
                res["correlation_names"].append(tab.title)
            res["correlations"] = components(figs_corr)

    else:
        res["has_correlation"] = False

    return res


def _format_missing(
    data: Dict[str, Any], cfg: Config, completions: Optional[Dict[str, Any]], ncols: int
) -> Dict[str, Any]:
    """Format of Missing section"""
    res: Dict[str, Any] = {}
    # missing
    if cfg.missingvalues.enable and completions is not None:
        res["has_missing"] = True
        itmdt = completions["miss"](data["miss"])

        rndrd = render_missing(itmdt, cfg)
        figs_missing: List[figure] = []
        for fig in rndrd["layout"]:
            fig.sizing_mode = "stretch_both"
            figs_missing.append(fig)
        res["missing"] = components(figs_missing)
        res["missing_tabs"] = ["Bar Chart", "Spectrum", "Heat Map"]
        # only display dendrogram when df has more than one column
        if ncols > 1:
            res["missing_tabs"].append("Dendrogram")
    return res


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


def format_basic(df: EDAFrame, cfg: Config) -> Dict[str, Any]:
    """
    Format basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    cfg
        The config dict user passed in. E.g. config =  {"hist.bins": 20}
        Without user's specifications, the default is "auto"
    Returns
    -------
    Dict[str, Any]
        A dictionary in which formatted data is stored.
        This variable acts like an API in passing data to the template engine.
    """
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    # aggregate all computations

    cfg.plot.report = True
    with catch_warnings():
        simplefilter("ignore", RuntimeWarning)
        data, completions = basic_computations(df, cfg)

    res_overview = _format_overview(data, cfg)
    res_variables = _format_variables(df, cfg, data)
    res_interaction = _format_interaction(data, cfg)
    res_correlations = _format_correlation(data, cfg)
    res_missing = _format_missing(data, cfg, completions, df.shape[1])
    res = {**res_overview, **res_variables, **res_interaction, **res_correlations, **res_missing}

    return res


def _compute_variables(df: EDAFrame, cfg: Config) -> Dict[str, Any]:
    """Computation of Variables section."""
    if not cfg.variables.enable:
        return {}

    def one(col: str) -> Any:
        try:
            dtype = df.get_eda_dtype(col)
            # An all-missing column cannot be summarised as numbers or dates,
            # so it is shown as a categorical column whose only value is "nan".
            if df.get_missing_cnt(col) == df.shape[0]:
                return nom_comps(df.get_col_as_str(col, na_as_str=True), cfg)
            if isinstance(dtype, (Nominal, GeoGraphy, GeoPoint, SmallCardNum)):
                return nom_comps(df.get_col_as_str(col, na_as_str=False), cfg)
            if isinstance(dtype, Continuous):
                return cont_comps(df.frame[col], cfg)
            if isinstance(dtype, DateTime):
                return {
                    "stats": calc_stats_dt(df.frame[col]),
                    "line": _calc_line_dt(df.frame[[col]], "auto"),
                }
            raise ValueError(f"unprocessed type in column{col}:{dtype}")
        except Exception:
            print(f"error happened in column: {col}", file=sys.stderr)
            raise

    cols = list(df.columns)
    return dict(zip(cols, pmap(one, cols, labels=cols)))


def _is_all_missing(df: EDAFrame, col: str) -> bool:
    return df.get_missing_cnt(col) == df.shape[0]


def _compute_overview(df: EDAFrame, cfg: Config) -> Dict[str, Any]:
    """Computation of Overview section."""
    data: Dict[str, Any] = {}
    if not cfg.overview.enable:
        return data

    def one(col: str) -> Any:
        col_dtype = df.get_eda_dtype(col)
        if _is_all_missing(df, col):
            srs = df.get_col_as_str(col, na_as_str=False).dropna().astype(object)
            return col, Nominal(), _nom_calcs(srs, cfg)
        if isinstance(col_dtype, Continuous):
            return col, Continuous(), _cont_calcs(df.frame[col].dropna(), cfg)
        if isinstance(col_dtype, (Nominal, GeoGraphy, GeoPoint, SmallCardNum)):
            srs = df.get_col_as_str(col, na_as_str=False).dropna()
            return col, Nominal(), _nom_calcs(srs, cfg)
        if isinstance(col_dtype, DateTime):
            return col, DateTime(), None
        raise RuntimeError(f"unprocessed data type: col:{col}, dtype: {type(col_dtype)}")

    data["ov"] = calc_stats(df, cfg)
    cols = list(df.columns)
    data["insights"] = pmap(one, cols, labels=cols)
    return data


def basic_computations(
    df: EDAFrame, cfg: Config
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Computations for the basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    df_num
        The DataFrame of numerical column (used for correlation). It is seperated from df since
        the small distinct value numerical column in df is regarded as categorical column, and
        will transform to str then used for other plots. But they should be regarded as numerical
        column in df_num and used in correlation. This is a temporary fix, in the future we should treat
        those small distinct value numerical columns as ordinary in both correlation plots and other plots.
    cfg
        The config dict user passed in. E.g. config =  {"hist.bins": 20}
        Without user's specifications, the default is "auto"
    """
    # pylint: disable=too-many-branches, protected-access, too-many-locals

    variables_data = _compute_variables(df, cfg)
    overview_data = _compute_overview(df, cfg)
    data: Dict[str, Any] = {**variables_data, **overview_data}

    df_num = df.select_num_columns()
    num_columns = df_num.columns
    cat_columns = [col for col in df.columns if col not in num_columns]
    data["num_cols"] = df_num.columns
    data["all_cols"] = df.columns
    # interactions
    if cfg.interactions.enable:
        data["interaction.scatter_source"] = sample_frame(
            df_num.frame, cfg.scatter.sample_size, cfg.scatter.sample_rate
        )

        data["interaction.other_plots"] = {}
        if cfg.interactions.cat_enable:
            other_plots = {}
            curr_cfg = Config.from_dict(display=["Box Plot", "Nested Bar Chart"])
            for cat_col in cat_columns:
                for other_col in df.columns:
                    if (cat_col == other_col) or (cat_col + "_" + other_col in other_plots):
                        continue
                    # print(f"cat col:{cat_col}, other col:{other_col}")
                    imdt = compute(df.frame, cat_col, other_col, cfg=curr_cfg)
                    box_plot = render(imdt, curr_cfg)["layout"][0]
                    other_plots[cat_col + "_" + other_col] = box_plot
                    other_plots[other_col + "_" + cat_col] = box_plot
            data["interaction.other_plots"] = other_plots

    # correlations
    if cfg.correlations.enable:
        data.update(zip(("cordx", "cordy", "corrs"), correlation_nxn(df_num, cfg)))

    # missing values
    completions = None
    if cfg.missingvalues.enable:
        gathered, completion = compute_missing_nullivariate(  # type: ignore[misc]
            df, cfg, _staged=True
        )
        data["miss"] = gathered
        completions = {"miss": completion}

    return data, completions
