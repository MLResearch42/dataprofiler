"""Computations for plot(df, x, y)."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ....errors import UnreachableError
from ...configs import Config
from ...dtypes import (
    Continuous,
    DateTime,
    DType,
    DTypeDef,
    GeoGraphy,
    GeoPoint,
    LatLong,
    Nominal,
    SmallCardNum,
)
from ...eda_frame import DataFrame, EDAFrame
from ...intermediate import Intermediate
from ...utils import (
    DTMAP,
    _calc_box_otlrs,
    _calc_box_stats,
    _calc_groups,
    _calc_line_dt,
    _calc_running_total_dt,
    _get_timeunit,
    _period_labels,
    finite,
    sample_frame,
    value_counts,
)
from .common import gen_new_df_with_used_cols

BOX_COLUMNS = ["grp", "q1", "q2", "q3", "lw", "uw", "otlrs"]


def _check_type_combination(
    instance_tuple: Tuple[DType, DType],
    type_tuple: Tuple[Any, Any],
) -> bool:
    """Check whether two instance types is instance of types in type_tuples"""

    xtype, ytype = instance_tuple[0], instance_tuple[1]
    type1, type2 = type_tuple[0], type_tuple[1]
    return (isinstance(xtype, type1) and isinstance(ytype, type2)) or (
        isinstance(xtype, type2) and isinstance(ytype, type1)
    )


def _pair(frame: EDAFrame, x: str, y: str, as_str: Tuple[str, ...] = ()) -> pd.DataFrame:
    """The two columns with incomplete rows dropped; ``as_str`` columns become strings."""
    cols = {
        col: frame.get_col_as_str(col) if col in as_str else frame.frame[col] for col in (x, y)
    }
    return pd.DataFrame(cols, copy=False).dropna()


def compute_bivariate(
    df: DataFrame,
    col1: Union[str, LatLong],
    col2: Union[str, LatLong],
    cfg: Config,
    dtype: Optional[DTypeDef],
) -> Intermediate:
    """Compute functions for plot(df, x, y).

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    col1
        A column name from the DataFrame
    col2
        A column name from the DataFrame
    cfg
        Config instance
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-branches,too-many-statements,too-many-boolean-expressions
    # pylint: disable = too-many-return-statements
    # pylint: disable = too-many-locals

    if col1 is None or col2 is None:
        raise ValueError(f"Input column name should not be None. Input columns: {col1}, y={col2}.")

    new_col_names, ndf = gen_new_df_with_used_cols(df, col1, col2, z=None)
    x = new_col_names[col1]
    y = new_col_names[col2]
    if x is None or y is None:
        raise ValueError("Two column names are required.")
    if x == y:
        raise ValueError(f"The two columns must be different, got {x!r} twice.")

    frame = EDAFrame(ndf, dtype)
    if frame.shape[0] == 0:
        raise ValueError("The DataFrame has no rows, there is nothing to plot.")

    xtype = frame.get_eda_dtype(x)
    ytype = frame.get_eda_dtype(y)

    if _check_type_combination((xtype, ytype), ((Nominal, SmallCardNum), Continuous)):
        x, y = (x, y) if isinstance(xtype, (Nominal, SmallCardNum)) else (y, x)
        # NA is dropped, so transforming to str will not introduce NA in the viz.
        tmp_df = _pair(frame, x, y, as_str=(x,))
        return Intermediate(
            x=x,
            y=y,
            data=_nom_cont_comps(tmp_df, cfg),
            visual_type="cat_and_num_cols",
        )

    if _check_type_combination((xtype, ytype), (DateTime, Continuous)):
        x, y = (x, y) if isinstance(xtype, DateTime) else (y, x)
        tmp_df = _pair(frame, x, y)
        dtnum: Dict[str, Any] = {}
        if len(tmp_df) > 0:
            # line chart
            if cfg.line.enable:
                dtnum["linedata_agg"] = _calc_line_dt(tmp_df, cfg.line.unit, agg=cfg.line.agg)
                dtnum["linedata_running_total"] = _calc_running_total_dt(tmp_df, cfg.line.unit)
            # box plot
            if cfg.box.enable:
                dtnum["boxdata"] = _calc_box_dt(tmp_df, cfg.box.unit)
        return Intermediate(
            x=x,
            y=y,
            linedata_agg=dtnum.get("linedata_agg", []),
            linedata_running_total=dtnum.get("linedata_running_total", []),
            boxdata=dtnum.get("boxdata", []),
            visual_type="dt_and_num_cols",
        )

    if _check_type_combination(
        (xtype, ytype), (DateTime, (Nominal, SmallCardNum, GeoGraphy, GeoPoint))
    ):
        x, y = (x, y) if isinstance(xtype, DateTime) else (y, x)
        tmp_df = _pair(frame, x, y, as_str=(y,))
        linedata: Any = []
        stackdata: Any = []
        if len(tmp_df) > 0:
            if cfg.line.enable:
                # line chart
                linedata = _calc_line_dt(
                    tmp_df,
                    cfg.line.unit,
                    ngroups=cfg.line.ngroups,
                    largest=cfg.line.sort_descending,
                )
            if cfg.stacked.enable:
                # stacked bar chart
                stackdata = _calc_stacked_dt(
                    tmp_df,
                    cfg.stacked.unit,
                    cfg.stacked.ngroups,
                    cfg.stacked.sort_descending,
                )
        return Intermediate(
            x=x,
            y=y,
            linedata=linedata,
            stackdata=stackdata,
            visual_type="dt_and_cat_cols",
        )

    if _check_type_combination((xtype, ytype), (GeoGraphy, Continuous)):
        x, y = (x, y) if isinstance(xtype, GeoGraphy) else (y, x)
        tmp_df = _pair(frame, x, y, as_str=(x,))
        return Intermediate(
            x=x, y=y, data=geo_cont_comps(tmp_df, cfg), visual_type="geo_and_num_cols"
        )

    if _check_type_combination((xtype, ytype), (GeoPoint, Continuous)):
        x, y = (x, y) if isinstance(xtype, GeoPoint) else (y, x)
        tmp_df = _pair(frame, x, y)
        return Intermediate(
            x=x, y=y, data=geop_cont_comps(tmp_df), visual_type="latlong_and_num_cols"
        )

    cat_types = (Nominal, SmallCardNum, GeoGraphy, GeoPoint)
    if isinstance(xtype, cat_types) and isinstance(ytype, cat_types):
        tmp_df = _pair(frame, x, y, as_str=(x, y))
        comps = tmp_df.groupby([x, y], sort=True).size()
        return Intermediate(
            x=x,
            y=y,
            data=comps,
            visual_type="two_cat_cols",
        )

    if isinstance(xtype, Continuous) and isinstance(ytype, Continuous):
        tmp_df = _pair(frame, x, y)
        # infinite values cannot be binned or drawn
        both = np.isfinite(tmp_df[x].to_numpy(dtype=float)) & np.isfinite(
            tmp_df[y].to_numpy(dtype=float)
        )
        if not both.all():
            tmp_df = tmp_df[both]

        data: Dict[str, Any] = {}
        if cfg.scatter.enable:
            # scatter plot data
            data["scat"] = sample_frame(tmp_df, cfg.scatter.sample_size, cfg.scatter.sample_rate)
        if cfg.hexbin.enable:
            # hexbin plot data
            data["hex"] = tmp_df
        if cfg.box.enable:
            # box plot
            data["box"] = _calc_box_cont(tmp_df, cfg)

        return Intermediate(
            x=x,
            y=y,
            data=data,
            visual_type="two_num_cols",
        )

    raise UnreachableError(
        f"Unprocessed type. x_name:{x}, x_type:{xtype}, y_name:{y}, y_type:{ytype}"
    )


def _top_groups(cnts: pd.Series, ngroups: int, descending: bool) -> List[Any]:
    """The ``ngroups`` most (or least) frequent group labels, most extreme first."""
    top = cnts.nlargest(ngroups) if descending else cnts.nsmallest(ngroups)
    return list(top.index)


def _grouped_box(df: pd.DataFrame, groups: List[Any]) -> pd.DataFrame:
    """Box plot statistics of column y for each of the given groups of column x."""
    x, y = df.columns
    grouped = dict(tuple(df[df[x].isin(groups)].groupby(x, sort=False, observed=True)[y]))
    rows = [{"grp": grp, **_box_comps(grouped[grp])} for grp in groups if grp in grouped]
    return pd.DataFrame(rows, columns=BOX_COLUMNS)


def _grouped_hist(df: pd.DataFrame, groups: List[Any], bins: int) -> pd.DataFrame:
    """A histogram of column y for each of the given groups of column x.

    All histograms share the same bin edges. The result has one row per group
    and a single column ``0`` holding ``(counts, edges)``.
    """
    x, y = df.columns
    sub = df[df[x].isin(groups)]
    vals = finite(sub[y])
    minv, maxv = (vals.min(), vals.max()) if len(vals) else (0.0, 1.0)
    grouped = dict(tuple(sub.groupby(x, sort=False, observed=True)[y]))
    hists = {
        grp: _hist(grouped[grp], bins, minv, maxv) for grp in groups if grp in grouped
    }
    out = pd.DataFrame({0: pd.Series(hists, dtype=object)})
    out.index.name = x
    return out


def _nom_cont_comps(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """
    Computations for a nominal and continuous column
    """
    data: Dict[str, Any] = {}
    x, _ = df.columns

    cnts = value_counts(df[x])
    data["ttl_grps"] = cnts.shape[0]

    if cfg.box.enable:
        data["box"] = _grouped_box(df, _top_groups(cnts, cfg.box.ngroups, cfg.box.sort_descending))
    if cfg.line.enable:
        data["hist"] = _grouped_hist(
            df, _top_groups(cnts, cfg.line.ngroups, cfg.line.sort_descending), cfg.line.bins
        )

    return data


def geo_cont_comps(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """
    Computations for a geography and continuous column

    Parameters
    ----------
    df
        DataFrame with one geography and one numerical column
    """
    x, y = df.columns
    data = _nom_cont_comps(df, cfg)
    # group the data to compute the mean
    data["value"] = df.groupby(x)[y].mean()

    return data


def geop_cont_comps(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computations for a geopoint and continuous column

    Parameters
    ----------
    df
        DataFrame with one geopoint and one numerical column
    """
    data: Dict[str, Any] = {}
    x, y = df.columns
    # group the data to compute the mean
    data["value"] = df.groupby(x)[y].mean()

    return data


def _calc_box_cont(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Box plot for a binned continuous variable
    """
    x, y = df.columns
    if len(df) == 0:
        return pd.DataFrame(columns=BOX_COLUMNS)
    # group the data into intervals
    bins = pd.cut(df[x], bins=cfg.box.bins, include_lowest=True)
    rows = [
        {"grp": grp, **_box_comps(srs)}
        for grp, srs in df[y].groupby(bins, observed=False, sort=True)
    ]
    return pd.DataFrame(rows, columns=BOX_COLUMNS)


def _box_comps(srs: pd.Series) -> Dict[str, Any]:
    """
    Box plot computations
    """
    arr = srs.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"q1": np.nan, "q2": np.nan, "q3": np.nan, "lw": np.nan, "uw": np.nan, "otlrs": []}

    data: Dict[str, Any] = {}
    # quartiles
    data["q1"], data["q2"], data["q3"] = np.quantile(arr, [0.25, 0.5, 0.75])
    iqr = data["q3"] - data["q1"]
    inside = (arr >= data["q1"] - 1.5 * iqr) & (arr <= data["q3"] + 1.5 * iqr)
    # inliers
    data["lw"], data["uw"] = arr[inside].min(), arr[inside].max()
    # outliers: randomly sample at most 100
    otlrs = arr[~inside]
    if otlrs.size > 100:
        otlrs = np.random.default_rng(0).choice(otlrs, 100, replace=False)
    data["otlrs"] = otlrs.tolist()

    return data


def _hist(srs: pd.Series, bins: int, minv: float, maxv: float) -> Any:
    """
    Compute a histogram on a given series
    """
    arr = srs.to_numpy(dtype=float)
    return np.histogram(arr[np.isfinite(arr)], bins=bins, range=(minv, maxv))


def _calc_box_dt(df: pd.DataFrame, unit: str) -> Tuple[pd.DataFrame, List[str], List[float], str]:
    """
    Calculate a box plot with date on the x axis.

    Parameters
    ----------
    df
        A dataframe with one datetime and one numerical column
    unit
        The unit of time over which to group the values
    """

    x, y = df.columns[0], df.columns[1]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 10) if unit == "auto" else unit
    if unit not in DTMAP:
        raise ValueError(f"Unknown time unit {unit!r}. Use one of {list(DTMAP)} or 'auto'.")
    grps = df.groupby(pd.Grouper(key=x, freq=DTMAP[unit][0]))  # time groups
    # box plot for the values in each time group
    df = pd.concat([_calc_box_stats(grp[y], key) for key, grp in grps], axis=1).T
    df["x"] = np.arange(1, len(df) + 1)
    # If grouping by week, make the label for the week the beginning Sunday
    if unit == "week":
        df.index = df.index - pd.to_timedelta(6, unit="D")
    df.index.name = "grp"
    df = df.reset_index()
    df["grp"] = _period_labels(df["grp"], DTMAP[unit][2])
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting
    for col in ("q1", "q2", "q3", "lw", "uw"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    outx, outy = _calc_box_otlrs(df)

    return df, outx, outy, DTMAP[unit][3]


def _calc_stacked_dt(
    df: pd.DataFrame,
    unit: str,
    ngroups: int,
    largest: bool,
) -> Tuple[pd.DataFrame, Dict[str, int], str]:
    """
    Calculate a stacked bar chart with date on the x axis

    Parameters
    ----------
    df
        A dataframe with one datetime and one categorical column
    unit
        The unit of time over which to group the values
    ngroups
        Number of groups for the categorical column
    largest
        Use the largest or smallest groups in the categorical column
    """
    # pylint: disable=too-many-locals

    x, y = df.columns[0], df.columns[1]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 10) if unit == "auto" else unit
    if unit not in DTMAP:
        raise ValueError(f"Unknown time unit {unit!r}. Use one of {list(DTMAP)} or 'auto'.")

    # get the largest groups
    df_grps, grp_cnt_stats, _ = _calc_groups(df, y, ngroups, largest)
    grouper = pd.Grouper(key=x, freq=DTMAP[unit][0])  # time grouper
    # table of counts with date groups as index and categorical values as column names
    dfr = df_grps.groupby([grouper, y]).size().unstack(fill_value=0).rename_axis(None)
    dfr.columns.name = None
    dfr.columns = [str(col) for col in dfr.columns]

    # if more than ngroups categorical values, aggregate the smallest groups into "Others"
    if grp_cnt_stats[f"{y}_ttl"] > grp_cnt_stats[f"{y}_shw"]:
        grp_cnts = df.groupby(pd.Grouper(key=x, freq=DTMAP[unit][0])).size()
        dfr = dfr.reindex(grp_cnts.index, fill_value=0)
        dfr["Others"] = grp_cnts - dfr.sum(axis=1)

    if unit == "week":  # make the label for the week the beginning Sunday
        dfr.index = dfr.index - pd.to_timedelta(6, unit="D")
    dfr.index = _period_labels(dfr.index.to_series(), DTMAP[unit][2]).to_numpy()

    return dfr, grp_cnt_stats, DTMAP[unit][3]
