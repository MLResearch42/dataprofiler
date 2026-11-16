"""Miscellaneous helpers shared by the compute and render code."""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bokeh.models import CustomJSTickFormatter, Legend
from bokeh.plotting import figure
from scipy import stats as _stats

from .dtypes import drop_null
from .eda_frame import to_pandas

LOGGER = logging.getLogger(__name__)

_NAN_PAIR = (float("nan"), float("nan"))


def to_dask(df: Any) -> pd.DataFrame:
    """Deprecated: EDA now runs in memory. Returns a pandas DataFrame."""
    return to_pandas(df)


def finite(srs: pd.Series) -> pd.Series:
    """Drop missing and infinite values from a numerical series."""
    arr = srs.to_numpy()
    if arr.dtype.kind == "f":
        mask = np.isfinite(arr)
        return srs if mask.all() else srs[mask]
    return srs.dropna()


def sample_n(arr: np.ndarray, n: int) -> np.ndarray:  # pylint: disable=C0103
    """Sample n values uniformly from the range of the `arr`,
    not from the distribution of `arr`'s elems."""

    if len(arr) <= n:
        return arr

    subsel = np.linspace(0, len(arr) - 1, n)
    subsel = np.floor(subsel).astype(int)
    return arr[subsel]


def relocate_legend(fig: figure, loc: str) -> figure:
    """Relocate legend(s) from center to `loc`."""
    remains = []
    targets = []
    for layout in fig.center:
        if isinstance(layout, Legend):
            targets.append(layout)
        else:
            remains.append(layout)
    fig.center = remains
    for layout in targets:
        fig.add_layout(layout, loc)

    return fig


def cut_long_name(name: str, max_len: int = 18) -> str:
    """If the name is longer than `max_len`,
    cut it to `max_len` length and append "..."""

    # Bug 136 Fixed
    name = str(name)
    cut_name = f"{name[:13]}...{name[len(name)-3:]}" if len(name) > max_len else name
    return cut_name


def fuse_missing_perc(name: str, perc: float) -> str:
    """Append (x.y%) to the name if `perc` is not 0."""
    if perc == 0:
        return name

    return f"{name} ({perc:.1%})"


# Dictionary for mapping the time unit to its formatting. Each entry is of the
# form unit:(unit code for pd.Grouper freq parameter, pandas to_period strftime
# formatting for line charts, pandas to_period strftime formatting for box plot,
# label format).
DTMAP = {
    "year": ("YE", "%Y", "%Y", "Year"),
    "quarter": ("QE", "Q%q %Y", "Q%q %Y", "Quarter"),
    "month": ("ME", "%B %Y", "%b %Y", "Month"),
    "week": ("W-SAT", "%d %B, %Y", "%d %b, %Y", "Week of"),
    "day": ("D", "%d %B, %Y", "%d %b, %Y", "Date"),
    "hour": ("h", "%d %B, %Y, %I %p", "%d %b, %Y, %I %p", "Hour"),
    "minute": ("min", "%d %B, %Y, %I:%M %p", "%d %b, %Y, %I:%M %p", "Minute"),
    "second": ("s", "%d %B, %Y, %I:%M:%S %p", "%d %b, %Y, %I:%M:%S %p", "Second"),
}


def _get_timeunit(min_time: pd.Timestamp, max_time: pd.Timestamp, dflt: int) -> str:
    """Auxillary function to find an appropriate time unit. Will find the
    time unit such that the number of time units are closest to dflt."""

    dt_secs = {
        "year": 60 * 60 * 24 * 365,
        "quarter": 60 * 60 * 24 * 91,
        "month": 60 * 60 * 24 * 30,
        "week": 60 * 60 * 24 * 7,
        "day": 60 * 60 * 24,
        "hour": 60 * 60,
        "minute": 60,
        "second": 1,
    }

    time_rng_secs = (max_time - min_time).total_seconds()
    prev_bin_cnt, prev_unit = 0, "year"
    for unit, secs_in_unit in dt_secs.items():
        cur_bin_cnt = time_rng_secs / secs_in_unit
        if abs(prev_bin_cnt - dflt) < abs(cur_bin_cnt - dflt):
            return prev_unit
        prev_bin_cnt = cur_bin_cnt
        prev_unit = unit

    return prev_unit


def _calc_box_stats(grp_srs: pd.Series, grp: Any, dlyd: bool = True) -> pd.DataFrame:
    """Tukey box plot statistics for one group of values.

    ``dlyd`` is ignored and only kept for backwards compatibility.
    """
    # pylint: disable=unused-argument
    stats: Dict[str, Any] = {}
    grp_srs = grp_srs.dropna()

    if len(grp_srs) == 0:
        stats["q1"] = stats["q2"] = stats["q3"] = np.nan
        stats["lw"] = stats["uw"] = np.nan
        stats["otlrs"] = []
        return pd.DataFrame({grp: stats})

    qntls = np.round(grp_srs.quantile([0.25, 0.50, 0.75]), 3)
    stats["q1"], stats["q2"], stats["q3"] = qntls[0.25], qntls[0.50], qntls[0.75]

    iqr = stats["q3"] - stats["q1"]
    stats["lw"] = grp_srs[grp_srs >= stats["q1"] - 1.5 * iqr].min()
    stats["uw"] = grp_srs[grp_srs <= stats["q3"] + 1.5 * iqr].max()

    otlrs = grp_srs[(grp_srs < stats["lw"]) | (grp_srs > stats["uw"])]
    if len(otlrs) > 100:  # sample 100 outliers
        otlrs = otlrs.sample(n=100, random_state=0)
    stats["otlrs"] = list(otlrs)

    return pd.DataFrame({grp: stats})


def _calc_box_otlrs(df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """Flatten the per-group outlier lists into x / y lists for plotting."""
    outx: List[str] = []  # list for the outlier groups
    outy: List[float] = []  # list for the outlier values
    for grp, otlrs in zip(df["grp"], df["otlrs"]):
        outx.extend([grp] * len(otlrs))
        outy.extend(otlrs)

    return outx, outy


def _period_labels(times: pd.Series, fmt: str) -> pd.Series:
    """Format timestamps with a Period format string (supports ``%q`` for quarters)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # dropping tz / nanoseconds is fine
        return times.dt.to_period("s").dt.strftime(fmt)


def _calc_line_dt(
    df: pd.DataFrame,
    unit: str,
    agg: Optional[str] = None,
    ngroups: Optional[int] = None,
    largest: Optional[bool] = None,
) -> Union[
    Tuple[Dict[str, Tuple[List[Any], List[Any], List[str]]], Dict[str, int], str],
    Tuple[pd.DataFrame, str, float],
    Tuple[pd.DataFrame, str],
]:
    """
    Calculate a line or multiline chart with date on the x axis. If df contains
    one datetime column, it will make a line chart of the frequency of values. If
    df contains a datetime and categorical column, it will compute the frequency
    of each categorical value in each time group. If df contains a datetime and
    numerical column, it will compute the aggregate of the numerical column grouped
    by the time groups. If df contains a datetime, categorical, and numerical column,
    it will compute the aggregate of the numerical column for values in the categorical
    column grouped by time.

    Parameters
    ----------
    df
        A dataframe whose first column is the datetime column
    unit
        The unit of time over which to group the values, or "auto"
    agg
        Aggregate to use for the numerical column
    ngroups
        Number of groups for the categorical column
    largest
        Use the largest or smallest groups in the categorical column
    """
    # pylint: disable=too-many-locals

    x = df.columns[0]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 100) if unit == "auto" else unit
    if unit not in DTMAP:
        raise ValueError(f"Unknown time unit {unit!r}. Use one of {list(DTMAP)} or 'auto'.")
    grouper = pd.Grouper(key=x, freq=DTMAP[unit][0])  # for grouping the time values

    # multiline charts
    if ngroups and largest is not None:
        hist_dict: Dict[str, Tuple[List[Any], List[Any], List[str]]] = {}
        agg = "freq" if agg is None else agg  # default agg if unspecified for notational concision

        # categorical column for grouping over, each resulting group is a line in the chart
        grpby_col = df.columns[1] if len(df.columns) == 2 else df.columns[2]
        df, grp_cnt_stats, largest_grps = _calc_groups(df, grpby_col, ngroups, largest)

        for grp, srs in df.groupby(grpby_col, sort=False):
            # calculate the frequencies or aggregate value in each time group
            if len(df.columns) == 3:
                dfr = srs.groupby(grouper)[df.columns[1]].agg(agg).reset_index()
            else:
                dfr = srs[[x]].groupby(grouper).size().reset_index()
            dfr.columns = [x, agg]
            # if grouping by week, make the label for the week the beginning Sunday
            if unit == "week":
                dfr[x] = dfr[x] - pd.to_timedelta(6, unit="D")
            hist_dict[grp] = (
                list(dfr[agg]),
                list(dfr[x]),
                list(_period_labels(dfr[x], DTMAP[unit][1])),
            )
        hist_dict = {grp: hist_dict[grp] for grp in largest_grps if grp in hist_dict}
        return hist_dict, grp_cnt_stats, DTMAP[unit][3]

    # single line charts
    nrows = len(df)
    if agg is None:  # frequency of datetime column
        miss_pct = round(df[x].isna().sum() / nrows * 100, 1) if nrows else 0.0
        dfr = drop_null(df[[x]]).groupby(grouper).size().reset_index()
        dfr.columns = [x, "freq"]
        dfr["pct"] = dfr["freq"] / nrows * 100 if nrows else 0.0
    else:  # aggregate over a second column
        dfr = df.groupby(grouper)[df.columns[1]].agg(agg).reset_index()
        dfr.columns = [x, agg]
    if unit == "week":
        dfr[x] = dfr[x] - pd.to_timedelta(6, unit="D")
    dfr["lbl"] = _period_labels(dfr[x], DTMAP[unit][1])

    return (dfr, DTMAP[unit][3], miss_pct) if agg is None else (dfr, DTMAP[unit][3])


def _calc_running_total_dt(df: pd.DataFrame, unit: str) -> Tuple[pd.DataFrame, str]:
    """
    Calculate a running total line for a df two columns: a datetime column and numerical column.

    Parameters
    ----------
    df
        A dataframe
    unit
        The unit of time over which to group the values
    """
    res_df, time_prefix = _calc_line_dt(df, unit, agg="sum")  # type: ignore[misc]
    res_df["sum"] = res_df["sum"].cumsum()
    res_df = res_df.rename(columns={"sum": "runningtotal"})
    return res_df, time_prefix


def _calc_groups(
    df: pd.DataFrame, x: str, ngroups: int, largest: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Keep only the rows belonging to the ``ngroups`` largest (or smallest) groups of x."""

    # group count statistics to inform the user of the sampled output
    grp_cnt_stats: Dict[str, int] = {}

    srs = df.groupby(x).size()
    srs_lrgst = srs.nlargest(n=ngroups) if largest else srs.nsmallest(n=ngroups)
    largest_grps = list(srs_lrgst.index)
    grp_cnt_stats[f"{x}_ttl"] = len(srs.index)

    df = df[df[x].isin(largest_grps)]
    grp_cnt_stats[f"{x}_shw"] = len(largest_grps)

    return df, grp_cnt_stats, largest_grps


def _quiet(func: Any, *args: Any) -> Any:
    """Run a scipy test, returning NaNs instead of raising or warning on tiny inputs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = func(*args)
            return (float(res[0]), float(res[1]))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return _NAN_PAIR


def normaltest(arr: Any) -> Tuple[float, float]:
    """D'Agostino-Pearson normality test. Returns ``(statistic, p-value)``."""
    return _quiet(_stats.normaltest, np.asarray(arr, dtype=float))


def skewtest(arr: Any) -> Tuple[float, float]:
    """Test whether the skew differs from a normal distribution's. ``(statistic, p-value)``."""
    return _quiet(_stats.skewtest, np.asarray(arr, dtype=float))


def chisquare(arr: Any) -> Tuple[float, float]:
    """Chi-square test against a uniform distribution. Returns ``(statistic, p-value)``."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return _NAN_PAIR
    return _quiet(_stats.chisquare, arr)


def ks_2samp(data1: Any, data2: Any) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns ``(statistic, p-value)``."""
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    data1, data2 = data1[~np.isnan(data1)], data2[~np.isnan(data2)]
    if data1.size == 0 or data2.size == 0:
        return _NAN_PAIR
    return _quiet(_stats.ks_2samp, data1, data2)


def gaussian_kde(arr: Any) -> Any:
    """A Gaussian kernel density estimate, or None when it cannot be computed.

    The estimate is undefined for fewer than two values or when all values are equal.
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or np.ptp(arr) == 0:
        return None
    try:
        return _stats.gaussian_kde(arr)
    except (np.linalg.LinAlgError, ValueError):
        return None


def skew(arr: Any) -> float:
    """Sample skewness (biased, matching scipy's default)."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(_stats.skew(arr))


def kurtosis(arr: Any) -> float:
    """Sample excess kurtosis (biased, matching scipy's default)."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(_stats.kurtosis(arr))


def histogram(
    arr: Any, bins: int, range: Optional[Tuple[float, float]] = None, density: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """``np.histogram`` that copes with empty input and a degenerate or missing range."""
    # pylint: disable=redefined-builtin
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if range is None or any(v is None or not np.isfinite(v) for v in range):
        range = (float(arr.min()), float(arr.max())) if arr.size else (0.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        counts, edges = np.histogram(
            arr, bins=bins, range=(float(range[0]), float(range[1])), density=density
        )
    if density:
        counts = np.nan_to_num(counts)
    return counts, edges


def value_counts(srs: pd.Series) -> pd.Series:
    """Counts of the distinct non-null values, unsorted.

    The result is named after the input column and its index is unnamed, which is
    the layout the render code expects on every pandas version.
    """
    cnts = srs.value_counts(sort=False)
    cnts.name = srs.name
    cnts.index.name = None
    return cnts


def str_lengths(srs: pd.Series) -> pd.Series:
    """Length of every string in a string series (nulls stay null)."""
    return srs.str.len()


def sample_frame(
    df: pd.DataFrame, sample_size: Optional[int], sample_rate: Optional[float] = None
) -> pd.DataFrame:
    """Randomly sample rows, by count when ``sample_size`` is given, else by rate."""
    if sample_size is not None:
        return df.sample(n=min(sample_size, len(df)), random_state=0)
    if sample_rate is not None:
        return df.sample(frac=sample_rate, random_state=0)
    return df


def tweak_figure(
    fig: figure,
    ptype: Optional[str] = None,
    show_yticks: bool = False,
    max_lbl_len: int = 15,
) -> None:
    """
    Set some common attributes for a figure
    """
    fig.axis.major_label_text_font_size = "9pt"
    fig.title.text_font_size = "10pt"
    fig.axis.minor_tick_line_color = "white"
    if ptype in ["pie", "qq", "heatmap"]:
        fig.ygrid.grid_line_color = None
    if ptype in ["bar", "pie", "hist", "kde", "qq", "heatmap", "line"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["bar", "hist", "line"] and not show_yticks:
        fig.ygrid.grid_line_color = None
        fig.yaxis.major_label_text_font_size = "0pt"
        fig.yaxis.major_tick_line_color = None
    if ptype in ["bar", "nested", "stacked", "heatmap", "box"]:
        fig.xaxis.major_label_orientation = np.pi / 3
        fig.xaxis.formatter = CustomJSTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (max_lbl_len, max_lbl_len)
        )
    if ptype in ["nested", "stacked", "box"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["nested", "stacked"]:
        fig.y_range.start = 0
        fig.x_range.range_padding = 0.03
    if ptype in ["line", "boxnum"]:
        fig.min_border_right = 20
        fig.xaxis.major_label_standoff = 7
        fig.xaxis.major_label_orientation = 0
        fig.xaxis.major_tick_line_color = None


def _format_ticks(ticks: List[float]) -> List[str]:
    """
    Format the tick values: large numbers are abbreviated with K, M, B and T.
    """
    formatted_ticks = []
    for tick in ticks:  # format the tick values
        if abs(tick) < 1e4 or abs(tick) >= 1e15 or not np.isfinite(tick):
            formatted_ticks.append(str(tick))
            continue
        before, after = f"{tick:e}".split("e")
        mod_exp = int(after) % 3
        factor = 1 if mod_exp == 0 else 10 if mod_exp == 1 else 100
        value = np.round(float(before) * factor, len(str(before)))
        value = int(value) if float(value).is_integer() else value
        if abs(tick) >= 1e12:
            formatted_ticks.append(str(value) + "T")
        elif abs(tick) >= 1e9:
            formatted_ticks.append(str(value) + "B")
        elif abs(tick) >= 1e6:
            formatted_ticks.append(str(value) + "M")
        else:
            formatted_ticks.append(str(value) + "K")

    return formatted_ticks


def _format_axis(fig: figure, minv: float, maxv: float, axis: str) -> None:
    """
    Put about five round, human-readable ticks on an axis.

    When the range is empty, infinite or undefined the axis is left to bokeh's
    defaults instead of failing.
    """  # pylint: disable=too-many-locals
    try:
        minv, maxv = float(minv), float(maxv)
    except (TypeError, ValueError):
        return
    if not (np.isfinite(minv) and np.isfinite(maxv)) or maxv <= minv:
        return

    # divisor for 5 ticks (5 results in ticks that are too close together)
    divisor = 4.5
    # interval
    gap = (maxv - minv) / divisor
    # get exponent from scientific notation
    _, after = f"{gap:.0e}".split("e")
    # round to this amount
    round_to = -1 * int(after)
    # round the first tick and the value between ticks
    first = float(np.round(minv, round_to))
    gap = float(np.round(gap, round_to))
    if gap <= 0:
        return

    # make the tick values
    nticks = int(min(np.floor((maxv - first) / gap + 1e-9), 50)) + 1
    ticks = np.round(first + gap * np.arange(nticks), max(round_to, 0) + 2)
    ticks = [int(tick) if float(tick).is_integer() else float(tick) for tick in ticks]
    formatted_ticks = _format_ticks(ticks)
    overrides = dict(zip(ticks, formatted_ticks))

    if axis == "x":
        fig.xgrid.ticker = ticks
        fig.xaxis.ticker = ticks
        fig.xaxis.major_label_overrides = overrides
        fig.xaxis.major_label_text_font_size = "10pt"
        fig.xaxis.major_label_standoff = 7
        fig.xaxis.major_tick_line_color = None
    elif axis == "y":
        fig.ygrid.ticker = ticks
        fig.yaxis.ticker = ticks
        fig.yaxis.major_label_overrides = overrides
        fig.yaxis.major_label_text_font_size = "10pt"
        fig.yaxis.major_label_standoff = 5


def _format_bin_intervals(bins_arr: np.ndarray) -> List[str]:
    """
    Auxillary function to format bin intervals in a histogram
    """
    edges = [float(val) for val in np.asarray(bins_arr, dtype=float)]
    # enough significant digits to tell neighbouring edges apart
    width = min((b - a for a, b in zip(edges, edges[1:]) if b > a), default=1.0)
    digits = int(min(max(3, -np.floor(np.log10(width)) + 2), 12)) if width > 0 else 3
    vals = [int(val) if val.is_integer() else round(val, digits) for val in edges]
    if len(vals) < 2:
        return []
    intervals = [f"[{vals[i]}, {vals[i + 1]})" for i in range(len(vals) - 2)]
    intervals.append(f"[{vals[-2]}, {vals[-1]}]")
    return intervals
