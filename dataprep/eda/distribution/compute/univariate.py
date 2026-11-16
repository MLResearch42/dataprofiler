"""
Computations for plot(df, x)
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ....assets.english_stopwords import english_stopwords as ess
from ...configs import Config
from ...dtypes import (
    Continuous,
    DateTime,
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
    _calc_line_dt,
    chisquare,
    finite,
    gaussian_kde,
    histogram,
    kurtosis,
    normaltest,
    skew,
    value_counts,
)
from .common import gen_new_df_with_used_cols

_STOPWORDS = frozenset(ess)


def compute_univariate(
    df: DataFrame,
    col: Union[str, LatLong],
    cfg: Config,
    dtype: Optional[DTypeDef],
) -> Intermediate:
    """
    Compute functions for plot(df, x)

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    col
        A column name from the DataFrame
    cfg
        Config instance
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """

    new_col_names, ndf = gen_new_df_with_used_cols(df, col, None, None)
    x = new_col_names[col]
    if x is None:
        raise ValueError("A column name is required.")

    frame = EDAFrame(ndf, dtype)
    if frame.shape[0] == 0:
        raise ValueError("The DataFrame has no rows, there is nothing to plot.")
    col_dtype = frame.get_eda_dtype(x)

    if isinstance(col_dtype, (Nominal, GeoPoint, SmallCardNum)):
        data = nom_comps(frame.get_col_as_str(x), cfg)
        return Intermediate(col=x, data=data, visual_type="categorical_column")

    if isinstance(col_dtype, Continuous):
        data = cont_comps(frame.frame[x], cfg)
        return Intermediate(col=x, data=data, visual_type="numerical_column")

    if isinstance(col_dtype, DateTime):
        data = calc_stats_dt(frame.frame[x])
        line: Any = []
        if cfg.line.enable and data["Distinct Count"] > 0:
            line = _calc_line_dt(frame.frame[[x]], cfg.line.unit)
        return Intermediate(col=x, data=data, line=line, visual_type="datetime_column")

    if isinstance(col_dtype, GeoGraphy):
        data = nom_comps(frame.get_col_as_str(x), cfg)
        return Intermediate(col=x, data=data, visual_type="geography_column")

    raise ValueError(f"unprocessed type. col:{x}, type:{col_dtype}")


def nom_comps(srs: pd.Series, cfg: Config) -> Dict[str, Any]:
    """
    All computations required for plot(df, Nominal). Assume srs is string column.
    """
    # pylint: disable=too-many-branches
    data: Dict[str, Any] = {}

    data["nrows"] = srs.shape[0]  # total rows
    srs = srs.dropna()  # drop null values
    grps = value_counts(srs)  # counts of unique values in the series
    data["geo"] = grps
    data["nuniq"] = grps.shape[0]  # total number of groups

    # compute bar and pie together unless the parameters are different
    if cfg.bar.enable or cfg.pie.enable or cfg.value_table.enable:
        # select the largest or smallest groups
        data["bar"] = (
            grps.nlargest(cfg.bar.bars) if cfg.bar.sort_descending else grps.nsmallest(cfg.bar.bars)
        )

        if cfg.bar.bars == cfg.pie.slices and cfg.bar.sort_descending == cfg.pie.sort_descending:
            data["pie"] = data["bar"]
        else:
            data["pie"] = (
                grps.nlargest(cfg.pie.slices)
                if cfg.pie.sort_descending
                else grps.nsmallest(cfg.pie.slices)
            )

        if cfg.bar.bars == cfg.value_table.ngroups and cfg.bar.sort_descending:
            data["value_table"] = data["bar"]
        elif cfg.pie.slices == cfg.value_table.ngroups and cfg.pie.sort_descending:
            data["value_table"] = data["pie"]
        else:
            data["value_table"] = grps.nlargest(cfg.value_table.ngroups)

        if cfg.insight.enable:
            data["chisq"] = chisquare(grps.to_numpy())

    # dataframe with the distinct values (column "index") and their counts
    df = grps.rename_axis("index").reset_index()
    # the length of every distinct value, and how often that value occurs
    lens = df["index"].str.len().to_numpy(dtype=float)
    weights = df[grps.name].to_numpy()

    if cfg.stats.enable or cfg.value_table.enable:
        data.update(_calc_nom_stats(srs, df, data["nrows"], data["nuniq"]))
    elif cfg.wordfreq.enable and cfg.insight.enable:
        data["len_stats"] = {
            "Minimum": lens.min() if lens.size else np.nan,
            "Maximum": lens.max() if lens.size else np.nan,
        }
    if cfg.wordlen.enable:
        if lens.size:
            data["len_hist"] = np.histogram(
                lens, bins=cfg.wordlen.bins, range=(lens.min(), lens.max()), weights=weights
            )
        else:
            data["len_hist"] = histogram(lens, cfg.wordlen.bins)
    if cfg.wordcloud.enable or cfg.wordfreq.enable:
        word_freqs = _calc_word_freq(
            df,
            cfg.wordfreq.top_words,
            cfg.wordfreq.stopword,
            cfg.wordfreq.lemmatize,
            cfg.wordfreq.stem,
        )
        if all(
            getattr(cfg.wordcloud, att) == getattr(cfg.wordfreq, att)
            for att in ("top_words", "stopword", "stem", "lemmatize")
        ):
            word_freqs_cloud = word_freqs
        else:
            word_freqs_cloud = _calc_word_freq(
                df,
                cfg.wordcloud.top_words,
                cfg.wordcloud.stopword,
                cfg.wordcloud.lemmatize,
                cfg.wordcloud.stem,
            )
        data["word_cnts_cloud"] = word_freqs_cloud["word_cnts"]
        data["nuniq_words_cloud"] = word_freqs_cloud["nuniq_words"]
        data["word_cnts_freq"] = word_freqs["word_cnts"]
        data["nwords_freq"] = word_freqs["nwords"]

    return data


def cont_comps(srs: pd.Series, cfg: Config) -> Dict[str, Any]:
    """
    All computations required for plot(df, Continuous)
    """
    # pylint: disable=too-many-branches
    data: Dict[str, Any] = {}

    data["nrows"] = srs.shape[0]  # total rows
    srs = srs.dropna()
    data["npres"] = srs.shape[0]  # number of present (not null) values
    srs = finite(srs)  # remove infinite values
    arr = srs.to_numpy()
    has_data = arr.size > 0
    minv, maxv = (arr.min(), arr.max()) if has_data else (np.nan, np.nan)
    # always available: the renderer uses them to detect constant columns
    data["min"], data["max"] = minv, maxv

    if cfg.hist.enable or cfg.qqnorm.enable and cfg.insight.enable:
        data["hist"] = histogram(arr, cfg.hist.bins, (minv, maxv))
        if cfg.insight.enable:
            data["norm"] = normaltest(data["hist"][0])
    if cfg.hist.enable and cfg.insight.enable:
        data["chisq"] = chisquare(data["hist"][0])
    # compute only the required amount of quantiles
    if cfg.qqnorm.enable:
        data["qntls"] = _quantiles(srs, np.linspace(0.01, 0.99, 99))
    elif cfg.stats.enable:
        data["qntls"] = _quantiles(srs, [0.05, 0.25, 0.5, 0.75, 0.95])
    elif cfg.box.enable:
        data["qntls"] = _quantiles(srs, [0.25, 0.5, 0.75])
    if cfg.stats.enable or cfg.hist.enable and cfg.insight.enable:
        data["skew"] = skew(arr)
    if cfg.stats.enable or cfg.qqnorm.enable:
        data["mean"] = arr.mean() if has_data else np.nan
        data["std"] = arr.std(ddof=1) if arr.size > 1 else np.nan
    if cfg.stats.enable:
        data["nreals"] = srs.shape[0]
        data["nzero"] = int((arr == 0).sum())
        data["nneg"] = int((arr < 0).sum())
        data["kurt"] = kurtosis(arr)
        data["mem_use"] = int(srs.memory_usage(deep=True))
    # compute the density histogram
    if cfg.kde.enable:
        # gaussian_kde needs at least two distinct values
        if has_data and not np.isclose(minv, maxv):
            data["dens"] = histogram(arr, cfg.kde.bins, (minv, maxv), density=True)
            # gaussian kernel density estimate, on a sample to bound its cost
            smp = arr if arr.size <= 1000 else np.random.default_rng(0).choice(arr, 1000, False)
            data["kde"] = gaussian_kde(smp)
        else:
            data["kde"] = None
    if cfg.box.enable:
        data.update(_calc_box(srs, data["qntls"], cfg))
    if cfg.value_table.enable:
        cnts = value_counts(srs)
        data["nuniq"] = cnts.shape[0]
        data["value_table"] = cnts.nlargest(cfg.value_table.ngroups)
    elif cfg.stats.enable:
        data["nuniq"] = int(srs.nunique())

    return data


def _quantiles(srs: pd.Series, qs: Any) -> pd.Series:
    """Quantiles of a series as a Series indexed by the quantile (NaN when empty)."""
    qs = np.asarray(qs, dtype=float)
    if srs.shape[0] == 0:
        return pd.Series(np.nan, index=qs, name=srs.name)
    return pd.Series(np.quantile(srs.to_numpy(dtype=float), qs), index=qs, name=srs.name)


def _calc_box(srs: pd.Series, qntls: pd.Series, cfg: Config) -> Dict[str, Any]:
    """
    Box plot calculations
    """
    idx = np.round(np.asarray(qntls.index, dtype=float), 2)
    lookup = dict(zip(idx, qntls.to_numpy()))
    # quartiles
    data: Dict[str, Any] = {
        f"qrtl{i + 1}": lookup[qnt] for i, qnt in enumerate((0.25, 0.5, 0.75))
    }
    arr = srs.to_numpy()

    # inter-quartile range
    iqr = data["qrtl3"] - data["qrtl1"]
    inside = (arr >= data["qrtl1"] - 1.5 * iqr) & (arr <= data["qrtl3"] + 1.5 * iqr)
    in_arr = arr[inside]
    # lower and upper whiskers
    data["lw"] = in_arr.min() if in_arr.size else np.nan
    data["uw"] = in_arr.max() if in_arr.size else np.nan

    # outliers: randomly sample at most 100 without replacement
    otlrs = arr[~inside]
    if otlrs.size > 100:
        data["otlrs"] = np.random.default_rng(0).choice(otlrs, 100, replace=False)
    else:
        data["otlrs"] = otlrs
    if cfg.insight.enable:
        data["notlrs"] = int(otlrs.size)

    return data


def _calc_word_freq(
    df: pd.DataFrame,
    top_words: int,
    stopword: bool,
    lemmatize: bool,
    stem: bool,
) -> Dict[str, Any]:
    """
    Parse a categorical column of text data into words, then compute
    the frequency distribution of words and the total number of words.

    ``df`` has two columns: the distinct values and how often each occurs.
    """
    col, cnt = df.columns[0], df.columns[1]

    # drop punctuation, lower-case and split on whitespace
    words = (
        df[col].astype(object).map(str).str.replace(r"[^\w+ ]", "", regex=True).str.lower().str.split()
    )
    exploded = pd.DataFrame({"word": words, cnt: df[cnt].to_numpy()}).explode("word")
    exploded = exploded.dropna(subset=["word"])
    if stopword:
        exploded = exploded[~exploded["word"].isin(_STOPWORDS)]

    # lemmatize and stem
    if lemmatize or stem:
        uniq = exploded["word"].unique()
        mapped = list(uniq)
        if lemmatize:
            lemmatizer = _get_lemmatizer()
            mapped = [lemmatizer(word) for word in mapped]
        if stem:
            stemmer = _get_stemmer()
            mapped = [stemmer(word) for word in mapped]
        exploded["word"] = exploded["word"].map(dict(zip(uniq, mapped)))

    # counts of words, named after the count column so the renderer finds it
    word_cnts = exploded.groupby("word", sort=False)[cnt].sum()
    word_cnts.index.name = col
    nwords = int(word_cnts.sum())  # total number of words
    nuniq_words = word_cnts.shape[0]  # total unique words
    fnl_word_cnts = word_cnts.nlargest(top_words)  # words with the highest frequency

    return {"word_cnts": fnl_word_cnts, "nwords": nwords, "nuniq_words": nuniq_words}


def _get_stemmer() -> Any:
    """The Porter stemmer from NLTK (needs the optional ``nltk`` package, no downloads)."""
    try:
        from nltk.stem import PorterStemmer  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "Stemming needs the optional 'nltk' package: pip install nltk"
        ) from err
    return PorterStemmer().stem


def _get_lemmatizer() -> Any:
    """The WordNet lemmatizer from NLTK (needs ``nltk`` and its 'wordnet' corpus)."""
    try:
        from nltk.stem import WordNetLemmatizer  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "Lemmatizing needs the optional 'nltk' package: pip install nltk"
        ) from err
    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("words")
    except LookupError as err:
        raise LookupError(
            "Lemmatizing needs NLTK's 'wordnet' corpus, which is not installed. On a "
            "machine with internet access run nltk.download('wordnet'); on an offline "
            "machine copy the corpus into an nltk_data folder, or use stemming instead "
            '(config={"wordfreq.stem": True}), which needs no extra data.'
        ) from err
    return lemmatizer.lemmatize


def _calc_nom_stats(
    srs: pd.Series,
    df: pd.DataFrame,
    nrows: int,
    nuniq: int,
) -> Dict[str, Any]:
    """
    Calculate statistics for a nominal column.

    ``srs`` is the column without nulls; ``df`` holds its distinct values and
    their counts. Everything is computed on the distinct values and weighted by
    the counts, which is much faster than scanning every row.
    """
    grp, col = df.columns
    vals = df[grp]
    cnts = df[col].to_numpy(dtype=float)
    npres = srs.shape[0]

    # overview stats
    stats = {
        "nrows": nrows,
        "npres": npres,
        "nuniq": nuniq,
        "mem_use": int(srs.memory_usage(deep=True)),
        "first_rows": srs.reset_index(drop=True).iloc[:5],
    }

    # length stats
    lens = vals.str.len().to_numpy(dtype=float)
    if npres > 0:
        mean = float(np.average(lens, weights=cnts))
        var = float(np.sum(cnts * (lens - mean) ** 2) / (npres - 1)) if npres > 1 else np.nan
        order = np.argsort(lens, kind="stable")
        median = _weighted_median(lens[order], cnts[order])
        leng = {
            "Mean": mean,
            "Standard Deviation": float(np.sqrt(var)),
            "Median": median,
            "Minimum": lens.min(),
            "Maximum": lens.max(),
        }
    else:
        leng = {
            "Mean": np.nan,
            "Standard Deviation": np.nan,
            "Median": np.nan,
            "Minimum": np.nan,
            "Maximum": np.nan,
        }

    # letter stats
    def count(pattern: str) -> int:
        return int((vals.str.count(pattern).to_numpy(dtype=float) * cnts).sum())

    lc_cnt = count(r"[a-z]")
    uc_cnt = count(r"[A-Z]")
    letter = {
        "Count": lc_cnt + uc_cnt,
        "Lowercase Letter": lc_cnt,
        "Space Separator": count(r"[ ]"),
        "Uppercase Letter": uc_cnt,
        "Dash Punctuation": count(r"[-]"),
        "Decimal Number": count(r"[0-9]"),
    }

    return {"stats": stats, "len_stats": leng, "letter_stats": letter}


def _weighted_median(sorted_vals: np.ndarray, weights: np.ndarray) -> float:
    """Median of values that each occur ``weights`` times; ``sorted_vals`` is ascending."""
    total = weights.sum()
    cum = np.cumsum(weights)
    # 0-based positions of the middle element(s) in the fully expanded data
    lo = np.searchsorted(cum, np.floor((total - 1) / 2) + 1)
    hi = np.searchsorted(cum, np.ceil((total - 1) / 2) + 1)
    return float((sorted_vals[lo] + sorted_vals[hi]) / 2)


def calc_stats_dt(srs: pd.Series) -> Dict[str, Any]:
    """
    Calculate stats from a datetime column
    """
    size = srs.shape[0]  # include nan
    count = int(srs.count())  # exclude nan
    uniq_count = int(srs.nunique())
    overview_dict = {
        "Distinct Count": uniq_count,
        "Approximate Unique (%)": uniq_count / count if count else 0.0,
        "Missing": size - count,
        "Missing (%)": 1 - (count / size) if size else 0.0,
        "Memory Size": int(srs.memory_usage(deep=True)),
        "Minimum": srs.min(),
        "Maximum": srs.max(),
    }

    return overview_dict
