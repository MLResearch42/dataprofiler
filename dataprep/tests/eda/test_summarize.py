"""summarize(df): the statistics of every column as a DataFrame."""

import numpy as np
import pandas as pd
import pytest

from ...datasets import load_dataset
from ...eda import summarize
from ...eda.summary import SUMMARY_COLUMNS


@pytest.fixture(scope="module")
def titanic() -> pd.DataFrame:
    return load_dataset("titanic")


@pytest.fixture
def mixed() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "num": rng.normal(10, 2, size=n),
            "with_inf": np.r_[[np.inf, -np.inf], rng.normal(size=n - 2)],
            "all_missing": np.nan,
            "const": 5,
            "flag": rng.random(n) > 0.5,
            "cat": pd.Categorical(rng.choice(list("abc"), n)),
            "nullable": pd.array(rng.integers(0, 100, n), dtype="Int64"),
            "when": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
            "lists": [[1, 2]] * n,
        }
    )
    df.loc[::10, "nullable"] = pd.NA
    return df


def test_schema_is_fixed(mixed: pd.DataFrame) -> None:
    summary = summarize(mixed)
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert list(summary.index) == list(mixed.columns)
    assert summary.index.name == "column"
    for col in ("count", "missing", "distinct", "top_count"):
        assert summary[col].dtype == np.int64
    # the same columns for a table without any numerical or datetime column
    assert list(summarize(mixed[["cat"]]).columns) == SUMMARY_COLUMNS


def test_numbers_match_pandas(titanic: pd.DataFrame) -> None:
    summary = summarize(titanic)
    for col in ("Age", "Fare"):
        srs = titanic[col].dropna()
        row = summary.loc[col]
        assert row["count"] == len(srs)
        assert row["missing"] == titanic[col].isna().sum()
        assert row["missing_pct"] == pytest.approx(100 * titanic[col].isna().mean())
        assert row["distinct"] == srs.nunique()
        assert row["mean"] == pytest.approx(srs.mean())
        assert row["std"] == pytest.approx(srs.std())
        assert row["min"] == srs.min() and row["max"] == srs.max()
        assert row["p25"] == pytest.approx(srs.quantile(0.25))
        assert row["median"] == pytest.approx(srs.median())
        assert row["p75"] == pytest.approx(srs.quantile(0.75))
        assert row["skewness"] == pytest.approx(srs.skew())
        assert row["kurtosis"] == pytest.approx(srs.kurt())
        assert row["zeros"] == (srs == 0).sum()
    sex = summary.loc["Sex"]
    assert (sex["top"], sex["top_count"], sex["distinct"]) == ("male", 577, 2)
    assert (sex["min_length"], sex["max_length"]) == (4, 6)
    assert sex["mean_length"] == pytest.approx(titanic["Sex"].str.len().mean())
    assert np.isnan(sex["mean"])
    assert summary.loc["Cabin", "missing"] == 687


def test_insights_match_plot(titanic: pd.DataFrame) -> None:
    summary = summarize(titanic)
    assert "Missing" in summary.loc["Age", "insights"]
    assert "Skewed" in summary.loc["Fare", "insights"]
    assert "High Cardinality" in summary.loc["Name", "insights"]
    assert summary.loc["Sex", "insights"] == ""
    assert summary[summary["missing_pct"] > 5].index.tolist() == ["Age", "Cabin"]
    # thresholds can be changed like in plot()
    strict = summarize(titanic, config={"insight.missing.threshold": 50})
    assert "Missing" not in strict.loc["Age", "insights"]
    assert "Missing" in strict.loc["Cabin", "insights"]


def test_dataset_attrs(titanic: pd.DataFrame) -> None:
    info = summarize(titanic).attrs["dataset"]
    assert info["rows"] == 891 and info["columns"] == 12
    assert info["duplicate_rows"] == 0
    assert info["missing_cells"] == int(titanic.isna().sum().sum())
    assert info["memory_bytes"] > 0
    assert info["sample"] is None
    doubled = summarize(pd.concat([titanic, titanic.head(10)])).attrs["dataset"]
    assert doubled["duplicate_rows"] == 10


def test_awkward_columns(mixed: pd.DataFrame) -> None:
    summary = summarize(mixed)
    assert summary.loc["with_inf", "infinite"] == 2
    assert np.isfinite(summary.loc["with_inf", "mean"])
    empty = summary.loc["all_missing"]
    assert (empty["count"], empty["missing"], empty["distinct"]) == (0, 200, 0)
    assert empty["insights"] == "Missing"  # not "Unique": there are no values
    assert summary.loc["const", "insights"].startswith("Constant")
    assert summary.loc["nullable", "dtype"] == "Int64"
    assert summary.loc["nullable", "missing"] == 20
    assert summary.loc["when", "type"] == "DateTime"
    assert summary.loc["when", "earliest"] == pd.Timestamp("2020-01-01")
    assert summary.loc["when", "latest"] == pd.Timestamp("2020-01-01") + pd.Timedelta(days=199)
    assert summary.loc["lists", "distinct"] == 1


def test_user_frame_not_modified(mixed: pd.DataFrame) -> None:
    before = mixed.copy()
    summarize(mixed)
    pd.testing.assert_frame_equal(mixed, before)


def test_dtype_override(titanic: pd.DataFrame) -> None:
    summary = summarize(titanic, dtype={"Fare": "nominal"})
    assert summary.loc["Fare", "type"] == "Nominal"
    assert np.isnan(summary.loc["Fare", "mean"])


def test_target_columns(titanic: pd.DataFrame) -> None:
    summary = summarize(titanic, target="Survived")
    assert list(summary.columns) == SUMMARY_COLUMNS + ["target_score", "target_method"]
    assert np.isnan(summary.loc["Survived", "target_score"])
    assert summary["target_score"].idxmax() == "Sex"


def test_sample(titanic: pd.DataFrame) -> None:
    summary = summarize(titanic, sample=200)
    assert summary["count"].max() == 200
    assert summary.attrs["dataset"]["rows"] == 200
    assert "200 of 891" in summary.attrs["dataset"]["sample"]
    # the same sample on every run
    pd.testing.assert_frame_equal(summary, summarize(titanic, sample=200))


def test_empty_inputs() -> None:
    with pytest.raises(ValueError, match="no columns"):
        summarize(pd.DataFrame())
    no_rows = summarize(pd.DataFrame({"a": pd.Series([], dtype=float)}))
    assert no_rows.loc["a", "count"] == 0
    assert no_rows.loc["a", "insights"] == ""
