"""rank_features(df, target) and plot(df, target=...)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ...datasets import load_dataset
from ...eda import Container, plot, rank_features
from ...eda.target import RANKING_COLUMNS, correlation_ratio, cramers_v


@pytest.fixture(scope="module")
def titanic() -> pd.DataFrame:
    return load_dataset("titanic")


@pytest.fixture
def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    group = rng.choice(list("abcd"), n)
    shift = pd.Series(group).map({"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}).to_numpy()
    label = np.where(x + rng.normal(scale=0.3, size=n) > 0, "yes", "no")
    return pd.DataFrame(
        {
            "x": x,
            "noise": rng.normal(size=n),
            "group": group,
            "row_id": [f"id{i}" for i in range(n)],
            "when": pd.date_range("2020-01-01", periods=n, freq="h"),
            "y": 2 * x + shift + rng.normal(scale=0.5, size=n),
            "label": label,
            # the label under another name, e.g. a column filled in after the outcome
            "leak": np.where(label == "yes", "Y", "N"),
        }
    )


def test_correlation_ratio() -> None:
    groups = np.array(list("aabbcc") * 50)
    values = pd.Series(groups).map({"a": 1.0, "b": 5.0, "c": 9.0}).to_numpy()
    assert correlation_ratio(values, groups) == pytest.approx(1.0)
    assert correlation_ratio(np.ones(300), groups) == 0.0
    rng = np.random.default_rng(0)
    assert correlation_ratio(rng.normal(size=300), groups) < 0.15
    # one group per row: nothing left to estimate the noise with
    assert np.isnan(correlation_ratio(np.arange(10.0), np.arange(10)))
    # without the bias correction it is the classic eta: sqrt(SS_between / SS_total)
    big = rng.normal(size=20000)
    big_groups = rng.choice(list("ab"), 20000)
    big = big + (big_groups == "a")
    means = pd.Series(big).groupby(big_groups).transform("mean").to_numpy()
    eta = np.sqrt(((means - big.mean()) ** 2).sum() / ((big - big.mean()) ** 2).sum())
    assert correlation_ratio(big, big_groups) == pytest.approx(eta, abs=1e-3)


def test_cramers_v() -> None:
    rng = np.random.default_rng(0)
    a = rng.choice(list("abc"), 5000)
    assert cramers_v(a, a)[0] == pytest.approx(1.0)
    assert cramers_v(a, rng.choice(list("xy"), 5000))[0] < 0.05
    # close to the uncorrected statistic when there are many rows
    b = np.where(rng.random(5000) < 0.7, a, rng.choice(list("abc"), 5000))
    table = pd.crosstab(a, b).to_numpy()
    expected = stats.contingency.association(table, method="cramer")
    assert cramers_v(a, b)[0] == pytest.approx(expected, abs=0.01)
    # many categories are grouped
    many = np.array([f"c{i}" for i in rng.integers(0, 500, 5000)])
    score, grouped = cramers_v(many, a)
    assert grouped and 0 <= score < 0.1


def test_ranking_table(synthetic: pd.DataFrame) -> None:
    ranking = rank_features(synthetic, "y")
    assert list(ranking.columns) == RANKING_COLUMNS
    assert ranking.index.name == "column"
    assert "y" not in ranking.index
    assert ranking.index[0] == "x"
    assert ranking["score"].dropna().is_monotonic_decreasing
    assert ranking.attrs == {"target": "y", "target_type": "numerical"}
    row = ranking.loc["x"]
    rho = stats.spearmanr(synthetic["x"], synthetic["y"]).statistic
    assert row["method"] == "Spearman"
    assert row["correlation"] == pytest.approx(rho)
    assert row["score"] == pytest.approx(abs(rho))
    assert ranking.loc["group", "method"] == "Correlation ratio"
    assert ranking.loc["group", "score"] > 0.3
    assert ranking.loc["noise", "score"] < 0.1
    assert "identifier" in ranking.loc["row_id", "note"]
    assert ranking.loc["row_id", "score"] < 0.1 or np.isnan(ranking.loc["row_id", "score"])
    assert ranking.loc["when", "type"] == "DateTime"
    assert ranking.loc["when", "method"] == "Spearman"


def test_categorical_target(synthetic: pd.DataFrame) -> None:
    ranking = rank_features(synthetic, "label")
    assert ranking.attrs["target_type"] == "categorical"
    assert ranking.loc["x", "method"] == "Correlation ratio"
    assert ranking.loc["group", "method"] == "Cramér's V"
    assert ranking.index[0] == "leak"
    assert "leakage" in ranking.loc["leak", "note"]
    assert ranking.loc["leak", "score"] > 0.9


def test_titanic(titanic: pd.DataFrame) -> None:
    ranking = rank_features(titanic, "Survived")
    assert ranking.index[0] == "Sex"
    assert ranking.loc["Age", "n"] == titanic["Age"].notna().sum()
    assert ranking.loc["Embarked", "n"] == 889
    # a numerical target can be treated as classes
    as_classes = rank_features(titanic, "Fare", dtype={"Fare": "nominal"})
    assert as_classes.attrs["target_type"] == "categorical"


def test_missing_values_and_constants() -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 11.0],
            "const": 3,
            "empty": np.nan,
        }
    )
    ranking = rank_features(df, "y")
    assert ranking.loc["x", "n"] == 10
    assert ranking.loc["const", "score"] == 0.0
    assert ranking.loc["const", "note"] == "constant column"
    assert np.isnan(ranking.loc["empty", "score"])
    assert ranking.index[-1] == "empty"  # undefined scores come last


def test_errors(synthetic: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="not found"):
        rank_features(synthetic, "nope")
    with pytest.raises(ValueError, match="two distinct values"):
        rank_features(synthetic.assign(c=1), "c")
    with pytest.raises(ValueError, match="no columns other than the target"):
        rank_features(synthetic[["y"]], "y")
    with pytest.raises(ValueError, match="at most one other column"):
        plot(synthetic, "x", "noise", target="y")


def test_integer_column_names() -> None:
    df = pd.DataFrame({0: np.arange(50.0), 1: np.arange(50.0) ** 2})
    assert rank_features(df, 1).loc["0", "score"] == pytest.approx(1.0)


def test_plot_target(titanic: pd.DataFrame) -> None:
    result = plot(titanic, target="Survived", progress=False)
    assert isinstance(result, Container)
    assert result.visual_type == "target_ranking"
    html = result.to_html()
    assert "Feature Ranking" in html and "Target Distribution" in html
    assert "Association with Survived" in html
    # notes are shown in the insight panel, escaped
    assert "identifier?" in html
    reg = plot(load_dataset("house_prices_train"), target="SalePrice", progress=False)
    assert "top 50 of 80 columns" in reg.to_html()


def test_plot_one_column_with_target(titanic: pd.DataFrame) -> None:
    with_target = plot(titanic, "Age", target="Survived", progress=False)
    plain = plot(titanic, "Age", "Survived", progress=False)
    assert with_target.visual_type == plain.visual_type


def test_html_in_column_names_is_escaped() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "<b>id</b>": [f"x{i}" for i in range(100)],
            "y": rng.normal(size=100),
        }
    )
    html = plot(df, target="y", progress=False).to_html()
    assert "<b>id</b>:" not in html
    assert "&lt;b&gt;id&lt;/b&gt;" in html
