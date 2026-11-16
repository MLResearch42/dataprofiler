"""The sample= option of the EDA functions."""

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...datasets import load_dataset
from ...eda import (
    create_diff_report,
    create_report,
    plot,
    plot_correlation,
    plot_diff,
    plot_missing,
)
from ...eda.eda_frame import sample_rows

NOTE = "Based on a random sample"


@pytest.fixture(scope="module")
def titanic() -> pd.DataFrame:
    return load_dataset("titanic")


def test_sample_rows(titanic: pd.DataFrame) -> None:
    sampled, note = sample_rows(titanic, 100)
    assert len(sampled) == 100
    assert note == (
        "Based on a random sample of 100 of 891 rows; counts and statistics describe the sample."
    )
    # rows keep their order, and the sample is the same every time
    assert sampled.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(sampled, sample_rows(titanic, 100)[0])
    assert len(sample_rows(titanic, 0.25)[0]) == round(0.25 * 891)
    assert len(sample_rows(titanic, np.int64(10))[0]) == 10


def test_no_sample_keeps_the_input(titanic: pd.DataFrame) -> None:
    assert sample_rows(titanic, None) == (titanic, None)
    for everything in (891, 5000, 1.0):
        same, note = sample_rows(titanic, everything)
        assert same is titanic and note is None


@pytest.mark.parametrize(
    "bad, error",
    [
        (0, ValueError),
        (-5, ValueError),
        (0.0, ValueError),
        (1.5, ValueError),
        (True, TypeError),
        ("100", TypeError),
    ],
)
def test_invalid_sample(titanic: pd.DataFrame, bad: object, error: type) -> None:
    with pytest.raises(error):
        sample_rows(titanic, bad)  # type: ignore[arg-type]


def test_dask_is_sampled_before_loading(titanic: pd.DataFrame) -> None:
    ddf = dd.from_pandas(titanic, npartitions=4)
    sampled, note = sample_rows(ddf, 200)
    assert isinstance(sampled, pd.DataFrame)
    assert 120 < len(sampled) < 280  # Dask samples a fraction of each partition
    assert note is not None and "of 891 rows" in note


def test_plots_say_they_show_a_sample(titanic: pd.DataFrame) -> None:
    outputs = [
        plot(titanic, sample=300, progress=False),
        plot(titanic, "Age", sample=300, progress=False),
        plot(titanic, "Age", "Sex", sample=300, progress=False),
        plot(titanic, target="Survived", sample=300, progress=False),
        plot_correlation(titanic, sample=300, progress=False),
        plot_missing(titanic, sample=300, progress=False),
    ]
    for output in outputs:
        assert "Based on a random sample of 300 of 891 rows" in output.to_html()
    assert NOTE not in plot(titanic, "Age", progress=False).to_html()
    assert NOTE not in plot(titanic, "Age", sample=10_000, progress=False).to_html()


def test_sampled_statistics(titanic: pd.DataFrame) -> None:
    html = plot(titanic, "Age", sample=300, progress=False).to_html()
    sampled = sample_rows(titanic, 300)[0]
    missing = int(sampled["Age"].isna().sum())
    assert f"<th>Missing</th><td>{missing}</td>" in html.replace("\n", "").replace(" ", "")


def test_diff_samples_each_frame(titanic: pd.DataFrame) -> None:
    frames = [titanic[titanic.Survived == 1], titanic[titanic.Survived == 0]]
    html = plot_diff(frames, sample=100).to_html()
    assert "Based on random samples of the rows (100, 100 rows)" in html
    report = create_diff_report({"yes": frames[0], "no": frames[1]}, sample=100, progress=False)
    assert "random samples of the rows" in report.to_html()


def test_report(titanic: pd.DataFrame) -> None:
    report = create_report(titanic, sample=0.5, progress=False)
    assert "Based on a random sample of 446 of 891 rows" in report.to_html()
    assert NOTE not in create_report(titanic.head(100), progress=False).to_html()
