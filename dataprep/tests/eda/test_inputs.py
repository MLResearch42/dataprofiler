"""EDA accepts many kinds of tabular input and normalises it without modifying it."""

import numpy as np
import pandas as pd
import pytest

from ...eda import plot, set_num_workers
from ...eda.eda_frame import EDAFrame, to_pandas
from ...eda.parallel import get_num_workers, pmap


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "num": rng.normal(size=100),
            "cat": rng.choice(list("abc"), 100),
            "dt": pd.date_range("2021-01-01", periods=100),
        }
    )


def test_user_frame_is_not_modified(df: pd.DataFrame) -> None:
    before = df.copy()
    df.columns = ["num", "cat", "dt"]
    plot(df, progress=False)
    plot(df, "cat", progress=False)
    pd.testing.assert_frame_equal(df, before)


def test_pyarrow_table(df: pd.DataFrame) -> None:
    pa = pytest.importorskip("pyarrow")
    plot(pa.Table.from_pandas(df), progress=False)


def test_dask_dataframe(df: pd.DataFrame) -> None:
    dd = pytest.importorskip("dask.dataframe")
    ddf = dd.from_pandas(df, npartitions=3)
    # dask may store strings with a different (Arrow) string dtype
    pd.testing.assert_frame_equal(to_pandas(ddf).reset_index(drop=True), df, check_dtype=False)
    plot(ddf, progress=False)


def test_series_and_dict_inputs(df: pd.DataFrame) -> None:
    plot(df["num"], progress=False)
    assert to_pandas({"a": [1, 2]}).shape == (2, 1)
    with pytest.raises(TypeError, match="Unsupported data type"):
        to_pandas(42)


def test_column_names_are_normalised() -> None:
    frame = EDAFrame(pd.DataFrame([[1, 2, 3]], columns=["a", "a", 7]))
    assert list(frame.columns) == ["a_1", "a_2", "7"]


def test_nullable_and_timezone_dtypes() -> None:
    frame = EDAFrame(
        pd.DataFrame(
            {
                "ints": pd.array([1, None, 3] * 5, dtype="Int64"),
                "tz": pd.date_range("2021", periods=15, tz="Europe/Paris"),
                "td": pd.to_timedelta(np.arange(15), unit="h"),
            }
        )
    )
    assert frame.frame["ints"].dtype == np.float64
    assert frame.frame["tz"].dt.tz is None
    assert frame.frame["td"].dtype == np.float64  # seconds


def test_unknown_column_message(df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="not found"):
        plot(df, "nope", progress=False)


def test_pmap_keeps_order_and_raises() -> None:
    assert pmap(lambda x: x * 2, range(50)) == [x * 2 for x in range(50)]

    def boom(x: int) -> int:
        if x == 7:
            raise RuntimeError("boom")
        return x

    with pytest.raises(RuntimeError, match="boom"):
        pmap(boom, range(20))


def test_num_workers_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        set_num_workers(1)
        assert get_num_workers() == 1
        set_num_workers(None)
        monkeypatch.setenv("DATAPREP_NUM_WORKERS", "3")
        assert get_num_workers() == 3
        with pytest.raises(ValueError):
            set_num_workers(0)
    finally:
        set_num_workers(None)
