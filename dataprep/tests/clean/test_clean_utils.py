"""
Tests of the helpers shared by the clean functions.
"""

import datetime
from collections import Counter
from decimal import Decimal

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...clean import utils


def test_is_null_and_is_missing() -> None:
    missing = [None, np.nan, float("nan"), np.float32("nan"), np.float64("nan"), pd.NA, pd.NaT]
    missing += [np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns")]
    for val in missing:
        assert utils.is_missing(val) is True
        assert utils.is_null(val) is True
    for val in ["", "NULL", "nan", "NaN", "N/A", "<NA>", "null"]:
        assert utils.is_missing(val) is False
        assert utils.is_null(val) is True
    for val in ["a", " ", "none", 0, 0.0, False, [], {}, (np.nan,), b"", Decimal("1")]:
        assert utils.is_null(val) is False
    assert utils.is_null(pd.Timestamp("2020-01-01")) is False
    assert utils.is_null(datetime.date(2020, 1, 1)) is False
    # every string of NULL_VALUES is a null string
    assert {val for val in utils.NULL_VALUES if isinstance(val, str)} == set(utils.NULL_STRINGS)


def test_in_null_values() -> None:
    assert utils.in_null_values(float("nan"), [np.nan]) is True
    assert utils.in_null_values(None, [np.nan]) is True
    assert utils.in_null_values(np.nan, ["?"]) is False
    assert utils.in_null_values("?", ["?", None]) is True
    assert utils.in_null_values("a", ["?", None]) is False
    assert utils.in_null_values("a", None) is False


def test_is_dask_and_check_dataframe() -> None:
    df = pd.DataFrame({"a": [1]})
    ddf = dd.from_pandas(df, npartitions=1)
    assert utils.is_dask(ddf) and utils.is_dask(ddf["a"])
    assert not utils.is_dask(df) and not utils.is_dask(None)
    utils.check_dataframe(df)
    utils.check_dataframe(ddf)
    with pytest.raises(ValueError, match="pandas or Dask DataFrame"):
        utils.check_dataframe(df["a"])
    assert utils.to_dask(df).npartitions == 1
    assert utils.to_dask(ddf) is ddf


def test_clean_column_keeps_rows_aligned() -> None:
    def formatter(val: object) -> tuple:
        return (f"<{val}>", len(str(val)), 2)

    # duplicated and unordered index, several partitions
    df = pd.DataFrame({"col": ["a", "bb", "ccc", "dddd", "e"]}, index=[3, 3, 1, 0, 7])
    expected = (["<a>", "<bb>", "<ccc>", "<dddd>", "<e>"], [1, 2, 3, 4, 1])
    for frame in (df, dd.from_pandas(df.reset_index(drop=True), npartitions=3)):
        result, stats = utils.clean_column(frame, "col", formatter, ["text", "size"])
        assert list(result.columns) == ["col", "text", "size"]
        assert (result["text"].tolist(), result["size"].tolist()) == expected
        assert stats == Counter({2: 5})
    result, _ = utils.clean_column(df, "col", formatter, ["text", "size"])
    assert list(result.index) == [3, 3, 1, 0, 7]
    result, _ = utils.clean_column(df, "col", formatter, ["text", "size"], inplace=True)
    assert list(result.columns) == ["text", "size"]
    assert list(df.columns) == ["col"]


def test_add_columns() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [0.5, 1.5]})
    result = utils.add_columns(df, {"b_clean": ["X", np.nan], "extra": [1, 2]}, replace="b")
    assert list(result.columns) == ["a", "b_clean", "c", "extra"]
    result = utils.add_columns(df, {"c": [9, 9]}, drop="a")
    assert list(result.columns) == ["b", "c"]
    assert result["c"].tolist() == [9, 9]
    # strings get the default string type of the installed pandas version
    result = utils.add_columns(df, {"new": ["p", np.nan]})
    assert result["new"].dtype == pd.Series(["p", np.nan]).dtype
    assert list(df.columns) == ["a", "b", "c"]
    assert utils.unzip([], 3) == [(), (), ()]
    assert utils.unzip([(1, "a"), (2, "b")], 2) == [(1, 2), ("a", "b")]


def test_validate_values() -> None:
    def is_a(val: object) -> bool:
        return isinstance(val, str) and val == "a"

    assert utils.validate_values("a", is_a) is True
    srs = pd.Series(["a", "b", None], index=[5, 6, 7], name="letters")
    result = utils.validate_values(srs, is_a)
    assert result.tolist() == [True, False, False]
    assert result.dtype == bool and result.name == "letters" and list(result.index) == [5, 6, 7]
    lazy = utils.validate_values(dd.from_pandas(srs, npartitions=2), is_a)
    assert utils.is_dask(lazy)
    assert lazy.compute().tolist() == [True, False, False]
    frame = utils.validate_values(srs.to_frame(), is_a, tabular=True)
    assert frame["letters"].tolist() == [True, False, False]
    assert utils.validate_values(pd.Series([], dtype=object), is_a).dtype == bool


def test_reports(capsys: pytest.CaptureFixture) -> None:
    utils.create_report_new("Thing", Counter({0: 1, 1: 1, 2: 1, 3: 1}), "coerce")
    out = capsys.readouterr().out
    assert out == (
        "Thing Cleaning Report:\n"
        "\t1 values cleaned (25.0%)\n"
        "\t1 values unable to be parsed (25.0%), set to NaN\n"
        "Result contains 2 (50.0%) values in the correct format and 2 null values (50.0%)\n"
    )
    utils.create_report_new("Thing", pd.Series({1: 2}), "ignore")
    assert "2 values unable to be parsed (100.0%), left unchanged" in capsys.readouterr().out
    # an empty column does not divide by zero
    utils.create_report_new("Thing", {}, "coerce")
    assert "0 (0.0%) values in the correct format" in capsys.readouterr().out
