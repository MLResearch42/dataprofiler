"""Tests for the EDA type system (dataprep.eda.dtypes)."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ...eda.dtypes import (
    Continuous,
    DateTime,
    GeoGraphy,
    GeoPoint,
    Nominal,
    SmallCardNum,
    detect_dtype,
    is_continuous,
    is_datetime,
    is_nominal,
    normalize_dtype,
)
from ...eda.eda_frame import EDAFrame


def test_nominal_pandas_dtypes() -> None:
    df = pd.DataFrame({"S": ["a"], "C": pd.Categorical(["c"]), "B": [False]})
    for col in df.columns:
        assert is_nominal(df[col].dtype), col
        assert not is_continuous(df[col].dtype), col


@pytest.mark.parametrize(
    "values",
    [[1, 2], [1.5, 2.5], pd.array([1, None], dtype="Int64"), [timedelta(1), timedelta(2)]],
)
def test_continuous_pandas_dtypes(values: object) -> None:
    assert is_continuous(pd.Series(values).dtype)


def test_datetimes_are_not_continuous() -> None:
    srs = pd.Series([datetime.now()])
    assert is_datetime(srs.dtype)
    assert not is_continuous(srs.dtype)
    tz = srs.dt.tz_localize("UTC")
    assert is_datetime(tz.dtype)


def test_detect_semantic_types() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=50),
            "small": rng.integers(0, 3, 50),
            "cat": rng.choice(list("xyz"), 50),
            "dt": pd.date_range("2020-01-01", periods=50),
            "country": ["Canada", "France", "Japan", "Brazil", "Kenya"] * 10,
            "point": ["49.2, -123.1", "40.7, -74.0"] * 25,
        }
    )
    frame = EDAFrame(df)
    expected = {
        "num": Continuous,
        "small": SmallCardNum,
        "cat": Nominal,
        "dt": DateTime,
        "country": GeoGraphy,
        "point": GeoPoint,
    }
    for col, cls in expected.items():
        assert isinstance(frame.get_eda_dtype(col), cls), col


def test_decimal_comma_numbers_are_not_geopoints() -> None:
    srs = pd.Series(["12,5", "3,25", "100,0"] * 10, name="x")
    assert isinstance(detect_dtype(srs), Nominal)


@pytest.mark.parametrize("spec", ["nominal", Nominal, Nominal()])
def test_user_override(spec: object) -> None:
    srs = pd.Series([1, 2, 3] * 10, name="zip")
    assert isinstance(detect_dtype(srs, {"zip": spec}), Nominal)
    assert isinstance(normalize_dtype(spec), Nominal)


def test_unknown_override_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dtype"):
        normalize_dtype("not-a-type")
