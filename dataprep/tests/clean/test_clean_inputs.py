"""
Smoke tests of the clean_* / validate_* functions that are not built on
python-stdnum: every function must work on pandas DataFrames (with Python
objects, with the default string type of the installed pandas version and with
the nullable "string" type) and on Dask DataFrames, with missing values present.
"""

from typing import Any, Dict, List

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ... import clean as clean_pkg
from .stdnum_cases import CASES

INPUT_KINDS = ["pandas-object", "pandas-default", "pandas-string", "dask"]

# name -> a valid value, the cleaned value with the default parameters, an invalid value
SIMPLE: Dict[str, Dict[str, Any]] = {
    "email": {"valid": "Abc@Example.com", "cleaned": "abc@example.com", "invalid": "not an email"},
    "phone": {"valid": "(555) 234-5678", "cleaned": "555-234-5678", "invalid": "12345"},
    "country": {"valid": "USA", "cleaned": "United States", "invalid": "zzzz"},
    "date": {"valid": "2003/09/25", "cleaned": "2003-09-25 00:00:00", "invalid": "This is Sep."},
    "ip": {"valid": "233.5.6.0", "cleaned": "233.5.6.0", "invalid": "0xabc"},
    "address": {"valid": "123 Pine Ave.", "cleaned": "123 Pine Ave.", "invalid": "hello"},
    "currency": {"valid": "1,234.5", "cleaned": 1234.5, "invalid": "abc"},
    "lat_long": {"valid": "41.5, -81.0", "cleaned": (41.5, -81.0), "invalid": "hello"},
}
MISSING = [None, np.nan, "NULL", ""]

# values of all kinds that a column may contain
ODD_VALUES = [1, 2.5, None, pd.NA, np.nan, ("a", 1), {"k": 1}, b"x", pd.Timestamp("2020-01-01"), True]


def make_df(values: List[Any], kind: str) -> Any:
    """DataFrame with the values in the column "col", next to another column."""
    index = list(range(100, 100 + len(values)))
    if kind == "pandas-object":
        return pd.DataFrame(
            {
                "col": pd.Series(values, index=index, dtype=object),
                "other": pd.Series(range(len(values)), index=index),
            }
        )
    df = pd.DataFrame({"col": values, "other": range(len(values))}, index=index)
    if kind == "pandas-string":
        df["col"] = df["col"].astype("string")
    if kind == "dask":
        return dd.from_pandas(df, npartitions=2)
    return df


def values_of(column: pd.Series) -> List[Any]:
    """The values of a column as Python objects, with None for missing values."""
    return [None if pd.isna(val) is True else val for val in column.astype(object).tolist()]


def test_every_public_function_is_smoke_tested() -> None:
    """Fail when a function is added to the package without adding it to the smoke tests."""
    covered = {"clean_headers", "clean_text", "default_text_pipeline", "clean_df", "clean_ml"}
    covered |= {"clean_json", "validate_json", "clean_url", "validate_url"}  # tested below
    covered |= {"clean_duplication", "clean_df_gui"}  # test_clean_duplication.py, test_clean_gui.py
    for name in list(SIMPLE) + list(CASES):  # CASES: test_clean_stdnum.py
        covered |= {f"clean_{name}", f"validate_{name}"}
    assert covered == set(clean_pkg.__all__)
    assert len(clean_pkg.__all__) == len(set(clean_pkg.__all__))


@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("name", sorted(SIMPLE))
def test_clean(name: str, kind: str, capsys: pytest.CaptureFixture) -> None:
    case = SIMPLE[name]
    clean = getattr(clean_pkg, f"clean_{name}")
    values = [case["valid"], case["invalid"]] + MISSING
    n_missing = len(MISSING)

    df = make_df(values, kind)
    result = clean(df, "col", progress=False)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["col", "other", "col_clean"]
    assert list(result.index) == list(range(100, 100 + len(values)))
    assert values_of(result["col_clean"]) == [case["cleaned"], None] + [None] * n_missing
    assert values_of(result["other"]) == list(range(len(values)))
    if kind != "dask":
        assert list(df.columns) == ["col", "other"]

    # the report counts the cleaned, the unparsable and the null values
    report = capsys.readouterr().out
    assert "Cleaning Report" in report
    assert "1 values unable to be parsed" in report
    assert f"and {n_missing + 1} null values" in report

    # no report on request
    result = clean(make_df(values, kind), "col", report=False, progress=False, inplace=True)
    assert capsys.readouterr().out == ""
    assert list(result.columns) == ["other", "col_clean"]

    # errors="ignore" keeps the invalid values, errors="raise" raises for them
    result = clean(make_df(values, kind), "col", errors="ignore", report=False, progress=False)
    ignored = values_of(result["col_clean"])
    assert ignored[0] == case["cleaned"]
    # clean_address returns the invalid value with its basic formatting applied
    assert ignored[1] == ("Hello" if name == "address" else case["invalid"])
    # the missing value markers stay missing. clean_email returns null strings as they are
    assert ignored[2:4] == [None, None]
    assert ignored[4:] == (["NULL", ""] if name == "email" else [None, None])
    with pytest.raises(ValueError):
        clean(make_df(values, kind), "col", errors="raise", report=False, progress=False)
    result = clean(
        make_df([case["valid"], None, np.nan], kind),
        "col",
        errors="raise",
        report=False,
        progress=False,
    )
    assert values_of(result["col_clean"]) == [case["cleaned"], None, None]
    with pytest.raises(ValueError, match="errors"):
        clean(make_df(values, kind), "col", errors="no_such_policy")


@pytest.mark.parametrize("name", sorted(SIMPLE))
def test_clean_empty_dataframe(name: str) -> None:
    clean = getattr(clean_pkg, f"clean_{name}")
    result = clean(pd.DataFrame({"col": pd.Series([], dtype=object)}), "col", progress=False)
    assert list(result.columns) == ["col", "col_clean"]
    assert len(result) == 0


@pytest.mark.parametrize("name", sorted(SIMPLE))
def test_clean_progress_bar_with_dask(name: str) -> None:
    """The progress bar (only used for Dask input) does not break the computation."""
    case = SIMPLE[name]
    clean = getattr(clean_pkg, f"clean_{name}")
    result = clean(make_df([case["valid"], case["invalid"]], "dask"), "col", report=False)
    assert values_of(result["col_clean"]) == [case["cleaned"], None]


@pytest.mark.parametrize("name", sorted(SIMPLE) + ["url", "json"])
def test_validate(name: str) -> None:
    validate = getattr(clean_pkg, f"validate_{name}")
    case = {
        "url": {"valid": "https://www.sfu.ca/path?x=1", "invalid": "hello"},
        "json": {"valid": '{"a": [1, 2]}', "invalid": '{"a": 1,}'},
    }.get(name) or SIMPLE[name]
    values = [case["valid"], case["invalid"]] + MISSING
    expected = [True, False] + [False] * len(MISSING)

    assert validate(case["valid"]) is True
    assert validate(case["invalid"]) is False
    for missing in (None, np.nan, pd.NA, pd.NaT, ""):
        assert validate(missing) is False

    for kind in ("pandas-object", "pandas-default", "pandas-string"):
        column = make_df(values, kind)["col"]
        result = validate(column)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert result.tolist() == expected
        assert list(result.index) == list(column.index)

    # Dask Series give a lazy result
    assert validate(make_df(values, "dask")["col"]).compute().tolist() == expected

    # columns of any type are accepted, and give booleans
    for column in (
        pd.Series([1, 2, 3]),
        pd.Series([1.5, np.nan]),
        pd.Series([True, False]),
        pd.Series(pd.to_datetime(["2020-01-01", None])),
        pd.Series(["a", "b"], dtype="category"),
        pd.Series(ODD_VALUES, dtype=object),
        pd.Series([], dtype=object),
    ):
        result = validate(column)
        assert result.dtype == bool
        assert len(result) == len(column)
        assert not result[column.isna().to_numpy()].any()


@pytest.mark.parametrize("name", sorted(SIMPLE) + ["url"])
def test_clean_odd_values(name: str) -> None:
    """Values of any type are either cleaned or set to NaN, they never raise."""
    clean = getattr(clean_pkg, f"clean_{name}")
    df = pd.DataFrame({"col": pd.Series(ODD_VALUES, dtype=object)})
    result = clean(df, "col", report=False, progress=False)
    assert len(result) == len(ODD_VALUES)
    for position, val in enumerate(ODD_VALUES):
        if val is None or val is pd.NA or (isinstance(val, float) and np.isnan(val)):
            assert pd.isna(result.iloc[position, -1])


@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_clean_url(kind: str, capsys: pytest.CaptureFixture) -> None:
    values = ["https://www.sfu.ca/path?x=1&token=abc", "hello"] + MISSING
    result = clean_pkg.clean_url(make_df(values, kind), "col", remove_auth=True, progress=False)
    assert list(result.columns) == ["col", "other", "col_details"]
    assert values_of(result["col_details"]) == [
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "col_clean": "https://www.sfu.ca/path",
            "queries": {"x": "1"},
        },
        None,
    ] + [None] * len(MISSING)
    assert "Removed 1 auth queries from 1 rows" in capsys.readouterr().out

    result = clean_pkg.clean_url(
        make_df(values, kind), "col", split=True, inplace=True, report=False, progress=False
    )
    assert list(result.columns) == ["other", "scheme", "host", "col_clean", "queries"]
    assert values_of(result["host"]) == ["www.sfu.ca", None] + [None] * len(MISSING)
    assert values_of(result["queries"])[0] == {"x": "1", "token": "abc"}


def test_clean_url_remove_auth_list_is_not_remembered() -> None:
    """Extra auth parameters only apply to the call they are passed to."""
    df = pd.DataFrame({"col": ["https://www.sfu.ca/path?x=1&mykey=abc"]})
    result = clean_pkg.clean_url(df, "col", remove_auth=["mykey"], report=False)
    assert result["col_details"][0]["queries"] == {"x": "1"}
    result = clean_pkg.clean_url(df, "col", remove_auth=True, report=False)
    assert result["col_details"][0]["queries"] == {"x": "1", "mykey": "abc"}


@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_clean_json(kind: str) -> None:
    values = ['{"name": "jane", "address": {"city": "Burnaby"}}', '{"name": "x",}', "[1, 2]"]
    df = make_df(values + MISSING, kind)
    result = clean_pkg.clean_json(df, "col")
    assert list(result.columns) == ["col", "other", "name", "address.city"]
    assert values_of(result["col"]) == [
        {"name": "jane", "address": {"city": "Burnaby"}},
        None,
        [1, 2],
    ] + [None] * len(MISSING)
    assert values_of(result["name"]) == ["jane"] + [None] * (2 + len(MISSING))
    assert values_of(result["address.city"]) == ["Burnaby"] + [None] * (2 + len(MISSING))
    if kind != "dask":
        # the input DataFrame still holds the JSON strings
        assert df["col"].tolist()[0] == values[0]

    result = clean_pkg.clean_json(make_df(values, kind), "col", errors="ignore")
    assert values_of(result["col"])[1] == '{"name": "x",}'
    with pytest.raises(ValueError, match="Unable to parse value"):
        clean_pkg.clean_json(make_df(values, kind), "col", errors="raise")


def test_clean_json_key_with_the_name_of_a_column() -> None:
    df = pd.DataFrame({"col": ['{"other": 5}'], "other": ["keep"]})
    result = clean_pkg.clean_json(df, "col")
    assert list(result.columns) == ["col", "other", "col_other"]
    assert result["other"].tolist() == ["keep"]
    assert result["col_other"].tolist() == [5]


@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_clean_lat_long_columns(kind: str) -> None:
    df = pd.DataFrame({"lat": ["41° 30′ N", "x", None], "lon": ["81° W", "20", np.nan]})
    if kind == "pandas-object":
        df = df.astype(object)
    elif kind == "pandas-string":
        df = df.astype("string")
    elif kind == "dask":
        df = dd.from_pandas(df, npartitions=2)

    result = clean_pkg.clean_lat_long(df, lat_col="lat", long_col="lon", report=False)
    assert list(result.columns) == ["lat", "lon", "latitude_longitude"]
    assert result["latitude_longitude"].tolist()[0] == (41.5, -81.0)

    result = clean_pkg.clean_lat_long(
        df, lat_col="lat", long_col="lon", split=True, output_format="ddh", report=False
    )
    assert list(result.columns) == ["lat", "lon", "lat_clean", "lon_clean"]
    assert values_of(result["lat_clean"]) == ["41.5° N", None, None]
    assert values_of(result["lon_clean"]) == ["81.0° W", "20.0° E", None]

    result = clean_pkg.clean_lat_long(
        df, lat_col="lat", long_col="lon", output_format="dm", inplace=True, report=False
    )
    assert list(result.columns) == ["latitude_longitude"]
    assert values_of(result["latitude_longitude"]) == ["41° 30′ N, 81° 0′ W", None, None]

    result = clean_pkg.clean_lat_long(df, lat_col="lat", report=False, progress=False)
    assert list(result.columns) == ["lat", "lon", "lat_clean"]
    assert values_of(result["lat_clean"]) == [41.5, None, None]

    with pytest.raises(ValueError):
        clean_pkg.clean_lat_long(df)


@pytest.mark.parametrize("kind", ["pandas-default", "dask"])
def test_clean_split(kind: str) -> None:
    df = make_df(["Abc@Example.com", "bad", None], kind)
    result = clean_pkg.clean_email(df, "col", split=True, report=False, progress=False)
    assert list(result.columns) == ["col", "other", "username", "domain"]
    assert values_of(result["domain"]) == ["example.com", None, None]

    df = make_df(["+1 (555) 234-5678 ext. 12", "bad", None], kind)
    result = clean_pkg.clean_phone(df, "col", split=True, report=False, progress=False)
    assert list(result.columns) == [
        "col",
        "other",
        "country_code",
        "area_code",
        "office_code",
        "station_code",
        "ext_num",
    ]
    assert result.iloc[0, 2:].tolist() == ["1", "555", "234", "5678", "12"]
    assert clean_pkg.clean_phone(df, "col", output_format="e164", report=False)[
        "col_clean"
    ].tolist()[0] == ("+15552345678 ext. 12")

    df = make_df(["123 Pine Ave., Vancouver, WA 98660", "hello", None], kind)
    result = clean_pkg.clean_address(
        df, "col", output_format="house_number street_name city", split=True, report=False
    )
    assert list(result.columns) == ["col", "other", "house_number", "street_name", "city"]
    assert result.iloc[0, 2:].tolist() == ["123", "Pine", "Vancouver"]

    df = make_df(["$1,200.50", "bad", None], kind)
    result = clean_pkg.clean_currency(
        df, "col", ignore_symbol="$", split=True, target_currency="eur", user_rate=2, report=False
    )
    assert list(result.columns) == [
        "col",
        "other",
        "input_currency_symbol",
        "col_clean",
        "conversion_rate",
        "target_symbol",
        "target_val",
    ]
    assert result.iloc[0, 2:].tolist() == ["US$", 1200.5, 2, "€", 2401.0]
    assert result.iloc[1:, 2:].isna().all().all()


def test_validate_country_two_letter_codes() -> None:
    """"NO" and "NE" are Norway and Niger, not the missing value "none"."""
    assert clean_pkg.validate_country("NO") is True
    assert clean_pkg.validate_country("ne") is True
    assert clean_pkg.validate_country("NA") is True  # Namibia
    assert clean_pkg.validate_country("none") is False
    df = pd.DataFrame({"col": ["NO", "NE", "NA", "None"]})
    result = clean_pkg.clean_country(df, "col", report=False)
    assert values_of(result["col_clean"]) == ["Norway", "Niger", "Namibia", None]


def test_clean_date_timezones_without_pytz() -> None:
    df = pd.DataFrame({"col": ["2003-09-25 10:36:28", "Thu Sep 25 10:36:28 2003", None]})
    result = clean_pkg.clean_date(
        df,
        "col",
        input_timezone="UTC",
        output_timezone="Asia/Tokyo",
        output_format="YYYY-MM-DD hh:mm:ss Z",
        report=False,
    )
    assert values_of(result["col_clean"]) == [
        "2003-09-25 19:36:28 UTC+09:00",
        "2003-09-25 19:36:28 UTC+09:00",
        None,
    ]
    with pytest.raises(ValueError, match="input_timezone"):
        clean_pkg.clean_date(df, "col", input_timezone="Mars/Olympus")
    with pytest.raises(ValueError, match="output_timezone"):
        clean_pkg.clean_date(df, "col", output_timezone="Mars/Olympus")


@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_clean_text(kind: str) -> None:
    values = ["<b>The</b> Zoë saw 3 cats at https://a.io !", "", None, np.nan, "NULL"]
    df = make_df(values, kind)
    result = clean_pkg.clean_text(df, "col")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["col", "other"]
    assert values_of(result["col"]) == ["zoe saw cats", None, None, None, None]
    if kind != "dask":
        assert df["col"].tolist()[0] == values[0]

    pipeline = [
        {"operator": "lowercase"},
        {"operator": "replace_digits", "parameters": {"value": "N"}},
        {"operator": lambda text: text if pd.isna(text) else text[:12]},
    ]
    result = clean_pkg.clean_text(make_df(values, kind), "col", pipeline=pipeline)
    assert values_of(result["col"])[:2] == ["<b>the</b> z", ""]
    assert all(val is None for val in values_of(result["col"])[2:4])

    result = clean_pkg.clean_text(make_df(values, kind), "col", stopwords={"zoe", "saw"})
    assert values_of(result["col"])[0] == "the cats at"
    assert [step["operator"] for step in clean_pkg.default_text_pipeline()][0] == "fillna"


@pytest.mark.parametrize("kind", ["pandas-default", "dask"])
def test_clean_headers(kind: str, capsys: pytest.CaptureFixture) -> None:
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["ISBN", "Zip Code", "Zip Code", "Año"])
    if kind == "dask":
        # Dask does not support duplicated column names
        df = dd.from_pandas(df.iloc[:, [0, 1, 3]], npartitions=1)
        result = clean_pkg.clean_headers(df)
        assert list(result.columns) == ["isbn", "zip_code", "ano"]
        assert result.compute().shape == (1, 3)
        return
    result = clean_pkg.clean_headers(df)
    assert list(result.columns) == ["isbn", "zip_code", "zip_code_1", "ano"]
    assert "4 values cleaned (100.0%)" in capsys.readouterr().out
    assert list(df.columns) == ["ISBN", "Zip Code", "Zip Code", "Año"]
    result = clean_pkg.clean_headers(df, case="const", remove_accents=False, report=False)
    assert list(result.columns) == ["ISBN", "ZIP_CODE", "ZIP_CODE_1", "AÑO"]
    assert capsys.readouterr().out == ""
