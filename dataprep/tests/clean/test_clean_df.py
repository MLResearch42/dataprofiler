"""
Tests of clean_df().
"""

import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...clean import clean_df


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["Abby", None, "NULL", "Bob", "Bob"],
            "Age": [12, np.nan, 40, 7, 7],
            "wEIGht": [32.5, np.nan, 1.5, 2.25, 2.25],
            "Email Address": ["abby@gmail.com", None, "x@y.org", "bob@a.io", "bob@a.io"],
            "Country of Birth": ["CA", None, "US", "France", "France"],
            "home lat": ["41.5", "-81.0", "N/A", "12.5", "12.5"],
            "big": [1, 2, 300000, 4, 4],
            "ip": ["1.2.3.4", "8.8.8.8", "::1", "nan", "nan"],
            "when": [datetime.date(2020, 1, 1), None] + [datetime.date(2020, 1, 3)] * 3,
        }
    )


def test_clean_df_default(capsys: pytest.CaptureFixture) -> None:
    df = make_df()
    before = df.copy()
    types, result = clean_df(df)

    assert types["semantic_data_type"].tolist() == [
        "string",
        "floating",
        "floating",
        "email",
        "country",
        "coordinate",
        "integer",
        "ip",
        "date",
    ]
    assert types["atomic_data_type"].tolist() == [
        "string",
        "floating",
        "floating",
        "string",
        "string",
        "string",
        "integer",
        "string",
        "date",
    ]
    assert list(types.index) == list(df.columns)

    assert list(result.columns) == [
        "name",
        "age",
        "w_ei_ght",
        "email_address",
        "country_of_birth",
        "home_lat",
        "big",
        "ip",
        "when",
    ]
    # null strings became missing values
    assert result["name"].isna().tolist() == [False, True, True, False, False]
    assert result["home_lat"].isna().tolist() == [False, False, True, False, False]
    assert result["ip"].isna().tolist() == [False, False, False, True, True]
    # numerical columns are downcast
    assert str(result["age"].dtype) == "Int8"
    assert str(result["w_ei_ght"].dtype) == "Float32"
    assert str(result["big"].dtype) == "Int32"
    assert result["age"].tolist()[0] == 12
    assert list(result.index) == [0, 1, 2, 3, 4]

    out = capsys.readouterr().out
    assert "['email', 'country', 'coordinate', 'ip']" in out
    assert "6 values cleaned (66.67%)" in out
    assert "Downcast Memory Report:" in out
    # the input DataFrame is not modified
    assert before.equals(df)


def test_clean_df_remove_and_duplicates(capsys: pytest.CaptureFixture) -> None:
    types, result = clean_df(
        make_df(), data_type_detection="atomic", standardize_missing_values="remove"
    )
    assert list(types.columns) == ["atomic_data_type"]
    assert len(result) == 1
    assert result["name"].tolist() == ["Abby"]
    assert "4 entries dropped (80.0%)" in capsys.readouterr().out

    _, result = clean_df(make_df(), remove_duplicate_entries=True, downcast_memory=False)
    assert len(result) == 4
    assert str(result["age"].dtype) == "Int64"
    assert "1 entries dropped (20.0%)" in capsys.readouterr().out


def test_clean_df_ignore_everything() -> None:
    df = make_df()
    before = df.copy()
    result = clean_df(
        df,
        clean_header=False,
        data_type_detection="none",
        standardize_missing_values="ignore",
        report=False,
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(df.columns)
    assert result["Name"].tolist()[2] == "NULL"
    assert str(result["Name"].dtype) == "category"
    assert str(result["Age"].dtype) == "float32"
    assert before.equals(df)

    result = clean_df(
        df,
        clean_header=False,
        data_type_detection="none",
        standardize_missing_values="ignore",
        downcast_memory=False,
        report=False,
    )
    assert before.equals(result)


def test_clean_df_edge_cases() -> None:
    # columns without any value, numeric column names, unhashable values
    df = pd.DataFrame(
        {0: [None, None], "empty": [np.nan, np.nan], "lists": [[1], [2]], "lat": [1.5, 2.5]}
    )
    types, result = clean_df(df, report=False)
    assert types.loc["lat", "semantic_data_type"] == "coordinate"
    assert len(result) == 2
    result = clean_df(
        df, data_type_detection="none", standardize_missing_values="ignore", report=False
    )
    assert result["lists"].tolist() == [[1], [2]]

    empty = clean_df(pd.DataFrame({"a": pd.Series([], dtype=object)}), report=False)[1]
    assert len(empty) == 0

    with pytest.raises(ValueError, match="data_type_detection"):
        clean_df(df, data_type_detection="magic")
    with pytest.raises(ValueError, match="standardize_missing_values"):
        clean_df(df, standardize_missing_values="magic")


def test_clean_df_dask_input() -> None:
    # without the column of Python dates, which Dask itself converts to strings
    df = make_df().drop(columns="when")
    types, result = clean_df(dd.from_pandas(df, npartitions=2), report=False)
    expected_types, expected = clean_df(df, report=False)
    assert types.equals(expected_types)
    assert isinstance(result, pd.DataFrame)
    assert result.astype(str).equals(expected.astype(str))
