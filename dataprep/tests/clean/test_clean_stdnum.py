"""
Smoke tests for every clean_<identifier> / validate_<identifier> pair that is
built on python-stdnum. The test data is in stdnum_cases.py.
"""

import doctest
import importlib
from typing import Any, Callable, Dict, List, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ... import clean as clean_pkg
from .stdnum_cases import CASES

NAMES = sorted(CASES)
# pandas DataFrame with Python objects, pandas DataFrame with the default string
# type of the installed pandas version, and Dask DataFrame
INPUT_KINDS = ["pandas-object", "pandas-default", "dask"]


def _functions(name: str) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    return getattr(clean_pkg, f"clean_{name}"), getattr(clean_pkg, f"validate_{name}")


def _make_df(name: str, kind: str) -> Any:
    """
    DataFrame with a valid number, an invalid value and the missing value
    markers, between two other columns. The index is not the default one.
    """
    case = CASES[name]
    values = [case["valid"], case["invalid"], None, np.nan, "NULL", ""]
    index = [10, 11, 12, 13, 14, 15]
    if kind == "pandas-object":
        return pd.DataFrame(
            {
                "id": pd.Series(range(6), index=index),
                "num": pd.Series(values, index=index, dtype=object),
                "other": pd.Series(["x"] * 6, index=index, dtype=object),
            }
        )
    df = pd.DataFrame({"id": range(6), "num": values, "other": ["x"] * 6}, index=index)
    if kind == "dask":
        return dd.from_pandas(df, npartitions=2)
    return df


def _values(column: pd.Series) -> List[Any]:
    """The values of a column as Python objects, with None for missing values."""
    return [None if pd.isna(val) is True else val for val in column.astype(object).tolist()]


def test_cases_cover_every_stdnum_function() -> None:
    """Every stdnum based function of the package has test data, and vice versa."""
    stdnum_functions = set()
    for public_name in clean_pkg.__all__:
        if not public_name.startswith("clean_"):
            continue
        module = importlib.import_module(f"{clean_pkg.__name__}.{public_name}")
        if hasattr(module, "_FORMATTERS"):
            stdnum_functions.add(public_name[len("clean_") :])
    assert stdnum_functions == set(CASES)
    assert len(CASES) == 235
    for name in CASES:
        assert f"validate_{name}" in clean_pkg.__all__


@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("name", NAMES)
def test_clean_output_formats(name: str, kind: str) -> None:
    """valid -> cleaned value, invalid and missing -> NaN, in every output format."""
    clean, _ = _functions(name)
    case = CASES[name]
    for output_format, expected in case["formats"].items():
        df = _make_df(name, kind)
        result = clean(df, "num", output_format=output_format, progress=False)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["id", "num", "other", "num_clean"]
        assert list(result.index) == [10, 11, 12, 13, 14, 15]
        cleaned = _values(result["num_clean"])
        assert cleaned[0] is not None
        assert cleaned[0] == expected
        assert cleaned[1:] == [None] * 5
        # the other columns, and the input DataFrame, are untouched
        assert _values(result["num"]) == [case["valid"], case["invalid"], None, None, "NULL", ""]
        assert _values(result["id"]) == list(range(6))
        if kind != "dask":
            assert list(df.columns) == ["id", "num", "other"]


@pytest.mark.parametrize("name", NAMES)
def test_clean_default_format_is_standard(name: str) -> None:
    clean, _ = _functions(name)
    result = clean(_make_df(name, "pandas-default"), "num", progress=False)
    assert _values(result["num_clean"])[0] == CASES[name]["formats"]["standard"]


@pytest.mark.parametrize("kind", ["pandas-default", "dask"])
@pytest.mark.parametrize("name", NAMES)
def test_clean_inplace_and_errors(name: str, kind: str) -> None:
    clean, _ = _functions(name)
    case = CASES[name]
    expected = case["formats"]["compact"]

    # inplace: the cleaned column takes the place of the original column
    result = clean(_make_df(name, kind), "num", output_format="compact", inplace=True, progress=False)
    assert list(result.columns) == ["id", "num_clean", "other"]
    assert _values(result["num_clean"]) == [expected, None, None, None, None, None]

    # errors="ignore": invalid values are kept, missing values stay missing
    result = clean(
        _make_df(name, kind), "num", output_format="compact", errors="ignore", progress=False
    )
    assert _values(result["num_clean"]) == [expected, case["invalid"], None, None, None, None]

    # errors="raise": invalid values raise, valid and missing values do not
    with pytest.raises(ValueError, match="Unable to parse value"):
        clean(_make_df(name, kind), "num", errors="raise", progress=False)
    only_valid = pd.DataFrame({"num": [case["valid"], None, np.nan]})
    if kind == "dask":
        only_valid = dd.from_pandas(only_valid, npartitions=1)
    result = clean(only_valid, "num", output_format="compact", errors="raise", progress=False)
    assert _values(result["num_clean"]) == [expected, None, None]


@pytest.mark.parametrize("name", NAMES)
def test_clean_invalid_arguments(name: str) -> None:
    clean, _ = _functions(name)
    df = _make_df(name, "pandas-default")
    with pytest.raises(ValueError, match="output_format"):
        clean(df, "num", output_format="no_such_format", progress=False)
    with pytest.raises(ValueError, match="errors"):
        clean(df, "num", errors="no_such_policy", progress=False)


@pytest.mark.parametrize("kind", ["pandas-default", "dask"])
@pytest.mark.parametrize("name", [name for name in NAMES if "split" in CASES[name]])
def test_clean_split(name: str, kind: str) -> None:
    clean, _ = _functions(name)
    case = CASES[name]
    components: Dict[str, Any] = case["split"]
    for output_format, expected in case["formats"].items():
        result = clean(
            _make_df(name, kind), "num", output_format=output_format, split=True, progress=False
        )
        assert list(result.columns) == ["id", "num", "other", "num_clean"] + list(components)
        assert _values(result["num_clean"]) == [expected, None, None, None, None, None]
        for column, component in components.items():
            assert _values(result[column]) == [component, None, None, None, None, None]

    result = clean(_make_df(name, kind), "num", split=True, inplace=True, progress=False)
    assert list(result.columns) == ["id", "num_clean", "other"] + list(components)


@pytest.mark.parametrize("name", NAMES)
def test_validate(name: str) -> None:
    _, validate = _functions(name)
    case = CASES[name]
    expected = [True, False, False, False, False, False]

    # single values
    assert validate(case["valid"]) is True
    assert validate(case["invalid"]) is False
    for missing in (None, np.nan, pd.NA, ""):
        assert validate(missing) is False

    # pandas Series of objects and of the default string type
    for kind in ("pandas-object", "pandas-default"):
        df = _make_df(name, kind)
        result = validate(df["num"])
        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert result.tolist() == expected
        assert list(result.index) == list(df.index)
        assert result.name == "num"

        # a column of a DataFrame
        assert validate(df, "num").tolist() == expected

        # every cell of a DataFrame
        frame = validate(df[["num", "other"]])
        assert isinstance(frame, pd.DataFrame)
        assert frame["num"].tolist() == expected
        assert frame["other"].tolist() == [False] * 6

    # Dask Series and DataFrame give lazy results
    ddf = _make_df(name, "dask")
    assert validate(ddf["num"]).compute().tolist() == expected
    assert validate(ddf, "num").compute().tolist() == expected
    assert validate(ddf[["num", "other"]]).compute()["num"].tolist() == expected


def _is_integer_like(name: str) -> bool:
    """True if the compact form of the valid example can be stored as an integer."""
    compact = CASES[name]["formats"]["compact"]
    return isinstance(compact, str) and compact.isdigit() and not compact.startswith("0")


@pytest.mark.parametrize("name", [name for name in NAMES if _is_integer_like(name)])
def test_numbers_stored_as_integers(name: str) -> None:
    """A number that consists of digits only is also cleaned when stored as an integer."""
    clean, validate = _functions(name)
    compact = CASES[name]["formats"]["compact"]
    df = pd.DataFrame({"num": [int(compact), 0]})
    result = clean(df, "num", output_format="compact", progress=False)
    assert _values(result["num_clean"])[0] == compact
    assert validate(df["num"]).tolist()[0] is True
    assert validate(int(compact)) is True


@pytest.mark.parametrize("name", NAMES)
def test_docstring_examples(name: str) -> None:
    """The examples in the docstrings show the actual output."""
    module = importlib.import_module(f"{clean_pkg.__name__}.clean_{name}")
    globs = {"pd": pd, "np": np, **vars(module)}
    finder = doctest.DocTestFinder(recurse=False)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    n_examples = 0
    for func in _functions(name):
        for test in finder.find(func, name=func.__name__, globs=globs):
            n_examples += len(test.examples)
            runner.run(test, out=lambda text: None)
    assert n_examples >= 6
    assert runner.summarize(verbose=False).failed == 0
