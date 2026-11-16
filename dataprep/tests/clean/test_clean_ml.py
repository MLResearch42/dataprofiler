"""
Tests of clean_ml(), of the Pipeline and of its components.
"""

from typing import Any, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...clean import clean_ml
from ...clean.components import component_dic
from ...clean.components.cat_encoding import operator_dic as cat_encoders
from ...clean.components.cat_imputation import operator_dic as cat_imputers
from ...clean.components.num_imputation import operator_dic as num_imputers
from ...clean.components.num_scaling import operator_dic as num_scalers
from ...clean.components.variance_threshold import operator_dic as thresholders
from ...clean.pipeline import Pipeline


def make_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "cat": ["a", "b", None, "a", "c"],
            "num": [1.0, 2.0, np.nan, 4.0, 10.0],
            "const": [5, 5, 5, 5, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )
    test = pd.DataFrame(
        {"cat": ["b", "a", "NULL"], "num": [3.0, np.nan, 0.0], "const": [5, 5, 5], "target": [1, 0, 1]}
    )
    return train, test


def test_default(tmp_path: Any) -> None:
    train, test = make_data()
    train_before, test_before = train.copy(), test.copy()
    clean_train, clean_test = clean_ml(train, test)

    # one-hot encoding of the categories of the training data: a, b, missing_value, c
    assert clean_train["cat"].tolist() == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert clean_test["cat"].tolist() == [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    # mean imputation (4.25) and standardization with the mean and std of the training data
    np.testing.assert_allclose(
        clean_train["num"], [-0.930949, -0.644503, 0.0, -0.071611, 1.647064], atol=1e-6
    )
    np.testing.assert_allclose(clean_test["num"], [-0.358057, 0.0, -1.217395], atol=1e-6)
    # a constant column is scaled to zeros
    assert clean_train["const"].tolist() == [0.0] * 5
    assert clean_test["const"].tolist() == [0.0] * 3
    # the target column is not transformed
    assert clean_train["target"].tolist() == [0, 1, 0, 1, 0]
    assert list(clean_train.columns) == ["cat", "num", "const", "target"]
    # the inputs are not modified
    assert train.equals(train_before) and test.equals(test_before)


def test_no_encoding_and_no_scaling() -> None:
    train, test = make_data()
    clean_train, clean_test = clean_ml(
        train, test, cat_encoding="no_encoding", num_scaling="no_scaling", fill_val="unknown"
    )
    assert clean_train["cat"].tolist() == ["a", "b", "unknown", "a", "c"]
    assert clean_test["cat"].tolist() == ["b", "a", "unknown"]
    assert clean_train["num"].tolist() == [1.0, 2.0, 4.25, 4.0, 10.0]
    assert clean_test["num"].tolist() == [3.0, 4.25, 0.0]
    assert clean_train["const"].tolist() == [5] * 5


def test_imputers_and_scalers() -> None:
    train, test = make_data()
    clean_train, clean_test = clean_ml(
        train,
        test,
        cat_imputation="most_frequent",
        cat_encoding="no_encoding",
        num_imputation="median",
        num_scaling="minmax",
    )
    assert clean_train["cat"].tolist() == ["a", "b", "a", "a", "c"]
    assert clean_test["cat"].tolist() == ["b", "a", "a"]
    # median 3.0, then (x - 1) / (10 - 1)
    np.testing.assert_allclose(clean_train["num"], [0, 1 / 9, 2 / 9, 3 / 9, 1])
    np.testing.assert_allclose(clean_test["num"], [2 / 9, 2 / 9, -1 / 9])

    clean_train, clean_test = clean_ml(
        train, test, num_imputation="most_frequent", num_scaling="maxabs", cat_encoding="no_encoding"
    )
    np.testing.assert_allclose(clean_train["num"], [0.1, 0.2, 0.1, 0.4, 1.0])
    np.testing.assert_allclose(clean_test["num"], [0.3, 0.1, 0.0])


def test_drop_and_variance_threshold() -> None:
    train, test = make_data()
    clean_train, clean_test = clean_ml(train, test, cat_imputation="drop", num_imputation="drop")
    # columns with missing values in the training data are dropped
    assert list(clean_train.columns) == ["const", "target"]
    assert list(clean_test.columns) == ["const", "target"]

    clean_train, clean_test = clean_ml(
        train, test, variance_threshold=True, variance=0.0, num_scaling="no_scaling"
    )
    # the constant column has no variance
    assert list(clean_train.columns) == ["cat", "num", "target"]
    assert list(clean_test.columns) == ["cat", "num", "target"]


def test_include_and_exclude_operators() -> None:
    train, test = make_data()
    clean_train, clean_test = clean_ml(train, test, exclude_operators=["one_hot"])
    assert clean_train["cat"].tolist() == ["a", "b", "missing_value", "a", "c"]
    np.testing.assert_allclose(clean_test["num"], [-0.358057, 0.0, -1.217395], atol=1e-6)

    clean_train, clean_test = clean_ml(train, test, include_operators=["one_hot", "standardize"])
    # no imputation: the missing value is a category, the unseen "NULL" is all zeros
    assert clean_train["cat"].tolist()[2] == [0.0, 0.0, 1.0, 0.0]
    assert clean_test["cat"].tolist()[2] == [0.0, 0.0, 0.0, 0.0]
    assert np.isnan(clean_train["num"].tolist()[2])


def test_customized_pipelines_and_user_defined_operator() -> None:
    class Doubler:
        """A user defined scaler."""

        def fit(self, col_df: pd.Series) -> Any:
            return self

        def transform(self, col_df: pd.Series) -> pd.Series:
            return col_df * 2

    train, test = make_data()
    clean_train, clean_test = clean_ml(
        train,
        test,
        customized_cat_pipeline=[
            {"cat_imputation": {"operator": "constant", "cat_null_value": [None], "fill_val": "ZZ"}}
        ],
        customized_num_pipeline=[
            {"num_imputation": {"operator": "mean", "num_null_value": None}},
            {"num_scaling": {"operator": Doubler}},
        ],
    )
    assert clean_train["cat"].tolist() == ["a", "b", "ZZ", "a", "c"]
    assert clean_test["cat"].tolist() == ["b", "a", "NULL"]
    assert clean_train["num"].tolist() == [2.0, 4.0, 8.5, 8.0, 20.0]
    assert clean_test["num"].tolist() == [6.0, 8.5, 0.0]


def test_dask_input() -> None:
    train, test = make_data()
    expected_train, expected_test = clean_ml(train, test, cat_encoding="no_encoding")
    clean_train, clean_test = clean_ml(
        dd.from_pandas(train, npartitions=2),
        dd.from_pandas(test, npartitions=2),
        cat_encoding="no_encoding",
    )
    assert isinstance(clean_train, pd.DataFrame) and isinstance(clean_test, pd.DataFrame)
    assert clean_train.astype(object).equals(expected_train.astype(object))
    assert clean_test.astype(object).equals(expected_test.astype(object))


def test_pipeline_fit_and_transform() -> None:
    pipeline = Pipeline(
        {
            "cat_pipeline": ["cat_imputation", "cat_encoding"],
            "cat_imputation": "constant",
            "cat_null_value": [None, "?"],
            "fill_val": "missing",
            "cat_encoding": None,
        },
        {"num_pipeline": ["num_imputation", "num_scaling"], "num_imputation": "mean",
         "num_null_value": None, "num_scaling": "minmax"},
    )
    train = pd.Series(["a", "?", None])
    assert pipeline.fit(train, train) is pipeline
    assert pipeline.is_num_type is False
    assert pipeline.transform(pd.Series(["?", "b"])).tolist() == ["missing", "b"]

    numbers = pd.Series([0.0, np.nan, 10.0])
    train_result, test_result = pipeline.fit_transform(numbers, pd.Series([5.0]))
    assert pipeline.is_num_type is True
    assert train_result.tolist() == [0.0, 0.5, 1.0]
    assert test_result.tolist() == [0.5]


def test_component_registries() -> None:
    assert set(component_dic) == {
        "cat_encoding",
        "cat_imputation",
        "num_imputation",
        "num_scaling",
        "variance_threshold",
    }
    assert set(cat_imputers) == {"constant", "most_frequent", "drop"}
    assert set(num_imputers) == {"mean", "median", "most_frequent", "drop"}
    assert set(num_scalers) == {"standardize", "minmax", "maxabs"}
    assert set(cat_encoders) == {"one_hot"}
    assert set(thresholders) == {"variance_threshold"}


@pytest.mark.parametrize("name", ["constant", "most_frequent", "drop"])
def test_cat_imputers(name: str) -> None:
    column = pd.Series(["a", "b", "a", None, np.nan, "?"])
    imputer = cat_imputers[name]([None, "?"], "filled")
    result = imputer.fit_transform(column)
    if name == "constant":
        assert result.tolist() == ["a", "b", "a", "filled", "filled", "filled"]
    elif name == "most_frequent":
        assert result.tolist() == ["a", "b", "a", "a", "a", "a"]
    else:
        assert len(result) == 0
        assert cat_imputers[name]([None], "").fit_transform(pd.Series(["a"])).tolist() == ["a"]


@pytest.mark.parametrize("name", ["mean", "median", "most_frequent", "drop"])
def test_num_imputers(name: str) -> None:
    column = pd.Series([1.0, 1.0, 4.0, 10.0, np.nan])
    result = num_imputers[name]([np.nan]).fit_transform(column)
    expected = {"mean": 4.0, "median": 2.5, "most_frequent": 1.0}
    if name == "drop":
        assert len(result) == 0
    else:
        assert result.tolist() == [1.0, 1.0, 4.0, 10.0, expected[name]]


def test_scalers_encoder_and_thresholder() -> None:
    column = pd.Series([-2.0, 0.0, 2.0])
    np.testing.assert_allclose(num_scalers["standardize"]().fit_transform(column), [-1, 0, 1])
    np.testing.assert_allclose(num_scalers["minmax"]().fit_transform(column), [0, 0.5, 1])
    np.testing.assert_allclose(num_scalers["maxabs"]().fit_transform(column), [-1, 0, 1])

    encoder = cat_encoders["one_hot"]().fit(pd.Series(["x", "y", "x", None]))
    assert encoder.transform(pd.Series(["y", np.nan, "unseen"])).tolist() == [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]

    assert thresholders["variance_threshold"](1.0).fit_transform(column).tolist() == [-2, 0, 2]
    assert len(thresholders["variance_threshold"](10.0).fit_transform(column)) == 0
