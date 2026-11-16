"""
module for testing the clean_duplication() function
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean.clean_duplication import UserInterface

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def clean_duplication_ui() -> UserInterface:
    df = pd.DataFrame(
        {
            "city": [
                "Québec",
                "Québec",
                "Québec",
                "Quebec",
                "Quebec",
                "quebec",
                "vancouver",
                "vancouver",
                "vancouverr",
                "Vancouver",
                "Vancouver",
                "Vancouver",
                "van",
                "Ottowa",
                "Ottowa",
                "otowa",
                "hello",
                np.nan,
            ]
        }
    )
    return UserInterface(df, "city", "df", 5)


def test_fingerprint_clusters(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._clustering_method_drop.value = "fingerprint"
    clusters = clean_duplication_ui._clusterer.get_page(0, 5)
    clusters_check = pd.Series(
        [[("Québec", 3), ("Quebec", 2), ("quebec", 1)], [("Vancouver", 3), ("vancouver", 2)]],
        name="city",
    )

    assert clusters_check.equals(clusters)


def test_ngram_clusters(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._clustering_method_drop.value = "ngram-fingerprint"
    clusters = clean_duplication_ui._clusterer.get_page(0, 5)
    clusters_check = pd.Series(
        [
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Vancouver", 3), ("vancouver", 2)],
        ],
        name="city",
    )
    # set the ngram size to 1
    clean_duplication_ui._ngram_text.value = "1"
    clusters2 = clean_duplication_ui._clusterer.get_page(0, 5)
    # check for either ordering of clusters, since they're
    # only sorted by the length of the cluster the order isn't
    # guaranteed
    clusters_check2 = pd.Series(
        [
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
            [("Ottowa", 2), ("otowa", 1)],
        ],
        name="city",
    )
    clusters_check3 = pd.Series(
        [
            [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Ottowa", 2), ("otowa", 1)],
        ],
        name="city",
    )

    assert clusters_check.equals(clusters)
    assert clusters_check2.equals(clusters2) or clusters_check3.equals(clusters2)


def test_phonetic_clusters(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._clustering_method_drop.value = "phonetic-fingerprint"
    clusters = clean_duplication_ui._clusterer.get_page(0, 5)
    # check for either ordering of clusters, since they're
    # only sorted by the length of the cluster the order isn't
    # guaranteed
    clusters_check = pd.Series(
        [
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
            [("Ottowa", 2), ("otowa", 1)],
        ],
        name="city",
    )

    clusters_check2 = pd.Series(
        [
            [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Ottowa", 2), ("otowa", 1)],
        ],
        name="city",
    )
    assert clusters_check.equals(clusters) or clusters_check2.equals(clusters)


# def test_levenshtein_clusters(clean_duplication_ui: UserInterface) -> None:
#     clean_duplication_ui._clustering_method_drop.value = "levenshtein"
#     clusters = clean_duplication_ui._clusterer.get_page(0, 5)
#     # check for either ordering of clusters, since they're
#     # only sorted by the length of the cluster the order isn't
#     # guaranteed
#     clusters_check = pd.Series(
#         [
#             [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
#             [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
#         ],
#         name="city",
#     )
#     clusters_check2 = pd.Series(
#         [
#             [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
#             [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
#         ],
#         name="city",
#     )
#     clean_duplication_ui._block_chars_text.value = "7"
#     clusters2 = clean_duplication_ui._clusterer.get_page(0, 5)
#     clusters_check3 = pd.Series(
#         [[("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)]],
#         name="city",
#     )
#     assert clusters_check.equals(clusters) or clusters_check2.equals(clusters)
#     assert clusters_check3.equals(clusters2)


def test_merge(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._clustering_method_drop.value = "fingerprint"
    # select the checkbox for the first cluster and
    # set the textbox contents to "hi"
    clean_duplication_ui._checks[0].value = True
    clean_duplication_ui._reprs[0].value = "hi"
    clean_duplication_ui._execute_merge({})
    # get the dataframe after merging
    df_clean = clean_duplication_ui._clusterer._df
    df_check = df_clean.copy()
    df_check["city"] = [
        "hi",
        "hi",
        "hi",
        "hi",
        "hi",
        "hi",
        "vancouver",
        "vancouver",
        "vancouverr",
        "Vancouver",
        "Vancouver",
        "Vancouver",
        "van",
        "Ottowa",
        "Ottowa",
        "otowa",
        "hello",
        np.nan,
    ]

    assert df_check.equals(df_clean)


def test_select_all(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._sel_all.value = True
    assert all(check.value for check in clean_duplication_ui._checks)


def _cities() -> pd.DataFrame:
    return pd.DataFrame(
        {"city": ["New York", "new york", "New  York.", "Boston", "Bostn", None], "n": range(6)}
    )


def test_levenshtein_clusters() -> None:
    ui = UserInterface(_cities(), "city", "df", 5)
    ui._clustering_method_drop.value = "levenshtein"
    clusters = [sorted(val for val, _ in cluster) for cluster in ui._clusterer.get_page(0, 5)]
    # every cluster holds the values within the radius of one of its values, largest first
    assert clusters[0] == ["New  York.", "New York", "new york"]
    assert all(len(cluster) > 1 for cluster in clusters)
    # only values that share a block of 6 characters are compared: "Bostn" is too short
    assert ["Bostn", "Boston"] not in clusters
    ui._block_chars_text.value = "4"
    clusters = [sorted(val for val, _ in cluster) for cluster in ui._clusterer.get_page(0, 5)]
    assert ["Bostn", "Boston"] in clusters
    # a radius of 0 only clusters identical strings
    ui._radius_text.value = "0"
    assert ui._clusterer.get_page(0, 5).tolist() == []
    # an invalid parameter shows a message instead of raising
    ui._radius_text.value = "two"
    assert ui._cluster_vbox.children == (ui._invalid_param_label,)


def test_merge_export_code_and_finish() -> None:
    df = _cities()
    before = df.copy()
    ui = UserInterface(df, "city", "cities", 5)
    clusters = ui._clusterer.get_page(0, 5).tolist()
    assert clusters == [[("New York", 1), ("new york", 1), ("New  York.", 1)]]
    assert ui._code_area.layout.display == "none"

    ui._checks[0].value = True
    ui._reprs[0].value = "New York"
    ui._execute_merge({})
    # the code that reproduces the merge is shown, nothing is injected in the notebook
    code = "cities['city'] = cities['city'].replace(['new york', 'New  York.'], 'New York')"
    assert ui._code_area.value == code
    assert ui._code_area.layout.display == "flex"
    assert ui._clusterer.get_page(0, 5).tolist() == []

    ui._close({})
    expected = before.copy()
    expected["city"] = ["New York", "New York", "New York", "Boston", "Bostn", None]
    assert ui.result is not None and ui.result.equals(expected)
    assert "`cities_clean`" in ui._result_label.value
    # the DataFrame of the user is not modified, and running the code gives the same result
    assert df.equals(before)
    exec(code, {"cities": df})  # pylint: disable=exec-used
    assert df.equals(expected)


def test_merge_without_export_code() -> None:
    ui = UserInterface(_cities(), "city", "df", 5)
    ui._export_code.value = False
    ui._checks[0].value = True
    ui._execute_merge({})
    assert ui._code_area.value == ""
    assert ui._clusterer.exported_code == []


def test_pages() -> None:
    values = [f"{word} {suffix}" for word in "abcdefg" for suffix in ("x", "X")]
    ui = UserInterface(pd.DataFrame({"col": values}), "col", "df", 3)
    assert len(ui._clusterer.clusters) == 7
    assert len(ui._clusterer.get_page(ui._page_pos, ui._page_pos + 3)) == 3
    ui._next_page({})
    ui._next_page({})
    assert ui._page_pos == 6
    assert len(ui._clusterer.get_page(ui._page_pos, ui._page_pos + 3)) == 1
    ui._prev_page({})
    assert ui._page_pos == 3


def test_dask_input_and_non_string_values() -> None:
    import dask.dataframe as dd  # pylint: disable=import-outside-toplevel

    ui = UserInterface(dd.from_pandas(_cities(), npartitions=2), "city", "df", 5)
    assert ui._clusterer.get_page(0, 5).tolist() == [
        [("New York", 1), ("new york", 1), ("New  York.", 1)]
    ]
    ui = UserInterface(pd.DataFrame({"num": [1, 1, 2, np.nan]}), "num", "df", 5)
    assert ui._clusterer.get_page(0, 5).tolist() == []


def test_clean_duplication_function() -> None:
    from ...clean import clean_duplication  # pylint: disable=import-outside-toplevel

    cities = _cities()
    box = clean_duplication(cities, "city")
    assert type(box).__name__ == "Box"
    assert len(box.children) == 5
    box = clean_duplication(cities, "city", df_var_name="frame", page_size=2)
    assert type(box).__name__ == "Box"
