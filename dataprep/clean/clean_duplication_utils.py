"""
Common functions and classes for the clean_duplication function.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations
from operator import itemgetter
from string import punctuation
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, List, Set
from unicodedata import category, combining, normalize

import pandas as pd

from .utils import apply_partitions, is_missing

if TYPE_CHECKING:
    import dask.dataframe as dd


class Clusterer:
    """
    Performs clustering methods on data.
    """

    # pylint: disable=too-many-instance-attributes

    clusters: pd.Series
    exported_code: List[str]
    _df: pd.DataFrame
    _counts: pd.Series
    _df_name: str
    _col: str
    _ngram: int
    _radius: int
    _block_size: int

    def __init__(self, df: pd.DataFrame | dd.DataFrame, col: str, df_name: str):
        self.clusters = pd.Series(dtype=object)
        self.exported_code = []
        # the data is clustered in memory: a Dask DataFrame is computed here
        parts = apply_partitions(df, lambda pdf: pdf)
        pdf = parts[0] if len(parts) == 1 else pd.concat(parts)
        # work on a copy so that the DataFrame passed in by the user is never modified
        self._df = pdf.copy(deep=False)
        self._counts = pd.Series(dtype=object)
        self._df_name = df_name
        self._col = col
        self._ngram = 2
        self._radius = 2
        self._block_size = 6
        # values are clustered as strings, missing values are left untouched
        self._df[self._col] = pd.Series(
            [val if is_missing(val) else str(val) for val in self._df[self._col].tolist()],
            index=self._df.index,
        )

    def cluster(self, cluster_method: str) -> None:
        """
        Create clusters using the given clustering method.
        """

        if cluster_method == "levenshtein":
            self._nearest_neighbours_cluster()
        else:
            self._key_collision_cluster(cluster_method)

    def _count_values(self) -> List[str]:
        """
        Count the distinct (non-missing) values of the column, and return them
        in order of first appearance.
        """
        self._counts = self._df[self._col].value_counts(sort=False)
        return list(self._counts.index)

    def _key_collision_cluster(self, cluster_method: str) -> None:
        """
        Create clusters using a key collision method.
        Clusters are a Pandas Series of lists (each list represents a cluster).
        """
        key_funcs: dict[str, Callable[[str], str]] = {
            "fingerprint": self._finger_print_key,
            "ngram-fingerprint": self._ngram_finger_print_key,
            "phonetic-fingerprint": self._phonetic_fingerprint_key,
        }
        key_func = key_funcs[cluster_method]

        # values with the same key belong to the same cluster
        groups: DefaultDict[str, List[str]] = defaultdict(list)
        for val in self._count_values():
            groups[key_func(val)].append(val)

        clusters = [groups[key] for key in sorted(groups) if len(groups[key]) > 1]
        # largest clusters first
        clusters.sort(key=len, reverse=True)
        self.clusters = pd.Series(clusters, name=self._col, dtype=object)

    def _nearest_neighbours_cluster(self) -> None:
        """
        Performs nearest neighbour clustering.
        Blocking is used to speed up the process, blocks are obtained where strings in the same
        block share a substring of a given blocking size. Only strings within the same block are
        compared using the levenshtein distance function.

        Method from OpenRefine: https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth
        and simile-vicino: https://code.google.com/archive/p/simile-vicino/
        """
        blocks: DefaultDict[str, Set[str]] = defaultdict(set)
        for val in self._count_values():
            self._populate_blocks(val, blocks, self._block_size)

        # compare every pair of strings in each block and add to cluster if
        # their distance is less than the given radius
        self.clusters = self._get_nearest_neighbour_clusters(blocks, self._radius)

    @staticmethod
    def _populate_blocks(val: str, blocks: DefaultDict[str, Set[str]], block_size: int) -> None:
        """
        Create n gram tokens of the given string and place the string into the block
        for each n gram.
        """
        tokens = _ngram_tokens(val, block_size)
        for token in tokens:
            blocks[token].add(val)

    @staticmethod
    def _get_nearest_neighbour_clusters(
        blocks: DefaultDict[str, Set[str]], radius: int
    ) -> pd.Series:
        """
        Compare every pair of strings in each block and add to cluster if
        their distance is less than the given radius.
        """
        try:
            # pylint: disable=import-outside-toplevel
            from rapidfuzz.distance.Levenshtein import distance as levenshtein_distance
        except ImportError as err:
            raise ImportError(
                "The levenshtein clustering method requires the rapidfuzz package. "
                "Install it with: pip install rapidfuzz"
            ) from err

        cluster_map: DefaultDict[str, Set[str]] = defaultdict(set)
        for block in blocks.values():
            for center, val in permutations(block, 2):
                if val in cluster_map[center]:
                    continue

                cluster_map[center].add(center)
                dist = levenshtein_distance(center, val)
                if dist <= radius or radius < 0:
                    cluster_map[center].add(val)

        # remove duplicate clusters and clusters of length 1
        unique_clusters = set(
            frozenset(cluster) for cluster in cluster_map.values() if len(cluster) > 1
        )
        # convert to list of lists
        clusters = [sorted(cluster) for cluster in unique_clusters]
        # sort by size of cluster, then alphabetically so the order is reproducible
        clusters.sort(key=lambda cluster: (-len(cluster), cluster))
        return pd.Series(clusters, dtype=object)

    @staticmethod
    def _finger_print_key(val: str) -> str:
        """
        Generates a fingerprint key from a given string.

        - remove leading and trailing whitespace
        - convert to lowercase
        - remove punctuation and control characters
        - normalize extended western characters to ASCII
        - split into whitespace separated tokens
        - sort tokens and remove duplicates
        - join tokens back together

        Method taken from OpenRefine:
        https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth
        """
        val = val.strip()
        val = val.lower()
        val = val.translate(str.maketrans("", "", punctuation))
        # remove control characters
        val = "".join(ch for ch in val if category(ch)[0] != "C")
        val = normalize_non_ascii(val)
        return " ".join(sorted(set(val.split())))

    @staticmethod
    def _phonetic_fingerprint_key(val: str) -> str:
        """
        Generates n-gram fingerprint from the given string.
        Uses the double metaphone algorithm.
        """
        try:
            from metaphone import doublemetaphone  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError(
                "The phonetic-fingerprint clustering method requires the Metaphone package. "
                "Install it with: pip install \"dataprep-next[duplication]\""
            ) from err

        primary, secondary = doublemetaphone(val)
        if primary == secondary:
            secondary = ""
        return f"{primary},{secondary}"

    def _ngram_finger_print_key(self, val: str) -> str:
        """
        Generates n-gram fingerprint from the given string.

        - convert to lowercase
        - remove punctuation, whitespace and control characters
        - get string n-grams
        - sort n-grams and remove duplicates
        - join sorted n grams back together
        - normalize extended western characters to ASCII

        Method taken from OpenRefine:
        https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth
        """
        return "".join(sorted(set(_ngram_tokens(val, self._ngram))))

    def _create_replace_calls(
        self, cluster_page: pd.Series, do_merge: List[bool], new_values: List[str]
    ) -> str:
        """
        Creates a string containing the required replace function calls.

        cluster_page
            Current page of clusters being displayed by the UI
        do_merge
            Boolean values indicating whether a cluster should be merged.
            If do_merge[i] == True then merge the i-th cluster.
        """
        df_name, col = self._df_name, f"'{self._col}'"
        replace_calls = []
        for idx, cluster in enumerate(cluster_page):
            if do_merge[idx]:
                # create the string that all the values in the cluster will be set to
                cluster_repr = new_values[idx].replace("'", "\\'")
                cluster_repr = f"'{cluster_repr}'"
                # create the strings to be replaced
                cluster_vals = [val.replace("'", "\\'") for val, _ in cluster]
                cluster_vals = [f"'{val}'" for val in cluster_vals if f"'{val}'" != cluster_repr]
                code = (
                    f"{df_name}[{col}] = {df_name}[{col}].replace"
                    f"([{', '.join(cluster_vals)}], {cluster_repr})"
                )
                replace_calls.append(code)

        return "\n" + "\n".join(replace_calls)

    def live_export_code(
        self, cluster_page: pd.Series, do_merge: List[bool], new_values: List[str]
    ) -> str:
        """
        Create the DataFrame replace calls that merge the selected clusters.

        The code is appended to ``exported_code`` and returned, so that the user
        interface can show it. Nothing is injected into the notebook: this works
        the same in Jupyter, JupyterLab and VS Code notebooks.
        """
        code = self._create_replace_calls(cluster_page, do_merge, new_values).strip("\n")
        if code:
            self.exported_code.append(code)
        return code

    def execute_merge_code(
        self,
        cluster_page: pd.Series,
        do_merge: List[bool],
        new_values: List[str],
    ) -> None:
        """
        Merge the clusters in the DataFrame.
        """
        for idx, cluster in enumerate(cluster_page):
            cluster_repr = new_values[idx]
            if do_merge[idx]:
                self._df[self._col] = self._df[self._col].replace(
                    [cluster_val for cluster_val, _ in cluster], cluster_repr
                )

    def final_df(self) -> pd.DataFrame:
        """
        Return the DataFrame with the merged values.

        When running inside IPython or a notebook kernel (Jupyter, JupyterLab,
        VS Code), the DataFrame is also assigned to the variable
        ``<name of the input DataFrame>_clean`` in the user's namespace.
        """
        final_df = self._df.copy()
        shell = _get_ipython_shell()
        if shell is not None:
            shell.user_ns[self.final_df_name] = final_df
        return final_df

    @property
    def final_df_name(self) -> str:
        """Name of the variable that receives the cleaned DataFrame."""
        return f"{self._df_name}_clean"

    def get_page(self, start: int, end: int) -> pd.Series:
        """
        Returns a page of clusters from start to end. Adds the counts to
        each cluster entry and sorts each cluster by the counts, so that the values
        in each cluster are displayed in the order of the most frequent.
        """
        counts = self._counts.to_dict()
        page = self.clusters.iloc[start:end]
        page = page.map(lambda cluster: [(val, int(counts[val])) for val in cluster])
        page = page.map(lambda x: sorted(x, key=itemgetter(1), reverse=True))
        return page

    def set_cluster_params(self, ngram: int, radius: int, block_size: int) -> None:
        """
        Set clustering parameters.
        """
        self._ngram = ngram
        self._radius = radius
        self._block_size = block_size


def _get_ipython_shell() -> Any:
    """
    Return the running IPython shell, or None when not running inside IPython.
    """
    try:
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return get_ipython()


def _ngram_tokens(val: str, n: int) -> List[str]:
    """
    Create n-gram tokens from the given string.
    """
    val = val.strip()
    val = val.lower()
    val = " ".join(val.split())
    val = val.translate(str.maketrans("", "", punctuation))
    # remove control characters
    val = "".join(ch for ch in val if category(ch)[0] != "C")
    val = normalize_non_ascii(val)
    return [val[i : i + n] for i in range(len(val) - n + 1)]


def normalize_non_ascii(val: str) -> str:
    """
    Normalize extended western characters to ascii. (remove accents)
    """
    nfkd_form = normalize("NFKD", val)
    return "".join([c for c in nfkd_form if not combining(c)])
