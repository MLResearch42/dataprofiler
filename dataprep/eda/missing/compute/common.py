"""Common parts for compute missing."""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from ...configs import Config
from ...dtypes import Continuous, DateTime, DType, GeoGraphy, GeoPoint, Nominal, SmallCardNum
from ...utils import histogram as _np_histogram
from ...utils import value_counts

LABELS = ["Orignal data", "After drop missing values"]

CATEGORICAL_LIKE = (Nominal, GeoGraphy, GeoPoint, SmallCardNum, DateTime)


def _categorical_counts(srs: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Counts and labels of the distinct values, most frequent first."""
    cnts = value_counts(srs).sort_values(ascending=False, kind="stable")
    return cnts.to_numpy(), np.asarray(cnts.index, dtype=object)


def uni_histogram(
    srs: pd.Series,
    srs_dtype: DType,
    cfg: Config,
    range: Optional[Tuple[float, float]] = None,  # pylint: disable=redefined-builtin
) -> Tuple[np.ndarray, ...]:
    """Calculate "histogram" for both numerical and categorical.

    Returns ``(counts, centers, edges)`` for numerical data and
    ``(counts, labels)`` for categorical data.
    """

    if isinstance(srs_dtype, Continuous):
        counts, edges = _np_histogram(srs.to_numpy(dtype=float), cfg.hist.bins, range)
        centers = (edges[:-1] + edges[1:]) / 2

        return counts, centers, edges

    if isinstance(srs_dtype, CATEGORICAL_LIKE):
        return _categorical_counts(srs)

    raise ValueError(f"Unsupported dtype {srs.dtype}")


def histogram(
    arr: Any,
    eda_dtype: DType,
    bins: Optional[int] = None,
    return_edges: bool = True,
    range: Optional[Tuple[float, float]] = None,  # pylint: disable=redefined-builtin
) -> Tuple[np.ndarray, ...]:
    """Calculate "histogram" for both numerical and categorical."""
    arr = np.asarray(arr)
    if len(arr.shape) != 1:
        raise ValueError("Histogram only supports 1-d array.")

    if isinstance(eda_dtype, Continuous):
        if bins is None:
            raise ValueError("num_bins cannot be None if calculating numerical histograms.")

        counts, edges = _np_histogram(arr, bins, range)
        centers = (edges[:-1] + edges[1:]) / 2

        if not return_edges:
            return counts, centers
        return counts, centers, edges

    if isinstance(eda_dtype, CATEGORICAL_LIKE):
        return _categorical_counts(pd.Series(arr, dtype=object))

    raise ValueError(f"Unsupported dtype {eda_dtype}")
