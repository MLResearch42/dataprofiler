"""Common components for compute correlation."""

import warnings
from enum import Enum, auto

import numpy as np
from scipy.stats import kendalltau as kendalltau_
from scipy.stats import rankdata


class CorrelationMethod(Enum):
    """Supported correlation methods"""

    # pylint: disable=invalid-name
    Pearson = auto()
    Spearman = auto()
    KendallTau = auto()


def nanrankdata(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Rank the data along an axis (ties get their average rank); NaNs stay NaN."""
    data = np.asarray(data, dtype=float)
    return rankdata(data, axis=axis, nan_policy="omit")


def kendalltau(a: np.ndarray, b: np.ndarray) -> float:  # pylint: disable=invalid-name
    """Kendall's tau-b of two 1-D arrays without NaNs (NaN when undefined)."""
    a, b = np.ravel(a), np.ravel(b)
    if a.size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(kendalltau_(a, b).correlation)


def corrcoef(arr: np.ndarray) -> float:
    """Pearson correlation of the two columns of a (rows x 2) array (NaN when undefined)."""
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] < 2:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(arr, rowvar=False)[0, 1])
