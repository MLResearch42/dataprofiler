"""
Initialize dictionary of numerical imputers.
"""

from .drop_imputer import DropImputer
from .mean_imputer import MeanImputer
from .median_imputer import MedianImputer
from .most_frequent_imputer import MostFrequentImputer

operator_dic = {
    "mean": MeanImputer,
    "median": MedianImputer,
    "most_frequent": MostFrequentImputer,
    "drop": DropImputer,
}
