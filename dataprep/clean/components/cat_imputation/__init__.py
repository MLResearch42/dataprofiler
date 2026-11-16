"""
Initialize dictionary of categorical imputers.
"""

from .constant_imputer import ConstantImputer
from .drop_imputer import DropImputer
from .most_frequent_imputer import MostFrequentImputer

operator_dic = {
    "constant": ConstantImputer,
    "most_frequent": MostFrequentImputer,
    "drop": DropImputer,
}
