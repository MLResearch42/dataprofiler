"""Deprecated alias of :mod:`dataprep.eda.dtypes`.

DataPrep used to carry two copies of the type system. They are now one module;
this file only exists so that ``from dataprep.eda.dtypes_v2 import ...`` keeps working.
"""

# pylint: disable=wildcard-import,unused-wildcard-import
from .dtypes import *  # noqa: F401,F403
from .dtypes import (  # noqa: F401
    CATEGORICAL_DTYPES,
    DATETIME_DTYPES,
    NUMERICAL_DTYPES,
    STRING_DTYPES,
    DTypeDict,
    DTypeOrStr,
    detect_without_known,
    get_dtype_cnts_and_num_cols,
    is_geography,
    is_geopoint,
    is_pandas_categorical,
    map_dtype,
)
