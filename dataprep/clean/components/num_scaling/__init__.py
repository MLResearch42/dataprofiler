"""
Initialize dictionary of numerical scalers.
"""

from .maxabs_scaler import MaxAbsScaler
from .minmax_scaler import MinmaxScaler
from .standard_scaler import StandardScaler

operator_dic = {
    "standardize": StandardScaler,
    "minmax": MinmaxScaler,
    "maxabs": MaxAbsScaler,
}
