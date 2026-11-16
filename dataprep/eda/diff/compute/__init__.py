"""Computations for plot_diff([df...])."""

from typing import Any, Dict, List, Optional, Union

from ....errors import DataprepError
from ...configs import Config
from ...dtypes import DTypeDef
from ...eda_frame import DataFrame, to_pandas
from ...intermediate import Intermediate
from .multiple_column import compare_multiple_col
from .multiple_df import compare_multiple_df

__all__ = ["compute_diff"]


def compute_diff(
    df: List[DataFrame],
    x: Optional[str] = None,
    *,
    cfg: Union[Config, Dict[str, Any], None] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    All in one compute function.

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    cfg: Union[Config, Dict[str, Any], None], default None
        When a user call plot(), the created Config object will be passed to compute().
        When a user call compute() directly, if he/she wants to customize the output,
        cfg is a dictionary for configuring. If not, cfg is None and
        default values will be used for parameters.
    display: Optional[List[str]], default None
        A list containing the names of the visualizations to display. Only exist when
        a user call compute() directly and want to customize the output
    x: Optional[str], default None
        A valid column name from the dataframe
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint:disable = too-many-branches
    if isinstance(cfg, dict):
        cfg = Config.from_dict(display, cfg)
    elif not cfg:
        cfg = Config()

    if isinstance(df, list):

        if len(df) < 2:
            raise DataprepError("plot_diff needs at least 2 DataFrames.")
        if len(df) > 5:
            raise DataprepError("Too many DataFrames, max: 5.")
        label = cfg.diff.label
        if not label:
            cfg.diff.label = [f"df{i+1}" for i in range(len(df))]
        elif len(df) != len(label):
            raise ValueError("Number of the given label doesn't match the number of DataFrames.")

        if cfg.diff.baseline > len(df) - 1:
            raise ValueError("Baseline is out of the boundary of the input.")

        df_list = [to_pandas(dfi) for dfi in df]

        if x is not None:
            x = str(x)
            if sum(x in map(str, dfi.columns) for dfi in df_list) < 2:
                raise DataprepError("x must exist in at least two DataFrames")
            return compare_multiple_col(df_list, x, cfg)
        return compare_multiple_df(df_list, cfg, dtype)

    raise TypeError(
        f"plot_diff expects a list of DataFrames, got {type(df).__name__}. "
        "Example: plot_diff([df1, df2])"
    )
