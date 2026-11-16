"""Computations for plot(df, x, y, z)."""

from typing import Optional

import pandas as pd

from ...configs import Config
from ...dtypes import (
    Continuous,
    DateTime,
    DTypeDef,
    GeoGraphy,
    GeoPoint,
    Nominal,
    SmallCardNum,
)
from ...eda_frame import DataFrame, EDAFrame
from ...intermediate import Intermediate
from ...utils import _calc_line_dt
from .common import gen_new_df_with_used_cols


def compute_trivariate(
    df: DataFrame,
    col1: str,
    col2: str,
    col3: str,
    cfg: Config,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """Compute functions for plot(df, x, y, z).

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    x
        A column name from the DataFrame
    y
        A column name from the DataFrame
    z
        A column name from the DataFrame
    cfg:
        Config instance
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    if len({col1, col2, col3}) != 3:
        raise ValueError(f"The three columns must be different, got {col1}, {col2}, {col3}.")
    _, ndf = gen_new_df_with_used_cols(df, col1, col2, col3)
    x, y, z = str(col1), str(col2), str(col3)
    frame = EDAFrame(ndf, dtype)

    if frame.shape[0] == 0:
        raise ValueError("The DataFrame has no rows, there is nothing to plot.")

    # Whichever order the columns were given in, find the datetime (x),
    # the numerical (y) and the categorical (z) one.
    categorical_types = (Nominal, GeoGraphy, SmallCardNum, GeoPoint)
    roles = {}
    for col in (x, y, z):
        col_type = frame.get_eda_dtype(col)
        if isinstance(col_type, DateTime):
            roles.setdefault("datetime", col)
        elif isinstance(col_type, Continuous):
            roles.setdefault("numerical", col)
        elif isinstance(col_type, categorical_types):
            roles.setdefault("categorical", col)
    if len(roles) != 3:
        types = ", ".join(f"{col}: {type(frame.get_eda_dtype(col)).__name__}" for col in (x, y, z))
        raise ValueError(
            "plot(df, x, y, z) needs one datetime, one numerical and one categorical column. "
            f"Detected types: {types}. Use the dtype argument to override a detected type."
        )
    x, y, z = roles["datetime"], roles["numerical"], roles["categorical"]

    # x: datetime, y: numerical, z: categorical (as strings)
    tmp_df = pd.DataFrame(
        {x: frame.frame[x], y: frame.frame[y], z: frame.get_col_as_str(z)}, copy=False
    ).dropna()
    if len(tmp_df) == 0:
        raise ValueError("No rows are left after dropping missing values.")

    # line chart
    data = _calc_line_dt(
        tmp_df, cfg.line.unit, cfg.line.agg, cfg.line.ngroups, cfg.line.sort_descending
    )
    return Intermediate(
        x=x,
        y=y,
        z=z,
        agg=cfg.line.agg,
        data=data,
        visual_type="dt_cat_num_cols",
    )
