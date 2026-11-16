"""
    This module implements the plot_missing(df) function
"""

from typing import Any, Dict, List, Optional, Union

from ..configs import Config
from ..container import Container
from ..dtypes import DTypeDef
from ..eda_frame import DataFrame, sample_rows
from ..parallel import Progress
from .compute import compute_missing
from .render import render_missing

__all__ = ["render_missing", "compute_missing", "plot_missing"]


def plot_missing(
    df: DataFrame,
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
    sample: Union[int, float, None] = None,
    progress: bool = True,
) -> Container:
    """
    This function is designed to deal with missing values
    There are three functions: plot_missing(df), plot_missing(df, x)
    plot_missing(df, x, y)

    Parameters
    ----------
    df
        the pandas data_frame for which plots are calculated for each column.
    col1
        a valid column name of the data frame.
    col2
        a valid column name of the data frame.
    config
        A dictionary for configuring the visualizations.
        E.g. config={"spectrum.bins": 20}
    display
        A list containing the names of the visualizations to display
        E.g. display=["Stats", "Spectrum"]
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous().
    sample
        Use a random sample of the rows instead of all of them: a number of rows
        (int) or a fraction of the rows (float). The sample is the same on every
        run, and the output says that it shows a sample.
    progress
        Enable the progress bar.

    Examples
    --------
    >>> from dataprep.datasets import load_dataset
    >>> from dataprep.eda import plot_missing
    >>> df = load_dataset("titanic")
    >>> plot_missing(df)  # doctest: +SKIP
    >>> plot_missing(df, "Age")  # doctest: +SKIP
    >>> plot_missing(df, "Age", "Fare")  # doctest: +SKIP
    """
    cfg = Config.from_dict(display, config)
    df, sample_note = sample_rows(df, sample)

    with Progress(enabled=progress):
        itmdt = compute_missing(df, col1, col2, cfg=cfg, dtype=dtype)

    to_render = render_missing(itmdt, cfg)
    to_render["sample_note"] = sample_note

    return Container(to_render, itmdt.visual_type, cfg)
