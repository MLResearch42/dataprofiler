"""
    This module implements the plot_correlation(df) function.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..configs import Config
from ..container import Container
from ..eda_frame import DataFrame, sample_rows
from ..parallel import Progress
from .compute import compute_correlation
from .render import render_correlation

__all__ = ["render_correlation", "compute_correlation", "plot_correlation"]


def plot_correlation(
    df: DataFrame,
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    sample: Union[int, float, None] = None,
    progress: bool = True,
) -> Container:
    """
    This function is designed to calculate the correlation between columns
    There are three functions: plot_correlation(df), plot_correlation(df, x)
    plot_correlation(df, x, y)
    There are also some parameters such as k and value_range to satisfy your requirement

    Parameters
    ----------
    df
        The pandas data_frame for which plots are calculated for each column.
    col1
        A valid column name of the data frame.
    col2
        A valid column name of the data frame.
    value_range
        Range of value.
    k
        Choose top-k element.
    config
        A dictionary for configuring the visualizations
        E.g. config={"scatter.sample_size": 5000}
    display
        A list containing the names of the visualizations to display
        E.g. display=["Pearson"]
    sample
        Use a random sample of the rows instead of all of them: a number of rows
        (int) or a fraction of the rows (float). The sample is the same on every
        run, and the output says that it shows a sample.
    progress
        Enable the progress bar.

    Examples
    --------
    >>> from dataprep.datasets import load_dataset
    >>> from dataprep.eda import plot_correlation
    >>> df = load_dataset("house_prices_train")
    >>> plot_correlation(df)  # doctest: +SKIP
    >>> plot_correlation(df, k=6)  # doctest: +SKIP
    >>> plot_correlation(df, "SalePrice")  # doctest: +SKIP
    >>> plot_correlation(df, "SalePrice", k=3)  # doctest: +SKIP
    >>> plot_correlation(df, "SalePrice", value_range=(0.5, 1))  # doctest: +SKIP
    >>> plot_correlation(df, "GrLivArea", "SalePrice")  # doctest: +SKIP

    Note
    ----
    Only numerical columns are used.
    """

    cfg = Config.from_dict(display, config)
    df, sample_note = sample_rows(df, sample)

    with Progress(enabled=progress):
        itmdt = compute_correlation(df, col1, col2, cfg=cfg, value_range=value_range, k=k)
    to_render = render_correlation(itmdt, cfg)
    if isinstance(to_render, dict):
        to_render["sample_note"] = sample_note

    return Container(to_render, itmdt.visual_type, cfg)
