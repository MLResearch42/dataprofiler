"""
This module implements the plot(df) function.
"""

from typing import Any, Dict, List, Optional, Union

from ..configs import Config
from ..container import Container
from ..dtypes import DTypeDef, LatLong
from ..eda_frame import DataFrame, sample_rows
from ..parallel import Progress
from ..target import compute_target, render_target
from .compute import compute
from .render import render

__all__ = ["plot", "compute", "render"]


def plot(
    df: DataFrame,
    col1: Optional[Union[str, LatLong]] = None,
    col2: Optional[Union[str, LatLong]] = None,
    col3: Optional[str] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
    target: Optional[str] = None,
    sample: Union[int, float, None] = None,
    progress: bool = True,
) -> Container:
    """Generates plots for exploratory data analysis.

    If no columns are specified, the distribution of
    each coloumn is plotted. A histogram is plotted if the
    column contains numerical values, a bar chart is plotted
    if the column contains categorical values, a line chart is
    plotted if the column is of type datetime.

    If one column (x) is specified, the
    distribution of x is plotted in various ways. If x
    contains categorical values, a bar chart and pie chart are
    plotted. If x contains numerical values, a histogram,
    kernel density estimate plot, box plot, and qq plot are plotted.
    If x contains datetime values, a line chart is plotted.

    If two columns (x and y) are specified, plots depicting
    the relationship between the variables will be displayed. If
    x and y contain numerical values, a scatter plot, hexbin
    plot, and binned box plot are plotted. If one of x and y
    contain categorical values and the other contains numerical values,
    a box plot and multiline histogram are plotted. If x and y
    contain categorical vales, a nested bar chart, stacked bar chart, and
    heat map are plotted. If one of x and y contains datetime values
    and the other contains numerical values, a line chart and a box plot
    are shown. If one of x and y contains datetime values and the other
    contains categorical values, a multiline chart and a stacked box plot
    are shown.

    If x, y, and z are specified, they must be one each of type datetime,
    numerical, and categorical. A multiline chart containing an aggregate
    on the numerical column grouped by the categorical column over time is
    plotted.


    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    col1: Optional[str], default None
        A valid column name from the dataframe
    col2: Optional[str], default None
        A valid column name from the dataframe
    col3: Optional[str], default None
        A valid column name from the dataframe
    config
        A dictionary for configuring the visualizations
        E.g. config={"hist.bins": 20}
    display
        A list containing the names of the visualizations to display
        E.g. display=["Histogram"]
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous().
    target
        The name of a target column. ``plot(df, target="y")`` ranks every other
        column by how strongly it is associated with the target and shows the
        distribution of the target (see :func:`dataprep.eda.rank_features`).
        ``plot(df, "x", target="y")`` is the same as ``plot(df, "x", "y")``.
    sample
        Plot a random sample of the rows instead of all of them: a number of rows
        (int) or a fraction of the rows (float). The sample is the same on every
        run, and the output says that it shows a sample.
    progress
        Enable the progress bar.

    Examples
    --------
    >>> from dataprep.datasets import load_dataset
    >>> from dataprep.eda import plot
    >>> titanic = load_dataset("titanic")
    >>> plot(titanic)  # doctest: +SKIP
    >>> plot(titanic, "Age")  # doctest: +SKIP
    >>> plot(titanic, "Age", "Sex")  # doctest: +SKIP
    >>> plot(titanic, target="Survived")  # doctest: +SKIP
    >>> plot(titanic, sample=500)  # doctest: +SKIP
    """
    cfg = Config.from_dict(display, config)
    df, sample_note = sample_rows(df, sample)

    if target is not None:
        columns = [col for col in (col1, col2, col3) if col is not None]
        if len(columns) > 1:
            raise ValueError(
                "With target=..., pass at most one other column: plot(df, target=...) "
                "ranks every column, plot(df, col, target=...) compares one column with it."
            )
        if columns:
            col1, col2, col3 = columns[0], target, None
        else:
            with Progress(enabled=progress):
                itmdt = compute_target(df, target, cfg, dtype)
            to_render = render_target(itmdt, cfg)
            to_render["sample_note"] = sample_note
            return Container(to_render, itmdt.visual_type, cfg)

    with Progress(enabled=progress):
        itmdt = compute(df, col1, col2, col3, cfg=cfg, dtype=dtype)

    to_render = render(itmdt, cfg)
    to_render["sample_note"] = sample_note

    return Container(to_render, itmdt.visual_type, cfg)
