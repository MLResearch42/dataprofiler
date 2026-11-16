"""
    This module implements the plot_diff function.
"""

from typing import Any, Dict, List, Optional, Union

from ..configs import Config
from ..container import Container
from ..dtypes import DTypeDef
from ..eda_frame import DataFrame, sample_rows
from ..parallel import Progress
from .compute import compute_diff
from .render import render_diff

__all__ = ["plot_diff", "compute_diff", "render_diff"]


def plot_diff(
    df: List[DataFrame],
    x: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
    progress: bool = False,
    *,
    sample: Union[int, float, None] = None,
) -> Container:
    """
    This function is to compute and visualize the differences between 2 or more(up to 5) datasets.

    Parameters
    ----------
    df
        The DataFrame(s) to be compared.
    x
        The column to be emphasized in the comparision.
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
    progress
        Whether to show the progress bar.
    sample
        Compare random samples of the rows instead of all of them: a number of
        rows (int) or a fraction of the rows (float), taken from each DataFrame.
        The samples are the same on every run, and the output says that it
        shows samples.

    Examples
    --------
    >>> from dataprep.datasets import load_dataset
    >>> from dataprep.eda import plot_diff
    >>> df_train = load_dataset('house_prices_train')
    >>> df_test = load_dataset('house_prices_test')
    >>> plot_diff([df_train, df_test])  # doctest: +SKIP
    """
    # pylint: disable=too-many-arguments
    cfg = Config.from_dict(display, config)
    df, sample_note = sample_each(df, sample)

    with Progress(enabled=progress):
        intermediate = compute_diff(df, x=x, cfg=cfg, dtype=dtype)
    to_render = render_diff(intermediate, cfg=cfg)
    to_render["sample_note"] = sample_note
    return Container(to_render, intermediate.visual_type, cfg)


def sample_each(dfs: Any, sample: Union[int, float, None]) -> Any:
    """``sample_rows`` applied to each of several DataFrames.

    Returns the DataFrames and a note for the output, or None if no rows were dropped.
    """
    if sample is None:
        return dfs, None
    if isinstance(dfs, dict):
        sampled_dict, note = sample_each(list(dfs.values()), sample)
        return dict(zip(dfs.keys(), sampled_dict)), note
    if not isinstance(dfs, (list, tuple)):
        return sample_rows(dfs, sample)
    results = [sample_rows(one, sample) for one in dfs]
    sampled = [one for one, _ in results]
    if not any(note for _, note in results):
        return sampled, None
    sizes = ", ".join(f"{len(one):,}" for one in sampled)
    note = (
        f"Based on random samples of the rows ({sizes} rows); "
        "counts and statistics describe the samples."
    )
    return sampled, note
