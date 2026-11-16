"""
    This module implements the create_report(df) function.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from jinja2 import ChoiceLoader, Environment, PackageLoader

from ..configs import Config
from ..container import bokeh_resources
from ..eda_frame import sample_rows
from .formatter import format_report
from .report import Report

__all__ = ["create_report"]

ENV_LOADER = Environment(
    loader=ChoiceLoader(
        [
            PackageLoader("dataprep", "eda/create_report/templates"),
            PackageLoader("dataprep", "eda/templates"),  # shared notebook-frame script
        ]
    ),
    autoescape=False,
)


def create_report(
    df: pd.DataFrame,
    *,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    title: Optional[str] = "DataPrep Report",
    mode: Optional[str] = "basic",
    sample: Union[int, float, None] = None,
    progress: bool = True,
) -> Report:
    """
    This function is to generate and render element in a report object.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    config
        A dictionary for configuring the visualizations
        E.g. config={"hist.bins": 20}
    display
        The list that contains the names of plots user wants to display,
        E.g. display =  ["bar", "hist"]
        Without user's specifications, the default is "auto"
    title: Optional[str], default "DataPrep Report"
        The title of the report, which will be shown on the navigation bar.
    mode: Optional[str], default "basic"
        This controls what type of report to be generated.
        Currently only the 'basic' is fully implemented.
    sample
        Build the report from a random sample of the rows: a number of rows (int)
        or a fraction of the rows (float). Much faster on large tables. The sample
        is the same on every run, and the report says that it shows a sample.
    progress
        Whether to show the progress bar.

    Examples
    --------
    >>> from dataprep.datasets import load_dataset
    >>> from dataprep.eda import create_report
    >>> df = load_dataset("titanic")
    >>> report = create_report(df)  # doctest: +SKIP
    >>> report  # show the report in a notebook  # doctest: +SKIP
    >>> report.save("titanic.html")  # save it as a standalone HTML file  # doctest: +SKIP
    >>> report.show_browser()  # open it in the browser  # doctest: +SKIP
    >>> create_report(df, sample=500)  # doctest: +SKIP
    """
    _suppress_warnings()
    cfg = Config.from_dict(display, config)
    df, sample_note = sample_rows(df, sample)
    context = {
        "resources": bokeh_resources(widgets=True),
        "title": title,
        "components": format_report(df, cfg, mode, progress),
        "sample_note": sample_note,
    }
    template_base = ENV_LOADER.get_template("base.html")
    report = template_base.render(context=context)
    return Report(report)


def _suppress_warnings() -> None:
    """
    suppress warnings in create_report
    """
    warnings.filterwarnings(
        "ignore",
        "The default value of regex will change from True to False in a future version",
        category=FutureWarning,
    )
