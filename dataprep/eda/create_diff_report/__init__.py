"""
    This module implements the create_diff_report([df1, df2]) function.
"""

import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from jinja2 import ChoiceLoader, Environment, PackageLoader

from ..configs import Config
from ..container import bokeh_resources
from ..create_report.report import Report
from ..diff import sample_each
from .diff_formatter import format_diff_report

__all__ = ["create_diff_report"]

ENV_LOADER = Environment(
    loader=ChoiceLoader(
        [
            PackageLoader("dataprep", "eda/create_diff_report/templates"),
            PackageLoader("dataprep", "eda/templates"),  # shared notebook-frame script
        ]
    ),
    autoescape=False,
)


def create_diff_report(
    df_list: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    title: Optional[str] = "DataPrep Report",
    mode: Optional[str] = "basic",
    progress: bool = True,
    *,
    sample: Union[int, float, None] = None,
) -> Report:
    """
    This function is to generate and render elements in a report object given multiple dataframes.

    Parameters
    ----------
    df_list
        The DataFrames for which data are calculated.
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
    progress
        Whether to show the progress bar.
    sample
        Compare random samples of the rows: a number of rows (int) or a fraction
        of the rows (float), taken from each DataFrame. The samples are the same
        on every run, and the report says that it shows samples.

    Examples
    --------
    >>> from dataprep.datasets import load_dataset
    >>> from dataprep.eda import create_diff_report
    >>> df_train = load_dataset('house_prices_train')
    >>> df_test = load_dataset('house_prices_test')
    >>> create_diff_report([df_train, df_test])  # doctest: +SKIP
    """
    # pylint: disable=too-many-arguments

    _suppress_warnings()
    cfg = Config.from_dict(display, config)
    df_list, sample_note = sample_each(df_list, sample)

    components = format_diff_report(df_list, cfg, mode, progress)

    dict_stats = defaultdict(list)

    for comps in components["dfs"]:
        for key, value in comps["overview"][0].items():
            if value is not None:
                dict_stats[key].append(value)

    context = {
        "resources": bokeh_resources(widgets=True),
        "title": title,
        "stats": dict_stats,
        "components": components,
        "is_diff_report": True,
        "df_labels": cfg.diff.label,
        "legend_labels": components["legend_lables"],
        "sample_note": sample_note,
    }

    # {% for div in value.plots[1] %}
    #             <div class="vp-plot">
    #                 {{ div }}
    #                 {% if key in context.components.dfs[1].variables %}
    #                 {{ context.components.dfs[1].variables[key].plots[1][loop.index0] }}
    #                 {% endif %}
    #             </div>

    # return context

    template_base = ENV_LOADER.get_template("base.html")
    report = template_base.render(context=context, zip=zip)
    return Report(report)


def _suppress_warnings() -> None:
    """
    suppress warnings in create_diff_report
    """
    warnings.filterwarnings(
        "ignore",
        "The default value of regex will change from True to False in a future version",
        category=FutureWarning,
    )
