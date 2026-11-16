"""
This module implements the Container class: the object returned by ``plot``,
``plot_correlation``, ``plot_missing`` and ``plot_diff``.

How the output is displayed
---------------------------
A container renders to one self-contained HTML document: the charts, the small
amount of CSS/JavaScript that drives the tabs, and the BokehJS library, all
inline. Nothing is fetched from the internet, so it works on closed networks.

In a notebook (Jupyter, JupyterLab, the VS Code notebook editor and the VS Code
interactive window) the document is shown inside an ``<iframe>``. The frame keeps
the styles and scripts of one output from leaking into the notebook or into
other outputs, which is what makes several plots in one notebook behave, and it
gives the charts a white background on dark editor themes.
"""

import html
import sys
import uuid
import webbrowser
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional, Union

from bokeh.resources import Resources
from jinja2 import Environment, PackageLoader

from ..utils import is_notebook
from .configs import Config
from .embed import components

__all__ = ["Container", "Context", "notebook_iframe", "bokeh_resources", "set_bokeh_resources"]

# The templates assemble trusted HTML fragments (bokeh scripts and divs), so
# auto-escaping is off; text that comes from the data is escaped with "|e".
ENV_LOADER = Environment(loader=PackageLoader("dataprep", "eda/templates"), autoescape=False)

TAB_VISUAL_TYPES = {
    "missing_impact_1v1",
    "missing_impact",
    "categorical_column",
    "geography_column",
    "numerical_column",
    "datetime_column",
    "cat_and_num_cols",
    "geo_and_num_cols",
    "latlong_and_num_cols",
    "two_num_cols",
    "two_cat_cols",
    "dt_and_num_cols",
    "dt_and_cat_cols",
    "dt_cat_num_cols",
    "correlation_impact",
    "correlation_single_heatmaps",
    "correlation_scatter",
    "comparison_continuous",
    "target_ranking",
}

GRID_VISUAL_TYPES = {"distribution_grid", "missing_impact_1vn", "comparison_grid"}

_RESOURCE_MODE = "inline"
_RESOURCE_CACHE: Dict[Any, str] = {}


def set_bokeh_resources(mode: str = "inline") -> None:
    """Choose where the BokehJS library used by the charts comes from.

    Parameters
    ----------
    mode
        ``"inline"`` (default) embeds BokehJS (about 1.4 MB) in every output, so
        plots work without internet access and still render when a notebook is
        reopened. ``"cdn"`` links to cdn.bokeh.org instead: outputs and saved
        notebooks are much smaller, but viewing them needs internet access.
    """
    global _RESOURCE_MODE  # pylint: disable=global-statement
    if mode not in ("inline", "cdn"):
        raise ValueError(f"mode must be 'inline' or 'cdn', got {mode!r}")
    _RESOURCE_MODE = mode


def bokeh_resources(widgets: bool = False) -> str:
    """The ``<script>`` tags that load BokehJS, limited to the parts DataPrep uses."""
    key = (_RESOURCE_MODE, widgets)
    if key not in _RESOURCE_CACHE:
        comps = ["bokeh", "bokeh-widgets"] if widgets else ["bokeh"]
        _RESOURCE_CACHE[key] = Resources(mode=_RESOURCE_MODE, components=comps).render()
    return _RESOURCE_CACHE[key]


def notebook_iframe(document: str, height: int = 500, auto_height: bool = True) -> str:
    """Wrap a complete HTML document in an ``<iframe>`` for display in a notebook.

    Parameters
    ----------
    document
        A full HTML document.
    height
        The height of the frame in pixels. With ``auto_height`` this is only the
        starting height.
    auto_height
        Let the frame grow and shrink with its content. The document has to
        include the ``frame.html`` template, which reports its height.
    """
    frame_id = f"dataprep-frame-{uuid.uuid4().hex[:12]}"
    # Only & and " need escaping inside a double quoted attribute.
    srcdoc = html.escape(document, quote=False).replace('"', "&quot;")
    # A <base target> makes links open in a new tab instead of inside the frame.
    listener = ""
    if auto_height:
        # Fallback for frontends where the frame cannot resize itself.
        listener = (
            "<script>(function(){"
            f"var f=document.getElementById('{frame_id}');"
            "if(!f)return;"
            "window.addEventListener('message',function(e){"
            "if(e.source===f.contentWindow&&e.data&&e.data.dataprepHeight){"
            "f.style.height=e.data.dataprepHeight+'px';}});"
            "})();</script>"
        )
    return (
        '<div class="dataprep-output" style="width:100%;">'
        f'<iframe id="{frame_id}" srcdoc="{srcdoc}" '
        f'style="width:100%;height:{int(height)}px;border:0;background:#fff;'
        'color-scheme:light;display:block;" '
        'allow="clipboard-write"></iframe>'
        f"{listener}</div>"
    )


def _open_in_browser(document: str) -> str:
    """Write a document to a temporary file and open it in the system browser."""
    # delete=False: the browser reads the file after this function has returned.
    with NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmpf:
        tmpf.write(document)
    webbrowser.open_new_tab(Path(tmpf.name).as_uri())
    return tmpf.name


class Container:
    """The result of a ``plot*`` call.

    In a notebook, leaving the object as the last expression of a cell displays
    it. Otherwise use one of the methods below.

    Examples
    --------
    >>> from dataprep.eda import plot
    >>> from dataprep.datasets import load_dataset
    >>> result = plot(load_dataset("titanic"), "Age")
    >>> result.show()             # display now, e.g. inside a loop  # doctest: +SKIP
    >>> result.save("age.html")   # a standalone, shareable HTML file  # doctest: +SKIP
    >>> result.show_browser()     # open in the system browser  # doctest: +SKIP
    """

    def __init__(self, to_render: Dict[str, Any], visual_type: str, cfg: Config) -> None:
        self.visual_type = visual_type
        self.context = Context(**to_render)
        self.context.rnd = uuid.uuid4().int % 10 ** 8
        if visual_type in GRID_VISUAL_TYPES:
            self.template_base = ENV_LOADER.get_template("grid_base.html")
        elif visual_type in TAB_VISUAL_TYPES:
            if visual_type == "missing_impact":
                self.context.highlight = False
            else:
                self.context.highlight = cfg.insight.enable
            if to_render.get("tabledata") or to_render.get("comparison_stats"):
                if "Stats" not in self.context.meta:
                    self.context.meta.insert(0, "Stats")
            if to_render.get("value_table"):
                self.context.meta.append("Value Table")
            if visual_type == "correlation_impact":
                self.template_base = ENV_LOADER.get_template("tab_base_corr.html")
            else:
                self.template_base = ENV_LOADER.get_template("tab_base.html")
        else:
            raise TypeError(f"Unsupported Visual Type: {visual_type}.")
        self._html: Optional[str] = None

    def to_html(self) -> str:
        """The plot as a complete, self-contained HTML document."""
        if self._html is None:
            self._html = self.template_base.render(context=self.context)
        return self._html

    def save(self, filename: Union[str, Path]) -> None:
        """Save the plot as a standalone HTML file.

        Parameters
        ----------
        filename
            Where to write the file. ``.html`` is appended when it is missing.
        """
        path = Path(filename).expanduser()
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_name(path.name + ".html")
        path.write_text(self.to_html(), encoding="utf-8")

    def __repr__(self) -> str:
        return (
            f"<dataprep plot ({self.visual_type}). "
            "Use .show(), .show_browser() or .save(path)>"
        )

    def _repr_html_(self) -> str:
        """Display itself inside a notebook."""
        return notebook_iframe(self.to_html(), height=self._initial_height())

    def _initial_height(self) -> int:
        """A first guess of the height; the frame adjusts itself once it is drawn."""
        return 620 if self.visual_type in GRID_VISUAL_TYPES else 520

    def show(self) -> None:
        """
        Display the plot now. Useful when plotting inside a loop or a function,
        where the result is not the last expression of a notebook cell.
        """
        if not is_notebook():
            print(
                "Not running in a notebook: use show_browser() to open the plot in a "
                "browser, or save(path) to write it to an HTML file.",
                file=sys.stderr,
            )
            return
        from IPython.display import HTML, display  # pylint: disable=import-outside-toplevel

        display(HTML(self._repr_html_()))

    def show_browser(self) -> None:
        """
        Open the plot in the system browser. Useful when plotting from a
        terminal or when the plot is very large.
        """
        _open_in_browser(self.to_html())


class Context:
    """
    The parameters handed to the template engine.

    The same templates render many kinds of plot, so most attributes are
    optional: reading an attribute that was never set gives ``None`` instead of
    raising, which lets a template simply skip the parts that do not apply.
    """

    _title = "DataPrep.EDA Report"
    _container_width = 650  # default width just in case nothing got passed in

    def __init__(self, **param: Any) -> None:
        self.title = self._title
        self.resources = bokeh_resources()
        self.container_width = self._container_width

        for attr, value in param.items():
            if attr == "layout":
                if len(value) == 0:
                    self.components = "", []
                else:
                    self.components = components(value)
            elif attr == "meta":
                value = list(value)
                setattr(self, attr, value)
                self.figs = [val for val in value if val != "Stats"]
            else:
                setattr(self, attr, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, None)

    def __getattr__(self, attr: str) -> None:
        # only reached when the attribute does not exist
        if attr.startswith("__"):
            raise AttributeError(attr)
        return None
