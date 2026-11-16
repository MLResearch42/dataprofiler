"""
This module implements the Report class: the object returned by
``create_report`` and ``create_diff_report``.
"""

import sys
from pathlib import Path
from typing import Optional, Union

from ...utils import is_notebook
from ..container import _open_in_browser, notebook_iframe

__all__ = ["Report"]

#: Height in pixels of the scrollable frame a report is shown in inside a notebook.
NOTEBOOK_HEIGHT = 800


class Report:
    """A profiling report: one self-contained HTML document.

    In a notebook, leaving the object as the last expression of a cell displays
    the report in a scrollable frame. Nothing is loaded from the internet.

    Examples
    --------
    >>> from dataprep.eda import create_report
    >>> from dataprep.datasets import load_dataset
    >>> report = create_report(load_dataset("titanic"))
    >>> report.save("titanic_report.html")  # doctest: +SKIP
    >>> report.show_browser()  # doctest: +SKIP
    """

    def __init__(self, report: str) -> None:
        self.report = report

    def to_html(self) -> str:
        """The report as a complete, self-contained HTML document."""
        return self.report

    def _repr_html_(self) -> str:
        """Display the report inside a notebook."""
        return notebook_iframe(self.report, height=NOTEBOOK_HEIGHT, auto_height=False)

    def __repr__(self) -> str:
        return (
            f"<dataprep Report: {len(self.report) / 1e6:.1f} MB of HTML. "
            "Use .save(path), .show() or .show_browser()>"
        )

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the report as a standalone HTML file.

        Parameters
        ----------
        path
            Where to save the report. A directory gets a ``report.html`` inside
            it, a name without an extension gets ``.html`` appended, and with no
            path at all the report is written to ``report.html`` in the current
            working directory.

        Returns
        -------
        pathlib.Path
            The file that was written.
        """
        if path is None:
            target = Path.cwd() / "report.html"
        else:
            target = Path(path).expanduser()
            if target.is_dir():
                target = target / "report.html"
            elif target.suffix == "":
                target = target.with_name(target.name + ".html")
            elif target.suffix.lower() not in (".html", ".htm"):
                raise ValueError(
                    f"Format '{target.suffix}' is not supported (supported formats: html)"
                )

        target.write_text(self.report, encoding="utf-8")
        print(f"Report has been saved to {target}!")
        return target

    def show_browser(self) -> None:
        """
        Open the report in the system browser. Useful when calling from a terminal.
        """
        _open_in_browser(self.report)

    def show(self, height: int = NOTEBOOK_HEIGHT) -> None:
        """
        Display the report now. Useful inside a loop or a function, where the
        report is not the last expression of a notebook cell.

        Parameters
        ----------
        height
            Height of the scrollable frame in pixels.
        """
        if not is_notebook():
            print(
                "Not running in a notebook: use show_browser() to open the report in a "
                "browser, or save(path) to write it to an HTML file.",
                file=sys.stderr,
            )
            return
        from IPython.display import HTML, display  # pylint: disable=import-outside-toplevel

        display(HTML(notebook_iframe(self.report, height=height, auto_height=False)))
