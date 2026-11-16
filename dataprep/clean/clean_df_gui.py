"""
A graphical user interface to clean a DataFrame with the functions of dataprep.clean.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

if TYPE_CHECKING:
    import dask.dataframe as dd

    from .gui.clean_gui import CleanGui

DISPLAY_MODES = ("auto", "browser", "iframe", "none")


def clean_df_gui(
    df: pd.DataFrame | dd.DataFrame,
    display: str = "auto",
    port: Optional[int] = None,
    host: str = "localhost",
) -> CleanGui:
    """
    Start the clean GUI for a DataFrame.

    The GUI is a web page served by a small web server that runs in a background
    thread of the current Python process. The address of the page is printed, so
    it can always be opened by hand. The server only accepts connections from
    this computer, and stops when the Python process (or notebook kernel) ends.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned. The
        DataFrame itself is not modified.
    display
        How to show the GUI.
            - 'browser': open the GUI in a new tab of the system web browser.
            - 'iframe': embed the GUI in the output of the notebook cell.
            - 'none': only print the address of the GUI.
            - 'auto': 'iframe' in Jupyter Notebook and JupyterLab, otherwise
              'browser' (VS Code notebooks, IPython, plain Python scripts).

        (default: 'auto')
    port
        The port of the web server. By default port 7680 is used, or a free
        port if 7680 is already taken.

        (default: None)
    host
        The network interface the web server listens on.

        (default: 'localhost')

    Returns
    -------
    A handle of the running GUI with:
        - ``url``: the address of the GUI,
        - ``df``: the DataFrame as currently shown in the GUI,
        - ``code``: Python code that reproduces the cleaning steps,
        - ``stop()``: stop the web server,
        - ``wait()``: block until interrupted (useful in a script).

    In a notebook the "Export" menu of the GUI also assigns the cleaned DataFrame
    to the variable ``cleaned_df`` and the code to the variable ``cleaning_code``.

    Examples
    --------
    >>> df = pd.DataFrame({"email": ["Abc@Example.com", "bad"]})
    >>> gui = clean_df_gui(df)  # doctest: +SKIP
    Clean GUI running at http://localhost:7680/
    >>> gui.df  # doctest: +SKIP
    >>> gui.stop()  # doctest: +SKIP
    """
    if display not in DISPLAY_MODES:
        raise ValueError(
            f'display {display} is invalid, it needs to be "auto", "browser", "iframe" or "none"'
        )

    # flask is only needed, and imported, when the GUI is used
    from .gui.clean_gui import launch  # pylint: disable=import-outside-toplevel

    gui = launch(df, host=host, port=port)
    print(f"Clean GUI running at {gui.url}")

    if display == "auto":
        display = _auto_display_mode()
    if display == "iframe":
        _display_iframe(gui.url)
    elif display == "browser":
        _open_browser(gui.url)
    return gui


def _notebook_shell() -> Any:
    """
    Return the IPython shell if running in a notebook kernel (Jupyter Notebook,
    JupyterLab, VS Code notebooks and interactive window), otherwise None.
    """
    try:
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    shell = get_ipython()
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return None
    return shell


def _in_vscode() -> bool:
    """
    Return True if the process was started by VS Code (which sets VSCODE_* variables
    in the environment of the notebook kernels and terminals it starts).
    """
    return any(name in os.environ for name in ("VSCODE_PID", "VSCODE_CWD", "VSCODE_IPC_HOOK_CLI"))


def _auto_display_mode() -> str:
    """
    Choose how to show the GUI: embedded in the cell output for Jupyter Notebook
    and JupyterLab, in the system web browser otherwise. VS Code notebooks do not
    reliably show pages of a local web server in a cell output.
    """
    if _notebook_shell() is not None and not _in_vscode():
        return "iframe"
    return "browser"


def _display_iframe(url: str) -> None:
    """Embed the GUI in the output of the current notebook cell."""
    from IPython.display import IFrame, display  # pylint: disable=import-outside-toplevel

    display(IFrame(url, width="100%", height=600))


def _open_browser(url: str) -> None:
    """Open the GUI in the system web browser, if there is one."""
    import webbrowser  # pylint: disable=import-outside-toplevel

    try:
        opened = webbrowser.open(url)
    except webbrowser.Error:
        opened = False
    if not opened:
        print("Could not open a web browser, open the address above by hand.")
