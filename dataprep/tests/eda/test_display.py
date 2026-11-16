"""Tests for how plots and reports are turned into HTML and displayed."""

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ...datasets import load_dataset
from ...eda import (
    Container,
    Report,
    create_diff_report,
    create_report,
    plot,
    plot_correlation,
    plot_diff,
    plot_missing,
    set_bokeh_resources,
)
from ...eda.container import bokeh_resources, notebook_iframe
from ...eda.embed import components

# XML namespaces in inline SVG icons are not network requests.
ALLOWED_URL = re.compile(r"https?://www\.w3\.org/")


@pytest.fixture(scope="module")
def titanic() -> pd.DataFrame:
    return load_dataset("titanic")


@pytest.fixture(scope="module")
def outputs(titanic: pd.DataFrame) -> dict:
    half = len(titanic) // 2
    return {
        "plot": plot(titanic, progress=False),
        "plot_x": plot(titanic, "Age", progress=False),
        "plot_xy": plot(titanic, "Age", "Sex", progress=False),
        "correlation": plot_correlation(titanic, progress=False),
        "missing": plot_missing(titanic, progress=False),
        "diff": plot_diff([titanic.iloc[:half], titanic.iloc[half:]], progress=False),
        "report": create_report(titanic, progress=False),
        "diff_report": create_diff_report([titanic.iloc[:half], titanic.iloc[half:]], progress=False),
    }


def _document(obj: object) -> str:
    return obj.to_html()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "name", ["plot", "plot_x", "plot_xy", "correlation", "missing", "diff", "report", "diff_report"]
)
def test_outputs_are_offline_and_complete(outputs: dict, name: str) -> None:
    doc = _document(outputs[name])
    assert doc.lstrip().lower().startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in doc.lower()
    # BokehJS is embedded, not linked. Its source contains URLs in comments and
    # error messages; everything DataPrep adds around it must contain none.
    library = [res for res in (bokeh_resources(), bokeh_resources(widgets=True)) if res in doc]
    assert library, "BokehJS must be inlined"
    own = doc.replace(library[0], "")
    urls = [url for url in re.findall(r"https?://[^\s\"'<>)]+", own) if not ALLOWED_URL.match(url)]
    assert urls == [], f"{name} references the internet: {urls[:5]}"
    # nothing is loaded from a server
    assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)=[\"']?https?:", doc, re.I)


@pytest.mark.parametrize("name", ["plot", "plot_x", "report"])
def test_notebook_output_is_an_isolated_frame(outputs: dict, name: str) -> None:
    out = outputs[name]._repr_html_()
    assert out.startswith('<div class="dataprep-output"')
    match = re.search(r'srcdoc="([^"]*)"', out)
    assert match, "the document must be embedded with srcdoc"
    # the attribute round-trips to the exact standalone document
    assert html.unescape(match.group(1)) == _document(outputs[name])


def test_frame_resizes_and_forwards_scrolling(outputs: dict) -> None:
    doc = _document(outputs["plot"])
    assert "dataprepHeight" in doc  # auto height
    assert "WheelEvent" in doc  # wheel scrolling handed on to the notebook
    report = _document(outputs["report"])
    assert "var autoHeight = false" in report  # reports scroll inside a fixed frame


def test_iframe_escapes_quotes_and_ampersands() -> None:
    out = notebook_iframe('<p title="a&b">x</p>', auto_height=False)
    assert 'srcdoc="&lt;p title=&quot;a&amp;b&quot;&gt;x&lt;/p&gt;"' in out


def test_container_save_and_repr(outputs: dict, tmp_path: Path) -> None:
    container = outputs["plot_x"]
    assert isinstance(container, Container)
    container.save(tmp_path / "age")  # extension is added
    assert (tmp_path / "age.html").read_text(encoding="utf-8") == container.to_html()
    assert "dataprep plot" in repr(container)


def test_report_save_paths(outputs: dict, tmp_path: Path) -> None:
    report = outputs["report"]
    assert isinstance(report, Report)
    assert report.save(tmp_path) == tmp_path / "report.html"
    assert report.save(tmp_path / "titanic") == tmp_path / "titanic.html"
    with pytest.raises(ValueError, match="not supported"):
        report.save(tmp_path / "titanic.pdf")
    assert "MB of HTML" in repr(report)


def test_cdn_resources_are_opt_in(titanic: pd.DataFrame) -> None:
    try:
        set_bokeh_resources("cdn")
        doc = plot(titanic, "Age", progress=False).to_html()
        assert "cdn.bokeh.org" in doc
    finally:
        set_bokeh_resources("inline")
    assert "cdn.bokeh.org" not in plot(titanic, "Age", progress=False).to_html()
    with pytest.raises(ValueError):
        set_bokeh_resources("internet")


def test_fast_components_match_bokeh() -> None:
    from bokeh.embed import components as bokeh_components
    from bokeh.plotting import figure

    def figs() -> list:
        out = []
        for i in range(3):
            fig = figure(width=200, height=200)
            fig.line([0, 1], [0, i])
            out.append(fig)
        return out

    script, divs = components(figs())
    ref_script, ref_divs = bokeh_components(figs())
    assert len(divs) == len(ref_divs) == 3
    # model ids differ between runs; the structure must not
    def strip(text: str) -> str:
        return re.sub(r'"(id|root_id|docid|elementid)":"[^"]+"|p\d{4,}', "", text)

    assert len(strip(script)) == len(strip(ref_script))


def test_many_columns_render_in_linear_time() -> None:
    # 120 columns used to take minutes because serialization was quadratic
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.normal(size=(200, 120)), columns=[f"c{i}" for i in range(120)])
    import time

    start = time.perf_counter()
    plot(wide, progress=False).to_html()
    assert time.perf_counter() - start < 60
