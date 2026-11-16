"""The core features must work when the optional packages are not installed."""

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import sys

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {"dask", "flask", "flask_cors", "nltk", "ipywidgets"}:
                raise ImportError(f"{name} is blocked for this test")

    sys.meta_path.insert(0, Block())

    import pandas as pd
    from dataprep.clean import clean_au_abn, clean_country, clean_date, clean_email, clean_text
    from dataprep.datasets import get_dataset_names, load_dataset
    from dataprep.eda import create_report, plot, plot_correlation, plot_missing
    from dataprep.eda import rank_features, summarize

    df = pd.DataFrame({
        "email": ["a@b.com", "bad"] * 10,
        "country": ["Canada", "zzz"] * 10,
        "date": ["2021-01-02", "nope"] * 10,
        "abn": ["51824753556", "1"] * 10,
        "text": ["Hello, World!", "the cat"] * 10,
        "x": range(20),
        "y": [v * 1.5 for v in range(20)],
    })
    for func, col in [(clean_email, "email"), (clean_country, "country"), (clean_date, "date")]:
        assert func(df, col, report=False)[f"{col}_clean"].notna().sum() == 10
    assert clean_au_abn(df, "abn")["abn_clean"].notna().sum() == 10
    clean_text(df, "text")
    plot(df, progress=False).to_html()
    plot_correlation(df, progress=False).to_html()
    plot_missing(df, progress=False).to_html()
    create_report(df, progress=False)
    titanic = load_dataset("titanic")
    assert "titanic" in get_dataset_names()
    assert summarize(titanic, sample=100).attrs["dataset"]["rows"] == 100
    assert rank_features(titanic, "Survived").index[0] == "Sex"
    plot(titanic, target="Survived", progress=False).to_html()
    assert not any(m.split(".")[0] in {"dask", "flask", "nltk"} for m in sys.modules)
    print("ok")
    """
)


def test_core_features_without_optional_packages() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().endswith("ok")
