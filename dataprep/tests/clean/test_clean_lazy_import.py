"""
Tests of the lazy loading of dataprep.clean and of its optional dependencies.
The checks run in fresh Python processes, because they depend on which modules
have been imported.
"""

import subprocess
import sys
import textwrap

import pytest

from ... import clean as clean_pkg

# third-party packages that must not be imported before they are needed
HEAVY = ["dask", "flask", "IPython", "ipywidgets", "pycrfsuite", "rapidfuzz", "metaphone"]
HEAVY += ["varname", "regex", "tqdm", "bokeh", "nltk"]


def run_python(code: str) -> str:
    """Run the code in a new interpreter and return what it printed."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def test_import_is_lazy() -> None:
    out = run_python(
        f"""
        import sys
        import dataprep.clean
        heavy = {HEAVY!r} + ["pandas", "numpy", "stdnum"]
        print(sorted(name for name in heavy if name in sys.modules))
        print(sorted(name for name in sys.modules if name.startswith("dataprep.clean.")))

        from dataprep.clean import clean_email, validate_email
        print(sorted(name for name in {HEAVY!r} + ["stdnum"] if name in sys.modules))
        print(sorted(name for name in sys.modules if name.startswith("dataprep.clean.")))
        """
    ).splitlines()
    # importing the package imports nothing else
    assert out[0] == "[]"
    assert out[1] == "[]"
    # importing one function only imports its module (and pandas)
    assert out[2] == "[]"
    assert out[3] == "['dataprep.clean.clean_email', 'dataprep.clean.utils']"


def test_functions_used_by_eda_are_light() -> None:
    out = run_python(
        f"""
        import sys
        import pandas as pd
        from dataprep.clean import validate_country, validate_lat_long
        head = pd.Series(["Canada", "41.5, -81.0", None, 3])
        print(validate_country(head).tolist(), validate_lat_long(head).tolist())
        print(sorted(name for name in {HEAVY!r} if name in sys.modules))
        """
    ).splitlines()
    assert out[0] == "[True, False, False, False] [False, True, False, False]"
    assert out[1] == "[]"


def test_star_import_and_pandas_input_do_not_import_optional_packages() -> None:
    out = run_python(
        f"""
        import sys
        import pandas as pd
        from dataprep.clean import *
        import dataprep.clean
        print(all(callable(globals()[name]) for name in dataprep.clean.__all__))
        df = pd.DataFrame({{"col": ["Abc@Example.com", "USA", "51824753556", None]}})
        clean_email(df, "col", report=False)
        clean_country(df, "col", report=False)
        clean_date(df, "col", report=False)
        clean_au_abn(df, "col")
        clean_text(df, "col")
        clean_df(df, report=False)
        print(sorted(name for name in {HEAVY!r} if name in sys.modules))
        """
    ).splitlines()
    assert out[0] == "True"
    # python-crfsuite is needed by the address detection of clean_df
    assert out[1] in ("['pycrfsuite']", "[]")


def test_attribute_is_the_function_even_if_the_module_was_imported_first() -> None:
    out = run_python(
        """
        import types
        import dataprep.clean.clean_ip
        from dataprep.clean.clean_au_abn import validate_au_abn
        import dataprep.clean as clean
        from dataprep.clean import clean_ip, clean_au_abn
        print(isinstance(clean.clean_ip, types.FunctionType), isinstance(clean_ip, types.FunctionType))
        print(isinstance(clean_au_abn, types.FunctionType), clean.validate_au_abn is validate_au_abn)
        print(isinstance(clean.utils, types.ModuleType), isinstance(clean.clean_date_utils, types.ModuleType))
        try:
            clean.no_such_name
        except AttributeError as error:
            print("AttributeError", error)
        """
    ).splitlines()
    assert out[0] == "True True"
    assert out[1] == "True True"
    assert out[2] == "True True"
    assert out[3] == "AttributeError module 'dataprep.clean' has no attribute 'no_such_name'"


def test_dir_and_all() -> None:
    names = dir(clean_pkg)
    assert set(clean_pkg.__all__) <= set(names)
    assert len(clean_pkg.__all__) == 497
    for name in clean_pkg.__all__:
        assert callable(getattr(clean_pkg, name)), name
        assert getattr(clean_pkg, name).__name__ == name
        assert getattr(clean_pkg, name).__doc__, name
    with pytest.raises(AttributeError):
        getattr(clean_pkg, "__no_such_dunder__")


def test_works_without_the_optional_packages() -> None:
    """
    Simulate an installation without the optional packages: the functions that do
    not need them work on pandas DataFrames, the others say what to install.
    """
    out = run_python(
        """
        import importlib.abc
        import sys

        # pandas is imported first: older pandas needs pytz itself, and only
        # DataPrep's own imports are under test.
        import pandas as pd

        ALREADY = {name.split(".")[0] for name in sys.modules}
        BLOCKED = {"dask", "flask", "flask_cors", "werkzeug", "ipywidgets", "IPython", "pycrfsuite",
                   "varname", "metaphone", "rapidfuzz", "regex", "tqdm", "pytz", "nltk", "bokeh"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        sys.meta_path.insert(0, Blocker())
        from dataprep.clean import *

        df = pd.DataFrame({"col": ["Abc@Example.com", "USA", "51824753556", "2003/09/25", None]})
        quiet = dict(report=False, progress=False)
        results = [
            clean_email(df, "col", **quiet), clean_phone(df, "col", **quiet),
            clean_country(df, "col", **quiet), clean_date(df, "col", **quiet),
            clean_lat_long(df, "col", **quiet), clean_ip(df, "col", **quiet),
            clean_url(df, "col", **quiet), clean_au_abn(df, "col", progress=False),
            clean_json(df, "col"), clean_text(df, "col"), clean_headers(df, report=False),
            clean_currency(df, "col", target_currency="eur", user_rate=2, **quiet),
            clean_df(df, report=False)[1], clean_ml(df, df, target="none")[0],
        ]
        print(all(isinstance(result, pd.DataFrame) for result in results))
        print(validate_country(df["col"]).tolist(), validate_au_abn(df["col"]).tolist())
        for call in (
            lambda: clean_address(df, "col", **quiet),
            lambda: clean_duplication(df, "col"),
            lambda: clean_df_gui(df, display="none"),
            lambda: clean_country(df, "col", fuzzy_dist=1, **quiet),
        ):
            try:
                call()
                print("no error")
            except ImportError as error:
                print("ImportError:", error)
        print(sorted((BLOCKED - ALREADY) & {name.split(".")[0] for name in sys.modules}))
        """
    ).splitlines()
    assert out[0] == "True"
    assert out[1] == "[False, True, False, False, False] [False, False, True, False, False]"
    assert "pip install python-crfsuite" in out[2]
    assert "dataprep-next[duplication]" in out[3]
    assert "pip install flask flask-cors" in out[4]
    assert "pip install regex" in out[5]
    assert out[6] == "[]"
