"""
Tests of clean_df_gui and of the Flask backend of the clean GUI. The server is
only ever started on the loopback interface, on a free port.
"""

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_cors")

from ... import clean as clean_pkg  # noqa: E402  pylint: disable=wrong-import-position
from ...clean import clean_df_gui  # noqa: E402  pylint: disable=wrong-import-position
from ...clean.gui import clean_gui  # noqa: E402  pylint: disable=wrong-import-position


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Email Addr": ["Abc@Example.com", "bad", None],
            "abn": ["51824753556", "x", np.nan],
            "coord": ["41.5, -81.0", "nope", None],
            "n": [1, 2, 3],
        }
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def client() -> Iterator[Any]:
    """Flask test client of the backend, cleaning a fresh copy of the test DataFrame."""
    clean_gui.STATE.reset(_df())
    with clean_gui.app.test_client() as test_client:
        yield test_client


def _clean_column(client: Any, func: str, column: str, params: Dict[str, str]) -> Any:
    return client.post(
        "/cleanSingleCol",
        data={"clean_func": func, "selected_col": column, "selected_params": str(params)},
    )


def test_index_and_static_files(client: Any) -> None:
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '<div id=app>' in html
    scripts = [part.split(">")[0].split(" ")[0] for part in html.split("src=")[1:]]
    styles = [part.split(" ")[0].split(">")[0] for part in html.split("href=")[1:]]
    assert len(scripts) == 3 and len(styles) == 1
    for asset in scripts + styles:
        asset_response = client.get(asset)
        assert asset_response.status_code == 200
        assert len(asset_response.data) > 500
        asset_response.close()
    app_response = client.get([src for src in scripts if "/app." in src][0])
    app_js = app_response.get_data(as_text=True)
    app_response.close()
    # the frontend talks to the server that served it, whatever its port
    assert "window.location.origin" in app_js
    assert "7680" not in app_js


def test_init_sheet_is_valid_json_with_missing_values(client: Any) -> None:
    response = client.get("/getInitSheet")
    assert response.status_code == 200
    payload = json.loads(response.get_data(as_text=True))  # strict JSON, no NaN literals
    assert [col["colName"] for col in payload["tableColumns"]] == ["Email Addr", "abn", "coord", "n"]
    assert payload["tableColumns"][0] == {
        "colName": "Email Addr",
        "colLabel": "Email Addr",
        "colWidth": 180,
    }
    assert payload["tableData"][0] == {
        "Email Addr": "Abc@Example.com",
        "abn": "51824753556",
        "coord": "41.5, -81.0",
        "n": "1",
    }
    assert payload["tableData"][2]["abn"] == "nan"


def test_clean_columns_undo_redo_and_log(client: Any) -> None:
    response = _clean_column(
        client,
        "clean_email",
        "Email Addr",
        {"remove_whitespace": "True", "fix_domain": "False", "split": "False"},
    )
    assert response.status_code == 200
    table = response.get_json()
    assert [row["Email Addr_clean"] for row in table["tableData"]] == ["abc@example.com", "nan", "nan"]

    response = _clean_column(client, "clean_au_abn", "abn", {"output_format": "standard"})
    assert [row["abn_clean"] for row in response.get_json()["tableData"]] == [
        "51 824 753 556",
        "nan",
        "nan",
    ]

    response = _clean_column(client, "clean_lat_long", "coord", {"output_format": "dd", "split": "True"})
    assert response.get_json()["tableData"][0]["latitude"] == "41.5"

    response = _clean_column(client, "clean_headers", "", {"case": "const", "remove_accents": "True"})
    assert [col["colName"] for col in response.get_json()["tableColumns"]][:2] == ["EMAIL_ADDR", "ABN"]

    log = client.get("/getOperationLog").get_json()["operationLog"]
    assert len(log) == 4
    assert log[1] == "clean_au_abn(df, column='abn', output_format='standard', progress=False)"

    undone = client.get("/undo").get_json()
    assert "Email Addr" in [col["colName"] for col in undone["tableColumns"]]
    assert len(client.get("/getOperationLog").get_json()["operationLog"]) == 3
    redone = client.get("/redo").get_json()
    assert "EMAIL_ADDR" in [col["colName"] for col in redone["tableColumns"]]
    assert len(client.get("/getOperationLog").get_json()["operationLog"]) == 4

    # the exported code reproduces the DataFrame shown in the GUI
    code = client.get("/exportExecutionLog").get_json()["code"]
    namespace: Dict[str, Any] = {"df": _df()}
    namespace.update({name: getattr(clean_pkg, name) for name in clean_gui.clean_list})
    exec(code, namespace)  # pylint: disable=exec-used
    assert namespace["df"].equals(clean_gui.STATE.df)

    # a new step after an undo discards the undone steps
    client.get("/undo")
    client.get("/undo")
    _clean_column(client, "clean_au_abn", "abn", {"output_format": "compact"})
    assert len(client.get("/getOperationLog").get_json()["operationLog"]) == 3
    client.get("/redo")
    assert len(client.get("/getOperationLog").get_json()["operationLog"]) == 3

    original = client.get("/getOriginData").get_json()
    assert [col["colName"] for col in original["tableColumns"]] == ["Email Addr", "abn", "coord", "n"]
    assert client.get("/getOperationLog").get_json()["operationLog"] == []


def test_clean_column_errors_are_reported(client: Any) -> None:
    response = _clean_column(client, "clean_email", "no such column", {})
    assert response.status_code == 400
    assert "can not be cleaned" in response.get_json()["error"]
    # only the clean functions of the package can be called
    for func in ("eval", "os.system", "validate_email", ""):
        assert _clean_column(client, func, "n", {}).status_code == 400
    assert _clean_column(client, "clean_au_abn", "abn", {"output_format": "nope"}).status_code == 400
    assert client.get("/getOperationLog").get_json()["operationLog"] == []
    assert client.post("/getFunctionParams", json={"clean_func": "eval"}).status_code == 400


def test_clean_whole_dataframe_and_exports(client: Any) -> None:
    response = client.post(
        "/cleanWholeDF",
        data={
            "clean_headers": "True",
            "data_type_detection": "semantic",
            "standardize_missing_values": "fill",
            "downcast_memory": "False",
        },
    )
    assert response.status_code == 200
    assert [col["colName"] for col in response.get_json()["tableColumns"]] == [
        "email_addr",
        "abn",
        "coord",
        "n",
    ]
    # the DataFrame keeps its data types, values are only converted to text for display
    assert str(clean_gui.STATE.df["n"].dtype) == "Int64"

    response = client.post("/exportCSV")
    assert response.mimetype == "text/csv"
    assert response.get_data(as_text=True).splitlines()[:2] == [
        "email_addr,abn,coord,n",
        "Abc@Example.com,51824753556,\"41.5, -81.0\",1",
    ]

    assert client.get("/exportDF").get_json()["status"] == "success"
    assert clean_gui.STATE.exported_df.equals(clean_gui.STATE.df)

    response = client.post(
        "/cleanWholeDF",
        data={
            "clean_headers": "False",
            "data_type_detection": "none",
            "standardize_missing_values": "ignore",
            "downcast_memory": "False",
        },
    )
    assert response.status_code == 200


def test_file_upload(client: Any) -> None:
    import io  # pylint: disable=import-outside-toplevel

    response = client.post(
        "/fileUpload",
        data={"file": (io.BytesIO(b"x,y\n1,a\n2,\n"), "data.csv")},
        content_type="multipart/form-data",
    )
    assert response.get_json()["tableData"] == [{"x": "1", "y": "a"}, {"x": "2", "y": "nan"}]
    assert client.get("/getOperationLog").get_json()["operationLog"] == []


# clean_duplication has its own user interface and is not offered by the frontend
GUI_FUNCTIONS = [func for func in clean_gui.clean_list if func != "clean_duplication"]


@pytest.mark.parametrize("func", GUI_FUNCTIONS)
def test_function_params_offer_valid_options(func: str, client: Any) -> None:
    """Every option that the GUI offers for a parameter is accepted by the clean function."""
    response = client.post("/getFunctionParams", json={"clean_func": func})
    assert response.status_code == 200
    payload = response.get_json()
    assert [col["value"] for col in payload["tableColumns"]] == ["Email Addr", "abn", "coord", "n"]
    assert set(payload["paramDefault"]) == set(payload["paramDic"])

    if func in ("clean_df", "clean_currency"):
        return
    column = {"clean_lat_long": "coord", "clean_email": "Email Addr"}.get(func, "abn")
    for param, options in payload["paramDic"].items():
        assert payload["paramDefault"][param] in [option["value"] for option in options]
        for option in options:
            clean_gui.STATE.reset(_df())
            response = _clean_column(client, func, column, {param: option["value"]})
            assert response.status_code == 200, (param, option, response.get_json())


def test_server_on_a_free_port(capsys: pytest.CaptureFixture) -> None:
    """Start the real server, talk to it over HTTP, and stop it."""
    df = _df()
    port = _free_port()
    gui = clean_df_gui(df, display="none", port=port)
    try:
        assert gui.url == f"http://localhost:{port}/"
        assert f"Clean GUI running at {gui.url}" in capsys.readouterr().out
        assert gui.running
        with urllib.request.urlopen(gui.url, timeout=10) as response:
            assert response.status == 200
            assert b"<div id=app>" in response.read()
        with urllib.request.urlopen(gui.url + "getInitSheet", timeout=10) as response:
            assert len(json.loads(response.read())["tableData"]) == 3
        request = urllib.request.Request(
            gui.url + "cleanSingleCol",
            data=urllib.parse.urlencode(
                {
                    "clean_func": "clean_au_abn",
                    "selected_col": "abn",
                    "selected_params": "{'output_format': 'compact'}",
                }
            ).encode(),
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            assert json.loads(response.read())["tableData"][0]["abn_clean"] == "51824753556"
        assert gui.df["abn_clean"].tolist()[0] == "51824753556"
        # the frontend requests the log with a doubled slash
        with urllib.request.urlopen(gui.url + "/getOperationLog", timeout=10) as response:
            assert len(json.loads(response.read())["operationLog"]) == 1
        assert "clean_au_abn(df, column='abn'" in gui.code
        # the DataFrame of the user is not modified
        assert list(df.columns) == ["Email Addr", "abn", "coord", "n"]

        # starting the GUI again replaces the running one
        second = clean_df_gui(df, display="none", port=_free_port())
        assert not gui.running and second.running
        second.stop()
    finally:
        gui.stop()
    assert not gui.running
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(gui.url, timeout=2)  # pylint: disable=consider-using-with


def test_display_modes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import sys  # pylint: disable=import-outside-toplevel
    import webbrowser  # pylint: disable=import-outside-toplevel

    from ...clean import clean_df_gui as _  # noqa: F401  pylint: disable=import-outside-toplevel

    module = sys.modules[clean_df_gui.__module__]
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)

    # outside a notebook, and in VS Code notebooks, the system browser is used
    monkeypatch.setattr(module, "_notebook_shell", lambda: None)
    assert module._auto_display_mode() == "browser"
    monkeypatch.setattr(module, "_notebook_shell", lambda: object())
    monkeypatch.setenv("VSCODE_PID", "1234")
    assert module._auto_display_mode() == "browser"
    for name in ("VSCODE_PID", "VSCODE_CWD", "VSCODE_IPC_HOOK_CLI"):
        monkeypatch.delenv(name, raising=False)
    assert module._auto_display_mode() == "iframe"

    gui = clean_df_gui(_df(), display="browser", port=_free_port())
    try:
        assert opened == [gui.url]
        assert gui.url in capsys.readouterr().out
    finally:
        gui.stop()
    with pytest.raises(ValueError, match="display"):
        clean_df_gui(_df(), display="popup")
