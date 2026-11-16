"""
Tests of clean_currency() and validate_currency(). No test uses the network:
the online rate services are replaced by fakes.
"""

import socket
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_currency, validate_currency
from ...clean import utils as clean_utils
from ...clean.utils import RatesNotAvailableError

currency_module = sys.modules[clean_currency.__module__]


@pytest.fixture()
def df() -> pd.DataFrame:
    return pd.DataFrame({"money": ["$1,234.50", "12", "abc", None, 0, "-3.2", np.nan]})


def test_clean_default(df: pd.DataFrame, capsys: pytest.CaptureFixture) -> None:
    result = clean_currency(df, "money")
    expected = df.copy()
    expected["money_clean"] = [np.nan, 12.0, np.nan, np.nan, 0.0, -3.2, np.nan]
    assert expected.equals(result)
    out = capsys.readouterr().out
    assert "Currency Cleaning Report:" in out
    assert "3 values cleaned (42.86%)" in out
    assert "2 values unable to be parsed (28.57%), set to NaN" in out


def test_clean_ignore_symbol_and_representation(df: pd.DataFrame) -> None:
    result = clean_currency(df, "money", ignore_symbol="$", report=False)
    assert result["money_clean"].tolist()[0] == 1234.5
    result = clean_currency(
        df, "money", ignore_symbol=["$"], target_representation="abbr", n_round=1, report=False
    )
    assert result["money_clean"].tolist()[:2] == ["1,234.5", "12.0"]
    result = clean_currency(df, "money", errors="ignore", inplace=True, report=False)
    assert list(result.columns) == ["money_clean"]
    assert result["money_clean"].tolist()[:3] == ["$1,234.50", 12.0, "abc"]
    with pytest.raises(ValueError, match="Unable to parse value"):
        clean_currency(df, "money", errors="raise", report=False)


def test_clean_split_without_target(df: pd.DataFrame) -> None:
    result = clean_currency(df, "money", split=True, input_currency="EUR", report=False)
    assert list(result.columns) == ["money", "input_currency_symbol", "money_clean"]
    assert result["input_currency_symbol"].tolist()[1] == "€"
    assert result["money_clean"].tolist()[1] == 12.0
    result = clean_currency(df, "money", split=True, input_currency="Bitcoin", report=False)
    assert result["input_currency_symbol"].tolist()[1] == "BTC"


def test_user_rate_needs_no_network(df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the rate services must not be used when user_rate is given")

    monkeypatch.setattr(clean_utils, "_fetch_json", fail)
    # fiat to fiat: multiply by the rate
    result = clean_currency(df, "money", target_currency="eur", user_rate=0.5, report=False)
    assert result["money_clean"].tolist()[1] == 6.0
    # fiat to crypto: divide by the price of the coin
    result = clean_currency(
        df, "money", target_currency="bitcoin", user_rate=48, n_round=4, report=False
    )
    assert result["money_clean"].tolist()[1] == 0.25
    # crypto to fiat: multiply by the price of the coin
    result = clean_currency(
        df,
        "money",
        input_currency="Ethereum",
        target_currency="USD",
        user_rate=2000,
        target_representation="abbr",
        report=False,
    )
    assert result["money_clean"].tolist()[1] == "US$24,000.00"


def test_live_rates_use_the_rate_services(df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_fetch(host: str, request_path: str, timeout: Any = None) -> Any:
        calls.append({"host": host, "path": request_path})
        if host == currency_module.FIAT_BASE_URL:
            return {"rates": {"EUR": 0.91234567}}
        return {"bitcoin": {"usd": 50000}}

    monkeypatch.setattr(clean_utils, "_fetch_json", fake_fetch)
    result = clean_currency(df, "money", target_currency="EUR", report=False)
    assert result["money_clean"].tolist()[1] == 10.95  # 12 * 0.9123
    assert calls[-1] == {
        "host": "api.ratesapi.io",
        "path": "/api/latest?base=USD&symbols=EUR&rtype=fpy",
    }
    result = clean_currency(df, "money", target_currency="bitcoin", n_round=5, report=False)
    assert result["money_clean"].tolist()[1] == 0.00024
    assert calls[-1] == {
        "host": "api.coingecko.com",
        "path": "/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
    }
    # same currency: no request
    n_calls = len(calls)
    clean_currency(df, "money", target_currency="usd", report=False)
    assert len(calls) == n_calls

    monkeypatch.setattr(clean_utils, "_fetch_json", lambda *args, **kwargs: {"rates": {}})
    with pytest.raises(RatesNotAvailableError, match="not available"):
        clean_currency(df, "money", target_currency="EUR", report=False)


def test_unreachable_rate_service_gives_a_clear_error(
    df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline (here: the name lookup fails), the error says what to do."""

    def no_dns(*args: Any, **kwargs: Any) -> Any:
        raise socket.gaierror(8, "nodename nor servname provided, or not known")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    with pytest.raises(RatesNotAvailableError) as error:
        clean_currency(df, "money", target_currency="EUR", report=False)
    message = str(error.value)
    assert "Could not fetch live exchange rates from https://api.ratesapi.io" in message
    assert "user_rate" in message


def test_rate_service_requests_have_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client  # pylint: disable=import-outside-toplevel

    seen: Dict[str, Any] = {}

    class FakeConnection:
        def __init__(self, host: str, timeout: Any = None) -> None:
            seen["host"], seen["timeout"] = host, timeout

        def request(self, method: str, path: str) -> None:
            raise TimeoutError("timed out")

        def close(self) -> None:
            seen["closed"] = True

    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    with pytest.raises(RatesNotAvailableError, match="timed out"):
        clean_utils._get_rate("USD", "EUR", "rates.example")
    assert seen == {"host": "rates.example", "timeout": clean_utils.RATES_TIMEOUT, "closed": True}
    assert 0 < clean_utils.RATES_TIMEOUT <= 30


def test_invalid_parameters(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not a known currency"):
        clean_currency(df, "money", input_currency="xxx")
    with pytest.raises(ValueError, match="check your input and target currencies"):
        clean_currency(df, "money", target_currency="xxx", user_rate=1)
    with pytest.raises(ValueError, match="crypto to crypto"):
        clean_currency(df, "money", input_currency="bitcoin", target_currency="ethereum")
    with pytest.raises(ValueError, match="user_rate"):
        clean_currency(df, "money", target_currency="eur", user_rate=0)
    with pytest.raises(ValueError, match="target_representation"):
        clean_currency(df, "money", target_representation="words")
    with pytest.raises(ValueError, match="df is invalid"):
        clean_currency([1, 2], "money")


def test_validate(df: pd.DataFrame) -> None:
    assert validate_currency(df["money"]).tolist() == [False, True, False, False, True, True, False]
    assert validate_currency(df["money"], symbol="$").tolist()[0] is True
    assert validate_currency("1,000") is True
    assert validate_currency("€5", symbol=["€", "$"]) is True
    assert validate_currency("five") is False
    assert validate_currency("nan") is False
