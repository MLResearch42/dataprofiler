"""
Shared fixtures of the dataprep.clean tests.
"""

import socket
from typing import Any, Iterator

import pytest

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


class NetworkAccessError(RuntimeError):
    """Raised when a test tries to reach a host other than this computer."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    The package must work on a closed network: fail any test that looks up or
    connects to a host other than this computer.
    """
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None and str(host) not in _LOCAL_HOSTS:
            raise NetworkAccessError(f"test tried to resolve the host {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(address, tuple) and str(host) not in _LOCAL_HOSTS:
            raise NetworkAccessError(f"test tried to connect to {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield
