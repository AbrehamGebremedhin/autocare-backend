"""Fixtures shared by the unit tests."""
import asyncio
import ipaddress
import socket

import pytest


class NetworkBlocked(AssertionError):
    """A test tried to reach a non-loopback host."""


class _Attempts(list):
    error = NetworkBlocked


def _loopback(host) -> bool:
    if host is None or host == "":
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


@pytest.fixture
def no_network(monkeypatch):
    """Fail any non-loopback connect or name lookup, at the socket and the asyncio layers (the Windows loop connects
    through ConnectEx, not socket.connect). Loopback passes: the Windows loop's own self-pipe needs it.
    Returns the list of blocked hosts; `.error` is the exception class raised."""
    attempts = _Attempts()

    def check(host) -> None:
        if not _loopback(host):
            attempts.append(host.decode("ascii", "replace") if isinstance(host, bytes) else host)
            raise NetworkBlocked(f"network use in a unit test: {host!r}")

    def host_of(address):
        return address[0] if isinstance(address, tuple) else None  # a str address is an AF_UNIX path

    def sync_connect(real):
        def wrapper(self, address, *args, **kwargs):
            check(host_of(address))
            return real(self, address, *args, **kwargs)
        return wrapper

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, sync_connect(getattr(socket.socket, name)))

    real_gai = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        check(host)
        return real_gai(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    def loop_method(real, name):
        if name == "sock_connect":
            async def wrapper(self, sock, address):
                check(host_of(address))
                return await real(self, sock, address)
        elif name == "create_connection":
            async def wrapper(self, protocol_factory, host=None, *args, **kwargs):
                check(host)
                return await real(self, protocol_factory, host, *args, **kwargs)
        else:
            async def wrapper(self, host, *args, **kwargs):
                check(host)
                return await real(self, host, *args, **kwargs)
        return wrapper

    base = asyncio.base_events.BaseEventLoop
    for cls in (base, *_all_subclasses(base)):
        for name in ("create_connection", "sock_connect", "getaddrinfo"):
            if name in cls.__dict__:
                monkeypatch.setattr(cls, name, loop_method(cls.__dict__[name], name))
    return attempts
