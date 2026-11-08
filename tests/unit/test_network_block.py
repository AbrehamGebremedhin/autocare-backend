"""The shared no_network fixture really blocks non-loopback network use, at every layer."""
import asyncio
import socket

import httpx
import pytest

pytestmark = pytest.mark.usefixtures("no_network")


async def test_asyncio_open_connection_is_blocked(no_network):
    with pytest.raises(no_network.error):
        await asyncio.open_connection("192.0.2.1", 80)
    assert no_network and no_network[0] == "192.0.2.1"


async def test_httpx_is_blocked(no_network):
    with pytest.raises(no_network.error):
        async with httpx.AsyncClient() as client:
            await client.get("http://example.invalid")
    assert "example.invalid" in no_network


def test_socket_layers_are_blocked(no_network):
    s = socket.socket()
    try:
        with pytest.raises(no_network.error):
            s.connect(("192.0.2.1", 80))
        with pytest.raises(no_network.error):
            s.connect_ex(("192.0.2.1", 80))
    finally:
        s.close()
    with pytest.raises(no_network.error):
        socket.getaddrinfo("example.invalid", 80)
    assert len(no_network) == 3


async def test_loop_sock_connect_and_getaddrinfo_are_blocked(no_network):
    loop = asyncio.get_running_loop()
    sock = socket.socket()
    sock.setblocking(False)
    try:
        with pytest.raises(no_network.error):
            await loop.sock_connect(sock, ("192.0.2.1", 80))
    finally:
        sock.close()
    with pytest.raises(no_network.error):
        await loop.getaddrinfo("example.invalid", 80)


async def test_loopback_passes_through(no_network):
    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()
    finally:
        server.close()
        await server.wait_closed()
    assert no_network == []
