"""The `live_mcp` marker, enforced while the tests run.

A test that carries the marker reaches the open OneKGPd server; every other test
in this tree runs offline, and this module is what makes that a property of the
run rather than a claim about it. It patches `socket.socket.connect` and
`socket.socket.connect_ex` for the duration of an unmarked test and refuses any
address that is not loopback, naming the host.

**Why a run-time refusal rather than a list.** Which tests connect cannot be read
off the source: a test can call `eval()` over a task built on the live URL and
never connect, because the task body refuses first, and some tests that do
connect **pass without a server**, because they assert on a header a failed run
still writes. So neither the marker nor a static rule over the source derives
from the other, and the fact was carried here as a measurement — a literal set of
test names, re-measured by hand whenever it moved. The refusal below needs no
such list: it observes the connection itself, which is the thing the marker is
about (`specification.md` §9, §16).

**The refusal is a `BaseException`, for the reason `pytest.fail` is one.** An
MCP transport surfaces an error to the model as tool-result text rather than
raising it, and an `except Exception` around a connection is ordinary code — so
a refusal an ordinary handler can catch would leave a green test that had
reached off the machine. Deriving from `BaseException` puts it outside every
such handler, and the test fails where the attempt was made.

**And the attempt is recorded before it is refused**, so that a handler wide
enough to catch even this one cannot hide it: the fixture fails the test at
teardown on anything left in the record.

**Loopback stays available to every test**, marked or not: a fixture that binds a
local port is not what the marker is about.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from typing import Any

import pytest

__all__ = [
    "NonLoopbackConnection",
    "attempted",
    "forget",
    "refusal_message",
    "refuse_non_loopback_connections",
]

LIVE_MCP_MARKER = "live_mcp"
"""The marker a test carries to be allowed off this machine."""


class NonLoopbackConnection(BaseException):
    """A test without the marker tried to open a connection off this machine.

    A `BaseException` and not an `Exception`, which is what keeps it out of the
    reach of ordinary error handling — the same reason `pytest.fail` raises one.
    """


_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
"""Captured at import, before anything has had a chance to patch them."""

_ATTEMPTED: list[str] = []


def attempted() -> list[str]:
    """Every non-loopback address the current test has reached for, in order."""
    return list(_ATTEMPTED)


def forget() -> None:
    """Drop the record. For a test whose own subject is the record itself."""
    _ATTEMPTED.clear()


def refusal_message(attempts: list[str]) -> str:
    """What the operator reads when an unmarked test reached off the machine."""
    return (
        f"this test opened, or tried to open, a connection to {', '.join(attempts)}. "
        f"Only a test marked `{LIVE_MCP_MARKER}` may leave this machine — mark it if "
        "reaching the server is what it is for, and point it at a local fixture "
        "otherwise. Loopback is available to every test"
    )


def _is_loopback(host: object) -> bool:
    """Whether `host` names this machine.

    A name that is not an address literal is remote unless it spells localhost,
    which is the safe direction: a name can only be resolved by asking a
    resolver, and the answer arrives too late to be the thing this decides. The
    empty host is the local one every `bind`-then-`connect` fixture produces.
    """
    text = host.decode() if isinstance(host, bytes) else str(host)
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return text in ("", "localhost", "localhost.localdomain")


def _refuse(address: Any) -> None:
    """Record and refuse, unless `address` is local to this machine.

    An `AF_UNIX` address is a path rather than a pair and never leaves the
    machine, so only a `(host, port, ...)` tuple is examined.
    """
    if not isinstance(address, tuple) or not address:
        return
    host = address[0]
    if _is_loopback(host):
        return
    target = f"{host}:{address[1]}" if len(address) > 1 else str(host)
    _ATTEMPTED.append(target)
    raise NonLoopbackConnection(refusal_message([target]))


def _guarded_connect(self: socket.socket, address: Any) -> None:
    _refuse(address)
    return _ORIGINAL_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    _refuse(address)
    return _ORIGINAL_CONNECT_EX(self, address)


@pytest.fixture(autouse=True)
def refuse_non_loopback_connections(request: pytest.FixtureRequest) -> Iterator[None]:
    """Hold an unmarked test to loopback, and fail it loudly if it reaches out.

    Imported into a conftest, where being autouse makes it cover every test in
    that tree. Restoring leaves the two methods as explicit attributes of
    `socket.socket` holding what `_socket.socket` defines, which is what they
    called before and what they call after.
    """
    if LIVE_MCP_MARKER in request.keywords:
        yield
        return
    forget()
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = _ORIGINAL_CONNECT  # type: ignore[method-assign]
        socket.socket.connect_ex = _ORIGINAL_CONNECT_EX  # type: ignore[method-assign]
        reached = attempted()
        forget()
    if reached:
        pytest.fail(refusal_message(reached), pytrace=False)
