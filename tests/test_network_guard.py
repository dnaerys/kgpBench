"""The `live_mcp` marker, enforced — both halves, and the teardown driven.

The marked half is not asserted here and cannot be: a test that asserts the
guard is inert for a marked test either carries the marker without reaching the
server, which is the over-marking the marker exists to prevent, or does not
carry it and is testing the other half. What holds it is the marked tests
themselves — under a guard that ignored the marker they fail, which is a
mutation over this module rather than an assertion in it.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import network_guard
import pytest
from network_guard import NonLoopbackConnection

UNROUTABLE = "192.0.2.1"
"""TEST-NET-1 (RFC 5737), reserved for documentation and routed nowhere.

The address is the control's own subject: with the refusal removed these tests
must still reach no server, so the negative case cannot be the real endpoint.
"""


def test_an_unmarked_test_cannot_connect_to_a_non_loopback_address():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.05)
        with pytest.raises(NonLoopbackConnection) as refusal:
            sock.connect((UNROUTABLE, 9))
    assert UNROUTABLE in str(refusal.value)
    assert "live_mcp" in str(refusal.value)
    network_guard.forget()


def test_the_refusal_covers_connect_ex_as_well_as_connect():
    """`connect_ex` reports an error rather than raising one, so a guard that
    covered only `connect` would leave a second door open at the same address."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.05)
        with pytest.raises(NonLoopbackConnection) as refusal:
            sock.connect_ex((UNROUTABLE, 9))
    assert UNROUTABLE in str(refusal.value)
    network_guard.forget()


def test_a_host_name_is_refused_without_being_resolved():
    """The decision cannot wait for a resolver: asking one is already a step off
    this machine, so a name that is not a spelling of localhost is remote."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.05)
        with pytest.raises(NonLoopbackConnection) as refusal:
            sock.connect(("no-such-host.invalid", 9))
    assert "no-such-host.invalid" in str(refusal.value)
    network_guard.forget()


def test_loopback_is_available_to_an_unmarked_test():
    """A real connection over a real socket, not an assertion about the rule."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1) as client:
            accepted, _ = server.accept()
            with accepted:
                accepted.sendall(b"local")
                assert client.recv(5) == b"local"
    assert network_guard.attempted() == []


def test_an_ordinary_handler_does_not_swallow_the_refusal():
    """`except Exception` around a connection is ordinary code, and an MCP
    transport turns an error into tool-result text rather than raising it — so
    the refusal is outside the reach of both, as `pytest.fail` is."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.05)
        with pytest.raises(NonLoopbackConnection):
            try:
                sock.connect((UNROUTABLE, 9))
            except Exception:  # noqa: BLE001 - the handler is this test's subject
                pytest.fail("an ordinary handler caught the refusal")
    network_guard.forget()


def test_a_refusal_caught_anyway_survives_in_the_record():
    """The second half of the protection, for a handler wide enough to catch
    even a `BaseException`: the attempt is recorded before it is refused, and
    the record is what the teardown reads."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.05)
        try:
            sock.connect((UNROUTABLE, 9))
        except BaseException:  # noqa: BLE001 - the swallow is this test's subject
            pass
    recorded = network_guard.attempted()
    network_guard.forget()
    assert recorded == [f"{UNROUTABLE}:9"]
    assert UNROUTABLE in network_guard.refusal_message(recorded)


def test_a_test_that_swallows_the_refusal_still_fails(tmp_path: Path):
    """The fixture driven, in a pytest of its own.

    Nothing in this suite runs the teardown against a test that reached out —
    every control above clears the record, because it has to pass. So the
    assertion that a run actually goes red is made over a run: three tests in a
    scratch directory, all three swallowing everything they can. The unmarked
    one with an ordinary handler fails where it reached; the unmarked one that
    catches even the refusal passes its call and is caught by the teardown; and
    the marked one is the control that what fails is the marker's absence and
    not the address.

    The first is reported twice — once where it reached out and once by the
    teardown reading the record — because the two halves of the protection are
    independent and neither can tell that the other has already fired. Pinned as
    it is rather than tidied: the noise is one line and the alternative is a
    teardown that decides whether to speak.
    """
    (tmp_path / "conftest.py").write_text(
        "from network_guard import refuse_non_loopback_connections  # noqa: F401\n"
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers = live_mcp: reaches a server\n"
    )
    (tmp_path / "test_swallowed.py").write_text(
        textwrap.dedent(
            f"""
            import socket

            import pytest


            def _reach(handled):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.05)
                    try:
                        sock.connect(({UNROUTABLE!r}, 9))
                    except handled:
                        pass


            def test_unmarked_with_an_ordinary_handler_fails():
                _reach(Exception)


            def test_unmarked_catching_everything_is_caught_at_teardown():
                _reach(BaseException)


            @pytest.mark.live_mcp
            def test_marked_swallows_and_passes():
                _reach(BaseException)
            """
        )
    )
    run = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-p", "no:cacheprovider", "-q"],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(Path(network_guard.__file__).parent),
        },
        cwd=str(tmp_path),
        timeout=300,
    )
    assert run.returncode != 0, run.stdout
    assert "1 failed, 2 passed, 2 errors" in run.stdout, run.stdout
    assert "test_unmarked_with_an_ordinary_handler_fails" in run.stdout
    assert "test_unmarked_catching_everything_is_caught_at_teardown" in run.stdout
    assert "test_marked_swallows_and_passes" not in run.stdout
    assert UNROUTABLE in run.stdout
