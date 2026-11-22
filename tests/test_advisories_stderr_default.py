"""The launchers write a deck's advisories on stderr by default.

momwire#1673 sank them because SimNEC's stream (``- -``) never drained the
engine's stderr pipe and the engine blocked once ~4 KB of advisories filled
it. SimNEC drains it now, so every launcher route -- one-shot, stream, and a
deck the warm server answers -- writes each advisory the user's deck raises
to the launcher's stderr, within a per-process cap
(``momwire_serve_client.ADVISORIES_CAP``) that keeps a host which still never
drains the pipe from hanging. ``MOMWIRE_ADVISORIES=off`` is today's silence.

The served route is the one that used to lose them even when asked for: the
warm server's process raised them, and its stderr is the server log.
"""

from __future__ import annotations

import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import momwire_serve_client as mech
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
# Raises DuplicateWire (wires 2 and 3 are one wire, reversed) and nothing
# that the test filters hide: the advisory the brief's repro names.
DUP_DECK = (FIXTURES / "razor_printout_small" / "dup_reversed_rp_ne_pq.nec").read_text()
# momwire#1673's deck: ~1.9 KB of advisories per structure.
BURIED_DECK = (FIXTURES / "buried_grazing_1672" / "shallow_s40.nec").read_text()

SHELL = (
    "import sys; from momwire.dropin.nec5._shell import main; "
    "sys.exit(main(sys.argv[1:], basis='razor2p'))"
)
# bs2 raises the DuplicateWire alone on DUP_DECK (0.5 KB): five decks of it
# fit under the cap, where razor2p's far-mesh advisory makes one deck 1.7 KB.
STREAM_SHELL = SHELL.replace("razor2p", "bs2")
CLIENT = (
    "import sys, momwire_eznec_client as c; "
    "sys.argv[0] = 'momwire-nec5-razor2p'; sys.exit(c.main())"
)
DUPLICATE = b"DuplicateWire"
# Spelt out, not `ENV`: the switch is the user's contract.
ENV = "MOMWIRE_ADVISORIES"


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != ENV}
    # The pytest filters hide two advisories by their opening words; a
    # launcher run by a host has no such filter.
    env.pop("PYTHONWARNINGS", None)
    env.update(extra)
    return env


def _stream_of(deck: str, copies: int) -> str:
    body = deck.replace("\nEN\n", "\nNX\n")
    return body * (copies - 1) + deck


def _shell(args, stdin: str = "", **extra: str) -> subprocess.CompletedProcess:
    shell = STREAM_SHELL if args == ["-", "-"] else SHELL
    return subprocess.run(
        [sys.executable, "-c", shell, *args],
        input=stdin.encode(),
        capture_output=True,
        env=_env(**extra),
        timeout=600,
        check=False,
    )


# --------------------------------------------------------------------------
# one process: one-shot and stream
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_a_one_shot_writes_the_duplicate_wire_advisory_on_stderr(tmp_path):
    deck = tmp_path / "d.nec"
    deck.write_text(DUP_DECK)
    out = tmp_path / "d.out"
    r = _shell([str(deck), str(out)])
    assert r.returncode == 0
    assert r.stderr.count(DUPLICATE) >= 1, r.stderr[:600]
    assert "ANTENNA INPUT PARAMETERS" in out.read_text(encoding="latin-1")


@pytest.mark.integration
def test_off_writes_nothing_and_the_printout_is_the_same(tmp_path):
    deck = tmp_path / "d.nec"
    deck.write_text(DUP_DECK)
    loud, quiet = tmp_path / "loud.out", tmp_path / "quiet.out"
    r_loud = _shell([str(deck), str(loud)])
    r_quiet = _shell([str(deck), str(quiet)], **{ENV: "off"})
    assert r_quiet.returncode == 0 and r_quiet.stderr == b"", r_quiet.stderr[:400]
    assert DUPLICATE in r_loud.stderr
    assert loud.read_bytes() == quiet.read_bytes()
    # `stderr`, #1673's opt-in, is still read: it names the default now
    r_old = _shell([str(deck), str(tmp_path / "old.out")], **{ENV: "stderr"})
    assert DUPLICATE in r_old.stderr


@pytest.mark.integration
def test_a_stream_prints_each_decks_advisory_for_that_deck():
    """Python displays a warning once per (text, category, line) per
    process: without a per-deck reset the second identical deck of a stream
    showed nothing."""
    copies = 3
    r = _shell(["-", "-"], _stream_of(DUP_DECK, copies))
    assert r.returncode == 0
    assert r.stdout.count(b"ANTENNA INPUT PARAMETERS") == copies
    one = r.stderr.count(DUPLICATE) // copies
    assert one >= 1 and r.stderr.count(DUPLICATE) == one * copies, r.stderr[:800]
    quiet = _shell(["-", "-"], _stream_of(DUP_DECK, copies), **{ENV: "off"})
    assert quiet.stderr == b"" and quiet.stdout == r.stdout


@pytest.mark.integration
def test_a_long_stream_stops_at_the_cap_with_one_note():
    copies = 30
    r = _shell(["-", "-"], _stream_of(DUP_DECK, copies))
    assert r.returncode == 0
    assert r.stdout.count(b"ANTENNA INPUT PARAMETERS") == copies
    text = r.stderr.decode("utf-8", "replace")
    assert text.count(mech.ADVISORIES_CAPPED.strip()) == 1, text[-400:]
    assert text.endswith(mech.ADVISORIES_CAPPED.replace("\n", os.linesep))
    shown = text[: -len(mech.ADVISORIES_CAPPED)]
    assert 1 < shown.count("DuplicateWire") < copies
    assert mech.advisory_cost(shown) <= mech.ADVISORIES_CAP


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "linux", reason="F_SETPIPE_SZ is Linux's")
def test_an_undrained_4kb_stderr_pipe_never_stalls_a_stream():
    """momwire#1673's harness: SimNEC's stream, one structure at a time, each
    printout awaited before the next is sent, and a 4096-byte stderr pipe
    that nobody reads until the stream is over. The cap is what lets it
    finish now that advisories are on stderr."""
    import fcntl

    structures = 6
    read_end, write_end = os.pipe()
    fcntl.fcntl(write_end, fcntl.F_SETPIPE_SZ, 4096)
    assert fcntl.fcntl(write_end, fcntl.F_GETPIPE_SZ) == 4096
    proc = subprocess.Popen(
        [sys.executable, "-c", STREAM_SHELL, "-", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=write_end,
        env=_env(),
    )
    os.close(write_end)
    piece = BURIED_DECK.replace("\nEN\n", "\nNX\n")
    try:
        for index in range(structures):
            proc.stdin.write(piece.encode())
            proc.stdin.flush()
            seen = b""
            deadline = time.monotonic() + 120
            # A streamed structure's printout ends on its NX card's echo.
            while not re.search(rb"INPUT LINE +\d+ +NX[^\n]*\n\Z", seen):
                left = deadline - time.monotonic()
                assert left > 0, f"structure {index + 1} never answered"
                ready, _, _ = select.select([proc.stdout], [], [], left)
                if ready:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    assert chunk, "the engine closed its stdout"
                    seen += chunk
        proc.stdin.close()
        proc.stdout.read()
        assert proc.wait(timeout=120) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    with os.fdopen(read_end, "rb") as err:
        stderr = err.read()
    assert b"CoarseCrossingNode" in stderr
    assert len(stderr) <= mech.ADVISORIES_CAP + len(mech.ADVISORIES_CAPPED)


# --------------------------------------------------------------------------
# a deck the warm server answers
# --------------------------------------------------------------------------


def _listening(room: Path) -> list[str]:
    return [
        line
        for log in room.glob("*.log")
        for line in log.read_text(errors="replace").splitlines()
        if "listening pid=" in line
    ]


def _stop(room: Path) -> None:
    for line in _listening(room):
        for word in line.split():
            if word.startswith("pid="):
                try:
                    os.kill(int(word.split("=", 1)[1]), signal.SIGTERM)
                except (OSError, ValueError):
                    pass


@pytest.fixture
def room():
    room = Path(tempfile.mkdtemp(prefix="mw-"))
    try:
        yield room
    finally:
        _stop(room)
        import shutil

        shutil.rmtree(room, ignore_errors=True)


def _client(deck: Path, out: Path, room: Path, **extra: str):
    return subprocess.run(
        [sys.executable, "-c", CLIENT, str(deck), str(out)],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        env=_env(MOMWIRE_PORTAL_RUNTIME_DIR=str(room), **extra),
        timeout=300,
        check=False,
    )


@pytest.mark.integration
def test_a_served_deck_carries_its_advisories_to_the_clients_stderr(tmp_path, room):
    """Two launches: the first spawns the warm server, the second is served
    warm, and each deck's DuplicateWire reaches THAT client's stderr -- not
    the server log, where it went (or would have gone) before. The second
    is the case Python's once-per-location filter hid in a long-lived
    process."""
    deck = tmp_path / "d.nec"
    deck.write_text(DUP_DECK)
    one_shot = tmp_path / "one.out"
    assert _shell([str(deck), str(one_shot)], **{ENV: "off"}).returncode == 0
    for index in range(2):
        out = tmp_path / f"served-{index}.out"
        r = _client(deck, out, room)
        assert r.returncode == 0, r.stderr[:400]
        assert len(_listening(room)) == 1, "the deck was not served warm"
        assert DUPLICATE in r.stderr, (index, r.stderr[:600])
        assert out.read_bytes() == one_shot.read_bytes()
    # A deck served with the advisories turned off writes none, and its
    # printout is the same file.
    out = tmp_path / "off.out"
    r = _client(deck, out, room, **{ENV: "off"})
    assert r.returncode == 0 and r.stderr == b"", r.stderr[:400]
    assert out.read_bytes() == one_shot.read_bytes()
    assert len(_listening(room)) == 1


# --------------------------------------------------------------------------
# the mechanics
# --------------------------------------------------------------------------


def test_the_switch_reads_off_and_nothing_else_as_off():
    assert mech.ADVISORIES_ENV == ENV
    assert mech.advisories_on({})
    assert mech.advisories_on({ENV: "stderr"})
    assert mech.advisories_on({ENV: "1"})
    for off in ("off", "OFF", " 0 ", "no", "false", "none"):
        assert not mech.advisories_on({ENV: off}), off


def test_the_budget_writes_whole_pieces_then_one_note_then_nothing():
    budget = mech.AdvisoryBudget(cap=100)
    piece = "x" * 39 + "\n"  # costs 41
    assert budget.take(piece) == piece
    assert budget.take(piece) == piece
    assert budget.take(piece) == mech.ADVISORIES_CAPPED
    assert budget.take("y\n") == ""
    assert budget.used == 82


def test_the_cost_is_an_upper_bound_for_any_console():
    text = "Ω λ — ok\n"
    cost = mech.advisory_cost(text)
    for encoding in ("utf-8", "cp1252", "cp437", "latin-1"):
        encoded = text.encode(encoding, "backslashreplace").replace(b"\n", b"\r\n")
        assert len(encoded) <= cost, encoding


def test_the_cap_and_its_note_stay_inside_a_4kb_pipe_with_room():
    assert mech.ADVISORIES_CAP + mech.advisory_cost(mech.ADVISORIES_CAPPED) < 4096 - 768


def test_an_advisories_frame_splits_off_ahead_of_the_rest():
    body = "a.py:1: DuplicateWire: two\n".encode()
    answer = (
        mech.ADVISORIES_FRAME_START
        + body.replace(b"\n", b"\r\n")
        + mech.ADVISORIES_FRAME_END
        + b"PRINTOUT"
    )
    shown, rest = mech.split_advisories(answer)
    assert rest == b"PRINTOUT"
    assert mech.advisories_text(shown) == body.decode()
    assert mech.split_advisories(b"PRINTOUT") == (None, b"PRINTOUT")
    with pytest.raises(ValueError):
        mech.split_advisories(mech.ADVISORIES_FRAME_START + b"never closed")


def test_the_seam_frames_a_requested_decks_advisories_and_only_then():
    from momwire.dropin.nec5 import _shell

    seam = _shell.seam(basis="razor2p")
    asked = mech.ADVISORIES_REQUEST.decode("latin-1") + DUP_DECK
    framed, _ = seam.answer(asked, "")
    plain, _ = seam.answer(DUP_DECK, "")
    start = mech.ADVISORIES_FRAME_START.decode("latin-1").replace("\r\n", "\n")
    end = mech.ADVISORIES_FRAME_END.decode("latin-1").replace("\r\n", "\n")
    assert framed.startswith(start)
    head, _, rest = framed[len(start) :].partition(end)
    assert "DuplicateWire" in head
    assert rest == plain  # the printout is the same text
    # and again: a warm server's second identical deck is not deduplicated
    again, _ = seam.answer(asked, "")
    assert again == framed
