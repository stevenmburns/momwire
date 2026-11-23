"""The launchers' advisories on stderr, in MOMWIRE_ADVISORIES' three modes.

momwire#1673 sank every advisory because SimNEC's stream (``- -``) never
drained the engine's stderr pipe. 0.97.0 (de085684) wrote each deck's
advisories by default once SimNEC drained it, and that cost the Windows
drop-in 15-21 ms on every warm EZNEC launch, which writes the constant
RazorFarMeshClass advisory to a console nobody reads (momwire#2119). So:

====  ================================================================
off   the DEFAULT (unset, empty, unknown): nothing, as in 0.95.0
deck  each distinct advisory once per deck, under a divider
all   every occurrence, a deck's repeats included
====  ================================================================

In every writing mode a divider names the deck before its first advisory,
the per-process cap (``ADVISORIES_CAP``) holds, and the printout is the
same bytes as with the advisories off.
"""

from __future__ import annotations

import io
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import momwire_serve_client as mech
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
# Raises DuplicateWire (wires 2 and 3 are one wire, reversed) and nothing
# that the test filters hide: the advisory the brief's repro names.
DUP_PATH = FIXTURES / "razor_printout_small" / "dup_reversed_rp_ne_pq.nec"
DUP_DECK = DUP_PATH.read_text()
# The same deck at three frequencies: the solver, and its DuplicateWire, are
# made once per frequency, so ONE deck raises the same advisory three times.
DUP_DECK_3FR = DUP_DECK.replace("FR 0,1,0,0,299.7925", "FR 0,3,0,0,290,5")
# momwire#1673's deck: ~1.9 KB of advisories per structure, none of them a
# DuplicateWire (ExtendedKernelDefault, NearSurfaceWire, CoarseCrossingNode).
BURIED_DECK = (FIXTURES / "buried_grazing_1672" / "shallow_s40.nec").read_text()
# DUP_DECK without its copy: no advisory at all.
CLEAN_DECK = "".join(
    line for line in DUP_DECK.splitlines(keepends=True) if not line.startswith("GW 3,")
)

SHELL = (
    "import sys; from momwire.dropin.nec5._shell import main; "
    "sys.exit(main(sys.argv[1:], basis='razor2p'))"
)
# bs2 raises the DuplicateWire alone on DUP_DECK (0.5 KB): five decks of it
# fit under the cap, where razor2p's far-mesh advisory makes one deck 1.7 KB.
BS2_SHELL = SHELL.replace("razor2p", "bs2")
CLIENT = (
    "import sys, momwire_eznec_client as c; "
    "sys.argv[0] = 'momwire-nec5-razor2p'; sys.exit(c.main())"
)
DUPLICATE = b"DuplicateWire"
# Spelt out, not `mech.ADVISORIES_ENV`: the switch is the user's contract.
ENV = "MOMWIRE_ADVISORIES"
# The divider, spelt out for the same reason.
DIVIDER = "momwire: advisories for "


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != ENV}
    # The pytest filters hide two advisories by their opening words; a
    # launcher run by a host has no such filter.
    env.pop("PYTHONWARNINGS", None)
    env.update(extra)
    return env


def _structure(deck: str) -> str:
    return deck.replace("\nEN\n", "\nNX\n")


def _stream_of(*decks: str) -> str:
    """The decks as one ``- -`` stream: NX between them, EN at the end."""
    return "".join(_structure(d) for d in decks[:-1]) + decks[-1]


def _shell(args, stdin: str = "", shell: str = BS2_SHELL, **extra: str):
    return subprocess.run(
        [sys.executable, "-c", shell, *args],
        input=stdin.encode(),
        capture_output=True,
        env=_env(**extra),
        timeout=600,
        check=False,
    )


def _dividers(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(DIVIDER)]


def _assert_no_lonely_divider(text: str) -> None:
    """Every divider line is followed by an advisory, never by another
    divider, the cap's note or the end."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(DIVIDER):
            assert index + 1 < len(lines), text[-400:]
            after = lines[index + 1]
            assert not after.startswith(DIVIDER), text[-400:]
            assert not after.startswith("momwire: further advisories"), text[-400:]


# --------------------------------------------------------------------------
# one process: one-shot and stream
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_unset_empty_and_unknown_write_nothing_and_change_no_byte(tmp_path):
    """The default (momwire#2119): nothing on stderr, from a one-shot or a
    stream, and the printout is the one a writing mode gets."""
    deck = tmp_path / "d.nec"
    deck.write_text(DUP_DECK)
    loud = tmp_path / "loud.out"
    assert DUPLICATE in _shell([str(deck), str(loud)], **{ENV: "deck"}).stderr
    stream = _stream_of(DUP_DECK, DUP_DECK)
    loud_stream = _shell(["-", "-"], stream, **{ENV: "all"})
    assert DUPLICATE in loud_stream.stderr
    # `stream` was a mode in the first draft of momwire#2119; it is not one.
    unset = (None, "", "  ", "louder", "stream", "off", "NONE")
    for index, setting in enumerate(unset):
        extra = {} if setting is None else {ENV: setting}
        out = tmp_path / f"quiet-{index}.out"
        r = _shell([str(deck), str(out)], **extra)
        assert r.returncode == 0 and r.stderr == b"", (setting, r.stderr[:400])
        assert out.read_bytes() == loud.read_bytes()
        s = _shell(["-", "-"], stream, **extra)
        assert s.returncode == 0 and s.stderr == b"", (setting, s.stderr[:400])
        assert s.stdout == loud_stream.stdout


@pytest.mark.integration
def test_a_one_shot_writes_its_advisory_under_a_divider_naming_the_deck(tmp_path):
    deck = tmp_path / "d.nec"
    deck.write_text(DUP_DECK)
    for setting in ("deck", "all", "stderr", " ON "):
        r = _shell([str(deck), str(tmp_path / "d.out")], **{ENV: setting})
        text = r.stderr.decode()
        assert r.returncode == 0
        assert text.startswith(f"{DIVIDER}{deck}{os.linesep}"), (setting, text[:300])
        assert _dividers(text) == [f"{DIVIDER}{deck}"]
        assert text.count("DuplicateWire") == 1, setting


@pytest.mark.integration
def test_a_mixed_stream_puts_each_advisory_under_its_own_structure():
    """A stream mixing the duplicate-wire deck, a deck with none and the
    buried deck: a divider for each structure that writes something, never
    for the one that writes nothing, and each advisory under its own."""
    mixed = _stream_of(DUP_DECK, CLEAN_DECK, BURIED_DECK)
    m = _shell(["-", "-"], mixed, **{ENV: "deck"})
    assert m.returncode == 0
    assert m.stdout.count(b"ANTENNA INPUT PARAMETERS") == 3
    text = m.stderr.decode()
    assert _dividers(text) == [
        f"{DIVIDER}structure 1 of the stream",
        f"{DIVIDER}structure 3 of the stream",
    ], text
    first, third = text.split(f"{DIVIDER}structure 3 of the stream")
    assert first.count("DuplicateWire") == 1 and "ExtendedKernel" not in first
    # (the cap cuts the buried deck's last advisory here: 0.5 + 2.9 KB)
    assert third.count("ExtendedKernelDefault") == 1
    assert "DuplicateWire" not in third
    _assert_no_lonely_divider(text)
    assert _shell(["-", "-"], mixed).stdout == m.stdout


@pytest.mark.integration
def test_deck_shows_each_structures_advisory_under_its_own_divider():
    r = _shell(["-", "-"], _stream_of(DUP_DECK, DUP_DECK, DUP_DECK), **{ENV: "deck"})
    assert r.returncode == 0
    text = r.stderr.decode()
    assert _dividers(text) == [
        f"{DIVIDER}structure {n} of the stream" for n in (1, 2, 3)
    ], text
    for chunk in text.split(DIVIDER)[1:]:
        assert chunk.count("DuplicateWire") == 1, chunk[:300]


@pytest.mark.integration
def test_all_shows_a_decks_own_repeats_and_deck_shows_them_once(tmp_path):
    deck = tmp_path / "fr3.nec"
    deck.write_text(DUP_DECK_3FR)
    counts = {}
    printouts = set()
    for setting in ("deck", "all", "off"):
        out = tmp_path / f"{setting}.out"
        r = _shell([str(deck), str(out)], **{ENV: setting})
        assert r.returncode == 0
        counts[setting] = r.stderr.count(DUPLICATE)
        printouts.add(out.read_bytes())
        if setting != "off":
            assert len(_dividers(r.stderr.decode())) == 1
    assert counts == {"deck": 1, "all": 3, "off": 0}
    assert len(printouts) == 1


@pytest.mark.integration
def test_a_long_deck_mode_stream_stops_at_the_cap_with_one_note():
    copies = 30
    r = _shell(["-", "-"], _stream_of(*[DUP_DECK] * copies), **{ENV: "deck"})
    assert r.returncode == 0
    assert r.stdout.count(b"ANTENNA INPUT PARAMETERS") == copies
    text = r.stderr.decode("utf-8", "replace")
    assert text.count(mech.ADVISORIES_CAPPED.strip()) == 1, text[-400:]
    assert text.endswith(mech.ADVISORIES_CAPPED.replace("\n", os.linesep))
    shown = text[: -len(mech.ADVISORIES_CAPPED)]
    assert 1 < shown.count("DuplicateWire") < copies
    assert len(_dividers(shown)) == shown.count("DuplicateWire")
    assert mech.advisory_cost(shown) <= mech.ADVISORIES_CAP
    _assert_no_lonely_divider(text)


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "linux", reason="F_SETPIPE_SZ is Linux's")
@pytest.mark.parametrize("setting", ["deck", "all"])
def test_an_undrained_4kb_stderr_pipe_never_stalls_a_stream(setting):
    """momwire#1673's harness: SimNEC's stream, one structure at a time, each
    printout awaited before the next is sent, and a 4096-byte stderr pipe
    that nobody reads until the stream is over. The cap is what lets it
    finish in a mode that writes every structure's advisories."""
    import fcntl

    structures = 6
    read_end, write_end = os.pipe()
    fcntl.fcntl(write_end, fcntl.F_SETPIPE_SZ, 4096)
    assert fcntl.fcntl(write_end, fcntl.F_GETPIPE_SZ) == 4096
    proc = subprocess.Popen(
        [sys.executable, "-c", BS2_SHELL, "-", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=write_end,
        env=_env(**{ENV: setting}),
    )
    os.close(write_end)
    piece = _structure(BURIED_DECK)
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
    # Without the cap six structures write ~11 KB here: the pipe would fill.
    assert mech.ADVISORIES_CAPPED.encode() in stderr
    assert len(stderr) <= mech.ADVISORIES_CAP + len(mech.ADVISORIES_CAPPED)
    _assert_no_lonely_divider(stderr.decode())


_NEC2_STEP = (
    "CM stepped\nCE\nGW 1 10 0 0 -3 0 0 0 0.0005\nGW 2 10 0 0 0 0 0 3 0.001\n"
    "GE 0\nEX 0 1 10 0 1 0\nFR 0 1 0 0 14 0\nXQ\n"
)


@pytest.mark.integration
@pytest.mark.parametrize("setting", ["deck", "all"])
def test_the_nec2_stream_divides_its_structures_too(setting):
    """NEC-2's ``- -`` is a session over its own seam (``deck_scoped``):
    each structure is one deck of the process's advisories."""
    launcher = Path(sys.executable).parent / "momwire-nec2c-sinusoidal"
    if not launcher.exists():
        pytest.skip("momwire-nec2c-sinusoidal is not installed beside this interpreter")
    stream = f"{_NEC2_STEP}NX\n{_NEC2_STEP}EN\n"
    runs = {}
    for value in (setting, "off"):
        runs[value] = subprocess.run(
            [str(launcher), "-", "-"],
            input=stream.encode(),
            capture_output=True,
            env=_env(**{ENV: value}),
            timeout=300,
            check=True,
        )
    text = runs[setting].stderr.decode()
    assert _dividers(text) == [
        f"{DIVIDER}structure {n} of the stream" for n in (1, 2)
    ], text
    assert text.count("RadiusStepJunction") == 2
    assert runs["off"].stderr == b""
    untimed = [
        b"\n".join(
            ln
            for ln in r.stdout.splitlines()
            # wall-clock lines: FILL, and the EN's RUN TIME (momwire#2133)
            if b"FILL:" not in ln and not ln.startswith(b" RUN TIME =")
        )
        for r in runs.values()
    ]
    assert untimed[0] == untimed[1]


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
def test_served_decks_in_each_mode(tmp_path, room):
    """One warm server, launches in turn: ``off``, unset and unknown write
    nothing; ``deck`` and ``all`` write the deck's advisories on every
    launch, under the divider naming the deck; the printout is the
    one-shot's every time."""
    deck = tmp_path / "d.nec"
    deck.write_text(DUP_DECK)
    one_shot = tmp_path / "one.out"
    assert _shell([str(deck), str(one_shot)], shell=SHELL).returncode == 0
    plan = [None, "deck", "deck", "stream", "off", "all"]
    shown = []
    for index, setting in enumerate(plan):
        out = tmp_path / f"served-{index}.out"
        r = _client(deck, out, room, **({} if setting is None else {ENV: setting}))
        assert r.returncode == 0, r.stderr[:400]
        assert len(_listening(room)) == 1, "the deck was not served warm"
        assert out.read_bytes() == one_shot.read_bytes()
        text = r.stderr.decode()
        shown.append(text.count("DuplicateWire"))
        if setting in (None, "off", "stream"):
            assert r.stderr == b"", (setting, r.stderr[:300])
        if text:
            assert text.startswith(f"{DIVIDER}{deck}{os.linesep}"), text[:300]
            assert _dividers(text) == [f"{DIVIDER}{deck}"]
            assert "RazorFarMeshClass" in text  # razor2p's, constant text
    assert shown == [0, 1, 1, 0, 0, 1]


# --------------------------------------------------------------------------
# the mechanics
# --------------------------------------------------------------------------


def test_the_switch_reads_three_modes_and_anything_else_as_off():
    assert mech.ADVISORIES_ENV == ENV
    assert mech.ADVISORIES_MODES == ("off", "deck", "all")
    table = {
        "stream": "off",
        " Stream ": "off",
        "deck": "deck",
        "DECK": "deck",
        "stderr": "deck",
        "on": "deck",
        "1": "deck",
        "yes": "deck",
        "true": "deck",
        "all": "all",
        "\tALL\n": "all",
        "off": "off",
        "OFF": "off",
        " 0 ": "off",
        "no": "off",
        "false": "off",
        "none": "off",
        "": "off",
        "   ": "off",
        "louder": "off",
        "every": "off",
    }
    for value, mode in table.items():
        assert mech.advisories_mode({ENV: value}) == mode, value
        assert mech.advisories_on({ENV: value}) == (mode != "off"), value
    assert mech.advisories_mode({}) == "off"
    assert not mech.advisories_on({})
    assert mech.advisories_request("all") == b"\x00ADVISORIES all\n"


def test_the_divider_is_pinned_and_its_label_is_bounded():
    assert mech.stream_label(2) == "structure 2 of the stream"
    assert mech.advisories_divider("C:\\EZNEC\\x.nec") == (
        "momwire: advisories for C:\\EZNEC\\x.nec\n"
    )
    long = "/d" * 400 + "/deck.nec"
    line = mech.advisories_divider(long)
    label = line[len(DIVIDER) : -1]
    assert label.startswith("...") and long.endswith(label[3:])
    assert mech.advisory_cost(label) <= mech.ADVISORIES_LABEL_ROOM
    assert len(label) == mech.ADVISORIES_LABEL_ROOM
    # A served deck's divider rides outside the server's cap: still inside
    # a 4 KB pipe with the note.
    worst = mech.advisory_cost(DIVIDER + "x" * mech.ADVISORIES_LABEL_ROOM + "\n")
    assert (
        mech.ADVISORIES_CAP + worst + mech.advisory_cost(mech.ADVISORIES_CAPPED) < 4096
    )


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


def _capped(mode: str, cap: int = mech.ADVISORIES_CAP):
    sink = io.StringIO()
    return sink, mech.CappedStderr(sink, mech.AdvisoryBudget(cap), mode=mode)


A = "a.py:1: DuplicateWire: wires 2 and 3\n  x\n"
B = "b.py:2: CoarseCrossingNode: node 4\n  y\n"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("deck", "D1 A B D2 A C D3 A"),
        ("all", "D1 A A B D2 A C D3 A"),
    ],
)
def test_the_policy_remembers_what_its_mode_says_and_divides_decks(mode, expected):
    """Three decks: (A, A, B), (A, C), (A). A divider before each deck's
    first written advisory, and none for a deck that writes nothing."""
    C = "c.py:3: NearSurfaceWire: wire 1\n  z\n"
    sink, err = _capped(mode)
    names = {A: "A", B: "B", C: "C"}
    for number, deck in enumerate(((A, A, B), (A, C), (A,)), start=1):
        err.begin_deck(mech.stream_label(number))
        for text in deck:
            err.write(text)
    got = []
    for line in sink.getvalue().splitlines():
        if line.startswith(DIVIDER):
            got.append("D" + line.split()[-4])
        elif not line.startswith(" "):
            got.append(next(n for t, n in names.items() if t.startswith(line)))
    assert " ".join(got) == expected


def test_a_divider_and_its_advisory_are_one_piece_of_the_cap():
    divider = mech.advisories_divider(mech.stream_label(1))
    # Room for the advisory alone, not with its divider: the note, no divider.
    sink, err = _capped("deck", cap=mech.advisory_cost(A) + 1)
    err.begin_deck(mech.stream_label(1))
    err.write(A)
    assert sink.getvalue() == mech.ADVISORIES_CAPPED
    # Room for both: both, in one write.
    sink, err = _capped("deck", cap=mech.advisory_cost(divider + A))
    err.begin_deck(mech.stream_label(1))
    err.write(A)
    err.write(B)
    assert sink.getvalue() == divider + A + mech.ADVISORIES_CAPPED


def _raise_twice():
    for _ in range(2):
        warnings.warn("twice from one line", UserWarning, stacklevel=1)
    warnings.warn("ignored by a caller", UserWarning, stacklevel=1)


def test_all_mode_shows_every_occurrence_and_keeps_ignore_filters():
    from momwire.dropin.nec5 import _shell

    before = list(warnings.filters)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ignored by a caller")
        outer = list(warnings.filters)
        got = {}
        for mode in ("deck", "all"):
            with _shell._deck_advisories(mode, record=True) as caught:
                _raise_twice()
            got[mode] = [str(w.message) for w in caught]
            assert warnings.filters == outer, mode
    assert warnings.filters == before
    assert got["deck"] == ["twice from one line"]
    assert got["all"] == ["twice from one line"] * 2


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


def test_a_launcher_divides_a_frame_only_when_it_holds_an_advisory():
    raw = A.replace("\n", "\r\n").encode()
    assert mech.served_advisories(raw, "x.nec") == f"{DIVIDER}x.nec\n{A}"
    assert mech.served_advisories(b"", "x.nec") == ""
    capped = mech.ADVISORIES_CAPPED.replace("\n", "\r\n").encode()
    assert mech.served_advisories(capped, "x.nec") == mech.ADVISORIES_CAPPED


_START = mech.ADVISORIES_FRAME_START.decode("latin-1").replace("\r\n", "\n")
_END = mech.ADVISORIES_FRAME_END.decode("latin-1").replace("\r\n", "\n")


def _asked(mode: str, deck: str = DUP_DECK) -> str:
    return mech.advisories_request(mode).decode("latin-1") + deck


def _frame(out: str) -> tuple[str, str]:
    assert out.startswith(_START), out[:80]
    head, _, rest = out[len(_START) :].partition(_END)
    return head, rest


def test_the_seam_frames_a_requested_decks_advisories_and_only_then():
    from momwire.dropin.nec5 import _shell

    seam = _shell.seam(basis="razor2p")
    plain, _ = seam.answer(DUP_DECK, "")
    for mode in ("deck", "all"):
        framed, _ = seam.answer(_asked(mode), "")
        head, rest = _frame(framed)
        assert "DuplicateWire" in head
        assert rest == plain  # the printout is the same text
        # and again: the seam alone remembers nothing between decks
        assert seam.answer(_asked(mode), "")[0] == framed
    three = _shell.seam(basis="razor2p").answer(_asked("all", DUP_DECK_3FR), "")[0]
    assert _frame(three)[0].count("DuplicateWire") == 3
    once = _shell.seam(basis="razor2p").answer(_asked("deck", DUP_DECK_3FR), "")[0]
    assert _frame(once)[0].count("DuplicateWire") == 1


def test_the_advisories_frame_follows_a_times_frame():
    from momwire.dropin.nec5 import _shell

    seam = _shell.seam(basis="razor2p")
    plain = seam.answer(_asked("deck"), "")[0]
    timed = seam.answer(mech.TIMES_REQUEST.decode("latin-1") + _asked("deck"), "")[0]
    t_start = mech.TIMES_FRAME_START.decode("latin-1").replace("\r\n", "\n")
    t_end = mech.TIMES_FRAME_END.decode("latin-1").replace("\r\n", "\n")
    assert timed.startswith(t_start)
    after = timed[timed.index(t_end) + len(t_end) :]
    # (a timed printout carries its own MOMWIRE_TIMES section)
    assert _frame(after)[0] == _frame(plain)[0]


def test_the_seams_frame_is_held_to_one_processs_cap():
    """A served deck's frame is the launcher's whole output for that deck:
    the seam caps it, whole advisories and then the note."""
    from momwire.dropin.nec5 import _shell

    item = "a" * 1000
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            warnings.warn(item, UserWarning, stacklevel=1)
    head = _shell.advisories(caught, "all")
    one = warnings.formatwarning(
        caught[0].message, caught[0].category, caught[0].filename, caught[0].lineno
    )
    fits = mech.ADVISORIES_CAP // mech.advisory_cost(one)
    assert 1 < fits < 5
    assert head == one * fits + mech.ADVISORIES_CAPPED
    assert _shell.advisories(caught, "deck") == one
