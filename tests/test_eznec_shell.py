"""The EZNEC seam's process shell and printout header (momwire#497, U1).

Two kinds of gate here, and they check different halves of the same claim.

**Byte gates** compare :func:`momwire.eznec.render_header` against the top of
the captured Windows printouts under ``tests/fixtures/eznec/`` — the user's
own licensed engine's output, kept verbatim.  The expected prefix is always
SLICED OUT of the fixture at the ``- - - STRUCTURE SPECIFICATION - - -``
heading, never transcribed into this file: a hand-copied banner would gate
the transcription, not the engine.  Comparison is after the manifest's
CRLF-to-LF normalization.

**Protocol gates** run ``python -m momwire.eznec`` as a real process, because
every obligation in the fault-injection table is about what a PROCESS leaves
behind: the file exists, the stamp echo is in it, the refusal is after the
echo, the exit code says nothing, stdin is untouched.  None of that can be
observed from inside the interpreter that rendered the string.

Nothing here asserts anything about stderr.  EZNEC never reads it (the
captures record it as 0 bytes), so it is not a channel and pinning it would
invent a contract the far side does not have.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import momwire.eznec as eznec
from momwire.eznec._shell import ARGUMENT_ERROR_INPUT, ARGUMENT_ERROR_OUTPUT

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "eznec"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text())

# The captures that shipped a printout as well as a deck — the byte-gate set.
GATED_IDS = tuple(
    entry["id"] for entry in MANIFEST["captures"] if entry.get("printout")
)

# The heading that begins U3's territory; the header ends at the newline
# before it.
_STRUCTURE = "- - - STRUCTURE SPECIFICATION - - -"

# The refusal frame, as the fault-injection experiment observed it reaching the
# operator.  U1 filled it with one stub reason for every deck; since U4 the
# reason names the card or the rung, so the gates below match on the FRAME and
# leave the sentence to the units that own it.
_ERROR_PREFIX = " ***** NEC ERROR - "
_ERROR_LINE = f"{_ERROR_PREFIX}{eznec.STUB_REFUSAL}"


def capture(cid: str) -> dict:
    for entry in MANIFEST["captures"]:
        if entry["id"] == cid:
            return entry
    raise AssertionError(f"no capture {cid} in the manifest")


def deck_text(cid: str) -> str:
    return (FIXTURE_DIR / capture(cid)["deck"]).read_bytes().decode("latin-1")


def printout_text(cid: str) -> str:
    """A captured printout, CRLF-normalized per the manifest."""
    raw = (FIXTURE_DIR / capture(cid)["printout"]).read_bytes().decode("latin-1")
    return raw.replace("\r\n", "\n")


def header_prefix(cid: str) -> str:
    """The captured printout down to (not including) the structure heading."""
    text = printout_text(cid)
    at = text.index(_STRUCTURE)
    return text[: text.rindex("\n", 0, at) + 1]


def run_engine(args, *, cwd=None, stdin=subprocess.DEVNULL, input=None):
    """``python -m momwire.eznec`` in a child process.

    The child is pointed at the SAME momwire the parent imported (an editable
    checkout, a worktree on ``PYTHONPATH``, or an installed wheel) by deriving
    the import root from the package's own file.
    """
    src_root = str(Path(eznec.__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [src_root, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    return subprocess.run(
        [sys.executable, "-m", "momwire.eznec", *args],
        cwd=None if cwd is None else str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        **({"input": input} if input is not None else {"stdin": stdin}),
    )


# --------------------------------------------------------------------------
# byte gates: the header is the captured engine's, column for column
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cid", GATED_IDS)
def test_header_matches_the_captured_printout_byte_for_byte(cid):
    """Every capture that shipped a printout gates its own header.

    This is the whole of U1's byte contract: the form-feed line, the build
    tag, the NEC-5 banner box, the comment box with the deck's ``CM`` cards
    echoed into it, and the blank lines down to the structure heading.
    """
    assert eznec.render_header(deck_text(cid)) == header_prefix(cid)


def test_the_two_named_gates_have_different_comment_blocks():
    """0043 and 0010 differ in title AND launch timestamp — so a header that
    matched both cannot be replaying one fixture's bytes.

    The echo is what EZNEC checks a printout's provenance by, so "tracks the
    deck" is the property that matters, not "matches a captured file".
    """
    assert header_prefix("0043") != header_prefix("0010")
    assert eznec.render_header(deck_text("0043")) == header_prefix("0043")
    assert eznec.render_header(deck_text("0010")) == header_prefix("0010")


def test_the_echo_carries_the_launch_stamp_and_the_title():
    header = eznec.render_header(deck_text("0043"))
    assert "EZNEC Pro/2+ v. 7.0.4  2026-08-20 08:14:05" in header
    assert "Vertical over real ground" in header


def test_a_five_card_comment_block_echoes_six_lines():
    """``CE`` echoes too, as its own blank line — five ``CM`` cards, six lines.

    Pinned because it is the one counting rule in the header that a reader
    would otherwise get wrong, and getting it wrong shifts every byte after
    it.
    """
    cards = eznec._printout.comment_cards(deck_text("0043"))
    assert len(cards) == 6
    assert cards[0] == " Vertical over real ground"
    assert cards[-1] == ""


def test_every_echoed_line_is_padded_to_the_observed_width():
    """The echo field is 103 columns wide including trailing blanks, and an
    empty ``CM`` prints as 103 spaces rather than as an empty line."""
    lines = eznec.render_header(deck_text("0043")).split("\n")
    rule = " " * 22 + "*" * 58
    top, bottom = lines.index(rule), len(lines) - 1 - lines[::-1].index(rule)
    echoed = lines[top + 2 : bottom - 1]
    assert len(echoed) == 6
    assert {len(line) for line in echoed} == {103}
    assert echoed[1] == " " * 103


def test_the_header_stops_before_the_structure_heading():
    header = eznec.render_header(deck_text("0043"))
    assert _STRUCTURE not in header
    assert header.endswith("\n\n\n\n\n\n")
    assert printout_text("0043").startswith(header)


def test_the_header_is_written_with_lf_endings():
    assert "\r" not in eznec.render_header(deck_text("0043"))


# --------------------------------------------------------------------------
# the refusal frame
# --------------------------------------------------------------------------


def test_a_refusal_is_the_header_then_one_error_line():
    deck = deck_text("0043")
    refusal = eznec.render_refusal(deck, eznec.STUB_REFUSAL)
    header = eznec.render_header(deck)
    assert refusal.startswith(header)
    assert refusal[len(header) :] == f"\n{_ERROR_LINE}\n"


def test_a_refusal_with_no_deck_has_no_comment_box():
    """The unreadable-input path: banner, then the error.  There are no ``CM``
    cards to echo, so there is no box to print them in."""
    refusal = eznec.render_refusal(None, "UNABLE TO READ INPUT FILE deck.nec")
    assert "NUMERICAL ELECTROMAGNETICS CODE (NEC-5)" in refusal
    assert refusal.count("*" * 58) == 0
    assert refusal.rstrip("\n").endswith(
        " ***** NEC ERROR - UNABLE TO READ INPUT FILE deck.nec"
    )


# --------------------------------------------------------------------------
# protocol gates: what the process leaves behind
# --------------------------------------------------------------------------


def test_a_valid_deck_is_refused_in_the_printout_at_exit_zero(tmp_path):
    """The fault-injection table's three obligations at once: the file is
    written, the stamp echo is in it, and the refusal sits AFTER the echo.

    0022 is the deck for it since #504 U4 — a 10.3 m vertical over a ``GN 0``
    earth asking for a NEAR ELECTRIC FIELD, and the LAST capture in the corpus
    that refuses anything.  It is the fifth tenant of this gate, and the
    turnover is the whole arc's shape written out one eviction at a time: 0043
    came back solved when rung 1 landed, 0047 when the Sommerfeld ground did,
    0045 when the MININEC ground did, and 0031 when the phased drive did.

    There is no sixth tenant waiting.  Every ground card, every network table
    and every drive the corpus writes is now served, so the only capture left
    to hold this gate open is the one naming a REQUEST — and when the near
    field lands, this gate needs a hand-edited deck or a new capture, which is
    a fact worth knowing before that unit starts rather than after.
    """
    deck = tmp_path / "EZN5.NEC"
    deck.write_bytes((FIXTURE_DIR / capture("0022")["deck"]).read_bytes())
    out = tmp_path / "NEC5.OUT"

    proc = run_engine([str(deck), str(out)])

    assert proc.returncode == 0
    assert out.exists()
    written = out.read_bytes().decode("latin-1").replace("\r\n", "\n")
    stamp = "EZNEC Pro/2+ v. 7.0.4"
    assert stamp in written
    assert _ERROR_PREFIX in written
    assert written.index(stamp) < written.index(_ERROR_PREFIX)
    reason = written.rsplit(_ERROR_PREFIX, 1)[1].strip()
    assert reason.startswith("NE (near electric field) is not served")
    assert written == eznec.render_refusal(deck_text("0022"), reason)


def test_the_printout_bytes_are_crlf_all_the_way_down(tmp_path):
    """momwire#512: EZNEC refuses an LF-only printout, behind a popup blaming
    a file that never existed ("Output file NEC.OUT is present, but was
    written earlier from another calculation").  Proven by controlled
    substitution on the Windows box: a wrapper changing NOTHING but line
    endings made EZNEC render this engine's results first try.

    So the WRITTEN BYTES are the gate — the byte-gates elsewhere compare the
    rendered string, which is exactly the layer this defect hid beneath.
    Every newline must be CRLF, results and refusals alike, and the file must
    end with one.
    """
    for cid in ("0043", "0022"):  # a served run and a refusal
        room = tmp_path / cid
        room.mkdir()
        deck = room / "EZN5.NEC"
        deck.write_bytes((FIXTURE_DIR / capture(cid)["deck"]).read_bytes())
        out = room / "NEC5.OUT"
        proc = run_engine([str(deck), str(out)])
        assert proc.returncode == 0
        raw = out.read_bytes()
        assert raw, cid
        assert raw.count(b"\n") == raw.count(b"\r\n"), f"{cid}: bare LF written"
        assert raw.endswith(b"\r\n"), cid
        assert b"\r\n" in raw, cid


def test_paths_resolve_against_the_working_directory(tmp_path):
    """EZNEC passes bare filenames and a cwd of the engine's own directory."""
    (tmp_path / "EZN5.NEC").write_bytes(
        (FIXTURE_DIR / capture("0010")["deck"]).read_bytes()
    )

    proc = run_engine(["EZN5.NEC", "NEC5.OUT"], cwd=tmp_path)

    assert proc.returncode == 0
    written = (tmp_path / "NEC5.OUT").read_bytes().decode("latin-1").replace("\r\n", "\n")
    assert written.startswith(header_prefix("0010"))


def test_a_path_with_spaces_needs_no_quoting_in_argv(tmp_path):
    """EZNEC quotes both paths on the command line; the shell unquotes them
    before argv, so the engine must not try to unquote anything itself."""
    directory = tmp_path / "EZNEC 7.0" / "Docs"
    directory.mkdir(parents=True)
    deck = directory / "EZN5.NEC"
    deck.write_bytes((FIXTURE_DIR / capture("0043")["deck"]).read_bytes())
    out = directory / "NEC5.OUT"

    proc = run_engine([str(deck), str(out)])

    assert proc.returncode == 0
    assert out.read_bytes().decode("latin-1").replace("\r\n", "\n").startswith(header_prefix("0043"))


def test_garbage_deck_bytes_still_leave_a_printout(tmp_path):
    """Line noise is not a crash: the engine has nothing to echo, so it says
    so in the only channel there is and exits 0 like everything else."""
    deck = tmp_path / "EZN5.NEC"
    deck.write_bytes(bytes(range(256)) * 8)
    out = tmp_path / "NEC5.OUT"

    proc = run_engine([str(deck), str(out)])

    assert proc.returncode == 0
    written = out.read_bytes().decode("latin-1").replace("\r\n", "\n")
    assert "NUMERICAL ELECTROMAGNETICS CODE (NEC-5)" in written
    assert _ERROR_PREFIX in written


def test_a_missing_deck_still_leaves_a_printout_naming_it(tmp_path):
    """Deleting the output on a read failure would send EZNEC down the
    "try a different location" path and blame the user's installation."""
    deck = tmp_path / "absent.nec"
    out = tmp_path / "NEC5.OUT"

    proc = run_engine([str(deck), str(out)])

    assert proc.returncode == 0
    assert out.exists()
    written = out.read_bytes().decode("latin-1").replace("\r\n", "\n")
    assert " ***** NEC ERROR - UNABLE TO READ INPUT FILE " in written
    assert str(deck) in written


def test_a_directory_in_place_of_a_deck_is_a_read_failure(tmp_path):
    out = tmp_path / "NEC5.OUT"

    proc = run_engine([str(tmp_path), str(out)])

    assert proc.returncode == 0
    assert " ***** NEC ERROR - UNABLE TO READ INPUT FILE " in out.read_text()


def test_one_argument_prints_the_observed_error_and_writes_nothing(tmp_path):
    """`NEC5CL_x13.exe deck.nec` prints exactly this line and exits 0.  There
    is no output path to write a printout to, so none is written."""
    deck = tmp_path / "EZN5.NEC"
    deck.write_bytes((FIXTURE_DIR / capture("0043")["deck"]).read_bytes())

    proc = run_engine([str(deck)], cwd=tmp_path)

    assert proc.returncode == 0
    assert proc.stdout.splitlines() == [ARGUMENT_ERROR_OUTPUT]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["EZN5.NEC"]


def test_no_arguments_errors_on_stdout_rather_than_prompting(tmp_path):
    """The real engine prompts on the console device here; that path is
    deliberately not implemented, and stdin is still never read."""
    proc = run_engine([], cwd=tmp_path)

    assert proc.returncode == 0
    assert proc.stdout.splitlines() == [ARGUMENT_ERROR_INPUT]
    assert list(tmp_path.iterdir()) == []


def test_more_than_two_arguments_is_an_unresolved_command_line(tmp_path):
    deck = tmp_path / "EZN5.NEC"
    deck.write_bytes(b"CM x\nCE\nEN\n")

    proc = run_engine([str(deck), "NEC5.OUT", "extra"], cwd=tmp_path)

    assert proc.returncode == 0
    assert proc.stdout.splitlines() == [ARGUMENT_ERROR_OUTPUT]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["EZN5.NEC"]


def test_an_internal_failure_still_reports_in_the_printout(tmp_path, monkeypatch):
    """The last line of defence, exercised in process: an unexpected exception
    becomes a ``NEC ERROR`` line rather than a traceback nobody can see."""
    out = tmp_path / "NEC5.OUT"
    deck = tmp_path / "EZN5.NEC"
    deck.write_bytes(b"CM x\nCE\nEN\n")

    def boom(*_args, **_kwargs):
        raise RuntimeError("the fill fell over")

    monkeypatch.setattr(eznec._shell, "run", boom, raising=True)

    assert eznec.main([str(deck), str(out)]) == 0
    written = out.read_text()
    assert " ***** NEC ERROR - INTERNAL ERROR IN MOMWIRE ENGINE - " in written
    assert "the fill fell over" in written


# --------------------------------------------------------------------------
# stdin is not a channel
# --------------------------------------------------------------------------


def _serve(tmp_path, name, **kwargs):
    deck = tmp_path / f"{name}.nec"
    deck.write_bytes((FIXTURE_DIR / capture("0043")["deck"]).read_bytes())
    out = tmp_path / f"{name}.out"
    proc = run_engine([str(deck), str(out)], **kwargs)
    return proc, out.read_bytes()


def test_a_deck_on_stdin_changes_nothing(tmp_path):
    """Same argv, empty stdin vs stdin carrying a whole deck: same printout,
    same stdout, same exit 0.  The portal's protocol is not this one."""
    quiet, quiet_bytes = _serve(tmp_path, "quiet")
    noisy, noisy_bytes = _serve(
        tmp_path, "noisy", input=deck_text("0010") + "\nEN\n" * 100
    )

    assert quiet.returncode == noisy.returncode == 0
    assert quiet.stdout == noisy.stdout == ""
    assert quiet_bytes == noisy_bytes


def test_stdin_is_left_entirely_unread(tmp_path):
    """Proved at the file offset: a child's stdin fd is a dup of the one
    handed to it, so the two SHARE a kernel offset.  If the engine had read a
    single byte the offset would have moved."""
    payload = tmp_path / "stdin.bin"
    payload.write_bytes(b"GW 1,11,0.,-.25,0.,0.,.25,0.,.0005\n" * 64)
    deck = tmp_path / "EZN5.NEC"
    deck.write_bytes((FIXTURE_DIR / capture("0043")["deck"]).read_bytes())
    out = tmp_path / "NEC5.OUT"

    fd = os.open(payload, os.O_RDONLY)
    try:
        proc = run_engine([str(deck), str(out)], stdin=fd)
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
    finally:
        os.close(fd)

    assert proc.returncode == 0
    assert out.exists()


# --------------------------------------------------------------------------
# the exit code is never a channel
# --------------------------------------------------------------------------


def test_the_package_is_shipped():
    """``setup.py`` lists packages explicitly, so a new one that nobody adds
    imports fine from a checkout and is simply absent from every wheel."""
    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    assert '"momwire.eznec"' in setup_py.read_text()


def test_every_path_exits_zero(tmp_path):
    """One test that says the rule out loud: served, refused, unreadable,
    mis-invoked — EZNEC ignores the status in all four, so it is always 0."""
    good = tmp_path / "good.nec"
    good.write_bytes((FIXTURE_DIR / capture("0043")["deck"]).read_bytes())
    junk = tmp_path / "junk.nec"
    junk.write_bytes(b"\x00\xff" * 500)

    invocations = (
        [str(good), str(tmp_path / "a.out")],
        [str(junk), str(tmp_path / "b.out")],
        [str(tmp_path / "gone.nec"), str(tmp_path / "c.out")],
        [str(good)],
        [],
        [str(good), str(tmp_path / "d.out"), "surplus"],
    )
    assert [run_engine(args, cwd=tmp_path).returncode for args in invocations] == [
        0
    ] * len(invocations)


def test_an_internal_crash_still_carries_the_comment_echo(tmp_path, monkeypatch):
    """A refusal without the stamp echo is one EZNEC discards as stale, so
    even the last-ditch internal-error printout must echo the deck's CM
    block (re-read from disk) ahead of its NEC ERROR line."""
    from momwire.eznec import _printout, _shell

    deck = FIXTURE_DIR / "decks" / "0043_vertical-over-real-ground.nec"
    deck_text = deck.read_text(encoding="latin-1")
    out = tmp_path / "NEC5.OUT"

    def boom(deck_path, printout_path):
        raise RuntimeError("injected")

    monkeypatch.setattr(_shell, "run", boom)
    assert _shell.main([str(deck), str(out)]) == 0
    text = out.read_text(encoding="latin-1")
    assert text.startswith(_printout.render_header(deck_text))
    assert "NEC ERROR - INTERNAL ERROR IN MOMWIRE ENGINE" in text
    assert "injected" in text
