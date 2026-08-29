"""The session executor's own gates (momwire#719 U4).

What is NOT here: the portal's resident byte behavior — ``resident_loop``
now delegates to :func:`momwire.serve.run_session`, so every existing
resident-protocol test in ``test_portal.py`` and every shared-server gate in
``test_portal_shared.py`` gates the moved loop already, against the same
fixtures as before the move. These tests gate what is genuinely new: the
:class:`~momwire.serve.Seam` contract both dialects now speak, and the
degenerate-case claim — the eznec one-shot as a session of exactly one deck
— that the #718 arc's resident server builds on.
"""

from __future__ import annotations

import io

import pytest

from momwire.eznec import _shell
from momwire.portal import _portal
from momwire.serve import Seam, run_session

DIPOLE_NEC5 = (
    "CM a served rung-1 deck\n"
    "CE\n"
    "GW 1,11,0.,-5.,10.,0.,5.,10.,.001\n"
    "GE 0\n"
    "FR 0,1,0,0,14.\n"
    "EX 4,1,6,0,1.,0.\n"
    "XQ 0\n"
    "EN\n"
)

GARBAGE = "QZ this is not a deck\n"


# --------------------------------------------------------------------------
# the Seam invariant: answer never raises, refusals ride the output
# --------------------------------------------------------------------------


def test_the_portal_seam_answers_garbage_inside_its_own_error_frame():
    out, err = _portal.portal_seam().answer(GARBAGE, "NX")
    assert "ERROR:" in out
    # The sentinel is unconditional — a refusal must never cost it, or the
    # host blocks in readLine() forever.
    assert "NX" in out
    assert err == ""


def test_the_eznec_seam_answers_exactly_what_the_shell_renders():
    for text in (DIPOLE_NEC5, GARBAGE):
        out, err = _shell.seam().answer(text, "")
        assert out == _shell.render(text)
        assert err == ""


def test_the_eznec_seam_frames_an_unforeseen_failure_instead_of_raising(monkeypatch):
    def boom(text, *, basis):
        raise RuntimeError("unforeseen")

    monkeypatch.setattr(_shell, "render", boom)
    out, err = _shell.seam().answer(DIPOLE_NEC5, "")
    assert "INTERNAL ERROR IN MOMWIRE ENGINE - RuntimeError: unforeseen" in out
    assert err == ""


def test_the_two_last_ditch_frames_share_one_spelling():
    """:func:`_shell.main`'s process-level catch and the seam's answer-level
    catch must frame identically — pinned by the shared helper, and here by
    the sentence itself so a fork of either spelling fails loudly."""
    reason = _shell._internal_error_reason(RuntimeError("x"))
    assert reason == "INTERNAL ERROR IN MOMWIRE ENGINE - RuntimeError: x"


# --------------------------------------------------------------------------
# the degenerate case: the eznec one-shot as a session of one deck
# --------------------------------------------------------------------------


def test_an_eznec_session_is_exactly_one_deck_answered_at_eof():
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_session(_shell.seam(), io.StringIO(DIPOLE_NEC5), stdout, stderr)
    assert code == 0
    assert stdout.getvalue() == _shell.render(DIPOLE_NEC5)
    assert stderr.getvalue() == ""


def test_a_whitespace_only_eznec_session_answers_nothing():
    stdout, stderr = io.StringIO(), io.StringIO()
    assert run_session(_shell.seam(), io.StringIO("\n  \n"), stdout, stderr) == 0
    assert stdout.getvalue() == ""


# --------------------------------------------------------------------------
# the generic loop mechanics, on a synthetic seam
# --------------------------------------------------------------------------


def _echo_seam(**overrides) -> Seam:
    fields = dict(
        name="echo",
        greeting=lambda: ["HELLO"],
        terminators=frozenset({"GO", "BYE"}),
        closing=frozenset({"BYE"}),
        eof_terminator="EOF",
        answer=lambda body, term: (f"[{term}:{body}]\n", ""),
    )
    fields.update(overrides)
    return Seam(**fields)


def test_the_loop_greets_once_frames_on_terminators_and_ends_on_closing():
    stdout = io.StringIO()
    code = run_session(
        _echo_seam(),
        io.StringIO("a\nb\nGO\nc\nBYE\nnever\nGO\n"),
        stdout,
        io.StringIO(),
    )
    assert code == 0
    # Frames after the closing terminator are never read — the session ended.
    assert stdout.getvalue() == "HELLO\n[GO:a\nb]\n[BYE:c]\n"


def test_eof_synthesizes_the_seam_s_terminator_for_a_non_blank_body():
    stdout = io.StringIO()
    run_session(_echo_seam(), io.StringIO("a\nGO\ntail\n"), stdout, io.StringIO())
    assert stdout.getvalue() == "HELLO\n[GO:a]\n[EOF:tail]\n"


def test_a_seam_without_eof_synthesis_discards_an_unterminated_body():
    stdout = io.StringIO()
    run_session(
        _echo_seam(eof_terminator=None),
        io.StringIO("a\nGO\ntail\n"),
        stdout,
        io.StringIO(),
    )
    assert stdout.getvalue() == "HELLO\n[GO:a]\n"


def test_the_lock_is_held_across_every_answer():
    entered = []

    class Lock:
        def __enter__(self):
            entered.append("in")

        def __exit__(self, *exc):
            entered.append("out")

    run_session(
        _echo_seam(),
        io.StringIO("a\nGO\nb\nGO\ntail\n"),
        io.StringIO(),
        io.StringIO(),
        solve_lock=Lock(),
    )
    assert entered == ["in", "out"] * 3


# --------------------------------------------------------------------------
# the portal seam through the generic loop — the delegation is total
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_resident_loop_is_the_generic_loop_over_the_portal_seam():
    """Byte-for-byte: the public entry and a hand-built run_session over
    ``portal_seam()`` answer an NX-then-EN stream identically. This is the
    U4 claim itself — resident_loop adds nothing but the seam."""
    stream = (
        "CE two decks\nGW 1 11 0. -5. 10. 0. 5. 10. 0.001\nGE 0\n"
        "EX 0 1 6 0 1.\nFR 0 1 0 0 14.0 1\nXQ\nNX\n"
        "CE second\nGW 1 11 0. -5. 10. 0. 5. 10. 0.001\nGE 0\n"
        "EX 0 1 6 0 1.\nFR 0 1 0 0 14.0 1\nXQ\nEN\n"
    )
    via_entry, via_seam = io.StringIO(), io.StringIO()
    assert _portal.resident_loop(io.StringIO(stream), via_entry, io.StringIO()) == 0
    assert (
        run_session(_portal.portal_seam(), io.StringIO(stream), via_seam, io.StringIO())
        == 0
    )

    def strip_timing(text: str) -> str:
        # The FILL cell is wall-clock — the one line two live solves of one
        # deck may legitimately disagree on (same rule as the corpus gates).
        return "\n".join(line for line in text.split("\n") if "FILL:" not in line)

    assert strip_timing(via_entry.getvalue()) == strip_timing(via_seam.getvalue())
    assert via_entry.getvalue().count("ANTENNA INPUT PARAMETERS") == 2
