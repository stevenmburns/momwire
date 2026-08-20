"""The momwire SimNEC portal daemon (issue #792, units 2 to 4).

Everything here runs against the committed oracle fixtures — no ``nec2c``
binary — and, bar the one cwd-independence test at the foot of the file, in
process. The oracle's *numbers* are not the contract
(a different basis and kernel will never match digit for digit); its *layout*
is, so the fixtures are compared structurally: same section sequence, same
column geometry, same token arity — for every deck in the corpus, not just a
representative handful. Values are checked against momwire itself
(self-consistency and reciprocity), plus one loose cross-engine smoke bound on
the free-space dipole's impedance. Cross-engine value agreement is
``test_portal_differential.py``.

Unit 3 adds, below the unit-2 sections: the whole-corpus layout gate, the
execute-card semantics of ``RP``/``NE``/``NH``, the pattern and near-field
tables, ``NT`` port algebra, and the robustness contract — a malformed deck
must be REPORTED and stepped over, never swallowed and never fatal, because
``Execute.processResponse`` blocks in ``readLine()`` with no timeout.

Unit 4 adds the packaging contract: the ``momwire-nec2c`` console script whose
NAME is what SimNEC's portal dialog accepts an engine on, and the fact that the
daemon runs from any working directory (SimNEC launches it via ``sh -c`` with
cwd=$HOME).
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from momwire.portal import _portal as nec_portal
from momwire.portal import deck_frame, main, run_deck

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "nec_portal"
ALL_NAMES = tuple(
    entry["name"]
    for entry in json.loads((FIXTURE_DIR / "manifest.json").read_text())["decks"]
)

# No corpus deck is refused: every fixture here is byte-compared against the
# oracle's own printout, and this tuple is the empty statement of that.
#
# It has been emptied twice, both times by a design doc reversing a hygiene
# decision. ``TL`` and ``NT`` were here from #930 on the argument that this
# engine's language is antenna-only; momwire#456 phase C took the opposite
# decision and the dialect SOLVES them, so the ``NETWORK_FIXTURES`` battery
# below pins the answer instead of the refusal. Then the four MININEC-ground
# decks — ``dipole_gd_second_medium`` and the three ``mininec_*`` forms
# momwire#487 captured — left it when the measurement came in: a ``GD`` under
# a ``GN 1`` at an execute card that will not read it is answered as PLAIN
# PERFECT GROUND by every reference engine, banner and all, so the refusal
# #458 installed was the divergence rather than the safety margin
# (momwire#487; ``docs/design/mininec-ground-idiom.md``). The tuple is kept
# rather than deleted because the gates below are written to run over
# ``ANTENNA_NAMES``, and a future refusal must land here rather than by
# quietly dropping a deck out of a list.
REFUSED_NAMES: tuple[str, ...] = ()
ANTENNA_NAMES = tuple(n for n in ALL_NAMES if n not in REFUSED_NAMES)

# momwire#415: the two decks whose STRUCTURE SPECIFICATION section carries a
# replication — the annotation row (``STRUCTURE REFLECTED ALONG THE AXES X * *
# - TAGS INCREMENTED BY 1`` / ``STRUCTURE ROTATED ABOUT Z-AXIS 4 TIMES -
# LABELS INCREMENTED BY 1``, nec2c geometry.c) and the FIRST/LAST SEG columns
# of a GW written after the card, which count the segments the replication
# added (the trailing wire is 19-27 / 37-45, not 10-18). Both halves are
# byte-asserted against the captured oracle section below
# (``test_the_gx_gr_structure_specification_reproduces_the_oracle``).
REPLICATED_SPEC_NAMES = ("dipole_gr_rotated_ring", "dipole_gx_reflected_pair")

# momwire#487: the decks whose structure TOUCHES the ground plane under a
# ``GE 1``. They are the first in the corpus to do so — every earlier ground
# fixture either stands clear of z = 0 (``dipole_pec_ground``,
# ``catalog_verticals_vertical``) or writes ``GE -1``. They arrived (#488)
# with a known two-line printout gap against the oracle, held out of the
# column-layout gate and asserted exactly; U2 CLOSED it, so they are in
# ``LAYOUT_NAMES`` with everything else and this tuple is what the
# ground-contact printout is asserted on directly
# (``test_the_ground_contact_printout_says_the_ends_are_connected``).
GROUND_CONTACT_NAMES = (
    "mininec_gd_reset_by_gn_rp0",
    "mininec_gp80_seam",
    "mininec_vertical_gd2_rp0",
    "mininec_vertical_rp0",
    "mininec_vertical_rp3_ch",
    "mininec_vertical_rp3_cliff_at_zero",
    "mininec_vertical_rp3_clt",
)

# The oracle's second ground-plane banner line. nec2c's ``conect()`` prints it
# for a POSITIVE ``GE`` and not for ``GE -1``/``GE 0``, and unconditionally on
# the geometry — a ``GE 1`` deck standing clear of the plane prints it too
# (measured on the oracle, 2026-08-20).
GROUND_CONTACT_BANNER = (
    "     WHERE WIRE ENDS TOUCH GROUND, CURRENT WILL BE "
    "INTERPOLATED TO IMAGE IN GROUND PLANE."
)


# nec2/Execute.versionA — the regex SimNEC applies to `<cmd> -version`.
VERSION_A = re.compile(r"nec2c\.ae6ty\.(.*)")

# nec2/Execute.processResponse's daemon sentinel.
NX_ECHO = re.compile(r"^\s*DATA CARD No:\s+(\d+) NX\b.*$", re.MULTILINE)

_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")

# The section banners SimNEC's state machine arms on, plus the ones a reader
# uses to find its way. Order is part of the contract.
_SECTION_MARKERS = (
    "COMMENTS",
    "STRUCTURE SPECIFICATION",
    "MULTIPLE WIRE JUNCTIONS",
    "SEGMENTATION DATA",
    "FREQUENCY",
    "STRUCTURE IMPEDANCE LOADING",
    "ANTENNA ENVIRONMENT",
    "NETWORK DATA",
    "STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS",
    "MATRIX TIMING",
    "ANTENNA INPUT PARAMETERS",
    "CURRENTS AND LOCATION",
    "POWER BUDGET",
    "FAR FIELD GROUND PARAMETERS",
    "RADIATION PATTERNS",
    "NEAR ELECTRIC FIELDS",
    "NEAR MAGNETIC FIELDS",
)

# Small decks only — the whole file has to stay in the fast lane.
REPRESENTATIVE = (
    "dipole_free_space",
    "dipole_pec_ground",
    "dipole_load_ld0",
    "dipole_gs_scaled",
    "dipole_rp2_linear_cliff",
    "split_dipole_qq",
    "two_source_sensor_lines",
)


def fixture_deck(name: str) -> str:
    """The deck body: the fixture minus its framing ``NX`` card."""
    return (FIXTURE_DIR / f"{name}.deck").read_text().split("\nNX")[0]


def printout(name: str) -> str:
    """Our printout for a fixture, through the daemon loop when the fixture is
    a multi-deck residency transcript."""
    deck = (FIXTURE_DIR / f"{name}.deck").read_text()
    if deck.count("\nNX") > 1:
        buffer = io.StringIO()
        assert (
            main([], stdin=io.StringIO(deck), stdout=buffer, stderr=io.StringIO()) == 0
        )
        return buffer.getvalue()
    return run_deck(deck.split("\nNX")[0])[0]


def fixture_out(name: str) -> str:
    return (FIXTURE_DIR / f"{name}.out").read_text()


def section_walk(text: str) -> list[str]:
    """The section banners in printout order."""
    walk = []
    for line in text.splitlines():
        stripped = line.strip(" -")
        for marker in _SECTION_MARKERS:
            if stripped == marker:
                walk.append(marker)
                break
    return walk


def layout_signature(line: str) -> tuple:
    """A line's format-determined shape: each token's END column plus whether
    it is a number.

    Right-aligned fixed-width fields put every token's end at a
    format-determined column no matter what the value is, so this compares
    ``%11.4E``-against-``%11.4E`` while letting momwire's numbers differ from
    the oracle's.
    """
    return tuple(
        (m.end(), "N" if _NUMBER.match(m.group()) else m.group())
        for m in re.finditer(r"\S+", line)
    )


def body_lines(text: str) -> list[str]:
    """Non-blank printout lines, minus the ones that legitimately differ:
    the version banner (ours says momwire) and the wall-clock timings the
    capture script canonicalises to zero."""
    return [
        line
        for line in text.splitlines()
        if line.strip()
        and "VERSION:" not in line
        and "FILL:" not in line
        and "ERROR-NEC2C" not in line
    ]


def aip_tables(text: str) -> list[list[list[str]]]:
    """The ANTENNA INPUT PARAMETERS rows, tokenised.

    Mirrors nec2/Execute's WAITINGFORSENSORS state for the NEC2C engine: a
    data row is exactly 11 whitespace tokens (``samplesWidth``) and the
    current sits at fields 4 and 5 (``samplesOffset``).
    """
    tables: list[list[list[str]]] = []
    collecting = False
    for line in text.splitlines():
        parts = line.split()
        if parts[:3] == ["No:", "No:", "REAL"]:
            collecting = True
            tables.append([])
            continue
        if not collecting:
            continue
        if len(parts) != 11:
            collecting = False
            continue
        tables[-1].append(parts)
    return tables


# --------------------------------------------------------------------------
# the version probe
# --------------------------------------------------------------------------


# Execute.testCommand's four probe regexes, verbatim from the 5.1a0 bytecode
# (issue #828 research; grammar doc 2026-08-09 addendum). A/B/C Double-parse
# group(1) against the 1.23 floor; NECd reads no group and sets no state.
VERSION_B = re.compile(r"5b4az\.ae6ty\.(.*)")
VERSION_C = re.compile(r"necpp\.nec2c\.(.*)")
VERSION_NECD = re.compile(r"(NEC\d+\D.*)")
# Options.getEngine()'s re-read of the stored version text: group(1) must be
# "2" or SimNEC's scripting layer reclassifies the engine (W7EL insulation
# refuses anything but Complex.TWO).
OPTIONS_ENGINE_PIECE = re.compile(r"[a-zA-Z]*([0-9])+?(.*)")


def test_version_probe_matches_executes_versionNECd_regex():
    """The honest identity (issue #828, Ward-sanctioned): the probe answers
    the versionNECd path, whose match sets no engine state — the engine enum,
    daemon class, and parse offsets all come from the executable FILENAME.
    The gates below replicate every constraint the bytecode research found
    load-bearing, so a probe edit that would break a live SimNEC fails here.
    """
    out = io.StringIO()
    assert main(["-version"], stdin=io.StringIO(""), stdout=out) == 0
    lines = out.getvalue().splitlines()
    assert len(lines) == 1, f"the probe must print exactly one line: {lines}"
    probe = lines[0].strip()

    # lookingAt() semantics = re.match, anchored at the start only.
    assert VERSION_NECD.match(probe), f"{probe!r} does not match (NEC\\d+\\D.*)"
    # Must NOT match A/B/C: those branches Double-parse the tail and a
    # non-numeric tail there is rejected as "nec2c version too old".
    for pat in (VERSION_A, VERSION_B, VERSION_C):
        assert not pat.match(probe), f"{probe!r} would take the {pat.pattern} path"
    # Case-sensitive NEC2 at position 0, non-digit right after the 2 (the
    # regex needs \D after the digit run), and Options.getEngine must read 2.
    assert probe.startswith("NEC2") and not probe[4].isdigit(), probe
    assert OPTIONS_ENGINE_PIECE.match(probe).group(1) == "2", probe
    assert probe.startswith("NEC2momwire.")


def test_legacy_probe_flag_answers_the_versionA_masquerade():
    """``--legacy-probe`` keeps the pre-#828 identity available for a SimNEC
    build that might predate versionNECd. That path Double-parses the tail
    against the 1.23 floor, so the shape constraints are the old ones: one
    dot, a bare number above the floor."""
    out = io.StringIO()
    assert main(["--legacy-probe", "-version"], stdin=io.StringIO(""), stdout=out) == 0
    probe = out.getvalue().strip()
    assert probe == nec_portal.LEGACY_PROBE_VERSION
    tail = VERSION_A.match(probe).group(1)
    assert tail.count(".") == 1 and float(tail) >= 1.23, tail


def test_printout_banner_carries_the_momwire_identity():
    """The banner is not version-checked (the regexes are anchored and it is
    prefixed ``VERSION:``), so it stays fixture-pinned while the PROBE now
    carries the honest identity too (#828). The banner keeps its historical
    shape on purpose: 40 committed fixtures pin it, and the offline .out
    import path (``ae6ty/FileStuff``) greps the ``(nec2c)`` box line."""
    text, _err = run_deck(fixture_deck("dipole_free_space"))
    assert f"VERSION:{nec_portal.BANNER_VERSION}" in text
    assert "momwire" in nec_portal.BANNER_VERSION
    assert "momwire" in nec_portal.PROBE_VERSION


# --------------------------------------------------------------------------
# structural interchangeability with the oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_section_walk_matches_the_oracle(name):
    ours, _err = run_deck(fixture_deck(name))
    assert section_walk(ours) == section_walk(fixture_out(name))


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_column_layout_matches_the_oracle(name):
    """Same lines, same columns — only the values inside them differ."""
    ours = body_lines(run_deck(fixture_deck(name))[0])
    theirs = body_lines(fixture_out(name))
    assert len(ours) == len(theirs)
    for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
        if "DIELECTRIC CONSTANT" in b:
            # The oracle glues the real and imaginary parts into one token,
            # so this line's "shape" is its value. It is an ignored section.
            continue
        assert layout_signature(a) == layout_signature(b), (
            f"{name} line {i}\n  ours   {a!r}\n  oracle {b!r}"
        )


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_nx_sentinel_is_byte_identical_modulo_the_card_ordinal(name):
    """The one line the Java side blocks on. Everything but the ordinal must
    match the oracle byte for byte (grammar doc §2)."""
    ours = NX_ECHO.search(run_deck(fixture_deck(name))[0])
    theirs = NX_ECHO.search(fixture_out(name))
    assert ours and theirs
    assert ours.group(1) == theirs.group(1)
    blank = re.compile(r"No:\s+\d+ NX")
    assert blank.sub("No: NX", ours.group(0)) == blank.sub("No: NX", theirs.group(0))


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_antenna_input_rows_are_eleven_tokens_with_the_current_at_4_and_5(name):
    ours = aip_tables(run_deck(fixture_deck(name))[0])
    theirs = aip_tables(fixture_out(name))
    assert [len(t) for t in ours] == [len(t) for t in theirs]
    assert ours, f"{name}: no ANTENNA INPUT PARAMETERS table"
    for table in ours:
        for row in table:
            assert len(row) == 11
            float(row[4])
            float(row[5])


def test_quiet_mode_suppresses_the_segmentation_block():
    """`CE QQ 1` is ae6ty's quiet directive; the jar's own test deck used
    it (the fixture is its YY-free successor, #839)."""
    quiet, _err = run_deck(fixture_deck("split_dipole_qq"))
    loud, _err = run_deck(fixture_deck("dipole_free_space"))
    assert "SEGMENTATION DATA" not in quiet
    assert "SEGMENTATION DATA" in loud
    assert "STRUCTURE SPECIFICATION" in quiet


def test_reduced_field_is_the_only_thing_written_to_stderr():
    """NEC2Daemon never drains the child's stderr, so anything beyond the
    `CM FF` line risks filling the pipe buffer and deadlocking the UI."""
    _out, err = run_deck(fixture_deck("split_dipole_qq_daemon_framed"))
    assert err == "reducedField:2\n"
    _out, err = run_deck(fixture_deck("dipole_free_space"))
    assert err == ""


# --------------------------------------------------------------------------
# the retired YY report card (#839)
# --------------------------------------------------------------------------


def test_yy_card_is_no_longer_a_known_card():
    """Ward's YY directive is retired (#839: abandoned upstream, its only
    sender a dev benchmark). A deck carrying one now takes the same
    unknown-card path a real nec2c would give it — never a silent no-op
    that prints a '-YY' row."""
    deck = fixture_deck("split_dipole_qq").replace("GE 0\n", "GE 0\nYY 1 4 2 4 5 4\n")
    out, _err = run_deck(deck)
    assert "unrecognised NEC card 'YY'" in out
    assert "    -YY" not in out


def test_two_source_y_matrix_is_reciprocal():
    """Y12 == Y21 out of the multi-EX probe SimNEC actually writes. momwire's
    Galerkin operator is symmetric, so this is a self-consistency pin on the
    whole port-algebra path, not an approximation."""
    tables = aip_tables(run_deck(fixture_deck("two_source_sensor_lines"))[0])
    assert len(tables) == 2 and all(len(t) == 2 for t in tables)
    y12 = complex(float(tables[0][1][4]), float(tables[0][1][5]))
    y21 = complex(float(tables[1][0][4]), float(tables[1][0][5]))
    assert abs(y12 - y21) <= 1e-6 * max(abs(y12), abs(y21))


# --------------------------------------------------------------------------
# residency
# --------------------------------------------------------------------------


def test_two_decks_through_one_loop_produce_two_frames():
    """NEC2Daemon.submit frames decks on stdin with NX and never restarts the
    process; the engine reprints its banner after each NX for the deck it
    expects next, so two decks show three banners."""
    text = io.StringIO()
    stdin = io.StringIO((FIXTURE_DIR / "resident_two_decks.deck").read_text())
    assert main([], stdin=stdin, stdout=text, stderr=io.StringIO()) == 0
    ours = text.getvalue()

    assert len(NX_ECHO.findall(ours)) == 2
    assert ours.count("VERSION:") == 3
    assert ours.count("STRUCTURE SPECIFICATION") == 2
    # Card numbering restarts inside each deck.
    assert re.findall(r"DATA CARD No:\s+(\d+) (\w\w)", ours) == [
        ("1", "EX"),
        ("2", "FR"),
        ("3", "XQ"),
        ("4", "NX"),
        ("1", "EX"),
        ("2", "FR"),
        ("3", "XQ"),
        ("4", "NX"),
    ]
    assert section_walk(ours) == section_walk(fixture_out("resident_two_decks"))


_EN_DIPOLE = (
    "CE standalone dipole\n"
    "GW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\n"
    "GE 0\n"
    "EX 0 1 5 0 1.\n"
    "FR 0 1 0 0 30. 0\n"
    "XQ\n"
)


def test_en_terminates_a_frame_and_ends_the_run():
    """#901: an unmodified .nec file — XQ then EN, no NX — redirected into the
    daemon must solve, echo EN as its own final data card the way genuine
    nec2c does, skip the next-deck banner reprint, and exit 0."""
    rc, out, err = _run_main([], deck=_EN_DIPOLE + "EN\n")
    assert rc == 0
    assert err == ""
    assert out.count("ANTENNA INPUT PARAMETERS") == 1
    assert re.findall(r"DATA CARD No:\s+(\d+) (\w\w)", out) == [
        ("1", "EX"),
        ("2", "FR"),
        ("3", "XQ"),
        ("4", "EN"),
    ]
    # EN ends the run: one start-up banner and nothing after the EN echo.
    assert out.count("VERSION:") == 1
    last = [ln for ln in out.splitlines() if ln.strip()][-1]
    assert " EN " in last


def test_en_after_an_nx_frame_echoes_and_exits_without_running():
    """EN arriving with an empty body (right after an NX frame) is echoed as
    card 1 of an empty deck and ends the run — nothing solves twice. Blank
    lines left behind must not become a deck of their own either."""
    rc, out, err = _run_main([], deck=_EN_DIPOLE + "NX\n\nEN\n\n")
    assert rc == 0
    assert err == ""
    assert out.count("ANTENNA INPUT PARAMETERS") == 1
    # Startup banner + the post-NX reprint, none after EN.
    assert out.count("VERSION:") == 2
    assert re.findall(r"DATA CARD No:\s+(\d+) (\w\w)", out)[-2:] == [
        ("4", "NX"),
        ("1", "EN"),
    ]


def _canonical_printout(text: str) -> str:
    """The capture script's timing canonicaliser, for equality claims between
    two independent runs: the wall-clock numbers are the only
    non-deterministic printout fields, and raw byte equality flakes exactly
    (and only) under a loaded full-suite xdist run — the momwire#403 lesson
    ``test_portal_shared.same_printout`` already encodes."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "nec_portal_capture.py"
    spec = importlib.util.spec_from_file_location("nec_portal_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.canonicalize_timings(text)


def test_unterminated_body_at_eof_solves_as_if_it_ended_with_en():
    """#458: end of input is a terminator. NEC's own card reader synthesizes
    an EN when stdin runs out mid-deck and runs what it has (seen live in the
    #413 4nec2 capture, on a bundled model ending at its NE card), so the same
    deck with and without its EN must produce the same printout — timings
    canonicalized, since the two solves run seconds apart. The earlier
    discard was invisible to every captured host — they read only the
    printout — and arrived as an empty answer blamed on the engine."""
    rc, out, err = _run_main([], deck=_EN_DIPOLE)
    rc_en, out_en, err_en = _run_main([], deck=_EN_DIPOLE + "EN\n")
    assert (rc, _canonical_printout(out), err) == (
        rc_en,
        _canonical_printout(out_en),
        err_en,
    )
    assert rc == 0
    assert err == ""
    assert out.count("ANTENNA INPUT PARAMETERS") == 1
    assert re.findall(r"DATA CARD No:\s+(\d+) (\w\w)", out)[-1] == ("4", "EN")


def test_whitespace_only_residual_at_eof_runs_nothing():
    """The EOF terminator is for a deck, not for the blank lines a caller
    leaves behind after its last NX: a residual body with nothing in it must
    still solve nothing and print nothing beyond the next-deck banner."""
    rc, out, err = _run_main([], deck=_EN_DIPOLE + "NX\n\n   \n")
    assert rc == 0
    assert err == ""
    # One solve — the NX frame's — and no second data-card echo after it.
    assert out.count("ANTENNA INPUT PARAMETERS") == 1
    assert re.findall(r"DATA CARD No:\s+(\d+) (\w\w)", out)[-1] == ("4", "NX")


def test_an_unsupported_card_still_emits_the_sentinel():
    """A deck we cannot run must not leave SimNEC blocked in readLine().

    ``SP`` is the live example after issue #873 took ``IS`` off this list
    (#800 took ``PT`` and ``MP``, #799 took ``TL``): the portal dialect can
    carry a surface patch, but momwire is a wire solver — so it takes the
    error path rather than silently solving whatever wires remain.

    Issue #829 (Ward's 2026-08-08 reply): the refusal now leads with an
    ``ERROR:`` line — token 0 exactly, which is what trips Execute's
    ``"NEC ERROR (1)"`` warning frame — with the oracle-shaped
    ``ERROR-NEC2C:`` line kept right after it for grep. Before #829 the
    prefix was chosen specifically to dodge that frame; Ward has since said
    the frame "should be fine" and that he intends to make the reader bail
    on it, so this test now pins the opposite of what it used to.
    """
    out, err = deck_frame(
        "CE patch\n"
        "GW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\n"
        "GW 2 9 1. 0. -2.5 1. 0. 2.5 0.001\n"
        "GE 0\n"
        "EX 0 1 5 0 1.\n"
        "SP 0 0 0. 0. 0. 0. 0. 1.\n"
        "FR 0 1 0 0 30. 0\n"
        "XQ\n"
    )
    text = "\n".join(out)
    assert NX_ECHO.search(text), "the NX sentinel is missing on the error path"
    assert "ERROR: SP" in text
    assert "ERROR-NEC2C: SP" in text
    assert any(line.split()[:1] == ["ERROR:"] for line in text.splitlines()), (
        "no line trips Execute's token-0 `ERROR:` warning frame"
    )
    assert err == []


# --------------------------------------------------------------------------
# issue #873: the IS insulated-sheath card rides momwire's jacket model
# --------------------------------------------------------------------------

_IS_DIPOLE = (
    "CE insulated dipole\n"
    "GW 1 21 0. 0. -2.5 0. 0. 2.5 0.0005\n"
    "GE 0\n"
    "EX 0 1 11 0 1.\n"
    "{is_card}"
    "FR 0 1 0 0 28.5 0\n"
    "XQ\n"
)


def _is_dipole_z(is_card: str = ""):
    out, err = deck_frame(_IS_DIPOLE.format(is_card=is_card))
    text = "\n".join(out)
    assert err == []
    assert "ERROR-NEC2C" not in text
    rows = _aip_rows(text)
    assert rows
    return complex(*rows[0][4:6])


def test_is_card_solves_and_loads_the_wire():
    """The acceptance solve (issue #873): a full-wire ``IS`` runs instead of
    refusing, and the jacket's King series inductance makes the dipole
    electrically longer — at a frequency just below bare resonance the
    reactance moves up by whole ohms, the velocity-factor shift."""
    z_bare = _is_dipole_z()
    z_ins = _is_dipole_z("IS 0 1 1 21 3.2 0. 0.0015\n")
    assert z_ins.imag > z_bare.imag + 3.0, (z_bare, z_ins)
    # The jacket is lossless: R moves only via the current redistribution,
    # not by tens of ohms.
    assert abs(z_ins.real - z_bare.real) < 10.0, (z_bare, z_ins)


def test_is_card_matches_a_direct_momwire_solve():
    """Portal path == the same jacket handed straight to the solver (issue
    #873 acceptance): same geometry, mesh, basis and insulation kwargs, so
    the two answers must sit within solver noise.

    The one test in this battery that could NOT move verbatim from
    antennaknobs (#846 phase III). Its independent construction there went
    through ``AntennaBuilder``/``WireSpec``/``MomwireEngine`` — the assembly
    layer the rewire deleted, and an import the isolation rule forbids. The
    claim is unchanged and the oracle is now more direct rather than less:
    the same dipole handed to :class:`BSplineSolver` with the insulation
    kwargs the ``IS`` card is supposed to become. A card silently swallowed
    on the way to the solver is exactly what this catches — the bare answer
    would come back instead.
    """
    import numpy as np
    from momwire import BSplineSolver

    solver = BSplineSolver(
        wires=[np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])],
        nsegs=21,
        wavelength=nec_portal.C_LIGHT / 28.5e6,
        wire_radius=0.0005,
        insulation_radius=0.0015,
        insulation_eps_r=3.2,
    )
    direct = solver.compute_impedance()[0]
    portal = _is_dipole_z("IS 0 1 1 21 3.2 0. 0.0015\n")
    # The portal value is read back from the PRINTED AIP row, so the bound
    # is the printout's own quantization, not solver noise (measured 6e-6).
    assert abs(portal - direct) / abs(direct) < 1e-4, (portal, direct)


def _is_refusal(deck: str) -> str:
    out, err = deck_frame(deck)
    text = "\n".join(out)
    assert err == []
    assert NX_ECHO.search(text), "refusal must still emit the NX sentinel"
    assert "ERROR: IS" in text
    return text


def test_is_partial_wire_range_refuses_by_name():
    text = _is_refusal(_IS_DIPOLE.format(is_card="IS 0 1 3 9 3.2 0. 0.0015\n"))
    assert "partial-wire" in text


def test_is_conductive_sheath_refuses_by_field():
    text = _is_refusal(_IS_DIPOLE.format(is_card="IS 0 1 1 21 3.2 0.01 0.0015\n"))
    assert "conductive sheath" in text


def test_is_jacket_inside_the_conductor_refuses_by_name():
    text = _is_refusal(_IS_DIPOLE.format(is_card="IS 0 1 1 21 3.2 0. 0.0002\n"))
    assert "conductor radius" in text


def test_is_after_an_execute_request_refuses():
    text = _is_refusal(
        _IS_DIPOLE.format(is_card="") + "IS 0 1 1 21 3.2 0. 0.0015\nXQ\n"
    )
    assert "execute" in text


# --------------------------------------------------------------------------
# issue #829: the `ERROR:` line must trip and ONLY trip on a real refusal
# --------------------------------------------------------------------------


def _first_tokens(text: str) -> list[str]:
    return [line.split()[0] for line in text.splitlines() if line.split()]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_no_clean_fixture_carries_a_token_0_error_line(name):
    """The corpus-wide half of the token-0 pin: a healthy deck must never
    show Execute's warning frame — a stray ``ERROR:`` on a clean run would be
    a false alarm in the SimNEC UI.

    Walked against the committed oracle ``.out`` bytes themselves, not our
    own printout, because that is the file most exposed to accidental
    real-string drift (e.g. a future oracle capture, a hand edit) and the one
    a corpus-wide gate is for. The oracle's own literal ``ERROR-NEC2C:``
    stdin-EOF string (grammar doc §8) is a DIFFERENT token — it never
    collides with ``ERROR:`` — which is exactly why that shape was safe to
    reuse here.
    """
    assert "ERROR:" not in _first_tokens(fixture_out(name)), (
        f"{name}: a clean oracle fixture carries a token-0 `ERROR:` line"
    )


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_no_clean_regenerated_printout_carries_a_token_0_error_line(name):
    """Same pin, our side: a handful of clean decks run back through this
    engine must not produce the warning frame either."""
    assert "ERROR:" not in _first_tokens(printout(name)), (
        f"{name}: our own clean printout carries a token-0 `ERROR:` line"
    )


def test_patch_antenna_refusal_shows_why_nothing_loaded():
    """Stand-in for the issue's live-session gate (no SimNEC on this box).

    The live case (issue #829) was an EZNEC patch-antenna design: SimNEC
    forwarded an ``SP``/``SM`` surface-patch deck, the daemon refused it
    silently under the pre-#829 ``ERROR-NEC2C:``-only shape, and the user was
    left staring at an empty session with no indication why. This synthesizes
    the same refusal class — ``SP`` — through ``main`` exactly as the daemon
    receives it on stdin, and pins the three things that scenario needs:
    the deck exits cleanly (rc 0, daemon stays resident — no reason for a
    live session to die over one bad deck), the refusal names the offending
    card, and the ``ERROR:`` token-0 line is present to trip SimNEC's own
    warning frame instead of leaving the UI blank.
    """
    deck = (
        "CE patch antenna\n"
        "GW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\n"
        "GE 0\n"
        "EX 0 1 5 0 1.\n"
        "SP 0 0 0. 0. 0. 0. 0. 1.\n"
        "FR 0 1 0 0 30. 0\n"
        "XQ\nNX\n"
    )
    rc, out, err = _run_main([], deck=deck)
    assert rc == 0, "the daemon must exit cleanly, not die on the bad deck"
    assert err == ""
    error_lines = [ln for ln in out.splitlines() if ln.split()[:1] == ["ERROR:"]]
    assert error_lines, "no token-0 `ERROR:` line — the user sees nothing again"
    assert "SP" in error_lines[0], f"the refusal does not name SP: {error_lines[0]!r}"
    assert NX_ECHO.search(out), "no NX sentinel on the error path"
    # Follow-up: re-run this scenario against a live SimNEC session on the
    # Windows box once Ward ships his reader bail-fix, to confirm the frame
    # actually surfaces to the user end to end (tracked as a note in #829,
    # not re-testable here — no SimNEC install on this machine).


# --------------------------------------------------------------------------
# numbers: momwire against itself, and one loose bound against the oracle
# --------------------------------------------------------------------------


def test_antenna_input_row_is_internally_consistent():
    """V = I·Z and Y = I/V and P = ½·Re(V·I*), read off the row's own columns.

    This is the identity a transposed real/imaginary pair breaks: the row
    still has 11 tokens and still parses, but the numbers stop agreeing.
    """
    for table in aip_tables(run_deck(fixture_deck("dipole_free_space"))[0]):
        for row in table:
            v = complex(float(row[2]), float(row[3]))
            i = complex(float(row[4]), float(row[5]))
            z = complex(float(row[6]), float(row[7]))
            y = complex(float(row[8]), float(row[9]))
            p = float(row[10])
            assert abs(i * z - v) <= 1e-3 * abs(v)
            assert abs(y - i / v) <= 1e-3 * abs(y)
            assert p == pytest.approx(0.5 * (v * i.conjugate()).real, rel=1e-3)


def test_swapping_the_current_columns_is_caught():
    """A mutation reviewers will try. Transposing fields 4 and 5 keeps the row
    shape and keeps every token parseable — only the identities above notice.
    """
    original = nec_portal.fmt_aip_row

    def transposed(tag, seg, voltage, current, impedance, admittance, power):
        return original(
            tag,
            seg,
            voltage,
            complex(current.imag, current.real),
            impedance,
            admittance,
            power,
        )

    nec_portal.fmt_aip_row = transposed
    try:
        table = aip_tables(run_deck(fixture_deck("dipole_free_space"))[0])[0]
    finally:
        nec_portal.fmt_aip_row = original
    row = table[0]
    v = complex(float(row[2]), float(row[3]))
    i = complex(float(row[4]), float(row[5]))
    z = complex(float(row[6]), float(row[7]))
    assert len(row) == 11  # still well-formed...
    assert abs(i * z - v) > 1e-3 * abs(v)  # ...and still wrong


def test_changing_a_section_header_is_caught():
    """The banner strings are what Execute's state machine arms on, so they
    are contract, not decoration."""
    original = nec_portal._AIP_HEADER
    nec_portal._AIP_HEADER = "                        --------- INPUT PARAMS ---------"
    try:
        ours, _err = run_deck(fixture_deck("dipole_free_space"))
    finally:
        nec_portal._AIP_HEADER = original
    assert section_walk(ours) != section_walk(fixture_out("dipole_free_space"))


def test_free_space_dipole_impedance_is_in_the_oracles_neighbourhood():
    """A smoke bound, not the differential harness: momwire's B-spline basis
    and nec2c's pulse basis disagree by a few percent on a 9-segment dipole
    and that is expected. 15% catches a wrong port, a wrong frequency, a
    missing ground, or a sign error."""
    ours = aip_tables(run_deck(fixture_deck("dipole_free_space"))[0])[0][0]
    theirs = aip_tables(fixture_out("dipole_free_space"))[0][0]
    z_ours = complex(float(ours[6]), float(ours[7]))
    z_theirs = complex(float(theirs[6]), float(theirs[7]))
    assert abs(z_ours - z_theirs) <= 0.15 * abs(z_theirs), (
        f"ours {z_ours} vs oracle {z_theirs}"
    )


def test_loaded_deck_spends_power_in_the_load():
    """LD 0 is a series R+L in the segment's current path, so the budget must
    show a structure loss and an efficiency below 100%."""
    text, _err = run_deck(fixture_deck("dipole_load_ld0"))
    budget = {
        line.split("=")[0].strip(): float(line.split("=")[1].split()[0])
        for line in text.splitlines()
        if "=" in line and ("POWER" in line or "LOSS" in line or "EFFICIENCY" in line)
    }
    assert budget["STRUCTURE LOSS"] > 0
    assert budget["INPUT POWER"] > budget["RADIATED POWER"] > 0
    assert 0 < budget["EFFICIENCY"] < 100
    assert budget["RADIATED POWER"] == pytest.approx(
        budget["INPUT POWER"] - budget["STRUCTURE LOSS"], rel=1e-3
    )


# --------------------------------------------------------------------------
# unit 3: the whole corpus, byte layout
# --------------------------------------------------------------------------


LAYOUT_NAMES = ANTENNA_NAMES


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_every_fixture_matches_the_oracle_column_layout(name):
    """The score the unit reports: every committed oracle printout, line for
    line and column for column.

    ``REPRESENTATIVE`` above keeps the diagnostics readable for the decks a
    reader is likely to be debugging; this one is the gate. Both are cheap —
    the whole corpus solves in about a second and a half.

    Nothing is held out. The ground-contact decks were, between #488 and
    momwire#487 U2, for a two-line printout gap that is now closed — see
    ``test_the_ground_contact_printout_says_the_ends_are_connected``.
    """
    ours = body_lines(printout(name))
    theirs = body_lines(fixture_out(name))
    assert len(ours) == len(theirs), (
        f"{name}: {len(ours)} body lines against the oracle's {len(theirs)}"
    )
    for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
        if "DIELECTRIC CONSTANT" in b:
            # The oracle glues the real and imaginary parts into one token, so
            # this line's "shape" is its value. It is an ignored section.
            continue
        assert layout_signature(a) == layout_signature(b), (
            f"{name} line {i}\n  ours   {a!r}\n  oracle {b!r}"
        )


@pytest.mark.parametrize("name", ANTENNA_NAMES)
def test_every_fixture_walks_the_oracle_section_order(name):
    assert section_walk(printout(name)) == section_walk(fixture_out(name))


def _first_segment_row(text: str) -> list[str]:
    """Segment 1's SEGMENTATION DATA row, as tokens (``I-`` is index 8)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if "SEGMENTATION DATA" in ln)
    return next(
        ln.split()
        for ln in lines[start:]
        if len(ln.split()) == 12 and ln.split()[0] == "1"
    )


@pytest.mark.parametrize("name", GROUND_CONTACT_NAMES)
def test_the_ground_contact_printout_says_the_ends_are_connected(name):
    """A wire END standing on the plane under a ``GE 1``, in both places the
    oracle records it — the gap #488 pinned, now closed and gated forwards.

    ``GE``'s SIGN is the ground-contact current expansion, and nec2c's
    ``conect()`` reads it twice over: it prints a SECOND ground-plane banner
    line naming the interpolation, and it tests each segment end against
    ``z = 0`` before it looks for a touching segment, writing the segment's
    OWN number in the connection column when it lands there. Both are the
    positive flag's alone — the ten ``GE -1`` fixtures print neither, which
    ``test_every_fixture_matches_the_oracle_column_layout`` covers by running
    over the whole corpus with nothing held out.

    What is still open underneath is momwire#489: the SOLVE does not consult
    the sign and interpolates either way, so a ``GE -1`` deck whose wire
    touches the plane gets the ``GE 1`` answer here and 57 - 4012j from the
    oracle. Every real deck in the class writes ``GE 1``, which is why these
    fixtures do and why their numbers agree to 0.91 %; this test is the
    printout half, and it says nothing about the physics half.
    """
    ours = printout(name)
    theirs = fixture_out(name)
    assert GROUND_CONTACT_BANNER in theirs, "the oracle stopped printing it"
    assert GROUND_CONTACT_BANNER in ours, "the interpolation banner went missing"
    assert _first_segment_row(theirs)[8] == "1", "the oracle's I- moved"
    assert _first_segment_row(ours)[8] == "1", (
        "segment 1 stands on the plane and reads as a free end again"
    )


def test_the_interpolation_banner_is_the_positive_ge_flags_alone():
    """The other side of the same claim, over the whole corpus: the banner
    follows ``GE``'s SIGN and nothing else.

    Ten fixtures write ``GE -1`` and get ``GROUND PLANE SPECIFIED.`` without
    the second line; the rest write ``GE 0`` and get neither. Read off both
    engines, so a deck cannot join either set on one side only.
    """
    for name in ANTENNA_NAMES:
        flag = next(
            (
                int(ln.split()[1])
                for ln in fixture_deck(name).splitlines()
                if ln.split()[:1] == ["GE"] and len(ln.split()) > 1
            ),
            0,
        )
        for who, text in (("ours", printout(name)), ("oracle", fixture_out(name))):
            assert ("GROUND PLANE SPECIFIED." in text) == (flag != 0), (who, name)
            assert (GROUND_CONTACT_BANNER in text) == (flag > 0), (who, name)
    assert set(GROUND_CONTACT_NAMES) == {
        name for name in ANTENNA_NAMES if GROUND_CONTACT_BANNER in fixture_out(name)
    }


# --------------------------------------------------------------------------
# momwire#415: GX / GR inside STRUCTURE SPECIFICATION
# --------------------------------------------------------------------------


def structure_specification(text: str) -> list[str]:
    """The STRUCTURE SPECIFICATION body: from its banner to the next one."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if "STRUCTURE SPECIFICATION" in ln)
    out = []
    for line in lines[start + 1 :]:
        if line.strip(" -") in _SECTION_MARKERS:
            break
        out.append(line.rstrip())
    return out


@pytest.mark.parametrize("name", REPLICATED_SPEC_NAMES)
def test_gx_and_gr_never_echo_as_data_card_lines(name):
    """The half of the echo units 1-2 got right, on both engines.

    ``GX``/``GR`` are geometry, and nec2c prints geometry inside STRUCTURE
    SPECIFICATION rather than in the ``DATA CARD No:`` list that starts after
    ``GE``. Units 1-2 added both to ``_portal._GEOMETRY_CARDS`` on that
    assumption; the committed oracle printouts are the proof, and this is the
    assertion that keeps it — an engine that regressed would print ``DATA CARD
    No:  1 GX`` where the oracle prints none.
    """
    mnemonic = "GX" if "gx" in name else "GR"
    echo = re.compile(rf"^\s*DATA CARD No:\s+\d+ {mnemonic}\b", re.MULTILINE)
    assert not echo.search(fixture_out(name)), "the oracle echoed it as a DATA CARD"
    assert not echo.search(printout(name))
    # ...and the card DID reach the geometry: the oracle's own segment total
    # counts the copies, and so does ours.
    total = re.compile(r"TOTAL SEGMENTS USED:\s+(\d+)")
    expected = {"dipole_gx_reflected_pair": "27", "dipole_gr_rotated_ring": "45"}[name]
    assert total.search(fixture_out(name)).group(1) == expected
    assert total.search(printout(name)).group(1) == expected


@pytest.mark.parametrize("name", REPLICATED_SPEC_NAMES)
def test_the_gx_gr_structure_specification_reproduces_the_oracle(name):
    """``_structure_rows`` reproduces the oracle's replicated sections to the
    byte: the annotation row on the line where the card fired, and a trailing
    GW numbered from the post-replication segment total (momwire#415)."""
    assert structure_specification(printout(name)) == structure_specification(
        fixture_out(name)
    )


# --------------------------------------------------------------------------
# unit 3: RP / NE / NH are execute cards
# --------------------------------------------------------------------------


def test_rp_runs_the_group_and_the_trailing_xq_runs_nothing():
    """nec2c executes on reading RP, so the deck's own ``XQ`` is a bare echo.

    ``dipole_rp_pattern`` is EX / FR / RP / XQ and the oracle prints ONE run:
    the RP echo, the whole solve, the pattern, then the XQ echo immediately
    followed by NX. An engine that ran the XQ too would print a second
    ANTENNA INPUT PARAMETERS table and a second pattern, and SimNEC would read
    a 2x1 sensor matrix where it expected 1x1.
    """
    text = run_deck(fixture_deck("dipole_rp_pattern"))[0]
    assert text.count("ANTENNA INPUT PARAMETERS") == 1
    assert text.count("RADIATION PATTERNS") == 1
    echoes = re.findall(r"DATA CARD No:\s+\d+ (\w\w)", text)
    assert echoes == ["EX", "FR", "RP", "XQ", "NX"]
    # The XQ echo is followed straight away by the NX echo — no blank-line
    # wrapper, because nothing ran.
    lines = [ln for ln in text.splitlines() if "DATA CARD No:" in ln]
    body = text.splitlines()
    xq = next(i for i, ln in enumerate(body) if ln == lines[-2])
    assert body[xq + 1] == lines[-1]


@pytest.mark.parametrize(
    ("name", "cards"),
    [
        ("dipole_ne_nearfield", ["EX", "FR", "NE", "XQ", "NX"]),
        ("dipole_nh_nearfield", ["EX", "FR", "NH", "XQ", "NX"]),
    ],
)
def test_near_field_cards_execute_too(name, cards):
    text = run_deck(fixture_deck(name))[0]
    assert re.findall(r"DATA CARD No:\s+\d+ (\w\w)", text) == cards
    assert text.count("ANTENNA INPUT PARAMETERS") == 1


def test_a_second_xq_after_a_fresh_ex_still_runs():
    """The no-op rule must not swallow a legitimate second group."""
    text = run_deck(fixture_deck("two_source_sensor_lines"))[0]
    assert text.count("ANTENNA INPUT PARAMETERS") == 2


# --------------------------------------------------------------------------
# unit 3: patterns
# --------------------------------------------------------------------------


def pattern_rows(text: str) -> list[list[str]]:
    rows, armed = [], False
    for line in text.splitlines():
        parts = line.split()
        if parts[:2] == ["DEGREES", "DEGREES"]:
            armed = True
            continue
        if not armed:
            continue
        if len(parts) not in (11, 12):
            armed = False
            continue
        rows.append(parts)
    return rows


def test_pattern_grid_follows_the_rp_card():
    """``RP 0 7 13 1001 0 0 30 30 1000``: 7 thetas x 13 phis, theta fastest."""
    rows = pattern_rows(run_deck(fixture_deck("dipole_rp_pattern"))[0])
    assert len(rows) == 7 * 13
    assert [float(r[0]) for r in rows[:7]] == [0, 30, 60, 90, 120, 150, 180]
    assert {float(r[1]) for r in rows} == {30.0 * i for i in range(13)}
    assert float(rows[7][1]) == 30.0 and float(rows[7][0]) == 0.0


def test_pattern_peak_gain_matches_the_engines_far_field():
    """The printout's gain is the workbench's gain.

    ``MomwireEngine.far_field`` normalises by source input power —
    ``eta0*k^2/(8*pi*P_in)`` — and so does this printout. If the two ever
    drift apart, a user comparing a SimNEC pattern against the antennaknobs
    plot for the same design sees two different antennas.
    """
    rows = pattern_rows(run_deck(fixture_deck("dipole_rp_pattern"))[0])
    peak = max(float(r[4]) for r in rows)
    # A half-wave dipole in free space: 2.15 dBi, and momwire's 9-segment
    # B-spline reads a shade under.
    assert 1.9 <= peak <= 2.3, peak


def test_a_pattern_null_prints_the_floor_and_blanks_the_sense_column():
    """Both come from nec2c's |E|^2 <= 1e-20 test, and both must agree.

    A row that keeps its SENSE word but floors its gain (or the reverse) means
    the two thresholds have drifted apart, and ``Execute``'s ptr arithmetic
    then reads the E-field columns off by one on exactly the rows where the
    field is real.
    """
    for name in ("dipole_rp_pattern", "dipole_rp_crossed_quadrature"):
        for row in pattern_rows(run_deck(fixture_deck(name))[0]):
            floored = float(row[4]) <= -900.0
            assert floored == (len(row) == 11), f"{name}: {row}"


def test_average_power_gain_is_about_unity_for_a_lossless_antenna():
    """A free-space dipole radiates everything it is fed, so the pattern
    averaged over 4*pi steradians has to come out near 1 — which is only true
    if the gain normaliser, the solid-angle quadrature and the E-field
    prefactor all agree with each other."""
    for name in ("dipole_rp_pattern", "dipole_rp_crossed_quadrature"):
        line = next(
            ln
            for ln in run_deck(fixture_deck(name))[0].splitlines()
            if "AVERAGE POWER GAIN" in ln
        )
        average = float(line.split(":")[1].split()[0])
        solid = float(line.split("(")[1].split(")")[0])
        assert 0.9 <= average <= 1.1, line
        # A full sphere is 4*PI; the crossed deck only sweeps the upper
        # hemisphere, so 2*PI.
        assert solid == (4.0 if name == "dipole_rp_pattern" else 2.0)


def test_pattern_e_field_scales_with_the_requested_range():
    """RFLD is a real range, not decoration: doubling it halves every E field
    and moves the EXP(-JKR)/R line with it."""
    deck = fixture_deck("dipole_rp_pattern")
    near = pattern_rows(run_deck(deck)[0])
    far = pattern_rows(run_deck(deck.replace("30 30 1000", "30 30 2000"))[0])
    for a, b in zip(near, far, strict=True):
        if float(a[4]) <= -900.0:
            continue
        assert float(b[-4]) == pytest.approx(0.5 * float(a[-4]), rel=1e-3)
        # ...while the GAIN, which is range-independent, does not move.
        assert float(b[4]) == pytest.approx(float(a[4]), abs=0.01)


# --------------------------------------------------------------------------
# unit 3: near fields
# --------------------------------------------------------------------------


def near_field_rows(text: str) -> list[list[str]]:
    rows, armed = [], False
    for line in text.splitlines():
        parts = line.split()
        if parts[:3] == ["METERS", "METERS", "METERS"]:
            armed = True
            continue
        if not armed:
            continue
        if len(parts) != 9:
            armed = False
            continue
        rows.append(parts)
    return rows


def test_near_field_grid_varies_x_fastest_then_y_then_z():
    rows = near_field_rows(run_deck(fixture_deck("dipole_ne_nearfield"))[0])
    points = [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
    assert points == [(x, 0.0, z) for z in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)]


def test_near_field_off_the_conductor_tracks_the_oracle():
    """The claim the mixed-potential form is here to support.

    Every grid point a metre off the wire must match nec2c to a few percent in
    magnitude and a degree in phase — that is a real cross-engine near-field
    agreement, not a layout check. Points ON the conductor are excluded and
    documented (grammar doc §11): a point-source quadrature has no business
    being evaluated inside the source.
    """
    for name in ("dipole_ne_nearfield", "dipole_nh_nearfield"):
        ours = near_field_rows(run_deck(fixture_deck(name))[0])
        theirs = near_field_rows(fixture_out(name))
        assert len(ours) == len(theirs)
        checked = 0
        for a, b in zip(ours, theirs, strict=True):
            if float(a[0]) == 0.0:  # on the wire (the dipole lies on the z axis)
                continue
            live = max(float(b[3 + 2 * c]) for c in range(3))
            for component in range(3):
                magnitude = 3 + 2 * component
                mine, oracle = float(a[magnitude]), float(b[magnitude])
                if oracle <= 1e-4 * live:
                    # A component the symmetry kills. nec2c prints its own
                    # fill dust there (2.4E-09) and it means nothing; we print
                    # an exact zero since momwire#464 floored this table, so
                    # all that is testable is that the component IS dead.
                    assert mine <= 1e-4 * live, f"{name}: {a} / {b}"
                    continue
                assert mine == pytest.approx(oracle, rel=0.02), f"{name}: {a} / {b}"
                assert float(a[magnitude + 1]) == pytest.approx(
                    float(b[magnitude + 1]), abs=1.0
                ), f"{name} phase: {a} / {b}"
                checked += 1
        assert checked >= 6, f"{name}: only {checked} components compared"


def test_near_field_on_the_conductor_is_documented_not_trusted():
    """The one place the near field does NOT track the oracle.

    nec2c prints the impressed source field on a driven segment (1.8 V/m =
    1 V over a 0.5559 m segment). We evaluate the same integral the rest of
    the table uses, at a point inside the source. It lands in the same decade
    and with the same sign, which is as much as a regularised point-source sum
    can claim — pinned here so the limitation stays visible rather than
    drifting silently.
    """
    row = next(
        r
        for r in near_field_rows(run_deck(fixture_deck("dipole_ne_nearfield"))[0])
        if (float(r[0]), float(r[2])) == (0.0, 0.0)
    )
    assert 0.5 <= float(row[7]) <= 5.0, row  # oracle prints 1.8000E+00
    assert abs(float(row[8])) > 150.0  # ...at -180 degrees


def test_a_pec_ground_doubles_the_near_field_sources():
    """A near-field grid over PEC ground must see the image, and it can only
    do that if the image's CHARGE is negated along with its current."""
    deck = (
        "CE dipole over pec ground with a near field grid\n"
        "GW 1 9 0. 0. 2.0 0. 0. 7.0 0.001\n"
        "GE -1\nGN 1\nEX 0 1 5 0 1.\nFR 0 1 0 0 14.1 0\n"
        "NE 0 1 1 1 5. 0. 4.5 0. 0. 0.\n"
        "XQ\n"
    )
    with_ground = near_field_rows(run_deck(deck)[0])
    free = near_field_rows(run_deck(deck.replace("GE -1\nGN 1\n", "GE 0\n"))[0])
    assert len(with_ground) == len(free) == 1
    assert float(with_ground[0][7]) != pytest.approx(float(free[0][7]), rel=1e-3)


# --------------------------------------------------------------------------
# TL and NT: solved, and printed (momwire#456 phase C)
# --------------------------------------------------------------------------
#
# The dialect used to be antenna-only and refused both cards BY NAME (#930).
# Phase C reversed that decision (design doc
# `docs/design/networks-move-into-the-engine.md`): unit 1 landed the parser,
# and this unit composes the resolved cards with the antenna's port admittance
# through `momwire.networks` and prints the two blocks nec2c prints for them.
#
# The battery below is the same seven decks the byte oracle and the
# cross-engine differential run, asked here for the printout's SHAPE — which
# banner, in which order, with which rows — because that is the half
# `test_portal_differential.py` deliberately does not check.

NETWORK_FIXTURES = (
    "dipole_nt_network",
    "dipole_tl_network",
    "dipole_tl_shunt_crossed",
    "dipole_tl_zero_length",
    "dipole_nt_all_zero",
    "dipole_ld_nt_colocated",
    "dipole_ex6_gyrator",
    "dipole_nt_after_xq",
)


@pytest.mark.parametrize("name", NETWORK_FIXTURES)
def test_every_network_fixture_answers(name):
    """No network deck may quietly fall back to the error path.

    The direct inversion of `test_the_network_fixtures_refuse_at_solve_time`,
    which stood here for one unit and pinned the staged message this replaces.
    """
    text = run_deck(fixture_deck(name))[0]
    assert "ERROR" not in text, "\n".join(
        ln for ln in text.splitlines() if "ERROR" in ln
    )
    assert "ANTENNA INPUT PARAMETERS" in text
    assert NX_ECHO.search(text), "no sentinel — SimNEC blocks forever"


@pytest.mark.parametrize("name", NETWORK_FIXTURES)
def test_a_network_deck_prints_both_blocks_in_the_oracle_order(name):
    """`NETWORK DATA`, then the excitation block, then the antenna's own.

    Order is the whole assertion: all three tables address `(tag, segment)`
    rows and two of them share a column layout, so a reader that met them in
    the wrong order would parse plausible numbers out of the wrong table.
    """
    text = run_deck(fixture_deck(name))[0]
    banners = [
        "---------- NETWORK DATA ----------",
        "STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS",
        "--------- ANTENNA INPUT PARAMETERS ---------",
    ]
    # LAST occurrence, not first: `dipole_nt_after_xq` runs two groups and only
    # the second has a network, so its first ANTENNA INPUT PARAMETERS precedes
    # every network banner in the file. Taken per group the order holds, and
    # the last of each is the last group's.
    at = [text.rindex(b) for b in banners]
    assert at == sorted(at), f"{name}: blocks out of order"


@pytest.mark.parametrize(
    ("name", "kinds"),
    (
        ("dipole_nt_network", ["NT"]),
        ("dipole_tl_network", ["TL"]),
        ("dipole_tl_shunt_crossed", ["TL", "NT"]),
        ("dipole_tl_zero_length", ["TL"]),
        ("dipole_ex6_gyrator", ["NT"]),
    ),
)
def test_the_network_table_header_re_emits_once_per_row_kind(name, kinds):
    """One banner, and one column header per KIND of row under it.

    `dipole_tl_shunt_crossed` is the deck that can show this: it carries a
    line and an admittance matrix, so its `NETWORK DATA` block has two column
    headers where every other network fixture has one. A renderer that emitted
    the header per ROW, or only ever once, agrees with every other fixture in
    the corpus and disagrees with this one.
    """
    text = run_deck(fixture_deck(name))[0]
    block = text[
        text.index("---------- NETWORK DATA ----------") : text.index(
            "STRUCTURE EXCITATION DATA"
        )
    ]
    assert block.count("---------- NETWORK DATA ----------") == 1
    assert block.count("TRANSMISSION LINE") == kinds.count("TL")
    assert block.count("ADMITTANCE MATRIX ELEMENTS") == kinds.count("NT")


def test_the_network_table_addresses_global_segment_numbers():
    """`TL 1 5 2 5` prints as `1 5 2 14`.

    The card addresses a segment RELATIVE to its tag and every table in this
    printout addresses it absolutely, so the row has to carry the translation.
    Segment 5 of tag 2 is the fourteenth segment of a structure whose first
    wire has nine, and printing `2 5` there would look entirely reasonable.
    """
    text = run_deck(fixture_deck("dipole_tl_network"))[0]
    row = next(
        ln for ln in text.splitlines() if "6.0000E+02" in ln and "STRAIGHT" in ln
    )
    assert row.split()[:4] == ["1", "5", "2", "14"]


def test_the_crossed_line_prints_its_impedance_positive_and_says_so():
    """NEC spells a crossed line as a NEGATIVE z0 and echoes |z0| with the
    word `CROSSED` in the type column. The sign is a polarity, not an
    impedance: a renderer that passed the card's field straight through would
    print `-4.5000E+02` and lose the word."""
    text = run_deck(fixture_deck("dipole_tl_shunt_crossed"))[0]
    row = next(ln for ln in text.splitlines() if "CROSSED" in ln)
    assert "4.5000E+02" in row and "-4.5000E+02" not in row


def test_a_zero_length_line_prints_the_distance_it_resolved_to():
    """`TL 1 5 2 5 450. 0.` — F2 = 0 means "measure it", and the LENGTH column
    shows what was measured. The two connection segments' centres are the
    origin and (1, 0, 0), so the answer is one metre, and it is visible in the
    printout before it is visible in any impedance."""
    text = run_deck(fixture_deck("dipole_tl_zero_length"))[0]
    row = next(
        ln for ln in text.splitlines() if "4.5000E+02" in ln and "STRAIGHT" in ln
    )
    assert "1.0000E+00" in row


def test_the_excitation_block_reads_the_segment_current_not_the_sources():
    """At a driven connection point the two tables print the SAME segment and
    DIFFERENT currents, and the difference is what the network carried.

    `dipole_nt_network` drives tag 1 segment 5 and hangs an NT on it, so the
    row appears in both blocks: the excitation block reports what the antenna
    took and `ANTENNA INPUT PARAMETERS` what the source delivered. An engine
    that printed one number twice would look completely plausible.
    """
    text = run_deck(fixture_deck("dipole_nt_network"))[0]
    excitation = text[
        text.index("STRUCTURE EXCITATION DATA") : text.index("ANTENNA INPUT PARAMETERS")
    ]
    antenna = text[text.index("ANTENNA INPUT PARAMETERS") :]
    at_gap = next(ln for ln in excitation.splitlines() if ln.split()[:2] == ["1", "5"])
    at_source = next(ln for ln in antenna.splitlines() if ln.split()[:2] == ["1", "5"])
    assert at_gap.split()[2:4] == at_source.split()[2:4], "same applied voltage"
    assert at_gap.split()[4:6] != at_source.split()[4:6], "the network carried nothing"


def test_the_gyrator_idiom_delivers_one_amp():
    """The manufactured `EX 6`: 52 of the 457 models bundled with 4nec2 build
    an ideal current source out of a phantom wire, an ordinary `EX 0` volt on
    it and a gyrator `NT` (Y12 = j1). The whole idiom is worth exactly one
    assertion — that the real segment ends up carrying 1 A — and nothing else
    in the corpus makes that claim.
    """
    text = run_deck(fixture_deck("dipole_ex6_gyrator"))[0]
    excitation = text[
        text.index("STRUCTURE EXCITATION DATA") : text.index("ANTENNA INPUT PARAMETERS")
    ]
    row = next(ln for ln in excitation.splitlines() if ln.split()[:2] == ["1", "5"])
    assert float(row.split()[4]) == pytest.approx(1.0, abs=1e-6)
    assert float(row.split()[5]) == pytest.approx(0.0, abs=1e-6)


def test_an_antenna_only_deck_prints_neither_block():
    """The blocks are the network's, not the printout's furniture."""
    text = run_deck(fixture_deck("dipole_free_space"))[0]
    assert "NETWORK DATA" not in text
    assert "STRUCTURE EXCITATION DATA" not in text


# The network-free twin of `dipole_nt_after_xq`: the same two dipoles, the same
# source and frequency, and no network card anywhere — so no gap is cut at the
# far endpoint at all. Written here rather than captured because its whole
# purpose is to be compared against a group of another deck.
_AFTER_XQ_CONTROL = (
    "CE control\n"
    "GW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\n"
    "GW 2 9 1.0 0. -2.5 1.0 0. 2.5 0.001\n"
    "GE 0\nEX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\n"
)


def _aip_row_lines(text: str) -> list[str]:
    """Every ANTENNA INPUT PARAMETERS data row VERBATIM, in printout order.

    The sibling ``_aip_rows`` below parses the same rows into floats; this one
    keeps the bytes, because the one thing it is used for is a byte comparison.
    """
    rows, armed = [], False
    for line in text.splitlines():
        if "ANTENNA INPUT PARAMETERS" in line:
            armed = True
        elif armed and len(line.split()) == 11 and line.split()[0].isdigit():
            rows.append(line)
            armed = False
    return rows


def test_a_network_read_after_an_execute_card_is_scoped_to_the_groups_after_it():
    """Retention runs FORWARD from the card and not backward.

    ``dipole_nt_after_xq`` runs the bare antenna, attaches an ``NT``, and runs
    again — one deck, two answers, in that order, which is what nec2c prints.
    No single-group deck can express this rule: every other network fixture
    states its cards before the first execute card, so an engine that ignored
    the scoping entirely would answer all of them correctly and both of this
    one's groups wrong.
    """
    text = run_deck(fixture_deck("dipole_nt_after_xq"))[0]
    before, after = text.split("DATA CARD No:   4 NT", 1)
    assert "NETWORK DATA" not in before, "the group that ran first got a network"
    assert "STRUCTURE EXCITATION DATA" not in before
    assert "NETWORK DATA" in after, "the group after the card got none"
    assert "STRUCTURE EXCITATION DATA" in after

    rows = _aip_row_lines(text)
    assert len(rows) == 2, rows
    assert rows[0].split()[4:6] != rows[1].split()[4:6], (
        "both groups answered the same current — the card moved nothing"
    )


def test_the_group_before_the_network_card_reproduces_the_control_exactly():
    """And it reproduces it to the BYTE, which is the stronger claim.

    The port set is the deck's, not the group's: the ``NT``'s far endpoint cuts
    a gap at segment 14 for the whole run, including the group that has no
    network. That gap has to be invisible there — an undriven, unloaded,
    un-networked port is pinned at zero volts and collapses to a plain shorted
    segment — and "invisible" is testable against a deck where the gap was
    never cut at all.

    Bit-identical rather than within-tolerance on purpose: this is the same
    matrix answering the same question, so anything less than equality would
    mean the gap is perturbing the structure, and a tolerance would hide how
    much.
    """
    ours = _aip_row_lines(run_deck(fixture_deck("dipole_nt_after_xq"))[0])[0]
    control = _aip_row_lines(run_deck(_AFTER_XQ_CONTROL)[0])[0]
    assert ours == control


def test_the_network_card_arms_the_next_execute_card_without_refilling():
    """An ``NT`` between two execute cards re-arms execution but rebuilds
    nothing: the oracle prints no FREQUENCY / LOADING / ENVIRONMENT / MATRIX
    TIMING preamble for the second group, the way a bare second ``XQ`` under
    one ``FR`` prints none. Pinned because the alternative — treating the card
    as an operator change — is a plausible reading that would insert a whole
    preamble block into the middle of this printout."""
    text = run_deck(fixture_deck("dipole_nt_after_xq"))[0]
    _before, after = text.split("DATA CARD No:   4 NT", 1)
    for banner in ("FREQUENCY :", "MATRIX TIMING", "STRUCTURE IMPEDANCE LOADING"):
        assert banner not in after, f"the NT group refilled: {banner}"


def test_a_network_card_after_the_execute_card_answers_without_it():
    """A card is retained from where it is READ, so a group that already ran
    is answered as though it were not there.

    That is the oracle's own behaviour and it is a strange-looking one — the
    deck states a network and the printout shows none — which is exactly why
    it is pinned: an engine that scoped the card to the whole DECK instead of
    to the groups after it would print a NETWORK DATA block here and change
    the impedance, and the deck would still look answered.
    """
    deck = (
        "CE nt after the execute card\n"
        "GW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\n"
        "GE 0\nEX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\n"
        "NT 1 3 1 7 0. 0.02 0. 0. 0. 0.02\n"
    )
    text = run_deck(deck)[0]
    assert "ERROR" not in text
    assert "ANTENNA INPUT PARAMETERS" in text
    assert "NETWORK DATA" not in text
    assert "is not part of this engine's nec2 dialect" not in text


# --------------------------------------------------------------------------
# issue #800: MP, the multicore hint SimNEC emits by itself
# --------------------------------------------------------------------------

MP_ADVISORY = "MP: multiProcessor 16 32"


def test_mp_deck_runs_instead_of_being_refused():
    """The reason the card had to land: SimNEC appends MP on structure SIZE
    alone (``NECSource.constructNECFile``, once ``sum(Wire.numSegments)``
    reaches ``getMPInfo()[0]``, default 256), so a big array arrives carrying
    one whether or not anybody asked. Refusing it refused the deck."""
    text = printout("dipole_mp_multiprocessor")
    assert "ERROR-NEC2C" not in text
    assert "ANTENNA INPUT PARAMETERS" in text


def test_mp_echo_and_advisory_are_the_oracles_bytes():
    """Both lines the card produces, against the committed oracle printout.

    The echo is layout — four integer fields, ``16`` and ``32`` in the first
    two — and the advisory is a literal, at column 0, so both can be compared
    verbatim rather than structurally.
    """
    ours = printout("dipole_mp_multiprocessor")
    theirs = fixture_out("dipole_mp_multiprocessor")

    echo = next(ln for ln in theirs.splitlines() if " MP" in ln.split("No:")[-1])
    assert echo in ours, f"our echo differs from the oracle's:\n  {echo!r}"
    assert echo.split()[4:6] == ["MP", "16"], echo

    assert MP_ADVISORY in theirs.splitlines(), "the fixture lost its advisory"
    assert MP_ADVISORY in ours.splitlines()


def test_mp_advisory_sits_between_the_environment_and_matrix_timing():
    """Its position — and the extra blank it carries — are the contract.

    nec2c prints the line straight after the ANTENNA ENVIRONMENT block with
    one blank of its own, so a multiprocessing run shows THREE blanks before
    MATRIX TIMING where a plain one shows two. Checked on both sides.
    """
    for text in (
        printout("dipole_mp_multiprocessor"),
        fixture_out("dipole_mp_multiprocessor"),
    ):
        lines = text.splitlines()
        index = lines.index(MP_ADVISORY)
        assert lines[index - 1].strip() == "FREE SPACE"
        assert lines[index + 1 : index + 4] == ["", "", ""]
        assert "MATRIX TIMING" in lines[index + 4]


def test_a_single_processor_mp_echoes_but_says_nothing():
    """``MP 1 32`` is still a card: it echoes, and prints no advisory. That
    threshold is why the corpus carries both forms."""
    ours = printout("dipole_mp_single_process")
    theirs = fixture_out("dipole_mp_single_process")
    assert "multiProcessor" not in ours
    assert "multiProcessor" not in theirs
    assert any(" MP" in ln and "DATA CARD No:" in ln for ln in ours.splitlines())


def test_mp_changes_nothing_but_the_card_lines():
    """The whole point of treating it as advisory, asserted rather than
    assumed: ``dipole_mp_multiprocessor`` IS ``dipole_free_space`` plus one
    card, and once the echo, the advisory and the shifted card ordinals are
    removed the two printouts are identical — ours and the oracle's alike.
    """

    def stripped(text: str) -> list[str]:
        """``body_lines`` — so the live FILL timing goes with the blanks and
        the banner — minus the MP card's own two lines and the card ordinals."""
        out = []
        for line in body_lines(text):
            if "multiProcessor" in line:
                continue
            if "DATA CARD No:" in line:
                # Inserting a card renumbers every echo after it, so the
                # ordinal goes; the MP echo itself goes with it.
                _ordinal, rest = line.split("No:")[1].strip().split(maxsplit=1)
                if rest.split()[0] != "MP":
                    out.append(rest)
                continue
            out.append(line)
        return out

    for plain, with_mp in (
        (printout("dipole_free_space"), printout("dipole_mp_multiprocessor")),
        (fixture_out("dipole_free_space"), fixture_out("dipole_mp_multiprocessor")),
    ):
        assert stripped(with_mp) == stripped(plain)


def test_mp_reprints_once_per_matrix_rebuild():
    """An FR sweep rebuilds per frequency, and the advisory goes with it."""
    deck = fixture_deck("dipole_fr_sweep").replace(
        "FR 0 3 0 0 28. 1.", "MP 16 32\nFR 0 3 0 0 28. 1."
    )
    text = run_deck(deck)[0]
    assert text.count("MATRIX TIMING") == 3
    assert text.count(MP_ADVISORY) == 3


def test_mp_is_not_an_arming_card():
    """Measured on the oracle: ``... XQ / MP 4 8 / XQ`` prints one block, not
    two. An MP alone does not make the next execute card a real run."""
    text = run_deck(
        "CE mp arming\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "EX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\nMP 4 8\nXQ\n"
    )[0]
    assert text.count("ANTENNA INPUT PARAMETERS") == 1
    assert text.count("MATRIX TIMING") == 1
    # ...and both execute cards are still echoed.
    assert len(re.findall(r"DATA CARD No:\s+\d+ XQ", text)) == 2


@pytest.mark.parametrize(
    ("card", "advisory"),
    [
        ("MP -3 -9", "MP: multiProcessor -3 -9"),
        # Measured on the oracle: -1 is NOT the silent single-processor case.
        # Its advisory test reads the field unsigned, so every negative prints
        # (and every negative also hangs it).
        ("MP -1 32", "MP: multiProcessor -1 32"),
    ],
)
def test_a_hostile_mp_never_hangs_the_daemon(card, advisory):
    """``MP -3 -9`` makes the ORACLE spin forever (measured: killed at 25 s).

    ``Execute.processResponse`` has no timeout, so an engine that inherited
    that behaviour would hang the SimNEC UI. This one echoes the card, prints
    its advisory with the numbers as given, finishes the run, and emits the
    sentinel.
    """
    out, err = deck_frame(
        "CE hostile mp\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        f"EX 0 1 5 0 1.\n{card}\nFR 0 1 0 0 30. 0\nXQ\n"
    )
    text = "\n".join(out)
    assert "ERROR-NEC2C" not in text
    assert advisory in text
    assert "ANTENNA INPUT PARAMETERS" in text
    assert NX_ECHO.search(text)
    assert err == []


# --------------------------------------------------------------------------
# issue #800: PT, the current-print toggle
# --------------------------------------------------------------------------


def currents_tables(text: str) -> list[list[list[str]]]:
    """The CURRENTS AND LOCATION rows, tokenised, one list per table.

    Ten tokens per row, which is what ends the table for a reader; the seg and
    tag numbers are fields 0 and 1.
    """
    tables: list[list[list[str]]] = []
    collecting = False
    for line in text.splitlines():
        parts = line.split()
        if parts[1:4] == ["CURRENTS", "AND", "LOCATION"]:
            tables.append([])
            collecting = False
            continue
        if parts[:3] == ["No:", "No:", "X"]:
            collecting = True
            continue
        if not collecting:
            continue
        if len(parts) != 10:
            collecting = False
            continue
        tables[-1].append(parts)
    return tables


def test_pt_minus_one_removes_the_whole_currents_section():
    """Not just the rows: the banner, the note, the blank and both column
    headers go too — which is why ``dipole_pt_toggle`` has one CURRENTS AND
    LOCATION section for its two runs, on both engines."""
    for text in (printout("dipole_pt_toggle"), fixture_out("dipole_pt_toggle")):
        assert text.count("CURRENTS AND LOCATION") == 1
        assert text.count("ANTENNA INPUT PARAMETERS") == 2


def test_pt_minus_two_restores_the_table():
    """``PT`` is a toggle held across execute cards, not a per-run flag: the
    second run of ``dipole_pt_toggle`` prints the full 18-segment table."""
    ours = currents_tables(printout("dipole_pt_toggle"))
    theirs = currents_tables(fixture_out("dipole_pt_toggle"))
    assert [len(t) for t in ours] == [len(t) for t in theirs] == [18]


def test_pt_zero_limits_the_table_to_the_named_tags_segments():
    """``PT 0 2 1 3`` prints tag 2's segments 1-3 — global 10 to 12. The
    addressing is EX's, so an absolute reading would print 1-3 instead and
    still look perfectly plausible."""
    ours = currents_tables(printout("dipole_pt_segment_range"))
    theirs = currents_tables(fixture_out("dipole_pt_segment_range"))
    assert len(ours) == len(theirs) == 1
    for rows in (ours[0], theirs[0]):
        assert [(r[0], r[1]) for r in rows] == [("10", "2"), ("11", "2"), ("12", "2")]


def test_an_all_zero_pt_range_prints_everything():
    """Measured on the oracle: ``PT 0 1 0 0`` and ``PT 0 2 0 0`` both print the
    whole table, so an empty range is "no restriction", not "no rows"."""
    for tag in (1, 2):
        deck = fixture_deck("dipole_pt_segment_range").replace(
            "PT 0 2 1 3", f"PT 0 {tag} 0 0"
        )
        assert len(currents_tables(run_deck(deck)[0])[0]) == 18


@pytest.mark.parametrize("flag", [-2, 1, 2, 3])
def test_the_other_pt_flags_print_the_ordinary_table(flag):
    """Stock NEC-2's receiving-pattern and normalised-current formats. This
    ae6ty build prints the plain full table for all of them — diffed against
    the same deck with no PT card at all — so they are read as no restriction
    rather than as a layout nobody has seen."""
    base = fixture_deck("dipole_pt_segment_range").replace("PT 0 2 1 3\n", "")
    deck = fixture_deck("dipole_pt_segment_range").replace("PT 0 2 1 3", f"PT {flag}")
    plain = [ln for ln in body_lines(run_deck(base)[0]) if "DATA CARD" not in ln]
    ours = [ln for ln in body_lines(run_deck(deck)[0]) if "DATA CARD" not in ln]
    assert ours == plain


def test_pt_is_not_an_arming_card():
    """It changes what a run prints, not what a run computes, so it cannot
    make a spent ``XQ`` into a fresh execution."""
    text = run_deck(
        "CE pt arming\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "EX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\nPT -1\nXQ\n"
    )[0]
    assert text.count("ANTENNA INPUT PARAMETERS") == 1
    assert text.count("CURRENTS AND LOCATION") == 1


# --------------------------------------------------------------------------
# issue #800 (tail): GD, the additional-ground-parameters card
# --------------------------------------------------------------------------

# The committed pairs: (with the card, the identical deck without it), one
# per ground the card can ride.
#
# ``dipole_gd_second_medium`` is the PEC one — the MININEC-type ground idiom
# on an elevated dipole. It was a refusal fixture between #458 and
# momwire#487 and is back to byte comparison with the rest.
#
# ``mininec_gd_without_gn_rp0`` (momwire#487) is ``dipole_rp_pattern`` plus a
# ``GD`` and NO ``GN`` at all, which is the third of the card's three
# situations: with no ground there is no image for a second medium to modify,
# so the card is inert for a reason unrelated to the two above and the same
# four assertions have to hold anyway.
GD_PAIRS = (
    ("dipole_gd_second_medium", "dipole_pec_ground"),
    ("dipole_gd_cliff_sommerfeld", "dipole_sommerfeld_ground"),
    ("mininec_gd_without_gn_rp0", "dipole_rp_pattern"),
)


def test_gd_deck_runs_instead_of_being_refused():
    """The reason the card had to land: SimNEC's EZNEC-derived examples
    (``Cardioid (EZNEC).ssn``, ``4-square (EZNEC).ssn``) carry a ``GD`` and
    ``NECSource`` forwards it, so refusing it failed those decks outright and
    the live session read back fabricated R = 0 / X = 0."""
    for name, _base in GD_PAIRS:
        text = printout(name)
        assert "ERROR-NEC2C" not in text, name
        assert "ANTENNA INPUT PARAMETERS" in text, name


# --------------------------------------------------------------------------
# momwire#487: the MININEC-type ground idiom, captured as the oracle answers it
# --------------------------------------------------------------------------

# The IDIOM itself: a ``GD`` in force under a ``GN 1`` at an execute card
# that will not read the record. momwire#458 refused this shape; momwire#487
# measured the oracle and found perfect-ground physics on the other side, so
# it is served letter-faithfully and each of these is byte-compared like any
# other fixture. Each is the idiom in a form the corpus did not already carry:
#
# * ``mininec_gp80_seam`` is 4nec2's OWN emitted deck, verbatim (capture 0038,
#   a three-element 80 m ground plane): ``GN 1`` + ``GD 0`` and no ``RP`` card
#   at all, which is the shape 26 of the 29 ``GN 3`` models in its bundle take.
#   It is the artifact the whole question is about, so it is in the corpus as
#   bytes rather than as a paraphrase.
# * ``mininec_vertical_rp0`` states the idiom on a quarter wave standing ON the
#   ground plane — the geometry the idiom exists for — with an explicit
#   ``RP 0``. ``dipole_gd_second_medium`` is the request-less form on an
#   elevated dipole, so neither the ground contact nor the request was covered.
# * ``mininec_vertical_gd2_rp0`` is that deck with the integer field set to 2,
#   which is the NEC-4 slot 4nec2 writes and the form every hand-written ``GD``
#   deck in its bundle uses. The oracle ignores the field: the two committed
#   printouts differ in that one echoed integer and in nothing else, which is
#   what says a parser may read F1-F4 and stop.
MININEC_IDIOM_NAMES = (
    "dipole_gd_second_medium",
    "mininec_gp80_seam",
    "mininec_vertical_rp0",
    "mininec_vertical_gd2_rp0",
)

# The rest of the same capture — the neighbouring shapes, each of which the
# refusal had to reach past for its own reason, and which are the controls
# that say serving the idiom did not swallow them. ``*_rp3_cliff_at_zero`` is
# the MININEC far field itself: the cliff at distance zero and height zero,
# which is exactly what 4nec2's manual says its ``GN 3`` expands to.
MININEC_NEIGHBOUR_NAMES = (
    "mininec_gd_reset_by_gn_rp0",
    "mininec_gd_without_gn_rp0",
    "mininec_vertical_rp3_clt",
    "mininec_vertical_rp3_ch",
    "mininec_vertical_rp3_cliff_at_zero",
)


@pytest.mark.parametrize("name", MININEC_IDIOM_NAMES)
def test_the_mininec_ground_fixtures_answer_plain_perfect_ground(name):
    """The flip, at the four forms of the idiom the corpus carries.

    The committed ``.out`` beside each of these is the oracle's own answer and
    it is PLAIN PERFECT GROUND — banner and all, the ``GD``'s only trace the
    DATA CARD echo. So the test is agreement rather than the divergence #458
    pinned here: the deck solves, prints the same banner, announces no second
    medium anywhere, and still emits the sentinel a blocked ``readLine()`` is
    waiting on. The numbers are ``test_portal_differential``'s; the bytes are
    ``test_every_fixture_matches_the_oracle_column_layout``'s, which these
    decks are no longer held out of.
    """
    theirs = fixture_out(name)
    assert "PERFECT GROUND" in theirs, f"{name}: the oracle moved"
    text = printout(name)
    assert "ERROR-NEC2C" not in text, name
    assert "PERFECT GROUND" in text, name
    assert "ANTENNA INPUT PARAMETERS" in text, name
    for banner in ("CLIFF", "SECOND MEDIUM", "FAR FIELD GROUND PARAMETERS"):
        assert banner not in text, f"{name}: {banner} on a request that cannot read it"
        assert banner not in theirs, f"{name}: the oracle grew a {banner} block"
    assert NX_ECHO.search(run_deck(fixture_deck(name))[0]), name


def test_the_gp80_seam_deck_is_the_request_less_form_4nec2_actually_emits():
    """The artifact, and what the oracle does with it.

    ``mininec_gp80_seam`` is not a paraphrase of the idiom — it is 4nec2's own
    emitted deck, so what it pins is the shape the frontend really sends: no
    ``RP`` card at all, the ``GD`` written AFTER the ``GN 1`` and after the
    ``EX``/``LD``, and an ``XQ`` that is the whole request. 26 of the 29
    ``GN 3`` models in the bundle run exactly this way, which is why it is the
    deck the whole question is about.

    The oracle's answer to it is one impedance table over PERFECT GROUND and
    nothing else: no pattern, no ground-parameters block, no announcement
    anywhere that a second medium was named. The ``GD`` reaches the printout
    once, in the DATA CARD echo, and reaches the answer not at all.
    """
    deck = fixture_deck("mininec_gp80_seam")
    mnemonics = [line.split()[0] for line in deck.splitlines()]
    assert "RP" not in mnemonics, "the seam deck grew a request"
    assert mnemonics.index("GD") == mnemonics.index("GN") + 1
    assert mnemonics[-1] == "XQ"

    out = fixture_out("mininec_gp80_seam")
    assert "PERFECT GROUND" in out
    assert "RADIATION PATTERNS" not in out
    assert "FAR FIELD GROUND PARAMETERS" not in out
    assert "SECOND MEDIUM" not in out
    echo = [ln for ln in out.splitlines() if " GD" in ln.split("No:")[-1]]
    assert len(echo) == 1, echo
    assert [float(v) for v in echo[0].split()[9:11]] == [13.0, 0.005]


def test_the_gd_integer_field_does_not_change_the_oracles_answer():
    """``GD 2`` is ``GD 0``, at the bytes.

    The two fixtures are one deck apart in that one integer, sharing a title
    line so nothing else can differ. The oracle's two printouts differ in
    exactly one line and that line is the ``GD`` echo — so the field is a slot
    NEC-4 reads and NEC-2 records and ignores, and an engine reading F1-F4 and
    stopping is faithful to it.
    """
    plain = fixture_out("mininec_vertical_rp0").splitlines()
    slot = fixture_out("mininec_vertical_gd2_rp0").splitlines()
    differing = [(a, b) for a, b in zip(plain, slot, strict=True) if a != b]
    assert len(differing) == 1, differing
    assert differing[0][0].split()[4] == "GD"
    assert differing[0][0].split()[5] == "0"
    assert differing[0][1].split()[5] == "2"
    assert differing[0][0].split()[6:] == differing[0][1].split()[6:]


@pytest.mark.parametrize("name", MININEC_NEIGHBOUR_NAMES)
def test_the_shapes_next_to_the_idiom_are_answered(name):
    """The idiom's edges, still where they were.

    Two of these have no second medium in force at the execute card (a ``GN``
    resets the four slots a ``GD`` writes, so ``GD``-then-``GN 1`` is plain
    perfect ground; a ``GD`` with no ``GN`` under it has no image to modify)
    and three ask a request that DOES read the record. They were the decks
    #458's refusal had to reach past, and they are what says the flip
    generalised the answer rather than moving the boundary: the cliff decks
    must still print a cliff, and the ``RP 0`` ones must still not.
    """
    text = printout(name)
    assert "ERROR-NEC2C" not in text, name
    assert "ANTENNA INPUT PARAMETERS" in text, name
    assert NX_ECHO.search(text), name
    reads_the_record = "RP 3 " in fixture_deck(name)
    assert ("CLIFF" in text) is reads_the_record, name
    assert ("CLIFF" in fixture_out(name)) is reads_the_record, name


def test_a_gn_card_resets_the_second_medium_a_gd_wrote():
    """The reset, at the answer rather than at the parser.

    ``GD`` then ``GN 1`` writes the same four /FPAT/ slots twice, so the
    second write wins and the deck reaches its execute card with nothing in
    force. Measured against the deck with its ``GD`` line taken out — which is
    the same deck — on our side, and against the oracle's own bytes for the
    ORDER: the committed ``mininec_gd_reset_by_gn_rp0`` printout is
    ``mininec_vertical_rp0``'s with the ``GN`` and ``GD`` echo lines swapped
    and nothing else moved, which is the whole of the difference one card
    order makes to NEC-2.
    """
    deck = fixture_deck("mininec_gd_reset_by_gn_rp0")
    bare = "".join(
        line + "\n" for line in deck.splitlines() if line.split()[:1] != ["GD"]
    )
    assert bare != deck, "the fixture lost its GD card"

    def stripped(text: str) -> list[str]:
        return [ln for ln in body_lines(text) if "DATA CARD No:" not in ln]

    assert stripped(run_deck(deck)[0]) == stripped(run_deck(bare)[0])

    idiom = fixture_out("mininec_vertical_rp0").splitlines()
    reset = fixture_out("mininec_gd_reset_by_gn_rp0").splitlines()
    differing = [(a, b) for a, b in zip(idiom, reset, strict=True) if a != b]
    assert [a.split()[4] for a, _b in differing] == ["GN", "GD"], differing
    assert [b.split()[4] for _a, b in differing] == ["GD", "GN"], differing


def test_the_cliff_requests_read_the_record_the_rp0_decks_ignore():
    """The measurement the whole idiom turns on, on ONE geometry.

    Four fixtures share the grounded vertical and its ``GN 1`` + ``GD``:
    ``mininec_vertical_rp0`` asks ``RP 0`` and gets the perfect-ground
    pattern, and the two ``RP 3`` decks ask a cliff and get the second medium.
    So "perfect-ground currents, real-ground far field" names a REQUEST MODE
    and not a ground type — which is why the ``RP 0`` decks above can be
    answered as perfect ground without losing anything, and why these two must
    not be.

    The two cliff decks carry one non-zero real each, and the oracle's own
    rows say which is which: at CLT = 10 m the specular point ``z*tan(theta)``
    clears the edge only in the last row of the sweep, so exactly the
    theta = 80 rows move; at CLT = 0 with CHT = 2 m the join is under every
    ray and almost every row does.
    """
    flat = _pattern_gains(fixture_out("mininec_vertical_rp0"))
    edge = _pattern_gains(fixture_out("mininec_vertical_rp3_clt"))
    depth = _pattern_gains(fixture_out("mininec_vertical_rp3_ch"))
    assert len(flat) == len(edge) == len(depth) == 45

    moved = {angles for angles, gain in edge.items() if gain != flat[angles]}
    assert moved and {theta for theta, _phi in moved} == {80.0}, sorted(moved)

    moved = {angles for angles, gain in depth.items() if gain != flat[angles]}
    assert len(moved) >= 40, sorted(moved)
    assert depth[(30.0, 0.0)] - flat[(30.0, 0.0)] < -10.0, "CHT moved nothing"

    for name in ("mininec_vertical_rp3_clt", "mininec_vertical_rp3_ch"):
        assert "CIRCULAR CLIFF" in fixture_out(name), name
    assert "CLIFF" not in fixture_out("mininec_vertical_rp0")


def test_the_cliff_at_zero_is_the_mininec_far_field_over_the_same_currents():
    """``mininec_vertical_rp3_cliff_at_zero`` — the request 4nec2's manual
    says its ``GN 3`` expands to, on the deck the idiom exists for.

    ``GN 1`` + ``GD 13. .005 0. 0.`` + ``RP 3``: distance zero and height
    zero, so medium 2 is under every ray and the whole pattern reflects off
    it, while the impedance stays the perfect-ground one. That is "PEC
    currents, real-ground far field" spelled as what it is — a REQUEST, not a
    ground type — and it is the same four fields ``mininec_vertical_rp0``
    carries and ignores.

    So the two decks differ in exactly one card and in the pattern alone: the
    ANTENNA INPUT PARAMETERS row is identical on the oracle's side, and the
    gains move by several dB at low angles (the finite ground eats what the
    perfect one reflected).
    """
    flat = fixture_out("mininec_vertical_rp0")
    cliff = fixture_out("mininec_vertical_rp3_cliff_at_zero")
    assert "PERFECT GROUND" in cliff, "the currents stopped being perfect-ground"
    assert "CIRCULAR CLIFF" in cliff
    assert aip_tables(flat)[0][0][6:8] == aip_tables(cliff)[0][0][6:8], (
        "the cliff reached the matrix"
    )

    flat_gain = _pattern_gains(flat)
    cliff_gain = _pattern_gains(cliff)
    # Every row but the theta = 0 ones, which are the vertical's own zenith
    # null (-999.99 dB): a reflection coefficient cannot move a zero.
    live = {a for a in flat_gain if a[0] != 0.0}
    moved = {a for a in live if cliff_gain[a] != flat_gain[a]}
    assert moved == live, sorted(live - moved)
    assert all(flat_gain[a] < -900.0 for a in flat_gain if a[0] == 0.0)
    assert cliff_gain[(80.0, 0.0)] - flat_gain[(80.0, 0.0)] < -5.0

    ours = _pattern_gains(printout("mininec_vertical_rp3_cliff_at_zero"))
    worst = max(abs(ours[a] - cliff_gain[a]) for a in cliff_gain)
    assert worst <= 0.5, f"cross-engine pattern gap {worst:.3f} dB"


def _pattern_gains(text: str) -> dict[tuple[float, float], float]:
    """{(theta, phi): TOTAL gain dB} off a RADIATION PATTERNS table."""
    gains, armed = {}, False
    for line in text.splitlines():
        tokens = line.split()
        if tokens[:2] == ["DEGREES", "DEGREES"]:
            armed = True
            continue
        if not armed:
            continue
        if len(tokens) not in (11, 12):
            armed = False
            continue
        gains[(float(tokens[0]), float(tokens[1]))] = float(tokens[4])
    return gains


@pytest.mark.parametrize(("name", "_base"), GD_PAIRS)
def test_gd_echo_is_the_oracles_bytes(name, _base):
    """The whole of the card's printed output, verbatim.

    ``dipole_gd_cliff_sommerfeld`` is the one that matters here: its card
    carries all four reals (``5. .001 20. -2.``), so the echo pins EPSR2,
    SIG2, CLT *and* CHT in their card-order columns. The Cardioid's own form
    leaves the last two at zero and would not.
    """
    ours = printout(name)
    theirs = fixture_out(name)
    echo = next(ln for ln in theirs.splitlines() if " GD" in ln.split("No:")[-1])
    assert echo in ours, f"our echo differs from the oracle's:\n  {echo!r}"
    assert echo.split()[4] == "GD"


def test_the_cliff_gd_echo_carries_all_four_reals_in_card_order():
    """``GD 0 0 0 0 5. .001 20. -2.`` -> EPSR2, SIG2, CLT, CHT, then zeros.

    Read off the oracle's own ``RP 2`` printout, which names them:
    ``EDGE DISTANCE= 20.00 METERS`` / ``HEIGHT= -2.00 METERS`` /
    ``RELATIVE DIELECTRIC CONST= 5.000`` / ``GROUND CONDUCTIVITY= 0.001``.
    """
    for text in (
        printout("dipole_gd_cliff_sommerfeld"),
        fixture_out("dipole_gd_cliff_sommerfeld"),
    ):
        echo = next(ln for ln in text.splitlines() if " GD" in ln.split("No:")[-1])
        reals = [float(v) for v in echo.split()[9:]]
        assert reals == [5.0, 0.001, 20.0, -2.0, 0.0, 0.0], echo
        assert [int(v) for v in echo.split()[5:9]] == [0, 0, 0, 0], echo


@pytest.mark.parametrize(("name", "_base"), GD_PAIRS)
def test_gd_adds_no_line_to_the_antenna_environment_block(name, _base):
    """Probed for and not there — including under ``GN 2``, where a second
    medium is the likeliest place an announcement would have appeared."""
    for text in (printout(name), fixture_out(name)):
        lines = text.splitlines()
        start = next(i for i, ln in enumerate(lines) if "ANTENNA ENVIRONMENT" in ln)
        end = next(i for i, ln in enumerate(lines) if "MATRIX TIMING" in ln)
        block = " ".join(lines[start:end]).upper()
        assert "SECOND MEDIUM" not in block, block
        assert "CLIFF" not in block, block
        assert "MEDIUM 2" not in block, block


@pytest.mark.parametrize(("name", "base"), GD_PAIRS)
def test_gd_changes_nothing_but_its_own_echo(name, base):
    """Asserted rather than assumed, exactly as ``MP`` is.

    Each fixture IS its base deck plus one card. Strip the ``GD`` echo and
    the ordinals it shifts and the two printouts are identical — ours and the
    oracle's alike. If that ever stops being true, ``GD`` has become physics
    and this engine is wrong to record it and move on.
    """

    def stripped(text: str) -> list[str]:
        out = []
        for line in body_lines(text):
            if "DATA CARD No:" in line:
                _ordinal, rest = line.split("No:")[1].strip().split(maxsplit=1)
                if rest.split()[0] != "GD":
                    out.append(rest)
                continue
            out.append(line)
        return out

    for plain, with_gd in (
        (printout(base), printout(name)),
        (fixture_out(base), fixture_out(name)),
    ):
        assert stripped(with_gd) == stripped(plain)


def test_gd_is_not_an_arming_card():
    """Measured on the oracle: ``... XQ / GD 2 0 0 0 13. .005 0. 0. / XQ``
    prints one block, not two."""
    text = run_deck(
        "CE gd arming\nGW 1 9 0. 0. 0.5 0. 0. 5.0 0.001\nGE 1\nGN 1\n"
        "EX 0 1 1 0 1.\nFR 0 1 0 0 14.1 0\nXQ\nGD 2 0 0 0 13. .005 0. 0.\nXQ\n"
    )[0]
    assert text.count("ANTENNA INPUT PARAMETERS") == 1
    assert text.count("MATRIX TIMING") == 1
    # ...and both execute cards are still echoed.
    assert len(re.findall(r"DATA CARD No:\s+\d+ XQ", text)) == 2


def test_the_comma_delimited_gd_simnec_sends_parses_like_the_spaced_form():
    """``NECSource`` writes the card comma-delimited — ``GD
    2,0,0,0,13.,.005,0.,0.`` is the literal Cardioid line. Measured identical
    to the spaced form on the oracle; identical here too.

    Over the Cardioid's own ``GN 1`` again: #458 moved this probe to a
    ``GN 2`` while the pairing refused, and momwire#487 moved it back, so the
    free-format read is measured on the literal line SimNEC sends.
    """
    head = "CE gd commas\nGW 1 9 0. 0. 0.5 0. 0. 5.0 0.001\nGE 1\nGN 1\n"
    tail = "EX 0 1 1 0 1.\nFR 0 1 0 0 14.1 0\nXQ\n"
    commas = run_deck(head + "GD 2,0,0,0,13.,.005,0.,0.\n" + tail)[0]
    spaces = run_deck(head + "GD 2 0 0 0 13. .005 0. 0.\n" + tail)[0]
    assert body_lines(commas) == body_lines(spaces)
    assert "ERROR-NEC2C" not in commas


def test_a_bare_gd_is_a_card_like_any_other():
    """The oracle echoes ``GD`` with four zero integers and six zero reals and
    runs the deck; nothing here may trip over the missing fields."""
    text = run_deck(
        "CE bare gd\nGW 1 9 0. 0. 0.5 0. 0. 5.0 0.001\nGE 1\nGN 1\nGD\n"
        "EX 0 1 1 0 1.\nFR 0 1 0 0 14.1 0\nXQ\n"
    )[0]
    assert "ERROR-NEC2C" not in text
    echo = next(ln for ln in text.splitlines() if " GD" in ln.split("No:")[-1])
    assert [float(v) for v in echo.split()[5:]] == [0.0] * 10, echo


_CLIFF_HEAD = (
    "CE gd cliff pattern\nGW 1 9 0. 0. 0.5 0. 0. 5.0 0.001\nGE 1\n"
    "GN 0 0 0 0 13. .005\nGD 0 0 0 0 5. .001 20. -2.\n"
    "EX 0 1 1 0 1.\nFR 0 1 0 0 14.1 0\n"
)


def test_the_rp_modes_that_would_use_a_gd_are_still_refused():
    """The fidelity gate, narrowed by issue #802 to the modes still out.

    NEC-2 reaches the second medium in the far field ALONE, and there only
    through ``RP``'s cliff and ground-screen modes. Two of those five now run
    (see the tests below); these four do not, and for reasons that are not the
    cliff's:

    * ``RP 1`` is the surface wave — a different banner (``RADIATED FIELDS
      NEAR GROUND``), a different row shape carrying ``E(RADIAL)``, and a
      ``GFLD`` this engine has no equivalent of;
    * ``RP 4``-``6`` all want a radial wire ground screen, which momwire
      cannot model. That is the same reason ``GN``'s ``NRADL`` field is
      refused, and running one as bare ground would be a wrong answer rather
      than a refusal. 5 and 6 carry a cliff too, but the screen is what stops
      them.

    Issue #829: every refused mode must also lead with a token-0 ``ERROR:``
    line so SimNEC's ``"NEC ERROR (1)"`` warning frame fires.
    """
    for mode in (1, 4, 5, 6):
        text = run_deck(_CLIFF_HEAD + f"RP {mode} 3 3 1001 0 0 30 30 1000\nXQ\n")[0]
        assert f"RP mode {mode} is not supported" in text, mode
        assert NX_ECHO.search(text), mode
        error_lines = [ln for ln in text.splitlines() if ln.split()[:1] == ["ERROR:"]]
        assert error_lines, mode
        assert f"RP mode {mode}" in error_lines[0], mode
        # A refusal must not leave half a report behind it either.
        assert "FAR FIELD GROUND PARAMETERS" not in text, mode
    # ...and the mode SimNEC actually writes runs, second medium or not.
    text = run_deck(_CLIFF_HEAD + "RP 0 3 3 1001 0 0 30 30 1000\nXQ\n")[0]
    assert "ERROR-NEC2C" not in text
    assert "RADIATION PATTERNS" in text
    assert "FAR FIELD GROUND PARAMETERS" not in text


def test_the_rp_cliff_modes_consume_the_gd_instead_of_refusing_it():
    """Issue #802: the two modes a ``GD`` is FOR now run.

    They were refused with 1 and 4-6 until #802, which was honest but left
    Ward's EZNEC-derived examples — whose cliff parameters land on ``RP 3`` —
    with no answer at all. Each must now print the block the oracle prints,
    name the right cliff in it, and echo the card's own four numbers into it
    in card order.
    """
    for mode, word in ((2, "LINEAR"), (3, "CIRCULAR")):
        text = run_deck(_CLIFF_HEAD + f"RP {mode} 3 3 1001 0 0 30 30 1000\nXQ\n")[0]
        assert "ERROR-NEC2C" not in text, mode
        assert "FAR FIELD GROUND PARAMETERS" in text, mode
        assert f"--- {word} CLIFF ---" in text, mode
        assert "RADIATION PATTERNS" in text, mode
        assert "EDGE DISTANCE=     20.00 METERS" in text, mode
        assert "HEIGHT=     -2.00 METERS" in text, mode
        assert "RELATIVE DIELECTRIC CONST=      5.000" in text, mode
        assert "GROUND CONDUCTIVITY=      0.001 MHOS" in text, mode


def test_the_second_medium_actually_moves_the_cliff_modes_gains():
    """Lifting the refusal is only worth it if the card now changes an answer.

    Same deck, same geometry, same ground — one digit of the ``RP`` card
    apart. ``RP 0`` cannot see the second medium at all, so its table is the
    control; ``RP 2`` and ``RP 3`` must both differ from it, and from EACH
    OTHER, because a straight edge and a circular one only agree along the
    azimuth that crosses both.

    The theta grid has to reach grazing to say anything. The reflection point
    of a segment at height ``z`` is ``z·tan(theta)`` out, so this 5 m vertical
    does not reach a 20 m edge until about 76 degrees — a 0/30/60 sweep sees a
    flat world and would pass this test with the card ignored.
    """

    def gains(mode):
        text = run_deck(_CLIFF_HEAD + f"RP {mode} 4 3 1001 0 0 28 30 1000\nXQ\n")[0]
        rows, armed = {}, False
        for line in text.splitlines():
            parts = line.split()
            if parts[:2] == ["DEGREES", "DEGREES"]:
                armed = True
            elif armed and len(parts) in (11, 12):
                rows[(float(parts[0]), float(parts[1]))] = float(parts[4])
            elif armed:
                armed = False
        return rows

    flat, linear, circular = gains(0), gains(2), gains(3)
    assert set(flat) == set(linear) == set(circular)
    assert linear != flat, "RP 2 read the second medium and nothing moved"
    assert circular != flat, "RP 3 read the second medium and nothing moved"
    assert linear != circular, "a linear and a circular cliff cannot agree everywhere"


def test_a_cliff_mode_with_no_second_medium_still_prints_the_block():
    """``RDPAT`` prints FAR FIELD GROUND PARAMETERS on the MODE, not on the
    card. A cliff mode whose deck never sent a ``GD`` and never put the
    fields on its ``GN`` gets the block with four zeros in it — measured on
    the oracle, and the reason :func:`_far_field_ground_lines` renders a
    missing record rather than skipping the block."""
    text = run_deck(
        "CE cliff mode with no gd\nGW 1 9 0. 0. 0.5 0. 0. 5.0 0.001\nGE 1\nGN 1\n"
        "EX 0 1 1 0 1.\nFR 0 1 0 0 14.1 0\nRP 2 3 2 1001 0 0 30 90 1000\nXQ\n"
    )[0]
    assert "ERROR-NEC2C" not in text
    assert "--- LINEAR CLIFF ---" in text
    assert "EDGE DISTANCE=      0.00 METERS" in text
    assert "RELATIVE DIELECTRIC CONST=      0.000" in text


# --------------------------------------------------------------------------
# issue #802: RFLD = 0, the gain-only form
# --------------------------------------------------------------------------


def _pattern_rows(text: str) -> list[list[str]]:
    """Every RADIATION PATTERNS data row, as raw token lists."""
    rows, armed = [], False
    for line in text.splitlines():
        parts = line.split()
        if parts[:2] == ["DEGREES", "DEGREES"]:
            armed = True
        elif armed and len(parts) in (11, 12):
            rows.append(parts)
        elif armed:
            armed = False
    return rows


def test_rfld_zero_drops_the_range_header_and_keeps_the_gain():
    """The gain-only form, against the same deck read out at 1000 m.

    Two NEC-2 thresholds are both spelt ``1e-20`` and only come apart here.
    ``DB10`` clamps the LINEAR POWER GAIN, which never depended on the
    range — so the three gain columns must be identical between the two runs,
    to the printed digit. The blank-SENSE test clamps the field as ``FFLD``
    returns it, BEFORE the range scaling — so it must reach the same verdict
    on both runs even though the printed E columns differ by a factor of
    ``RFLD``, three decades here.

    An engine that floored on the printed field instead would pass at 1000 m
    and quietly grow lobes in its own nulls at ``RFLD = 0``.
    """
    head = (
        "CE gain only\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "EX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\n"
    )
    at_range = run_deck(head + "RP 0 5 3 1001 0 0 45 45 1000\nXQ\n")[0]
    gain_only = run_deck(head + "RP 0 5 3 1001 0 0 45 45 0\nXQ\n")[0]

    # Shape: the RANGE / EXP(-JKR)/R pair is printed at a range and not at all
    # without one, and it is the ONLY line that leaves.
    assert "RANGE:" in at_range and "EXP(-JKR)/R:" in at_range
    assert "RANGE:" not in gain_only and "EXP(-JKR)/R:" not in gain_only
    assert "RADIATION PATTERNS" in gain_only
    assert len(at_range.splitlines()) - len(gain_only.splitlines()) == 3

    ranged, bare = _pattern_rows(at_range), _pattern_rows(gain_only)
    assert len(ranged) == len(bare) == 15
    for a, b in zip(ranged, bare, strict=True):
        assert len(a) == len(b), f"the SENSE column changed with the range: {a} / {b}"
        # theta, phi, VERTC, HORIZ, TOTAL — none of them see the range.
        assert a[:5] == b[:5], f"a gain column moved with the range: {a} / {b}"
        # ...and E goes up by exactly RFLD, the 1/r the range header carried.
        e_at, e_bare = float(a[-4]), float(b[-4])
        if e_at:
            assert e_bare / e_at == pytest.approx(1000.0, rel=1e-3), (a, b)


# --------------------------------------------------------------------------
# unit 3: robustness — the daemon must survive anything on stdin
# --------------------------------------------------------------------------

BAD_DECKS = {
    "a card nobody has ever seen": (
        "CE bogus\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "ZZ 1 2 3\nEX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\n",
        "ZZ",
    ),
    "a non-numeric field": (
        "CE malformed\nGW 1 9 0. 0. -2.5 0. 0. banana 0.001\nGE 0\n"
        "EX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\n",
        "banana",
    ),
    "a one-letter mnemonic": (
        "CE short mnemonic\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "G\nEX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\n",
        "MNEMONIC",
    ),
    "a zero-segment wire": (
        "CE zero segments\nGW 1 0 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "EX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\n",
        "ERROR-NEC2C",
    ),
    "a segment that does not exist": (
        "CE bad segment\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "EX 0 1 99 0 1.\nFR 0 1 0 0 30. 0\nXQ\n",
        "out of range",
    ),
    "no excitation at all": (
        "CE undriven\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\nFR 0 1 0 0 30. 0\nXQ\n",
        "no EX card",
    ),
    "a fractional MP field": (
        # The oracle refuses this one too, with NON-NUMERICAL CHARACTER '.' IN
        # INTEGER FIELD — MP's two fields are #Proc and blockSize and neither
        # has a fractional reading.
        "CE bad mp\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "EX 0 1 5 0 1.\nMP 2.7 8.3\nFR 0 1 0 0 30. 0\nXQ\n",
        "MP field",
    ),
    "an MP with a word in it": (
        "CE worded mp\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "EX 0 1 5 0 1.\nMP lots fast\nFR 0 1 0 0 30. 0\nXQ\n",
        "lots",
    ),
    "a GD with a word in it": (
        # The ORACLE's free-format reader silently SKIPS a non-numeric token
        # and shifts the rest left, so `GD 2 0 0 0 marsh .005 0. 0.` echoes
        # .005 as EPSR2 — a wrong answer dressed as a right one. This engine
        # names the token instead, on the same error path every other card
        # uses, and still emits the sentinel.
        "CE worded gd\nGW 1 9 0. 0. 0.5 0. 0. 5.0 0.001\nGE 1\nGN 1\n"
        "GD 2 0 0 0 marsh .005 0. 0.\nEX 0 1 1 0 1.\nFR 0 1 0 0 14.1 0\nXQ\n",
        "marsh",
    ),
    "a current source we do not model": (
        "CE ex type 6\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
        "EX 6 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\n",
        "EX type 6",
    ),
}


@pytest.mark.parametrize(("label", "case"), sorted(BAD_DECKS.items()))
def test_a_bad_deck_reports_and_still_emits_the_sentinel(label, case):
    """Whatever is wrong, the NX echo goes out.

    ``Execute.processResponse`` blocks in ``readLine()`` with no timeout
    (grammar doc §2, §10.1). A replacement engine that dies, or that reports
    an error and forgets the sentinel, hangs the SimNEC UI rather than showing
    a message — which is strictly worse than a wrong answer.

    Issue #829 (Ward's 2026-08-08 reply): every refusal now also trips
    Execute's ``"NEC ERROR (1)"`` warning frame on purpose (token 0 exactly
    ``ERROR:``), so the user sees *why* nothing loaded instead of staring at
    a session that looks like it just did nothing. That used to be exactly
    what this test forbade; #829 reverses the call.
    """
    deck, marker = case
    out, err = deck_frame(deck)
    text = "\n".join(out)
    assert NX_ECHO.search(text), f"{label}: no NX sentinel"
    assert marker in text, f"{label}: error does not mention {marker!r}:\n{text[-500:]}"
    assert any(line.split()[:1] == ["ERROR:"] for line in text.splitlines()), (
        f"{label}: no line trips Execute's token-0 `ERROR:` warning frame"
    )
    assert err == []


def test_the_daemon_survives_a_bad_deck_and_runs_the_next_one():
    """Residency is the whole point: one broken deck must not end the process.

    SimNEC never restarts the engine between decks (``NEC2Daemon.destroy()``
    is the only teardown), so a deck that raises has to be reported and
    stepped over, leaving the loop ready for the next ``NX``.
    """
    stdin = io.StringIO(
        "CE bogus\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\nZZ 1\nNX\n"
        + fixture_deck("dipole_free_space")
        + "\nNX\n"
        + fixture_deck("dipole_rp_pattern")
        + "\nNX\n"
    )
    out = io.StringIO()
    assert main([], stdin=stdin, stdout=out, stderr=io.StringIO()) == 0
    text = out.getvalue()
    assert len(NX_ECHO.findall(text)) == 3, "one sentinel per deck, bad ones included"
    assert text.count("ERROR-NEC2C") == 1
    # ...and the decks after the bad one really ran.
    assert text.count("ANTENNA INPUT PARAMETERS") == 2
    assert text.count("RADIATION PATTERNS") == 1


def test_a_blank_deck_still_frames():
    """An empty body between two NX cards is legal input and must not hang."""
    out, err = deck_frame("")
    assert NX_ECHO.search("\n".join(out))
    assert err == []


# --------------------------------------------------------------------------
# packaging: the console script SimNEC is pointed at (unit 4)
# --------------------------------------------------------------------------

ENTRY_POINT_NAME = "momwire-nec2c"


def _console_scripts() -> dict[str, str]:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    return tomllib.loads(text)["project"]["scripts"]


def test_the_console_script_name_passes_simnecs_filename_check():
    """``nec2/NEC2PortalDialog`` accepts an engine on its FILENAME alone.

    The check is a lowercased substring test for ``nec2c`` on the configured
    command's file name — nothing inside the file is consulted until the
    ``-version`` probe. Renaming the entry point to something tidier
    (``momwire-portal``) makes SimNEC refuse it outright, so the name is
    contract, not cosmetics.
    """
    scripts = _console_scripts()
    assert ENTRY_POINT_NAME in scripts, (
        f"pyproject [project.scripts] must ship {ENTRY_POINT_NAME!r}; has {sorted(scripts)}"
    )
    assert "nec2c" in ENTRY_POINT_NAME.lower()
    # The sibling dialect prefixes select a DIFFERENT column layout
    # (checkNEC42Fields, samplesWidth 12) that this engine does not emit.
    assert "nec5" not in ENTRY_POINT_NAME.lower()
    assert "nec42" not in ENTRY_POINT_NAME.lower()


def test_the_console_script_name_carries_no_out_substring():
    """The second filename trap, documented since 2026-08-08 and untested
    until #846 phase III.

    ``nec2/Execute.testCommand`` refuses outright — "Can't execute 'out'
    file:" — any command whose FULL PATH contains the substring ``out``
    (grammar doc, "Corrections to the body above"). It is a guard against a
    user pointing the engine field at a printout file, and it does not care
    that our name is a console script rather than a ``.out``.

    A test can only pin the half we own. ``momwire-nec2c`` is chosen so the
    NAME can never trip it — which is not free: ``momwire-nec2c-output``,
    ``nec2c-out``, or any tidy-up along those lines would make SimNEC refuse
    the engine with a message that names neither momwire nor the real cause.

    **The other half is the user's install prefix and cannot be tested from
    here.** SimNEC matches the whole path, so a venv at
    ``~/checkouts/routing/.venv`` or ``C:\\Users\\scout\\...`` refuses this
    engine no matter what it is called. That belongs in the usage page's
    troubleshooting, not in an assertion: the fix is to install momwire
    somewhere else, and nothing in this repo can make that unnecessary.
    """
    assert "out" not in ENTRY_POINT_NAME.lower(), (
        f"{ENTRY_POINT_NAME!r} contains 'out'; SimNEC's testCommand refuses "
        "any command path containing that substring"
    )


def test_the_console_script_target_resolves():
    """The ``module:attr`` string must actually import — a typo here is only
    discovered by a user whose SimNEC session dies at the version probe."""
    target = _console_scripts()[ENTRY_POINT_NAME]
    module_name, _, attr = target.partition(":")
    assert attr, f"{target!r} names no callable"
    entry = getattr(importlib.import_module(module_name), attr)
    assert entry is nec_portal.main
    assert callable(entry)


def test_the_entry_point_runs_from_an_unrelated_cwd(tmp_path):
    """SimNEC launches the engine via ``sh -c`` with cwd=$HOME.

    Nothing in the daemon may resolve a relative path — not the fixtures, not
    the momwire import, not a config file. Both halves of the protocol are
    exercised out of a directory that has no relationship to the checkout: the
    ``-version`` probe SimNEC gates on, and one real deck framed by ``NX``.
    """
    src_root = str(Path(nec_portal.__file__).resolve().parents[2])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([src_root, os.environ.get("PYTHONPATH", "")]),
    }
    argv = [sys.executable, "-m", "momwire.portal"]

    probe = subprocess.run(
        [*argv, "-version"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == [nec_portal.PROBE_VERSION]

    solve = subprocess.run(
        argv,
        cwd=tmp_path,
        env=env,
        input=fixture_deck("dipole_free_space") + "\nNX\n",
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert solve.returncode == 0, solve.stderr
    assert NX_ECHO.search(solve.stdout), "no sentinel — SimNEC would block forever"
    assert "ANTENNA INPUT PARAMETERS" in solve.stdout
    assert f"VERSION:{nec_portal.BANNER_VERSION}" in solve.stdout


def test_selftest_passes_and_reports(tmp_path, monkeypatch):
    """`momwire-nec2c --selftest` is the deployment smoke for boxes with no
    checkout (the Windows live-session path): it must pass here, from an
    unrelated cwd, and print the PASS verdict on its own line."""
    import io

    from momwire.portal import main

    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    rc = main(["--selftest"], stdout=out)
    text = out.getvalue()
    assert rc == 0
    assert text.rstrip().endswith("PASS")
    assert "FAIL" not in text


# --- the --basis flag --------------------------------------------------------


def _run_main(argv, deck=""):
    import io

    from momwire.portal import _portal as nec_portal

    out = io.StringIO()
    err = io.StringIO()
    rc = nec_portal.main(argv, stdin=io.StringIO(deck), stdout=out, stderr=err)
    return rc, out.getvalue(), err.getvalue()


def test_basis_flag_unknown_name_fails_fast_and_nonzero():
    """A typo'd basis must fail the -version probe at configure time, not
    silently serve the default."""
    rc, out, _ = _run_main(["--basis", "nope", "-version"])
    assert rc == 3 and "choices:" in out


def test_basis_flag_rides_the_version_probe():
    """SimNEC probes `<full command line> -version`; the probe line must be
    unchanged (Double-parsed by Execute) with any valid --basis present."""
    from momwire.portal import PROBE_VERSION

    rc, out, _ = _run_main(["--basis", "sinusoidal-galerkin-converged", "-version"])
    assert rc == 0 and out == f"{PROBE_VERSION}\n"


def test_basis_flag_solves_and_stamps_the_banner():
    """The alternate basis answers a deck, and the PRINTOUT banner records
    which physics answered (+sgc) — the -version line never does."""
    deck = (
        "CE basis\n"
        "GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n"
        "GE 0\nEX 0 1 6 0 1.\nFR 0 1 0 0 14.0 1\nXQ\nNX\n"
    )
    rc, out, err = _run_main(["--basis=sinusoidal-galerkin-converged"], deck=deck)
    assert rc == 0 and err == ""
    assert "VERSION:nec2c.ae6ty.momwire.9.1+sgc" in out
    assert "ANTENNA INPUT PARAMETERS" in out
    rows = [
        ln
        for ln in out.splitlines()
        if ln.startswith("    1 ") and len(ln.split()) == 11
    ]
    assert rows, "no AIP data row under the alternate basis"


def test_default_basis_banner_is_unchanged():
    deck = (
        "CE basis\n"
        "GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n"
        "GE 0\nEX 0 1 6 0 1.\nFR 0 1 0 0 14.0 1\nXQ\nNX\n"
    )
    rc, out, _ = _run_main([], deck=deck)
    assert rc == 0
    assert "VERSION:nec2c.ae6ty.momwire.9.1\n" in out and "+sg" not in out


# --- --basis sinusoidal: the NEC-closest rung (issue #822) -------------------

# Point-matched sinusoidal has neither the B-spline family's KCL-port solve nor
# the Galerkin family's ported operator, so `_y_and_port_coeffs` grows a third
# branch that reproduces `compute_y_matrix`'s algebra. These pin the copy to
# momwire's own: if either drifts, the daemon's single fill and the library's
# per-source refill stop being the same solve.


def _sin_solver(**kwargs):
    """A 9-segment 5 m dipole on plain SinusoidalSolver, fed at its centre."""
    import numpy as np
    from momwire import SinusoidalSolver

    return SinusoidalSolver(
        wires=[np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])],
        n_per_edge_per_wire=[[9]],
        feeds=[(0, 2.5, 1.0)],
        wavelength=nec_portal.C_LIGHT / 14.0e6,
        wire_radius=0.001,
        **kwargs,
    )


def test_the_sinusoidal_shim_reproduces_momwires_own_y_matrix():
    import numpy as np

    y_shim, _x = nec_portal._y_and_port_coeffs(_sin_solver())
    y_lib = np.asarray(_sin_solver().compute_y_matrix(), dtype=np.complex128)
    assert np.allclose(y_shim, y_lib, rtol=1e-10, atol=0.0)


def test_the_sinusoidal_shim_columns_are_the_one_volt_drive_coefficients():
    """Column j of X must be the solve momwire would do for a 1 V drive at
    port j — that identity is what lets `solve_group` reuse one fill for
    every excitation (``coeffs = X @ V``)."""
    import numpy as np

    _y, x = nec_portal._y_and_port_coeffs(_sin_solver())
    _z, alpha = _sin_solver().compute_impedance()
    driven = x @ np.array([1.0 + 0.0j])
    assert np.allclose(driven, alpha, rtol=1e-10, atol=0.0)


# The classes that exercise everything the shim's coefficients feed: a
# Sommerfeld and a perfect ground (both carried on the fill), a lumped load
# and a series RLC (the power-budget and port-algebra paths), a multi-wire
# structure with a junction, the PT readout, and a pattern (which resamples the
# solved current through `currents_at_knots`).
#
# The two network decks left this roster in #930 — TL and NT are out of
# dialect — and the classes they stood for (a second gap the drive does not
# reach; port algebra outside the fill) are covered by the loaded decks and the
# multi-wire one that replaced them.  `dipole_gd_second_medium` stood in the
# PEC slot until #458 refused it (the MININEC-type ground idiom).  momwire#487
# un-refused it and it did not come back here: the class it exercised on the
# fill was the image ground, its own base deck holds that slot, and a `GD`
# never reached a fill to begin with.
_SIN_HARD_FIXTURES = (
    "dipole_sommerfeld_ground",
    "dipole_pec_ground",
    "dipole_load_ld0",
    "split_dipole_qq",
    "dipole_load_ld4",
    "dipole_pt_toggle",
    "dipole_rp_pattern",
)


def _aip_rows(text: str) -> list[list[float]]:
    """Every ANTENNA INPUT PARAMETERS data row, as floats past the tag/seg
    pair."""
    rows, inside = [], False
    for line in text.splitlines():
        if line.strip(" -") == "ANTENNA INPUT PARAMETERS":
            inside = True
            continue
        if not inside:
            continue
        tokens = line.split()
        if len(tokens) == 11 and tokens[0].isdigit() and tokens[1].isdigit():
            rows.append([float(t) for t in tokens[2:]])
        elif rows and not line.strip():
            inside = False
    return rows


@pytest.mark.parametrize("name", _SIN_HARD_FIXTURES)
def test_sinusoidal_basis_answers_the_hard_fixture_classes(name):
    import math

    deck = (FIXTURE_DIR / f"{name}.deck").read_text()
    rc, out, err = _run_main(["--basis", "sinusoidal"], deck=deck)
    assert rc == 0 and err == ""
    assert "ERROR-NEC2C" not in out, f"{name} took the error path under +sin"
    missing = set(section_walk(fixture_out(name))) - set(section_walk(out))
    assert not missing, f"{name} lost sections under +sin: {sorted(missing)}"
    rows = _aip_rows(out)
    assert rows, f"no AIP data row for {name} under +sin"
    assert all(math.isfinite(v) for row in rows for v in row), (
        f"non-finite AIP value for {name} under +sin"
    )


def test_sinusoidal_basis_impedance_tracks_the_nec2c_fixture():
    """The point-matched sinusoidal basis is the closest of the roster to
    NEC-2's own formulation, so the free-space dipole's driving-point
    impedance has to sit near the oracle's — a looser bound than a digit
    match, but tight enough to catch a wrong RHS scaling or a dropped
    ``-1/h``."""
    deck = (FIXTURE_DIR / "dipole_free_space.deck").read_text()
    rc, out, err = _run_main(["--basis", "sinusoidal"], deck=deck)
    assert rc == 0 and err == ""
    ours = complex(*_aip_rows(out)[0][4:6])
    theirs = complex(*_aip_rows(fixture_out("dipole_free_space"))[0][4:6])
    assert abs(ours - theirs) / abs(theirs) < 0.05, f"{ours} vs {theirs}"


def test_sinusoidal_basis_answer_differs_from_the_default_basis():
    """Disabled-path probe: every other +sin test would still pass if the
    ``_BASES`` entry silently served the default B-spline solver (its answer
    is inside the 5% oracle bound too, and the banner stamps regardless of
    what solved). The two bases genuinely disagree on this deck — measured
    79.205+45.150j (+sin) vs 79.524+46.003j (default), 1.0% apart — so a
    collapse to the default is detectable."""
    deck = (FIXTURE_DIR / "dipole_free_space.deck").read_text()
    _rc, out_sin, _err = _run_main(["--basis", "sinusoidal"], deck=deck)
    _rc, out_default, _err = _run_main([], deck=deck)
    z_sin = complex(*_aip_rows(out_sin)[0][4:6])
    z_default = complex(*_aip_rows(out_default)[0][4:6])
    assert abs(z_sin - z_default) / abs(z_default) > 0.003, (z_sin, z_default)


def test_sinusoidal_basis_stamps_the_banner():
    deck = (
        "CE basis\n"
        "GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n"
        "GE 0\nEX 0 1 6 0 1.\nFR 0 1 0 0 14.0 1\nXQ\nNX\n"
    )
    rc, out, err = _run_main(["--basis", "sinusoidal"], deck=deck)
    assert rc == 0 and err == ""
    assert "VERSION:nec2c.ae6ty.momwire.9.1+sin" in out
    assert _aip_rows(out)


def test_sinusoidal_has_no_converged_variant():
    """The zero-width point gap has no collocation RHS (momwire#212), so the
    flag must not offer a name the solver would refuse — same constraint the
    CLI's MOMWIRE_BASIS_VARIANTS records."""
    rc, out, _ = _run_main(["--basis", "sinusoidal-converged", "-version"])
    assert rc == 3 and "sinusoidal-converged" not in out.split("choices:")[1]


# --- bspline-d1 (issue #821): the degree axis, same solver class -----------


def test_bspline_d1_basis_flag_solves_and_stamps_the_banner():
    """`bspline-d1` is BSplineSolver with degree=1 bound — same public
    `compute_port_solution()` path as plain `bspline` (momwire#232; degree is
    just a constructor knob), so it answers a deck and stamps +bs1."""
    deck = (
        "CE basis\n"
        "GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n"
        "GE 0\nEX 0 1 6 0 1.\nFR 0 1 0 0 14.0 1\nXQ\nNX\n"
    )
    rc, out, err = _run_main(["--basis=bspline-d1"], deck=deck)
    assert rc == 0 and err == ""
    assert "VERSION:nec2c.ae6ty.momwire.9.1+bs1" in out
    assert "ANTENNA INPUT PARAMETERS" in out
    rows = [
        ln
        for ln in out.splitlines()
        if ln.startswith("    1 ") and len(ln.split()) == 11
    ]
    assert rows, "no AIP data row under the bspline-d1 basis"


@pytest.mark.parametrize(
    "name",
    _SIN_HARD_FIXTURES,
)
def test_bspline_d1_basis_solves_hard_fixture_classes(name):
    """bs1 is degree=1 on the exact same BSplineSolver class the default
    basis uses (engines/momwire.py:_parity_for_solver changes the mesh
    parity, not the code path), so it must clear every hard fixture class the
    default `bspline` basis clears: Sommerfeld ground, GD second medium, TL
    and NT networks, LD4 loading, PT toggling, RP patterns."""
    rc, out, err = _run_main(
        ["--basis", "bspline-d1"], deck=fixture_deck(name) + "\nNX\n"
    )
    assert rc == 0
    assert err == ""
    assert "ANTENNA INPUT PARAMETERS" in out
    tables = aip_tables(out)
    assert tables and tables[0], f"no AIP data rows for {name}"
    for table in tables:
        for row in table:
            for tok in row:
                assert math.isfinite(float(tok)), f"{name}: non-finite token {tok!r}"


def test_bspline_d1_free_space_impedance_within_loose_bound_of_committed_oracle():
    """The oracle fixture is nec2c's own answer (not ours) — the same "loose
    cross-engine smoke bound" style as the default-basis test at the top of
    this file, applied to the alternate degree.

    Measured (issue #821 build): R=78.06 vs oracle 79.24 (1.5% off — inside
    5%); X=40.27 vs oracle 45.36 (11.2% off). The coarser tent basis (bs1,
    degree=1) is a full segmentation-order coarser than the oracle's own
    basis, so 5% was optimistic for X on a 2-segment-mesh-equivalent free
    dipole; 15% is the bound this measurement actually supports. R stays at
    the tighter 5% since it tracks a shallower dependency on basis order."""
    _rc, out, _err = _run_main(
        ["--basis", "bspline-d1"],
        deck=fixture_deck("dipole_free_space") + "\nNX\n",
    )
    ours = aip_tables(out)[0][0]
    theirs = aip_tables(fixture_out("dipole_free_space"))[0][0]
    r_ours, x_ours = float(ours[6]), float(ours[7])
    r_theirs, x_theirs = float(theirs[6]), float(theirs[7])
    assert abs(r_ours - r_theirs) / r_theirs < 0.05, (r_ours, r_theirs)
    assert abs(x_ours - x_theirs) / abs(x_theirs) < 0.15, (x_ours, x_theirs)


def test_bspline_d1_answer_differs_from_the_default_degree():
    """Disabled-path probe: if `_build_engine` dropped the `{"degree": 1}`
    kwargs and served the default degree-2 solver, every other +bs1 test
    would still pass — the d2 answer is inside both oracle bounds and the
    banner stamps regardless of what solved. The two degrees genuinely
    disagree here — measured X=40.27 (d1) vs X=46.00 (d2), 12% apart — so a
    silently-ignored kwarg is detectable."""
    deck = fixture_deck("dipole_free_space") + "\nNX\n"
    _rc, out_d1, _err = _run_main(["--basis", "bspline-d1"], deck=deck)
    _rc, out_d2, _err = _run_main([], deck=deck)
    x_d1 = float(aip_tables(out_d1)[0][0][7])
    x_d2 = float(aip_tables(out_d2)[0][0][7])
    assert abs(x_d1 - x_d2) / abs(x_d2) > 0.05, (x_d1, x_d2)


# --- hmatrix / arrayblock (issue #830): the large-array accelerators --------

# These two entries are NOT a physics axis: HMatrixSolver and ArrayBlockSolver
# are BSplineSolver subclasses that solve the SAME operator with a compressed
# representation and GMRES instead of a dense fill and an LU. That makes the
# roster's usual disabled-path probe (differ-from-default) useless — a silent
# collapse to plain `bspline` would print the same digits — so the armour here
# is two-sided: the printout must AGREE with the default basis, and a spy must
# see the accelerated solve actually run.


def _accel_pair(cls, volts=(1.0, 0.0), **kwargs):
    """Two 9-segment 5 m dipoles 3 m apart, both centre-fed — a two-port
    structure small enough to solve densely for comparison."""
    import numpy as np

    return cls(
        wires=[
            np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]]),
            np.array([[3.0, 0.0, -2.5], [3.0, 0.0, 2.5]]),
        ],
        n_per_edge_per_wire=[[9], [9]],
        feeds=[(0, 2.5, volts[0]), (1, 2.5, volts[1])],
        wavelength=nec_portal.C_LIGHT / 14.0e6,
        wire_radius=0.001,
        **kwargs,
    )


def _route_spy(solver):
    """Record which of the two B-spline solve routes momwire takes, without
    disturbing either. Instance-attribute spies intercept the calls
    ``compute_port_solution()`` makes internally (momwire#232 replaced the
    portal's own spy shim, but the route question is still observable from
    outside the public API this way)."""
    routes: list[str] = []
    dense = solver._solve_with_kcl_ports

    def spy_dense(z, v, kcl_a, overwrite=False):
        routes.append("dense")
        return dense(z, v, kcl_a, overwrite=overwrite)

    solver._solve_with_kcl_ports = spy_dense
    accel = getattr(solver, "_solve_hmatrix", None)
    if accel is not None:

        def spy_accel(h, kcl_a, b):
            routes.append(f"accel:{type(h).__name__}")
            return accel(h, kcl_a, b)

        solver._solve_hmatrix = spy_accel
    return routes


def _accel_classes():
    from momwire import ArrayBlockSolver, HMatrixSolver

    return {"hmatrix": HMatrixSolver, "arrayblock": ArrayBlockSolver}


@pytest.mark.parametrize("basis", ["hmatrix", "arrayblock"])
def test_the_accelerated_shim_reproduces_momwires_own_y_matrix(basis):
    """The accelerated subclasses never reach `_solve_with_kcl_ports` — their
    `compute_y_matrix` runs the constrained GMRES in `_solve_hmatrix` — so the
    shim spies that instead, and the Y it hands back has to be the library's
    own to the iterative tolerance."""
    import numpy as np

    cls = _accel_classes()[basis]
    y_shim, _x = nec_portal._y_and_port_coeffs(_accel_pair(cls))
    y_lib = np.asarray(_accel_pair(cls).compute_y_matrix(), dtype=np.complex128)
    assert np.allclose(y_shim, y_lib, rtol=1e-8, atol=0.0)


@pytest.mark.parametrize("basis", ["hmatrix", "arrayblock"])
def test_the_accelerated_shim_captures_the_gmres_solve_not_a_dense_fallback(basis):
    """The one thing no printout test can see: WHICH solve answered. Without
    it, `_BASES` could name the accelerated class while every deck quietly
    took the dense path (`_hmatrix_unsupported`) and nothing would fail."""
    cls = _accel_classes()[basis]
    solver = _accel_pair(cls)
    assert not solver._hmatrix_unsupported()
    routes = _route_spy(solver)
    _y, x = nec_portal._y_and_port_coeffs(solver)
    assert routes and all(r.startswith("accel:") for r in routes), routes
    assert x.shape[1] == 2


@pytest.mark.parametrize("basis", ["hmatrix", "arrayblock"])
def test_the_accelerated_shim_columns_are_the_one_volt_drive_coefficients(basis):
    """Column j of X must be the coefficients momwire would solve for a 1 V
    drive at port j and nothing else on — the identity `solve_group` leans on
    when it turns one fill into every excitation (``coeffs = X @ V``). Checked
    against the DENSE B-spline solve of the same mesh, which is the answer the
    accelerator approximates (measured max relative deviation 8e-16 for
    hmatrix, 2e-11 for arrayblock, both far inside the 1e-6 solve_tol)."""
    import numpy as np
    from momwire import BSplineSolver

    cls = _accel_classes()[basis]
    _y, x = nec_portal._y_and_port_coeffs(_accel_pair(cls))
    for j, volts in enumerate([(1.0, 0.0), (0.0, 1.0)]):
        _z, alpha = _accel_pair(BSplineSolver, volts=volts).compute_impedance()
        assert np.allclose(x[:, j], alpha, rtol=1e-6, atol=1e-9 * np.abs(alpha).max())


def test_the_dense_fallback_route_is_still_captured():
    """`_hmatrix_unsupported()` is singular enrichment and nothing else — not
    mesh size, not ground — so this is the ONLY way to reach the dense path on
    an accelerated class. The portal never asks for enrichment, but the shim
    keeps the dense spy wired so a momwire that grows a new fallback degrades
    to a slower answer rather than a `PortalError`."""
    import numpy as np
    from momwire import HMatrixSolver

    solver = _accel_pair(HMatrixSolver, use_singular_enrichment=True)
    assert solver._hmatrix_unsupported()
    routes = _route_spy(solver)
    y_shim, x = nec_portal._y_and_port_coeffs(solver)
    assert routes == ["dense"], routes
    y_lib = np.asarray(
        _accel_pair(HMatrixSolver, use_singular_enrichment=True).compute_y_matrix(),
        dtype=np.complex128,
    )
    assert np.allclose(y_shim, y_lib, rtol=1e-8, atol=0.0)
    assert x.shape == (18, 2)


# The same seven classes the roster gates on (#826/#827), shared with the
# sinusoidal gate above so one edit moves both. MEASURED: all seven take the
# ACCELERATED route under
# both bases — the ground decks included, because `_hmatrix_unsupported()`
# tests only `use_singular_enrichment` and every ground model the portal emits
# (PEC image, reflection coefficient, Sommerfeld) is carried on the fast path.
# No fixture class reaches the dense fallback.
_ACCEL_HARD_FIXTURES = _SIN_HARD_FIXTURES


@pytest.mark.parametrize("basis", ["hmatrix", "arrayblock"])
@pytest.mark.parametrize("name", _ACCEL_HARD_FIXTURES)
def test_the_accelerators_answer_the_hard_fixture_classes(name, basis):
    rc, out, err = _run_main(["--basis", basis], deck=fixture_deck(name) + "\nNX\n")
    assert rc == 0 and err == ""
    assert "ERROR-NEC2C" not in out, f"{name} took the error path under {basis}"
    missing = set(section_walk(fixture_out(name))) - set(section_walk(out))
    assert not missing, f"{name} lost sections under {basis}: {sorted(missing)}"
    tables = aip_tables(out)
    assert tables and tables[0], f"no AIP data rows for {name} under {basis}"
    for table in tables:
        for row in table:
            for tok in row:
                assert math.isfinite(float(tok)), f"{name}: non-finite token {tok!r}"


@pytest.mark.parametrize("basis", ["hmatrix", "arrayblock"])
def test_the_accelerators_agree_with_the_default_basis_and_the_oracle(basis):
    """Agreement, not difference, is the probe here: same physics as
    `bspline`, so the accelerated answer must land on the default's to the
    iterative solve tolerance (measured: identical to every printed digit,
    79.524+46.003j) while still clearing the 5% oracle bound (measured 0.77%
    from nec2c's 79.240+45.364j). A collapse to a DIFFERENT basis is what this
    catches; a collapse to dense `bspline` is caught by the spy test above."""
    deck = fixture_deck("dipole_free_space") + "\nNX\n"
    rc, out, err = _run_main(["--basis", basis], deck=deck)
    assert rc == 0 and err == ""
    _rc, out_default, _err = _run_main([], deck=deck)
    ours = aip_tables(out)[0][0]
    theirs = aip_tables(out_default)[0][0]
    oracle = aip_tables(fixture_out("dipole_free_space"))[0][0]
    z_ours = complex(float(ours[6]), float(ours[7]))
    z_default = complex(float(theirs[6]), float(theirs[7]))
    z_oracle = complex(float(oracle[6]), float(oracle[7]))
    assert abs(z_ours - z_default) / abs(z_default) < 0.005, (z_ours, z_default)
    assert abs(z_ours - z_oracle) / abs(z_oracle) < 0.05, (z_ours, z_oracle)


@pytest.mark.parametrize("basis,suffix", [("hmatrix", "+hm"), ("arrayblock", "+ab")])
def test_the_accelerators_stamp_the_banner(basis, suffix):
    deck = (
        "CE basis\n"
        "GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n"
        "GE 0\nEX 0 1 6 0 1.\nFR 0 1 0 0 14.0 1\nXQ\nNX\n"
    )
    rc, out, err = _run_main(["--basis", basis], deck=deck)
    assert rc == 0 and err == ""
    assert f"VERSION:nec2c.ae6ty.momwire.9.1{suffix}" in out
    assert aip_tables(out)[0]


@pytest.mark.parametrize("basis", ["hmatrix", "arrayblock"])
def test_the_accelerators_ride_the_version_probe_unchanged(basis):
    from momwire.portal import PROBE_VERSION

    rc, out, _err = _run_main(["--basis", basis, "-version"])
    assert rc == 0 and out == f"{PROBE_VERSION}\n"


def test_the_lattice_fft_path_engages_and_the_shim_still_agrees():
    """The engaged-path probe for `arrayblock`'s reason to exist. A 4x4 grid
    of identical 3-segment half-wave dipoles meets every FFT gate (one block
    shape, a regular lattice, P >= 16), and `require_lattice_fft=True` turns a
    miss into `LatticeFFTUnavailable` naming the unmet gate rather than a
    silent degradation to the parent H-matrix. The spy then pins that the
    operator the shim's X came out of really is the spectral one, and the
    answer is checked against the dense solve of the same mesh.

    Runtime ~0.4 s (48 bases): the lattice gate is about the FFT bookkeeping,
    not about size, so 16 three-segment elements engage it and this stays a
    normal-suite test rather than a `heavy_mesh` one."""
    import numpy as np
    from momwire import ArrayBlockSolver, BSplineSolver

    wavelength = nec_portal.C_LIGHT / 14.0e6
    arm = 0.25 * wavelength
    pitch = 0.5 * wavelength
    wires = [
        np.array([[ix * pitch, iy * pitch, -arm], [ix * pitch, iy * pitch, arm]])
        for ix in range(4)
        for iy in range(4)
    ]

    def build(cls, **kwargs):
        return cls(
            wires=wires,
            n_per_edge_per_wire=[[3]] * len(wires),
            feeds=[(0, arm, 1.0), (5, arm, 0.0)],
            wavelength=wavelength,
            wire_radius=0.001,
            **kwargs,
        )

    solver = build(ArrayBlockSolver, require_lattice_fft=True)
    routes = _route_spy(solver)
    y_fft, x_fft = nec_portal._y_and_port_coeffs(solver)
    assert routes == ["accel:LatticeArrayBlock"], routes

    y_dense, x_dense = nec_portal._y_and_port_coeffs(build(BSplineSolver))
    assert np.allclose(y_fft, y_dense, rtol=1e-6, atol=0.0)
    assert np.allclose(x_fft, x_dense, rtol=1e-5, atol=1e-8 * np.abs(x_dense).max())


def test_require_lattice_fft_names_the_unmet_gate_on_a_single_pair():
    """The other half of the engaged-path proof: on a deck with nothing to
    exploit the FFT gate is genuinely unmet and momwire says which one — so
    the passing case above cannot be a vacuous assertion."""
    from momwire import ArrayBlockSolver, LatticeFFTUnavailable

    solver = _accel_pair(ArrayBlockSolver, require_lattice_fft=True)
    with pytest.raises(LatticeFFTUnavailable):
        nec_portal._y_and_port_coeffs(solver)


# --- --basis razor / razor-nec5: the NEC-5 formulation twin ------------------
#
# #432 put razor on `deck.BASES`, but `RazorSolver` had no
# `compute_port_solution()` — the portal's `_y_and_port_coeffs` (every basis
# above goes through the ONE call, no per-family branch since #232) raised
# `AttributeError` the moment a deck actually solved under either roster
# name, which is why neither had a live gate here. The sharing audit's #429
# rank-9 item closes it: `_y_and_port_coeffs` drives razor exactly like every
# other family now.
#
# Known gap (filed as its own follow-up issue, #439, not fixed here):
# `_port_signs` assumes every `PortPlan` site has a
# matching `RazorSolver.feeds` entry, which holds for a driven site and for
# a site that is BOTH fed and loaded (`_sites()` merges the two into one
# `PortSite`), but not for a LOAD-ONLY site on a DIFFERENT segment from
# every `EX` — razor bakes that one straight into `lumped_loads`
# (`docs/razor-solver.md` "A load-only site is not a port here"), so it
# never reaches `RazorSolver.feeds` at all, and `_port_signs` indexes past
# the end of the list. The load-and-ground deck below sidesteps it by
# loading the FED segment (a legal, documented razor configuration in its
# own right — `docs/razor-solver.md`'s "Z_driven identity"), which is what
# the gate below actually exercises end to end.


def _razor_solver(**kwargs):
    """A 9-segment 5 m dipole on plain RazorSolver, fed at its centre — the
    same shape `_sin_solver` builds, for the same shim comparison."""
    import numpy as np
    from momwire import RazorSolver

    return RazorSolver(
        wires=[np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])],
        n_per_edge_per_wire=[[8]],
        feeds=[(0, 2.5, 1.0)],
        wavelength=nec_portal.C_LIGHT / 14.0e6,
        wire_radius=0.001,
        **kwargs,
    )


def test_the_razor_shim_reproduces_momwires_own_y_matrix():
    import numpy as np

    y_shim, _x = nec_portal._y_and_port_coeffs(_razor_solver())
    y_lib = np.asarray(_razor_solver().compute_y_matrix(), dtype=np.complex128)
    assert np.allclose(y_shim, y_lib, rtol=1e-10, atol=0.0)


def test_the_razor_shim_columns_are_the_one_volt_drive_coefficients():
    """Column j of X must be the solve momwire would do for a 1 V drive at
    port j — the identity `solve_group` leans on to turn one fill into
    every excitation (``coeffs = X @ V``)."""
    import numpy as np

    _y, x = nec_portal._y_and_port_coeffs(_razor_solver())
    _z, alpha = _razor_solver().compute_impedance()
    driven = x @ np.array([1.0 + 0.0j])
    assert np.allclose(driven, alpha, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize(
    "basis,suffix", [("razor", "+razor"), ("razor-nec5", "+razor5")]
)
def test_razor_basis_flag_solves_and_stamps_the_banner(basis, suffix):
    deck = (
        "CE razor basis\n"
        "GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n"
        "GE 0\nEX 0 1 6 0 1.\nFR 0 1 0 0 14.0 1\nXQ\nNX\n"
    )
    rc, out, err = _run_main(["--basis", basis], deck=deck)
    assert rc == 0 and err == ""
    assert f"VERSION:nec2c.ae6ty.momwire.9.1{suffix}" in out
    assert "ANTENNA INPUT PARAMETERS" in out
    rows = _aip_rows(out)
    assert rows, f"no AIP data row under --basis {basis}"
    assert all(math.isfinite(v) for row in rows for v in row)


# A deck exercising a LOAD and a GROUND together in one pass: a PEC ground
# (`GN 1`, wire clear of the plane — the same geometry
# `dipole_pec_ground.deck` uses) plus a series RLC load (`LD 0`) on the SAME
# segment the `EX` card drives, so the site is one `RazorSolver` port that is
# both fed and loaded (`docs/razor-solver.md`'s "Z_driven identity" case) —
# legal today, and the case the requirement asks this gate to cover.
_RAZOR_LOAD_AND_GROUND_DECK = (
    "CE razor load and ground\n"
    "GW 1 9 0. 0. 2.0 0. 0. 7.0 0.001\n"
    "GE -1\n"
    "GN 1\n"
    "LD 0 1 5 5 50. 1.e-6 0.\n"
    "EX 0 1 5 0 1.\n"
    "FR 0 1 0 0 14.1 0\n"
    "XQ\nNX\n"
)


@pytest.mark.parametrize("basis", ["razor", "razor-nec5"])
def test_razor_basis_answers_a_deck_with_a_load_and_a_ground(basis):
    """The portal gap #432 left open — no live solve under either razor
    roster name — is closed by `RazorSolver.compute_port_solution` (the
    sharing audit's #429 rank-9 item): both names answer a deck exercising
    a load AND a ground in one pass, with finite AIP data."""
    rc, out, err = _run_main(["--basis", basis], deck=_RAZOR_LOAD_AND_GROUND_DECK)
    assert rc == 0 and err == ""
    assert "ERROR-NEC2C" not in out, f"razor basis {basis} took the error path"
    assert "ANTENNA INPUT PARAMETERS" in out
    rows = _aip_rows(out)
    assert rows, f"no AIP data row for the load+ground deck under {basis}"
    assert all(math.isfinite(v) for row in rows for v in row)


# --- the EK card, honoured for real (issue #849) ------------------------------
#
# Until momwire 0.26.0 the portal parsed `EK`, threaded it through the execute
# groups, and printed its announcement — but momwire had one kernel, so the
# card moved no number and the grammar doc called it advisory. It is real now,
# and the tests below are the four halves of "real": the operator moves, it
# moves to MOMWIRE's own extended kernel and not to something this file invented
# on the way, it moves in nec2c's direction and by nec2c's amount, and the
# bases that cannot serve it say so.
#
# Every EK deck in the fixture corpus is a 0.001 m wire at Δ/a ≈ 500, where the
# extended kernel is a 0.02 % correction by construction — the card's whole
# subject is the O(a²) term the reduced kernel drops. So the decks here are
# FAT: 11 segments of a 5 m dipole at a = 0.2 m is Δ/a = 2.27, and both engines
# move by ~8 % across the card.

# λ = 10 m at 30 MHz; 11 segments over 5 m is Δ = 0.4545 m, so Δ/a = 2.27.
_FAT_WIRE = "GW 1 11 0. 0. -2.5 0. 0. 2.5 0.2\n"


def _fat_deck(kernel_card: str = "") -> str:
    card = f"{kernel_card}\n" if kernel_card else ""
    return (
        f"CE ek fat wire\n{_FAT_WIRE}GE 0\n{card}"
        "FR 0 1 0 0 30. 1\nEX 0 1 6 0 1.\nXQ\nNX\n"
    )


# nec2c 5b4az.ae6ty.1.23 (SimNEC 5.1's shipped `nec2c-ubuntu-x86` — the version
# the repo pins as `minimumNEC2CVersion`), on `_fat_deck()` and
# `_fat_deck("EK")` respectively. Captured 2026-08-10 the same way
# scripts/nec_portal_capture.py captures the corpus: the deck on stdin, the
# ANTENNA INPUT PARAMETERS row read off stdout.
NEC2C_FAT_REDUCED = complex(1.2368e02, 2.8585e01)
NEC2C_FAT_EXTENDED = complex(1.1357e02, 2.8980e01)


def _first_z(text: str) -> complex:
    """The impedance of the first ANTENNA INPUT PARAMETERS row."""
    columns = aip_impedances(text)
    assert columns, f"no AIP row:\n{text[-800:]}"
    return complex(float(columns[0][0]), float(columns[0][1]))


def _relative(a: complex, b: complex) -> float:
    return abs(a - b) / max(abs(a), abs(b))


@pytest.fixture
def restore_basis():
    """`--basis` is a process-global (`_active_basis`), set by `main` and never
    reset by `run_deck` — the ordering hazard `cache_reset` documents. A test
    that asks for an alternate basis puts it back, so it cannot decide what a
    later test's `run_deck`/`printout` call solves with."""
    from momwire.portal import _portal as nec_portal

    saved = (nec_portal._active_basis, nec_portal._active_basis_name)
    yield
    nec_portal._active_basis, nec_portal._active_basis_name = saved


def test_the_ek_card_moves_the_impedance_on_a_fat_wire(restore_basis):
    """The card is no longer advisory: same deck, one card, a different answer.

    Asserted on the DEFAULT basis as well as the NEC-closest one, because the
    passthrough is the engine's and every basis in the roster but Galerkin has
    to carry it.
    """
    for basis in ([], ["--basis", "sinusoidal"]):
        rc_off, off, err_off = _run_main(basis, deck=_fat_deck())
        rc_on, on, err_on = _run_main(basis, deck=_fat_deck("EK"))
        assert (rc_off, rc_on) == (0, 0) and not err_off and not err_on
        assert "ERROR-NEC2C" not in off and "ERROR-NEC2C" not in on
        shift = _relative(_first_z(off), _first_z(on))
        assert shift >= 0.02, f"{basis}: EK moved the impedance by only {shift:.3%}"


def test_the_ek_answer_is_momwires_own_extended_kernel_solve(restore_basis):
    """Which extended kernel — the identity behind the number.

    A deck through the portal under `--basis sinusoidal` must be the same solve
    as `SinusoidalSolver(extended_kernel=...)` built straight off the deck's own
    mesh: one straight wire, 11 uniform segments, the gap at the centre. The
    only slack allowed is the printout's own `%13.5E` rounding, so this catches
    a kernel flag that reached a DIFFERENT constructor, a mesh the portal built
    differently under EK, or an announcement printed over a reduced-kernel fill.
    """
    import numpy as np
    from momwire import SinusoidalSolver

    for card, kwargs in (("", {}), ("EK", {"extended_kernel": True})):
        solver = SinusoidalSolver(
            wires=[np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])],
            n_per_edge_per_wire=[[11]],
            feeds=[(0, 2.5, 1.0)],
            wavelength=299_792_458.0 / 30e6,
            wire_radius=0.2,
            **kwargs,
        )
        direct, _coeffs = solver.compute_impedance()
        _rc, out, _err = _run_main(["--basis", "sinusoidal"], deck=_fat_deck(card))
        assert _relative(_first_z(out), direct) <= 1e-4, (
            f"EK={card!r}: portal {_first_z(out)} vs direct {direct}"
        )


def test_the_fat_wire_ek_pair_tracks_nec2c_on_both_sides_of_the_card(restore_basis):
    """The card's physics, against the reference engine rather than ourselves.

    Both kernels are held at the differential harness's ordinary 5 % bar, and
    the SHIFT is held too — a passthrough that quietly did nothing would keep
    the reduced side inside 5 % and fail here, which is the mode this exists
    to catch. The sinusoidal basis is used because it is the roster's
    NEC-closest rung (issue #822); on this deck it lands 0.5 % from nec2c
    reduced and 3.1 % extended.
    """
    _rc, off, _e = _run_main(["--basis", "sinusoidal"], deck=_fat_deck())
    _rc, on, _e = _run_main(["--basis", "sinusoidal"], deck=_fat_deck("EK"))
    z_off, z_on = _first_z(off), _first_z(on)
    assert _relative(z_off, NEC2C_FAT_REDUCED) <= 0.05, f"{z_off} vs nec2c reduced"
    assert _relative(z_on, NEC2C_FAT_EXTENDED) <= 0.05, f"{z_on} vs nec2c extended"
    # nec2c's own move across the card is -10.1 Ω of resistance (8.0 %); ours
    # must be the same sign and the same order, not merely "different".
    theirs = NEC2C_FAT_EXTENDED - NEC2C_FAT_REDUCED
    ours = z_on - z_off
    assert ours.real < 0 and theirs.real < 0, f"{ours} vs {theirs}"
    assert 0.5 <= abs(ours) / abs(theirs) <= 2.0, f"{ours} vs {theirs}"


@pytest.mark.parametrize("card", ["EK", "EK 0", "EK 1", "EK 2", "EK -2"])
def test_every_ek_field_but_minus_one_is_the_extended_kernel(card):
    """What the oracle actually does with the card's one field.

    Measured on nec2c 5b4az.ae6ty.1.23: `EK`, `EK 0`, `EK 1`, `EK 2` and even
    `EK -2` all print THE EXTENDED THIN WIRE KERNEL WILL BE USED, and only
    `EK -1` does not — NEC-2's card reader sets the flag on an EK card and
    clears it for `-1` alone. The portal used to REFUSE anything outside
    {0, -1}, which is the #814 failure class exactly: a deck the reference
    engine runs, turned into a fabricated SimNEC readout by our refusal.
    """
    rc, out, err = _run_main([], deck=_fat_deck(card))
    assert rc == 0 and err == ""
    assert "ERROR-NEC2C" not in out, out[-600:]
    assert "THE EXTENDED THIN WIRE KERNEL WILL BE USED" in out
    assert _first_z(out) == _first_z(_run_main([], deck=_fat_deck("EK"))[1])


def test_ek_minus_one_is_the_reduced_kernel():
    """The other side of the same rule, and the no-card default with it."""
    z_bare = _first_z(_run_main([], deck=_fat_deck())[1])
    rc, out, _err = _run_main([], deck=_fat_deck("EK -1"))
    assert rc == 0 and "THE EXTENDED THIN WIRE KERNEL WILL BE USED" not in out
    assert _first_z(out) == z_bare


def test_two_groups_of_one_deck_under_two_kernels_get_two_operators():
    """`dipole_ek_rearm`'s shape, on a wire fat enough to show it.

    One deck, one frequency, two execute groups — the second under a kernel the
    first was not. The per-deck fill cache is keyed on the frequency ALONE
    unless the kernel is in the key too, so this is the intra-deck half of the
    staleness gate: getting it wrong prints the first group's answer twice and
    every other assertion in this file still passes.
    """
    deck = (
        f"CE ek rearm fat\n{_FAT_WIRE}GE 0\nEK -1\n"
        "FR 0 1 0 0 30. 1\nEX 0 1 6 0 1.\nXQ\nEK\nXQ\nNX\n"
    )
    rc, out, err = _run_main([], deck=deck)
    assert rc == 0 and err == "" and "ERROR-NEC2C" not in out
    rows = aip_impedances(out)
    assert len(rows) == 2, rows
    first, second = complex(*map(float, rows[0])), complex(*map(float, rows[1]))
    assert first == _first_z(_run_main([], deck=_fat_deck("EK -1"))[1])
    assert second == _first_z(_run_main([], deck=_fat_deck("EK"))[1])
    assert _relative(first, second) >= 0.02, f"{first} vs {second}"


@pytest.mark.parametrize(
    "basis", ["sinusoidal-galerkin", "sinusoidal-galerkin-converged"]
)
def test_a_galerkin_basis_serves_an_ek_deck(basis, restore_basis):
    """The un-refusal (momwire 0.27.0), on the class of deck that reaches it
    in real life.

    Until momwire 0.27.0 the Galerkin fill refused NEC's extended kernel and
    this test pinned the documented refusal frame. momwire#246 implemented the
    kernel on the Galerkin family, #287 lifted the last (Sommerfeld) ground
    refusal and #299 made non-collinear decks sound, so the live `NECSource`
    path — which sends `EK` on EVERY deck (grammar doc §17) — now gets an
    ANSWER from the Galerkin portal entries. The deck is the fat Δ/a ≈ 2.4
    dipole, where the kernel is a real correction: the EK answer must differ
    measurably from the reduced one (same 2 % floor the sinusoidal EK test
    uses), and both frames must keep their NX sentinels.
    """
    rc, out, err = _run_main(["--basis", basis], deck=_fat_deck("EK"))
    assert rc == 0 and err == ""
    assert "Traceback" not in out and "ERROR-NEC2C" not in out
    assert NX_ECHO.search(out), "no NX sentinel on the EK-served path"
    ek_rows = aip_impedances(out)
    assert ek_rows, out[-800:]
    # ... and the reduced answer on the same deck differs by a real margin,
    # so the card is being honoured rather than silently dropped.
    rc, plain, _err = _run_main(["--basis", basis], deck=_fat_deck())
    assert rc == 0 and "ERROR-NEC2C" not in plain
    z_ek = complex(*map(float, ek_rows[0]))
    z_plain = complex(*map(float, aip_impedances(plain)[0]))
    assert _relative(z_ek, z_plain) >= 0.02, f"{z_ek} vs {z_plain}"


def test_the_thin_wire_ek_fixtures_barely_move_and_move_towards_nec2c():
    """The armor's own measurement, kept next to the numbers that justify it.

    43 of the 45 corpus fixtures carry no `EK` and are byte-identical across
    this change. The two that do carry it are Δ/a ≈ 500 dipoles, where the
    extended kernel is the correction it is designed to be small: measured,
    both move 0.017 % — inside momwire's own no-op bar — and both move CLOSER
    to nec2c (0.761 % → 0.745 %). `dipole_ek_rearm`'s first group is `EK -1`
    and must not move at all, which is the assertion that the reduced path is
    still the reduced path.
    """

    def render(name):
        # Through `main` rather than `printout`, so the process-global basis is
        # this test's own whatever ran before it.
        return _run_main([], deck=fixture_deck(name) + "\nNX\n")[1]

    extended = _first_z(render("dipole_ek_extended"))
    oracle = complex(7.9240e01, 4.5364e01)  # the fixture's own captured row
    reduced = _first_z(render("dipole_free_space"))  # same wire, no EK card
    assert _relative(extended, reduced) <= 1e-3, (
        f"a Δ/a≈500 wire moved {_relative(extended, reduced):.4%} across EK"
    )
    assert _relative(extended, oracle) < _relative(reduced, oracle)

    rows = aip_impedances(render("dipole_ek_rearm"))
    assert len(rows) == 2
    first, second = complex(*map(float, rows[0])), complex(*map(float, rows[1]))
    assert first == reduced, "the EK -1 group is no longer the reduced-kernel one"
    assert second == extended


# --- the cross-deck solver cache (issue #823) --------------------------------
#
# The cache's whole claim is that the printout is IDENTICAL whether a deck was
# solved or served — so the printout cannot be the evidence that it WAS served.
# These tests read `nec_portal._cache_stats` for that half and hold the
# printout to the other half: that a served answer is the answer the arriving
# deck asked for, byte for byte against the same deck rendered from an empty
# cache. Nothing here parses a timing line.
#
# A stale factor would be silent and wrong, so the battery below is the point
# of the feature: one mutation per class of card that moves the operator, each
# asserting a MISS, and one per class that does not, each asserting a HIT.
#
# Serving is OFF in the shipped default and opted into with `--cache`, so a
# test OF the cache has to ask for it: `cache_reset` is the state every test
# here starts from, and the `main()`-driven ones pass the flag. The default-off
# and dry-run modes get their own tests further down.


def cache_reset(serving: bool = True) -> None:
    """Empty cache, default basis, no stats file, serving as asked.

    The basis pin is not decoration: ``main()`` resets ``_active_basis`` at the
    START of its next invocation, not on exit, so a ``--basis`` probe test
    leaves its basis behind for every direct ``deck_frame`` render that
    follows. A fresh-subprocess comparison after such a test would then diff
    two different physics — an ordering hazard, not a cache bug (it predates
    the cache; the subprocess-identity tests are just the first to be sensitive
    to it)."""
    nec_portal._active_basis = nec_portal._BASES["bspline"]
    nec_portal._cache_serving = serving
    nec_portal._cache_stats_path = None
    nec_portal._reset_solver_cache()


def cache_render(body: str) -> str:
    """One deck through the daemon's own per-deck path, cache as it stands."""
    return "\n".join(deck_frame(body)[0])


def cold_render(body: str) -> str:
    """The same deck from an EMPTY cache — what a fresh process prints."""
    cache_reset()
    return cache_render(body)


def cache_counts() -> dict:
    return dict(nec_portal._cache_stats)


def deck_chunks(transcript: str) -> list[str]:
    """A daemon transcript split after each ``NX`` echo — one chunk per deck."""
    chunks: list[str] = []
    current: list[str] = []
    for line in transcript.splitlines():
        current.append(line)
        if NX_ECHO.match(line):
            chunks.append("\n".join(current))
            current = []
    return chunks


def fill_lines(text: str) -> list[str]:
    """The MATRIX TIMING lines `body_lines` drops. A hit reuses the cached
    entry's measured fill, so between two passes of one process even these
    repeat to the digit — the token arity never moves either way."""
    return [line for line in text.splitlines() if "FILL:" in line]


def frame_lines(text: str) -> list[str]:
    """`body_lines` minus the ASCII banner box, so ONE deck's in-process frame
    and a whole process's transcript of the same deck compare."""
    return [
        line
        for line in body_lines(text)
        if "|" not in line and set(line.strip()) != {"_"}
    ]


def aip_impedances(text: str) -> list[tuple[str, str]]:
    """The impedance columns of every ANTENNA INPUT PARAMETERS row."""
    return [(row[6], row[7]) for table in aip_tables(text) for row in table]


@pytest.mark.parametrize("name", ANTENNA_NAMES)
def test_a_second_pass_of_every_fixture_prints_what_the_first_did(name):
    """The identity gate, over the whole corpus: every fixture sent TWICE down
    one process, compared under the harness's own canonicalisation.

    Fixtures that are multi-deck residency transcripts are compared deck for
    deck, first half against second. `misses` may not grow on the second pass —
    that is the assertion that the second half was SERVED rather than re-solved
    into an identical answer, which is what makes the identity meaningful.
    """
    text = (FIXTURE_DIR / f"{name}.deck").read_text()
    if not text.endswith("\n"):
        text += "\n"
    buffer = io.StringIO()
    rc = main(
        ["--cache"], stdin=io.StringIO(text * 2), stdout=buffer, stderr=io.StringIO()
    )
    assert rc == 0
    chunks = deck_chunks(buffer.getvalue())
    half = len(chunks) // 2
    assert half >= 1 and len(chunks) == 2 * half
    for first, second in zip(chunks[:half], chunks[half:]):
        assert body_lines(second) == body_lines(first)
        assert fill_lines(second) == fill_lines(first)
    assert nec_portal._cache_stats["hits"] >= half
    assert nec_portal._cache_stats["misses"] <= half


@pytest.mark.parametrize("name", ("dipole_free_space", "dipole_rp_pattern"))
def test_a_served_deck_matches_a_genuinely_fresh_process(name):
    """And the served printout is not merely self-consistent: it is what a
    process that never saw the deck before prints. `dipole_rp_pattern` carries
    the far-field path, where a stale solver would show up as a wrong pattern
    table rather than a wrong impedance."""
    text = fixture_deck(name) + "\nNX\n"
    proc = subprocess.run(
        [sys.executable, "-m", "momwire.portal"],
        input=text,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0 and proc.stderr == ""
    buffer = io.StringIO()
    # The subprocess is deliberately STOCK — no `--cache` — so this compares a
    # served answer against the shipped default's, not against itself.
    assert (
        main(
            ["--cache"],
            stdin=io.StringIO(text * 2),
            stdout=buffer,
            stderr=io.StringIO(),
        )
        == 0
    )
    chunks = deck_chunks(buffer.getvalue())
    assert len(chunks) == 2 and nec_portal._cache_stats["hits"] == 1
    assert frame_lines(chunks[1]) == frame_lines(proc.stdout)


# A deck carrying one of everything the key has to watch, so each mutation
# below is a single-token edit against the same base: a wire over a
# reflection-coefficient ground, scaled, loaded, driven off-centre.
CACHE_BASE = (
    "CM cross-deck cache probe\n"
    "CE\n"
    "GW 1 9 0. 0. 5.0 0. 0. 10.0 0.001\n"
    "GS 0 0 1.0\n"
    "GE -1\n"
    "GN 0 0 0 0 13. 0.005\n"
    "LD 0 1 3 3 50. 1.e-6 0.\n"
    "EX 0 1 5 0 1.\n"
    "FR 0 1 0 0 14.1 0\n"
    "XQ\n"
)


def mutate(old: str, new: str, base: str = CACHE_BASE) -> str:
    """A deck with one substring replaced — asserted unique, so a mutant can
    never silently edit a card it did not mean to."""
    assert base.count(old) == 1, old
    return base.replace(old, new)


# The same base carrying a SECOND load, so a load's VALUES can be mutated
# without moving the port set — the case the port component of the key cannot
# catch. (It carried a TL and an NT branch until #930 put both out of dialect;
# a load is the remaining card whose value lives outside the fill.)
CACHE_LOAD_BASE = mutate(
    "LD 0 1 3 3 50. 1.e-6 0.\n",
    "LD 0 1 3 3 50. 1.e-6 0.\nLD 4 1 7 7 25. -40.\n",
)

# A free-space deck at Δ/a = 2.27 — the regime where the extended kernel is
# worth several percent (issue #849). The plain CACHE_BASE wire is 500× too
# thin for an EK mutation to move a printed digit, so a cross-deck cache test
# on it can only ever assert the MISS; this one asserts the answer too.
CACHE_FAT_BASE = (
    "CM cross-deck cache probe, fat\n"
    "CE\n"
    "GW 1 11 0. 0. -2.5 0. 0. 2.5 0.2\n"
    "GE 0\n"
    "EX 0 1 6 0 1.\n"
    "FR 0 1 0 0 30. 0\n"
    "XQ\n"
)


# (label, base deck, mutant deck, does this move the printed numbers?)
_OPERATOR_MUTATIONS = (
    ("GW endpoint", CACHE_BASE, mutate("0. 0. 10.0 0.001", "0. 0. 10.4 0.001"), True),
    ("wire radius", CACHE_BASE, mutate("0.001", "0.0025"), True),
    ("segment count", CACHE_BASE, mutate("GW 1 9", "GW 1 11"), True),
    ("GS scale", CACHE_BASE, mutate("GS 0 0 1.0", "GS 0 0 1.1"), True),
    # The GE flag is the ground-plane ANNOTATION on this deck (the wire is well
    # clear of z = 0, and GN carries the physics), so it moves the printout
    # without moving a number. It is in the key because on a deck whose wire
    # touches the plane it moves both.
    ("GE flag", CACHE_BASE, mutate("GE -1", "GE 0"), False),
    ("GN parameter", CACHE_BASE, mutate("13. 0.005", "20. 0.005"), True),
    ("GN removed", CACHE_BASE, mutate("GN 0 0 0 0 13. 0.005\n", ""), True),
    ("LD value", CACHE_BASE, mutate("50. 1.e-6", "150. 1.e-6"), True),
    ("LD removed", CACHE_BASE, mutate("LD 0 1 3 3 50. 1.e-6 0.\n", ""), True),
    ("EX moved", CACHE_BASE, mutate("EX 0 1 5", "EX 0 1 4"), True),
    (
        "LD added",
        CACHE_BASE,
        mutate("EX 0 1 5 0 1.\n", "LD 4 1 7 7 25. -40.\nEX 0 1 5 0 1.\n"),
        True,
    ),
    # A load's VALUES across an unchanged pair of segments: the port set is
    # identical, so only the `loads` component of the key can catch these.
    (
        "LD 4 resistance",
        CACHE_LOAD_BASE,
        mutate("25. -40.", "60. -40.", CACHE_LOAD_BASE),
        True,
    ),
    (
        "LD 4 reactance",
        CACHE_LOAD_BASE,
        mutate("25. -40.", "25. 40.", CACHE_LOAD_BASE),
        True,
    ),
    (
        "LD 0 inductance",
        CACHE_LOAD_BASE,
        mutate("50. 1.e-6 0.", "50. 2.e-6 0.", CACHE_LOAD_BASE),
        True,
    ),
    # EK used to be the conservative key entry — kept for the principle that a
    # card whose meaning is "compute the operator differently" must never be
    # answered from an entry built without it, back when momwire had no other
    # kernel and the flag moved no number. Issue #849 made it an ordinary one.
    # This deck's wire is Δ/a ≈ 500, where the extended kernel is a 0.02 %
    # correction that does not survive `%13.5E`, so `moves_numbers` stays False
    # HERE and the fat-wire deck below carries the "and it answers with its own
    # physics" half.
    ("EK toggled", CACHE_BASE, mutate("EX 0 1 5 0 1.\n", "EK\nEX 0 1 5 0 1.\n"), False),
    (
        "EK toggled (fat wire)",
        CACHE_FAT_BASE,
        mutate("GE 0\n", "GE 0\nEK\n", CACHE_FAT_BASE),
        True,
    ),
)


@pytest.mark.parametrize(
    "base,mutant,moves_numbers",
    [(m[1], m[2], m[3]) for m in _OPERATOR_MUTATIONS],
    ids=[m[0] for m in _OPERATOR_MUTATIONS],
)
def test_an_operator_change_misses_the_cross_deck_cache(base, mutant, moves_numbers):
    """The care point. One mutation per class of card that moves the operator;
    each must be a MISS and each must answer with its own physics, checked
    against the same deck rendered from an empty cache."""
    baseline = cold_render(base)
    assert "ERROR-NEC2C" not in baseline, baseline
    before = cache_counts()
    served = cache_render(mutant)
    after = cache_counts()
    assert after["hits"] == before["hits"], "served a stale operator"
    assert after["misses"] == before["misses"] + 1
    assert "ERROR-NEC2C" not in served, served
    assert body_lines(served) == body_lines(cold_render(mutant))
    if moves_numbers:
        assert aip_impedances(served) != aip_impedances(baseline)


# Cards that change what is PRINTED or how the answer is read out, never the
# operator behind it. Each must be served from the base's entry.
_READOUT_MUTATIONS = (
    ("CM text", mutate("CM cross-deck cache probe", "CM something else entirely")),
    (
        "card formatting",
        mutate(
            "GW 1 9 0. 0. 5.0 0. 0. 10.0 0.001",
            "GW,1,9,0.,0.,5.,0.,0.,10.,.001",
        ),
    ),
    ("EX voltage", mutate("EX 0 1 5 0 1.", "EX 0 1 5 0 2.5")),
    ("RP grid", mutate("XQ\n", "RP 0 7 13 1001 0 0 30 30 1000\nXQ\n")),
    ("PT print control", mutate("EX 0 1 5 0 1.\n", "PT 0 1 3 5\nEX 0 1 5 0 1.\n")),
    ("PT card", mutate("XQ\n", "PT -1\nXQ\n")),
    ("MP card", mutate("XQ\n", "MP 16 32\nXQ\n")),
)


@pytest.mark.parametrize(
    "mutant",
    [m[1] for m in _READOUT_MUTATIONS],
    ids=[m[0] for m in _READOUT_MUTATIONS],
)
def test_a_readout_change_hits_the_cross_deck_cache_and_answers_fresh(mutant):
    """The other half of the care point: a deck that differs only in what it
    prints must be SERVED — no parse, no mesh, no fill — and must still print
    exactly what a cold cache prints for it."""
    cold_render(CACHE_BASE)
    before = cache_counts()
    served = cache_render(mutant)
    after = cache_counts()
    assert after["hits"] == before["hits"] + 1, "re-solved an operator it had"
    assert after["misses"] == before["misses"]
    assert after["fills"] == before["fills"], "a hit at one frequency must not fill"
    assert "ERROR-NEC2C" not in served, served
    assert body_lines(served) == body_lines(cold_render(mutant))


def test_a_new_frequency_reuses_the_geometry_and_pays_only_the_fill():
    """The issue's third bullet: a crew member handed the same structure at
    another frequency skips the parse and the mesh — a solver-level HIT — and
    pays exactly one new fill inside it."""
    mutant = mutate("14.1", "21.1")
    cold_render(CACHE_BASE)
    before = cache_counts()
    served = cache_render(mutant)
    after = cache_counts()
    assert after["hits"] == before["hits"] + 1
    assert after["misses"] == before["misses"]
    assert after["fills"] == before["fills"] + 1
    assert body_lines(served) == body_lines(cold_render(mutant))


# The GD proof. The second medium reaches NEC's far field only through RP's
# cliff modes, so it is deliberately OUT of the key and a GD knob-drag HITS.
# What makes that safe is that a hit rebinds `portal_deck` to the arriving
# deck, and the comparison here is against a FRESH PROCESS rather than against
# the served run itself — so this stays a proof if GD ever grows a far field.
# The ground is `GN 1` — the MININEC-type ground idiom, which is where the
# live GD knob-drags actually come from. #458 moved this probe to a `GN 2`
# while the pairing refused (a refused deck caches nothing) and momwire#487
# moved it back.
GD_BASE = (
    "CE gd probe\n"
    "GW 1 9 0. 0. 2.0 0. 0. 7.0 0.001\n"
    "GE -1\nGN 1\nGD 2,0,0,0,13.,.005,0.,0.\n"
    "EX 0 1 5 0 1.\nFR 0 1 0 0 14.1 0\nXQ\n"
)


def test_a_gd_card_change_hits_the_cache_and_still_answers_fresh():
    moved = GD_BASE.replace("13.,.005", "80.,.01")
    assert moved != GD_BASE
    cold_render(GD_BASE)
    before = cache_counts()
    served = cache_render(moved)
    assert cache_counts()["hits"] == before["hits"] + 1
    assert "ERROR-NEC2C" not in served, served
    proc = subprocess.run(
        [sys.executable, "-m", "momwire.portal"],
        input=moved + "NX\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0 and proc.stderr == ""
    assert frame_lines(served) == frame_lines(proc.stdout)


def test_a_hit_rebinds_the_arriving_deck_onto_the_cached_solver():
    """The invariant the GD exclusion rests on. Everything a cached instance
    derived from its original deck is in the key and therefore identical, but
    `portal_deck` is a live reference the printout reads through — so a hit
    hands it the deck actually being rendered, and no cached instance can carry
    a stale card that the key deliberately does not watch."""
    cold_render(GD_BASE)
    solver = next(iter(nec_portal._solver_cache.values()))
    assert (
        solver.portal_deck.second_medium == nec_portal.parse_deck(GD_BASE).second_medium
    )
    moved = GD_BASE.replace("13.,.005", "80.,.01")
    cache_render(moved)
    assert (
        solver.portal_deck.second_medium == nec_portal.parse_deck(moved).second_medium
    )


def test_a_second_medium_change_through_a_warm_cache_moves_the_cliff_pattern():
    """The combined pin for #823 × #842. The cliff modes made the second
    medium load-bearing in the far field, and it is read through
    ``solver.portal_deck.second_medium`` — exactly the attribute a cache hit
    rebinds. So a GD edit through a WARM cache must (a) hit, (b) move the
    RADIATION PATTERNS rows, and (c) match a fresh process byte for byte. A
    stale second medium would pass (a) and fail (b) or (c)."""
    base = fixture_deck("dipole_rp2_linear_cliff")
    moved = base.replace("GD 0 0 0 0 5. .001 10. -2.", "GD 0 0 0 0 80. .04 10. -2.")
    assert moved != base
    first = cold_render(base)
    before = cache_counts()
    served = cache_render(moved)
    assert cache_counts()["hits"] == before["hits"] + 1
    pattern_rows = lambda text: [  # noqa: E731 - two-line local helper
        ln for ln in text.splitlines() if len(ln.split()) == 12 and "." in ln
    ]
    assert pattern_rows(served) != pattern_rows(first), (
        "the warm-cache answer ignored the new second medium"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "momwire.portal"],
        # fixture_deck strips the framing NX AND the trailing newline.
        input=moved + "\nNX\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0 and proc.stderr == ""
    fresh = frame_lines(proc.stdout)
    assert fresh, "the fresh process rendered nothing — deck framing bug"
    assert frame_lines(served) == fresh


def test_a_cached_entry_is_re_sized_by_the_fills_it_grew():
    """An entry GROWS after it is stored — a sweep adds an `at()` fill per
    frequency — so the size taken at insertion drifts low exactly on the deck
    the cache exists to serve. The entry that just rendered is re-walked when
    the next deck arrives, which is what keeps the cap honest."""
    cache_reset()
    cache_render(CACHE_BASE)
    key = next(iter(nec_portal._solver_cache))
    at_insert = nec_portal._cache_sizes[key]
    for mhz in ("18.1", "21.1", "24.1", "28.1", "50.1"):
        cache_render(mutate("14.1", mhz))
    cache_render(_bound_deck(99.0))
    # Six frequencies through one geometry, plus the arrival that re-sized it.
    counts = cache_counts()
    assert (counts["hits"], counts["misses"], counts["fills"]) == (5, 2, 7)
    assert nec_portal._cache_sizes[key] > at_insert


def test_a_repeated_probe_skips_the_solve_it_paid_for():
    """The reason the feature exists. Measured at authoring time on
    `catalog_wire_w8jk`, the biggest committed deck (106 segments): 154 ms cold
    against 11 ms served, a factor of fourteen — what is left in the served
    pass is the readout algebra and the printout itself, which no cache can
    skip. The ASSERT is on the counters, not the clock: even a "deliberately
    loose" wall-clock ratio proved flaky under full-suite load (warm BLAS
    shrinks the cold fill, a GC pause inflates the served one), and the
    deterministic form of "costs less than the solve it skips" is that the
    second pass performs zero geometry parses and zero fills."""
    body = fixture_deck("catalog_wire_w8jk")
    cache_reset()
    cache_render(body)
    after_cold = dict(nec_portal._cache_stats)
    assert (after_cold["misses"], after_cold["fills"]) == (1, 1)
    cache_render(body)
    after_served = nec_portal._cache_stats
    assert after_served["hits"] == 1
    assert after_served["misses"] == after_cold["misses"], "second pass re-parsed"
    assert after_served["fills"] == after_cold["fills"], "second pass re-filled"


def _bound_deck(z: float) -> str:
    return (
        "CE bound probe\n"
        f"GW 1 9 0. 0. {z} 0. 0. {z + 5.0} 0.001\n"
        "GE 0\nEX 0 1 5 0 1.\nFR 0 1 0 0 14.1 0\nXQ\n"
    )


def test_the_cache_evicts_by_bytes_and_an_evicted_geometry_re_solves(monkeypatch):
    """The bound. The cap is patched to about two and a half entries rather
    than filling the shipped few hundred MB, because what needs proving is the
    eviction ORDER and that an evicted structure comes back correct — not the
    value of a constant."""
    cache_reset()
    assert "ERROR-NEC2C" not in cache_render(_bound_deck(10.0))
    first_key = next(iter(nec_portal._solver_cache))
    # The second arrival re-sizes the first entry now that its fill is done, so
    # this is a grown entry's size and not an empty one's.
    cache_render(_bound_deck(20.0))
    monkeypatch.setattr(
        nec_portal, "_CACHE_BYTES_CAP", int(nec_portal._cache_sizes[first_key] * 2.5)
    )
    for z in (30.0, 40.0, 50.0, 60.0):
        cache_render(_bound_deck(z))
    assert nec_portal._cache_stats["evictions"] >= 2
    assert nec_portal._cache_stats["bytes"] <= nec_portal._CACHE_BYTES_CAP
    assert len(nec_portal._solver_cache) <= 3
    assert first_key not in nec_portal._solver_cache
    assert first_key not in nec_portal._cache_sizes

    # Newest still resident.
    before = cache_counts()
    cache_render(_bound_deck(60.0))
    assert cache_counts()["hits"] == before["hits"] + 1

    # Oldest gone — and it re-solves to the same printout it gave when it was
    # resident, which is what "degrades to today's behaviour" has to mean.
    before = cache_counts()
    served = cache_render(_bound_deck(10.0))
    assert cache_counts()["misses"] == before["misses"] + 1
    assert body_lines(served) == body_lines(cold_render(_bound_deck(10.0)))


def test_the_cache_evicts_the_least_RECENTLY_used_not_the_oldest(monkeypatch):
    """A knob returned to a value probed long ago is the hit this feature is
    for, so a re-used entry has to be young again. Without the reorder on a
    hit this is a FIFO and the entry just served would be the next to go."""
    cache_reset()
    for z in (10.0, 20.0, 30.0):
        cache_render(_bound_deck(z))
    oldest, middle, _newest = list(nec_portal._solver_cache)
    monkeypatch.setattr(
        nec_portal, "_CACHE_BYTES_CAP", int(nec_portal._cache_sizes[oldest] * 2.5)
    )
    before = cache_counts()
    cache_render(_bound_deck(10.0))  # touched: the oldest becomes the newest
    assert cache_counts()["hits"] == before["hits"] + 1
    cache_render(_bound_deck(40.0))
    assert oldest in nec_portal._solver_cache
    assert middle not in nec_portal._solver_cache


@pytest.mark.parametrize(
    "refused",
    [
        # Refused while PARSING — never reaches the cache at all.
        CACHE_BASE.replace("EX 0 1 5 0 1.\n", "SP 0 0 0. 0. 0. 0. 0.\n"),
        # Refused while BUILDING the solver: tag 2 does not exist, and a deck
        # with no EX has no ports. Both raise out of `DeckSolver.__init__`,
        # after the key has been computed and before anything is stored.
        CACHE_BASE.replace("EX 0 1 5", "EX 0 2 5"),
        CACHE_BASE.replace("EX 0 1 5 0 1.\n", ""),
    ],
    ids=["parse refusal", "unknown tag", "no EX card"],
)
def test_a_refused_deck_neither_poisons_nor_consults_the_cache(refused):
    """#829's error path against #823's cache. A refusal must move nothing —
    no entry, no statistic — and the same structure sent valid afterwards must
    solve fresh and print what a cold cache prints for it."""
    cache_reset()
    before = cache_counts()
    out = cache_render(refused)
    assert "ERROR-NEC2C" in out, out
    assert cache_counts() == before
    assert not nec_portal._solver_cache
    assert not nec_portal._cache_sizes

    served = cache_render(CACHE_BASE)
    assert "ERROR-NEC2C" not in served, served
    assert cache_counts()["misses"] == before["misses"] + 1
    assert body_lines(served) == body_lines(cold_render(CACHE_BASE))


def test_a_refusal_after_a_hit_leaves_the_hit_entry_intact():
    """The other order: a good deck, a refused one, then the good deck again —
    the refusal must not have disturbed the entry standing behind it."""
    cold_render(CACHE_BASE)
    cache_render(CACHE_BASE.replace("EX 0 1 5", "EX 0 2 5"))
    before = cache_counts()
    served = cache_render(CACHE_BASE)
    assert cache_counts()["hits"] == before["hits"] + 1
    assert body_lines(served) == body_lines(cold_render(CACHE_BASE))


def test_the_cache_is_per_invocation_like_the_basis():
    """`main` empties the cache exactly where it re-reads `--basis`: engine
    state is per invocation, so a second call cannot be served from the
    first's — which is also what keeps entries built under one basis from
    occupying the cap under another."""
    deck = fixture_deck("dipole_free_space") + "\nNX\n"
    for _ in range(2):
        _rc, _out, _err = _run_main(["--cache"], deck=deck)
        counts = cache_counts()
        assert (counts["hits"], counts["misses"], counts["fills"]) == (0, 1, 1)
        assert len(nec_portal._solver_cache) == 1


def test_the_operator_key_carries_the_basis():
    """One process has one `--basis`, so this can never differ between two live
    decks — the key carries it anyway so it cannot be read wrong, and so a
    future in-process basis switch cannot serve the wrong physics."""
    deck = nec_portal.parse_deck(CACHE_BASE)
    default = nec_portal._operator_key(deck)
    original = nec_portal._active_basis
    try:
        nec_portal._active_basis = nec_portal._BASES["sinusoidal"]
        assert nec_portal._operator_key(deck) != default
    finally:
        nec_portal._active_basis = original
    assert nec_portal._operator_key(deck) == default


# --- the three cache modes: off, dry-run, serving -----------------------------
#
# Serving is opt-in and the shipped default is OFF, because the workload the
# cache exploits is a live SimNEC session's re-probe rate and nobody has
# measured one. So the default path has to be provably the pre-#823 path — not
# "the cache with nothing in it" — and there has to be a way to measure the
# session without serving anything, which is what `--cache-stats` alone is.
#
# The evidence here is a spy on the two things the off path may not do:
# construct fewer solvers than there are decks, and compute an operator key.


def _spy_cache_machinery(monkeypatch) -> tuple[list, list]:
    """(solver constructions, operator-key computations) for one test.

    A counter cannot prove the off path, because the off path does not count —
    that is the property under test. These do, from outside."""
    built: list[str] = []
    keyed: list[str] = []
    real_init = nec_portal.DeckSolver.__init__
    real_key = nec_portal._operator_key

    def init_spy(self, deck):
        built.append("build")
        real_init(self, deck)

    def key_spy(deck):
        keyed.append("key")
        return real_key(deck)

    monkeypatch.setattr(nec_portal.DeckSolver, "__init__", init_spy)
    monkeypatch.setattr(nec_portal, "_operator_key", key_spy)
    return built, keyed


def _twice(name: str = "dipole_free_space") -> str:
    """One fixture deck, framed and sent twice — the repeat probe in miniature."""
    text = fixture_deck(name) + "\nNX\n"
    return text * 2


def _stream_main(argv: list[str], stdin_text: str) -> list[str]:
    """`main` over a stdin stream, returning one chunk per deck."""
    buffer = io.StringIO()
    rc = main(argv, stdin=io.StringIO(stdin_text), stdout=buffer, stderr=io.StringIO())
    assert rc == 0
    return deck_chunks(buffer.getvalue())


def test_the_cache_is_off_unless_the_command_line_asks(monkeypatch):
    """The shipped default. Two identical decks, and the second is genuinely
    re-solved: two constructions, and NO key computed — the off branch is the
    pre-#823 path to the byte, not a cache that happens to be empty."""
    built, keyed = _spy_cache_machinery(monkeypatch)
    chunks = _stream_main([], _twice())
    assert len(chunks) == 2
    assert body_lines(chunks[1]) == body_lines(chunks[0])
    assert len(built) == 2, "the second deck was served rather than re-solved"
    assert keyed == [], "the off path computed a cache key"
    assert nec_portal._cache_mode() == "off"
    assert not nec_portal._solver_cache and not nec_portal._cache_sizes
    assert set(nec_portal._cache_stats.values()) == {0}


def test_the_cache_flag_turns_serving_on(monkeypatch):
    """And `--cache` is all it takes: one construction for two decks, the
    second answered from the first's factors, same printout."""
    built, keyed = _spy_cache_machinery(monkeypatch)
    chunks = _stream_main(["--cache"], _twice())
    assert len(chunks) == 2
    assert body_lines(chunks[1]) == body_lines(chunks[0])
    assert len(built) == 1 and len(keyed) == 2
    assert nec_portal._cache_mode() == "serving"
    counts = cache_counts()
    assert (counts["hits"], counts["misses"], counts["fills"]) == (1, 1, 1)


def test_cache_stats_alone_counts_the_hits_without_serving(monkeypatch, tmp_path):
    """The zero-risk live experiment. `--cache-stats` on its own solves every
    deck fresh — stock behaviour, stock answers, nothing retained — and records
    how many of them a cache WOULD have served. That is the number the
    default-off decision is waiting on, obtainable from a real session without
    putting a served answer anywhere near it."""
    path = tmp_path / "stats.json"
    built, keyed = _spy_cache_machinery(monkeypatch)
    chunks = _stream_main(["--cache-stats", str(path)], _twice())
    assert len(chunks) == 2
    assert body_lines(chunks[1]) == body_lines(chunks[0])
    assert nec_portal._cache_mode() == "dry-run"
    assert len(built) == 2, "a dry run served a deck"
    assert len(keyed) == 2, "a dry run has to key every deck to count it"
    assert not nec_portal._solver_cache and not nec_portal._cache_sizes

    stats = json.loads(path.read_text())
    assert stats["mode"] == "dry-run"
    assert (stats["hits"], stats["misses"]) == (1, 1)
    assert stats["decks_rendered"] == 2
    assert stats["fills"] == 2, "a dry run still pays every fill — that is the point"
    assert (stats["entries"], stats["bytes"], stats["evictions"]) == (0, 0, 0)


def test_a_dry_run_answers_exactly_what_a_stock_process_answers(tmp_path):
    """Counting may not perturb the physics: a dry-run deck is compared to the
    same deck through a process carrying no flags at all."""
    path = tmp_path / "stats.json"
    text = fixture_deck("dipole_rp_pattern") + "\nNX\n"
    proc = subprocess.run(
        [sys.executable, "-m", "momwire.portal"],
        input=text,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0 and proc.stderr == ""
    chunks = _stream_main(["--cache-stats", str(path)], text)
    assert len(chunks) == 1
    assert frame_lines(chunks[0]) == frame_lines(proc.stdout)


def test_both_flags_together_count_the_real_cache(tmp_path):
    """`--cache --cache-stats PATH` is serve AND measure — the mode to run once
    the dry run says the hit rate is worth having."""
    path = tmp_path / "stats.json"
    chunks = _stream_main(["--cache", "--cache-stats", str(path)], _twice())
    assert len(chunks) == 2
    stats = json.loads(path.read_text())
    assert stats["mode"] == "serving"
    assert (stats["hits"], stats["misses"], stats["fills"]) == (1, 1, 1)
    assert stats["entries"] == 1 and stats["bytes"] > 0


def test_the_stats_file_is_rewritten_after_every_deck(tmp_path):
    """SimNEC ends a session with `Process.destroy()` — a kill, not an EOF — so
    a file written at exit is a file that never appears. It is written at every
    deck boundary instead, which this reads BETWEEN the two decks by holding up
    the stdin stream."""
    path = tmp_path / "stats.json"
    text = fixture_deck("dipole_free_space") + "\nNX\n"
    midway: list[dict] = []

    def stream():
        yield from io.StringIO(text)
        # `main` asks for this line only after deck one has been framed.
        midway.append(json.loads(path.read_text()))
        yield from io.StringIO(text)

    buffer = io.StringIO()
    rc = main(
        [f"--cache-stats={path}"],
        stdin=stream(),
        stdout=buffer,
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert midway and midway[0]["decks_rendered"] == 1
    assert midway[0]["mode"] == "dry-run"
    assert json.loads(path.read_text())["decks_rendered"] == 2
    # And the transcript is untouched by any of it.
    assert len(deck_chunks(buffer.getvalue())) == 2


def test_a_refused_deck_still_counts_in_the_stats_denominator(tmp_path):
    """The hit rate needs an honest denominator, and a refused deck is a deck
    the session sent. It moves no cache statistic — that is the #829 contract —
    but it is counted as rendered."""
    path = tmp_path / "stats.json"
    good = fixture_deck("dipole_free_space") + "\nNX\n"
    bad = "CE refused\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\nSP 0 0\nNX\n"
    chunks = _stream_main(["--cache-stats", str(path)], good + bad)
    assert len(chunks) == 2 and "ERROR-NEC2C" in chunks[1]
    stats = json.loads(path.read_text())
    assert stats["decks_rendered"] == 2
    assert (stats["hits"], stats["misses"]) == (0, 1)


def test_the_cache_flags_ride_the_version_probe_unchanged():
    """SimNEC probes `<full command line> -version`, so every flag the portal
    dialog can carry has to leave that line alone."""
    for argv in (
        ["--cache", "-version"],
        ["--cache-stats", "/nonexistent/dir/stats.json", "-version"],
        ["--cache", "--cache-stats=/nonexistent/dir/stats.json", "-version"],
        ["--basis", "sinusoidal", "--cache", "-version"],
    ):
        rc, out, _err = _run_main(argv)
        assert (rc, out) == (0, f"{nec_portal.PROBE_VERSION}\n"), argv


def test_cache_stats_without_a_path_fails_fast_and_nonzero():
    """Same contract as an unknown `--basis`: a malformed portal-dialog line is
    caught by the configure-time probe, not by the first deck of a session."""
    for argv in (["--cache-stats"], ["--cache-stats", "--cache"], ["--cache-stats="]):
        rc, out, _err = _run_main([*argv, "-version"])
        assert rc == 3 and "--cache-stats" in out, argv


def test_an_unwritable_stats_path_costs_the_measurement_not_the_session(tmp_path):
    """This engine may write NOTHING to stdout or stderr, so a bad path can
    only be allowed to lose the file. The decks still answer."""
    path = tmp_path / "no-such-dir" / "stats.json"
    buffer = io.StringIO()
    errors = io.StringIO()
    rc = main(
        ["--cache-stats", str(path)],
        stdin=io.StringIO(_twice()),
        stdout=buffer,
        stderr=errors,
    )
    assert rc == 0 and errors.getvalue() == ""
    assert not path.exists()
    chunks = deck_chunks(buffer.getvalue())
    assert len(chunks) == 2 and "ERROR-NEC2C" not in buffer.getvalue()


# --------------------------------------------------------------------------
# #846: the portal depends on momwire alone, and momwire does not depend on it
# --------------------------------------------------------------------------

PORTAL_DIR = Path(nec_portal.__file__).resolve().parent
MOMWIRE_SRC = PORTAL_DIR.parent


def _import_lines(path: Path) -> list[tuple[int, str]]:
    """Every line of ``path`` that starts an import, numbered.

    Read at the SOURCE rather than at ``sys.modules``: an import that only
    fires on an error path, or inside a function, would hide from a runtime
    check of what got loaded. The grep is the honest instrument for a rule
    about what the code is ALLOWED to reach for.
    """
    return [
        (number, line.strip())
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.match(r"\s*(from|import)\s", line)
    ]


def test_the_portal_imports_nothing_from_antennaknobs():
    """Half one of the rule: the portal reaches DOWN, never sideways.

    The portal's whole point is that it is momwire's front door — it must
    install and run with momwire and nothing else, which is what let phase III
    move the file into ``momwire/portal/`` at all. Before the phase II rewire
    it reached for five antennaknobs internals: ``AntennaBuilder``,
    ``MomwireEngine``, ``parse_nec``, ``network._series_rlc_impedance`` and
    ``network_reduce``'s TL pair. There are none left, and there may be none
    again: momwire does not depend on antennaknobs, so an import added here
    would break every install that has only the engine.
    """
    offenders = [
        f"{path.name}:{number}: {line}"
        for path in sorted(PORTAL_DIR.glob("*.py"))
        for number, line in _import_lines(path)
        if re.search(r"\bantennaknobs\b", line)
    ]
    assert not offenders, (
        "momwire/portal must depend on momwire alone (#846); found:\n  "
        + "\n  ".join(offenders)
    )


def test_nothing_outside_the_portal_imports_from_the_portal():
    """Half two, and the one that keeps the engine's identity: the isolation
    rule of design doc #846 §4, enforced as a test rather than a convention.

    ``momwire/portal/`` may import momwire's public solver API and
    ``momwire.deck``. **Nothing outside ``momwire/portal/`` may import from
    ``momwire.portal``.** Colocation is what deleted portal-vs-solver version
    skew; this rule is what stops it costing momwire its identity. SimNEC
    protocol cruft — banner spoofing, column-exact NEC-2 tables, timing
    canonicalization, the ``out`` lore — now lives in the physics repo, and
    the only thing keeping it from becoming load-bearing for a solver is that
    no solver is permitted to look at it.

    Walked over the whole of ``src/momwire`` bar the portal itself, at the
    source, for the same reason as its sibling above: an import inside a
    function would satisfy any check of what got loaded at import time.

    Mutation-verified at authoring time with the hardest case: a
    ``from momwire.portal import PROBE_VERSION`` added INSIDE A FUNCTION in
    ``src/momwire/razor.py`` fails this test by name and file, while both its
    neighbours below stay green — which is exactly the reach a ``sys.modules``
    check cannot see.
    """
    offenders = [
        f"{path.relative_to(MOMWIRE_SRC)}:{number}: {line}"
        for path in sorted(MOMWIRE_SRC.rglob("*.py"))
        if PORTAL_DIR not in path.parents
        for number, line in _import_lines(path)
        if re.search(
            r"\bmomwire\.portal\b|\bfrom\s+\.portal\b|\bfrom\s+\.\s+import\s+.*\bportal\b",
            line,
        )
    ]
    assert not offenders, (
        "nothing outside momwire/portal may import from it (#846 §4); found:\n  "
        + "\n  ".join(offenders)
    )


def test_importing_momwire_does_not_drag_in_the_portal():
    """The rule's runtime shadow, and the reason a user feels it.

    ``import momwire`` is the engine. If the package eagerly imported its own
    front door, every solver script would pay for the protocol layer and the
    rule above would be true only on paper. Checked in a fresh interpreter
    because this test session has certainly imported the portal already.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import momwire, sys; print('momwire.portal' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", proc.stdout


# --------------------------------------------------------------------------
# the two readings of one deck must agree (#846 phase II)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_the_portal_and_the_dialect_agree_about_every_deck(name):
    """The proof obligation the double parse takes on.

    ``parse_deck`` reads a deck for the PRINTOUT (the card echo, the report
    cards, the ``MP``/``PT`` state) while ``momwire.deck.parse`` reads the same
    deck for the SOLVE. Two readers of one text can drift, so this walks the
    whole corpus and asserts they agree on everything they both have an
    opinion about:

    * which execute cards ran, in order (arming — the #933 fix's own claim);
    * each group's frequency list and extended-kernel flag;
    * each group's DRIVE, reconstructed from the portal's ``(tag, segment)``
      source list against the model's voltage vector over the union feed set.

    The last one is the load-bearing check: the portal keeps its own source
    list because a source explicitly driven at 0 V is a printed row while an
    undriven port is not, and a voltage vector cannot make that distinction.
    Everything else about the drive must still match.
    """
    from momwire.deck import DeckError
    from momwire.deck import parse as dialect_parse

    for body in _deck_bodies(name):
        try:
            deck = nec_portal.parse_deck(body)
        except DeckError:
            continue  # a refusal fixture: there is no second reading to agree with
        model = dialect_parse(body, dialect="nec2")

        assert len(deck.groups) == len(model.groups), f"{name}: group count"
        ports = nec_portal._union_ports(deck)
        assert ports == [
            (tag, seg) for tag, seg in ports
        ]  # discovery order, stated for the reader
        assert len(ports) == len(model.feeds), f"{name}: union port set size"

        for index, (ours, theirs) in enumerate(zip(deck.groups, model.groups)):
            assert (ours is None) == (theirs is None), f"{name}: group {index} armed"
            if ours is None:
                continue
            assert ours.freqs_mhz == theirs.frequencies, f"{name}: group {index} FR"
            assert ours.ek == theirs.extended_kernel, f"{name}: group {index} EK"
            assert ours.refilled == theirs.refilled, f"{name}: group {index} refill"
            drive = [0j] * len(ports)
            for tag, seg, volts in ours.sources:
                drive[ports.index((tag, seg))] = volts
            assert tuple(drive) == theirs.voltages, f"{name}: group {index} drive"


def _deck_bodies(name: str) -> list[str]:
    """A fixture's deck bodies, split at the ``NX`` cards that frame them."""
    text = (FIXTURE_DIR / f"{name}.deck").read_text()
    return [body for body in text.split("\nNX") if body.strip()]


# --------------------------------------------------------------------------
# #933: GN between execute cards re-arms
# --------------------------------------------------------------------------

# `XQ / GN 1 / XQ` over a wire well clear of the plane, so the ground is the
# only thing that changes between the two runs. Measured on the oracle
# (nec2c-ubuntu-x86, 2026-08-15) — fixture `dipole_gn_rearm`.
GN_REARM_DECK = (
    "CE gn between executes re-arms\n"
    "GW 1 9 0. 0. 2.0 0. 0. 7.0 0.001\n"
    "GE -1\n"
    "EX 0 1 5 0 1.\n"
    "FR 0 1 0 0 14.1 0\n"
    "XQ\n"
    "GN 1\n"
    "XQ\n"
)


def test_a_gn_between_execute_cards_runs_a_second_group():
    """#933. The old ``GN`` branch ``continue``d before the arming test, so a
    ground card between two execute cards armed nothing and the second ``XQ``
    printed no block at all. nec2c runs it."""
    text = run_deck(GN_REARM_DECK)[0]
    assert text.count("ANTENNA INPUT PARAMETERS") == 2, (
        "the second XQ did not run — GN failed to re-arm"
    )


def test_the_second_group_answers_over_the_ground_the_gn_card_named():
    """And it re-arms because the OPERATOR moved, not merely the printout: the
    environment block names the new ground and the impedance moves with it."""
    text = run_deck(GN_REARM_DECK)[0]
    first, second = text.split("ANTENNA ENVIRONMENT")[1:3]
    assert "FREE SPACE" in first.split("MATRIX TIMING")[0]
    assert "PERFECT GROUND" in second.split("MATRIX TIMING")[0]

    free, ground = (complex(row[4], row[5]) for row in _aip_rows(text))
    assert free != ground, "the ground did not reach the matrix"
    # The oracle's own numbers for this probe: 12.843 - j938.79 in free space,
    # 18.771 - j937.26 over perfect ground. A basis difference moves both, but
    # the DIRECTION and rough size of the shift is physics, not basis.
    assert ground.real > free.real
    assert (ground.real - free.real) / free.real > 0.1


def test_the_gn_rearm_block_is_a_partial_refill():
    """The shape the oracle prints for it: LOADING / ENVIRONMENT / MATRIX
    TIMING, and NO FREQUENCY block — the operator was rebuilt, but the
    frequency list was not (there is no new ``FR``). Identical to the
    ``dipole_ek_rearm`` shape, which is why the flag is ``_OPERATOR_CARDS``
    rather than a GN special case."""
    text = run_deck(GN_REARM_DECK)[0]
    second = text.split("ANTENNA INPUT PARAMETERS")[1]
    # Everything between the first block's table and the second's banner.
    between = second.split("--------- ANTENNA INPUT PARAMETERS")[0]
    assert "STRUCTURE IMPEDANCE LOADING" in between
    assert "ANTENNA ENVIRONMENT" in between
    assert "MATRIX TIMING" in between
    assert "FREQUENCY :" not in between, "a GN re-arm must not reprint FREQUENCY"


def test_each_group_carries_the_model_groups_own_environment():
    """The source of truth is ``momwire.deck``'s ``ExecuteGroup.environment``
    (momwire#370). The portal recovered the per-group ground by re-parsing the
    deck prefix that ended at each execute card until then — correct, and
    O(groups·deck), and a second reader of the ``GN``/``GD`` rules living
    beside the dialect's."""
    deck = nec_portal.parse_deck(GN_REARM_DECK)
    model = deck.model
    assert [g.environment for g in deck.groups] == [g.environment for g in model.groups]
    assert deck.groups[0].environment.ground is None
    assert deck.groups[1].environment.ground == "pec"
    # The printout's view derives from it rather than being stored beside it.
    assert deck.groups[0].ground.kind == "free"
    assert deck.groups[1].ground.kind == "pec"
    # And the DECK-level record is the deck's last environment, unchanged.
    assert deck.ground.kind == "pec"


def test_the_deck_solver_fills_each_group_over_its_own_environment():
    """The environment reaches the OPERATOR, not merely the printout: two
    groups of one deck at one frequency get two fills and two impedances."""
    deck = nec_portal.parse_deck(GN_REARM_DECK)
    solver = nec_portal.DeckSolver(deck)
    free, ground = (
        solver.solve_group(group, group.freqs_mhz[0]) for group in deck.groups
    )
    assert free["solver"] is not ground["solver"]
    assert free["ground_z"] is None
    assert ground["ground_z"] == 0.0
    z_free = free["v_gap"][0] / free["i_port"][0]
    z_ground = ground["v_gap"][0] / ground["i_port"][0]
    assert z_free != z_ground


def test_the_deck_solver_translates_the_geometry_once():
    """momwire#370: the mesh is the STRUCTURE's, not the operating point's,
    so every fill replays one prepared handle — the same coordinate arrays,
    not a fresh walk per frequency."""
    deck = nec_portal.parse_deck(GN_REARM_DECK)
    solver = nec_portal.DeckSolver(deck)
    first = solver.at(14.1)["solver"]
    second = solver.at(21.0)["solver"]
    assert first is not second
    for left, right in zip(first.wires_polylines, second.wires_polylines):
        assert left is right


def test_the_prefix_reparse_workaround_is_gone():
    """The grep the rewire is measured by: nothing in the portal stamps a
    ground onto the model any more, and no second parse recovers one."""
    source = Path(nec_portal.__file__).read_text()
    assert "replace(model" not in source
    assert "_environments_per_group" not in source
    assert "parse_dialect" in source  # the ONE parse, still there
    assert source.count("parse_dialect(") == 1


def test_the_gn_rearm_fixture_matches_the_oracle_section_walk():
    """The committed oracle capture of the same probe, section for section."""
    assert section_walk(printout("dipole_gn_rearm")) == section_walk(
        fixture_out("dipole_gn_rearm")
    )


def test_an_ex_between_execute_cards_re_arms_without_a_refill():
    """The control. Moving the DRIVE re-arms — the second run is real — but
    the matrix is untouched, so the oracle prints no preamble at all and
    neither does this engine."""
    deck = GN_REARM_DECK.replace("GN 1\n", "EX 0 1 3 0 1.\n")
    text = run_deck(deck)[0]
    assert text.count("ANTENNA INPUT PARAMETERS") == 2
    between = text.split("--------- ANTENNA INPUT PARAMETERS")[1]
    between = between.split("--------- ANTENNA INPUT PARAMETERS")[0]
    assert "STRUCTURE IMPEDANCE LOADING" not in between
    assert "MATRIX TIMING" not in between
