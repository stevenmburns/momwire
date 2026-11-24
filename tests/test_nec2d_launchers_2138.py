"""The NEC-2D launchers, ``momwire-nec2d[-<basis>]`` (momwire#2138).

Two contracts, held separately:

* the NAME: a program name carrying ``nec2d`` is a NEC-2 engine (the deck
  dialect, the refusals and the solve of ``momwire-nec2c``) that prints LLNL's
  Fortran NEC-2 layout, NEC-2D's, which is what 4nec2's ``nec2dxs`` engines
  write and the only layout 4nec2 reads.  The basis rides past ``nec2d-`` as it
  rides past ``nec2c-``; every launcher that ships the nec2c pair ships this
  one too.
* the LAYOUT: section by section and line by line NEC-2D's, every non-numeric
  token in its column and every number in its field, so 4nec2 finds each
  table by its heading and each row where it looks for it.  Held against the
  ``nec2dxs`` printouts of the portal corpus under ``fixtures/nec2d_oracle``
  when they are present, and always against the structural facts 4nec2 is
  known to read (the headings, the FREQUENCY and FILL lines, the impedance
  row's position, the closing EN echo and RUN TIME).

And one physics contract: ``momwire-nec2d-sinusoidal`` and
``momwire-nec2c-sinusoidal`` print one solve, so their input impedances agree
to every digit the nec2c printout carries.
"""

import io
import math
import re
import tomllib
from pathlib import Path

import pytest

from momwire.deck import NEC2_BASES
from momwire.deck._solver import (
    dialect_from_program_name,
    nec2_layout_from_program_name,
)
from momwire.dropin.nec2c import _nec2d, engine_scope, main
from momwire.dropin.nec2c import _portal as nec_portal

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PORTAL = FIXTURES / "nec_portal"
ORACLE = FIXTURES / "nec2d_oracle"

DIPOLE = (
    "CM a dipole 2138\n"
    "CE\n"
    "GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n"
    "GE 0\n"
    "LD 5 0 0 0 5.8E7\n"
    "EX 0 1 6 0 1.\n"
    "FR 0 1 0 0 14.0 0\n"
    "XQ\n"
)


def _printout(prog: str, text: str, tmp_path: Path) -> str:
    """The deck file form (``prog in out``), LF endings."""
    deck = tmp_path / "deck.nec"
    out = tmp_path / "deck.out"
    deck.write_bytes(text.encode("latin-1"))
    with engine_scope():
        assert main([str(deck), str(out)], stdin=io.StringIO(""), prog=prog) == 0
    return out.read_bytes().decode("latin-1").replace("\r\n", "\n")


def _stdin_names(prog: str, text: str, tmp_path: Path) -> str:
    """4nec2's form: ``ENGINE < nec2d.tmp``, the two file names on stdin."""
    deck = tmp_path / "in.nec"
    out = tmp_path / "in.out"
    deck.write_bytes(text.encode("latin-1"))
    names = io.StringIO(f"{deck}\n{out}\n")
    with engine_scope():
        assert main([], stdin=names, stdout=io.StringIO(), prog=prog) == 0
    return out.read_bytes().decode("latin-1").replace("\r\n", "\n")


# --------------------------------------------------------------------------
# the layout signature: what "layout-identical" means here
# --------------------------------------------------------------------------

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")
_TOKEN = re.compile(r"\S+")


def _kind(token: str) -> str:
    body = token.lstrip("+-").upper()
    if "E" in body:
        mantissa = body.split("E")[0]
        return "E%d" % len(mantissa.split(".")[1]) if "." in mantissa else "E"
    if "." in body:
        return "F%d" % len(body.split(".")[1])
    return "I"


def layout(line: str) -> tuple:
    """A line's layout: each text token with its first column, each number
    with its LAST column (Fortran fields are right-justified) and its format
    (``I``, ``F<d>`` or ``E<d>``).  Two printouts of one deck by two engines
    differ in their numbers' digits, never in this."""
    out = []
    pos = 0
    for m in _NUMBER.finditer(line):
        start, end = m.span()
        if start and line[start - 1] not in " =:(-+," and not line[start - 1].isdigit():
            continue
        if end < len(line) and line[end] not in " ,)*-+":
            continue
        out += [
            ("T", pos + t.start(), t.group()) for t in _TOKEN.finditer(line[pos:start])
        ]
        out.append(("N", end, _kind(m.group())))
        pos = end
    out += [("T", pos + t.start(), t.group()) for t in _TOKEN.finditer(line[pos:])]
    return tuple(out)


def layout_mismatches(ours: str, theirs: str) -> list[tuple[int, str, str]]:
    """``(line number, ours, theirs)`` for every line whose layout differs;
    line 2, momwire's engine stamp where NEC-2D prints a blank, excused."""
    a, b = ours.split("\n"), theirs.split("\n")
    bad = []
    for n in range(max(len(a), len(b))):
        la = a[n] if n < len(a) else "<end>"
        lb = b[n] if n < len(b) else "<end>"
        if la.startswith(" momwire ") and not lb.strip():
            continue
        if layout(la) != layout(lb):
            bad.append((n + 1, la, lb))
    return bad


def test_the_layout_signature_sees_text_and_field_changes():
    row = "     1    21 1.00000E+00 0.00000E+00 1.37472E-02 1.55136E-03"
    assert layout(row) == layout(row.replace("1.37472", "9.99999"))
    assert layout(row) == layout(row.replace(" 1.55136E-03", "-1.55136E-03"))
    assert layout(row) != layout(row.replace("1.37472E-02", "1.3747E-02 "))
    assert layout(row) != layout(" " + row[:-1])
    heading = "                                - - - MATRIX TIMING - - -"
    assert layout(heading) != layout(heading.replace("MATRIX", "MATRIC"))


# --------------------------------------------------------------------------
# the name
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prog", "layout_name"),
    [
        ("momwire-nec2d", "nec2d"),
        ("momwire-nec2d-sinusoidal", "nec2d"),
        (r"C:\momwire\Momwire-NEC2D-SG.EXE", "nec2d"),
        ("/usr/bin/momwire-nec2d-bs1", "nec2d"),
        ("momwire-nec2c", "nec2c"),
        ("momwire-nec2c-sinusoidal.exe", "nec2c"),
        # nec2c wins a name carrying both, as it wins every other marker
        ("momwire-nec2c-nec2d", "nec2c"),
        ("python", "nec2c"),
    ],
)
def test_the_name_selects_the_layout(prog, layout_name):
    assert nec2_layout_from_program_name(prog) == layout_name


def test_a_nec2d_name_is_a_nec2_engine_everywhere_a_name_is_read():
    for prog in ("momwire-nec2d", "momwire-nec2d-sinusoidal.exe"):
        assert dialect_from_program_name(prog) == "nec2"
    assert nec_portal._filename_basis("momwire-nec2d-sinusoidal") == "sinusoidal"
    assert nec_portal._filename_basis("momwire-nec2d") is None
    assert nec_portal._filename_basis("Momwire-NEC2D-SG.exe") == "sg"


def test_the_version_line_names_the_nec2d_launcher():
    out = io.StringIO()
    with engine_scope():
        assert main(["-version"], stdout=out, prog="momwire-nec2d-sinusoidal") == 0
    line = out.getvalue().strip()
    assert line.startswith("NEC2 momwire-nec2d-sinusoidal [sb] "), line
    out = io.StringIO()
    with engine_scope():
        assert main(["-version"], stdout=out, prog="momwire-nec2c") == 0
    assert out.getvalue().startswith("NEC2 momwire-nec2c [sb] ")


def test_an_unknown_nec2d_basis_fails_at_the_probe():
    out = io.StringIO()
    with engine_scope():
        assert main(["-version"], stdout=out, prog="momwire-nec2d-nonsense") == 3
    assert "nonsense" in out.getvalue()


def test_pip_ships_the_nec2c_roster_under_nec2d():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    nec2d = {n: t for n, t in scripts.items() if n.startswith("momwire-nec2d")}
    assert nec2d.pop("momwire-nec2d") == scripts["momwire-nec2c"]
    for name, target in nec2d.items():
        assert target == scripts["momwire-nec2c"], name
    assert {n.removeprefix("momwire-nec2d-") for n in nec2d} == set(NEC2_BASES)


def test_the_drop_in_ships_the_nec2d_pair_through_the_nec2_route():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eznec_freeze_build_2138", ROOT / "scripts" / "eznec_freeze" / "build.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    assert tuple(build.NEC2D_LAUNCHERS) == ("momwire-nec2d", "momwire-nec2d-sinusoidal")
    assert set(build.NEC2D_LAUNCHERS) <= set(build.launcher_stems())
    # The native launcher forwards a nec2d name to the engine's NEC-2 route,
    # whose layout the engine then reads off the same name.
    source = (
        ROOT / "scripts" / "eznec_client_c" / "momwire_eznec_client.c"
    ).read_text()
    assert '#define NEC2D_MARKER "nec2d"' in source
    assert "strstr(name, NEC2D_MARKER)" in source
    readme = build.launcher_table(".exe")
    assert "momwire-nec2d-sinusoidal.exe" in readme


# --------------------------------------------------------------------------
# the layout, structurally
# --------------------------------------------------------------------------


def test_the_printout_walks_nec2d_sections_in_order(tmp_path):
    text = _printout("momwire-nec2d", DIPOLE, tmp_path)
    lines = text.split("\n")
    assert lines[0] == "1"
    assert lines[1].startswith(" momwire ")
    assert lines[6] == " " * 36 + "NUMERICAL ELECTROMAGNETICS CODE (NEC-2D)"
    headings = [ln.strip() for ln in lines if re.fullmatch(r" +- - .* - -", ln)]
    assert headings == [
        "- - - - COMMENTS - - - -",
        "- - - STRUCTURE SPECIFICATION - - -",
        "- - - - SEGMENTATION DATA - - - -",
        "- - - - - - FREQUENCY - - - - - -",
        "- - - STRUCTURE IMPEDANCE LOADING - - -",
        "- - - ANTENNA ENVIRONMENT - - -",
        "- - - MATRIX TIMING - - -",
        "- - - ANTENNA INPUT PARAMETERS - - -",
        "- - - CURRENTS AND LOCATION - - -",
        "- - - POWER BUDGET - - -",
    ]
    assert re.search(r"\n {36}FREQUENCY= 1\.4000E\+01 MHZ\n {36}WAVELENGTH= ", text)
    assert re.search(
        r"\n {24}FILL= +\d+\.\d{3} SEC\.,  FACTOR= +\d+\.\d{3} SEC\.\n", text
    )
    # The impedance row: two blank-free header lines below the heading's blank.
    at = lines.index(" " * 42 + "- - - ANTENNA INPUT PARAMETERS - - -")
    assert lines[at + 1] == ""
    assert lines[at + 2].startswith("   TAG   SEG.    VOLTAGE (VOLTS)")
    assert re.fullmatch(r" +1 +6( ?-?\d\.\d{5}E[-+]\d\d){9}", lines[at + 4])
    # The run closes with the EN echo, a blank and RUN TIME as F10.3.
    assert lines[-4].startswith(" ***** DATA CARD NO.  5   EN   0")
    assert lines[-3] == ""
    assert re.fullmatch(r" RUN TIME = {0,5}\d+\.\d{3}", lines[-2])
    assert lines[-1] == ""


def test_the_card_echo_is_nec2d_s(tmp_path):
    text = _printout("momwire-nec2d", DIPOLE, tmp_path)
    assert (
        " ***** DATA CARD NO.  1   LD   5     0     0     0  5.80000E+07"
        "  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00\n"
    ) in text
    # an integer wider than its I3 field prints as asterisks
    assert _nec2d._echo(1, "NT", [9901, 1, 4, 1], [0.0] * 6).startswith(
        " ***** DATA CARD NO.  1   NT ***     1     4     1"
    )


def test_the_end_of_a_deck_without_en_echoes_the_last_cards_fields(tmp_path):
    """4nec2 writes no EN: NEC-2D's reader then closes the deck itself, and
    its EN echo carries the last card read."""
    deck = DIPOLE.replace("XQ\n", "RP 0 3 2 1000 0 0 45 90\n")
    for text in (
        _printout("momwire-nec2d", deck, tmp_path),
        _stdin_names("momwire-nec2d", deck, tmp_path),
    ):
        lines = text.split("\n")
        assert lines[-4].startswith(
            " ***** DATA CARD NO.  5   EN   0     3     2  1000  0.00000E+00"
        ), lines[-4]
        assert lines[-2].startswith(" RUN TIME =")


def test_the_conductivity_and_a_trap_on_it_draw_the_loaded_twice_note(tmp_path):
    deck = DIPOLE.replace("EX 0", "LD 1 1 3 3 1.0E5 3.3E-6 1.7E-11\nEX 0")
    text = _printout("momwire-nec2d", deck, tmp_path)
    assert "\n     ALL" in text
    assert (
        "\n\n          NOTE, SOME OF THE ABOVE SEGMENTS HAVE BEEN LOADED TWICE - "
        "IMPEDANCES ADDED\n"
    ) in text


def test_a_structure_chained_by_nx_starts_a_new_page(tmp_path):
    text = _printout("momwire-nec2d", DIPOLE + "NX\n" + DIPOLE + "EN\n", tmp_path)
    lines = text.split("\n")
    nx = next(i for i, ln in enumerate(lines) if "DATA CARD NO.  5   NX" in ln)
    assert lines[nx + 1] == "1"
    assert text.count("NUMERICAL ELECTROMAGNETICS CODE (NEC-2D)") == 2
    assert text.count(" RUN TIME =") == 1


def test_a_refused_deck_says_why_in_the_printout(tmp_path):
    text = _printout(
        "momwire-nec2d", DIPOLE.replace("GE 0", "SP 0 0 1 1 1 0 0 1\nGE 0"), tmp_path
    )
    assert " ***** NEC ERROR - " in text
    assert text.rstrip().split("\n")[-1].startswith(" RUN TIME =")


def test_a_pattern_prints_rdpat_s_table(tmp_path):
    deck = DIPOLE.replace("XQ\n", "RP 0 3 2 1503 0 0 45 90 0 0\n")
    text = _printout("momwire-nec2d", deck, tmp_path)
    assert " " * 48 + "- - - RADIATION PATTERNS - - -" in text
    assert "  THETA     PHI        VERT.   HOR.    TOTAL      AXIAL" in text
    assert "   AVERAGE POWER GAIN=" in text
    assert " " * 37 + "- - - - NORMALIZED GAIN - - - -" in text
    assert " " * 37 + "      TOTAL GAIN" in text
    row = next(ln for ln in text.split("\n") if ln.startswith("    0.00     0.00 "))
    assert re.fullmatch(
        r" +-?\d+\.\d\d +-?\d+\.\d\d {3}( *-?\d+\.\d\d){3} +\d\.\d{5} +-?\d+\.\d\d"
        r"  (LINEAR|RIGHT |LEFT  |      )( +\d\.\d{5}E[-+]\d\d +-?\d+\.\d\d){2}",
        row,
    ), row


# --------------------------------------------------------------------------
# the layout against nec2dxs, and the physics against nec2c
# --------------------------------------------------------------------------

_ORACLE_DECKS = sorted(p.stem for p in ORACLE.glob("*.out")) if ORACLE.is_dir() else []


@pytest.mark.skipif(
    not _ORACLE_DECKS,
    reason="the nec2dxs oracle printouts (tests/fixtures/nec2d_oracle) are not committed",
)
@pytest.mark.parametrize("name", _ORACLE_DECKS)
def test_the_layout_is_nec2dxs_s(name, tmp_path):
    deck = (PORTAL / f"{name}.deck").read_text(encoding="latin-1")
    ours = _printout("momwire-nec2d-sinusoidal", deck, tmp_path)
    theirs = (ORACLE / f"{name}.out").read_bytes().decode("latin-1")
    theirs = theirs.replace("\r\n", "\n")
    bad = layout_mismatches(ours, theirs)
    assert not bad, "\n".join(
        f"{n}:\n  ours  |{a}\n  theirs|{b}" for n, a, b in bad[:8]
    )


_AIP = re.compile(r" *(\d+) +(\d+)((?: *-?\d\.\d+E[-+]\d\d){9})$")


def _impedances(text: str) -> list[tuple[str, str, float, float]]:
    """Every port row's (tag, seg, R, X): the input parameters and the
    network connection points, in both layouts."""
    out = []
    for line in text.split("\n"):
        m = _AIP.match(line)
        if m:
            cells = re.findall(r"-?\d\.\d+E[-+]\d\d", m.group(3))
            out.append((m.group(1), m.group(2), float(cells[4]), float(cells[5])))
    return out


@pytest.mark.parametrize(
    "name",
    [
        "dipole_free_space",
        "dipole_load_ld0",
        "dipole_nt_network",
        "dipole_tl_network",
        "dipole_sommerfeld_ground",
        "two_source_sensor_lines",
        "catalog_multiband_trap_dipole",
    ],
)
def test_nec2d_and_nec2c_print_one_solve(name, tmp_path):
    deck = (PORTAL / f"{name}.deck").read_text(encoding="latin-1")
    ours = _impedances(_printout("momwire-nec2d-sinusoidal", deck, tmp_path))
    theirs = _impedances(_printout("momwire-nec2c-sinusoidal", deck, tmp_path))
    assert ours and len(ours) == len(theirs)
    for (t1, s1, r1, x1), (t2, s2, r2, x2) in zip(ours, theirs, strict=True):
        assert (t1, s1) == (t2, s2)
        # nec2c prints four decimals and NEC-2D five, of one number: the
        # five-digit value stands within half a four-digit unit of the
        # four-digit one, give or take its own rounding (so a five-digit value
        # ending in 5 may round either way).
        for five, four in ((r1, r2), (x1, x2)):
            unit = 10.0 ** (math.floor(math.log10(abs(four))) - 4) if four else 1e-30
            assert abs(five - four) <= 0.55 * unit, (five, four)
