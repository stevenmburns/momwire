"""momwire#1110: EZNEC's load probes beside a CURRENT drive - the mixed drive.

momwire#1099 served EZNEC's ``EX 0,tag,node,0,1.E-10,0.`` load probe beside
another ``EX 0``.  Every EZNEC model driven by a current source - which is what
``EX 4`` is, and what EZNEC writes for a normalized-power drive - writes the
same probe beside an ``EX 4`` instead, and this seam refused the pair outright.
So a terminated folded dipole, or any loaded model fed that way, could not run.

The captures gated here are EZNEC Pro/2+ sessions with our licensed NEC-5's
printout for each (tests/fixtures/eznec_load_probes_1110/, README there): 0192
is a terminated folded dipole whose drive reaches the structure through an
``NT``, 0193-0196 are one 11-segment dipole carrying each of the four shapes
EZNEC's RLC load window can write, and 0183 is that same dipole with no load
and no probe - the LOADLESS CONTROL, which is where the tolerance below comes
from.  M_0120 is capture 0120 with a probe hand-placed at a ``TL`` end, and it
is here to stay REFUSED.

Gates:

1. all five loaded decks are SERVED, one ANTENNA INPUT PARAMETERS row per EX
   card in deck order, TAG / SEG. / end-digit byte-equal to the oracle;
2. the boundary conditions are restored byte-equal - the ``EX 4``'s set current
   and the probe's written 1e-10 V (the U1 rule, once per row);
3. the drive row is byte-identical to the same deck rendered with its probe
   lines removed: the probe moves NOTHING the printout can show, which is the
   whole reason this shape is servable rather than a mixed-source solve;
4. the drive row's Z and the probe row's current agree with NEC-5's to the
   basis tolerance stated below, and the probe row's own arithmetic closes;
5. the loading table is unmoved by the probe;
6. the two refusals that stay, by name: a probe at a network connection point,
   and a genuine mixed drive.

**The tolerance.**  bspline and NEC-5 solve these decks to different digits by
design, and on EZNEC's 11-segment mesh NEC-5 is itself under-converged, so the
disagreement is the MESH's and not the load's.  0183 says how big it is with no
load on the deck at all: 85.101 + 45.802j ohms against NEC-5's 79.948 +
29.919j, which is 19.56 % of |Z|.  So the gate on that dipole family is that a
load must not make the disagreement WORSE than the bare wire's - 20 %, with
0183 itself held to it, and the four loaded decks measuring 17.30 / 15.98 /
14.10 / 14.10 % inside it.  0192's 69-segment folded dipole is a properly
meshed deck and needs no such allowance: 0.24 % on the drive row and 0.23 % on
the probe row, held at 3 %.  The probe CURRENTS on the dipole family land far
inside the drive row's allowance (2.17 / 3.35 / 0.94 / 0.94 %) and are held at
5 % rather than 20 %, because a tolerance that is not measured is not a gate.

A tighter number on 0193-0196 would be gating bspline against an
under-converged oracle, which is what the seam rule forbids; what those four
fixtures are really for is everything else in the list, all of which is exact.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_load_probes_1110"

# The dipole family: EZNEC's 11 segments, where NEC-5 is under-converged and
# the loadless control measures the gap (module docstring).
DIPOLE_TOLERANCE = 0.20
DIPOLE_CURRENT_TOLERANCE = 0.05
# 0192's 69-segment folded dipole, where the two engines agree to a quarter of
# a percent and nothing has to be allowed for.
FOLDED_TOLERANCE = 0.03

# 0192 is a 138-segment folded dipole over a Sommerfeld ground printing a
# 181-point pattern - a ~30 second render, which is the integration lane's
# business rather than the default lane's.  The dipole family is seconds and
# stays unmarked, where a regression is seen soonest.
SERVED = (
    pytest.param("0192", marks=pytest.mark.integration),
    "0193",
    "0194",
    "0195",
    "0196",
    # Capture 0202 (2026-09-18): the deck Dan AC6LA's "first blood" post (QRZ
    # 1003328 #100) describes, written by EZNEC itself - Cardioid.ez with an
    # 18 ohm load at segment 2 of each 6-segment vertical, both sources type I,
    # so two EX 4 and two probes. It replaces the hand reconstruction that
    # stood here for a day (README); NEC-5's printout for it is the oracle.
    "0202",
)
# EZNEC writes exactly this card per lumped load, and the CM line above it says
# so in words.  Stripping them is how the "the probe moves nothing" gate gets
# its other render.
PROBE_LINE = re.compile(r"^EX 0,.*1\.E-10,0\.\s*$")


def deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


def oracle(name: str) -> list[str]:
    return (FIXTURES / f"{name}.out").read_text().splitlines()


def tolerance(name: str) -> float:
    # 0202, the cardioid, is EZNEC's six-segment sample mesh, the same
    # under-converged class as the eleven-segment dipoles: NEC-5 and bspline
    # sit 12-15 % apart on its drive rows and 1 % on its probe rows.
    return FOLDED_TOLERANCE if name == "0192" else DIPOLE_TOLERANCE


def without_probes(text: str) -> str:
    stripped = [line for line in text.splitlines() if not PROBE_LINE.match(line)]
    assert len(stripped) < len(text.splitlines()), "this deck carries no probe"
    return "".join(line + "\n" for line in stripped)


def _rows(lines: list[str]) -> list[str]:
    start = (
        next(i for i, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line)
        + 4
    )
    out = []
    for line in lines[start:]:
        if not line.strip():
            break
        out.append(line)
    return out


def _cells(row: str) -> list[float]:
    return [float(x) for x in row[12:].split()]


def _loading(lines: list[str]) -> list[str]:
    """The STRUCTURE IMPEDANCE LOADING block, header row and all.

    It ends at the allocation line the shell writes next, which is the one
    marker every one of these printouts has between the table and the timings.
    """
    start = next(
        i for i, line in enumerate(lines) if "STRUCTURE IMPEDANCE LOADING" in line
    )
    out = []
    for line in lines[start:]:
        if line.startswith("ALLOCATE"):
            break
        out.append(line)
    return out


def _n_drives(name: str) -> int:
    """How many EX 4 cards the deck writes: its ANTENNA INPUT PARAMETERS
    rows are those drives first, in deck order, then one row per probe. One
    on every deck here but the cardioid, which drives two verticals."""
    return deck(name).count("\nEX 4,")


def _split(rows: list[str], name: str) -> tuple[list[str], list[str]]:
    n = _n_drives(name)
    return rows[:n], rows[n:]


def _refusal(text: str) -> str:
    printout = render(text)
    assert " ***** NEC ERROR - " in printout, "this deck was SERVED"
    return printout.rsplit(" ***** NEC ERROR - ", 1)[1].strip()


@pytest.mark.parametrize("name", SERVED)
def test_each_probed_deck_is_served_one_row_per_card_in_deck_order(name):
    rendered = render(deck(name))
    assert "NEC ERROR" not in rendered
    rows = _rows(rendered.splitlines())
    assert len(rows) == deck(name).count("\nEX ")
    # TAG, SEG. and the trailing end digit, byte for byte
    assert [row[:12] for row in rows] == [row[:12] for row in _rows(oracle(name))]


@pytest.mark.parametrize("name", SERVED)
def test_both_boundary_conditions_are_restored_byte_equal(name):
    drives, probes = _split(_rows(render(deck(name)).splitlines()), name)
    their_drives, their_probes = _split(_rows(oracle(name)), name)
    assert drives and probes
    # each EX 4's set current: the CURRENT columns of its drive row, the
    # deck's own amplitude and phase (1.4142 at 0 deg on the single-drive
    # decks; 1.4142 at 0 and at -90 deg on the cardioid)
    for ours, theirs in zip(drives, their_drives, strict=True):
        assert ours[36:60] == theirs[36:60]
        assert "1.4142E+00" in ours[36:60]
    # each probe's written 1e-10 V: the VOLTAGE columns of its row
    for ours, theirs in zip(probes, their_probes, strict=True):
        assert ours[12:36] == theirs[12:36] == "  1.0000E-10  0.0000E+00"


@pytest.mark.parametrize("name", SERVED)
def test_the_drive_row_is_byte_identical_to_the_probe_stripped_deck(name):
    with_probe, _ = _split(_rows(render(deck(name)).splitlines()), name)
    without = _rows(render(without_probes(deck(name))).splitlines())
    assert len(without) == _n_drives(name)
    assert with_probe == without


@pytest.mark.parametrize("name", SERVED)
def test_the_drive_row_impedance_agrees_with_the_oracle(name):
    drives, _ = _split(_rows(render(deck(name)).splitlines()), name)
    their_drives, _ = _split(_rows(oracle(name)), name)
    for row, their_row in zip(drives, their_drives, strict=True):
        ours, theirs = _cells(row), _cells(their_row)
        z_ours, z_theirs = complex(ours[4], ours[5]), complex(theirs[4], theirs[5])
        assert abs(z_ours - z_theirs) <= tolerance(name) * abs(z_theirs), (
            z_ours,
            z_theirs,
        )


def test_the_loadless_control_is_what_the_dipole_tolerance_is_measured_from():
    """0183 has no load, no probe and one EX 4, and it is this far off.

    The four loaded decks are gated at the same number, so what their gate
    actually says is that the LOAD does not make the mesh disagreement worse -
    which is the only claim an under-converged oracle supports.
    """
    ours = _cells(_rows(render(deck("0183")).splitlines())[0])
    theirs = _cells(_rows(oracle("0183"))[0])
    z_ours, z_theirs = complex(ours[4], ours[5]), complex(theirs[4], theirs[5])
    gap = abs(z_ours - z_theirs) / abs(z_theirs)
    assert 0.15 <= gap <= DIPOLE_TOLERANCE, gap


@pytest.mark.parametrize("name", SERVED)
def test_the_probe_row_carries_the_load_current_and_its_own_arithmetic(name):
    _, probes = _split(_rows(render(deck(name)).splitlines()), name)
    _, their_probes = _split(_rows(oracle(name)), name)
    limit = FOLDED_TOLERANCE if name == "0192" else DIPOLE_CURRENT_TOLERANCE
    for row, their_row in zip(probes, their_probes, strict=True):
        ours, theirs = _cells(row), _cells(their_row)
        i_ours, i_theirs = complex(ours[2], ours[3]), complex(theirs[2], theirs[3])
        assert abs(i_ours - i_theirs) <= limit * abs(i_theirs), (i_ours, i_theirs)
        # Z = V/I, Y = I/V, P = 0.5 Re(V I*), all of the row's own numbers
        v = complex(ours[0], ours[1])
        z, y, p = complex(ours[4], ours[5]), complex(ours[6], ours[7]), ours[8]
        assert abs(z - v / i_ours) <= 1e-3 * abs(z)
        assert abs(y - i_ours / v) <= 1e-3 * abs(y)
        assert abs(p - 0.5 * (v * i_ours.conjugate()).real) <= 1e-3 * abs(p)


@pytest.mark.parametrize("name", SERVED)
def test_the_loading_table_is_unmoved_by_the_probe(name):
    with_probe = _loading(render(deck(name)).splitlines())
    without = _loading(render(without_probes(deck(name))).splitlines())
    assert with_probe and with_probe == without


def test_the_parallel_lc_and_its_equivalent_reactance_answer_alike():
    """0195's parallel LC and 0196's LD 4 are the same ohms at 299.7925 MHz.

    The oracle prints the two decks' row tables identically, digit for digit,
    which is what says the four RLC shapes reach the solve as one impedance
    rather than as four code paths; this holds the seam to the same identity.
    """
    assert _rows(oracle("0195")) == _rows(oracle("0196"))
    assert _rows(render(deck("0195")).splitlines()) == _rows(
        render(deck("0196")).splitlines()
    )


def test_a_probe_at_a_network_connection_point_refuses_by_name():
    """The shape the serve is NOT: capture 0120 with a probe at a TL end.

    NEC-5 answers it, and answers it differently - the two drive rows move
    from 0.22391 + 48.345j and 49.710 - 4.6729j ohms to 8.7300 + 58.204j and
    59.304 - 5.8734j, about 10 ohms each.  A 1e-10 V source cannot do that to
    a passive port, so the card is a port-voltage constraint there and not a
    probe, and the seam rule says refuse rather than reproduce a wrong answer.
    """
    reason = _refusal(deck("M_0120_ex4_plus_probe"))
    assert reason.startswith("the EX 0 at 1,-1 is one of EZNEC's 1e-10 V load probes")
    assert "the TL card between 3,1 and 1,-1" in reason
    assert "port-voltage constraint" in reason


def test_a_genuine_mixed_drive_refuses_by_name():
    """A real generator voltage beside a set current: 0 of 80 captures."""
    genuine = deck("0193").replace("EX 0,1,8,0,1.E-10,0.", "EX 0,1,8,0,1.,0.")
    reason = _refusal(genuine)
    assert (
        "the EX 0 at 1,8 sets 1 V, which is a generator rather than a probe" in reason
    )
    assert "GENUINE mixed drive" in reason


def test_a_tiny_voltage_at_an_unloaded_node_is_not_a_load_probe():
    """The middle condition, on its own: no LD under it, so it reads nothing.

    Off by one node from a served deck, and refused - which is what keeps the
    serve to the shape EZNEC writes rather than to any small voltage.
    """
    stray = deck("0193").replace("EX 0,1,8,0,1.E-10,0.", "EX 0,1,4,0,1.E-10,0.")
    reason = _refusal(stray)
    assert "the EX 0 at 1,4 sits at a node no LD card loads" in reason
