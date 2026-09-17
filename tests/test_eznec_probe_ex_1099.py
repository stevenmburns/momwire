"""momwire#1099: EZNEC's 1e-10 V probe source beside a lumped load.

EZNEC Pro/4+ writes one extra ``EX 0,tag,node,0,1.E-10,0.`` per lumped load
("1.E-10 volt sources for recording currents at load locations") and reads the
load current back from that row of ANTENNA INPUT PARAMETERS. This seam refused
every deck with two ``EX 0`` cards, so a model with one R/L/C load could not
run at all (Dan AC6LA's spiral loop, QRZ 1003328 #84, 2026-09-16).

Gates, against our licensed NEC-5's printout for a deck of our own
(tests/fixtures/eznec_probe_ex_1099/, README there):

1. the deck is served, with one ANTENNA INPUT PARAMETERS row per EX card in
   deck order, TAG / SEG. / end-digit byte-equal to the oracle;
2. the drive row is byte-identical with and without the probe - the probe
   moves nothing the printout can show, exactly as the oracle's two printouts
   agree to every digit;
3. the probe row restores its written 1e-10 V, carries the current through
   the loaded node (within the basis tolerance of the oracle's), and prints
   Z = V/I and P = 0.5 Re(V I*) of its own numbers;
4. the refusal that stays: a probe at a TL/NT connection point beside an EX 4
   (momwire#1110 serves the mixed drive only OFF the network - at a connection
   point NEC-5 applies the card as a port-voltage constraint and its answer
   moves by ohms).

The mixed EX 4 + EX 0 drive itself is no longer refused: momwire#1110 serves
it wherever every EX 0 is one of these probes, and gates it on five EZNEC
captures in tests/test_eznec_load_probes_1110.py.  Several EX 0 THROUGH a
TL/NT network is no longer refused either: momwire#1105 serves it, gated on a
licensed NEC-5 printout in tests/test_eznec_multi_voltage_network_1105.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_probe_ex_1099"
PROBE = (FIXTURES / "synthetic_probe_ex.nec").read_text()
NOPROBE = (FIXTURES / "synthetic_probe_ex_noprobe.nec").read_text()
ORACLE = (FIXTURES / "synthetic_probe_ex.out").read_text().splitlines()
ORACLE_NOPROBE = (FIXTURES / "synthetic_probe_ex_noprobe.out").read_text().splitlines()

# bspline against the licensed engine on a 21-segment dipole: the two solve
# the same deck to different digits by design (see the 1092 fixture README);
# the currents agree to about a percent, which is what this holds.
BASIS_TOLERANCE = 0.03


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


def _refusal(text: str) -> str:
    printout = render(text)
    assert " ***** NEC ERROR - " in printout, "this deck was SERVED"
    return printout.rsplit(" ***** NEC ERROR - ", 1)[1].strip()


def test_the_probe_deck_is_served_one_row_per_card_in_deck_order():
    rendered = render(PROBE)
    assert "NEC ERROR" not in rendered
    rows = _rows(rendered.splitlines())
    oracle = _rows(ORACLE)
    assert (
        [r[:12] for r in rows]
        == [r[:12] for r in oracle]
        == [
            "   1    11 1",
            "   1     5 1",
        ]
    )


def test_the_drive_row_is_byte_identical_with_and_without_the_probe():
    with_probe = _rows(render(PROBE).splitlines())
    without = _rows(render(NOPROBE).splitlines())
    assert len(without) == 1
    assert with_probe[0] == without[0]
    # the oracle says the same of itself
    assert _rows(ORACLE)[0] == _rows(ORACLE_NOPROBE)[0]


def test_the_probe_row_restores_its_volts_and_carries_the_load_current():
    rows = _rows(render(PROBE).splitlines())
    ours = _cells(rows[1])
    theirs = _cells(_rows(ORACLE)[1])
    # the written 1e-10 V is a boundary condition, printed as written
    assert rows[1][12:36] == _rows(ORACLE)[1][12:36] == "  1.0000E-10  0.0000E+00"
    i_ours, i_theirs = complex(ours[2], ours[3]), complex(theirs[2], theirs[3])
    assert abs(i_ours - i_theirs) <= BASIS_TOLERANCE * abs(i_theirs), (i_ours, i_theirs)
    # and the row's own arithmetic: Z = V/I, Y = I/V, P = 0.5 Re(V I*)
    v = complex(ours[0], ours[1])
    z, y, p = complex(ours[4], ours[5]), complex(ours[6], ours[7]), ours[8]
    assert abs(z - v / i_ours) <= 1e-3 * abs(z)
    assert abs(y - i_ours / v) <= 1e-3 * abs(y)
    assert abs(p - 0.5 * (v * i_ours.conjugate()).real) <= 1e-3 * abs(p)


def test_the_drive_row_agrees_with_the_oracle_to_the_basis_tolerance():
    ours = _cells(_rows(render(PROBE).splitlines())[0])
    theirs = _cells(_rows(ORACLE)[0])
    z_ours, z_theirs = complex(ours[4], ours[5]), complex(theirs[4], theirs[5])
    assert abs(z_ours - z_theirs) <= BASIS_TOLERANCE * abs(z_theirs), (z_ours, z_theirs)


def test_the_loading_table_is_unmoved_by_the_probe():
    def loading(lines):
        start = next(
            i for i, l in enumerate(lines) if "STRUCTURE IMPEDANCE LOADING" in l
        )
        return [l for l in lines[start : start + 12] if l.strip().startswith("1 ")]

    assert loading(render(PROBE).splitlines()) == loading(render(NOPROBE).splitlines())


def test_a_probe_at_a_network_connection_point_still_refuses_by_name():
    """This deck's own shape with the drive turned into an EX 4 is SERVED
    since momwire#1110; hang a TL off the probe's node and it is not.

    Kept here, where the mixed-drive refusal used to be, because this is the
    line that moved rather than a line that vanished: what a probe beside an
    EX 4 needs is to be OFF the network.
    """
    mixed = PROBE.replace("EX 0,1,11,0,1.414214,0.", "EX 4,1,11,0,1.414214,0.")
    assert "NEC ERROR" not in render(mixed)
    at_port = mixed.replace(
        "EX 0,1,5,0,1.E-10,0.\n",
        "EX 0,1,5,0,1.E-10,0.\nTL 1,5,1,19,50.,0.,0.,0.,0.,0.\n",
    )
    reason = _refusal(at_port)
    assert reason.startswith("the EX 0 at 1,5 is one of EZNEC's 1e-10 V load probes")
    assert "the TL card between 1,5 and 1,19" in reason
    assert "port-voltage constraint" in reason


def test_several_voltages_through_a_network_are_served_now():
    """The refusal that used to live here, inverted (momwire#1105, 2026-09-17).

    Neither ``EX 0`` node is on the added ``TL`` here (the drive is at 1,11,
    the probe at 1,5, the line hangs off 1,3), so this is the plain
    all-voltage-through-a-network shape, not the network-connection-point one
    :func:`test_a_probe_at_a_network_connection_point_still_refuses_by_name`
    covers.  The dedicated fixture test
    (tests/test_eznec_multi_voltage_network_1105.py) is where that shape is
    gated against a licensed NEC-5 printout; this is the synthetic deck the
    refusal used to be pinned on, now asserting it serves instead.
    """
    with_line = PROBE.replace(
        "EX 0,1,5,0,1.E-10,0.\n",
        "EX 0,1,5,0,1.E-10,0.\nTL 1,3,1,19,50.,0.,0.,0.,0.,0.\n",
    )
    printout = render(with_line)
    assert " ***** NEC ERROR - " not in printout, printout
    rows = _rows(printout.splitlines())
    assert len(rows) == 2
    # Both EX 0 cards restore their own written volts byte-exact.
    assert rows[0][12:36] == "  1.4142E+00  0.0000E+00"
    assert rows[1][12:36] == "  1.0000E-10  0.0000E+00"


@pytest.mark.parametrize("deck", [PROBE, NOPROBE])
def test_both_fixtures_parse_to_the_oracle_row_count(deck):
    n_ex = deck.count("\nEX ")
    assert len(_rows(render(deck).splitlines())) == n_ex
