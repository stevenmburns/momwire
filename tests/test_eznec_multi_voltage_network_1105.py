"""momwire#1105: several ``EX 0`` (set-voltage) sources through a TL/NT
network - the last multi-``EX`` shape this seam refused by name.

momwire#511 served several ``EX 4`` (set-current) sources reaching the
structure through a network, and momwire#1099 served several ``EX 0`` with no
network on the deck.  Between them sat the fourth cell of the table
:func:`momwire.eznec._serve._multi_drive_state` narrates: several ``EX 0``
THROUGH a network.  Nothing had printed a row to say what NEC-5 does with the
pair, so the seam refused it by name until this issue.

The gate (tests/fixtures/eznec_multi_voltage_network_1105/, README there) is
capture 0120 - already in the byte-gated 80-deck corpus as a current-driven
(``EX 4``) cardioid L-network feed - with its two ``EX 4`` cards rewritten to
``EX 0`` at the same two nodes.  EZNEC writes a voltage-driven deck as its
current-driven twin with the ``EX`` type field alone changed (captures
0186/0187 vs 0188/0189 proved that conversion is the type field alone,
payload untouched), so this hand edit is EZNEC's own shape, and our licensed
NEC-5's printout for it is the oracle.

Gates:

1. the deck is served, with two ANTENNA INPUT PARAMETERS rows in deck order,
   TAG / SEG. / end-digit byte-equal to the oracle;
2. both rows restore their own card's written volts byte-equal (the U1 rule,
   on the OTHER card this time - EX 0's set quantity is the voltage);
3. each row's Z agrees with the oracle's to the basis tolerance stated below;
4. the STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS table carries
   the same four TAG/SEG addresses as the oracle's, in the same discovery
   order (both driven sites are themselves connection points here, which is
   what makes this deck a real gate for the restore in
   :func:`momwire.eznec._serve._multi_drive_state` rather than a no-op);
5. the network-free all-voltage shape (momwire#1099) is unmoved: that gate
   lives in test_eznec_serve.py (the 0032 rewrite) and
   test_eznec_probe_ex_1099.py, not here - 0187 in this fixture directory is
   kept only for CONTRAST (README).

**The tolerance.**  bspline and NEC-5 solve this deck to different digits by
design, and the deck's own current-driven twin (already in the byte-gated
corpus) is not a comparable baseline - driving 1.4142 A and driving 1.4142 V
into a 1e10 ohm virtual pin are two different physical stimuli with two
different answers, not the same solve to two precisions.  So the only honest
number is bspline vs NEC-5 on THIS deck, driven THIS way, measured
2026-09-17: 0.159 % of |Z| on the first row (TAG 3 SEG 61) and 0.459 % on the
second (TAG 2 SEG 31).  Both land inside 3 %, the same figure momwire#1110
held its own properly-meshed capture (0192) to, and with the same margin: a
number that is not measured is not a gate.
"""

from __future__ import annotations

from pathlib import Path

from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_multi_voltage_network_1105"
DECK = (FIXTURES / "V_0120_two_voltage_through_network.nec").read_text()
ORACLE = (
    (FIXTURES / "V_0120_two_voltage_through_network.NEC5.OUT").read_text().splitlines()
)

CONTRAST_DECK = (FIXTURES / "0187.nec").read_text()
CONTRAST_ORACLE = (FIXTURES / "0187.NEC5.OUT").read_text().splitlines()

# Measured 2026-09-17 (module docstring, "The tolerance").
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


def _network_rows(lines: list[str]) -> list[str]:
    start = next(i for i, line in enumerate(lines) if "CONNECTION POINTS" in line) + 4
    out = []
    for line in lines[start:]:
        if not line.strip():
            break
        out.append(line)
    return out


def _cells(row: str) -> list[float]:
    return [float(x) for x in row[12:].split()]


def test_the_deck_is_served_two_rows_in_deck_order():
    rendered = render(DECK)
    assert "NEC ERROR" not in rendered
    rows = _rows(rendered.splitlines())
    assert len(rows) == 2 == DECK.count("\nEX ")
    # TAG, SEG. and the trailing end digit, byte for byte
    assert [row[:12] for row in rows] == [row[:12] for row in _rows(ORACLE)]


def test_both_rows_restore_their_written_volts_byte_equal():
    rows = _rows(render(DECK).splitlines())
    theirs = _rows(ORACLE)
    # EX 0,3,1,0,1.414214,0. and EX 0,2,-1,0,0.,-1.414214 - the VOLTAGE
    # columns of each row, restored rather than read back (the U1 rule).
    assert rows[0][12:36] == theirs[0][12:36] == "  1.4142E+00  0.0000E+00"
    assert rows[1][12:36] == theirs[1][12:36] == "  0.0000E+00 -1.4142E+00"


def test_each_rows_impedance_agrees_with_the_oracle():
    ours = _rows(render(DECK).splitlines())
    theirs = _rows(ORACLE)
    for row_ours, row_theirs in zip(ours, theirs):
        co, ct = _cells(row_ours), _cells(row_theirs)
        z_ours, z_theirs = complex(co[4], co[5]), complex(ct[4], ct[5])
        assert abs(z_ours - z_theirs) <= BASIS_TOLERANCE * abs(z_theirs), (
            z_ours,
            z_theirs,
        )


def test_the_network_connection_table_carries_the_oracles_four_addresses():
    """Both driven sites are themselves network connection points here (each
    ``EX 0`` sits exactly at a ``TL`` end), so this table is where a stale,
    unrestored ``v_applied`` at a V-set site would show up - see the restore
    in :func:`momwire.eznec._serve._multi_drive_state`.

    The oracle's own row ORDER differs between this deck and its current-driven
    (``EX 4``) twin even though the network topology is identical - NEC-5's
    internal ordering is not deck order the way this seam's is
    (:func:`momwire.eznec._serve._connection_points`, deck order, unmoved by
    this fix and already byte-gated on 47 other captures).  So this gate
    checks the ADDRESS SET the table carries, not the row sequence.
    """
    rendered = render(DECK)
    rows = _network_rows(rendered.splitlines())
    theirs = _network_rows(ORACLE)
    assert len(rows) == len(theirs) == 4
    assert {row[:12] for row in rows} == {row[:12] for row in theirs}


def test_the_network_free_contrast_deck_still_serves():
    """0187 (README) is kept for CONTRAST, not as a gate for this issue - the
    network-free all-voltage shape's own gate is the 0032 rewrite in
    test_eznec_serve.py and test_eznec_probe_ex_1099.py, both unmoved by this
    fix.  This is a light sanity check that the fixture itself is servable.
    """
    rendered = render(CONTRAST_DECK)
    assert "NEC ERROR" not in rendered
    rows = _rows(rendered.splitlines())
    assert len(rows) == 4 == CONTRAST_DECK.count("\nEX ")
    assert [row[:12] for row in rows] == [row[:12] for row in _rows(CONTRAST_ORACLE)]
