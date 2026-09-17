momwire#1105 - several EX 0 (set-voltage) sources through a TL/NT network.

Every captured multi-EX deck before this issue was either all EX 4 (current
sources) through a network (momwire#511, momwire#1110's mixed drive), or
several EX 0 with no network on the deck (momwire#1099).  Several EX 0
THROUGH a network was refused by name because nothing had printed a row to
say what NEC-5 does with the pair.

V_0120_two_voltage_through_network.nec
    capture 0120 (the cardioid L-network feed, already in the byte-gated
    corpus at tests/fixtures/eznec/decks/0120_cardioid-l-network-feed.nec)
    with its two `EX 4` cards rewritten to `EX 0` at the same two nodes and
    the same volts a unit-current EX 4 solve would need to deliver 1.414214 A
    there.  EZNEC writes a voltage-driven deck as its current-driven twin
    with the EX type field alone changed - captures 0186/0187 vs 0188/0189
    proved that conversion is the type field alone, payload untouched - so
    this hand edit is EZNEC's own shape, not a synthetic one.
V_0120_two_voltage_through_network.NEC5.OUT
    our licensed NEC-5's printout for it (2026-09-17).

0187.nec
    the network-free all-voltage four-square (already served since #1099):
    four EX 0 cards, no TL or NT card on the deck.  Kept here for CONTRAST,
    not as a gate for this issue - the network-free shape's byte gate is
    test_eznec_serve.py's 0032 rewrite, which predates and is unmoved by this
    fix.
0187.NEC5.OUT
    our licensed NEC-5's printout for it (2026-09-17).

What the V_0120 oracle shows: two ANTENNA INPUT PARAMETERS rows in deck
order (TAG 3 SEG 61 end 1, then TAG 2 SEG 31 end 2), each restoring its own
card's written volts byte-exact in the VOLTAGE columns and carrying a SOLVED
current - this is the mirror image of the EX 4 twin, where the current is set
and the voltage carries the network's answer (the current-driven capture at
tests/fixtures/eznec/printouts/0120_cardioid-l-network-feed.out prints
3.1666E-01 / 6.8370E+01 V at the same first node, an entirely different
number, because driving 1.4142 A and driving 1.4142 V into a 1e10 ohm virtual
pin are two different physical stimuli).  The STRUCTURE EXCITATION DATA AT
NETWORK CONNECTION POINTS table carries FOUR rows, not three - both driven
sites are themselves network connection points here (each EX 0 sits exactly
at a TL end), plus the two undriven connection points on the antenna wires -
same four TAG/SEG addresses the EX-4-driven twin's own connection table
prints, in the same discovery order (`_connection_points` walks the TL/NT
cards in deck order, which this fix does not touch).

The tolerance. bspline and NEC-5 solve this deck to different digits by
design, and the current-driven twin at the same mesh already disagrees with
NEC-5 (a separate solve, not a comparable number - see above), so the only
honest baseline is bspline vs NEC-5 on THIS deck, driven THIS way. Measured
2026-09-17: row 1 (TAG 3 SEG 61) Z = 12.673-6.2491j ohms against the oracle's
12.692-6.2371j, 0.16% of |Z|; row 2 (TAG 2 SEG 31) Z = -7.6388-12.519j against
-7.5822-12.483j, 0.46% of |Z|. Both land inside 3%, which is what this
fixture's tests hold them to - the same figure momwire#1110 used for a
properly meshed deck (its 0192 gate), and comfortably above what is actually
measured here.

V_0120 is a hand edit of an existing capture (0120, byte-gated as itself
elsewhere); 0187 is a fresh EZNEC Pro/2+ capture made for this issue, not part
of the 80-deck manifest. Either way the oracle is the licensed NEC-5's
printed output, quoted as numbers only. Kept out of tests/fixtures/eznec/ so
the 80-capture corpus and its manifest stay untouched.
