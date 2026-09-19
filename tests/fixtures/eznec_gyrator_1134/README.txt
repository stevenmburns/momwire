momwire#1134 - EZNEC's NT gyrator current source, and where it reports.

NEC-2 has no current-source EX card. EZNEC writes one anyway, by parking a
phantom wire ~100 lambda from the antenna and using its segments as circuit
NODES: an EX 0 voltage source on one of them, and an NT GYRATOR - Y11 = Y22 =
0, Y12 = Y21 = j - tying it to the segment it really drives. The deck names
the construction itself:

    CM ! Wire #3 for I srcs, shorted/open TL, and/or parallel loads.
    CM ! NT #1-2 are EZNEC current sources

A gyrator inverts impedance, so a source read where it SITS reports the
reciprocal of the antenna's driving point. The SOLVE is right; only the
readout port was wrong, which is what makes this a readout fix and not a
geometry one - the phantom wire stays in the structure.


Dan AC6LA's Cardioid - one antenna, three dialects
--------------------------------------------------
Two lambda/4 verticals 0.25 m apart at 299.7925 MHz over perfect ground, 18
ohm loads, driven in quadrature. Saved by EZNEC/Pro+ v. 7.0 in three NEC
dialects and reported by Dan; copied here verbatim from antennaknobs'
tests/fixtures/eznec_gyrator_1595/, which is where AK#1595 fixed the same
defect on the other seam.

Cardioidmodnec2.nec   NEC-2: the phantom wire (tag 3), two EX 0 and two NT
                      gyrators. THE SUBJECT.
Cardioidmodnec4.nec   NEC-4.2: native EX 6 current sources. Documentary
                      reference only - this seam refuses EX 6 by name ("not
                      part of this engine's nec5 dialect") - but it is what
                      pins the SIGN: the currents it asks for are
                      1.414214 + 0j on tag 1 and -1.414214j on tag 2, and
                      those are exactly what -Y12*V delivers from the NEC-2
                      deck's own two EX 0 cards.
Cardioidmodnec5.nec   NEC-5: native EX 4 current sources. The third dialect.
                      NOT an oracle for the NEC-2 deck's numbers: the two
                      dialects address different points - NEC-5 addresses
                      knots, NEC-2 segment centres - so their feeds AND their
                      loads sit half a segment apart and the served
                      impedances are 7-9 % apart even when both are read
                      correctly. Kept for provenance, not for a tolerance.


The controlled pair - what actually decides the readout
--------------------------------------------------------
Because the three dialects above do not agree numerically with each other,
the decisive gate is a pair of our own: one antenna, one drive point, one
delivered current, spelled two ways in ONE dialect.

native_current.nec    a 10-segment free-space dipole, EX 4 of 1.414214 A at
                      node 5 (its centre knot).
gyrator_current.nec   the same dipole, plus a phantom wire parked at 173
                      lambda, an EX 0 of 1.414214j V on it, and an NT gyrator
                      with Y12 = j tying that node to node 5 of the dipole.
                      -Y12*V = -j * 1.414214j = 1.414214 A, the native deck's
                      EX 4 exactly.

Measured (bspline, the seam's default basis):

    native            80.32694752428672 + 45.40508723322069j
    gyrator, fixed    80.32694752427943 + 45.405087233207354j
    relative          1.6e-13

    gyrator, before   0.009434642555716667 - 0.0053329672431875145j
    1 / native        0.009434644813007521 - 0.0053329658844504125j
    relative          2.4e-07

So the pair is pinned as AGREEMENT with the native deck, never as a ratio and
never as a reciprocal - dividing one side by the other is the mistake AK#1595
shipped, and it verifies nothing a sign or a scale error would not also pass.
The residual 1.6e-13 is the phantom wire's own coupling at 173 lambda plus the
three extra unknowns it puts in the matrix (12 against the native deck's 9);
nothing else differs.

Decks of our own except the three Cardioid files, and kept out of
tests/fixtures/eznec/ so the 80-capture corpus and its manifest stay
untouched. No licensed-engine printout is quoted here: the oracle is a second
deck in the same dialect through the same solver.
