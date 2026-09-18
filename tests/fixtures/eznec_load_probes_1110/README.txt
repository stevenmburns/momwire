momwire#1110 - EZNEC's 1e-10 V load probes beside an EX 4 current drive.

EZNEC Pro writes one extra `EX 0,tag,node,0,1.E-10,0.` per lumped load
("1.E-10 volt sources for recording currents at load locations") and reads the
load current out of that row of ANTENNA INPUT PARAMETERS. momwire#1099 served
that probe beside another EX 0; this seam still refused it beside an EX 4,
which is the pairing every current-driven EZNEC model with a load writes.

Unlike tests/fixtures/eznec_probe_ex_1099/, these are CAPTURES: EZNEC Pro/2+
v. 7.0.4 sessions exported in NEC-5 format, run through our licensed NEC-5
(2026-09-17). The printouts are the engine's output and are quoted as numbers
only, never as internals.

Kept out of tests/fixtures/eznec/ so the 80-capture corpus and its manifest
stay untouched (tests/test_eznec_reproducibility.py).

  0183.nec/.out  Dipole1: 11 segments, EX 4 at 1,6, no load and NO probe.
                 The LOADLESS CONTROL. bspline reads 85.101 + 45.802j ohms
                 here against NEC-5's 79.948 + 29.919j - 19.56 % of |Z| -
                 because EZNEC's own 11-segment mesh leaves NEC-5
                 under-converged. That number is where the dipole family's
                 tolerance comes from, and why it is 20 % and not 3 %: the
                 gate those four decks can carry is that a LOAD does not make
                 the mesh disagreement worse, not that the two engines agree.

  0193.nec/.out  The same dipole with a load at 1,8 and its probe beside it,
  0194           one deck per shape EZNEC's RLC load window writes:
  0195             0193  LD 0  18 ohm series R
  0196             0194  LD 0  series L+C (1e-7 H, 5e-12 F)
                   0195  LD 1  parallel L+C, the same values
                   0196  LD 4  -243.3432j ohms, the equivalent reactance
                 0195 and 0196 are the same impedance at 299.7925 MHz and the
                 oracle prints their row tables identically, digit for digit.

  0192.nec/.out  Terminated folded dipole, 69 segments a side over GN 0, fed
                 EX 4 on a virtual wire that reaches the structure through an
                 NT (ports 5,1 and 1,34), with an 820 ohm LD 0 at 2,34 and its
                 probe. The network case: the probe is NOT at either NT port,
                 which is what makes it a probe. A properly meshed deck, and
                 the two engines agree to 0.24 % on the drive row and 0.23 %
                 on the probe row.

  M_0120_ex4_plus_probe.nec/.out
                 The REFUSAL fixture, and the reason the "not at a connection
                 point" condition exists. Capture 0120 (cardioid, two EX 4
                 reaching the structure through a TL/TL/NT) with a probe hand
                 placed at 1,-1, which IS a TL end. NEC-5 ANSWERS it - and
                 answers it differently: the two drive rows move from
                 0.22391 + 48.345j and 49.710 - 4.6729j ohms (the unprobed
                 capture, tests/fixtures/eznec/printouts/0120_*.out) to
                 8.7300 + 58.204j and 59.304 - 5.8734j, about 10 ohms each.
                 A 1e-10 V source cannot do that to a passive port, so at a
                 connection point the card is a port-voltage CONSTRAINT and
                 not a probe. One printout is not a rule, so this shape stays
                 refused by name (docs/design/seam-rule.md: reproduce every
                 byte the host reads, never reproduce a wrong answer).
                 Hand-edited from 0120, not an EZNEC session.

  M_cardioid_two_ex4_two_probes.nec/.out
                 Dan AC6LA's "first blood" deck (QRZ 1003328 post #100,
                 2026-09-17): EZNEC Pro/4+'s Cardioid sample, two 6-segment
                 verticals over GN 1, current sources EX 4 at the bases and an
                 18 ohm LD 4 at segment 2 of each with its probe. Dan attached
                 the EX 0 twin written eight minutes earlier, so this is the
                 EX 4 deck the refusal sentence he quoted describes,
                 reconstructed by changing the two type fields alone (the
                 conversion momwire#1105 measured on 0186-0189). Hand-edited,
                 not an EZNEC session; its NEC-5 printout is the oracle.
                 momwire 0.58.0 refused it with Dan's sentence; 0.59.0 serves
                 it. Drive rows NEC-5 32.786 - 24.844j / 64.140 + 10.899j
                 against bspline 33.663 - 18.699j / 66.418 + 18.087j, 15 % and
                 12 % of |Z| - EZNEC's six-segment mesh, the dipole class -
                 and the probe rows 1.2005 - 0.0550j / -0.0986 - 1.2509j A
                 against 1.2088 - 0.0599j / -0.1085 - 1.2639j, within 1.3 %.

What the served oracles show: one ANTENNA INPUT PARAMETERS row per EX card in
deck order; the EX 4's set current and the probe's 1e-10 V both printed as
written; the probe row carrying the current through the loaded node, with
Z = V/I, Y = I/V and P = 0.5 Re(V I*) of its own numbers; and the loading
table unchanged.

One claim these five captures cannot make on their own is that the probe leaves
the DRIVE row alone, because none of them was also run without its probe. The
#1099 fixture's pair is the engine's own evidence for that on the all-EX-0
shape, the arithmetic says the same here (1e-10 V against the hundred-odd volts
a 1.414214 A drive puts across its own port is a relative 1e-12, seven decades
below the last of the five digits a row prints), and
tests/test_eznec_load_probes_1110.py holds the seam to it directly by rendering
each deck twice, with the probe lines and without.
