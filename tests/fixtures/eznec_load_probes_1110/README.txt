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

  M_0195_two_loads_one_node.nec/.out
                 momwire#1787 (QRZ, 2026-11-01): two parallel-LC traps at ONE
                 node, the shape EZNEC writes for stacked traps - one 1e-10 V
                 probe per lumped load, so two identical EX 0 cards at 1,8.
                 Capture 0195 with a second LD 1 at 1,8 (2e-7 H, 3e-12 F) and
                 its probe; hand-edited, not an EZNEC session, run through the
                 licensed NEC-5 (2026-11-01). NEC-5 solves the pair as ONE
                 passive probe: the drive row equals the one-probe deck's to
                 every printed digit, and the probe row is printed once per
                 card, identically (not as a 2e-10 V source).

  N42_*.nec/.out, N2_*.nec/.out
                 momwire#1807: two lumped loads on ONE SEGMENT in the NEC-4.2
                 and NEC-2 dialects, which write no probes (the NEC-4.2 form
                 of M_0195 above). Synthetic decks on 0183's 11-segment
                 dipole, not EZNEC sessions. The N42_ printouts are licensed
                 NEC-4.2's, the release build nec42-release (sha256
                 5fc0d508e4cd0926...) on Haswell, each deck copied as m.nec
                 into a fresh directory and run as `nec42-release m.nec
                 m.out` with OMP_NUM_THREADS=1; the N2_ printouts are nec2c
                 1.3's (`nec2c -i deck -o out`). Run 2026-11-04, black box,
                 numbers only.
                   N42_0195_two_loads_one_segment  the issue's repro: the
                                 two parallel-LC traps of M_0195 at 1,8,
                                 EX 6 drive. NEC-4.2: 57.4228 - j197.494,
                                 the drive row of one LD 4 of their summed
                                 reactance (-j577.06).
                   N42_series_r_pair / _single     10 + 20 ohm series R
                                 against one 30 ohm: identical drive rows.
                   N42_mixed_three                 LD 0, LD 4 and a ranged
                                 LD 1 overlapping on segment 8.
                   N42_separated_by_ex / _single   an EX card between the
                                 two LDs: NEC-4.2 DISCARDS the first, and
                                 prints the second card's deck exactly.
                   N2_series_r_pair / _single, N2_mixed_three  the same
                                 shapes on nec2c, which agrees.
                 Both engines ADD the impedances of consecutive LD cards on
                 one segment and close the loading table with a "LOADED
                 TWICE - IMPEDANCES ADDED" note, each in its own layout; any
                 other card between two LD cards discards the earlier
                 loads. tests/test_two_loads_one_segment_1807.py holds the
                 loading tables byte for byte and the drive rows to the
                 slot's gap (the bare dipole is itself 0.97 % of |Z| off).

  N42_run_*, N42_ex*_fed_load_*, N42_ex6_phased_*, N2_run_*
                 momwire#1812 and momwire#1813, the same dipole and the same
                 two engines, run the same way on 2026-11-05 (nec2c 1.3-4).
                   N42_run_fr_other_segment / _single_segment3  10 ohm on
                                 segment 8, FR, 20 ohm on segment 3: the deck
                                 answers as the 20 ohm deck alone.
                   N42_run_ld5_dropped   an LD 5 conductivity, FR, 20 ohm:
                                 the conductivity is dropped too.
                   N42_run_zero_drops    10 ohm, FR, a zero-valued LD: the
                                 bare dipole (N42_run_bare).
                   N42_run_ek_keeps / _cm_keeps  an EK or a CM between two
                                 LD cards on one segment: the run is kept and
                                 the pair stacks (as N42_series_r_single).
                   N42_run_pq_keeps      a PQ between two LD cards on
                                 different segments: both loads stay.
                   N42_ex6_fed_load_* / N42_ex0_fed_load_*  EX 6 and EX 0 on
                                 segment 6 with an LD 0 (30 ohm), LD 1
                                 (500 ohm parallel 1e-7 H) or LD 4 (30-j40)
                                 on that segment: identical drive impedances,
                                 the load included.
                   N42_ex6_phased_fed_load / _unloaded  EX 6 on the loaded
                                 segment 6 beside an EX 0 on segment 3: the
                                 EX 6 row moves by the 30 ohm, exactly.
                   N2_run_fr_other_segment / _single_segment3, _ld5_dropped
                                 the nec2c forms, which agree.
                   N2_run_ek_breaks / _ek_single  nec2c's EK ENDS a run,
                                 where NEC-4.2's does not.
                   N2_run_pt_keeps       a PT keeps the run; the pair stacks.
                 tests/test_ld_runs_1812.py holds them.

  0202.nec/.out  The deck Dan AC6LA's "first blood" post (QRZ 1003328 #100,
                 2026-09-17) describes: EZNEC Pro/2+ 7.0.4's Cardioid.ez (the
                 plain 6-segment one), both sources type I as shipped, an
                 18 ohm load at 25 % of each vertical (segment 2 of 6) entered
                 through the RLC window, so EZNEC writes two EX 4 at the
                 bases and two 1e-10 V probes at the loads - captured on the
                 Windows sitting of 2026-09-18 (capture 0202, AK branch
                 capture/2026-09-18-writers), predicted card for card before
                 the run. Dan's own deck carried LD 4 (the R+jX window) where
                 this one carries LD 0 (the RLC window); the drive shape is
                 the same and that is what is under test. A hand reconstruction
                 of Dan's deck stood here for one day (momwire PR #1118) and
                 is replaced by this capture. 0.58.0 refused this shape with
                 the sentence Dan quoted; 0.59.0 serves it. Drive rows NEC-5
                 32.092 - 25.434j / 63.354 + 10.308j against bspline
                 (six-segment mesh, the dipole class); probe rows
                 1.1997 - 0.0540j / -0.0975 - 1.2501j A.

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
