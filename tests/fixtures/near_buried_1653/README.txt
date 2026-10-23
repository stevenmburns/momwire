momwire#1653 unit 1 - near fields (NE/NH) on decks with buried wires, and at
points in the soil.

Synthetic decks, written for this issue (decks.py; no user's deck): the shapes
of a buried radial screen under a vertical (Hub / Slant style) and a simple
buried dipole. 7 MHz, soil 13 / 0.005 Sommerfeld, wire radius 1 mm.

  stem               geometry
  d1_buried_dipole   5 m horizontal dipole at z = -0.3 m, centre fed
  d2_elev_screen     10 m vertical from z = 1 m, fed in its bottom segment,
                     over 8 detached 5 m radials at z = -0.15 m
  d3_hub             8 radials from a buried hub at -0.15 m, a 2-segment drop
                     to the z = 0 crossing node, a 10 m vertical fed in its
                     2nd segment
  d4_slant           a 10 m vertical on the z = 0 node, 8 radials slanting
                     from the node down to -0.15 m at 5 m

Every deck asks NE 0 along +x at z = 1, 0.01, -0.05 and -0.5 m (0.5-19.5 m),
NE 0 at z = 1 and -0.05 m (20.5-210.5 m), NH 0 at z = 1 and -0.05 m
(0.5-19.5 m) and NE 1 at R = 300 m, theta 10-160 deg.

Each stem comes in two dialects:

  <stem>.nec5.nec   NEC-5 dialect (GE -1 0, GN 0 ... NOFILE): what momwire's
                    NEC-5 seam reads, and the NEC-5 reference's input.
  <stem>.nec42.nec  NEC-4.2 dialect (GE -1, GN 2 ... NOFILE).

The .out files are the licensed engines' printouts of exactly those decks,
run 2026-10-23 on Haswell as black boxes (deck in, printout out, one fresh
directory per run, OMP_NUM_THREADS=1), quoted as numbers only:

  <stem>.nec5.ripplefix.out  NEC-5, exe/nec5-ripplefix, the static clean-room
                             build d0c1383 with the near-ground ripple fixed,
                             sha256 4f27b568bb35890bbe08cf7c67ce0661e251353f0e596289e004ba8f567e255b
  <stem>.nec42.out           NEC-4.2, main-pre exe/nec42,
                             sha256 02c6fc87c8cb75cbfc7444809d7e0656fcbeaf5b83689efb9e919956739647cd

ANTENNA INPUT PARAMETERS (ohms):

  deck               NEC-5 ripplefix       NEC-4.2 main-pre      razor-2p
  d1_buried_dipole   124.92 + j8.5311      128.27 + j15.849      124.88 + j8.5068
  d2_elev_screen     28.115 - j3938.1      51.308 - j9627.7      28.109 - j3938.0
  d3_hub             48.930 - j21.524      48.822 - j17.220      48.978 - j20.808
  d4_slant           48.436 - j30.740      47.177 - j29.396      48.284 - j31.914

What they are for: d1 and d2 are the decks whose SOLVES agree (razor-2p's Z
to 0.03 % of NEC-5's), so their near-field tables gate the READOUT, in all
four (source side, observer side) regimes (tests/test_near_buried_1653.py).
d3 and d4 carry the crossing node, where razor-2p's and NEC-5's currents
differ by 1-2 % (momwire#1149): their tables differ by that same uniform
factor, which is recorded and not gated. NEC-4.2's own impedance differs
from NEC-5's on every deck (d2: its end-segment source), so it is the second
opinion and not the gate.
