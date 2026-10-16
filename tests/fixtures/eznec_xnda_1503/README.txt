momwire#1503: 4nec2 writes RP 0 with XNDA 1003 by default (Generate > Far
Field, default Expert settings). These decks are one dipole with one RP card
at XNDA 1000, 1001, 1002 and 1003; the .nec5.out files are the NEC-5 engine's
printouts for them, captured BLACK BOX on Rocketlake (2026-10-15) with the
clean-room NEC-5 build ~/wardbench/nec5cl-cleanroom (`ENGINE DECK OUT`).
No NEC source or manual was read.

What they show (and tests/test_eznec_xnda_1503.py pins, on these captures
and on momwire's own printouts):
  - A = 3 prints exactly what A = 1 prints: the pattern rows and the
    average-gain trailer;
  - A = 2 prints the pattern heading and its column lines, no rows, then the
    average-gain trailer;
  - A = 0 prints the rows and no trailer.
The other digits (X = 0 major/minor axis gains, N >= 1 a NORMALIZED GAIN
table, D = 1 directive gains) each change the printout and are refused by
name.
