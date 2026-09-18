momwire#1116 - a one-field GE card.

Two OCF (off-center-fed dipole) decks from Mike WA7ARK, posted to QRZ thread
1003328 (#101 and #104, 2026-09-17), "Created from AutoEZ" and "Written by
EZNEC/Pro+ v. 7.0 in NEC-5 format" per their own CM lines. Both carry a bare
`GE 0` where every one of the 87 captured EZNEC decks carries the two-field
`GE n,-1`; the AutoEZ-side export path writes the shorter form, which the
issue's open thread question left unresolved (AutoEZ writing the deck itself,
or a hand edit) -- either way NEC-5 accepts it and takes its own default for
the missing second field.

  WA7ARK-OCF-LoadOnly.nec        The OCF with its LD 0 load, fed at 1,378.
  WA7ARK-OCF-Load-Xfmr-TL.nec    The same OCF plus an NT transformer and an
                                  NT lossy transmission line on a virtual
                                  wire (tag 3), fed at the virtual source
                                  3,2.

Kept out of tests/fixtures/eznec/ so the 80-capture corpus and its manifest
(tests/test_eznec_reproducibility.py) stay untouched.

The `.out` files are our licensed NEC-5 (nec5cl) printouts for these decks
exactly as captured, run 2026-09-17, and are quoted as numbers only, never as
internals. Both decks serve completely (impedance and RP) once the reader
accepts the one-field GE. Measured ANTENNA INPUT PARAMETERS rows:

  deck                          feed node   NEC-5 impedance
  WA7ARK-OCF-LoadOnly            1,378       275.53 - 1441.0j ohms
  WA7ARK-OCF-Load-Xfmr-TL        3,2         49.31 + 104.23j ohms

momwire's own bspline serve (post-fix) lands at 274.72 - 1431.8j and
49.98 + 104.96j respectively -- under 1 % on both R and X of the licensed
numbers above, which is what tests/test_eznec_one_field_ge_1116.py gates.

NEC-5 itself echoes the bare `GE 0` card in its own printout as
`GE    0    0` -- it takes 0 as I2's default, not a blank -- which is the
number the reader's own defaulting (`Card.i` reading a missing field as 0)
already reproduces without a separate sentinel.
