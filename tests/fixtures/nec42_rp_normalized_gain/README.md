# RP with XNDA N >= 1 on the NEC-4.2 slot: the NORMALIZED GAIN table

Every deck here is synthetic, written for this probe by `gen_decks.py` (its
docstring and comments say what each family varies); no user deck. Each
`nec42/<deck>.out` is licensed NEC-4.2's printout of `decks/<deck>.nec`,
run as a black box: deck in, printout out, nothing else read (no NEC-4.2
source, build tree or manual). `free1500_big`'s printout is gzipped
(`gzip -n -9`, 426 KB plain).

- Engine: licensed NEC-4.2 (Burke, G. & Poggio, A. (2011). Numerical
  Electromagnetics Code version 4.2. LLNL-CODE-491368), the unmodified
  reference build `nec42` on the Haswell box
  (`~/stevenmburns/bench-nec42/exe/`, sha256 `02c6fc87c8cb75cb...`), the
  build `../eznec_nec42/README.txt` names for p54-p97.
- Protocol: each deck copied as `m.nec` into a fresh directory and run as
  `nec42 m.nec m.out` with `OMP_NUM_THREADS=1`; `m.out` kept as
  `nec42/<deck>.out`. Run 2026-11-17 for the 11-17 exe census (NEC-4.2
  decks refused with "RP with XNDA ... is not served by this engine's NEC-4
  printout").
- Committed as test fixtures with the licensee's permission (the
  `../eznec_nec42/` precedent). They are OUTPUT only.

The families (`../rp_normalized_gain/`'s NEC-5 set, in NEC-4.2's dialect:
no XQ after the RP, and the grounds this slot serves in place of GN 0):

    free<XNDA>          a tilted dipole, 40 ohm load, free space, 7 x 4
                        directions; N = 0..5 at D = 0 and 1 (A = 1), and
                        N = 5 at A = 0, 2, 3 (D = 0 and 1)
    free1<N>01_gnor*    GNOR (RP's tenth field) = 10, 0.5, -3, N = 1 and 5
    free1501_range      RFLD = 1000
    free1500_sweep      theta -180 to 178 in 2, one phi
    free1500_big        37 x 73 = 2701 directions
    free_tworp*         two RP cards in one run: N = 5 / 3, 5 / 0, 0 / 5
    small<n>            n = 1, 2, 4, 5 directions (the column split)
    cut<XNDA>           one phi (a theta cut): N = 0 and 5, A = 1, 2, 3
    phicut<XNDA>        one theta (a phi cut): N = 0 and 5, A = 1, 2
    mfreq<XNDA>         three frequencies
    zdip1401            a z dipole: every horizontal gain is the null
    xdip<XNDA>          crossed dipoles fed in quadrature (a real minor axis)
    gn1_* / gd_*        a near-vertical dipole over GN 1, and GN 1 + GD
    gn2_* / gn3_*       the same over GN 2 / GN 3 with NOFILE (13, 0.005 S/m):
                        XNDA 1501, 1511, 1301, and 1501 through the horizon

What the printouts show (`tests/test_nec42_rp_normalized_gain.py` pins each):

- The table is the NEC-5 printout's (`../rp_normalized_gain/README.txt`)
  to the byte: after the pattern (rows, and the average-gain trailer when A
  asks for it), four blank lines, the `- - - - NORMALIZED GAIN - - - -`
  heading, the gain's name, `NORMALIZATION FACTOR = <F9.2> DB`, three
  column-header lines and the cells, F10.2 theta, F9.2 phi, F10.2 dB, three
  groups filled column by column, the first n mod 3 one longer; then four
  blank lines before the next echo (six without a table).
- The same names for N = 1..5, the same factor (GNOR when nonzero, else the
  largest gain), the same cells (the gain less the factor), one table per
  RP card per frequency, A = 2 the table with no rows, D = 1 the factor
  moved and the cells not.
- On a linearly polarised structure the minor axis is round-off here too
  (free1201 prints a factor of -164.70 dB); momwire prints a linear row's
  minor axis as the null.
- A single cut (one theta or one phi) does not read A: at N = 0 and N = 5,
  A = 1 and 3 print no average-gain trailer and A = 2 prints the rows
  (cut*, phicut*, small*), NEC-5's rule (momwire#1962). The slot printed a
  trailer there before.

Regenerating a deck changes nothing here unless its text changes; a changed
deck needs a fresh NEC-4.2 run before its printout is replaced.
