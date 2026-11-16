RP 0 with XNDA N >= 1: NEC's NORMALIZED GAIN table, measured BLACK BOX.

Every deck here is synthetic (written for this probe, no user deck). The
.nec5.out files are the licensed NEC-5 engine's printouts for them: our
Linux build of x13 (~/nec5-timing/nec5cl-x13 on Skylake, 2026-11-15),
`nec5cl DECK OUT` from the deck's own directory at OMP / OPENBLAS threads 1,
one fresh directory per deck (the engine caches SOMMPD.NEX in its cwd).
Only the printouts were read: no NEC-4.2 or NEC-5 source, no NEC-4.2 manual.
What the N digit selects is the public NEC-2 user's manual's (RP card,
XNDA: N = 1 major axis, 2 minor axis, 3 vertical, 4 horizontal, 5 total).
free1500_big's printout is gzipped (424 KB plain).

The decks:
  free<XNDA>         a tilted dipole (both polarisations nonzero), 40 ohm
                     load, free space, 7 x 4 directions; N = 0..5, D = 0/1
  free15<D><A>       the same at A = 0, 2, 3
  free1<N>01_gnor*   GNOR (RP's tenth field) = 10, 0.5, -3
  free1501_range     RFLD = 1000
  free1500_sweep     theta -180 to 178 in 2, one phi (the W8IO deck's form)
  free1500_big       37 x 73 = 2701 directions
  free_tworp*        two RP cards in one run, N = 5 / 3, 5 / 0, 0 / 5
  zdip1401           a z dipole: every horizontal gain is the null
  xdip<XNDA>         crossed dipoles fed in quadrature (elliptical; a real
                     minor axis)
  mfreq<XNDA>        three frequencies
  somm<XNDA>         a near-vertical dipole over 13 / 0.005 (GN 0,
                     Sommerfeld), N = 1..5, D = 0/1
  somm1501_below     the same, theta 0..180 (rows past the horizon)
  somm1501_mfreq     the same, two frequencies
  small<n>           n = 1, 2, 4, 5 directions (the column split)

What they show (tests/test_rp_normalized_gain_table.py pins each):
  - After the pattern (rows, and the average-gain trailer when A asks for
    it), four blank lines, then
        - - - - NORMALIZED GAIN - - - -        (column 38)
        <blank>
        <the gain's name, right-aligned to column 53>
        NORMALIZATION FACTOR = <F9.2> DB       (column 39)
        <blank>
        three column-header lines, always three groups
        the cells
    then four blank lines before whatever prints next (six without a
    table).
  - The names: MAJOR AXIS GAIN, MINOR AXIS GAIN, VERTICAL GAIN,
    HORIZONTAL GAIN, TOTAL GAIN for N = 1..5.
  - A cell is F10.2 theta, F9.2 phi, F10.2 dB; cells six blanks apart, no
    trailing blanks. The directions are the pattern's own, in its order
    (theta fastest), filled column by column into three groups, the first
    n mod 3 groups one longer: 28 cells print 10 / 9 / 9, 2701 print
    901 / 900 / 900, 4 print 2 / 1 / 1. No cap on the count.
  - Each cell is the row's gain in dB less the factor, at full precision
    (a cell just under the peak prints -0.00): VERT. for N = 3, HOR. for 4,
    TOTAL for 5, the ellipse's major / minor axis share of the total for
    1 / 2. A null row (-999.99) prints -999.99 less the factor.
  - The factor is GNOR when GNOR is nonzero (in dB: GNOR 10 prints 10.00
    and every cell 10 dB lower), else the table's largest gain.
  - D = 1 normalises the directive gains: the cells are D = 0's, the
    factor moves by 10 log10(Pin / Prad).
  - A = 0..3 does not move the table; A = 2 prints it with no pattern rows.
  - One table per RP card per frequency, right after that card's pattern;
    an N = 0 card in the same run prints none.
  - Past the horizon over ground the rows are nulls, and so are their cells.
  - On a linearly polarised structure the minor axis is round-off: x13
    prints -164.43 dB as the factor of free1201 and its cells as dust.
    momwire prints a linear row's minor-axis gain as the null instead.
