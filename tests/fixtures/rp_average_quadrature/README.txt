RP 0's AVERAGE POWER GAIN: the quadrature, measured BLACK BOX (follow-up to
momwire#1962 / #1967).

Every deck here is synthetic (written for this probe, no user deck): the
tilted dipole of ../rp_normalized_gain/ at 28.5 MHz, in free space, or raised
over GN 1 for the two gn1_* decks. The .nec5.out files are the licensed NEC-5
engine's printouts for them: our Linux build of x13 (~/nec5-timing/nec5cl-x13
on Skylake, 2026-11-16), `nec5cl DECK OUT` from a fresh directory holding
only the deck, at OMP / OPENBLAS threads 1. Only the printouts were read: no
NEC-4.2 or NEC-5 source, no NEC-4.2 manual.

The decks (RP 0 NTH NPH XNDA THETS PHIS DTH DPH), each deck's second comment
line saying what it varies:
  p2_d90 p3_d45 p4_d30_s10 p5_d10    phi short of a turn, 2 to 5 phis
  p19_d10 p36_d10 p37_d10            half a turn, a turn less a step, a turn
  p3_d180 p2_d180 p2_d360 p5_d120    big steps, 0..360 closed, past a turn
  p3_dm45 t3_dm30_p2                 a descending phi, a descending theta
  t2_p3 t5_full_p2                   two thetas; theta 0..180
  p2_d90_a2 p2_d90_a3                A = 2 and A = 3
  gn1_p2 gn1_p2_full                 over GN 1, theta 0..90 and 0..180
  z_*                                a zero step on an axis with >1 angle

THE RULE (how it was pinned). x13 prints |E(theta)| and |E(phi)| to six
figures on every row, and the gain is a constant times |E|^2 for one deck
geometry and ground. So each capture's printed average, divided by a
candidate quadrature of its own |E|^2 rows, implies that constant; the right
quadrature implies the SAME constant on every capture. It does: phi by the
trapezoid rule over the NPH - 1 steps (the first and last phi count half),
theta by momwire's clipped bands, gives one constant to 4e-6 across all 16
free-space captures and one across both GN 1 captures. momwire's rule (the
first NPH - 1 phis at full weight) spreads it by 40 %, and a sin(theta)
trapezoid in theta by 3 %. The two rules agree over a closed turn (the end
phis are one direction), which is why the full-sphere captures never told
them apart. test_rp_average_quadrature recomputes this.

What the captures show besides:
  - The solid angle is |sum of bands| * (NPH - 1) * |DPH|: momwire's already,
    a turn less a step (p36_d10) reads 3.8889 pi, past a turn (p5_d120)
    5.3333 pi, over GN 1 the whole requested theta range (1.0000 pi for
    0..180, though nothing radiates below the horizon).
  - A descending phi or theta reads as its ascending mirror (p3_dm45 and
    p3_d45 print the same trailer; t3_dm30_p2 and p2_d90 too).
  - A zero step on an axis with more than one angle spans nothing: x13
    prints NaN for the average and for the 4*pi power, and 0.0000 for the
    solid angle (z_*; at A = 2 too). momwire printed numbers.
  - A = 2 and A = 3 average as A = 1 (A = 2 without the rows).

The NEC-4.2 slot keeps its old trailer (`_serve._pattern`'s nec4_average);
the nec2c portal shares `average_gain` and keeps its old rule too. Neither
is measured here.
