RP 0 with ONE theta or ONE phi: XNDA's A digit, measured BLACK BOX (momwire#1962).

Every deck here is synthetic (written for this probe, no user deck), the
tilted dipole of ../rp_normalized_gain/ at 28.5 MHz in free space. The
.nec5.out files are the licensed NEC-5 engine's printouts for them: our Linux
build of x13 (~/nec5-timing/nec5cl-x13 on Skylake, 2026-11-16), `nec5cl DECK
OUT` from a fresh directory holding only the deck, at OMP / OPENBLAS threads 1.
Only the printouts were read: no NEC-4.2 or NEC-5 source, no NEC-4.2 manual.

The decks (RP 0 NTH NPH XNDA THETS PHIS DTH DPH):
  cut_phi_dphi10   3 1 1501 30 0 30 10   one phi, a nonzero phi step
  cut_phi_a2       3 1 1502 30 0 30 0    one phi, A = 2
  cut_phi_a3       3 1 1503 30 0 30 0    one phi, A = 3
  cut_theta        1 3 1501 30 0 30 45   one theta, a nonzero theta step
  cut_theta_a2     1 3 1502 30 0 30 45   one theta, A = 2
  cut_two_phi      3 2 1501 30 0 30 90   two phis: the control

What they show, with ../rp_normalized_gain/small{1,2,4,5} (one phi, DPH = 0,
A = 1; 1, 2, 4 and 5 thetas):
  - A card with NTH = 1 or NPH = 1 prints as at A = 0: no average-gain
    trailer at A = 1 or A = 3, and at A = 2 the pattern rows print (A = 2
    otherwise prints the trailer with no rows, ../rp_normalized_gain's
    free1502). The step on the single axis does not matter (DPH = 0, 10;
    DTH = 30), nor does it on the other.
  - Two phis (cut_two_phi) average as every other capture does.

Not here, because momwire does not print them yet (follow-ups):
  - with both counts above one and a zero step on one axis (RP 0 3 2 1501
    30 0 30 0, or DTH = 0 with three phis), x13 prints the trailer with NaN
    for the average gain and the power and 0.0000 for the solid angle;
    momwire prints numbers.
  - cut_two_phi's trailer VALUE: x13's average over a phi range short of a
    full turn is not momwire's (9.92383E-01 here against momwire's
    7.18063E-01, the same solid angle), as on ../rp_normalized_gain's
    free_tworp* and mfreq* captures. The tests compare this trailer by shape.
