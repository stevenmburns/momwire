"""momwire#510 — is the grazing floor the interpolation grid's θ resolution?

Where experiments 1 and 3 left it
---------------------------------
The floor is in the finite ground (``GN 1`` is exact at 1.09e-4 λ), it is
keyed to h/λ and not h/Δ, it gets WORSE under mesh refinement, and it is the
HORIZONTAL wire in the grazing zone rather than the grazing feedpoint — a lone
vertical whose bottom segment grazes is exact.  Read together that says
catastrophic cancellation in the *numerical* ground: the same cancellation
over a closed-form image is exact, so it is the evaluation that fails, not the
physics.

``probe_contact_direct_remainder.py`` split that evaluation into two axes at
CONTACT and found the integration tolerance SATURATED (rtol 1e-7 through 1e-13
all give the same residual) with the whole ~0.003 Ω shift belonging to the
INTERPOLATION.  Contact is a different regime, so the split has to be asked
again here — but the grid's own layout comment already names the premise this
deck breaks::

    # The grazing band (region 0) keeps the 0.01-lambda spacing: there the
    # layer variable h = R1 sin(theta) stretches it out in R1, no physical
    # deck queries small R1 at grazing (two points at grazing-small R1 means
    # both are ON the plane — radial screens, which are refused) ...

0033 **is** an elevated radial screen and it is **not** refused.  And the
premise's mechanism fails for it specifically: the layer variable
h = R₁·sinθ is what "stretches out in R₁" for a general pair, but for two
points that both sit at a fixed grazing height it is PINNED at z_s + z_o
however large R₁ grows.  It does not stretch.

What this measures
------------------
``--mode theta`` is pure geometry against the shipped grid — no solver, no
binary.  For 0033's own mesh it walks every source/observer pair, computes the
(R₁, θ) that pair hands the grid, and asks the grid ITSELF (constructed at
this deck's ε̃, k and extent, so the numbers are the shipped lattice's and not
a re-derivation) which θ cell each query lands in.  If the queries pile into
the first cell, the cubic stencil is interpolating a function of h across a
cell whose h-span is orders of magnitude wider than the h being asked about.

``--mode direct`` is the decisive one: swap the interpolated grid for direct
evaluation and re-run 0033 against the binary at several heights.  If the
error collapses, the floor is the interpolation lattice — a keying defect with
a fix, the same shape momwire#443 already fixed once for the contact boundary
layer — and #510 stops being a refuse-only formulation limit.  If it does not,
the integration or the formulation underneath is next.

``--mode reflcoef`` partitions what is left.  ``refl-coef`` and ``sommerfeld``
share the image fold and differ only by the remainder Q, so forcing the seam's
ground kwargs to refl-coef — one monkeypatch on ``_serve._ground_kwargs``,
everything else about the deck, drive, mesh and route held — asks whether the
error lives in the shared half.  The deck route has no card for it (``GN 0``
maps to sommerfeld and should), which is why it is patched rather than
spelled.  refl-coef is outside its own documented 0.1-0.5 λ validity window
down here, so its ABSOLUTE error is not evidence of anything; what is evidence
is whether it tracks sommerfeld.

WHAT IT MEASURED (2026-08-25)
-----------------------------
``--mode theta`` — 0033's own mesh, 325 pairs, against the shipped lattice:

  h/λ            1.09e-4    1e-3    3e-3    1e-2    3e-2     1e-1
  median θ (deg)   0.115   1.058   3.171   10.46   23.66       54
  in first cell    64.6 %  64.6 %  64.6 %  43.1 %     0 %      0 %
  razor err%      171.86   44.18    3.91    0.06    0.00     0.01

The grazing band's θ cell is **10°** (region 0, R₁ < 0.2 λ) and 5° beyond it,
and at the native height the MEDIAN query sits at 0.115° — 87× inside the
first cell.  The error tracks how deep into that first cell the queries fall,
which is exactly what an interpolation-resolution failure looks like.

``--mode direct`` — and it is NOT that.  With the lattice removed entirely and
the surfaces evaluated directly at rtol 1e-11:

  h/λ        grid err%   direct err%
  1.09e-4      435.18       435.23    bspline
  1.09e-4      171.86       172.08    razor-nec5
  1e-3          44.18        44.19    razor-nec5
  3e-3           3.91         3.91    razor-nec5
  1e-2           0.06         0.06    razor-nec5

Identical to three figures.  **The interpolation lattice is exonerated, and so
is the integration tolerance** (direct evaluation IS the fill function, swept
at a tolerance two decades past production).  A textbook-looking resolution
story measured out — the surfaces are evidently smooth enough in θ near zero
that a cubic across 10° carries them.

``--mode reflcoef`` — and refl-coef does NOT track sommerfeld here, which is
where this parts company with #624's contact finding.  They separate, and the
gap grows exactly as the error does (razor-nec5, |Z_somm − Z_refl|): 10.0 Ω at
1e-2 λ, 22.9 at 3e-3, 71.9 at 1e-3, 176.3 at 1.09e-4.  Since the two differ
only by the remainder Q, that gap IS Q's contribution.

The sharpest form, and the one to work from — the GROUND CORRECTION,
Z(``GN 0``) − Z(``GN 1``), which is what the finite ground is worth on top of
a perfect image.  razor-nec5 over ``GN 1`` reproduces the binary to 0.00 % at
every height, so the binary's own PEC answer is a common baseline both sides
share exactly:

  h/λ       true (binary)        momwire (razor)      overshoot
  1e-2      1.577  +0.670j       1.582  +0.739j        1.02x
  3e-3      3.084  +8.700j       3.391 +13.009j        1.46x
  1e-3      4.473 +20.397j       9.065 +63.307j        3.06x
  1.09e-4  13.900 +61.346j      82.971+144.743j        2.65x

**momwire over-computes the finite-ground correction by up to ~3× at grazing
and is exact at 1e-2 λ.**  Backing refl-coef's undershoot out, Q is worth
176 Ω at the native height where it should be worth 71 Ω.

So the defect is in **how Q is scaled or folded into the system, not in
computing it** — its evaluation is verified correct by ``--mode direct`` and
its contribution is 2.5× too large anyway.  That is a bounded, well-localized
target rather than a formulation dead end.  It also rhymes with #624's
row-halving suspicion (a model-independent scale factor on a near-plane row)
without contradicting experiment 3: different geometric trigger and different
symptom, possibly one defect family.

(bspline's overshoot column is not readable the same way — its own ~5 % basis
error rides on a correction that is only 1.7 Ω at 1e-2 λ, which is what makes
that row 4.77×.  razor's column is the clean one, and it is clean precisely
because razor over PEC is exact.)

``--mode soil`` — and the scale factor is NOT one number.  |Δ_mw / Δ_true| for
razor-nec5 across the five golden half-spaces:

  h/λ        sea    vgood     avg    poor    diel      arg spread
  1e-2      1.066   1.025   1.019   0.978   0.997      0.0 .. 2.0 deg
  3e-3      2.048   1.482   1.456   1.661   0.889     -8.8 .. 7.8
  1e-3      5.455   3.183   3.063   3.261   1.946    -77.7 .. 12.3
  1.09e-4   4.436   3.033   2.652   3.232  61.188    -45.8 .. 1.9

Two readings, and the first is a control: **at 1e-2 λ the correction is right
on all five half-spaces** (0.98-1.07, phase ≤ 2°), so the floor is a property
of HEIGHT and not of any one soil.

Below it the overshoot is **soil-DEPENDENT** — 2.65× to 4.44× across the four
real soils at the native height, with the phase wandering −16° to −57°.  A
single real constant does not describe that, so **row-halving is ruled out as
the mechanism** and the analogy to #624 does not carry.  It was worth testing
and it is dead.

What replaces it is a better lead: **the lossless dielectric is catastrophic**
— 61× at the native height, against 2.6-4.4× for soils that conduct.  It is
also the one half-space whose true correction has a NEGATIVE real part
(−5.288 + 7.670j: a lossless dielectric lowers the resistance relative to a
perfect image).  momwire answers 110.7 + 559.2j to that.

``--mode sigma`` / ``--mode epsr`` — the two ORTHOGONAL axes, because the five
golden soils confound them (``diel`` is the least conductive AND least dense,
``sea`` both the most).  Absolute error in the ground correction, |Δ_mw −
Δ_true| in Ω, razor-nec5 at the native height:

  sigma       0     1e-6    1e-5    1e-4    3e-4    1e-3    3e-3    1e-2    1e-1
  eps_r 5   300.2   300.3   300.7   279.8   205.8   136.3   120.4   117.1   116.3
  eps_r 20   26.6    26.6    26.9    29.3    35.0    54.1    86.7   111.2   116.2

  eps_r    1.05    1.5     2.5     3.0    3.10    3.6      5      13      20     81
  |err|   18.95  96.71  563.58 2480.3  3343.9  939.6  300.7   63.92  26.85  67.53

**Two separate things, and the controls separate them.**

*One:* a **shared, soil-independent grazing defect** — both σ sweeps converge
on the SAME ~116 Ω plateau once the ground conducts (tan δ ≳ 5), from opposite
directions.  Across the four lossy golden soils the ABSOLUTE error is 82-136 Ω
while the true correction varies 23-63 Ω, so it is an ADDITIVE error that
barely depends on the half-space — not a multiplicative one.  That is a
sharper statement than the soil mode's ratio column supports on its own, and
it is a second reason row-halving is the wrong suspect: a mis-scaled row would
give error ∝ Δ_true.

*Two:* a **razor-specific resonance in ε_r at ≈ 3.1**, at deep grazing with
low loss.  Resolved finely, and it is a pole, not a bump:

  eps_r    2.60    2.80    2.90    3.00    3.05    3.10    3.20    3.40    3.60
  razor   695.7  1171.9  1650.9  2480.3  3002.4  3343.9  2797.1  1442.3   939.6
   arg    -44.1   -46.5   -53.2   -69.3   -84.1  -104.1  -142.2  -170.5  -177.6
  bspline  99.3   109.6   114.3   118.7   120.7   122.8   126.7   133.8   140.3

A magnitude peak with a ~135° PHASE SWEEP through it, in razor alone; bspline
walks smoothly 99 → 140 Ω across the same window with no feature at all.  That
reads as a spurious pole in razor's ASSEMBLY, and ``--mode direct --eps-r 3.0``
confirms it is not the Sommerfeld evaluation: grid 2466.91 % against direct
2470.40 %, so the one-soil exoneration above survives its worst corner.

**Correcting the reading this probe reached one run earlier.**  The ε_r = 5
σ-sweep's monotone fall (300 → 116 Ω) looked like momwire#282 stage 2's "a
loss term must vanish with σ" shape.  It is not: ε_r = 5 sits on the
resonance's SHOULDER, and what σ damps there is the pole.  Clear of it at
ε_r = 20 the trend REVERSES (26.6 → 116 Ω).  So #282's σ resemblance is not
confirmed, and the earlier note claiming it was is superseded by the
orthogonal sweep.  ε_r ≈ 3 is also outside real ground (5-81), which is why
no captured deck meets the pole — it is a diagnostic pointer, not a user
symptom.

Runs the binary in ``--mode direct`` / ``--mode reflcoef``, so: antennaknobs
``--radial-lens`` — the pole MOVES with geometry.  |err| in Ω at the native
height, razor-nec5, scanning ε_r at four radial lengths:

  eps_r      1.8    2.2    2.6    3.0    3.4    4.0     5.0     6.5     8.0
  L=20      21.5   29.4   36.1   41.9   46.9   53.4    62.1    72.0    79.5
  L=30      61.5   89.4  117.2  145.5  175.2  223.7   322.8   557.7  1087.4
  L=39.6   171.8  330.0  695.7 2480.3 1442.3  565.6   300.7   184.2   133.5
  L=60     332.5  209.0  139.0   76.4   15.2  208.2  4217.2   948.2   640.5

So it is not one ε_r: L = 39.6 peaks near 3.0, L = 60 near 5.0, and L = 20/30
show no peak below 8 at all.  A coefficient singularity would sit at one ε̃
whatever the antenna is.  At L = 60, ε_r = 5 the solved impedance reaches
3600 + 2678j — an ANTI-RESONANCE, an open circuit where the antenna has no
business having one.

``--mode cond`` — **and it is NOT a singular matrix.**  Conditioning of what
``scipy.linalg.solve`` actually inverts, at the native height, L = 39.6:

  eps_r      1.8    2.2    2.6    2.9    3.0    3.1    3.2    3.6    4.0    5.0     8.0    13.0
  razor cond 442.2  180.7  109.3  84.23  78.33  73.26  68.88  60.23  66.81  82.95   126.2  1410
  razor |err| 171.8  330.0  695.7 1650.9 2480.3 3343.9 2797.1  939.6  565.6  300.7   133.5  63.9
  bspl cond  113.8  119.0  119.6  119.9  120.0  120.1  120.2  128.6  143.5  172.4   221.4  256.4

The correlation is **inverse**: razor's operator is at its BEST conditioned
(cond 73, σ_min 136) exactly where the error is WORST, and at its worst
(cond 1410) where the error is smallest (63.9 Ω).  bspline's is smooth and
featureless throughout.  So the matrix is fine and its ENTRIES are wrong —
the anti-resonance is not rank loss, and ill-conditioning joins the
interpolation, the integration tolerance and row-halving on the list of
things this is not.

``--mode wire`` — **the minimal reproducer.**  ONE horizontal wire, 39.624 m,
five segments, driven at node 2: no junction, no screen, no vertical.

  h/λ        eps_r 2.5   3.0   3.1    5.0    13.0     (razor |err|, Ω)
  1e-2            0.23  0.20  0.22   0.26    0.27
  1e-3           33.0  39.7  40.9   57.1    79.7
  1.09e-4       599.6 753.7 783.0 1283.9  3281.7

It reproduces completely — clean at 1e-2 λ (ratio 0.996-1.005), broken below.
**The reproducer drops from 24 unknowns to 6**, which makes entry-by-entry
diagnosis tractable and gives any fix a unit test that runs in milliseconds.
It also confirms experiment 3's reading from the smallest possible model: the
horizontal wire alone is sufficient, the screen and the junction are not part
of the mechanism.

Two cautions this reproducer raises.  The wire is a far more sensitive object
than 0033 — Z over ``GN 1`` at 1e-2 λ is 0.0395 − 1035.2j, a real part of
0.04 Ω, so the ground correction dwarfs the PEC answer and bspline's own basis
error is amplified with it (bspline is 35-39 Ω out even at the clean rung, and
its column here is not readable as basis-independent evidence).  And razor
UNDERSHOOTS here at 1e-3 (ratio 0.73-0.83) where on 0033 it overshot 3.06× —
so the error's SIGN is geometry-dependent, which is a third reason no single
scale factor describes it.

``--mode matrix`` — **the finding, and it needs no binary at all.**

Reciprocity first, and it is clean: ‖Z − Zᵀ‖/‖Z‖ sits at 1e-15 to 1e-17 for
both trunks, at every height, over PEC and over ``GN 0`` alike.  No asymmetry
bug.

Then WHERE the ground correction lives, |ΔZ| averaged by |i−j|:

  razor-nec5     |i-j| = 0      1       2       3
  h/λ = 1e-2            13.2    5.4    1.03    0.145
  h/λ = 1e-3            62.8   34.5   0.915   0.0958
  h/λ = 1.09e-4        439     235    0.873   0.11

The correction blows up **only in the near-diagonal band** — 33× on the
diagonal and 43× on the first off-diagonal between 1e-2 and 1.09e-4 λ — while
**every entry at |i−j| ≥ 2 is height-independent**.  Whatever is wrong is in
the self and nearest-neighbour ground terms, the ones where a source and its
own image nearly coincide (R₁ → 2h).

That growth is not by itself an error: a horizontal wire close to a dielectric
genuinely has a large near-field coupling to it.  What settles it is the
LIMIT, which is computable from momwire's own machinery.  As h → 0 the
near-diagonal correction is the incomplete cancellation of an antiparallel
image — PEC images a horizontal current at exactly −1, a half-space at −Γ — so

    |ΔZ| / |PEC image term|  ->  |1 − Γ| = |2/(ε̃ + 1)|

and the PEC image term is measurable without any reference at all, as
Z(``GN 1``) − Z(``GN -1``).  For average soil at 1.832 MHz, ε̃ = 13 − 49.06j
and the limit is **0.0392**:

  |ΔZ| / |PEC image|, bands 0 / 1 / 2
  h/λ         bspline                    razor-nec5
  1e-2        0.0603  0.0785  0.0411     0.0656  0.0813  0.0292
  1e-3        0.0498  0.0728  0.0399     0.0620  0.0716  0.0201
  1.09e-4     0.0484  0.0740  0.0408     0.2357  0.2537  0.0192

**bspline is height-STABLE and razor is not.**  bspline holds ~0.05/0.074/0.041
at all three heights — converging to a constant, as the physics requires — and
razor tracks it down to 1e-3 and then jumps 4× at the last height, to 0.236 and
0.254, exactly where the quasi-static limit is most valid.

The absolute agreement with 0.0392 is only approximate (bspline sits within a
factor ~2, which is what a leading-order estimate is worth), and the finding
does not rest on it.  It rests on razor's own ratio MOVING 0.062 → 0.236
between two heights where it should be settling to a constant, on a band whose
neighbours are height-independent, in a basis whose sibling does settle.

So the defect is named as precisely as this arc can name it without reading
the assembly: **razor's self and nearest-neighbour finite-ground entries carry
about 4× too much of the perfect-image term at deep grazing.**  The surfaces
are right, the medium is right, the matrix is well conditioned and symmetric,
and bspline's identical band converges.

And it hands the fix a **reference-free acceptance test**: on the one-wire
reproducer, |ΔZ|/|PEC image| in the near-diagonal band must tend to a constant
as h → 0, and that constant must be |2/(ε̃+1)|.  No binary, no captured deck,
six unknowns, milliseconds.

``--mode nqp`` — **and that is the whole of #510.**

Reading razor's assembly with the band named: the remainder Q rides the T1
window and is integrated over each SOURCE SEGMENT with ``n_qp_sommerfeld``
Gauss points, default **3**.  For a grazing horizontal pair the Q integrand
has a spike of width ~2h where the observer sits over the source's image —
h = 1.78 cm inside a 7.92 m segment, a relative width of 0.0022.  Three Gauss
points cannot see a feature that narrow, and the spike sharpens as h → 0.
That is exactly the band, and exactly the height dependence, ``--mode matrix``
measured.

It is consistent with every earlier elimination, which is what made it worth
testing: the direct-grid bypass changed how each quadrature point's SURFACE is
evaluated, not how many points there are, so exonerating the surfaces said
nothing about the order.  And momwire#282 raising n_qp 3 → 12 at CONTACT moved
its answer 0.03 Ω — which is why this was not tried sooner, and is not a
counterexample: contact is a VERTICAL wire, its image is collinear, and there
is no spike to miss.

The near-diagonal band, |ΔZ|/|PEC image| against the 0.0392 limit:

  n_qp              3       6      12      24      48      96
  h/λ 1e-3     0.0620  0.0457  0.0469  0.0474  0.0475  0.0475
  h/λ 1.09e-4  0.2357  0.0416  0.0419  0.0425  0.0433  0.0440

The 4× excess is gone at n_qp = 6 and the band lands on bspline's value.

End to end against the binary, razor-nec5 on 0033 — err % of |Z|:

  h/λ        n_qp=3     12      24      48      96     192
  1.09e-4    171.86  61.02   44.15   26.05    9.87    1.44
  2e-4       239.54  31.86   19.18    7.74    1.32    0.08
  5e-4       230.91  10.02    3.24    0.37    0.05    0.05
  1e-3        44.18   2.52    0.28
  3e-3         3.91   0.02    0.01
  1e-2         0.06   0.01    0.01

**171.86 % → 1.44 % at the captured height.**  0033 and 0034 have been
"divergent" for want of Gauss points.  #510 is a quadrature-ORDER defect, it
is entirely fixable, and nothing else in the elimination list has to be
revisited.

The order needed scales like the spike is narrow: h/Δ = 0.0103 wants ~48,
h/Δ = 0.00225 wants ~192, so roughly **n_qp ≈ 0.4·Δ/h** — a keying rule of the
same shape momwire#443 already applied to the grid's boundary layer.  Keying
is the design point rather than a global raise: 192 points on every pair would
be ruinous, and the spike exists only where an observer sits over a source
segment's near-coincident image.

Runs the binary in every mode but ``theta``, ``matrix`` and ``nqp``, so:
antennaknobs venv, ``NEC5_EXE`` set.

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/probe_grazing_ground_path.py --mode wire
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from momwire import _sommerfeld
from probe_grazing_height_floor import (
    FREQ_MHZ,
    N_SEG,
    RADIAL_LEN,  # noqa: F401  (re-exported for --radial-lens' default)
    RADIUS,
    SOIL,
    TRUNKS,
    VERT_LEN,
    WL,
    _num,
    deck,
    momwire_z,
)

EPS0 = 8.8541878128e-12

# The heights the two earlier probes bracket the floor with.
THETA_HEIGHTS = (1.09e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DIRECT_HEIGHTS = (1.09e-4, 1e-3, 3e-3, 1e-2)

# The soil sweep's rungs: two deep inside the broken zone, one on the shoulder
# and one clean, so a soil-independent ratio can be told from a soil-dependent
# one at each depth rather than only at the worst.
SOIL_HEIGHTS = (1.09e-4, 1e-3, 3e-3, 1e-2)

# The two ORTHOGONAL half-space axes, momwire#282 stage 2's shape.  The five
# golden soils confound σ with ε_r — `diel` is both the least conductive and
# the least dense, `sea` both the most — so neither can be read off them.
# `sigma` holds ε_r at poor soil's 5.0 and walks the conductivity through five
# decades; `epsr` holds σ small enough that the loss tangent stays ≲ 0.1
# everywhere on the sweep and walks the permittivity.
SIGMA_SWEEP = (0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
EPSR_SWEEP = (1.05, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 13.0, 20.0, 40.0, 81.0)
AXIS_HEIGHTS = (1.09e-4, 1e-3, 1e-2)


def medium_eps_t() -> complex:
    """``GN 0 ... 13. .005`` as the complex ε̃ the grid is keyed on."""
    omega = 2.0 * np.pi * FREQ_MHZ * 1e6
    return SOIL[0] - 1j * SOIL[1] / (omega * EPS0)


def node_points(h: float) -> np.ndarray:
    """Every mesh node of 0033 at radial height ``h``.

    Segment CENTRES would be the more faithful query set, but the pair
    geometry this mode is about — R₁ and θ of a source/observer pair — is set
    by the heights and the horizontal separation, and nodes and centres span
    the same ranges of both.  Nodes are what the deck states, so nodes are
    what is walked.
    """
    pts = [(0.0, 0.0, h + i * VERT_LEN / N_SEG) for i in range(N_SEG + 1)]
    for ux, uy in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)):
        for i in range(1, N_SEG + 1):
            s = i * RADIAL_LEN / N_SEG
            pts.append((ux * s, uy * s, h))
    return np.array(pts)


def pair_queries(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R₁, θ) for every ordered pair, in the grid's own convention.

    R₁ is source to the IMAGE of the observer — the distance the half-space
    surfaces are tabulated against — and θ is that ray's angle above the
    interface, so a pair of points at a common grazing height gives
    R₁ ≈ horizontal separation and θ ≈ (z_s + z_o)/R₁.
    """
    d = pts[:, None, :2] - pts[None, :, :2]
    rho = np.hypot(d[..., 0], d[..., 1])
    zsum = pts[:, None, 2] + pts[None, :, 2]
    r1 = np.hypot(rho, zsum)
    theta = np.arcsin(np.clip(zsum / np.maximum(r1, 1e-300), 0.0, 1.0))
    iu = np.triu_indices(len(pts), k=1)
    return r1[iu], theta[iu]


def grid_for(h: float):
    """The shipped grid this deck's solve would build, so the θ lattice read
    below is the real one rather than a re-derivation of the layout rules."""
    k2 = 2.0 * np.pi / WL
    pts = node_points(h)
    r1, _ = pair_queries(pts)
    return _sommerfeld.get_grid(
        medium_eps_t(), k2, float(r1.max()), omega=k2 * 299792458.0
    )


def theta_cells(grid) -> list[tuple[float, float, float, float]]:
    """(r0, r1, θ0, Δθ) of each region, in metres and degrees."""
    out = []
    for reg in grid._regions:
        r_lo = float(reg["r0"])
        r_hi = r_lo + float(reg["dr"]) * (int(reg["n_r"]) - 1)
        out.append(
            (
                r_lo,
                r_hi,
                float(np.degrees(reg["th0"])),
                float(np.degrees(reg["dth"])),
            )
        )
    return out


def mode_theta() -> list[dict]:
    print("0033's mesh — every pair's (R1, theta) against the shipped lattice")
    print(f"lambda = {WL:.2f} m, soil eps_r={SOIL[0]} sigma={SOIL[1]}\n")

    rows = []
    for hw in THETA_HEIGHTS:
        h = hw * WL
        pts = node_points(h)
        r1, theta = pair_queries(pts)
        th_deg = np.degrees(theta)
        grid = grid_for(h)
        cells = theta_cells(grid)
        # The coarsest first-cell width any query actually lands in: each
        # region owns an R1 band, so a query's cell is its band's.
        first = []
        for rr, tt in zip(r1, th_deg):
            widths = [
                dth for (r0, r1e, th0, dth) in cells if r0 - 1e-9 <= rr <= r1e + 1e-9
            ]
            first.append(min(widths) if widths else float("nan"))
        first = np.array(first)
        inside = np.isfinite(first) & (th_deg < first)
        print(f"h/lambda = {hw:<9.3g}  h = {h:.5g} m   ({len(r1)} pairs)")
        print("   grid theta cells (deg), by R1 band:")
        for r0, r1e, th0, dth in cells:
            print(
                f"      R1 {r0 / WL:>7.4f} .. {r1e / WL:<7.4f} lambda   "
                f"theta0 {th0:>4.1f}  dtheta {dth:.3f}"
            )
        print(
            f"   queried theta: min {th_deg.min():.4g} deg, "
            f"median {np.median(th_deg):.4g}, max {th_deg.max():.4g}"
        )
        print(
            f"   pairs inside their band's FIRST theta cell: "
            f"{inside.sum()} / {len(r1)}  ({100.0 * inside.mean():.1f} %)\n"
        )
        rows.append(
            dict(
                h_over_wl=hw,
                h=h,
                n_pairs=int(len(r1)),
                theta_min_deg=float(th_deg.min()),
                theta_median_deg=float(np.median(th_deg)),
                theta_max_deg=float(th_deg.max()),
                frac_in_first_cell=float(inside.mean()),
                cells=[list(c) for c in cells],
            )
        )
    return rows


class DirectGrid:
    """:class:`SommerfeldGrid`'s query contract, answered without a lattice.

    Lifted from ``probe_contact_direct_remainder.py`` — same contract, same
    memoization — and carrying the attributes the remainder path reads.  The
    cache is exact, not an approximation: a deck's obs/src halves repeat their
    (R₁, θ) bit-for-bit.
    """

    def __init__(self, eps_t, k2, r1_max, omega, mu, rtol):
        self.eps_t = complex(eps_t)
        self.k2 = float(k2)
        self.r1_max = float(r1_max)
        self.r_near = float(r1_max)
        self.omega = float(omega)
        self.mu = float(mu)
        self.rtol = float(rtol)
        self._cache: dict[tuple[float, float], tuple[complex, ...]] = {}
        self.calls = 0
        self.misses = 0

    def eval(self, R1, theta):
        r_b, th_b = np.broadcast_arrays(
            np.asarray(R1, dtype=float), np.asarray(theta, dtype=float)
        )
        shape = r_b.shape
        r_f = r_b.ravel()
        th_f = np.clip(th_b.ravel(), 0.0, 0.5 * np.pi)
        self.calls += r_f.size
        keys = [(float(a), float(b)) for a, b in zip(r_f, th_f)]
        fresh = sorted({kk for kk in keys if kk not in self._cache})
        if fresh:
            self.misses += len(fresh)
            surf = _sommerfeld.iv_surfaces_direct(
                self.eps_t,
                self.k2,
                np.array([kk[0] for kk in fresh]),
                np.array([kk[1] for kk in fresh]),
                rtol=self.rtol,
                omega=self.omega,
                mu=self.mu,
            )
            for i, kk in enumerate(fresh):
                self._cache[kk] = tuple(surf[s][i] for s in _sommerfeld._SURF_KEYS)
        out = np.empty((len(_sommerfeld._SURF_KEYS), r_f.size), dtype=np.complex128)
        for i, kk in enumerate(keys):
            out[:, i] = self._cache[kk]
        return {
            key: out[s].reshape(shape) for s, key in enumerate(_sommerfeld._SURF_KEYS)
        }


class MaskedAcc:
    """The accelerator module with only its GRID-consuming kernels hidden.

    Both remainder kernels are selected by ``hasattr``, so masking those two
    names routes the remainder — and only the remainder — through the numpy
    body that calls ``grid.eval``, leaving the rest of the fill accelerated.
    """

    _HIDDEN = ("sommerfeld_remainder_bspline_Q", "remainder_field_proj_batch")

    def __init__(self, acc):
        self._acc = acc

    def __getattr__(self, name):
        if name in MaskedAcc._HIDDEN:
            raise AttributeError(name)
        return getattr(self._acc, name)


def with_direct_grid(fn, rtol: float):
    """Run ``fn`` with every solver's Sommerfeld grid replaced by direct
    evaluation.

    ``_sommerfeld.get_grid`` is the single funnel — bspline reaches it through
    ``_somm_grid`` and razor through ``_potential_ground`` — so one patch
    covers both trunks whatever route the seam takes to build them.
    """
    from momwire import bspline

    # `_potential_ground` — razor's route — holds no `_acc` of its own: it
    # reaches the kernels through `_sommerfeld.remainder_field_proj`, so
    # masking `_sommerfeld`'s covers both trunks.  It still picks up the
    # patched grid, because `get_grid` is looked up on the module per call.
    saved_get = _sommerfeld.get_grid
    saved = {m: m._acc for m in (bspline, _sommerfeld)}
    grids: list[DirectGrid] = []

    def _direct(eps_t, k2, r1_max, omega, mu=_sommerfeld._MU0, cancel_flag=0):
        g = DirectGrid(eps_t, k2, r1_max, omega, mu, rtol)
        grids.append(g)
        return g

    _sommerfeld.get_grid = _direct
    for mod, acc in saved.items():
        if acc is not None:
            mod._acc = MaskedAcc(acc)
    try:
        return fn(), grids
    finally:
        _sommerfeld.get_grid = saved_get
        for mod, acc in saved.items():
            mod._acc = acc


def mode_direct(rtol: float, soil: tuple[float, float] = SOIL) -> list[dict]:
    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    print("0033 against the binary, shipped grid vs direct evaluation")
    print(f"direct rtol = {rtol:g}; lambda = {WL:.2f} m")
    print(f"soil: eps_r = {soil[0]}, sigma = {soil[1]}\n")
    hdr = (
        f"{'h/lambda':>10} {'trunk':>11} | {'nec5cl':>21} | "
        f"{'grid':>21} {'err%':>8} | {'direct':>21} {'err%':>8} | {'calls':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for hw in DIRECT_HEIGHTS:
        text = deck(hw * WL, soil=soil)
        z_ref = complex(eng.run_deck(text)[0][0][2])
        for trunk in TRUNKS:
            z_grid = momwire_z(text, trunk)
            (z_dir, grids) = with_direct_grid(lambda: momwire_z(text, trunk), rtol)
            calls = sum(g.misses for g in grids)
            e_g = 100.0 * abs(z_grid - z_ref) / abs(z_ref) if z_grid else float("nan")
            e_d = 100.0 * abs(z_dir - z_ref) / abs(z_ref) if z_dir else float("nan")
            print(
                f"{hw:>10.3g} {trunk:>11} | "
                f"{z_ref.real:>10.4f}{z_ref.imag:>+10.4f}j | "
                f"{z_grid.real:>10.4f}{z_grid.imag:>+10.4f}j {e_g:>8.2f} | "
                f"{z_dir.real:>10.4f}{z_dir.imag:>+10.4f}j {e_d:>8.2f} | {calls:>8}",
                flush=True,
            )
            rows.append(
                dict(
                    h_over_wl=hw,
                    trunk=trunk,
                    z_ref=[z_ref.real, z_ref.imag],
                    z_grid=[z_grid.real, z_grid.imag],
                    z_direct=[z_dir.real, z_dir.imag],
                    err_grid_pct=e_g,
                    err_direct_pct=e_d,
                    direct_evals=calls,
                )
            )
    return rows


def mode_reflcoef() -> list[dict]:
    """Does the error live in the half refl-coef and sommerfeld SHARE?

    The seam has no card for refl-coef and should not grow one — ``GN 0`` is
    a Sommerfeld ground and serving it as anything else would be a lie. So
    the ground kwargs are patched for the length of one solve, which holds
    the deck, the drive, the mesh and the whole route fixed and moves exactly
    one thing.
    """
    from momwire.eznec import _serve

    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    saved = _serve._ground_kwargs

    def _as_refl(deck_, medium):
        kw = saved(deck_, medium)
        if kw.get("ground_model") == "sommerfeld":
            kw = dict(kw, ground_model="refl-coef")
        return kw

    print("0033 against the binary — sommerfeld vs refl-coef, everything")
    print("else held (same deck, drive, mesh, seam route)\n")
    print("refl-coef is outside its own 0.1-0.5 lambda window down here, so")
    print("its absolute error proves nothing; whether it TRACKS does.\n")
    hdr = (
        f"{'h/lambda':>10} {'trunk':>11} | {'nec5cl':>21} | "
        f"{'sommerfeld':>21} {'err%':>8} | {'refl-coef':>21} {'err%':>8} | "
        f"{'|somm-refl|':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for hw in DIRECT_HEIGHTS:
        text = deck(hw * WL)
        z_ref = complex(eng.run_deck(text)[0][0][2])
        z_pec = complex(eng.run_deck(deck(hw * WL, pec=True))[0][0][2])
        for trunk in TRUNKS:
            z_s = momwire_z(text, trunk)
            _serve._ground_kwargs = _as_refl
            try:
                z_r = momwire_z(text, trunk)
            finally:
                _serve._ground_kwargs = saved
            if z_s is None or z_r is None:
                continue
            e_s = 100.0 * abs(z_s - z_ref) / abs(z_ref)
            e_r = 100.0 * abs(z_r - z_ref) / abs(z_ref)
            print(
                f"{hw:>10.3g} {trunk:>11} | "
                f"{z_ref.real:>10.4f}{z_ref.imag:>+10.4f}j | "
                f"{z_s.real:>10.4f}{z_s.imag:>+10.4f}j {e_s:>8.2f} | "
                f"{z_r.real:>10.4f}{z_r.imag:>+10.4f}j {e_r:>8.2f} | "
                f"{abs(z_s - z_r):>11.4f}",
                flush=True,
            )
            rows.append(
                dict(
                    h_over_wl=hw,
                    trunk=trunk,
                    z_ref=[z_ref.real, z_ref.imag],
                    z_pec_ref=[z_pec.real, z_pec.imag],
                    z_sommerfeld=[z_s.real, z_s.imag],
                    z_reflcoef=[z_r.real, z_r.imag],
                    err_sommerfeld_pct=e_s,
                    err_reflcoef_pct=e_r,
                    gap=abs(z_s - z_r),
                    d_true=[(z_ref - z_pec).real, (z_ref - z_pec).imag],
                    d_momwire=[(z_s - z_pec).real, (z_s - z_pec).imag],
                    overshoot=abs(z_s - z_pec) / abs(z_ref - z_pec),
                )
            )

    # The sharpest form of the question. razor-nec5 over `GN 1` reproduces the
    # binary to 0.00 % at every height (experiment 1), so the binary's own PEC
    # answer is a legitimate common baseline and the GROUND CORRECTION — what
    # the finite ground is worth on top of a perfect image — can be compared
    # directly instead of through two absolute impedances.
    print("\n\nthe ground correction itself: Z(GN 0) - Z(GN 1)\n")
    hdr2 = (
        f"{'h/lambda':>10} {'trunk':>11} | {'true (binary)':>21} | "
        f"{'momwire':>21} | {'overshoot':>9}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for row in rows:
        dt = complex(*row["d_true"])
        dm = complex(*row["d_momwire"])
        print(
            f"{row['h_over_wl']:>10.3g} {row['trunk']:>11} | "
            f"{dt.real:>10.4f}{dt.imag:>+10.4f}j | "
            f"{dm.real:>10.4f}{dm.imag:>+10.4f}j | {row['overshoot']:>8.2f}x"
        )
    return rows


def mode_soil() -> list[dict]:
    """Is the overshoot a model-INDEPENDENT scale factor?

    The reflcoef mode measured the ground correction Z(``GN 0``) − Z(``GN 1``)
    overshooting by up to ~3x on average soil.  A scale factor that is the
    same across five very different half-spaces is a scaling defect; one that
    tracks ε̃ is a physics term with the wrong weight.  That is the question
    #624's spike asked from the contact side and answered "model-independent"
    — here it is asked at grazing, where the symptom is 100x bigger.

    The COMPLEX ratio is what is read, not its magnitude.  A pure scale factor
    is real: if Δ_momwire / Δ_true sits at (say) 2.6 + 0j on every soil, the
    correction is being multiplied by a constant.  If its phase wanders with
    the soil, it is not one number and row-halving is the wrong suspect.

    razor-nec5 only.  bspline's ~5 % basis error rides on the same correction
    and contaminates the ratio (its 1e-2 λ row came out 4.77x on a 1.7 Ω
    correction), and razor's column is exact over ``GN 1`` at every height,
    which is what makes the PEC baseline a shared one.
    """
    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole
    from golden_contact_nec5 import GROUND_EPS

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    print("the ground correction Z(GN 0) - Z(GN 1) across five half-spaces")
    print("razor-nec5; the PEC baseline is soil-independent and exact\n")
    print("a model-INDEPENDENT scale factor is a REAL ratio, constant in soil;")
    print("a mis-weighted physics term tracks eps~ and wanders in phase.\n")

    rows = []
    for hw in SOIL_HEIGHTS:
        z_pec = complex(eng.run_deck(deck(hw * WL, pec=True))[0][0][2])
        print(f"h/lambda = {hw:<9.3g}   Z(GN 1) = {z_pec.real:.4f}{z_pec.imag:+.4f}j")
        hdr = (
            f"   {'soil':>6} {'eps_r':>7} {'sigma':>8} | {'true':>20} | "
            f"{'momwire':>20} | {'ratio':>18} {'|.|':>7} {'arg':>8}"
        )
        print(hdr)
        print("   " + "-" * (len(hdr) - 3))
        for name, soil in GROUND_EPS.items():
            text = deck(hw * WL, soil=soil)
            z_ref = complex(eng.run_deck(text)[0][0][2])
            z_mw = momwire_z(text, "razor-nec5")
            if z_mw is None:
                continue
            d_true = z_ref - z_pec
            d_mw = z_mw - z_pec
            ratio = d_mw / d_true if abs(d_true) > 0 else complex("nan")
            print(
                f"   {name:>6} {soil[0]:>7.1f} {soil[1]:>8.5g} | "
                f"{d_true.real:>9.4f}{d_true.imag:>+9.4f}j | "
                f"{d_mw.real:>9.4f}{d_mw.imag:>+9.4f}j | "
                f"{ratio.real:>8.3f}{ratio.imag:>+8.3f}j "
                f"{abs(ratio):>7.3f} {np.degrees(np.angle(ratio)):>7.1f}d",
                flush=True,
            )
            rows.append(
                dict(
                    h_over_wl=hw,
                    soil=name,
                    eps_r=soil[0],
                    sigma=soil[1],
                    z_pec=[z_pec.real, z_pec.imag],
                    z_ref=[z_ref.real, z_ref.imag],
                    z_mw=[z_mw.real, z_mw.imag],
                    d_true=[d_true.real, d_true.imag],
                    d_momwire=[d_mw.real, d_mw.imag],
                    ratio=[ratio.real, ratio.imag],
                    ratio_mag=abs(ratio),
                    ratio_arg_deg=float(np.degrees(np.angle(ratio))),
                )
            )
        print()
    return rows


def mode_axis(
    which: str,
    *,
    values: tuple[float, ...] | None = None,
    heights: tuple[float, ...] | None = None,
    trunks: tuple[str, ...] = ("razor-nec5",),
    hold_eps: float = 5.0,
    radial_len: float = RADIAL_LEN,
) -> list[dict]:
    """The overshoot along ONE half-space axis, with the other held.

    The soil mode found the ground correction overshooting 2.6-4.4x on the
    four conducting soils and 61x on the lossless dielectric, which resembles
    momwire#282 stage 2's contact discrepancy — single-peaked in ε_r, falling
    monotonically in σ.  But the five golden soils confound the two axes
    (``diel`` is the least conductive AND the least dense; ``sea`` is both the
    most), so the resemblance cannot be read off them.  These are #282's own
    two orthogonal sweeps, asked at grazing.

    A LOSS story has to vanish with σ.  If the overshoot is largest at σ = 0
    and falls monotonically as the ground becomes conductive, the resemblance
    is real and #282's stage-2 record is machinery rather than analogy.  If it
    is flat in σ and structured in ε_r alone, the two findings only look
    alike.

    The loss tangent is printed because it, not σ, is what says whether a
    half-space is "lossy" at a given frequency: at 1.832 MHz average soil's
    tan δ is 3.8 and the lossless dielectric's is 0.039, which is the gap the
    soil mode was actually straddling.
    """
    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    omega = 2.0 * np.pi * FREQ_MHZ * 1e6
    if which == "sigma":
        cases = [(hold_eps, s) for s in (values or SIGMA_SWEEP)]
        print(f"sigma axis at eps_r = {hold_eps} — a LOSS term must vanish with sigma")
    else:
        cases = [(e, 1e-5) for e in (values or EPSR_SWEEP)]
        print("eps_r axis at sigma = 1e-5 — loss tangent <= 0.1 throughout")
    print(f"ground correction Z(GN 0) - Z(GN 1), {', '.join(trunks)}\n")

    rows = []
    for hw in heights or AXIS_HEIGHTS:
        z_pec = complex(
            eng.run_deck(deck(hw * WL, pec=True, radial_len=radial_len))[0][0][2]
        )
        print(
            f"h/lambda = {hw:<9.3g}  radial {radial_len:g} m  "
            f"Z(GN 1) = {z_pec.real:.4f}{z_pec.imag:+.4f}j"
        )
        hdr = (
            f"   {'eps_r':>7} {'sigma':>9} {'tan_d':>9} {'trunk':>11} | "
            f"{'true':>20} | {'momwire':>20} | {'|ratio|':>9} {'arg':>7} | "
            f"{'|err|':>10}"
        )
        print(hdr)
        print("   " + "-" * (len(hdr) - 3))
        for eps_r, sigma in cases:
            text = deck(hw * WL, soil=(eps_r, sigma), radial_len=radial_len)
            z_ref = complex(eng.run_deck(text)[0][0][2])
            d_true = z_ref - z_pec
            tan_d = sigma / (omega * EPS0 * eps_r)
            for trunk in trunks:
                z_mw = momwire_z(text, trunk)
                if z_mw is None:
                    continue
                d_mw = z_mw - z_pec
                ratio = d_mw / d_true if abs(d_true) > 0 else complex("nan")
                print(
                    f"   {eps_r:>7.2f} {sigma:>9.3g} {tan_d:>9.3g} {trunk:>11} | "
                    f"{d_true.real:>9.4f}{d_true.imag:>+9.4f}j | "
                    f"{d_mw.real:>9.4f}{d_mw.imag:>+9.4f}j | "
                    f"{abs(ratio):>9.3f} {np.degrees(np.angle(ratio)):>6.1f}d | "
                    f"{abs(d_mw - d_true):>10.4f}",
                    flush=True,
                )
                rows.append(
                    dict(
                        axis=which,
                        h_over_wl=hw,
                        eps_r=eps_r,
                        sigma=sigma,
                        tan_delta=tan_d,
                        trunk=trunk,
                        radial_len=radial_len,
                        z_pec=[z_pec.real, z_pec.imag],
                        d_true=[d_true.real, d_true.imag],
                        d_momwire=[d_mw.real, d_mw.imag],
                        ratio_mag=abs(ratio),
                        ratio_arg_deg=float(np.degrees(np.angle(ratio))),
                        abs_err=abs(d_mw - d_true),
                    )
                )
        print()
    return rows


def mode_cond(
    values: tuple[float, ...] | None,
    heights: tuple[float, ...] | None,
    radial_len: float,
) -> list[dict]:
    """Is the eps_r pole a SINGULAR MATRIX?

    The radial-length scan moved the pole (39.6 m -> eps_r ~ 3.0, 60 m ->
    ~5.0) and at 60 m the solved impedance reaches 3600 + 2678j, which is an
    ANTI-RESONANCE — an open circuit where the antenna has no business having
    one.  A moving, geometry-dependent anti-resonance in one basis only is
    what a spurious internal resonance looks like: the assembled operator
    losing rank at a particular (geometry, eps~) pair.

    So this reads the conditioning directly, and it reads it at the SOLVE
    rather than at any one assembly method.  Hooking ``_assemble_Z`` per class
    was tried first and is wrong twice over: razor reaches its solve through
    more than one assembly path so the hook never fired, and bspline's
    ``_assemble_Z`` returns a partial block whose conditioning came out
    identical (522.6) at every ε_r, which is the tell that it is not the
    matrix being inverted.  ``scipy.linalg.solve`` is where the real operator
    passes, whatever route built it, so that is what is wrapped — and the
    worst-conditioned call of each solve is the one reported.

    bspline is carried as the control because it has no pole across this
    window — if its conditioning is flat where razor's spikes, the finding is
    razor's operator and not the physics of a half-space at eps~ ~ 3.
    """
    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    import scipy.linalg

    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    seen: dict[str, list] = {}
    current = {"trunk": None}
    orig_solve = scipy.linalg.solve

    def patched_solve(a, b, *args, **kw):
        arr = np.asarray(a)
        if current["trunk"] and arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            sv = np.linalg.svd(arr, compute_uv=False)
            seen.setdefault(current["trunk"], []).append(
                (float(sv[0] / sv[-1]), float(sv[-1]), int(sv.size))
            )
        return orig_solve(a, b, *args, **kw)

    scipy.linalg.solve = patched_solve
    saved = {"scipy": orig_solve}

    print("conditioning of the assembled operator across the eps_r pole")
    print(f"radial {radial_len:g} m; cond and sigma_min of what the solve inverts\n")
    rows = []
    try:
        for hw in heights or (1.09e-4,):
            z_pec = complex(
                eng.run_deck(deck(hw * WL, pec=True, radial_len=radial_len))[0][0][2]
            )
            print(
                f"h/lambda = {hw:<9.3g}   Z(GN 1) = {z_pec.real:.4f}{z_pec.imag:+.4f}j"
            )
            hdr = (
                f"   {'eps_r':>7} {'trunk':>11} {'N':>4} | {'cond':>12} "
                f"{'sigma_min':>12} | {'|err| ohm':>11}"
            )
            print(hdr)
            print("   " + "-" * (len(hdr) - 3))
            for eps_r in values or EPSR_SWEEP:
                text = deck(hw * WL, soil=(eps_r, 1e-5), radial_len=radial_len)
                z_ref = complex(eng.run_deck(text)[0][0][2])
                for trunk in ("razor-nec5", "bspline"):
                    seen.pop(trunk, None)
                    current["trunk"] = trunk
                    z_mw = momwire_z(text, trunk)
                    current["trunk"] = None
                    if z_mw is None or not seen.get(trunk):
                        continue
                    cond, smin, n = max(seen[trunk], key=lambda t: t[0])
                    err = abs((z_mw - z_pec) - (z_ref - z_pec))
                    print(
                        f"   {eps_r:>7.2f} {trunk:>11} {n:>4} | {cond:>12.4g} "
                        f"{smin:>12.4g} | {err:>11.4f}",
                        flush=True,
                    )
                    rows.append(
                        dict(
                            h_over_wl=hw,
                            eps_r=eps_r,
                            trunk=trunk,
                            radial_len=radial_len,
                            n_unknowns=n,
                            cond=cond,
                            sigma_min=smin,
                            abs_err=err,
                        )
                    )
            print()
    finally:
        scipy.linalg.solve = saved["scipy"]
    return rows


def wire_deck(h, *, n_seg=5, pec=False, soil=SOIL, length=RADIAL_LEN):
    """ONE horizontal wire at height ``h``, driven at an interior node.

    The smallest thing that could carry the defect.  Experiment 3 said the
    trigger is a horizontal wire in the grazing zone and not the junction or
    the feedpoint, so a single radial with no vertical, no junction and no
    screen should either reproduce or exonerate that reading.

    Driven at node 2 rather than an end, because a free wire END carries no
    basis function and the binary refuses it by name (``SORVT1: ERROR -
    Voltage source specified where there is no basis function``).
    """
    gn = "GN 1,0,0,0" if pec else f"GN 0,0,0,0,{_num(soil[0])},{_num(soil[1])},1.,0."
    return (
        "CM momwire#510 — one horizontal wire, grazing\n"
        f"CM h = {h:.6g} m = {h / WL:.4g} wavelengths, L = {length:g} m\nCE\n"
        f"GW 1,{n_seg},0.,0.,{_num(h)},{_num(length)},0.,{_num(h)},{_num(RADIUS)}\n"
        "GE 1,-1\n"
        f"FR 0,1,0,0,{_num(FREQ_MHZ)}\n" + gn + "\nEX 0,1,2,0,1.414214,0.\nXQ 0\nEN\n"
    )


def mode_wire(values, heights) -> list[dict]:
    """Does ONE horizontal wire reproduce the defect?

    Everything so far has been measured on 0033 — five wires, a junction, a
    screen.  If a single grazing horizontal wire carries the same signature,
    the reproducer drops from 24 unknowns to 6, the remaining diagnosis
    becomes entry-by-entry tractable, and any fix gets a unit test that runs
    in milliseconds.  If it does NOT, the screen or the junction is part of
    the mechanism after all and experiment 3's reading needs qualifying.
    """
    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    print("ONE horizontal wire, no junction, no screen, no vertical")
    print(f"L = {RADIAL_LEN} m, a = {RADIUS} m, driven at node 2\n")
    rows = []
    for hw in heights or (1.09e-4, 1e-3, 1e-2):
        z_pec = complex(eng.run_deck(wire_deck(hw * WL, pec=True))[0][0][2])
        print(f"h/lambda = {hw:<9.3g}   Z(GN 1) = {z_pec.real:.4f}{z_pec.imag:+.4f}j")
        hdr = (
            f"   {'eps_r':>7} {'trunk':>11} | {'true':>20} | {'momwire':>20} | "
            f"{'|ratio|':>9} {'arg':>7} | {'|err|':>10}"
        )
        print(hdr)
        print("   " + "-" * (len(hdr) - 3))
        for eps_r in values or (2.5, 3.0, 3.1, 5.0, 13.0):
            text = wire_deck(hw * WL, soil=(eps_r, 1e-5))
            z_ref = complex(eng.run_deck(text)[0][0][2])
            d_true = z_ref - z_pec
            for trunk in TRUNKS:
                z_mw = momwire_z(text, trunk)
                if z_mw is None:
                    continue
                d_mw = z_mw - z_pec
                ratio = d_mw / d_true if abs(d_true) > 0 else complex("nan")
                print(
                    f"   {eps_r:>7.2f} {trunk:>11} | "
                    f"{d_true.real:>9.4f}{d_true.imag:>+9.4f}j | "
                    f"{d_mw.real:>9.4f}{d_mw.imag:>+9.4f}j | "
                    f"{abs(ratio):>9.3f} {np.degrees(np.angle(ratio)):>6.1f}d | "
                    f"{abs(d_mw - d_true):>10.4f}",
                    flush=True,
                )
                rows.append(
                    dict(
                        h_over_wl=hw,
                        eps_r=eps_r,
                        trunk=trunk,
                        d_true=[d_true.real, d_true.imag],
                        d_momwire=[d_mw.real, d_mw.imag],
                        ratio_mag=abs(ratio),
                        abs_err=abs(d_mw - d_true),
                    )
                )
        print()
    return rows


def _capture_Z(text, trunk):
    """The matrix ``scipy.linalg.solve`` actually inverts for this deck.

    Wrapping the solve rather than any assembly method, for the reason
    `mode_cond` records: razor reaches its solve by more than one assembly
    path and bspline's `_assemble_Z` returns a partial block.
    """
    import scipy.linalg

    grabbed = []
    orig = scipy.linalg.solve

    def patched(a, b, *args, **kw):
        arr = np.asarray(a)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            grabbed.append(arr.copy())
        return orig(a, b, *args, **kw)

    scipy.linalg.solve = patched
    try:
        momwire_z(text, trunk)
    finally:
        scipy.linalg.solve = orig
    return max(grabbed, key=lambda m: m.shape[0]) if grabbed else None


def mode_matrix(heights) -> list[dict]:
    """Read the ground-correction matrix itself, on the one-wire reproducer.

    Two checks, both self-contained — no binary, no reference implementation,
    which matters because every external reference this arc has (the licensed
    engine, the PEC baseline) gives an IMPEDANCE and not a matrix.

    RECIPROCITY.  A reciprocal medium gives a symmetric operator.  razor's
    testing is razor-blade rather than Galerkin so its Z need not be symmetric
    by construction, which is exactly why the PEC column is carried: whatever
    asymmetry the formulation has is visible there, at every height, and what
    matters is whether the FINITE-ground column departs from it as h -> 0.

    LOCALISATION.  |dZ_ij| binned by |i-j| says which pairs carry the ground
    correction.  For a horizontal wire over a half-space the correction should
    fall off with separation; a correction that is flat or rising in |i-j| is
    a coupling that does not decay, which is what an error in the grazing
    limit would look like.
    """
    # The quasi-static image coefficient. As h -> 0 the near-diagonal ground
    # correction is the INCOMPLETE cancellation of an antiparallel image: PEC
    # images a horizontal current at -1 exactly, a half-space at -Gamma, so
    # dZ is proportional to (1 - Gamma) times whatever the PEC image itself
    # contributes. In the near field that Gamma is the static (eps~-1)/(eps~+1),
    # so the predicted ratio is 2/(eps~+1) -- 0.0392 for average soil at
    # 1.832 MHz. The PEC image term is measurable without any reference:
    # Z(GN 1) - Z(GN -1), momwire's own free-space and perfect-image
    # machinery, both of which razor reproduces exactly.
    eps_t = medium_eps_t()
    predicted = abs(2.0 / (eps_t + 1.0))

    print("the ground-correction matrix on the one-wire reproducer")
    print(f"L = {RADIAL_LEN} m, 5 segments, driven at node 2")
    print(f"eps~ = {eps_t:.4g}; quasi-static |1 - Gamma| = {predicted:.5f}\n")
    rows = []
    for hw in heights or (1e-2, 1e-3, 1.09e-4):
        print(f"h/lambda = {hw:.3g}")
        for trunk in TRUNKS:
            free = wire_deck(hw * WL).replace(
                f"GN 0,0,0,0,{_num(13.0)},{_num(0.005)},1.,0.", "GN -1"
            )
            z_f = _capture_Z(free, trunk)
            z_p = _capture_Z(wire_deck(hw * WL, pec=True), trunk)
            z_g = _capture_Z(wire_deck(hw * WL, soil=(13.0, 0.005)), trunk)
            if z_p is None or z_g is None or z_p.shape != z_g.shape:
                print(f"   {trunk}: no comparable matrix")
                continue
            dz = z_g - z_p

            def asym(m):
                return float(np.linalg.norm(m - m.T) / max(np.linalg.norm(m), 1e-300))

            n = z_p.shape[0]
            prof = []
            for d in range(n):
                idx = [(i, i + d) for i in range(n - d)]
                prof.append(
                    float(np.mean([abs(dz[i, j]) for i, j in idx])) if idx else 0.0
                )
            print(
                f"   {trunk:>11} N={n:<3} "
                f"asym(PEC) {asym(z_p):.3e}  asym(GN0) {asym(z_g):.3e}  "
                f"asym(dZ) {asym(dz):.3e}  |dZ|/|Z_pec| "
                f"{np.linalg.norm(dz) / np.linalg.norm(z_p):.4f}"
            )
            print(
                "                 |dZ| by |i-j|: " + "  ".join(f"{v:.3g}" for v in prof)
            )
            rows_extra = None
            if z_f is not None and z_f.shape == z_p.shape:
                img = z_p - z_f  # what the PERFECT image alone contributes
                band = [
                    (
                        float(np.mean([abs(dz[i, i + d]) for i in range(n - d)])),
                        float(np.mean([abs(img[i, i + d]) for i in range(n - d)])),
                    )
                    for d in range(min(3, n))
                ]
                rows_extra = [(a / b if b else None) for a, b in band]
                print(
                    "                 |dZ|/|PEC image|: "
                    + "  ".join("nan" if r is None else f"{r:.4f}" for r in rows_extra)
                    + f"   (quasi-static predicts {predicted:.4f})"
                )
            rows.append(
                dict(
                    h_over_wl=hw,
                    trunk=trunk,
                    n=n,
                    asym_pec=asym(z_p),
                    asym_gn0=asym(z_g),
                    asym_dz=asym(dz),
                    dz_rel=float(np.linalg.norm(dz) / np.linalg.norm(z_p)),
                    dz_profile=prof,
                    dz_over_image=rows_extra,
                    quasistatic_predicted=predicted,
                )
            )
        print()
    return rows


def mode_nqp(values, heights) -> list[dict]:
    """Is the near-diagonal band an UNDER-RESOLVED source quadrature?

    Reading razor's assembly: the remainder Q rides the T1 window and is
    integrated over each source segment with ``n_qp_sommerfeld`` Gauss points,
    default 3.  For a grazing horizontal pair the Q integrand has a spike of
    width ~h where the observer sits over the source's image — h = 1.78 cm in
    a 7.92 m segment, h/delta = 0.0022.  Three Gauss points cannot resolve
    that, and the spike sharpens as h -> 0, which is exactly the band and
    exactly the height dependence ``--mode matrix`` measured.

    It is consistent with every earlier elimination, which is what makes it
    worth testing rather than merely plausible: the direct-grid bypass changed
    how each quadrature point's SURFACE is evaluated, not how many points
    there are, so exonerating the surfaces says nothing about this.

    momwire#282's probe raised n_qp 3 -> 12 at CONTACT and moved the answer
    0.03 ohm, which is why this was not tried sooner — but contact is a
    vertical wire, where the image is collinear and there is no spike.

    If the near-diagonal ratio converges toward bspline's ~0.05-0.07 as n_qp
    rises, the defect is the quadrature order and the fix is to key it to
    h/delta.  If it is flat, the order is innocent and the formulation is next.
    """
    from momwire import RazorSolver

    orig_init = RazorSolver.__init__
    forced = {"n": None}

    def patched_init(self, *a, **kw):
        if forced["n"] is not None:
            kw["n_qp_sommerfeld"] = forced["n"]
        return orig_init(self, *a, **kw)

    RazorSolver.__init__ = patched_init

    eps_t = medium_eps_t()
    predicted = abs(2.0 / (eps_t + 1.0))
    print("razor's Q source quadrature order, on the one-wire reproducer")
    print(f"quasi-static limit |2/(eps~+1)| = {predicted:.5f}")
    print("bspline's same band sits at ~0.05 / 0.074 at every height\n")

    rows = []
    try:
        for hw in heights or (1e-3, 1.09e-4):
            print(f"h/lambda = {hw:.3g}   (h/delta = {hw * WL / (RADIAL_LEN / 5):.4g})")
            hdr = f"   {'n_qp':>5} | {'|dZ|/|PEC image| band 0':>24} {'band 1':>9}"
            print(hdr)
            print("   " + "-" * (len(hdr) - 3))
            free = wire_deck(hw * WL).replace(
                f"GN 0,0,0,0,{_num(13.0)},{_num(0.005)},1.,0.", "GN -1"
            )
            for nq in values or (3, 6, 12, 24, 48, 96):
                forced["n"] = int(nq)
                z_f = _capture_Z(free, "razor-nec5")
                z_p = _capture_Z(wire_deck(hw * WL, pec=True), "razor-nec5")
                z_g = _capture_Z(wire_deck(hw * WL, soil=(13.0, 0.005)), "razor-nec5")
                forced["n"] = None
                if z_f is None or z_p is None or z_g is None:
                    continue
                dz, img, n = z_g - z_p, z_p - z_f, z_p.shape[0]
                band = [
                    float(np.mean([abs(dz[i, i + d]) for i in range(n - d)]))
                    / max(
                        float(np.mean([abs(img[i, i + d]) for i in range(n - d)])),
                        1e-300,
                    )
                    for d in (0, 1)
                ]
                print(f"   {nq:>5} | {band[0]:>24.4f} {band[1]:>9.4f}")
                rows.append(
                    dict(h_over_wl=hw, n_qp=int(nq), band0=band[0], band1=band[1])
                )
            print()
    finally:
        RazorSolver.__init__ = orig_init
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=(
            "theta",
            "direct",
            "reflcoef",
            "soil",
            "sigma",
            "epsr",
            "cond",
            "wire",
            "matrix",
            "nqp",
        ),
        default="theta",
    )
    p.add_argument("--rtol", type=float, default=1e-11)
    # `--mode direct`'s first run was on average soil alone, and the eps_r
    # axis then found a 2480 ohm peak at eps_r = 3 with no loss in it. An
    # exoneration measured at one half-space does not cover that one, so the
    # soil is settable and the peak gets asked the same question.
    p.add_argument("--eps-r", type=float, default=None)
    p.add_argument("--sigma", type=float, default=1e-5)
    # Axis-mode overrides, for resolving a feature the default lattice only
    # samples: the eps_r sweep found a 2480 ohm spike between its 2.5 and 4.0
    # rungs and how SHARP that spike is decides whether it reads as a pole.
    p.add_argument("--values", type=float, nargs="+", default=None)
    p.add_argument("--heights", type=float, nargs="+", default=None)
    p.add_argument("--trunks", nargs="+", default=["razor-nec5"])
    # Which eps_r the sigma axis holds. Default 5.0 is poor soil's, which sits
    # on the SHOULDER of razor's eps_r ~ 3.1 spike -- so a second sigma sweep
    # well clear of it is what separates "loss damps a pole" from "loss damps
    # the shared grazing error".
    p.add_argument("--hold-eps", type=float, default=5.0)
    # Does the eps_r ~ 3.1 pole MOVE with geometry? An algebraic singularity
    # in a coefficient sits at one eps_r whatever the antenna is; a spurious
    # guided resonance tracks the radial length. Accepts several lengths and
    # runs the whole eps_r scan at each.
    p.add_argument("--radial-lens", type=float, nargs="+", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.mode == "theta":
        rows = mode_theta()
    elif args.mode == "direct":
        soil = SOIL if args.eps_r is None else (args.eps_r, args.sigma)
        rows = mode_direct(args.rtol, soil)
    elif args.mode == "reflcoef":
        rows = mode_reflcoef()
    elif args.mode == "soil":
        rows = mode_soil()
    elif args.mode == "nqp":
        rows = mode_nqp(
            tuple(args.values) if args.values else None,
            tuple(args.heights) if args.heights else None,
        )
    elif args.mode == "matrix":
        rows = mode_matrix(tuple(args.heights) if args.heights else None)
    elif args.mode == "wire":
        rows = mode_wire(
            tuple(args.values) if args.values else None,
            tuple(args.heights) if args.heights else None,
        )
    elif args.mode == "cond":
        rows = mode_cond(
            tuple(args.values) if args.values else None,
            tuple(args.heights) if args.heights else None,
            (args.radial_lens or [RADIAL_LEN])[0],
        )
    else:
        rows = []
        for rl in args.radial_lens or [RADIAL_LEN]:
            rows += mode_axis(
                args.mode,
                values=tuple(args.values) if args.values else None,
                hold_eps=args.hold_eps,
                heights=tuple(args.heights) if args.heights else None,
                trunks=tuple(args.trunks),
                radial_len=rl,
            )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
