"""The point-matched sinusoidal scheme at a bend (stevenmburns/momwire#421).

`SinusoidalSolver` sits ~12.6 Ω from `BSplineSolver`, `SinusoidalGalerkinSolver`
and `RazorSolver` on an inverted-V, in free space, at N=192. #421 filed that as
a suspected defect in our bend handling.

It is not ours. It is NEC-2's, reproduced faithfully — which is this solver's
job. Measured against `nec2c 1.3` on the same geometry, our point-matched row
tracks nec2c to 0.088-0.096 Ω at every included angle from 180° to 60°, the
same residual a STRAIGHT wire shows, while nec2c and the other three
formulations separate by the full 12.6 Ω. The gap is a property of the
scheme, not of the translation.

And it does not converge away. Under uniform refinement at 90°:

    N     nec2c    mw sin   sin-gal   bspline    razor
    24   -352.79  -352.89   -344.67   -344.11   -341.85
    96   -349.12  -349.21   -338.76   -338.83   -338.09
    192  -349.73  -349.81   -337.57   -337.63   -337.17
    384  -351.55  -351.62   -336.81   -336.85   -336.53
    768  -354.78  -354.83   -336.28   -336.31   -336.04

The three well-behaved formulations land within 0.27 Ω of each other at
N=768. The NEC-2 scheme turns around after N≈96 and walks the other way, and
our twin walks with it to 0.05-0.10 Ω the whole time.

That is the same discriminator #573 measured on the W7EL coupled loop: not
"the sinusoidal basis" and not "point testing" as such, but whether the
tested scalar-potential term telescopes. `SinusoidalGalerkinSolver` uses the
SAME basis on the SAME mesh and converges with everyone else; what differs is
the testing rule, and NEC-2's midpoint sampling of grad(phi) leaves a residual
at a kink exactly as it does around a loop.

These gates pin the property so a future reader does not re-file it as a bug,
and so a change that accidentally "fixed" it — which would mean the twin had
stopped being a twin — fails loudly. They are written WITHOUT nec2c so they
run everywhere; the nec2c column above is the evidence, not the gate.
"""

import numpy as np
import pytest

from momwire import (
    BSplineSolver,
    RazorSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
)

WL = 299792458.0 / 14.0e6
ARM = 0.19 * WL  # each arm, so 0.38 λ of wire in total
RADIUS = 1.0e-3


def _invvee(included_deg):
    """Three-anchor bent wire, apex up, arms of fixed length.

    Only the included angle changes, so arm length and total wire length are
    constant down the sweep and the bend is the only variable.
    """
    half = np.deg2rad(included_deg / 2.0)
    dx, dz = ARM * np.sin(half), -ARM * np.cos(half)
    apex = 0.3 * WL + abs(dz) / 2
    return np.array([[-dx, 0.0, apex + dz], [0.0, 0.0, apex], [dx, 0.0, apex + dz]])


def _z(cls, included_deg, n, **extra):
    # `SinusoidalGalerkinSolver` carries NEC's SEGMENT gap throughout this
    # module. Its rows are scored against `SinusoidalSolver` — the same basis
    # on the same mesh, differing only in the TESTING — and that reading is
    # only clean while the source is held: with a different feed model in the
    # sibling, "the gap is the bend" would be measuring the bend plus the
    # feed. `SinusoidalSolver` can carry no other source (the zero-width gap
    # has no collocation RHS, momwire#212), so the naming happens here, and it
    # happens at all because momwire#654 moved this class's default to the
    # point gap.
    if cls is SinusoidalGalerkinSolver:
        extra.setdefault("feed_model", "segment")
    z, _ = cls(
        wires=[_invvee(included_deg)],
        n_per_edge_per_wire=[[n // 2, n // 2]],
        wire_radius=RADIUS,
        wavelength=WL,
        **extra,
    ).compute_impedance()
    return complex(z)


def _siblings(included_deg, n):
    """The three formulations that agree with each other."""
    return {
        "bspline": _z(BSplineSolver, included_deg, n, degree=2),
        "sin-galerkin": _z(SinusoidalGalerkinSolver, included_deg, n),
        "razor": _z(RazorSolver, included_deg, n),
    }


def test_a_straight_wire_shows_no_gap():
    """The control. At 180° the same three-anchor wire is straight, and the
    point-matched row joins the other three — so whatever the gap is, it is
    about the BEND and not about this deck, this feed or this basis."""
    n = 96
    sin = _z(SinusoidalSolver, 180.0, n)
    sibs = _siblings(180.0, n)
    for name, z in sibs.items():
        assert abs(sin - z) < 0.5, (name, sin, z)


@pytest.mark.parametrize(
    "included,floor",
    [(120.0, 3.0), (90.0, 8.0), (60.0, 15.0)],
)
def test_the_gap_is_the_bend_and_grows_with_it(included, floor):
    """The gap scales with bend sharpness: measured 0.12 Ω at 180°, then
    1.18 / 4.50 / 10.46 / 19.31 Ω at 150 / 120 / 90 / 60°.

    Floors, not windows, because the point being pinned is that the gap is
    real and bend-driven — an upper bar would be pinning NEC-2's error to a
    precision nobody should depend on.
    """
    n = 96
    sin = _z(SinusoidalSolver, included, n)
    sibs = _siblings(included, n)
    gaps = {name: abs(sin - z) for name, z in sibs.items()}
    assert min(gaps.values()) > floor, gaps
    # ...and the three siblings still agree with EACH OTHER, so the outlier
    # is the point-matched row and not a three-way disagreement.
    vals = list(sibs.values())
    spread = max(abs(a - b) for a in vals for b in vals)
    assert spread < 1.5, sibs


def test_the_siblings_converge_and_the_nec2_scheme_does_not():
    """The heart of #421, and the reason it is not a bug.

    Refining makes the three well-behaved formulations agree ever more
    closely; it does NOT bring the point-matched row in. Measured at 90°, the
    sibling spread falls 2.26 → 0.27 Ω from N=24 to N=768 while the
    point-matched gap does not fall at all — nec2c's own value turns around
    after N≈96 and walks away, and ours walks with it.

    N=192 is the ceiling here to keep this off the slow lane; the turnaround
    is already visible by then in the nec2c column quoted in the module
    docstring.
    """
    gaps, spreads = [], []
    for n in (48, 96, 192):
        sin = _z(SinusoidalSolver, 90.0, n)
        sibs = _siblings(90.0, n)
        vals = list(sibs.values())
        spreads.append(max(abs(a - b) for a in vals for b in vals))
        gaps.append(min(abs(sin - z) for z in vals))

    # the siblings are converging on each other
    assert spreads[-1] < spreads[0], spreads
    assert spreads[-1] < 1.0, spreads
    # the point-matched row is not joining them
    assert min(gaps) > 8.0, gaps
    assert gaps[-1] >= gaps[0] - 1.0, gaps


def test_the_same_basis_under_galerkin_testing_is_clean():
    """The discriminator, stated as a gate (#573's result, at a kink).

    `SinusoidalGalerkinSolver` is the SAME three-term basis on the SAME mesh;
    only the testing rule differs. It sits with `bspline` and `razor`, not
    with `SinusoidalSolver` — so the bend gap cannot be blamed on the basis,
    and any future "fix" that moves the basis rather than the testing rule is
    aiming at the wrong thing.
    """
    n = 96
    sg = _z(SinusoidalGalerkinSolver, 90.0, n)
    bs = _z(BSplineSolver, 90.0, n, degree=2)
    sin = _z(SinusoidalSolver, 90.0, n)
    assert abs(sg - bs) < 1.0, (sg, bs)
    assert abs(sg - sin) > 8.0, (sg, sin)


# ----------------------------------------------------------------------
# The refinement audit — why the gap is not a meshing or feed artefact
# ----------------------------------------------------------------------
# A cross-solver gap that grows with N is exactly the shape a REFINEMENT
# artefact takes, so these two controls exist before the conclusion does.
# Both were run at review, and both clear it.
#
# The mesh itself is clean: at every N both edges get exactly n//2 segments
# of identical length (4.0686 m), dmin == dmax, nothing stuck at a coarser
# rung. nec2c meshed exactly the requested counts at every rung too (24, 48,
# 96, 192, 384, 768), feeding tag 1 segment n//2 throughout.
#
# The FEED, however, is genuinely not the same object across these solvers:
#
#   SinusoidalSolver   a segment delta gap, snapped to the segment whose
#                      CENTRE is nearest the requested arclength — so on an
#                      equal-armed vee it sits 0.5 segments from the apex and
#                      that distance HALVES with every refinement.
#   BSplineSolver      a point feed at the requested arclength exactly — on
#                      this deck the apex itself, fixed for every N.
#
# That difference cannot be removed (the point-matched model has no
# sub-segment feed), so it is controlled for instead.


def test_the_straight_wire_converges_for_every_formulation():
    """Control 1: the moving feed does not, by itself, cause divergence.

    The point-matched feed sits half a segment from the wire midpoint at
    EVERY N here too, so if that moving delta gap were what drives the bend
    result, it would drive one here. It does not — measured 0.311 → 0.005 Ω
    from N=24 to N=768, all four formulations landing on −319.0 Ω.
    """
    gaps = []
    for n in (24, 96, 384):
        sin = _z(SinusoidalSolver, 180.0, n)
        sg = _z(SinusoidalGalerkinSolver, 180.0, n)
        gaps.append(abs(sin - sg))
    assert gaps[-1] < gaps[0], gaps
    assert gaps[-1] < 0.1, gaps


def test_the_gap_survives_feeding_every_solver_at_the_same_point():
    """Control 2: it is not the feed POSITION either.

    Pin `feed_arclength` to the point-matched solver's own feed — the centre
    of the segment it would have chosen — so the siblings are driven exactly
    where it is driven, instead of at the apex. The gap is unmoved: measured
    8.2 / 10.5 / 14.8 Ω at N = 24 / 96 / 384, against 8.2 / 10.5 / 14.8 with
    the default feeds. Matching the feed point moves the siblings by ~0.2 Ω,
    not by 12.
    """
    n = 96
    s = SinusoidalSolver(
        wires=[_invvee(90.0)],
        n_per_edge_per_wire=[[n // 2, n // 2]],
        wire_radius=RADIUS,
        wavelength=WL,
    )
    g = s._build_geometry()
    h = np.asarray(g["seg_h"])
    fs = g["feed_seg"]
    arc = float(np.cumsum(h)[fs] - 0.5 * h[fs])

    sin = _z(SinusoidalSolver, 90.0, n, feed_arclength=arc)
    sg = _z(SinusoidalGalerkinSolver, 90.0, n, feed_arclength=arc)
    bs = _z(BSplineSolver, 90.0, n, degree=2, feed_arclength=arc)

    assert abs(sg - bs) < 1.5, (sg, bs)  # siblings still agree
    assert abs(sin - sg) > 8.0, (sin, sg)  # and the gap is still there
