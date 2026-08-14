"""RazorSolver vs actual NEC-5: pinning the twin relationship itself
(momwire#309, unit 4 / antennaknobs#896).

Every other razor test compares RazorSolver against something computed in
Python (`bench_tri_razor.py`'s own instrument, or `BSplineSolver`). This file
is the only one that compares it against real NEC-5 output, so it is the one
that would actually catch a change that silently breaks the twin property
(a kernel sign flip, a testing-path bug, a quadrature regression) while every
in-Python pin stays green because the bug is common to both sides.

`NEC5_LADDER` below is transcribed from a licensed NEC-5 binary
(LLNL-CODE-746721) run on the ByDipole1 wire IN FREE SPACE — length
10.18946 m, radius 1.0262e-3 m, 14 MHz, center-fed at the knot (EX at segment
N/2, end 2) — the same wire and feed convention `tests/test_razor.py` and
`scripts/bench_tri_razor.py` (antennaknobs repo) already use. Citation: NEC-5
(LLNL-CODE-746721), free-space ByDipole1 ladder, 2026-08-13.

**What the numbers actually look like.** RazorSolver tracks NEC-5's reactance
walk in SHAPE (both are O(1/N), same pair-walk ratios — test 2) and the two
share a Richardson limit (test 3), but they are not close pointwise at any N
in this ladder: at N=24 the gap is already (0.11, 1.33) Ω and only closes to
(0.010, 0.11) Ω by N=100. That is `bench_tri_razor.py`'s own finding — its
`razor` lane (this class) and its `nec5` lane never coincide, only converge
together — and it is why test 1 uses per-N tolerances that shrink with N
rather than one flat bound: the twin is a claim about the WALK, not about
matching NEC-5's printout to a fixed tolerance at any single N.

**The gap has a name now** (momwire#316 residue study): the pointwise
difference is a clean C/N² term produced by NEC-5 evaluating ∫A·dl with the
two-point trapezoid at the path-end centroids — every potential at element
centroids, the classic mixed-potential idiom. `nec5_quadrature=True` adopts
that rule, and test 5 pins the consequence: NEC-5's printout is matched at
EVERY rung up to a CONSTANT ≈ −0.004−0.037j Ω residual (an N-independent
kernel nuance, the last 0.04 Ω of the twin story).
"""

import numpy as np
import pytest

from momwire import RazorSolver
from momwire.bspline import BSplineSolver

# NEC-5, free-space ByDipole1 ladder (see module docstring for provenance).
NEC5_LADDER = {
    12: 65.559 - 42.679j,
    16: 66.298 - 38.036j,
    20: 66.681 - 35.631j,
    24: 66.911 - 34.179j,
    28: 67.063 - 33.212j,
    32: 67.171 - 32.523j,
    36: 67.251 - 32.006j,
    40: 67.312 - 31.605j,
    44: 67.361 - 31.283j,
    48: 67.401 - 31.019j,
    52: 67.434 - 30.799j,
    56: 67.462 - 30.612j,
    60: 67.486 - 30.45j,
    64: 67.507 - 30.31j,
    68: 67.525 - 30.186j,
    72: 67.541 - 30.077j,
    76: 67.555 - 29.979j,
    80: 67.568 - 29.891j,
    84: 67.579 - 29.811j,
    88: 67.59 - 29.738j,
    92: 67.599 - 29.672j,
    96: 67.608 - 29.611j,
    100: 67.616 - 29.554j,
}

# ByDipole1 in free space (same wire as tests/test_razor.py).
BD1_LEN = 10.18946
BD1_RAD = 0.0010262
BD1_WL = 299792458.0 / 14.0e6
BD1_WIRE = [np.array([[0.0, 0.0, 0.0], [0.0, BD1_LEN, 0.0]])]

# Every N any test below needs, solved once via the module-scoped fixture.
_RAZOR_NS = (12, 24, 48, 72, 96, 100)
_BSPLINE_NS = (12, 24, 48, 96)


def _razor_z(n):
    z, _ = RazorSolver(
        wires=BD1_WIRE, nsegs=n, wire_radius=BD1_RAD, wavelength=BD1_WL
    ).compute_impedance()
    return z


def _bspline_z(n):
    z, _ = BSplineSolver(
        wires=BD1_WIRE, nsegs=n, wire_radius=BD1_RAD, wavelength=BD1_WL, degree=1
    ).compute_impedance()
    return z


@pytest.fixture(scope="module")
def ladders():
    """Every RazorSolver / BSplineSolver(degree=1) point every test below
    needs, solved once and shared — nothing here is solved twice. Measured
    ~1.6 s total on this wire (the N=100 razor solve alone is ~0.6 s), well
    under the file's 10 s budget and under the 2 s `slow` threshold even for
    whichever test triggers the fixture first, so nothing in this file is
    marked `@pytest.mark.slow`.
    """
    return {
        "razor": {n: _razor_z(n) for n in _RAZOR_NS},
        "bspline": {n: _bspline_z(n) for n in _BSPLINE_NS},
    }


# --------------------------------------------------------------------------
# 1. pointwise tracking — the gap shrinks with N, it does not vanish
# --------------------------------------------------------------------------
def test_razor_tracks_nec5_from_n24_up(ladders):
    """RazorSolver vs NEC-5, component-wise, at N = 24, 48, 72, 100.

    Measured gaps (R, X) in Ω: N=24 (0.11, 1.33), N=48 (0.03, 0.35),
    N=72 (0.015, 0.17), N=100 (0.010, 0.11) — R is small everywhere on this
    ladder, X is where NEC-5's coarse-mesh excess actually shows up, and
    both shrink monotonically with N. The per-N bounds below shrink to match
    (with headroom), so this pins the WALK shape rather than a fixed
    tolerance a coarse rung could accidentally satisfy by coincidence.

    Rungs below N=24 are deliberately not gated here: NEC-5 carries extra
    coarse-mesh excess there (a secondary kernel/feed detail the twin does
    not chase — see the module docstring of `razor.py`), and the gap at
    N=12 is over 5 Ω on X, an order of magnitude past what these bounds
    would tolerate.
    """
    razor = ladders["razor"]
    tol = {
        24: (0.2, 1.5),
        48: (0.1, 0.45),
        72: (0.05, 0.22),
        100: (0.03, 0.14),
    }
    for n, (r_tol, x_tol) in tol.items():
        d = razor[n] - NEC5_LADDER[n]
        assert abs(d.real) <= r_tol, f"N={n}: R gap {d.real} vs tol {r_tol}"
        assert abs(d.imag) <= x_tol, f"N={n}: X gap {d.imag} vs tol {x_tol}"


# --------------------------------------------------------------------------
# 2. the pair-walk signature — both lanes are O(1/N), same ratio band
# --------------------------------------------------------------------------
def test_pair_walk_signature_matches_nec5(ladders):
    """X(2N) - X(N) for N = 12, 24, 48, on both lanes.

    Razor's own diffs: (+4.487, +2.182, +1.172) Ω — pinned against fresh
    solves so a formulation change shows up here even if test 1's absolute
    gate is loose enough to absorb it. NEC-5's constants give
    (+8.500, +3.160, +1.408) Ω — twice razor's coefficient at the coarsest
    pair, which is exactly the coarse-mesh excess test 1 declines to gate.

    Both series are still O(1/N): consecutive-pair ratios sit in the same
    [1.7, 2.8] band on both lanes (both a bit under the naive doubling of
    2.0, both above 1.0 which would mean "not shrinking"), even though
    NEC-5's own first ratio (2.69) sits well above razor's (2.06) — the
    coefficient differs, the O(1/N) shape does not.
    """
    razor = ladders["razor"]
    razor_diffs = [razor[2 * n].imag - razor[n].imag for n in (12, 24, 48)]
    nec5_diffs = [NEC5_LADDER[2 * n].imag - NEC5_LADDER[n].imag for n in (12, 24, 48)]

    razor_expected = [4.487, 2.182, 1.172]
    for d, e in zip(razor_diffs, razor_expected):
        assert abs(d - e) <= 0.05, f"razor diffs {razor_diffs} vs {razor_expected}"

    razor_ratios = [razor_diffs[i] / razor_diffs[i + 1] for i in range(2)]
    nec5_ratios = [nec5_diffs[i] / nec5_diffs[i + 1] for i in range(2)]
    assert all(1.7 <= r <= 2.8 for r in razor_ratios), razor_ratios
    assert all(1.7 <= r <= 2.8 for r in nec5_ratios), nec5_ratios


# --------------------------------------------------------------------------
# 3. Richardson limits — the two O(1/N) walks share a limit
# --------------------------------------------------------------------------
def test_richardson_limits_agree(ladders):
    """Extrapolate each lane's (48, 96) pair to its N -> infinity limit.

    For an O(1/N) walk, Z_inf = 2*Z(2N) - Z(N); at N=48 that removes each
    lane's first-order term. Razor and NEC-5 disagree by (0.11, 1.33) Ω at
    N=24 (test 1) but their extrapolated limits agree to (0.01, 0.12) Ω —
    comfortably inside the 0.25 Ω gate — which is the twin claim in its
    sharpest form: two different coefficients on the same 1/N walk, one
    shared limit.
    """
    razor = ladders["razor"]
    z_razor_inf = 2.0 * razor[96] - razor[48]
    z_nec5_inf = 2.0 * NEC5_LADDER[96] - NEC5_LADDER[48]
    d = z_razor_inf - z_nec5_inf
    assert abs(d.real) <= 0.25, f"R limits {z_razor_inf} vs {z_nec5_inf}"
    assert abs(d.imag) <= 0.25, f"X limits {z_razor_inf} vs {z_nec5_inf}"


# --------------------------------------------------------------------------
# 4. negative control — Galerkin testing on the SAME basis does not walk
# --------------------------------------------------------------------------
def test_galerkin_tent_does_not_walk_like_nec5(ladders):
    """BSplineSolver(degree=1): same tent basis as RazorSolver, Galerkin
    testing instead of razor-blade. Its X-pair diffs for N = 12, 24, 48
    are (2.402, 0.796, 0.357) Ω against razor's (4.487, 2.182, 1.172) Ω —
    every pair smaller, so the walk is not merely a tent-basis artifact.

    The N=12 -> 24 pair is excluded from the tighter "under half" bound:
    both low-order bases carry their own coarse-mesh transient there (the
    same N<24 caveat test 1 applies to razor vs NEC-5), so bspline's 2.402
    is only 54% of razor's 4.487 rather than comfortably under half. Past
    that shared transient (N=24 -> 48 and N=48 -> 96), Galerkin testing
    walks at well under half razor's rate — this is the negative control:
    the O(1/N) walk comes from razor-blade TESTING, not from the tent basis
    the two solvers share.
    """
    razor = ladders["razor"]
    bspline = ladders["bspline"]
    razor_diffs = [razor[2 * n].imag - razor[n].imag for n in (12, 24, 48)]
    bspline_diffs = [bspline[2 * n].imag - bspline[n].imag for n in (12, 24, 48)]

    for bd, rd in zip(bspline_diffs, razor_diffs):
        assert bd < rd, f"bspline {bspline_diffs} vs razor {razor_diffs}"

    for bd, rd in zip(bspline_diffs[1:], razor_diffs[1:]):
        assert bd < 0.5 * rd, f"bspline {bspline_diffs[1:]} vs razor {razor_diffs[1:]}"


# --------------------------------------------------------------------------
# 5. nec5_quadrature: the identified rule closes the gap to a constant
# --------------------------------------------------------------------------
def test_nec5_quadrature_matches_every_rung():
    """`nec5_quadrature=True` (momwire#316) swaps the converged path
    quadrature for NEC-5's identified two-point centroid trapezoid. The
    residual against the printouts must then be (a) small at EVERY rung —
    no shrinking-tolerance ladder needed — and (b) CONSTANT across rungs
    (measured −0.0036−0.0371j Ω, spread under 0.001 Ω): an N-independent
    kernel nuance, not discretization. Constancy is the sharp claim — a
    quadrature regression would reintroduce an N-dependent term long
    before it moved the mean."""
    gaps = []
    for n in (12, 24, 48, 72, 100):
        z, _ = RazorSolver(
            wires=BD1_WIRE,
            nsegs=n,
            wire_radius=BD1_RAD,
            wavelength=BD1_WL,
            nec5_quadrature=True,
        ).compute_impedance()
        gaps.append(NEC5_LADDER[n] - z)
    for n, g in zip((12, 24, 48, 72, 100), gaps):
        assert abs(g.real) <= 0.01, f"N={n}: R residual {g.real}"
        assert abs(g.imag) <= 0.05, f"N={n}: X residual {g.imag}"
    spread = max(abs(a - b) for a in gaps for b in gaps)
    assert spread <= 0.003, f"residual not constant: {gaps}"


def test_nec5_quadrature_split_identity_holds():
    """The mode's per-wing element-local weights (h_wing/2 at the wing
    centroid) must keep the junction machinery exact: the two-wire
    colinear split of ByDipole1 solves identically to the single wire."""
    kw = dict(wire_radius=BD1_RAD, wavelength=BD1_WL, nec5_quadrature=True)
    z_one, _ = RazorSolver(wires=BD1_WIRE, nsegs=20, **kw).compute_impedance()
    h = BD1_LEN / 20
    two = [
        np.array([[0.0, 0.0, 0.0], [0.0, 8 * h, 0.0]]),
        np.array([[0.0, 8 * h, 0.0], [0.0, BD1_LEN, 0.0]]),
    ]
    z_two, _ = RazorSolver(
        wires=two,
        n_per_edge_per_wire=[8, 12],
        feeds=[(1, BD1_LEN / 2 - 8 * h, 1.0)],
        **kw,
    ).compute_impedance()
    assert abs(z_two - z_one) / abs(z_one) < 1e-9, f"{z_two} vs {z_one}"


def test_nec5_quadrature_default_off():
    """The default stays the converged quadrature — the unmodified pins in
    tests/test_razor.py are the real guard; this just pins the flag."""
    s = RazorSolver(wires=BD1_WIRE, nsegs=12, wavelength=BD1_WL)
    assert s.nec5_quadrature is False
