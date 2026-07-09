"""TriangularSolver ≡ BSplineSolver(degree=1) transfer-of-trust pins.

TriangularSolver is retirement-bound (docs/triangular-retirement-plan.md).
These tests pin the fact that makes the retirement safe: the two solvers
are the SAME numerical scheme — same tent basis, same quadratures, same
assembly — to solve-order roundoff, whenever the delta-gap feed lands on
a mesh knot. That holds automatically at even per-wire segment counts
(the parity antennaknobs enforces for d=1), at kink knots, and whenever
`feed_arclength` names a knot.

The ONE behavioral difference is feed placement when the requested
arclength falls between knots (odd nsegs, midpoint feed): triangular
snaps to the nearest interior knot, bspline excites the exact arclength
by splitting the delta-gap across the two straddling tents. That is a
convention difference, not a quadrature difference — pinned below by
showing bspline reproduces triangular exactly when fed AT the snapped
knot.

After the retirement lands, the triangular halves of these tests are
deleted and the bspline d=1 values become pinned constants.
"""

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.triangular import TriangularSolver

from test_momwire import _fandipole_two_band_sim

# Solve-order roundoff on well-conditioned systems: measured ≤ 8e-11 Ω
# absolute on dipoles; give junction/multi-wire paths an order of margin.
RTOL = 1e-8


def _pair(wires, nsegs, tri_kwargs=None, bs_kwargs=None, **common):
    zt, _ = TriangularSolver(
        wires=wires, nsegs=nsegs, **(tri_kwargs or {}), **common
    ).compute_impedance()
    zb, _ = BSplineSolver(
        wires=wires, nsegs=nsegs, degree=1, **(bs_kwargs or {}), **common
    ).compute_impedance()
    return zt, zb


@pytest.mark.parametrize("half", [5.29, 2.65])  # resonant 0.481λ, short 0.24λ
@pytest.mark.parametrize("nsegs", [20, 40])
def test_d1_matches_triangular_dipole_even_nsegs(half, nsegs):
    wires = [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]
    zt, zb = _pair(wires, nsegs)
    assert abs(zt - zb) <= RTOL * abs(zt), f"tri={zt} d1={zb}"


def test_d1_matches_triangular_odd_nsegs_when_fed_at_snapped_knot():
    """Odd nsegs, midpoint feed: triangular snaps the feed to the nearest
    interior knot. Feeding bspline AT that knot reproduces triangular to
    roundoff — the entire odd-N disagreement is feed placement, nothing
    else (this is the corrected conclusion of the PR #101-era
    'genuinely different quadratures' investigation)."""
    half, nsegs = 2.65, 21
    wires = [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]
    h = 2 * half / nsegs
    knot_arc = round(half / h) * h  # arclength of the knot nearest midpoint
    zt, _ = TriangularSolver(wires=wires, nsegs=nsegs).compute_impedance()
    zb, _ = BSplineSolver(
        wires=wires, nsegs=nsegs, degree=1, feed_arclength=knot_arc
    ).compute_impedance()
    assert abs(zt - zb) <= RTOL * abs(zt), f"tri={zt} d1={zb}"
    # And the midpoint-fed bspline genuinely differs (the convention gap
    # is real at odd N — don't port odd-N triangular values verbatim).
    zb_mid, _ = BSplineSolver(wires=wires, nsegs=nsegs, degree=1).compute_impedance()
    assert abs(zt - zb_mid) > 1e3 * RTOL * abs(zt)


def test_d1_matches_triangular_two_wire_yagi():
    hd = 0.962 * 22 / 4
    refl = 1.05 * hd
    wires = [
        np.array([[0.0, 0.0, -hd], [0.0, 0.0, hd]]),
        np.array([[-2.2, 0.0, -refl], [-2.2, 0.0, refl]]),
    ]
    zt, zb = _pair(wires, 20)
    assert abs(zt - zb) <= RTOL * abs(zt), f"tri={zt} d1={zb}"


def test_d1_matches_triangular_v_dipole():
    # 30° V: two edges meeting at the feed kink — the feed knot is the
    # kink itself, on-knot for any per-edge count.
    hd = 0.962 * 22 / 4
    c30, s30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    wires = [
        np.array(
            [[-hd * c30, 0.0, hd * s30], [0.0, 0.0, 0.0], [hd * c30, 0.0, hd * s30]]
        )
    ]
    zt, zb = _pair(wires, 10)
    assert abs(zt - zb) <= RTOL * abs(zt), f"tri={zt} d1={zb}"


def test_d1_matches_triangular_pec_ground():
    # Horizontal dipole at moderate height over a PEC image plane.
    hd = 0.962 * 22 / 4
    wires = [np.array([[-hd, 0.0, 7.0], [hd, 0.0, 7.0]])]
    zt, zb = _pair(wires, 20, ground_z=0.0)
    assert abs(zt - zb) <= RTOL * abs(zt), f"tri={zt} d1={zb}"


def test_d1_matches_triangular_fandipole_k3_junctions():
    """K=3 junction directional bases + KCL Schur solve: same discrete
    system in both solvers (feed wire has a knot at the requested
    arclength, so the feed convention is moot here)."""
    wavelength = 299_792_458.0 / 14.3e6
    zt, _ = _fandipole_two_band_sim(20, wavelength).compute_impedance()
    zb, _ = _fandipole_two_band_sim(
        20, wavelength, solver_cls=BSplineSolver, degree=1
    ).compute_impedance()
    assert abs(zt - zb) <= 10 * RTOL * abs(zt), f"tri={zt} d1={zb}"


def test_d1_matches_triangular_multifeed_y_matrix():
    hd = 0.962 * 22 / 4
    wires = [
        np.array([[0.0, 0.0, -hd], [0.0, 0.0, hd]]),
        np.array([[3.0, 0.0, -hd], [3.0, 0.0, hd]]),
    ]
    feeds = [(0, None, 1.0), (1, None, 1.0 + 0.5j)]
    common = dict(wires=wires, nsegs=20, feeds=feeds)
    Yt = TriangularSolver(**common).compute_y_matrix()
    Yb = BSplineSolver(degree=1, **common).compute_y_matrix()
    assert np.allclose(Yt, Yb, rtol=RTOL, atol=0), f"tri={Yt} d1={Yb}"


def test_d1_matches_triangular_swept():
    hd = 0.962 * 22 / 4
    wires = [np.array([[0.0, 0.0, -hd], [0.0, 0.0, hd]])]
    k0 = 2 * np.pi / 22
    k_array = np.linspace(0.9 * k0, 1.1 * k0, 7)
    zt = TriangularSolver(wires=wires, nsegs=20).compute_impedance_swept(k_array)
    zb = BSplineSolver(wires=wires, nsegs=20, degree=1).compute_impedance_swept(k_array)
    assert np.allclose(zt, zb, rtol=RTOL, atol=0), (
        f"max rel diff {np.max(np.abs(zt - zb) / np.abs(zt)):.2e}"
    )
