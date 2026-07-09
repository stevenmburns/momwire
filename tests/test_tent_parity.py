"""Tent-basis (BSplineSolver degree=1) pinned-value regressions.

These pins are the surviving half of the TriangularSolver transfer-of-trust
tests (docs/triangular-retirement-plan.md). Before the retirement, every
value below was asserted equal to TriangularSolver's answer to ≤1e-8
relative on the same mesh — the two solvers are the SAME numerical scheme
(same tent basis, same quadratures, same assembly) whenever the delta-gap
feed lands on a mesh knot, which holds at even per-wire segment counts
(the parity antennaknobs enforces for d=1), at kink knots, and whenever
`feed_arclength` names a knot. The constants therefore ARE the retired
triangular solver's values; a drift here means the tent-basis pipeline
changed behavior, not that a tolerance was mis-guessed.

The one convention difference triangular had: for a feed arclength BETWEEN
knots (odd nsegs, midpoint feed) it snapped to the nearest interior knot,
while bspline excites the exact arclength by splitting the delta-gap
across the two straddling tents (a poor representation on a linear basis —
snap-or-even-N is the better choice, and the last test pins the gap so
the convention difference stays visible).
"""

import numpy as np
import pytest

from momwire.bspline import BSplineSolver

from test_momwire import _fandipole_two_band_sim

HD = 0.962 * 22 / 4

# Relative gate: the pins were recorded at solve-order-roundoff agreement
# with the retired TriangularSolver; 1e-6 leaves room for BLAS/OMP
# reduction-order jitter across machines while still catching any real
# change in quadrature, assembly, or feed handling.
RTOL = 1e-6


def _z(wires, nsegs, **kw):
    z, _ = BSplineSolver(wires=wires, nsegs=nsegs, degree=1, **kw).compute_impedance()
    return z


@pytest.mark.parametrize(
    "half,nsegs,z_pin",
    [
        (5.29, 20, 69.069365 - 20.348461j),  # resonant 0.481λ
        (5.29, 40, 69.403082 - 19.237949j),
        (2.65, 20, 11.984894 - 942.538771j),  # short 0.24λ (big |X|)
        (2.65, 40, 11.965807 - 941.022355j),
    ],
)
def test_d1_dipole_pinned(half, nsegs, z_pin):
    wires = [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]
    z = _z(wires, nsegs)
    assert abs(z - z_pin) <= RTOL * abs(z_pin), f"z={z} pin={z_pin}"


def test_d1_two_wire_yagi_pinned():
    refl = 1.05 * HD
    wires = [
        np.array([[0.0, 0.0, -HD], [0.0, 0.0, HD]]),
        np.array([[-2.2, 0.0, -refl], [-2.2, 0.0, refl]]),
    ]
    z = _z(wires, 20)
    z_pin = 27.793081 + 2.327531j
    assert abs(z - z_pin) <= RTOL * abs(z_pin), f"z={z}"


def test_d1_v_dipole_pinned():
    c30, s30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    wires = [
        np.array(
            [[-HD * c30, 0.0, HD * s30], [0.0, 0.0, 0.0], [HD * c30, 0.0, HD * s30]]
        )
    ]
    z = _z(wires, 10)
    z_pin = 53.636373 - 32.936517j
    assert abs(z - z_pin) <= RTOL * abs(z_pin), f"z={z}"


def test_d1_pec_ground_pinned():
    wires = [np.array([[-HD, 0.0, 7.0], [HD, 0.0, 7.0]])]
    z = _z(wires, 20, ground_z=0.0)
    z_pin = 93.328701 - 12.162851j
    assert abs(z - z_pin) <= RTOL * abs(z_pin), f"z={z}"


def test_d1_fandipole_k3_junctions_pinned():
    wavelength = 299_792_458.0 / 14.3e6
    z, _ = _fandipole_two_band_sim(
        20, wavelength, solver_cls=BSplineSolver, degree=1
    ).compute_impedance()
    z_pin = 60.319831 - 0.820675j
    assert abs(z - z_pin) <= 10 * RTOL * abs(z_pin), f"z={z}"


def test_d1_multifeed_y_matrix_pinned():
    wires = [
        np.array([[0.0, 0.0, -HD], [0.0, 0.0, HD]]),
        np.array([[3.0, 0.0, -HD], [3.0, 0.0, HD]]),
    ]
    feeds = [(0, None, 1.0), (1, None, 1.0 + 0.5j)]
    Y = BSplineSolver(wires=wires, nsegs=20, degree=1, feeds=feeds).compute_y_matrix()
    assert abs(Y[0, 0] - (0.02086267 + 0.02461764j)) < 1e-7
    assert abs(Y[0, 1] - (-0.01344524 - 0.02291742j)) < 1e-7
    assert abs(Y[0, 1] - Y[1, 0]) < 1e-10  # reciprocity


def test_d1_swept_pinned():
    wires = [np.array([[0.0, 0.0, -HD], [0.0, 0.0, HD]])]
    k0 = 2 * np.pi / 22
    k_array = np.linspace(0.9 * k0, 1.1 * k0, 7)
    zs = BSplineSolver(wires=wires, nsegs=20, degree=1).compute_impedance_swept(k_array)
    for got, pin in [
        (zs[0], 50.602751 - 180.306570j),
        (zs[-1], 94.257307 + 139.987897j),
    ]:
        assert abs(got - pin) <= RTOL * abs(pin), f"got={got} pin={pin}"


def test_d1_odd_nsegs_feed_convention():
    """At odd nsegs with a midpoint feed the delta-gap falls BETWEEN knots.
    The retired TriangularSolver snapped it to the nearest interior knot;
    bspline excites the exact arclength. Pin both numbers so the
    convention difference stays visible: the knot-fed value equals the
    old triangular answer, the midpoint-fed value is the (worse-
    conditioned) between-knots excitation. Callers wanting triangular-
    compatible results at odd N must feed at a knot (or use even N —
    which antennaknobs' parity coercion already guarantees).
    """
    half, nsegs = 2.65, 21
    wires = [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]
    h = 2 * half / nsegs
    knot_arc = round(half / h) * h
    z_knot = _z(wires, nsegs, feed_arclength=knot_arc)
    z_mid = _z(wires, nsegs)
    pin_knot = 11.989311 - 944.752814j  # == retired triangular's snap answer
    pin_mid = 13.238609 - 992.756924j
    assert abs(z_knot - pin_knot) <= RTOL * abs(pin_knot), f"z={z_knot}"
    assert abs(z_mid - pin_mid) <= RTOL * abs(pin_mid), f"z={z_mid}"
    assert abs(z_knot - z_mid) > 1e3 * RTOL * abs(pin_knot)
