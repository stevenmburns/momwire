"""`pair_extents_below`'s AVX2 lanes against its scalar loop — perf item 9
(momwire#1290).

The walk's result is a max and a min over the node pairs' r1^2 and hh^2/rho^2.
The lanes form each pair's values with the scalar loop's operations, and a max
or a min is a selection, so the claim is equality of bytes with `lanes=False`:
on random clouds at every count around the vector width, on clouds with
duplicated nodes (rho = 0, a +inf quotient) and nodes on the interface (a 0/0
quotient the minimum drops), and through a live buried solve with
`bspline._PAIR_EXTENTS_LANES` flipped. The red control is a cloud whose
minimising pair sits in the lanes: the extents the gate compares must move
when that pair's depth moves by 1e-12 relative.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sg_cplx_far_fill_simd_1224 import _hub4

import momwire._accel as _accel
from momwire import bspline
from momwire import sinusoidal_galerkin as sg

pytestmark = pytest.mark.skipif(
    _accel.acc is None or not hasattr(_accel.acc, "pair_extents_below"),
    reason="no compiled pair_extents_below in this build",
)


def _both(x, y, d):
    f = _accel.acc.pair_extents_below
    return f(x, y, d, lanes=True), f(x, y, d, lanes=False)


def _bytes(t):
    return np.asarray(t, dtype=np.float64).tobytes()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 31, 64, 257])
def test_random_clouds(n):
    rng = np.random.default_rng(n)
    x, y = rng.uniform(-20, 20, (2, n))
    d = rng.uniform(1e-3, 2.0, n)
    lanes, scalar = _both(x, y, d)
    assert _bytes(lanes) == _bytes(scalar)


@pytest.mark.parametrize("n", [6, 13, 40])
def test_duplicated_and_interface_nodes(n):
    rng = np.random.default_rng(100 + n)
    x, y = rng.uniform(-5, 5, (2, n))
    d = rng.uniform(0.0, 1.0, n)
    x[3], y[3] = x[1], y[1]  # rho = 0, hh > 0: +inf
    d[2] = d[5] = 0.0  # interface nodes; with x/y equal below, 0/0
    x[5], y[5] = x[2], y[2]
    lanes, scalar = _both(x, y, d)
    assert _bytes(lanes) == _bytes(scalar)
    assert np.isfinite(lanes[1])


def test_red_control_the_minimising_pair_moves_the_minimum():
    rng = np.random.default_rng(7)
    n = 40
    x, y = rng.uniform(-20, 20, (2, n))
    d = rng.uniform(0.5, 2.0, n)
    # Node 9 shallow and close to node 1: the pair (1, 9), in the lanes of row 1.
    x[9], y[9], d[9] = x[1] + 15.0, y[1], 1e-3
    d[1] = 1e-3
    base, _ = _both(x, y, d)
    d2 = d.copy()
    d2[9] = d[9] * (1.0 + 1e-12)
    moved, _ = _both(x, y, d2)
    assert _bytes(moved) != _bytes(base)


def test_live_buried_solve_with_the_lanes_off(monkeypatch):
    calls = []
    f = _accel.acc.pair_extents_below

    def spy(*a, **kw):
        calls.append(kw.get("lanes"))
        return f(*a, **kw)

    monkeypatch.setattr(_accel.acc, "pair_extents_below", spy)
    Z1, I1 = sg.SinusoidalGalerkinSolver(**_hub4()).compute_impedance()
    monkeypatch.setattr(bspline, "_PAIR_EXTENTS_LANES", False)
    Z0, I0 = sg.SinusoidalGalerkinSolver(**_hub4()).compute_impedance()
    assert True in calls and False in calls, calls
    assert np.atleast_1d(Z1).tobytes() == np.atleast_1d(Z0).tobytes()
    assert np.asarray(I1).tobytes() == np.asarray(I0).tobytes()
