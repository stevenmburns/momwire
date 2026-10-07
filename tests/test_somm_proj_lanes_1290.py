"""`remainder_field_proj_batch`'s AVX2 lanes against its per-pair loop —
perf item 9 (momwire#1290).

The above-ground Sommerfeld remainder's projection runs a row's sources in
blocks: geometry and the libm calls (hypot, atan2, the polar's sincos) per
pair, the stencil and the projection four pairs to a vector, the surface read
two surfaces to a vector, the momwire#1258 continuation per pair where it
applies, and a block's last nb mod 4 pairs through `proj_one`. Each lane does
the scalar stage's operations in its order (separate multiplies and adds, the
build being -ffp-contract=off), so the claim is equality of bytes with
`lanes=False`, the per-pair loop:

  * on every call a live above-ground and a mixed solve make, the lanes'
    pair counter moving by exactly the calls' pairs;
  * on synthetic observer/source sets over the same grids, at source counts
    around the vector width and the block, with stacked pairs (rho = 0, the
    unsafe-azimuth arm), grazing and steep pairs (both theta clamps and every
    region), and pairs past the table's edge (the continuation);
  * on whole solves, G, Z and the currents with `_sommerfeld._PROJ_LANES`
    off.

On Windows (MSVC, /fp:fast) the lanes run too since momwire#1371, and every
lanes-vs-walk comparison here takes the derived win32 tolerance instead of bit
equality (`assert_lanes_match`, tests/_lane_gate.py). Linux and macOS keep the
bit gates.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sg_cplx_far_fill_simd_1224 import _hub4
from test_sg_real_far_fill_staged_1290 import _array

import momwire._accel as _accel
from momwire import _sommerfeld
from momwire import sinusoidal_galerkin as sg
from _lane_gate import assert_lanes_match


def _direct_sg(*args, **kw):
    """This module's subject is the DIRECT-field fill's own machinery, so
    every solver here names it (`SinusoidalGalerkinSolver`'s default fill is
    the mixed-potential one since momwire#1354, gated in
    `test_sinusoidal_mp_1354.py`)."""
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver as _cls

    kw.setdefault("fill", "direct")
    return _cls(*args, **kw)


pytestmark = pytest.mark.skipif(
    _accel.acc is None or not hasattr(_accel.acc, "somm_proj_lanes_built"),
    reason="no compiled remainder projection with the lanes in this build",
)

DECKS = {"array-above": lambda: _array(True, n=21), "hub4": _hub4}

# The lanes are the AVX2 build's alone; elsewhere `lanes=True` is the
# per-pair loop and the byte claims below hold trivially.
LANES_BUILT = bool(getattr(_accel.acc, "somm_proj_lanes_built", lambda: False)())


def _record(make):
    fn = _accel.acc.remainder_field_proj_batch
    seen = []

    def spy(*a, **kw):
        seen.append(
            [np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in a]
        )
        return fn(*a, **kw)

    before = _accel.acc.somm_proj_lane_pairs()
    _accel.acc.remainder_field_proj_batch = spy
    try:
        _direct_sg(**make()).compute_impedance()
    finally:
        _accel.acc.remainder_field_proj_batch = fn
    return seen, _accel.acc.somm_proj_lane_pairs() - before


def _both(args):
    fn = _accel.acc.remainder_field_proj_batch
    return fn(*args, lanes=True), fn(*args, lanes=False)


@pytest.fixture(scope="module")
def recorded():
    return {name: _record(make) for name, make in DECKS.items()}


@pytest.mark.parametrize("name", sorted(DECKS))
def test_live_calls_equal_the_per_pair_loop(name, recorded):
    calls, lane_pairs = recorded[name]
    assert calls, f"{name}: the remainder projection never ran"
    pairs = sum(a[0].shape[0] * a[2].shape[0] for a in calls)
    assert lane_pairs == (pairs if LANES_BUILT else 0)
    for i, args in enumerate(calls):
        lanes, scalar = _both(args)
        assert_lanes_match(lanes, scalar, f"{name}: call {i}")


def _synthetic(args, n_src, seed):
    """`args`' grid and ground with new observers and sources: a mix of
    ordinary pairs, stacked pairs (same x, y), near-grazing pairs, steep ones,
    and pairs past the table's edge."""
    rng = np.random.default_rng(seed)
    gz = float(args[4])
    r1_max = float(args[6])
    m = 9
    obs = np.column_stack(
        [rng.uniform(-3, 3, m), rng.uniform(-3, 3, m), gz + rng.uniform(1e-3, 2.0, m)]
    )
    src = np.column_stack(
        [
            rng.uniform(-3, 3, n_src),
            rng.uniform(-3, 3, n_src),
            gz + rng.uniform(1e-3, 2.0, n_src),
        ]
    )
    kind = np.arange(n_src) % 5
    src[kind == 1, :2] = obs[0, :2]  # stacked over observer 0: rho = 0
    src[kind == 2, 2] = gz + 1e-6  # grazing
    src[kind == 3, :2] = obs[0, :2] + 1e-4  # steep
    far = kind == 4  # past the table's edge
    src[far, 0] = obs[0, 0] + 1.5 * r1_max
    t_obs = rng.standard_normal((m, 3))
    t_obs /= np.linalg.norm(t_obs, axis=1, keepdims=True)
    t_src = rng.standard_normal((n_src, 3))
    t_src /= np.linalg.norm(t_src, axis=1, keepdims=True)
    t_src[kind == 0] = (0.0, 0.0, 1.0)  # vertical: the tangent's own unsafe arm
    return [obs, t_obs, src, t_src, *args[4:]]


@pytest.mark.parametrize("n_src", [1, 3, 4, 5, 63, 64, 65, 131])
@pytest.mark.parametrize("name", sorted(DECKS))
def test_synthetic_pairs_on_every_arm(name, n_src, recorded):
    calls, _ = recorded[name]
    lanes, scalar = _both(_synthetic(calls[0], n_src, seed=n_src))
    assert_lanes_match(lanes, scalar)
    assert np.all(np.isfinite(lanes))


@pytest.mark.parametrize("name", sorted(DECKS))
def test_whole_solve_with_the_lanes_off(name, monkeypatch):
    def capture():
        seen = {}
        solve = sg._solve_in_place

        def spy(G, rhs):
            seen.setdefault("G", np.array(G, copy=True))
            return solve(G, rhs)

        monkeypatch.setattr(sg, "_solve_in_place", spy)
        Z, cur = _direct_sg(**DECKS[name]()).compute_impedance()
        monkeypatch.setattr(sg, "_solve_in_place", solve)
        return seen["G"], np.atleast_1d(Z), np.asarray(cur)

    before = _accel.acc.somm_proj_lane_pairs()
    G1, Z1, I1 = capture()
    assert (_accel.acc.somm_proj_lane_pairs() > before) == LANES_BUILT
    monkeypatch.setattr(_sommerfeld, "_PROJ_LANES", False)
    mid = _accel.acc.somm_proj_lane_pairs()
    G0, Z0, I0 = capture()
    assert _accel.acc.somm_proj_lane_pairs() == mid
    assert_lanes_match(G1, G0)
    assert_lanes_match(Z1, Z0)
    assert_lanes_match(I1, I0)
