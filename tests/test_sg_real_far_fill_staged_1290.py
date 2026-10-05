"""The reduced real-k Galerkin far fill's rows, staged for the vector unit —
perf item 9 (momwire#1290).

`sg_real_rows_staged` runs the real fill's per-pair body as loops over the
sources the compiler vectorises, keeps the reference's own libmvec sweep over
the reference's own (source, kind) table, and keeps log1p, asinh and
sin_minus_arg's sin as the scalar libm calls the reference makes. Every value
is the reference's expression in its association (-ffp-contract=off, no
reduction across lanes), so the claim is EQUALITY OF BYTES with the per-pair
body, which stays callable as `reference=True` and reachable from a solve as
`sinusoidal_galerkin._SG_REAL_STAGED = False`. Any difference is a
transcription error, so there is no tolerance to derive.

What this file pins:

  * bytes, staged vs reference, on every reduced real fill call of free,
    above-ground and mixed decks made through the production seam, with and
    without the ground fold's `out`/`scale`;
  * that production took the staged body, by the kernel's per-body counters;
  * bytes on synthetic rows that put lanes on each arm of the assembly's
    selects (sin_minus_arg's series / sin, t_sing's series / far arm, X's two
    rationalisations) at source counts off the vector width and the tile;
  * whole solves: Z and the matrix handed to the solve, flag on vs off.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sg_cplx_far_fill_simd_1224 import _hub4

import momwire._accel as _accel
from momwire import sinusoidal_galerkin as sg


def _direct_sg(*args, **kw):
    """This module's subject is the DIRECT-field fill's own machinery, so
    every solver here names it (`SinusoidalGalerkinSolver`'s default fill is
    the mixed-potential one since momwire#1354, gated in
    `test_sinusoidal_mp_1354.py`)."""
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver as _cls

    kw.setdefault("fill", "direct")
    return _cls(*args, **kw)


pytestmark = pytest.mark.skipif(
    not sg._HAVE_GALERKIN_FAR_FILL or not sg._HAVE_SG_REAL_STAGED,
    reason="no compiled real far fill with the staged rows in this build",
)

C0 = 299792458.0
WL7 = C0 / 7e6
SOMM = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")


def _array(ground, n=21):
    """Three parallel dipoles 8.5 m apart, 10 m up: the timed decks' shape."""
    wires = [
        np.array([(-10.0, 8.5 * j, 10.0), (10.0, 8.5 * j, 10.0)]) for j in range(3)
    ]
    d = dict(
        wires=wires,
        n_per_edge_per_wire=[[n] for _ in wires],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
    )
    if ground:
        d.update(SOMM)
    return d


def _vee():
    """A bent, fed vee and a skew parasite: pairs off the parallel family, so
    X and the endpoint offsets take both signs."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (6.0, 2.0, 3.0)]),
            np.array([(0.0, 0.0, 0.0), (-5.0, 3.0, 1.0)]),
            np.array([(2.0, -4.0, -1.0), (-3.0, -1.0, 4.0)]),
        ],
        n_per_edge_per_wire=[[17], [15], [19]],
        junctions=[[(0, "start"), (1, "start")]],
        feeds=[(0, 3.0, 1 + 0j)],
        wavelength=WL7 / 3.0,
        wire_radius=2e-3,
    )


DECKS = {
    "array-free": lambda: _array(False),
    "array-above": lambda: _array(True),
    "vee-free": _vee,
    "hub4": _hub4,
}


def _record(make):
    """Every reduced real fill call one solve makes, as argument copies, and
    how far the per-body counters moved during it. The spy only records and
    forwards; the replay happens after it is removed."""
    fill = _accel.acc.sinusoidal_galerkin_far_fill
    seen = []

    def spy(*a, **kw):
        seen.append(
            [np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in a]
        )
        return fill(*a, **kw)

    before = _accel.acc.sg_real_far_fill_calls()
    _accel.acc.sinusoidal_galerkin_far_fill = spy
    try:
        _direct_sg(**make()).compute_impedance()
    finally:
        _accel.acc.sinusoidal_galerkin_far_fill = fill
    after = _accel.acc.sg_real_far_fill_calls()
    return seen, (after[0] - before[0], after[1] - before[1])


def _same_bytes(a, b):
    return all(x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))


def _both(args, **kw):
    fill = _accel.acc.sinusoidal_galerkin_far_fill
    return fill(*args, **kw), fill(*args, reference=True, **kw)


@pytest.mark.parametrize("name", sorted(DECKS))
def test_staged_rows_equal_the_reference_body_to_the_byte(name):
    """Production took the staged body on every call (the counters moved by
    exactly the calls seen, the reference's not at all); each call replayed
    gives the reference's bytes; and so does the ground fold into a random
    triple at a complex scale."""
    calls, (staged_n, ref_n) = _record(DECKS[name])
    assert calls, f"{name}: the real far fill never ran"
    assert staged_n == len(calls), (staged_n, len(calls))
    assert ref_n == 0, f"{name}: a production fill took the reference body"
    for i, args in enumerate(calls):
        staged, ref = _both(args)
        assert _same_bytes(staged, ref), f"{name}: call {i} differs"

    rng = np.random.default_rng(1290)
    shape = (calls[-1][10].shape[0], calls[-1][3].shape[0])
    base = [
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape) for _ in range(3)
    ]
    fill = _accel.acc.sinusoidal_galerkin_far_fill
    outs = []
    for reference in (False, True):
        out = tuple(np.array(b, copy=True) for b in base)
        fill(*calls[-1], out=out, scale=complex(-1.0, 0.3), reference=reference)
        outs.append(out)
    assert _same_bytes(*outs), f"{name}: the folded triple differs"


def _synthetic(H, n_src, z_obs, k, seed=0):
    """A direct call: sources of half-length H scattered in a box with random
    tangents, observers along random segments. `z_obs` shifts the observers
    along z, which moves P = 2Hz/(r1 + r2) and so the two sin_minus_arg
    arguments k(H +- P) across 0.1."""
    rng = np.random.default_rng(seed)
    H = np.broadcast_to(np.asarray(H, dtype=np.float64), (n_src,)).copy()
    src_c = rng.uniform(-3, 3, (n_src, 3))
    src_t = rng.standard_normal((n_src, 3))
    src_t /= np.linalg.norm(src_t, axis=1, keepdims=True)
    nq, M = 4, 3
    obs_c = rng.uniform(-3, 3, (M * nq, 3)) + np.array([0.0, 0.0, z_obs])
    obs_t = rng.standard_normal((M * nq, 3))
    obs_t /= np.linalg.norm(obs_t, axis=1, keepdims=True)
    a = rng.uniform(5e-4, 2e-3, M * nq)
    gx, gw = np.polynomial.legendre.leggauss(8)
    nnz = 2 * M
    w = rng.standard_normal((nnz, nq)) + 1j * rng.standard_normal((nnz, nq))
    starts = np.arange(0, nnz + 1, 2, dtype=np.int64)
    return [obs_c, obs_t, a, src_c, src_t, H, k, 377.0, gx, gw, w, starts]


@pytest.mark.parametrize("n_src", [1, 3, 4, 5, 255, 256, 257, 517])
@pytest.mark.parametrize(
    "H, k",
    [
        (0.02, 0.6),  # |k(H +- P)| mostly under 0.1: the series arm
        (0.9, 0.6),  # mostly over: sin(u) - u
        (0.2, 0.6),  # both, lane by lane
        (2.5, 2.0),  # long sources: |X| >= 1, t_sing's far arm
    ],
)
def test_synthetic_rows_on_every_arm(n_src, H, k):
    for z_obs in (0.0, 4.0):
        staged, ref = _both(_synthetic(H, n_src, z_obs, k, seed=n_src))
        assert _same_bytes(staged, ref), (n_src, H, k, z_obs)


def _capture(make):
    """(G, Z): G as `_solve_in_place` receives it, before the solve
    overwrites it."""
    seen = {}
    solve = sg._solve_in_place

    def spy(G, rhs):
        seen.setdefault("G", np.array(G, copy=True))
        return solve(G, rhs)

    sg._solve_in_place = spy
    try:
        Z, cur = _direct_sg(**make()).compute_impedance()
    finally:
        sg._solve_in_place = solve
    return seen["G"], np.atleast_1d(Z), np.asarray(cur)


@pytest.mark.parametrize("name", ["array-above", "vee-free"])
def test_whole_solve_is_bit_identical_with_the_staged_body_off(name, monkeypatch):
    G1, Z1, I1 = _capture(DECKS[name])
    monkeypatch.setattr(sg, "_SG_REAL_STAGED", False)
    before = _accel.acc.sg_real_far_fill_calls()
    G0, Z0, I0 = _capture(DECKS[name])
    after = _accel.acc.sg_real_far_fill_calls()
    assert after[0] == before[0] and after[1] > before[1], (before, after)
    assert G1.tobytes() == G0.tobytes()
    assert Z1.tobytes() == Z0.tobytes()
    assert I1.tobytes() == I0.tobytes()
