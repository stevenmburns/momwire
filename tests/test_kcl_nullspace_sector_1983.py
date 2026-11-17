"""`kcl_nullspace` on the sector route (momwire#1983).

The sector route (`rotational_symmetry=True`) used to ignore the flag: its
harmonic solves carried the junction rows through a Schur step of their own
and never reached `_solve_with_kcl`. Now each harmonic's block is reduced to
the null space of its own rows before the factor, by the same step the dense
route takes (`_kcl_nullspace.solve_reduced`). Those rows are the dense
pattern with weights (sqrt(N) on harmonic 0, DFT phases on harmonic h), so
the plan is `plan_weighted`, and harmonic h tests with the conjugate rows.

What is held:

- the algebra: `plan_weighted`'s Q spans the null space of weighted and
  complex rows, its reduction is the dense Qᵀ Z Q (Q^H Z Q when testing with
  the conjugate), and the reduced solve is the Schur answer;
- a near-cancelled entry is refused by name, never divided by;
- on the route, on: the reduction runs (the route word and `COUNTS`), no
  dense Schur solve is reached, and impedance, Y and port columns match the
  route with the flag off within `RTOL` and the dense solve within the
  route's own bars;
- off: the route is bit for bit what it was, and names nothing.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from momwire import _kcl_nullspace
from momwire import _rotational_symmetry as rs
from momwire.bspline import BSplineSolver

from test_rotational_off_axis_1897 import C_TOL, FEED, LIFT, SOMM, WL, cage
from test_rotational_symmetry_1029 import solver as screen_solver

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

ENV = "MOMWIRE_X_KCL_NULLSPACE"

# The registry's `drift` for the feature, the bar for on against off on the
# route and for on against the dense solve's Z and Y. Measured on Skylake
# (2026-11-17), worst on these decks: on/off 4.4e-14 (cage port columns),
# on/dense 7.5e-14 (cage Y, free space). The cage's route and dense solves
# agree that well; the route's own bars (`C_TOL` for the columns) are looser
# because other decks' rotated copies round apart.
RTOL = 1e-12


@pytest.fixture(autouse=True)
def _env_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def build(*, rot=True, ground=None, feeds=None, **kw):
    wires, npe = cage(lift=LIFT if ground else 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            degree=2,
            wavelength=WL,
            wire_radius=0.002,
            feeds=[FEED] if feeds is None else feeds,
            rotational_symmetry=rot,
            **(SOMM if ground else {}),
            **kw,
        )


def posts(n=4):
    return [(2 + i, 0.2, 0j) for i in range(n)]


# ---- the algebra -------------------------------------------------------------


def _weighted_rows(rng, n, sizes, phases=False):
    """Disjoint rows with weights: sqrt(2) and 1, or unit phases."""
    cols = rng.permutation(n)
    A = np.zeros((len(sizes), n), dtype=np.complex128)
    at = 0
    for i, s in enumerate(sizes):
        for c in cols[at : at + s]:
            if phases:
                A[i, c] = np.exp(2j * math.pi * rng.integers(4) / 4) * rng.choice(
                    [-1, 1]
                )
            else:
                A[i, c] = rng.choice([-1.0, 1.0]) * rng.choice([1.0, math.sqrt(2.0)])
        at += s
    return A


def _q(p):
    return np.stack([_kcl_nullspace.expand(col, p) for col in np.eye(p.n_free)], 1)


@pytest.mark.parametrize("phases", [False, True], ids=["weighted", "phased"])
def test_q_spans_the_null_space(phases):
    rng = np.random.default_rng(1983)
    A = _weighted_rows(rng, 30, [3, 2, 4], phases=phases)
    p = _kcl_nullspace.plan_weighted(A, conj_test=True)
    assert not isinstance(p, str)
    Q = _q(p)
    assert np.max(np.abs(A @ Q)) < 1e-15
    assert np.linalg.matrix_rank(Q) == 30 - 3
    assert np.max(np.abs(p.a)) <= 1.0
    assert p.conj_test == phases  # real rows test with Q itself


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("phases", [False, True], ids=["weighted", "phased"])
def test_the_reduction_is_the_dense_projection(phases, order):
    rng = np.random.default_rng(7)
    n = 25
    A = _weighted_rows(rng, n, [3, 3], phases=phases)
    p = _kcl_nullspace.plan_weighted(A, conj_test=True)
    Q = _q(p)
    P = np.conj(Q) if phases else Q
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    got = _kcl_nullspace.reduce_matrix(Z.copy(order=order), p, overwrite=True)
    assert np.allclose(got, P.T @ Z @ Q, rtol=0, atol=1e-13)


@pytest.mark.parametrize("phases", [False, True], ids=["weighted", "phased"])
def test_the_reduced_solve_is_the_schur_answer(phases):
    """Against `solve_ports`' Schur step, multipliers on the conjugate rows."""
    rng = np.random.default_rng(11)
    n = 30
    A = _weighted_rows(rng, n, [3, 2, 2], phases=phases)
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Z += n * np.eye(n)
    V = rng.standard_normal((n, 2)) + 0j
    lu = rs._lapack.lu_factor_strict(Z.copy())
    schur = rs._constrained(lu, V, A, np.conj(A).T)
    p = _kcl_nullspace.plan_weighted(A, conj_test=True)
    got = _kcl_nullspace.solve_reduced(Z.copy(), V, p)
    assert _rel(got, schur) < 1e-13
    fac = _kcl_nullspace.factor_reduced(Z.copy(), p)
    assert np.array_equal(_kcl_nullspace.solve_factored(fac, V, p), got)


def test_a_near_cancelled_entry_is_refused_by_name():
    A = np.zeros((1, 6), dtype=np.complex128)
    A[0, :3] = (1.0, 1e-17, -1.0)
    assert _kcl_nullspace.plan_weighted(A) == "a near-zero constraint entry"
    A[0, 1] = 0.5
    assert not isinstance(_kcl_nullspace.plan_weighted(A), str)


def test_the_unit_plan_is_unchanged():
    """`plan` (the dense route's) still refuses a weight, and its plans
    carry real weights tested with Q itself."""
    rng = np.random.default_rng(5)
    A = _weighted_rows(rng, 20, [3, 2]).real
    A[0, np.flatnonzero(A[0])[0]] = math.sqrt(2.0)
    assert _kcl_nullspace.plan(A) == "an entry other than +-1"
    p = _kcl_nullspace.plan(np.sign(A))
    assert p.conj_test is False and p.a.dtype == np.float64
    assert p.row_a is p.a


# ---- on the route --------------------------------------------------------------


def _no_dense_schur(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the sector route reached a dense KCL solve")

    for name in (
        "_solve_with_kcl",
        "_solve_with_kcl_ports",
        "_solve_with_kcl_batch",
        "_solve_with_kcl_swept_ports",
    ):
        monkeypatch.setattr(BSplineSolver, name, boom)


@pytest.mark.parametrize("ground", [None, "sommerfeld"], ids=["free", "sommerfeld"])
def test_impedance_on_the_route_takes_the_reduction(ground, monkeypatch):
    """The cage: the ring and spoke junctions are off the axis, so harmonic
    0's rows are orbit sums carrying sqrt(N) weights."""
    dense = build(rot=False, ground=ground).compute_impedance()
    off = build(ground=ground)
    z_off, c_off = off.compute_impedance()
    before = _kcl_nullspace.COUNTS["reduced"]
    _no_dense_schur(monkeypatch)
    on = build(ground=ground, kcl_nullspace=True)
    z_on, c_on = on.compute_impedance()
    assert on.kcl_nullspace_route == "reduced"
    assert _kcl_nullspace.COUNTS["reduced"] == before + 1
    assert off.kcl_nullspace_route is None
    assert _rel(z_on, z_off) < RTOL
    assert _rel(c_on, c_off) < RTOL
    assert abs(z_on - dense[0]) / abs(dense[0]) < RTOL
    assert _rel(c_on, dense[1]) <= C_TOL


def test_the_buried_screen(monkeypatch):
    """momwire#1029's screen: the hub junction is on the axis."""
    z_off = screen_solver(kcl_nullspace=False).compute_impedance()[0]
    _no_dense_schur(monkeypatch)
    on = screen_solver(kcl_nullspace=True)
    z_on = on.compute_impedance()[0]
    assert on.kcl_nullspace_route == "reduced"
    assert _rel(z_on, z_off) < RTOL


@pytest.mark.parametrize("ground", [None, "sommerfeld"], ids=["free", "sommerfeld"])
def test_ports_on_the_route_take_the_reduction_in_every_harmonic(ground, monkeypatch):
    """1 + N ports, every harmonic: harmonic h's rows carry DFT phases and
    are tested with their conjugates."""
    feeds = [FEED, *posts()]
    dense = build(rot=False, ground=ground, feeds=feeds).compute_port_solution()
    off = build(ground=ground, feeds=feeds).compute_port_solution()
    seen = []
    real = _kcl_nullspace.factor_reduced

    def spy(Z, p, **k):
        seen.append(p.conj_test)
        return real(Z, p, **k)

    monkeypatch.setattr(_kcl_nullspace, "factor_reduced", spy)
    _no_dense_schur(monkeypatch)
    s = build(ground=ground, feeds=feeds, kcl_nullspace=True)
    on = s.compute_port_solution()
    assert s.kcl_nullspace_route == "reduced"
    assert len(seen) == 4 and seen.count(False) >= 1 and any(seen)
    assert _rel(on.y, off.y) < RTOL
    assert _rel(on.coeffs, off.coeffs) < RTOL
    assert _rel(on.y, dense.y) < RTOL
    assert _rel(on.coeffs, dense.coeffs) <= C_TOL


def test_an_unreducible_harmonic_takes_schur_and_says_why(monkeypatch):
    """All or none, named: the answer is the flag-off route's, bit for bit."""
    feeds = [FEED, *posts()]
    off = build(feeds=feeds).compute_port_solution()
    monkeypatch.setattr(
        _kcl_nullspace, "plan_weighted", lambda rows, **k: "a test's refusal"
    )
    s = build(feeds=feeds, kcl_nullspace=True)
    on = s.compute_port_solution()
    assert s.kcl_nullspace_route.startswith("schur: harmonic ")
    assert s.kcl_nullspace_route.endswith(": a test's refusal")
    assert np.array_equal(on.y, off.y)
    assert np.array_equal(on.coeffs, off.coeffs)


def test_off_never_plans(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("planned with the flag off")

    monkeypatch.setattr(_kcl_nullspace, "plan_weighted", boom)
    s = build(feeds=[FEED, *posts()])
    s.compute_port_solution()
    s.compute_impedance()
    assert s.kcl_nullspace_route is None
