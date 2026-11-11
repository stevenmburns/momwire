"""BSplineSolver's junction constraints eliminated before the factor
(momwire#1898, the opt-in feature `kcl_nullspace`).

Off, every constrained solve is the Schur route (`_solve_with_kcl` and its
three siblings). On, a constraint matrix of the exact shape (+-1 entries, no
column in two rows, no empty row) is reduced to its null space,
(Qᵀ Z Q) y = Qᵀ v and x = Q y (`momwire._kcl_nullspace`); any other falls
back to the Schur route and says why.

What is held:

- the algebra: A Q = 0, `reduce_matrix` is the dense Qᵀ Z Q in either memory
  order, in place or not, and the reduced solve is the constrained solution;
- the refusals: a non-unit entry, a shared column, an empty row, each named
  on `kcl_nullspace_route`, and the Schur answer returned bit for bit;
- off by default, and off never enters the reduction;
- on, against off, within `RTOL` on every entry: impedance, Y matrix, port
  solution columns, the swept impedance and the swept port solutions, for
  gap feeds, junction ports, node gaps, singular enrichment, a PEC image and
  a buried Sommerfeld screen;
- the unknown count drops by exactly one per constraint row;
- the sector route (`rotational_symmetry=True`) keeps its own constraint
  path, untouched by the flag, and `pair_reuse` composes.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _features, _kcl_nullspace
from momwire.bspline import BSplineSolver

from test_port_solution import enrichment, junction_ports, pec_ground
from test_rotational_symmetry_1029 import solver as screen_solver

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

ENV = "MOMWIRE_X_KCL_NULLSPACE"

# |on - off| / |off|, per entry (Z, coefficients, Y, columns). Measured on
# Haswell, worst 2.6e-14 on these decks (degree 1); a different elimination
# order moves bits, never more. The registry's `drift` is the ELT whip's.
RTOL = 1e-12

WL = 10.0
L = 0.48 * WL


@pytest.fixture(autouse=True)
def _env_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def tee(z0=0.0, **extra):
    """A fed dipole with a three-wire junction at one tip and a two-wire bend
    at the other: KCL rows with 3 and 2 entries, both signs. `z0` lifts it."""
    h = L / 2
    wires = [
        np.array([(0.0, 0.0, z0 - h), (0.0, 0.0, z0 + h)]),
        np.array([(0.0, 0.0, z0 + h), (0.4, 0.0, z0 + h)]),
        np.array([(0.0, 0.0, z0 + h), (-0.3, 0.2, z0 + h)]),
        np.array([(0.5, 0.0, z0 - h), (0.0, 0.0, z0 - h)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[21], [4], [3], [5]],
        wavelength=WL,
        wire_radius=1e-3,
        junctions=[
            [(0, "end"), (1, "start"), (2, "start")],
            [(3, "end"), (0, "start")],
        ],
        feeds=[(0, None, 1.0 + 0j)],
        **extra,
    )


def node_gap(**extra):
    """The tee driven by a node gap at the three-wire junction as well."""
    return tee(node_gaps=[(1, "start", 0.5 + 0j)], **extra)


def pec_tee(**extra):
    """The tee over a PEC plane, and a grounded vertical whose foot is a
    grounded junction (no KCL row, `_kcl_row_junctions`)."""
    kw = tee(z0=L / 2 + 0.3, ground_z=0.0, **extra)
    kw["wires"] = kw["wires"] + [
        np.array([(2.0, 0.0, 0.0), (2.0, 0.0, 0.3)]),
        np.array([(2.0, 0.0, 0.3), (2.0, 0.0, 2.0)]),
    ]
    kw["n_per_edge_per_wire"] = kw["n_per_edge_per_wire"] + [[2], [8]]
    kw["junctions"] = kw["junctions"] + [[(4, "end"), (5, "start")]]
    return kw


# Decks with constraint rows left after the ports take theirs.
DECKS = {
    "tee": tee,
    "tee_degree3": lambda: tee(degree=3),
    "tee_degree1": lambda: tee(degree=1),
    "junction_port_and_kcl": lambda: tee(junction_ports=[(1, 0.5 + 0j)]),
    "node_gap": node_gap,
    "enrichment": enrichment,
    "pec_tee": pec_tee,
}
# Decks with none: every junction a port, or no junction. The flag has
# nothing to reduce and the answer is the Schur route's, bit for bit.
NO_ROWS = {"junction_ports_only": junction_ports, "pec_ground": pec_ground}


def _pair(kw):
    return (
        BSplineSolver(**kw, kcl_nullspace=False),
        BSplineSolver(**kw, kcl_nullspace=True),
    )


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ---- the algebra -------------------------------------------------------------


def _random_rows(rng, n, sizes):
    cols = rng.permutation(n)
    A = np.zeros((len(sizes), n))
    at = 0
    for r, m in enumerate(sizes):
        A[r, cols[at : at + m]] = rng.choice([-1.0, 1.0], size=m)
        at += m
    return A


def _dense_q(p):
    return np.stack([_kcl_nullspace.expand(col, p) for col in np.eye(p.n_free)], 1)


def test_q_spans_the_null_space_with_full_rank():
    rng = np.random.default_rng(1898)
    A = _random_rows(rng, 40, [3, 2, 1, 6, 2])
    p = _kcl_nullspace.plan(A)
    Q = _dense_q(p)
    assert p.n_free == 40 - 5
    assert np.array_equal(A @ Q, np.zeros((5, p.n_free)))
    assert np.linalg.matrix_rank(Q) == p.n_free
    # The lowest-index end of each row is the one eliminated.
    assert all(p.elim[r] == np.flatnonzero(A[r]).min() for r in range(5))


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_reduce_matrix_is_the_dense_projection(order, overwrite):
    rng = np.random.default_rng(7)
    n = 300  # more than one chunk of column operations
    A = _random_rows(rng, n, [2, 3] * 60)
    p = _kcl_nullspace.plan(A)
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q = _dense_q(p)
    want = Q.T @ Z @ Q
    Zin = np.array(Z, order=order)
    got = _kcl_nullspace.reduce_matrix(Zin, p, overwrite=overwrite)
    assert got.shape == (p.n_free, p.n_free)
    assert got.flags.f_contiguous or got.flags.c_contiguous
    assert np.allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())
    if not overwrite:
        assert np.array_equal(Zin, Z)


def test_the_reduced_solve_is_the_constrained_solution():
    rng = np.random.default_rng(3)
    n = 60
    A = _random_rows(rng, n, [3, 2, 2, 4])
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Z = Z + Z.T + n * np.eye(n)
    v = rng.standard_normal(n) + 0j
    s = BSplineSolver(**tee(), kcl_nullspace=True)
    x = s._solve_with_kcl(Z, v, A)
    assert s.kcl_nullspace_route == "reduced"
    s.kcl_nullspace_route = None
    s._kcl_nullspace = False
    x_schur = s._solve_with_kcl(Z, v, A)
    assert s.kcl_nullspace_route is None
    assert np.max(np.abs(A @ x)) < 1e-12 * np.max(np.abs(x))
    assert _rel(x, x_schur) < 1e-12


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda A: A.__setitem__((0, np.flatnonzero(A[0])[0]), 0.5), "+-1"),
        (lambda A: A.__setitem__((1, np.flatnonzero(A[0])[0]), 1.0), "two"),
        (lambda A: A.__setitem__(2, 0.0), "empty"),
    ],
)
def test_an_inexact_shape_falls_back_to_schur_and_says_why(edit, reason):
    rng = np.random.default_rng(11)
    n = 30
    A = _random_rows(rng, n, [3, 2, 2])
    edit(A)
    assert reason in _kcl_nullspace.plan(A)
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Z += n * np.eye(n)
    v = rng.standard_normal(n) + 0j
    before = _kcl_nullspace.COUNTS.copy()
    on = BSplineSolver(**tee(), kcl_nullspace=True)
    off = BSplineSolver(**tee(), kcl_nullspace=False)
    if reason == "empty":
        # An empty row leaves the Schur system singular too: the fallback
        # is still taken and named, and fails exactly as off does.
        for s in (on, off):
            with pytest.raises(np.linalg.LinAlgError):
                s._solve_with_kcl(Z, v, A)
    else:
        assert np.array_equal(on._solve_with_kcl(Z, v, A), off._solve_with_kcl(Z, v, A))
    assert on.kcl_nullspace_route == f"schur: {_kcl_nullspace.plan(A)}"
    assert reason in on.kcl_nullspace_route
    assert _kcl_nullspace.COUNTS[on.kcl_nullspace_route] == (
        before[on.kcl_nullspace_route] + 1
    )
    assert off.kcl_nullspace_route is None


# ---- the switch --------------------------------------------------------------


def test_off_by_default_and_off_never_reduces(monkeypatch):
    assert BSplineSolver(**tee())._kcl_nullspace is False
    monkeypatch.setenv(ENV, "1")
    assert BSplineSolver(**tee())._kcl_nullspace is True
    assert BSplineSolver(**tee(), kcl_nullspace=False)._kcl_nullspace is False
    monkeypatch.delenv(ENV)

    def boom(*_a, **_k):
        raise AssertionError("the reduction ran with the flag off")

    monkeypatch.setattr(_kcl_nullspace, "plan", boom)
    s = BSplineSolver(**tee())
    s.compute_impedance()
    s.compute_port_solution()
    assert s.kcl_nullspace_route is None


def test_the_registry_entry():
    f = _features.FEATURES["kcl_nullspace"]
    assert (f.reach, f.kind, f.issue, f.env, f.kwarg) == (
        1,
        "python",
        1898,
        ENV,
        "kcl_nullspace",
    )
    assert f.drift >= RTOL


# ---- on against off, every entry ---------------------------------------------


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_impedance_y_and_columns_match_the_schur_route(deck):
    off, on = _pair(DECKS[deck]())
    z_off, c_off = off.compute_impedance()
    z_on, c_on = on.compute_impedance()
    assert on.kcl_nullspace_route == "reduced"
    assert _rel(z_on, z_off) < RTOL
    assert _rel(c_on, c_off) < RTOL
    s_off, s_on = off.compute_port_solution(), on.compute_port_solution()
    assert _rel(s_on.y, s_off.y) < RTOL
    assert _rel(s_on.coeffs, s_off.coeffs) < RTOL
    assert _rel(on.compute_y_matrix(), off.compute_y_matrix()) < RTOL


@pytest.mark.parametrize("deck", sorted(NO_ROWS))
def test_a_deck_with_no_constraint_rows_is_untouched(deck):
    off, on = _pair(NO_ROWS[deck]())
    assert np.array_equal(on.compute_impedance()[0], off.compute_impedance()[0])
    assert np.array_equal(on.compute_port_solution().y, off.compute_port_solution().y)
    assert on.kcl_nullspace_route is None


@pytest.mark.parametrize("deck", ["tee", "junction_port_and_kcl", "node_gap"])
def test_the_swept_entries_match_the_schur_route(deck):
    off, on = _pair(DECKS[deck]())
    ks = 2 * np.pi / WL * np.array([0.95, 1.0, 1.07])
    assert on._swept_batched_available()
    assert _rel(on.compute_impedance_swept(ks), off.compute_impedance_swept(ks)) < RTOL
    for a, b in zip(
        on.compute_port_solution_swept(ks), off.compute_port_solution_swept(ks)
    ):
        assert _rel(a.y, b.y) < RTOL
        assert _rel(a.coeffs, b.coeffs) < RTOL
    assert on.kcl_nullspace_route == "reduced"


def test_the_columns_satisfy_kcl_and_the_count_drops_by_one_per_row():
    kw = tee()
    s = BSplineSolver(**kw, kcl_nullspace=True)
    geom = s._build_geometry()
    supp_seg, _polys, kcl_A, _wk, _wbg = s._build_basis_polynomials(geom)
    p = _kcl_nullspace.plan(kcl_A)
    assert p.n_free == supp_seg.shape[0] - kcl_A.shape[0]
    x = s.compute_port_solution().coeffs
    assert np.max(np.abs(kcl_A @ x)) < 1e-13 * np.max(np.abs(x))


# ---- composition -------------------------------------------------------------


def test_the_sector_route_keeps_its_own_constraint_path():
    """`rotational_symmetry=True` solves each harmonic under its own rows
    (`_rotational_symmetry._harmonic_constraints`, complex DFT combinations,
    not of the reducible shape) and never reaches `_solve_with_kcl`: the flag
    changes no bit there. The dense solve of the same buried screen, on,
    matches off."""
    z_sym_off = screen_solver(kcl_nullspace=False).compute_impedance()[0]
    sym_on = screen_solver(kcl_nullspace=True)
    z_sym_on = sym_on.compute_impedance()[0]
    assert z_sym_on == z_sym_off
    assert sym_on.kcl_nullspace_route is None
    dense_off = screen_solver(rotational_symmetry=False, kcl_nullspace=False)
    dense_on = screen_solver(rotational_symmetry=False, kcl_nullspace=True)
    z_off = dense_off.compute_impedance()[0]
    z_on = dense_on.compute_impedance()[0]
    assert dense_on.kcl_nullspace_route == "reduced"
    assert _rel(z_on, z_off) < RTOL


def test_pair_reuse_composes():
    """A fill hint and the solve's reduction are independent: a Yagi whose
    reflector and driver carry junctions, with the director hint, on and off."""
    from test_pair_reuse_1809 import LAM, _director_hint, _yagi

    wires, n = _yagi(3)
    r_end = wires[0][1]
    d_end = wires[1][1]
    wires = wires + [
        np.array([r_end, r_end + (0.0, 0.0, 0.3)]),
        np.array([d_end, d_end + (0.0, 0.2, 0.0)]),
        np.array([d_end, d_end + (0.0, 0.0, -0.25)]),
    ]
    n = n + [[3], [2], [2]]
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=n,
        feeds=[(1, None, 1.0 + 0j)],
        wavelength=LAM,
        wire_radius=0.006,
        junctions=[
            [(0, "end"), (5, "start")],
            [(1, "end"), (6, "start"), (7, "start")],
        ],
    )
    z_plain = BSplineSolver(**kw).compute_impedance()[0]
    s = BSplineSolver(**kw, pair_reuse=_director_hint(3), kcl_nullspace=True)
    z_both = s.compute_impedance()[0]
    assert s.kcl_nullspace_route == "reduced"
    assert _rel(z_both, z_plain) < RTOL
