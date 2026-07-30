"""Milestone M1 gate (momwire#182): sinusoidal **Galerkin** solver.

The load-bearing gate is **G1**: a correct Galerkin fill produces a symmetric
system matrix (exact reciprocity), so ``‖Z − Zᵀ‖ / ‖Z‖`` must collapse to
~1e-10. Collocation cannot fake that symmetry, and quadrature sloppiness
breaks it — both directions are asserted here.

The chosen M1 fill scheme (issue #182, "let G1 decide") reuses the closed-form
Eqs 76-79 source-field evaluators and Gauss-quadratures only the test integral.
The empirical finding this test pins: that scheme is a structurally correct
Galerkin fill — symmetry is exact on the diagonal and at roundoff for
``|i-j| ≥ 2`` — and the entire residual lives on the near-singular ADJACENT
test integral, which converges to <1e-10 as the test quadrature is refined
(``n_qp_test`` ≈ 128 under naive quadrature; M2 replaces that near-singular
integral with Richmond closed forms so the same 1e-10 is reached cheaply).
"""

import numpy as np
import pytest

from momwire import (
    BSplineSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
)

HD = 0.962 * 22 / 4  # halfdriver ~5.29 m at the solver-default wavelength 22

# n_qp_test that drives naive-quadrature symmetry below 1e-10 on every
# validation geometry (bent/junction adjacency converges slower than the
# straight dipole; see the module docstring). M2's closed forms will make
# this unnecessary.
NQP_CONVERGED = 128


def _sym_ratio(G):
    return np.linalg.norm(G - G.T) / np.linalg.norm(G)


def _dipole(**over):
    dip = np.array([[0.0, -HD, 0.0], [0.0, HD, 0.0]])
    return dict(wires=[dip], n_per_edge_per_wire=[[41]], nsegs=41, **over)


def _vee(**over):
    L = 2 * HD
    half = L / 2
    a = np.radians(30)
    vee = np.array(
        [
            [0.0, -half * np.cos(a), -half * np.sin(a)],
            [0.0, 0.0, 0.0],
            [0.0, +half * np.cos(a), -half * np.sin(a)],
        ]
    )
    return dict(wires=[vee], n_per_edge_per_wire=[[20, 20]], nsegs=40, **over)


def _k2_junction(**over):
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    return dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        nsegs=15,
        junctions=[[(0, "end"), (1, "start")]],
        **over,
    )


def _k3_star(**over):
    L = 3.0
    arms = [
        np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]]),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [L * np.cos(np.radians(210)), L * np.sin(np.radians(210)), 0.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [L * np.cos(np.radians(330)), L * np.sin(np.radians(330)), 0.0],
            ]
        ),
    ]
    return dict(
        wires=arms,
        n_per_edge_per_wire=[[12], [12], [12]],
        feed_wire_index=0,
        feed_arclength=1.5,
        nsegs=12,
        junctions=[[(0, "start"), (1, "start"), (2, "start")]],
        **over,
    )


GEOMETRIES = {
    "dipole": _dipole,
    "vee": _vee,
    "k2_junction": _k2_junction,
    "k3_star": _k3_star,
}


# ---------------------------------------------------------------------------
# G1 — the milestone gate
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_g1_galerkin_matrix_is_symmetric(geom_name):
    """G1: the Galerkin system matrix is symmetric to 1e-10 (exact reciprocity)
    once the near-singular adjacent test integral is quadrature-converged."""
    sim = SinusoidalGalerkinSolver(**GEOMETRIES[geom_name](n_qp_test=NQP_CONVERGED))
    geom = sim._build_geometry()
    G, _ = sim._assemble_Z(geom, sim.k)
    r = _sym_ratio(G)
    assert r < 1e-10, f"{geom_name}: ‖G-Gᵀ‖/‖G‖ = {r:.3e} exceeds the G1 gate"


@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_g1_symmetry_is_quadrature_limited(geom_name):
    """The G1 residual is test-quadrature error, not a broken fill: refining
    ``n_qp_test`` drives ‖G-Gᵀ‖/‖G‖ monotonically down toward the gate. A
    structural (fill) bug would floor the asymmetry at some level refinement
    cannot cross. (The 'quadrature sloppiness breaks it' half of G1.)"""
    ratios = []
    for nq in (4, 16, 64):
        sim = SinusoidalGalerkinSolver(**GEOMETRIES[geom_name](n_qp_test=nq))
        geom = sim._build_geometry()
        G, _ = sim._assemble_Z(geom, sim.k)
        ratios.append(_sym_ratio(G))
    # Coarse quadrature is visibly asymmetric ("sloppiness breaks it")...
    assert ratios[0] > 1e-6, (
        f"{geom_name}: coarse quad unexpectedly symmetric ({ratios[0]:.2e})"
    )
    # ...and refinement strictly and monotonically reduces it toward the gate.
    assert ratios[0] > ratios[1] > ratios[2], (
        f"{geom_name}: asymmetry not monotonically decreasing under "
        f"refinement: {[f'{r:.2e}' for r in ratios]}"
    )


@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_collocation_matrix_is_not_symmetric(geom_name):
    """Contrast: the point-matched SinusoidalSolver assembles a NON-symmetric
    matrix on the same geometry — symmetry is a property of the Galerkin
    testing, which collocation cannot fake."""
    sim = SinusoidalSolver(**GEOMETRIES[geom_name]())
    geom = sim._build_geometry()
    G, _ = sim._assemble_Z(geom, sim.k)
    assert _sym_ratio(G) > 1e-3, "collocation matrix unexpectedly near-symmetric"


# ---------------------------------------------------------------------------
# Loose accuracy sanity (G1's "within ~15% of dense BSplineSolver" clause)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_impedance_within_sanity_band_of_bspline(geom_name):
    """M1 is about structure, not accuracy — but a correct fill must at least
    land the driving-point impedance in a loose band around dense
    BSplineSolver at the default quadrature."""
    z_gk, alpha = SinusoidalGalerkinSolver(
        **GEOMETRIES[geom_name]()
    ).compute_impedance()
    z_bs, _ = BSplineSolver(**GEOMETRIES[geom_name](), degree=2).compute_impedance()
    assert np.isfinite(z_gk.real) and np.isfinite(z_gk.imag)
    assert np.isfinite(alpha).all()
    rel = abs(z_gk - z_bs) / abs(z_bs)
    assert rel < 0.15, f"{geom_name}: |Zgk-Zbs|/|Zbs| = {rel:.3%} outside sanity band"


def test_ground_and_loading_are_not_yet_supported():
    """M1 is free-space, lossless: the deferred features fail loudly rather
    than silently producing wrong numbers (ground → M4, loading → later)."""
    # Elevated dipole so the free-space geometry build succeeds and the
    # NotImplementedError comes from the Galerkin assembly, not the in-plane
    # ground degeneracy check.
    dip_hi = np.array([[0.0, -HD, 2.0], [0.0, HD, 2.0]])
    common = dict(wires=[dip_hi], n_per_edge_per_wire=[[41]], nsegs=41)
    with pytest.raises(NotImplementedError):
        SinusoidalGalerkinSolver(**common, ground_z=0.0).compute_impedance()
    with pytest.raises(NotImplementedError):
        SinusoidalGalerkinSolver(**common, wire_conductivity=5.8e7).compute_impedance()
