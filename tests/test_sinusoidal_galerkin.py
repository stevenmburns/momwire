"""Milestones M1/M2 gates (momwire#182): sinusoidal **Galerkin** solver.

**G1** — a correct Galerkin fill produces a symmetric system matrix (exact
reciprocity), so ``‖Z − Zᵀ‖ / ‖Z‖`` must collapse to ~1e-10. Collocation
cannot fake that symmetry, and quadrature sloppiness breaks it — both
directions are asserted here.

**G2** — the near-singular test integrals are treated well enough that the
driving-point impedance agrees with dense ``BSplineSolver`` and PyNEC within
the band the point-matched solver occupies on the same geometries, and is
quadrature-*converged* rather than tuned (stable under doubling the rules).

The M1 fill scheme reuses the closed-form Eqs 76-79 source-field evaluators
and quadratures only the test integral. M1 established that this is
structurally a correct Galerkin fill but needed ``n_qp_test`` ≈ 128 — on all
N² pairs — to reach the G1 gate, because a uniform rule cannot resolve the
width-``a`` endpoint spike that node-sharing pairs put in the integrand.

M2 replaces that global refinement with a targeted split: far pairs keep the
cheap uniform rule, near pairs are re-integrated on an endpoint-graded
composite rule. The headline result pinned below is that **G1 now passes at
the DEFAULT quadrature** — the gate no longer costs 16× the fill.

Measured on this geometry set at the defaults (n_qp_test=8, n_qp_near=8):

    geometry      ‖G-Gᵀ‖/‖G‖   M1 same nq   M1 nq=128 (its converged setting)
    dipole          8.3e-12      1.7e-06      1.4e-11
    vee             1.2e-11      7.8e-05      1.8e-11
    k2_junction     1.4e-11      3.7e-05      2.0e-11
    k3_star         2.9e-11      1.4e-04      1.9e-11

i.e. M2 reaches M1's converged floor (which is the source-side ``n_qp_const``
quadrature, not the test rule) at 7-13× less fill time — 35-60 ms here against
237-771 ms for the M1 setting that first met the gate.
"""

import numpy as np
import pytest

from momwire import (
    BSplineSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
)
from momwire.sinusoidal_galerkin import _graded_endpoint_rule

WL = 22.0
FREQ_MHZ = 299792458.0 / WL / 1e6
HD = 0.962 * WL / 4  # halfdriver ~5.29 m at the solver-default wavelength 22

# G1 gate value, and the level a uniform rule must FAIL to reach at the same
# node count for the "sloppiness breaks it" contrast to mean anything.
G1_GATE = 1e-10
UNIFORM_FLOOR = 1e-6


def _sym_ratio(G):
    return np.linalg.norm(G - G.T) / np.linalg.norm(G)


def _matrix(sim):
    geom = sim._build_geometry()
    G, _ = sim._assemble_Z(geom, sim.k)
    return G


# ---------------------------------------------------------------------------
# Geometries, each paired with its PyNEC deck where one is used
# ---------------------------------------------------------------------------

DIP_A, DIP_B = (0.0, -HD, 0.0), (0.0, HD, 0.0)


def _dipole(**over):
    dip = np.array([DIP_A, DIP_B])
    return dict(wires=[dip], n_per_edge_per_wire=[[41]], nsegs=41, **over)


_A30 = np.radians(30)
VEE_A = (0.0, -HD * np.cos(_A30), -HD * np.sin(_A30))
VEE_B = (0.0, 0.0, 0.0)
VEE_C = (0.0, +HD * np.cos(_A30), -HD * np.sin(_A30))


def _vee(**over):
    vee = np.array([VEE_A, VEE_B, VEE_C])
    return dict(wires=[vee], n_per_edge_per_wire=[[20, 20]], nsegs=40, **over)


K2_A, K2_B, K2_C = (0.0, -5.0, 0.0), (0.0, 0.0, -2.0), (0.0, 5.0, 0.0)
# 31 per wire, NOT 15: at 15 both sinusoidal solvers are still ~12% off their
# own refined answer (154.8 → 137.5 by N=31), so a G2 gate there would be
# measuring mesh error rather than the singular-integral treatment M2 changes.
K2_N = 31


def _k2_junction(**over):
    return dict(
        wires=[np.array([K2_A, K2_B]), np.array([K2_B, K2_C])],
        n_per_edge_per_wire=[[K2_N], [K2_N]],
        feed_wire_index=0,
        feed_arclength=2.5,
        nsegs=K2_N,
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

# (momwire kwargs factory, NEC wires, NEC feed tag, NEC feed segment 1-based)
G2_CASES = {
    "dipole": (_dipole, [(1, 41, DIP_A, DIP_B)], 1, 21),
    "vee": (_vee, [(1, 20, VEE_A, VEE_B), (2, 20, VEE_B, VEE_C)], 2, 1),
    # feed_arclength 2.5 on the 5.385 m first wire lands on momwire segment
    # K2_N//2 - 1 (0-based), i.e. NEC's 1-based K2_N//2.
    "k2_junction": (
        _k2_junction,
        [(1, K2_N, K2_A, K2_B), (2, K2_N, K2_B, K2_C)],
        1,
        K2_N // 2,
    ),
}


def _pynec_z(wires_nec, feed_tag, feed_seg, radius=0.0005):
    """Free-space driving-point impedance from PyNEC on the twin deck."""
    nec = pytest.importorskip("PyNEC")
    c = nec.nec_context()
    geo = c.get_geometry()
    for tag, n, p0, p1 in wires_nec:
        geo.wire(tag, n, *p0, *p1, radius, 1.0, 1.0)
    c.geometry_complete(0)
    c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)
    c.ex_card(0, feed_tag, feed_seg, 0, 1.0, 0.0, 0, 0, 0, 0)
    c.fr_card(0, 1, FREQ_MHZ, 0)
    c.xq_card(0)
    return complex(c.get_input_parameters(0).get_impedance()[0])


# ---------------------------------------------------------------------------
# G1 — the symmetry gate, now met at default quadrature (M2's headline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_g1_galerkin_matrix_is_symmetric(geom_name):
    """G1: the Galerkin system matrix is symmetric to 1e-10 (exact reciprocity)
    at the DEFAULT quadrature. Under M1's uniform rule this same assertion
    needed n_qp_test=128."""
    r = _sym_ratio(_matrix(SinusoidalGalerkinSolver(**GEOMETRIES[geom_name]())))
    assert r < G1_GATE, f"{geom_name}: ‖G-Gᵀ‖/‖G‖ = {r:.3e} exceeds the G1 gate"


@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_near_correction_is_what_buys_the_gate(geom_name):
    """The graded near-pair rule — not extra nodes everywhere — is what closes
    G1. At identical `n_qp_test`, disabling the correction (M1 behaviour)
    leaves the matrix visibly asymmetric."""
    mk = GEOMETRIES[geom_name]
    r_off = _sym_ratio(_matrix(SinusoidalGalerkinSolver(**mk(near_correction=False))))
    r_on = _sym_ratio(_matrix(SinusoidalGalerkinSolver(**mk())))
    assert r_off > UNIFORM_FLOOR, (
        f"{geom_name}: uniform rule unexpectedly symmetric ({r_off:.2e}) — the "
        "M1-vs-M2 contrast this gate rests on has gone away"
    )
    assert r_on < G1_GATE < r_off


@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_uniform_rule_symmetry_is_quadrature_limited(geom_name):
    """With the correction OFF, the G1 residual is test-quadrature error, not a
    broken fill: refining `n_qp_test` drives ‖G-Gᵀ‖/‖G‖ monotonically down
    toward the gate. A structural (fill) bug would floor the asymmetry at a
    level refinement cannot cross. (The 'quadrature sloppiness breaks it' half
    of G1, preserved from M1.)"""
    mk = GEOMETRIES[geom_name]
    ratios = [
        _sym_ratio(
            _matrix(SinusoidalGalerkinSolver(**mk(n_qp_test=nq, near_correction=False)))
        )
        for nq in (4, 16, 64)
    ]
    assert ratios[0] > UNIFORM_FLOOR, (
        f"{geom_name}: coarse quad unexpectedly symmetric ({ratios[0]:.2e})"
    )
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
    assert _sym_ratio(_matrix(sim)) > 1e-3, (
        "collocation matrix unexpectedly near-symmetric"
    )


# ---------------------------------------------------------------------------
# G2 — accuracy against the external references, and quadrature convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geom_name", list(G2_CASES))
def test_g2_matches_bspline_at_least_as_well_as_collocation(geom_name):
    """G2, dense-BSpline half: against the other Galerkin scheme in the tree,
    sinusoidal-Galerkin is at least as close as the point-matched solver is.

    Measured deltas — galerkin / collocation: dipole 0.08%/0.28%,
    vee 0.11%/1.15%, k2_junction 1.15%/1.19%.
    """
    mk = G2_CASES[geom_name][0]
    z_gk, _ = SinusoidalGalerkinSolver(**mk()).compute_impedance()
    z_co, _ = SinusoidalSolver(**mk()).compute_impedance()
    z_bs, _ = BSplineSolver(**mk(), degree=2).compute_impedance()
    d_gk = abs(z_gk - z_bs) / abs(z_bs)
    d_co = abs(z_co - z_bs) / abs(z_bs)
    assert d_gk <= d_co + 5e-3, (
        f"{geom_name}: galerkin is {d_gk:.4%} from dense bspline, worse than "
        f"collocation's {d_co:.4%}"
    )


@pytest.mark.parametrize("geom_name", list(G2_CASES))
def test_g2_tracks_pynec_within_the_collocation_band(geom_name):
    """G2, PyNEC half: sinusoidal-Galerkin lands within ~1.5% of NEC-2.

    It deliberately does NOT match the point-matched solver's ~0.08% NEC
    tracking — that tightness is collocation's NEC heritage (same basis, same
    point matching), and issue #182 lists "will NOT track PyNEC as closely"
    as an explicit non-goal, not a defect. Measured: dipole 0.27%, vee 1.02%,
    k2_junction 0.20%.
    """
    mk, wires, tag, seg = G2_CASES[geom_name]
    z_nec = _pynec_z(wires, tag, seg)
    z_gk, _ = SinusoidalGalerkinSolver(**mk()).compute_impedance()
    d = abs(z_gk - z_nec) / abs(z_nec)
    assert d < 1.5e-2, (
        f"{geom_name}: galerkin {z_gk:.4f} vs pynec {z_nec:.4f} = {d:.4%}"
    )


@pytest.mark.parametrize("geom_name", list(G2_CASES))
@pytest.mark.parametrize("radius", [0.0005, 0.005, 0.05])
def test_g2_impedance_is_quadrature_converged(geom_name, radius):
    """G2's "converged, not tuned" clause: doubling BOTH quadrature knobs moves
    the impedance by <0.5%. Swept over radius because the near-pair integrand's
    feature width IS the radius — the graded rule derives its panel count from
    a/h, so this also checks that derivation across two decades (#156's
    fat-radius territory). Measured shifts are ≤1e-8."""
    mk = G2_CASES[geom_name][0]
    z1, _ = SinusoidalGalerkinSolver(**mk(wire_radius=radius)).compute_impedance()
    z2, _ = SinusoidalGalerkinSolver(
        **mk(wire_radius=radius, n_qp_test=16, n_qp_near=16)
    ).compute_impedance()
    shift = abs(z2 - z1) / abs(z1)
    assert shift < 5e-3, f"{geom_name} a={radius}: {shift:.3%} shift on doubling n_qp"


@pytest.mark.parametrize("geom_name", list(G2_CASES))
def test_g2_symmetry_holds_across_the_radius_sweep(geom_name):
    """The G1 gate is not a thin-wire accident: it holds as a/h sweeps from
    ~0.004 to ~1.5, i.e. from very thin to fatter than NEC-2 will even accept
    (PyNEC refuses the last radius on the bent geometries). Symmetry in fact
    IMPROVES as the wire fattens — the endpoint spike the graded rule targets
    is regularized by a, so it broadens as a grows."""
    mk = G2_CASES[geom_name][0]
    ratios = [
        _sym_ratio(_matrix(SinusoidalGalerkinSolver(**mk(wire_radius=r))))
        for r in (0.0005, 0.005, 0.05, 0.2)
    ]
    assert all(r < G1_GATE for r in ratios), (
        f"{geom_name}: G1 lost over the radius sweep: {[f'{r:.2e}' for r in ratios]}"
    )
    assert ratios[0] > ratios[-1], (
        f"{geom_name}: fattening the wire did not ease the near-singular "
        f"integral as expected: {[f'{r:.2e}' for r in ratios]}"
    )


@pytest.mark.parametrize(
    "geom_name,radius",
    [("dipole", 0.0005), ("k2_junction", [0.0005, 0.004])],
)
def test_near_correction_reproduces_brute_force_refinement(geom_name, radius):
    """The correction's own correctness check, independent of symmetry: the
    graded near-pair rule must land on the SAME matrix that simply refining the
    uniform rule converges to. This is the strongest available statement — it
    validates the graded rule against the honest reference rather than against
    a property (symmetry) that the rule could in principle satisfy for the
    wrong reason.

    The uniform reference is converged by n_qp_test=128 (its entries stop
    moving by 1e-9 from there to 512). The mixed-radius case is included
    because the correction carries its own observer-radius column — a wrong
    radius there would be invisible on every uniform-radius geometry.
    """
    mk = GEOMETRIES[geom_name]
    sim = SinusoidalGalerkinSolver(**mk(wire_radius=radius))
    G_corrected = _matrix(sim)
    G_brute = _matrix(
        SinusoidalGalerkinSolver(
            **mk(wire_radius=radius, near_correction=False, n_qp_test=128)
        )
    )
    rel = np.linalg.norm(G_corrected - G_brute) / np.linalg.norm(G_brute)
    assert rel < 1e-7, f"{geom_name} a={radius}: correction differs by {rel:.3e}"


def test_mixed_per_wire_radii_break_reciprocity_by_construction():
    """G1 is a UNIFORM-radius property, and that is a statement about NEC's
    thin-wire kernel rather than about the fill.

    The Eqs 76-79 evaluators regularize with the OBSERVER segment's radius
    (rh = √(ρ² + a_obs²)), so with mixed radii the (i,j) entry is regularized
    by a_i and the (j,i) entry by a_j: the kernel itself is asymmetric, and no
    testing scheme can restore a symmetry the kernel does not have. Pinned
    here so the ~8% asymmetry below is never mistaken for a quadrature bug —
    brute-force refinement reproduces it exactly (that equality is what
    `test_near_correction_reproduces_brute_force_refinement` asserts).
    """
    sim = SinusoidalGalerkinSolver(**_k2_junction(wire_radius=[0.0005, 0.004]))
    assert sim._uniform_radius is None  # the mixed path is genuinely taken
    r_mixed = _sym_ratio(_matrix(sim))
    r_uniform = _sym_ratio(_matrix(SinusoidalGalerkinSolver(**_k2_junction())))
    assert r_uniform < G1_GATE < r_mixed


# ---------------------------------------------------------------------------
# The M2 machinery itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [0.5, 0.05, 4e-3, 1e-5])
def test_graded_rule_is_a_valid_partition_of_the_test_segment(eps):
    """The graded rule must remain an exact quadrature: nodes inside [-1, 1],
    weights summing to the interval length, and polynomials integrated exactly
    (each panel carries a full Gauss rule, so composite exactness is 2n-1)."""
    x, w = _graded_endpoint_rule(eps, 8, np.polynomial.legendre.leggauss)
    assert x.size == w.size and np.all(np.abs(x) <= 1.0)
    assert np.isclose(w.sum(), 2.0, rtol=0, atol=1e-14)
    for p, exact in ((0, 2.0), (2, 2 / 3), (4, 2 / 5), (6, 2 / 7)):
        assert np.isclose((w * x**p).sum(), exact, rtol=1e-13)


def test_graded_rule_resolves_the_endpoint_scale():
    """Panels must shrink to the feature width at the ends while staying coarse
    in the middle — that is the whole point of grading over refining."""
    x, _ = _graded_endpoint_rule(1e-3, 8, np.polynomial.legendre.leggauss)
    # Nodes land within the first eps-wide panel at each end...
    assert (x < -1.0 + 1e-3).any() and (x > 1.0 - 1e-3).any()
    # ...without carpeting the interior: a uniform rule of the same size would
    # put ~x.size/2 nodes in |x| < 0.5; grading puts far fewer.
    assert (np.abs(x) < 0.5).sum() < 0.25 * x.size


def test_near_pairs_selects_self_and_node_sharing_only():
    """On a straight uniformly-meshed dipole the geometric criterion must pick
    exactly the self pair and the two collinear neighbours — the pairs whose
    test integrand carries the width-`a` endpoint spike — and nothing further
    out. The set must also be symmetric, or the corrected matrix could not be."""
    sim = SinusoidalGalerkinSolver(**_dipole())
    geom = sim._build_geometry()
    mm, nn = sim._near_pairs(geom)
    n_segs = geom["n_segs"]
    pairs = set(zip(mm.tolist(), nn.tolist()))
    expected = {(i, j) for i in range(n_segs) for j in range(n_segs) if abs(i - j) <= 1}
    assert pairs == expected
    assert pairs == {(j, i) for i, j in pairs}, "near-pair set is not symmetric"


def test_near_pairs_catches_close_approach_without_a_shared_node():
    """Selection is geometric, not topological: two parallel wires closer than
    a segment length put a sub-segment scale into each other's test integrand
    even though they share no junction, and must be corrected too."""
    y = np.linspace(-2.0, 2.0, 2)
    gap = 0.05  # << the 0.4 m segment length below
    sim = SinusoidalGalerkinSolver(
        wires=[
            np.array([[0.0, y[0], 0.0], [0.0, y[1], 0.0]]),
            np.array([[gap, y[0], 0.0], [gap, y[1], 0.0]]),
        ],
        n_per_edge_per_wire=[[10], [10]],
        nsegs=10,
    )
    geom = sim._build_geometry()
    mm, nn = sim._near_pairs(geom)
    pairs = set(zip(mm.tolist(), nn.tolist()))
    # segment 0 of wire 0 (index 0) and segment 0 of wire 1 (index 10) face
    # each other across the gap with no shared node.
    assert (0, 10) in pairs and (10, 0) in pairs


# ---------------------------------------------------------------------------
# Sanity + deferred features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_impedance_within_sanity_band_of_bspline(geom_name):
    """A correct fill lands the driving-point impedance in a loose band around
    dense BSplineSolver on every validation geometry."""
    z_gk, alpha = SinusoidalGalerkinSolver(
        **GEOMETRIES[geom_name]()
    ).compute_impedance()
    z_bs, _ = BSplineSolver(**GEOMETRIES[geom_name](), degree=2).compute_impedance()
    assert np.isfinite(z_gk.real) and np.isfinite(z_gk.imag)
    assert np.isfinite(alpha).all()
    rel = abs(z_gk - z_bs) / abs(z_bs)
    assert rel < 0.15, f"{geom_name}: |Zgk-Zbs|/|Zbs| = {rel:.3%} outside sanity band"


def test_ground_and_loading_are_not_yet_supported():
    """Free-space, lossless: the deferred features fail loudly rather than
    silently producing wrong numbers (ground → M4, loading → later)."""
    # Elevated dipole so the free-space geometry build succeeds and the
    # NotImplementedError comes from the Galerkin assembly, not the in-plane
    # ground degeneracy check.
    dip_hi = np.array([[0.0, -HD, 2.0], [0.0, HD, 2.0]])
    common = dict(wires=[dip_hi], n_per_edge_per_wire=[[41]], nsegs=41)
    with pytest.raises(NotImplementedError):
        SinusoidalGalerkinSolver(**common, ground_z=0.0).compute_impedance()
    with pytest.raises(NotImplementedError):
        SinusoidalGalerkinSolver(**common, wire_conductivity=5.8e7).compute_impedance()
