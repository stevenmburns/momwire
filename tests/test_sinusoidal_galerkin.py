"""Milestones M1/M2/M3 gates (momwire#182): sinusoidal **Galerkin** solver.

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

**G3** — the variational payoff, as a test: at coarse mesh the Galerkin
impedance is closer to the fine-mesh answer than the point-matched solver's is,
and the convergence curve is monotone. See the ``M3`` section below for the
gate, the two experimental-design constraints it turned up (the feed must not
move between mesh levels; the delta-gap reactance has no mesh limit, so the
"converged reference" is a *choice* whose influence has to be bounded), and the
first basis-vs-testing instrument readings.
"""

import functools

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


# ---------------------------------------------------------------------------
# M3 — the variational claim, as a test
#
# Two experimental-design constraints had to be established before the gate
# could mean anything; both are asserted below rather than merely asserted-to.
#
# 1. **The feed must not move between mesh levels.** ``feed_arclength`` snaps to
#    the segment whose CENTRE is nearest the request, so unless that point is a
#    segment centre at every N in the sweep, refining the mesh also translates
#    the delta-gap by up to h/2. That is an O(h) perturbation of the *problem*,
#    and it dwarfs the difference between two testing schemes solving the same
#    problem. The M1/M2 geometries are not built for this (the vee's default
#    feed is its apex — a node for every mesh, so the gap creeps toward it as N
#    grows), hence the separate M3 factories below: odd segment counts with the
#    feed at a wire/edge midpoint, which is a segment centre for every odd N.
#
# 2. **The delta-gap reactance has no mesh limit**, because the gap width IS
#    the segment length: refining the mesh also shrinks the source, and the
#    gap's near-field energy grows like ln(1/h). R settles by N≈161; X drifts
#    logarithmically for as long as the thin-wire kernel stays valid. So "the
#    converged reference" the milestone asks for does not exist as a unique
#    number, and every error quoted here is an error *relative to a chosen
#    reference*. The gate is therefore stated against a whole family of
#    defensible references at once — the verdict is the property that survives
#    all of them.
#
# Measured with those two constraints honoured (errColl/errGal on |Z|, worst
# case over the six references, at N = 11/15/21):
#
#     geometry      ratio    reading
#     dipole         2.06    testing-limited
#     vee            1.41    testing-limited
#     k2_junction    2.97    testing-limited (strongest payoff)
#     k3_star        1.01    BASIS-limited — testing buys nothing here
#
# The k3_star row is the instrument doing its job: the sinusoidal basis, not
# the testing scheme, is what limits that geometry (swapping the basis for
# B-splines buys 1.4×; swapping the testing buys 1.01×). It is the high-Q
# near-open case — antennaknobs#478's territory.
# ---------------------------------------------------------------------------

# Coarse meshes the gate is stated at (the milestone's "e.g. 11-21 segments"),
# and the wider ladder the monotonicity check walks.
M3_COARSE = (11, 15, 21)
M3_LADDER = (11, 15, 21, 31, 41)


def _m3_dipole(n, **over):
    """Feed defaults to the wire midpoint = the centre segment for odd n."""
    dip = np.array([DIP_A, DIP_B])
    return dict(wires=[dip], n_per_edge_per_wire=[[n]], nsegs=n, **over)


def _m3_vee(n, **over):
    """Fed at the midpoint of the FIRST edge, deliberately not at the apex:
    the apex is a node for every mesh, so the default (total/2) feed would sit
    h/2 away from it and creep as n grows — constraint 1 above."""
    vee = np.array([VEE_A, VEE_B, VEE_C])
    return dict(
        wires=[vee],
        n_per_edge_per_wire=[[n, n]],
        nsegs=2 * n,
        feed_arclength=0.5 * HD,
        **over,
    )


def _m3_k2(n, **over):
    """Feed defaults to the midpoint of wire 0 — a segment centre for odd n.
    (The M1/M2 ``_k2_junction`` pins feed_arclength=2.5, which is NOT a segment
    centre at general n and so jitters by up to h/2 across a sweep.)"""
    return dict(
        wires=[np.array([K2_A, K2_B]), np.array([K2_B, K2_C])],
        n_per_edge_per_wire=[[n], [n]],
        feed_wire_index=0,
        nsegs=n,
        junctions=[[(0, "end"), (1, "start")]],
        **over,
    )


def _m3_k3(n, **over):
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
        n_per_edge_per_wire=[[n], [n], [n]],
        feed_wire_index=0,
        nsegs=n,
        junctions=[[(0, "start"), (1, "start"), (2, "start")]],
        **over,
    )


M3_GEOMETRIES = {
    "dipole": _m3_dipole,
    "vee": _m3_vee,
    "k2_junction": _m3_k2,
    "k3_star": _m3_k3,
}

# Fine-mesh reference impedances, measured at N = 161/241/321. `sin_*_321` are
# straight fine solves; `rich_*` are Richardson extrapolations (Z = Z∞ + c/N
# fitted on the last two). They span 0.9e-3 to 2.8e-3 of |Z| — that spread IS
# the reference ambiguity constraint 2 describes, and the gate below requires
# its verdict to hold at every corner of it. Re-derived by the `slow`
# reproduction test at the bottom, so these cannot rot silently.
M3_REFS = {
    "dipole": dict(
        sin_gal_321=69.639094 - 18.056294j,
        sin_coll_321=69.631876 - 18.107822j,
        bspline2_321=69.633780 - 18.065315j,
        rich_sin_gal=69.634798 - 18.007989j,
        rich_sin_coll=69.633551 - 18.018862j,
        rich_bspline2=69.633701 - 18.010747j,
    ),
    "vee": dict(
        sin_gal_321=97.960367 - 61.297119j,
        sin_coll_321=97.947410 - 61.346335j,
        bspline2_321=97.945040 - 61.306749j,
        rich_sin_gal=97.934880 - 61.214816j,
        rich_sin_coll=97.934600 - 61.216284j,
        rich_bspline2=97.930347 - 61.214625j,
    ),
    "k2_junction": dict(
        sin_gal_321=124.493290 + 0.392380j,
        sin_coll_321=124.479522 + 0.340718j,
        bspline2_321=124.492289 + 0.367751j,
        rich_sin_gal=124.513127 + 0.444632j,
        rich_sin_coll=124.513126 + 0.445724j,
        rich_bspline2=124.514726 + 0.445185j,
    ),
    "k3_star": dict(
        sin_gal_321=13.438931 - 951.615461j,
        sin_coll_321=13.438415 - 951.658449j,
        bspline2_321=13.416848 - 950.833391j,
        rich_sin_gal=13.379066 - 949.328997j,
        rich_sin_coll=13.380080 - 949.324632j,
        rich_bspline2=13.369902 - 949.004252j,
    ),
}

# Worst-case errColl/errGal over the whole reference family at M3_COARSE,
# floored a little below the measured value so the gate pins the finding
# without pinning the last digit of a float.
M3_GATE_RATIO = {
    "dipole": 2.0,
    "vee": 1.38,
    "k2_junction": 2.9,
    "k3_star": 1.01,
}

# N from which |Z(N) − Z_ref| decreases monotonically. Three geometries are
# monotone across the whole ladder; the junction is not — see
# `test_k2_pre_asymptotic_error_maximum_is_not_quadrature`.
M3_MONOTONE_FROM = {"dipole": 11, "vee": 11, "k2_junction": 21, "k3_star": 11}

_SCHEMES = {
    "gal": lambda kw: SinusoidalGalerkinSolver(**kw),
    "coll": lambda kw: SinusoidalSolver(**kw),
    "bspl": lambda kw: BSplineSolver(**kw, degree=2),
}


@functools.lru_cache(maxsize=None)
def _m3_z(scheme, geom_name, n):
    """Driving-point impedance, memoized — the M3 tests walk the same handful
    of (scheme, geometry, N) solves several times over."""
    kw = M3_GEOMETRIES[geom_name](n)
    return _SCHEMES[scheme](kw).compute_impedance()[0]


def _rel_err(z, ref):
    return abs(z - ref) / abs(ref)


# --- G3: the payoff gate ---------------------------------------------------


@pytest.mark.parametrize("geom_name", list(M3_GEOMETRIES))
def test_g3_galerkin_beats_collocation_at_coarse_mesh(geom_name):
    """G3: at 11-21 segments the Galerkin impedance error is strictly smaller
    than the point-matched solver's on the same geometry and mesh.

    This is the milestone's payoff-check: Galerkin's impedance is stationary to
    first order in the current error, so it should buy accuracy at coarse mesh
    even though it costs more per fill. Failure here would mean something is
    wrong even with G1/G2 green.
    """
    ref = M3_REFS[geom_name]["rich_sin_gal"]
    for n in M3_COARSE:
        e_gal = _rel_err(_m3_z("gal", geom_name, n), ref)
        e_coll = _rel_err(_m3_z("coll", geom_name, n), ref)
        assert e_gal < e_coll, (
            f"{geom_name} N={n}: galerkin {e_gal:.4%} is not better than "
            f"collocation {e_coll:.4%} — the variational payoff is absent"
        )


@pytest.mark.parametrize("geom_name", list(M3_GEOMETRIES))
def test_g3_verdict_survives_every_candidate_reference(geom_name):
    """The G3 verdict must not be an artifact of which reference was picked.

    It cannot simply be read off "the converged answer", because there isn't
    one: the delta-gap reactance drifts logarithmically forever (see
    `test_delta_gap_reactance_has_no_mesh_limit`). So the comparison is run
    against all six defensible references at once — three fine solves and three
    Richardson extrapolations, spanning ~1e-3 of |Z| — and the gate is the
    WORST ratio over that family. Anything that survives the whole spread is a
    property of the schemes, not of the reference.
    """
    worst = min(
        _rel_err(_m3_z("coll", geom_name, n), ref)
        / _rel_err(_m3_z("gal", geom_name, n), ref)
        for ref in M3_REFS[geom_name].values()
        for n in M3_COARSE
    )
    assert worst > 1.0, (
        f"{geom_name}: the G3 verdict flips under some reference choice "
        f"(worst errColl/errGal = {worst:.3f}) — it is a reference artifact"
    )
    assert worst >= M3_GATE_RATIO[geom_name], (
        f"{geom_name}: worst-case errColl/errGal fell to {worst:.3f}, below "
        f"the pinned {M3_GATE_RATIO[geom_name]}"
    )


@pytest.mark.parametrize("geom_name", list(M3_GEOMETRIES))
def test_g3_convergence_is_monotone(geom_name):
    """G3's second clause: the Galerkin error decreases monotonically with mesh
    refinement. Asserted from `M3_MONOTONE_FROM` upward — which is the whole
    ladder except on the junction geometry, whose pre-asymptotic behaviour the
    next test characterizes rather than hides."""
    ref = M3_REFS[geom_name]["rich_sin_gal"]
    ns = [n for n in M3_LADDER if n >= M3_MONOTONE_FROM[geom_name]]
    errs = [_rel_err(_m3_z("gal", geom_name, n), ref) for n in ns]
    assert all(b < a for a, b in zip(errs, errs[1:])), (
        f"{geom_name}: galerkin error not monotone over N={ns}: "
        f"{[f'{e:.4%}' for e in errs]}"
    )


def test_k2_pre_asymptotic_error_maximum_is_not_quadrature():
    """The junction geometry's Galerkin error RISES from N=7 to N≈17 before
    decaying — the one place the monotonicity clause does not hold from the
    coarsest mesh, so it gets an explicit cause rather than a carve-out.

    It is not quadrature: refining BOTH rules 4× moves Z by ~1e-8, four orders
    below the feature. It is not the reference either: the raw sequence itself
    turns around (X: 0.302 → 0.247 at N=17 → 0.393 at N=321), so no choice of
    reference makes |Z(N) − ref| monotone there. What is left is discretization
    — a fixed ~0.08 Ω error component that decays on its own schedule. The
    reason collocation looks monotone on the same geometry is simply that its
    error is 3-8× larger and buries the same wobble.
    """
    ref = M3_REFS["k2_junction"]["rich_sin_gal"]
    coarse = (7, 11, 17)
    errs = [_rel_err(_m3_z("gal", "k2_junction", n), ref) for n in coarse]
    assert errs[0] < errs[1] < errs[2], (
        f"the documented pre-asymptotic rise is gone: {[f'{e:.4%}' for e in errs]}"
    )
    # ...and it is discretization, not an under-resolved integral.
    for n in coarse:
        z1 = _m3_z("gal", "k2_junction", n)
        z2 = SinusoidalGalerkinSolver(
            **_m3_k2(n, n_qp_test=32, n_qp_near=32)
        ).compute_impedance()[0]
        assert abs(z2 - z1) / abs(z1) < 1e-6, (
            f"N={n}: 4× quadrature moved Z by {abs(z2 - z1) / abs(z1):.2e} — the "
            "coarse-mesh maximum is quadrature error after all"
        )


def test_the_variational_payoff_is_in_the_reactance():
    """Where the G3 win actually comes from — pinned because the headline
    |Z| ratio hides it.

    Galerkin converges faster in X on every geometry, but in R it is *slower*
    than collocation on three of the four (dipole 0.47×, vee 0.61×, k3_star
    0.97×; only the junction wins on both). The net |Z| verdict is a reactance
    win partly given back on resistance.

    That is the useful direction: the open sin↔bs2 investigations this solver
    was built to arbitrate (antennaknobs#521, #478, momwire#156) are all
    reactance-dominated.
    """
    r_ratio, x_ratio = {}, {}
    for name in M3_GEOMETRIES:
        ref = M3_REFS[name]["rich_sin_gal"]
        n = M3_COARSE[0]
        zg, zc = _m3_z("gal", name, n), _m3_z("coll", name, n)
        r_ratio[name] = abs(zc.real - ref.real) / abs(zg.real - ref.real)
        x_ratio[name] = abs(zc.imag - ref.imag) / abs(zg.imag - ref.imag)
    assert all(v > 1.0 for v in x_ratio.values()), (
        f"reactance is where the payoff was measured to be: {x_ratio}"
    )
    assert all(r_ratio[g] < 1.0 for g in ("dipole", "vee", "k3_star")), (
        f"collocation was measured to converge FASTER in R on these: {r_ratio}"
    )
    assert r_ratio["k2_junction"] > 1.0, (
        f"the junction was the one geometry winning on both parts: {r_ratio}"
    )


# --- The instrument: attributing a gap to basis vs testing ------------------


def test_k3_star_gap_is_basis_limited_not_testing_limited():
    """The first reading the basis × testing matrix was built to produce.

    On the high-Q near-open star, swapping the TESTING (collocation → Galerkin,
    same basis) buys ~1%, while swapping the BASIS (sinusoidal → B-spline, both
    Galerkin) buys ~40%. So this geometry's coarse-mesh error is a property of
    the sinusoidal basis, and no amount of testing work will move it. This is
    antennaknobs#478's near-open high-Q territory, and the classification is
    the deliverable — it is explicitly not a defect to fix.
    """
    ref = M3_REFS["k3_star"]["rich_sin_gal"]
    n = 15
    e_gal = _rel_err(_m3_z("gal", "k3_star", n), ref)
    e_coll = _rel_err(_m3_z("coll", "k3_star", n), ref)
    e_bspl = _rel_err(_m3_z("bspl", "k3_star", n), ref)
    assert e_coll / e_gal < 1.05, f"testing unexpectedly matters here: {e_coll / e_gal}"
    assert e_gal / e_bspl > 1.25, (
        f"basis unexpectedly does not matter: {e_gal / e_bspl}"
    )


def test_k2_junction_gap_is_testing_limited():
    """The opposite reading, on the same axes: at the junction the testing
    scheme is what matters (3.5× at N=15) and the sinusoidal basis is not the
    limiter — sin-Galerkin is in fact several times closer than dense B-spline
    at the same mesh. Junction accuracy is a testing-scheme property, which is
    the encouraging direction for the M5 junction ports."""
    ref = M3_REFS["k2_junction"]["rich_sin_gal"]
    n = 15
    e_gal = _rel_err(_m3_z("gal", "k2_junction", n), ref)
    e_coll = _rel_err(_m3_z("coll", "k2_junction", n), ref)
    e_bspl = _rel_err(_m3_z("bspl", "k2_junction", n), ref)
    assert e_coll / e_gal > 3.0, f"testing stopped mattering here: {e_coll / e_gal}"
    assert e_gal < e_bspl, (
        f"sin-galerkin {e_gal:.4%} no longer beats dense bspline {e_bspl:.4%}"
    )


def test_all_three_schemes_agree_at_fine_mesh():
    """Precondition for every attribution above: sinusoidal-Galerkin,
    sinusoidal-collocation and B-spline-Galerkin land on the SAME answer once
    refined (≤4e-4 of |Z| between their extrapolations on every geometry).

    So the sin↔bs2 discrepancies these tests classify are convergence-RATE
    effects, not different answers — consistent with M2's finding that an 11%
    junction "basis effect" evaporated under refinement. Anything M6 reports as
    a basis effect is therefore a statement about coarse-mesh cost, not about
    what the schemes ultimately converge to.
    """
    for name, refs in M3_REFS.items():
        lim = [refs["rich_sin_gal"], refs["rich_sin_coll"], refs["rich_bspline2"]]
        spread = max(abs(a - b) for a in lim for b in lim) / abs(lim[0])
        assert spread < 4e-4, f"{name}: fine-mesh limits disagree by {spread:.2e}"


# --- The two experimental-design constraints, asserted ---------------------


def test_a_drifting_feed_destroys_the_convergence_sequence():
    """Constraint 1, demonstrated rather than asserted: `feed_arclength` snaps
    to the nearest segment CENTRE, so a feed point that is not a segment centre
    at every N moves as the mesh refines.

    The vee fed at its apex is the sharp case — the apex is a node for every
    mesh, so the gap sits h/2 away and creeps inward. Its collocation sequence
    turns around near N=31 and then walks AWAY from its own fine-mesh answer,
    which would read as "collocation does not converge" if taken at face value.
    Fed at an edge midpoint instead — a segment centre for every odd n — the
    same solver on the same wires is cleanly monotone. Any future convergence
    study (M4's grounds, M6's sweeps) has to honour this.
    """
    ns = (11, 21, 31, 61, 81)

    def vee_apex(n):  # default feed_arclength = total/2 = the apex
        return dict(
            wires=[np.array([VEE_A, VEE_B, VEE_C])],
            n_per_edge_per_wire=[[n, n]],
            nsegs=2 * n,
        )

    drift = [SinusoidalSolver(**vee_apex(n)).compute_impedance()[0].real for n in ns]
    fixed = [_m3_z("coll", "vee", n).real for n in ns]
    assert not all(b < a for a, b in zip(drift, drift[1:])), (
        f"the drifting-feed artifact this test guards against is gone: {drift}"
    )
    assert all(b < a for a, b in zip(fixed, fixed[1:])), (
        f"the fixed-feed sequence should be monotone: {fixed}"
    )


@pytest.mark.slow
def test_delta_gap_reactance_has_no_mesh_limit():
    """Constraint 2: R settles under refinement, X does not.

    The delta-gap's width IS the segment length, so refining the mesh shrinks
    the source too, and the gap's stored near-field energy grows like ln(1/h).
    The signature is in the successive-halving ratio of ΔX: a 1/N-convergent
    sequence gives 2.0, a logarithmic one gives 1.0. Measured on the dipole
    (N = 81 → 641): 1.401, 1.315, 1.168 — walking toward 1, not 2.

    R over the same range is not monotone either, but it is *settled*: it moves
    by only a few mΩ per doubling (−2.6, +0.8, +2.1 mΩ, i.e. ~4e-5 of |Z|,
    wandering inside a band rather than trending), while X is still walking
    60-110 mΩ per doubling in one direction. It is that ~20× separation, not
    monotonicity, that makes R the trustworthy part.

    This is why G3 is stated against a family of references: "the converged
    reactance" is not a number this model has. (Pushed no further than N=641:
    by N=2561 the segment length is only 8 radii and the thin-wire kernel is
    itself invalid, so those points measure nothing.)
    """
    ns = [81, 161, 321, 641]
    zs = [SinusoidalSolver(**_m3_dipole(n)).compute_impedance()[0] for n in ns]
    dX = [zs[i + 1].imag - zs[i].imag for i in range(len(zs) - 1)]
    dR = [abs(zs[i + 1].real - zs[i].real) for i in range(len(zs) - 1)]
    ratios = [dX[i] / dX[i + 1] for i in range(len(dX) - 1)]
    assert all(r < 1.6 for r in ratios), (
        f"ΔX halving ratios {ratios} no longer indicate a logarithmic drift"
    )
    assert all(b < a for a, b in zip(ratios, ratios[1:])), (
        f"ratios should keep falling toward the logarithmic 1.0: {ratios}"
    )
    assert max(dR) < 0.01, f"R was supposed to be settled to a few mΩ: {dR}"
    assert min(abs(x) for x in dX) > 10 * max(dR), (
        f"X should still be moving an order+ more than R wanders: dX={dX} dR={dR}"
    )


@pytest.mark.slow
def test_m3_reference_constants_are_reproducible():
    """The pinned `M3_REFS` are the only measured inputs the M3 gate trusts, so
    they are re-derived here from the N = 161/241/321 solves they came from.
    Slow (the 321-segment-per-wire Galerkin fills dominate), but it keeps the
    constants from rotting silently under a future fill change."""
    fine = (161, 241, 321)

    def richardson(zs):
        c = (zs[-2] - zs[-1]) / (1.0 / fine[-2] - 1.0 / fine[-1])
        return zs[-1] - c / fine[-1]

    for name, refs in M3_REFS.items():
        series = {s: [_m3_z(s, name, n) for n in fine] for s in ("gal", "coll", "bspl")}
        got = {
            "sin_gal_321": series["gal"][-1],
            "sin_coll_321": series["coll"][-1],
            "bspline2_321": series["bspl"][-1],
            "rich_sin_gal": richardson(series["gal"]),
            "rich_sin_coll": richardson(series["coll"]),
            "rich_bspline2": richardson(series["bspl"]),
        }
        for key, pinned in refs.items():
            assert abs(got[key] - pinned) / abs(pinned) < 1e-5, (
                f"{name}.{key}: pinned {pinned:.6f}, recomputed {got[key]:.6f}"
            )


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
