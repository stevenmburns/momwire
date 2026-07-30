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

**G4** — ground models, one at a time (PEC image → reflection-coefficient →
Sommerfeld), each by reusing that ground's existing field evaluator under the
same test quadrature. Symmetry must survive the ground terms, and the
impedance must hold up against the per-ground references the tree already has.
See the ``M4`` section at the bottom for the measured tables, the two places
the gate had to be stated more carefully than the milestone's one-liner (the
Sommerfeld source-side quadrature near the plane; the per-ground restatement of
G2's PyNEC amendment), and the one inherited defect it turned up.
"""

import functools

import numpy as np
import pytest
import scipy.linalg

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


def test_loading_is_not_yet_supported():
    """Still deferred, and it fails loudly rather than silently producing wrong
    numbers (wire loading → after M5). Grounds arrived in M4 — see below."""
    dip_hi = np.array([[0.0, -HD, 2.0], [0.0, HD, 2.0]])
    common = dict(wires=[dip_hi], n_per_edge_per_wire=[[41]], nsegs=41)
    with pytest.raises(NotImplementedError):
        SinusoidalGalerkinSolver(**common, wire_conductivity=5.8e7).compute_impedance()


# ---------------------------------------------------------------------------
# M4 — ground models: PEC image → reflection-coefficient → Sommerfeld
#
# Each ground is wired by reusing that ground's EXISTING source-field evaluator
# under the M2 test quadrature — no new field kernels. The two G4 clauses:
#
# **Symmetry.** ‖G−Gᵀ‖/‖G‖ ≤ 1e-10 with the ground terms in, on uniform-radius
# geometries (M2 established that mixed radii break reciprocity in the KERNEL,
# so symmetry is not a valid oracle there). Measured at the default quadrature,
# n = 21, against the free-space value on the same wires:
#
#     geometry      free      pec       refl      sommerfeld
#     m4_dipole     2.9e-12   3.4e-12   3.1e-12   4.3e-12
#     m4_vertical   6.8e-12   6.6e-12   6.5e-12   6.9e-12
#     m4_lshape     3.9e-11   3.9e-11   4.0e-11   4.0e-11
#     m4_monopole   9.2e-12   1.3e-11   2.4e-12   6.7e-10  ← see below
#
# i.e. every ground block lands the matrix back on the free-space floor. The
# one entry above the gate is Sommerfeld on the ground-CONTACT monopole, and it
# is the source-side remainder quadrature rather than anything M4 added: it
# falls 6.7e-10 → 7.0e-12 → 6.3e-12 as `n_qp_sommerfeld` goes 3 → 5 → 7 and
# then stops, exactly the way M2's free-space floor turned out to be the
# source-side `n_qp_const`. Pinned by
# `test_g4_sommerfeld_symmetry_near_the_plane_is_source_quadrature_limited`.
#
# **Impedance.** The per-ground references that already exist in this tree are
# the PyNEC/nec2c goldens in `golden_refl_coef_ground.py` (gn −1 / gn 0 / gn 2)
# and the dense B-spline solve on the same geometry. Against them:
#
#   * the **ground increment** Z(ground) − Z(free space) agrees with dense
#     B-spline's to 0.04-0.08% for the PEC and Sommerfeld grounds, across the
#     whole 0.1-0.5 λ height window on both fixture geometries;
#   * |Z_gal − gn| sits at 0.96-1.30× |Z_bs2 − gn| — the same NEC-tracking floor
#     the tree's OTHER Galerkin scheme occupies;
#   * |Z_gal − Z_bs2| is smaller than |Z_coll − Z_bs2| on every case, and is
#     unchanged from its free-space value (dipole 0.257 Ω free → 0.253/0.257
#     under PEC/Sommerfeld; inverted_l 0.038 → 0.035/0.036).
#
# What is NOT claimed, per ground, is the point-matched solver's ~0.1 Ω
# tracking of NEC. That is the same amendment G2 made in M2 and for the same
# reason — collocation's tightness to NEC is its shared-heritage artifact, and
# issue #182 lists not reproducing it as an explicit non-goal. The evidence
# that it is heritage rather than a ground-block defect is the last bullet:
# the ground adds nothing to the sin↔bs2 gap, it just carries the free-space
# gap through.
#
# **Convergence.** Following the M3 constraints — feed fixed at a segment
# centre for every N in the sweep, and a NAMED reference family rather than
# "the converged value" — the variational payoff survives every ground:
#
#     geometry      pec    refl   somm    reading
#     m4_dipole     1.88   2.04   2.19    testing-limited
#     m4_vertical   1.74   1.90   1.86    testing-limited
#     m4_lshape     1.01   1.01   1.01    BASIS-limited (M3's k3_star again)
#     m4_monopole   1.64    —      —      testing-limited
#
# (worst-case errColl/errGal over the reference family at N = 11/15/21). The
# ratio barely moves between grounds on a given geometry, which says the
# testing payoff is a property of the free-space fill that the ground blocks
# inherit rather than something the ground changes.
# ---------------------------------------------------------------------------

from fixtures_refl_coef_geoms import GEOMS  # noqa: E402
from golden_refl_coef_ground import GOLDEN  # noqa: E402

# The three grounds, keyed by the GOLDEN column that references each one.
M4_GROUNDS = {
    "pec": ("pec", {}),
    "refl": ("finite-fast", {"ground_eps": (10.0, 0.002)}),
    "somm": ("finite", {"ground_eps": (10.0, 0.002), "ground_model": "sommerfeld"}),
}

# Heights (in wavelengths) of the fixture acceptance window. 0.05 and 0.02 are
# deliberately excluded: they sit outside the documented refl-coef acceptance
# window, where the dense-B-spline finite ground is itself 11.5 Ω from NEC gn 0
# and so cannot serve as a reference for anybody.
M4_WINDOW = (0.1, 0.2, 0.35, 0.5)
M4_FIXTURES = [("dipole", f) for f in M4_WINDOW] + [
    ("inverted_l", f) for f in M4_WINDOW
]

MONO_L = 0.25 * WL  # quarter-wave grounded vertical
GND_H = 0.1 * WL  # elevation of the "over ground" geometries


def _m4_dipole(n, **over):
    """Horizontal dipole 0.1 λ over the plane. Feed defaults to the wire
    midpoint = the centre segment for odd n (M3 constraint 1)."""
    w = np.array([[0.0, -HD, GND_H], [0.0, HD, GND_H]])
    return dict(wires=[w], n_per_edge_per_wire=[[n]], nsegs=n, **over)


def _m4_vertical(n, **over):
    """Centre-fed VERTICAL dipole with its base 0.1 λ up. Vertical current over
    ground is the case where the image is parallel rather than anti-parallel,
    and where the Fresnel dyad's p̂ correction switches off (ρ_v = −ρ_h at
    normal incidence) — a different corner of the ground blocks than the
    horizontal dipole exercises."""
    w = np.array([[0.0, 0.0, GND_H], [0.0, 0.0, GND_H + 2 * HD]])
    return dict(wires=[w], n_per_edge_per_wire=[[n]], nsegs=n, **over)


def _m4_lshape(n, **over):
    """Vertical + horizontal arm meeting 0.1 λ above the plane: a junction (so
    the N⁻/N⁺ basis coupling is live) with both current orientations present.
    Feed defaults to the midpoint of wire 0 — a segment centre for odd n."""
    w0 = np.array([[0.0, 0.0, GND_H], [0.0, 0.0, GND_H + 3.0]])
    w1 = np.array([[0.0, 0.0, GND_H], [3.0, 0.0, GND_H]])
    return dict(
        wires=[w0, w1],
        n_per_edge_per_wire=[[n], [n]],
        nsegs=n,
        feed_wire_index=0,
        junctions=[[(0, "start"), (1, "start")]],
        **over,
    )


def _m4_monopole(n, **over):
    """Quarter-wave vertical whose base LIES IN the plane — the #151
    ground-connected basis, and the only geometry here where a segment and its
    own image share a node. Centre-fed rather than base-fed so the feed is a
    segment centre at every odd n (a base feed would sit at h/2 and creep as
    the mesh refines — M3 constraint 1)."""
    w = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_L]])
    return dict(wires=[w], n_per_edge_per_wire=[[n]], nsegs=n, **over)


M4_GEOMETRIES = {
    "m4_dipole": _m4_dipole,
    "m4_vertical": _m4_vertical,
    "m4_lshape": _m4_lshape,
    "m4_monopole": _m4_monopole,
}

# (geometry, ground) pairs the gates run over. The ground-contact monopole is
# PEC-only: see `test_finite_ground_at_a_ground_contact_is_an_inherited_defect`
# for why the finite grounds have no meaning there on EITHER sinusoidal solver.
M4_CASES = [
    (g, gnd)
    for g in M4_GEOMETRIES
    for gnd in M4_GROUNDS
    if not (g == "m4_monopole" and gnd != "pec")
]

M4_N = 21  # mesh the symmetry / quadrature gates are measured at


def _m4_solver(cls, geom_name, ground, n=M4_N, **over):
    kw = M4_GEOMETRIES[geom_name](n, **over)
    return cls(**kw, ground_z=0.0, **M4_GROUNDS[ground][1])


# --- G4, clause 1: symmetry survives every ground --------------------------


@pytest.mark.parametrize("geom_name,ground", M4_CASES)
def test_g4_symmetry_holds_with_the_ground_terms_in(geom_name, ground):
    """G4's symmetry clause: adding a ground must not cost reciprocity.

    The image block is one more source block through the same test quadrature,
    so it should land the matrix back on the free-space floor — and it does, to
    within ~15% of the free-space value on every case (table in the section
    header). Sommerfeld on the ground-contact monopole is the single exception
    and is `n_qp_sommerfeld`-limited, not M4-limited; it is refined to the gate
    here and characterized by the next test.

    Uniform radius throughout: M2 showed the Eqs 76-79 kernel is itself
    asymmetric under mixed per-wire radii, so symmetry is not an oracle there.
    """
    over = {}
    if geom_name == "m4_monopole" and ground == "somm":
        over["n_qp_sommerfeld"] = 5
    r = _sym_ratio(
        _matrix(_m4_solver(SinusoidalGalerkinSolver, geom_name, ground, **over))
    )
    r_free = _sym_ratio(
        _matrix(SinusoidalGalerkinSolver(**M4_GEOMETRIES[geom_name](M4_N)))
    )
    assert r < G1_GATE, f"{geom_name}/{ground}: ‖G-Gᵀ‖/‖G‖ = {r:.3e}"
    assert r < max(10.0 * r_free, 1e-11), (
        f"{geom_name}/{ground}: {r:.3e} is well above the free-space floor "
        f"{r_free:.3e} on the same wires — the ground block, not the fill"
    )


def test_g4_sommerfeld_symmetry_near_the_plane_is_source_quadrature_limited():
    """The one symmetry number above the gate at default settings, given a
    cause rather than a carve-out.

    On the ground-contact monopole the Sommerfeld remainder's SOURCE-side
    Gauss rule (`n_qp_sommerfeld`, default 3 — the point-matched solver's
    default, untouched here) is what limits reciprocity: the remainder kernel
    varies fastest when the image point is closest, i.e. exactly at a wire
    touching the plane. Measured 6.65e-10 (q=3) → 7.01e-12 (q=5) → 6.26e-12
    (q=7) → 6.27e-12 (q=9): a clean quadrature convergence onto the same
    ~6e-12 floor the PEC ground reaches, not a structural asymmetry.

    Refining the TEST rule instead does nothing (6.65e-10 → 6.60e-10 at
    n_qp_test=16), which is what identifies the source side as the limiter —
    the same diagnosis M2 made for the free-space floor.
    """
    ratios = [
        _sym_ratio(
            _matrix(
                _m4_solver(
                    SinusoidalGalerkinSolver, "m4_monopole", "somm", n_qp_sommerfeld=q
                )
            )
        )
        for q in (3, 5, 7)
    ]
    assert ratios[0] > G1_GATE, f"the documented q=3 residual is gone: {ratios[0]:.2e}"
    assert ratios[1] < G1_GATE and ratios[2] < G1_GATE, (
        f"refining the source rule no longer reaches the gate: {ratios}"
    )
    # ...and it is not the test rule.
    r_test = _sym_ratio(
        _matrix(
            _m4_solver(
                SinusoidalGalerkinSolver,
                "m4_monopole",
                "somm",
                n_qp_test=16,
                n_qp_near=16,
            )
        )
    )
    assert r_test > G1_GATE, (
        f"doubling the TEST rule reached the gate ({r_test:.2e}) — the limiter "
        "is not the source-side remainder quadrature after all"
    )


def test_image_block_gets_its_own_near_pair_set():
    """The structural reason ground contact keeps its symmetry: near pairs are
    selected against the MIRRORED source geometry for an image block.

    On a wire whose base lies in the plane, base segment 0 and its own image
    share the in-plane node, so the image block carries the same width-`a`
    endpoint spike M2 built the graded rule for — and the selection finds
    exactly that one pair. A wire well clear of the plane has no near image
    pairs at all, so the correction costs nothing there.
    """
    sim = SinusoidalGalerkinSolver(**_m4_monopole(M4_N), ground_z=0.0)
    geom = sim._build_geometry()
    img_c, img_t = sim._image_source_centers_tangents(geom)
    mm, nn = sim._near_pairs(geom, src_c=img_c, src_t=img_t)
    assert set(zip(mm.tolist(), nn.tolist())) == {(0, 0)}

    high = SinusoidalGalerkinSolver(**_m4_dipole(M4_N), ground_z=0.0)
    g2 = high._build_geometry()
    ic, it = high._image_source_centers_tangents(g2)
    assert high._near_pairs(g2, src_c=ic, src_t=it)[0].size == 0


@pytest.mark.parametrize("ground", list(M4_GROUNDS))
def test_ground_symmetry_needs_the_near_correction(ground):
    """Same M1-vs-M2 contrast as free space, now with a ground in: disabling
    the graded near-pair rule at the same node count leaves the grounded matrix
    visibly asymmetric (2.1e-6 dipole, 1.6e-4 L-shape, 1.6e-6 to 6.1e-5 on the
    ground-contact monopole, where the image block contributes its own share)."""
    for geom_name in ("m4_dipole", "m4_lshape"):
        r_off = _sym_ratio(
            _matrix(
                _m4_solver(
                    SinusoidalGalerkinSolver, geom_name, ground, near_correction=False
                )
            )
        )
        r_on = _sym_ratio(
            _matrix(_m4_solver(SinusoidalGalerkinSolver, geom_name, ground))
        )
        assert r_on < G1_GATE < UNIFORM_FLOOR < r_off, (
            f"{geom_name}/{ground}: off={r_off:.2e} on={r_on:.2e}"
        )


# --- G4, clause 1b: the exact limits each ground must reproduce -------------


@pytest.mark.parametrize("geom_name", ["m4_dipole", "m4_vertical", "m4_lshape"])
def test_g4_finite_grounds_collapse_to_the_pec_image(geom_name):
    """ε̃ → ∞ must collapse BOTH finite grounds onto the PEC image block: the
    Fresnel dyad degenerates (ρ_v → 1, ρ_h → −1, so the p̂ correction vanishes)
    and NEC's Sommerfeld decomposition has C2 → 1 with the remainder scaling
    away like 1/√ε̃. This is the sign/normalization pin for both blocks —
    a flipped sign anywhere would be a ~2× miss, not a 1e-8 one.

    Measured at ground_eps = 1e16: refl 5.9e-8 / 2.6e-9 / 7.7e-11 and
    sommerfeld 7.7e-8 / 2.0e-9 / 9.4e-11 over the three geometries; the
    point-matched solver's own collapse on its fixtures is 1.1e-8 / 4.8e-9.
    """
    kw = M4_GEOMETRIES[geom_name](M4_N)
    z_pec, _ = SinusoidalGalerkinSolver(**kw, ground_z=0.0).compute_impedance()
    z_refl, _ = SinusoidalGalerkinSolver(
        **kw, ground_z=0.0, ground_eps=1e16 + 0j
    ).compute_impedance()
    z_somm, _ = SinusoidalGalerkinSolver(
        **kw, ground_z=0.0, ground_eps=1e16 + 0j, ground_model="sommerfeld"
    ).compute_impedance()
    assert abs(z_refl - z_pec) < 1e-5 * abs(z_pec)
    assert abs(z_somm - z_pec) < 1e-5 * abs(z_pec)


@pytest.mark.parametrize("geom_name", ["m4_dipole", "m4_vertical", "m4_lshape"])
def test_g4_sommerfeld_free_space_limit_is_exact(geom_name):
    """ε̃ → 1: C2 = 0 and the remainder integrands vanish identically, so the
    grounded solve must reproduce the no-ground solve. Measured EXACTLY equal
    (0.0 relative) on all three geometries — the two blocks cancel term by
    term, they do not merely nearly cancel.

    Excluded: the ground-contact monopole, where setting `ground_z` also
    changes the BASIS (#151 makes the in-plane end a ground connection instead
    of a free end), so ε̃ = 1 is not the same problem as no ground at all.
    """
    kw = M4_GEOMETRIES[geom_name](M4_N)
    z_free, _ = SinusoidalGalerkinSolver(**kw).compute_impedance()
    z_g, _ = SinusoidalGalerkinSolver(
        **kw, ground_z=0.0, ground_eps=1.0 + 0j, ground_model="sommerfeld"
    ).compute_impedance()
    assert abs(z_g - z_free) < 1e-12 * abs(z_free)


@pytest.mark.parametrize("geom_name,ground", M4_CASES)
def test_g4_ground_impedance_is_quadrature_converged(geom_name, ground):
    """G2's "converged, not tuned" clause, restated with the ground terms in:
    doubling BOTH test-quadrature knobs moves Z by well under 0.5%. Measured
    shifts are 1e-9 to 1.7e-7 across the whole (geometry × ground) matrix, so
    none of the ground numbers below are quadrature artifacts."""
    z1, _ = _m4_solver(SinusoidalGalerkinSolver, geom_name, ground).compute_impedance()
    z2, _ = _m4_solver(
        SinusoidalGalerkinSolver, geom_name, ground, n_qp_test=16, n_qp_near=16
    ).compute_impedance()
    shift = abs(z2 - z1) / abs(z1)
    assert shift < 5e-3, f"{geom_name}/{ground}: {shift:.3%} on doubling n_qp"


# --- G4, clause 2: impedance against the existing per-ground references -----


def _fixture_z(cls, name, frac, ground, **over):
    kw = dict(GEOMS[(name, frac)])
    if cls is BSplineSolver:
        over.setdefault("degree", 2)
    return cls(**kw, ground_z=0.0, **M4_GROUNDS[ground][1], **over).compute_impedance()[
        0
    ]


def _fixture_free_z(cls, name, frac):
    kw = dict(GEOMS[(name, frac)])
    extra = {"degree": 2} if cls is BSplineSolver else {}
    return cls(**kw, **extra).compute_impedance()[0]


@pytest.mark.parametrize("name,frac", M4_FIXTURES)
@pytest.mark.parametrize("ground", ["pec", "somm"])
def test_g4_ground_increment_matches_dense_bspline(name, frac, ground):
    """The sharpest available statement about the ground blocks themselves.

    Comparing total impedances mixes the ground block with the free-space fill,
    whose sin↔bs2 gap is a basis/testing effect M3 already classified. Comparing
    the ground INCREMENT Z(ground) − Z(free space) isolates the ground block —
    and against the tree's independently-validated B-spline Galerkin ground
    blocks it agrees to **0.04-0.08%** of the increment on every case here,
    over both fixture geometries and the whole 0.1-0.5 λ height window (the
    increment itself is 1.1-54 Ω, so this is not a small-signal artifact).

    `refl` is excluded from this comparison on purpose, and not because it
    fails: the two solvers implement DIFFERENT finite-ground models there. The
    sinusoidal solvers are field-based and apply NEC's Fresnel field dyad
    exactly; the mixed-potential B-spline has to approximate the image-charge
    weighting through `ground_phi_mode` (`_ground_refl.PHI_MODES` documents
    that as the one place the mixed-potential form cannot copy NEC). Their
    increments differ by 0.17-16%, which measures that modelling difference,
    not this fill. The refl ground is gated against the gn 0 golden below.
    """
    d_gal = _fixture_z(SinusoidalGalerkinSolver, name, frac, ground) - _fixture_free_z(
        SinusoidalGalerkinSolver, name, frac
    )
    d_bs = _fixture_z(BSplineSolver, name, frac, ground) - _fixture_free_z(
        BSplineSolver, name, frac
    )
    rel = abs(d_gal - d_bs) / abs(d_bs)
    assert rel < 1.5e-3, (
        f"{name} h={frac} {ground}: ground increment differs from dense bspline "
        f"by {rel:.3%} (Δ_gal={d_gal:.4f}, Δ_bs2={d_bs:.4f})"
    )


@pytest.mark.parametrize("name,frac", M4_FIXTURES)
@pytest.mark.parametrize("ground", list(M4_GROUNDS))
def test_g4_tracks_the_gn_golden_at_the_other_galerkin_scheme_floor(name, frac, ground):
    """G4's golden clause, stated the way G2's PyNEC clause had to be.

    |Z_gal − gn| is 0.96-1.30× |Z_bs2 − gn| across the matrix: sinusoidal
    Galerkin lands on the NEC-family goldens at the same cross-solver floor the
    tree's OTHER Galerkin scheme occupies (~1.4 Ω on these fixtures, which is
    the floor `docs/refl-coef-ground-plan.md` measured independently).

    It deliberately does NOT reproduce the point-matched solver's 0.05-0.14 Ω
    tracking of the same goldens. That tightness is collocation's NEC heritage
    — same basis, same point matching — and #182 lists not reproducing it as an
    explicit non-goal. The evidence that this is heritage and not a ground-block
    defect is `test_the_ground_adds_nothing_to_the_sin_bspline_gap` below: the
    gap is already there in free space, at the same size, on the same wires.
    """
    gn = GOLDEN[(name, frac, 10.0, 0.002)][M4_GROUNDS[ground][0]]
    d_gal = abs(_fixture_z(SinusoidalGalerkinSolver, name, frac, ground) - gn)
    d_bs = abs(_fixture_z(BSplineSolver, name, frac, ground) - gn)
    assert d_gal < 1.4 * d_bs, (
        f"{name} h={frac} {ground}: |gal-gn| = {d_gal:.4f} against dense "
        f"bspline's {d_bs:.4f} — no longer at the other Galerkin scheme's floor"
    )


@pytest.mark.parametrize("name,frac", M4_FIXTURES)
@pytest.mark.parametrize("ground", list(M4_GROUNDS))
def test_g4_closer_to_dense_bspline_than_collocation_is(name, frac, ground):
    """G2's dense-B-spline clause, per ground: against the other Galerkin scheme
    in the tree, sinusoidal Galerkin is closer than the point-matched solver on
    every grounded case. Measured |gal−bs2| 0.035-1.62 Ω against |coll−bs2|
    1.45-2.52 Ω over the matrix."""
    z_bs = _fixture_z(BSplineSolver, name, frac, ground)
    d_gal = abs(_fixture_z(SinusoidalGalerkinSolver, name, frac, ground) - z_bs)
    d_coll = abs(_fixture_z(SinusoidalSolver, name, frac, ground) - z_bs)
    assert d_gal < d_coll, (
        f"{name} h={frac} {ground}: galerkin {d_gal:.4f} from dense bspline, "
        f"worse than collocation's {d_coll:.4f}"
    )


@pytest.mark.parametrize("name", ["dipole", "inverted_l"])
@pytest.mark.parametrize("ground", ["pec", "somm"])
def test_the_ground_adds_nothing_to_the_sin_bspline_gap(name, ground):
    """Why the golden clause above is a heritage statement and not a defect.

    The sin-Galerkin ↔ dense-B-spline gap under ground is the SAME gap the two
    schemes already have in free space on the same wires — 0.257 Ω free vs
    0.253 (PEC) / 0.257 (Sommerfeld) on the dipole, 0.038 free vs 0.035 / 0.036
    on the inverted-L. The ground block adds essentially nothing to it; it just
    carries the free-space difference through. So the residual against NEC is
    the free-space fill's, and no amount of ground work would move it.

    `refl` excluded for the modelling reason given two tests up.
    """
    frac = 0.2
    gap_free = abs(
        _fixture_free_z(SinusoidalGalerkinSolver, name, frac)
        - _fixture_free_z(BSplineSolver, name, frac)
    )
    gap_gnd = abs(
        _fixture_z(SinusoidalGalerkinSolver, name, frac, ground)
        - _fixture_z(BSplineSolver, name, frac, ground)
    )
    assert 0.8 < gap_gnd / gap_free < 1.25, (
        f"{name}/{ground}: sin↔bs2 gap {gap_gnd:.4f} under ground vs "
        f"{gap_free:.4f} in free space — the ground block is contributing"
    )


# --- G4, clause 3: the M3 payoff, per ground -------------------------------
#
# Both M3 constraints are honoured: every `_m4_*` factory puts the feed at a
# point that is a segment centre for every odd n (so the delta gap does not
# translate as the mesh refines), and the reference is a NAMED FAMILY rather
# than "the converged value" — which, per M3, does not exist for a delta-gap
# reactance. Family = the N=161 solve and the Richardson extrapolation of
# N = 81/121/161, for each scheme that implements the SAME ground model.
#
# B-spline is in the family for `pec` and `somm` (identical physics: the PEC
# image, and NEC's C2-image-plus-remainder decomposition) but NOT for `refl`,
# where it approximates the image-charge weighting through `ground_phi_mode`
# and lands 1 Ω / 2% away from both sinusoidal schemes — a different model, not
# a different discretization, so it cannot serve as a reference for this one.

M4_REFS = {
    ("m4_dipole", "pec"): dict(
        gal_161=20.995250 + 2.666325j,
        rich_gal=20.999742 + 2.766010j,
        coll_161=20.991305 + 2.574722j,
        rich_coll=20.998852 + 2.739493j,
        bspl_161=20.995314 + 2.664842j,
        rich_bspl=21.000269 + 2.765637j,
    ),
    ("m4_dipole", "refl"): dict(
        gal_161=48.106418 - 4.643412j,
        rich_gal=48.110072 - 4.560912j,
        coll_161=48.097236 - 4.733933j,
        rich_coll=48.108123 - 4.587350j,
    ),
    ("m4_dipole", "somm"): dict(
        gal_161=56.409201 - 8.712697j,
        rich_gal=56.408636 - 8.639097j,
        coll_161=56.399501 - 8.801966j,
        rich_coll=56.406698 - 8.665026j,
        bspl_161=56.405042 - 8.723017j,
        rich_bspl=56.408964 - 8.641944j,
    ),
    ("m4_vertical", "pec"): dict(
        gal_161=75.632971 - 26.411502j,
        rich_gal=75.609852 - 26.356312j,
        coll_161=75.618689 - 26.498808j,
        rich_coll=75.606594 - 26.382214j,
        bspl_161=75.618372 - 26.428028j,
        rich_bspl=75.608020 - 26.361113j,
    ),
    ("m4_vertical", "refl"): dict(
        gal_161=72.422670 - 22.749400j,
        rich_gal=72.405074 - 22.691248j,
        coll_161=72.409158 - 22.837169j,
        rich_coll=72.402045 - 22.717164j,
    ),
    ("m4_vertical", "somm"): dict(
        gal_161=72.107419 - 24.213443j,
        rich_gal=72.088192 - 24.154544j,
        coll_161=72.093827 - 24.300908j,
        rich_coll=72.085090 - 24.180389j,
        bspl_161=72.094569 - 24.228621j,
        rich_bspl=72.086715 - 24.158961j,
    ),
    ("m4_lshape", "pec"): dict(
        gal_161=11.968607 - 1085.995621j,
        rich_gal=11.875187 - 1081.538405j,
        coll_161=11.965616 - 1086.082421j,
        rich_coll=11.875475 - 1081.571768j,
        bspl_161=11.933778 - 1084.412462j,
        rich_bspl=11.863879 - 1081.026114j,
    ),
    ("m4_lshape", "refl"): dict(
        gal_161=10.932084 - 1087.785369j,
        rich_gal=10.846718 - 1083.313835j,
        coll_161=10.929226 - 1087.871391j,
        rich_coll=10.846979 - 1083.347318j,
    ),
    ("m4_lshape", "somm"): dict(
        gal_161=12.579734 - 1088.065235j,
        rich_gal=12.481400 - 1083.591743j,
        coll_161=12.576230 - 1088.151274j,
        rich_coll=12.481679 - 1083.625009j,
        bspl_161=12.543057 - 1086.476087j,
        rich_bspl=12.469523 - 1083.077592j,
    ),
    ("m4_monopole", "pec"): dict(
        gal_161=74.917206 + 44.165358j,
        rich_gal=74.957605 + 44.213649j,
        coll_161=74.913681 + 44.115541j,
        rich_coll=74.956952 + 44.202932j,
        bspl_161=74.929388 + 44.141531j,
        rich_bspl=74.962429 + 44.218677j,
    ),
}

M4_FINE = (81, 121, 161)

# Worst-case errColl/errGal over the whole reference family at N = 11/15/21,
# floored a little below the measured value.
M4_GATE_RATIO = {
    ("m4_dipole", "pec"): 1.85,
    ("m4_dipole", "refl"): 2.0,
    ("m4_dipole", "somm"): 2.15,
    ("m4_vertical", "pec"): 1.70,
    ("m4_vertical", "refl"): 1.87,
    ("m4_vertical", "somm"): 1.83,
    ("m4_lshape", "pec"): 1.005,
    ("m4_lshape", "refl"): 1.005,
    ("m4_lshape", "somm"): 1.005,
    ("m4_monopole", "pec"): 1.60,
}


@functools.lru_cache(maxsize=None)
def _m4_z(scheme, geom_name, ground, n):
    kw = M4_GEOMETRIES[geom_name](n)
    kw = dict(kw, ground_z=0.0, **M4_GROUNDS[ground][1])
    return _SCHEMES[scheme](kw).compute_impedance()[0]


@pytest.mark.parametrize("geom_name,ground", M4_CASES)
def test_g4_variational_payoff_survives_each_ground(geom_name, ground):
    """G3's payoff, re-measured with each ground in: at 11-21 segments the
    Galerkin impedance is still closer to the fine-mesh answer than the
    point-matched solver's, and the verdict survives every reference in the
    family (M3 constraint 2 — there is no single converged value to measure
    against, so the gate is the worst ratio over all defensible choices).

    Measured worst-case errColl/errGal: dipole 1.88/2.04/2.19,
    vertical 1.74/1.90/1.86, L-shape 1.01/1.01/1.01, contact monopole 1.64
    (pec/refl/somm). Two things worth reading off that:

    * the ratio barely moves between grounds on a given geometry, so the
      testing payoff is a property of the free-space fill that the ground
      blocks inherit rather than something the ground changes;
    * the L-shape is BASIS-limited at 1.01, which is M3's k3_star reading
      again — near-open, |X| ≈ 1085, and no amount of testing work moves it.
    """
    refs = M4_REFS[(geom_name, ground)]
    worst = min(
        _rel_err(_m4_z("coll", geom_name, ground, n), ref)
        / _rel_err(_m4_z("gal", geom_name, ground, n), ref)
        for ref in refs.values()
        for n in M3_COARSE
    )
    assert worst > 1.0, (
        f"{geom_name}/{ground}: the payoff flips under some reference choice "
        f"(worst errColl/errGal = {worst:.3f})"
    )
    assert worst >= M4_GATE_RATIO[(geom_name, ground)], (
        f"{geom_name}/{ground}: worst-case errColl/errGal fell to {worst:.3f}, "
        f"below the pinned {M4_GATE_RATIO[(geom_name, ground)]}"
    )


@pytest.mark.parametrize("geom_name,ground", M4_CASES)
def test_g4_ground_convergence_is_monotone(geom_name, ground):
    """The convergence curve with a ground in must still decay monotonically —
    over the whole M3 ladder (11 → 41) on every geometry × ground here, unlike
    free space where the junction geometry has a pre-asymptotic maximum."""
    ref = M4_REFS[(geom_name, ground)]["rich_gal"]
    errs = [_rel_err(_m4_z("gal", geom_name, ground, n), ref) for n in M3_LADDER]
    assert all(b < a for a, b in zip(errs, errs[1:])), (
        f"{geom_name}/{ground}: not monotone over N={M3_LADDER}: "
        f"{[f'{e:.4%}' for e in errs]}"
    )


def test_the_m4_reference_family_is_tight_enough_to_decide():
    """The reference families span 1.3e-3 to 9.1e-3 of |Z| — wide enough that
    the M3 warning applies (no single converged number exists) and narrow
    enough that the payoff verdicts above survive all of them. Asserted so a
    future reference refresh cannot silently widen the family until the gate
    stops meaning anything."""
    for key, refs in M4_REFS.items():
        vals = list(refs.values())
        spread = max(abs(a - b) for a in vals for b in vals) / abs(vals[0])
        assert spread < 1.5e-2, f"{key}: reference family spans {spread:.2e} of |Z|"


# --- A finding, pinned rather than fixed -----------------------------------


def test_finite_ground_at_a_ground_contact_is_an_inherited_defect():
    """A wire END LYING IN the plane plus a FINITE ground is broken on BOTH
    sinusoidal solvers, and M4 did not introduce it.

    #151's ground-connected basis folds a segment's image-side extension back
    onto the segment — the end current is completed by its own image. That is
    exact for a PEC plane (where the image IS the mirrored current) and it is
    the reason the PEC contact monopole agrees with every other basis to <0.1%.
    With a Fresnel-weighted or Sommerfeld image the image is no longer a
    physical continuation of the wire, so the basis's built-in continuation is
    inconsistent with the field model, and refining the mesh does not help:

        n     galerkin           collocation        dense bspline
        11    21.15  -965.56j    64.55  -484.08j    41.66 +22.12j
        21    21.12  -920.65j    53.54  -632.64j    41.71 +22.35j
        41    21.12  -896.35j    41.53  -741.49j    41.74 +22.50j
        81    21.12  -883.02j    32.39  -806.50j    41.76 +22.60j

    B-spline (value-1 end basis, no self-image folding) converges cleanly;
    neither sinusoidal scheme goes anywhere near it. So this is a basis defect
    shared with the point-matched solver, which per the standing rules is
    recorded rather than "fixed" from the Galerkin side. It is also why the
    finite grounds are gated on ELEVATED geometries throughout M4.

    (The PEC contact case, by contrast, is gated normally above — it is the
    only ground for which the #151 basis is exact.)
    """
    e = {"ground_eps": (10.0, 0.002)}
    ns = (21, 81)
    zb = [
        BSplineSolver(
            **_m4_monopole(n), ground_z=0.0, **e, degree=2
        ).compute_impedance()[0]
        for n in ns
    ]
    # B-spline is converged: it barely moves over a 4x refinement.
    assert abs(zb[1] - zb[0]) / abs(zb[0]) < 0.02, f"bspline reference moved: {zb}"
    for cls in (SinusoidalSolver, SinusoidalGalerkinSolver):
        zs = [
            cls(**_m4_monopole(n), ground_z=0.0, **e).compute_impedance()[0] for n in ns
        ]
        # ...and both sinusoidal schemes are nowhere near it, at either mesh.
        assert all(abs(z - zb[i]) > 5.0 * abs(zb[i]) for i, z in enumerate(zs)), (
            f"{cls.__name__} is no longer far from the bspline reference "
            f"({zs} vs {zb}) — the documented defect may have been fixed; "
            "re-derive this test rather than deleting it"
        )
    # The PEC contact case IS sound on the same wires — the contrast that
    # localizes the defect to the non-PEC image, not to ground contact itself.
    z_pec = [
        cls(**_m4_monopole(21), ground_z=0.0).compute_impedance()[0]
        for cls in (SinusoidalSolver, SinusoidalGalerkinSolver, BSplineSolver)
    ]
    assert max(abs(z - z_pec[0]) for z in z_pec) < 0.01 * abs(z_pec[0])


@pytest.mark.slow
def test_m4_reference_constants_are_reproducible():
    """The pinned `M4_REFS` are the only measured inputs the M4 convergence
    gates trust, so they are re-derived here from the N = 81/121/161 solves
    they came from — three schemes × three grounds × four geometries. Slow
    (~30 s); keeps the constants from rotting silently under a fill change."""

    def richardson(zs):
        c = (zs[-2] - zs[-1]) / (1.0 / M4_FINE[-2] - 1.0 / M4_FINE[-1])
        return zs[-1] - c / M4_FINE[-1]

    for (geom_name, ground), refs in M4_REFS.items():
        series = {
            s: [_m4_z(s, geom_name, ground, n) for n in M4_FINE]
            for s in ("gal", "coll", "bspl")
            if f"{s}_161" in refs
        }
        got = {}
        for s, zs in series.items():
            got[f"{s}_161"] = zs[-1]
            got[f"rich_{s}"] = richardson(zs)
        assert set(got) == set(refs), (geom_name, ground, sorted(got), sorted(refs))
        for key, pinned in refs.items():
            assert abs(got[key] - pinned) / abs(pinned) < 1e-5, (
                f"{geom_name}/{ground}.{key}: pinned {pinned:.6f}, "
                f"recomputed {got[key]:.6f}"
            )


# ---------------------------------------------------------------------------
# M5 — the network-parameter paths, made honestly Galerkin
#
# M4's finding 2: `compute_y_matrix` / `compute_y_matrix_swept` /
# `compute_impedance_swept` were inherited verbatim from the point-matched
# solver, which pairs THIS solver's Galerkin matrix with collocation's point
# RHS (a bare −1/h at the feed segment's row). That is #177's hybrid trap, and
# it was not subtle: the inherited Y read Z = 0.0489 − 0.0129j on the N=41
# dipole where `compute_impedance` reads 69.687 − 18.217j — wrong by 1400×,
# because the Galerkin RHS lives in BASIS index space (the delta gap spread
# over every basis touching the feed segment) and the collocation one in
# SEGMENT index space. Ports are a Y-matrix feature, so this had to be fixed
# before anything could be built on it.
#
# All four entry points now share `_drive_columns` and `_port_currents`, so
# they agree with `compute_impedance` to roundoff (5.6e-16 measured).
#
# The remaining wrinkle is the delta-gap feed's READOUT, and it is a real
# finding rather than a leftover. The Galerkin drive for NEC's delta gap
# spreads E_app = V/Δ over the whole feed segment, so its exact dual readout
# is the gap-AVERAGED current; the readout this solver actually uses is the
# point-matched solver's, the current at the segment CENTRE. They differ at
# O(h), which is why a two-gap-feed Y is symmetric only to ~1e-5:
#
#     N     centre (default)   variational   collocation (inherited)
#     21    6.7e-05            1.8e-12       9.1e-05
#     41    2.4e-05            1.6e-12       2.7e-05
#     81    6.5e-06            4.2e-12       7.3e-06
#
# `feed_readout="variational"` makes the pairing dual and the Y exactly
# reciprocal. It is NOT the default, and the reason is measured: it costs the
# M3 payoff gate on k3_star (worst-case errColl/errGal 1.014 → 0.797, i.e. the
# variational readout is where collocation starts winning). Keeping the
# centre readout also keeps the two sinusoidal cells differing in exactly one
# thing — the testing — which is the whole point of the instrument.
# ---------------------------------------------------------------------------


def _two_feed_dipole(n=41, **over):
    dip = np.array([DIP_A, DIP_B])
    return dict(
        wires=[dip],
        n_per_edge_per_wire=[[n]],
        nsegs=n,
        feeds=[(0, HD * 0.3, 1.0 + 0j), (0, HD * 1.1, 0.25 - 0.5j)],
        **over,
    )


@pytest.mark.parametrize("readout", ["centre", "variational"])
@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_y_matrix_agrees_with_compute_impedance(geom_name, readout):
    """M4 finding 2, fixed: the Y matrix is now the same Galerkin drive and
    the same readout `compute_impedance` uses, so I = Y·V reproduces the
    impedance readout to roundoff (worst 5.6e-16 measured). Holds under both
    readout conventions — each is internally consistent."""
    kw = GEOMETRIES[geom_name](wavelength=WL, feed_readout=readout)
    s = SinusoidalGalerkinSolver(**kw)
    z = np.atleast_1d(s.compute_impedance()[0])
    volts = np.array([v for _, _, v in s.feeds], dtype=np.complex128)
    i_y = SinusoidalGalerkinSolver(**kw).compute_y_matrix() @ volts
    np.testing.assert_allclose(volts / z, i_y, rtol=1e-11)


def test_multi_feed_y_matrix_agrees_with_compute_impedance():
    """Same, with two gap feeds at different voltages — the case where a
    single-column RHS cannot hide a drive/readout mismatch."""
    s = SinusoidalGalerkinSolver(**_two_feed_dipole(wavelength=WL))
    z = np.atleast_1d(s.compute_impedance()[0])
    volts = np.array([v for _, _, v in s.feeds], dtype=np.complex128)
    Y = SinusoidalGalerkinSolver(**_two_feed_dipole(wavelength=WL)).compute_y_matrix()
    np.testing.assert_allclose(volts / z, Y @ volts, rtol=1e-11)


def test_the_inherited_y_matrix_was_a_collocation_rhs_on_a_galerkin_matrix():
    """Pins WHAT M4's finding 2 actually was, so the fix cannot be undone by
    a well-meaning "simplify by reusing the base class" refactor.

    The inherited implementation put −1/h_feed on the feed SEGMENT's row of a
    matrix whose rows are indexed by BASIS, then read the centre current. On
    the N=41 dipole that gives 0.0489 − 0.0129j against the true 69.687 −
    18.217j: not a subtle inconsistency, a different problem.
    """
    s = SinusoidalGalerkinSolver(**_dipole(wavelength=WL))
    geom = s._build_geometry()
    G, seg_view = s._assemble_Z(geom, s.k)
    fi = geom["feed_segs"][0]
    b_coll = np.zeros(geom["n_segs"], dtype=np.complex128)
    b_coll[fi] = -1.0 / geom["seg_h"][fi]
    z_hybrid = 1.0 / s._feed_segment_current(
        scipy.linalg.solve(G, b_coll), seg_view, fi
    )
    z_true = complex(np.atleast_1d(s.compute_impedance()[0])[0])
    assert abs(z_hybrid - z_true) / abs(z_true) > 0.9
    assert abs(z_true / z_hybrid) > 100.0


@pytest.mark.parametrize("readout", ["centre", "variational"])
def test_impedance_swept_matches_per_k(readout):
    """The swept path is the per-k path, including the ports ordering and the
    readout convention."""
    ks = np.array([0.9, 1.0, 1.1]) * SinusoidalGalerkinSolver(**_dipole()).k
    swept = SinusoidalGalerkinSolver(
        **_two_feed_dipole(n=21, wavelength=WL, feed_readout=readout)
    ).compute_impedance_swept(ks)
    y_swept = SinusoidalGalerkinSolver(
        **_two_feed_dipole(n=21, wavelength=WL, feed_readout=readout)
    ).compute_y_matrix_swept(ks)
    for i, kk in enumerate(ks):
        s = SinusoidalGalerkinSolver(
            **_two_feed_dipole(n=21, wavelength=WL, feed_readout=readout)
        )
        s._set_k(float(kk))
        np.testing.assert_allclose(
            swept[i], np.atleast_1d(s.compute_impedance()[0]), rtol=1e-10
        )
        np.testing.assert_allclose(y_swept[i], s.compute_y_matrix(), rtol=1e-10)


@pytest.mark.parametrize("n", [21, 41, 81])
def test_gap_feed_readout_is_not_its_drives_dual(n):
    """The measured cost of keeping NEC's centre-current readout.

    Y symmetry is exact iff the readout functional IS the drive functional.
    The Galerkin delta-gap drive integrates E_app = V/Δ over the feed
    segment, so its dual is the gap-AVERAGED current; the default readout is
    the centre current. Measured asymmetry (see the section header): 6.7e-05
    / 2.4e-05 / 6.5e-06 at N = 21/41/81 — an O(h) effect that decays, and one
    the point-matched solver has slightly MORE of on the same wires.

    `feed_readout="variational"` closes it to ~1e-12, which is the proof that
    the residue is the readout convention and nothing in the fill.
    """

    def asym(cls, **kw):
        Y = cls(
            **_two_feed_dipole(n=n, wavelength=WL, **kw),
        ).compute_y_matrix()
        Y = np.asarray(Y)
        return np.linalg.norm(Y - Y.T) / np.linalg.norm(Y)

    a_centre = asym(SinusoidalGalerkinSolver)
    a_var = asym(SinusoidalGalerkinSolver, feed_readout="variational")
    a_coll = asym(SinusoidalSolver)
    assert a_var < 1e-10, f"the dual pairing is not reciprocal: {a_var:.3e}"
    assert 1e-6 < a_centre < 2e-4, a_centre
    assert a_centre < a_coll, (
        f"the Galerkin centre-readout Y ({a_centre:.3e}) is no longer less "
        f"asymmetric than the point-matched one ({a_coll:.3e})"
    )


def test_feed_readout_is_validated():
    with pytest.raises(ValueError, match="feed_readout"):
        SinusoidalGalerkinSolver(**_dipole(feed_readout="galerkin"))


@pytest.mark.slow
def test_the_variational_readout_costs_the_m3_payoff_on_k3_star():
    """Why `feed_readout="variational"` is not the default, as a number.

    Re-runs M3's payoff comparison on k3_star — the geometry M3 identified as
    BASIS-limited, where the testing payoff is already only 1.01× — under both
    readouts, against that readout's OWN fine-mesh reference family (N=321
    solve + a Richardson extrapolation off N=241/321, following M3's rule that
    a verdict must survive every defensible reference).

    Measured worst-case errColl/errGal: 1.014 with the centre readout (the
    value `M3_GATE_RATIO` pins) and 0.797 with the variational one — i.e. the
    exactly-reciprocal readout is where the point-matched solver starts
    winning, and G3 would go red. Slow (~90 s): four fine k3_star solves.
    """
    fine = (241, 321)

    def both_readouts(n):
        s = SinusoidalGalerkinSolver(**_m3_k3(n))
        geom = s._build_geometry()
        G, seg_view = s._assemble_Z(geom, s.k)
        U = s._drive_columns(geom, seg_view, s.k)
        alpha = scipy.linalg.solve(G, U[:, 0])
        centre = 1.0 / s._feed_segment_current(alpha, seg_view, geom["feed_segs"][0])
        return complex(centre), complex(1.0 / -(U[:, 0] @ alpha))

    fine_z = {n: both_readouts(n) for n in fine}
    coarse_z = {n: both_readouts(n) for n in M3_COARSE}
    worst = {}
    for idx, label in ((0, "centre"), (1, "variational")):
        z241, z321 = fine_z[241][idx], fine_z[321][idx]
        refs = (z321, z321 + (z321 - z241) * (241.0 / (321.0 - 241.0)))
        worst[label] = min(
            _rel_err(_m3_z("coll", "k3_star", n), ref) / _rel_err(coarse_z[n][idx], ref)
            for ref in refs
            for n in M3_COARSE
        )
    assert worst["centre"] > 1.0, worst
    assert worst["centre"] >= M3_GATE_RATIO["k3_star"], worst
    assert worst["variational"] < 1.0, (
        "the variational readout no longer costs the k3_star payoff "
        f"({worst}) — reconsider making it the default"
    )
