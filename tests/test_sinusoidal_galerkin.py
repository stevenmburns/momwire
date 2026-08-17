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

import contextlib
import functools

import numpy as np
import pytest
import scipy.linalg
import scipy.spatial.distance

from momwire import (
    BSplineSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
    _field_ground,
)
from momwire.sinusoidal_galerkin import (
    _basis_value,
    _graded_endpoint_rule,
    _plain_projection,
)

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
# `_near_pairs` row-chunked prefilter (issue #334): `reach`/`cdist`/the
# scaled-reach product/the `<=` boolean used to be built as one (N, N_src)
# shot regardless of N — a ~25 bytes/element transient that collapses to
# O(hits) index pairs. The row-chunked production code must select the
# EXACT SAME pairs, in the EXACT SAME order, as that single-shot form, and
# its own transient must stay bounded independent of N.
# ---------------------------------------------------------------------------


def _near_pairs_unchunked_reference(sim, geom, src_c=None, src_t=None, n_samples=5):
    """The pre-#334 single-shot prefilter body, kept here independent of the
    production `_near_pairs` as the oracle the row-chunked version is
    checked against."""
    c = geom["seg_centers"]
    t = geom["seg_tangents"]
    cs = c if src_c is None else src_c
    ts = t if src_t is None else src_t
    hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)
    reach = hh[:, None] + hh[None, :]

    dc = scipy.spatial.distance.cdist(c, cs)
    cand = np.argwhere(dc <= (1.0 + sim.near_factor) * reach)
    m, n = cand[:, 0], cand[:, 1]

    s = np.linspace(-1.0, 1.0, n_samples)

    def _gap(ai, bi, ca, ta, cb, tb):
        pts = (
            ca[ai][:, None, :]
            + (hh[ai][:, None] * s[None, :])[:, :, None] * ta[ai][:, None, :]
        )
        d = pts - cb[bi][:, None, :]
        u = np.einsum("pgd,pd->pg", d, tb[bi])
        u = np.clip(u, -hh[bi][:, None], hh[bi][:, None])
        perp = d - u[..., None] * tb[bi][:, None, :]
        return np.linalg.norm(perp, axis=-1).min(axis=1)

    gap = np.minimum(
        _gap(m, n, c, t, cs, ts),
        _gap(n, m, cs, ts, c, t),
    )
    keep = gap <= sim.near_factor * reach[m, n]
    return m[keep], n[keep]


@pytest.mark.parametrize("geom_name", list(GEOMETRIES))
def test_near_pairs_chunked_matches_unchunked_reference(geom_name):
    """Default (auto-picked) chunking must select the exact same (m, n)
    pairs, in the exact same order, as the pre-#334 single-shot reference —
    not merely the same set."""
    sim = SinusoidalGalerkinSolver(**GEOMETRIES[geom_name]())
    geom = sim._build_geometry()
    mm, nn = sim._near_pairs(geom)
    want_m, want_n = _near_pairs_unchunked_reference(sim, geom)
    assert np.array_equal(mm, want_m)
    assert np.array_equal(nn, want_n)


@pytest.mark.parametrize("chunk_rows", [1, 3, 7, 100])
def test_near_pairs_chunked_matches_unchunked_reference_forced_tail(chunk_rows):
    """Force small `chunk_rows` values on the 41-segment dipole so the final
    row-block is genuinely partial for every value here (41 is not a
    multiple of 1, 3, 7, or 100) — this exercises the tail concatenation,
    not just interior full blocks. `chunk_rows=1` is the extreme case: every
    block is a single row."""
    sim = SinusoidalGalerkinSolver(**_dipole())
    geom = sim._build_geometry()
    mm, nn = sim._near_pairs(geom, chunk_rows=chunk_rows)
    want_m, want_n = _near_pairs_unchunked_reference(sim, geom)
    assert np.array_equal(mm, want_m)
    assert np.array_equal(nn, want_n)


def test_near_pairs_chunked_matches_unchunked_reference_image_block():
    """Same equality check against the M4 image-block call shape
    (`src_c`/`src_t` passed explicitly, mirrored geometry) — the branch
    every ground model's near-correction actually goes through — with a
    forced `chunk_rows` well below N so the tail is exercised here too."""
    sim = SinusoidalGalerkinSolver(**_m4_monopole(M4_N), ground_z=0.0)
    geom = sim._build_geometry()
    img_c, img_t = sim._image_source_centers_tangents(geom)
    mm, nn = sim._near_pairs(geom, src_c=img_c, src_t=img_t, chunk_rows=4)
    want_m, want_n = _near_pairs_unchunked_reference(
        sim, geom, src_c=img_c, src_t=img_t
    )
    assert np.array_equal(mm, want_m)
    assert np.array_equal(nn, want_n)


@pytest.mark.memgate
def test_near_pairs_prefilter_holds_no_n_squared_transient():
    """Tracemalloc gate: the prefilter's peak transient must stay a few MB
    on a ~2,000-segment geometry, not the ~25 bytes/element `(N, N)` stack
    (reach + cdist + scaled-reach + bool) the row-chunking replaced.

    Arithmetic for the threshold at N = 2,000 (near_factor=0.5 default, a
    straight uniformly-meshed wire so the eventual hit set is the usual
    O(N) self/neighbour pairs, not the transient this gate measures):

      * default chunking picks `chunk_rows = max(1, 4_000_000 // (25 *
        2000)) = 80`, so each `(chunk_rows, N)` block's reach/cdist/
        scaled-reach/bool stack is capped at `80 * 2000 * 25 = 4.0 MB`.
      * measured peak (this test, tracemalloc): ~6.5 MB — a couple of
        those blocks' worth alive at once during the elementwise chain,
        plus the small O(N) `cand_blk`/`m_parts`/`n_parts` accumulation.
      * the retired single-shot form at this N would need
        `25 * 2000**2 = 100 MB` for its reach/cdist/scaled-reach/bool
        stack alone.

    12 MB therefore sits ~1.85x above the measured peak and ~8x below the
    single-shot transient this replaces.
    """
    import tracemalloc

    n = 2000
    wire = np.array([[0.0, 0.0, -50.0], [0.0, 0.0, 50.0]])
    sim = SinusoidalGalerkinSolver(
        wires=[wire],
        n_per_edge_per_wire=[[n]],
        nsegs=n,
        wavelength=200.0,
    )
    geom = sim._build_geometry()

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        mm, nn = sim._near_pairs(geom)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert mm.size > 0
    assert peak < 12_000_000, (
        f"_near_pairs prefilter peaked {peak / 1e6:.2f} MB at N={n} "
        f"(25 * N**2 = {25 * n**2 / 1e6:.1f} MB) — an N-squared transient "
        "is back"
    )


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
#
# 2026-07-31 (momwire#203): the six `*_gal` entries were re-measured after the
# basis evaluation was folded onto the well-scaled shape set. They moved by
# 1.7e-7 to 1.9e-5 of |Z|; the `*_coll` and `*_bspline2` entries reproduced to
# 1e-9, which is what confines the move to the Galerkin path. The old values
# were the ones carrying error, not these: at the vee's N=321 the literal
# `σA + B·sin + σC·cos` evaluates a 8.2e-6-sized number from O(1) terms and so
# is 1.1e-11 relative off an 80-bit evaluation of the same coefficients, and
# cond(G) = 2.9e5 there — 3e-6 of |Z|, reproduced to within a factor of ~1 by
# perturbing the folded evaluation by exactly that much (the Richardson keys
# then amplify it ~4×, which is why `vee.rich_sin_gal` is the one that moved
# past 1e-5). Nothing about the physics changed; the solver stopped throwing
# away five digits it had already computed.
M3_REFS = {
    "dipole": dict(
        sin_gal_321=69.639093 - 18.056307j,
        sin_coll_321=69.631876 - 18.107822j,
        bspline2_321=69.633780 - 18.065315j,
        rich_sin_gal=69.634796 - 18.008036j,
        rich_sin_coll=69.633551 - 18.018862j,
        rich_bspline2=69.633701 - 18.010747j,
    ),
    "vee": dict(
        sin_gal_321=97.960292 - 61.297668j,
        sin_coll_321=97.947410 - 61.346335j,
        bspline2_321=97.945040 - 61.306749j,
        rich_sin_gal=97.934589 - 61.216983j,
        rich_sin_coll=97.934600 - 61.216284j,
        rich_bspline2=97.930347 - 61.214625j,
    ),
    "k2_junction": dict(
        sin_gal_321=124.493250 + 0.392167j,
        sin_coll_321=124.479522 + 0.340718j,
        bspline2_321=124.492289 + 0.367751j,
        rich_sin_gal=124.513021 + 0.444108j,
        rich_sin_coll=124.513126 + 0.445724j,
        rich_bspline2=124.514726 + 0.445185j,
    ),
    "k3_star": dict(
        sin_gal_321=13.438927 - 951.615918j,
        sin_coll_321=13.438415 - 951.658449j,
        bspline2_321=13.416848 - 950.833391j,
        rich_sin_gal=13.379051 - 949.330644j,
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


def test_loading_is_served_in_the_galerkin_form():
    """Wire loading landed after M5 (momwire#395) as the overlap term the
    testing scheme asks for; the refusal that used to stand here is gone.
    The physics, the oracles and the cross-scheme agreement live in
    `tests/test_wire_loading.py` — this row only pins that the solve runs
    and that the loading is what moved it."""
    dip_hi = np.array([[0.0, -HD, 2.0], [0.0, HD, 2.0]])
    common = dict(wires=[dip_hi], n_per_edge_per_wire=[[41]], nsegs=41)
    z0, _ = SinusoidalGalerkinSolver(**common).compute_impedance()
    z1, _ = SinusoidalGalerkinSolver(
        **common, wire_conductivity=5.8e7
    ).compute_impedance()
    assert z1.real > z0.real  # a resistive loading can only add loss


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


@contextlib.contextmanager
def _without_the_282_contact_correction():
    """Assemble with #282's ground-contact charge correction switched off.

    The correction is a SOURCE-side term (see
    `SinusoidalSolver._contact_charge_kernel`): it removes the charge a
    basis leaves on the plane, and nothing removes it from the test
    function's side, because a test function has no charge to remove. So on
    a deck that touches a FINITE ground it makes the fill non-self-adjoint
    by construction — measured below. Everything in this module that is
    about the fill's own reciprocity therefore measures it with the
    correction off, which is what those tests were always about.
    """
    cls = SinusoidalGalerkinSolver
    orig = cls._contact_charge_correction_tested
    cls._contact_charge_correction_tested = lambda self, G, geom, k, sv, ctx: G
    try:
        yield
    finally:
        cls._contact_charge_correction_tested = orig


def test_g4_sommerfeld_symmetry_near_the_plane_is_source_quadrature_limited():
    """The one symmetry number above the gate at default settings, given a
    cause rather than a carve-out.

    On the ground-contact monopole the Sommerfeld remainder's SOURCE-side
    Gauss rule (`n_qp_sommerfeld`, default 3 — the point-matched solver's
    default, untouched here) is what limits reciprocity: the remainder kernel
    varies fastest when the image point is closest, i.e. exactly at a wire
    touching the plane. Measured 6.70e-10 (q=3) → 7.24e-12 (q=5) → 5.80e-12
    (q=7): a clean quadrature convergence onto the same ~6e-12 floor the PEC
    ground reaches, not a structural asymmetry.

    Refining the TEST rule instead does nothing, which is what identifies the
    source side as the limiter — the same diagnosis M2 made for the
    free-space floor.

    Measured with #282's contact-charge correction OFF: that correction is
    deliberately one-sided and dominates this number when it is on (3.9e-3,
    quadrature-independent — `test_the_282_contact_correction_is_not
    _self_adjoint` pins it). The quadrature statement below is about the
    Sommerfeld remainder's own fill and is unchanged by #282.
    """
    with _without_the_282_contact_correction():
        _g4_sommerfeld_symmetry_body()


def _g4_sommerfeld_symmetry_body():
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


def test_the_282_contact_correction_is_not_self_adjoint():
    """#282's price, measured rather than assumed.

    The contact-charge correction removes the point charge a wire end lying
    in a FINITE ground plane leaves behind — a source-model defect (see
    `SinusoidalSolver._contact_charge_kernel`), so it lands on that basis's
    COLUMN and on nothing else. The fill it produces is therefore not
    symmetric on such a deck: measured 3.9e-3 here against the 5.8e-12 the
    same deck reaches with the correction off, and it does NOT move with any
    quadrature rule, which is what says "structural" rather than "under-
    resolved". A PEC ground or a wire clear of the plane is untouched (the
    correction is identically zero there) and keeps the 1e-13 floor.

    The alternative was a fill that stays symmetric and diverges under mesh
    refinement — the #282 defect itself, ~30x off the cross-basis reference
    and getting worse with every refinement. This test exists so the trade
    is a recorded decision and not a silent regression: if a symmetric
    formulation of the same removal is ever found, this number should fall
    back to the quadrature floor and this test should be deleted.
    """
    somm = _sym_ratio(
        _matrix(_m4_solver(SinusoidalGalerkinSolver, "m4_monopole", "somm"))
    )
    assert somm > 1e-4, (
        f"the contact correction no longer breaks self-adjointness ({somm:.2e}) "
        "— if that is a real fix, delete this test and tighten G4's"
    )
    # Quadrature-independent: it is the correction, not an under-resolved rule.
    fine = _sym_ratio(
        _matrix(
            _m4_solver(
                SinusoidalGalerkinSolver,
                "m4_monopole",
                "somm",
                n_qp_test=16,
                n_qp_near=16,
                n_qp_sommerfeld=7,
            )
        )
    )
    assert abs(fine - somm) / somm < 0.05, (
        f"the asymmetry moved with the rules ({somm:.2e} -> {fine:.2e}) — then "
        "it is quadrature, not the correction"
    )
    # PEC contact and the elevated wire keep the floor: the correction is a
    # no-op without a finite ground, and without a contact.
    pec = _sym_ratio(
        _matrix(_m4_solver(SinusoidalGalerkinSolver, "m4_monopole", "pec"))
    )
    lifted = _sym_ratio(
        _matrix(_m4_solver(SinusoidalGalerkinSolver, "m4_vertical", "somm"))
    )
    assert pec < G1_GATE, f"PEC contact lost its symmetry floor: {pec:.2e}"
    assert lifted < G1_GATE, f"the elevated wire lost its symmetry floor: {lifted:.2e}"


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


def test_finite_ground_at_a_ground_contact_converges_since_282():
    """A wire END LYING IN the plane plus a FINITE ground, which used to be
    broken on BOTH sinusoidal solvers and is now a convergence gate.

    #151's ground-connected basis lets the end current be nonzero by treating
    the plane as a junction with the segment's own image. Over a PEC plane
    that is exact. Over a finite ground the image carries only ρ of the wire
    current, so the wire's end charge no longer cancels against the image's
    and a point charge is left sitting ON the plane — whose potential at the
    nearest collocation point grows like 1/Δ. That was #282: refining the
    mesh made the answer worse, on both sinusoidal solvers and in nec2c
    itself (measured, GN 2, the deck below at NS = 11/21/41: 80.95-106.25j,
    101.26-206.92j, 121.18-314.57j — our own Sommerfeld column to 0.1%).

        n     galerkin           collocation        dense bspline
        11    21.29  -964.39j    63.63   +49.07j    41.66 +22.12j     BEFORE
        21    21.37  -918.65j    63.81   +49.32j    41.71 +22.35j
        81    25.68  -850.88j    64.00   +49.62j    41.76 +22.60j

        11    63.75  +49.44j     63.63   +49.07j    41.66 +22.12j     AFTER
        21    63.87  +49.54j     63.81   +49.32j    41.71 +22.35j
        81    64.02  +49.70j     64.00   +49.62j    41.76 +22.60j

    #282 removes that charge — it is double-counting, not physics: ρ < 1 is
    already the model's statement that the earth takes the current the plane
    does not reflect, and the mixed-potential solvers never had the charge at
    all (`BSplineSolver` builds its charge term from the basis derivative, so
    a ground-contact basis has no end charge — which is why it converges).
    Both sinusoidal schemes now settle to 0.4% over a 7x refinement and agree
    with EACH OTHER to 0.3%.

    What they do NOT do is agree with the b-spline reference: 63.9 vs 41.7 in
    R on this thin-wire deck (Δ/a = 238, where the contact term is at its most
    violent). That gap is recorded, not gated — the two families now differ by
    a cross-basis contact-model difference of fixed size rather than by a
    divergence, and NEC-2 forbids this configuration outright (a wire may not
    connect to a finite ground; only GN 1 or a ground screen).
    """
    e = {"ground_eps": (10.0, 0.002)}
    ns = (11, 21, 81)
    zb = [
        BSplineSolver(
            **_m4_monopole(n), ground_z=0.0, **e, degree=2
        ).compute_impedance()[0]
        for n in ns
    ]
    # B-spline is converged: it barely moves over a 7x refinement.
    assert abs(zb[-1] - zb[0]) / abs(zb[0]) < 0.02, f"bspline reference moved: {zb}"
    zs = {}
    for cls in (SinusoidalSolver, SinusoidalGalerkinSolver):
        z = [
            cls(**_m4_monopole(n), ground_z=0.0, **e).compute_impedance()[0] for n in ns
        ]
        zs[cls.__name__] = z
        spread = abs(z[-1] - z[0]) / abs(z[0])
        assert spread < 0.02, (
            f"{cls.__name__} no longer converges at a finite-ground contact "
            f"(spread {spread:.3f} over NS={ns}): {z}"
        )
        # ...and monotonically, toward its own limit rather than away.
        errs = [abs(v - z[-1]) for v in z]
        assert errs[0] > errs[1], f"{cls.__name__} is not settling: {z}"
    # The two sinusoidal schemes agree with each other — the same contact
    # model through two different testing schemes.
    a, b = zs["SinusoidalSolver"], zs["SinusoidalGalerkinSolver"]
    assert max(abs(x - y) / abs(x) for x, y in zip(a, b)) < 0.01, (
        f"the sinusoidal family disagrees with itself at contact: {a} vs {b}"
    )
    # The recorded cross-basis gap, pinned loosely so a real convergence onto
    # the b-spline answer would show up as a failure to be celebrated.
    gap = abs(a[-1] - zb[-1]) / abs(zb[-1])
    assert 0.2 < gap < 1.0, (
        f"the sinusoidal/bspline contact gap moved to {gap:.3f} — if it "
        "closed, re-derive this test around the agreement"
    )
    # The PEC contact case is sound on the same wires — the contrast that
    # localizes what is left to the non-PEC image, not to ground contact.
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


def test_feed_model_is_validated():
    with pytest.raises(ValueError, match="feed_model"):
        SinusoidalGalerkinSolver(**_dipole(feed_model="delta"))


def test_coll_feed_model_is_validated():
    """The point-matched solver takes the same keyword with the same message,
    so a caller matching the two solvers' feed models writes one spelling."""
    with pytest.raises(ValueError, match="feed_model"):
        SinusoidalSolver(**_dipole(feed_model="delta"))
    assert SinusoidalSolver(**_dipole()).feed_model == "segment"
    assert SinusoidalSolver(**_dipole(feed_model="segment")).feed_model == "segment"


def test_coll_point_feed_model_is_refused_not_defaulted():
    """momwire#212: the zero-width gap is outside the COLLOCATION stack, and
    the refusal says so rather than silently serving the segment gap.

    A caller who flips `SinusoidalGalerkinSolver`'s default to `"point"` and
    tries to feed-match its point-matched sibling has to be told the pairing
    does not exist — a silent fallback would produce exactly the two-feed-model
    comparison report §16 exists to prevent. Derivation in
    `SinusoidalSolver._reject_point_feed_model`; measured refutation in
    `test_m6_instrument_report.py`.
    """
    with pytest.raises(NotImplementedError, match="zero-width gap"):
        SinusoidalSolver(**_dipole(feed_model="point"))
    # The sibling's option is unaffected — this is a testing-scheme limit.
    assert SinusoidalGalerkinSolver(**_dipole(feed_model="point")).feed_model == "point"


@pytest.mark.parametrize("n", [21, 41, 81])
@pytest.mark.parametrize("readout", ["centre", "variational"])
def test_point_gap_feed_y_is_symmetric_in_both_readouts(n, readout):
    """`feed_model="point"` makes the gap-feed Y block self-dual (momwire#192).

    The zero-width gap's drive column IS the basis-evaluation vector at the
    feed segment's centre, which is what the DEFAULT centre readout reads, so
    drive and readout are the same functional and Y = −UᵀG⁻¹U up to the fill's
    own reciprocity floor — no `feed_readout="variational"` opt-in and none of
    the M3 payoff traded. Measured 3.1e-13 / 3.9e-12 / 5.6e-12 at N = 21/41/81,
    identical under both readouts.

    The segment-gap control is the adversarial half: it is the O(h) asymmetry
    `test_gap_feed_readout_is_not_its_drives_dual` pins (6.7e-5 / 2.4e-5 /
    6.5e-6), so this test cannot pass on a `feed_model` that silently did
    nothing.
    """

    def asym(**kw):
        Y = np.asarray(
            SinusoidalGalerkinSolver(
                **_two_feed_dipole(n=n, wavelength=WL, feed_readout=readout, **kw)
            ).compute_y_matrix()
        )
        return np.linalg.norm(Y - Y.T) / np.linalg.norm(Y)

    a_point = asym(feed_model="point")
    a_segment = asym(feed_model="segment")
    assert a_point < 1e-10, f"the point gap is not self-dual: {a_point:.3e}"
    if readout == "centre":
        # The control: without the option this IS the measurement, and it is
        # five orders worse.
        assert a_segment > 1e-6, a_segment
        assert a_point < 1e-3 * a_segment, (a_point, a_segment)


@pytest.mark.parametrize("n", [21, 41, 81])
def test_point_gap_readouts_coincide(n):
    """Under `feed_model="point"` the two `feed_readout` branches are the same
    functional, so the knob stops having consequences at gap feeds.

    `_port_currents`'s centre branch sums σ(A+C)·α over the feed segment's
    support entries; its variational branch is −U[:, j]·α, and the point-gap
    drive column is that same vector scattered onto basis rows. Measured
    driving-point spread between the readouts: 0.0 / 0.0 / 1.7e-16 at
    N = 21/41/81 — against the segment gap's 2.2e-3 / 1.4e-3 / 8.2e-4, which
    is the O(h) gap-average-vs-centre difference the default trades.
    """

    def z(**kw):
        s = SinusoidalGalerkinSolver(**_two_feed_dipole(n=n, wavelength=WL, **kw))
        return np.atleast_1d(s.compute_impedance()[0])

    point = [z(feed_model="point", feed_readout=r) for r in ("centre", "variational")]
    np.testing.assert_allclose(point[0], point[1], rtol=1e-14)

    segment = [z(feed_readout=r) for r in ("centre", "variational")]
    spread = np.abs(segment[0] - segment[1]).max() / np.abs(segment[1]).max()
    assert spread > 1e-5, spread

    # The identity behind it, on an arbitrary current vector rather than on a
    # solved one: the drive column IS the centre-evaluation functional.
    s = SinusoidalGalerkinSolver(
        **_two_feed_dipole(n=n, wavelength=WL, feed_model="point")
    )
    geom = s._build_geometry()
    seg_view = s._basis_coefs(geom, s.k)
    U = s._drive_columns(geom, seg_view, s.k)
    rng = np.random.default_rng(0)
    alpha = rng.standard_normal(geom["n_segs"]) + 1j * rng.standard_normal(
        geom["n_segs"]
    )
    for j, fseg in enumerate(geom["feed_segs"]):
        dual = -(U[:, j] @ alpha)
        centre = s._feed_segment_current(alpha, seg_view, fseg)
        assert abs(dual - centre) <= 1e-14 * abs(centre), (j, dual, centre)


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


# ---------------------------------------------------------------------------
# momwire#194 — the far fill is blocked over test segments
# ---------------------------------------------------------------------------
# The profile (scripts/profile_sinusoidal_galerkin.py) showed the unblocked
# fill's (N·nq, N, n_qp_const) kernel scratch peaking at 18.6 GiB by N=1601
# and OOMing at N≈2000 — the M6 census ceiling. Blocking bounds the scratch;
# these tests pin that it changes NOTHING else: the arithmetic per matrix
# entry is identical. The gate is a few ULPs rather than array_equal
# (issue #236): numpy may pick a different einsum/reduction kernel per
# block shape, and reassociating a per-entry sum moves the last bits —
# observed as a deterministic ULP-level k2_junction mismatch on one venv's
# numpy while three CI platforms were bit-exact. Anything past a few ULPs
# is still a real blocking bug (an indexing or projector error is orders
# of magnitude larger), so the gate keeps its teeth.


def _assert_ulp_close(A, B, max_ulp=8):
    """ULP gate at the MATRIX'S complex scale, not per entry: reassociation
    noise is absolute at the magnitude of the summands, and G's real part is
    a cancellation residue ~5 orders below its imag part (measured: dipole
    re 8e-7 vs im 3.4e-2), so a per-entry relative gate would demand the
    impossible exactly where cancellation happens. Measured worst on the
    affected venv: 2.2 ULPs at scale (k2_junction re); a real blocking bug
    (indexing, projector) lands at the scale itself, ~16 orders above tol."""
    scale = max(np.max(np.abs(A)), np.max(np.abs(B)))
    tol = max_ulp * np.spacing(scale)
    worst = np.max(np.abs(A - B))
    assert worst <= tol, (
        f"worst |A-B|={worst:.3e} = {worst / np.spacing(scale):.1f} ULPs "
        f"at matrix scale {scale:.3e} (gate: {max_ulp})"
    )


def _G_at_block_budget(monkeypatch, budget, cls=SinusoidalGalerkinSolver, **kw):
    """G at a forced fill-workspace budget, on the NUMPY path.

    Blocking is a property of the numpy fill — the fused C++ far fill never
    forms the scratch the budget bounds and ignores it — so the accelerator is
    switched off here or the comparison would be two identical C++ fills.
    """
    from momwire import sinusoidal_galerkin as _sg

    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", False)
    monkeypatch.setattr(_sg, "_FILL_WORKSPACE_BYTES", budget)
    return _matrix(cls(**kw))


@pytest.mark.parametrize("geom_name", ["dipole", "vee", "k2_junction", "k3_star"])
def test_far_fill_blocking_is_ulp_exact(monkeypatch, geom_name):
    """Forced one-segment blocks reproduce the single-block G to a few ULPs."""
    kw = GEOMETRIES[geom_name]()
    G_one_block = _G_at_block_budget(monkeypatch, 1 << 62, **kw)
    G_tiny_blocks = _G_at_block_budget(monkeypatch, 1, **kw)
    _assert_ulp_close(G_one_block, G_tiny_blocks)


@pytest.mark.parametrize(
    "ground_kw",
    [
        dict(ground_z=-6.0),
        dict(ground_z=-6.0, ground_eps=(13.0, 0.005)),
    ],
    ids=["pec-image", "refl-coef"],
)
def test_far_fill_blocking_is_ulp_exact_over_ground(monkeypatch, ground_kw):
    """The ground blocks share `_tested_contribs`, so blocking must be
    invisible there too — including the per-pair Fresnel projector, whose
    tables are indexed by GLOBAL test-segment index from inside a block."""
    kw = _dipole(**ground_kw)
    G_one_block = _G_at_block_budget(monkeypatch, 1 << 62, **kw)
    G_tiny_blocks = _G_at_block_budget(monkeypatch, 1, **kw)
    _assert_ulp_close(G_one_block, G_tiny_blocks)


# ---------------------------------------------------------------------------
# momwire#194 step 2 — the C++ far fill
# ---------------------------------------------------------------------------
# `sinusoidal_galerkin_far_fill` fuses the Eqs 76-79 kernel with the test
# reduction for the PLAIN-projected blocks (free space, PEC image). The numpy
# fill stays the reference implementation, so the gate is a matrix-level
# agreement: the same G to reassociation. `1e-13` is the house tolerance for
# the sinusoidal accelerator agreements (test_momwire's field-tensor test uses
# it too); measured here it sits at 1.3e-14 - 5.4e-14, the two paths differing
# only in summation order — the C++ scalar arithmetic against numpy's ufunc
# chains. It is NOT thread-count sensitive: the kernel parallelizes over test
# segments and each segment's rows are summed by one thread in the numpy
# path's own quadrature-node order.

ACCEL_AGREEMENT = 1e-13


def _G_accel_and_numpy(monkeypatch, **kw):
    """(G with the accelerator, G with it switched off) on one geometry."""
    from momwire import sinusoidal_galerkin as _sg

    G_acc = _matrix(SinusoidalGalerkinSolver(**kw))
    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", False)
    G_py = _matrix(SinusoidalGalerkinSolver(**kw))
    return G_acc, G_py


def _rel_matrix_delta(G_acc, G_py):
    return np.linalg.norm(G_acc - G_py) / np.linalg.norm(G_py)


@pytest.mark.parametrize("geom_name", ["dipole", "vee", "k2_junction", "k3_star"])
def test_accelerated_far_fill_matches_numpy(monkeypatch, geom_name):
    """Free space: the C++ fill reproduces the numpy matrix entrywise."""
    G_acc, G_py = _G_accel_and_numpy(monkeypatch, **GEOMETRIES[geom_name]())
    assert _rel_matrix_delta(G_acc, G_py) < ACCEL_AGREEMENT


@pytest.mark.parametrize(
    "over",
    [
        dict(ground_z=-6.0),
        dict(ground_z=-6.0, ground_eps=(13.0, 0.005)),
        dict(ground_z=-6.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"),
        dict(wire_radius=[0.0005, 0.004]),
    ],
    ids=["pec-image", "refl-coef", "sommerfeld", "mixed-radius"],
)
def test_accelerated_far_fill_matches_numpy_per_block(monkeypatch, over):
    """Each dispatch branch, against its numpy reference.

    The PEC image block goes through the kernel like free space (same plain
    projector, mirrored sources). The refl-coef and Sommerfeld grounds keep
    their numpy image blocks while their free-space block is accelerated —
    the mixed case, which is where a wrong dispatch would show. Mixed per-wire
    radii exercise the kernel's per-OBSERVER radius argument (the point-matched
    kernel takes a scalar and splits into radius runs instead).
    """
    kw = _k2_junction(**over) if "wire_radius" in over else _dipole(**over)
    G_acc, G_py = _G_accel_and_numpy(monkeypatch, **kw)
    assert _rel_matrix_delta(G_acc, G_py) < ACCEL_AGREEMENT


def test_accelerated_fill_meets_the_g1_symmetry_gate():
    """The accelerated fill is held to G1 itself, not just to agreement with
    numpy: a reduction that mis-pairs weights with rows would still agree with
    a numpy path making the same mistake, but could not stay symmetric."""
    from momwire import sinusoidal_galerkin as _sg

    if not _sg._HAVE_GALERKIN_FAR_FILL:
        pytest.skip("accelerator not built")
    for name, factory in GEOMETRIES.items():
        G = _matrix(SinusoidalGalerkinSolver(**factory()))
        assert _sym_ratio(G) < G1_GATE, name


def test_far_fill_dispatch_is_projector_selective(monkeypatch):
    """The kernel serves the plain projector only. Free space takes it once;
    a PEC image takes it twice (free-space + image block); the refl-coef
    ground takes it once and leaves its Fresnel-weighted image block on numpy.
    """
    from momwire import sinusoidal_galerkin as _sg

    if not _sg._HAVE_GALERKIN_FAR_FILL:
        pytest.skip("accelerator not built")

    calls = []
    real = SinusoidalGalerkinSolver._far_fill_accel

    def counted(self, *a, **kw):
        calls.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(SinusoidalGalerkinSolver, "_far_fill_accel", counted)
    for over, expected in (
        ({}, 1),
        (dict(ground_z=-6.0), 2),
        (dict(ground_z=-6.0, ground_eps=(13.0, 0.005)), 1),
    ):
        calls.clear()
        _matrix(SinusoidalGalerkinSolver(**_dipole(**over)))
        assert len(calls) == expected, over


# ---------------------------------------------------------------------------
# momwire#332 unit C — the ground block folds into the free-space triple
# ---------------------------------------------------------------------------
# The tested assembly's structural cost is three (nnz, N) contribution arrays,
# one per folded shape — nnz-major, and what the fill is FOR. What #332 unit C
# took away is the grounded path's habit of building a second such triple for
# the ground and a third for `free − ground`, so that any ground cost 3x the
# free-space residency. `_fold_ground_block` subtracts in place instead: the
# numpy fill accumulates as it blocks (one triple total), the fused C++ fill
# still returns its own and is folded in on return (two). Both spellings do
# the same float64 subtraction per matrix entry, so the answer is bit-equal
# rather than reassociated — which is what the first gate below pins.


def _whole_slab_remainder(sim, ctx, geom, eps_t):
    """The Sommerfeld remainder as a triple of its own, off the WHOLE
    (3, N·nq, N) tensor — `_tested_sommerfeld_remainder`'s pre-#332-unit-D
    body, kept here so the oracle below still spans old arithmetic against
    new.

    The shipped path streams instead: the evaluator's observer chunks are
    reduced and folded one at a time and the tensor never exists whole. What
    this reference holds fixed is the reduction the streaming version has to
    reproduce — every entry's nq nodes summed in node order, off a slab that
    was built in one piece — so an agreement here says the chunk boundaries
    really do fall between test segments rather than through one.
    """
    N, nq = ctx["N"], ctx["nq"]
    S = sim._field_tensor_sommerfeld_remainder(
        geom,
        sim.k,
        eps_t,
        obs_centers=ctx["obs_c"],
        obs_tangents=ctx["obs_t"],
        cos_shape="cos-1",
    )
    return tuple(
        sim._tested_contrib_rows(
            ctx["w_entry"], ctx["m_of_entry"], nq, s.reshape(N, nq, N)
        )
        for s in S
    )


def _differenced_grounded_G(sim, geom):
    """G with the ground built as a whole parallel triple and differenced —
    `_assemble_Z`'s pre-#332 spelling, kept here as the fold's reference.

    Deliberately a second spelling of the orchestration rather than a call
    into `_fold_ground_block` with a scratch destination: what is under test
    is that folding in place reaches the same entries the difference did,
    including the near-correction cells, which the fold has to write BEFORE
    the far half rather than over the top of it.

    The Sommerfeld half reaches further back still, to before unit D: its
    remainder comes from `_whole_slab_remainder` above, so the comparison
    covers the streamed reduction as well as the fold.

    The three-way ground branch is spelled out from `ground_eps` /
    `ground_model` here on purpose, and stayed that way through momwire#397
    unit 3 while `_fold_ground_block` stopped reading those strings: what this
    reference has to be independent of is the production DISPATCH. The dyad
    itself is the shared builder's (`_image_refl_prep(...).weights(...)
    .project`, one spelling since unit 1) because a hand-rolled second
    Fresnel chain would be measuring the projector, not the fold.
    """
    from momwire import _ground_refl

    k = sim.k
    seg_view = sim._basis_coefs(geom, k)
    ctx = sim._test_context(geom, seg_view, k)
    contribs = sim._tested_contribs(geom, k, ctx, _plain_projection)

    src_c, src_t = sim._image_source_centers_tangents(geom)
    if sim.ground_eps is None:
        gnd = sim._tested_contribs(
            geom, k, ctx, _plain_projection, src_c, src_t, mirror=True
        )
    else:
        eps_t = _ground_refl.eps_tilde(sim.ground_eps, sim.omega, sim.eps)
        if sim.ground_model == "sommerfeld":
            c2 = (eps_t - 1.0) / (eps_t + 1.0)
            img = sim._tested_contribs(
                geom, k, ctx, _plain_projection, src_c, src_t, mirror=True
            )
            rem = _whole_slab_remainder(sim, ctx, geom, eps_t)
            gnd = tuple(c2 * a - b for a, b in zip(img, rem))
        else:
            gnd = sim._tested_contribs(
                geom,
                k,
                ctx,
                sim._image_refl_prep(geom).weights(eps_t).project,
                src_c,
                src_t,
                mirror=True,
            )
    contribs = tuple(c - g for c, g in zip(contribs, gnd))

    G = sim._scatter_coef_product(ctx, contribs)
    sim._ek_bracket_correction_tested(
        G, geom, k, ctx, _field_ground.field_ground_for(sim, geom, k, sim.omega)
    )
    sim._contact_charge_correction_tested(G, geom, k, seg_view, ctx)
    return G


# Every (geometry, ground) M4 runs, reduced; plus the extended kernel on all
# of them but the L. The L is left out of the EK half only for its cost — the
# bracket correction on a junction is ~1 s a solve at any mesh, because the
# graded rule the near cells take grows as the mesh COARSENS — and what it
# would add over the vertical is a junction, which the free-space fill and
# the scatter share with every case here.
#
# The ground-contact monopole is here under the FINITE grounds too, which
# `M4_CASES` excludes on physics grounds (see
# `test_finite_ground_at_a_ground_contact_is_an_inherited_defect`) — it is the
# only geometry whose IMAGE block has near pairs at all, so it is the only one
# that exercises the fold's near-correction reordering, and on a non-plain
# projector that reordering is on the shipped dispatch rather than behind the
# accelerator. Whether the answer means anything physically does not bear on
# whether two spellings of it agree.
#
# Under `somm` it is the GROUND-STANDING deck the streamed remainder needs
# (#332 unit D) — the reduction it streams is keyed on the test-segment
# support runs, so a geometry whose bases reach across a ground contact is
# where a mis-sliced run would show.
_FOLD_CASES = (
    [(g, gnd, False) for g, gnd in M4_CASES]
    + [(g, gnd, True) for g, gnd in M4_CASES if g != "m4_lshape"]
    + [("m4_monopole", gnd, ek) for gnd in ("refl", "somm") for ek in (False, True)]
)


@pytest.mark.parametrize("geom_name,ground,extended_kernel", _FOLD_CASES)
@pytest.mark.parametrize("accel", [True, False], ids=["accel", "numpy-blocked"])
def test_folded_ground_is_bit_equal_to_the_differenced_spelling(
    monkeypatch, geom_name, ground, accel, extended_kernel
):
    """Exact equality, not a tolerance: the fold is the same subtraction per
    entry, so anything at all here is a bug rather than reassociation.

    Both dispatches are covered because they fold at different moments — the
    numpy fill subtracts each block as it reduces it and has to run the near
    correction FIRST (it overwrites its cells, and under the fold what it
    would overwrite is the free-space value), while the accelerated fill folds
    the kernel's own return afterwards. The numpy cases run at a one-segment
    block budget so the near-correction cells are spread over many blocks, and
    the extended kernel rides along on both because its delta reaches the
    ground blocks through exactly these arrays.
    """
    from momwire import sinusoidal_galerkin as _sg

    if not accel:
        monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", False)
        monkeypatch.setattr(_sg, "_FILL_WORKSPACE_BYTES", 1)
    over = {"extended_kernel": True} if extended_kernel else {}
    # Coarser than the M4 physics gates' 21: what is under test is an
    # arithmetic identity per matrix entry, which needs every branch of the
    # dispatch present and nothing at all from the mesh. Odd, so the feed
    # stays a segment centre and the geometries keep their usual shape.
    sim = _m4_solver(SinusoidalGalerkinSolver, geom_name, ground, n=11, **over)
    geom = sim._build_geometry()
    G_folded, _ = sim._assemble_Z(geom, sim.k)
    G_differenced = _differenced_grounded_G(sim, geom)
    assert np.array_equal(G_folded, G_differenced), (
        f"{geom_name}/{ground}: folding the ground moved the matrix by "
        f"{np.abs(G_folded - G_differenced).max():.3e}"
    )


def _residency_deck(n, **ground_kwargs):
    """`n` short segments on one straight wire, 4 m over the plane — the
    residency gates' deck. N² and nnz·N are both large enough at n = 300 that
    the triples dominate the trace, small enough that even the numpy-path
    refl-coef fill stays a couple of seconds.
    """
    ys = np.linspace(-HD, HD, n + 1)
    wires = [np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, 4.0)])]
    return SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=[[1] * n],
        nsegs=n,
        wavelength=WL,
        **ground_kwargs,
    )


@pytest.mark.memgate
@pytest.mark.parametrize(
    "ground_kwargs",
    [
        pytest.param({"ground_z": 0.0}, id="pec"),
        pytest.param(
            {"ground_z": 0.0, "ground_eps": (13.0, 0.005)},
            id="refl-coef",
        ),
    ],
)
def test_grounded_fill_holds_no_parallel_ground_triple(monkeypatch, ground_kwargs):
    """Tracemalloc gate on the real `_assemble_Z` path (issue #332).

    Budget is stated in TRIPLES — 3 × 16·nnz·N, the fill's own structural
    unit — because that is what the defect was denominated in. At the N = 300
    segments this geometry builds, nnz = 898 (~3N) and one triple is
    12.93 MB against G's 1.44 MB, so the triples are what the peak is made
    of. Measured above G, cold:

      * differenced (the pre-#332 spelling): 37.73 MB = 2.92 triples for the
        PEC image and 41.33 MB = 3.20 for the refl-coef ground — free-space
        triple, parallel ground triple, differenced result, plus the
        refl-coef ground's specular and Fresnel tables;
      * folded: 27.07 MB = 2.09 triples PEC (the destination plus the fused
        kernel's own return, which is allocated in C++ and cannot be
        accumulated into) and 24.00 MB = 1.86 refl-coef (one destination
        triple, and its Fresnel tables).

    2.5 triples therefore sits above both folded numbers and below both
    differenced ones, by ~1.2x each way. That margin is thin by this suite's
    standards and it is thin for a structural reason: the accelerated path
    can only ever go from three triples to two, so no threshold separating
    them has more room than 3:2 to spend.

    Two module constants are shrunk for the trace, both fixed working sets
    that do not scale with the triples and would otherwise bury them:
    `_FILL_WORKSPACE_BYTES` (the numpy fill's per-block kernel scratch, which
    the refl-coef ground pays and which is a whole gigabyte by default) and
    `_PAIR_BLOCK` (the near correction's per-pair-block scratch, which was
    ~124 MB at the flat 512 that used to ship, regardless of N; momwire#383
    now sizes that block to `_NEAR_WORKSPACE_BYTES` and it is ~8 MB on this
    reduced-kernel deck, so the shrink to 8 pairs is a ceiling under a
    budget rather than the whole of the lever). Same lever as the
    point-matched gates' `swept_mem_mb = 1`.

    The fill runs COLD, on purpose: a warm-up call would move the refl-coef
    ground's cached specular tables out of the traced region, and the lesson
    of the point-matched gate's own cold twin is that anything a warm-up
    hides is something this gate stops being able to see.
    """
    import tracemalloc

    from momwire import sinusoidal_galerkin as _sg

    monkeypatch.setattr(_sg, "_FILL_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(_sg, "_PAIR_BLOCK", 8)

    n = 300
    sim = _residency_deck(n, **ground_kwargs)
    geom = sim._build_geometry()
    N = geom["n_segs"]
    assert N == n
    nnz = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)[
        "w_entry"
    ].shape[0]
    triple = 3 * 16 * nnz * N

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        G, _ = sim._assemble_Z(geom, sim.k)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    transient = peak - G.nbytes
    assert transient < 2.5 * triple, (
        f"grounded fill peaked {transient / 1e6:.2f} MB above G = "
        f"{transient / triple:.2f} contribution triples (one triple = "
        f"{triple / 1e6:.2f} MB) — a parallel ground triple is back"
    )


# ---------------------------------------------------------------------------
# momwire#332 unit D — the Sommerfeld remainder streams instead of slabbing
# ---------------------------------------------------------------------------
# The Sommerfeld ground's second half is a smooth remainder field, evaluated by
# the point-matched solver's own tensor builder with this solver's test
# quadrature points as its observers. That makes its tensor (3, N·n_qp_test, N)
# — `n_qp_test` = 8 times the matrix, and several times the (nnz, N) triple it
# immediately reduces to. Unit C's fold could not touch it: it was still the
# largest single thing the grounded assembly held.
#
# Unit D reduces it INSIDE the evaluator's existing observer-chunk loop and
# folds each chunk's rows straight off the C2-scaled image, so nothing bigger
# than one chunk of it is ever live. The two gates below are the arithmetic
# one (chunking must not reassociate a test entry's quadrature sum) and the
# residency one.


_SOMM_GROUND = {
    "ground_z": 0.0,
    "ground_eps": (10.0, 0.002),
    "ground_model": "sommerfeld",
}


def _count_remainder_chunks(monkeypatch):
    """Count the evaluator's observer chunks by counting the grid-interpolation
    calls, which is one per chunk and happens nowhere else in a fill."""
    from momwire import _sommerfeld as _sm

    calls = []
    original = _sm.remainder_field_proj

    def counting(*a, **kw):
        calls.append(a[0].shape[0])
        return original(*a, **kw)

    monkeypatch.setattr(_sm, "remainder_field_proj", counting)
    return calls


# Chunk budgets in complex entries, against the n = 11 decks below: n_src =
# N·n_qp_sommerfeld = 33, so the shipped 1<<19 is ~15,900 observer rows and
# swallows all 88 of them whole, while 660 is 20 rows (16 after the alignment
# rounding = 2 test segments) and 1 is the degenerate floor, where the
# alignment is the only thing keeping a chunk from splitting a test segment.
_CHUNK_CASES = [
    pytest.param(1, 11, id="chunk-one-segment"),
    pytest.param(660, 6, id="chunk-two-segments"),
    pytest.param(1 << 19, 1, id="chunk-shipped"),
]


@pytest.mark.parametrize("chunk_elems,n_chunks", _CHUNK_CASES)
@pytest.mark.parametrize("extended_kernel", [False, True], ids=["reduced", "ek"])
@pytest.mark.parametrize("geom_name", ["m4_dipole", "m4_monopole"])
def test_streamed_remainder_is_bit_equal_at_every_chunk(
    monkeypatch, geom_name, extended_kernel, chunk_elems, n_chunks
):
    """Exact equality against the whole-slab reduction, at three chunk sizes
    down to one test segment per chunk.

    This is the gate the streaming design is FOR. A test entry's contribution
    is a sum over its `n_qp_test` quadrature nodes, accumulated one node at a
    time; split those nodes across two chunks and the two halves could only
    meet as a partial sum, which is a reassociation of the same products
    (#203/#205's subject). The evaluator is therefore told to round its chunks
    down to whole test segments (`row_group=nq`), and what that buys is
    `array_equal` rather than a tolerance — at the shipped chunk, where the
    88 observer rows of this deck fit in one piece, and at one segment per
    chunk, where they do not.

    The reference is built at the SHIPPED chunk in every case, so the
    comparison also covers the evaluator's own claim that its per-chunk
    einsum is chunk-independent — the contraction is over the source
    quadrature axis alone, which no observer boundary touches.

    Both decks are here because the reduction is keyed on the test segments'
    support runs: the elevated dipole's bases stop at the wire ends, the
    ground-standing monopole's reach across a ground contact (#151), and a
    mis-sliced run would land differently in the two. The extended kernel
    rides along because it reaches the C2-scaled image half of this ground
    while the remainder itself stays reduced (momwire#287) — which is exactly
    the composition the fold has to keep associated.
    """
    from momwire import sinusoidal as _sin

    over = {"extended_kernel": True} if extended_kernel else {}
    sim = SinusoidalGalerkinSolver(
        **M4_GEOMETRIES[geom_name](11, **over), **_SOMM_GROUND
    )
    geom = sim._build_geometry()
    G_ref = _differenced_grounded_G(sim, geom)

    monkeypatch.setattr(_sin, "_REMAINDER_CHUNK_ELEMS", chunk_elems)
    chunks = _count_remainder_chunks(monkeypatch)
    G_streamed, _ = sim._assemble_Z(geom, sim.k)

    assert len(chunks) == n_chunks, f"chunking not exercised: rows per chunk {chunks}"
    assert np.array_equal(G_streamed, G_ref), (
        f"{geom_name}: streaming the remainder at {chunk_elems} entries per "
        f"chunk moved the matrix by {np.abs(G_streamed - G_ref).max():.3e}"
    )


@pytest.mark.memgate
def test_sommerfeld_fill_streams_the_remainder(monkeypatch):
    """Tracemalloc gate on the real `_assemble_Z` path, Sommerfeld ground
    (issue #332 unit D) — the sibling of
    `test_grounded_fill_holds_no_parallel_ground_triple` above, same deck,
    same shrunk constants, same TRIPLE unit (3 × 16·nnz·N).

    This ground is the expensive one and it was the last thing #332 had not
    reached. Measured above G on the n = 300 deck, cold, where one triple is
    12.93 MB and G is 1.44 MB:

      * before unit D: 81.05 MB = 6.27 triples. Unit C's fold had already
        taken the composition from 3.91 to 2.91 triples, and the remainder's
        own (3, N·n_qp_test, N) tensor — 34.56 MB = 2.67 triples here — plus
        the (nnz, N) gathers its reduction made on top put it all back;
      * after: 41.52 MB = 3.21 triples. Two triples are the accelerated
        floor (the free-space destination and the image block the fused C++
        kernel returns); the rest is ~8.4 MB of ONE observer chunk, which
        `_REMAINDER_CHUNK_ELEMS` fixes regardless of N, and ~7.4 MB of cold
        SommerfeldGrid.
      * at N = 400 the same two: 143.78 MB = 56.16x G before, 60.63 MB =
        23.68x G after — the fixed 16 MB shrinks against the triples as the
        mesh grows, so 6.25 -> 2.64 triples there.

    4.5 triples sits 1.40x above the streamed number and 1.39x below the
    slabbed one — a wider margin each way than the fold gate's 1.2x, and it
    is wider for the same structural reason stated there in reverse: what
    unit D removed scales with `n_qp_test` rather than with 3:2.
    """
    import tracemalloc

    from momwire import sinusoidal_galerkin as _sg

    monkeypatch.setattr(_sg, "_FILL_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(_sg, "_PAIR_BLOCK", 8)

    n = 300
    sim = _residency_deck(n, **_SOMM_GROUND)
    geom = sim._build_geometry()
    N = geom["n_segs"]
    assert N == n
    nnz = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)[
        "w_entry"
    ].shape[0]
    triple = 3 * 16 * nnz * N

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        G, _ = sim._assemble_Z(geom, sim.k)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    transient = peak - G.nbytes
    assert transient < 4.5 * triple, (
        f"sommerfeld fill peaked {transient / 1e6:.2f} MB above G = "
        f"{transient / triple:.2f} contribution triples (one triple = "
        f"{triple / 1e6:.2f} MB) — the remainder tensor is materialized again"
    )


# ---------------------------------------------------------------------------
# momwire#198 — sparse scatter/coefficient assembly
# ---------------------------------------------------------------------------
# `_assemble_Z`'s tail forms G = Σ_shape (R @ contrib[shape]) @ M[shape]. Both
# factors carry one nonzero per support entry, so the dense spelling (an
# nnz-way `np.add.at` plus three (n_basis,N)@(N,n_basis) matmuls) does O(N³)
# BLAS work on matrices that are 3/N full. The sparse spelling is the same
# product, and the crossover is the point-matched solver's own
# `_DENSE_ASSEMBLY_THRESHOLD` — measured here on the Galerkin tail at N ≈ 57
# (N=40: 0.31 ms dense / 0.48 ms sparse; N=80: 1.54 / 0.95; N=400: 49.5 / 8.1).
#
# The tolerance used to be 1e-12 rather than the 1e-13 house figure the
# accelerator agreements use, and the reason was conditioning rather than
# sloppiness: the const- and cos-shape products very nearly cancelled, so any
# reassociation of that sum landed ~1e-13 away and neither spelling was the
# accurate one (against an 80-bit reference built from the same tested
# contributions, dense sat 1.16e-11 and sparse 1.26e-11 away at N=401).
#
# 2026-07-31 (momwire#203): the sum is folded before the product now, so there
# is nothing left for the reassociation to amplify — the two spellings land
# 1.1e-16 to 1.5e-16 apart on the set below, and the ported pair, which the
# M5b corrections used to amplify by ~40×, comes in at 2e-17. One gate covers
# both; see the #203 section at the bottom for the before/after table.

SPARSE_ASSEMBLY_AGREEMENT = 1e-15  # measured 1.1e-16 - 1.5e-16 on this set
# How far either spelling sits from the EXACT product of the same float64
# contributions — a different quantity from the gate above (which is how far
# they sit from each other) and a looser one, because it grows with mesh:
# 1.2e-15 at N=41, 2.8e-15 at N=401, 1.4e-14 at N=801, 8.3e-14 at N=2401 (#203;
# the literal pairing was at 7.3e-13 / 1.2e-11 / 7.6e-11 / 4.5e-10).
EXACT_PRODUCT_DISTANCE = 1e-14


def _G_by_assembly_path(monkeypatch, *, sparse, ported=False, **kw):
    """G with the assembly-tail path forced, by moving the dense/sparse
    threshold past (or below) this geometry's N."""
    from momwire import sinusoidal_galerkin as _sg

    monkeypatch.setattr(_sg, "_DENSE_ASSEMBLY_THRESHOLD", 0 if sparse else 1 << 30)
    sim = SinusoidalGalerkinSolver(**kw)
    geom = sim._build_geometry()
    assemble = sim._assemble_Z_ported if ported else sim._assemble_Z
    return assemble(geom, sim.k)[0]


def _both_assembly_paths(monkeypatch, *, ported=False, **kw):
    return (
        _G_by_assembly_path(monkeypatch, sparse=False, ported=ported, **kw),
        _G_by_assembly_path(monkeypatch, sparse=True, ported=ported, **kw),
    )


@pytest.mark.parametrize("geom_name", ["dipole", "vee", "k2_junction", "k3_star"])
def test_sparse_assembly_matches_dense(monkeypatch, geom_name):
    G_dense, G_sparse = _both_assembly_paths(monkeypatch, **GEOMETRIES[geom_name]())
    assert _rel_matrix_delta(G_sparse, G_dense) < SPARSE_ASSEMBLY_AGREEMENT


@pytest.mark.parametrize(
    "over",
    [
        dict(ground_z=-6.0),
        dict(ground_z=-6.0, ground_eps=(13.0, 0.005)),
        dict(ground_z=-6.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"),
    ],
    ids=["pec-image", "refl-coef", "sommerfeld"],
)
def test_sparse_assembly_matches_dense_over_ground(monkeypatch, over):
    """The ground blocks are subtracted from the free-space contributions
    BEFORE the scatter, so they reach the product through the same arrays —
    but each ground adds its own cancellation to the sum being reassociated."""
    G_dense, G_sparse = _both_assembly_paths(monkeypatch, **_dipole(**over))
    assert _rel_matrix_delta(G_sparse, G_dense) < SPARSE_ASSEMBLY_AGREEMENT


@pytest.mark.parametrize("geom_name", ["k2_junction", "k3_star"])
@pytest.mark.parametrize("ported", [False, True], ids=["reaction-form", "ported"])
def test_sparse_assembly_matches_dense_with_junction_ports(
    monkeypatch, geom_name, ported
):
    """Junction ports are the case where "one nonzero per support entry" has
    to be checked rather than assumed: a port column is supported on all K of
    its members, so M[shape] gets K nonzeros in that column and G grows to
    (N+P, N+P). `_assemble_Z_ported` goes through `_assemble_Z` too."""
    kw = GEOMETRIES[geom_name](junction_ports=[0])
    G_dense, G_sparse = _both_assembly_paths(monkeypatch, ported=ported, **kw)
    assert G_dense.shape == (kw["nsegs"] * len(kw["wires"]) + 1,) * 2
    assert _rel_matrix_delta(G_sparse, G_dense) < SPARSE_ASSEMBLY_AGREEMENT


@pytest.mark.parametrize("geom_name", ["dipole", "vee", "k2_junction", "k3_star"])
def test_sparse_assembly_meets_the_g1_symmetry_gate(monkeypatch, geom_name):
    """G1 held against the sparse path itself, not just against the dense
    matrix: an index swapped between the scatter and the coefficient matrix
    would still be close to dense in norm but could not stay symmetric."""
    G = _G_by_assembly_path(monkeypatch, sparse=True, **GEOMETRIES[geom_name]())
    assert _sym_ratio(G) < G1_GATE


@pytest.mark.parametrize("geom_name", ["dipole", "vee", "k2_junction", "k3_star"])
@pytest.mark.parametrize("ports", [None, [0]], ids=["no-ports", "junction-port"])
def test_support_entries_name_distinct_segment_basis_pairs(geom_name, ports):
    """The invariant the sparse coefficient matrices rest on. `csc_matrix`
    SUMS duplicate (row, col) triplets where the dense `M[m, i] = coef`
    assignment keeps the last, so a CSR carrying a (segment, basis) pair twice
    would make the two paths disagree by more than reassociation."""
    if ports is not None and geom_name in ("dipole", "vee"):
        pytest.skip("no junction to hang a port on")
    kw = (
        GEOMETRIES[geom_name]()
        if ports is None
        else GEOMETRIES[geom_name](junction_ports=ports)
    )
    sim = SinusoidalGalerkinSolver(**kw)
    geom = sim._build_geometry()
    view = sim._basis_coefs(geom, sim.k)
    n_segs = geom["n_segs"]
    m_of_entry = np.repeat(np.arange(n_segs), np.diff(view["starts"]))
    pairs = np.stack([m_of_entry, view["jbasis"]], axis=1)
    assert np.unique(pairs, axis=0).shape[0] == pairs.shape[0]


def _tested_pieces(sim):
    """The float64 (context, tested contributions) every spelling of the
    coefficient product starts from — one fill, shared by the 80-bit reference
    and by the float64 products measured against it. On the folded shape set
    (const, sin, cos−1) since #205."""
    geom = sim._build_geometry()
    k = sim.k
    ctx = sim._test_context(geom, sim._basis_coefs(geom, k), k)
    return ctx, sim._tested_contribs(geom, k, ctx, _plain_projection)


def _coefficient_product(ctx, contribs, n_ports, dtype, *, folded):
    """G = Σ_shape (scatter contrib[shape]) @ M[shape], at a chosen precision
    and on a chosen shape set.

    `contribs` arrives on the FOLDED shape set — since #205 that is the only
    thing the fill produces — so it is the literal spelling that has to be
    reconstructed here: `folded=False` adds the const contributions back into
    the cos ones to recover the cos-shape table, then pairs const↔σA and
    cos↔σC as the pre-#203 product did. `folded=True` is the shipped pairing,
    const↔σ(A+C) and (cos−1)↔σC, straight off the fill. At 80-bit both are the
    exact product, which is how the reference is built.
    """
    n_segs = ctx["N"]
    n_basis = n_segs + n_ports
    i_of_entry, m_of_entry = ctx["i_of_entry"], ctx["m_of_entry"]
    c_const, c_sin, c_cos = (np.asarray(c, dtype=dtype) for c in contribs)
    if folded:
        coefs = (ctx["sigAC"], ctx["B"], ctx["sigC"])
    else:
        c_cos = c_cos + c_const
        coefs = (ctx["sigA"], ctx["B"], ctx["sigC"])

    G = np.zeros((n_basis, n_basis), dtype=dtype)
    for contrib, coef in zip((c_const, c_sin, c_cos), coefs):
        T = np.zeros((n_basis, n_segs), dtype=dtype)
        np.add.at(T, i_of_entry, contrib)
        M = np.zeros((n_segs, n_basis), dtype=dtype)
        M[m_of_entry, i_of_entry] = np.asarray(coef, dtype=dtype)
        G = G + T @ M
    return G


def _longdouble_coefficient_product(sim):
    """G's coefficient product in 80-bit arithmetic, from the SAME float64
    tested contributions every assembly path starts from — so the only
    difference from any of them is the precision of the sum they reassociate.

    The scatter runs at 80-bit too (#203): the shipped fold happens BEFORE the
    scatter, so a reference that scattered in float64 would have one spelling's
    rounding already baked into it and would flatter that spelling. And the
    reference takes the FOLDED pairing (#205): reconstructing the cos-shape
    table to run the literal one would walk the 80-bit product into the very
    cancellation it is meant to arbitrate, which costs it four digits.
    """
    ctx, contribs = _tested_pieces(sim)
    return _coefficient_product(
        ctx, contribs, len(sim.junction_ports), np.clongdouble, folded=True
    )


def test_sparse_assembly_is_no_less_accurate_than_dense(monkeypatch):
    """Both spellings sit the same distance from the exact product, which is
    what keeps `SPARSE_ASSEMBLY_AGREEMENT` a statement about reassociation
    rather than a slackened gate."""
    kw = _dipole()
    G_dense, G_sparse = _both_assembly_paths(monkeypatch, **kw)
    G_ref = _longdouble_coefficient_product(SinusoidalGalerkinSolver(**kw))
    ref = np.asarray(G_ref, dtype=np.complex128)
    err_dense = _rel_matrix_delta(G_dense, ref)
    err_sparse = _rel_matrix_delta(G_sparse, ref)
    assert err_sparse < 3 * err_dense, (err_sparse, err_dense)
    assert err_sparse < EXACT_PRODUCT_DISTANCE


# ---------------------------------------------------------------------------
# momwire#203 — the const/cos cancellation, folded at both of its sites
# ---------------------------------------------------------------------------
# The three-term basis is normalized to its own segment-centre current A+C,
# and that is O((kΔ)²) while A and C are each O(1). So every literal spelling
# of the basis — `σA + B·sin kξ + σC·cos kξ` for a value, `T_c @ M_A +
# T_co @ M_C` for the Galerkin product — computes a small number by cancelling
# two large ones, and throws away ε·8/(kΔ)² relative. On the half-wave dipole
# that is 5.5e-14 at N=41 but 1.75e-10 at N=2401, growing like N².
#
# Two sites, one fold, measured on the dipole ladder:
#
#   N     fval vs 80-bit eval      product vs 80-bit product   ‖G−Gᵀ‖/‖G‖
#         literal    folded        literal    folded           lit / fold / exact
#   41    5.5e-14 →  9.1e-17       7.3e-13 →  1.2e-15          1.03 / 1.06 / 1.05 e-11
#   401   2.9e-12 →  3.0e-15       1.2e-11 →  2.8e-15          1.48 / 1.45 / 1.45 e-10
#   801   2.3e-11 →  8.0e-15       7.6e-11 →  1.4e-14          1.98 / 2.04 / 2.04 e-10
#   2401  1.8e-10 →  7.2e-14       4.5e-10 →  8.3e-14          1.57 / 1.45 / 1.45 e-9
#
# The last column is the finding that contradicts #203's own second gate. The
# issue expected G1 to fall back toward "the fill's reciprocity floor" once the
# product was fixed. It does not move, because it was ALREADY at that floor:
# the exact 80-bit product of the same float64 contributions has the same G1 to
# three digits (`test_reciprocity_floor_is_set_by_the_fill_not_the_product`).
# G−Gᵀ is inherited from the contributions, and the same cancellation sets it
# one level up — the const- and cos-shape SOURCE fields nearly coincide, so the
# ε·‖T_const‖ rounding the kernel leaves in them is amplified by ‖T‖/‖G‖. That
# fingerprint is exact: G1 = 1.4 × ε‖T_const‖/‖G‖ at N=801 AND at N=2401.
# Fixing it means giving the field kernel the (cos kξ − 1) source shape itself,
# which is a change to the C++ far fill's output contract — filed separately,
# out of scope here.
#
# What the fold DOES buy beyond the product: the solve. cond(G) is 2.9e5 at the
# vee's N=321 and ~4e6 at the dipole's N=2401, so the evaluation error above is
# what set the impedance's last digits — Z moved 5.2e-6 relative at the vee's
# N=321 and 5.6e-4 at the dipole's N=2401 when the fold landed, and M3's `*_gal`
# constants were re-pinned for it.

_203_N = 401  # fine enough for the cancellation to bite, cheap enough to gate


def _basis_eval_errors(sim):
    """(literal, folded) relative error of the basis evaluation over every
    support entry × test node, against an 80-bit evaluation of the same float64
    coefficients — so this measures the SPELLING, not the coefficients."""
    geom = sim._build_geometry()
    k = sim.k
    view = sim._basis_coefs(geom, k)
    hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)
    gx, _gw = sim._leggauss_cached(sim.n_qp_test)
    m_of_entry = np.repeat(np.arange(geom["n_segs"]), np.diff(view["starts"]))
    xi = (hh[:, None] * gx[None, :])[m_of_entry]
    sig = view["sigma"].astype(np.complex128)
    sigA, B, sigC = sig * view["A"], view["B"], sig * view["C"]

    literal = (
        sigA[:, None] + B[:, None] * np.sin(k * xi) + sigC[:, None] * np.cos(k * xi)
    )
    folded = _basis_value((sig * view["AC"])[:, None], B[:, None], sigC[:, None], k, xi)
    kl, xl = np.longdouble(k), xi.astype(np.longdouble)
    exact = (
        sigA.astype(np.clongdouble)[:, None]
        + B.astype(np.clongdouble)[:, None] * np.sin(kl * xl)
        + sigC.astype(np.clongdouble)[:, None] * np.cos(kl * xl)
    )
    return (
        _rel_matrix_delta(literal, np.asarray(exact, dtype=np.complex128)),
        _rel_matrix_delta(folded, np.asarray(exact, dtype=np.complex128)),
    )


def test_folded_basis_evaluation_is_orders_more_accurate():
    """Site 1: the value. `_basis_value` must be the accurate spelling, not
    merely a different one — the coefficients are held fixed and only the
    arithmetic that combines them changes."""
    sim = SinusoidalGalerkinSolver(**_m3_dipole(_203_N))
    literal, folded = _basis_eval_errors(sim)
    assert folded < 1e-14, folded
    assert literal > 100 * folded, (literal, folded)


def test_folded_basis_evaluation_is_the_one_the_fill_uses():
    """…and the fill actually uses it: `_test_context`'s weights are the folded
    value times the arc weight, exactly."""
    sim = SinusoidalGalerkinSolver(**_dipole())
    geom = sim._build_geometry()
    view = sim._basis_coefs(geom, sim.k)
    ctx = sim._test_context(geom, view, sim.k)
    hh = ctx["hh"]
    gx, gw = sim._leggauss_cached(sim.n_qp_test)
    xi = (hh[:, None] * gx[None, :])[ctx["m_of_entry"]]
    fval = _basis_value(
        ctx["sigAC"][:, None], ctx["B"][:, None], ctx["sigC"][:, None], sim.k, xi
    )
    expect = (gw[None, :] * hh[ctx["m_of_entry"]][:, None]) * fval
    assert np.array_equal(ctx["w_entry"], expect)


def test_folded_coefficient_product_is_orders_closer_to_the_exact_product():
    """Site 2: the product. Both spellings are formed from the SAME float64
    contributions, so the 80-bit product of those contributions is the exact
    answer to the question each is answering."""
    sim = SinusoidalGalerkinSolver(**_m3_dipole(_203_N))
    ctx, contribs = _tested_pieces(sim)
    ref = np.asarray(
        _coefficient_product(ctx, contribs, 0, np.clongdouble, folded=True),
        dtype=np.complex128,
    )
    literal = _coefficient_product(ctx, contribs, 0, np.complex128, folded=False)
    folded = _coefficient_product(ctx, contribs, 0, np.complex128, folded=True)
    err_lit = _rel_matrix_delta(literal, ref)
    err_fold = _rel_matrix_delta(folded, ref)
    assert err_fold < EXACT_PRODUCT_DISTANCE, err_fold
    assert err_lit > 1000 * err_fold, (err_lit, err_fold)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_shipped_assembly_is_the_folded_product(monkeypatch, sparse):
    """Neither assembly tail gets to keep the literal pairing: both land on the
    exact product at a distance the literal spelling cannot reach."""
    kw = _m3_dipole(_203_N)
    sim = SinusoidalGalerkinSolver(**kw)
    ctx, contribs = _tested_pieces(sim)
    ref = np.asarray(
        _coefficient_product(ctx, contribs, 0, np.clongdouble, folded=True),
        dtype=np.complex128,
    )
    err_lit = _rel_matrix_delta(
        _coefficient_product(ctx, contribs, 0, np.complex128, folded=False), ref
    )
    G = _G_by_assembly_path(monkeypatch, sparse=sparse, **kw)
    err = _rel_matrix_delta(G, ref)
    assert err < EXACT_PRODUCT_DISTANCE, err
    assert err < err_lit / 1000, (err, err_lit)


def test_sin_minus_arg_is_accurate_on_both_branches():
    """`_drive_columns`' `sin u − u`, against a 50-digit series. Both branches
    are live at meshes this solver runs: u = kΔ/2 is 0.137 on M3's 11-segment
    coarse dipole (subtraction) and 0.037 at N=41 (series)."""
    from decimal import Decimal, getcontext

    from momwire.sinusoidal_galerkin import _sin_minus_arg

    getcontext().prec = 50

    def exact(x):
        term = total = Decimal(x)
        for n in range(1, 40):
            term = -term * Decimal(x) * Decimal(x) / Decimal((2 * n) * (2 * n + 1))
            total += term
        return total - Decimal(x)

    for u in (1e-4, 1e-3, 1e-2, 0.037, 0.0999, 0.1, 0.137, 0.5, 1.0):
        ref = exact(u)
        got = Decimal(float(_sin_minus_arg(u)))
        assert abs((got - ref) / ref) < Decimal("1e-13"), (u, got, ref)
    # vectorized, and the two branches meet without a step
    assert np.allclose(
        _sin_minus_arg(np.array([0.09999, 0.10001])),
        [-1.66525e-04, -1.66625e-04],
        rtol=1e-6,
    )


def test_reciprocity_floor_is_set_by_the_fill_not_the_product():
    """#203's second gate, refuted by measurement — and #205's, met.

    The issue read G1's mesh degradation (3.4e-10 at N=801, 1.3e-9 at N=2401)
    as the coefficient product's conditioning noise, and expected the fold to
    remove it. It did not: the EXACT 80-bit product of the same float64
    contributions was just as asymmetric, so G−Gᵀ is inherited from the
    contributions rather than created by the product.

    That is still the finding — and it is why the fix had to be a kernel one.
    Both halves are asserted:

    * the floor is the FILL's, so the 80-bit product of the same
      contributions has the same G1 the float64 one does (it did before #205
      at 1.45e-10, and it does after at 1.34e-13);
    * and the fill's floor is now `FOLDED_FILL_G1`, not `G1_GATE` — the
      contributions arrive on the (cos kξ − 1) shape, so nothing here is left
      to amplify.
    """
    sim = SinusoidalGalerkinSolver(**_m3_dipole(_203_N))
    ctx, contribs = _tested_pieces(sim)
    exact = _coefficient_product(ctx, contribs, 0, np.clongdouble, folded=True)
    folded = _coefficient_product(ctx, contribs, 0, np.complex128, folded=True)
    r_exact, r_folded = _sym_ratio(exact), _sym_ratio(folded)
    assert r_folded < FOLDED_FILL_G1, r_folded
    assert abs(r_folded - float(r_exact)) < 0.05 * r_folded, (r_folded, r_exact)


# ---------------------------------------------------------------------------
# momwire#205 — the (cos kξ − 1) source shape, in the field kernel
# ---------------------------------------------------------------------------
# #203 folded the coefficient product and did not move G1, because the same
# A ≈ −C cancellation runs one level up: the const-shape and cos-shape SOURCE
# fields nearly coincide, so subtracting the two closed forms leaves
# ε·|const-shape field| in a quantity that is O((kH)²) of it. That error is
# what `_folded_cos_fields` removes, by taking the two closed forms apart and
# spelling each surviving term at its own size (`cos_shape="cos-1"`).
#
# Measured on the half-wave dipole (the accelerated and numpy fills agree to
# every digit shown; `assemble` is the free-space fill+product wall clock on
# the dev box, 8 threads):
#
#   N      ‖G−Gᵀ‖/‖G‖               field error / |const|      assemble
#          before      after        before     after           before  after
#   41     1.06e-11 →  2.15e-13     6.9e-14 →  1.3e-16         0.03 s  0.05 s
#   401    1.45e-10 →  1.34e-13     7.5e-13 →  4.7e-16         0.19 s  0.29 s
#   801    2.04e-10 →  1.27e-13        —          —            0.52 s  0.77 s
#   2401   1.45e-09 →  5.51e-14     3.9e-12 →  1.6e-15         3.40 s  6.82 s
#
# So the growth with mesh is not merely broken: G1 now FALLS as the mesh
# refines, because what is left is no longer the (kΔ)⁻² amplification but the
# fill's own structural asymmetry between an integrated test side and an
# analytic source side. The cost is the phase table, which roughly doubled —
# the folded spelling needs the half-angles of the node offsets (see the fill
# kernel's `S`), and that sweep is most of the kernel's time.
#
# What is NOT fixed: one power of kH is left in the regular part of ∫G₀, whose
# per-node terms are O(kH) and whose sum is O((kH)²) — removing it needs the
# second difference (node pair against endpoint pair) in closed form. At the
# meshes above it sits below the structural floor, which is why G1 stops
# improving rather than continuing to fall.

# The dipole ladder's floor: measured 5.5e-14 (N=2401) to 2.2e-13 (N=41).
FOLDED_FILL_G1 = 1e-12
# How far the folded fill may sit from the pre-#205 fill: the two are the same
# quantity, so this is the size of the error being removed, not a tolerance on
# the physics. Measured 1.2e-11 - 1.8e-11 relative on the four validation
# geometries at their (coarse) gate meshes, where the amplification is
# smallest; on the N=2401 dipole the same difference is ~1e-9.
PRE_205_AGREEMENT = 1e-10
# Same statement one level down, on the field tables rather than on G, and
# scaled by the const-shape field the ε lives on. Measured 4.5e-11 — which IS
# the literal difference's own error at that sample, since the folded field is
# four orders closer to exact than the thing it is being compared with.
PRE_205_FIELD_AGREEMENT = 1e-9


def _decimal_cos_minus_one_fields(k, rho, z, H, gx, gw, eta, prec=60):
    """(E_z, E_ρ) of the cos kξ − 1 source shape at `prec` digits, as the
    LITERAL difference of the two Eqs 76-79 closed forms over the same source
    quadrature.

    The reference has no spelling to defend: at 60 digits the cancellation it
    walks into costs ~15 digits and leaves 45, so what it measures is the
    float64 spelling and nothing else. (An 80-bit reference cannot serve here
    — the same cancellation eats most of ITS margin too, which is exactly the
    #203 finding.)
    """
    from decimal import Decimal as D
    from decimal import localcontext

    with localcontext() as ctx:
        ctx.prec = prec
        pi = D("3.14159265358979323846264338327950288419716939937510582097494459231")

        def dsin(x):
            term = total = x
            for n in range(1, 80):
                term = -term * x * x / D((2 * n) * (2 * n + 1))
                total += term
                if abs(term) < abs(total) * D(10) ** -prec:
                    break
            return total

        def dcos(x):
            term, total = D(1), D(1)
            for n in range(1, 80):
                term = -term * x * x / D((2 * n - 1) * (2 * n))
                total += term
                if abs(term) < D(10) ** -prec:
                    break
            return total

        def emjkr(r):  # e^{−jkr}, as a Python complex of Decimals
            return (dcos(k * r), -dsin(k * r))

        def cmul(a, b):
            return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])

        def cdiv(a, s):
            return (a[0] / s, a[1] / s)

        def cadd(a, b):
            return (a[0] + b[0], a[1] + b[1])

        def csub(a, b):
            return (a[0] - b[0], a[1] - b[1])

        def cscale(a, s):
            return (a[0] * s, a[1] * s)

        def jmul(a):  # multiply by j
            return (-a[1], a[0])

        k, rho, z, H = (D(repr(float(v))) for v in (k, rho, z, H))
        d2, d1 = z - H, z + H
        r2 = (rho * rho + d2 * d2).sqrt()
        r1 = (rho * rho + d1 * d1).sqrt()
        G2, G1 = cdiv(emjkr(r2), r2), cdiv(emjkr(r1), r1)
        # T_e = (1 + jkr_e)·G_e/r_e²
        T2 = cdiv(cmul((D(1), k * r2), G2), r2 * r2)
        T1 = cdiv(cmul((D(1), k * r1), G1), r1 * r1)

        def asinh(x):
            return ((x * x + 1).sqrt() + x).ln()

        int_G0 = (asinh((H - z) / rho) - asinh((-H - z) / rho), D(0))
        for t, w in zip(gx, gw):
            rq = (rho * rho + (z - H * D(repr(float(t)))) ** 2).sqrt()
            int_G0 = cadd(
                int_G0,
                cscale(
                    csub(cdiv(emjkr(rq), rq), (D(1) / rq, D(0))), D(repr(float(w))) * H
                ),
            )
        sH, cH = dsin(k * H), dcos(k * H)
        # E_z = pref_z·[…]; E_ρ = pref_ρ·[…], with pref_z = jη/(4πk) and
        # pref_ρ = −jη/(4πkρ). η is the SOLVER's, to the bit: its own value
        # differs from √(μ₀/ε₀) in the 10th digit, which at these field ratios
        # is 1.7e-12 of the const-shape scale — thirty times the error under
        # test.
        eta = D(repr(float(eta)))
        pz = eta / (D(4) * pi * k)  # ×j
        pr = -eta / (D(4) * pi * k * rho)  # ×j

        ez_const = cscale(
            cadd(csub(cscale(T2, d2), cscale(T1, d1)), cscale(int_G0, k * k)), -pz
        )
        # bracket_e = −k·sin(kz'_e)·G_e − cos(kz'_e)·Δz_e·T_e, at z'_2 = +H
        # and z'_1 = −H.
        ez_cos = cscale(
            csub(
                csub(cscale(G2, -k * sH), cscale(T2, d2 * cH)),
                csub(cscale(G1, k * sH), cscale(T1, d1 * cH)),
            ),
            pz,
        )
        erho_const = cscale(csub(T2, T1), pr * rho * rho)
        b2 = csub(cscale(csub(G2, cscale(T2, d2 * d2)), cH), cscale(G2, k * d2 * sH))
        b1 = cadd(cscale(csub(G1, cscale(T1, d1 * d1)), cH), cscale(G1, k * d1 * sH))
        erho_cos = cscale(csub(b2, b1), pr)

        def out(a):
            return complex(float(a[0]), float(a[1]))

        return (
            out(jmul(csub(ez_cos, ez_const))),
            out(jmul(csub(erho_cos, erho_const))),
            out(jmul(ez_const)),
            out(jmul(erho_const)),
        )


def _205_sample(n_seg):
    """(ρ, z) pairs spanning the bands the fill actually visits at mesh
    `n_seg` on the M3 dipole: the self pair, the touching neighbours, the
    mid-range where kr ~ 1 and the far end of the wire."""
    sim = SinusoidalGalerkinSolver(**_m3_dipole(n_seg))
    geom = sim._build_geometry()
    H = 0.5 * float(geom["seg_h"][0])
    a = float(sim._uniform_radius)
    rho0 = np.sqrt(a * a)
    zs = np.concatenate(
        [np.arange(0, 4) * 2.0 * H, np.geomspace(4.0 * H, 0.45 * WL, 12)]
    )
    rhos = np.array([rho0, 10 * rho0, 0.3, 2.0])
    Z, R = (v.ravel() for v in np.meshgrid(zs, rhos))
    return sim, H, np.sqrt(R * R + a * a), Z


def _folded_field_errors(n_seg):
    """(literal, folded) worst error of the (cos kξ − 1) field over the
    sample, each divided by the CONST-shape field at the same pair — the
    scale an ε·‖T_const‖ rounding sits at, and therefore the scale at which
    the error reaches G."""
    sim, H, rho, z = _205_sample(n_seg)
    k = sim.k
    gx, gw = sim._leggauss_cached(sim.n_qp_const)
    Hs = np.full_like(z, H)
    args = dict(
        k=k,
        H=Hs,
        z=z,
        rho=rho,
        dz1=z + Hs,
        dz2=z - Hs,
        r1=np.sqrt(rho**2 + (z + Hs) ** 2),
        r2=np.sqrt(rho**2 + (z - Hs) ** 2),
        gx=gx,
        gw=gw,
        pref_z=1j * sim.eta / (4.0 * np.pi * k),
        pref_rho=-1j * sim.eta / (4.0 * np.pi * k * rho),
    )
    args["G1"] = np.exp(-1j * k * args["r1"]) / args["r1"]
    args["G2"] = np.exp(-1j * k * args["r2"]) / args["r2"]
    args["sH"] = np.sin(k * Hs)
    z_qp = Hs[:, None] * gx
    args["dz_qp"] = z[:, None] - z_qp
    args["r_qp"] = np.sqrt(rho[:, None] ** 2 + args["dz_qp"] ** 2)
    erho_f, ez_f = SinusoidalSolver._folded_cos_fields(**args)

    lit = np.empty((len(z), 2), dtype=complex)
    fold = np.empty_like(lit)
    scale = np.empty(len(z))
    for i in range(len(z)):
        ez_d, erho_d, ez_c, erho_c = _decimal_cos_minus_one_fields(
            k, rho[i], z[i], H, gx, gw, sim.eta
        )
        cm = SinusoidalGalerkinSolver(**_m3_dipole(n_seg))._field_components_bcast(
            k,
            obs_c=np.array([[0.0, rho[i], z[i]]]),
            obs_t=np.array([[0.0, 0.0, 1.0]]),
            a=0.0,
            src_c=np.array([[0.0, 0.0, 0.0]]),
            src_t=np.array([[0.0, 0.0, 1.0]]),
            src_hh=np.array([H]),
        )
        lit[i] = (
            cm["Ez_cos"][0] - cm["Ez_const"][0],
            cm["Erho_cos"][0] - cm["Erho_const"][0],
        )
        fold[i] = (ez_f[i], erho_f[i])
        exact = np.array([ez_d, erho_d])
        scale[i] = max(abs(ez_c), abs(erho_c))
        lit[i] -= exact
        fold[i] -= exact
    return (
        float((np.abs(lit).max(axis=1) / scale).max()),
        float((np.abs(fold).max(axis=1) / scale).max()),
    )


@pytest.mark.parametrize("n_seg", [41, _203_N])
def test_folded_cos_shape_is_orders_more_accurate_than_the_difference(n_seg):
    """The spelling, against a 60-digit evaluation of the same scheme.

    Both numbers are errors in the SAME quantity computed from the SAME
    inputs on the SAME source quadrature, so what separates them is the
    arithmetic and nothing else — and the gap widens with mesh, because the
    literal difference's error is ε·|const| while the folded one's is at the
    answer's own size.
    """
    literal, folded = _folded_field_errors(n_seg)
    assert folded < 5e-14, folded  # measured 3.1e-15 at N=41, 1.4e-16 at N=401
    assert literal > 100 * folded, (literal, folded)


def test_folded_cos_shape_reproduces_the_literal_difference():
    """…and it is the same quantity: the folded field agrees with the literal
    difference to the literal difference's own accuracy, which is the only
    agreement available (the two disagree BY the error being removed)."""
    sim, H, rho, z = _205_sample(41)
    k = sim.k
    obs = np.stack([np.zeros_like(z), rho, z], axis=1)
    t = np.tile(np.array([0.0, 0.0, 1.0]), (len(z), 1))
    src_c = np.zeros((1, 3))
    src_t = np.array([[0.0, 0.0, 1.0]])
    hh = np.array([H])
    kw = dict(obs_c=obs[:, None, :], obs_t=t[:, None, :], a=0.0)
    lit = sim._field_components_bcast(
        k, src_c=src_c[None], src_t=src_t[None], src_hh=hh[None], **kw
    )
    fold = sim._field_components_bcast(
        k,
        src_c=src_c[None],
        src_t=src_t[None],
        src_hh=hh[None],
        cos_shape="cos-1",
        **kw,
    )
    # Scaled by the pair's const-shape field — where an ε·‖T_const‖ rounding
    # sits — and by the LARGER of the two components, since E_ρ vanishes
    # identically on the broadside pairs of this sample and a per-component
    # scale would divide by zero there.
    scale = np.abs(lit["Ez_const"]) + np.abs(lit["Erho_const"])
    for comp in ("Ez", "Erho"):
        d = lit[f"{comp}_cos"] - lit[f"{comp}_const"]
        rel = np.abs(fold[f"{comp}_cos"] - d) / scale
        assert rel.max() < PRE_205_FIELD_AGREEMENT, (comp, rel.max())


def _G_pre_205(monkeypatch, dtype=np.complex128, ld_kernel=False, **kw):
    """G as the pre-#205 fill built it: the LITERAL cos shape out of the field
    kernel, folded by subtracting the const contributions after the test
    reduction (#203's `_assemble_Z` tail).

    With `ld_kernel` the kernel runs at 80 bits and the fold happens there,
    which is the exact-arithmetic fill both float64 spellings are measured
    against — the fold has to be inside the kernel or the reference inherits
    the very cancellation it exists to arbitrate.
    """
    from momwire import sinusoidal_galerkin as _sg

    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", False)
    sim = SinusoidalGalerkinSolver(**kw)
    inner = sim._field_components_bcast

    def literal(k, **kwargs):
        kwargs["cos_shape"] = "cos"
        if not ld_kernel:
            return inner(k, **kwargs)
        cast = {
            key: (
                np.asarray(v, dtype=np.longdouble) if isinstance(v, np.ndarray) else v
            )
            for key, v in kwargs.items()
        }
        cm = inner(np.longdouble(k), **cast)
        out = {}
        for key, v in cm.items():
            if key in ("Ez_cos", "Erho_cos"):
                v = v - cm[key.replace("cos", "const")]
            out[key] = np.asarray(
                v, dtype=np.complex128 if v.dtype.kind == "c" else float
            )
        return out

    monkeypatch.setattr(sim, "_field_components_bcast", literal)
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    contribs = sim._tested_contribs(geom, sim.k, ctx, _plain_projection)
    if not ld_kernel:  # #203's assembly-level fold, on the literal shapes
        c_const, c_sin, c_cos = contribs
        contribs = (c_const, c_sin, c_cos - c_const)
    return _coefficient_product(
        ctx, contribs, len(sim.junction_ports), dtype, folded=True
    )


@pytest.mark.parametrize("geom_name", ["dipole", "vee", "k2_junction", "k3_star"])
def test_folded_fill_agrees_with_the_pre_205_fill_and_is_closer_to_exact(
    monkeypatch, geom_name
):
    """The matrix gate. Same fill, same quadrature, same product — so the two
    spellings must agree, and they do, to the size of the error #205 removes.
    Which of them is the accurate one is then settled against a fill whose
    kernel ran at 80 bits: the folded one is orders closer, and the residue
    is not a difference of scheme.
    """
    kw = GEOMETRIES[geom_name]()
    G_new = _matrix(SinusoidalGalerkinSolver(**kw))
    G_old = _G_pre_205(monkeypatch, **kw)
    G_ref = _G_pre_205(monkeypatch, ld_kernel=True, **kw)
    assert _rel_matrix_delta(G_new, G_old) < PRE_205_AGREEMENT
    err_new = _rel_matrix_delta(G_new, G_ref)
    err_old = _rel_matrix_delta(G_old, G_ref)
    # 47× (k3_star, 12 segments per arm) to 1100× (the 41-segment dipole):
    # the gain is the amplification removed, which is ~(kΔ)⁻², so the coarser
    # the validation geometry the less there was to win.
    assert err_new < err_old / 20, (err_new, err_old)


@pytest.mark.parametrize("geom_name", ["dipole", "vee", "k2_junction", "k3_star"])
def test_folded_fill_asymmetry_is_now_structural_not_rounding(monkeypatch, geom_name):
    """What is left of G−Gᵀ after the fold: nothing that arithmetic can fix.

    The same fill with its kernel run at 80 bits — 2000× the precision — has
    the SAME G1 to three digits on every validation geometry (2.15e-13 dipole,
    8.35e-12 vee, 2.94e-12 k2_junction, 2.21e-11 k3_star). So the residual is
    the fill's structural asymmetry between an integrated test side and an
    analytic source side, not its rounding — which is the statement #203 could
    not make, since there G1 fell by 50× when the kernel precision went up.

    The dipole, where the rounding used to dominate, additionally has to reach
    `FOLDED_FILL_G1`; the bent geometries carry a larger structural residual
    at their much coarser gate meshes and are not held to it.
    """
    kw = GEOMETRIES[geom_name]()
    r_new = _sym_ratio(_matrix(SinusoidalGalerkinSolver(**kw)))
    r_ref = _sym_ratio(_G_pre_205(monkeypatch, ld_kernel=True, **kw))
    assert abs(r_new - r_ref) < 0.05 * r_ref, (r_new, r_ref)
    if geom_name == "dipole":
        assert r_new < FOLDED_FILL_G1, r_new
