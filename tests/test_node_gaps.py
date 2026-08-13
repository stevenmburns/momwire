"""Series node gaps (#305): a delta-gap EMF in series with a named
wire-end at a junction node — the apex feed.

Distinct from `junction_ports` (#172, the KCL row promoted to a
net-inflow port): a node gap KEEPS its junction's KCL row as a
constraint (current is continuous through a series EMF) and its
drive/readout column is the σ-signed unit indicator of the named
wire-end's directional basis (σ = +1 start / −1 end, the outflow sign),
so I_port is the current flowing from the node into the named wire and
the pairing is Galerkin-reciprocal.

The apex colinear identity itself (split dipole apex-fed == single-wire
gap, the #300-class oracle) lives with its bridge-spelling siblings in
`test_colinear_split_identity.py`; this module carries the port
algebra, the validation rules, the family cross-checks and the
degree-3 star.

Values measured 2026-08-12 on the reference build; pins sit just above
measurement per the M5b convention (`test_junction_ports.py`).
"""

import math

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire.array_block import ArrayBlockSolver
from momwire.hmatrix import HMatrixSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

L = 5.2
WAVELENGTH = 299.792458 / 27.0
N = 64  # single-wire pitch; the split reuses it per arm


def _split_wires():
    a = L / 4
    return [
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, a)]),
        np.array([(0.0, 0.0, a), (0.0, 0.0, L)]),
    ]


_SPLIT_KW = dict(
    n_per_edge_per_wire=[[N // 4], [3 * N // 4]],
    junctions=[[(0, "end"), (1, "start")]],
    wavelength=WAVELENGTH,
    wire_radius=1e-3,
)


def _apex_bspline(**over):
    kw = dict(
        wires=_split_wires(),
        feeds=[],
        node_gaps=[(0, "end", 1.0 + 0j)],
        degree=2,
        **_SPLIT_KW,
    )
    kw.update(over)
    return BSplineSolver(**kw)


def _z(solver):
    z, _ = solver.compute_impedance()
    return complex(np.atleast_1d(z)[0])


# ---------------------------------------------------------------------------
# validation rules
# ---------------------------------------------------------------------------


def test_validation_rule_battery():
    """Every rejection names its rule; the legal spellings construct."""
    ok = dict(wires=_split_wires(), feeds=[], degree=2, **_SPLIT_KW)

    # Legal: either member of the degree-2 junction, feeds=[] included.
    BSplineSolver(node_gaps=[(0, "end", 1.0 + 0j)], **ok)
    BSplineSolver(node_gaps=[(1, "start", 1.0 + 0j)], **ok)

    with pytest.raises(ValueError, match="not a member of any junction"):
        BSplineSolver(node_gaps=[(0, "start", 1.0 + 0j)], **ok)
    with pytest.raises(ValueError, match="'start' or 'end'"):
        BSplineSolver(node_gaps=[(0, "p1", 1.0 + 0j)], **ok)
    with pytest.raises(ValueError, match="out of range"):
        BSplineSolver(node_gaps=[(7, "end", 1.0 + 0j)], **ok)
    with pytest.raises(ValueError, match="listed twice"):
        BSplineSolver(node_gaps=[(0, "end", 1.0 + 0j), (0, "end", 0j)], **ok)
    with pytest.raises(ValueError, match="already carries a node gap"):
        BSplineSolver(node_gaps=[(0, "end", 1.0 + 0j), (1, "start", 0j)], **ok)
    with pytest.raises(ValueError, match="expected .wire_index"):
        BSplineSolver(node_gaps=[(0, "end")], **ok)

    # A junction cannot be shunt-ported and series-gapped at once.
    with pytest.raises(ValueError, match="also a junction port"):
        BSplineSolver(node_gaps=[(0, "end", 1.0 + 0j)], junction_ports=[(0, 0j)], **ok)

    # A single-member junction has no through path.
    lone = dict(ok, junctions=[[(0, "end")], [(1, "start")]])
    with pytest.raises(ValueError, match="single member"):
        BSplineSolver(node_gaps=[(0, "end", 1.0 + 0j)], **lone)


def test_point_matched_sinusoidal_takes_no_node_gaps():
    """The point-matched family cannot express the port: a node-localized
    EMF samples to a zero collocation RHS (the #177/#212 class), so the
    constructor never grew the keyword and refuses it as unknown."""
    with pytest.raises(TypeError, match="node_gaps"):
        SinusoidalSolver(
            wires=_split_wires(),
            feeds=[],
            node_gaps=[(0, "end", 1.0 + 0j)],
            **_SPLIT_KW,
        )


def test_grounded_junction_is_rejected_at_solve():
    """#151 grounds a node through its image — a series gap between a
    wire and the ground stake is a different (unsupported) object. A
    ground-touching vee vertex: two members, so only the ground rule
    can fire, and it fires at solve time (groundedness is geometry)."""
    s = BSplineSolver(
        wires=[
            np.array([(0.0, 0.0, 0.0), (2.0, 0.0, 1.0)]),
            np.array([(0.0, 0.0, 0.0), (-2.0, 0.0, 1.0)]),
        ],
        feeds=[],
        node_gaps=[(0, "start", 1.0 + 0j)],
        n_per_edge_per_wire=[[16], [16]],
        junctions=[[(0, "start"), (1, "start")]],
        wavelength=WAVELENGTH,
        wire_radius=1e-3,
        degree=2,
        ground_z=0.0,
    )
    with pytest.raises(ValueError, match="grounded"):
        s.compute_impedance()


# ---------------------------------------------------------------------------
# port algebra
# ---------------------------------------------------------------------------


def test_zero_volt_node_gap_is_the_plain_solve():
    """A 0 V node gap adds a readout without touching operator, RHS or
    constraints — the driven feed's answer is bit-identical."""
    plain = BSplineSolver(
        wires=_split_wires(), feeds=[(1, L / 4, 1.0 + 0j)], degree=2, **_SPLIT_KW
    )
    gapped = BSplineSolver(
        wires=_split_wires(),
        feeds=[(1, L / 4, 1.0 + 0j)],
        node_gaps=[(0, "end", 0j)],
        degree=2,
        **_SPLIT_KW,
    )
    z_plain, _ = plain.compute_impedance()
    z_gapped, _ = gapped.compute_impedance()
    assert complex(np.atleast_1d(z_gapped)[0]) == complex(np.atleast_1d(z_plain)[0])


def test_member_choice_is_immaterial_at_degree_two():
    """At a two-wire vertex the through current is single: the gap 'in
    series with' either member is the same port, exactly (#305)."""
    za = _z(_apex_bspline(node_gaps=[(0, "end", 1.0 + 0j)]))
    zb = _z(_apex_bspline(node_gaps=[(1, "start", 1.0 + 0j)]))
    assert abs(za - zb) < 1e-9, f"{za:.6f} vs {zb:.6f}"


@pytest.mark.parametrize("family", ["bspline", "sg"])
def test_mixed_port_y_is_symmetric(family):
    """Apex node gap × mid-arm gap feed: drive == readout per column, so
    the two-port Y is machine-symmetric on both families."""
    kw = dict(
        wires=_split_wires(),
        feeds=[(1, L / 4, 1.0 + 0j)],
        node_gaps=[(0, "end", 0j)],
        **_SPLIT_KW,
    )
    if family == "bspline":
        s = BSplineSolver(degree=2, **kw)
    else:
        # variational: the honest gap-feed readout. Under the default
        # centre readout the FEED column is not its drive's dual (a
        # documented O(h) caveat of this family, ~6e-5 here); the node
        # gap column is exactly dual under either.
        s = SinusoidalGalerkinSolver(feed_readout="variational", **kw)
    y = s.compute_port_solution().y
    assert np.abs(y - y.T).max() / np.abs(y).max() < 1e-12


def test_family_port_orientations_agree():
    """The SG spelling normalizes to the COMPLEMENT bipartition so both
    families read I_port as 'current from the node into the named wire'
    — pinned on the off-diagonal Y sign and size, which a pure-sign
    convention bug would flip (diagonals cannot see it)."""
    kw = dict(
        wires=_split_wires(),
        feeds=[(1, L / 4, 1.0 + 0j)],
        node_gaps=[(0, "end", 0j)],
        **_SPLIT_KW,
    )
    y_bs = BSplineSolver(degree=2, **kw).compute_port_solution().y
    y_sg = SinusoidalGalerkinSolver(**kw).compute_port_solution().y
    # measured 2026-08-12: entrywise ~3% at this mesh (the basis gap),
    # catastrophic (≈200%) if either orientation flips.
    assert np.abs(y_bs - y_sg).max() / np.abs(y_bs).max() < 0.10


def test_iterative_families_reproduce_the_dense_port():
    """HMatrix/ArrayBlock consume the dense `_port_columns` — the node
    gap column must survive their route untouched (measured 6e-14)."""
    kw = dict(
        wires=_split_wires(),
        feeds=[(1, L / 4, 1.0 + 0j)],
        node_gaps=[(0, "end", 0j)],
        degree=2,
        **_SPLIT_KW,
    )
    y_ref = BSplineSolver(**kw).compute_port_solution().y
    # tight ACA/GMRES tolerances per test_junction_ports_iterative — the
    # defaults (aca_tol=1e-4) already land ~7e-9, this removes the slack.
    for cls in (HMatrixSolver, ArrayBlockSolver):
        y = cls(aca_tol=1e-9, solve_tol=1e-10, **kw).compute_port_solution().y
        assert np.abs(y - y_ref).max() / np.abs(y_ref).max() < 1e-9, cls.__name__


def test_iterative_families_drive_the_node_gap():
    """The accelerated `compute_impedance` route must DRIVE the gap, not
    just read it: before the fix it assembled a zero RHS for a node-gap-
    driven deck and returned an EMPTY z (feeds-shaped) while the dense
    fallback answered — a silent wrong-shape, wrong-physics result. Free
    space keeps the accelerated route (no dense fallback) engaged."""
    kw = dict(
        wires=_split_wires(),
        feeds=[],
        node_gaps=[(0, "end", 1.0 + 0j)],
        degree=2,
        **_SPLIT_KW,
    )
    z_ref, _ = BSplineSolver(**kw).compute_impedance()
    z_ref = complex(np.atleast_1d(z_ref)[0])
    for cls in (HMatrixSolver, ArrayBlockSolver):
        z, coeffs = cls(aca_tol=1e-9, solve_tol=1e-10, **kw).compute_impedance()
        z = complex(np.atleast_1d(z)[0])
        assert abs(z - z_ref) < 1e-6 * abs(z_ref), (cls.__name__, z, z_ref)
        assert np.abs(coeffs).max() > 0, cls.__name__


def test_swept_ports_match_per_k():
    """The swept path hoists the k-independent port columns once; each
    per-k Y must equal the single-k solve's."""
    s = _apex_bspline(feeds=[(1, L / 4, 1.0 + 0j)], node_gaps=[(0, "end", 0j)])
    k0 = 2 * np.pi / WAVELENGTH
    ks = k0 * np.array([0.9, 1.0, 1.1])
    swept = [ps.y for ps in s._port_solutions_swept(ks)]
    for k, y_k in zip(ks, swept):
        s1 = _apex_bspline(
            feeds=[(1, L / 4, 1.0 + 0j)],
            node_gaps=[(0, "end", 0j)],
            wavelength=2 * np.pi / k,
        )
        y_ref = s1.compute_port_solution().y
        assert np.abs(y_k - y_ref).max() / np.abs(y_ref).max() < 1e-9


# ---------------------------------------------------------------------------
# cross-family anchors
# ---------------------------------------------------------------------------

# The d=2 single-wire point gap at an interior knot is this codebase's
# best zero-width feed; the SG node EMF is exact-in-basis. Measured
# 2026-08-12: they land 0.0007 Ω apart at N=64 — pinned an order above.
BSPLINE_SINGLE_ANCHOR_GATE = 0.1


def test_sg_apex_lands_on_the_zero_width_anchor():
    zs = BSplineSolver(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, L)])],
        n_per_edge_per_wire=[[N]],
        feeds=[(0, L / 4, 1.0 + 0j)],
        wavelength=WAVELENGTH,
        wire_radius=1e-3,
        degree=2,
    )
    z_anchor = _z(zs)
    sg = SinusoidalGalerkinSolver(
        wires=_split_wires(), feeds=[], node_gaps=[(0, "end", 1.0 + 0j)], **_SPLIT_KW
    )
    z_sg = _z(sg)
    assert abs(z_sg - z_anchor) < BSPLINE_SINGLE_ANCHOR_GATE, (
        f"anchor {z_anchor:.4f} vs SG apex {z_sg:.4f}"
    )


def test_sg_apex_survives_the_extended_kernel():
    """EK moves the apex reading by its usual tube-physics delta, not by
    an end-bracket blowup at the ported node (#299's cancellation holds:
    the split is colinear and equal-radius, so the node predicate keeps
    both brackets and they cancel). Measured |ΔZ| = 0.039 Ω."""
    kw = dict(
        wires=_split_wires(), feeds=[], node_gaps=[(0, "end", 1.0 + 0j)], **_SPLIT_KW
    )
    z_red = _z(SinusoidalGalerkinSolver(**kw))
    z_ek = _z(SinusoidalGalerkinSolver(extended_kernel=True, **kw))
    assert abs(z_ek - z_red) < 0.5, f"reduced {z_red:.4f} vs EK {z_ek:.4f}"


# ---------------------------------------------------------------------------
# ground models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ground_kw",
    [
        {"ground_z": 0.0},
        {"ground_z": 0.0, "ground_eps": (13, 0.005)},
        {
            "ground_z": 0.0,
            "ground_eps": (13, 0.005),
            "ground_model": "sommerfeld",
        },
    ],
    ids=["pec", "refl-coef", "sommerfeld"],
)
def test_node_gap_runs_over_every_ground_model(ground_kw):
    """A node gap rides the ordinary span — no node-charge machinery — so
    it inherits each family's full ground roster, with NONE of the
    junction-port ground caveats (SG junction ports refuse finite
    grounds; SG node gaps don't have to). Pinned as: both families
    solve, agree within the basis gap, and actually saw the ground
    (measured 2026-08-12: PEC moves Z by ~7 Ω vs free space here)."""
    import math

    ang, arm, height = math.radians(30), 2.6, 8.0
    apex = (0.0, 0.0, height)
    e1 = (0.0, arm * math.cos(ang), height - arm * math.sin(ang))
    e2 = (0.0, -arm * math.cos(ang), height - arm * math.sin(ang))
    kw = dict(
        wires=[[apex, e1], [apex, e2]],
        n_per_edge_per_wire=[16, 16],
        feeds=[],
        junctions=[[(0, "start"), (1, "start")]],
        node_gaps=[(0, "start", 1.0 + 0j)],
        wavelength=11.1,
        wire_radius=1e-3,
    )

    def _solve(cls, extra):
        z, _ = cls(**kw, **extra).compute_impedance()
        return complex(np.atleast_1d(z)[0])

    z_free = _solve(BSplineSolver, {"degree": 2})
    z_bs = _solve(BSplineSolver, {"degree": 2, **ground_kw})
    z_sg = _solve(SinusoidalGalerkinSolver, ground_kw)
    assert abs(z_bs - z_free) > 2.0  # the ground is actually in the answer
    assert abs(z_bs - z_sg) < 1.0, f"bspline {z_bs:.2f} vs SG {z_sg:.2f}"


# ---------------------------------------------------------------------------
# the degree-3 star
# ---------------------------------------------------------------------------

# Asymmetric 3-arm star (lengths 2.0 / 2.6 / 3.1 m, bent out of plane):
# the gap in series with each arm is a DIFFERENT port with a different
# answer — NEC-5's tag/end addressing, not a convention. Measured
# 2026-08-12 at [14, 16, 18] segments, d=2, r=1 mm, λ=11.1 m.
STAR_PINS = {
    0: 34.434 - 155.061j,
    1: 372.211 + 64.838j,
    2: 77.119 + 129.644j,
}


def _star_solver(arm):
    arms = [
        [(0.0, 0.0, 0.0), (Lm * math.cos(a), Lm * math.sin(a), 0.3 * Lm)]
        for Lm, a in [(2.0, 0.0), (2.6, 2.1), (3.1, 4.2)]
    ]
    return BSplineSolver(
        wires=arms,
        n_per_edge_per_wire=[14, 16, 18],
        degree=2,
        feeds=[],
        junctions=[[(0, "start"), (1, "start"), (2, "start")]],
        node_gaps=[(arm, "start", 1.0 + 0j)],
        wavelength=11.1,
        wire_radius=1e-3,
    )


@pytest.mark.parametrize("arm", [0, 1, 2])
def test_star_arm_answers_are_pinned(arm):
    assert abs(_z(_star_solver(arm)) - STAR_PINS[arm]) < 0.5


def test_star_arms_are_three_different_ports():
    zs = [_z(_star_solver(arm)) for arm in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(zs[i] - zs[j]) > 30.0, (zs[i], zs[j])


def test_sg_swept_ports_match_per_k():
    """The SG family's swept path with a node-gap drive: hoisted port
    columns per k must equal the single-k solve (measured 7e-10)."""
    kw = dict(
        wires=_split_wires(),
        feeds=[],
        node_gaps=[(0, "end", 1.0 + 0j)],
        **_SPLIT_KW,
    )
    s = SinusoidalGalerkinSolver(**kw)
    k0 = 2 * np.pi / WAVELENGTH
    ks = k0 * np.array([0.9, 1.0, 1.1])
    swept = np.atleast_1d(s.compute_impedance_swept(ks)).ravel()
    for k, z_sw in zip(ks, swept):
        s1 = SinusoidalGalerkinSolver(**dict(kw, wavelength=2 * np.pi / k))
        z1 = complex(np.atleast_1d(s1.compute_impedance()[0])[0])
        assert abs(z_sw - z1) < 1e-6 * abs(z1), (k, z_sw, z1)
