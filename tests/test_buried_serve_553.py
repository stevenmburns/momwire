"""The buried serve: per-segment medium, the mixed-medium fill, the seam.

momwire#553 unit 5 — the integration unit. U1 continued the moment kernels
to complex k, U2 built the below/below remainder, U3 the transmitted family,
U4 the in-medium wavelength; this is where a deck with a wire below the
interface becomes an ANSWER.

The gates below are grouped as G-U5-n and each one records its measured
number through `record_property`, so a reader of the JUnit XML sees the
envelope rather than just a green tick.

What is NOT here, and why
-------------------------
The unit's brief named the two phase-0 ANCHOR decks — a 10 m contact
monopole over one buried radial (92.130 − 70.141j ohm printed) and over a
four-radial fan (90.051 − 70.731j) — as its serve gates. They refuse, and
`test_gu5_9_*` is that refusal. The reason is `G-U5-3`'s own measurement:
both decks stand a wire END in the plane, and momwire's buried fill cannot
hold a ground CONTACT and a buried wire at the same time. See
`_medium_spec.contact_with_buried_refusal` for the mechanism (the fifth
inversion — a field-form/mixed-potential boundary term whose licence is "the
remainder is small") and G-U5-3 for the 2.5-relative-against-1.0e-5 contrast
that priced it.

The anchors' NUMBERS are nonetheless gated, at `G-U5-12`, as xfails that arm
themselves the day the refusal comes out. momwire#567's scoping pass found
that the two decks were only ever asserted as SUBSTRINGS of refusal prose:
landing the fix would not have satisfied them, and the FAILS-IF-FIXED gate
at G-U5-3 was loose enough to pass a fix that left the contact row wrong by
120 %. Both of those are closed here.

The engine-referenced gate that remains is the FULLY buried one, and it did
not come out where the brief expected either: see `G-U5-8`, which records
the disagreement as a measurement and pins momwire against three
oracles that ARE available instead.
"""

import math
import time

import numpy as np
import pytest
from golden_buried_anchor_nec5 import ANCHOR_FOUR_RADIAL, ANCHOR_LONE_RADIAL

from momwire import (
    RazorSolver,
    _ground_refl,
    _medium_spec,
    _sommerfeld_below,
    _sommerfeld_transmitted,
)
from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver

C0 = 299792458.0
SOIL_A = (13.0, 0.005)
F7 = 7e6
WL7 = C0 / F7

# The ELEVATED deck's own scale: what the ε̃ → 1 collapse reads when every
# basis vanishes at its own support ends and the two testing conventions
# therefore agree. It is the threshold of `test_gu5_3_eps_one_*` and the
# verdict threshold of the contact tripwire below, and it is one constant so
# that the two cannot drift apart.
_ELEVATED_COLLAPSE_REL = 5e-5

# The contact deck's measured band is 2.3–2.5 across quadrature orders 3, 4,
# 6 and 8 (see the tripwire's docstring). 2.0 is ~15 % under the low end —
# the house CI-margin rule — so ordinary run-to-run variation cannot fire the
# tripwire while any real movement of the boundary term does.
_CONTACT_TERM_FLOOR = 2.0


# ----------------------------------------------------------------------
# decks
# ----------------------------------------------------------------------


def _mono(top=11.0, bottom=1.0):
    return np.array([(0.0, 0.0, top), (0.0, 0.0, bottom)])


def _radial(length=5.0, depth=0.15, direction=(1.0, 0.0)):
    dx, dy = direction
    return np.array(
        [(0.0, 0.0, -depth), (dx * length, dy * length, -depth)], dtype=float
    )


def served_deck(mult=1, clearance=1.0, depth=0.15, eps=SOIL_A, free=False, **kw):
    """The unit's SERVED serve-gate deck: an elevated 10 m monopole over a
    detached 5 m radial, no ground contact anywhere.

    The elevation is what makes it servable (G-U5-3): every basis vanishes at
    its own support ends, so the fill's two testing conventions agree and the
    ε̃ → 1 collapse is exact to interpolation.
    """
    n = 15 * mult
    arc = (round(0.4333 * n) - 0.5) / n * 10.0
    ground = (
        {} if free else dict(ground_z=0.0, ground_eps=eps, ground_model="sommerfeld")
    )
    return BSplineSolver(
        wires=[_mono(clearance + 10.0, clearance), _radial(depth=depth)],
        n_per_edge_per_wire=[[n], [10 * mult]],
        feeds=[(0, arc, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **ground,
        **kw,
    )


def buried_dipole(n=11, length=1.0, depth=0.15, vertical=True, eps=SOIL_A, free=False):
    """The phase-0 buried dipoles: fully below the interface, centre fed."""
    if vertical:
        pts = np.array([(0.0, 0.0, -(depth + length)), (0.0, 0.0, -depth)])
    else:
        pts = np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    fed = (n + 1) // 2
    arc = (fed - 0.5) / n * length
    ground = (
        {} if free else dict(ground_z=0.0, ground_eps=eps, ground_model="sommerfeld")
    )
    return (
        BSplineSolver(
            wires=[pts],
            n_per_edge_per_wire=[[n]],
            feeds=[(0, arc, 1 + 0j)],
            wavelength=WL7,
            wire_radius=0.001,
            **ground,
        ),
        fed,
    )


def contact_deck(depth=0.15):
    """The phase-0 lone-radial ANCHOR: a contact monopole over a buried
    radial. Refused — see the module docstring.

    Provenance, against the sentence the refusals print: a 10 m monopole
    standing 10 → 0 (its lower end IN the plane, which is the contact), one
    DETACHED 5 m radial 15 cm down, eps_r 13 / sigma 0.005 S/m, 7 MHz. The
    feed arclength 4.3333 m is the centre of segment 7 of 15, i.e. the
    engine deck's `EX 4,1,7`.
    """
    return dict(
        wires=[np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)]), _radial(depth=depth)],
        n_per_edge_per_wire=[[15], [10]],
        feeds=[(0, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def fan_deck(depth=0.15):
    """The phase-0 four-radial ANCHOR: `contact_deck`'s monopole over a fan
    of four detached radials, same soil and same feed site."""
    wires = [np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)])]
    npe = [[15]]
    for direction in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        wires.append(_radial(depth=depth, direction=direction))
        npe.append([10])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        # The four radials all start at (0, 0, -depth) — one buried node, so
        # one junction. Left undeclared until momwire#590's tripwire surfaced
        # it, they were solved as four DISCONNECTED wires; the refusal test
        # never noticed because it only reads the refusal message. The
        # monopole's own end is 15 cm above at (0, 0, 0) and is correctly not
        # a member.
        junctions=[[(1, "start"), (2, "start"), (3, "start"), (4, "start")]],
        feeds=[(0, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def element_currents(solver, coeffs):
    """Per-segment complex current, the quantity the engine's table prints."""
    _mid, moment, _nodes, _delta = solver.element_currents(coeffs)
    geom = solver._build_geometry()
    dl = geom["seg_r"] - geom["seg_l"]
    h = np.linalg.norm(dl, axis=1)
    return np.einsum("ij,ij->i", moment, dl / h[:, None]) / h


# ======================================================================
# G-U5-1 — the medium label, and the geometries still refused
# ======================================================================


def test_gu5_1_a_strictly_buried_wire_is_labelled_below():
    s = served_deck()
    assert s._wire_media() == (_medium_spec.ABOVE, _medium_spec.BELOW)
    assert s._has_buried_wires()
    geom = s._build_geometry()
    below = s._below_segments(geom)
    assert below[:15].sum() == 0
    assert below[15:].all()


def test_gu5_1_an_all_above_deck_labels_nothing_below():
    s = served_deck(free=False, depth=-0.5)  # radial 0.5 m ABOVE the plane
    assert s._wire_media() == (_medium_spec.ABOVE, _medium_spec.ABOVE)
    assert not s._has_buried_wires()


def test_gu5_1_free_space_never_invents_an_interface():
    s = served_deck(free=True)
    assert s._wire_media() == (_medium_spec.ABOVE, _medium_spec.ABOVE)
    assert not s._has_buried_wires()


def test_gu5_1_a_crossing_wire_refuses_naming_phase_two():
    wires = [np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 3.0)])]
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[10]],
        feeds=[(0, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    with pytest.raises(ValueError) as exc:
        s._wire_media()
    msg = str(exc.value)
    assert "crosses the ground interface" in msg
    assert "phase 2" in msg
    assert "74.761" in msg and "57.730" in msg


def test_gu5_1_an_end_in_the_plane_from_below_is_crossing_not_buried():
    """The tolerance decides, and it is the solver's own."""
    wires = [np.array([(0.0, 0.0, 0.0), (5.0, 0.0, -0.15)])]
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[10]],
        feeds=[(0, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    with pytest.raises(ValueError, match="crosses the ground interface"):
        s._wire_media()


@pytest.mark.parametrize(
    "ground,needle",
    [
        (dict(ground_z=0.0), "PERFECTLY CONDUCTING"),
        (
            dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="refl-coef"),
            "plane-wave boundary condition",
        ),
    ],
)
def test_gu5_1_buried_over_a_ground_with_no_lower_medium_refuses(ground, needle):
    s = BSplineSolver(
        wires=[_mono(), _radial()],
        n_per_edge_per_wire=[[15], [10]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **ground,
    )
    with pytest.raises(ValueError) as exc:
        s._wire_media()
    msg = str(exc.value)
    assert needle in msg
    assert "ground_model='sommerfeld'" in msg


def test_gu5_1_the_below_wire_grows_no_ground_contact_basis():
    """A buried wire never reaches the plane, so it has no contact END to
    tag and no in-plane EDGE to diagnose — the two questions the old
    "dips below the ground plane" raise stood in front of."""
    s = served_deck()
    start, end = s._wire_endpoint_status()
    assert start == ["free", "free"] and end == ["free", "free"]


# ======================================================================
# G-U5-2 — byte stability: the all-above path is structurally untouched
# ======================================================================


def test_gu5_2_an_all_above_deck_never_enters_the_buried_fill(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("an all-above deck reached the buried fill")

    monkeypatch.setattr(BSplineSolver, "_compute_Z_operator_buried", boom)
    z, _ = served_deck(depth=-0.5).compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)


def test_gu5_2_the_eps_argument_costs_the_shipped_assembly_no_bits():
    """`_assemble_Z(..., eps=None)` and the explicit float spelling are the
    same bytes — the seam is a default, not a code path."""
    s = served_deck(depth=-0.5)
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    J = s._build_J_blocks(geom, s.k)
    a = s._assemble_Z(J, supp_seg, polys, geom)
    b = s._assemble_Z(J, supp_seg, polys, geom, eps=s.eps)
    assert np.array_equal(a, b)


def test_gu5_2_the_weighted_image_eps_argument_costs_no_bits():
    s = served_deck(depth=-0.5)
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    J = s._build_J_blocks(geom, s.k)
    td = s._image_tangent_dot(geom["tangents"]).astype(np.complex128)
    w = np.full(td.shape, 0.5 + 0.25j)
    a = s._image_Z_weighted(J, supp_seg, polys, td, w)
    b = s._image_Z_weighted(J, supp_seg, polys, td, w, eps=s.eps)
    assert np.array_equal(a, b)


# ======================================================================
# G-U5-3 — the eps -> 1 collapse, and the boundary term that priced the
#          contact refusal
# ======================================================================


def _cross_blocks(s):
    """The two transmitted Galerkin blocks and the free-space mixed-potential
    block over the SAME cross pairs — the pair whose agreement at ε̃ = 1 is
    the fill's own self-consistency statement."""
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    below = s._below_segments(geom)
    b_idx = np.nonzero(below)[0]
    a_idx = np.nonzero(~below)[0]
    eps_t, _eps_m, k_p, k_m, _c2, _a_m = s._buried_medium()
    n = geom["n_segs_total"]
    mask = np.zeros((n, n), bool)
    mask[np.ix_(a_idx, b_idx)] = True
    mask[np.ix_(b_idx, a_idx)] = True
    z_mp = s._assemble_Z(s._build_J_blocks(geom, k_p) * mask, supp_seg, polys, geom)
    obs_a, t_a, w_a = s._buried_nodes(geom, a_idx)
    obs_b, t_b, w_b = s._buried_nodes(geom, b_idx)
    plan = s._buried_serve_plan(geom, a_idx, obs_a, obs_b, k_p, k_m)
    grid = _sommerfeld_transmitted.get_grid_below_above(
        eps_t,
        k_p,
        plan["r_cross_max"],
        plan["zp_min"],
        plan["zp_max"],
        s.omega,
        mu=s.mu,
        r_min=plan["r_cross_min"],
    )

    def up(o, to, sr, ts):
        return _sommerfeld_transmitted.transmitted_field_proj_below_to_above(
            o, to, sr, ts, 0.0, k_p, k_m, grid
        )

    def down(o, to, sr, ts):
        return _sommerfeld_transmitted.transmitted_field_proj_above_to_below(
            o, to, sr, ts, 0.0, k_p, k_m, grid
        )

    t_ab = s._field_galerkin_block(
        supp_seg, polys, up, a_idx, b_idx, obs_a, t_a, w_a, obs_b, t_b, w_b
    )
    t_ba = s._field_galerkin_block(
        supp_seg, polys, down, b_idx, a_idx, obs_b, t_b, w_b, obs_a, t_a, w_a
    )
    return z_mp, t_ab, t_ba


@pytest.mark.slow
def test_gu5_3_eps_one_collapses_the_cross_block_onto_free_space(record_property):
    """At ε̃ = 1 the transmitted family IS the free-space dipole (U3's own
    G-U3-5), so the whole cross block must reproduce the mixed-potential
    block over the same pairs. It is an identity of the FILL, independent of
    whether a ground at ε̃ = 1 is a sensible model."""
    s = served_deck(eps=(1.0, 0.0))
    z_mp, t_ab, t_ba = _cross_blocks(s)
    worst = float(np.max(np.abs(z_mp + t_ab + t_ba)) / np.max(np.abs(z_mp)))
    record_property("cross_vs_mixed_potential_rel", worst)
    assert worst < _ELEVATED_COLLAPSE_REL, (
        f"the cross block is not the free-space block: {worst:.3e}"
    )


@pytest.mark.slow
def test_gu5_3_a_ground_contact_breaks_that_collapse_by_o_one(record_property):
    """The measurement that refuses contact + buried, kept as a gate that
    FAILS IF FIXED: if this ever drops to the elevated deck's own scale, the
    boundary term has been dealt with and the refusal should come out.

    The disagreement is entirely on the CONTACT basis — every other basis
    row agrees to 1e-8 — and it does not move with quadrature order (3, 4,
    6 and 8 all read 2.3–2.5), which is what says it is a boundary term and
    not a resolution problem.

    THREE BANDS, BECAUSE `> 1.0` WAS A LIE
    --------------------------------------
    The trigger was `assert worst > 1.0` until momwire#567's scoping pass
    priced what that let through: the 2.3–2.5 band lived in prose and in the
    refusal strings, never in an assertion, so a fix that took the boundary
    term from 2.5 to 1.2 shipped GREEN while the contact row was still wrong
    by 120 %. The bands below separate the three verdicts that number can
    carry — dealt with, moved, unchanged — and only the first of them says
    the refusal may come out.
    """
    s = BSplineSolver(**{**contact_deck(), "ground_eps": (1.0, 0.0)})
    # The deck REFUSES (test_gu5_9_*), which is the point — the measurement
    # that refuses it has to reach past the refusal to stay honest, so the
    # medium labels are seeded directly instead of asked for.
    s._cached_wire_media = (_medium_spec.ABOVE, _medium_spec.BELOW)
    z_mp, t_ab, t_ba = _cross_blocks(s)
    d = np.abs(z_mp + t_ab + t_ba)
    scale = np.max(np.abs(z_mp))
    worst = float(np.max(d) / scale)
    # The contact basis is the LAST one supported on an above segment: bases
    # run wire by wire, the monopole is wire 0, and its boundary basis at the
    # grounded end is the one momwire#151 keeps rather than drops.
    geom = s._build_geometry()
    supp_seg, _polys, *_ = s._build_basis_polynomials(geom)
    n_above = int(np.count_nonzero(~s._below_segments(geom)))
    contact = int(np.max(np.nonzero(supp_seg[:, 0] < n_above)[0]))
    # Z is symmetric, so the contact basis shows up in a ROW and in a COLUMN;
    # striking both is what "localized on one basis" means here.
    rest = np.delete(np.delete(d, contact, axis=0), contact, axis=1)
    others = float(np.max(rest) / scale)
    record_property("contact_basis_rel", worst)
    record_property("every_other_basis_rel", others)
    if worst < _ELEVATED_COLLAPSE_REL:
        pytest.fail(
            f"the contact boundary term is now dealt with at the ELEVATED "
            f"deck's own scale ({worst:.3e} < {_ELEVATED_COLLAPSE_REL:.0e}): "
            "the contact+buried refusal should come out — test_gu5_9_*, and "
            "its three copies in src (_medium_spec.py, eznec/_serve.py, and "
            "bspline.py's capability string) — and this gate should be "
            "replaced by the G-U5-12 anchor serve gates, which arm the "
            "moment it does"
        )
    if worst <= _CONTACT_TERM_FLOOR:
        pytest.fail(
            f"the contact boundary term MOVED but is not at the elevated "
            f"deck's scale ({worst:.3e}, was 2.3–2.5 across quadrature "
            f"orders 3/4/6/8): do NOT lift the refusal on this. A fill that "
            "halves the term still leaves the contact row wrong by O(1), "
            "which is the ship-it-green hole momwire#567 found in the old "
            "`> 1.0` trigger. Find what moved it first"
        )
    assert others < 1e-2, f"the break is not localized on one basis: {others:.3e}"


@pytest.mark.slow
def test_gu5_3_a_fully_buried_deck_collapses_to_free_space(record_property):
    """No above wires at all: the medium machinery must vanish EXACTLY.
    A_m = 0 kills the image, D₁ = D₂ = 0 kills the remainder, k_m = k₀
    leaves the direct block, and there is no cross block to interpolate —
    so this one is machine precision, not an envelope."""
    s_free, _ = buried_dipole(free=True)
    s_one, _ = buried_dipole(eps=(1.0, 0.0))
    zf, _ = s_free.compute_impedance()
    zc, _ = s_one.compute_impedance()
    rel = abs(zc - zf) / abs(zf)
    record_property("fully_buried_collapse_rel", float(rel))
    assert rel < 1e-12, f"eps_t -> 1 did not collapse: {rel:.3e}"


def test_gu5_3_the_image_and_remainder_coefficients_vanish_at_eps_one():
    s = served_deck(eps=(1.0, 0.0))
    eps_t, _eps_m, _k_p, _k_m, c2, a_m = s._buried_medium()
    assert eps_t == 1
    assert c2 == 0 and a_m == 0
    surf = _sommerfeld_below.iv_surfaces_direct_below(
        1.0, 0.147, np.array([1.0]), np.array([0.5])
    )
    assert all(np.all(v == 0) for v in surf.values())


# ======================================================================
# G-U5-4 — reciprocity
# ======================================================================


@pytest.mark.slow
def test_gu5_4_the_two_cross_directions_are_one_transpose(record_property):
    """`transmitted_field_proj_above_to_below` is served as the reciprocity
    transpose of the SAME tables, so at the Galerkin level the two blocks
    must be each other's transpose to rounding. Any deviation is a bug."""
    s = served_deck()
    _z_mp, t_ab, t_ba = _cross_blocks(s)
    rel = float(np.max(np.abs(t_ab - t_ba.T)) / np.max(np.abs(t_ab)))
    record_property("cross_transpose_rel", rel)
    assert rel < 1e-12, f"the cross blocks are not transposes: {rel:.3e}"


@pytest.mark.slow
def test_gu5_4_the_mixed_medium_Z_is_symmetric(record_property):
    s = served_deck()
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    z = s._compute_Z_operator_buried(geom, supp_seg, polys)
    rel = float(np.max(np.abs(z - z.T)) / np.max(np.abs(z)))
    record_property("Z_symmetry_rel", rel)
    assert rel < 1e-11, f"the mixed-medium Z is not symmetric: {rel:.3e}"


# ======================================================================
# G-U5-5 — the served deck answers, and its own ladder converges
# ======================================================================


@pytest.mark.slow
def test_gu5_5_the_served_deck_converges_with_refinement(record_property):
    """momwire's OWN ladder: monotone-ish, and rung-to-rung deltas that
    shrink. The dipole-layer failure mode this rules out is a fill that
    DIVERGES with refinement, which is what an under-resolved Sommerfeld
    grid or a wrong near-field quadrature would look like."""
    zs = []
    for mult in (1, 3, 5):
        # ODD multipliers only — an even one moves the fed segment's centre
        # (the phase-0 capture rule), and a ladder that changes its feed
        # position between rungs measures two things at once.
        z, _ = served_deck(mult).compute_impedance()
        zs.append(z)
    d1 = abs(zs[1] - zs[0])
    d2 = abs(zs[2] - zs[1])
    record_property("ladder", [f"{z.real:.4f}{z.imag:+.4f}j" for z in zs])
    record_property("delta_x1_x3", float(d1))
    record_property("delta_x3_x5", float(d2))
    assert d2 < 0.35 * d1, f"the ladder is not converging: {d1:.4g} -> {d2:.4g}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "vertical,length,counts",
    [(True, 1.0, (11, 21, 41, 81)), (False, 10.0, (21, 41, 81))],
    ids=["bvd1", "bhd10"],
)
def test_gu5_5_a_fully_buried_ladder_converges(
    vertical, length, counts, record_property
):
    """The in-medium mesh ladder, on the decks with no interface crossing at
    all. What this rules out is the failure mode a wrong in-medium kernel
    would show: an answer that DIVERGES with refinement rather than
    settling. Convergence here is slow (a thin wire in a lossy medium) and
    that is fine — what is gated is monotone and shrinking."""
    zs = []
    for n in counts:
        s, _ = buried_dipole(n=n, length=length, vertical=vertical)
        z, _ = s.compute_impedance()
        zs.append(z)
    deltas = [abs(zs[i + 1] - zs[i]) for i in range(len(zs) - 1)]
    record_property("ladder", [f"{z.real:.4f}{z.imag:+.4f}j" for z in zs])
    record_property("deltas", [float(d) for d in deltas])
    assert all(b < a for a, b in zip(deltas, deltas[1:])), deltas
    assert all(zs[i + 1].real < zs[i].real for i in range(len(zs) - 1))


@pytest.mark.slow
def test_gu5_5_the_buried_wire_moves_the_answer(record_property):
    """A gate against a fill that quietly serves the above wire alone: the
    buried radial must CHANGE the elevated monopole's impedance."""
    z_with, _ = served_deck().compute_impedance()
    s = BSplineSolver(
        wires=[_mono(11.0, 1.0)],
        n_per_edge_per_wire=[[15]],
        feeds=[(0, (round(0.4333 * 15) - 0.5) / 15 * 10.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    z_without, _ = s.compute_impedance()
    rel = abs(z_with - z_without) / abs(z_without)
    record_property("with", f"{z_with:.4f}")
    record_property("without", f"{z_without:.4f}")
    record_property("radial_worth_rel", float(rel))
    # 8e-5 of |Z| on this deck — small, and that is the physics rather than a
    # missing coupling: a DETACHED radial 15 cm into lossy soil, 1 m under an
    # elevated monopole, is a weakly excited parasite. What the gate rules out
    # is a fill that serves the above wire ALONE, which would read exactly 0.
    assert rel > 1e-5, f"the buried wire did not reach the answer: {rel:.3e}"


# ======================================================================
# G-U5-6 — the serve domain refuses by name, never clamps
# ======================================================================


@pytest.mark.slow
def test_gu5_6_a_buried_structure_past_the_below_cap_refuses():
    s = BSplineSolver(
        wires=[_mono(), _radial(length=40.0)],
        n_per_edge_per_wire=[[15], [20]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    with pytest.raises(ValueError) as exc:
        s.compute_impedance()
    msg = str(exc.value)
    assert "below/below pair separation" in msg
    assert "in-medium wavelengths" in msg
    assert "no honest clamp" in msg


@pytest.mark.slow
def test_gu5_6_a_grazing_buried_pair_refuses_with_the_lateral_wave():
    """Depth 1 mm over a 5 m radial is theta = 0.023 deg — under the 1 deg
    floor the below/below surfaces are tabulated from."""
    s = BSplineSolver(
        wires=[_mono(), _radial(length=5.0, depth=0.001)],
        n_per_edge_per_wire=[[15], [10]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    with pytest.raises(ValueError) as exc:
        s.compute_impedance()
    msg = str(exc.value)
    assert "grazing floor" in msg
    assert "lateral wave" in msg.lower() or "LOGARITHMIC" in msg


@pytest.mark.slow
def test_gu5_6_a_source_deeper_than_the_zprime_ladder_refuses():
    s = BSplineSolver(
        wires=[_mono(), _radial(length=1.0, depth=4.0)],
        n_per_edge_per_wire=[[15], [10]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    with pytest.raises(ValueError) as exc:
        s.compute_impedance()
    assert "z' ladder" in str(exc.value)


def test_gu5_6_the_serve_plan_reads_the_domain_the_grid_will_have():
    """`grid_extent` answers with the BUCKETED radius, so a serve check
    cannot pass a geometry the grid then refuses from inside its fill."""
    k2 = 2.0 * np.pi * F7 / C0
    r_min, r_max, th = _sommerfeld_transmitted.grid_extent(k2, 11.15, 0.15, r_min=0.028)
    assert r_max >= 11.15
    assert r_min <= 0.028
    assert th == pytest.approx(
        _sommerfeld_transmitted.grazing_floor(r_max, 0.15), rel=1e-12
    )


def test_gu5_6_the_extended_r_min_lands_on_the_default_lattice():
    lam_p = WL7
    default = 0.001 * lam_p
    for req in (0.03, 0.005, 0.0006):
        got = _sommerfeld_transmitted.r_min_bucket(req, lam_p)
        assert got <= req
        steps = math.log(default / got) / 0.10
        assert abs(steps - round(steps)) < 1e-9
    assert _sommerfeld_transmitted.r_min_bucket(None, lam_p) == default
    assert _sommerfeld_transmitted.r_min_bucket(10.0, lam_p) == default
    with pytest.raises(ValueError, match="extension floor"):
        _sommerfeld_transmitted.r_min_bucket(1e-9 * lam_p, lam_p)


@pytest.mark.slow
def test_gu5_6_the_extended_r_min_interpolates(record_property):
    """The measurement behind `_R_MIN_LAMBDA_P` becoming an argument: the
    extended decade is not a near zone at a FIXED source depth, because the
    true separation is bounded below by |z'|."""
    k2 = 2.0 * np.pi * F7 / C0
    om = 2.0 * np.pi * F7
    eps_t = _ground_refl.eps_tilde(SOIL_A, om, 8.8541878128e-12)
    _sommerfeld_below.k_medium(eps_t, k2)  # the branch the grid keys on
    grid = _sommerfeld_transmitted.get_grid_below_above(
        eps_t, k2, 1.0, 0.15, 0.15, om, r_min=5e-4
    )
    worst = 0.0
    rng = np.random.default_rng(7)
    for _ in range(24):
        r = 10 ** rng.uniform(math.log10(6e-4), math.log10(0.04))
        th = np.radians(rng.uniform(0.5, 89.0))
        got = grid.eval(np.array([r]), np.array([th]), np.array([-0.15]))
        ref = _sommerfeld_transmitted.t_surfaces_direct(
            eps_t, k2, r * np.cos(th), r * np.sin(th), -0.15, rtol=1e-10, omega=om
        )
        for key in got:
            a = complex(np.asarray(got[key]).ravel()[0])
            b = complex(np.asarray(ref[key]).ravel()[0])
            if abs(b) > 0:
                worst = max(worst, abs(a - b) / abs(b))
    record_property("extended_r_min_worst_rel", float(worst))
    assert worst < 1e-5, f"the extended log axis does not interpolate: {worst:.3e}"


# ======================================================================
# G-U5-7 — ports, loads and the configurations a buried deck may not reach
# ======================================================================


@pytest.mark.slow
def test_gu5_7_wire_loading_passes_through_a_buried_wire(record_property):
    """The internal impedance of a conductor is the METAL's — k_c =
    sqrt(jωμσ_metal) and the jacket's own ε_r — with no term that knows what
    the wire is standing in. So loading must compose with the medium, and
    the way to say that in a test is that switching the buried wire's metal
    moves the answer by the loading term and nothing else."""
    bare, _ = buried_dipole()
    z_bare, _ = bare.compute_impedance()
    s = BSplineSolver(
        wires=[np.array([(0.0, 0.0, -1.15), (0.0, 0.0, -0.15)])],
        n_per_edge_per_wire=[[11]],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
        wire_conductivity=1.0e5,
    )
    z_loaded, _ = s.compute_impedance()
    record_property("bare", f"{z_bare:.4f}")
    record_property("loaded", f"{z_loaded:.4f}")
    assert z_loaded.real > z_bare.real
    assert np.isfinite(z_loaded.imag)


@pytest.mark.slow
def test_gu5_7_a_fed_buried_wire_is_legal():
    """bhd1/bvd1 are FED buried dipoles; a detached buried radial is not
    fed. Both are served, and the port algebra is basis-level."""
    s, fed = buried_dipole()
    z, coeffs = s.compute_impedance()
    cur = element_currents(s, coeffs)
    assert np.isfinite(z.real) and z.real > 0
    assert abs(cur[fed - 1]) == pytest.approx(np.max(np.abs(cur)), rel=1e-9)


def test_gu5_7_singular_enrichment_refuses_a_buried_deck():
    s = served_deck(use_singular_enrichment=True)
    with pytest.raises(NotImplementedError, match="use_singular_enrichment"):
        s.compute_impedance()


def test_gu5_7_the_extended_kernel_refuses_a_buried_deck():
    s = served_deck(extended_kernel=True)
    with pytest.raises(NotImplementedError, match="extended_kernel"):
        s.compute_impedance()


def test_gu5_7_the_dense_budget_refuses_rather_than_chunking():
    s = served_deck(8, swept_mem_mb=1)
    with pytest.raises(NotImplementedError) as exc:
        s.compute_impedance()
    assert "DENSE moment tensor" in str(exc.value)
    assert "swept_mem_mb" in str(exc.value)


def test_gu5_7_the_fast_operator_refuses_a_buried_deck():
    s = HMatrixSolver(
        wires=[_mono(), _radial()],
        n_per_edge_per_wire=[[15], [10]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    with pytest.raises(NotImplementedError) as exc:
        s.build_hmatrix()
    msg = str(exc.value)
    assert "fast operator has no per-segment medium" in msg
    assert "does NOT fall back" in msg


# ======================================================================
# G-U5-8 — the fully-buried decks against every oracle available
# ======================================================================


@pytest.mark.slow
def test_gu5_8_a_deep_buried_dipole_is_the_infinite_medium_dipole(record_property):
    """The strongest available in-medium oracle, and it needs no engine: as
    the source sinks, the image and the remainder both go away and the
    answer must run into the SAME wire solved in an infinite medium — which
    is the k_m direct block on its own, assembled with ε̃_m.

    It also prices the interface: the difference between a 15 cm burial and
    the infinite-medium limit is what the image and remainder blocks are
    worth on this deck.
    """
    s, _ = buried_dipole(depth=0.15)
    z_shallow, _ = s.compute_impedance()
    s_deep, _ = buried_dipole(depth=1.5)
    z_deep, _ = s_deep.compute_impedance()

    ref = BSplineSolver(
        wires=[np.array([(0.0, 0.0, -1.15), (0.0, 0.0, -0.15)])],
        n_per_edge_per_wire=[[11]],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
    )
    eps_t = _ground_refl.eps_tilde(SOIL_A, ref.omega, ref.eps)
    k_m = _sommerfeld_below.k_medium(eps_t, ref.k)
    geom = ref._build_geometry()
    supp_seg, polys, kcl_A, wk, wbg = ref._build_basis_polynomials(geom)
    z_op = ref._assemble_Z(
        ref._build_J_blocks(geom, k_m), supp_seg, polys, geom, eps=ref.eps * eps_t
    )
    v, port_vectors, _x, volts, kcl_con = ref._feed_drive_and_readout(
        geom, wk, wbg, supp_seg.shape[0], kcl_A
    )
    c = ref._solve_with_kcl(ref._apply_loading(z_op), v, kcl_con)
    z_inf = volts[0] / (port_vectors[0] @ c[: port_vectors[0].shape[0]])

    rel_deep = abs(z_deep - z_inf) / abs(z_inf)
    rel_shallow = abs(z_shallow - z_inf) / abs(z_inf)
    record_property("z_infinite_medium", f"{z_inf:.4f}")
    record_property("z_depth_1.5m", f"{z_deep:.4f}")
    record_property("z_depth_0.15m", f"{z_shallow:.4f}")
    record_property("deep_vs_infinite_rel", float(rel_deep))
    record_property("shallow_vs_infinite_rel", float(rel_shallow))
    assert rel_deep < 5e-3, f"the deep limit is not the medium: {rel_deep:.3e}"
    assert rel_shallow > rel_deep, "burying deeper did not reduce the interface term"


@pytest.mark.slow
def test_gu5_8_the_engine_current_tables_disagree_and_that_is_recorded(
    record_property,
):
    """**A FINDING, gated as one.**

    The brief for this unit named the engine's printed bhd1/bvd1 current
    tables as momwire's gate. They are not usable as one, and this test
    records the measurement rather than a tolerance nobody could meet.

    On `ne-bvd1-d0.15-A-7MHz` the engine prints a current that is ~1.0 A at
    the driven element and under 0.011 A one element away, every off-feed
    element at exactly 180.0 deg; momwire solves the smooth triangular taper
    a short dipole has (0.88 one element away, 0.11 at the tips), and prints
    341.6 − 329.5j ohm where the engine prints −0.0025 + 3.4712j — a printed
    NEGATIVE resistance on a passive antenna.

    Three independent checks say momwire's side is the physical one, and
    none of them is the engine: the ε̃ → 1 collapse is exact to 4e-15
    (G-U5-3), the deep-burial limit runs into the infinite-medium solve to
    5e-4 (above), and a quasi-static two-electrode estimate of the same
    wire — R_dc = 2·ln(2L/a)/(2πσL) divided by (1 + jωε/σ) — gives
    434 − 440j, within 1.3x. So the disagreement is recorded as a follow-up
    against the ENGINE's buried-dipole solve, not as a momwire tolerance.

    The gate is written to FAIL IF IT AGREES: if a later change brings the
    two together, this test is what says the finding is closed.
    """
    from golden_buried_currents_nec5 import DECKS

    d = DECKS["ne-bvd1-d0.15-A-7MHz"]
    s, fed = buried_dipole(n=11, length=1.0, depth=0.15)
    z, coeffs = s.compute_impedance()
    got = element_currents(s, coeffs)
    got = got / got[fed - 1]
    ref = np.array(
        [m * np.exp(1j * math.radians(p)) for _e, _zc, m, p in d["currents"]]
    )
    ref = ref / ref[fed - 1]
    worst = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    record_property("momwire_Z", f"{z:.4f}")
    record_property("engine_Z", str(d["input_z"]))
    record_property("worst_current_rel", worst)
    record_property("momwire_tip_current", float(abs(got[0])))
    record_property("engine_tip_current", float(abs(ref[0])))
    assert worst > 0.5, (
        "momwire and the engine now AGREE on the buried-dipole current "
        f"distribution ({worst:.3e}) — the momwire#553 U5 finding is closed "
        "and this gate should be replaced by the envelope it always wanted"
    )
    assert d["input_z"].real < 0.0, "the engine's printed R is no longer negative"


# ======================================================================
# G-U5-9 — the anchors, and the refusal that costs the unit its two decks
#
# The refusal is what these tests read. The anchors' NUMBERS are gated at
# G-U5-12, which arms itself the moment the refusal comes out.
# ======================================================================


def test_gu5_9_the_lone_radial_anchor_refuses_naming_both_gates():
    s = BSplineSolver(**contact_deck())
    with pytest.raises(ValueError) as exc:
        s.compute_impedance()
    msg = str(exc.value)
    assert "stands an END in the ground plane" in msg
    assert "92.130 - 70.141j" in msg
    assert "90.051 - 70.731j" in msg
    assert "phase 2" in msg


def test_gu5_9_the_four_radial_anchor_refuses_the_same_way():
    s = BSplineSolver(**fan_deck())
    with pytest.raises(ValueError, match="stands an END in the ground plane"):
        s.compute_impedance()


def test_gu5_9_a_contact_deck_with_no_buried_wire_still_serves():
    """The refusal is about the COMBINATION: neither half alone is refused."""
    s = BSplineSolver(
        wires=[np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)])],
        n_per_edge_per_wire=[[15]],
        feeds=[(0, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    z, _ = s.compute_impedance()
    assert z.real > 0


# ======================================================================
# G-U5-10 — the quadrature order the cross block needed
# ======================================================================


def test_gu5_10_the_buried_field_quadrature_is_not_the_remainders_default():
    s = served_deck()
    assert s._n_qp_buried_field() > s.n_qp_sommerfeld
    s.n_qp_sommerfeld = 99
    assert s._n_qp_buried_field() == 99


@pytest.mark.slow
def test_gu5_10_the_cross_block_needs_the_higher_order(record_property):
    """The fifth inversion, measured: at ε̃ = 1 the cross block must be the
    free-space block, and at the remainder families' own `n_qp_sommerfeld`
    it is not — because the transmitted integral is the whole field, near
    zone and all, where a remainder is smooth."""
    s = served_deck(eps=(1.0, 0.0), clearance=0.4)
    z_mp, t_ab, t_ba = _cross_blocks(s)
    scale = np.max(np.abs(z_mp))
    good = float(np.max(np.abs(z_mp + t_ab + t_ba)) / scale)
    s_coarse = served_deck(eps=(1.0, 0.0), clearance=0.4)
    s_coarse._n_qp_buried_field = lambda: 3
    z_mp3, t3_ab, t3_ba = _cross_blocks(s_coarse)
    coarse = float(np.max(np.abs(z_mp3 + t3_ab + t3_ba)) / np.max(np.abs(z_mp3)))
    record_property("q_measured", good)
    record_property("q_3", coarse)
    assert coarse > 3.0 * good, (
        f"n_qp_sommerfeld's order is no worse than the measured one "
        f"({coarse:.3e} vs {good:.3e}) — the fifth inversion has gone away"
    )


# ======================================================================
# G-U5-11 — the fill's cost, so a regression in it is visible
# ======================================================================


@pytest.mark.slow
def test_gu5_11_the_grids_are_shared_across_a_ladder(record_property):
    """The two buried grids are the whole cost of a buried solve (~50 s of
    numpy at the anchor scale), and they are keyed so a convergence ladder
    pays for them ONCE. A rung that re-fills is a cache-key regression."""
    from momwire._sommerfeld import _GRID_CACHE

    _GRID_CACHE.clear()
    t0 = time.time()
    served_deck(1).compute_impedance()
    cold = time.time() - t0
    t0 = time.time()
    served_deck(2).compute_impedance()
    warm = time.time() - t0
    record_property("cold_s", cold)
    record_property("warm_s", warm)
    assert warm < 0.5 * cold, f"the second rung re-filled a grid: {warm:.1f}s"


# ======================================================================
# G-U5-12 — the two banked anchors, as GATES rather than as strings
#
# Until momwire#567's scoping pass, the two anchor numbers existed in this
# repo only as substrings: three copies of refusal prose in
# src and four tests asserting that the prose still contains them. Nothing
# ever compared an impedance to them, so landing the contact+buried fix
# would not have satisfied them and nobody would have noticed.
#
# These gates close that. They go through the FRONT DOOR — no seeded media,
# no reached-past refusal — so today they hit the refusal and xfail, and the
# day the refusal comes out they score the deck against the engine without
# anyone editing this file. That is the whole design: a gate that arms
# itself is a gate that cannot be forgotten.
# ======================================================================

# The anchors themselves — our licensed NEC-5 engine's printout for the two
# phase-0 ANCHOR decks — are imported at the top of this file from
# `golden_buried_anchor_nec5`, which `scripts/capture_buried_anchor_nec5.py`
# regenerates from the banked cards (momwire#567 phase 0 re-ran the engine
# and found the fan bank was a transcription 0.67 ohm off the printout).
# These are the numbers `_medium_spec._REFUSE_CONTACT_WITH_BURIED` and
# `eznec._serve._REFUSE_BURIED_WITH_CONTACT` print; the binding is gated
# below so the constants and the prose cannot drift apart.

# PROVISIONAL, in absolute ohms, and it should be re-derived rather than
# inherited. 4.0 is `test_contact_nec5_lane._ENVELOPE["poor"]` — the loosest
# bar momwire's shipped ground-CONTACT row already lives under, on a plain
# contact monopole with nothing buried. These decks carry that same contact
# node PLUS a cross-medium block that does not exist yet, so nothing here
# can be argued tighter than the contact node's own worst shipped miss, and
# the number is deliberately not imported: re-deriving the contact lane's
# envelope should NOT silently move this one. momwire#567 phase 0 measures
# what agreement is actually achievable; this holds the slot until it does.
ANCHOR_ENVELOPE_OHM = 4.0

# What each trunk refuses these decks with TODAY. Since momwire#651 routed
# razor's buried readings through `_medium_spec.wire_media`, BOTH trunks
# refuse the anchor decks with the SAME shared contact+buried sentence.
# Razor's own buried fill is still missing (its own-gap sentence covers the
# wholly-buried DETACHED decks bspline serves), so arming razor's row needs
# both momwire#567 and the momwire#651 continuation.
_TRUNK_REFUSAL = {
    "bspline": (
        BSplineSolver,
        "stands an END in the ground plane",
        "the contact+buried combination is refused — momwire#567",
    ),
    "razor": (
        RazorSolver,
        "stands an END in the ground plane",
        "the shared contact+buried sentence (momwire#651); razor's own "
        "buried fill is still missing, so momwire#567 alone will not arm "
        "this row",
    ),
}

_ANCHOR_DECK = {
    "lone-radial": (contact_deck, ANCHOR_LONE_RADIAL),
    "four-radial": (fan_deck, ANCHOR_FOUR_RADIAL),
}


def test_gu5_12_the_anchor_constants_are_the_numbers_the_refusals_print():
    """The constants above and the copies of refusal prose in src are four
    spellings of two numbers. This is the only test that ties them."""
    from momwire.eznec import _serve

    for anchor in (ANCHOR_LONE_RADIAL, ANCHOR_FOUR_RADIAL):
        printed = f"{anchor.real:.3f} - {abs(anchor.imag):.3f}j"
        assert printed in _medium_spec._REFUSE_CONTACT_WITH_BURIED
        assert printed in _serve._REFUSE_BURIED_WITH_CONTACT


@pytest.mark.parametrize("trunk", sorted(_TRUNK_REFUSAL))
@pytest.mark.parametrize("deck", sorted(_ANCHOR_DECK))
def test_gu5_12_the_anchor_deck_answers_what_the_engine_printed(
    trunk, deck, record_property
):
    """The serve gate the two anchors never had.

    It is NOT marked slow, because today it costs a refusal and nothing
    else. When momwire#567 arms it, each row becomes a full buried solve
    (~50 s at this scale, G-U5-11) and the marker has to go on then — that
    is a deliberate item for whoever lifts the refusal, not an oversight
    here.
    """
    solver, refusal, why = _TRUNK_REFUSAL[trunk]
    build, anchor = _ANCHOR_DECK[deck]
    try:
        z, _ = solver(**build()).compute_impedance()
    except ValueError as exc:
        assert refusal in str(exc), (
            f"{trunk} no longer refuses the {deck} anchor with the sentence "
            f"this gate keys on — it says {str(exc)[:160]!r}. Either the "
            "deck now serves (delete the try/except and let the anchor "
            "comparison stand) or the refusal was reworded"
        )
        pytest.xfail(f"{trunk} refuses the {deck} anchor: {why}")
    miss = abs(z - anchor)
    record_property("momwire_Z", f"{z:.4f}")
    record_property("engine_Z", f"{anchor:.4f}")
    record_property("anchor_miss_ohm", float(miss))
    assert miss <= ANCHOR_ENVELOPE_OHM, (
        f"{trunk} answers {z:.4f} on the {deck} anchor where the engine "
        f"prints {anchor:.4f} — {miss:.4f} ohm apart, outside the "
        f"provisional {ANCHOR_ENVELOPE_OHM:g} ohm envelope"
    )


# ======================================================================
# momwire#651 — razor refuses buried geometry with the SHARED sentences
# ======================================================================


def _fed_crossing_deck():
    return dict(
        wires=[np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 3.0)])],
        n_per_edge_per_wire=[[10]],
        feeds=[(0, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def _detached_buried_deck(**ground):
    return dict(
        wires=[_mono(11.0, 1.0), _radial(depth=0.5)],
        n_per_edge_per_wire=[[15], [10]],
        feeds=[(0, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        **ground,
    )


_PARITY_DECKS = {
    "crossing": _fed_crossing_deck,
    "contact-with-buried": contact_deck,
    "buried-under-pec": _detached_buried_deck,
    "buried-under-refl-coef": lambda: _detached_buried_deck(
        ground_eps=SOIL_A, ground_model="refl-coef"
    ),
}


def _refusal_message(solver, build):
    """The sentence the trunk says, at whichever stage it fires — razor
    refuses at construction, bspline at its (lazy) media reading."""
    with pytest.raises(ValueError) as exc:
        s = solver(**build)
        s._wire_media()
    return str(exc.value)


@pytest.mark.parametrize("case", sorted(_PARITY_DECKS))
def test_651_the_two_trunks_refuse_buried_geometry_identically(case):
    """momwire#651: crossing, contact+buried and no-lower-medium decks are
    refused by BOTH trunks with byte-identical `_medium_spec` sentences.
    The one legitimate divergence — the wholly-buried DETACHED deck bspline
    serves — is the next test's."""
    build = _PARITY_DECKS[case]()
    assert _refusal_message(BSplineSolver, build) == _refusal_message(
        RazorSolver, build
    )


def test_651_razors_own_gap_sentence_names_the_serving_trunk():
    """The detached buried deck is LEGAL — bspline labels it and serves.
    Razor's refusal must say the gap is razor's own, name the trunk that
    serves the deck, and point at the momwire#651 continuation."""
    build = _detached_buried_deck(ground_eps=SOIL_A, ground_model="sommerfeld")
    assert _medium_spec.BELOW in BSplineSolver(**build)._wire_media()
    with pytest.raises(ValueError) as exc:
        RazorSolver(**build)
    msg = str(exc.value)
    assert "RazorSolver has no buried fill" in msg
    assert "BSplineSolver" in msg
    assert "momwire#651" in msg
