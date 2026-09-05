"""momwire#524 phase 2 — the crossing serve (G-524).

A wholly-below wire whose end stands in the ground plane, junction-joined
there to an above wire, is SERVED: the cross pair is filled with the
complete designed mixed-potential spelling on graded axes
(`_crossing_fill`), the self families get their missing by-parts
bnd + corner content, and continuity of current through the node plus the
AGARD slope condition emerge from the fill with no constraint row and no
merged dof (split ≡ merged ≡ V-constrained, measured to the digit).

The serve gate is momwire's OWN evidence, adjudicated 2026-08-26:

* mesh stability — Δ(crossing − mono) moves 0.02 Ω between the g1/g2
  interface-graded meshes (138.7671−102.9889j → 138.7691−102.9893j).
  Both prints are at the then-default n_qp_pair=4 and each carries
  ~0.43 Ω of quadrature error (momwire#760); the claim survives because
  it is about the DIFFERENCE between two meshes at one quadrature, and
  the error is common to both. `CROSSING_G1` itself is re-banked at
  converged quadrature below;
* the ε̃ = 1 collapse — at ground_eps = 1 the interface vanishes and the
  crossing deck IS a free-space 12 m wire, reproduced to 0.0124 Ω
  (0.002 %) through a corner telescoping of magnitude ~204,000;
* the high-σ collapse — the crossing answer falls onto the shipped
  contact-mono column exactly as σ → ∞ (|Δ| 86 → 2.8 across
  σ = 0.005 → 5 S/m), which is the one limit where the contact fiction
  is physical.

The licensed engine's crossing print (74.761 − 57.730j Ω on the g-class
deck) is a DIFFERENT EXPERIMENT, not a miss: its own printed junction
currents violate its AGARD condition divergently (I(0⁻) antiphase, ~√n
growth, ~2 A KCL deficit into the interface point), so its junction is
two contact ends plus a point-electrode sink. It is documented here and
NEVER gated against — the house rule about cross-formulation agreement.

THE FAN WIDENING (session 8) serves 1 above × N below. The composition
past K = 2 carries a node-mesh convergence class (see `test_g524_7`),
RESOLVED by the #674 study (scratch/674-study): the slow term is the
ABOVE tent's interface-adjacent h (first order — rise-only grading
doesn't move it), matched per-arm node grading restores ~2.6-order
convergence, and the ε̃ = 1 residual extrapolates to zero (0.0004 Ω,
two independent rung pairs). Lossy transmitted kernels amplify the
class ~30× — the base-mesh soil-A prints (fan 143.9327−26.2135j; hub
spelling 140.9839−43.6025j, a DIFFERENT structure; probe38) stay
RECORDS, and so does the study's own graded print: every #674 number
was taken at a FIXED n_qp_pair=4, which momwire#760 measures 6.8 Ω
from the quadrature limit. `FAN_SOIL_A_N2` is re-banked at converged
quadrature and is the only one of them that is a gate. The ε̃ = 1
collapses run in the merge-to-main `crossgate` lane (multi-minute
certification solves, the memgate reasoning) while the PR slow lane
keeps `test_g524_4` as the per-PR crossing regression pin. The HUB spelling's collapse is a banked
record (0.2194 Ω, probe38) with no gate of its own: its only unique
content — the hub-tent by-parts terms — was measured to contribute
exactly zero to the digit (probe39), and everything else it would
exercise, `test_g524_7` already does.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from momwire import _bspline_kernels, _crossing_fill, _medium_spec, _near_interface
from momwire.bspline import DEFAULT_N_QP_PAIR, BSplineSolver

from test_buried_serve_553 import SOIL_A, WL7, contact_deck

A_WIRE = 0.001

# Quadrature order for the ε̃ = 1 COLLAPSE adjudicators, pinned on both sides
# of every one of them (momwire#760).
#
# Those tests compare a buried fill at ε̃ = 1 against a free-space truth, and
# two of them build the truth by STRIPPING the ground keys off the same build
# dict -- so before #760 both sides simply inherited one global default and the
# match was automatic. It is not automatic any more: the buried fill resolves
# its own, higher default, and stripping `ground_z` flips the truth side back
# to the free-space one. Left implicit, three of the four adjudicators fail by
# almost exactly the quadrature gap rather than by anything about the
# composition (g524_5 missed by 0.0938 ohm, against the 0.0929 ohm this deck
# moves between q=8 and converged).
#
# The collapse is a statement about COMPOSITION -- that the buried machinery
# telescopes to the free-space answer when the interface vanishes -- and it is
# only meaningful when both sides integrate the same way. Pinned at the
# free-space default so the banked envelopes keep the meaning they were
# measured with; the VALUE does not matter to the identity, the MATCH does.
_COLLAPSE_N_QP = DEFAULT_N_QP_PAIR


# The interface-graded meshes the phase-2 probes banked (probe18 GRADES):
# vertices walk toward z = 0 so the node segments shrink without the
# uniform-mesh blow-up the graded ladder measured.
_GRADES = {
    1: dict(
        below=([-2.0, -0.5, -0.1], [3, 2, 2]), above=([0.1, 0.5, 10.0], [2, 2, 19])
    ),
    2: dict(
        below=([-2.0, -0.5, -0.1, -0.025], [3, 2, 3, 2]),
        above=([0.025, 0.1, 0.5, 10.0], [2, 3, 2, 19]),
    ),
}

# The soil-A exact-EM crossing answer, converged in the QUADRATURE axis
# as well as the mesh axis (momwire#760).
#
# probe27 (session 5) banked 138.7671 - 102.9889j here, mesh-stable
# g1 -> g2, and #692's near-density default reproduced it to 1e-4. Both
# were taken at the then-default cross-edge quadrature of 4 — which is
# also why they agree so exactly: the bank IS the q=4 print, and the
# quadrature axis was never in the 0.05 envelope. Walking it at fixed
# mesh: q=4 0.4333 / q=8 0.0929 / q=16 0.0087 / q=32 0.0003 ohm from the
# limit, settled to 4 decimals by q=64 (q=64 -> 96 moves 0.0000).
#
# This deck is the mild member of the class #760 documents. It costs
# ~0.4 ohm rather than the fan anchor's 6.8, and it recovers its rate by
# q=32 instead of crawling at ~C/q — the crossing node here has no
# coincident rises, which is the geometry that destroys the rate.
# Its envelope is the NODE axis (g1<->g2) plus quadrature. The FAR mesh is
# a third axis this print does not carry: scaling the far mesh moves degree
# 2 by 0.36 ohm at x3 and 0.61 at x9 (G1-B, test_bspline_pair_g1b), so the
# 0.05 gate below is a regression gate at the g1 far mesh, not a
# far-mesh-converged answer.
CROSSING_G1 = 138.9619 - 102.6019j

# As FAN_SOIL_A_N2_QP, for the same reason and past the same n_qp <= 8
# accelerator cap (momwire#762). Costs ~1.4 s on this deck; the shipped
# default of 8 lands 0.0929 ohm out, outside the 0.05 gate below.
CROSSING_G1_QP = 64
# The g2 mesh's q=4 print, kept as the other half of the mesh-movement
# record above. Not a gate, and deliberately NOT re-banked: its only job
# is the Δ against CROSSING_G1's q=4 print, which is why the pair has to
# stay at one quadrature.
CROSSING_G2 = 138.7691 - 102.9893j


def crossing_deck(level=1, **override):
    g = _GRADES[level]
    below_pts = np.array([(0.0, 0.0, z) for z in g["below"][0] + [0.0]])
    above_pts = np.array([(0.0, 0.0, z) for z in [0.0] + g["above"][0]])
    build = dict(
        wires=[below_pts, above_pts],
        n_per_edge_per_wire=[g["below"][1], g["above"][1]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(1, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    build.update(override)
    return build


# ----------------------------------------------------------------------
# G-524-1 — the served spelling labels; everything else refuses by name
# ----------------------------------------------------------------------


def test_g524_1_junction_spelling_is_served():
    s = BSplineSolver(**crossing_deck())
    assert s._wire_media() == (_medium_spec.BELOW, _medium_spec.ABOVE)
    assert s._crossing_junctions() == (0,)


def test_g524_1_midspan_crossing_still_refuses_naming_the_spelling():
    wires = [np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 3.0)])]
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[10]],
        feeds=[(0, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    with pytest.raises(ValueError) as exc:
        s._wire_media()
    msg = str(exc.value)
    assert "crosses the ground interface mid-span" in msg
    assert "split the wire AT the interface" in msg
    assert "declare the junction" in msg


def test_g524_1_touching_end_without_a_junction_still_refuses():
    """The exemption is the JUNCTION's, not the geometry's: a lone below
    wire with its end in the plane and no partner above it is still the
    crossing refusal (a bare interface touchdown has no declared
    continuation). N.B. omitting `junctions` on the two-wire deck does
    NOT reach this case — coincident wire ends are auto-detected as a
    junction, and that deck IS the served spelling."""
    build = crossing_deck()
    build["wires"] = [build["wires"][0]]
    build["n_per_edge_per_wire"] = [build["n_per_edge_per_wire"][0]]
    build.pop("junctions")
    build["feeds"] = [(0, 1.0, 1 + 0j)]
    s = BSplineSolver(**build)
    with pytest.raises(ValueError, match="crosses the ground interface"):
        s._wire_media()


def test_g524_1_detached_buried_screen_is_untouched():
    """The momwire#553 serve class must not route through the crossing
    fill: a detached buried wire has no crossing junction."""
    build = crossing_deck()
    build["wires"] = [
        np.array([(0.0, 0.0, -0.5), (5.0, 0.0, -0.5)]),
        np.array([(0.0, 0.0, 1.0), (0.0, 0.0, 11.0)]),
    ]
    build["n_per_edge_per_wire"] = [[5], [10]]
    build.pop("junctions")
    s = BSplineSolver(**build)
    assert s._wire_media() == (_medium_spec.BELOW, _medium_spec.ABOVE)
    assert s._crossing_junctions() == ()


# ----------------------------------------------------------------------
# G-524-2 — scope refusals, each by name
# ----------------------------------------------------------------------


def fan_rise_deck(n_radials=4, depth=0.15, **override):
    """The connected radial screen, rise-spelled (momwire#524 fan
    widening): `contact_deck`'s monopole junction-joined at the node to
    N radials that each run at depth and RISE to the surface. The N rise
    segments are geometrically coincident on (0,0,−depth) → (0,0,0) —
    legal thin-wire geometry (mutual ≡ self at ρ = 0 under the
    ρ_eff = √(ρ² + a²) regularization), and the spelling the free-space
    junction machinery solves identically for the ε̃ = 1 adjudicator.
    Feed = arclength 4.3333 on the 10 → 0 monopole. NOT the engine's
    `EX 4,1,7` — that card drives the far NODE at arc 4.6667 (momwire#706);
    these gates are momwire-internal (both sides of every comparison feed
    at 4.3333), so the banked prints stand. The old trap also stands: an
    improvised feed at 10 − 4.333 is silently ~50 Ω wrong."""
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))[:n_radials]
    wires = [
        np.array([(5.0 * dx, 5.0 * dy, -depth), (0.0, 0.0, -depth), (0.0, 0.0, 0.0)])
        for dx, dy in dirs
    ]
    npe = [[10, 2] for _ in dirs]
    mono_i = len(wires)
    wires.append(np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)]))
    npe.append([15])
    build = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[[(i, "end") for i in range(n_radials)] + [(mono_i, "end")]],
        feeds=[(mono_i, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    build.update(override)
    return build


def test_g524_2_node_fan_is_served_n4_labeling():
    """The fan widening's labeling test: one above member, four below
    members — the crossing junction labels, every member is exempted in
    `wire_media`, and the scope check passes."""
    s = BSplineSolver(**fan_rise_deck())
    media = s._wire_media()
    assert media == (_medium_spec.BELOW,) * 4 + (_medium_spec.ABOVE,)
    assert s._crossing_junctions() == (0,)
    assert s._grounded_junction_ends() == frozenset(
        [(0, "end"), (1, "end"), (2, "end"), (3, "end"), (4, "end")]
    )


def test_g524_2_two_above_members_refused_by_name():
    """The widening is 1 above × N below ONLY: a second above member
    puts an above-tent × above-tent interface corner on the deck, a pair
    class no adjudicator has measured."""
    build = crossing_deck()
    build["wires"] = build["wires"] + [np.array([(0.0, 0.0, 0.0), (3.0, 0.0, 5.0)])]
    build["n_per_edge_per_wire"] = build["n_per_edge_per_wire"] + [[5]]
    build["junctions"] = [[(0, "end"), (1, "start"), (2, "start")]]
    s = BSplineSolver(**build)
    with pytest.raises(NotImplementedError, match="more than one above member"):
        s._crossing_junctions()


def test_g524_2_above_side_other_junction_refused_by_name():
    build = crossing_deck()
    build["wires"] = [
        build["wires"][0],
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)]),
        np.array([(0.0, 0.0, 5.0), (0.0, 0.0, 10.0)]),
    ]
    build["n_per_edge_per_wire"] = [build["n_per_edge_per_wire"][0], [12], [11]]
    build["junctions"] = [
        [(0, "end"), (1, "start")],
        [(1, "end"), (2, "start")],
    ]
    s = BSplineSolver(**build)
    with pytest.raises(NotImplementedError, match="OTHER junction"):
        s._crossing_junctions()


# The #674 study's per-arm node grading (probe18's geometric walk, at
# the K = 5 node): vertices approach (0,0,0) on the rises from below and
# the monopole from above, MATCHED across the interface, far mesh at
# base.
#
# THE STUDY'S VERDICT WAS RE-DERIVED IN #760 AND DID NOT SURVIVE. It read:
# "the K>2 composition error is the ABOVE tent's interface-adjacent h at
# first order (rise-only grading leaves 0.2214 of the base 0.2269 Ω ε̃ = 1
# residual; mono-only drops it to 0.0171), and matched grading restores
# ~2.6-order convergence". Every number there was measured at a fixed
# n_qp_pair = 4, and on a crossing node the cross-edge quadrature error is
# first order in that knob. Re-run across it:
#
#   - the ε̃ = 1 composition error is 0.0000 Ω at n_qp_pair = 32, on every
#     rung of probe1's uniform ladder. There is no composition error; the
#     "clean first order" ladder was measuring quadrature.
#   - the above/rise asymmetry (0.0171 vs 0.2214) is 0.0002 vs 0.0001 at
#     n_qp_pair = 32. Neither arm dominates.
#
# The GRADING still earns its place here, which is why this deck is
# unchanged: at the shipped default it is worth ~4.5 Ω on the soil-A fan.
# What it is not is a mesh convergence class — at n_qp_pair = 64 the same
# grading is worth 0.08 Ω, less than the 0.12 Ω a far-mesh doubling moves.
_FAN_GRADES = {
    # rung: rise (z-vertices −depth → 0, npe), mono (z-vertices 10 → 0, npe)
    "n2": (
        ([-0.15, -0.05, -0.0125, 0.0], [2, 2, 2]),
        ([10.0, 0.5, 0.05, 0.0125, 0.0], [19, 2, 3, 2]),
    ),
    "n3": (
        ([-0.15, -0.05, -0.0125, -0.0031, 0.0], [2, 2, 2, 2]),
        ([10.0, 0.5, 0.05, 0.0125, 0.0031, 0.0], [19, 2, 3, 2, 2]),
    ),
}

# The soil-A 4-rise fan under matched node grading, converged in the
# QUADRATURE axis as well as the mesh axis (momwire#760).
#
# The #674 study banked 142.1923 - 36.4707j here and called it converged
# on the strength of its dense-near mesh measurements — n2->n3 movement
# 0.0059 ohm, observed order 3.4, Richardson Z* 142.1918 - 36.4771j,
# far-mesh doubling 0.022 ohm, dense-vs-split <= 5e-4. Every one of those
# was taken at a FIXED n_qp_pair=4, so the 0.05 envelope was built from
# two error axes out of three, and the missing one dominates: walking
# cross-edge quadrature at fixed mesh puts q=4 **6.808 ohm** from the
# limit. Mesh convergence at fixed quadrature converges to the wrong
# limit; that is the whole lesson, and #674's Richardson extrapolated
# within the wrong one.
#
# The ladder (numpy fallback past the accelerator's n_qp <= 8 cap, which
# reproduces the C++ path BIT-IDENTICALLY at q=4 and q=8, so it is one
# continuous measurement): q=4 6.808 / q=8 2.556 / q=16 0.698 /
# q=32 0.105 / q=64 0.005 ohm from the limit, with q=128 -> 160 moving
# 0.00001. The error falls as ~C/q with C ~ 33 — FIRST order in the
# number of Gauss points, i.e. Gauss-Legendre has lost its rate on a
# near-singular transmitted kernel, which is why brute order is needed
# and why #762 (lift the cap by tiling) and #760 (a singularity-aware
# rule) both exist.
#
# The old 143.9327 - 26.2135j base-mesh print stays a record, never a
# gate, and now carries the same fixed-q=4 caveat.
FAN_SOIL_A_N2 = 140.9358 - 43.1622j

# Cross-edge quadrature order at which the anchor deck sits 0.005 ohm
# from its limit — 10x inside the gate below. Past the accelerated
# kernel's n_qp <= 8 refusal, so the anchor runs the numpy twin; ~6 s,
# which is what this lane's multi-minute certification budget is for.
FAN_SOIL_A_N2_QP = 64


def fan_rise_deck_graded(rung="n2", **override):
    """`fan_rise_deck` with the #674 matched per-arm node grading spliced
    into the wire polylines. The monopole vertices only subdivide the
    existing 10 → 0 line, so the EX 4,1,7 feed arclength is untouched."""
    (rise_pts, rise_npe), (mono_pts, mono_npe) = _FAN_GRADES[rung]
    build = fan_rise_deck(**override)
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
    build["wires"] = [
        np.array([(5.0 * dx, 5.0 * dy, -0.15)] + [(0.0, 0.0, z) for z in rise_pts])
        for dx, dy in dirs
    ] + [np.array([(0.0, 0.0, z) for z in mono_pts])]
    build["n_per_edge_per_wire"] = [[10] + list(rise_npe) for _ in dirs] + [
        list(mono_npe)
    ]
    return build


def hub_deck(n_radials=4, depth=0.15, **override):
    """The screen's OTHER spelling: one rise carrying the node, N radials
    junction-joined to it at a buried hub (0, 0, −depth). Topologically
    `fan_rise_deck`'s twin, ELECTRICALLY a different structure: the fan's
    N coincident rises are a bundle conductor, not one wire — the two
    spellings' ε̃ = 1 truths sit ~9 Ω apart (probe38/39), so they are
    never gated against each other. The hub's by-parts end terms cancel
    through its own KCL row to the DIGIT (probe39 measured the stripped
    and unstripped solves identical through production)."""
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))[:n_radials]
    wires = [
        np.array([(5.0 * dx, 5.0 * dy, -depth), (0.0, 0.0, -depth)]) for dx, dy in dirs
    ]
    npe = [[10] for _ in dirs]
    rise_i = len(wires)
    wires.append(np.array([(0.0, 0.0, -depth), (0.0, 0.0, 0.0)]))
    npe.append([2])
    mono_i = rise_i + 1
    wires.append(np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)]))
    npe.append([15])
    build = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(i, "end") for i in range(n_radials)] + [(rise_i, "start")],
            [(rise_i, "end"), (mono_i, "end")],
        ],
        feeds=[(mono_i, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    build.update(override)
    return build


def test_g524_2_buried_hub_other_junction_is_served():
    """The below-side interior junction (the buried hub) passes scope:
    the crossing junction is the rise↔monopole node, the hub is an
    allowed wholly-below OTHER junction with its own KCL row."""
    s = BSplineSolver(**hub_deck())
    media = s._wire_media()
    assert media == (_medium_spec.BELOW,) * 5 + (_medium_spec.ABOVE,)
    assert s._crossing_junctions() == (1,)
    assert s._grounded_junction_ends() == frozenset([(4, "end"), (5, "end")])


def test_g524_2_mixed_radii_refused_by_name():
    build = crossing_deck(wire_radius=[0.001, 0.002])
    s = BSplineSolver(**build)
    with pytest.raises(NotImplementedError, match="radius rule"):
        s._crossing_junctions()


# ----------------------------------------------------------------------
# G-524-3 — the designed kernel's own identity pin (cheap, machine class)
# ----------------------------------------------------------------------


def test_g524_2_node_graded_fan_plans_without_the_cross_grid():
    """A crossing deck never builds the transmitted grid — its cross pair
    is the designed DIRECT evaluation — so the θ-floor cost law must not
    refuse it. The probe38 grading ladder found the g2-rung fan refused
    at θ = 0.129° for a grid the path never queries (quadrature nodes
    0.84 mm below the plane)."""
    build = fan_rise_deck()
    build["n_per_edge_per_wire"] = [[20, 6] for _ in range(4)] + [[30]]
    s = BSplineSolver(**build)
    geom = s._build_geometry()
    below = s._below_segments(geom)
    b_idx = np.nonzero(below)[0]
    a_idx = np.nonzero(~below)[0]
    _eps_t, _eps_m, k_p, k_m, _c2, _a_m = s._buried_medium()
    obs_a, _t, _w = s._buried_nodes(geom, a_idx)
    obs_b, _t, _w = s._buried_nodes(geom, b_idx)
    plan = s._buried_serve_plan(geom, a_idx, obs_a, obs_b, k_p, k_m, crossing=True)
    assert "r1_above" in plan and "r1_below" in plan
    assert "r_cross_max" not in plan
    with pytest.raises(ValueError, match="COST law"):
        s._buried_serve_plan(geom, a_idx, obs_a, obs_b, k_p, k_m)


def test_g524_3_triple_memo_is_bit_identical_and_dedups(monkeypatch):
    """momwire#680 U1: `designed_tables` evaluates each exact
    (ρ, z, z′) triple once per call, and a cache hit is the SAME floats
    — bit-identical to the unmemoized loop by construction. A symmetric
    deck's cross mesh repeats triples IEEE-exactly (the 4-radial fan is
    exactly 4.00× duplicated, probe40), which is what this buys.

    Forced onto the numpy walk of the POINT route: the counting
    monkeypatch and the bit-equality assertions are statements about the
    REFERENCE path. The other two routes reach neither — the C++ twin
    routes around `six_point` entirely (gated at 1e-12 relative, never
    bit, in test_near_interface_accel_680) and the column route evaluates
    a whole (ρ, z′) group at once (momwire#895, gated the same way in
    test_near_interface_columns_895). The dedup this pins is upstream of
    all three and identical on all three."""
    monkeypatch.setattr(_near_interface, "_ROUTE", "point")
    monkeypatch.setattr(_near_interface, "_FORCE_NUMPY", True)
    k = 2.0 * np.pi / WL7
    rho = np.array([[0.3, 0.5, 0.3], [0.3, 0.5, 0.3]])
    z = np.array([[0.2], [0.4]]) * np.ones((1, 3))
    zp = -0.15
    calls = []
    real = _near_interface.six_point

    def counting(eps_t, k2, r, zz, zzp, **kw):
        calls.append((r, zz, zzp))
        return real(eps_t, k2, r, zz, zzp, **kw)

    monkeypatch.setattr(_near_interface, "six_point", counting)
    tables = _near_interface.designed_tables(4.0 - 0.5j, k, rho, z, zp, rtol=1e-8)
    # 6 mesh cells but 4 unique triples per z-row × 2 rows = 4 evaluations
    # per row: (0.3, z) and (0.5, z) each once.
    assert len(calls) == 4
    assert len(set(calls)) == 4
    for i, r in ((0, 0.3), (1, 0.5), (2, 0.3)):
        for row, zz in ((0, 0.2), (1, 0.4)):
            ref = real(4.0 - 0.5j, k, r, zz, zp, rtol=1e-8)
            for kk, key_name in enumerate(_near_interface.KEYS):
                got = tables[key_name][row, i]
                assert got == ref[kk], (key_name, row, i)
    # The duplicated columns are the SAME floats, not merely close.
    for key_name in _near_interface.KEYS:
        assert np.array_equal(tables[key_name][:, 0], tables[key_name][:, 2])


def test_g524_3_eps1_kernel_identity():
    """At ε̃ = 1: U_T = k²V_T = e^{−jkR}/R exactly and W ≡ ∂zW ≡ 0 —
    the transmitted family collapses to the free-space kernel (pinned at
    2.2e-16 by the derivation's probe ledger, gated at 1e-12 here)."""
    k = 2.0 * np.pi / WL7
    for rho, z, zp in ((A_WIRE, 0.0, 0.0), (0.3, 0.2, -0.4), (0.0, 1.0, -0.5)):
        six = _near_interface.six_point(1.0, k, rho, z, zp, rtol=1e-12)
        R = np.hypot(rho, z - zp)
        g = np.exp(-1j * k * R) / R
        assert abs(six[0] - g) <= 1e-12 * abs(g)
        assert abs(k * k * six[1] - g) <= 1e-12 * abs(g)
        assert abs(six[2]) <= 1e-12 * abs(g)
        assert abs(six[3]) <= 1e-12 * abs(g) / max(R, A_WIRE)


# ----------------------------------------------------------------------
# G-524-4 / G-524-5 — the serve gates (slow lane)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_g524_4_soil_a_crossing_anchor(record_property, monkeypatch):
    """The adjudicated soil-A crossing answer, against probe27's number
    re-derived at converged quadrature (momwire#760). The envelope covers
    the g1<->g2 mesh movement (0.021 Ω) and the quadrature residual
    (0.0003 Ω at the order used here), NOT an engine comparison — the
    engine's 74.761 − 57.730j crossing print is a different experiment
    (see the module docstring) and is deliberately absent here.

    Runs on the numpy twin to get past the accelerator's n_qp <= 8 cap,
    as `test_g674_2_soil_a_fan_anchor` does and for the same reason; the
    two paths are bit-identical where both can run."""
    for flag in (n for n in dir(_bspline_kernels) if n.startswith("_HAVE_")):
        monkeypatch.setattr(_bspline_kernels, flag, False)
    z, _ = BSplineSolver(
        **crossing_deck(1), n_qp_pair=CROSSING_G1_QP
    ).compute_impedance()
    record_property("momwire_Z", f"{z:.4f}")
    record_property("banked_Z", f"{CROSSING_G1:.4f}")
    record_property("n_qp_pair", str(CROSSING_G1_QP))
    assert abs(z - CROSSING_G1) <= 0.05, (
        f"crossing serve answers {z:.4f} on the g1 adjudication deck at "
        f"n_qp_pair={CROSSING_G1_QP} where the banked exact-EM answer is "
        f"{CROSSING_G1:.4f} — {abs(z - CROSSING_G1):.4f} ohm apart (mesh "
        "envelope 0.021 ohm, quadrature 0.0003; NEVER re-gate this "
        "against the engine's crossing print)"
    )


@pytest.mark.slow
@pytest.mark.crossgate
def test_g524_6_rise_deck_eps1_collapse(record_property):
    """The P3 rise class through the same adjudicator: an above wire
    ENDING at the node (σ_aσ_b = +1, the orientation-carried corner
    sign's other branch) joined to a BENT below wire (15 cm rise + 5 m
    horizontal radial). At ε̃ = 1 the deck is one bent free-space wire
    solved independently by the shipped fill (measured 0.0019 Ω apart).
    This is the gate that catches an orientation-blind corner — that
    bug wrecked this deck to 10−1007j while leaving every
    starts-at-node deck untouched."""
    pts = np.array(
        [(5.0, 0.0, -0.15), (0.0, 0.0, -0.15), (0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]
    )
    z_truth, _ = BSplineSolver(
        wires=[pts],
        n_per_edge_per_wire=[[10, 2, 15]],
        feeds=[(0, 5.0 + 0.15 + 10.0 - 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        n_qp_pair=_COLLAPSE_N_QP,
    ).compute_impedance()
    rise = np.array([(5.0, 0.0, -0.15), (0.0, 0.0, -0.15), (0.0, 0.0, 0.0)])
    mono = np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)])
    z, _ = BSplineSolver(
        wires=[rise, mono],
        n_per_edge_per_wire=[[10, 2], [15]],
        junctions=[[(0, "end"), (1, "end")]],
        feeds=[(1, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=(1.0, 0.0),
        ground_model="sommerfeld",
        n_qp_pair=_COLLAPSE_N_QP,
    ).compute_impedance()
    record_property("momwire_Z", f"{z:.4f}")
    record_property("free_space_truth", f"{z_truth:.4f}")
    assert abs(z - z_truth) <= 0.05, (
        f"the rise-deck ε̃ = 1 solve answers {z:.4f} where the free-space "
        f"bent-wire truth is {z_truth:.4f} — {abs(z - z_truth):.4f} ohm "
        "apart; check the corner's orientation sign first"
    )


@pytest.mark.slow
@pytest.mark.crossgate
def test_g524_5_eps1_collapse_reproduces_free_space(record_property):
    """probe29's adjudicator through the production path: at ε̃ = 1 the
    interface vanishes and the crossing deck IS a free-space 12 m wire,
    solved independently by the shipped free-space fill. The corner the
    composition must telescope through is ~204,000 in magnitude; passing
    at the 0.05 Ω class is the arithmetic being RIGHT where truth is
    known (measured 0.0124 Ω, 0.002 %)."""
    g = _GRADES[1]
    pts = [(0.0, 0.0, z) for z in g["below"][0] + [0.0] + g["above"][0]]
    z_truth, _ = BSplineSolver(
        wires=[np.array(pts)],
        n_per_edge_per_wire=[g["below"][1] + g["above"][1]],
        feeds=[(0, 2.0 + 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        n_qp_pair=_COLLAPSE_N_QP,
    ).compute_impedance()
    z, _ = BSplineSolver(
        **crossing_deck(1, ground_eps=(1.0, 0.0), n_qp_pair=_COLLAPSE_N_QP)
    ).compute_impedance()
    record_property("momwire_Z", f"{z:.4f}")
    record_property("free_space_truth", f"{z_truth:.4f}")
    assert abs(z - z_truth) <= 0.05, (
        f"the ε̃ = 1 crossing solve answers {z:.4f} where the free-space "
        f"single-wire truth is {z_truth:.4f} — {abs(z - z_truth):.4f} ohm "
        "apart; the complete composition no longer collapses"
    )


@pytest.mark.slow
@pytest.mark.crossgate
def test_g524_7_fan_eps1_collapse(record_property):
    """The fan widening's adjudicator (probe38): at ε̃ = 1 the 4-rise fan
    deck IS a free-space 5-wire junction deck, solved independently by
    the native junction machinery (KCL row, shipped free-space fill).

    The composition is NOT ε̃=1-exact past K = 2: the residual is a
    measured CONVERGENCE class in the node mesh, not bookkeeping —
    probe38 banked 0.0043 Ω (N=1) → 0.1327 (N=2) → 0.2269 (N=4) on this
    mesh, shrinking 0.2269 → 0.1487 → 0.1060 down the node-grading
    ladder with no plateau (a corner-sign error would miss by the ~1e5
    corner magnitude instead — the −1000j class). The gate holds the
    measured value with CI headroom; tightening it means GRADING the
    node, not touching the corner loops — which is exactly what the
    #674 study did: `test_g674_1` runs this adjudicator on the graded
    rung at a 60× tighter envelope. This BASE-mesh gate stays as the
    ungraded pin (the two miss differently: a corner-loop defect moves
    both, a grading-machinery defect only g674_1)."""
    build = fan_rise_deck(ground_eps=(1.0, 0.0), n_qp_pair=_COLLAPSE_N_QP)
    truth = {
        k: v
        for k, v in build.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    z_truth, _ = BSplineSolver(**truth).compute_impedance()
    z, _ = BSplineSolver(**build).compute_impedance()
    record_property("momwire_Z", f"{z:.4f}")
    record_property("free_space_truth", f"{z_truth:.4f}")
    assert abs(z - z_truth) <= 0.30, (
        f"the ε̃ = 1 fan solve answers {z:.4f} where the free-space "
        f"5-wire junction truth is {z_truth:.4f} — {abs(z - z_truth):.4f} "
        "ohm apart (measured 0.2268 on this mesh through #692's near "
        "density, a node-mesh convergence class); a jump past this "
        "envelope is bookkeeping, not convergence"
    )


# ----------------------------------------------------------------------
# G-674 — the K>2 node's convergence study, banked (momwire#674)
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.crossgate
def test_g674_1_graded_fan_eps1_collapse(record_property):
    """The #674 resolution of g524_7's 0.30-Ω caveat: matched per-arm
    node grading (the n2 rung) collapses the K = 5 composition residual
    to the MEASUREMENT FLOOR — 0.0001 Ω against the independently-solved
    free-space truth (the admissibility split itself sits ~2e-4 from
    dense on this mm-graded deck). The gate's 0.005 envelope is 50× the
    measured value and still 60× tighter than g524_7 — a miss here with
    g524_7 green means the grading machinery (vertex splicing, short-
    segment quadrature), not the corner loops."""
    build = fan_rise_deck_graded("n2", ground_eps=(1.0, 0.0), n_qp_pair=_COLLAPSE_N_QP)
    truth = {
        k: v
        for k, v in build.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    z_truth, _ = BSplineSolver(**truth).compute_impedance()
    z, _ = BSplineSolver(**build).compute_impedance()
    record_property("momwire_Z", f"{z:.4f}")
    record_property("free_space_truth", f"{z_truth:.4f}")
    assert abs(z - z_truth) <= 0.005, (
        f"the n2-graded ε̃ = 1 fan solve answers {z:.4f} where the "
        f"free-space truth is {z_truth:.4f} — {abs(z - z_truth):.4f} ohm "
        "apart (measured 0.0001 on this rung); the graded composition no "
        "longer collapses"
    )


@pytest.mark.slow
@pytest.mark.crossgate
def test_g674_2_soil_a_fan_anchor(record_property, monkeypatch):
    """The soil-A fan anchor, converged in all three axes: the n2-graded
    deck's 140.9358 - 43.1622j at cross-edge quadrature 64, where it sits
    0.005 ohm from its own limit against the 0.05 gate — the node axis
    (n2->n3 0.0059 ohm) and far-mesh axis (0.022 ohm) the #674 study
    measured, plus the quadrature axis that study held fixed at 4 and
    never measured (momwire#760).

    Running at 64 needs the numpy twin: the accelerated kernel refuses
    n_qp > 8 (momwire#762, a cache-blocking constant rather than a real
    limit). Bypassing it is the suite's established idiom and is sound
    here for a reason worth stating — the two paths agree BIT-IDENTICALLY
    at q=4 and q=8 on this very deck, so the twin is an oracle, not an
    approximation.

    What this must NOT be re-gated against, both of which are records:
    #674's own 142.1923 - 36.4707j (6.8 ohm away — it is the q=4 print,
    not a converged answer), and the engine's detached-stake
    90.051 - 70.731j (a different experiment, module docstring)."""
    for flag in (n for n in dir(_bspline_kernels) if n.startswith("_HAVE_")):
        monkeypatch.setattr(_bspline_kernels, flag, False)
    z, _ = BSplineSolver(
        **fan_rise_deck_graded("n2"), n_qp_pair=FAN_SOIL_A_N2_QP
    ).compute_impedance()
    record_property("momwire_Z", f"{z:.4f}")
    record_property("banked_Z", f"{FAN_SOIL_A_N2:.4f}")
    record_property("n_qp_pair", str(FAN_SOIL_A_N2_QP))
    assert abs(z - FAN_SOIL_A_N2) <= 0.05, (
        f"the n2-graded soil-A fan answers {z:.4f} at n_qp_pair="
        f"{FAN_SOIL_A_N2_QP} where the banked converged answer is "
        f"{FAN_SOIL_A_N2:.4f} — {abs(z - FAN_SOIL_A_N2):.4f} ohm apart "
        "(node axis 0.0059, far-mesh 0.022, quadrature 0.005; NEVER "
        "re-gate against #674's q=4 print or the engine print)"
    )


# ----------------------------------------------------------------------
# G-698 — the crossing exemption is EARNED, not granted by geometry
#
# `_grounded_junction_ends` silences `_medium_spec.wire_media`'s
# contact+buried refusal for every junction whose shared point sits in the
# plane. momwire#698: a junction that does not actually CROSS still got
# that silence, and `_crossing_junctions` then declined the deck, so it
# fell through to the OLD field-form transmitted block and was SERVED —
# the contact basis's O(1) boundary term unaccounted for, which is the one
# configuration the refusal exists to prevent. Measured on main @ 202e0f6:
# the one-member spelling answered 86.9322 − 23.8620j, 46.57 Ω from the
# engine's 92.130 − 70.141j and worse than the refusal prose's then-quoted
# 13–17 Ω best-consistent-spelling gap. (Both figures predate the
# momwire#706 feed correction — historical record, direction unaffected:
# the shipped cell stays ~51 Ω wrong at the matched feed.)
# ----------------------------------------------------------------------


_CONTACT_SENTENCE = re.escape("stands an END in the ground plane (ground CONTACT)")


def test_g698_one_member_junction_at_the_contact_end_still_refuses():
    """The issue's repro: a one-member group declared at the contact end
    of the lone-radial anchor. One wire end cannot join two media, so it
    can never be the crossing junction the exemption is for."""
    build = dict(contact_deck())
    build["junctions"] = [[(0, "end")]]
    s = BSplineSolver(**build)
    with pytest.raises(ValueError, match=_CONTACT_SENTENCE):
        s.compute_impedance()


def test_g698_the_one_member_bypass_refuses_at_the_media_reading():
    """WHERE it refuses is the fix's design decision: the member-count
    test is a pure-geometry NECESSARY condition, so it lives in
    `_grounded_junction_ends` and the deck refuses at exactly the point
    the undeclared deck does — no fill is planned, no grid is built."""
    build = dict(contact_deck())
    build["junctions"] = [[(0, "end")]]
    s = BSplineSolver(**build)
    assert s._grounded_junction_ends() == frozenset()
    with pytest.raises(ValueError, match=_CONTACT_SENTENCE):
        s._wire_media()


@pytest.mark.parametrize("junctions", [None, [[(0, "start")]], [[(0, "end")]]])
def test_g698_the_controls_all_refuse_the_same_way(junctions):
    """The repro and the two controls the issue recorded beside it, in one
    row each and refusing with ONE sentence: the undeclared deck and the
    one-member group at the deck's OTHER (top, out-of-plane) end were
    already refused on main, and the contact-end spelling — the bypass —
    now joins them instead of answering."""
    build = dict(contact_deck())
    if junctions is not None:
        build["junctions"] = junctions
    with pytest.raises(ValueError, match=_CONTACT_SENTENCE):
        BSplineSolver(**build)._wire_media()


def test_g698_a_one_member_junction_off_the_plane_stays_legal():
    """momwire#172 is not narrowed: only the crossing-exemption EFFECT of
    a one-member group changes. A lone group at a free end of an
    all-above deck still solves (its KCL row pins I_end = 0, which is
    numerically the free end it already was)."""
    s = BSplineSolver(
        wires=[np.array([(0.0, 0.0, 1.0), (0.0, 0.0, 11.0)])],
        n_per_edge_per_wire=[[15]],
        junctions=[[(0, "end")]],
        feeds=[(0, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    assert s._wire_media() == (_medium_spec.ABOVE,)
    z, _ = s.compute_impedance()
    assert z.real > 0


def _stranded_grounded_junction_deck():
    """Two ABOVE wires meeting at a shared point IN the plane, plus a
    detached buried radial: a 2-member grounded junction that cannot
    cross (both members above), so the geometric exemption is granted and
    never earned. The whole class momwire#698's belt-and-braces closes —
    the member count alone would let this one through."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]),
            np.array([(0.0, 0.0, 0.0), (5.0, 0.0, 3.0)]),
            np.array([(1.0, 0.0, -0.15), (6.0, 0.0, -0.15)]),
        ],
        n_per_edge_per_wire=[[15], [8], [10]],
        junctions=[[(0, "start"), (1, "start")]],
        feeds=[(0, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def test_g698_an_all_above_grounded_junction_does_not_earn_the_exemption():
    """The belt-and-braces (2): the labels say both members are ABOVE, so
    `_crossing_junctions` never validates the junction and the contact
    ends it exempted are stranded on the field-form block. Refused with
    the contact+buried sentence rather than served.

    This one refuses LATER than the repro above — at the first
    `_crossing_junctions` call, which the buried fill makes before it
    plans any grid — because "does this junction cross" is a MEDIA
    question and the exemption set is built before the labels exist."""
    s = BSplineSolver(**_stranded_grounded_junction_deck())
    assert s._grounded_junction_ends() == frozenset([(0, "start"), (1, "start")])
    assert s._wire_media() == (
        _medium_spec.ABOVE,
        _medium_spec.ABOVE,
        _medium_spec.BELOW,
    )
    with pytest.raises(ValueError, match=_CONTACT_SENTENCE):
        s._crossing_junctions()
    with pytest.raises(ValueError, match=_CONTACT_SENTENCE):
        s.compute_impedance()


def test_g698_the_served_crossing_decks_still_earn_their_exemption():
    """The audit must not fire on the decks it shares a function with:
    every contact end of the served crossing and fan spellings IS a
    member of a validated crossing junction."""
    assert BSplineSolver(**crossing_deck())._crossing_junctions() == (0,)
    assert BSplineSolver(**fan_rise_deck())._crossing_junctions() == (0,)
    assert BSplineSolver(**hub_deck())._crossing_junctions() == (1,)


# ======================================================================
# G-696 — the node-mesh advisory: say when a crossing node's REGION is
# unresolved for the #674 class, and stay quiet when it is not
# ======================================================================
#
# The physics is measured in the G-674 gates; what is gated HERE is the
# reporting. The one design claim worth pinning is the choice of
# quantity: the advisory reads the FINEST mesh within reach of the node,
# not the segment touching it, because the touching segment does not
# predict the error (a feed gap in front of a graded chain is a resolved
# node). See `_crossing_fill`'s constants for the two-deck calibration.
#
# These run without solving: the advisory reads geometry only, and the
# decks it reads are the ones the G-674 gates solve.


def _arms(build):
    s = BSplineSolver(**build)
    return s._crossing_node_members(s._crossing_junctions(), s._wire_media())


def gap_then_graded_deck(**override):
    """The shape that broke the first spelling of this advisory: a 50 mm
    ungraded feed gap at the node, with a chain graded to 6.25 mm right
    behind it. Transcribed from antennaknobs' buried-radial vertical in
    its FIXED state, whose soil-A answer moves 0.115 ohm across an 8x
    sweep of that gap — a resolved node wearing a coarse first segment."""
    build = fan_rise_deck_graded("n2", **override)
    mono_i = len(build["wires"]) - 1
    build["wires"] = list(build["wires"])
    build["n_per_edge_per_wire"] = [list(n) for n in build["n_per_edge_per_wire"]]
    # Only the ABOVE arm differs from the banked graded rung: its grading
    # now sits BEHIND a 50 mm ungraded gap at the node instead of running
    # all the way in. The rises stay at the n2 rung, as they are on the
    # antennaknobs deck this is transcribed from.
    build["wires"][mono_i] = np.array(
        [(0.0, 0.0, 10.0), (0.0, 0.0, 0.15), (0.0, 0.0, 0.05), (0.0, 0.0, 0.0)]
    )
    build["n_per_edge_per_wire"][mono_i] = [15, 16, 1]
    return build


def test_g696_1_the_gated_quantity_is_the_resolved_scale_not_the_touching_segment():
    """The design claim. On the gap-then-graded arm the touching segment
    is 50 mm while the node region resolves to 6.25 mm — an 8x spread
    inside ONE arm. Gating the touching segment would fire here; gating
    the resolved scale does not."""
    above = [a for a in _arms(gap_then_graded_deck()) if a.side == "above"]
    assert len(above) == 1
    assert above[0].h_adjacent == pytest.approx(0.050, rel=1e-12)
    assert above[0].h_resolved == pytest.approx(0.00625, rel=1e-12)


def test_g696_2_every_crossing_member_is_reported_once():
    """One above member and N below members, per the fan widening's own
    scope — the advisory sees the whole node, not just the above arm."""
    arms = _arms(fan_rise_deck(n_radials=4))
    assert len(arms) == 5
    assert sorted(a.side for a in arms) == ["above"] + ["below"] * 4


def test_g696_3_the_base_fan_warns_and_names_the_above_arm():
    """The coarse anchor. Its above arm is a single 667 mm edge that never
    resolves, so it must warn and name that arm.

    #674 measured this deck at 0.2269 ohm (eps~=1) and 7.48 ohm of soil-A
    mesh move; #760 re-derived both across the quadrature axis that study
    held at n_qp_pair=4 and neither survives as a MESH number — the eps~=1
    residual is 0.0000 ohm at n_qp_pair=32. The deck still warns, and should:
    at the shipped order the node is worth ~4.5 ohm. What changed is the
    reason, which is why the message now cites #760 as well."""
    with pytest.warns(_crossing_fill.CoarseCrossingNode) as rec:
        worst = _crossing_fill.warn_coarse_node(_arms(fan_rise_deck()))
    assert worst.h_resolved == pytest.approx(10.0 / 15.0, rel=1e-12)
    assert worst.side == "above"
    msg = str(rec[0].message)
    assert "666.7 mm" in msg and "above member" in msg
    assert "momwire#674" in msg and "momwire#696" in msg
    # The re-derivation is part of the claim now, not a footnote to it.
    assert "momwire#760" in msg
    # WITHDRAWN by #760: the message must no longer attribute a dominant arm.
    assert "dominant term" not in msg


@pytest.mark.parametrize(
    "deck,why",
    [
        (fan_rise_deck_graded, "the banked FAN_SOIL_A_N2 rung, 0.0001 ohm off"),
        (gap_then_graded_deck, "a feed gap in front of a graded chain"),
    ],
)
def test_g696_4_resolved_nodes_are_silent(deck, why):
    """Both converged shapes stay quiet. The second is the regression
    that the first spelling of this advisory got wrong: it gated the
    touching segment, fired on a deck measured fine, and needed its bar
    threaded through a 50-75 mm window to avoid doing so."""
    build = deck() if deck is gap_then_graded_deck else deck("n2")
    arms = _arms(build)
    assert max(a.h_resolved for a in arms) == pytest.approx(0.00625, rel=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", _crossing_fill.CoarseCrossingNode)
        assert _crossing_fill.warn_coarse_node(arms) is not None, why


def test_g696_5_the_bar_sits_on_674s_own_graded_rungs():
    """Not a number split between two anchors: the bar IS #674's coarsest
    graded rung (25 mm, 0.0036 ohm at eps~=1) and 4x its converged recipe
    rung (6.25 mm, 0.0001 ohm)."""
    assert _crossing_fill.NODE_H_BAR == 0.025
    assert _crossing_fill.NODE_H_BAR == pytest.approx(4 * 0.00625, rel=1e-12)


@pytest.mark.parametrize("reach", [0.06, 0.15, 0.5, 1.0])
def test_g696_6_the_verdict_is_insensitive_to_the_reach(reach, monkeypatch):
    """The reason for gating the resolved scale rather than threading a
    needle: over a 17x span of the reach parameter, both calibration
    decks land on the same side every time. The earlier spelling flipped
    on a 20% move of its threshold."""
    monkeypatch.setattr(_crossing_fill, "NODE_REACH", reach)
    coarse = max(a.h_resolved for a in _arms(fan_rise_deck()))
    fine = max(a.h_resolved for a in _arms(gap_then_graded_deck()))
    assert coarse > _crossing_fill.NODE_H_BAR
    assert fine <= _crossing_fill.NODE_H_BAR


def test_g696_7_a_single_coarse_edge_cannot_escape_by_being_longer_than_the_reach():
    """The first edge always counts however long it is. The base fan's
    above arm is one 667 mm edge — longer than the reach — and must not
    read as 'nothing measured' and slip through."""
    (above,) = [a for a in _arms(fan_rise_deck()) if a.side == "above"]
    assert above.h_resolved == pytest.approx(10.0 / 15.0, rel=1e-12)
    assert _crossing_fill.NODE_REACH < 10.0 / 15.0


def test_g696_8_the_advisory_never_refuses_and_never_remeshes():
    """Advisory ONLY: the deck is the deck. A coarse node is a legitimate
    thing to ask for — every rung of a convergence ladder but the last is
    one — so this path must not raise, and must not touch the mesh."""
    s = BSplineSolver(**fan_rise_deck())
    before = [list(npe) for npe in s.n_per_edge_per_wire]
    with pytest.warns(_crossing_fill.CoarseCrossingNode):
        _crossing_fill.warn_coarse_node(
            s._crossing_node_members(s._crossing_junctions(), s._wire_media())
        )
    assert [list(npe) for npe in s.n_per_edge_per_wire] == before


def test_g696_9_an_empty_deck_reports_nothing_and_stays_quiet():
    """No crossing junction, no advice — and `None` back rather than a
    raise, so a diagnostic caller can read it unconditionally."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _crossing_fill.warn_coarse_node([]) is None
