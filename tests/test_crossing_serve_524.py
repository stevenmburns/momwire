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
  interface-graded meshes (138.7671−102.9889j → 138.7691−102.9893j);
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
RECORDS, but the GRADED fan is converged and banked: `FAN_SOIL_A_N2`
with its G-674 gates. The ε̃ = 1 collapses run in the merge-to-main
`crossgate` lane (multi-minute certification solves, the memgate
reasoning) while the PR slow lane keeps `test_g524_4` as the per-PR
crossing regression pin. The HUB spelling's collapse is a banked
record (0.2194 Ω, probe38) with no gate of its own: its only unique
content — the hub-tent by-parts terms — was measured to contribute
exactly zero to the digit (probe39), and everything else it would
exercise, `test_g524_7` already does.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _medium_spec, _near_interface
from momwire.bspline import BSplineSolver

from test_buried_serve_553 import SOIL_A, WL7

A_WIRE = 0.001

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

# Banked by probe27 (session 5) and re-measured through this production
# path: the soil-A exact-EM crossing answer, mesh-stable g1 -> g2.
# Through #692's near-density default the g1 production print is
# 138.7670 − 102.9889j (1e-4 from this adjudicated bank, envelope 0.05);
# g2 is unchanged to the digit. The bank stays the dense-quadrature
# adjudication.
CROSSING_G1 = 138.7671 - 102.9889j
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
    Feed = arclength 4.3333 on the 10 → 0 monopole (EX 4,1,7 — the trap:
    an improvised feed at 10 − 4.333 is silently ~50 Ω wrong)."""
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
# base. The study's verdict (scratch/674-study, probes 1-3): the K>2
# composition error is the ABOVE tent's interface-adjacent h at first
# order (rise-only grading leaves 0.2214 of the base 0.2269 Ω ε̃ = 1
# residual; mono-only drops it to 0.0171), and matched grading restores
# ~2.6-order convergence — n2 collapses the residual to 0.0001 Ω for a
# base-mesh solve cost.
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

# Banked by the #674 study (probe3, split lane = the production path)
# and RE-BANKED through #692's near-density default (moved 4e-4): the
# soil-A 4-rise fan CONVERGED under matched node grading. The #674
# study's dense-near measurements — n2→n3 movement 0.0059 Ω, observed
# order 3.4, Richardson Z* 142.1918 − 36.4771j, far-mesh doubling
# 0.022 Ω, dense-vs-split ≤ 5e-4 — stand as the convergence evidence
# (post-flip n2→n3 is 0.0060). The base-mesh 143.9327 − 26.2135j record
# stands in probe38's JSON as the unconverged-mesh print, never a gate.
FAN_SOIL_A_N2 = 142.1923 - 36.4707j


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

    Forced onto the numpy walk: the counting monkeypatch and the
    bit-equality assertions are statements about the REFERENCE path (the
    C++ twin routes around `six_point` entirely and is gated at 1e-12
    relative, never bit, in test_near_interface_accel_680)."""
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
def test_g524_4_soil_a_crossing_anchor(record_property):
    """The adjudicated soil-A crossing answer through the production path,
    against probe27's banked number. The envelope is the g1↔g2 mesh
    movement (0.021 Ω), NOT an engine comparison — the engine's
    74.761 − 57.730j crossing print is a different experiment (see the
    module docstring) and is deliberately absent here."""
    z, _ = BSplineSolver(**crossing_deck(1)).compute_impedance()
    record_property("momwire_Z", f"{z:.4f}")
    record_property("banked_Z", f"{CROSSING_G1:.4f}")
    assert abs(z - CROSSING_G1) <= 0.05, (
        f"crossing serve answers {z:.4f} on the g1 adjudication deck where "
        f"the banked exact-EM answer is {CROSSING_G1:.4f} — "
        f"{abs(z - CROSSING_G1):.4f} ohm apart (mesh envelope 0.021 ohm; "
        "NEVER re-gate this against the engine's crossing print)"
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
    ).compute_impedance()
    z, _ = BSplineSolver(**crossing_deck(1, ground_eps=(1.0, 0.0))).compute_impedance()
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
    build = fan_rise_deck(ground_eps=(1.0, 0.0))
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
    build = fan_rise_deck_graded("n2", ground_eps=(1.0, 0.0))
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
def test_g674_2_soil_a_fan_anchor(record_property):
    """The soil-A fan anchor the #674 study earned: the n2-graded deck's
    142.1922 − 36.4711j, CONVERGED in the node axis (n2→n3 movement
    0.0059 Ω at observed order 3.4; Richardson Z* 142.1918 − 36.4771j)
    with the far-mesh doubling worth 0.022 Ω. The 0.05 envelope covers
    both axes with the session-7 headroom. probe38's base-mesh
    143.9327 − 26.2135j stands as the unconverged-mesh record — 12.6 Ω
    away, all of it the node class the grading removed. NEVER re-gate
    this against the engine's detached-stake 90.051 − 70.731j (a
    different experiment, module docstring)."""
    z, _ = BSplineSolver(**fan_rise_deck_graded("n2")).compute_impedance()
    record_property("momwire_Z", f"{z:.4f}")
    record_property("banked_Z", f"{FAN_SOIL_A_N2:.4f}")
    assert abs(z - FAN_SOIL_A_N2) <= 0.05, (
        f"the n2-graded soil-A fan answers {z:.4f} where the banked "
        f"converged answer is {FAN_SOIL_A_N2:.4f} — "
        f"{abs(z - FAN_SOIL_A_N2):.4f} ohm apart (node axis 0.0059, "
        "far-mesh axis 0.022; NEVER re-gate against the engine print)"
    )
