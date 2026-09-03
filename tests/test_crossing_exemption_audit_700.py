"""The crossing exemption, audited by call site — momwire#700.

`_medium_spec.wire_media` refuses a ground CONTACT end on a deck that also
carries a buried wire. A crossing junction is EXEMPT from that refusal,
because the crossing fill is what carries the contact end's current into the
lower medium. This file is the audit of who grants that exemption, what each
grant skips, and what it is keyed on.

THE CALL-SITE TABLE
-------------------

=====================================  ==================================  =========================
site                                   what the exemption skips there      keyed on
=====================================  ==================================  =========================
`_medium_spec.wire_media`              the contact-with-buried refusal,    GEOMETRY, via
(`crossing_ends=`)                     and the mid-span crossing refusal   `grounded_crossing_exemption`
                                       for the below wire's plane-touching
                                       anchor
`_crossing_junctions` (both trunks)    nothing — this is where the         MEDIA: the labels plus
                                       exemption is GIVEN BACK by any      the validated crossing
                                       grounded junction that did not      junction
                                       turn out to cross (momwire#698)
`BSplineSolver._buried_serve_plan`     the CROSS-MEDIUM section only:      the DECLARED crossing
(`crossing=`)                          the transmitted grid's range cap,   junction, and only with
                                       depth cap and grazing floor. The    an above segment present
                                       below/below R1 cap and theta floor
                                       are asked before it and always run
`RazorSolver._assemble_Z_below_plane`  the points within `_JUNCTION_TOL`   the DECLARED crossing
(`plan_skip=`)                         of a crossing node, from the        node's coordinates; the
                                       below/below R1 + theta scan         ordinary below-plane
                                                                           dispatch passes None
`RazorSolver._find_junctions`          the crossing group's `grounded`     MEDIA, via
(the grounded-flag demotion)           flag: no contact tent, no ghost     `_crossing_junctions`
                                       wing, no plane-reference T2 drop
`BSplineSolver._grounded_junctions`    the junction's KCL row, dropped     GEOMETRY ONLY, and NOT
                                       outright                            narrowed by anything
=====================================  ==================================  =========================

WHAT momwire#700 FOUND
----------------------

The last row is keyed on pure geometry — "the junction's shared point is in
the plane" — with no media question asked and no #698 narrowing applied. On a
deck with an above member that is right: the crossing fill owns the node. On a
WHOLLY-BELOW deck it was not, and nothing else caught it:

* `grounded_crossing_exemption` granted the exemption on member count and
  plane contact alone, so `wire_media` went silent;
* `_crossing_junctions` — the one place the exemption is given back — was
  never called, because BOTH of its call sites in the B-spline buried fill sat
  behind `a_idx.size` (the deck has an above segment) and a wholly-below deck
  short-circuits them;
* `_grounded_junctions` then dropped the node's KCL row on a deck with no
  contact tent, no image continuation and no crossing block.

Measured on the repro below before the fix: `_crossing_junctions()` RAISED the
contact-with-buried refusal while `compute_impedance()` on the same solver
returned 104.53 - 30.36j ohm — the audit answering differently by call site,
in one solver, on one deck. 99.3 % of the current arriving at the node left
the model there, against 5e-17 for the identical topology half a metre lower.
Razor never had the defect: it asks `_crossing_junctions` unconditionally at
construction, which is exactly the asymmetry momwire#700's title names.

THE PHYSICAL STATEMENT
----------------------

The exemption means one thing: **the node's own plane-touching sample is a
sampling artefact of a point the CROSSING block owns, never a grazing pair.**
A point in the interface has depth 0, so its pair with itself reads
`atan2(0, 0)` = 0 deg — below any floor, from a pair that is not a physical
pair. Where a crossing block owns that point, dropping it from the scan is
bookkeeping and one nanometre lower the same deck fills without complaint.
Where no crossing block owns it, the same point is a wire end delivering
current into a medium the deck has no conductor in, and the refusal is the
answer. That is why every key in the table is either "the DECLARED crossing
node" or "the validated crossing junction", and never "any point at depth 0".
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from momwire import _medium_spec, _sommerfeld_below
from momwire import razor as _razor
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_crossing_serve_524 import (  # noqa: E402
    _stranded_grounded_junction_deck,
    crossing_deck,
    fan_rise_deck,
    fan_rise_deck_graded,
    hub_deck,
)

WL7 = 299792458.0 / 7.0e6
A_WIRE = 1e-3
EPS0 = 8.8541878128e-12
SOIL_A = complex(13.0, -0.005 / (2 * np.pi * 7.0e6 * EPS0))

_CROSSING_SENTENCE = "crosses the ground interface mid-span"
_CONTACT_SENTENCE = "that COMBINATION is not served"


def wholly_below_inplane_deck(node_z=0.0, **override):
    """momwire#700's repro: two BELOW wires meeting at a shared anchor.

    With `node_z = 0` the shared anchor is IN the plane, so the junction is
    grounded on pure geometry and has two members — every condition the
    exemption used to ask. Nothing is above the interface, so no member can
    cross and the exemption is not earned.

    With `node_z < 0` it is the same topology as an ordinary buried hub, and
    that is the control: same wires, same mesh, junction off the plane.

    The mesh is the one every number in the module docstring was measured on
    — 10 per edge on both wires, at `9746ccc` — so those numbers are this
    deck's and not a cousin's.
    """
    build = dict(
        wires=[
            np.array([(0.0, 0.0, -2.0), (0.0, 0.0, node_z)]),
            np.array([(0.0, 0.0, node_z), (5.0, 0.0, -0.15 + node_z)]),
        ],
        n_per_edge_per_wire=[[10], [10]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    build.update(override)
    return build


# ======================================================================
# Site 1 — `_medium_spec.wire_media(crossing_ends=)`: the exemption itself
# ======================================================================


def test_site1_a_wholly_below_junction_does_not_earn_the_exemption():
    """The necessary condition momwire#700 adds: a grounded junction no
    member of which reaches above the plane cannot span the interface, so it
    grants nothing. Pure geometry, asked before any label exists."""
    s = BSplineSolver(**wholly_below_inplane_deck())
    assert s._grounded_junction_ends() == frozenset()


def test_site1_the_served_spellings_still_earn_it():
    """The condition must not fire on the decks it shares a function with:
    each of these has an above member at the node."""
    assert BSplineSolver(**crossing_deck())._grounded_junction_ends() == frozenset(
        [(0, "end"), (1, "start")]
    )
    assert len(BSplineSolver(**fan_rise_deck())._grounded_junction_ends()) == 5
    assert len(BSplineSolver(**hub_deck())._grounded_junction_ends()) == 2
    # momwire#698's deck keeps its geometric exemption — both members are
    # above the plane, so only the MEDIA audit can take it back. Belt and
    # braces, not belt instead of braces.
    assert BSplineSolver(
        **_stranded_grounded_junction_deck()
    )._grounded_junction_ends() == frozenset([(0, "start"), (1, "start")])


def test_site1_both_trunks_refuse_the_repro_with_one_sentence():
    """momwire#700's headline. The two trunks reach this deck by different
    routes — declared junctions here, detected there — and must answer with
    the same refusal from the same call site."""
    build = wholly_below_inplane_deck()
    with pytest.raises(ValueError, match=_CROSSING_SENTENCE) as b_exc:
        BSplineSolver(**build).compute_impedance()
    with pytest.raises(ValueError, match=_CROSSING_SENTENCE) as r_exc:
        RazorSolver(**build)
    assert str(b_exc.value) == str(r_exc.value)


def test_site1_the_two_trunks_share_one_exemption_function(monkeypatch):
    """Structural, and the reason momwire#700 exists: the DETECTED-vs-declared
    group source is the only thing the two copies may differ by, so there is
    only one copy left."""
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", True)
    build = hub_deck()
    declared = _medium_spec.grounded_crossing_exemption(
        build["wires"], build["ground_z"], build["junctions"]
    )
    r = RazorSolver(**build)
    detected = _medium_spec.grounded_crossing_exemption(
        build["wires"], build["ground_z"], (g["ends"] for g in r._find_junctions())
    )
    assert declared == detected == BSplineSolver(**build)._grounded_junction_ends()


# ======================================================================
# Site 2 — `_crossing_junctions`: where the exemption is given back
# ======================================================================


class _Reached(Exception):
    """Raised from a spy to stop a fill the instant the spied call happens —
    the gate is about REACHABILITY, and the grid behind it costs seconds."""


def test_site2_the_audit_is_reachable_on_a_wholly_below_deck(monkeypatch):
    """The defect was a VALIDATION behind a short-circuit. Both B-spline call
    sites read `a_idx.size and self._crossing_junctions()`, so on a deck with
    no above segment the audit never ran. It runs now — and the real audit
    runs, not just the call: the spy delegates before it stops the fill."""
    calls = {"n": 0}
    real = BSplineSolver._crossing_junctions

    def spy(self):
        out = real(self)
        calls["n"] += 1
        raise _Reached(out)

    s = BSplineSolver(**wholly_below_inplane_deck(node_z=-0.5))
    assert _medium_spec.BELOW in s._wire_media()
    assert not np.any(~s._below_segments(s._build_geometry())), (
        "control deck must be wholly below, or this gate proves nothing"
    )
    monkeypatch.setattr(BSplineSolver, "_crossing_junctions", spy)
    with pytest.raises(_Reached) as exc:
        s.compute_impedance()
    assert calls["n"] == 1
    assert exc.value.args[0] == (), "a wholly-below deck has no crossing junction"


def test_site2_razor_asks_it_at_construction():
    """Razor's side of the asymmetry, pinned so it cannot drift back: the
    audit runs before any fill, from `_refuse_buried_geometry`."""
    with pytest.raises(ValueError, match=_CONTACT_SENTENCE):
        RazorSolver(**_stranded_grounded_junction_deck())


def test_site2_the_media_audit_still_catches_the_all_above_grounded_junction():
    """momwire#698's own deck, unchanged: geometry grants, media takes back."""
    s = BSplineSolver(**_stranded_grounded_junction_deck())
    with pytest.raises(ValueError, match=_CONTACT_SENTENCE):
        s._crossing_junctions()


# ======================================================================
# Site 3 — `_buried_serve_plan(crossing=)`: the CROSS-MEDIUM section only
# ======================================================================


def _below_extents(s):
    """R1_max, theta_min and lambda_m over the quadrature nodes the fill
    queries — `_buried_serve_plan`'s own below/below measurement."""
    geom = s._build_geometry()
    b_idx = np.nonzero(s._below_segments(geom))[0]
    obs_b, _t, _w = s._buried_nodes(geom, b_idx)
    d = s.ground_z - obs_b[:, 2]
    rho = np.hypot(
        obs_b[:, 0][:, None] - obs_b[:, 0][None, :],
        obs_b[:, 1][:, None] - obs_b[:, 1][None, :],
    )
    hh = d[:, None] + d[None, :]
    lam_m = 2.0 * np.pi / abs(s._buried_medium()[3])
    return (
        float(np.hypot(rho, hh).max()),
        float(np.degrees(np.arctan2(hh, rho)).min()),
        lam_m,
        rho,
    )


def test_site3_the_crossing_flag_skips_the_cross_medium_section_only():
    """What `crossing=True` buys, stated as the difference between the two
    plans on one deck: the transmitted grid's three limits go, the
    below/below pair's two stay."""
    s = BSplineSolver(**fan_rise_deck())
    geom = s._build_geometry()
    below = s._below_segments(geom)
    b_idx, a_idx = np.nonzero(below)[0], np.nonzero(~below)[0]
    obs_a, _ta, _wa = s._buried_nodes(geom, a_idx)
    obs_b, _tb, _wb = s._buried_nodes(geom, b_idx)
    _e, _em, k_p, k_m, _c2, _am = s._buried_medium()
    kw = dict(k_p=k_p, k_m=k_m)
    crossing_plan = s._buried_serve_plan(geom, a_idx, obs_a, obs_b, crossing=True, **kw)
    ordinary_plan = s._buried_serve_plan(
        geom, a_idx, obs_a, obs_b, crossing=False, **kw
    )
    cross_keys = {"r_cross_max", "r_cross_min", "zp_min", "zp_max"}
    assert set(crossing_plan) & cross_keys == set()
    assert cross_keys <= set(ordinary_plan)
    # the below/below sizing is identical either way, which is the point
    assert crossing_plan["r1_below"] == ordinary_plan["r1_below"]
    assert crossing_plan["r1_above"] == ordinary_plan["r1_above"]


def test_site3_the_below_below_floor_is_never_exempted():
    """A crossing deck taken past the below/below grazing floor still
    refuses. `crossing=True` does not reach that check — it is asked before
    the early return, over the same nodes, on every buried deck."""
    floor = _sommerfeld_below._SOMM_BELOW_TH_MIN_DEG
    # A SHALLOW fan: radials 9 m out at 1 cm depth. theta_min ~ 0.06 deg,
    # under the floor, while R1_max ~ 18 m stays inside the 2-wavelength cap
    # (20.0 m at soil A) — the cap fires first on any deck made grazing by
    # LENGTH, so the deck has to be made grazing by DEPTH to read the floor.
    build = fan_rise_deck(depth=0.01)
    build["wires"] = [
        np.array([(9.0 * dx, 9.0 * dy, -0.01), (0.0, 0.0, -0.01), (0.0, 0.0, 0.0)])
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1))
    ] + [build["wires"][-1]]
    s = BSplineSolver(**build)
    r1, th_min, lam_m, _rho = _below_extents(s)
    assert th_min < floor, (th_min, floor)
    assert r1 / lam_m < _sommerfeld_below._SOMM_BELOW_R1_CAP_LAMBDA_M, (
        "the R1 cap must NOT be what fires here, or the gate proves the wrong limit"
    )
    assert s._crossing_junctions() == (0,), "the deck must still be a crossing deck"
    with pytest.raises(ValueError, match="grazing"):
        s.compute_impedance()


# ======================================================================
# Site 4 — `RazorSolver._assemble_Z_below_plane(plan_skip=)`
# ======================================================================


def test_site4_the_ordinary_below_dispatch_passes_no_plan_skip(monkeypatch):
    """`plan_skip` is the DECLARED crossing node's coordinates and nothing
    else. A wholly-below deck reaching razor's ordinary below-plane fill must
    arrive with None, or a plane-touching end would be dropped from the
    R1/theta scan with no crossing block owning it.

    Spied by name — momwire#813's gate 9 rule: anything reaching past
    `_refuse_buried_geometry` must call the medium fill BY NAME, because a
    constructor stub that skips the refusal also skips the dispatch flags it
    sets and lands silently on razor's ORDINARY fill.
    """
    monkeypatch.setattr(_razor, "_SERVE_BELOW_PLANE", True)
    seen = []

    def spy(self, geom, prepared, k, omega, *, plan_skip=None):
        seen.append(plan_skip)
        raise _Reached(plan_skip)

    monkeypatch.setattr(RazorSolver, "_assemble_Z_below_plane", spy)
    s = RazorSolver(
        wires=[np.array([(0.0, 0.0, -0.15), (5.0, 0.0, -0.15)])],
        n_per_edge_per_wire=[[6]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    assert s._below_plane and not s._crossing
    with pytest.raises(_Reached):
        s.compute_impedance()
    assert seen == [None]


# ======================================================================
# Site 5 — the grounded-flag demotion, and site 6 — the KCL drop
# ======================================================================


def test_site6_the_kcl_drop_is_pure_geometry_and_the_refusal_is_what_guards_it():
    """The one site the audit does NOT narrow, recorded rather than changed.

    `_grounded_junctions` still answers "the shared point is in the plane"
    with no media question, because on every deck that reaches a fill the
    crossing block owns that node. What keeps a wholly-below deck away from
    it is the refusal at site 1 — so this gate holds both halves together:
    the flag is still set, and the deck still never reaches a fill.
    """
    s = BSplineSolver(**wholly_below_inplane_deck())
    assert s._grounded_junctions() == frozenset([0])
    with pytest.raises(ValueError, match=_CROSSING_SENTENCE):
        s.compute_impedance()


def test_site6_the_control_keeps_its_kcl_row_and_closes():
    """The measurement that made momwire#700's case, kept as a gate: the
    identical topology with the node half a metre lower is an ordinary buried
    hub, keeps its KCL row, and closes current at the node to machine zero.
    Before the fix the in-plane spelling of this deck solved with 99.3 % of
    the node current unaccounted for."""
    s = BSplineSolver(**wholly_below_inplane_deck(node_z=-0.5))
    assert s._grounded_junctions() == frozenset()
    ps = s.compute_port_solution()
    cur = s.currents_at_knots(ps.coeffs[:, 0])
    i_in, i_out = cur[0][-1], cur[1][0]
    scale = max(abs(i_in), abs(i_out))
    assert abs(i_in - i_out) / scale < 1e-12, (i_in, i_out)


# ======================================================================
# The fan-deck margin gate
# ======================================================================


def test_crossing_deck_1_is_blind_to_the_below_below_floor():
    """Named so no future gate is written on the cheap deck: every below/below
    pair on `crossing_deck(1)` is coaxial, so the floor reads 90 deg and the
    deck cannot exercise it at all."""
    s = BSplineSolver(**crossing_deck(1))
    _r1, th_min, _lam, rho = _below_extents(s)
    assert th_min == pytest.approx(90.0)
    assert not np.any(rho > 0), "a non-coaxial below pair would make this deck read"


def test_the_fan_reads_the_floor_and_its_margin_is_soil_and_grading_invariant():
    """The fan-deck margin gate, recorded against the LIVE constants rather
    than a literal — the floor moved 1.0 -> 0.1 deg in momwire#842 and the
    margin claim moved with it.

    The margin is `atan2(2d, span)` of the radial screen: fixed depth, fixed
    extent. Grading the node moves the shallowest node 48x (2.5e-3 m to
    5.2e-5 m across the rungs) and moves theta_min not at all, and soil moves
    lambda_m but cannot move an angle. So "grade it harder and the floor gets
    thin" is not a thing that can happen on this deck.
    """
    floor = _sommerfeld_below._SOMM_BELOW_TH_MIN_DEG
    _r1, th_fan, _lam, rho = _below_extents(BSplineSolver(**fan_rise_deck()))
    assert np.any(rho > 0), "the fan must have non-coaxial below pairs"
    assert th_fan == pytest.approx(1.72418, abs=1e-4)
    assert th_fan > 10.0 * floor, (th_fan, floor)

    shallow = []
    for rung in ("n2", "n3"):
        s = BSplineSolver(**fan_rise_deck_graded(rung=rung))
        geom = s._build_geometry()
        obs, _t, _w = s._buried_nodes(geom, np.nonzero(s._below_segments(geom))[0])
        shallow.append(float((s.ground_z - obs[:, 2]).min()))
        assert _below_extents(s)[1] == pytest.approx(th_fan, abs=1e-9)
    assert shallow[0] / shallow[1] > 3.0, shallow


def test_on_the_fan_it_is_the_r1_cap_that_soil_moves():
    """The other half of the same correction. Soil cannot move an angle, but
    it moves lambda_m, so what a wetter soil walks the fan into is the
    below/below R1 CAP — 2.0x of margin at soil A, refused by eps_r 30 /
    sigma 0.03. Derived from the live constants and the deck's own R1, never
    from a recorded literal."""
    cap_wl = _sommerfeld_below._SOMM_BELOW_R1_CAP_LAMBDA_M
    r1, _th, lam_a, _rho = _below_extents(BSplineSolver(**fan_rise_deck()))
    assert 1.5 < cap_wl / (r1 / lam_a) < 3.0, r1 / lam_a

    wet = complex(30.0, -0.03 / (2 * np.pi * 7.0e6 * EPS0))
    s = BSplineSolver(**fan_rise_deck(ground_eps=wet))
    r1_w, th_w, lam_w, _rho = _below_extents(s)
    assert th_w == pytest.approx(_th, abs=1e-9), "soil must not move the angle"
    assert r1_w / lam_w > cap_wl, (r1_w / lam_w, cap_wl)
    with pytest.raises(ValueError, match="reach"):
        s.compute_impedance()
