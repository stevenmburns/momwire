"""razor-2p's DETACHED buried route — momwire#1149 U1.

A detached deck carries wires above the interface and wires below it, with
no junction in the plane: an elevated vertical over a buried radial screen
(antennaknobs' `verticals.elevated_buried_counterpoise`), or momwire#553's
own serve-gate deck. Nothing conducts across the interface, so razor's
crossing assembly serves it with ZERO crossing tents: each medium's fill on
its own sub-geometry (`_medium_geometry`), plus the trunk's two cross blocks
on razor's path axes (`cross_complete_block` / `_reversed`, `corner=False`),
with nothing chopped. Loading goes on afterwards on the full geometry, as the
wholly-below route applies it.

The gates are REFERENCE-FREE or convergence ones (the 2026-09-22 decision on
momwire#1149). The deck is scoping probe 5's: `crossing_deck(1)` (graded, so
reciprocity is not vacuous — a straight uniform wire is symmetric by
construction) with the above wire lifted by `gap` and the buried wire
lowered by `gap` and moved `dx` sideways, ports at above arclength 4.5 and
buried arclength 1.0, and `xm` multiplying every edge count but each wire's
plane-end one. Measured 2026-09-22 on this route
(`scratch/razor-buried-u1/probe_u1_gates.py`, JSON beside it):

  * **2-port non-reciprocity decays ~4x per doubling**: 1.69e-2 -> 4.18e-3
    -> 1.02e-3 -> 2.26e-4 at x1..x8 (ratios 4.05 / 4.08 / 4.54). The
    crossing deck, for contrast, sits flat at 3.24e-2 (momwire#1149 U2).
  * **eps~ = 1 collapses block by block**: the cross blocks to 7e-14 / 2e-13
    of their own scale at x1 / x2, the same-medium blocks to 4e-19.
  * **Z converges onto bspline's transmitted-grid route**, an independent
    Galerkin spelling that shares no cross-block code: |dZ12| 0.543 ->
    0.287 -> 0.162 -> 0.098 ohm, |dZ11| 26.6 -> 14.7 -> 8.26 -> 4.67,
    |dZ22| 36.1 -> 19.1 -> 10.3 -> 5.77 (each ratio 0.53-0.61 per doubling,
    razor's first-order walk against a converging Galerkin answer).
  * **the stand-off ladder** (gap 0.3 -> 0.01 m, dx 0.5 and 0 — the coaxial
    dx = 0 case puts the two wire ends 2*gap apart across the plane) serves
    at every rung and decays at 3.9-4.1x per doubling at every gap; the
    razor/bspline gap at x4 moves only with the geometry (|dZ12| 0.16 ->
    0.58 at dx = 0 as the ends close to 20 mm).
  * **loading converges onto bspline's**: the copper-skin shift of Z11 is
    0.512+0.492j vs 0.544+0.524j at x1 and 0.507+0.486j vs 0.517+0.497j at
    x4 (gap 0.046 -> 0.025 -> 0.015 ohm).
  * **the catalog counterpoise** (`elevated_buried_counterpoise` at its
    defaults, spelled in momwire kwargs): razor and bspline differ by 5-9 %
    on a -j44 k-ohm driving point, and they differ by the SAME amount with
    the screen removed and in free space (4928.35 / 4929.64 / 4952.95 ohm at
    x1). The disagreement is razor's first-order class on a feed 25 mm from
    a free end, not the buried route: the screen's own contribution agrees
    to 1.4 / 1.3 / 0.26 / 0.36 ohm at x1..x8 (`probe_f_control.py`).

Instrument, not a gate (momwire#1149 decision 2): at equal mesh against the
licensed NEC-5 reference (verified against our licensed materials, scoping
probe 10), this deck's razor Z11 is within 0.10 ohm (0.06 at x8), Z22
within 0.023 and Z12 within 0.003 at x1..x8; the reference's own
non-reciprocity is 8.6e-3 -> 1.0e-4 on the same ladder.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill, _wire_loading  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402

C0 = 299792458.0


def detached(m=1, gap=0.3, dx=0.5, *, ground=True, eps=(13.0, 0.005), **kw):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    below, above = d["wires"]
    d["wires"] = [
        below + np.array([dx, 0.0, -gap]),
        above + np.array([0.0, 0.0, gap]),
    ]
    d.pop("junctions")
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if ground:
        d["ground_eps"] = eps
    else:
        for k in ("ground_z", "ground_eps", "ground_model"):
            d.pop(k)
    d.update(kw)
    return d


def two_port(cls, deck, **kw):
    Y = np.asarray(cls(**deck, **kw).compute_y_matrix())
    return np.linalg.inv(Y), abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])


def razor2(deck):
    return two_port(RazorSolver, deck, nec5_quadrature=True)


def _decays(nonrec, bar=3.0):
    ratios = [a / b for a, b in zip(nonrec, nonrec[1:])]
    assert all(r >= bar for r in ratios), (nonrec, ratios)
    return ratios


# ----------------------------------------------------------------------
# the route
# ----------------------------------------------------------------------


def test_the_detached_deck_takes_the_detached_route(monkeypatch):
    """Through the REAL constructor, with no crossing tent, and with both
    cross blocks actually filled once each (counted, so this cannot pass on
    a route that skipped them)."""
    calls = {"fwd": 0, "rev": 0}
    fwd = _crossing_fill.cross_complete_block
    rev = _crossing_fill.cross_complete_block_reversed

    def f(*a, **k):
        calls["fwd"] += 1
        return fwd(*a, **k)

    def r(*a, **k):
        calls["rev"] += 1
        return rev(*a, **k)

    monkeypatch.setattr(_crossing_fill, "cross_complete_block", f)
    monkeypatch.setattr(_crossing_fill, "cross_complete_block_reversed", r)
    s = RazorSolver(**detached(), nec5_quadrature=True)
    assert s._detached and not s._crossing and not s._below_plane
    assert s._crossing_tents(s._build_geometry()) == []
    z, _ = s.compute_impedance()
    assert np.all(np.isfinite(np.asarray(z)))
    assert calls == {"fwd": 1, "rev": 1}


def test_the_row_serves_it():
    """The buried cell is what a consumer asks for a detached deck (no
    junction in the plane), and it is served."""
    assert RazorSolver.capabilities.refusal("buried", "sommerfeld") is None
    assert RazorSolver.capabilities.refusal("buried") is None


def test_mixed_radii_are_refused_by_the_declared_sentence():
    with pytest.raises(ValueError) as exc:
        RazorSolver(**detached(wire_radius=[0.001, 0.002]))
    declared = RazorSolver.capabilities.refusal("per_wire_radius", "detached")
    assert str(exc.value).endswith(declared)


def test_the_extended_kernel_is_refused_by_the_declared_sentence():
    with pytest.raises(NotImplementedError) as exc:
        RazorSolver(**detached(), extended_kernel=True)
    declared = RazorSolver.capabilities.refusal("buried", "extended_kernel")
    assert str(exc.value).endswith(declared)


def test_the_switch_off_refuses_it(monkeypatch):
    monkeypatch.setattr(_razor, "_SERVE_BELOW_PLANE", False)
    with pytest.raises(ValueError) as exc:
        RazorSolver(**detached())
    assert str(exc.value).endswith(_razor._BURIED_FILL_REFUSAL)


# ----------------------------------------------------------------------
# reference-free: reciprocity decays under refinement
# ----------------------------------------------------------------------


def test_reciprocity_decays_at_least_3x_per_doubling():
    """Measured 4.05 / 4.08 (and 4.54 to x8). A consistent discretisation of
    the reciprocal operator decays; the crossing deck's node does not."""
    nonrec = [razor2(detached(m))[1] for m in (1, 2, 4)]
    assert nonrec[0] < 3e-2, nonrec
    _decays(nonrec)


@pytest.mark.parametrize("dx", [0.5, 0.0])
def test_reciprocity_decays_at_the_shallowest_stand_off(dx):
    """gap = 0.01 m: the two wire ends 20 mm apart across the plane (coaxial
    at dx = 0), where the plan and the near-interface columns bite. Measured
    3.94 / 4.02 (dx 0.5) and 3.90 / 3.96 (dx 0)."""
    nonrec = [razor2(detached(m, gap=0.01, dx=dx))[1] for m in (1, 2, 4)]
    _decays(nonrec)


# ----------------------------------------------------------------------
# reference-free: the eps~ = 1 collapse, block by block
# ----------------------------------------------------------------------


@pytest.mark.parametrize("lane", [True, False])
def test_eps_one_collapses_to_free_space_block_by_block(lane):
    """Measured on the matrix, per block: a whole-Y comparison would be
    dominated by the self terms (5e4 ohm) and hide the cross blocks (7.7
    ohm), which are the part this route adds. Cross blocks 7e-14 of their
    own scale, same-medium blocks at rounding."""
    s1 = RazorSolver(**detached(eps=(1.0, 0.0)), nec5_quadrature=lane)
    sf = RazorSolver(**detached(ground=False), nec5_quadrature=lane)
    assert s1._detached
    g1, gf = s1._build_geometry(), sf._build_geometry()
    Z1, Zf = s1._assemble_Z(g1, s1.k), sf._assemble_Z(gf, sf.k)
    off = np.asarray(g1["basis_offsets"])
    below, above = np.arange(off[0], off[1]), np.arange(off[1], off[2])
    for name, rows, cols, bar in (
        ("above x below", above, below, 1e-10),
        ("below x above", below, above, 1e-10),
        ("above x above", above, above, 1e-12),
        ("below x below", below, below, 1e-12),
    ):
        b1, bf = Z1[np.ix_(rows, cols)], Zf[np.ix_(rows, cols)]
        rel = np.max(np.abs(b1 - bf)) / np.max(np.abs(bf))
        assert rel < bar, f"{name}: {rel:.3e}"


def test_a_swept_solve_is_the_single_frequency_solves():
    """The detached route reached through the sweep's shared prepare: every k
    must equal a fresh single-frequency solver at that wavelength (the
    scoping study measured 9e-13 / 4e-12 ohm on the other two families)."""
    base = detached()
    lam0 = base["wavelength"]
    lams = (lam0, lam0 / 1.05)
    ks = [2 * math.pi / lam for lam in lams]
    swept = np.asarray(
        RazorSolver(**base, nec5_quadrature=True).compute_impedance_swept(ks)
    )
    for i, lam in enumerate(lams):
        single, _ = RazorSolver(
            **{**base, "wavelength": lam}, nec5_quadrature=True
        ).compute_impedance()
        assert np.max(np.abs(swept[i] - np.asarray(single))) < 1e-9, (i, swept[i])


# ----------------------------------------------------------------------
# loading: applied once, on the full geometry
# ----------------------------------------------------------------------


def test_loading_is_the_full_geometry_stencil_applied_once():
    """The per-medium sub-geometries carry no loading (a first draft built
    their stencils too, and died on the sub-geometry's missing keys); the
    whole term goes on once afterwards. So loaded minus unloaded is exactly
    the full-geometry stencil at this omega — not twice it, not half."""
    kw = dict(wire_conductivity=3.5e7)
    s0 = RazorSolver(**detached(), nec5_quadrature=True)
    sl = RazorSolver(**detached(**kw), nec5_quadrature=True)
    g = sl._build_geometry()
    Z0 = s0._assemble_Z(s0._build_geometry(), s0.k)
    ZL = sl._assemble_Z(g, sl.k)
    L = np.zeros_like(ZL)
    sl._apply_loading(
        L,
        sl._loading_stencil(g),
        _wire_loading.loading_for(sl, sl.c * sl.k, g),
    )
    assert np.any(L != 0)
    err = np.max(np.abs((ZL - Z0) - L))
    assert err <= 1e-12 * np.max(np.abs(Z0)), err
    assert err <= 1e-6 * np.max(np.abs(L)), err


# ----------------------------------------------------------------------
# convergence onto bspline (slow: bspline's transmitted grid is ~8 s a rung)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_z_converges_onto_bsplines_transmitted_grid_route(record_property):
    """Equal-mesh gap shrinking per doubling, not equality at one mesh
    (razor's first-order walk against a converging Galerkin answer). Measured
    ratios 0.53-0.61; x8/x1 0.16-0.18."""
    gaps = {"z11": [], "z22": [], "z12": []}
    for m in (1, 2, 4, 8):
        Zr, _ = razor2(detached(m))
        Zb, _ = two_port(BSplineSolver, detached(m))
        for key, (i, j) in (("z11", (0, 0)), ("z22", (1, 1)), ("z12", (0, 1))):
            gaps[key].append(abs(Zr[i, j] - Zb[i, j]))
        record_property(f"x{m}", f"razor Z12 {Zr[0, 1]:.4f} bspline {Zb[0, 1]:.4f}")
    for key, g in gaps.items():
        ratios = [b / a for a, b in zip(g, g[1:])]
        assert all(r < 0.7 for r in ratios), (key, g, ratios)
        assert g[-1] / g[0] < 0.25, (key, g)


@pytest.mark.slow
@pytest.mark.parametrize("dx", [0.5, 0.0])
@pytest.mark.parametrize("gap", [0.3, 0.1, 0.03])
def test_the_stand_off_ladder_decays(gap, dx):
    """Served and reciprocity-decaying at every stand-off (measured 3.9-4.1x
    per doubling everywhere); 0.01 m is the fast lane's
    `test_reciprocity_decays_at_the_shallowest_stand_off`."""
    _decays([razor2(detached(m, gap=gap, dx=dx))[1] for m in (1, 2, 4)])


@pytest.mark.slow
@pytest.mark.parametrize("dx", [0.5, 0.0])
def test_the_shallowest_stand_off_is_within_bsplines_reach(dx):
    """gap = 0.01 m at x4, against bspline's transmitted grid: measured
    |dZ12| 0.29 (dx 0.5) / 0.58 (dx 0) ohm and |dZ11| 7.4 on |Z11| ~ 965,
    against 0.16 / 8.3 at gap 0.3 — the gap moves with the geometry and
    does not blow up as the ends close."""
    Zr, _ = razor2(detached(4, gap=0.01, dx=dx))
    Zb, _ = two_port(BSplineSolver, detached(4, gap=0.01, dx=dx))
    assert abs(Zr[0, 1] - Zb[0, 1]) < 1.0, (Zr[0, 1], Zb[0, 1])
    assert abs(Zr[0, 0] - Zb[0, 0]) / abs(Zb[0, 0]) < 0.015, (Zr[0, 0], Zb[0, 0])


@pytest.mark.slow
def test_loading_converges_onto_bsplines():
    """The copper-skin shift of Z11 and Z22, razor vs bspline: gap 0.046 ->
    0.025 -> 0.015 ohm on Z11."""
    kw = dict(wire_conductivity=3.5e7)
    gaps = []
    for m in (1, 2, 4):
        shift = {}
        for name, cls, extra in (
            ("razor", RazorSolver, {"nec5_quadrature": True}),
            ("bspline", BSplineSolver, {}),
        ):
            Z0, _ = two_port(cls, detached(m), **extra)
            ZL, _ = two_port(cls, detached(m, **kw), **extra)
            shift[name] = np.diag(ZL - Z0)
        assert np.all(shift["razor"].real > 0)
        gaps.append(np.max(np.abs(shift["razor"] - shift["bspline"])))
    assert all(b < a for a, b in zip(gaps, gaps[1:])), gaps
    assert gaps[-1] / gaps[0] < 0.5, gaps


def _counterpoise(m, screen=True):
    """antennaknobs `verticals.elevated_buried_counterpoise` at its defaults
    (7.1 MHz), spelled in momwire kwargs: a lambda/4 radiator from 0.5 m whose
    first 0.05 m is the house eps-gap, fed at its middle (the feed stays 25 mm
    above the free foot on every rung), and four radials 0.6 x lambda/4 at
    depth 0.15 m from a buried hub."""
    lam = C0 / 7.1e6
    h = 0.25 * lam
    wires = [np.array([(0, 0, 0.5), (0, 0, 0.55), (0, 0, 0.5 + h)])]
    if screen:
        for t in (0.0, math.pi / 2, math.pi, 1.5 * math.pi):
            tip = np.array([0.6 * h * math.cos(t), 0.6 * h * math.sin(t), -0.15])
            tip[np.abs(tip) < 1e-9] = 0.0
            wires.append(np.array([(0.0, 0.0, -0.15), tuple(tip)]))
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[2 * m, 20 * m]] + [[6 * m]] * (len(wires) - 1),
        feeds=[(0, 0.025, 1 + 0j)],
        wavelength=lam,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )


@pytest.mark.slow
def test_the_catalog_counterpoise_adds_nothing_to_the_razor_bspline_gap():
    """razor and bspline differ by 5-9 % on this -j44 k-ohm driving point
    with or without the screen, and in free space: razor's first-order class
    on a feed 25 mm from a free end. What the detached route owns is the
    screen's CONTRIBUTION, and that agrees to ~1 ohm (0.03 % of the gap):
    measured 1.37 / 1.33 / 0.26 / 0.36 ohm at x1..x8."""
    for m in (1, 2, 4):
        z = {}
        for screen in (True, False):
            deck = _counterpoise(m, screen)
            rs = RazorSolver(**deck, nec5_quadrature=True)
            assert rs._detached is screen
            z[screen] = (
                complex(rs.compute_impedance()[0]),
                complex(BSplineSolver(**deck).compute_impedance()[0]),
            )
        d_razor = z[True][0] - z[False][0]
        d_bspline = z[True][1] - z[False][1]
        gap = abs(z[True][0] - z[True][1])
        # not vacuous: the screen moves the driving point, the same way
        assert abs(d_bspline) > 1.0 and d_razor.real < 0 and d_bspline.real < 0
        assert abs(d_razor - d_bspline) < 2.0, (m, d_razor, d_bspline)
        assert abs(d_razor - d_bspline) < 1e-3 * gap, (m, d_razor, d_bspline, gap)
