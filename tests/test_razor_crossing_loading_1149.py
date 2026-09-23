"""razor-2p wire loading on CROSSING decks — momwire#1149 U3.

Until U3 a crossing deck refused every kind of loading
(`_CROSSING_LOADING_REFUSAL`), because the crossing tent is filled as two
half tents, one per medium, and the loading term on it had not been derived.

**The derivation** (`RazorSolver._loading_stencil`, "The crossing tent").
Razor's loading term is the testing-path integral of the surface impedance
times each tent, ``L[m, n] = ∫_{P_m} Z_s Λ_n dl``. It has no kernel, so it is
a sum of per-SEGMENT closed forms (3h/8, h/8). A crossing node is a knot, so
no segment straddles the plane and every segment lies wholly in one medium.
The crossing row's path is split at the knot by the fill too, and on each
half the current is the WHOLE tent's (only one half tent is nonzero on each
segment), so the two half-path terms are the stencil's per-segment entries
of the unsplit tent. A bare conductor's Z_s is its internal impedance, set by
the metal, not by what lies outside. So the full-geometry stencil, applied
once after the crossing assembly as the detached route already does, is the
loading of the crossing deck, crossing tent included — and the per-medium
sub-geometries' stencils sum to it IDENTICALLY, which is why building those
is not a red control (measured 0.0 on all three decks here).

What stays refused is a dielectric JACKET on a buried wire of a crossing
deck: the jacket's series term is the thin-sheath formula against a
free-space exterior (`_CROSSING_BURIED_JACKET_REFUSAL`). A jacket on an above
wire is served.

The gates, U1/U2 style — real constructor, counted branches, red controls.
Measured 2026-09-22 (`scratch/razor-buried-u3/`):

  * **loaded - unloaded = one full-geometry stencil**: 3.4e-12 / 2.6e-11 /
    4.7e-13 ohm on crossing_deck / the two-radius rod / hub_deck(4), against
    stencils of 0.076 / 0.64 / 0.10 ohm and fills of 5e4..2e5 ohm;
  * **the loading SHIFT converges onto bspline's** (copper-class 3.5e7 S/m,
    every edge refined, feed on a knot): crossing_deck 0.0257 -> 0.0145 ->
    0.0084 -> 0.0050 ohm (x1..x8), hub_deck(4) 0.041 -> 0.023 -> 0.013 ->
    0.0074, the rise/2 rod 0.108 -> 0.060 -> 0.034 -> 0.019 (ratios
    0.55-0.59). Red control, the buried segments' loading dropped: flat at
    0.25 / 0.41 / 2.1 ohm. Dropping only the crossing tent's own entries is
    NOT a red control for this bar — it removes O(h_node) of path and
    converges too (0.042 -> 0.0068 on crossing_deck) — so it is caught
    where the tent's share is O(1), a lumped load AT the crossing knot;
  * **a lumped load at the crossing knot** equals razor's own 2-port algebra
    to 3e-13 ohm (one diagonal entry is a Sherman-Morrison update of a port
    there) and converges onto bspline's 2-port algebra 1.46 -> 0.79 -> 0.45
    -> 0.26 ohm on an 84 ohm shift;
  * **an above-wire jacket** (b = 2 mm, eps_r 3): shift gap to bspline 0.49
    -> 0.30 -> 0.19 -> 0.12 ohm on a 37 ohm shift;
  * **loading off is bit-identical to main** on crossing_deck, hub_deck(4)
    and the rod, and so is every non-crossing route loaded or not (probe 5,
    main's razor.py run beside the branch's);
  * reciprocity still decays at >= 3x per doubling with loading on, and a
    swept solve equals single-frequency solves.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _medium_spec, _wire_loading  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck, hub_deck  # noqa: E402

A = 0.25e-3
SIGMA = dict(wire_conductivity=3.5e7)
Z_NODE = 50.0 + 20.0j


def refined(d, m):
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


def crossing(m=1, **kw):
    d = crossing_deck(1, **kw)
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return refined(d, m)


def hub(m=1, **kw):
    d = hub_deck(**kw)
    d["feeds"] = [(len(d["wires"]) - 1, 4.0, 1 + 0j)]
    return refined(d, m)


def rod(m=1, **kw):
    """The v0.61.0 ground-rod shape: a two-radius crossing, rise at a/2."""
    d = crossing_deck(2, wire_radius=[A / 2, A], **kw)
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return refined(d, m)


DECKS = {"crossing": crossing, "hub4": hub, "rod_rise2": rod}


def razor(d, **kw):
    return RazorSolver(
        **{k: v for k, v in d.items() if k != "junctions"}, nec5_quadrature=True, **kw
    )


def z(cls, d, **kw):
    s = razor(d, **kw) if cls is RazorSolver else cls(**d, **kw)
    return complex(s.compute_impedance()[0])


def ratios(xs):
    return [b / a for a, b in zip(xs, xs[1:])]


def full_stencil(s, geom, omega):
    L = np.zeros((geom["n_basis_total"],) * 2, dtype=np.complex128)
    s._apply_loading(
        L, s._loading_stencil(geom), _wire_loading.loading_for(s, omega, geom)
    )
    return L


def algebra(Zm, z_l):
    """Driving point of port 0 with port 1 terminated in `z_l`."""
    return Zm[0, 0] - Zm[0, 1] * Zm[1, 0] / (Zm[1, 1] + z_l)


def two_port_node(m=1):
    d = crossing(m)
    d["feeds"] = [(1, 4.5, 1 + 0j), (1, 0.0, 1 + 0j)]
    return d


# ----------------------------------------------------------------------
# the route, counted
# ----------------------------------------------------------------------


def test_a_loaded_crossing_deck_is_served_through_the_constructor(monkeypatch):
    """The real constructor, the crossing route, and the loading applied
    ONCE per fill on the FULL geometry: `_apply_loading` sees the whole
    deck's basis, and `_loading_stencil` is never built on a sub-geometry."""
    applied, built = [], []
    orig_apply, orig_st = RazorSolver._apply_loading, RazorSolver._loading_stencil

    def apply(Z, stencil, spec):
        applied.append(Z.shape[0])
        return orig_apply(Z, stencil, spec)

    def st(self, geom):
        built.append("seg_offsets" in geom)
        return orig_st(self, geom)

    monkeypatch.setattr(RazorSolver, "_apply_loading", staticmethod(apply))
    monkeypatch.setattr(RazorSolver, "_loading_stencil", st)
    s = razor(crossing(), **SIGMA)
    assert s._crossing and not s._detached and not s._below_plane
    assert s.buried_serve_refusal() is None
    zl = complex(s.compute_impedance()[0])
    n = s._build_geometry()["n_basis_total"]
    assert applied == [n], applied
    assert built == [True], built
    z0 = z(RazorSolver, crossing())
    assert (zl - z0).real > 0


def test_the_row_serves_loading_on_a_crossing_deck():
    caps = RazorSolver.capabilities
    assert caps.wire_loading is True
    assert caps.refusal("wire_loading", "crossing_junction") is None
    assert "wire_loading+crossing_junction" not in caps.refusals
    assert not hasattr(_razor, "_CROSSING_LOADING_REFUSAL")


# ----------------------------------------------------------------------
# the derivation, pinned
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DECKS))
def test_loaded_minus_unloaded_is_one_full_geometry_stencil(name):
    """Not twice it, not the per-medium halves of it: exactly the stencil of
    the whole deck, the crossing tent as one two-wing tent."""
    mk = DECKS[name]
    s0, sl = razor(mk()), razor(mk(**SIGMA))
    g = sl._build_geometry()
    Z0 = s0._assemble_Z(s0._build_geometry(), s0.k)
    ZL = sl._assemble_Z(g, sl.k)
    L = full_stencil(sl, g, sl.c * sl.k)
    tents = [m for m, _ in sl._crossing_tents(g)]
    assert tents and np.all(np.abs(np.diag(L)[tents]) > 0)
    err = np.max(np.abs((ZL - Z0) - L))
    assert err <= 1e-12 * np.max(np.abs(Z0)), err
    assert err <= 1e-8 * np.max(np.abs(L)), err


@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_per_medium_stencils_sum_to_the_full_one_identically(name):
    """The derivation's corollary: every segment lies in one medium and the
    term has no kernel, so the two sub-geometries' stencils (each crossing
    tent a half tent with a sigma = 0 ghost wing), mapped back through the
    basis map, ARE the full stencil. Which is why building them is not a red
    control for anything, and why the crossing tent needs no special case."""
    s = razor(DECKS[name](**SIGMA))
    g = s._build_geometry()
    spec = _wire_loading.loading_for(s, s.omega, g)
    L = full_stencil(s, g, s.omega)
    media = s._wire_media()
    off = np.asarray(g["seg_offsets"])
    Lm = np.zeros_like(L)
    for side in (_medium_spec.ABOVE, _medium_spec.BELOW):
        sub, rows, _chop = s._medium_geometry(g, side)
        st = s._loading_stencil(dict(sub, n_segs_total=sub["seg_h"].shape[0]))
        seg_i = np.concatenate(
            [np.arange(off[w], off[w + 1]) for w, m in enumerate(media) if m == side]
        )
        np.add.at(
            Lm,
            (rows[st["rows"]], rows[st["cols"]]),
            spec.z_seg[seg_i[st["seg"]]] * st["vals"],
        )
    assert np.array_equal(Lm, L)


# ----------------------------------------------------------------------
# a lumped load AT the crossing knot: the tent's share is O(1)
# ----------------------------------------------------------------------


def _node_lumped(m=1):
    return razor(crossing(m), lumped_loads=[(1, 0.0, Z_NODE)])


def test_a_lumped_load_at_the_crossing_knot_is_its_own_port_algebra():
    """One diagonal entry of the crossing tent is a Sherman-Morrison update
    of a port at that knot, so the loaded solve equals razor's own 2-port
    algebra to rounding (measured 1.7e-13 ohm on an 84 ohm shift)."""
    Zm = np.linalg.inv(np.asarray(razor(two_port_node()).compute_y_matrix()))
    s = _node_lumped()
    g = s._build_geometry()
    idx, _ = _wire_loading.loading_for(s, s.omega, g).lumped
    assert list(idx) == [m for m, _ in s._crossing_tents(g)]
    zl = complex(s.compute_impedance()[0])
    assert abs(zl - algebra(Zm, Z_NODE)) < 1e-9, (zl, algebra(Zm, Z_NODE))


def test_without_the_crossing_tents_entries_the_node_load_is_lost(monkeypatch):
    """The red control for the gate above, and for the crossing tent's own
    entries generally: drop every loading entry on a crossing tent's row or
    column and the node load disappears (84 ohm), where on the distributed
    term the same omission is only O(h_node) and converges away."""
    s = _node_lumped()
    tents = np.array([m for m, _ in s._crossing_tents(s._build_geometry())])
    orig = RazorSolver._apply_loading

    def without_tents(Z, stencil, spec):
        keep = ~(np.isin(stencil["rows"], tents) | np.isin(stencil["cols"], tents))
        st = {k: v[keep] for k, v in stencil.items()}
        lumped = spec.lumped
        if lumped is not None:
            li = ~np.isin(lumped[0], tents)
            lumped = (lumped[0][li], lumped[1][li])
        return orig(Z, st, _wire_loading.LoadingSpec(spec.z_wire, spec.z_seg, lumped))

    monkeypatch.setattr(RazorSolver, "_apply_loading", staticmethod(without_tents))
    Zm = np.linalg.inv(np.asarray(razor(two_port_node()).compute_y_matrix()))
    zl = complex(_node_lumped().compute_impedance()[0])
    assert abs(zl - algebra(Zm, Z_NODE)) > 10.0


# ----------------------------------------------------------------------
# reference-free: reciprocity, the sweep
# ----------------------------------------------------------------------


def test_reciprocity_still_decays_with_loading_on():
    """The loading term is symmetric, so it must not spoil the node's decay:
    crossing_deck's 2-port (above 4.5 / buried 1.0), every edge refined."""

    def nonrec(m):
        d = crossing_deck(1)
        d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
        Y = np.asarray(razor(refined(d, m), **SIGMA).compute_y_matrix())
        return abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])

    xs = [nonrec(m) for m in (1, 2, 4)]
    assert all(1 / r >= 3.0 for r in ratios(xs)), xs


def test_a_swept_loaded_solve_is_the_single_frequency_solves():
    """Z_s moves with omega (skin effect), so the stencil must ride the
    prepare half and the values the per-k half on the crossing route too."""
    base = crossing(**SIGMA, lumped_loads=[(0, 1.0, 5 + 3j)])
    lam0 = base["wavelength"]
    lams = (lam0, lam0 / 1.05)
    ks = [2 * math.pi / lam for lam in lams]
    swept = np.asarray(razor(base).compute_impedance_swept(ks))
    for i, lam in enumerate(lams):
        single = complex(razor({**base, "wavelength": lam}).compute_impedance()[0])
        assert abs(swept[i] - single) < 1e-9, (i, swept[i], single)


# ----------------------------------------------------------------------
# what stays refused: a jacket in the soil
# ----------------------------------------------------------------------


def _jacket(on):
    b = [np.nan, np.nan]
    e = [np.nan, np.nan]
    b[on], e[on] = 0.002, 3.0
    return dict(insulation_radius=b, insulation_eps_r=e)


def test_a_buried_jacket_on_a_crossing_deck_is_refused_by_the_declared_cell():
    """Wire 0 of crossing_deck is the buried one. The fill and the pre-flight
    say the same sentence, and it is the row's."""
    s = razor(crossing(**_jacket(0)))
    declared = RazorSolver.capabilities.refusal("crossing_junction", "insulation")
    assert declared is _razor._CROSSING_BURIED_JACKET_REFUSAL
    assert s.buried_serve_refusal() == declared
    with pytest.raises(NotImplementedError) as exc:
        s.compute_impedance()
    assert str(exc.value) == declared


def test_an_above_jacket_on_a_crossing_deck_is_served():
    s = razor(crossing(**_jacket(1)))
    assert s.buried_serve_refusal() is None
    assert np.isfinite(complex(s.compute_impedance()[0]))


def test_the_jacket_refusal_is_the_crossing_decks_only():
    """Scope, pinned so a change to it is deliberate: razor's detached route
    serves a buried jacket today (as every bspline buried route does), and
    U3 does not change that (momwire#1149, open question)."""
    from test_razor_detached_1149 import detached

    s = razor(detached(**_jacket(0)))
    assert s._detached and s.buried_serve_refusal() is None
    assert np.all(np.isfinite(np.asarray(s.compute_impedance()[0])))


# ----------------------------------------------------------------------
# convergence onto bspline (slow)
# ----------------------------------------------------------------------


def _shift(cls, mk, m, **load):
    return z(cls, mk(m, **load)) - z(cls, mk(m))


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_loading_shift_converges_onto_bsplines(name):
    """|shift_razor - shift_bspline| shrinking per doubling: measured ratios
    0.55-0.59 on all three decks to x8."""
    mk = DECKS[name]
    g = [
        abs(_shift(RazorSolver, mk, m, **SIGMA) - _shift(BSplineSolver, mk, m, **SIGMA))
        for m in (1, 2, 4)
    ]
    assert all(r <= 0.65 for r in ratios(g)), g


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(DECKS))
def test_without_the_buried_segments_loading_the_shift_does_not_converge(
    name, monkeypatch
):
    """The red control for the gate above: the loading of every BELOW
    segment dropped (the U0-era worry, one medium's stencil only). The gap
    to bspline sits flat at 0.25 / 0.41 / 2.1 ohm."""
    orig = RazorSolver._loading_stencil

    def above_only(self, geom):
        out = orig(self, geom)
        media = self._wire_media()
        off = np.asarray(geom["seg_offsets"])
        below = np.concatenate(
            [np.arange(off[w], off[w + 1]) for w, m in enumerate(media) if m == "below"]
        )
        keep = ~np.isin(out["seg"], below)
        return {k: v[keep] for k, v in out.items()}

    mk = DECKS[name]
    zb = [_shift(BSplineSolver, mk, m, **SIGMA) for m in (1, 2)]
    monkeypatch.setattr(RazorSolver, "_loading_stencil", above_only)
    g = [abs(_shift(RazorSolver, mk, m, **SIGMA) - b) for m, b in zip((1, 2), zb)]
    assert all(r > 0.9 for r in ratios(g)), g


@pytest.mark.slow
def test_the_node_load_converges_onto_bsplines_port_algebra():
    """bspline has no lumped kwarg; its answer is the 2-port algebra over a
    port at the knot. Measured 1.46 -> 0.79 -> 0.45 ohm (x1..x4)."""
    g = []
    for m in (1, 2, 4):
        zb0 = z(BSplineSolver, crossing(m))
        Zm = np.linalg.inv(
            np.asarray(BSplineSolver(**two_port_node(m)).compute_y_matrix())
        )
        zr0 = z(RazorSolver, crossing(m))
        zr = complex(_node_lumped(m).compute_impedance()[0])
        g.append(abs((zr - zr0) - (algebra(Zm, Z_NODE) - zb0)))
    assert all(r <= 0.65 for r in ratios(g)), g


@pytest.mark.slow
def test_an_above_jacket_converges_onto_bsplines():
    """The jacket's shift (series L' and the a' kernel radius, which makes
    the node a two-radius one): 0.49 -> 0.30 -> 0.19 ohm (x1..x4)."""
    g = [
        abs(
            _shift(RazorSolver, crossing, m, **_jacket(1))
            - _shift(BSplineSolver, crossing, m, **_jacket(1))
        )
        for m in (1, 2, 4)
    ]
    assert all(r <= 0.7 for r in ratios(g)), g
