"""The exact ring kernel above ground on a deck with buried wires (momwire#1619
stage 1).

Until this, any wire below the ground refused `exact_kernel=True`
(`BURIED_REFUSAL`: the kernel is written for a real wavenumber) and "auto"
resolved False on every buried deck, so a fat vertical over buried radials
never got the exact kernel however short its segments. Above ground the
wavenumber is real, so stage 1 serves the combination IN PART on razor-2p and
bs2, by the rule the extended kernel follows there (momwire#1170 on bs2,
momwire#1614 on razor):

* a coaxial pair whose two segments are both above ground is corrected;
* a pair touching a buried segment keeps the fill's kernel, and so does a
  stepped line's cross job into a buried wire;
* razor's chopped rows keep the fill's kernel at the KNOT observer, the
  crossing node in the plane;
* "auto" reads the above-ground segments only, and a wholly buried deck is
  still refused under True and resolves False under "auto".

The decks are synthetic stand-ins for the deck shape the issue measured: a
19.05 mm vertical about 5 m tall over buried radials of 1.026 mm at 14 MHz,
GN 0 13/0.005, its base on a hub (a one-segment stub down to the radials), on
radials that slope from the base into the soil, or lifted clear of a buried
hub it does not touch (no crossing node).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from momwire import BSplineSolver, RazorSolver, _medium_spec
from momwire import _exact_kernel as EK
from momwire import _exact_kernel_seam as SEAM
from momwire import _razor_exact_plan as PLAN

A_V, A_R = 0.01905, 0.0010262
L_V, L_R = 5.0253, 5.1816
WL14 = 299792458.0 / 14e6
GROUND = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
FAMILIES = {
    "razor-2p": (RazorSolver, dict(nec5_quadrature=True)),
    "bs2": (BSplineSolver, {}),
}


def hub(nv, nr=4, nrad=6, stub_a=A_R, stub_n=1, radial_a=A_R):
    """The vertical on a hub: radials 25.4 mm down, a stub from the base."""
    wires = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, L_V)])]
    for i in range(nr):
        th = 2 * math.pi * i / nr
        x, y = L_R * math.cos(th), L_R * math.sin(th)
        wires.append(np.array([(0.0, 0.0, -0.0254), (x, y, -0.0254)]))
    wires.append(np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -0.0254)]))
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[nv]] + [[nrad]] * nr + [[stub_n]],
        wire_radius=[A_V] + [radial_a] * nr + [stub_a],
    )


def slant(nv, nr=4, nrad=6):
    """The vertical on radials that leave its base and slope into the soil."""
    wires = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, L_V)])]
    for i in range(nr):
        th = 2 * math.pi * i / nr
        x, y = L_R * math.cos(th), L_R * math.sin(th)
        wires.append(np.array([(0.0, 0.0, 0.0), (x, y, -0.0508)]))
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[nv]] + [[nrad]] * nr,
        wire_radius=[A_V] + [A_R] * nr,
    )


def detached(nv, nr=4, nrad=6):
    """The vertical lifted 0.3 m clear of the ground over a buried hub of
    radials it does not touch: no crossing node, the detached route."""
    wires = [np.array([(0.0, 0.0, 0.3), (0.0, 0.0, 0.3 + L_V)])]
    for i in range(nr):
        th = 2 * math.pi * i / nr
        x, y = L_R * math.cos(th), L_R * math.sin(th)
        wires.append(np.array([(0.0, 0.0, -0.0254), (x, y, -0.0254)]))
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[nv]] + [[nrad]] * nr,
        wire_radius=[A_V] + [A_R] * nr,
    )


DECKS = {"hub": hub, "slant": slant, "detached": detached}
# (family, deck) for the solving gates. bs2's detached deck takes the
# transmitted blocks, several seconds a solve: the push lane's.
SOLVES = [
    pytest.param(
        f,
        d,
        id=f"{f}-{d}",
        marks=[pytest.mark.slow] if (f, d) == ("bs2", "detached") else [],
    )
    for f in sorted(FAMILIES)
    for d in sorted(DECKS)
]
# Delta/a on the vertical: 8.8 at 30 segments, 2.2 at 120.
COARSE, FINE = 30, 120


def solver(family, deck, *, exact="auto", ek=True, **kw):
    cls, extra = FAMILIES[family]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(
            **deck,
            wavelength=WL14,
            feeds=[(0, 0.3, 1 + 0j)],
            extended_kernel=ek,
            exact_kernel=exact,
            **GROUND,
            **extra,
            **kw,
        )


def _z(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(s.compute_impedance()[0])


def _matrix(s):
    """The loaded Z the solve factors, and the per-basis 'wholly buried' mask."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = s._build_geometry()
        if isinstance(s, RazorSolver):
            Z = s._assemble_Z(geom, s.k)
            above = s._above_segments(geom)
            wing = geom["wing_seg"]
            sig = geom["wing_sigma"]
            touches = ((above[wing]) & (sig != 0)).any(axis=1)
            return np.asarray(Z), ~touches
        supp, polys = s._build_basis_polynomials(geom)[:2]
        Z = s._compute_Z_operator(geom, supp, polys)
        below = np.asarray(s._below_segments(geom), dtype=bool)
        live = np.any(polys != 0.0, axis=2)
        touches = (~below[np.where(supp >= 0, supp, 0)] & live).any(axis=1)
        return np.asarray(Z), ~touches


# -- "auto" ---------------------------------------------------------------------


@pytest.mark.parametrize("deck", sorted(DECKS))
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_auto_engages_on_a_fat_vertical_over_buried_radials(family, deck):
    """The issue's case: the vertical's Delta/a under 3 engages "auto" on a
    buried deck; at 8.8 it does not."""
    assert solver(family, DECKS[deck](FINE)).exact_kernel is True
    assert solver(family, DECKS[deck](COARSE)).exact_kernel is False


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_auto_reads_the_above_ground_segments_only(family):
    """A fat, short BURIED stub (h/a 0.67, coaxial with the vertical and at
    its radius, the buried side one radius throughout) does not engage
    "auto": its pairs are stage 2's. The segments asked are the vertical's
    alone."""
    s = solver(family, hub(COARSE, stub_a=A_V, stub_n=2, radial_a=A_V))
    assert s.exact_kernel is False
    h, a = s._exact_kernel_h_and_a()
    assert h.size == COARSE and np.all(a == A_V)
    assert np.all(h >= SEAM.EXACT_KERNEL_AUTO_H_OVER_A * a)


BURIED_DIPOLE = dict(
    wires=[np.array([(-1.0, 0.0, -0.3), (1.0, 0.0, -0.3)])],
    n_per_edge_per_wire=[[19]],
    wire_radius=0.0125,
)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_wholly_buried_deck_is_still_refused(family):
    """Nothing above ground, nothing to correct: True refuses by name at the
    fill and "auto" resolves False, as before."""
    assert solver(family, BURIED_DIPOLE).exact_kernel is False
    s = solver(family, BURIED_DIPOLE, exact=True)
    with pytest.raises(NotImplementedError, match="buried wires"):
        s.compute_impedance()


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_true_is_served_on_a_mixed_deck(family):
    s = solver(family, hub(COARSE), exact=True)
    assert s._exact_kernel_route_refusal(None, False) is None
    assert s._exact_kernel_resolved() == (True, True)


# -- which pairs ------------------------------------------------------------------


@pytest.mark.parametrize(("family", "deck"), SOLVES)
def test_a_wholly_buried_basis_keeps_its_row_and_column(family, deck):
    """Every pair touching a buried segment keeps the fill's kernel, so the
    row and column of a basis with no above-ground support are the same
    bits with the exact kernel on or off -- and the vertical's are not."""
    d = DECKS[deck](FINE)
    z_off, buried = _matrix(solver(family, d, exact=False))
    z_on, buried_on = _matrix(solver(family, d, exact=True))
    assert np.array_equal(buried, buried_on)
    assert buried.sum() >= 4 * 5
    assert np.array_equal(z_on[buried, :], z_off[buried, :])
    assert np.array_equal(z_on[:, buried], z_off[:, buried])
    moved = np.abs(z_on - z_off)
    assert moved[np.ix_(~buried, ~buried)].max() > 1e-6 * np.abs(z_off).max()


def test_bs2_jobs_are_cut_to_the_above_ground_segments():
    """bs2's jobs on a buried deck: every segment above ground, and the line
    stepping from the vertical into the coaxial buried stub (a radius step,
    a cross job in free space) is no job at all."""
    s = solver("bs2", hub(FINE), exact=True)
    geom = s._build_geometry()
    below = np.asarray(s._below_segments(geom), dtype=bool)
    every = s._exact_kernel_jobs(geom)
    assert any(lab is not None for _S, _e, lab in every)
    jobs = s._exact_kernel_jobs(geom, above_only=True)
    assert jobs and all(lab is None for _S, _e, lab in jobs)
    for S, edges, _lab in jobs:
        assert not below[S].any()
        assert all(not below[e[0].start] for e in edges)
    assert sum(S.size for S, _e, _l in jobs) == int((~below).sum())


def test_razor_plans_the_above_medium_only(monkeypatch):
    """razor's correction runs on the above medium's own sub-geometry: every
    segment it plans over is above ground, and the crossing tent's half
    reaches it as a chopped row."""
    seen = []
    orig = PLAN.build_plan

    def spy(geom, *args, **kw):
        seen.append((geom, kw.get("chop", args[7] if len(args) > 7 else None)))
        return orig(geom, *args, **kw)

    monkeypatch.setattr(PLAN, "ENABLED", False)
    monkeypatch.setattr(PLAN, "build_plan", spy)
    s = solver("razor-2p", hub(FINE), exact=True)
    assert s._wire_media()[-1] == _medium_spec.BELOW
    _z(s)
    assert len(seen) == 1
    geom, chop = seen[0]
    z = np.concatenate(
        [
            geom["seg_p0"][:, 2],
            geom["seg_p0"][:, 2] + geom["seg_h"] * geom["seg_t"][:, 2],
        ]
    )
    assert z.min() >= -1e-12
    assert chop is not None and len(chop) == 1


def test_razor_knot_observer_is_not_corrected():
    """A chopped row's T2 ends at the knot, in the plane: its coefficient
    there is zeroed (the knot keeps the fill's kernel), and only there."""
    s = solver("razor-2p", hub(FINE), exact=True)
    geom = s._build_geometry()
    geom_a, _rows, chop = s._medium_geometry(geom, _medium_spec.ABOVE)
    rows = np.array(sorted(chop), dtype=np.int64)
    seg_a = np.asarray(s._medium_seg_radius(geom_a), dtype=float)
    pts, tans, wts = s._testing_paths(geom_a)
    args = (geom_a, seg_a, pts, tans, wts, None, True, 1 << 20)
    plain, _ = PLAN.build_plan(*args)
    chopped, _ = PLAN.build_plan(*args, chop=rows)
    c_phi = 1j * 2 * np.pi * 14e6 * s.eps
    zeroed = 0
    for g0, g1 in zip(plain, chopped, strict=True):
        p0, p1 = PLAN.row_cP(g0, c_phi), PLAN.row_cP(g1, c_phi)
        diff = np.flatnonzero(p0 != p1)
        assert np.all(p1[diff] == 0.0)
        assert np.all(np.isin(g1.r_m[diff], rows))
        zeroed += diff.size
    assert zeroed == rows.size


# -- the mechanism -------------------------------------------------------------------


def _reduced(z, a, k):
    R = np.sqrt(np.asarray(z) ** 2 + a * a)
    return np.exp(-1j * k * R) / (4 * np.pi * R)


def _extended(z, a, k):
    R = np.sqrt(np.asarray(z) ** 2 + a * a)
    kr = k * R
    c1 = 1 + 1j * kr
    c2 = 3 * c1 - kr * kr
    fac = 1 + 0.25 * a**4 / R**4 * c2 - 0.5 * a * a / R**2 * c1
    return np.exp(-1j * kr) / (4 * np.pi * R) * fac


@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize(("family", "deck"), SOLVES)
def test_swapped_kernel_reproduces_the_buried_fill(monkeypatch, family, deck, ek):
    """Hand the correction the fill's OWN kernel: the buried deck's Z must not
    move beyond the fill's quadrature accuracy. This is what shows the base
    removed is the one the buried fill's above medium put there."""
    kern = _extended if ek else _reduced
    kw = {} if family == "razor-2p" else dict(n_qp_pair_same_edge=8)
    d = DECKS[deck](FINE)
    base = _z(solver(family, d, exact=False, ek=ek, **kw))
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    monkeypatch.setattr(EK, "ring_kernel", kern)
    monkeypatch.setattr(
        EK,
        "ring_kernel_split",
        lambda z, a, k: (np.zeros(np.shape(z), complex), kern(z, a, k)),
    )
    cls = FAMILIES[family][0]
    calls = []
    orig = cls._add_exact_kernel_correction

    def spy(self, *args, **kwargs):
        calls.append(1)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(cls, "_add_exact_kernel_correction", spy)
    swapped = _z(solver(family, d, exact=True, ek=ek, **kw))
    assert calls, "the correction did not run"
    assert abs(swapped - base) <= 1e-8 * abs(base), (swapped, base)
