"""razor's cross-block axes grade a segment that APPROACHES the plane —
momwire#1152.

`_crossing_fill.axis_data` graded only segments TOUCHING the interface. On a
coaxial detached deck whose mast starts `gap` above the plane on one long
segment, with a buried rise ending `gap` below it (`detached_hub`: 0.67 m
mast segment), that segment got plain Gauss-12 against an observer 2*gap
away across the plane, and razor's REVERSED cross block collapsed at
eps~ = 1 only to 6.1e-6 at gap 0.01 m and 9.9e-9 at 0.05 m, where every
other block and deck reaches 1e-13.

`grade_near_plane=True`, passed by razor only, grades such a segment toward
its nearer end, first panel max(d, a). Measured 2026-09-22
(`scratch/razor-buried-u3/`, probes 1 and 7):

  * detached_hub reversed block at eps~ = 1: 6.1e-6 -> 2.3e-13 (gap 0.01),
    9.9e-9 -> 3.0e-14 (0.05), 2.4e-11 -> 1.2e-14 (0.1);
  * razor's driving point moves by <= 1e-10 ohm on crossing_deck (x1, x4),
    hub_deck(4), the rise/2 rod and U1's detached deck, and by 9e-8 ohm on
    detached_hub at gap 0.01 — the quadrature error it removes;
  * BSplineSolver and SinusoidalGalerkinSolver never pass the flag, so their
    axes are unchanged by construction (gated below by recording the calls).

Note the issue's own deck description ("the coaxial dx = 0, gap 0.01 m
detached deck") matches U1's `detached(gap=0.01, dx=0)` in words, but that
deck collapses at 1e-15 already: its plane-end segments are 50 mm, so the
gap is a fifth of the segment. The 6e-6 lives on `detached_hub`, whose mast
segment is 67 times the gap, and that is the deck gated here.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import SOIL_A, hub_deck  # noqa: E402

ORIG = _crossing_fill.axis_data


def detached_hub(gap, *, ground=True, eps=SOIL_A):
    """hub_deck(2) with the node pulled apart: two radials at 0.15 m depth
    into a buried hub, a rise from the hub up to -gap, and the 10 m monopole
    (15 segments, 0.67 m each) from +gap up. Ports on the monopole (4.0) and
    radial 0 (1.0). `scratch/razor-buried-u2b/common.py`'s deck at uniform
    radii."""
    d = hub_deck(n_radials=2)
    d.pop("junctions")
    w = d["wires"]
    w[2] = np.array([(0.0, 0.0, -0.15), (0.0, 0.0, -gap)])
    w[3] = np.array([(0.0, 0.0, 10.0), (0.0, 0.0, gap)])
    d["feeds"] = [(3, 4.0, 1 + 0j), (0, 1.0, 1 + 0j)]
    if ground:
        d["ground_eps"] = eps
    else:
        for k in ("ground_z", "ground_eps", "ground_model"):
            d.pop(k)
    return d


def _blocks(gap):
    """Per-block relative eps~ = 1 collapse against razor's free-space fill."""
    s1 = RazorSolver(**detached_hub(gap, eps=(1.0, 0.0)), nec5_quadrature=True)
    sf = RazorSolver(**detached_hub(gap, ground=False), nec5_quadrature=True)
    assert s1._detached
    g1, gf = s1._build_geometry(), sf._build_geometry()
    Z1, Zf = s1._assemble_Z(g1, s1.k), sf._assemble_Z(gf, sf.k)
    off = np.asarray(g1["basis_offsets"])
    media = s1._wire_media()
    side = {
        m: np.concatenate(
            [np.arange(off[w], off[w + 1]) for w, lab in enumerate(media) if lab == m]
        )
        for m in ("above", "below")
    }
    out = {}
    for name, rows, cols in (
        ("above x below", side["above"], side["below"]),
        ("below x above", side["below"], side["above"]),
    ):
        b1, bf = Z1[np.ix_(rows, cols)], Zf[np.ix_(rows, cols)]
        out[name] = float(np.max(np.abs(b1 - bf)) / np.max(np.abs(bf)))
    return out


@pytest.fixture
def ungraded(monkeypatch):
    """The pre-#1152 axis: the flag forced off, everything else untouched."""

    def off(*a, **kw):
        kw["grade_near_plane"] = False
        return ORIG(*a, **kw)

    monkeypatch.setattr(_crossing_fill, "axis_data", off)


@pytest.mark.parametrize("gap", [0.01, 0.05])
def test_the_coaxial_near_plane_deck_collapses_at_eps1(gap):
    """Measured 2.3e-13 / 3.0e-14 on the reversed block (was 6.1e-6 /
    9.9e-9); the forward block was already at 6e-16."""
    rel = _blocks(gap)
    assert all(r <= 1e-10 for r in rel.values()), rel


def test_without_the_grading_the_reversed_block_misses(ungraded):
    """The red control: the same deck on the old axis, 6.1e-6."""
    rel = _blocks(0.01)
    assert rel["below x above"] > 1e-7, rel


def test_only_approaching_segments_are_graded(monkeypatch):
    """The 2d test keeps it narrow. On detached_hub(0.01) exactly two
    segments are newly graded: the mast's first (10 mm off the plane, 0.67 m
    long) and the rise's top one (10 mm below it, 70 mm long). The level
    radials at 0.15 m depth never are, and neither is any segment further up
    the mast, because each is less than twice as far off as its lower end."""
    graded = []

    def spy(ctx, seg_idx, *a, grade_near_plane=False, **kw):
        ax = ORIG(ctx, seg_idx, *a, grade_near_plane=grade_near_plane, **kw)
        base = ORIG(ctx, seg_idx, *a, grade_near_plane=False, **kw)
        for g in seg_idx:
            if ax["seg_runs"][int(g)][1] != base["seg_runs"][int(g)][1]:
                graded.append(int(g))
        return ax

    monkeypatch.setattr(_crossing_fill, "axis_data", spy)
    s = RazorSolver(**detached_hub(0.01), nec5_quadrature=True)
    geom = s._build_geometry()
    s._assemble_Z(geom, s.k)
    seg_off = np.asarray(geom["seg_offsets"])
    mast_first = int(seg_off[3 + 1]) - 1  # the mast is spelled top-down
    rise_top = int(seg_off[2 + 1]) - 1
    assert sorted(set(graded)) == sorted({mast_first, rise_top}), graded
    p0 = (
        geom["seg_p0"][mast_first]
        + geom["seg_h"][mast_first] * geom["seg_t"][mast_first]
    )
    assert abs(p0[2] - 0.01) < 1e-12


def test_bspline_never_passes_the_flag(monkeypatch):
    """bspline's crossing axes are shared code: they must be the axes they
    were, and they are by construction, because the flag defaults off and
    only razor passes it. Recorded on bspline's crossing fill (its detached
    route is the transmitted grid and reads no crossing axis)."""
    seen = []

    def spy(*a, **kw):
        seen.append(kw.get("grade_near_plane", False))
        return ORIG(*a, **kw)

    monkeypatch.setattr(_crossing_fill, "axis_data", spy)
    from test_crossing_serve_524 import crossing_deck

    BSplineSolver(**crossing_deck(1)).compute_impedance()
    assert seen and not any(seen), seen
