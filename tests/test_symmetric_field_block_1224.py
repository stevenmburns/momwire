"""The below/below field block projects each unordered pair once (momwire#1224).

`_field_galerkin_block_symmetric` serves the square below/below block of a
buried fill: its observer and source axes are one node set, the projected
remainder table is reciprocal, and so chunk c projects only against sources
from its own first segment on and assembles that table twice -- into Q, and
with the chunk's own columns masked into Q.T. The rectangular route stays
reachable (`_FIELD_GALERKIN_SYMMETRIC = False`) and is the reference.

Gates:

- G-1224-S1  the license: on a real chunk the projected table equals its
             transpose to rounding, measured by swapping the kernel's
             observer and source arguments.
- G-1224-S2  the buried Z: halved route against the rectangle at 1e-12 of
             max|Z|, NOT bit-identical (a route that never ran would be), and
             the halved route taken over more than one chunk.
- G-1224-S3  the projection count: the halved route asks the kernel for
             1/2 + chunk/(2n) of the rectangle's pairs, counted at the
             projector rather than inferred from a timing.
- G-1224-S4  `symmetric=True` over two different axes is refused, not
             assembled.
- G-1224-S5  an observer restriction (`rows=`) keeps the rectangle: the
             restricted fill is the reference route's rows bit for bit.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import momwire.bspline as _bs
from momwire import BSplineSolver, _sommerfeld_below

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_field_galerkin_914 import screen_deck  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:crossing node"),
    pytest.mark.skipif(
        not _bs._HAVE_FIELD_GALERKIN_STRIDED,
        reason="the halved route needs the strided field-Galerkin target",
    ),
]


def _fill(symmetric, monkeypatch, rows=None):
    """The buried Z of the 12-radial screen through one route, plus a record
    of every projector call (observer count, source count) and how many
    times the halved route ran."""
    monkeypatch.setattr(_bs, "_FIELD_GALERKIN_SYMMETRIC", symmetric)
    calls = {"sym": 0, "proj": []}
    real_sym = BSplineSolver._field_galerkin_block_symmetric
    real_proj = _sommerfeld_below.remainder_field_proj_below

    def spy_sym(self, *a, **kw):
        calls["sym"] += 1
        return real_sym(self, *a, **kw)

    def spy_proj(obs, t_obs, src, *a, **kw):
        calls["proj"].append((len(obs), len(src)))
        return real_proj(obs, t_obs, src, *a, **kw)

    monkeypatch.setattr(BSplineSolver, "_field_galerkin_block_symmetric", spy_sym)
    monkeypatch.setattr(_sommerfeld_below, "remainder_field_proj_below", spy_proj)
    s = BSplineSolver(**screen_deck(12))
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = s._compute_Z_operator_buried(geom, supp_seg, polys, rows=rows)
    return Z, calls


@pytest.fixture(scope="module")
def both():
    mp = pytest.MonkeyPatch()
    try:
        Zs, cs = _fill(True, mp)
        mp.undo()
        Zr, cr = _fill(False, mp)
    finally:
        mp.undo()
    return Zs, cs, Zr, cr


def test_g1224_s1_the_projected_table_is_reciprocal():
    s = BSplineSolver(**screen_deck(12))
    seen = []
    real = _sommerfeld_below.remainder_field_proj_below

    def grab(obs, t_obs, src, t_src, *rest):
        if not seen:
            seen.append((obs, t_obs, src, t_src, rest))
        return real(obs, t_obs, src, t_src, *rest)

    mp = pytest.MonkeyPatch()
    mp.setattr(_sommerfeld_below, "remainder_field_proj_below", grab)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.compute_impedance()
    finally:
        mp.undo()
    obs, t_obs, src, t_src, rest = seen[0]
    P = real(obs, t_obs, src, t_src, *rest)
    PT = real(src, t_src, obs, t_obs, *rest).T
    rel = float(np.max(np.abs(P - PT)) / np.max(np.abs(P)))
    assert rel < 1e-14, rel


def test_g1224_s2_the_halved_route_is_the_rectangle(both):
    Zs, cs, Zr, cr = both
    assert cs["sym"] == 1 and cr["sym"] == 0, (cs["sym"], cr["sym"])
    # more than one chunk, or the mirror's masking was never exercised
    assert len(cs["proj"]) > 1, cs["proj"]
    rel = float(np.max(np.abs(Zs - Zr)) / np.max(np.abs(Zr)))
    assert rel <= 1e-12, rel
    assert not np.array_equal(Zs, Zr), "bit-identical: the halved route never ran"


def test_g1224_s3_the_kernel_is_asked_for_half_the_pairs(both):
    _Zs, cs, _Zr, cr = both
    full = sum(m * n for m, n in cr["proj"])
    half = sum(m * n for m, n in cs["proj"])
    n_src = cr["proj"][0][1]
    chunk = cr["proj"][0][0]
    # sum over chunks of c * (n - i0) is n^2/2 + n*chunk/2, on the node axis
    expect = 0.5 + chunk / (2.0 * n_src)
    assert half / full <= expect + 1e-9, (half / full, expect)
    assert half / full >= 0.5, half / full


def test_g1224_s4_two_different_axes_are_refused():
    s = BSplineSolver(**screen_deck(12))
    nodes = np.zeros((6, 3))
    W = np.ones((3, 1, 6))
    other = nodes + 1.0
    with pytest.raises(ValueError, match="same nodes"):
        s._field_galerkin_block(
            np.zeros((1, 3), dtype=np.int64),
            np.zeros((1, 3, 3)),
            None,
            np.array([0]),
            np.array([0]),
            nodes,
            nodes,
            W,
            other,
            nodes,
            W,
            symmetric=True,
        )


def test_g1224_s5_an_observer_restriction_keeps_the_rectangle(monkeypatch):
    rows = np.arange(0, 40, dtype=np.int64)
    Zs, cs = _fill(True, monkeypatch, rows=rows)
    monkeypatch.undo()
    Zr, _cr = _fill(False, monkeypatch, rows=rows)
    assert cs["sym"] == 0
    assert np.array_equal(Zs, Zr)
