"""The below/below replay kernel's lanes and work items (momwire#1290).

`remainder_field_proj_batch_below` runs its blocked loop's stencil, stencil
read and projection four pairs to a vector in the AVX2 build, each lane doing
the scalar stage's own operations in its order, and hands out (observer row,
span of sources) work items dynamically instead of whole rows. Neither may
move a bit, so the gates are uint64:

- G-1290-L1  on every call of a live buried solve, the lanes are the scalar
             blocked loop (`lanes=False`) and the per-pair composition
             (`blocked=False`), and so are the three query extremes.
- G-1290-L2  the same at source counts around the block (64), the quad (4)
             and the span (1024), so short quads, short blocks and a short
             last span are all in the frame.
- G-1290-L3  synthetic geometry that crosses every band and zone edge and
             both clamps (theta under the floor and past pi/2's side, R1 past
             the table): the lanes' nested selects are the scalar ternaries.
- G-1290-L4  the items: any subset of observer rows is the full call's rows,
             so where a row's spans run cannot reach an entry.
- G-1290-L5  end to end, the buried Z with the lanes and with
             `_BELOW_LANES = False`, bit for bit, the lanes counted as run.

The scalar loops are the only route on the baseline, arm64 and MSVC builds
(`below_lanes_1290` False there), where L1-L4 hold trivially and still run.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire import _sommerfeld_below as below

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_field_galerkin_914 import screen_deck  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:crossing node"),
    pytest.mark.skipif(
        not hasattr(below._acc, "below_lanes_1290"),
        reason="the .so predates the below/below lanes: rebuild",
    ),
]


@pytest.fixture(scope="module")
def live_args():
    """Every argument tuple the kernel was handed during one buried solve."""
    seen = []
    real = below._acc.remainder_field_proj_batch_below

    def spy(*a, **kw):
        seen.append(a)
        return real(*a, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(below._acc, "remainder_field_proj_batch_below", spy)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            BSplineSolver(**screen_deck(12)).compute_impedance()
    finally:
        mp.undo()
    assert seen, "the solve never reached the below/below replay kernel"
    return real, seen


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _same(got, ref):
    """uint64 equality per part, with an unfilled region's NaNs (L3 reads
    tables the solve never filled) compared as NaN-for-NaN: which NaN payload
    an operation propagates is not part of the claim."""
    g = np.ascontiguousarray(got).view(np.float64)
    r = np.ascontiguousarray(ref).view(np.float64)
    nan = np.isnan(g)
    if not np.array_equal(nan, np.isnan(r)):
        return False
    return np.array_equal(g[~nan].view(np.uint64), r[~nan].view(np.uint64))


def _three(real, args):
    lanes = real(*args)
    scalar = real(*args, lanes=False)
    pair = real(*args, blocked=False)
    return lanes, scalar, pair


def test_g1290_l1_the_lanes_are_the_scalar_loop(live_args):
    real, seen = live_args
    for i, args in enumerate(seen):
        lanes, scalar, pair = _three(real, args)
        for ref in (scalar, pair):
            assert np.array_equal(_bits(lanes[0]), _bits(ref[0])), i
            assert lanes[1:] == ref[1:], (i, lanes[1:], ref[1:])
        assert np.all(np.isfinite(lanes[0])), i


@pytest.mark.parametrize("n_src", [1, 3, 4, 5, 63, 64, 65, 67, 1023, 1024, 1025, 2087])
def test_g1290_l2_every_block_and_span_shape(live_args, n_src):
    real, seen = live_args
    args = list(max(seen, key=lambda a: len(a[2])))
    src, t_src = args[2], args[3]
    # an evenly spread subset (with repeats past the source count), so the
    # short quad, block and span each hold far and near pairs
    take = np.linspace(0, len(src) - 1, n_src).astype(int)
    args[2], args[3] = src[take], t_src[take]
    lanes, scalar, pair = _three(real, args)
    assert lanes[0].shape[1] == n_src
    for ref in (scalar, pair):
        assert np.array_equal(_bits(lanes[0]), _bits(ref[0]))
        assert lanes[1:] == ref[1:]


def test_g1290_l3_every_edge_and_clamp(live_args):
    real, seen = live_args
    args = list(max(seen, key=lambda a: len(a[2])))
    ground_z = args[4]
    r1_max = args[11]
    rng = np.random.default_rng(1290)
    # Observers and sources from just under the plane to deep, horizontally
    # from on top of each other to past the table: theta from under the
    # grazing floor to pi/2, R1 from ~0 to past r1_max.
    n_obs, n_src = 9, 701
    reach = 1.5 * r1_max

    def points(n):
        p = np.empty((n, 3))
        rad = reach * rng.random(n) ** 2
        ang = 2 * np.pi * rng.random(n)
        p[:, 0], p[:, 1] = rad * np.cos(ang), rad * np.sin(ang)
        p[:, 2] = ground_z - 10.0 ** rng.uniform(-6, np.log10(reach), n)
        return p

    def tangents(n):
        t = rng.standard_normal((n, 3))
        t[: n // 7] = (0.0, 0.0, 1.0)  # vertical: the tangent's horizontal fallback
        return t / np.linalg.norm(t, axis=1)[:, None]

    args[0], args[1] = points(n_obs), tangents(n_obs)
    args[2], args[3] = points(n_src), tangents(n_src)
    # a source directly under an observer: rho = 0, the projection's
    # safe-rho fallback
    args[2][5] = args[0][3] - np.array([0.0, 0.0, 1.0])
    lanes, scalar, pair = _three(real, args)
    for ref in (scalar, pair):
        assert _same(lanes[0], ref[0])
        assert lanes[1:] == ref[1:]
    assert lanes[1] > r1_max  # the R1 clamp was exercised


def test_g1290_l4_rows_are_independent_of_the_items(live_args):
    real, seen = live_args
    args = list(max(seen, key=lambda a: len(a[2])))
    full = real(*args)[0]
    for rows in (slice(0, 1), slice(1, 4), slice(len(args[0]) - 3, None)):
        sub = list(args)
        sub[0], sub[1] = args[0][rows], args[1][rows]
        assert np.array_equal(_bits(real(*sub)[0]), _bits(full[rows]))


def test_g1290_l5_the_buried_z_with_and_without_the_lanes(monkeypatch):
    def fill(lanes):
        monkeypatch.setattr(below, "_BELOW_LANES", lanes)
        count = {"lanes": 0, "scalar": 0}
        real = below._acc.remainder_field_proj_batch_below

        def spy(*a, **kw):
            count["scalar" if kw.get("lanes", True) is False else "lanes"] += 1
            return real(*a, **kw)

        monkeypatch.setattr(below._acc, "remainder_field_proj_batch_below", spy)
        s = BSplineSolver(**screen_deck(12))
        geom = s._build_geometry()
        supp_seg, polys, *_ = s._build_basis_polynomials(geom)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = s._compute_Z_operator_buried(geom, supp_seg, polys)
        monkeypatch.undo()
        return Z, count

    Zl, cl = fill(True)
    Zs, cs = fill(False)
    assert cl["lanes"] > 0 and cl["scalar"] == 0, cl
    assert cs["scalar"] > 0 and cs["lanes"] == 0, cs
    assert np.array_equal(_bits(Zl), _bits(Zs))
