"""The remainder order's per-source scan in C++ (`_accel_qp_pairs.cpp`).

`_quadrature.remainder_qp_pairs` lists the segment pairs whose remainder order
the geometry raises. Its numpy body was one Python iteration per source
segment; the scan answers every source it is CERTAIN of and hands the rest
back to that body. The list is an integer list, so the gate is equality, and
it is read on the decks whose sources sit exactly on the guard's edges (the
hub crossings, #1201) as well as on generic ones. Each gate also reads the
route counters, so a scan that silently never ran cannot pass.
"""

import numpy as np
import pytest

from momwire import _quadrature
from momwire import bspline as _bs
from momwire._accel import acc

from test_crossing_serve_524 import hub_deck
from test_remainder_pair_order_1189 import gnd_screen_deck

pytestmark = pytest.mark.skipif(
    acc is None or not hasattr(acc, "remainder_qp_pairs_scan"),
    reason="needs the C++ accelerator",
)


def _segments(build):
    seg_l, seg_r = [], []
    for w, npe in zip(build["wires"], build["n_per_edge_per_wire"]):
        pts = np.asarray(w, float)
        for e in range(len(pts) - 1):
            n = npe[e]
            t = np.linspace(0.0, 1.0, n + 1)
            p = pts[e] + t[:, None] * (pts[e + 1] - pts[e])
            seg_l.append(p[:-1])
            seg_r.append(p[1:])
    return np.concatenate(seg_l), np.concatenate(seg_r)


def _nodes(seg_l, seg_r, q):
    xg, _ = np.polynomial.legendre.leggauss(q)
    tq = 0.5 * (xg + 1.0)
    return seg_l[:, None, :] + tq[None, :, None] * (seg_r - seg_l)[:, None, :]


def _both(monkeypatch, *args, **kw):
    monkeypatch.setattr(_quadrature, "_QP_FORCE_NUMPY", True)
    ref = _quadrature.remainder_qp_pairs(*args, **kw)
    monkeypatch.setattr(_quadrature, "_QP_FORCE_NUMPY", False)
    before = dict(_quadrature._QP_SCAN_ROUTES)
    got = _quadrature.remainder_qp_pairs(*args, **kw)
    ran = _quadrature._QP_SCAN_ROUTES["cpp"] - before["cpp"]
    unsure = _quadrature._QP_SCAN_ROUTES["unsure"] - before["unsure"]
    return ref, got, ran, unsure


def _assert_same(ref, got):
    for a, b in zip(ref, got):
        assert a.dtype == b.dtype == np.int64
        np.testing.assert_array_equal(a, b)


def _buried_hub(x):
    d = hub_deck(n_radials=16)
    npe = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [[n * x for n in e] for e in npe]
    return d


DECKS = {
    "gnd_screen": gnd_screen_deck,
    "hub16": lambda: _buried_hub(1),
    "hub16x4": lambda: _buried_hub(4),
}


@pytest.mark.parametrize("name", sorted(DECKS))
@pytest.mark.parametrize("graded", [False, True])
def test_scan_lists_the_numpy_pairs(monkeypatch, name, graded):
    seg_l, seg_r = _segments(DECKS[name]())
    base = 3
    nodes = _nodes(seg_l, seg_r, base)
    c = _bs._REMAINDER_GRADED_C if graded else _bs._REMAINDER_QP_C
    tol = _bs._REMAINDER_GRADED_EDGE_TOL if graded else 0.0
    ref, got, ran, unsure = _both(
        monkeypatch,
        nodes,
        seg_l,
        seg_r,
        0.0,
        base,
        _bs._REMAINDER_QP_CAP,
        c,
        edge_tol=tol,
    )
    _assert_same(ref, got)
    # The scan ran, and it answered most sources itself.
    assert ran == 1
    assert unsure < seg_l.shape[0] // 4


def test_scan_on_random_near_ground_wires(monkeypatch):
    """Wires at random heights and tilts down to 1e-4 of their length over
    the plane: the generic case, many orders above base."""
    rng = np.random.default_rng(1201)
    n = 120
    a = rng.uniform(-5, 5, (n, 3))
    a[:, 2] = 10.0 ** rng.uniform(-4, 0, n)
    b = a + rng.normal(0, 1, (n, 3))
    b[:, 2] = np.abs(b[:, 2]) + 1e-4
    nodes = _nodes(a, b, 4)
    for tol in (0.0, _bs._REMAINDER_GRADED_EDGE_TOL):
        ref, got, ran, unsure = _both(
            monkeypatch,
            nodes,
            a,
            b,
            0.0,
            4,
            _bs._REMAINDER_QP_CAP,
            4.0,
            edge_tol=tol,
        )
        assert ref[0].size > 0
        _assert_same(ref, got)
        assert ran == 1


def test_uncertain_sources_take_the_numpy_body(monkeypatch):
    """An observer node exactly at a mirrored source's end makes t_raw
    exactly 0: the scan must hand that source back, not decide it."""
    seg_l = np.array([[0.0, 0.0, 0.01], [1.0, 0.0, 0.01]])
    seg_r = np.array([[1.0, 0.0, 0.01], [2.0, 0.0, 0.01]])
    # observer nodes on the mirror line's ends
    nodes = np.array(
        [
            [[0.0, 0.0, -0.01], [0.5, 0.0, -0.005]],
            [[1.0, 0.0, -0.01], [1.5, 0.0, -0.005]],
        ]
    )
    ref, got, ran, unsure = _both(
        monkeypatch, nodes, seg_l, seg_r, 0.0, 2, _bs._REMAINDER_QP_CAP, 4.0
    )
    _assert_same(ref, got)
    assert ran == 1 and unsure >= 1
