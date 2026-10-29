"""The keyed remainder runs in one C++ call (momwire#1752 K2).

`_remainder_pair_correction` hands every keyed run's base- and keyed-order
moments to `sommerfeld_remainder_bspline_runs` at once. Each request is the
single `sommerfeld_remainder_bspline_Q` call's arguments, so the batched fill
must be the per-call fill byte for byte, and must not depend on the thread
count.
"""

import warnings

import numpy as np
import pytest

from momwire import bspline as _bs
from momwire.bspline import BSplineSolver

from test_remainder_pair_order_1189 import surface_screen_deck

pytestmark = pytest.mark.skipif(
    _bs._acc is None or not hasattr(_bs._acc, "sommerfeld_remainder_bspline_runs"),
    reason="the extension has no sommerfeld_remainder_bspline_runs",
)


def _Z(monkeypatch, batched):
    if not batched:
        monkeypatch.setattr(
            BSplineSolver, "_remainder_runs_batched", lambda self, g, gr, req: None
        )
    calls = {"n": 0}
    orig = BSplineSolver._remainder_runs_batched

    def count(self, *a):
        calls["n"] += 1
        return orig(self, *a)

    if batched:
        monkeypatch.setattr(BSplineSolver, "_remainder_runs_batched", count)
    s = BSplineSolver(**surface_screen_deck(12))
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = np.array(s._compute_Z_operator(geom, supp_seg, polys), copy=True)
    monkeypatch.undo()
    return Z, calls["n"], s


def test_the_batched_runs_are_the_per_call_bytes(monkeypatch):
    zb, n, _s = _Z(monkeypatch, batched=True)
    assert n == 1, n  # one call for the whole correction
    zc, _n, _s = _Z(monkeypatch, batched=False)
    assert zb.tobytes() == zc.tobytes()
