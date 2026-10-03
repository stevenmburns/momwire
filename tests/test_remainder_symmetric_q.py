"""The fused remainder kernel's symmetric route is the banded kernel, bit for bit.

`sommerfeld_remainder_bspline_Q` serves the dense block (obs == src) by tiling
the basis axis and evaluating each unordered node pair's grid interpolation
once for both orientations (see the block comment above
`remainder_Q_symmetric` in `_accel_somm.cpp`). Nothing is approximated: the
shared half of the projection reads the pair only through rho and hh, which
are bit-identical under the swap, and every Jf and Q entry is then formed by
the banded kernel's own expressions in the banded kernel's order. So the gate
is equality with `reference=True` (the banded kernel), on decks that reach
every branch of the share: a node over its own segment (rho = 0, the
tangent-direction fallback), vertical and horizontal wires, a junction whose
support map is not one contiguous run, both degrees, and budgets small enough
to shrink the tiles to one basis and the bands to one segment.

The equality is asserted where the build is known not to contract a*b+c on
its own (GCC/clang, -ffp-contract=off, momwire#1194). MSVC's /fp:fast may
contract the two routes' identical expressions differently, so there it is a
reassociation-sized tolerance instead.
"""

import sys

import numpy as np
import pytest

from momwire import _ground_refl
from momwire import bspline as _bs
from momwire.bspline import BSplineSolver

from fixtures_refl_coef_geoms import GEOMS

SOMM = {"ground_eps": (10.0, 0.002), "ground_model": "sommerfeld"}
EXACT = sys.platform != "win32"


class _Kw:
    """`_acc` proxy adding keyword arguments to the fused kernel's calls."""

    def __init__(self, real, **kw):
        self._real = real
        self._kw = kw

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if name != "sommerfeld_remainder_bspline_Q":
            return attr

        def wrapped(*args, **kwargs):
            kwargs.update(self._kw)
            return attr(*args, **kwargs)

        return wrapped


def _q(monkeypatch, s, real, **kw):
    geom = s._build_geometry()
    supp_seg, polys, *_rest = s._build_basis_polynomials(geom)
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    monkeypatch.setattr(_bs, "_acc", _Kw(real, **kw))
    return s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)


def _assert_same(got, ref):
    if EXACT:
        assert np.array_equal(got, ref), (
            "symmetric route moved bits: max rel "
            f"{np.abs(got - ref).max() / np.abs(ref).max():.3e}"
        )
    else:
        assert np.abs(got - ref).max() <= 1e-13 * np.abs(ref).max()


def _need_kernel():
    if _bs._acc is None or not hasattr(_bs._acc, "sommerfeld_remainder_bspline_Q"):
        pytest.skip("fused sommerfeld kernel unavailable")
    assert hasattr(_bs._acc, "remainder_q_symmetric_calls"), (
        "the .so predates the symmetric route: rebuild (`make build`)"
    )


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    "name, frac", [("dipole", 0.05), ("inverted_l", 0.1), ("yagi", 0.2)]
)
def test_symmetric_route_is_the_banded_kernel(name, frac, degree, monkeypatch):
    _need_kernel()
    real = _bs._acc
    kw = dict(GEOMS[(name, frac)], ground_z=0.0, degree=degree, **SOMM)
    s = BSplineSolver(**kw)
    n0 = real.remainder_q_symmetric_calls()
    ref = _q(monkeypatch, s, real, reference=True)
    assert real.remainder_q_symmetric_calls() == n0  # reference: banded only
    got = _q(monkeypatch, s, real)
    assert real.remainder_q_symmetric_calls() == n0 + 1  # the route ran
    assert np.any(ref != 0.0)
    _assert_same(got, ref)


# 1 byte: one-basis tiles AND one-segment bands, so every support straddles a
# tile and a band boundary. 1 << 16 and 1 << 20 walk both up through partial
# straddling.
@pytest.mark.parametrize("budget", [1, 1 << 16, 1 << 20])
def test_symmetric_route_under_small_budgets(budget, monkeypatch):
    _need_kernel()
    real = _bs._acc
    s = BSplineSolver(**dict(GEOMS[("inverted_l", 0.1)], ground_z=0.0, **SOMM))
    ref = _q(monkeypatch, s, real, reference=True, max_jf_bytes=budget)
    n0 = real.remainder_q_symmetric_calls()
    got = _q(monkeypatch, s, real, max_jf_bytes=budget)
    assert real.remainder_q_symmetric_calls() == n0 + 1
    _assert_same(got, ref)


def test_rectangular_calls_keep_the_banded_kernel():
    """A rows-only call (obs != src) takes the banded kernel, and its rows
    are the dense block's rows -- which now come from the symmetric route --
    so this holds the two routes together through the production seam
    (`_ObserverRows`, momwire#1131), not only through `reference=`."""
    _need_kernel()
    kw = dict(GEOMS[("yagi", 0.2)], ground_z=0.0, **SOMM)
    s = BSplineSolver(**kw)
    geom = s._build_geometry()
    supp_seg, polys, *_rest = s._build_basis_polynomials(geom)
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    n0 = _bs._acc.remainder_q_symmetric_calls()
    full = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)
    assert _bs._acc.remainder_q_symmetric_calls() == n0 + 1

    # Wires are laid out segment-contiguously; request the middle one whole.
    counts = [int(np.sum(e)) for e in kw["n_per_edge_per_wire"]]
    start = counts[0]
    seg_rows = np.arange(start, start + counts[1], dtype=np.int64)
    restrict = _bs._ObserverRows(
        seg_rows, supp_seg, polys, geom["n_segs_total"], compact=True
    )
    assert 0 < restrict.basis_rows.size < supp_seg.shape[0]
    part = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t, restrict=restrict)
    assert _bs._acc.remainder_q_symmetric_calls() == n0 + 1  # rows: banded
    _assert_same(part, full[restrict.basis_rows])
