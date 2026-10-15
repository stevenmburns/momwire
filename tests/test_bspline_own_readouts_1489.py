"""momwire#1489: bs2's B-spline evaluation without scipy.interpolate.

`_bspline_basis_rows` is scipy's design matrix reproduced: the same span
search and Cox-de Boor recurrence, in scipy's evaluation order. The basis
polynomials every bs2 fill is built from come out of it, so a roundoff
difference here would move every impedance. scipy's `BSpline` stays the
REFERENCE, imported inside each test only.
"""

import platform

import numpy as np
import pytest

from momwire.bspline import (
    _bspline_basis_rows,
    _bspline_design_matrix_dense,
)


_EPS = np.finfo(float).eps


def _clamped_knots(rng, k, n_seg, uniform):
    if uniform:
        h = np.full(n_seg, rng.uniform(0.01, 30.0) / n_seg)
    else:
        h = rng.uniform(0.001, 2.0, n_seg)
    arc = np.concatenate([[0.0], np.cumsum(h)])
    return arc, h, np.concatenate([np.zeros(k), arc, np.full(k, arc[-1])])


# scipy's compiled de Boor is fused into FMAs where the compiler may
# contract a*b + c (clang and gcc on arm64; x86-64 wheels target a baseline
# without FMA), and momwire's numpy evaluation never fuses -- the stance its
# own C++ takes with -ffp-contract=off. So the two are equal to the bit on
# x86-64 and only to a few ulps elsewhere; the BIT gate is where it can hold.
_SCIPY_UNFUSED = platform.machine().lower() in ("x86_64", "amd64")

# Worst |momwire - scipy| on a fusing platform, in eps (basis values lie in
# [0, 1]), measured over this test's own samples on macOS arm64 (GitHub's
# macos-14 runner, scipy 1.18.1, numpy 2.5.4): 0 / 0.5 / 1.5 at degree
# 1 / 2 / 3 (degree 1 has no a*b + c to fuse). The tolerance is the next
# power of two at or above twice the worst, the rule the readouts use below.
_FUSED_TOL_EPS = {1: 0.0, 2: 1.0, 3: 4.0}


@pytest.mark.parametrize("k", [1, 2, 3])
def test_basis_rows_are_scipys_design_matrix_to_the_bit(k):
    """Values AND first-basis offsets, bit for bit on x86-64 (a uint64 view,
    so even a zero's sign counts), on clamped knots like the solver's:
    uniform and non-uniform meshes, one to many segments, and the samples the
    solver takes (d+1 per segment, so every interior knot is hit exactly)
    plus every knot and random interior points. Where scipy's build fuses,
    the offsets stay exact and the values agree to the measured tolerance."""
    from scipy.interpolate import BSpline

    rng = np.random.default_rng(1489 + k)
    worst = 0.0
    for trial in range(300):
        n_seg = int(rng.integers(1, 40))
        arc, h, t = _clamped_knots(rng, k, n_seg, uniform=trial % 3 == 0)
        unit = np.linspace(0.0, 1.0, k + 1)
        x = np.concatenate(
            [
                (arc[:-1, None] + h[:, None] * unit[None, :]).ravel(),
                arc,
                rng.uniform(0.0, arc[-1], 40),
            ]
        )
        x = np.clip(x, 0.0, arc[-1])
        dm = BSpline.design_matrix(x, t, k)
        ref_vals = dm.data.reshape(-1, k + 1)
        ref_first = dm.indices.reshape(-1, k + 1)[:, 0]
        vals, first = _bspline_basis_rows(x, t, k)
        assert np.array_equal(first, ref_first)
        dense = _bspline_design_matrix_dense(x[:7], t, k)
        ref_dense = BSpline.design_matrix(x[:7], t, k).toarray()
        if _SCIPY_UNFUSED:
            assert np.array_equal(vals.view(np.uint64), ref_vals.view(np.uint64))
            assert np.array_equal(dense.view(np.uint64), ref_dense.view(np.uint64))
        worst = max(worst, float(np.abs(vals - ref_vals).max()) / _EPS)
        worst = max(worst, float(np.abs(dense - ref_dense).max()) / _EPS)
    if not _SCIPY_UNFUSED:
        assert worst <= _FUSED_TOL_EPS[k], f"degree {k}: {worst} eps"


def test_basis_rows_take_the_right_span_on_a_knot_and_the_last_at_the_end():
    """The span rule, spelled out at degree 1 where it is visible: on an
    interior knot the row starts at the basis whose support OPENS there; at
    the final knot it closes the last span."""
    t = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 3.0])
    vals, first = _bspline_basis_rows(np.array([0.0, 1.0, 2.0, 3.0]), t, 1)
    assert first.tolist() == [0, 1, 2, 2]
    assert vals.tolist() == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]


def test_basis_rows_refuse_what_scipys_design_matrix_refused():
    t = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    for bad in (2.0 + 1e-12, -1e-18):
        with pytest.raises(ValueError, match="outside the knot range"):
            _bspline_basis_rows(np.array([0.5, bad]), t, 2)
    with pytest.raises(ValueError, match="not a finite number"):
        _bspline_basis_rows(np.array([np.nan]), t, 2)
    vals, first = _bspline_basis_rows(np.zeros(0), t, 2)
    assert vals.shape == (0, 3)
    assert first.shape == (0,)
