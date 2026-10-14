"""The frozen Gauss-Legendre table (`momwire._quadrature`, `_leggauss_table`).

Three claims, each its own test:

* the table is numpy's own rule, bit for bit, on the platform it was
  generated on (Linux x86_64) — so freezing it moved nothing there;
* every tabled rule is a Gauss-Legendre rule, on every platform — exact
  for polynomials of degree 2n-1 to rounding;
* every caller in the package goes through `_quadrature.leggauss`, so no
  solver path can quietly reach the host's eigensolver again.

On macOS arm64 the first claim does NOT hold for numpy (its eigensolve
differs in the last ulp: CI's macOS lane showed order 12 move), which is
the reason the table exists; the test skips there and says so.
"""

import platform
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import _leggauss_table
from momwire._quadrature import leggauss

SRC = Path(__file__).resolve().parent.parent / "src" / "momwire"
ORDERS = range(1, _leggauss_table.N_MAX + 1)
REFERENCE = sys.platform == "linux" and platform.machine() == "x86_64"


def test_table_covers_one_to_n_max():
    assert sorted(_leggauss_table.HALVES) == list(ORDERS)


def _ulps(a, b):
    """Largest distance between `a` and `b` in units in the last place."""
    ia = a.view(np.int64).astype(np.int64)
    ib = b.view(np.int64).astype(np.int64)
    # Map the sign-magnitude bit patterns onto one monotone integer line.
    ia = np.where(ia < 0, np.int64(-(2**63)) - ia, ia)
    ib = np.where(ib < 0, np.int64(-(2**63)) - ib, ib)
    return int(np.max(np.abs(ia - ib))) if a.size else 0


def test_table_is_numpys_rule_to_the_bit():
    moved = {}
    for n in ORDERS:
        x, w = leggauss(n)
        rx, rw = np.polynomial.legendre.leggauss(n)
        if x.tobytes() != rx.tobytes() or w.tobytes() != rw.tobytes():
            moved[n] = max(_ulps(x, rx), _ulps(w, rw))
    if not REFERENCE:
        # Off the reference platform the difference IS the point of the table;
        # say how big it is (pytest's warnings summary carries it to the CI
        # log) rather than fail.
        worst = max(moved.values(), default=0)
        warnings.warn(
            f"leggauss table vs numpy {np.__version__} on {sys.platform}/"
            f"{platform.machine()}: {len(moved)} of {len(ORDERS)} orders differ, "
            f"worst {worst} ulp; orders {sorted(moved)}",
            stacklevel=1,
        )
        pytest.skip("numpy's rule is the table's only on Linux x86_64")
    assert not moved, (
        f"orders {sorted(moved)} differ from numpy {np.__version__} here (ulps "
        f"{moved}); the table was generated with numpy "
        f"{_leggauss_table.NUMPY_VERSION}"
    )


@pytest.mark.parametrize("n", ORDERS)
def test_tabled_rule_is_gauss_legendre(n):
    x, w = leggauss(n)
    assert x.shape == w.shape == (n,)
    assert np.all(np.diff(x) > 0) and -1 < x[0] and x[-1] < 1
    assert np.array_equal(x, -x[::-1]) and np.array_equal(w, w[::-1])
    assert np.all(w > 0)
    # Exact for every monomial up to degree 2n - 1: the integral of t^k over
    # [-1, 1] is 2/(k+1) for even k, 0 for odd k.
    for k in range(2 * n):
        exact = 2.0 / (k + 1) if k % 2 == 0 else 0.0
        assert abs(np.sum(w * x**k) - exact) <= 64 * n * np.finfo(float).eps


def test_rules_are_read_only_and_cached():
    x, w = leggauss(12)
    assert not x.flags.writeable and not w.flags.writeable
    assert leggauss(12)[0] is x and leggauss(np.int64(12))[0] is x


def test_beyond_the_table_falls_back_to_numpy():
    n = _leggauss_table.N_MAX + 1
    x, w = leggauss(n)
    rx, rw = np.polynomial.legendre.leggauss(n)
    assert x.tobytes() == rx.tobytes() and w.tobytes() == rw.tobytes()


def test_no_module_calls_numpys_leggauss_directly():
    allowed = {"_quadrature.py"}
    pattern = re.compile(
        r"legendre\.leggauss\(|from numpy\.polynomial\.legendre import"
    )
    hits = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if p.name not in allowed and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert not hits, f"route these through momwire._quadrature.leggauss: {hits}"


def test_table_reaches_every_capped_order():
    # The keyed remainder orders run up to their caps, and the buried field's
    # separation rule up to its factor cap times its base order: every order in
    # between is reachable, so the table is contiguous to the largest cap.
    from momwire import _below_interface, bspline, razor

    caps = [
        bspline._REMAINDER_QP_CAP,
        razor._REMAINDER_QP_CAP,
        _below_interface.N_QP_BURIED_FIELD * _below_interface._MAX_NEAR_Q_FACTOR,
    ]
    assert max(caps) <= _leggauss_table.N_MAX
