"""Design D4's vector bracket, pinned against the scalar spelling it replaced.

razor's complex-k moment loop (momwire#796) evaluates e^{-jkR} - 1 at every
quadrature node. It used to call expm1, exp, sincos and sin(y/2) per node,
scalar, which made the complex fill 5x the real one per entry. D4 moved the
bracket into its own translation unit, `_accel_razor_cplx.cpp`, spelled with
the three functions glibc 2.28's libmvec vectorises (exp, sin, cos) and a
degree-13 Taylor series for expm1 above a = Im(k) R = -0.35.

That is a numerics change, not a refactor: libmvec's last bits are not scalar
libm's, and the series is not libm's expm1. So it is pinned here against the
scalar reference, `_stable.expm1_neg_jkR` -- the numpy twin of the loop D4
replaced, term for term -- at 1e-13 relative per node. The real bound is a few
ulp: the worst measured over this grid is 7.2e-16 (GCC 13, glibc 2.39,
AVX2). 1e-13 is the gate, and a wrong series misses it by orders of
magnitude.

The grid is dense where the series can go wrong: across the switch at
a = -0.35 from both sides, down to a -> 0 where expm1's relative accuracy is
the whole point, out to deep decay where exp(a) - 1 -> -1, and at the node
counts that exercise the padding (1..5, 64, 65).
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire._accel import acc as _acc
from momwire._stable import expm1_neg_jkR

pytestmark = pytest.mark.skipif(
    _acc is None or not hasattr(_acc, "razor_cplx_brackets"),
    reason="accelerator without the D4 bracket (razor_cplx_brackets)",
)

SWITCH = -0.35
BAR = 1e-13


def _rel(k, R):
    got = _acc.razor_cplx_brackets(R, k)
    ref = expm1_neg_jkR(np.complex128(k), R)
    return np.abs(got - ref) / np.abs(ref)


def _a_grid():
    """Values of a = Im(k) R <= 0 concentrated where the series switches."""
    near = SWITCH + np.linspace(-0.02, 0.02, 801)
    ulps = np.array([np.nextafter(SWITCH, -1.0), SWITCH, np.nextafter(SWITCH, 0.0)])
    small = -np.logspace(-12, np.log10(0.35), 400)
    large = -np.logspace(np.log10(0.35), 3, 400)
    return np.concatenate([near, ulps, small, large, [0.0]])


@pytest.mark.parametrize("k_re", [1e-6, 0.3, 0.58, 2.1, 40.0])
def test_bracket_matches_the_scalar_spelling_across_the_switch(k_re):
    # Choose R so that a = k_im * R lands on the grid for a fixed Im k. The
    # oscillation y = k_re * R then sweeps with it, from y ~ 0 to many turns.
    k_im = -0.238
    a = _a_grid()
    R = np.where(a == 0.0, 1e-3, a / k_im)
    rel = _rel(complex(k_re, k_im), R)
    assert np.all(np.isfinite(rel))
    assert rel.max() <= BAR, (
        f"max rel {rel.max():.2e} at a = {a[rel.argmax()]:.6g}, k_re = {k_re}"
    )


@pytest.mark.parametrize("k", [0.58 - 0.238j, 2.1 - 1e-9j, 30.0 - 25.0j, 1.0 - 0.0j])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 63, 64, 65, 131])
def test_every_node_count_is_served(k, n):
    # The kernel pads each chunk to a multiple of four and walks orders above
    # 64 in chunks; neither may change what a real node receives.
    R = np.linspace(1e-3, 7.0, n)
    rel = _rel(k, R)
    assert rel.max() <= BAR, f"n = {n}, k = {k}: max rel {rel.max():.2e}"
    # And the node's value does not depend on its neighbours or on n.
    one = np.array([_acc.razor_cplx_brackets(R[i : i + 1], k)[0] for i in range(n)])
    np.testing.assert_array_equal(_acc.razor_cplx_brackets(R, k), one)


def test_an_independent_oracle_agrees():
    # The scalar spelling is itself a few-ulp approximation. mpmath at 50
    # digits is not, so a shared defect in the two float spellings cannot
    # hide behind their agreement.
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    k = complex(0.58, -0.238)
    a = np.concatenate(
        [SWITCH + np.linspace(-1e-3, 1e-3, 21), -np.logspace(-10, 2, 60)]
    )
    R = a / k.imag
    got = _acc.razor_cplx_brackets(R, k)
    worst = 0.0
    for r, g in zip(R, got, strict=True):
        exact = complex(mp.expm1(-1j * mp.mpc(k.real, k.imag) * mp.mpf(float(r))))
        worst = max(worst, abs(g - exact) / abs(exact))
    assert worst <= BAR, f"max rel vs mpmath {worst:.2e}"
