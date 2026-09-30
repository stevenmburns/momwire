"""Bit-identity gate for the mixed fill's banded Φ reduction — momwire#1224.

`SinusoidalSolver._assemble_Z_mixed` used to reduce its three whole Φ
tables as `G = Φ₀@M₀; G += Φ₁@M₁; G += Φ₂@M₂`, each product a Φ-sized
copy plus a Φ-sized result inside scipy. `_reduce_phi_rows_in_place` runs
the same three products by observer-row bands, written into Φ₀'s own
storage. Above the dense threshold each M is a `csc_matrix`, and scipy's
`ndarray @ csc` sums each output element over its column's stored entries
along Φ's observer axis, so a band must be the same bits as the same rows
of the whole product.

Gates, on the real Φ and M of an inverted-L over 16 buried radials at
x = 4 (N = 734, `invl_deck`), captured from the solver's own call:
  * the banded result equals the old whole-matrix spelling byte-for-byte
    at band heights that force many bands (1, 7, 100, N − 1) and at one
    that does not (N, and larger);
  * a counting proxy on each M records the row count of every product, so
    the band path is proven to have run with exactly the expected bands —
    the test cannot pass with the reduction silently taken whole;
  * the solver's own call is on the sparse path and returns Φ₀'s storage.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
from test_crossing_serve_524 import invl_deck

from momwire import sinusoidal as sn


class _CountingM:
    """An M that records the observer rows of every `Φ_band @ M`.

    `__array_ufunc__ = None` makes numpy defer `ndarray @ self` to
    `__rmatmul__`, which then runs the exact product the solver runs."""

    __array_ufunc__ = None

    def __init__(self, M):
        self.M = M
        self.rows = []

    def __rmatmul__(self, other):
        self.rows.append(other.shape[0])
        return other @ self.M


@pytest.fixture(scope="module")
def captured():
    """The solver's own `(Φ, Ms, band)` at invl x4, copied on the way in,
    and what its call returned."""
    seen = {}
    orig = sn._reduce_phi_rows_in_place

    def spy(Phi, Ms, band):
        seen["Phi"] = [P.copy() for P in Phi]
        seen["Ms"] = list(Ms)
        seen["band"] = band
        seen["phi0_id"] = id(Phi[0])
        out = orig(Phi, Ms, band)
        seen["out_is_phi0"] = id(out) == seen["phi0_id"]
        seen["out"] = out.copy()
        return out

    mp = pytest.MonkeyPatch()
    mp.setattr(sn, "_reduce_phi_rows_in_place", spy)
    try:
        s = sn.SinusoidalSolver(**invl_deck(x=4))
        s.compute_impedance()
    finally:
        mp.undo()
    assert "Phi" in seen, "the mixed fill never reached the banded reduction"
    return seen


def _whole(Phi, Ms):
    """The pre-#1224 spelling: whole products, `(A + B) + C`."""
    G = Phi[0] @ Ms[0]
    G += Phi[1] @ Ms[1]
    G += Phi[2] @ Ms[2]
    return G


def test_solver_call_is_sparse_and_in_place(captured):
    N = captured["Phi"][0].shape[0]
    assert N == 734
    assert all(isinstance(M, scipy.sparse.csc_matrix) for M in captured["Ms"])
    assert captured["out_is_phi0"]
    # The solver's own result is the whole-matrix answer, bit for bit.
    ref = _whole(captured["Phi"], captured["Ms"])
    assert captured["out"].tobytes() == ref.tobytes()


@pytest.mark.parametrize("band", [1, 7, 100, 733, 734, 5000])
def test_banded_reduction_is_bit_equal(captured, band):
    Phi0, Ms0 = captured["Phi"], captured["Ms"]
    N = Phi0[0].shape[0]
    ref = _whole(Phi0, Ms0)

    Phi = [P.copy() for P in Phi0]
    Ms = [_CountingM(M) for M in Ms0]
    out = sn._reduce_phi_rows_in_place(Phi, Ms, band)

    expected = [min(band, N - r0) for r0 in range(0, N, band)]
    for M in Ms:
        assert M.rows == expected
    assert (len(expected) > 1) == (band < N)
    assert out is Phi[0]
    assert np.array_equal(out, ref)
    assert out.tobytes() == np.ascontiguousarray(ref).tobytes()
