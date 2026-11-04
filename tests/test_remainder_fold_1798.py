"""The Sommerfeld remainder folds into Z with no n_basis^2 Q, to the bit (momwire#1798).

`_Z_sommerfeld_remainder_into` subtracts (or adds) each finished entry of the
dense remainder block as the fused kernel's symmetric route forms it, where
the dense route materialised Q beside Z and then took `Z -= Q`: 2.5 GB of the
ELT whip's 5.0 GB peak. The gate is byte equality with the dense route, in
both signs and both layouts (the chunked fill's Z is Fortran order, the dense
fill's C order), on decks with and without #1189's raised pairs -- the
entries those pairs correct are exempted from the fold, corrected on the side
and folded last, which is the one place the order could slip.

`_fortran_in_place` is the LU half: a C-order Z rearranged in its own buffer
so that `lu_factor(overwrite_a=True)` factors it without its Fortran copy.

Asserted bit for bit where the build does not contract a*b+c on its own
(GCC/clang, -ffp-contract=off); MSVC's /fp:fast gets a roundoff tolerance.
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


def _need_kernel():
    if _bs._acc is None or not hasattr(_bs._acc, "sommerfeld_remainder_bspline_Q_into"):
        pytest.skip("fused sommerfeld fold kernel unavailable")


def _same(got, ref):
    got = np.ascontiguousarray(got)
    ref = np.ascontiguousarray(ref)
    if EXACT:
        assert got.tobytes() == ref.tobytes(), (
            f"the fold moved bits: max abs {np.abs(got - ref).max():.3e}"
        )
    else:
        assert np.abs(got - ref).max() <= 1e-13 * np.abs(ref).max()


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(
    "name, frac, raised",
    [("dipole", 0.05, False), ("yagi", 0.2, False), ("inverted_l", 0.02, True)],
)
def test_fold_is_the_dense_route(name, frac, raised, sign, order):
    _need_kernel()
    s = BSplineSolver(**dict(GEOMS[(name, frac)], ground_z=0.0, **SOMM))
    geom = s._build_geometry()
    supp_seg, polys, *_rest = s._build_basis_polynomials(geom)
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    pairs = s._remainder_qp_pairs(geom["seg_l"], geom["seg_r"], s.ground_z)
    assert (pairs is not None and pairs[0].size > 0) == raised
    Q = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)
    nb = Q.shape[0]
    rng = np.random.default_rng(1798)
    Z0 = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
    ref = Z0 - Q if sign < 0 else Z0 + Q
    T = np.array(Z0, order=order)
    n0 = _bs._acc.remainder_q_symmetric_calls()
    out = s._Z_sommerfeld_remainder_into(T, sign, geom, supp_seg, polys, eps_t)
    assert out is T
    assert _bs._acc.remainder_q_symmetric_calls() == n0 + 1  # the fused fold ran
    _same(T, ref)


def test_fold_rejects_a_copying_target():
    """T is written in place: a target the binding would have to convert
    (here complex64) is refused rather than silently filled as a copy."""
    _need_kernel()
    s = BSplineSolver(**dict(GEOMS[("dipole", 0.05)], ground_z=0.0, **SOMM))
    geom = s._build_geometry()
    supp_seg, polys, *_rest = s._build_basis_polynomials(geom)
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    nb = supp_seg.shape[0]
    with pytest.raises(RuntimeError, match="complex128"):
        s._Z_sommerfeld_remainder_into(
            np.zeros((nb, nb), dtype=np.complex64), -1, geom, supp_seg, polys, eps_t
        )


@pytest.mark.parametrize("n", [1, 7, 256, 300, 600])
def test_fortran_in_place_is_the_fortran_copy(n):
    rng = np.random.default_rng(n)
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    ref = np.asfortranarray(Z)
    buf = Z.__array_interface__["data"][0]
    F = _bs._fortran_in_place(Z)
    assert F.flags.f_contiguous
    assert F.__array_interface__["data"][0] == buf  # Z's own buffer
    assert np.asarray(F).tobytes(order="F") == ref.tobytes(order="F")
    already = np.asfortranarray(ref)
    assert _bs._fortran_in_place(already) is already
