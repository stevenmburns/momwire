"""One singular-matrix policy for every dense LU route (momwire#1981).

`_lapack.lu_factor_strict` is `lu_factor` with the refusal `solve` already
had: an exactly zero pivot raises ``LinAlgError("singular matrix")``. Before
it, bspline's `_lu_solve` and SG's `_solve_in_place` refused while the
point-matched solve, the sector route's harmonic blocks and the array
preconditioner only took `lu_factor`'s warning and went on to inf/NaN.

What is held:

- the factors are `lu_factor`'s to the bit, in place or copied, and
  `factor_and_solve` is `scipy.linalg.solve`'s answer to the bit;
- with `MOMWIRE_CONDITION_CHECK` off no lange or gecon runs (momwire#1290's
  saving); on, an ill-conditioned matrix warns as `solve` does;
- every route raises the same type with the same message on a singular
  matrix: bspline / razor (`_lu_solve`), SG, the point-matched solve, the
  sector route's impedance and port solves, the array preconditioner.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg

from momwire import _lapack, _rotational_symmetry
from momwire import bspline as _bspline
from momwire import sinusoidal_galerkin as _sg
from momwire.array_block import _BlockJacobiAugPrecond
from momwire.sinusoidal import SinusoidalSolver

from test_rotational_symmetry_1029 import solver as screen_solver

SINGULAR = "singular matrix"


@pytest.fixture(autouse=True)
def _condition_off(monkeypatch):
    monkeypatch.delenv(_lapack.CONDITION_CHECK_ENV, raising=False)


def _rand(rng, n):
    return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))


def _singular(n=6, order="F"):
    A = np.ones((n, n), dtype=np.complex128)
    return np.asarray(A, order=order)


def _raises_singular(fn, *args, **kwargs):
    """`fn` raises the policy's error; `lu_factor`'s warning may precede it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
        with pytest.raises(np.linalg.LinAlgError) as exc:
            fn(*args, **kwargs)
    assert str(exc.value) == SINGULAR
    return exc.value


# ---- the policy --------------------------------------------------------------


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_the_factors_are_lu_factors_bits(order, overwrite):
    rng = np.random.default_rng(1981)
    A = np.asarray(_rand(rng, 40), order=order)
    ref_lu, ref_piv = _lapack.lu_factor(A.copy(order=order))
    lu, piv = _lapack.lu_factor_strict(A.copy(order=order), overwrite_a=overwrite)
    assert np.array_equal(lu, ref_lu)
    assert np.array_equal(piv, ref_piv)


def test_factor_and_solve_is_scipy_solve_to_the_bit():
    rng = np.random.default_rng(1290)
    A = _rand(rng, 50)
    b = rng.standard_normal((50, 3)) + 0j
    ref = scipy.linalg.solve(A, b)
    assert np.array_equal(_lapack.factor_and_solve(A.copy(), b), ref)
    F = np.asfortranarray(A)
    assert np.array_equal(
        _lapack.factor_and_solve(F, np.asfortranarray(b), True, True), ref
    )


def test_no_condition_estimate_unless_asked(monkeypatch):
    seen = []
    real = scipy.linalg.get_lapack_funcs

    def spy(names, *a, **k):
        seen.append(names)
        return real(names, *a, **k)

    monkeypatch.setattr(_lapack.scipy.linalg, "get_lapack_funcs", spy)
    rng = np.random.default_rng(3)
    A = _rand(rng, 30)
    _lapack.lu_factor_strict(A)
    assert not any(n in ("lange", "gecon") for n in seen)
    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1")
    lu_on, _ = _lapack.lu_factor_strict(A)
    assert "lange" in seen and "gecon" in seen
    assert np.array_equal(lu_on, _lapack.lu_factor(A)[0])


def test_the_condition_warning_is_opt_in(monkeypatch):
    rng = np.random.default_rng(19)
    q1, _ = np.linalg.qr(_rand(rng, 30))
    q2, _ = np.linalg.qr(_rand(rng, 30))
    ill = (q1 * np.logspace(0, -19, 30)) @ q2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _lapack.lu_factor_strict(ill)
    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1")
    with pytest.warns(scipy.linalg.LinAlgWarning, match="ill-conditioned"):
        _lapack.lu_factor_strict(ill)


def test_a_singular_matrix_raises_with_or_without_the_estimate(monkeypatch):
    _raises_singular(_lapack.lu_factor_strict, _singular())
    _raises_singular(_lapack.factor_and_solve, _singular(), np.ones(6))
    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1")
    _raises_singular(_lapack.lu_factor_strict, _singular())


def test_solve_says_the_same_sentence():
    """The refusal the policy copies: `_lapack.solve`'s own, on a general
    matrix (a zero column keeps its pivot exactly zero; a structured one,
    like all ones, goes to scipy, whose wording differs)."""
    rng = np.random.default_rng(1981)
    A = _rand(rng, 6)
    A[:, 1] = 0
    _raises_singular(_lapack.solve, A, np.ones(6))
    _raises_singular(_lapack.lu_factor_strict, A)


# ---- every route -------------------------------------------------------------


def test_bspline_and_razor_lu_solve():
    _raises_singular(_bspline._lu_solve, _singular(), np.ones(6), overwrite_a=True)
    _raises_singular(_bspline._lu_solve, _singular(order="C"), np.ones(6))


def test_sg_solve_in_place():
    _raises_singular(_sg._solve_in_place, _singular(), np.ones(6))


def _dipole(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SinusoidalSolver(
            wires=[np.array([(0.0, 0.0, -0.24), (0.0, 0.0, 0.24)])],
            n_per_edge_per_wire=[[11]],
            wire_radius=1e-3,
            feeds=[(0, None, 1.0 + 0j)],
            wavelength=1.0,
            **kw,
        )


def test_the_point_matched_solve(monkeypatch):
    """It returned inf/NaN impedance before; now it refuses as the rest do."""
    s = _dipole()
    real = s._assemble_Z

    def zero(*a, **k):
        G, view = real(*a, **k)
        return np.zeros_like(G), view

    monkeypatch.setattr(s, "_assemble_Z", zero)
    _raises_singular(s.compute_impedance)
    assert getattr(s, "Z_factors", None) is None  # nothing stashed


def _zero_sector_fill(monkeypatch):
    real = _rotational_symmetry.sector_fill

    def zero(*a, **k):
        Z, row_of = real(*a, **k)
        return np.zeros_like(Z), row_of

    monkeypatch.setattr(_rotational_symmetry, "sector_fill", zero)


@pytest.mark.parametrize("kcl_nullspace", [False, True])
def test_the_sector_route_impedance(monkeypatch, kcl_nullspace):
    _zero_sector_fill(monkeypatch)
    _raises_singular(screen_solver(kcl_nullspace=kcl_nullspace).compute_impedance)


@pytest.mark.parametrize("kcl_nullspace", [False, True])
def test_the_sector_route_port_solution(monkeypatch, kcl_nullspace):
    _zero_sector_fill(monkeypatch)
    s = screen_solver(kcl_nullspace=kcl_nullspace)
    _raises_singular(s.compute_port_solution)


def test_the_array_preconditioner():
    block = SimpleNamespace(
        n=6,
        groups=[np.arange(6)],
        shape_of_elem=np.zeros(1, dtype=np.int64),
        shape_blocks=[np.ones((6, 6), dtype=np.complex128)],
    )
    _raises_singular(_BlockJacobiAugPrecond, block, None)
