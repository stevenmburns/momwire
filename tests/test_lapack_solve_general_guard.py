"""`_lapack.solve`'s no-host fallback runs SciPy's getrf + getrs directly
only on a matrix `scipy.linalg.solve` provably solves as general; every
structured matrix SciPy >= 1.18 detects (diagonal, tridiagonal, banded,
triangular, symmetric, Hermitian) goes to SciPy unchanged. Both ways the
answer is bit-identical to `scipy.linalg.solve`."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from momwire import _lapack


@pytest.fixture(autouse=True)
def _no_host(monkeypatch):
    # the guard governs the no-host fallback; pin that branch
    monkeypatch.setattr(_lapack, "_host_for", lambda dtype: None)


def _spy(monkeypatch):
    calls = []
    original = scipy.linalg.solve

    def counting(*a, **k):
        calls.append(1)
        return original(*a, **k)

    monkeypatch.setattr(_lapack.scipy.linalg, "solve", counting)
    return calls, original


def _general(n, dtype, rng):
    a = rng.standard_normal((n, n))
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.standard_normal((n, n))
    return (a + n * np.eye(n)).astype(dtype)


def _rhs(n, k, dtype, rng):
    b = rng.standard_normal((n, k) if k else n)
    if np.issubdtype(dtype, np.complexfloating):
        b = b + 1j * rng.standard_normal(b.shape)
    return b.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.complex64])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_a_general_matrix_takes_getrf_and_matches_scipy_bit_for_bit(
    monkeypatch, dtype, k
):
    rng = np.random.default_rng(1)
    a, b = _general(300, dtype, rng), _rhs(300, k, dtype, rng)
    calls, original = _spy(monkeypatch)
    x = _lapack.solve(a, b)
    assert calls == []  # the direct getrf + getrs path ran
    ref = original(a, b)
    assert x.dtype == ref.dtype and x.shape == ref.shape
    assert np.array_equal(x, ref)


def _structured(kind, n, rng):
    a = _general(n, np.complex128, rng)
    if kind == "symmetric":
        return (a + a.T) / 2
    if kind == "hermitian":
        return (a + a.conj().T) / 2 + n * np.eye(n)
    if kind == "upper":
        return np.triu(a)
    if kind == "lower":
        return np.tril(a)
    if kind == "diagonal":
        return np.diag(np.diag(a))
    if kind == "tridiagonal":
        return np.triu(np.tril(a, 1), -1)
    if kind == "banded":
        return np.triu(np.tril(a, 5), -7)
    if kind == "corner-zero":  # full except one corner: banded upper side
        a = a.copy()
        a[0, n - 1] = 0
        return a
    if kind == "first-row-symmetric":  # can't be proven general cheaply
        a = a.copy()
        a[0, 1:] = a[1:, 0]
        return a
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    [
        "symmetric",
        "hermitian",
        "upper",
        "lower",
        "diagonal",
        "tridiagonal",
        "banded",
        "corner-zero",
        "first-row-symmetric",
    ],
)
def test_anything_not_proven_general_goes_to_scipy_unchanged(monkeypatch, kind):
    rng = np.random.default_rng(2)
    a = _structured(kind, 120, rng)
    b = _rhs(120, 2, np.complex128, rng)
    calls, original = _spy(monkeypatch)
    x = _lapack.solve(a, b)
    assert calls == [1]
    assert np.array_equal(x, original(a, b))


def test_a_singular_general_matrix_raises_as_scipy_does():
    rng = np.random.default_rng(3)
    a = _general(50, np.complex128, rng)
    a[:, 7] = 0  # an exact zero pivot; the matrix is still unstructured
    assert _lapack._scipy_takes_gesv(a, a.dtype)
    b = np.ones(50, np.complex128)
    with pytest.raises(np.linalg.LinAlgError):
        scipy.linalg.solve(a, b)
    with pytest.raises(np.linalg.LinAlgError):
        _lapack.solve(a, b)
