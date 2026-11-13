"""`_lapack.solve` prepares A in one threaded pass (momwire#1916 part A):
`fortran_copy_1norm` is the Fortran copy, the finiteness check and lange's
1-norm at once. None of it decides a value, so:

- the copy is numpy's Fortran copy, byte for byte;
- the norm is scipy's `lange("1")`, bit for bit, at every thread count;
- the solve's answer is the unprepared solve's, bit for bit, on both routes;
- a non-finite A still raises, and the ill-conditioning warning still fires
  exactly where it fired.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg

from momwire import _lapack
from momwire._accel import acc

pytestmark = pytest.mark.skipif(
    acc is None or not getattr(acc, "solve_prep_1916", False),
    reason="accelerator None or without solve_prep_1916",
)


@pytest.fixture(autouse=True)
def _condition_check_on(monkeypatch):
    """These tests hold `solve` to scipy's contract, warning included, so the
    opt-in condition estimate is on (`_lapack.condition_check`)."""
    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1")


def _rand(rng, shape, dtype):
    a = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (64, 64), (130, 77), (77, 130), (300, 300)])
def test_copy_and_norm_are_numpys_and_lapacks_bits(dtype, shape):
    rng = np.random.default_rng(1916)
    a = _rand(rng, shape, dtype)
    # A wide dynamic range, so the sums' order would show if it moved.
    a *= np.exp(rng.uniform(-30, 30, shape))
    ref = np.array(a, order="F", copy=True)
    lange = scipy.linalg.get_lapack_funcs("lange", (ref,))
    want = lange("1", ref)
    for nt in (1, 3, 0):
        f, finite, anorm = acc.fortran_copy_1norm(a, nt)
        assert f.flags.f_contiguous and f.dtype == a.dtype and f.shape == a.shape
        assert f.tobytes(order="A") == ref.tobytes(order="A")
        assert finite
        assert anorm == want, (nt, anorm, want)


@pytest.mark.parametrize("where", ["re-nan", "im-inf", "re-ninf"])
def test_a_non_finite_entry_is_seen(where):
    rng = np.random.default_rng(2)
    a = _rand(rng, (100, 100), np.complex128)
    v = {"re-nan": complex(np.nan, 0.0), "im-inf": complex(0.0, np.inf)}.get(
        where, complex(-np.inf, 1.0)
    )
    a[97, 3] = v
    _, finite, _ = acc.fortran_copy_1norm(a)
    assert not finite


def _outcome(fn, *args):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            x = fn(*args)
            exc = None
        except Exception as e:  # noqa: BLE001 - the type IS the observation
            x, exc = None, type(e)
    return x, exc, sorted({w.category.__name__ for w in seen})


def _ill(rng, n, dtype):
    q1, _ = np.linalg.qr(_rand(rng, (n, n), dtype))
    q2, _ = np.linalg.qr(_rand(rng, (n, n), dtype))
    return np.ascontiguousarray((q1 * np.logspace(0, -19, n)) @ q2)


def _cases(dtype):
    rng = np.random.default_rng(3)
    n = 200
    fine = _rand(rng, (n, n), dtype) + n * np.eye(n, dtype=dtype)
    nan = fine.copy()
    nan[150, 7] = np.nan
    b = _rand(rng, (n,), dtype)
    return {
        "fine": (fine, b),
        "fine, many rhs": (fine, _rand(rng, (n, 3), dtype)),
        "ill": (_ill(rng, n, dtype), b),
        "nan": (nan, b),
        "nan rhs": (fine, np.where(np.arange(n) == 5, np.nan, b)),
    }


@pytest.fixture(params=["default", "host"])
def route(request, monkeypatch):
    if request.param == "default":
        monkeypatch.setattr(_lapack, "_host_for", lambda dtype: None)
    else:
        # scipy's own getrf / getrs, through the host route's code path.
        monkeypatch.setattr(_lapack, "_host_for", lambda dtype: "scipy")

        def getrf(fns, a):
            f = scipy.linalg.get_lapack_funcs("getrf", (a,))
            lu, piv, info = f(a, overwrite_a=True)
            assert lu is a or np.shares_memory(lu, a)
            return piv + 1, info

        def getrs(fns, lu, ipiv, b, trans=0):
            f = scipy.linalg.get_lapack_funcs("getrs", (lu,))
            x, info = f(lu, ipiv - 1, b, trans=trans, overwrite_b=True)
            b[...] = x
            return info

        monkeypatch.setattr(_lapack, "_getrf", getrf)
        monkeypatch.setattr(_lapack, "_getrs", getrs)
    return request.param


@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
def test_solve_is_the_unprepared_solve_raises_and_warns_alike(
    route, dtype, monkeypatch
):
    calls = []
    original = acc.fortran_copy_1norm

    class Spy:
        def __getattr__(self, name):
            return getattr(acc, name)

        def fortran_copy_1norm(self, *a):
            calls.append(1)
            return original(*a)

    import momwire._accel as accel_mod

    monkeypatch.setattr(accel_mod, "acc", Spy())
    for name, (a, b) in _cases(dtype).items():
        calls.clear()
        monkeypatch.setattr(_lapack, "_SOLVE_PREP_ACCEL", True)
        got = _outcome(_lapack.solve, a, b)
        assert calls == [1], name  # the prepared pass ran
        monkeypatch.setattr(_lapack, "_SOLVE_PREP_ACCEL", False)
        calls.clear()
        want = _outcome(_lapack.solve, a, b)
        assert calls == [], name
        assert got[1:] == want[1:], name
        if want[0] is not None:
            assert got[0].tobytes() == want[0].tobytes(), name
    # The cases mean something: the warning and the error both occur.
    monkeypatch.setattr(_lapack, "_SOLVE_PREP_ACCEL", True)
    cases = _cases(dtype)
    assert _outcome(_lapack.solve, *cases["ill"])[2] == ["LinAlgWarning"]
    assert _outcome(_lapack.solve, *cases["fine"])[2] == []
    assert _outcome(_lapack.solve, *cases["nan"])[1] is ValueError


def test_what_the_pass_cannot_take_goes_the_old_way(monkeypatch):
    rng = np.random.default_rng(4)
    a = _rand(rng, (100, 100), np.complex128) + 100 * np.eye(100)
    for arg, dtype in [
        (np.asfortranarray(a), np.complex128),  # not C-contiguous
        (a.astype(np.complex64), np.complex64),  # a dtype it does not take
        (a.real.copy(), np.complex128),  # cast by a complex rhs
        (a[:40, :40].copy(), np.complex128),  # below the size floor
    ]:
        assert _lapack._solve_prep(arg, np.dtype(dtype)) is None
    assert _lapack._solve_prep(a, np.dtype(np.complex128)) is not None


@pytest.mark.parametrize("case", ["fine", "ill", "fine, many rhs"])
def test_gecon_beside_getrs_is_the_serial_solve(case, monkeypatch):
    """At n >= `_GECON_BESIDE_MIN_N` the no-host solve runs gecon on a second
    thread beside getrs: the same answer, the same warning."""
    monkeypatch.setattr(_lapack, "_host_for", lambda dtype: None)
    rng = np.random.default_rng(5)
    n = _lapack._GECON_BESIDE_MIN_N + 37
    if case == "ill":
        a = _ill(rng, n, np.complex128)
    else:
        a = _rand(rng, (n, n), np.complex128) + n * np.eye(n)
    b = _rand(rng, (n, 3) if "many" in case else (n,), np.complex128)
    started = []
    real = _lapack._gecon_beside

    def spy(*a):
        started.append(1)
        return real(*a)

    monkeypatch.setattr(_lapack, "_gecon_beside", spy)
    beside = _outcome(_lapack.solve, a, b)
    assert started == [1]
    monkeypatch.setattr(_lapack, "_GECON_BESIDE_MIN_N", 10**9)
    serial = _outcome(_lapack.solve, a, b)
    assert started == [1]
    assert beside[1:] == serial[1:]
    assert beside[0].tobytes() == serial[0].tobytes()
    assert beside[2] == (["LinAlgWarning"] if case == "ill" else [])
