"""`_lapack.solve`'s condition estimate is opt-in (`MOMWIRE_CONDITION_CHECK`).

Off (the default): a well-conditioned solve runs no gecon; the free pivot
screen still flags an ill-conditioned one, which then gets gecon and the
warning. On: gecon runs on every solve. Both routes (host, no-host), both
sides of the size where gecon runs beside getrs. Either way the answer is
the same bits: the estimate never decides a value.
"""

from __future__ import annotations

import ctypes
import sys
import warnings

import numpy as np
import pytest
import scipy.linalg
from scipy.linalg import cython_lapack

from momwire import _lapack


def _rand(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _ill(rng, n):
    q1, _ = np.linalg.qr(_rand(rng, (n, n)))
    q2, _ = np.linalg.qr(_rand(rng, (n, n)))
    return np.ascontiguousarray((q1 * np.logspace(0, -19, n)) @ q2)


def _capsule_address(name):
    cap = cython_lapack.__pyx_capi__[name]
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype = ctypes.c_char_p
    api.PyCapsule_GetName.argtypes = [ctypes.py_object]
    api.PyCapsule_GetPointer.restype = ctypes.c_void_p
    api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))


@pytest.fixture(params=["host", "no host"])
def route(request, monkeypatch):
    """The two routes `solve` takes itself (neither hands off to scipy)."""
    if request.param == "host":
        table = {"abi": 1, **{n: _capsule_address(n) for n in ("zgetrf", "zgetrs")}}
        monkeypatch.setattr(sys, "_mw_host_lapack", table, raising=False)
    else:
        monkeypatch.delattr(sys, "_mw_host_lapack", raising=False)
        monkeypatch.setattr(_lapack, "_host_for", lambda dtype: None)
    return request.param


@pytest.fixture
def gecon_calls(monkeypatch):
    """Counts every gecon `solve` obtains and runs."""
    calls = []
    real = scipy.linalg.get_lapack_funcs

    def spy(names, arrays=(), *a, **kw):
        fns = real(names, arrays, *a, **kw)
        if not isinstance(names, (tuple, list)):
            return fns
        out = []
        for name, fn in zip(names, fns):
            if name == "gecon":

                def counted(*args, _fn=fn, **kwargs):
                    calls.append(1)
                    return _fn(*args, **kwargs)

                out.append(counted)
            else:
                out.append(fn)
        return out

    monkeypatch.setattr(scipy.linalg, "get_lapack_funcs", spy)
    return calls


def _run(a, b):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        x = _lapack.solve(a, b)
    return x, sorted({w.category.__name__ for w in seen})


def _fine(rng, n):
    return _rand(rng, (n, n)) + n * np.eye(n)


SIZES = [200, _lapack._GECON_BESIDE_MIN_N + 37]


@pytest.mark.parametrize("n", SIZES)
def test_a_well_conditioned_solve_runs_no_gecon_unless_asked(
    route, gecon_calls, n, monkeypatch
):
    rng = np.random.default_rng(1924)
    a, b = _fine(rng, n), _rand(rng, (n, 2))
    monkeypatch.delenv(_lapack.CONDITION_CHECK_ENV, raising=False)
    assert not _lapack.condition_check()
    x_off, warned_off = _run(a, b)
    assert gecon_calls == []
    assert warned_off == []

    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1")
    x_on, warned_on = _run(a, b)
    assert gecon_calls == [1]
    assert warned_on == []
    assert x_on.tobytes() == x_off.tobytes()


@pytest.mark.parametrize("n", SIZES)
def test_the_pivot_screen_catches_an_ill_conditioned_matrix_by_default(
    route, gecon_calls, n, monkeypatch
):
    """Off: the screen flags the factors, gecon runs once and the warning
    fires, as with the check on; the same bits either way."""
    rng = np.random.default_rng(1925)
    a, b = _ill(rng, n), _rand(rng, (n, 2))
    monkeypatch.delenv(_lapack.CONDITION_CHECK_ENV, raising=False)
    x_off, warned_off = _run(a, b)
    assert gecon_calls == [1]
    assert warned_off == ["LinAlgWarning"]

    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1")
    x_on, warned_on = _run(a, b)
    assert gecon_calls == [1, 1]
    assert warned_on == ["LinAlgWarning"]
    assert x_on.tobytes() == x_off.tobytes()


def test_the_screen_reads_the_diagonal_ratio():
    lu = np.diag([1.0, 0.5, 2.0 * _lapack.PIVOT_SCREEN]).astype(np.complex128)
    assert not _lapack._pivots_suspect(lu)
    lu[2, 2] = 0.5 * _lapack.PIVOT_SCREEN
    assert _lapack._pivots_suspect(lu)
    lu[2, 2] = 0.0
    assert _lapack._pivots_suspect(lu)
    lu[2, 2] = np.nan
    assert _lapack._pivots_suspect(lu)
    lu[2, 2] = 1.0
    lu[0, 1] = 1e30  # off-diagonal entries are not read
    assert not _lapack._pivots_suspect(lu)


@pytest.mark.parametrize(
    ("value", "on"), [("", False), ("0", False), ("1", True), ("yes", True)]
)
def test_the_switch_reads_the_environment(value, on, monkeypatch):
    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, value)
    assert _lapack.condition_check() is on
