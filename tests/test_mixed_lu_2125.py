"""`mixed_lu` (momwire#2125): complex64 getrf + complex128 iterative refinement
in `_lapack.solve` / `_lapack.solve_transposed`, opt-in, with a fallback to the
ordinary complex128 factor.

Every test here proves the route RAN (the outcome counters move, and the
recorded outcome is the one the test is about), so none of them passes with
the feature absent; the fallback tests compare BYTES with the flag-off call,
so they fail if the route answered instead of falling back. Each was checked
to fail with the mixed route disabled and with its fallback disabled.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from momwire import _features, _lapack
from momwire.deck import build_solver, parse_nec2
from momwire.sinusoidal import SinusoidalSolver

pytestmark = pytest.mark.reach(1)

ENV = "MOMWIRE_X_MIXED_LU"

DIPOLE = (
    "CM d\nCE\nGW 1 41 0 0 -2.5 0 0 2.5 0.001\nGE 0\n"
    "EX 0 1 21 0 1 0\nFR 0 1 0 0 28.0\nXQ\nEN\n"
)
TWO_PORT = (
    "CM d\nCE\nGW 1 41 0 0 -2.5 0 0 2.5 0.001\nGW 2 41 1 0 -2.5 1 0 2.5 0.001\n"
    "GE 0\nEX 0 1 21 0 1 0\nEX 0 2 21 0 1 0\nFR 0 1 0 0 28.0\nXQ\nEN\n"
)


@pytest.fixture
def no_features(monkeypatch):
    for f in _features.FEATURES.values():
        for var in (f.env, *f.aliases):
            monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _rng(seed=0):
    return np.random.default_rng(seed)


def _well(n=160, seed=0):
    """A C-ordered complex matrix with kappa of a few: mixed refines it."""
    rng = _rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    a += 4 * np.sqrt(n) * np.eye(n)
    return np.ascontiguousarray(a)


def _conditioned(kappa, n=120, seed=1):
    """A C-ordered complex matrix with 2-norm condition number `kappa`."""
    rng = _rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    v, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(kappa), n)
    return np.ascontiguousarray((u * s) @ v)


def _rhs(n, k=1, seed=2):
    rng = _rng(seed)
    b = rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
    return b[:, 0] if k == 1 else b


def _counts():
    return dict(_lapack.MIXED_SOLVES), dict(_lapack.MIXED_FALLBACKS)


def _moved(before):
    s0, f0 = before
    s = {k: v - s0[k] for k, v in _lapack.MIXED_SOLVES.items() if v != s0[k]}
    f = {k: v - f0[k] for k, v in _lapack.MIXED_FALLBACKS.items() if v != f0[k]}
    return s, f


def _flag_off_and_on(fn, a, b):
    """(flag-off answer, flag-off matrix after, flag-on answer, flag-on
    matrix after, counter moves), each call on its own copy of `a`."""
    a_off, a_on = a.copy(order="K"), a.copy(order="K")
    x_off = fn(a_off, b)
    before = _counts()
    with _lapack.mixed_lu_scope(True):
        x_on = fn(a_on, b)
    return x_off, a_off, x_on, a_on, _moved(before)


# --- the route answers -----------------------------------------------------


@pytest.mark.parametrize("k", [1, 3])
def test_a_well_conditioned_system_is_refined_and_a_is_kept(k):
    a, b = _well(), _rhs(160, k)
    x_off, _a_off, x_on, a_on, (solves, fallbacks) = _flag_off_and_on(
        _lapack.solve_transposed, a, b
    )
    assert solves["refined"] == 1 and not fallbacks
    assert solves["sweeps"] >= _lapack.MIXED_SWEEPS_MIN
    assert _lapack.MIXED_LAST["outcome"] == "refined"
    # The refined route reads a and never writes it (the fallback needs it).
    assert a_on.tobytes() == a.tobytes()
    assert x_on.shape == x_off.shape and x_on.dtype == np.complex128
    assert np.abs(x_on - x_off).max() <= 1e-13 * np.abs(x_off).max()
    # The stop: the backward error at complex128's unit roundoff.
    bb = b.reshape(160, -1)
    eta = np.abs(bb - a @ x_on.reshape(160, -1)).sum(0) / (
        np.abs(a).sum(0).max() * np.abs(x_on.reshape(160, -1)).sum(0)
    )
    assert eta.max() <= np.sqrt(160) * _lapack._U64


def test_solve_takes_the_route_too():
    a, b = _well(seed=3), _rhs(160, 2)
    x_off, _a, x_on, _b, (solves, fallbacks) = _flag_off_and_on(_lapack.solve, a, b)
    assert solves == {"refined": 1, "sweeps": solves["sweeps"]} and not fallbacks
    assert np.abs(x_on - x_off).max() <= 1e-13 * np.abs(x_off).max()


def test_solve_declines_a_fortran_ordered_matrix():
    a = np.asfortranarray(_well(seed=4))
    b = _rhs(160)
    x_off, _a, x_on, _b, (solves, fallbacks) = _flag_off_and_on(_lapack.solve, a, b)
    assert solves == {"declined": 1} and not fallbacks
    assert x_on.tobytes() == x_off.tobytes()


def test_one_correction_is_never_enough(monkeypatch):
    # 2 I and a right-hand side held exactly in complex64: the complex64
    # solve is already exact (backward error 0, under u64 from the start), and
    # the route still takes two corrections before it stops. With the minimum
    # lowered to one it stops after one.
    n = 64
    a = np.ascontiguousarray(2.0 * np.eye(n, dtype=np.complex128))
    b = _rhs(n).astype(np.complex64).astype(np.complex128)
    with _lapack.mixed_lu_scope(True):
        _lapack.solve_transposed(a.copy(), b)
    assert _lapack.MIXED_LAST["outcome"] == "refined"
    assert _lapack.MIXED_LAST["eta"][0] <= _lapack.MIXED_TOLERANCE
    assert _lapack.MIXED_LAST["sweeps"] == _lapack.MIXED_SWEEPS_MIN == 2
    monkeypatch.setattr(_lapack, "MIXED_SWEEPS_MIN", 1)
    with _lapack.mixed_lu_scope(True):
        _lapack.solve_transposed(a.copy(), b)
    assert _lapack.MIXED_LAST["sweeps"] == 1


# --- the fallbacks: the flag-off bits, every time -------------------------------


def _assert_fallback(a, b, reason, fn=None):
    fn = fn or _lapack.solve_transposed
    x_off, a_off, x_on, a_on, (solves, fallbacks) = _flag_off_and_on(fn, a, b)
    assert solves == {"fallback": 1}, solves
    assert fallbacks == {reason: 1}, fallbacks
    assert _lapack.MIXED_LAST["outcome"] == reason
    # The ordinary route then ran on the untouched matrix: its answer, and
    # its factors left in a, byte for byte.
    assert x_on.tobytes() == x_off.tobytes()
    assert a_on.tobytes() == a_off.tobytes()


@pytest.mark.parametrize("kappa", [1e8, 1e10])
def test_an_ill_conditioned_system_falls_back_to_the_flag_off_bits(kappa):
    # complex64's factor cannot contract at kappa >= 1e8 (rate ~ kappa u32 > 1).
    _assert_fallback(_conditioned(kappa), _rhs(120), "stall")


def test_a_slow_contraction_is_not_waited_for():
    # kappa 1e7: refinement still converges (it contracts by about 0.1 a
    # sweep), but more slowly than MIXED_CONTRACTION, so the first slow sweep
    # sends it to the ordinary route rather than a dozen sweeps later.
    a, b = _conditioned(1e7), _rhs(120)
    _assert_fallback(a, b, "stall")
    assert max(_lapack.MIXED_LAST["ratio"]) > _lapack.MIXED_CONTRACTION


@pytest.mark.parametrize("scale", [1e39, 1e-40], ids=["overflow", "underflow"])
def test_a_matrix_complex64_cannot_hold_falls_back_before_the_cast(scale):
    # 1e39 exceeds complex64's largest finite value (3.4e38), and 1e-40 is
    # below its smallest normal one: the cast guard sends both to the ordinary
    # route before any complex64 copy is made ("range", not "nonfinite").
    a = _well(seed=6) * scale
    _assert_fallback(a, _rhs(160), "range")


def test_a_matrix_singular_in_complex64_falls_back():
    # 1 + 1e-10 rounds to 1 in complex64: singular there, not in complex128.
    a = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]], dtype=np.complex128)
    _assert_fallback(a, np.array([1.0, 2.0], dtype=np.complex128), "singular")


def test_a_singular_system_still_raises_the_ordinary_error():
    a = np.ones((4, 4), dtype=np.complex128)
    with _lapack.mixed_lu_scope(True), pytest.raises(np.linalg.LinAlgError):
        _lapack.solve_transposed(a, np.ones(4, dtype=np.complex128))


# --- what declines without trying ------------------------------------------------


def test_too_many_right_hand_sides_decline():
    a = _well(n=200, seed=7)
    k = _lapack.MIXED_MAX_NRHS
    _off, _a, _on, _b, (solves, _f) = _flag_off_and_on(
        _lapack.solve_transposed, a, _rhs(200, k)
    )
    assert solves["refined"] == 1
    b = _rhs(200, k + 1)
    x_off, _a, x_on, _b, (solves, fallbacks) = _flag_off_and_on(
        _lapack.solve_transposed, a, b
    )
    assert solves == {"declined": 1} and not fallbacks
    assert x_on.tobytes() == x_off.tobytes()


def test_a_real_system_declines():
    a = np.ascontiguousarray(_well(seed=8).real)
    b = _rhs(160).real
    x_off, _a, x_on, _b, (solves, _f) = _flag_off_and_on(_lapack.solve_transposed, a, b)
    assert solves == {"declined": 1}
    assert x_on.tobytes() == x_off.tobytes()


def test_a_host_lapack_declines(monkeypatch):
    # A published host carries no complex64 getrf: the route steps aside.
    before = _counts()
    with _lapack.mixed_lu_scope(True):
        assert not _lapack._mixed_wanted(object(), np.dtype(np.complex128), np.ones(3))
    assert _moved(before)[0] == {"declined": 1}


# --- the flag reaches only its own solve -------------------------------------


def test_off_by_default_and_scoped():
    a, b = _well(seed=9), _rhs(160)
    before = _counts()
    _lapack.solve_transposed(a.copy(), b)
    assert _moved(before) == ({}, {})
    with _lapack.mixed_lu_scope(True):
        with _lapack.mixed_lu_scope(False):
            _lapack.solve_transposed(a.copy(), b)
    assert _moved(before) == ({}, {})


def test_the_scope_belongs_to_its_thread():
    a, b = _well(seed=10), _rhs(160)
    seen = []

    def other():
        seen.append(_lapack._MIXED.get())

    with _lapack.mixed_lu_scope(True):
        t = threading.Thread(target=other)
        t.start()
        t.join()
        assert _lapack._MIXED.get()
    assert seen == [False]
    del a, b


def test_the_factor_clock_counts_the_mixed_route():
    a, b = _well(n=300, seed=11), _rhs(300)
    t0 = _lapack.factor_seconds()
    before = _counts()
    with _lapack.mixed_lu_scope(True):
        _lapack.solve_transposed(a, b)
    assert _moved(before)[0]["refined"] == 1
    assert _lapack.factor_seconds() > t0


# --- the solver: kwarg, variable, and a request that says no ------------------


@pytest.mark.parametrize("deck", [DIPOLE, TWO_PORT], ids=["1port", "2port"])
def test_the_sinusoidal_port_solve_runs_it_on_the_variable(no_features, deck):
    y_off = build_solver(parse_nec2(deck), basis="sinusoidal").solver
    y_off = y_off.compute_port_solution().y
    no_features.setenv(ENV, "1")
    solver = build_solver(parse_nec2(deck), basis="sinusoidal").solver
    assert solver.features_bound["mixed_lu"] is True
    before = _counts()
    y_on = solver.compute_port_solution().y
    assert _moved(before)[0]["refined"] == 1
    assert np.abs(y_on - y_off).max() <= 1e-11 * np.abs(y_off).max()


def _dipole(**kw):
    ys = np.linspace(-2.5, 2.5, 2)
    wire = np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, 4.0)])
    return SinusoidalSolver(
        wires=[wire], nsegs=41, wavelength=10.7, wire_radius=0.001, **kw
    )


def test_the_keyword_says_no_over_the_variable(no_features):
    # A request must be able to say no: the kwarg wins over the variable.
    no_features.setenv(ENV, "1")
    assert _dipole().features_bound["mixed_lu"] is True
    solver = _dipole(mixed_lu=False)
    assert solver.features_bound["mixed_lu"] is False
    before = _counts()
    solver.compute_port_solution()
    assert _moved(before) == ({}, {})
    no_features.delenv(ENV)
    solver = _dipole(mixed_lu=True)
    before = _counts()
    solver.compute_port_solution()
    assert _moved(before)[0]["refined"] == 1


def test_the_swept_impedance_runs_it(no_features):
    no_features.setenv(ENV, "1")
    solver = build_solver(parse_nec2(DIPOLE), basis="sinusoidal").solver
    before = _counts()
    solver.compute_impedance_swept(np.array([solver.k, 1.01 * solver.k]))
    assert _moved(before)[0]["refined"] == 2


def test_the_galerkin_family_ignores_it(no_features):
    no_features.setenv(ENV, "1")
    solver = build_solver(parse_nec2(DIPOLE), basis="sg").solver
    assert solver.features_bound["mixed_lu"] is False
    before = _counts()
    solver.compute_port_solution()
    assert _moved(before) == ({}, {})


def test_the_registry_entry():
    f = _features.FEATURES["mixed_lu"]
    assert (f.reach, f.kind, f.kwarg, f.env, f.issue) == (
        1,
        "python",
        "mixed_lu",
        ENV,
        2125,
    )
    assert f.offer and not f.default and f.label and f.summary
    # A last-digits feature: antennaknobs labels it "(faster)".
    assert f.drift < _features.DEFAULT_DRIFT_MAX
    assert f.solvers == ("sinusoidal",)
