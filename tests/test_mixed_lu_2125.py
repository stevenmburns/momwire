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
from momwire.deck import build_solver, parse, parse_nec2
from momwire.razor import RazorSolver
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


@pytest.fixture(autouse=True)
def any_size(request, monkeypatch):
    """The size rule off (MIXED_MIN_N, `_mixed_nrhs_cap`), so these small
    systems take the route; the test of the rule itself opts out."""
    if request.node.name == "test_the_size_rule":
        return
    monkeypatch.setattr(_lapack, "MIXED_MIN_N", 0)
    monkeypatch.setattr(_lapack, "_mixed_nrhs_cap", lambda n: _lapack.MIXED_MAX_NRHS)


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


def test_the_sweep_cap_grows_with_n_and_loosens_the_stall_bar():
    # Small orders keep the five-sweep cap and its 2**-6 bar exactly.
    assert _lapack._mixed_sweeps_cap(1024, 1) == _lapack.MIXED_SWEEPS_MAX == 5
    assert _lapack._mixed_stall(5) == _lapack.MIXED_CONTRACTION
    # The cap grows as n / nrhs; the bar is the rate that finishes inside it.
    assert _lapack._mixed_sweeps_cap(9329, 1) == 14
    assert _lapack._mixed_sweeps_cap(9329, 3) == 5
    assert _lapack._mixed_stall(14) == 2.0**-3
    assert _lapack._mixed_stall(1000) == _lapack.MIXED_STALL_LOOSEST
    for cap in range(5, 40):
        theta = _lapack._mixed_stall(cap)
        assert -29 / np.log2(theta) <= cap or theta == _lapack.MIXED_STALL_LOOSEST


def test_a_slow_contraction_is_waited_for_when_the_cap_allows(monkeypatch):
    # kappa 3e6 contracts by about 0.02 a sweep, slower than the base bar
    # (MIXED_CONTRACTION, 2**-6), so at the five-sweep cap it stalls. With
    # the cap a large order gets, it refines, within 4x of the gap between
    # the ordinary answer (`solve_transposed` factors a^T) and a second
    # complex128 factor's, pivoting on a.
    a, b = _conditioned(3e6), _rhs(120)
    with _lapack.mixed_lu_scope(True):
        before = _counts()
        _lapack.solve_transposed(a.copy(), b)
    assert _moved(before)[1] == {"stall": 1}
    monkeypatch.setattr(_lapack, "MIXED_SWEEPS_N", 8)  # n = 120: cap 15
    x_off, _a_off, x_on, _a_on, (solves, fallbacks) = _flag_off_and_on(
        _lapack.solve_transposed, a, b
    )
    assert solves["refined"] == 1 and not fallbacks, (solves, fallbacks)
    assert _lapack.MIXED_LAST["cap"] == 15
    assert max(_lapack.MIXED_LAST["ratio"][:-1]) > _lapack.MIXED_CONTRACTION
    import scipy.linalg as sl

    x_other = sl.lu_solve(sl.lu_factor(a), b)
    scatter = np.abs(x_other - x_off).max()
    assert np.abs(x_on - x_off).max() <= 4 * scatter, (
        np.abs(x_on - x_off).max(),
        scatter,
    )


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


def test_the_size_rule():
    cap = _lapack._mixed_nrhs_cap
    assert (cap(1024), cap(2048), cap(4392), cap(12534)) == (3, 13, 64, 182)
    for n, k, outcome in [
        (1023, 1, "declined"),
        (1024, 3, "refined"),
        (1024, 4, "declined"),
    ]:
        a, b = _well(n=n, seed=7), _rhs(n, k)
        x_off, _a, x_on, _b, (solves, fallbacks) = _flag_off_and_on(
            _lapack.factor_and_solve, a, b
        )
        assert solves.get(outcome) == 1 and not fallbacks, (n, k, solves)
        if outcome == "declined":
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


# --- factor_and_solve: bs2, bs1, razor2p, sg --------------------------------


@pytest.mark.parametrize("order", ["F", "C"])
def test_factor_and_solve_refines_either_order_and_keeps_a(order):
    a = np.asarray(_well(seed=12), order=order)
    b = _rhs(160, 2)
    x_off, a_off, x_on, a_on, (solves, fallbacks) = _flag_off_and_on(
        lambda m, v: _lapack.factor_and_solve(m, v, overwrite_a=True), a, b
    )
    assert solves["refined"] == 1 and not fallbacks
    assert np.abs(x_on - x_off).max() <= 1e-13 * np.abs(x_off).max()
    # Refined, a is untouched although overwrite_a asked; flag off, an
    # F-ordered a held its factors (factored in place).
    assert a_on.tobytes() == a.tobytes()
    if order == "F":
        assert a_off.tobytes() != a.tobytes()


@pytest.mark.parametrize("order", ["F", "C"])
def test_factor_and_solve_fallback_is_the_in_place_flag_off_route(order):
    a = np.asarray(_conditioned(1e9), order=order)
    b = _rhs(120)
    x_off, a_off, x_on, a_on, (solves, fallbacks) = _flag_off_and_on(
        lambda m, v: _lapack.factor_and_solve(m, v, overwrite_a=True), a, b
    )
    assert solves == {"fallback": 1} and fallbacks == {"stall": 1}
    assert x_on.tobytes() == x_off.tobytes()
    # The fallback factored a in place exactly as the flag-off call did.
    assert a_on.tobytes() == a_off.tobytes()


def test_factor_and_solve_declines_a_strided_matrix():
    a = _well(n=200, seed=13)[::2, ::2]  # neither C- nor F-contiguous
    b = _rhs(100)
    x_off = _lapack.factor_and_solve(a, b)
    before = _counts()
    with _lapack.mixed_lu_scope(True):
        x_on = _lapack.factor_and_solve(a, b)
    assert _moved(before) == ({"declined": 1}, {})
    assert x_on.tobytes() == x_off.tobytes()


T_JUNCTION = (
    "CM t\nCE\nGW 1 21 0 0 0 0 0 5 0.001\nGW 2 15 0 0 5 0 3 7 0.001\n"
    "GW 3 15 0 0 5 0 -3 7 0.001\nGE 0\nEX 0 1 11 0 1 0\nFR 0 1 0 0 28.0\nXQ\nEN\n"
)


def _razor(**kw):
    wire = [[[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]]]
    return RazorSolver(
        wires=wire,
        nsegs=21,
        wire_radius=1e-3,
        wavelength=20.0,
        nec5_quadrature=True,
        **kw,
    )


def _engine(name, deck):
    if name == "razor2p":
        return _razor()
    return build_solver(parse_nec2(deck), basis=name).solver


@pytest.mark.parametrize("deck", [DIPOLE, T_JUNCTION], ids=["dipole", "junction"])
@pytest.mark.parametrize("name", ["bs2", "bs1", "sg", "razor2p"])
def test_each_engine_runs_it_on_the_variable(no_features, name, deck):
    if name == "razor2p" and deck is T_JUNCTION:
        pytest.skip("one razor2p geometry is enough")
    y_off = _engine(name, deck).compute_port_solution().y
    no_features.setenv(ENV, "1")
    solver = _engine(name, deck)
    assert solver.features_bound["mixed_lu"] is True
    before = _counts()
    y_on = solver.compute_port_solution().y
    solves, fallbacks = _moved(before)
    assert solves.get("refined", 0) >= 1 and not fallbacks, (solves, fallbacks)
    assert np.abs(y_on - y_off).max() <= 1e-11 * np.abs(y_off).max()


def test_a_badly_scaled_matrix_refines_past_its_backward_error(no_features):
    # sg on a SimNEC stream deck (45 unknowns, rows and columns spanning
    # 1.5e5, kappa_1 3e7): the normwise backward error reaches u64 after two
    # corrections while the answer is still 5e-10 from the flag-off one; the
    # forward test (MIXED_FORWARD_TOLERANCE) takes the sweep that closes it.
    from pathlib import Path

    deck = Path(__file__).parents[1] / "scripts/core_split/nx_decks"
    model = parse((deck / "simnec_stream_2281585.nec").read_text(), dialect="nec4")
    z_off = 1 / build_solver(model, basis="sg").solver.compute_port_solution().y[0, 0]
    no_features.setenv(ENV, "1")
    before = _counts()
    z_on = 1 / build_solver(model, basis="sg").solver.compute_port_solution().y[0, 0]
    assert _moved(before)[0]["refined"] == 1
    assert _lapack.MIXED_LAST["eta"][2] <= _lapack.MIXED_TOLERANCE
    assert _lapack.MIXED_LAST["sweeps"] >= 3
    assert abs(z_on - z_off) <= 1e-11 * abs(z_off)


@pytest.mark.parametrize("name", ["bs2", "sg", "razor2p"])
def test_each_engine_impedance_runs_it(no_features, name):
    no_features.setenv(ENV, "1")
    solver = _engine(name, DIPOLE)
    before = _counts()
    solver.compute_impedance()
    assert _moved(before)[0].get("refined", 0) >= 1


def test_the_kcl_lambda_solve_stays_outside_the_scope(no_features, monkeypatch):
    # bs2's Schur step solves a small (n_c x n_c) system with `_lapack.solve`
    # after the big factor; only the big factor is in the scope.
    no_features.setenv(ENV, "1")
    seen = []
    real = _lapack.solve

    def spy(a, b):
        seen.append(_lapack._MIXED.get())
        return real(a, b)

    monkeypatch.setattr(_lapack, "solve", spy)
    solver = _engine("bs2", T_JUNCTION)
    solver.compute_port_solution()
    assert seen and not any(seen)


@pytest.mark.parametrize("name", ["bs2", "razor2p", "sg"])
def test_flag_off_still_factors_z_in_place(no_features, monkeypatch, name):
    calls = []
    real = _lapack.lu_factor

    def spy(a, overwrite_a=False, check_finite=True):
        a1 = np.asarray(a)
        calls.append((overwrite_a, a1.flags.f_contiguous, a1.shape[0]))
        return real(a, overwrite_a=overwrite_a, check_finite=check_finite)

    monkeypatch.setattr(_lapack, "lu_factor", spy)
    _engine(name, DIPOLE).compute_port_solution()
    big = max(calls, key=lambda c: c[2])
    assert big[0] and big[1], calls  # overwrite_a on an F-ordered Z: in place
    calls.clear()
    no_features.setenv(ENV, "1")
    _engine(name, DIPOLE).compute_port_solution()
    assert not calls  # refined: no complex128 factor at all


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
    assert f.solvers == ("bspline", "razor", "sinusoidal", "sinusoidal_galerkin")
