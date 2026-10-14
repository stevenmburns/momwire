"""The razor-2p C++ solve prototype (`_razor_solve_core.cpp`) against the
Python route it replaces, byte for byte.

`razor._RAZOR_CXX = True` routes `compute_port_solution` through one native
call (every fill window, then SciPy's own zgetrf/zgetrs) where
`RazorSolver._cxx_route_refusal` allows it. Each case solves the same
solver configuration both ways and requires the filled Z (before the LU),
the solved coefficients and Y to be the same bytes. Every agreement is
counted only after `razor._CXX_ROUTE` and the context's counters show the
C++ branch RAN -- a route that silently fell back would compare Python with
Python and pass.
"""

import numpy as np
import pytest

from momwire import CancelToken, RazorSolver, SolveAborted, razor

pytestmark = pytest.mark.skipif(
    not razor._HAVE_RAZOR_SOLVE_CXX, reason="accelerator without razor_solve_proto"
)

DIPOLE = [[(0.0, -5.0, 6.0), (0.0, 5.0, 6.0)]]
BASE = dict(
    wire_radius=0.002,
    wavelength=22.0,
    nec5_quadrature=True,
    exact_kernel="auto",
)

CASES = {
    "free": dict(wires=DIPOLE, nsegs=21, feed_arclength=5.0),
    "free_ek": dict(wires=DIPOLE, nsegs=21, feed_arclength=5.0, extended_kernel=True),
    "pec": dict(wires=DIPOLE, nsegs=21, feed_arclength=5.0, ground_z=0.0),
    "pec_ek": dict(
        wires=DIPOLE, nsegs=21, feed_arclength=5.0, ground_z=0.0, extended_kernel=True
    ),
    # A wire touching the plane: its first tent is a GROUNDED row.
    "pec_grounded": dict(
        wires=[[(0.0, 0.0, 0.0), (0.0, 0.0, 5.3)]],
        nsegs=15,
        feeds=[(0, 0.4, 1.0 + 0j)],
        ground_z=0.0,
    ),
    # A three-wire junction and two driven ports.
    "junction_two_ports": dict(
        wires=[
            [(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 0.0, 9.0)],
        ],
        n_per_edge_per_wire=[[9], [9], [7]],
        feeds=[(0, 5.0 * 4 / 9, 1.0 + 0j), (2, 3.0 * 3 / 7, 0.5 + 0.25j)],
    ),
    "kinked_pec": dict(
        wires=[[(0.0, -5.0, 4.0), (0.0, 0.0, 6.0), (0.0, 5.0, 4.0)]],
        n_per_edge_per_wire=[[11, 11]],
        feeds=[(0, 5.4, 1.0 + 0j)],
        ground_z=0.0,
    ),
}


@pytest.fixture
def capture_z(monkeypatch):
    """The Python route's Z as `_lu_solve` receives it (before it factors)."""
    seen = {}
    orig = razor._bspline._lu_solve

    def lu(Z, rhs, **kw):
        seen["Z"] = np.array(Z, order="F", copy=True)
        return orig(Z, rhs, **kw)

    monkeypatch.setattr(razor._bspline, "_lu_solve", lu)
    monkeypatch.setattr(razor, "_CXX_WANT_Z", True)
    monkeypatch.setattr(razor, "_CXX_ROUTE", {})
    return seen


def _both(monkeypatch, seen, kw):
    monkeypatch.setattr(razor, "_RAZOR_CXX", False)
    seen.clear()
    kw = {**BASE, **kw}
    py = RazorSolver(**kw).compute_port_solution()
    z_py = seen["Z"]
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    solver = RazorSolver(**kw)
    cx = solver.compute_port_solution()
    return py, z_py, cx, solver


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_cxx_route_is_the_python_route_to_the_bit(monkeypatch, capture_z, name):
    ctx = razor._cxx_context()
    before = ctx.counters
    py, z_py, cx, solver = _both(monkeypatch, capture_z, CASES[name])
    after = razor._cxx_context().counters
    assert razor._CXX_ROUTE == {"cxx": 1}
    assert after["solves"] == before["solves"] + 1
    assert after["lu_calls"] == before["lu_calls"] + 1
    if CASES[name].get("ground_z") is not None:
        assert after["image_windows"] > before["image_windows"]
    assert solver._cxx_z.tobytes() == z_py.tobytes()
    assert cx.coeffs.tobytes() == py.coeffs.tobytes()
    assert cx.y.tobytes() == py.y.tobytes()


def test_many_windows_are_the_one_window_to_the_bit(monkeypatch, capture_z):
    # A budget of one basis row per window on both routes: 20 windows here.
    monkeypatch.setattr(razor, "_CHUNK_ELEMS", 1)
    before = razor._cxx_context().counters["windows"]
    py, z_py, cx, solver = _both(monkeypatch, capture_z, CASES["pec"])
    assert razor._cxx_context().counters["windows"] - before == 20
    assert razor._CXX_ROUTE == {"cxx": 1}
    assert solver._cxx_z.tobytes() == z_py.tobytes()
    assert cx.coeffs.tobytes() == py.coeffs.tobytes()


@pytest.mark.parametrize(
    "extra, reason",
    [
        (dict(lumped_loads=[(0, 5.0, 50 + 0j)]), "loading"),
        (dict(ground_z=0.0, ground_eps=13.0 - 1.2j), "finite-ground"),
        (dict(nec5_quadrature=False, n_qp_path=2), "path-rule"),
    ],
)
def test_an_uncovered_solve_stays_on_python_and_says_why(
    monkeypatch, capture_z, extra, reason
):
    kw = dict(CASES["free"], **extra)
    py, _, cx, _ = _both(monkeypatch, capture_z, kw)
    assert razor._CXX_ROUTE == {f"python:{reason}": 1}
    assert cx.coeffs.tobytes() == py.coeffs.tobytes()


def test_the_default_is_the_python_route(monkeypatch):
    monkeypatch.setattr(razor, "_RAZOR_CXX", None)
    monkeypatch.delenv("MOMWIRE_RAZOR_CXX", raising=False)
    monkeypatch.setattr(razor, "_CXX_ROUTE", {})
    RazorSolver(**BASE, **CASES["free"]).compute_port_solution()
    assert razor._CXX_ROUTE == {}


def test_the_environment_switch_routes(monkeypatch):
    monkeypatch.setattr(razor, "_RAZOR_CXX", None)
    monkeypatch.setenv("MOMWIRE_RAZOR_CXX", "1")
    monkeypatch.setattr(razor, "_CXX_ROUTE", {})
    RazorSolver(**BASE, **CASES["free"]).compute_port_solution()
    assert razor._CXX_ROUTE == {"cxx": 1}


def test_a_tripped_token_aborts_inside_the_native_call(monkeypatch):
    # The Python checkpoints are silenced, so only the C++ poll can see it.
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    token = CancelToken()
    solver = RazorSolver(**BASE, **CASES["free"], cancel=token)
    monkeypatch.setattr(solver, "_checkpoint", lambda: None)
    token.cancel()
    with pytest.raises(SolveAborted):
        solver.compute_port_solution()


def test_the_lapack_is_scipys_own():
    # The capsules the binding reads exist with the Fortran signatures the
    # core declares; a SciPy that renamed or retyped them must fail here.
    from scipy.linalg import cython_lapack

    capi = cython_lapack.__pyx_capi__
    assert "int *, int *, __pyx_t_double_complex *, int *, int *, int *" in str(
        capi["zgetrf"]
    )
    assert "char *, int *, int *, __pyx_t_double_complex *" in str(capi["zgetrs"])


def test_a_released_pool_keeps_the_context_and_its_counters(monkeypatch, capture_z):
    # Every solve releases the buffers here; the counters must carry on and
    # the next solve must regrow the pool and still be the Python route.
    monkeypatch.setattr(razor, "_CXX_POOL_MAX_BASIS", 0)
    ctx = razor._cxx_context()
    before = ctx.counters
    for _ in range(2):
        py, z_py, cx, solver = _both(monkeypatch, capture_z, CASES["pec"])
        assert solver._cxx_z.tobytes() == z_py.tobytes()
        assert cx.coeffs.tobytes() == py.coeffs.tobytes()
    after = razor._cxx_context().counters
    assert razor._cxx_context() is ctx
    assert after["solves"] == before["solves"] + 2
    assert after["pool_grows"] > before["pool_grows"]
    assert ctx.n_basis == 0
