"""Phase 1 cooperative-cancellation tests.

A CancelToken threads into a solver via the ``cancel=`` constructor kwarg; the
Python-level checkpoints (``self._checkpoint()``) poll it at phase boundaries,
sweep iterations, and H-matrix/array build + matvec loops, raising
``SolveAborted`` promptly. The default no-token path must be byte-identical to
before.

Mid-solve cancellation is driven deterministically by shadowing the instance's
``_checkpoint`` with a counter that trips the token on the Nth poll — no timer
threads, so the tests are not timing-flaky.
"""

import time

import numpy as np
import pytest

from momwire import (
    ArrayBlockSolver,
    BSplineSolver,
    CancelToken,
    HMatrixSolver,
    SinusoidalSolver,
    SolveAborted,
    TriangularSolver,
)

HALF = 0.962 * 22 / 4
DIPOLE = [np.array([[0.0, 0.0, -HALF], [0.0, 0.0, HALF]])]
BASE_SOLVERS = [SinusoidalSolver, TriangularSolver, BSplineSolver]


def _dipole(cls, *, cancel=None, nsegs=60):
    return cls(
        wires=DIPOLE,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        wavelength=22.0,
        cancel=cancel,
    )


def _dipole_array(cls, *, cancel=None, n_elem=4, nsegs=14):
    """A row of separated dipoles — exercises the H-matrix / array-block far
    blocks and the GMRES matvec path."""
    ys = np.linspace(-3.0, 3.0, n_elem)
    wires = [np.array([[0.0, y, -HALF], [0.0, y, HALF]]) for y in ys]
    return cls(
        wires=wires,
        n_per_edge_per_wire=[[nsegs]] * n_elem,
        nsegs=nsegs,
        wavelength=22.0,
        feeds=[(i, None, 1.0 + 0.0j) for i in range(n_elem)],
        cancel=cancel,
    )


def _trip_on_nth_checkpoint(solver, token, n):
    """Shadow ``solver._checkpoint`` so the Nth call cancels ``token`` (then the
    real checkpoint raises). Returns a one-element list holding the call count.
    """
    calls = [0]
    real = solver._checkpoint

    def counting():
        calls[0] += 1
        if calls[0] == n:
            token.cancel()
        real()

    solver._checkpoint = counting
    return calls, real


# --------------------------------------------------------------------------
# Pre-cancelled: the first checkpoint raises before any heavy work.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", BASE_SOLVERS, ids=lambda c: c.__name__)
def test_precancelled_compute_impedance_raises(cls):
    token = CancelToken()
    token.cancel()
    with pytest.raises(SolveAborted):
        _dipole(cls, cancel=token).compute_impedance()


@pytest.mark.parametrize(
    "cls", [HMatrixSolver, ArrayBlockSolver], ids=lambda c: c.__name__
)
def test_precancelled_array_solver_raises(cls):
    # Covers the build-loop / matvec checkpoints on the H-matrix and array-block
    # solve paths, not just the dense phase boundaries.
    token = CancelToken()
    token.cancel()
    with pytest.raises(SolveAborted):
        _dipole_array(cls, cancel=token).compute_impedance()


def test_precancel_is_fast_relative_to_full_solve():
    # The pre-cancelled solve raises at the first checkpoint (after geometry,
    # before the fill), so it must be a small fraction of a full solve.
    t0 = time.perf_counter()
    _dipole(SinusoidalSolver, nsegs=120).compute_impedance()
    full = time.perf_counter() - t0

    token = CancelToken()
    token.cancel()
    t0 = time.perf_counter()
    with pytest.raises(SolveAborted):
        _dipole(SinusoidalSolver, cancel=token, nsegs=120).compute_impedance()
    aborted = time.perf_counter() - t0

    assert aborted < 0.5 * full, f"abort {aborted*1e3:.1f}ms vs full {full*1e3:.1f}ms"


# --------------------------------------------------------------------------
# No-token path is unchanged (result byte-identical to omitting the kwarg).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", BASE_SOLVERS, ids=lambda c: c.__name__)
def test_no_token_result_identical(cls):
    z_ref, a_ref = _dipole(cls).compute_impedance()  # cancel defaults to None
    z_none, a_none = _dipole(cls, cancel=None).compute_impedance()
    assert np.array_equal(np.atleast_1d(z_ref), np.atleast_1d(z_none))
    assert np.array_equal(a_ref, a_none)


# --------------------------------------------------------------------------
# Mid-sweep cancellation: raises well before the sweep completes.
# --------------------------------------------------------------------------
def test_container_matvec_checks_token():
    # GMRES calls the operator's matvec every iteration; a tripped token must
    # break out within one matvec (not just at build time). Test the containers
    # directly so this pins the solve-time seam, not the build-loop one.
    from momwire._aca import HMatrix
    from momwire.array_block import ArrayBlock

    token = CancelToken()
    token.cancel()

    h = HMatrix(2, near=[], far=[], cancel=token)
    with pytest.raises(SolveAborted):
        h.matvec(np.zeros(2, dtype=np.complex128))
    with pytest.raises(SolveAborted):
        h.matmat(np.zeros((2, 1), dtype=np.complex128))

    ab = ArrayBlock(
        0,
        groups=[],
        shape_of_elem=np.array([], dtype=int),
        shape_blocks={},
        coupling=[],
        cancel=token,
    )
    with pytest.raises(SolveAborted):
        ab.matvec(np.zeros(0, dtype=np.complex128))


def test_deterministic_cancel_mid_sweep():
    token = CancelToken()
    s = _dipole(SinusoidalSolver, cancel=token, nsegs=40)
    k_arr = np.linspace(0.8 * s.k, 1.2 * s.k, 40)
    calls, _ = _trip_on_nth_checkpoint(s, token, n=5)
    with pytest.raises(SolveAborted):
        s.compute_impedance_swept(k_arr)
    # The swept loop checkpoints once per frequency; tripping on the 5th means
    # it aborted after ~4 of 40 frequencies, not the full sweep.
    assert calls[0] == 5


# --------------------------------------------------------------------------
# An aborted solve leaves no instance-cache residue that corrupts a re-run.
# --------------------------------------------------------------------------
def test_abort_leaves_no_instance_cache_residue():
    z_ref, a_ref = _dipole(SinusoidalSolver, nsegs=40).compute_impedance()

    token = CancelToken()
    s = _dipole(SinusoidalSolver, cancel=token, nsegs=40)
    # Trip the 2nd checkpoint: after the fill/assembly, before the dense solve —
    # squarely mid-solve.
    _, real = _trip_on_nth_checkpoint(s, token, n=2)
    with pytest.raises(SolveAborted):
        s.compute_impedance()

    # Clear the token and re-run on the SAME instance: it must match a fresh
    # solver, proving the aborted attempt left no partial state behind.
    s._cancel = None
    s._checkpoint = real
    z2, a2 = s.compute_impedance()
    assert np.allclose(z2, z_ref)
    assert np.allclose(a2, a_ref)
