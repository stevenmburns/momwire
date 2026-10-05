"""Cooperative cancellation for in-flight solves.

A :class:`CancelToken` is a single int32 in shared memory. A caller creates one,
threads it into a solver (``cancel=`` on the constructor), and flips it from any
thread the moment the solve becomes stale. The solver polls it at cheap seams —
phase boundaries, sweep/ACA/GMRES loop iterations — via :meth:`_checkpoint`, and
raises :class:`SolveAborted` promptly instead of running the doomed solve to
completion. The raw flag address (:attr:`CancelToken.ptr`) is also handed to the
C++ kernels so they can poll it per outer-loop iteration without touching the
GIL (that is a later phase; the plumbing lives here).

A relaxed racy read of the flag is fine: the only transition is 0 -> 1, and a
missed read costs at most one extra poll interval. Default ``cancel=None`` makes
every checkpoint a single ``is not None`` test — zero cost for existing callers.
"""

import contextlib
import threading

import numpy as np


class SolveAborted(Exception):
    """Solve was cancelled via a :class:`CancelToken`; no result was produced."""


class CancelToken:
    """A one-shot, thread-safe cancellation flag shared with a solve.

    The token owns the flag's memory; while a solver holds the token, the raw
    pointer handed to the C++ kernels stays valid for the duration of any kernel
    call. Cancellation is idempotent and monotonic (0 -> 1, never back).
    """

    def __init__(self):
        self._flag = np.zeros(1, dtype=np.int32)

    def cancel(self):
        """Request cancellation. Safe to call from any thread; idempotent."""
        self._flag[0] = 1

    @property
    def cancelled(self):
        """True once :meth:`cancel` has been called."""
        return bool(self._flag[0])

    def raise_if_cancelled(self):
        """Raise :class:`SolveAborted` if cancellation has been requested."""
        if self._flag[0]:
            raise SolveAborted()

    @property
    def ptr(self):
        """Raw address of the flag's int32, for the C++ kernels to poll."""
        return self._flag.ctypes.data


class _Cancelable:
    """Mixin giving a solver a cancel token and a cheap checkpoint helper.

    Solvers set ``self._cancel`` in their constructor (default ``None``); the
    class-level ``None`` covers any construction path that doesn't. When no
    token is present, :meth:`_checkpoint` is a single ``is not None`` test.
    """

    _cancel = None

    def _checkpoint(self):
        """Raise :class:`SolveAborted` if this solver's token has been tripped."""
        cancel = self._cancel
        if cancel is not None:
            cancel.raise_if_cancelled()

    @property
    def _cancel_flag(self):
        """Raw flag address for the C++ kernels' ``cancel_flag`` argument.

        0 when no token is present, which the kernels treat as "no cancellation"
        — so passing this unconditionally is free on the default path.
        """
        cancel = self._cancel
        return cancel.ptr if cancel is not None else 0


# --------------------------------------------------------------------------
# Ambient token (momwire#1342).
#
# The crossing fill (`_crossing_fill`) and the near-interface tables
# (`_near_interface`) are free functions: the fill receives its solver's data
# as a `CrossingContext`, and the tables sit several calls below it with no
# context at all (`designed_rows`, `sheet_plan`, `PlaneSheet.cover`, the column
# twin). A token argument would have to be threaded through ~40 signatures,
# four memo classes and the razor, bspline, sinusoidal and Galerkin callers.
# Instead each public fill entry installs its context's token for the duration
# of the call (`scope`), and the seams below read it (`poll`, `flag`).
#
# Thread-local on purpose. The scope is entered and read on the thread that
# runs the fill; a pool thread started beneath it sees no token and so polls
# nothing, which costs latency and never correctness (the owning thread polls
# again when it collects the pool's result). A scope restores the PREVIOUS
# token on exit, so a fill nested inside another (the two-radius node calls
# two single-radius fills) leaves the outer one installed.
_ambient = threading.local()


@contextlib.contextmanager
def scope(token):
    """Make `token` (a `CancelToken` or None) the ambient one for the block."""
    prev = getattr(_ambient, "token", None)
    _ambient.token = token
    try:
        yield
    finally:
        _ambient.token = prev


def poll():
    """Raise :class:`SolveAborted` if the ambient token has been tripped. One
    attribute read when there is none."""
    token = getattr(_ambient, "token", None)
    if token is not None and token._flag[0]:
        raise SolveAborted()


def flag():
    """The ambient token's raw flag address for a kernel's `cancel_flag`, or 0
    (no cancellation) when none is installed."""
    token = getattr(_ambient, "token", None)
    return token.ptr if token is not None else 0
