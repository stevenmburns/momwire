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
