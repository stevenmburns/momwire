"""How many threads one drop-in answer runs on (momwire#2023).

With nothing set, the drop-in's OpenMP and OpenBLAS pools both ran at full
width on every deck: on a 12-core, 16-thread laptop a 60-segment deck had 31
busy threads and took four times as long as it does on one. The standalone
executable sizes its pools by the deck, and this module gives the Python path
the same rule. The rule itself is the executable's: `_thread_policy.h`, compiled
into both and reached here through the accelerator (:func:`deck_threads`).

Precedence, for each pool on its own:

=========  =========================================================
OpenMP     ``OMP_NUM_THREADS``, else ``MOMWIRE_THREADS``, else the deck
OpenBLAS   ``OPENBLAS_NUM_THREADS``, else ``MOMWIRE_THREADS``, else
           ``GOTO_NUM_THREADS`` or ``OMP_NUM_THREADS`` (which OpenBLAS itself
           reads when it loads), else the deck
=========  =========================================================

A pool pinned by its own variable is left exactly as its library started it.
``MOMWIRE_THREADS=N`` sets both pools to N, as it does in the executable, and
may raise them. The deck's rule only ever lowers a pool.

A busy resident server divides the cores among its workers
(:mod:`momwire.serve._pool`). That share caps everything above, and the BLAS
share stays at two or more, because OpenBLAS's one-thread path rounds
differently from its threaded one. The deck's rule keeps BLAS threaded for the
same reason: a deck on one OpenMP thread gets two BLAS threads, as it does in
the executable. Only an explicit count of one puts BLAS on its serial path.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import sys

FORCE_ENV = "MOMWIRE_THREADS"
OPENMP_ENV = "OMP_NUM_THREADS"
BLAS_ENV = "OPENBLAS_NUM_THREADS"
# Read by OpenBLAS itself when it loads, after its own variable.
BLAS_FALLBACK_ENVS = ("GOTO_NUM_THREADS", OPENMP_ENV)

# The smallest BLAS count the rule or a worker's share leaves: see above.
BLAS_FLOOR = 2


def _acc():
    """The accelerator, on use: a resident worker imports this module before
    it knows which seam it serves."""
    from . import _accel  # noqa: PLC0415

    return _accel.acc


def env_count(name: str, environ=None) -> int:
    """A thread variable's count, 0 when it is unset or not a positive count.

    ``OMP_NUM_THREADS`` may be a per-level list (``4,2``): its first entry is
    the count this process's parallel regions start with.
    """
    environ = os.environ if environ is None else environ
    try:
        count = int(environ.get(name, "").split(",")[0].strip() or 0)
    except ValueError:
        return 0
    return max(count, 0)


def deck_threads(
    n_basis: int, crossing_pairs: int = 0, below_pairs: int = 0
) -> int | None:
    """The executable's count for a deck of this size on this process's
    cores, or None without the compiled rule (a pure-Python install, whose
    pools are then left alone)."""
    rule = getattr(_acc(), "_threads_for", None)
    if rule is None:
        return None
    return int(rule(int(n_basis), int(crossing_pairs), int(below_pairs)))


def allowed_threads() -> int:
    """The logical CPUs this process may run on."""
    pools = getattr(_acc(), "_pool_threads", None)
    if pools is not None:
        return int(pools()["allowed_threads"])
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def targets(
    rule: int | None,
    *,
    environ=None,
    share: int | None = None,
    allowed: int | None = None,
) -> dict[str, int]:
    """The count each pool is set to, by API; a pool absent is left alone.

    ``rule`` is :func:`deck_threads`' count (None: no deck measured yet),
    ``share`` the resident pool's per-worker share (None: a lone worker).
    """
    forced = env_count(FORCE_ENV, environ)
    plan: dict[str, int] = {}
    if not env_count(OPENMP_ENV, environ):
        if forced:
            plan["openmp"] = forced
        elif rule is not None:
            plan["openmp"] = rule
    if not env_count(BLAS_ENV, environ):
        if forced:
            plan["blas"] = forced
        elif rule is not None and not any(
            env_count(name, environ) for name in BLAS_FALLBACK_ENVS
        ):
            cpus = allowed_threads() if allowed is None else allowed
            plan["blas"] = rule if rule > 1 else min(BLAS_FLOOR, cpus)
    if share:
        if "openmp" in plan:
            plan["openmp"] = min(plan["openmp"], share)
        if "blas" in plan:
            plan["blas"] = min(plan["blas"], max(share, BLAS_FLOOR))
    return plan


_controller = None
_controller_modules = -1


def _pools():
    """The process's thread pools, rescanned once new modules have loaded (a
    first solve can bring in another native library)."""
    global _controller, _controller_modules
    if _controller is None or len(sys.modules) != _controller_modules:
        from threadpoolctl import ThreadpoolController  # noqa: PLC0415

        _controller = ThreadpoolController()
        _controller_modules = len(sys.modules)
    return _controller


class _Scope:
    """One answer's pools: set as its solves are measured, restored after."""

    def __init__(self, environ=None, share: int | None = None) -> None:
        self.environ = os.environ if environ is None else environ
        self.share = share
        self.rule: int | None = None
        self.entry: dict[str, tuple[object, int]] = {}

    def apply(self) -> None:
        plan = targets(self.rule, environ=self.environ, share=self.share)
        if not plan:
            return
        forced = env_count(FORCE_ENV, self.environ)
        for lib in _pools().lib_controllers:
            want = plan.get(lib.user_api)
            if want is None:
                continue
            _lib, entry = self.entry.setdefault(lib.filepath, (lib, lib.num_threads))
            # Only MOMWIRE_THREADS may raise a pool past where it stood.
            count = want if forced else min(want, entry)
            if lib.num_threads != count:
                lib.set_num_threads(count)

    def measured(self, n_basis: int, crossing: int, below: int) -> None:
        rule = deck_threads(n_basis, crossing, below)
        if rule is None or (self.rule is not None and rule <= self.rule):
            return
        # The largest solve so far: a deck's later, smaller solve keeps it,
        # as the executable sizes a whole deck by its largest.
        self.rule = rule
        self.apply()

    def used(self, api: str) -> int | None:
        counts = [
            lib.num_threads for lib in _pools().lib_controllers if lib.user_api == api
        ]
        return max(counts) if counts else None

    def restore(self) -> None:
        for lib, entry in self.entry.values():
            if lib.num_threads != entry:
                lib.set_num_threads(entry)


_SCOPE: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar(
    "momwire_thread_scope", default=None
)
_SHARE: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "momwire_thread_share", default=None
)


@contextlib.contextmanager
def frame_share(share: int | None):
    """The resident pool's per-worker share for the frame inside."""
    token = _SHARE.set(share)
    try:
        yield
    finally:
        _SHARE.reset(token)


@contextlib.contextmanager
def answering(environ=None):
    """One answer's thread counts; an inner call joins the outer answer."""
    if _SCOPE.get() is not None:
        yield
        return
    scope = _Scope(environ, _SHARE.get())
    token = _SCOPE.set(scope)
    try:
        scope.apply()
        yield
    finally:
        _SCOPE.reset(token)
        _record(scope)
        scope.restore()


def measured(n_basis: int, crossing_pairs: int = 0, below_pairs: int = 0) -> None:
    """A solve of this size is about to fill: size the answer's pools."""
    scope = _SCOPE.get()
    if scope is not None:
        scope.measured(n_basis, crossing_pairs, below_pairs)
        _record(scope)


def _record(scope: _Scope) -> None:
    """The counts in force, kept on the answer's timing tally, whose ``times:``
    line is written after this scope has closed."""
    from . import _times  # noqa: PLC0415 - `_times` reads this module

    times = _times.current()
    if times is not None:
        times.used_threads = (scope.used("openmp"), scope.used("blas"))


def used(api: str) -> int | None:
    """The count ``api`` (``"openmp"`` or ``"blas"``) runs on in the answer in
    progress, read from the pool; None outside an answer."""
    scope = _SCOPE.get()
    return None if scope is None else scope.used(api)
