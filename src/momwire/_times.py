"""Opt-in per-stage timing for the NEC-5 seam: ``MOMWIRE_TIMES=1``.

Where an answer's time goes, for a user who wants to see it: the printout's
``MATRIX TIMING`` (``FILL=`` / ``FACTOR=``) and ``RUN TIME`` lines carry
measured seconds in NEC-5's own places instead of the zeros momwire writes
by default, and each answer writes one ``times:`` line on stderr
(:func:`momwire_serve_client.times_line`, the same keys the standalone
executables write).

The tally is EXCLUSIVE wall time per stage, one answer at a time: a
:func:`stage` names the stage the code inside it is in, a nested one switches
the clock to the inner stage and back, so a ground table filled in the middle
of the fill is charged to ``ground`` and not to ``fill`` as well. The C++
extension keeps the same tally for the ground tables and the LU it runs
(``_mw_times.h``), and each switch here folds its share in.

OFF (the default) it is one :class:`contextvars.ContextVar` read per stage
boundary and nothing else: no clock is read, nothing is written, and no
number the solve computes is touched. The answer is held in a context
variable, so a resident server's concurrent requests keep their own tallies
(the C++ half is per process: two TIMED requests at once in one process share
its ground and factor counts).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import time

import momwire_serve_client as _mech

from . import _accel

# The stages, in the `times:` line's order (`_mw_times.h`'s, with the write).
STAGES = ("seam", "ground", "fill", "factor", "rp", "near", "run", "printout", "write")

_CURRENT: contextvars.ContextVar[Times | None] = contextvars.ContextVar(
    "momwire_times", default=None
)
_OFF = contextlib.nullcontext()


def import_ms() -> float:
    """Milliseconds since the `momwire` package started importing (the
    one-shot's ``load_ms``, read when its answer starts)."""
    import momwire

    t0 = getattr(momwire, "_IMPORT_T0", None)
    return 0.0 if t0 is None else (time.perf_counter() - t0) * 1e3


def env_on(environ=None) -> bool:
    """``MOMWIRE_TIMES`` set to anything but ``""`` / 0 / off / false / no."""
    return _mech.times_requested(os.environ if environ is None else environ)


def _cxx():
    acc = _accel.acc
    return acc if acc is not None and hasattr(acc, "stage_times_read") else None


class Times:
    """One answer's tally (see the module docstring)."""

    def __init__(self, *, route: str, start_ms: float = 0.0, load_ms: float = 0.0):
        now = time.perf_counter()
        self.route = route
        self.origin = now
        self.start_ms = start_ms
        self.load_ms = load_ms
        self.ms = dict.fromkeys((*STAGES, "none"), 0.0)
        self.counts = {"built": 0, "disk": 0, "blended": 0}
        self.runs = 0
        self._cur = "none"
        self._last = now
        self._acc = _cxx()
        self._cxx_last = (0.0, 0.0, 0, 0, 0)
        if self._acc is not None:
            self._acc.stage_times_begin()

    def _switch(self, to: str) -> str:
        now = time.perf_counter()
        elapsed = (now - self._last) * 1e3
        if self._acc is not None:
            cur = tuple(self._acc.stage_times_read())
            last = self._cxx_last
            d_ground, d_factor = cur[0] - last[0], cur[1] - last[1]
            self._cxx_last = cur
            # The C++ half's ground and LU ran inside this stretch: theirs.
            self.ms["ground"] += d_ground
            self.ms["factor"] += d_factor
            elapsed -= d_ground + d_factor
            for key, a, b in zip(
                ("built", "disk", "blended"), cur[2:], last[2:], strict=True
            ):
                self.counts[key] += a - b
        self.ms[self._cur] += max(elapsed, 0.0)
        self._last = now
        prev, self._cur = self._cur, to
        return prev

    def settle(self) -> None:
        self._switch(self._cur)

    def elapsed_ms(self, name: str) -> float:
        self.settle()
        return self.ms[name]

    def run_seconds(self) -> float:
        """The printout's RUN TIME: this answer's seconds so far, the
        process's start and load included where they are known."""
        return (self.start_ms + self.load_ms) / 1e3 + time.perf_counter() - self.origin

    def line(self, *, threads: int | None = None) -> str:
        """The ``times:`` line (:func:`momwire_serve_client.times_line`)."""
        self.settle()
        if self._acc is not None:
            self._acc.stage_times_end()
            self._acc = None
        total = self.start_ms + self.load_ms + (time.perf_counter() - self.origin) * 1e3
        cells = {"start": self.start_ms, "load": self.load_ms}
        cells.update((name, self.ms[name]) for name in STAGES)
        return _mech.times_line(
            self.route,
            cells,
            total_ms=total,
            threads=default_threads() if threads is None else threads,
            runs=self.runs,
            counts=self.counts,
        )


def default_threads() -> int:
    """The thread count the Python path runs with: MOMWIRE_THREADS, else
    every hardware thread (the launcher's default)."""
    try:
        forced = int(os.environ.get("MOMWIRE_THREADS", "") or 0)
    except ValueError:
        forced = 0
    return forced if forced > 0 else (os.cpu_count() or 1)


def current() -> Times | None:
    return _CURRENT.get()


@contextlib.contextmanager
def answering(route: str, *, start_ms: float = 0.0, load_ms: float = 0.0):
    """Time one answer: yields its :class:`Times` (the caller writes its
    line), the base stage ``seam`` until something inner claims the clock."""
    times = Times(route=route, start_ms=start_ms, load_ms=load_ms)
    token = _CURRENT.set(times)
    times._switch("seam")
    try:
        yield times
    finally:
        _CURRENT.reset(token)
        if times._acc is not None:
            times.settle()
            times._acc.stage_times_end()
            times._acc = None


class _Stage:
    __slots__ = ("_name", "_prev", "_times")

    def __init__(self, times: Times, name: str):
        self._times = times
        self._name = name

    def __enter__(self):
        self._prev = self._times._switch(self._name)
        return self

    def __exit__(self, *exc):
        self._times._switch(self._prev)
        return False


def stage(name: str):
    """The code inside is in stage ``name`` (a no-op when nothing is timed)."""
    times = _CURRENT.get()
    if times is None:
        return _OFF
    return _Stage(times, name)


def staged(name: str):
    """A decorator: the function runs in stage ``name`` (see :func:`stage`)."""

    def deco(fn):
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)

        return timed

    return deco


def count(kind: str, n: int = 1) -> None:
    """A ground table obtained: ``built``, ``disk`` or ``blended``."""
    times = _CURRENT.get()
    if times is not None:
        times.counts[kind] += n


def count_run() -> None:
    times = _CURRENT.get()
    if times is not None:
        times.runs += 1
