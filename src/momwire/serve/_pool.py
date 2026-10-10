"""The resident server's worker pool: one known address, N solving processes.

momwire#1418. The resident server answered every connection in ONE process
behind one global solve lock, so SimNEC's crew of N engine processes queued
behind each other: a crew of 2 ran at the speed of a crew of 1 (antennaknobs
``scratch/dan-simnec-speed/REPORT.md``, cause 1). The front end now keeps the
address the clients already know and hands each frame to a WORKER process;
the protocol, the native launcher and the Python client cannot tell.

Handoff: the front end proxies, it does not pass the socket
--------------------------------------------------------------
The front end keeps the client's connection and runs the session framing it
always ran (:func:`momwire.serve.run_session`). What moves is the one call
the lock used to wrap -- ``seam.answer(body, terminator)`` -- which now goes
to a worker over that worker's stdin/stdout pipes as a length-prefixed JSON
message, and the ``(stdout, stderr)`` chunks come back the same way. Chosen
over handing the accepted socket itself to the worker because:

* **it is the same code on Windows and POSIX.** Passing a socket needs
  ``SCM_RIGHTS`` on POSIX and ``WSADuplicateSocket`` (``socket.share``) on
  Windows -- two mechanisms, two failure surfaces, and only one of them runs
  on the box the suite runs on. Anonymous pipes to a child process are one
  mechanism everywhere.
* **a crashed worker cannot take the caller's answer with it.** The front end
  still owns the connection when its worker dies, so the caller gets the
  seam's own internal-error frame -- exactly what an exception inside an
  in-process solve produced -- rather than a reset socket.
* **the cost is a copy of a deck and a printout through a pipe**: kilobytes
  to a few megabytes against a solve of tens of milliseconds and up.

Workers are started with :mod:`subprocess` running an explicit command, not
:mod:`multiprocessing`: nothing is forked and nothing is pickled, so there is
no fork-only behaviour to lean on by accident, and the frozen Windows engine
needs no ``freeze_support`` re-entry -- a worker is that same exe started with
``--serve-worker`` (``scripts/eznec_freeze/entry.py``), and a source install
runs ``python -m momwire.serve._worker``.

Sizing
------
* **Spawn on demand.** One worker is started with the server. A frame that
  finds every live worker busy starts another, up to the cap, and meanwhile
  takes whichever worker frees first -- so a spawn never adds its start-up
  time to the frame that asked for it.
* **The cap** is ``min(cores, RAM / 4 / 150 MB)``: one worker per core the
  process may run on, and never more than a quarter of physical memory at the
  ~150 MB a warm worker holds (measured on the SimNEC replay decks). On an
  8-thread box with 8 GB that is 8 workers, ~1.2 GB in all.
  ``MOMWIRE_SERVE_WORKERS=N`` overrides it; ``MOMWIRE_SERVE_WORKERS=0`` turns
  the pool off and answers in-process under the old lock.
* **Threads are divided.** A frame is solved with its OpenMP pool limited to
  ``cores // live workers`` and its BLAS pool to the same share but never
  below two; a lone worker is not limited at all, so a single caller keeps
  exactly today's thread count. The floor is for the answers, not the
  speed: OpenBLAS's one-thread path rounds differently from its threaded
  one (measured: the 4-square fixture's near-zero columns move between
  ``OPENBLAS_NUM_THREADS=1`` and ``2``, while 2, 4 and 8 agree to the bit),
  so a share of one would make a crew's printout differ in its last digits
  from a lone server's. BLAS is a sliver of these solves, so two threads
  per worker oversubscribe nothing that matters. No limit ever RAISES a
  count the worker was started with.
* **Idle workers are reaped** on the server's own idle rule: one that has
  answered nothing for ``--idle-timeout`` seconds is stopped, keeping one.

What is NOT shared any more is in-process state: each worker has its own
caches (the far-field phase cache, warm quadrature tables). A multi-run
deck's shared fill (momwire#1397) is unaffected -- one deck is one frame, and
one frame is answered by one worker.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
import subprocess
import sys
import threading
import time
from dataclasses import replace

# The pool's size override: unset = the automatic cap, ``0`` = no pool (the
# in-process, one-lock server), ``N`` = at most N workers.
WORKERS_ENV = "MOMWIRE_SERVE_WORKERS"

# ``1`` writes one log line per answered frame naming the worker pid and the
# worker-side wall interval. Off by default: a resident server answers
# thousands of frames and its log is a diagnostic, not a ledger.
TRACE_ENV = "MOMWIRE_SERVE_TRACE"

# The frozen engine's spelling of "be a worker" (entry.py routes it).
WORKER_FLAG = "--serve-worker"

# What a warm worker holds, in MB, for the memory half of the cap.
WORKER_MB = 150

# How long a worker may take to import and say it is ready.
_READY_TIMEOUT = 120.0

# After a worker fails to start, how long to answer in-process before trying
# to start one again, so a broken install does not pay a process launch per
# frame.
_SPAWN_BACKOFF = 60.0

# How long ``close`` waits for a worker to leave on its own.
_CLOSE_WAIT = 5.0

_HEADER = struct.Struct(">Q")

# Windows: the server runs detached, without a console, and a console child
# of a console-less parent gets a NEW console window unless told otherwise.
_CREATE_NO_WINDOW = 0x08000000


class WorkerLost(RuntimeError):
    """The worker answering a frame died before it answered."""


class WorkerRaised(RuntimeError):
    """``seam.answer`` raised inside a worker (a seam broke its contract)."""


# ---------------------------------------------------------------- wire ---


def write_message(stream, obj) -> None:
    """One message: an 8-byte big-endian length, then that much ASCII JSON.

    ``ensure_ascii`` so any ``str`` -- lone surrogates included -- crosses
    and comes back as the same ``str``.
    """
    data = json.dumps(obj, ensure_ascii=True).encode("ascii")
    stream.write(_HEADER.pack(len(data)) + data)
    stream.flush()


def _read_exact(stream, size: int) -> bytes | None:
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            return None
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def read_message(stream):
    """The next message, or None at a clean end of stream."""
    header = _read_exact(stream, _HEADER.size)
    if header is None:
        return None
    (size,) = _HEADER.unpack(header)
    data = _read_exact(stream, size)
    if data is None:
        raise EOFError("the stream ended inside a message")
    return json.loads(data.decode("ascii"))


# -------------------------------------------------------------- sizing ---


def usable_cores() -> int:
    """The CPUs this process may run on (its affinity where the OS says)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def physical_memory() -> int | None:
    """Total physical memory in bytes, or None where it cannot be asked."""
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def default_max_workers(cores: int | None = None, memory: int | None = None) -> int:
    """``min(cores, RAM / 4 / WORKER_MB)``, and at least one."""
    cap = usable_cores() if cores is None else cores
    memory = physical_memory() if memory is None else memory
    if memory:
        cap = min(cap, memory // 4 // (WORKER_MB << 20))
    return max(1, int(cap))


def configured_max_workers(environ=None) -> tuple[int, str]:
    """The cap this server runs with, and the sentence its log says it in.

    ``0`` means no pool. A value that is not a non-negative integer is
    reported and ignored rather than refused: a typo in an environment
    variable must not cost the user their engine.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(WORKERS_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = -1
        if value >= 0:
            return value, f"{WORKERS_ENV}={value}"
        auto = default_max_workers()
        return auto, f"{WORKERS_ENV}={raw!r} is not a count; using {auto}"
    auto = default_max_workers()
    return auto, f"automatic: {usable_cores()} cores, {WORKER_MB} MB per worker"


def worker_command() -> list[str]:
    """How to start a worker from THIS process.

    A frozen engine is its own interpreter: the same exe, told to be a worker
    by a flag only this module spells. A source install is ``-m``.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, WORKER_FLAG]
    return [sys.executable, "-m", "momwire.serve._worker"]


def thread_budget(live: int, cores: int) -> int | None:
    """Each worker's share of the cores; None (no limit) for a lone worker."""
    if live <= 1:
        return None
    return max(1, cores // live)


# -------------------------------------------------------------- worker ---


class _Worker:
    """One child process, its pipes, and its bookkeeping (under the pool's
    condition)."""

    def __init__(self, proc: subprocess.Popen, pid: int) -> None:
        self.proc = proc
        # The pid the WORKER reported, not ``proc.pid``: a Windows venv's
        # ``python.exe`` is a launcher that runs the real interpreter as its
        # child, so the process solving is not the process started. The
        # launcher still owns the pipes, so its exit is still the worker's.
        self.pid = pid
        self.busy = False
        self.last_used = time.monotonic()
        self.answered = 0

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, message) -> None:
        write_message(self.proc.stdin, message)

    def receive(self):
        reply = read_message(self.proc.stdout)
        if reply is None:
            raise EOFError("the worker closed its pipe")
        return reply

    def exit_code(self, wait: float = 2.0):
        try:
            return self.proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            return None

    def stop(self, wait: float = _CLOSE_WAIT) -> None:
        """Close its stdin (a worker leaves at end of input), then insist."""
        with contextlib.suppress(OSError, ValueError):
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                self.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=wait)


class _NoWorker(Exception):
    """No worker exists and none can be started right now."""


# ---------------------------------------------------------------- pool ---


class WorkerPool:
    """Workers for one seam. Thread-safe; one instance per server process.

    ``spec`` is what a worker needs to build the seam itself:
    ``{"factory": "module:callable", "kwargs": {...}}``, optionally
    ``"status": "module:callable"`` for the one line the worker reports when
    ready (the accelerator line, for the eznec seam). JSON-able on purpose: it
    crosses into the worker as the first message on its stdin.
    """

    def __init__(
        self,
        spec: dict,
        log,
        *,
        max_workers: int,
        idle_timeout: float,
        cores: int | None = None,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        trace: bool | None = None,
    ) -> None:
        self.spec = spec
        self.log = log
        self.max_workers = max(1, int(max_workers))
        self.idle_timeout = idle_timeout
        self.cores = usable_cores() if cores is None else max(1, int(cores))
        self.command = worker_command() if command is None else list(command)
        self.env = env
        self.trace = (
            os.environ.get(TRACE_ENV, "") not in ("", "0") if trace is None else trace
        )
        self.cond = threading.Condition()
        self.workers: list[_Worker] = []
        self.starting = 0
        self.waiting = 0
        self.closed = False
        self.spawn_failed_at: float | None = None
        self.spawned = 0
        # The in-process fallback's lock: today's one global solve lock, for
        # the case where no worker can be started at all.
        self.fallback_lock = threading.Lock()

    # -- logging ------------------------------------------------------

    def note(self, text: str) -> None:
        try:
            self.log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pool: {text}\n")
            self.log.flush()
        except (OSError, ValueError):
            pass

    # -- life cycle ----------------------------------------------------

    def start(self) -> None:
        """Start the first worker now, so the first frame does not wait for a
        whole import on top of the server's own."""
        with self.cond:
            self._spawn_locked()

    def close(self) -> None:
        with self.cond:
            self.closed = True
            workers, self.workers = self.workers, []
            self.cond.notify_all()
        for worker in workers:
            worker.stop()
        if workers:
            self.note(f"stopped {len(workers)} worker(s)")

    def live(self) -> list[int]:
        """The pids of the live workers (for logs and tests)."""
        with self.cond:
            return [w.pid for w in self.workers if w.alive()]

    # -- spawning --------------------------------------------------------

    def _spawn_locked(self) -> None:
        self.starting += 1
        threading.Thread(target=self._spawn, daemon=True).start()

    def _spawn(self) -> None:
        started = time.monotonic()
        proc = None
        try:
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = _CREATE_NO_WINDOW
            proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                close_fds=True,
                **kwargs,
            )
            threading.Thread(target=self._drain, args=(proc,), daemon=True).start()
            write_message(proc.stdin, self.spec)
            ready = self._await_ready(proc)
        except BaseException as exc:  # noqa: BLE001 - reported, then the fallback answers
            if proc is not None:
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                    proc.wait(timeout=_CLOSE_WAIT)
            self.note(
                f"a worker failed to start ({type(exc).__name__}: {exc}); "
                f"frames are answered in-process for the next {_SPAWN_BACKOFF:g}s"
            )
            with self.cond:
                self.starting -= 1
                self.spawn_failed_at = time.monotonic()
                self.cond.notify_all()
            return
        worker = _Worker(proc, int(ready["ready"]))
        with self.cond:
            self.starting -= 1
            if self.closed:
                stale = worker
            else:
                stale = None
                self.workers.append(worker)
                self.spawned += 1
                self.spawn_failed_at = None
                count = len(self.workers)
            self.cond.notify_all()
        if stale is not None:
            stale.stop()
            return
        self.note(
            f"worker pid={worker.pid} ready in {time.monotonic() - started:.2f}s "
            f"({count} of at most {self.max_workers})"
            + (f": {ready['status']}" if ready.get("status") else "")
        )

    def _await_ready(self, proc: subprocess.Popen) -> dict:
        """The worker's first message, or an exception: it died, or it took
        longer than ``_READY_TIMEOUT`` (then it is killed)."""
        timer = threading.Timer(_READY_TIMEOUT, proc.kill)
        timer.daemon = True
        timer.start()
        try:
            ready = read_message(proc.stdout)
        finally:
            timer.cancel()
        if ready is None or "ready" not in ready:
            code = None
            with contextlib.suppress(subprocess.TimeoutExpired):
                code = proc.wait(timeout=_CLOSE_WAIT)
            raise WorkerLost(f"the worker exited before it was ready (code {code})")
        return ready

    def _drain(self, proc: subprocess.Popen) -> None:
        """A worker's stderr, line by line, into the server log -- the same
        place an in-process solve's warnings went."""
        prefix = f"[worker {proc.pid}] "
        for raw in iter(proc.stderr.readline, b""):
            try:
                self.log.write(prefix + raw.decode("utf-8", "replace"))
                self.log.flush()
            except (OSError, ValueError):
                pass
        with contextlib.suppress(OSError):
            proc.stderr.close()

    # -- checkout ----------------------------------------------------------

    def _acquire(self) -> tuple[_Worker, int | None]:
        with self.cond:
            self.waiting += 1
            try:
                while True:
                    if self.closed:
                        raise WorkerLost("the server is stopping")
                    self._discard_dead_locked()
                    for worker in self.workers:
                        if not worker.busy:
                            worker.busy = True
                            return worker, thread_budget(len(self.workers), self.cores)
                    backing_off = (
                        self.spawn_failed_at is not None
                        and time.monotonic() - self.spawn_failed_at < _SPAWN_BACKOFF
                    )
                    if (
                        not backing_off
                        and self.starting < self.waiting
                        and len(self.workers) + self.starting < self.max_workers
                    ):
                        self._spawn_locked()
                    if not self.workers and not self.starting and backing_off:
                        raise _NoWorker
                    self.cond.wait(timeout=1.0)
            finally:
                self.waiting -= 1

    def _release(self, worker: _Worker) -> None:
        with self.cond:
            worker.busy = False
            worker.last_used = time.monotonic()
            worker.answered += 1
            idle = self._reap_locked()
            self.cond.notify_all()
        for gone in idle:
            gone.stop()
            self.note(f"worker pid={gone.pid} idle {self.idle_timeout:g}s; stopped")

    def _discard_dead_locked(self) -> None:
        dead = [w for w in self.workers if not w.busy and not w.alive()]
        for worker in dead:
            self.workers.remove(worker)
            self.note(
                f"worker pid={worker.pid} exited while idle "
                f"(code {worker.proc.returncode}); dropped"
            )

    def _reap_locked(self) -> list[_Worker]:
        now = time.monotonic()
        idle = [
            w
            for w in self.workers
            if not w.busy and now - w.last_used >= self.idle_timeout
        ]
        # Keep one: the server itself idles out on the same clock, and a
        # caller that is still here should not pay a spawn.
        if len(idle) == len(self.workers):
            idle = idle[1:]
        for worker in idle:
            self.workers.remove(worker)
        return idle

    # -- answering ---------------------------------------------------------

    def answer(
        self, seam, body: str, terminator: str, crashed, label: str = ""
    ) -> tuple[str, str]:
        """``seam.answer(body, terminator)``, solved in a worker.

        ``crashed(body, terminator, exc)`` frames a worker that died mid-frame
        the way the seam frames an exception of its own. A frame that never
        reached its worker (the pipe was already broken) is retried on
        another; one that did is not, because it may be what killed it.
        """
        for _attempt in range(3):
            try:
                worker, threads = self._acquire()
            except _NoWorker:
                with self.fallback_lock:
                    return seam.answer(body, terminator)
            try:
                worker.send(
                    {"body": body, "terminator": terminator, "threads": threads}
                )
            except (OSError, ValueError):
                self._lost(worker, label, "before the frame reached it")
                continue
            try:
                reply = worker.receive()
            except (OSError, ValueError, EOFError):
                code = self._lost(worker, label, "while solving")
                return crashed(
                    body,
                    terminator,
                    WorkerLost(
                        f"the worker process (pid {worker.pid}) exited with "
                        f"code {code} while solving"
                    ),
                )
            self._release(worker)
            if self.trace:
                self.note(
                    f"{label}worker pid={reply.get('pid')} t0={reply.get('t0'):.6f} "
                    f"t1={reply.get('t1'):.6f} threads={threads}"
                )
            if "raised" in reply:
                raise WorkerRaised(f"{reply['raised']}: {reply.get('message', '')}")
            return reply["out"], reply["err"]
        raise WorkerLost("no worker took the frame after three attempts")

    def _lost(self, worker: _Worker, label: str, when: str):
        with self.cond:
            if worker in self.workers:
                self.workers.remove(worker)
            self.cond.notify_all()
        with contextlib.suppress(OSError):
            worker.proc.kill()
        code = worker.exit_code()
        self.note(f"{label}worker pid={worker.pid} lost {when} (exit code {code})")
        return code

    def bind(self, seam, crashed, label: str = ""):
        """``seam`` with its answer solved in this pool; framing unchanged."""

        def answer(body: str, terminator: str) -> tuple[str, str]:
            return self.answer(seam, body, terminator, crashed, label)

        return replace(seam, answer=answer)
