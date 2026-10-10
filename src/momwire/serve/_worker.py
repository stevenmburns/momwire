"""One worker of the resident server's pool (momwire#1418).

``python -m momwire.serve._worker`` -- or the frozen engine started with
``--serve-worker`` -- reads messages on stdin and writes replies on stdout,
in :mod:`momwire.serve._pool`'s framing:

1. the spec: which seam to build (``"module:callable"`` and its kwargs);
   the worker builds it and answers ``{"ready": pid, "status": ...}``;
2. then frames, ``{"body", "terminator", "threads"}``, each answered with
   ``{"out", "err", "pid", "t0", "t1"}`` -- the seam's two chunks, verbatim,
   plus who answered and when (wall clock, so a trace from several workers
   lines up);
3. end of stdin is the end of the worker.

stdout carries nothing else: the protocol takes a duplicate of file
descriptor 1 and the descriptor itself is pointed at the null device, so a
stray ``print`` -- or a C library writing to fd 1 -- cannot corrupt a
message. stderr is the server's log (the pool drains it there).
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import time

from ._pool import read_message, write_message


def _resolve(path: str):
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _claim_stdout():
    """A private binary handle on the original stdout; fd 1 goes to null."""
    sys.stdout.flush()
    fd = os.dup(1)
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415 - Windows only

        msvcrt.setmode(fd, os.O_BINARY)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    sys.stdout = sys.stderr
    return os.fdopen(fd, "wb")


class _Threads:
    """Limit the BLAS and OpenMP pools for one frame, or not at all.

    A limit only ever LOWERS a pool: each API is held to the smaller of the
    pool's share and the count this worker already runs with, so a server
    started under ``OMP_NUM_THREADS=1`` (a bisect, a test lane) keeps one
    thread rather than being raised to its share. Thread count is not
    numerically inert -- a reduction split across threads rounds
    differently -- so raising a pinned count would move answers.

    The controller is rebuilt whenever new modules have been imported since
    it was built, because it only sees the native libraries loaded at
    construction and a first solve can load more.
    """

    def __init__(self) -> None:
        self._controller = None
        self._modules = -1

    def limit(self, threads):
        if not threads:
            return contextlib.nullcontext()
        if self._controller is None or len(sys.modules) != self._modules:
            from threadpoolctl import ThreadpoolController  # noqa: PLC0415

            self._controller = ThreadpoolController()
            self._modules = len(sys.modules)
        limits = {}
        for api in ("blas", "openmp"):
            counts = [
                lib.num_threads
                for lib in self._controller.lib_controllers
                if lib.user_api == api
            ]
            if counts:
                limits[api] = min(int(threads), *counts)
        if not limits:
            return contextlib.nullcontext()
        return self._controller.limit(limits=limits)


def worker_main(argv: list[str] | None = None) -> int:
    del argv  # everything arrives on stdin
    out = _claim_stdout()
    stdin = sys.stdin.buffer
    spec = read_message(stdin)
    if spec is None:
        return 0
    seam = _resolve(spec["factory"])(**spec.get("kwargs", {}))
    status = ""
    if spec.get("status"):
        try:
            status = _resolve(spec["status"])()
        except Exception as exc:  # noqa: BLE001 - a status line must not stop the worker
            status = f"status unavailable: {type(exc).__name__}: {exc}"
    write_message(out, {"ready": os.getpid(), "status": status})

    threads = _Threads()
    pid = os.getpid()
    while True:
        frame = read_message(stdin)
        if frame is None:
            return 0
        t0 = time.time()
        try:
            with threads.limit(frame.get("threads")):
                chunk_out, chunk_err = seam.answer(frame["body"], frame["terminator"])
            reply = {"out": chunk_out, "err": chunk_err}
        except Exception as exc:  # noqa: BLE001 - carried to the front end, which re-raises
            reply = {"raised": type(exc).__name__, "message": str(exc)}
        reply.update(pid=pid, t0=t0, t1=time.time())
        write_message(out, reply)


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(worker_main(sys.argv[1:]))
