"""Seams for the worker-pool tests (momwire#1418), built INSIDE a worker.

Not a test module: ``tests/test_serve_pool.py`` puts this directory on the
worker's ``PYTHONPATH`` and names these factories in the pool's spec, so a
worker process imports this file exactly as a real worker imports
``momwire.eznec._shell``.

``stub`` answers by command, one per body:

* ``SLEEP <s>`` -- sleep, then report who answered and when (wall clock);
* ``CRASH`` -- the process dies mid-frame, as a segfault or an OOM kill would;
* ``RAISE`` -- ``answer`` raises (a seam breaking its contract);
* ``THREADS`` -- the BLAS pool's thread count as this frame sees it;
* ``PRINT`` -- a stray ``print`` before answering, which must not reach the
  protocol.

``crashing_eznec`` is the REAL eznec seam, except that a deck whose comment
says ``CRASHME`` kills the worker solving it.
"""

from __future__ import annotations

import json
import os
import time

from momwire.serve import Seam


def stub() -> Seam:
    def answer(body: str, terminator: str) -> tuple[str, str]:
        word, _, arg = body.strip().partition(" ")
        if word == "SLEEP":
            t0 = time.time()
            time.sleep(float(arg))
            return (
                json.dumps({"pid": os.getpid(), "t0": t0, "t1": time.time()}),
                f"slept {arg}\n",
            )
        if word == "CRASH":
            os._exit(17)
        if word == "RAISE":
            raise RuntimeError("the stub seam was told to raise")
        if word == "THREADS":
            from threadpoolctl import threadpool_info

            info = threadpool_info()
            blas = [i["num_threads"] for i in info if i["user_api"] == "blas"]
            omp = [i["num_threads"] for i in info if i["user_api"] == "openmp"]
            return json.dumps({"pid": os.getpid(), "blas": blas, "openmp": omp}), ""
        if word == "PRINT":
            print("this must not reach the protocol")
            return "printed", ""
        return f"echo {body}|{terminator}", ""

    return Seam(
        name="stub",
        greeting=lambda: [],
        terminators=frozenset(),
        closing=frozenset(),
        eof_terminator="",
        answer=answer,
    )


def crashing_eznec(**kwargs) -> Seam:
    from momwire.eznec._shell import seam

    real = seam(**kwargs)

    def answer(body: str, terminator: str) -> tuple[str, str]:
        if "CRASHME" in body:
            os._exit(17)
        return real.answer(body, terminator)

    return Seam(
        name=real.name,
        greeting=real.greeting,
        terminators=real.terminators,
        closing=real.closing,
        eof_terminator=real.eof_terminator,
        answer=answer,
    )
