"""The resident server's worker pool (momwire#1418).

The claim is concurrency, so the gate is concurrency observed, not inferred
from a faster clock: N frames sent at once must be answered by N DISTINCT
worker processes whose solve intervals OVERLAP -- each worker reports its
own pid and its own wall-clock interval, so a pool that quietly serialised
(one lock, one worker, a queue) cannot pass by being fast. Then the
properties the issue promises around it: the bytes a caller reads are the
one-shot file's, a crashed worker reaches its caller as the seam's own
internal-error frame and leaves the server answering, a lone worker keeps
today's thread count, the cap holds, idle workers go, and a pool that cannot
start a worker still answers.

Two levels. The pool's own tests drive ``WorkerPool`` against the stub seams
in ``tests/_pool_seams.py`` (built inside the worker, like a real seam); the
end-to-end tests start the REAL daemon, ``python -m momwire.eznec --serve``,
and talk to it over its socket the way the native launcher does.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

import momwire_serve_client as mech
from momwire.eznec import _resident, _shell
from momwire.serve import _pool
from momwire.serve._pool import (
    WorkerLost,
    WorkerPool,
    WorkerRaised,
    configured_max_workers,
    default_max_workers,
    read_message,
    thread_budget,
    write_message,
)
from momwire.serve._server import Server

HERE = Path(__file__).resolve().parent
DECKS = HERE / "fixtures" / "eznec" / "decks"
SRC_ROOT = str(Path(_shell.__file__).resolve().parents[2])

# How long a concurrency gate keeps trying before it says the frames never
# overlapped. Rounds repeat because workers are spawned ON DEMAND: the first
# round may well be answered by fewer workers than frames, which is the
# design, not a failure.
_DEADLINE = 45.0


class _Log(io.StringIO):
    """A log several threads write to, readable while they do."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            return super().write(text)

    def text(self) -> str:
        with self._lock:
            return self.getvalue()


def _worker_env() -> dict[str, str]:
    """The worker sees this checkout's momwire AND the stub seams."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [SRC_ROOT, str(HERE), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    return env


def _stub_seam():
    sys.path.insert(0, str(HERE))
    try:
        import _pool_seams
    finally:
        sys.path.remove(str(HERE))
    return _pool_seams.stub()


def _crashed(body, terminator, exc):
    return f"CRASHED {type(exc).__name__}: {exc}", ""


@pytest.fixture
def make_pool():
    pools = []

    def make(**kwargs):
        log = _Log()
        spec = kwargs.pop("spec", {"factory": "_pool_seams:stub"})
        kwargs.setdefault("max_workers", 3)
        kwargs.setdefault("idle_timeout", 3600.0)
        kwargs.setdefault("env", _worker_env())
        pool = WorkerPool(spec, log, **kwargs)
        pools.append(pool)
        pool.start()
        return pool, log

    yield make
    for pool in pools:
        pool.close()


def _ask_pool(pool, body: str):
    return pool.answer(_stub_seam(), body, "", _crashed, "[test] ")


def _round(n: int, fire) -> list:
    """``fire(i)`` from ``n`` threads released together; their results."""
    barrier = threading.Barrier(n)
    results: list = [None] * n
    errors: list = []

    def run(i):
        try:
            barrier.wait(timeout=30)
            results[i] = fire(i)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    if errors:
        raise errors[0]
    return results


def _overlap(intervals) -> bool:
    """True when ALL intervals share one instant: max(start) < min(end)."""
    return max(t0 for t0, _ in intervals) < min(t1 for _, t1 in intervals)


# ------------------------------------------------------------ pure units --


def test_a_message_crosses_the_pipe_as_the_same_str():
    """Latin-1 printout bytes, CRLF, and a lone surrogate all round-trip."""
    payload = {"out": "Ω é \r\n \udcff end", "err": "", "threads": None}
    stream = io.BytesIO()
    write_message(stream, payload)
    write_message(stream, {"n": 2})
    stream.seek(0)
    assert read_message(stream) == payload
    assert read_message(stream) == {"n": 2}
    assert read_message(stream) is None


def test_a_truncated_message_is_an_error_not_an_end():
    stream = io.BytesIO()
    write_message(stream, {"out": "x" * 100})
    stream = io.BytesIO(stream.getvalue()[:-10])
    with pytest.raises(EOFError):
        read_message(stream)


def test_a_lone_worker_is_not_limited_and_a_crew_divides_the_cores():
    assert thread_budget(1, 8) is None
    assert thread_budget(2, 8) == 4
    assert thread_budget(3, 8) == 2
    assert thread_budget(8, 8) == 1
    assert thread_budget(16, 8) == 1


def test_the_cap_is_cores_bounded_by_a_quarter_of_memory():
    gib = 1 << 30
    assert default_max_workers(cores=8, memory=16 * gib) == 8
    # 2 GiB / 4 / 150 MB = 3
    assert default_max_workers(cores=8, memory=2 * gib) == 3
    assert default_max_workers(cores=8, memory=100 << 20) == 1
    assert default_max_workers(cores=4, memory=None) <= 4


def test_the_environment_sets_or_disables_the_pool():
    assert configured_max_workers({_pool.WORKERS_ENV: "0"})[0] == 0
    assert configured_max_workers({_pool.WORKERS_ENV: "5"})[0] == 5
    auto, why = configured_max_workers({_pool.WORKERS_ENV: "many"})
    assert auto == default_max_workers()
    assert "not a count" in why
    assert configured_max_workers({})[0] == default_max_workers()


def test_a_source_install_runs_the_worker_module():
    assert _pool.worker_command() == [sys.executable, "-m", "momwire.serve._worker"]


def test_a_frozen_engine_is_its_own_worker(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert _pool.worker_command() == [sys.executable, _pool.WORKER_FLAG]


# -------------------------------------------------------- the pool itself --


@pytest.mark.integration
def test_concurrent_frames_run_in_distinct_workers_at_once(make_pool):
    """THE gate: three frames at once, three worker pids, one shared instant.

    Repeated until it holds or the deadline passes, because the pool spawns on
    demand; a pool that serialises (one lock, one worker, a queue) never
    satisfies it however long it runs.
    """
    pool, log = make_pool(max_workers=3)
    deadline = time.monotonic() + _DEADLINE
    seen = []
    while time.monotonic() < deadline:
        replies = _round(3, lambda i: _ask_pool(pool, "SLEEP 0.4"))
        reports = [json.loads(out) for out, _err in replies]
        seen.append(reports)
        pids = {r["pid"] for r in reports}
        if len(pids) == 3 and _overlap([(r["t0"], r["t1"]) for r in reports]):
            assert pids <= set(pool.live())
            assert os.getpid() not in pids
            return
    pytest.fail(f"three frames never ran in three workers at once: {seen}")


@pytest.mark.integration
def test_the_cap_holds(make_pool):
    pool, _log = make_pool(max_workers=2)
    pids = set()
    for _ in range(3):
        replies = _round(4, lambda i: _ask_pool(pool, "SLEEP 0.3"))
        pids |= {json.loads(out)["pid"] for out, _err in replies}
    assert len(pids) <= 2
    assert pool.spawned <= 2


@pytest.mark.integration
def test_a_worker_that_dies_mid_frame_reaches_its_caller_and_only_its_caller(
    make_pool,
):
    pool, log = make_pool(max_workers=2)
    _ask_pool(pool, "SLEEP 0")
    out, err = _ask_pool(pool, "CRASH")
    lost = re.fullmatch(
        r"CRASHED WorkerLost: the worker process \(pid (\d+)\) exited with "
        r"code (\S+) while solving",
        out,
    )
    assert lost, out
    assert err == ""
    dead = int(lost.group(1))
    assert f"worker pid={dead} lost while solving" in log.text()
    assert dead not in pool.live()
    # The pool is still serving, from a worker that is not the dead one.
    assert json.loads(_ask_pool(pool, "SLEEP 0")[0])["pid"] != dead


@pytest.mark.integration
def test_a_seam_that_raises_raises_at_the_caller(make_pool):
    """The in-process server's behaviour: the exception reaches the
    connection's handler (which logs it and closes), not a printout."""
    pool, _log = make_pool(max_workers=1)
    with pytest.raises(WorkerRaised, match="RuntimeError: the stub seam was told"):
        _ask_pool(pool, "RAISE")
    assert _ask_pool(pool, "hello") == ("echo hello|", "")


@pytest.mark.integration
def test_a_stray_print_in_a_worker_does_not_reach_the_protocol(make_pool):
    pool, _log = make_pool(max_workers=1)
    assert _ask_pool(pool, "PRINT") == ("printed", "")
    assert _ask_pool(pool, "after") == ("echo after|", "")


@pytest.mark.integration
def test_a_worker_s_stderr_chunk_comes_back_and_its_stderr_reaches_the_log(
    make_pool,
):
    pool, log = make_pool(max_workers=1)
    _out, err = _ask_pool(pool, "SLEEP 0")
    assert err == "slept 0\n"
    deadline = time.monotonic() + 10
    while "this must not reach the protocol" not in log.text():
        _ask_pool(pool, "PRINT")
        assert time.monotonic() < deadline, log.text()
        time.sleep(0.05)
    assert "[worker " in log.text()


def _crew_of_two(pool):
    deadline = time.monotonic() + _DEADLINE
    while len(pool.live()) < 2:
        _round(2, lambda i: _ask_pool(pool, "SLEEP 0.3"))
        assert time.monotonic() < deadline


def _wide_env(width: str) -> dict[str, str]:
    """Worker env pinned to ``width`` threads, whatever the lane exported
    (the xdist lane pins one, and a limit never raises a count)."""
    env = _worker_env()
    env.update(OMP_NUM_THREADS=width, OPENBLAS_NUM_THREADS=width)
    return env


@pytest.mark.integration
def test_a_lone_worker_keeps_its_threads_and_a_crew_divides_them(make_pool):
    pool, log = make_pool(max_workers=2, cores=8, trace=True, env=_wide_env("8"))
    lone = json.loads(_ask_pool(pool, "THREADS")[0])
    assert re.search(r"worker pid=\d+ t0=\S+ t1=\S+ threads=None", log.text())
    # OpenBLAS caps itself at the box's cores, so the lone count is read, not
    # assumed (and macOS numpy's Accelerate is not a pool threadpoolctl
    # reports at all); OpenMP takes the environment's 8 as given.
    assert lone["openmp"] == [8] * len(lone["openmp"])
    _crew_of_two(pool)
    crew = json.loads(_ask_pool(pool, "THREADS")[0])
    assert log.text().rstrip().endswith("threads=4")
    assert crew["blas"] == [min(4, n) for n in lone["blas"]]
    assert crew["openmp"] == [4] * len(lone["openmp"])
    assert crew["openmp"], "no OpenMP runtime seen: the division went unobserved"


@pytest.mark.integration
def test_a_share_of_one_keeps_blas_on_its_threaded_path(make_pool):
    """OpenBLAS's one-thread path rounds differently from its threaded one,
    so BLAS is never limited below two; OpenMP takes the share of one."""
    pool, log = make_pool(max_workers=2, cores=2, trace=True, env=_wide_env("8"))
    lone = json.loads(_ask_pool(pool, "THREADS")[0])
    _crew_of_two(pool)
    crew = json.loads(_ask_pool(pool, "THREADS")[0])
    assert log.text().rstrip().endswith("threads=1")
    assert crew["blas"] == [min(2, n) for n in lone["blas"]]
    assert crew["openmp"] == [1] * len(lone["openmp"])
    assert crew["openmp"], "no OpenMP runtime seen: the division went unobserved"


@pytest.mark.integration
def test_a_share_never_raises_a_pinned_thread_count(make_pool):
    """A server started under a one-thread pin keeps one thread in a crew:
    thread count moves round-off, so raising it would move answers."""
    pool, log = make_pool(max_workers=2, cores=8, trace=True, env=_wide_env("1"))
    _crew_of_two(pool)
    crew = json.loads(_ask_pool(pool, "THREADS")[0])
    assert log.text().rstrip().endswith("threads=4")
    assert crew["blas"] or crew["openmp"], "no thread pool seen at all"
    assert all(n == 1 for n in crew["blas"])
    assert all(n == 1 for n in crew["openmp"])


@pytest.mark.integration
def test_idle_workers_are_reaped_on_the_idle_rule_keeping_one(make_pool):
    pool, log = make_pool(max_workers=2, idle_timeout=1.0)
    deadline = time.monotonic() + _DEADLINE
    while len(pool.live()) < 2:
        _round(2, lambda i: _ask_pool(pool, "SLEEP 0.3"))
        assert time.monotonic() < deadline
    time.sleep(1.5)
    _ask_pool(pool, "SLEEP 0")
    assert len(pool.live()) == 1
    assert "idle 1s; stopped" in log.text()


@pytest.mark.integration
def test_a_pool_that_cannot_start_a_worker_answers_in_process(make_pool):
    pool, log = make_pool(
        max_workers=2, command=[sys.executable, "-c", "raise SystemExit(3)"]
    )
    assert _ask_pool(pool, "hello") == ("echo hello|", "")
    assert "a worker failed to start" in log.text()
    assert pool.live() == []


# --------------------------------------------------------- the real daemon --


@pytest.fixture
def short_room():
    """A short private directory for anything that binds (see
    ``test_serve_resident.short_room`` for why not ``tmp_path``)."""
    room = Path(tempfile.mkdtemp(prefix="mw-"))
    try:
        yield room
    finally:
        shutil.rmtree(room, ignore_errors=True)


def _ask(path: str, deck_bytes: bytes) -> bytes:
    conn = mech.connect(path)
    assert conn is not None, f"nothing is listening at {path}"
    try:
        conn.sendall(deck_bytes)
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        conn.close()
    return b"".join(chunks)


def _oracle(deck: Path, tmp_path: Path) -> bytes:
    out = tmp_path / f"{deck.stem}.oracle.out"
    _shell.run(deck, out)
    return out.read_bytes()


@pytest.fixture
def daemon(short_room):
    """The real eznec daemon, three workers, one trace line per frame."""
    path = str(short_room / f"srv{mech.address_suffix()}")
    log_path = short_room / "srv.log"
    env = _worker_env()
    env[_pool.WORKERS_ENV] = "3"
    env[_pool.TRACE_ENV] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "momwire.eznec",
            "--serve",
            "--socket",
            path,
            "--log",
            str(log_path),
            "--idle-timeout",
            "600",
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while "listening pid=" not in (
        log_path.read_text(errors="replace") if log_path.exists() else ""
    ):
        assert proc.poll() is None, log_path.read_text(errors="replace")
        assert time.monotonic() < deadline, "the daemon never listened"
        time.sleep(0.05)
    try:
        yield path, log_path, proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def _first_differences(answers, expected) -> str:
    """The first few differing printout lines, for a red byte comparison."""
    rows = []
    for i, (got, want) in enumerate(zip(answers, expected)):
        a, b = got.split(b"\r\n"), want.split(b"\r\n")
        rows += [
            f"deck {i} line {n}: got {x!r} want {y!r}"
            for n, (x, y) in enumerate(zip(a, b))
            if x != y
        ][:4]
        if len(a) != len(b):
            rows.append(f"deck {i}: {len(a)} lines, want {len(b)}")
    return "\n".join(rows[:12])


_TRACE = re.compile(r"\[conn (\d+)\] worker pid=(\d+) t0=(\S+) t1=(\S+) threads=")


@pytest.mark.integration
def test_the_daemon_answers_a_crew_in_parallel_with_the_one_shot_bytes(
    daemon, tmp_path
):
    """Three launcher-shaped clients at once against the real daemon: every
    answer is the one-shot shell's file to the byte, and the daemon's own
    trace shows three worker pids, none of them the daemon, solving at once."""
    path, log_path, proc = daemon
    decks = sorted(DECKS.glob("*.nec"))[:3]
    expected = [_oracle(d, tmp_path) for d in decks]
    payloads = [d.read_bytes() for d in decks]
    deadline = time.monotonic() + _DEADLINE
    while True:
        answers = _round(3, lambda i: _ask(path, payloads[i]))
        assert answers == expected, _first_differences(answers, expected)
        rows = [
            (int(c), int(p), float(a), float(b))
            for c, p, a, b in _TRACE.findall(log_path.read_text(errors="replace"))
        ]
        last = rows[-3:]
        if (
            len(last) == 3
            and len({r[1] for r in last}) == 3
            and _overlap([(r[2], r[3]) for r in last])
        ):
            assert proc.pid not in {r[1] for r in last}
            return
        assert time.monotonic() < deadline, f"never three at once: {rows}"


@pytest.mark.integration
def test_a_crashed_worker_reaches_its_caller_as_the_seam_s_error_frame(
    short_room, tmp_path
):
    """A worker that dies solving a deck answers that caller with the very
    ``NEC ERROR`` printout an exception inside the solve produced in-process,
    and the server goes on answering the next caller from a fresh worker."""
    path = str(short_room / f"srv{mech.address_suffix()}")
    log = _Log()
    basis = _shell._serve.BASIS
    pool = WorkerPool(
        {"factory": "_pool_seams:crashing_eznec", "kwargs": {"basis": basis}},
        log,
        max_workers=2,
        idle_timeout=3600.0,
        env=_worker_env(),
    )
    pool.start()

    def connection(conn, number, log_stream, solve_lock):
        _resident._connection(
            conn, number, log_stream, solve_lock, basis=basis, pool=pool
        )

    server = Server(path, idle_timeout=3600.0, log=log, connection=connection)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        assert server.listening.wait(10)
        deck = sorted(DECKS.glob("*.nec"))[0]
        crash_body = b"CM CRASHME\r\n" + deck.read_bytes()
        answer = _ask(path, crash_body).decode("latin-1")
        reason = re.search(r"INTERNAL ERROR IN MOMWIRE ENGINE - WorkerLost: .*", answer)
        assert reason, answer
        # Exactly the seam's own failure frame, transcoded as the wire does.
        exc = WorkerLost(reason.group(0).split("WorkerLost: ", 1)[1].rstrip("\r"))
        # The body as the session loop hands it to the seam: universal
        # newlines, one line per card, no trailing newline.
        body = "\n".join(crash_body.decode("latin-1").splitlines())
        frame, _ = _shell.seam_failure(body, exc, basis=basis)
        assert answer == frame.replace("\n", "\r\n")
        # The front end is still up, and the next caller is answered properly.
        assert _ask(path, deck.read_bytes()) == _oracle(deck, tmp_path)
    finally:
        server.stop()
        thread.join(timeout=10)
        pool.close()
