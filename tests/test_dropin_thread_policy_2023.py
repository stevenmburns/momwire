"""The drop-in sizes its thread pools by the deck (momwire#2023).

The rule is the standalone executable's (`_thread_policy.h`). The answers
below are read from the pools themselves, in a fresh interpreter whose thread
variables are the case's: the suite's own one-thread pin (tests/conftest.py)
would otherwise decide every count here.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from momwire import _accel, _thread_policy
from momwire.dropin.nec5 import _serve

pytestmark = pytest.mark.skipif(
    _accel.acc is None, reason="no compiled accelerator loaded"
)

ROOT = Path(__file__).resolve().parents[1]
_VARS = (
    "MOMWIRE_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MOMWIRE_TIMES",
)
_HAS_MASK = hasattr(os, "sched_setaffinity")


def _deck(wires: int, segments: int) -> str:
    """Parallel half-wave wires in free space at 14 MHz."""
    cards = ["CM thread policy", "CE"]
    for i in range(wires):
        x = 3.0 * i
        cards.append(f"GW {i + 1},{segments},{x},-5.,0.,{x},5.,0.,1E-3")
    centre = segments // 2 + 1
    cards += ["GE 0", "FR 0,1,0,0,14.", f"EX 0,1,{centre},0,1.,0.", "XQ", "EN"]
    return "\n".join(cards) + "\n"


SMALL = _deck(1, 21)  # 21 segments: one thread
MIDDLE = _deck(10, 20)  # 200 segments: four threads, or every core below that

# Runs one answer through the real seam; reports the pools as the fill saw
# them and as the answer left them.
_PROBE = """
import json, sys
from threadpoolctl import threadpool_info
from momwire import _thread_policy, _times
from momwire.dropin.nec5 import _serve, _shell

def pools():
    out = {}
    for info in threadpool_info():
        out.setdefault(info["user_api"], set()).add(info["num_threads"])
    return {api: sorted(counts) for api, counts in out.items()}

seen = {"entry": pools()}
real = _serve._port_solution
def spy(solver, shared):
    seen["fill"] = pools()
    return real(solver, shared)
_serve._port_solution = spy
request = json.load(sys.stdin)
share = request.get("share")
with _times.answering("test") as times, _thread_policy.frame_share(share):
    printout = _shell.seam(basis="razor2p").answer(request["deck"], "")[0]
seen["left"] = pools()
seen["line"] = times.line()
seen["printout"] = printout
json.dump(seen, sys.stdout)
"""


def _answer(deck: str, env=None, *, share=None, cpus=None) -> dict:
    clean = {k: v for k, v in os.environ.items() if k not in _VARS}
    clean.update(env or {})
    preexec = (lambda: os.sched_setaffinity(0, cpus)) if cpus is not None else None
    done = subprocess.run(
        [sys.executable, "-c", _PROBE],
        input=json.dumps({"deck": deck, "share": share}),
        env=clean,
        preexec_fn=preexec,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _cores() -> int:
    return int(_accel.acc._pool_threads()["allowed_cores"])


def _cpus() -> int:
    return int(_accel.acc._pool_threads()["allowed_threads"])


def _blas(seen: dict, stage: str = "fill") -> list[int] | None:
    """The BLAS counts, or None where no OpenBLAS is loaded to ask (macOS)."""
    return seen[stage].get("blas")


# ------------------------------------------------------------ the rule ---


@pytest.mark.parametrize(
    ("size", "want"),
    [
        ((1, 0, 0), lambda c: 1),
        ((127, 0, 0), lambda c: 1),
        ((128, 0, 0), lambda c: min(4, c)),
        ((383, 0, 0), lambda c: min(4, c)),
        ((384, 0, 0), lambda c: c),
        ((10, 63, 0), lambda c: 1),
        ((10, 64, 0), lambda c: c),
        ((10, 0, 39999), lambda c: 1),
        ((10, 0, 40000), lambda c: c),
    ],
)
def test_the_rule_is_the_executables(size, want):
    assert _thread_policy.deck_threads(*size) == want(_cores())


def test_the_executable_and_the_drop_in_share_one_rule():
    exe = (ROOT / "standalone" / "momwire_razor2p.cpp").read_text(encoding="utf-8")
    assert '#include "_thread_policy.h"' in exe
    assert "mw_threads::for_deck(physical_cores()," in exe
    # No second copy of the table is left behind in the executable.
    assert not re.search(r"\b(CROSSING_PAIRS|BELOW_PAIRS)\s*=", exe)
    accel = (ROOT / "src/momwire/_accelerators.cpp").read_text(encoding="utf-8")
    assert "mw_threads::for_deck(mw::cpu::allowed_cores()," in accel


# ------------------------------------------------------ the precedence ---


@pytest.mark.parametrize(
    ("rule", "env", "want"),
    [
        # Nothing set: the deck decides, and one OpenMP thread keeps BLAS on
        # its threaded path.
        (1, {}, {"openmp": 1, "blas": 2}),
        (4, {}, {"openmp": 4, "blas": 4}),
        # No deck measured yet: nothing to do.
        (None, {}, {}),
        # MOMWIRE_THREADS fixes both, whatever the deck, and 1 means 1.
        (4, {"MOMWIRE_THREADS": "3"}, {"openmp": 3, "blas": 3}),
        (1, {"MOMWIRE_THREADS": "1"}, {"openmp": 1, "blas": 1}),
        (None, {"MOMWIRE_THREADS": "3"}, {"openmp": 3, "blas": 3}),
        # A pool's own variable beats it, for that pool alone.
        (4, {"MOMWIRE_THREADS": "3", "OMP_NUM_THREADS": "2"}, {"blas": 3}),
        (4, {"MOMWIRE_THREADS": "3", "OPENBLAS_NUM_THREADS": "2"}, {"openmp": 3}),
        # A pinned pool is left alone and the other still follows the deck.
        (4, {"OPENBLAS_NUM_THREADS": "1"}, {"openmp": 4}),
        # OpenBLAS reads OMP_NUM_THREADS itself when it loads: both are pinned.
        (4, {"OMP_NUM_THREADS": "2"}, {}),
        (4, {"GOTO_NUM_THREADS": "2"}, {"openmp": 4}),
        # What is not a count is not set.
        (4, {"MOMWIRE_THREADS": "junk"}, {"openmp": 4, "blas": 4}),
        (4, {"MOMWIRE_THREADS": "0"}, {"openmp": 4, "blas": 4}),
        (4, {"OMP_NUM_THREADS": ""}, {"openmp": 4, "blas": 4}),
    ],
)
def test_the_precedence(rule, env, want):
    assert _thread_policy.targets(rule, environ=env, allowed=8) == want


@pytest.mark.parametrize(
    ("rule", "env", "share", "want"),
    [
        (4, {}, 2, {"openmp": 2, "blas": 2}),
        # The share may leave OpenMP on one thread, never BLAS.
        (4, {}, 1, {"openmp": 1, "blas": 2}),
        (1, {}, 3, {"openmp": 1, "blas": 2}),
        # The share caps MOMWIRE_THREADS as well.
        (4, {"MOMWIRE_THREADS": "8"}, 2, {"openmp": 2, "blas": 2}),
    ],
)
def test_a_crews_share_caps_every_count(rule, env, share, want):
    assert _thread_policy.targets(rule, environ=env, share=share, allowed=8) == want


def test_one_cpu_has_no_second_blas_thread():
    assert _thread_policy.targets(1, environ={}, allowed=1) == {"openmp": 1, "blas": 1}


@pytest.mark.parametrize(
    ("raw", "want"),
    [("", 0), ("junk", 0), ("-2", 0), ("0", 0), ("3", 3), (" 3 ", 3), ("4,2", 4)],
)
def test_a_thread_variable_is_read_as_a_count(raw, want):
    assert _thread_policy.env_count("X", {"X": raw}) == want


# -------------------------------------------------------- the measures ---


def _piece(elements: int, z: float):
    points = np.array([[0.0, 0.0, z], [1.0, 0.0, z]])
    return SimpleNamespace(n_elements=elements, points=points)


def test_a_solves_size_counts_elements_not_polyline_points():
    mesh = SimpleNamespace(pieces=[_piece(61, 1.0), _piece(20, 2.0)])
    assert _serve._solve_size(mesh, None) == (81, 0, 0)


def test_a_crossing_deck_reports_its_block():
    mesh = SimpleNamespace(pieces=[_piece(30, 1.0), _piece(10, -1.0)])
    assert _serve._solve_size(mesh, object()) == (40, 300, 0)


def test_a_wholly_buried_deck_reports_its_pairs():
    mesh = SimpleNamespace(pieces=[_piece(30, -1.0), _piece(10, -2.0)])
    assert _serve._solve_size(mesh, object()) == (40, 0, 1600)


def test_no_ground_means_nothing_is_below():
    mesh = SimpleNamespace(pieces=[_piece(30, -1.0)])
    assert _serve._solve_size(mesh, None) == (30, 0, 0)


# ------------------------------------------------- the pools themselves ---


def test_a_small_deck_runs_on_one_openmp_thread_and_threaded_blas():
    seen = _answer(SMALL)
    assert seen["fill"]["openmp"] == [1]
    if _blas(seen):
        assert _blas(seen) == [min(2, _cpus())]


def test_a_middling_deck_runs_on_the_rules_count():
    seen = _answer(MIDDLE)
    assert seen["fill"]["openmp"] == [min(4, _cores())]
    if _blas(seen):
        assert _blas(seen) == [min(4, _cores())]


def test_the_answer_gives_the_pools_back():
    seen = _answer(MIDDLE)
    assert seen["left"] == seen["entry"]


@pytest.mark.parametrize("deck", [SMALL, MIDDLE], ids=["small", "middling"])
def test_momwire_threads_fixes_the_count_for_every_deck(deck):
    seen = _answer(deck, {"MOMWIRE_THREADS": "3"})
    assert seen["fill"]["openmp"] == [3]
    if _blas(seen):
        assert _blas(seen) == [3]


def test_momwire_threads_one_is_serial_blas():
    seen = _answer(MIDDLE, {"MOMWIRE_THREADS": "1"})
    assert seen["fill"]["openmp"] == [1]
    if _blas(seen):
        assert _blas(seen) == [1]


def test_a_pools_own_variable_beats_momwire_threads():
    seen = _answer(
        MIDDLE,
        {"MOMWIRE_THREADS": "3", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"},
    )
    assert seen["fill"]["openmp"] == [2]
    if _blas(seen):
        assert _blas(seen) == [1]


def test_an_explicit_pin_is_left_where_it_was():
    seen = _answer(SMALL, {"OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3"})
    assert seen["fill"]["openmp"] == [3]
    if _blas(seen):
        assert _blas(seen) == [3]


def test_a_crews_share_holds_a_frame_to_it():
    seen = _answer(MIDDLE, share=1)
    assert seen["fill"]["openmp"] == [1]
    if _blas(seen):
        assert _blas(seen) == [min(2, _cpus())]


@pytest.mark.skipif(not _HAS_MASK, reason="no affinity masks on this platform")
def test_the_affinity_mask_bounds_a_large_deck():
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs two CPUs")
    # Asked of a deck that wants every core: the mask's cores are all it gets.
    seen = _answer(_deck(10, 40), cpus=set(allowed[:1]))
    assert seen["fill"]["openmp"] == [1]


# ----------------------------------------------------- what is reported ---


def test_the_times_line_reports_the_counts_used():
    seen = _answer(SMALL)
    assert " threads=1 " in seen["line"]
    seen = _answer(MIDDLE, {"MOMWIRE_THREADS": "3"})
    assert " threads=3 " in seen["line"]


def test_the_printout_reports_the_counts_used():
    seen = _answer(MIDDLE, {"MOMWIRE_TIMES": "1", "MOMWIRE_THREADS": "3"})
    assert "THREADS = 3 (FILL), " in seen["printout"]
    if _blas(seen):
        assert "THREADS = 3 (FILL), 3 (FACTOR)" in seen["printout"]


# --------------------------------------------- every OpenBLAS in the process ---

# A copy of SciPy's OpenBLAS is loaded after the pools were set, in the middle
# of the answer, as a library first imported by the solve would be.
_LATE_PROBE = """
import ctypes, json, os, shutil, sys, tempfile, types
from threadpoolctl import ThreadpoolController, threadpool_info
from momwire.dropin.nec5 import _serve, _shell

def counts():
    return {os.path.basename(i["filepath"]): i["num_threads"]
            for i in threadpool_info() if i["user_api"] == "blas"}

seen = {}
real = _serve._port_solution
def spy(solver, shared):
    source = next(i["filepath"] for i in threadpool_info()
                  if i["user_api"] == "blas" and "openblas64" not in i["filepath"])
    copy = os.path.join(tempfile.mkdtemp(), "libscipy_openblas-latecopy.so")
    shutil.copy(source, copy)
    seen["handle"] = ctypes.CDLL(copy) and 1
    sys.modules["_late_blas_probe"] = types.ModuleType("_late_blas_probe")
    return real(solver, shared)
_serve._port_solution = spy
real_finish = _serve._finish_run
def finish(deck, prepared):
    seen["after_solve"] = counts()
    return real_finish(deck, prepared)
_serve._finish_run = finish
_shell.render(sys.stdin.read(), basis=sys.argv[1])
seen["left"] = counts()
json.dump(seen, sys.stdout)
"""


def _late(deck: str, basis: str, cpus=None) -> dict:
    clean = {k: v for k, v in os.environ.items() if k not in _VARS}
    preexec = (lambda: os.sched_setaffinity(0, cpus)) if cpus is not None else None
    done = subprocess.run(
        [sys.executable, "-c", _LATE_PROBE, basis],
        input=deck,
        env=clean,
        preexec_fn=preexec,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.skipif(sys.platform != "linux", reason="copies a Linux shared library")
@pytest.mark.parametrize("basis", ["razor2p", "bs2"])
def test_a_library_loaded_during_the_answer_is_held_to_the_count(basis):
    # MIDDLE's rule is min(4, cores): below OpenBLAS's default width on a
    # machine of more than four cores, so the late copy has to be held to it.
    # A deck of 384 segments or more gets every core, which a late library
    # already has, and the check would pass whether or not it was held.
    seen = _late(MIDDLE, basis)
    after = seen["after_solve"]
    assert len(after) >= 3, after  # numpy's, SciPy's and the late copy
    assert set(after.values()) == {min(4, _cores())}, after


@pytest.mark.skipif(sys.platform != "linux", reason="copies a Linux shared library")
def test_a_late_library_stays_inside_the_affinity_mask():
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs two CPUs")
    mask = set(allowed[:2])
    seen = _late(_deck(10, 40), "razor2p", cpus=mask)
    assert max(seen["after_solve"].values()) <= len(mask), seen["after_solve"]


# ----------------------------------------------- a pool that starts at one ---

_mech = pytest.importorskip("momwire_serve_client")


@pytest.mark.parametrize(
    ("env", "pinned"),
    [
        ({}, True),
        ({"MOMWIRE_THREADS": "3"}, True),
        ({"OPENBLAS_NUM_THREADS": "4"}, False),
        ({"GOTO_NUM_THREADS": "4"}, False),
        ({"OMP_NUM_THREADS": "4"}, False),
    ],
)
def test_the_pool_starts_small_unless_the_user_gave_a_count(env, pinned):
    environ = {_mech.BLAS_POOL_TEST_ENV: "1", **env}
    assert _mech.pin_blas_pool(environ) is pinned
    assert (environ.get("OPENBLAS_NUM_THREADS") == "1") is pinned or not pinned
    assert (_mech.BLAS_POOL_ENV in environ) is pinned


def test_starting_the_pool_twice_is_the_same_start():
    environ = {_mech.BLAS_POOL_TEST_ENV: "1"}
    assert _mech.pin_blas_pool(environ)
    assert _mech.pin_blas_pool(environ)
    assert environ["OPENBLAS_NUM_THREADS"] == "1"


@pytest.mark.skipif(sys.platform == "win32", reason="the start is the default there")
def test_off_windows_the_pool_starts_as_it_always_did():
    environ: dict[str, str] = {}
    assert not _mech.pin_blas_pool(environ)
    assert environ == {}


def test_a_small_start_leaves_the_serial_path_before_any_deck():
    plan = _thread_policy.targets(None, environ={}, allowed=8, started_small=True)
    assert plan == {"blas": 2}


_SMALL_PROBE = """
import json, os, sys
import momwire_serve_client as mech
assert mech.pin_blas_pool()
from threadpoolctl import threadpool_info
from momwire import _thread_policy
from momwire.dropin.nec5 import _serve, _shell

def blas():
    return sorted({i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"})

seen = {"loaded": blas()}
real = _serve._port_solution
def spy(solver, shared):
    seen["fill"] = blas()
    return real(solver, shared)
_serve._port_solution = spy
_shell.render(sys.stdin.read(), basis="razor2p")
seen["left"] = blas()
seen["threads"] = len(os.listdir("/proc/self/task")) if os.path.isdir("/proc/self/task") else None
seen["env"] = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", mech.BLAS_POOL_ENV)}
with _thread_policy.answering():
    _thread_policy.unsized()
    seen["unsized"] = blas()
json.dump(seen, sys.stdout)
"""


def _small(deck: str, env=None) -> dict:
    clean = {k: v for k, v in os.environ.items() if k not in _VARS}
    clean.update({_mech.BLAS_POOL_TEST_ENV: "1"}, **(env or {}))
    done = subprocess.run(
        [sys.executable, "-c", _SMALL_PROBE],
        input=deck,
        env=clean,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize(
    ("deck", "want"),
    [(SMALL, lambda c: min(2, _cpus())), (MIDDLE, lambda c: min(4, c))],
    ids=["small", "middling"],
)
def test_a_pool_started_small_grows_to_the_decks_count(deck, want):
    seen = _small(deck)
    if not seen["loaded"]:
        pytest.skip("no OpenBLAS loaded to ask")
    assert seen["loaded"] == [1]
    assert seen["fill"] == [want(_cores())]
    assert seen["left"] == [1]
    # The variables were the start's, not the user's.
    assert seen["env"] == {"OPENBLAS_NUM_THREADS": None, _mech.BLAS_POOL_ENV: None}


@pytest.mark.skipif(sys.platform != "linux", reason="counts /proc/self/task")
def test_a_pool_started_small_never_starts_more_workers_than_the_deck_needs():
    seen = _small(SMALL)
    rule = min(2, _cpus())
    # The main thread, one OpenMP team of one, and each OpenBLAS's workers.
    assert seen["threads"] <= 1 + 2 * (rule - 1) + 1, seen


def test_momwire_threads_grows_a_pool_started_small():
    seen = _small(MIDDLE, {"MOMWIRE_THREADS": "3"})
    if seen["loaded"]:
        assert seen["fill"] == [3]


def test_an_unmeasured_answer_gets_the_physical_cores():
    seen = _small(SMALL)
    if seen["loaded"]:
        assert seen["unsized"] == [_cores()]
