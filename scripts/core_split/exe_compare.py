"""momwire-razor2p (standalone/, unit 4) against `momwire-nec5-razor-2p`, deck
by deck, whole printout file against whole printout file.

    OPENBLAS_NUM_THREADS=T OMP_NUM_THREADS=T \\
    python scripts/core_split/exe_compare.py OUT.jsonl --exe EXE [--launcher L] \\
        [--workers W] DECK_OR_DIR...

Every deck is answered four ways:

* ``exe``      -- the executable as shipped (MOMWIRE_THREADS=T): the C++ route
                  for a deck it covers, the embedded Python for the rest;
* ``fallback`` -- the same executable with every deck forced through the
                  embedded Python (MW_EXE_FORCE_FALLBACK=1), the executable's
                  OpenBLAS published as `sys._mw_host_lapack`;
* ``launcher`` -- the real `momwire-nec5-razor-2p` (its own resident server,
                  started fresh in a private runtime directory with this
                  process's environment, so at T threads; the server divides
                  the cores among its live workers, so an auto-mode gate runs
                  with --workers 1, a lone caller, as EZNEC is);
* ``python``   -- `printout_compare.python_printout` in this process: what the
                  launcher writes, on numpy's and scipy's own libraries.

Every printout must equal every other, byte for byte. For a deck the C++ route
answered, each finished run's arrays (the solve's Y, `_finish`'s coeffs,
`_port_state`'s four vectors) must also equal the fallback's, byte for byte:
that is the C++ route against Python on the SAME library.

Each deck is counted by the route that answered it in the ``exe`` pass
(`cxx:serial`, `cxx:threads`, or `fallback:<why>`); a gate in which every deck
took the fallback measures nothing, so both counts are printed. One JSON line
per deck; a summary on stdout; exit status 1 on any mismatch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SPLIT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPLIT))

import printout_compare as pc  # noqa: E402

ARRAYS = pc.ARRAYS
HOOK = SPLIT / "exe_arrays_hook.py"


def _arg(args, flag, default=None):
    if flag in args:
        k = args.index(flag)
        value = args[k + 1]
        del args[k : k + 2]
        return value
    return default


def _read_runs(directory: Path) -> list[dict]:
    runs = []
    k = 0
    while (directory / f"{k}_y.bin").exists():
        runs.append({a: (directory / f"{k}_{a}.bin").read_bytes() for a in ARRAYS})
        k += 1
    return runs


def _answer(exe, launcher, deck: Path, work: Path, threads: str, launcher_env) -> dict:
    """The three subprocess answers for one deck."""
    work.mkdir(parents=True)
    out = {}
    env = dict(os.environ)
    if threads != "auto":
        env["MOMWIRE_THREADS"] = threads
    (work / "cxx").mkdir()
    subprocess.run(
        [exe, str(deck), str(work / "exe.out")],
        env=dict(
            env, MW_EXE_ROUTE_LOG=str(work / "route"), MW_EXE_ARRAYS=str(work / "cxx")
        ),
        capture_output=True,
        check=False,
    )
    out["route"] = (work / "route").read_text().split(" ", 1)[0]
    out["exe"] = (work / "exe.out").read_bytes()
    out["cxx_runs"] = _read_runs(work / "cxx")
    (work / "py").mkdir()
    subprocess.run(
        [exe, str(deck), str(work / "fallback.out")],
        env=dict(
            env,
            MW_EXE_FORCE_FALLBACK="1",
            MW_EXE_PYSTARTUP=str(HOOK),
            MW_EXE_PYARRAYS=str(work / "py"),
        ),
        capture_output=True,
        check=False,
    )
    out["fallback"] = (work / "fallback.out").read_bytes()
    out["py_runs"] = _read_runs(work / "py")
    calls = work / "py" / "host_calls.json"
    out["host_calls"] = json.loads(calls.read_text()) if calls.exists() else {}
    if launcher is not None:
        subprocess.run(
            [launcher, str(deck), str(work / "launcher.out")],
            env=launcher_env,
            capture_output=True,
            check=False,
        )
        out["launcher"] = (work / "launcher.out").read_bytes()
    return out


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    out_path = Path(args.pop(0))
    exe = _arg(args, "--exe")
    launcher = _arg(args, "--launcher")
    workers = int(_arg(args, "--workers", "4"))
    if exe is None:
        raise SystemExit("--exe EXE is required")
    # One count for every side, or neither set: "auto", the executable's own
    # size policy against the launcher's default (every hardware thread).
    threads = os.environ.get("OPENBLAS_NUM_THREADS")
    if os.environ.get("OMP_NUM_THREADS") != threads:
        raise SystemExit(
            "set OPENBLAS_NUM_THREADS and OMP_NUM_THREADS to one count, or neither"
        )
    if threads is None:
        threads = "auto"
    decks = pc._decks(args)
    work = Path(tempfile.mkdtemp(prefix="exe_compare_"))
    runtime = work / "runtime"  # the launcher's private resident server
    runtime.mkdir()
    launcher_env = dict(os.environ, MOMWIRE_PORTAL_RUNTIME_DIR=str(runtime))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_answer, exe, launcher, d, work / str(i), threads, launcher_env)
            for i, d in enumerate(decks)
        ]
        counts, routes, mismatches, runs_compared = Counter(), Counter(), 0, 0
        with open(out_path, "w") as out:
            for deck, fut in zip(decks, futures, strict=True):
                python = pc.python_printout(deck.read_bytes().decode("latin-1"))
                a = fut.result()
                rec = {"deck": str(deck), "route": a["route"]}
                routes[a["route"]] += 1
                ref = a.get("launcher", python)
                problems = []
                if python != ref:
                    problems.append(
                        "python != launcher: " + pc.first_difference(ref, python)
                    )
                for name in ("exe", "fallback"):
                    if a[name] != ref:
                        problems.append(f"{name}: " + pc.first_difference(ref, a[name]))
                # In auto mode the two routes run at different counts by design
                # (the C++ route by size, the fallback at the launcher's
                # default), and the LU's last bits follow the count: arrays are
                # compared only at a fixed count; printouts always.
                if a["route"].startswith("cxx") and threads != "auto":
                    if len(a["cxx_runs"]) != len(a["py_runs"]):
                        problems.append(
                            f"runs {len(a['py_runs'])} (python) / {len(a['cxx_runs'])} (c++)"
                        )
                    for k, (pr, cr) in enumerate(zip(a["py_runs"], a["cxx_runs"])):
                        runs_compared += 1
                        problems += [
                            f"run {k}: {n} differs" for n in ARRAYS if pr[n] != cr[n]
                        ]
                    rec["runs"] = len(a["cxx_runs"])
                if a["py_runs"]:
                    hc = a["host_calls"]
                    rec["host_calls"] = hc
                    if not (hc.get("getrf", 0) > 0 and hc.get("gemm", 0) > 0):
                        problems.append(
                            f"the fallback's runs were not on the host: {hc}"
                        )
                rec["same"] = not problems
                if problems:
                    rec["problems"] = problems
                    mismatches += 1
                    counts[a["route"] + "|MISMATCH"] += 1
                else:
                    counts[a["route"]] += 1
                out.write(json.dumps(rec) + "\n")
                out.flush()
    shutil.rmtree(work, ignore_errors=True)
    cxx = sum(v for k, v in routes.items() if k.startswith("cxx"))
    print(
        f"decks {len(decks)} at {threads} thread(s): C++ route {cxx}, fallback {len(decks) - cxx}"
    )
    print("by route: " + ", ".join(f"{k} {v}" for k, v in routes.most_common()))
    print(
        f"runs whose arrays were compared, C++ vs fallback ({', '.join(ARRAYS)}): {runs_compared}"
    )
    print(f"compared against: {'launcher + python' if launcher else 'python'}")
    print(f"mismatches {mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
