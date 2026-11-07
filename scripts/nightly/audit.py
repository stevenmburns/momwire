"""One cell of the Skylake nightly's exe == Python audit.

    python audit.py --tree MW --exes EXES_DIR --cell avx2-T1 --out OUT.jsonl \\
        [--only ONLY.txt] [--local-corpus FILE] [--save-printouts DIR] [--workers W]

Every deck of the corpus (corpus.txt beside this file, plus an optional local
list) is answered by each of the four no-Python executables that serve its
dialect and by the Python route those executables replay, in this process's
workers (`_shell.render_files`, written as `_shell.printout_bytes` writes it):

    nec5-bspline      momwire-nec5-bspline-standalone      basis bspline,    NEC-5
    nec5-razor-2p     momwire-nec5-razor-2p-standalone     basis razor-2p,   NEC-5
    nec42-bspline     momwire-nec42-bspline-standalone     basis bspline,    NEC-4.2
    nec42-sinusoidal  momwire-nec42-sinusoidal-standalone  basis sinusoidal, NEC-4.2

(the CMake target names, long form since momwire#1831).

A cell is a variant and a thread count, `<avx2|sse2>-T<n>`: the executables
built for that variant, Python's accelerator forced to it
(MOMWIRE_FORCE_VARIANT), and MOMWIRE_THREADS = OMP_NUM_THREADS =
OPENBLAS_NUM_THREADS = n on both sides.

The two printouts must be byte-identical once the NEC-4.2 slot's two measured
time lines are masked. A deck the executable DECLINES (its route log says
`refused:<why>`) is recorded, not failed, but Python is still asked: an
executable that declines what Python serves is a finding (triage.py).

One JSON line per (executable, deck), flushed as it lands. A local-corpus
deck is labelled `local#<n>` (its line in the local list) and its path is kept
only in `local_path`, which nothing posted ever reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> (CMake MW_EXE_BASIS, the target it builds, the Python basis, dialect)
EXES = {
    "nec5-bspline": (
        "bspline_nec5",
        "momwire-nec5-bspline-standalone",
        "bspline",
        "nec5",
    ),
    "nec5-razor-2p": (
        "razor2p",
        "momwire-nec5-razor-2p-standalone",
        "razor-2p",
        "nec5",
    ),
    "nec42-bspline": (
        "bspline_nec42",
        "momwire-nec42-bspline-standalone",
        "bspline",
        "nec42",
    ),
    "nec42-sinusoidal": (
        "sin42",
        "momwire-nec42-sinusoidal-standalone",
        "sinusoidal",
        "nec42",
    ),
}
DECK_SUFFIXES = (".nec", ".deck")
TIMEOUT_S = 900
ERR = b"NEC ERROR - "

# The NEC-4.2 slot's two measured times (run_audit_nec42.py's mask): their
# numbers only, by exact line pattern, and the count of masked lines kept.
FILL = re.compile(rb"^( {24}FILL=)[ 0-9.]{9}( SEC\.,  FACTOR=)[ 0-9.]{9}( SEC\.)$")
RUN_TIME = re.compile(rb"^( RUN TIME =)[ 0-9.]{10}$")


def parse_cell(cell: str) -> tuple[str, int]:
    m = re.fullmatch(r"(avx2|sse2)-T(\d+)", cell)
    if not m:
        raise SystemExit(f"cell {cell!r} is not <avx2|sse2>-T<n>")
    return m.group(1), int(m.group(2))


def mask(printout: bytes) -> tuple[bytes, int]:
    out, n = [], 0
    for line in printout.split(b"\r\n"):
        m = FILL.match(line)
        if m:
            out.append(m.group(1) + b"<t>" + m.group(2) + b"<t>" + m.group(3))
            n += 1
            continue
        m = RUN_TIME.match(line)
        if m:
            out.append(m.group(1) + b"<t>")
            n += 1
            continue
        out.append(line)
    return b"\r\n".join(out), n


def outcome(printout: bytes) -> str:
    if not printout:
        return "empty"
    return "refused" if ERR in printout else "served"


def first_difference(py: bytes, exe: bytes) -> int:
    a, b = py.split(b"\r\n"), exe.split(b"\r\n")
    for k, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return k + 1
    return min(len(a), len(b)) + 1


def read_corpus(tree: Path, local: Path | None) -> list[dict]:
    """[{label, path, dialects, group}] for the committed manifest (in-repo
    decks, labelled by their repo path) and the optional local list."""
    decks = []
    for ln in (HERE / "corpus.txt").read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        dialects, rel = ln.split()
        d = tree / rel
        if not d.is_dir():
            raise SystemExit(f"corpus.txt: {rel} is not a directory in {tree}")
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in DECK_SUFFIXES:
                decks.append(
                    dict(
                        label=f"{rel}/{p.name}",
                        path=str(p),
                        dialects=dialects.split(","),
                        group=rel,
                    )
                )
    if local and local.is_file():
        n = 0
        for ln in local.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            dialects, path = ln.split("\t", 1)
            n += 1
            decks.append(
                dict(
                    label=f"local#{n}",
                    path=path,
                    dialects=dialects.split(","),
                    group="local",
                    local=True,
                )
            )
    return decks


# ------------------------------------------------------------------ worker
_VARIANT = None
_SAVE = None


def _init(save):
    global _VARIANT, _SAVE
    import momwire._accel as accel

    _VARIANT = accel.VARIANT
    _SAVE = save


class _Alarm(Exception):
    pass


def _alarm(*_):
    raise _Alarm


def _python(text: str, basis: str, dialect: str) -> tuple[bytes, str | None]:
    from momwire.dropin.nec5 import _shell

    kw = dict(basis=basis) if dialect == "nec5" else dict(basis=basis, dialect="nec4")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT_S)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = _shell.render_files(text, **kw)[0]
            except _Alarm:
                raise
            except Exception as exc:  # noqa: BLE001 - the launcher's own last line of defence
                out = _shell.seam_failure(text, exc, **kw)[0]
        return _shell.printout_bytes(out), None
    except _Alarm:
        return b"", "timeout"
    except Exception as exc:  # noqa: BLE001 - recorded; the audit goes on
        return b"", f"{type(exc).__name__}: {exc}"[:300]
    finally:
        signal.alarm(0)


def one(job: dict) -> dict:
    exe_name, deck, exe = job["exe"], job["deck"], job["exe_path"]
    _, _, basis, dialect = EXES[exe_name]
    rec = {"exe": exe_name, "deck": deck["label"], "group": deck["group"]}
    if deck.get("local"):
        rec["local_path"] = deck["path"]
    rec["py_variant"] = _VARIANT
    with tempfile.TemporaryDirectory() as t:
        route, out = Path(t) / "route", Path(t) / "exe.out"
        env = dict(os.environ, MW_EXE_ROUTE_LOG=str(route))
        t0 = time.perf_counter()
        try:
            p = subprocess.run(
                [exe, deck["path"], str(out)],
                env=env,
                capture_output=True,
                timeout=TIMEOUT_S,
            )
            rec["exe_rc"] = p.returncode
            rec["route"] = (
                route.read_text().split(" ", 1)[0].strip() if route.exists() else "none"
            )
            exe_bytes = out.read_bytes() if out.exists() else b""
        except subprocess.TimeoutExpired:
            rec["route"], exe_bytes = "timeout", b""
        rec["exe_s"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    py_bytes, err = _python(
        Path(deck["path"]).read_bytes().decode("latin-1"), basis, dialect
    )
    rec["py_s"] = round(time.perf_counter() - t0, 3)
    if err:
        rec["py_error"] = err
    exe_m, rec["exe_masked"] = mask(exe_bytes)
    py_m, rec["py_masked"] = mask(py_bytes)
    rec["exe_outcome"], rec["py_outcome"] = outcome(exe_bytes), outcome(py_bytes)
    rec["exe_sha"] = hashlib.sha256(exe_m).hexdigest()
    rec["py_sha"] = hashlib.sha256(py_m).hexdigest()
    rec["identical"] = exe_bytes == py_bytes
    rec["identical_masked"] = (
        exe_m == py_m and rec["exe_masked"] == rec["py_masked"] and bool(py_bytes)
    )
    if not rec["identical_masked"] and exe_bytes and py_bytes:
        rec["diff_line"] = first_difference(py_m, exe_m)
    if _SAVE:
        key = hashlib.sha1(f"{exe_name}|{deck['label']}".encode()).hexdigest()[:16]
        rec["key"] = key
        Path(_SAVE, f"{key}.py").write_bytes(py_m)
        if not rec["identical_masked"] and exe_bytes:
            Path(_SAVE, f"{key}.exe").write_bytes(exe_m)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True, type=Path, help="the momwire checkout")
    ap.add_argument("--exes", required=True, type=Path, help="exe build root")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", type=Path, help='lines "<exe>\\t<deck label>"')
    ap.add_argument("--exe-names", default=",".join(EXES))
    ap.add_argument("--local-corpus", type=Path)
    ap.add_argument("--save-printouts", type=Path)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    variant, threads = parse_cell(a.cell)

    # The environment every child reads (numpy is never imported here).
    sys.path.insert(0, str(a.tree / "src"))
    import momwire_serve_client

    momwire_serve_client.pin_cpu_env()
    n = str(threads)
    os.environ.update(
        MOMWIRE_THREADS=n,
        OMP_NUM_THREADS=n,
        OPENBLAS_NUM_THREADS=n,
        MOMWIRE_BELOW_ANCHORS="off",
        MOMWIRE_FORCE_VARIANT=variant,
        PYTHONWARNINGS="ignore",
    )

    only = None
    if a.only:
        only = {
            tuple(ln.split("\t", 1))
            for ln in a.only.read_text().splitlines()
            if ln.strip()
        }
    decks = read_corpus(a.tree.resolve(), a.local_corpus)
    jobs = []
    for name in a.exe_names.split(","):
        cmake_basis, target, _, dialect = EXES[name]
        exe = a.exes / f"{cmake_basis}-{variant}" / target
        if not exe.is_file():
            raise SystemExit(f"no executable {exe}")
        for d in decks:
            if dialect in d["dialects"] and (
                only is None or (name, d["label"]) in only
            ):
                jobs.append(dict(exe=name, deck=d, exe_path=str(exe)))
    workers = a.workers or max(1, (os.cpu_count() or 8) // threads)
    if a.save_printouts:
        a.save_printouts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"{a.cell}: {len(jobs)} jobs, {workers} workers", flush=True)
    with (
        Pool(
            workers,
            initializer=_init,
            initargs=(a.save_printouts,),
            maxtasksperchild=40,
        ) as pool,
        a.out.open("w") as f,
    ):
        for k, rec in enumerate(pool.imap_unordered(one, jobs), 1):
            rec["cell"] = a.cell
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"{a.cell}: done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
