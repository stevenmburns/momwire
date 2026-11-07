"""The Skylake nightly: momwire main's exe == Python audit and perf sweep.

    python nightly.py run [--force] [--ref origin/main] [--dry-run] [--date D]
                          [--cells C,...] [--no-sweep] [--no-bisect]
    python nightly.py bisect NIGHT_DIR [--dry-run]   # re-bisect a red night
    python nightly.py report NIGHT_DIR --dry-run     # print the issue body
    python nightly.py install                        # the systemd user units

Runs ON Skylake from ~/antennas/nightly/harness (a copy of scripts/nightly
taken from the ref under test, so a bisect that checks out an older commit
never pulls the harness out from under itself). README.md beside this file
is the full description: what runs, the cell rotation, severity, re-running a
night, bisecting by hand.

Nothing here reverts, opens a pull request or edits code. It writes results
under ~/antennas/nightly/results/<date>/ and edits ONE pinned issue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()
ROOT = Path(os.environ.get("MW_NIGHTLY_ROOT", HOME / "antennas" / "nightly"))
LOCK = HOME / "antennas" / ".nightly.lock"
MW = ROOT / "mw"
VENV = ROOT / "venv"
PY = VENV / "bin" / "python"
EXES = ROOT / "exes"
OPENBLAS = ROOT / "openblas"
AK = ROOT / "ak"
RESULTS = ROOT / "results"
HARNESS = ROOT / "harness"
STATE = ROOT / "state.json"
LOCAL_CORPUS = ROOT / "local_corpus.txt"  # optional, never committed
OB_SEED = HOME / "antennas" / "ob-cache" / "src"  # an OpenBLAS clone to copy from

REPO = "stevenmburns/momwire"
ISSUE_TITLE = "Skylake nightly status"
# The antennaknobs commit whose scratch/pretag-sweep and src/ the sweep runs.
# Fixed on purpose (the sweep compares momwire trees); moving it re-baselines
# the sweep, which triage.py notes rather than flags.
AK_REPO = "https://github.com/stevenmburns/antennaknobs"
AK_SHA = "d451584f329c928030014a975cb0fe79524f9805"

EVERY_NIGHT = "avx2-T1"
# {avx2, sse2} x {T1, T2, T4} minus avx2-T1: one a night, by date, so five
# consecutive nights cover the whole space.
ROTATION = ("avx2-T2", "avx2-T4", "sse2-T1", "sse2-T2", "sse2-T4")
BUILD_JOBS = "6"
KEEP_NIGHTS = 14
RESULTS_CAP_BYTES = 1 << 30
BISECT_BUDGET_S = 100 * 60

sys.path.insert(0, str(HERE))
import audit  # noqa: E402
import triage  # noqa: E402


def rotation_cell(day: dt.date) -> str:
    return ROTATION[day.toordinal() % len(ROTATION)]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def env_base() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{HOME / '.local' / 'bin'}:{VENV / 'bin'}:{env.get('PATH', '')}"
    return env


def sh(cmd, *, cwd=None, logf: Path | None = None, env=None, check=True, timeout=None):
    """Run a command; its output to `logf` (appended) or captured."""
    env = env or env_base()
    if logf:
        with logf.open("a") as f:
            f.write(f"\n$ {' '.join(map(str, cmd))}\n")
            f.flush()
            p = subprocess.run(
                list(map(str, cmd)),
                cwd=cwd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
    else:
        p = subprocess.run(
            list(map(str, cmd)),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    if check and p.returncode:
        tail = "" if logf else (p.stderr or p.stdout)[-400:]
        raise RuntimeError(
            f"{cmd[0]} {cmd[1] if len(cmd) > 1 else ''} rc={p.returncode} {tail}"
        )
    return p


def git(*args) -> str:
    return sh(["git", "-C", MW, *args]).stdout.strip()


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def save_state(s: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(STATE)


def write_meta(night: Path, meta: dict) -> None:
    (night / "night.json").write_text(json.dumps(meta, indent=1))


# ------------------------------------------------------------------ harness
def refresh_harness(ref: str) -> bool:
    """Copy scripts/nightly (and printout_tolerance.py) from `ref` into
    HARNESS. True when the copy changed, i.e. the caller should re-exec."""
    have = sh(["git", "-C", MW, "ls-tree", "-d", ref, "scripts/nightly"]).stdout.strip()
    if not have:
        return False  # the ref predates the nightly: keep the installed harness
    new = ROOT / "harness.new"
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir()
    for path in ("scripts/nightly", "scripts/core_split/printout_tolerance.py"):
        arc = subprocess.run(
            ["git", "-C", str(MW), "archive", ref, path],
            capture_output=True,
            check=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(new)], input=arc, check=True)
    flat = ROOT / "harness.flat"
    shutil.rmtree(flat, ignore_errors=True)
    shutil.copytree(new / "scripts" / "nightly", flat)
    shutil.copy2(new / "scripts/core_split/printout_tolerance.py", flat)
    shutil.rmtree(new)

    def digest(d: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(d.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(p.relative_to(d).as_posix().encode() + p.read_bytes())
        return h.hexdigest()

    if HARNESS.is_dir() and digest(HARNESS) == digest(flat):
        shutil.rmtree(flat)
        return False
    shutil.rmtree(HARNESS, ignore_errors=True)
    flat.rename(HARNESS)
    return True


# -------------------------------------------------------------------- build
def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def build_python(logf: Path, state: dict) -> None:
    env = env_base()
    env.update(CC="ccache gcc", CXX="ccache g++")
    ph = file_hash(MW / "pyproject.toml")
    if state.get("pyproject") != ph or not PY.exists():
        if not PY.exists():
            sh([sys.executable, "-m", "venv", VENV], logf=logf)
            sh([PY, "-m", "pip", "install", "-q", "--upgrade", "pip"], logf=logf)
            sh(
                [
                    PY,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "setuptools>=61.0",
                    "wheel",
                    "pybind11",
                ],
                logf=logf,
            )
        sh(
            [PY, "-m", "pip", "install", "-q", "-e", ".[test]", "--no-build-isolation"],
            cwd=MW,
            logf=logf,
            env=env,
        )
        sh(
            [PY, "-m", "pip", "install", "-q", "matplotlib"], logf=logf
        )  # the sweep's antennaknobs
        state["pyproject"] = ph
    # make build: MOMWIRE_REQUIRE_ACCEL=1, so a failed compile fails here
    # rather than leaving a stale .so behind (never bare build_ext, never -j).
    sh(["make", "build", f"PYTHON={PY}"], cwd=MW, logf=logf)
    # build/ (setuptools' objects, ~1.4 GB) is kept: with it an unchanged
    # tree rebuilds in seconds, without it every night is a cold compile.
    for v in ("avx2", "sse2"):
        e = env_base()
        e["MOMWIRE_FORCE_VARIANT"] = v
        got = sh(
            [PY, "-c", "import momwire._accel as a; print(a.VARIANT)"], env=e, cwd=ROOT
        ).stdout.strip()
        if got != v:
            raise RuntimeError(f"forced {v}, momwire._accel.VARIANT is {got}")


def openblas(logf: Path) -> Path:
    h = file_hash(MW / "standalone" / "build_openblas.sh")
    d = OPENBLAS / h
    if (d / "libmwxp.a").is_file() and (d / "libmwnp.a").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "src").exists() and OB_SEED.is_dir():
        sh(["git", "clone", "-q", "--shared", OB_SEED, d / "src"], logf=logf)
    sh(["bash", MW / "standalone" / "build_openblas.sh", d], logf=logf)
    shutil.rmtree(d / "src", ignore_errors=True)
    for old in OPENBLAS.iterdir():
        if old.is_dir() and old.name != h:
            shutil.rmtree(old)  # the nightly's own earlier archives only
    return d


def build_exes(pairs, ob: Path, logf: Path) -> dict:
    """Build (exe name, variant) pairs; their -version lines."""
    versions = {}
    for name, v in pairs:
        cmake_basis, target, _, _ = audit.EXES[name]
        b = EXES / f"{cmake_basis}-{v}"
        sh(
            [
                "cmake", "-S", MW / "standalone", "-B", b,
                "-DMW_EXE_NO_PYTHON=ON", f"-DMW_EXE_BASIS={cmake_basis}",
                f"-DMW_EXE_VARIANT={v}", f"-DMW_OPENBLAS_LIB={ob / 'libmwxp.a'}",
                f"-DMW_OPENBLAS_NUMPY_LIB={ob / 'libmwnp.a'}", "-DMW_EXE_STATIC_GOMP=ON",
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ],
            logf=logf,
        )  # fmt: skip
        sh(["cmake", "--build", b, "-j", BUILD_JOBS, "--target", target], logf=logf)
        versions[f"{name}-{v}"] = sh(
            [b / target, "-version"], check=False
        ).stdout.strip()
    return versions


def ensure_ak(logf: Path) -> Path:
    if not (AK / ".git").is_dir():
        sh(
            ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", AK_REPO, AK],
            logf=logf,
        )
        sh(
            [
                "git",
                "-C",
                AK,
                "sparse-checkout",
                "set",
                "src/antennaknobs",
                "scratch/pretag-sweep",
            ],
            logf=logf,
        )
    head = sh(["git", "-C", AK, "rev-parse", "HEAD"], check=False).stdout.strip()
    if head != AK_SHA or not (AK / "src" / "antennaknobs").is_dir():
        sh(["git", "-C", AK, "fetch", "-q", "origin", AK_SHA], logf=logf)
        sh(["git", "-C", AK, "checkout", "-q", "--detach", AK_SHA], logf=logf)
    (AK / "src" / ".ak_sha").write_text(AK_SHA + "\n")
    return AK / "scratch" / "pretag-sweep"


# --------------------------------------------------------------------- runs
def run_cell(
    cell: str, night: Path, *, only: Path | None = None, exe_names=None
) -> None:
    out = night / f"audit-{cell}.jsonl"
    cmd = [
        PY,
        HERE / "audit.py",
        "--tree",
        MW,
        "--exes",
        EXES,
        "--cell",
        cell,
        "--out",
        out,
    ]
    if cell == EVERY_NIGHT:
        cmd += ["--save-printouts", night / "py-avx2-T1"]
    if LOCAL_CORPUS.is_file():
        cmd += ["--local-corpus", LOCAL_CORPUS]
    if only:
        cmd += ["--only", only]
    if exe_names:
        cmd += ["--exe-names", ",".join(exe_names)]
    sh(cmd, logf=night / f"audit-{cell}.log", cwd=ROOT)


def run_sweep(night: Path, harness: Path, order_by: Path | None, case_ids=None) -> None:
    """driver.py fast at the pinned antennaknobs. With `case_ids`, a copy of
    the harness restricted to those cases (each keeps its tier: parallel or
    timed), for a bisect."""
    night, harness = night.resolve(), harness.resolve()
    sha = git("rev-parse", "HEAD")
    (MW / ".sweep_sha").write_text(f"{sha} {git('describe', '--always', '--tags')}\n")
    if case_ids is not None:
        sub = night / "sweep-harness"
        shutil.rmtree(sub, ignore_errors=True)
        shutil.copytree(
            harness, sub, ignore=shutil.ignore_patterns("results", "__pycache__")
        )
        doc = json.loads((harness / "cases.json").read_text())
        doc["cases"] = [c for c in doc["cases"] if c["id"] in case_ids]
        (sub / "cases.json").write_text(json.dumps(doc))
        timed = [t for t in driver_timed(harness) if t in case_ids]
        (sub / "timed.txt").write_text("".join(t + "\n" for t in timed))
        harness = sub
    cmd = [
        "python3", harness / "driver.py", "fast", "--python", PY, "--tree", MW,
        "--ak-src", AK / "src", "--cases", harness / "cases.json",
        "--out", night / "sweep.jsonl", "--arm", sha[:12], "--jobs", "8",
    ]  # fmt: skip
    if order_by and order_by.is_file():
        cmd += ["--order-by", order_by]
    sh(cmd, logf=night / "sweep.log", cwd=harness)


def driver_timed(harness: Path) -> list[str]:
    lines = (harness / "timed.txt").read_text().splitlines()
    return [ln.split("#", 1)[0].strip() for ln in lines if ln.split("#", 1)[0].strip()]


def env_facts() -> dict:
    def out(*cmd):
        return sh(list(cmd), check=False).stdout.strip()

    return dict(
        host=os.uname().nodename,
        gcc=out("gcc", "--version").splitlines()[0] if out("gcc", "--version") else None,
        python=out(str(PY), "-c", "import sys,numpy,scipy;print(sys.version.split()[0],numpy.__version__,scipy.__version__)"),
    )  # fmt: skip


# ------------------------------------------------------------------ the run
def night_dir(day: dt.date) -> Path:
    base = RESULTS / day.isoformat()
    d, k = base, 1
    while d.exists():
        k += 1
        d = base.with_name(f"{day.isoformat()}-{k}")
    d.mkdir(parents=True)
    return d


def do_night(a) -> int:
    state = load_state()
    sh(["git", "-C", MW, "fetch", "-q", "origin"])
    if a.refresh_harness and refresh_harness(a.ref):
        log("harness changed at the ref: re-exec")
        os.execv(
            sys.executable,
            [
                sys.executable,
                str(HARNESS / "nightly.py"),
                *sys.argv[1:],
                "--no-refresh-harness",
            ],
        )
    sha = git("rev-parse", a.ref)
    green = state.get("last_green") or {}
    if not a.force and sha == green.get("sha"):
        log(
            f"main has not moved since the last green night ({sha[:12]}, {green.get('date')}); skipping"
        )
        state["last_skip"] = dict(sha=sha, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        save_state(state)
        return 0
    day = (
        dt.date.fromisoformat(a.date)
        if a.date
        else dt.datetime.now().astimezone().date()
    )
    cells = a.cells.split(",") if a.cells else [EVERY_NIGHT, rotation_cell(day)]
    night = night_dir(day)
    meta = dict(
        date=day.isoformat(),
        sha=sha,
        subject=None,
        cells=cells,
        rotation=f"{ROTATION.index(cells[1]) + 1} of {len(ROTATION)}" if len(cells) > 1 and cells[1] in ROTATION else None,
        ak_sha=AK_SHA,
        harness=file_hash(HERE / "nightly.py"),
        local_corpus=sum(1 for ln in LOCAL_CORPUS.read_text().splitlines() if ln.strip() and not ln.startswith("#")) if LOCAL_CORPUS.is_file() else 0,
        phases={},
        errors={},
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )  # fmt: skip
    log(f"night {night.name}: {sha[:12]}, cells {cells}")
    t_all = time.time()

    def phase(name, fn):
        t0 = time.time()
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 - a harness finding; the night goes on
            meta["errors"][name] = f"{type(exc).__name__}: {exc}"[:400]
            log(f"{name} FAILED: {exc}")
            return False
        finally:
            meta["phases"][name] = round(time.time() - t0, 1)
            write_meta(night, meta)

    def checkout():
        git("checkout", "-q", "--detach", sha)
        git("reset", "-q", "--hard", sha)
        meta["subject"] = git("log", "-1", "--format=%s", sha)

    ok = phase("checkout", checkout)
    ok = ok and phase("build-python", lambda: build_python(night / "build.log", state))
    obd = {}
    ok = ok and phase(
        "openblas", lambda: obd.setdefault("d", openblas(night / "build.log"))
    )
    variants = sorted({c.split("-")[0] for c in cells})
    if ok:
        pairs = [(n, v) for v in variants for n in audit.EXES]
        phase(
            "build-exes",
            lambda: meta.update(
                exe_versions=build_exes(pairs, obd["d"], night / "build.log")
            ),
        )
    meta["env"] = env_facts()
    if ok and "build-exes" not in meta["errors"]:
        for cell in cells:
            phase(f"audit-{cell}", lambda cell=cell: run_cell(cell, night))
    if ok and not a.no_sweep:
        prev = Path(green["dir"]) / "sweep.jsonl" if green.get("dir") else None
        phase("sweep", lambda: run_sweep(night, ensure_ak(night / "build.log"), prev))
    meta["wall_s"] = round(time.time() - t_all, 1)
    write_meta(night, meta)

    base = (
        Path(green["dir"]) if green.get("dir") and Path(green["dir"]).is_dir() else None
    )
    result = triage.triage(night, base, HERE / "known.json")
    if a.fake_bad:
        result = force_red(night, result, a.fake_bad)
    log(f"triage: {result['status']} {result['counts']}")
    if result["status"] == "red" and base and not a.no_bisect:
        t0 = time.time()
        try:
            bis = bisect(night, base, result, good_sha=a.bisect_good)
        except Exception as exc:  # noqa: BLE001 - the report still goes out
            bis = dict(error=f"{type(exc).__name__}: {exc}"[:400])
        meta["phases"]["bisect"] = round(time.time() - t0, 1)
        (night / "bisect.json").write_text(json.dumps(bis, indent=1))
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["wall_s"] = round(time.time() - t_all, 1)
    write_meta(night, meta)
    pack(night)

    prev_status = state.get("last_status")
    report(night, state, dry_run=a.dry_run, prev_status=prev_status)
    if not a.dry_run:
        state["last_status"] = result["status"]
        state["last_night"] = dict(
            sha=sha, date=day.isoformat(), dir=str(night), status=result["status"]
        )
        if result["status"] == "green":
            state["last_green"] = dict(sha=sha, date=day.isoformat(), dir=str(night))
        save_state(state)
    prune(state)
    log(f"done in {(time.time() - t_all) / 60:.1f} min: {result['status']}")
    return 0


def pack(night: Path) -> None:
    d = night / "py-avx2-T1"
    if d.is_dir():
        with tarfile.open(night / "py-avx2-T1.tar.xz", "w:xz") as tf:
            tf.add(d, arcname="py-avx2-T1")
        shutil.rmtree(d)


def prune(state: dict) -> None:
    if not RESULTS.is_dir():
        return
    keep = {
        (state.get("last_green") or {}).get("dir"),
        (state.get("last_night") or {}).get("dir"),
    }
    nights = sorted((p for p in RESULTS.iterdir() if p.is_dir()), key=lambda p: p.name)

    def size(p):
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    while nights:
        total = sum(size(p) for p in nights)
        if len(nights) <= KEEP_NIGHTS and total <= RESULTS_CAP_BYTES:
            break
        victim = next((p for p in nights if str(p) not in keep), None)
        if victim is None:
            break
        shutil.rmtree(victim)
        nights.remove(victim)


# ------------------------------------------------------------------- bisect
def _targets(result: dict) -> list[dict]:
    return [
        f
        for f in result["findings"]
        if f["severity"] in triage.RED and f["kind"] not in ("sweep-time", "harness")
    ]


def force_red(night: Path, result: dict, fake_bad: str) -> dict:
    """The forced-red rehearsal (--fake-bad, with --dry-run): one synthetic
    severe finding, which a bisect probe calls bad at `fake_bad` and every
    commit after it, so the triage, bisect and report paths all run on a
    tree that is in fact green."""
    global FAKE_BAD
    FAKE_BAD = git("rev-parse", fake_bad)
    f = triage.finding(
        "fake",
        "severe",
        f"forced red: commits from {FAKE_BAD[:12]} on are treated as bad",
        case="rehearsal",
    )
    f["recommend"] = triage.RECOMMEND["severe"]
    result["findings"].insert(0, f)
    result["counts"]["severe"] = result["counts"].get("severe", 0) + 1
    result["status"] = "red"
    (night / "triage.json").write_text(json.dumps(result, indent=1))
    return result


FAKE_BAD = None


def probe(sha: str, targets: list[dict], base: Path, work: Path) -> str:
    """'bad' when any target reproduces at `sha` (against the green night),
    'good' when none does, 'skip' when `sha` does not build."""
    d = work / sha[:12]
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    git("checkout", "-q", "--detach", sha)
    git("reset", "-q", "--hard", sha)
    logf = d / "build.log"
    try:
        build_python(logf, load_state())
        audit_t = [f for f in targets if "exe" in f]
        pairs = sorted({(f["exe"], f["cell"].split("-")[0]) for f in audit_t})
        if pairs:
            build_exes(pairs, openblas(logf), logf)
    except Exception:  # noqa: BLE001 - an unbuildable commit is skipped
        return "skip"
    if any(f["kind"] == "fake" for f in targets):
        bad = sh(
            ["git", "-C", MW, "merge-base", "--is-ancestor", FAKE_BAD, sha], check=False
        )
        return "bad" if bad.returncode == 0 else "good"
    write_meta(d, dict(sha=sha, ak_sha=AK_SHA, errors={}))
    by_cell = {}
    for f in audit_t:
        by_cell.setdefault(f["cell"], set()).add((f["exe"], f["deck"]))
    for cell, pairs2 in by_cell.items():
        only = d / f"only-{cell}.txt"
        only.write_text("".join(f"{e}\t{k}\n" for e, k in sorted(pairs2)))
        run_cell(cell, d, only=only, exe_names=sorted({e for e, _ in pairs2}))
    cases = {f["case"] for f in targets if f["kind"].startswith("sweep-")}
    if cases:
        run_sweep(d, ensure_ak(logf), None, case_ids=cases)
    got = {
        f["id"]
        for f in triage.triage(d, base, HERE / "known.json")["findings"]
        if f["severity"] in triage.RED
    }
    pack(d)
    return "bad" if got & {f["id"] for f in targets} else "good"


def bisect(night: Path, base: Path, result: dict, good_sha: str | None = None) -> dict:
    meta = json.loads((night / "night.json").read_text())
    good_sha = git(
        "rev-parse", good_sha or json.loads((base / "night.json").read_text())["sha"]
    )
    bad_sha = meta["sha"]
    targets = _targets(result)
    if not targets:
        return dict(note="nothing to bisect (only timing or harness findings)")
    # A handful of each kind is enough to name the commit; the rest ride along.
    picked, per_kind = [], {}
    for f in targets:
        if per_kind.get(f["kind"], 0) < 3:
            picked.append(f)
            per_kind[f["kind"]] = per_kind.get(f["kind"], 0) + 1
    commits = git(
        "rev-list", "--first-parent", "--reverse", f"{good_sha}..{bad_sha}"
    ).split()
    work = night / "bisect"
    t0 = time.time()
    steps = []
    out = dict(
        good=good_sha,
        bad=bad_sha,
        commits=len(commits),
        probed=[f["id"] for f in picked],
        steps=steps,
    )
    v = probe(good_sha, picked, base, work)
    steps.append([good_sha, v])
    log(f"bisect: last green {good_sha[:12]} -> {v}")
    if v != "good":
        out["result"] = (
            "reproduces at the last green commit: not a regression in this range"
        )
        return out
    lo, hi = -1, len(commits) - 1  # commits[hi] is bad (tonight)
    skipped = set()
    while hi - lo > 1:
        if time.time() - t0 > BISECT_BUDGET_S:
            out["result"] = "budget reached"
            break
        cand = [k for k in range(lo + 1, hi) if k not in skipped]
        if not cand:
            break
        mid = cand[len(cand) // 2]
        v = probe(commits[mid], picked, base, work)
        steps.append([commits[mid], v])
        log(f"bisect: {commits[mid][:12]} -> {v}")
        if v == "skip":
            skipped.add(mid)
        elif v == "bad":
            hi = mid
        else:
            lo = mid
    first = commits[hi]
    out["first_bad"] = first
    out["first_bad_subject"] = git("log", "-1", "--format=%s", first)
    out["range"] = [commits[lo] if lo >= 0 else good_sha, first]
    out["exact"] = all(k in skipped for k in range(lo + 1, hi))
    out.setdefault(
        "result",
        "first bad commit"
        if out["exact"]
        else "narrowed to a range (unbuildable commits inside)",
    )
    out["wall_s"] = round(time.time() - t0, 1)
    git("checkout", "-q", "--detach", bad_sha)
    return out


# ------------------------------------------------------------------ report
def _cell_table(night: Path) -> list[str]:
    rows = [
        "| cell | executable | decks | C++ route | declined | exe = Python |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell, recs in triage.audit_rows(night).items():
        by = {}
        for (exe, _), r in recs.items():
            s = by.setdefault(exe, [0, 0, 0, 0])
            s[0] += 1
            route = r.get("route", "")
            s[1] += route.startswith("cxx")
            s[2] += route.startswith("refused:")
            s[3] += bool(route.startswith("cxx") and r.get("identical_masked"))
        for exe, (n, cxx, dec, same) in sorted(by.items()):
            rows.append(f"| {cell} | {exe} | {n} | {cxx} | {dec} | {same} of {cxx} |")
    return rows


def _where(f: dict) -> str:
    if f.get("case"):
        return f"`{f['case']}`"
    return f"{f.get('cell')} {f.get('exe')} `{f.get('deck')}`"


def render_report(night: Path, prev_status: str | None) -> tuple[str, str | None]:
    meta = json.loads((night / "night.json").read_text())
    tr = json.loads((night / "triage.json").read_text())
    bis = (
        json.loads((night / "bisect.json").read_text())
        if (night / "bisect.json").is_file()
        else None
    )
    status = tr["status"].upper()
    sha = meta["sha"]
    lines = [
        f"## {status}: night {meta['date']}, main `{sha[:12]}`",
        "",
        f"`{sha[:12]}` {meta.get('subject') or ''}",
        "",
        f"- compared with: {'green night ' + str(tr.get('base_date')) + ', `' + tr['base'][:12] + '`' if tr.get('base') else 'no earlier green night (first run or baseline reset)'}",
        f"- cells: {', '.join(meta['cells'])}" + (f" (rotation {meta['rotation']})" if meta.get("rotation") else ""),
        f"- wall: {meta.get('wall_s', 0) / 60:.0f} min ({', '.join(f'{k} {v / 60:.1f}' for k, v in meta.get('phases', {}).items())} min)",
        f"- findings: {', '.join(f'{k} {v}' for k, v in sorted(tr['counts'].items())) or 'none'}",
        f"- sweep: antennaknobs `{meta['ak_sha'][:12]}`, {sum(1 for _ in open(night / 'sweep.jsonl')) if (night / 'sweep.jsonl').is_file() else 0} cases",
        f"- box: {meta.get('env', {}).get('host')}, {meta.get('env', {}).get('gcc')}, Python/numpy/SciPy {meta.get('env', {}).get('python')}",
    ]  # fmt: skip
    if meta.get("local_corpus"):
        lines.append(
            f"- local corpus: {meta['local_corpus']} decks (reported by index only)"
        )
    lines += ["", *_cell_table(night), ""]
    shown = [f for f in tr["findings"] if f["severity"] != "known"]
    order = {"severe": 0, "moderate": 1, "minor": 2}
    shown.sort(key=lambda f: (order.get(f["severity"], 3), f["id"]))
    if shown:
        lines += [
            "| severity | kind | where | detail | proposed action |",
            "|---|---|---|---|---|",
        ]
        for f in shown[:40]:
            detail = f["detail"].replace("|", "/")
            lines.append(
                f"| {f['severity']} | {f['kind']} | {_where(f)} | {detail} | {f['recommend']} |"
            )
        if len(shown) > 40:
            lines.append(
                f"\n{len(shown) - 40} more in the night's triage.json on Skylake."
            )
    known = sum(1 for f in tr["findings"] if f["severity"] == "known")
    if known:
        lines.append(
            f"\n{known} finding(s) matched the known list (scripts/nightly/known.json)."
        )
    if bis:
        lines.append("")
        if bis.get("first_bad"):
            lines.append(
                f"**Bisect** ({bis.get('commits')} commits since the last green night): {bis.get('result')}: `{bis['first_bad'][:12]}` {bis.get('first_bad_subject')}"
                + (
                    ""
                    if bis.get("exact")
                    else f" (range `{bis['range'][0][:12]}..{bis['range'][1][:12]}`)"
                )
            )
        else:
            lines.append(
                f"**Bisect:** {bis.get('result') or bis.get('note') or bis.get('error')}"
            )
    lines += [
        "",
        "Severity is proposed by the script; nothing reverts on its own. "
        "Edited nightly by `scripts/nightly/nightly.py` on Skylake (README there); "
        "a comment marks each red night and each return to green.",
    ]
    body = "\n".join(lines)
    comment = None
    if tr["status"] == "red":
        comment = "\n".join(
            lines[:1]
            + [""]
            + [
                ln
                for ln in lines
                if ln.startswith("| severe") or ln.startswith("| moderate")
            ][:20]
            + ([""] + [ln for ln in lines if ln.startswith("**Bisect")])
        )
    elif prev_status == "red":
        comment = f"Green again: night {meta['date']}, main `{sha[:12]}` {meta.get('subject') or ''}"
    return body, comment


def find_issue(state: dict) -> int | None:
    if state.get("issue"):
        return state["issue"]
    out = sh(
        [
            "gh",
            "issue",
            "list",
            "-R",
            REPO,
            "--state",
            "open",
            "--search",
            f'in:title "{ISSUE_TITLE}"',
            "--json",
            "number,title",
        ]
    ).stdout
    for it in json.loads(out or "[]"):
        if it["title"] == ISSUE_TITLE:
            return it["number"]
    return None


def report(night: Path, state: dict, *, dry_run: bool, prev_status: str | None) -> None:
    body, comment = render_report(night, prev_status)
    tr = json.loads((night / "triage.json").read_text())
    # A red night on a commit already reported red, with the same findings,
    # gets no second comment.
    sig = hashlib.sha256(
        json.dumps(
            sorted(f["id"] for f in tr["findings"] if f["severity"] in triage.RED)
        ).encode()
    ).hexdigest()
    last = state.get("last_red_comment") or {}
    meta = json.loads((night / "night.json").read_text())
    if (
        comment
        and tr["status"] == "red"
        and last.get("sha") == meta["sha"]
        and last.get("sig") == sig
    ):
        comment = None
    (night / "issue-body.md").write_text(body)
    if comment:
        (night / "issue-comment.md").write_text(comment)
    if dry_run:
        print("===== would-be issue body =====\n" + body)
        print("===== would-be comment =====\n" + (comment or "(none)"))
        return
    num = find_issue(state)
    if num is None:
        url = sh(
            [
                "gh",
                "issue",
                "create",
                "-R",
                REPO,
                "--title",
                ISSUE_TITLE,
                "--body-file",
                night / "issue-body.md",
            ]
        ).stdout.strip()
        num = int(url.rstrip("/").rsplit("/", 1)[1])
        sh(["gh", "issue", "pin", str(num), "-R", REPO], check=False)
    else:
        sh(
            [
                "gh",
                "issue",
                "edit",
                str(num),
                "-R",
                REPO,
                "--body-file",
                night / "issue-body.md",
            ]
        )
    state["issue"] = num
    if comment:
        sh(
            [
                "gh",
                "issue",
                "comment",
                str(num),
                "-R",
                REPO,
                "--body-file",
                night / "issue-comment.md",
            ]
        )
        if tr["status"] == "red":
            state["last_red_comment"] = dict(sha=meta["sha"], sig=sig)


# ----------------------------------------------------------------- install
UNIT = """[Unit]
Description=momwire Skylake nightly (scripts/nightly in stevenmburns/momwire)

[Service]
Type=oneshot
Nice=5
MemoryMax=24G
MemorySwapMax=0
TimeoutStartSec=6h
Environment=PATH={home}/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart={py} {harness}/nightly.py run
StandardOutput=append:{root}/nightly.log
StandardError=append:{root}/nightly.log
"""
TIMER = """[Unit]
Description=momwire Skylake nightly, 01:30 local

[Timer]
OnCalendar=*-*-* 01:30:00
Persistent=false

[Install]
WantedBy=timers.target
"""


def install(_a) -> int:
    d = HOME / ".config" / "systemd" / "user"
    d.mkdir(parents=True, exist_ok=True)
    (d / "momwire-skylake-nightly.service").write_text(
        UNIT.format(
            home=HOME,
            py=sys.executable if not PY.exists() else PY,
            harness=HARNESS,
            root=ROOT,
        )
    )
    (d / "momwire-skylake-nightly.timer").write_text(TIMER)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "momwire-skylake-nightly.timer"],
        check=True,
    )
    print(
        subprocess.run(
            ["loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ref", default="origin/main")
    r.add_argument(
        "--force", action="store_true", help="run even if main has not moved"
    )
    r.add_argument(
        "--dry-run",
        action="store_true",
        help="print the issue body; post nothing, keep the state",
    )
    r.add_argument("--date", help="the night's date (picks the rotation cell)")
    r.add_argument("--cells", help="comma list, instead of avx2-T1 + the rotation cell")
    r.add_argument("--no-sweep", action="store_true")
    r.add_argument("--no-bisect", action="store_true")
    r.add_argument(
        "--fake-bad", help="rehearsal: force red, a bisect calls this commit on bad"
    )
    r.add_argument(
        "--bisect-good", help="bisect from this commit, not the last green night's"
    )
    r.add_argument("--no-refresh-harness", dest="refresh_harness", action="store_false")
    b = sub.add_parser("bisect")
    b.add_argument("night", type=Path)
    b.add_argument("--base", type=Path, help="the green night (default: the state's)")
    b.add_argument("--good", help="bisect from this commit, not the green night's")
    p = sub.add_parser("report")
    p.add_argument("night", type=Path)
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("install")
    a = ap.parse_args()
    if a.cmd == "install":
        return install(a)
    if a.cmd == "run" and a.fake_bad and not a.dry_run:
        ap.error("--fake-bad is a rehearsal: it needs --dry-run")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"{LOCK} is held: another nightly is running")
        return 1
    if a.cmd == "run":
        return do_night(a)
    state = load_state()
    if a.cmd == "report":
        report(a.night, state, dry_run=a.dry_run, prev_status=state.get("last_status"))
        return 0
    base = a.base or Path(state["last_green"]["dir"])
    result = json.loads((a.night / "triage.json").read_text())
    out = bisect(a.night, base, result, good_sha=a.good)
    (a.night / "bisect.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
