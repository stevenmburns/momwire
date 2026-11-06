"""Hang diagnosis (0.87.0 pre-tag, macOS): run a command, and when its
progress file stops growing, sample every process in its tree (and every
momwire process on the box) before killing it.

usage: watch.py MODE OUTDIR BUDGET_S
  MODE replica: the macos-nopython exe_compare step, razor2p then bs2, again
                and again, a cold Sommerfeld store each iteration
  MODE exe:     the razor2p and bs2 executables alone over every deck, three
                at a time, each under a timeout
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path.cwd()
D = ROOT / "dist/momwire-nec5-macos-arm64"
STALL_S = 300


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return f"[timeout: {cmd}]"


def ps_table():
    rows = {}
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,etime=,stat=,pcpu=,command="],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        f = line.split(None, 5)
        if len(f) == 6:
            rows[int(f[0])] = (int(f[1]), line)
    return rows


def descendants(root, rows):
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += [c for c, (pp, _) in rows.items() if pp == p]
    return out


def capture(root, outdir, tag):
    rows = ps_table()
    pids = descendants(root, rows)
    extra = [p for p, (_, line) in rows.items()
             if ("momwire" in line or "exe_compare" in line) and p not in pids
             and "watch.py" not in line]
    d = outdir / f"hang-{tag}-{int(time.time())}"
    d.mkdir(parents=True)
    with open(d / "ps.txt", "w") as f:
        f.write("tree:\n")
        for p in pids:
            f.write(rows.get(p, (0, f"{p} gone"))[1] + "\n")
        f.write("other momwire processes:\n")
        for p in extra:
            f.write(rows[p][1] + "\n")
        f.write("\nps -M (threads):\n")
        for p in pids + extra:
            f.write(sh(f"ps -M -p {p}") + "\n")
    for p in pids + extra:
        line = rows.get(p, (0, ""))[1]
        (d / f"sample-{p}.txt").write_text(line + "\n" + sh(f"sudo sample {p} 3 -mayDie"))
        if "python" in line.lower() or "momwire-nec5-razor-2p" in line or "momwire-nec5-bspline" in line or "venv/bin" in line:
            (d / f"pyspy-{p}.txt").write_text(line + "\n" + sh(f"sudo py-spy dump --native --pid {p}", 180))
    # a second sample later: a livelock moves, a deadlock does not
    time.sleep(20)
    for p in pids + extra:
        (d / f"sample2-{p}.txt").write_text(sh(f"sudo sample {p} 2 -mayDie"))
    (d / "lsof.txt").write_text("\n".join(sh(f"sudo lsof -p {p}") for p in pids + extra))
    return d, pids + extra


def kill_all(pids):
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


def cold_store():
    shutil.rmtree(Path.home() / "Library/Caches/momwire", ignore_errors=True)


def replica(outdir, budget):
    t_end = time.time() + budget
    log = open(outdir / "replica.jsonl", "a", buffering=1)
    it = 0
    env = dict(os.environ, PATH=f"{ROOT}/venv/bin:" + os.environ["PATH"],
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    while time.time() < t_end:
        for b in ("razor2p", "bs2"):
            if time.time() >= t_end:
                break
            cold_store()
            extra, launcher = [], f"{ROOT}/venv/bin/momwire-nec5-razor-2p"
            if b == "bs2":
                extra = ["--basis", "bspline", "scripts/core_split/bs2_stream_decks"]
                launcher = f"{ROOT}/venv/bin/momwire-nec5-bspline"
            prog = outdir / f"exe-mac-{b}-{it}.jsonl"
            cmd = [f"{ROOT}/venv/bin/python", "scripts/core_split/exe_compare.py", str(prog),
                   "--standalone", "--rtol", "macos", "--exe", str(D / f"momwire-nec5-{b}-macos"),
                   "--launcher", launcher, "--workers", "3", *extra,
                   "tests/fixtures", "scripts/core_split/nx_decks"]
            t0 = time.time()
            p = subprocess.Popen(cmd, env=env, stdout=open(outdir / f"out-{b}-{it}.txt", "w"),
                                 stderr=subprocess.STDOUT, start_new_session=True)
            last_n, last_t, rec = -1, time.time(), None
            while True:
                try:
                    rc = p.wait(timeout=10)
                    rec = {"it": it, "b": b, "rc": rc, "s": round(time.time() - t0, 1)}
                    break
                except subprocess.TimeoutExpired:
                    pass
                n = sum(1 for _ in open(prog)) if prog.exists() else 0
                if n != last_n:
                    last_n, last_t = n, time.time()
                elif time.time() - last_t > STALL_S:
                    d, pids = capture(p.pid, outdir, f"{b}-{it}")
                    rec = {"it": it, "b": b, "rc": "HANG", "s": round(time.time() - t0, 1),
                           "lines": n, "capture": str(d)}
                    kill_all(pids)
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    p.wait()
                    break
                if time.time() > t_end + 600:
                    break
            log.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
        it += 1


def decks():
    out = []
    for top in ("tests/fixtures", "scripts/core_split/nx_decks"):
        out += sorted(q for q in Path(top).rglob("*") if q.is_file() and q.suffix in (".nec", ".deck"))
    return out


def exe_only(outdir, budget):
    t_end = time.time() + budget
    log = open(outdir / "exe.jsonl", "a", buffering=1)
    work = outdir / "work"
    work.mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MOMWIRE_THREADS="1")
    all_decks = decks()
    hangs = 0

    def one(args):
        nonlocal hangs
        k, b, deck = args
        out = work / f"{k}.out"
        p = subprocess.Popen([str(D / f"momwire-nec5-{b}-macos"), str(deck), str(out)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        try:
            rc = p.wait(timeout=STALL_S)
            rec = None if rc in (0, 1, 2) else {"b": b, "deck": str(deck), "rc": rc}
        except subprocess.TimeoutExpired:
            hangs += 1
            d, pids = capture(p.pid, outdir, f"exe-{b}-{k}")
            kill_all(pids)
            p.wait()
            rec = {"b": b, "deck": str(deck), "rc": "HANG", "capture": str(d)}
        out.unlink(missing_ok=True)
        if rec is not None:
            rec["s"] = round(time.time() - t0, 2)
        return rec

    rnd, n = 0, 0
    with ThreadPoolExecutor(3) as pool:
        while time.time() < t_end:
            cold_store()
            jobs = [(f"{rnd}-{b}-{i}", b, dk) for i, dk in enumerate(all_decks) for b in ("razor2p", "bs2")]
            for rec in pool.map(one, jobs):
                n += 1
                if rec is not None:
                    log.write(json.dumps(rec) + "\n")
            log.write(json.dumps({"round": rnd, "runs_so_far": n, "hangs": hangs}) + "\n")
            print(f"round {rnd}: {n} runs, {hangs} hangs", flush=True)
            rnd += 1


if __name__ == "__main__":
    mode, outdir, budget = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    (replica if mode == "replica" else exe_only)(outdir, budget)
