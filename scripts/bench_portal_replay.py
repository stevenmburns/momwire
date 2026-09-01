#!/usr/bin/env python3
"""Replay a captured SimNEC deck stream against a crew of portals (#375).

The live matrix's problem is the workload: a human at the GUI cannot do the
same thing twice.  This driver takes the golden stream a single instrumented
session captured (``portal_bench_shim.py`` records every deck's full text
and arrival time) and replays it byte-identically against any
(crew, threads) configuration, headless — so every cell of the matrix sees
the same decks in the same order with the same think-time gaps, and the runs
can wait for an idle box.

What is emulated: SimNEC's Workforce fan-out, as dispatch-to-first-idle-
member with queueing when all are busy.  What is NOT emulated: SimNEC's
member-respawn behaviour (the crew here is fixed for the run) and whatever
affinity its scheduler really uses — so crew-16 numbers here are the
fan-out effect only, and the live 51-process churn is a separate finding
this driver deliberately excludes.

Metrics per deck: queue wait (arrival to dispatch) and solve latency
(dispatch to NX sentinel) — their sum is what a SimNEC user experiences.
Plus per-process cache stats and a peak-RSS trace, same as the live
harness; ``bench_portal_crew.py collect`` reads the output dir unchanged.

Usage
-----
    bench_portal_replay.py --capture GOLDEN_DIR --crew 4 --threads 2 \\
        --run-dir ~/antennas/375-matrix/replay-crew4-t2 [--gaps 1.0]

``--capture`` takes a run dir with decks.*.jsonl (rows without a ``deck``
field, from the pre-capture shim, are refused).  ``--gaps 0`` fires decks
as fast as the crew accepts them (throughput mode); ``--gaps 1.0`` (the
default) honours the captured think-time gaps (latency mode).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

SENTINEL_RE = re.compile(r"^\s*DATA CARD No:\s+\d+\s+NX\b")
DECK_TIMEOUT_S = 120.0


def load_stream(capture_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(capture_dir.glob("decks.*.jsonl")):
        for line in f.read_text().splitlines():
            rows.append(json.loads(line))
    if not rows:
        sys.exit(f"no decks.*.jsonl rows under {capture_dir}")
    missing = sum(1 for r in rows if "deck" not in r)
    if missing:
        sys.exit(
            f"{missing}/{len(rows)} captured rows carry no deck text -- "
            "re-capture the golden session with the current shim"
        )
    rows.sort(key=lambda r: r["t_in"])
    t0 = rows[0]["t_in"]
    for r in rows:
        r["offset"] = r["t_in"] - t0
    return rows


class Member:
    """One resident portal; a reader thread flips it idle on the sentinel."""

    def __init__(self, cmd: list[str], env: dict[str, str]):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self.lock = threading.Lock()
        self.busy = False
        self.current: dict | None = None
        self.done: list[dict] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if SENTINEL_RE.match(line):
                with self.lock:
                    rec = self.current
                    if rec is not None:
                        rec["t_done"] = time.perf_counter()
                        rec["solve_ms"] = round(
                            (rec["t_done"] - rec["t_disp"]) * 1000, 2
                        )
                        self.done.append(rec)
                    self.current = None
                    self.busy = False

    def dispatch(self, rec: dict) -> None:
        with self.lock:
            self.busy = True
            rec["t_disp"] = time.perf_counter()
            self.current = rec
        assert self.proc.stdin is not None
        self.proc.stdin.write(rec["deck"])
        self.proc.stdin.flush()

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - teardown falls back to kill(); a bench must not die closing a child
            self.proc.kill()


def rss_sampler(run_dir: Path, stop: threading.Event) -> None:
    out = (run_dir / "rss.jsonl").open("a")
    while not stop.is_set():
        total, count = 0, 0
        pgrep = subprocess.run(
            ["pgrep", "-f", "momwire-nec2c"], capture_output=True, text=True
        ).stdout
        for pid in pgrep.split():
            try:
                with open(f"/proc/{pid}/status") as fh:
                    for ln in fh:
                        if ln.startswith("VmRSS:"):
                            total += int(ln.split()[1])
                            count += 1
                            break
            except OSError:
                pass
        out.write(
            json.dumps(
                {
                    "t": time.time(),
                    "portal_kb": total,
                    "n_portals": count,
                    "shim_kb": 0,
                    "java_kb": 0,
                }
            )
            + "\n"
        )
        out.flush()
        stop.wait(0.5)
    out.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--crew", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument(
        "--gaps",
        type=float,
        default=1.0,
        help="think-time scale: 1.0 captured pace, 0 flat out",
    )
    ap.add_argument("--basis", default=None)
    args = ap.parse_args()

    stream = load_stream(args.capture)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    portal = str(Path(sys.executable).parent / "momwire-nec2c")
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = str(args.threads)

    members: list[Member] = []
    for i in range(args.crew):
        cmd = [portal, "--cache", "--cache-stats", str(run_dir / f"cache.m{i}.json")]
        if args.basis:
            cmd += ["--basis", args.basis]
        members.append(Member(cmd, env))

    stop = threading.Event()
    threading.Thread(target=rss_sampler, args=(run_dir, stop), daemon=True).start()

    print(
        f"replaying {len(stream)} decks: crew={args.crew} "
        f"threads={args.threads} gaps=x{args.gaps}",
        flush=True,
    )
    t0 = time.perf_counter()
    for n, rec in enumerate(stream):
        due = t0 + rec["offset"] * args.gaps
        while time.perf_counter() < due:
            time.sleep(min(0.005, due - time.perf_counter()))
        rec["t_arrive"] = time.perf_counter()
        member = None
        deadline = time.perf_counter() + DECK_TIMEOUT_S
        while member is None:
            member = next((m for m in members if not m.busy), None)
            if member is None:
                if time.perf_counter() > deadline:
                    sys.exit(
                        f"deck {n}: no member freed in {DECK_TIMEOUT_S}s; aborting"
                    )
                time.sleep(0.0005)
        rec["queue_ms"] = round((time.perf_counter() - rec["t_arrive"]) * 1000, 2)
        member.dispatch(rec)
        if n and n % 100 == 0:
            print(f"  {n}/{len(stream)}", flush=True)

    while any(m.busy for m in members):
        time.sleep(0.005)
    wall = time.perf_counter() - t0
    stop.set()
    for m in members:
        m.close()

    log = (run_dir / f"decks.{os.getpid()}.jsonl").open("w")
    for i, m in enumerate(members):
        for rec in m.done:
            log.write(
                json.dumps(
                    {
                        "seq": rec["seq"],
                        "member": i,
                        "fr": rec.get("fr", ""),
                        "geom": rec.get("geom", ""),
                        "cards": rec.get("cards", 0),
                        "t_in": rec["t_arrive"],
                        "t_out": rec["t_done"],
                        "queue_ms": rec["queue_ms"],
                        "dt_ms": round(rec["queue_ms"] + rec["solve_ms"], 2),
                        "solve_ms": rec["solve_ms"],
                    }
                )
                + "\n"
            )
    log.close()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "crew": args.crew,
                "threads": args.threads,
                "replay": True,
                "gaps": args.gaps,
                "capture": str(args.capture),
                "basis": args.basis or "bspline",
                "date": _dt.datetime.now().isoformat(timespec="seconds"),
                "wall_s": round(wall, 2),
                "portal": portal,
            },
            indent=1,
        )
    )
    print(
        f"done in {wall:.1f}s -> {run_dir} "
        "(summarize with bench_portal_crew.py collect)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
