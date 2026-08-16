#!/usr/bin/env python3
"""Benchmark the momwire portal end to end, driven the way SimNEC drives it.

SimNEC runs its NEC engine as a resident child process: a deck goes down the
child's stdin terminated by an ``NX`` card, and the solve is complete when the
``DATA CARD No: n NX`` echo line comes back on stdout — that is the line
SimNEC blocks on in ``readLine()``.  This script reproduces that drive
pattern exactly and times every deck unit to its sentinel, for:

* the oracle binary SimNEC ships (``nec2c.ae6ty``), and
* ``momwire-nec2c`` under each ``--basis``.

The corpus is the committed fixture deck set under
``tests/fixtures/nec_portal/`` — the same decks the byte-layout conformance
suite runs, so every engine here has already been shown to *answer* them;
this script measures how fast, down one warm process, over repeated passes
(pass 1 is the cold pass; later passes are the steady state a SimNEC session
lives in).

A second phase re-runs the default basis with ``--cache`` and with the
``--cache-stats`` dry run, to measure what the structure cache does to a
stream that re-sends structures it has already sent (each pass after the
first re-sends the whole corpus, which is the sweep/knob-drag shape).

Decks a given engine refuses still answer with the sentinel (that is the
portal's contract); they are timed like any other unit and flagged
``error`` in the results, and the cross-engine comparison should be read on
the clean subset.

Usage
-----
    python scripts/bench_portal_e2e.py --out /tmp/portal-e2e.json
    python scripts/bench_portal_e2e.py --passes 5 --skip-basis hmatrix

The oracle is located like nec_portal_capture.py does: ``NEC_PORTAL_ORACLE``
in the environment, else the SimNEC 2/3 install path.  ``--no-oracle`` runs
the portal bases alone.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "nec_portal"

DEFAULT_ORACLE = Path(
    "/home/smburns/.SimNEC/2/3/Examples/nec2c.ae6ty/bin/nec2c-ubuntu-x86"
)
ORACLE_ENV = "NEC_PORTAL_ORACLE"

# The echo line SimNEC blocks on: solve done when this arrives.
SENTINEL_RE = re.compile(r"^\s*DATA CARD No:\s+\d+\s+NX\b")
ERROR_RE = re.compile(r"^ERROR")

DECK_TIMEOUT_S = 60.0

PORTAL_BASES = (
    "bspline",
    "bspline-d1",
    "sinusoidal",
    "sinusoidal-galerkin",
    "hmatrix",
    "arrayblock",
)


def deck_units(path: Path) -> list[str]:
    """Split a deck file into NX-terminated units, as SimNEC sends them."""
    units: list[str] = []
    current: list[str] = []
    for line in path.read_text().splitlines():
        current.append(line)
        if line.split() and line.split()[0].upper() == "NX":
            units.append("\n".join(current) + "\n")
            current = []
    # Trailing cards with no NX (an EN-terminated or truncated file) are not
    # part of the resident protocol; the corpus has none, but be explicit.
    if any(l.strip() for l in current):
        raise ValueError(f"{path.name}: trailing cards after last NX")
    return units


class ResidentEngine:
    """One resident engine process, driven over stdin/stdout like SimNEC."""

    def __init__(self, cmd: list[str]):
        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def solve(self, unit: str) -> dict:
        """Send one NX-terminated unit; block to its sentinel; time it."""
        assert self.proc.stdin is not None
        t0 = time.perf_counter()
        self.proc.stdin.write(unit)
        self.proc.stdin.flush()
        saw_error = False
        while True:
            try:
                line = self._lines.get(timeout=DECK_TIMEOUT_S)
            except queue.Empty:
                return {"t": None, "error": True, "timeout": True}
            if line is None:
                return {"t": None, "error": True, "died": True}
            if ERROR_RE.match(line):
                saw_error = True
            if SENTINEL_RE.match(line):
                return {"t": time.perf_counter() - t0, "error": saw_error}

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def bench_engine(
    name: str, cmd: list[str], corpus: list[tuple[str, str]], passes: int
) -> dict:
    """Run `passes` full-corpus passes down one resident process."""
    print(f"  [{name}] {' '.join(cmd)}", flush=True)
    engine = ResidentEngine(cmd)
    runs: dict[str, list[dict]] = {label: [] for label, _ in corpus}
    try:
        for p in range(passes):
            t_pass = time.perf_counter()
            for label, unit in corpus:
                result = engine.solve(unit)
                result["pass"] = p
                runs[label].append(result)
                if result.get("died") or result.get("timeout"):
                    print(
                        f"    !! {label}: engine "
                        f"{'died' if result.get('died') else 'timed out'}"
                        f" on pass {p}; respawning",
                        flush=True,
                    )
                    engine.close()
                    engine = ResidentEngine(cmd)
            print(
                f"    pass {p}: {time.perf_counter() - t_pass:.2f}s total", flush=True
            )
    finally:
        engine.close()
    return {"cmd": cmd, "runs": runs}


def read_version(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd + ["-version"], capture_output=True, text=True, timeout=30
        ).stdout
        return out.splitlines()[0].strip() if out else ""
    except Exception:
        return ""


def summarize(results: dict, corpus_labels: list[str], passes: int) -> None:
    """Print a per-engine aggregate over the warm passes, clean subset."""
    # A deck is 'clean' for the comparison if NO engine errored on it.
    dirty: set[str] = set()
    for engine in results.values():
        for label, rs in engine["runs"].items():
            if any(r["error"] for r in rs):
                dirty.add(label)
    clean = [l for l in corpus_labels if l not in dirty]
    print(
        f"\nClean subset: {len(clean)}/{len(corpus_labels)} deck units "
        f"(excluded: {', '.join(sorted(dirty)) or 'none'})"
    )
    header = (
        f"{'engine':<28} {'cold total':>10} {'warm total':>10} "
        f"{'warm/deck med':>13} {'warm/deck max':>13}"
    )
    print(header)
    print("-" * len(header))
    for name, engine in results.items():
        cold = sum(engine["runs"][l][0]["t"] or 0 for l in clean)
        warm_totals = []
        for p in range(1, passes):
            warm_totals.append(sum(engine["runs"][l][p]["t"] or 0 for l in clean))
        warm_per_deck = [
            r["t"] for l in clean for r in engine["runs"][l][1:] if r["t"] is not None
        ]
        if not warm_per_deck:
            continue
        print(
            f"{name:<28} {cold:>9.2f}s {min(warm_totals):>9.2f}s "
            f"{statistics.median(warm_per_deck) * 1000:>11.1f}ms "
            f"{max(warm_per_deck) * 1000:>11.1f}ms"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("/tmp/bench-portal-e2e.json"))
    ap.add_argument(
        "--passes",
        type=int,
        default=3,
        help="full-corpus passes per engine (pass 1 is cold)",
    )
    ap.add_argument("--oracle", type=Path, default=None)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--skip-basis", action="append", default=[], choices=PORTAL_BASES)
    ap.add_argument("--no-cache-phase", action="store_true")
    args = ap.parse_args()

    corpus: list[tuple[str, str]] = []
    for deck in sorted(FIXTURE_DIR.glob("*.deck")):
        for i, unit in enumerate(deck_units(deck)):
            label = deck.stem if i == 0 else f"{deck.stem}#{i}"
            corpus.append((label, unit))
    labels = [label for label, _ in corpus]
    print(
        f"Corpus: {len(corpus)} deck units from "
        f"{len(set(l.split('#')[0] for l in labels))} fixture decks"
    )

    portal = Path(sys.executable).parent / "momwire-nec2c"
    if not portal.is_file():
        print(f"momwire-nec2c not found next to {sys.executable}", file=sys.stderr)
        return 2

    engines: list[tuple[str, list[str]]] = []
    if not args.no_oracle:
        oracle = args.oracle or Path(os.environ.get(ORACLE_ENV, DEFAULT_ORACLE))
        if oracle.is_file() and os.access(oracle, os.X_OK):
            engines.append(("oracle", [str(oracle)]))
        else:
            print(f"oracle not runnable at {oracle}; skipping", file=sys.stderr)
    for basis in PORTAL_BASES:
        if basis in args.skip_basis:
            continue
        engines.append((basis, [str(portal), "--basis", basis]))

    results: dict[str, dict] = {}
    for name, cmd in engines:
        results[name] = bench_engine(name, cmd, corpus, args.passes)
        results[name]["version"] = read_version(cmd)

    cache_stats = None
    if not args.no_cache_phase:
        stats_path = Path("/tmp") / f"bench-portal-cache-{os.getpid()}.json"
        name = "bspline+cache"
        results[name] = bench_engine(
            name,
            [str(portal), "--cache", "--cache-stats", str(stats_path)],
            corpus,
            args.passes,
        )
        if stats_path.is_file():
            cache_stats = json.loads(stats_path.read_text())
            stats_path.unlink()
            print(f"  cache stats: {cache_stats}")

    summarize(results, labels, args.passes)

    args.out.write_text(
        json.dumps(
            {
                "date": _dt.datetime.now().isoformat(timespec="seconds"),
                "host": platform.node(),
                "python": sys.version.split()[0],
                "passes": args.passes,
                "corpus": labels,
                "cache_stats": cache_stats,
                "engines": results,
            },
            indent=1,
        )
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
