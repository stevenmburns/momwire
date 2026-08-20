#!/usr/bin/env python3
"""Sequence fuzzer for the ``dipole_ne_nearfield`` byte-oracle flake (#464).

Instrument 1 of the issue's hunt plan, same shape as the momwire#403 hunt:
one resident ``momwire-nec2c-shared`` server, a crew of client processes
firing CONCURRENTLY at it (the "loaded full-suite xdist run" the flake has
only ever reproduced under), each shipping a randomised permutation of the
fixture corpus with ``dipole_ne_nearfield`` pinned last. Every served answer
is byte-compared, after ``canonicalize_timings``, against ``stock()`` -- a
fresh in-process run of the SAME concatenated deck string -- exactly the
comparison ``tests/test_portal_shared.py``'s byte oracle makes. A mismatch
dumps both printouts and the permutation that produced them.

This only tests whether SEQUENCE/LOAD moves the printed bytes. It does not
by itself prove the mechanism is the NE table's missing floor -- that is
argued from the fixed value (``EX = 9.1237E-17`` against a co-component of
``2.6506E-01``, ~3e-16 relative: pure double cancellation dust) and fixed
independently of whether this fuzzer ever catches it moving.

Usage
-----
    .venv/bin/python scripts/dipole_ne_nearfield_fuzz.py --rounds 60 --workers 8
    .venv/bin/python scripts/dipole_ne_nearfield_fuzz.py --inject-fault

``--inject-fault`` is the self-test the issue's gate 2 asks for: it corrupts
one digit of the served printout before the comparison on the first round
only, to prove the harness actually flags a mismatch it is looking for
rather than silently discarding one. It is a probe of THIS SCRIPT, not of
the engine, and is not meant to run alongside a real fuzz sweep.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "nec_portal"
CLIENT_ARGV = [sys.executable, "-m", "momwire_nec2c_client"]
TARGET = "dipole_ne_nearfield"


def _load_capture_module():
    path = REPO_ROOT / "scripts" / "nec_portal_capture.py"
    spec = importlib.util.spec_from_file_location("nec_portal_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = _load_capture_module()

ALL_NAMES = tuple(sorted(p.stem for p in FIXTURE_DIR.glob("*.deck")))
OTHER_NAMES = tuple(n for n in ALL_NAMES if n != TARGET)


def fixture_deck(name: str) -> str:
    return (FIXTURE_DIR / f"{name}.deck").read_text()


def run_client(
    room: str, decks: str, timeout: float = 120.0
) -> subprocess.CompletedProcess:
    return subprocess.run(
        CLIENT_ARGV,
        input=decks,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": room},
    )


def stock(decks: str) -> str:
    # In-process, same reasoning as tests/test_portal_shared.py's stock():
    # configure_engine resets the basis and the cache on entry, so this call
    # is indistinguishable from a fresh process.
    from momwire.portal import main as portal_main

    out = io.StringIO()
    rc = portal_main([], io.StringIO(decks), out, io.StringIO())
    assert rc == 0
    return out.getvalue()


def same_printout(left: str, right: str) -> tuple[bool, str, str]:
    a = CAPTURE.canonicalize_timings(left)
    b = CAPTURE.canonicalize_timings(right)
    return a == b, a, b


def random_sequence(rng: random.Random, min_others: int, max_others: int) -> list[str]:
    n = rng.randint(min_others, max_others)
    others = rng.sample(OTHER_NAMES, n)
    rng.shuffle(others)
    return [*others, TARGET]


def inject_digit_fault(text: str) -> str:
    """Flip one digit in the NE table's EX column -- the self-test corruption
    for ``--inject-fault``. Finds the first ``NEAR ELECTRIC FIELDS`` row and
    bumps a digit in its first magnitude field by one (wrapping 9 -> 0)."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 9 and parts[0].replace(".", "").replace("-", "").isdigit():
            # crude: bump the first digit character found in the EX magnitude
            # field (columns 30:43 per fmt_near_field_row's 13.4E width).
            start = 30
            field = line[start : start + 13]
            for j, ch in enumerate(field):
                if ch.isdigit():
                    bumped = str((int(ch) + 1) % 10)
                    field = field[:j] + bumped + field[j + 1 :]
                    lines[i] = line[:start] + field + line[start + 13 :]
                    return "".join(lines)
    raise AssertionError("no NE table row found to inject a fault into")


def run_round(
    room: str, rng: random.Random, dump_dir: Path | None, round_id: int, inject: bool
):
    seq = random_sequence(rng, 3, 10)
    decks = "".join(fixture_deck(n) for n in seq)
    served = run_client(room, decks)
    if served.returncode != 0:
        return {
            "round": round_id,
            "seq": seq,
            "ok": False,
            "reason": f"client rc={served.returncode}: {served.stderr[-500:]}",
        }
    served_out = served.stdout
    if inject and round_id == 0:
        served_out = inject_digit_fault(served_out)
    expected = stock(decks)
    ok, a, b = same_printout(served_out, expected)
    if not ok and dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{round_id}-{os.getpid()}-{time.monotonic_ns()}"
        (dump_dir / f"{stamp}.seq.txt").write_text("\n".join(seq))
        (dump_dir / f"{stamp}-served.txt").write_text(a)
        (dump_dir / f"{stamp}-stock.txt").write_text(b)
    return {"round": round_id, "seq": seq, "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument(
        "--workers", type=int, default=8, help="concurrent client processes per batch"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-dir", type=Path, default=None)
    ap.add_argument(
        "--inject-fault",
        action="store_true",
        help="self-test: corrupt round 0's served output and confirm it is flagged",
    )
    args = ap.parse_args()

    # Each round carries its OWN Random, derived from the seed and the round
    # index, so a sweep is reproducible whatever order the pool runs them in.
    room = tempfile.mkdtemp(prefix="mw464fuzz-")
    dump_dir = args.dump_dir
    mismatches = []
    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    run_round,
                    room,
                    random.Random(args.seed * 100_003 + i),
                    dump_dir,
                    i,
                    args.inject_fault,
                )
                for i in range(args.rounds)
            ]
            for fut in futures:
                result = fut.result()
                if not result["ok"]:
                    mismatches.append(result)
    finally:
        # Reap the resident server the same way tests/test_portal_shared.py's
        # runtime fixture does: SIGTERM by pid, not by name.
        log_files = list(Path(room).glob("*.log"))
        text = "".join(p.read_text(errors="replace") for p in log_files)
        for pid_str, _sock in re.findall(
            r"^\S+ listening pid=(\d+) socket=(\S+)$", text, re.MULTILINE
        ):
            try:
                os.kill(int(pid_str), signal.SIGTERM)
            except OSError:
                pass
        time.sleep(0.2)
        shutil.rmtree(room, ignore_errors=True)

    dt = time.monotonic() - t0
    print(
        f"rounds={args.rounds} workers={args.workers} seed={args.seed} elapsed={dt:.1f}s"
    )
    print(f"mismatches={len(mismatches)}")
    for m in mismatches:
        reason = m.get("reason", "byte mismatch after canonicalize_timings")
        print(f"  round {m['round']}: {reason}")
        print(f"    sequence: {' -> '.join(m['seq'])}")
    if args.inject_fault:
        hit = any(m["round"] == 0 for m in mismatches)
        print(
            f"inject-fault self-test: {'CAUGHT' if hit else 'MISSED (BUG IN FUZZER)'}"
        )
        return 0 if hit else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
