"""The fat executable's printouts, byte for byte against the one-variant builds.

    python scripts/windows_fat/fat_compare.py FAT VARIANT SINGLE DECK_OR_DIR [...]
        [--auto] [--sde SDE --cpu CPU] [--coretype NAME] [--copies policy,threads]

Every `*.nec` under the DIRs (and every DECK named) is answered by FAT with MOMWIRE_FORCE_VARIANT=
VARIANT and by SINGLE (the one-variant executable of that VARIANT), each
twice: at the size policy's thread count (mostly the serial copy, on these
decks) and through the threaded copy (MW_EXE_COPY=threads), so all four of
the fat build's copies are reached. The printouts must be the same bytes, and
so must the exit status and stdout. Prints one line per mismatch and the
counts; exits 1 on any mismatch, or when nothing was compared.

--auto: FAT picks its own variant by CPU (no MOMWIRE_FORCE_VARIANT).
--sde/--cpu: both run under Intel SDE emulating CPU (`wsm`, `snb`, `hsw`),
which faults on any instruction that CPU lacks. Both, not FAT alone: the
DLLs beside them (OpenBLAS, the UCRT's math, the VC++ runtime) pick their
own kernels by CPUID too, so only the same emulated CPU makes the same bytes.
--sde-extra: more SDE options, comma-separated (`-chip_check_exe_only`: fault
only on the executable's own instructions, not the DLLs').
--coretype: OPENBLAS_CORETYPE for both.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TIMEOUT = 900


def run(
    cmd: list[str], deck: Path, out: Path, env: dict[str, str]
) -> tuple[int, bytes, bytes]:
    p = subprocess.Popen(
        [*cmd, str(deck), str(out)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = p.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        text = out.read_text(errors="replace") if out.exists() else ""
        p.kill()
        p.communicate()
        out.unlink(missing_ok=True)
        print(
            f"HANG {cmd[-1]} {deck}: printout"
            f" {'complete' if 'RUN TIME' in text else 'incomplete'} ({len(text)} chars)",
            flush=True,
        )
        return -999, b"HANG", b""
    body = out.read_bytes() if out.exists() else b""
    out.unlink(missing_ok=True)  # the next run writes the same path
    if b"SDE-ERROR" in stderr:
        # an instruction the emulated CPU lacks: a failure even when both
        # executables fault alike (run 44010094276 counted those as equal)
        print(f"SDE FAULT {cmd[-1]} {deck}: {stderr[-400:]!r}", flush=True)
        return -998, b"SDE-ERROR", b""
    if p.returncode != 0:
        print(
            f"exit {p.returncode & 0xFFFFFFFF:#010x} {cmd[-1]} {deck}: {stderr[-400:]!r}",
            flush=True,
        )
    return p.returncode, stdout, body


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fat")
    ap.add_argument("variant", choices=("avx2", "sse2"))
    ap.add_argument("single")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--sde")
    ap.add_argument("--cpu")
    ap.add_argument("--sde-extra", default="", help="comma-separated SDE options")
    ap.add_argument("--coretype")
    ap.add_argument("--copies", default="policy,threads")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    extra = [x for x in a.sde_extra.split(",") if x]
    wrap = [a.sde, f"-{a.cpu}", *extra, "--"] if a.sde else []
    fat_cmd = [*wrap, a.fat]
    one_cmd = [*wrap, a.single]
    decks = sorted(
        {Path(d)} if Path(d).is_file() else set(Path(d).rglob("*.nec")) for d in a.dirs
    )
    decks = sorted(set().union(*decks))
    copies = a.copies.split(",")
    work = Path(tempfile.mkdtemp())
    jobs = [(i, deck, copy) for i, deck in enumerate(decks) for copy in copies]

    def one(job):
        i, deck, copy = job
        env = dict(os.environ)
        env["MOMWIRE_FORCE_VARIANT"] = a.variant
        env.pop("MOMWIRE_THREADS", None)
        if a.coretype:
            env["OPENBLAS_CORETYPE"] = a.coretype
        if copy == "threads":
            env["MW_EXE_COPY"] = "threads"
            env["MOMWIRE_THREADS"] = "2"
        else:
            env.pop("MW_EXE_COPY", None)
        fat_env = dict(env)
        if a.auto:
            fat_env.pop("MOMWIRE_FORCE_VARIANT", None)
        # the same printout path for both, in case a printout names it
        out = work / f"{i}-{copy}.out"
        x = run(fat_cmd, deck, out, fat_env)
        y = run(one_cmd, deck, out, env)
        return deck, copy, x, y

    same = differ = refused = faulted = nonzero = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for deck, copy, x, y in pool.map(one, jobs):
            if -998 in (x[0], y[0]):
                faulted += 1
                continue
            if x[0] != 0 or y[0] != 0:
                nonzero += 1
            if x == y:
                same += 1
                if b"not served by this build" in x[2]:
                    refused += 1
            else:
                differ += 1
                what = (
                    "HANG(fat)"
                    if x[0] == -999
                    else "HANG(one)"
                    if y[0] == -999
                    else "rc"
                    if x[0] != y[0]
                    else "stdout"
                    if x[1] != y[1]
                    else "printout"
                )
                print(
                    f"DIFFER {what} {copy} {deck} (exit fat {x[0] & 0xFFFFFFFF:#x},"
                    f" one {y[0] & 0xFFFFFFFF:#x})",
                    flush=True,
                )
    how = f"under SDE -{a.cpu}" if a.sde else "native"
    pick = "picked by CPU" if a.auto else "forced"
    print(
        f"{a.variant} ({pick}, {how}): {same} identical ({refused} of them the same refusal),"
        f" {differ} different, {faulted} SDE faults, {nonzero} with a nonzero exit,"
        f" {len(decks)} decks x {len(copies)} copies"
    )
    return 1 if differ or faulted or not same else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
