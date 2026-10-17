"""SimNEC's streaming call (``- -``, momwire#1512), the no-Python executable
against ``momwire-nec5-razor-2p``, through the same pipes.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
    python scripts/core_split/stream_compare.py OUT.jsonl --exe EXE \\
        --launcher L [--workers W] [--timeout S] STREAM_OR_DIR...

Each input is the stdin bytes of one streaming call (the ``simnec_stream_*``
decks in ``nx_decks/`` are SimNEC's own, captured by ``simcap.py``). Both
engines are run on it twice:

* ``lockstep`` -- as SimNEC drives it: the input is cut into structures at
  each complete ``NX`` / ``EN`` line (``next_structure``), one structure is
  written, and nothing more until the engine has flushed that structure's
  printout, read up to its ``NX`` echo (or to the end of stdout). An engine
  that held a printout back would stall here, so the gate fails on the
  timeout. Each structure's output is recorded as its own segment.
* ``whole`` -- every byte written at once and stdin closed: the same
  printouts, however the pipe happens to chunk them.

Byte for byte: the executable's segments must be the launcher's segments,
each run's ``whole`` must be the concatenation of its segments, and the
launcher's stream must be what its file form prints for the same bytes, less
the truncated empty structure a file ending on ``NX`` is refused for (the
stream just ends there). One JSON line per input; a summary on stdout; exit
status 1 on any mismatch.
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SPLIT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPLIT))

import printout_compare as pc  # noqa: E402

from momwire.deck._nec5 import next_structure  # noqa: E402

_NX_ECHO = re.compile(rb"(?:^|\n) \*{5} INPUT LINE +\d+ +NX\b[^\n]*\r\n\Z")
_TRUNCATED = b"deck has no EN card"


def _arg(args, flag, default=None):
    if flag in args:
        k = args.index(flag)
        value = args[k + 1]
        del args[k : k + 2]
        return value
    return default


def pieces(data: bytes) -> list[bytes]:
    """The writes SimNEC makes: one structure each, cut after its NX / EN
    line; whatever follows the last cut is one more write."""
    text = data.decode("latin-1")
    out = []
    while text:
        cut = next_structure(text, final=True)
        if cut is None:
            out.append(text)
            break
        out.append(text[: cut[0]])
        if cut[1] == "EN":
            break
        text = text[cut[0] :]
    return [p.encode("latin-1") for p in out]


def _read_until(proc, buf: bytearray, timeout: float) -> str:
    """Read stdout until it ends on an NX echo ("nx"), or closes ("eof")."""
    fd = proc.stdout.fileno()
    deadline = time.monotonic() + timeout
    while True:
        if _NX_ECHO.search(bytes(buf[-400:])):
            return "nx"
        left = deadline - time.monotonic()
        if left <= 0:
            return "timeout"
        ready, _, _ = select.select([fd], [], [], left)
        if not ready:
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            return "eof"
        buf += chunk


def lockstep(cmd, data: bytes, env, cwd, timeout: float, writes=None) -> dict:
    """SimNEC's drive: each write (``pieces(data)`` unless ``writes`` says
    otherwise), then stdout read up to the NX echo before the next."""
    proc = subprocess.Popen(
        [*cmd, "-", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=cwd,
        bufsize=0,
    )
    segments, states = [], []
    closed = False
    for piece in pieces(data) if writes is None else writes:
        try:
            proc.stdin.write(piece)
            proc.stdin.flush()
        except BrokenPipeError:
            states.append("stdin-closed")
            closed = True
            break
        buf = bytearray()
        state = _read_until(proc, buf, timeout)
        segments.append(bytes(buf))
        states.append(state)
        if state != "nx":
            closed = state == "eof"
            break
    if not closed:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
    tail = bytearray()
    state = _read_until(proc, tail, timeout)
    while state == "nx":  # an NX echo in what came after stdin closed
        state = _read_until(proc, tail, timeout)
    if state == "timeout":
        proc.kill()
    segments.append(bytes(tail))
    states.append("close:" + state)
    return {"segments": segments, "states": states, "rc": proc.wait()}


def whole(cmd, data: bytes, env, cwd, timeout: float) -> dict:
    p = subprocess.run(
        [*cmd, "-", "-"],
        input=data,
        capture_output=True,
        env=env,
        cwd=cwd,
        timeout=timeout * 4,
        check=False,
    )
    return {"out": p.stdout, "rc": p.returncode}


def compare(exe, launcher, path: Path, work: Path, timeout: float) -> dict:
    work.mkdir(parents=True)
    data = path.read_bytes()
    env = dict(os.environ)
    rec = {"stream": str(path), "pieces": len(pieces(data))}
    runs = {}
    for name, cmd in (("launcher", [launcher]), ("exe", [exe])):
        runs[name] = {
            "lockstep": lockstep(cmd, data, env, work, timeout),
            "whole": whole(cmd, data, env, work, timeout),
        }
    problems = []
    ref, got = runs["launcher"], runs["exe"]
    for name, r in runs.items():
        if "timeout" in " ".join(r["lockstep"]["states"]):
            problems.append(
                f"{name}: a structure's printout never arrived: {r['lockstep']['states']}"
            )
        if b"".join(r["lockstep"]["segments"]) != r["whole"]["out"]:
            problems.append(
                f"{name}: lockstep != whole: "
                + pc.first_difference(
                    b"".join(r["lockstep"]["segments"]), r["whole"]["out"]
                )
            )
        if r["lockstep"]["rc"] != 0 or r["whole"]["rc"] != 0:
            problems.append(f"{name}: exit {r['lockstep']['rc']} / {r['whole']['rc']}")
    if got["lockstep"]["states"] != ref["lockstep"]["states"]:
        problems.append(
            f"flush order: {got['lockstep']['states']} (exe) / {ref['lockstep']['states']} (launcher)"
        )
    for k, (a, b) in enumerate(
        zip(ref["lockstep"]["segments"], got["lockstep"]["segments"], strict=False)
    ):
        if a != b:
            problems.append(f"segment {k}: " + pc.first_difference(a, b))
    if len(ref["lockstep"]["segments"]) != len(got["lockstep"]["segments"]):
        problems.append("segment counts differ")
    if ref["whole"]["out"] != got["whole"]["out"]:
        problems.append(
            "whole: " + pc.first_difference(ref["whole"]["out"], got["whole"]["out"])
        )
    # the launcher's stream against its own file form
    deck = work / "stream.nec"
    deck.write_bytes(data)
    subprocess.run(
        [launcher, str(deck), str(work / "file.out")],
        env=env,
        cwd=work,
        capture_output=True,
        check=False,
    )
    file_form = (work / "file.out").read_bytes()
    streamed = ref["whole"]["out"]
    rest = file_form[len(streamed) :]
    if not file_form.startswith(streamed) or (rest and _TRUNCATED not in rest):
        problems.append(
            "stream != file form: " + pc.first_difference(file_form, streamed)
        )
    rec["bytes"] = len(streamed)
    rec["segments"] = [len(s) for s in ref["lockstep"]["segments"]]
    rec["same"] = not problems
    if problems:
        rec["problems"] = problems
    return rec


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    out_path = Path(args.pop(0))
    exe = _arg(args, "--exe")
    launcher = _arg(args, "--launcher")
    workers = int(_arg(args, "--workers", "4"))
    timeout = float(_arg(args, "--timeout", "600"))
    if exe is None or launcher is None:
        raise SystemExit("--exe EXE and --launcher L are required")
    # each call runs in its own directory: the commands as absolute paths
    exe, launcher = (
        str(Path(c).resolve()) if os.sep in c else c for c in (exe, launcher)
    )
    streams = pc._decks(args)
    work = Path(tempfile.mkdtemp(prefix="stream_compare_"))
    # the file form's resident server: a private one, started with this
    # process's environment (MOMWIRE_FORCE_VARIANT included)
    (work / "runtime").mkdir()
    os.environ["MOMWIRE_PORTAL_RUNTIME_DIR"] = str(work / "runtime")
    mismatches = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(out_path, "w") as out:
        futures = [
            pool.submit(compare, exe, launcher, s, work / str(i), timeout)
            for i, s in enumerate(streams)
        ]
        for fut in futures:
            rec = fut.result()
            mismatches += not rec["same"]
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(
                f"{'ok      ' if rec['same'] else 'MISMATCH'} {rec['stream']}: "
                f"{rec['pieces']} writes, {rec['bytes']} bytes"
            )
            for problem in rec.get("problems", []):
                print("    " + problem)
    print(f"streams {len(streams)}, mismatches {mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
