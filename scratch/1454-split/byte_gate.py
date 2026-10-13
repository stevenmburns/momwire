"""Render the corpus through razor-2p and keep every printout (momwire#1454
step 3's byte gate).

    python byte_gate.py OUTDIR --mode {auto,true} [--series {on,off}]
                        [--jobs N] [--limit SECONDS] [--basis NAME]

Each deck is rendered in both EZNEC dialects, twice in one process (rep 0
is the cold call, rep 1 the repeat that meets the warm caches), with
``exact_kernel`` bound as the roster binds it (``auto``) or forced on
(``true``). Only lines containing ``TIME`` are masked. One JSON line per deck
goes to ``OUTDIR/index.jsonl`` with the sha256 of every masked printout, how
many exact-kernel corrections ran (a gate over decks that never take the
correction measures nothing) and, where the tree has them, the series
counters; the masked printouts themselves go to ``OUTDIR/<sha>.txt``.

Run it once per tree (``PYTHONPATH=<tree>/src``) and compare the two
directories with ``compare.py``. ``--series`` is read by a tree that has the
static/dynamic split and ignored by one that does not, so the same command
line serves base and change.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import types
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _mask(text):
    return "\n".join(x for x in text.splitlines() if "TIME" not in x)


class _Limit(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Limit


def _work(args):
    source, path, digest, mode, series, limit, outdir, basis = args
    warnings.simplefilter("ignore")
    import momwire
    from momwire import RazorSolver, _razor_exact_plan as P
    from momwire.deck import _solver
    from momwire.eznec import _shell

    try:
        from momwire import _exact_kernel_series as ES
    except ImportError:
        ES = None
    if ES is not None:
        ES.ENABLED = series == "on"
    if mode == "true":
        bases = dict(_solver.BASES)
        cls, kw = bases[basis]
        bases[basis] = (cls, types.MappingProxyType({**kw, "exact_kernel": True}))
        _solver.BASES = types.MappingProxyType(bases)
    calls = [0]
    orig = RazorSolver._add_exact_kernel_correction

    def counted(self, *a, **k):
        calls[0] += 1
        return orig(self, *a, **k)

    RazorSolver._add_exact_kernel_correction = counted
    text = Path(path).read_bytes().decode("latin-1")
    rec = {
        "source": source,
        "path": str(path),
        "deck": digest,
        "tree": str(Path(momwire.__file__).resolve().parents[2]),
        "prints": {},
        "corrections": 0,
    }
    P.clear()
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.perf_counter()
    try:
        signal.alarm(int(limit))
        for dialect in ("nec5", "nec4"):
            for rep in (0, 1):
                try:
                    out = _mask(_shell.render(text, basis=basis, dialect=dialect))
                except _Limit:
                    raise
                except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
                    out = f"RAISED {type(exc).__name__}: {exc}"
                sha = hashlib.sha256(out.encode("utf-8", "replace")).hexdigest()
                rec["prints"][f"{dialect}/{rep}"] = sha
                f = Path(outdir) / f"{sha}.txt"
                if not f.exists():
                    f.write_text(out, encoding="utf-8", errors="replace")
        signal.alarm(0)
    except _Limit:
        rec["skipped"] = f"over {limit} s"
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["corrections"] = calls[0]
    rec["stats"] = dict(P.STATS)
    rec["series"] = dict(getattr(P, "SERIES_STATS", {}))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--mode", choices=("auto", "true"), required=True)
    ap.add_argument("--series", choices=("on", "off"), default="on")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--only", default=None, help="substring a deck path must contain")
    ap.add_argument("--basis", default="razor-2p", help="roster name (bspline is bs2)")
    ns = ap.parse_args()
    from corpus import decks

    found, absent = decks()
    if ns.only:
        found = [d for d in found if ns.only in str(d[1])]
    out = Path(ns.outdir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (s, str(p), d, ns.mode, ns.series, ns.limit, str(out), ns.basis)
        for s, p, d in found
    ]
    with mp.get_context("spawn").Pool(ns.jobs, maxtasksperchild=1) as pool:
        with open(out / "index.jsonl", "w") as fh:
            for rec in pool.imap(_work, jobs):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
    print(f"{len(jobs)} decks -> {out}  (absent sources: {absent})")


if __name__ == "__main__":
    main()
