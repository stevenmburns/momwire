"""max |dZ| / |Z| between the series and the direct rule, deck by deck
(momwire#1454 step 3).

    python dz.py OUT.jsonl [--jobs N] [--limit SECONDS] [--only SUBSTR]

Every deck of the corpus is rendered once through razor-2p with
``exact_kernel`` forced on. Each exact-kernel correction it runs is applied
twice to the same matrix, once by the direct rule and once by the series,
and the two results are compared entry by entry:

* ``dz``      max |Z_series - Z_direct| over max |Z| (the matrix's scale);
* ``dz_el``   max over entries of |Z_series - Z_direct| / |Z_direct|, with
              the size of that entry against the matrix's largest: the
              worst relative change sits on an entry that is itself a
              cancelled remainder of the two potentials.

Z here is the free-space block with the correction applied, the last point
at which the two routes differ. Also recorded: how many groups took each
route, and how many unique pairs took each series rule.
"""

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import types
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")


ROUTES = (
    "series_groups",
    "direct_groups",
    "pairs_near",
    "pairs_centred",
    "pairs_plain_far",
)


class _Limit(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Limit


def measure(render, rec):
    """Run `render()` with every correction applied both ways; fill `rec`."""
    import numpy as np

    from momwire import (
        RazorSolver,
        _exact_kernel as EK,
        _exact_kernel_series as ES,
        _razor_exact_plan as P,
    )

    orig = RazorSolver._add_exact_kernel_correction
    made, direct_groups = [], [0]
    Series, point_rows = ES.PointSeriesAccel, EK.point_rows

    class Recorded(Series):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    def counted_rows(*a, **k):
        if ES.ENABLED:
            direct_groups[0] += 1
        return point_rows(*a, **k)

    def both(self, Z, geom, k, omega):
        Z0 = Z.copy(order="F")
        ES.ENABLED = False
        orig(self, Z, geom, k, omega)
        ES.ENABLED = True
        orig(self, Z0, geom, k, omega)
        ES.ENABLED = False
        d = np.abs(Z0 - Z)
        rec["corrections"] += 1
        rec["dz"] = max(rec["dz"], float(d.max() / np.abs(Z).max()))
        az = np.abs(Z)
        el = np.where(az > 0, d / np.where(az > 0, az, 1.0), 0.0)
        if el.max() >= rec["dz_el"]:
            rec["dz_el"] = float(el.max())
            # how small the entry it happens on is, against the matrix
            rec["dz_el_entry"] = float(az.flat[el.argmax()] / az.max())

    rec.update(corrections=0, dz=0.0, dz_el=0.0)
    ES.PointSeriesAccel = Recorded
    EK.point_rows = counted_rows
    RazorSolver._add_exact_kernel_correction = both
    P.clear()
    try:
        render()
    finally:
        RazorSolver._add_exact_kernel_correction = orig
        ES.PointSeriesAccel, EK.point_rows = Series, point_rows
        ES.ENABLED = False
    kinds = np.sum([s.kinds for s in made], axis=0) if made else np.zeros(3, int)
    rec["series_groups"] = len(made)
    rec["direct_groups"] = direct_groups[0]
    rec["pairs_near"], rec["pairs_centred"], rec["pairs_plain_far"] = map(int, kinds)


def _work(args):
    source, path, digest, limit = args
    warnings.simplefilter("ignore")
    from momwire.deck import _solver
    from momwire.eznec import _shell

    bases = dict(_solver.BASES)
    cls, kw = bases["razor-2p"]
    bases["razor-2p"] = (cls, types.MappingProxyType({**kw, "exact_kernel": True}))
    _solver.BASES = types.MappingProxyType(bases)
    text = Path(path).read_bytes().decode("latin-1")
    rec = {"source": source, "path": str(path), "deck": digest}
    signal.signal(signal.SIGALRM, _alarm)
    try:
        signal.alarm(int(limit))
        measure(lambda: _shell.render(text, basis="razor-2p"), rec)
        signal.alarm(0)
    except _Limit:
        rec["skipped"] = f"over {limit} s"
    except Exception as exc:  # noqa: BLE001 -- a refused deck is recorded, not hidden
        signal.alarm(0)
        rec["raised"] = f"{type(exc).__name__}: {exc}"[:200]
        for key in ROUTES:
            rec.setdefault(key, 0)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--only", default=None)
    ns = ap.parse_args()
    from corpus import decks

    found, absent = decks()
    if ns.only:
        found = [d for d in found if ns.only in str(d[1])]
    jobs = [(s, str(p), d, ns.limit) for s, p, d in found]
    recs = []
    with mp.get_context("spawn").Pool(ns.jobs, maxtasksperchild=1) as pool:
        with open(ns.out, "w") as fh:
            for rec in pool.imap(_work, jobs):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                recs.append(rec)
    used = [r for r in recs if "skipped" not in r]
    eng = [r for r in used if r["corrections"]]
    print(
        f"decks {len(recs)} (absent sources {absent}); skipped {len(recs) - len(used)}"
    )
    print("decks whose render raised:", sum(1 for r in used if "raised" in r))
    print(
        f"decks taking the correction {len(eng)}, corrections {sum(r['corrections'] for r in eng)}"
    )
    for key in ("dz", "dz_el"):
        w = max(eng, key=lambda r: r[key])
        print(f"max {key}: {w[key]:.3e}  in {w['path']}")
    w = max(eng, key=lambda r: r["dz_el"])
    print(f"  (that entry is {w['dz_el_entry']:.1e} of its matrix's largest)")
    tot = {
        k: sum(r[k] for r in eng)
        for k in (
            "series_groups",
            "direct_groups",
            "pairs_near",
            "pairs_centred",
            "pairs_plain_far",
        )
    }
    print(f"routes: {tot}")
    print(
        "decks with a group on the direct rule:",
        sum(1 for r in eng if r["direct_groups"]),
    )


if __name__ == "__main__":
    main()
