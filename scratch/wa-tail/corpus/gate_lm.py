"""Render every finite-ground deck (corpus + fixtures + harvest GN 0/2) with razor-2p and
bspline in THIS tree, cold Sommerfeld cache; record printout text, full-precision Z of
every run (port state), and the values of every Sommerfeld table the renders filled.
argv: out_prefix  (writes out_prefix.jsonl and out_prefix.npz)"""

import hashlib
import json
import os
import re
import sys
import warnings

os.environ["MOMWIRE_SOMM_CACHE"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.simplefilter("ignore")
from multiprocessing import Pool  # noqa: E402 — the thread env above must precede numpy

import numpy as np  # noqa: E402 — OpenBLAS reads the thread env at load

S = os.path.dirname(os.path.abspath(__file__))


def decks():
    out = []
    for p in open(os.path.join(S, "somm_decks.txt")).read().split("\n"):
        if p:
            out.append((p, open(p, encoding="latin-1").read()))
    for i, l in enumerate(open(os.path.expanduser("~/harvest-2a.jsonl"))):
        t = json.loads(l)
        if any(re.match(r"\s*GN[\s,]+[02]\b", x.upper()) for x in t.splitlines()):
            out.append((f"harvest#{i}", t))
    return out


def work(item):
    name, text, basis = item
    from momwire import _sommerfeld as s, _sommerfeld_below as sb
    from momwire.eznec import _shell, _serve

    grids = []
    og, ogb = s.get_grid, sb.get_grid_below

    def g1(*a, **k):
        g = og(*a, **k)
        grids.append(("above", repr((a, sorted(k.items()))), g))
        return g

    def g2(*a, **k):
        g = ogb(*a, **k)
        grids.append(("below", repr((a, sorted(k.items()))), g))
        return g

    s.get_grid, sb.get_grid_below = g1, g2
    import momwire.razor as rz

    for nm in dir(rz):
        if getattr(rz, nm, None) is og:
            setattr(rz, nm, g1)
    caps = []
    ps = _serve._port_state

    def cap(*a, **k):
        st = ps(*a, **k)
        try:
            caps.append(
                [
                    [complex(v).real, complex(v).imag]
                    for v in (st.v_applied / st.i_source)
                ]
            )
        except Exception:  # noqa: BLE001 — a run without a port state records None
            caps.append(None)
        return st

    _serve._port_state = cap
    s._GRID_CACHE.clear()
    try:
        sb.reset_sweep_state()
    except Exception:  # noqa: BLE001 — older trees have no sweep state
        pass
    try:
        txt = _shell.render(text, basis=basis)
        err = None
    except Exception as e:  # noqa: BLE001 — a refused deck is a recorded result
        txt, err = "", repr(e)[:300]
    arrays = {}
    for fam, key, g in grids:
        regs = getattr(g, "_regions", [])
        for ri, reg in enumerate(regs):
            if reg.get("filled", True) and reg.get("vals") is not None:
                k = (
                    f"{fam}-r{ri}-"
                    + hashlib.sha1(f"{fam}|{key}|{ri}".encode()).hexdigest()[:16]
                )
                arrays[k] = np.asarray(reg["vals"]).copy()
    s.get_grid, sb.get_grid_below = og, ogb
    _serve._port_state = ps
    return dict(
        name=name,
        basis=basis,
        sha=hashlib.sha256(txt.encode("latin-1", "replace")).hexdigest(),
        text=txt,
        Z=caps,
        err=err,
        grids=sorted(arrays),
    ), arrays


if __name__ == "__main__":
    prefix = sys.argv[1]
    items = [(n, t, b) for n, t in decks() for b in ("razor-2p", "bspline")]
    allarr = {}
    with (
        Pool(int(os.environ.get("GATE_JOBS", "8")), maxtasksperchild=50) as pool,
        open(prefix + ".jsonl", "w") as f,
    ):
        for rec, arrays in pool.imap_unordered(work, items, chunksize=2):
            f.write(json.dumps(rec) + "\n")
            for k, v in arrays.items():
                allarr.setdefault(k, v)
    np.savez(prefix + ".npz", **allarr)
    print(f"{len(items)} renders, {len(allarr)} table regions")
