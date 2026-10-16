"""The below/below projection kernel before and after a move (momwire#1512 B2).

    python scripts/core_split/below_proj_hash.py record DIR DECK_OR_DIR...
    python scripts/core_split/below_proj_hash.py replay DIR

`record` renders each deck with razor-2p (the Python route) and saves the
arguments and the answer of every `remainder_field_proj_batch_below` call
the fill makes (blocked and lanes as called), one .npz per call. `replay`
calls the CURRENT build's kernel on each saved argument set, the lanes off
and on and the per-pair loop too, and compares every answer with the
recorded one byte for byte. Run `record` on the build before the move and
`replay` on the build after it: a kernel moved verbatim answers identically.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np


def _decks(args):
    out = []
    for arg in args:
        p = Path(arg)
        out += (
            sorted(q for q in p.rglob("*.nec") if q.is_file()) if p.is_dir() else [p]
        )
    return out


def record(out_dir, decks, per_deck=3):
    from momwire import _accel, razor
    from momwire.eznec import _shell

    acc = _accel.acc
    orig = acc.remainder_field_proj_batch_below
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"n": 0, "deck": 0}

    def wrapped(*args, **kw):
        res = orig(*args, **kw)
        if state["deck"] < per_deck:
            state["deck"] += 1
            fixed = list(args[:15])
            regs = [np.asarray(a) for a in args[15:19]]
            vals = list(args[19])
            np.savez(
                out_dir / f"call{state['n']:05d}.npz",
                obs=args[0],
                t_obs=args[1],
                src=args[2],
                t_src=args[3],
                scal=np.array(fixed[4:7] + fixed[7:11], dtype=object),
                grid=np.array(fixed[11:15], dtype=object),
                r0=regs[0],
                dr=regs[1],
                th0=regs[2],
                dth=regs[3],
                n_vals=len(vals),
                **{f"v{i}": v for i, v in enumerate(vals)},
                out=res[0],
                ext=np.array(res[1:]),
                kw=np.array(sorted(kw.items()), dtype=object),
            )
            state["n"] += 1
        return res

    acc.remainder_field_proj_batch_below = wrapped
    saved = razor._RAZOR_CXX
    razor._RAZOR_CXX = False
    try:
        for d in decks:
            state["deck"] = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _shell.render(d.read_text(encoding="latin-1"), basis="razor-2p")
                except Exception:  # noqa: BLE001 -- a refusing deck records nothing
                    pass
    finally:
        acc.remainder_field_proj_batch_below = orig
        razor._RAZOR_CXX = saved
    print(f"recorded {state['n']} calls")


def replay(out_dir):
    from momwire import _accel

    acc = _accel.acc
    bad = n = 0
    for f in sorted(out_dir.glob("call*.npz")):
        z = np.load(f, allow_pickle=True)
        scal = list(z["scal"])
        grid = list(z["grid"])
        vals = [z[f"v{i}"] for i in range(int(z["n_vals"]))]
        regs = [z["r0"], z["dr"], z["th0"], z["dth"]]
        args = [z["obs"], z["t_obs"], z["src"], z["t_src"], *scal, *grid, *regs, vals]
        kw = dict(z["kw"].tolist()) if z["kw"].size else {}
        ref = z["out"].tobytes()
        for variant in (kw, {"lanes": False}, {"blocked": False}, {"lanes": True}):
            got = acc.remainder_field_proj_batch_below(*args, **variant)
            n += 1
            same = got[0].tobytes() == ref and np.array_equal(np.array(got[1:]), z["ext"])
            if not same:
                bad += 1
                print(f"DIFFER {f.name} {variant}")
    print(f"replayed {n} answers, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(Path(sys.argv[2]), _decks(sys.argv[3:]))
    else:
        sys.exit(replay(Path(sys.argv[2])))
