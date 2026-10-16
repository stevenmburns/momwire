"""The near-interface kernels before and after a move (momwire#1512 B4a).

    python scripts/core_split/near_core_hash.py record DIR DECK_OR_DIR...
    python scripts/core_split/near_core_hash.py replay DIR

`record` renders each deck with razor-2p (the Python route) and saves the
arguments and answer of up to 4 calls per deck of each of
`near_interface_six_columns` (the column cache dropped: it is value-neutral,
and the replay runs without it as well as with a fresh one) and
`near_interface_grid_sheet`. `replay` calls the current build on each and
compares the answers byte for byte, at 1 thread and at the recorded count.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np


def _decks(args):
    out = []
    for arg in args:
        p = Path(arg)
        out += sorted(p.rglob("*.nec")) if p.is_dir() else [p]
    return out


def record(out_dir, decks, per=4):
    from momwire import _near_interface, razor
    from momwire.eznec import _shell

    nia = _near_interface._nia
    out_dir.mkdir(parents=True, exist_ok=True)
    origs = {name: getattr(nia, name) for name in ("near_interface_six_columns", "near_interface_grid_sheet")}
    count = {"n": 0}
    seen = {}

    def wrap(name):
        orig = origs[name]

        def f(*args, **kw):
            if name == "near_interface_grid_sheet":
                out = args[10]
                before = np.array(out, copy=True)
                res = orig(*args, **kw)
                if seen.get(name, 0) < per:
                    seen[name] = seen.get(name, 0) + 1
                    a = list(args)
                    a[10] = before
                    pickle.dump(
                        (name, a, kw, np.array(out, copy=True)),
                        open(out_dir / f"call{count['n']:05d}.pkl", "wb"),
                    )
                    count["n"] += 1
                return res
            res = orig(*args, **kw)
            if seen.get(name, 0) < per:
                seen[name] = seen.get(name, 0) + 1
                kw2 = dict(kw)
                kw2.pop("cache", None)
                kw2.pop("cancel_flag", None)
                pickle.dump(
                    (name, list(args), kw2, np.array(res, copy=True)),
                    open(out_dir / f"call{count['n']:05d}.pkl", "wb"),
                )
                count["n"] += 1
            return res

        return f

    for name in origs:
        setattr(nia, name, wrap(name))
    saved = razor._RAZOR_CXX
    razor._RAZOR_CXX = False
    try:
        for d in decks:
            seen.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _shell.render(d.read_text(encoding="latin-1"), basis="razor-2p")
                except Exception:  # noqa: BLE001
                    pass
    finally:
        for name, f in origs.items():
            setattr(nia, name, f)
        razor._RAZOR_CXX = saved
    print(f"recorded {count['n']} calls")


def replay(out_dir):
    from momwire import _near_interface

    nia = _near_interface._nia
    bad = n = 0
    for f in sorted(out_dir.glob("call*.pkl")):
        name, args, kw, ref = pickle.load(open(f, "rb"))
        fn = getattr(nia, name)
        variants = [{}, {"n_threads": 1}] if name == "near_interface_six_columns" else [{}]
        for v in variants:
            a = list(args)
            if name == "near_interface_six_columns":
                if "n_threads" in v:
                    a[9] = 1
                for cache in (None, nia.ColumnCache(16 << 20)):
                    got = fn(*a, **kw, cache=cache)
                    n += 1
                    if np.asarray(got).tobytes() != ref.tobytes():
                        bad += 1
                        print(f"DIFFER {f.name} {name} {v} cache={cache is not None}")
            else:
                out = np.array(a[10], copy=True)
                a[10] = out
                fn(*a, **kw)
                n += 1
                if out.tobytes() != ref.tobytes():
                    bad += 1
                    print(f"DIFFER {f.name} {name}")
    print(f"replayed {n} answers, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(Path(sys.argv[2]), _decks(sys.argv[3:]))
    else:
        sys.exit(replay(Path(sys.argv[2])))
