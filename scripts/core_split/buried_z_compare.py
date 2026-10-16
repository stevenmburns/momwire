"""razor-2p's BURIED fill in C++ (momwire#1512 step 3) against the Python
route, stage by stage, deck by deck.

    python scripts/core_split/buried_z_compare.py OUT.jsonl DECK_OR_DIR...

Each deck goes through the Python launcher's own route (`_shell.render(text,
basis="razor-2p")`, the C++ route forced off), with `RazorSolver.__init__`'s
keyword arguments recorded and the matrix snapshotted at the buried
assembly's stage boundaries (`razor._assemble_Z_crossing` and
`_assemble_Z_below_plane`):

* ``above``  -- crossing / detached: Z after the above medium's block is
  scattered into the zero target (`(0 + Z_a)`);
* ``below``  -- after the below medium's window folds;
* ``cross``  -- after both cross blocks and the node charges (what
  `_assemble_Z_crossing` returns, before the loading);
* ``final``  -- the matrix the LU factors (`_assemble_Z_from_prepared`'s
  answer, loading included); a wholly-below deck has only this one.

The C++ side (`_accelerators.razor_buried_stages(kwargs)`) builds the same
solver's front half from the recorded arguments and fills the same stages,
as far as the port reaches; every stage it answers must be Python's, BYTE
for byte. A stage it does not reach yet is counted, never compared. One JSON
line per solve; a summary on stdout; exit status 1 on any mismatch.

`MOMWIRE_BELOW_ANCHORS` is whatever the environment says, on both sides
(the drivers read Python's below tables through the test bridge until the
table itself is ported, `MW_TEST_BELOW_TABLES`).
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

STAGES = ("above", "below", "cross", "final")


def _decks(args):
    out = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            out += sorted(
                q for q in p.rglob("*") if q.is_file() and q.suffix in (".nec", ".deck")
            )
        else:
            out.append(p)
    return out


def python_stages(text):
    """``[dict(kwargs, stages={name: Z})]`` per RazorSolver the Python route
    solved, in order."""
    from momwire import RazorSolver, razor
    from momwire.eznec import _shell

    solves = []
    current = {}
    orig_init = RazorSolver.__init__
    orig_below = RazorSolver._assemble_Z_below_plane
    orig_cross = RazorSolver._assemble_Z_crossing
    orig_from = RazorSolver._assemble_Z_from_prepared

    def init(self, **kw):
        current.clear()
        current["kwargs"] = dict(kw)
        current["stages"] = {}
        solves.append(current.copy())
        solves[-1]["stages"] = current["stages"]
        return orig_init(self, **kw)

    def below(self, geom, prepared, k, omega, **kw):
        into = kw.get("into")
        if into is not None:
            current["stages"]["above"] = np.array(into[0], copy=True, order="F")
        out = orig_below(self, geom, prepared, k, omega, **kw)
        if into is not None:
            current["stages"]["below"] = np.array(into[0], copy=True, order="F")
        return out

    def cross(self, geom, k, omega, **kw):
        z = orig_cross(self, geom, k, omega, **kw)
        current["stages"]["cross"] = np.array(z, copy=True, order="F")
        return z

    def from_prepared(self, geom, prepared, k, omega, **kw):
        z = orig_from(self, geom, prepared, k, omega, **kw)
        current["stages"]["final"] = np.array(z, copy=True, order="F")
        return z

    saved = razor._RAZOR_CXX
    razor._RAZOR_CXX = False
    RazorSolver.__init__ = init
    RazorSolver._assemble_Z_below_plane = below
    RazorSolver._assemble_Z_crossing = cross
    RazorSolver._assemble_Z_from_prepared = from_prepared
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _shell.render(text, basis="razor-2p")
    except Exception:  # noqa: BLE001 -- a raising deck is the seam gate's, not this one's
        pass
    finally:
        razor._RAZOR_CXX = saved
        RazorSolver.__init__ = orig_init
        RazorSolver._assemble_Z_below_plane = orig_below
        RazorSolver._assemble_Z_crossing = orig_cross
        RazorSolver._assemble_Z_from_prepared = orig_from
    return [s for s in solves if s["stages"]]


def main(argv):
    from momwire import _accel

    acc = _accel.acc
    out_path = Path(argv[1])
    decks = _decks(argv[2:])
    tally = Counter()
    bad = 0
    with out_path.open("w") as out:
        for deck in decks:
            text = deck.read_text(encoding="latin-1")
            for k, py in enumerate(python_stages(text)):
                row = {"deck": str(deck), "solve": k}
                got = acc.razor_buried_stages(py["kwargs"])
                row["route"] = got.get("route")
                row["reached"] = []
                mism = []
                for name in STAGES:
                    if name not in py["stages"]:
                        continue
                    if name not in got:
                        tally[f"unreached:{name}"] += 1
                        continue
                    row["reached"].append(name)
                    a = np.asfortranarray(py["stages"][name])
                    b = np.asfortranarray(got[name])
                    if a.shape != b.shape or a.tobytes() != b.tobytes():
                        d = np.abs(a - b) if a.shape == b.shape else None
                        mism.append(
                            {
                                "stage": name,
                                "shape": [list(a.shape), list(b.shape)],
                                "n_diff": None if d is None else int(np.count_nonzero(a != b)),
                                "max_rel": None
                                if d is None
                                else float(d.max() / max(np.abs(a).max(), 1e-300)),
                            }
                        )
                    tally[f"compared:{name}"] += 1
                row["mismatches"] = mism
                if mism:
                    bad += 1
                    print(f"MISMATCH {deck} solve {k}: {mism}", flush=True)
                out.write(json.dumps(row) + "\n")
                out.flush()
    print(f"decks {len(decks)}")
    for key in sorted(tally):
        print(f"  {key}: {tally[key]}")
    print(f"mismatches {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
