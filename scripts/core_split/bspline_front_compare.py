"""PROTOTYPE (momwire#1517 step 1): the C++ bspline front half
(`_bspline_front_core.cpp`), unit by unit, against `BSplineSolver`'s own.

    python scripts/core_split/bspline_front_compare.py DECK_OR_DIR...

For every deck the C++ seam prepares for bspline, `BSplineSolver` is built
from the seam's own keyword arguments (no fill) and each ported unit's
arrays are compared byte for byte (`front_compare.differences`). Units:
`_wire_spec.drop_duplicated_wires` (the dedup and its refusals) and
`_build_geometry`. Exit 1 on any mismatch.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import front_compare as fc
from seam_compare import _decks


def _py(value):
    """numpy arrays as arrays, lists of ints as lists, scalars as Python."""
    import numpy as np

    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_py(v) for v in value]
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.float64)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def main():
    from momwire import BSplineSolver, _accel

    acc = _accel.acc
    if acc is None or not hasattr(acc, "bspline_front_geometry"):
        sys.exit("no accelerator with the bspline front prototype loaded")
    warnings.simplefilter("ignore")
    counts, mismatches = Counter(), 0
    for path in _decks(sys.argv[1:]):
        text = path.read_bytes().decode("latin-1")
        for k, entry in enumerate(acc.bspline_front_geometry(text.encode("latin-1"))):
            try:
                solver = BSplineSolver(**entry["kwargs"])
                geom = solver._build_geometry()
            except Exception as exc:  # noqa: BLE001 -- counted by type
                want = (type(exc).__name__, str(exc))
                if "raised" in entry:
                    counts["raised-alike"] += 1
                    if tuple(entry["raised"]) != want:
                        mismatches += 1
                        print(f"MISMATCH {path}[{k}]: {entry['raised']} vs {want}")
                else:
                    counts[f"python-raised:{type(exc).__name__}"] += 1
                continue
            if "raised" in entry or "declined" in entry:
                mismatches += 1
                print(
                    f"MISMATCH {path}[{k}]: C++ {entry.get('raised') or entry['declined']}"
                )
                continue
            dd = solver._wire_dedup
            want = (
                None if dd is None else (dd.n_input, tuple(dd.kept), tuple(dd.removed))
            )
            d = fc.differences(want, entry["dedup"], "dedup")
            counts["deduped"] += dd is not None
            d += fc.differences(_py(dict(geom)), entry["geom"], "geom")
            counts["compared"] += 1
            if d:
                mismatches += 1
                print(f"MISMATCH {path}[{k}]: {d[:4]}")
    print(dict(counts), "mismatches", mismatches)
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
