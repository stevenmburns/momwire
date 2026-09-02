"""#762 protocol: spy on _crossing_fill's entry points during a bspline solve.

Records every array the fill's entry points return, for the crossing deck at
soil A and at eps~ = 1, so the same run before and after a change to the
module can be compared with `array_equal`. Usage:

    python spy762.py <out.npz>
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import BSplineSolver
from momwire import _crossing_fill as CF
from test_crossing_serve_524 import crossing_deck, fan_rise_deck
from test_buried_serve_553 import SOIL_A

REC = {}


def _flatten(v):
    """Every complex scalar in a nested impedance return, in one flat list."""
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(_flatten(x))
        return out
    arr = np.asarray(v)
    if arr.dtype == object:
        return _flatten(list(arr))
    return [complex(x) for x in np.atleast_1d(arr).ravel()]


def wrap(name):
    real = getattr(CF, name)

    def spy(*a, **kw):
        out = real(*a, **kw)
        if isinstance(out, tuple):
            for i, x in enumerate(out):
                REC[f"{name}#{len(REC)}.{i}"] = np.asarray(x)
        elif isinstance(out, dict):
            for kk, v in out.items():
                if isinstance(v, np.ndarray):
                    REC[f"{name}#{len(REC)}.{kk}"] = v
        else:
            REC[f"{name}#{len(REC)}"] = np.asarray(out)
        return out

    setattr(CF, name, spy)


def main(out_path):
    for n in (
        "axis_data",
        "cross_complete_block",
        "cross_complete_block_split",
        "self_completions",
    ):
        wrap(n)
    decks = [
        ("crossing", lambda: dict(crossing_deck(1))),
        ("fan2", lambda: dict(fan_rise_deck(n_radials=2))),
    ]
    for dname, mk in decks:
        for label, ge in (("soilA", SOIL_A), ("eps1", (1.0, 0.0))):
            deck = mk()
            deck["ground_eps"] = ge
            s = BSplineSolver(**deck)
            z = s.compute_impedance()
            REC[f"Z:{dname}:{label}"] = np.array(
                sorted(_flatten(z), key=lambda c: (c.real, c.imag))
            )
    np.savez(out_path, **REC)
    print(f"{len(REC)} arrays -> {out_path}")


main(sys.argv[1])
