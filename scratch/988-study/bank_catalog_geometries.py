"""Regenerate tests/fixtures/catalog_geometries.json (momwire#988).

Captures `BSplineSolver(degree=2)` constructor kwargs for the antennaknobs
catalog designs the fallback tests use, WITHOUT solving: the spy raises as soon
as it has the kwargs, so a 4,000-wire deck costs a mesh build rather than a
98-second solve.

Run from the momwire tree with antennaknobs importable:

    PYTHONPATH=<momwire>/src <ak-venv>/bin/python \
        scratch/988-study/bank_catalog_geometries.py out.json

`verticals.elt_whip` is deliberately NOT banked: 4,007 wires and 8,074 vertices
is ~894 KB of JSON for its two rungs. It stays on the antennaknobs path and is
allowlisted in `tests/test_no_hidden_antennaknobs_skips_988.py`.
"""

import importlib
import json
import sys
import warnings

import numpy as np

from momwire import BSplineSolver

DESIGNS = [
    "arrays.folded_invveearray",
    "arrays.moxonarray",
    "beams.moxon",
    "beams.owa_yagi",
    "broadband.lpda",
    "dipoles.dipole_turnstile",
    "loops.diamond_loop_turnstile",
    "loops.horizontal_loop",
    "loops.skyloop_lmatch",
    "loops.triangular_skyloop",
    "wire.rhombic",
    "wire.sterba",
    "wire.sterba_bl",
]
GROUND = ("finite", 13.0, 0.005)


class _Grabbed(Exception):
    """Raised once the constructor kwargs are in hand, to skip the solve."""


def _clean(v):
    if isinstance(v, complex):
        return {"__c__": [v.real, v.imag]}
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def capture(design, mult):
    from antennaknobs.engines.momwire import MomwireEngine

    captured = {}
    original = BSplineSolver.__init__

    def spy(self, *a, **kw):
        captured.update(kw)
        raise _Grabbed

    builder = importlib.import_module(f"antennaknobs.designs.{design}").Builder()
    builder.nominal_nsegs = int(builder.nominal_nsegs) * mult
    BSplineSolver.__init__ = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MomwireEngine(
                builder,
                solver=BSplineSolver,
                solver_kwargs={"degree": 2},
                ground=GROUND,
            ).impedance()
    except _Grabbed:
        pass
    finally:
        BSplineSolver.__init__ = original
    return {k: _clean(v) for k, v in captured.items() if k != "cancel"}


def main():
    cells = {}
    for design in DESIGNS:
        for mult, rung in ((2, "default"), (4, "refined")):
            rec = capture(design, mult)
            cells[f"{design}|{rung}"] = rec
            print(
                f"{design:32s} {rung:8s} wires={len(rec['wires']):3d}",
                file=sys.stderr,
            )
    json.dump({"cells": cells}, open(sys.argv[1], "w"), indent=1, sort_keys=True)
    print(f"banked {len(cells)} cells", file=sys.stderr)


if __name__ == "__main__":
    main()
