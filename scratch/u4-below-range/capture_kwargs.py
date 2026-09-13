"""Capture the solver kwargs antennaknobs' momwire engine builds for the U4
synthetic deck, stopping BEFORE the solver's own __init__ runs (no fill, no Z)."""

import json
import sys
from pathlib import Path

import numpy as np

from momwire import _sommerfeld_below as below

below._SOMM_BELOW_R1_CAP_LAMBDA_M = 5.0  # past the construction-time preflight
from momwire.bspline import BSplineSolver  # noqa: E402 - imported after the cap patch, deliberately


class Stop(Exception):
    pass


calls = []


def init(self, *a, **kw):
    calls.append((type(self).__name__, a, kw))
    raise Stop


BSplineSolver.__init__ = init
from antennaknobs.cli import _GROUND_UNSET, file_ground_default, make_engine_factory  # noqa: E402 - imported after the cap patch, deliberately
from antennaknobs.file_designs import builder_from_file  # noqa: E402 - imported after the cap patch, deliberately

deck = Path(sys.argv[1])
refine = int(sys.argv[2])
b = builder_from_file(str(deck), refine=refine)
try:
    eng = make_engine_factory("momwire", file_ground_default(_GROUND_UNSET, b))
    eng(b()).impedance()
    print("NO STOP: the solver was never constructed")
except Stop:
    pass


def enc(v):
    if isinstance(v, np.ndarray):
        return enc(v.tolist())
    if isinstance(v, complex):
        return {"re": v.real, "im": v.imag}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        return {k: enc(x) for k, x in v.items()}
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


name, a, kw = calls[0]
print("class", name, "positional", len(a), "calls", len(calls))
for k, v in kw.items():
    s = repr(v)
    print(f"  {k}: {s[:200]}")
Path(sys.argv[3]).write_text(json.dumps(dict(cls=name, kwargs=enc(kw)), indent=1))
