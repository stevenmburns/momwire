"""Capture the BSplineSolver kwargs antennaknobs MomwireEngine builds for
verticals.elt_whip:sym4 at 406 MHz, free space (needs the AK branch
elt-whip-sym4 installed). Writes sym4_kwargs.pkl for mkfixture.py."""

import pickle

from antennaknobs.cli import get_builder
from antennaknobs.engines.momwire import MomwireEngine
from momwire import BSplineSolver

seen = {}
orig = BSplineSolver.__init__


def spy(self, *a, **kw):
    seen["kw"] = kw
    seen["a"] = a
    raise SystemExit("captured")


BSplineSolver.__init__ = spy
b = get_builder("verticals.elt_whip:sym4")()
b.freq = 406.0
eng = MomwireEngine(b, ground="free")
try:
    eng.impedance()
except SystemExit:
    pass
kw = seen["kw"]
for k, v in kw.items():
    if k == "wires":
        print(k, len(v))
        continue
    s = repr(v)
    print(k, s[:300])
pickle.dump(kw, open("sym4_kwargs.pkl", "wb"))
