"""Capture the BSplineSolver kwargs antennaknobs' MomwireEngine builds for
verticals.elt_whip:hex6 at 406 MHz, free space, and write them in the
#1897 fixture's format (geometry only) to argv[1] (.json.gz)."""

import gzip
import json
import sys

from antennaknobs.cli import get_builder
from antennaknobs.engines.momwire import MomwireEngine
from momwire import BSplineSolver

seen = {}


def spy(self, *a, **kw):
    seen["kw"] = kw
    raise SystemExit("captured")


BSplineSolver.__init__ = spy
b = get_builder("verticals.elt_whip:hex6")()
b.freq = 406.0
try:
    MomwireEngine(b, ground="free").impedance()
except SystemExit:
    pass
kw = seen["kw"]
print(
    {
        k: (
            v
            if k not in ("wires", "n_per_edge_per_wire", "wire_radius", "junctions")
            else len(v)
        )
        for k, v in kw.items()
        if k != "feeds"
    }
)
print("feeds", kw["feeds"])
doc = {
    "source": "antennaknobs verticals.elt_whip:hex6, MomwireEngine(ground=free) "
    "BSplineSolver kwargs at 406 MHz",
    "wires": [[[float(x) for x in v] for v in w] for w in kw["wires"]],
    "n_per_edge_per_wire": [[int(n) for n in e] for e in kw["n_per_edge_per_wire"]],
    "wire_radius": [float(r) for r in kw["wire_radius"]],
    "feeds": [
        [int(w), float(a), [complex(v).real, complex(v).imag]]
        for w, a, v in kw["feeds"]
    ],
    "junctions": [[[int(w), str(e)] for w, e in j] for j in kw["junctions"]],
    "wavelength": float(kw["wavelength"]),
}
with gzip.open(sys.argv[1], "wt") as f:
    json.dump(doc, f, separators=(",", ":"))
