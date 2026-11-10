"""Write antennaknobs verticals.elt_whip:sym4 (branch elt-whip-sym4, 71ada770)
at 406 MHz, free space, as the BSplineSolver kwargs MomwireEngine builds."""

import gzip
import json
import pickle
import sys


kw = pickle.load(open("sym4_kwargs.pkl", "rb"))
doc = {
    "source": "antennaknobs verticals.elt_whip:sym4 (elt-whip-sym4 @ 71ada770), "
    "MomwireEngine(ground=free) BSplineSolver kwargs at 406 MHz",
    "wires": [[[float(x) for x in v] for v in w] for w in kw["wires"]],
    "n_per_edge_per_wire": [[int(n) for n in e] for e in kw["n_per_edge_per_wire"]],
    "wire_radius": [float(r) for r in kw["wire_radius"]],
    "feeds": [
        [int(w), float(a), [complex(v).real, complex(v).imag]]
        for w, a, v in kw["feeds"]
    ],
    "junctions": [[[int(w), str(e)] for w, e in j] for j in kw["junctions"]],
    "wavelength": float(kw["wavelength"]),
    "post_c_farad": 0.9e-12,
}
others = sorted(
    set(kw)
    - {
        "wires",
        "n_per_edge_per_wire",
        "wire_radius",
        "feeds",
        "junctions",
        "wavelength",
    }
)
print("other kwargs", {k: kw[k] for k in others})
with gzip.open(sys.argv[1], "wt") as f:
    json.dump(doc, f, separators=(",", ":"))
