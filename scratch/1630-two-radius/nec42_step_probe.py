"""Is NEC-4.2's two-radius failure the INTERFACE or the 19:1 radius STEP?
The 1-radial hub deck (19 mm vertical, 1 mm drop + radial), main radii, mid
feed, r = 1..8: (a) as built over GN 2; (b) lifted 10 m into free space
(GM), no ground: a radius-step junction with no interface; (c) the step
ratio swept in free space at r = 4. Black box. Usage: OUT.jsonl"""

from __future__ import annotations

import json
import sys

import decks
from ref_ladder import NEC5, NEC42, run


def deck(r, a_v, a_b, eng, free):
    ws = decks.wires("hub", r, a_v=a_v, a_b=a_b, n_rad=1)
    lines = ["CM step probe", "CE"] + [decks._gw(*w) for w in ws]
    if free:
        lines += ["GM 0 0 0 0 0 0 0 10", "GE 0" if eng == "nec42" else "GE 0 0"]
    elif eng == "nec42":
        lines += ["GE -1", "GN 2 0 0 0 13 0.005 NOFILE"]
    else:
        lines += ["GE -1 0", "GN 0 0 0 0 13 0.005 1.0 0.0 NOFILE"]
    ex = f"EX 0 1 {2 * r + 1} 0 1 0" if eng == "nec42" else f"EX 0 1 {2 * r} 2 1 0"
    lines += [ex, f"FR 0 1 0 0 {decks.FREQ_MHZ} 0", "XQ 0", "EN"]
    return "\n".join(lines) + "\n"


with open(sys.argv[1], "a") as fh:
    jobs = [("free", r, 0.019, 0.001) for r in (1, 2, 4, 8)]
    jobs += [
        ("free_ratio", 4, a_v, 0.001) for a_v in (0.001, 0.002, 0.004, 0.008, 0.019)
    ]
    if len(sys.argv) > 2 and sys.argv[2] == "ground":
        jobs = [
            ("gnd_ratio", 4, a_v, 0.001)
            for a_v in (0.001, 0.0011, 0.0015, 0.002, 0.004, 0.019)
        ]
    for name, r, a_v, a_b in jobs:
        for eng, exe in (("nec42", NEC42), ("nec5", NEC5)):
            (R, X), _ = run(
                exe, deck(r, a_v, a_b, eng, free=not name.startswith("gnd"))
            )
            fh.write(
                json.dumps(dict(case=name, r=r, a_v=a_v, engine=eng, R=R, X=X)) + "\n"
            )
            fh.flush()
