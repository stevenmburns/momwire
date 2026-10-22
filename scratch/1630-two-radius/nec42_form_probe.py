"""Which deck feature makes NEC-4.2 non-convergent on the #1630 hub deck?
One-radius (1 mm) hub deck, mid feed, r = 1..8, varying one factor at a time.
Black box: deck in, Z out. Usage: python nec42_form_probe.py OUT.jsonl"""

from __future__ import annotations

import json
import sys

import decks
from ref_ladder import NEC5, NEC42, run


def deck(r, depth, n_rad, drop_segs, L_rad, a=0.001, eng="nec42"):
    decks.DEPTH = depth
    decks.L_RAD = L_rad
    ws = decks.wires("hub", r, a_v=a, a_b=a, n_rad=n_rad)
    ws[1] = (ws[1][0], drop_segs(r), *ws[1][2:])
    lines = ["CM probe", "CE"] + [decks._gw(*w) for w in ws]
    if eng == "nec42":
        lines += ["GE -1", "GN 2 0 0 0 13 0.005 NOFILE", f"EX 0 1 {2 * r + 1} 0 1 0"]
    else:
        lines += [
            "GE -1 0",
            "GN 0 0 0 0 13 0.005 1.0 0.0 NOFILE",
            f"EX 0 1 {2 * r} 2 1 0",
        ]
    lines += [f"FR 0 1 0 0 {decks.FREQ_MHZ} 0", "XQ 0", "EN"]
    return "\n".join(lines) + "\n"


CASES = {
    "base": dict(depth=0.0254, n_rad=16, drop_segs=lambda r: r, L_rad=4.0),
    "deep15": dict(depth=0.15, n_rad=16, drop_segs=lambda r: r, L_rad=4.0),
    "rad4": dict(depth=0.0254, n_rad=4, drop_segs=lambda r: r, L_rad=4.0),
    "rad1": dict(depth=0.0254, n_rad=1, drop_segs=lambda r: r, L_rad=4.0),
    "drop1": dict(depth=0.0254, n_rad=16, drop_segs=lambda r: 1, L_rad=4.0),
    "deep15_rad1": dict(depth=0.15, n_rad=1, drop_segs=lambda r: r, L_rad=4.0),
    "norad": dict(depth=0.0254, n_rad=0, drop_segs=lambda r: r, L_rad=4.0),
    "norad_deep15": dict(depth=0.15, n_rad=0, drop_segs=lambda r: r, L_rad=4.0),
}

with open(sys.argv[1], "a") as fh:
    for name, kw in CASES.items():
        for r in (1, 2, 4, 8):
            for eng, exe in (("nec42", NEC42), ("nec5", NEC5)):
                try:
                    (R, X), dt = run(exe, deck(r, eng=eng, **kw))
                    rec = dict(R=R, X=X)
                except Exception as e:  # noqa: BLE001 - recorded, run goes on
                    rec = dict(error=repr(e)[:200])
                rec.update(case=name, r=r, engine=eng)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
