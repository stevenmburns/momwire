"""Follow-up to nec42_form_probe: is NEC-4.2's 16-radial failure the hub
junction itself (free space), the radial count, or the depth? One radius
(1 mm), mid feed. Usage: python nec42_form_probe2.py OUT.jsonl"""

from __future__ import annotations

import json
import sys

import decks
from ref_ladder import NEC5, NEC42, run


def deck(r, depth, n_rad, ground, eng):
    decks.DEPTH = depth
    ws = decks.wires("hub", r, a_v=0.001, a_b=0.001, n_rad=n_rad)
    lines = ["CM probe2", "CE"] + [decks._gw(*w) for w in ws]
    if not ground:
        # free space: shift the whole structure up so nothing is at z<0
        lines = (
            ["CM probe2", "CE"] + [decks._gw(*w) for w in ws] + ["GM 0 0 0 0 0 0 0 10"]
        )
        lines += ["GE 0" if eng == "nec42" else "GE 0 0"]
    elif eng == "nec42":
        lines += ["GE -1", "GN 2 0 0 0 13 0.005 NOFILE"]
    else:
        lines += ["GE -1 0", "GN 0 0 0 0 13 0.005 1.0 0.0 NOFILE"]
    ex = f"EX 0 1 {2 * r + 1} 0 1 0" if eng == "nec42" else f"EX 0 1 {2 * r} 2 1 0"
    lines += [ex, f"FR 0 1 0 0 {decks.FREQ_MHZ} 0", "XQ 0", "EN"]
    return "\n".join(lines) + "\n"


CASES = {
    "free16": dict(depth=0.0254, n_rad=16, ground=False),
    "gnd8": dict(depth=0.0254, n_rad=8, ground=True),
    "gnd16_deep1m": dict(depth=1.0, n_rad=16, ground=True),
}
with open(sys.argv[1], "a") as fh:
    for name, kw in CASES.items():
        for r in (1, 2, 4, 8):
            for eng, exe in (("nec42", NEC42), ("nec5", NEC5)):
                try:
                    (R, X), _dt = run(exe, deck(r, eng=eng, **kw))
                    rec = dict(R=R, X=X)
                except Exception as e:  # noqa: BLE001 - recorded, run goes on
                    rec = dict(error=repr(e)[:200])
                rec.update(case=name, r=r, engine=eng)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
