"""Probe 10: the licensed binary on probe 5's DETACHED deck, full 2x2 Z.

Above wire: z 0.3 -> 0.4 -> 0.8 -> 10.3 at x = 0 (segs 2, 2m, 19m).
Buried wire: z -2.3 -> -0.8 -> -0.4 -> -0.3 at x = 0.5 (segs 3m, 2m, 2).
Ports: above knot z = 4.8 (tag 6, seg 8m end 2); buried knot z = -1.3 (tag 1,
seg 2m end 2). Y from two runs (drive one, 1e-20 V probe on the other), then
Z = inv(Y) - compared with probe5_detached_cross_blocks.json's razor (cross
blocks only, bypass route) and bspline (transmitted grid) Z11/Z22/Z12.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/smburns/antennas/antennaknobs/src")
from antennaknobs.engines.nec5 import NEC5Engine, run_deck  # noqa: E402

EXE = "/home/smburns/antennas/NEC5-downloads/nec5-linux/nec5cl"
HERE = pathlib.Path(__file__).resolve().parent


def f(x):
    return f"{float(x):.6E}"


def deck(m, v_up, v_dn):
    edges = [
        ((0.5, -2.3), (0.5, -0.8), 3 * m),
        ((0.5, -0.8), (0.5, -0.4), 2 * m),
        ((0.5, -0.4), (0.5, -0.3), 2),
        ((0, 0.3), (0, 0.4), 2),
        ((0, 0.4), (0, 0.8), 2 * m),
        ((0, 0.8), (0, 10.3), 19 * m),
    ]
    lines = ["CM probe10 detached", "CE"]
    for tag, ((x0, z0), (x1, z1), ns) in enumerate(edges, start=1):
        lines.append(
            f"GW {tag} {ns} {f(x0)} {f(0)} {f(z0)} {f(x1)} {f(0)} {f(z1)} {f(0.001)}"
        )
    lines += [
        "GE -1 0",
        f"GN 0 0 0 0 {f(13.0)} {f(0.005)} {f(1.0)} {f(0.0)} NOFILE",
        f"EX 0 6 {8 * m} 2 {f(v_up)} {f(0.0)}",
        f"EX 0 1 {2 * m} 2 {f(v_dn)} {f(0.0)}",
        f"FR 0 1 0 0 {f(7.0)} {f(0.0)}",
        "XQ 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


def currents(m, v_up, v_dn):
    r = NEC5Engine._parse_input_parameters(
        run_deck(EXE, deck(m, v_up, v_dn), timeout=300), current=True
    )[0]
    assert r[0][0] == 6 and r[1][0] == 1
    return r[0][2], r[1][2]


mw = json.loads((HERE / "probe5_detached_cross_blocks.json").read_text())
out = []
for m in (1, 2, 4, 8):
    i_up_a, i_dn_a = currents(m, 1.0, 1e-20)
    i_up_b, i_dn_b = currents(m, 1e-20, 1.0)
    Y = np.array([[i_up_a, i_up_b], [i_dn_a, i_dn_b]])
    Z = np.linalg.inv(Y)
    nonrec = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
    row = dict(
        x=m,
        z11=[Z[0, 0].real, Z[0, 0].imag],
        z22=[Z[1, 1].real, Z[1, 1].imag],
        z12=[Z[0, 1].real, Z[0, 1].imag],
        nonrec=nonrec,
    )
    out.append(row)
    print(
        f"x{m} NEC-5    Z11 {Z[0, 0].real:9.3f}{Z[0, 0].imag:+9.3f}j Z22 {Z[1, 1].real:8.3f}"
        f"{Z[1, 1].imag:+9.3f}j Z12 {Z[0, 1].real:7.3f}{Z[0, 1].imag:+7.3f}j nonrec {nonrec:.2e}"
    )
    for r in mw:
        if r["x"] == m and r["medium"] == "soilA":
            print(
                f"   {r['solver']:10s} Z11 {r['z11'][0]:9.3f}{r['z11'][1]:+9.3f}j Z22 {r['z22'][0]:8.3f}"
                f"{r['z22'][1]:+9.3f}j Z12 {r['z12'][0]:7.3f}{r['z12'][1]:+7.3f}j nonrec {r['nonrec']:.2e}"
            )
(HERE / "probe10_nec5_detached.json").write_text(json.dumps(out, indent=1))
