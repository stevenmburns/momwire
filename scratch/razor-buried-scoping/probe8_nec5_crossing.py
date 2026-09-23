"""Probe 8: the licensed NEC-5 binary on probe 2/3's crossing deck (black box).

Same geometry as crossing_deck(1) (graded edges -> one GW per edge, junctions
by coincident ends, the conductor continuing through z = 0), soil A under the
documented buried spelling `GE -1 0` + `GN 0 ... NOFILE` (AK's wrapper), 7 MHz,
radius 1 mm, far mesh x1/x2/x4/x8 exactly as the momwire probes.

(1) single port, EX 0 on the knot at z = 4.5 (tag 6, segment 8m, end 2);
(2) the 2-port Y: above port z = 4.5 and buried port z = -1.0 (tag 1, segment
    2m, end 2), each driven in turn with the other a 1e-20 V probe source
    (a zero-amplitude EX reads as 1 V - the AK#1629 trap).
"""

import json
import pathlib
import sys

sys.path.insert(0, "/home/smburns/antennas/antennaknobs/src")
from antennaknobs.engines.nec5 import NEC5Engine, run_deck  # noqa: E402

EXE = "/home/smburns/antennas/NEC5-downloads/nec5-linux/nec5cl"
A = 0.001
EPS, SIG = 13.0, 0.005
F_MHZ = 7.0


def n(x):
    return f"{float(x):.6E}"


def deck(m, sources):
    edges = [
        (-2.0, -0.5, 3 * m),
        (-0.5, -0.1, 2 * m),
        (-0.1, 0.0, 2),
        (0.0, 0.1, 2),
        (0.1, 0.5, 2 * m),
        (0.5, 10.0, 19 * m),
    ]
    lines = ["CM probe8 crossing deck", "CE"]
    for tag, (z0, z1, ns) in enumerate(edges, start=1):
        lines.append(
            f"GW {tag} {ns} {n(0)} {n(0)} {n(z0)} {n(0)} {n(0)} {n(z1)} {n(A)}"
        )
    lines.append("GE -1 0")
    lines.append(f"GN 0 0 0 0 {n(EPS)} {n(SIG)} {n(1.0)} {n(0.0)} NOFILE")
    for tag, seg, v in sources:
        lines.append(f"EX 0 {tag} {seg} 2 {n(v)} {n(0.0)}")
    lines.append(f"FR 0 1 0 0 {n(F_MHZ)} {n(0.0)}")
    lines.append("XQ 0")
    lines.append("EN")
    return "\n".join(lines) + "\n"


def rows(text, current):
    return NEC5Engine._parse_input_parameters(text, current=current)[0]


out = []
for m in (1, 2, 4, 8, 16):
    up = (6, 8 * m)
    dn = (1, 2 * m)
    txt = run_deck(EXE, deck(m, [(*up, 1.0)]), timeout=300)
    z = rows(txt, current=False)[0][2]
    # 2-port: drive above, probe below; then drive below, probe above
    ta = run_deck(EXE, deck(m, [(*up, 1.0), (*dn, 1e-20)]), timeout=300)
    tb = run_deck(EXE, deck(m, [(*up, 1e-20), (*dn, 1.0)]), timeout=300)
    # rows come back in EX-card order with ABSOLUTE segment numbers
    ra, rb = rows(ta, current=True), rows(tb, current=True)
    assert ra[0][0] == 6 and ra[1][0] == 1 and rb[0][0] == 6 and rb[1][0] == 1
    y21 = ra[1][2]  # current at the buried port with the above port driven
    y12 = rb[0][2]
    nonrec = abs(y12 - y21) / abs(y12)
    out.append(
        dict(
            x=m,
            z=[z.real, z.imag],
            y12=[y12.real, y12.imag],
            y21=[y21.real, y21.imag],
            nonrec=nonrec,
        )
    )
    print(
        f"x{m:<2d} NEC-5 Z(feed@4.5) {z.real:9.4f}{z.imag:+9.4f}j   2-port nonrec {nonrec:.3e}",
        flush=True,
    )

pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
