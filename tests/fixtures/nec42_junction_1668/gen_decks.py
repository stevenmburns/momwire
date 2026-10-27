"""The momwire#1668 gate decks: NEC-4 dialect, EX 0 voltage sources, free
space, every deck fed on a segment BESIDE a solved junction (where a wrong
charge ratio moves Z most), each run through licensed NEC-4.2 as a black box
(README.md). Regenerate with

    python tests/fixtures/nec42_junction_1668/gen_decks.py tests/fixtures/nec42_junction_1668/decks

Td_*  a 1 mm T (two collinear 2 m arms and a perpendicular 2 m arm) whose
      perpendicular arm is meshed differently from the collinear pair: the
      collinear/perpendicular segment counts are 20/40, 40/20 and 20/10.
Sd_*  a 1 mm / 5 mm stepped dipole (two 3 m halves) with 20/40 and 40/20
      segments, fed beside the step on the thick and on the thin side.
V*_   three 1 mm 2 m arms, two at +-theta from +z and the third along -z,
      20 segments each: theta = 30 and 150 degrees (60 is the symmetric Y).
X4_   four 1 mm 2 m arms, collinear pair along x plus arms along +z and +y.
Tp3_  the 20-segment T with a 3 mm perpendicular arm (radii differ at a
      three-member node).
B60_  a 1 mm / 5 mm two-member bend, 60 degrees between the members.
H*_   decks built after the rule was fitted and never used to fit it: an
      asymmetric 3-member node in three dimensions (2 / 0.7 / 4 mm, 15 / 24 /
      10 segments, 10 MHz), a five-member star of mixed radii (21 MHz), a
      2 mm / 12 mm step with 17 / 29 segments at 28 MHz, and a dipole with a
      6 mm centre section and a 1 mm stub (two solved nodes, one of them a T).
"""

import math
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "nec42_radius_step_1644"
    ),
)
from gen_decks import MM, deck


def arm(ang_deg, elev_deg, length):
    a, e = math.radians(ang_deg), math.radians(elev_deg)
    return (
        length * math.cos(e) * math.cos(a),
        length * math.cos(e) * math.sin(a),
        length * math.sin(e),
    )


def tee(n1, n3, r=1.0, r3=None):
    r3 = r if r3 is None else r3
    return [
        (1, n1, (-2, 0, 0), (0, 0, 0), r * MM),
        (2, n1, (0, 0, 0), (2, 0, 0), r * MM),
        (3, n3, (0, 0, 0), (0, 0, 2), r3 * MM),
    ]


def main(out):
    os.makedirs(out, exist_ok=True)
    for n1, n3 in ((20, 40), (40, 20), (20, 10)):
        deck(out, f"Td_{n1}_{n3}_w3s1", "T", tee(n1, n3), 3, 1)
    for n1, n2 in ((20, 40), (40, 20)):
        w = [
            (1, n1, (0, 0, -3), (0, 0, 0), 1 * MM),
            (2, n2, (0, 0, 0), (0, 0, 3), 5 * MM),
        ]
        deck(out, f"Sd_{n1}_{n2}_w2s1", "step unequal D", w, 2, 1)
        deck(out, f"Sd_{n1}_{n2}_w1s{n1}", "step unequal D", w, 1, n1)
    for th in (30, 150):
        t = math.radians(th)
        a = (2 * math.sin(t), 0, 2 * math.cos(t))
        b = (-2 * math.sin(t), 0, 2 * math.cos(t))
        w = [
            (1, 20, (0, 0, 0), a, MM),
            (2, 20, (0, 0, 0), b, MM),
            (3, 20, (0, 0, 0), (0, 0, -2), MM),
        ]
        deck(out, f"V{th}_w3s1", f"V {th}", w, 3, 1)
    w4 = tee(20, 20) + [(4, 20, (0, 0, 0), (0, 2, 0), MM)]
    deck(out, "X4_n20_w3s1", "4-arm", w4, 3, 1)
    deck(out, "Tp3.0_n20_w3s1", "T", tee(20, 20, 1.0, 3.0), 3, 1)
    p = math.radians(60)
    w = [
        (1, 20, (3, 0, 0), (0, 0, 0), MM),
        (2, 20, (0, 0, 0), (3 * math.cos(p), 0, 3 * math.sin(p)), 5 * MM),
    ]
    deck(out, "B60_n20_w2s1", "bend 60", w, 2, 1)
    w = [
        (1, 15, (0, 0, 0), arm(0, 0, 2.5), 2 * MM),
        (2, 24, (0, 0, 0), arm(110, 20, 3.0), 0.7 * MM),
        (3, 10, (0, 0, 0), arm(230, -35, 1.6), 4 * MM),
    ]
    for t in (1, 2, 3):
        deck(out, f"H1_w{t}s1", "holdout asym 3", w, t, 1, f=10.0)
    w = []
    star = (
        (0, 0, 1, 12),
        (72, 10, 2, 12),
        (144, -10, 1, 16),
        (216, 30, 3, 10),
        (288, -20, 1.5, 12),
    )
    for i, (ang, el, r, n) in enumerate(star):
        w.append((i + 1, n, (0, 0, 0), arm(ang, el, 1.8), r * MM))
    deck(out, "H2_w4s1", "holdout 5-star", w, 4, 1, f=21.0)
    w = [
        (1, 17, (0, 0, -1.5), (0, 0, 0), 2 * MM),
        (2, 29, (0, 0, 0), (0, 0, 1.5), 12 * MM),
    ]
    deck(out, "H3_f28_w2s1", "holdout step", w, 2, 1, f=28.0)
    w = [
        (1, 10, (0, -3, 0), (0, -1, 0), 1 * MM),
        (2, 10, (0, -1, 0), (0, 1, 0), 6 * MM),
        (3, 10, (0, 1, 0), (0, 3, 0), 1 * MM),
        (4, 8, (0, 1, 0), (0, 1, 1.2), 1 * MM),
    ]
    deck(out, "H5_w4s1", "holdout two nodes", w, 4, 1)


if __name__ == "__main__":
    main(sys.argv[1])
