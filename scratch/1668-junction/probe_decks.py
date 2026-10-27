"""The momwire#1668 probe decks, every family the ratio fits in `results/`
were taken on (README.md). Writes one directory per family:

    python scratch/1668-junction/probe_decks.py OUT

e1  the 1 mm / 10 mm step and its 10 mm twin at 40 and 160 segments per half,
    fed at eight positions either side of the step; the 1 mm T at 20 and 80
    segments per arm fed at six.
e2  T meshes 10 / 40 / 160, T radii 0.2 and 5 mm, unequal-mesh T (Td), a
    four-arm node (X4), unequal-mesh steps (Sd), a T whose perpendicular arm
    is 0.5 or 3 mm (Tp); each fed beside the node on two arms.
e3  three 1 mm arms, two at +-theta from +z, one along -z (V), theta 30..150.
e4  Td and Sd fed further from the node (the ratio fit's feed check).
e5  two-member 1 mm / 5 mm bends at 60..150 degrees (B).
e6  the T with its wires reversed and reordered (NEC-4.2 reads them alike).
e7  the holdouts (H), built after the readings were fitted.

The `../../tests/fixtures/nec42_junction_1668` decks are a subset, written by
that directory's own generator.
"""

import math
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "tests",
        "fixtures",
        "nec42_radius_step_1644",
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


def step(n1, n2, r1, r2, half=3.0):
    return [
        (1, n1, (0, 0, -half), (0, 0, 0), r1 * MM),
        (2, n2, (0, 0, 0), (0, 0, half), r2 * MM),
    ]


def e1(out):
    for n in (40, 160):
        for lo, lab in ((1, "s1-10"), (10, "u10")):
            w = step(n, n, lo, 10)
            feeds = ((2, 1), (2, 2), (2, 3), (2, 5), (2, 10), (1, n), (1, n - 1))
            for tag, seg in (*feeds, (1, n - 2)):
                cm = f"{lab} n{n} feed w{tag} s{seg}"
                deck(out, f"{lab}_n{n}_w{tag}s{seg}", cm, w, tag, seg)
    for n in (20, 80):
        for tag, seg in ((3, 1), (3, 2), (3, 3), (3, n // 2), (1, n), (1, n // 2)):
            cm = f"T n{n} feed w{tag} s{seg}"
            deck(out, f"T_n{n}_w{tag}s{seg}", cm, tee(n, n), tag, seg)


def e2(out):
    for n in (10, 40, 160):
        deck(out, f"T_n{n}_w1s{n}", "T", tee(n, n), 1, n)
        deck(out, f"T_n{n}_w3s1", "T", tee(n, n), 3, 1)
    for r in (0.2, 5.0):
        deck(out, f"Tr{r}_n20_w3s1", "T", tee(20, 20, r), 3, 1)
        deck(out, f"Tr{r}_n20_w1s20", "T", tee(20, 20, r), 1, 20)
    for n1, n3 in ((20, 40), (40, 20), (20, 10)):
        deck(out, f"Td_{n1}_{n3}_w3s1", "T", tee(n1, n3), 3, 1)
        deck(out, f"Td_{n1}_{n3}_w1s{n1}", "T", tee(n1, n3), 1, n1)
    w4 = [*tee(20, 20), (4, 20, (0, 0, 0), (0, 2, 0), MM)]
    deck(out, "X4_n20_w3s1", "4-arm", w4, 3, 1)
    deck(out, "X4_n20_w1s20", "4-arm", w4, 1, 20)
    for n1, n2 in ((20, 40), (40, 20)):
        deck(out, f"Sd_{n1}_{n2}_w2s1", "step unequal D", step(n1, n2, 1, 5), 2, 1)
        w = step(n1, n2, 1, 5)
        deck(out, f"Sd_{n1}_{n2}_w1s{n1}", "step unequal D", w, 1, n1)
    for r3 in (0.5, 3.0):
        deck(out, f"Tp{r3}_n20_w3s1", "T", tee(20, 20, 1.0, r3), 3, 1)
        deck(out, f"Tp{r3}_n20_w1s20", "T", tee(20, 20, 1.0, r3), 1, 20)


def e3(out):
    for th in (30, 45, 60, 75, 90, 105, 135, 150):
        t = math.radians(th)
        a = (2 * math.sin(t), 0, 2 * math.cos(t))
        b = (-2 * math.sin(t), 0, 2 * math.cos(t))
        w = [
            (1, 20, (0, 0, 0), a, MM),
            (2, 20, (0, 0, 0), b, MM),
            (3, 20, (0, 0, 0), (0, 0, -2), MM),
        ]
        deck(out, f"V{th}_w3s1", f"V {th}", w, 3, 1)
        deck(out, f"V{th}_w1s1", f"V {th}", w, 1, 1)


def e4(out):
    for n1, n3 in ((20, 40), (40, 20)):
        for tag, seg in ((3, 3), (3, 6), (1, n1 - 2), (1, n1 - 5), (3, n3 // 2)):
            deck(out, f"Td_{n1}_{n3}_w{tag}s{seg}", "T", tee(n1, n3), tag, seg)
    for n1, n2 in ((20, 40), (40, 20)):
        w = step(n1, n2, 1, 5)
        for tag, seg in ((2, 3), (2, 6), (1, n1 - 2), (1, n1 - 5)):
            deck(out, f"Sd_{n1}_{n2}_w{tag}s{seg}", "step unequal D", w, tag, seg)


def e5(out):
    for phi in (60, 90, 120, 150):
        p = math.radians(phi)
        for n in (20, 40):
            end = (3 * math.cos(p), 0, 3 * math.sin(p))
            w = [(1, n, (3, 0, 0), (0, 0, 0), MM), (2, n, (0, 0, 0), end, 5 * MM)]
            deck(out, f"B{phi}_n{n}_w1s{n}", f"bend {phi}", w, 1, n)
            deck(out, f"B{phi}_n{n}_w2s1", f"bend {phi}", w, 2, 1)


def e6(out):
    n = 20
    c1f, c1r = ((-2, 0, 0), (0, 0, 0)), ((0, 0, 0), (-2, 0, 0))
    c2f, c2r = ((0, 0, 0), (2, 0, 0)), ((2, 0, 0), (0, 0, 0))
    pf, pr = ((0, 0, 0), (0, 0, 2)), ((0, 0, 2), (0, 0, 0))
    variants = {
        "fff": (c1f, c2f, pf),
        "rff": (c1r, c2f, pf),
        "frf": (c1f, c2r, pf),
        "ffr": (c1f, c2f, pr),
        "rrr": (c1r, c2r, pr),
    }
    for name, (a, b, c) in variants.items():
        w = [(1, n, *a, MM), (2, n, *b, MM), (3, n, *c, MM)]
        deck(out, f"To_{name}", f"T orient {name}", w, 3, 1 if c is pf else n)
    w = [
        (1, n, (0, 0, 0), (0, 0, 2), MM),
        (2, n, (-2, 0, 0), (0, 0, 0), MM),
        (3, n, (0, 0, 0), (2, 0, 0), MM),
    ]
    deck(out, "To_pfirst", "T perp first", w, 1, 1)


def e7(out):
    w = [
        (1, 15, (0, 0, 0), arm(0, 0, 2.5), 2 * MM),
        (2, 24, (0, 0, 0), arm(110, 20, 3.0), 0.7 * MM),
        (3, 10, (0, 0, 0), arm(230, -35, 1.6), 4 * MM),
    ]
    for t in (1, 2, 3):
        deck(out, f"H1_w{t}s1", "holdout asym 3", w, t, 1, f=10.0)
    star = (
        (0, 0, 1, 12),
        (72, 10, 2, 12),
        (144, -10, 1, 16),
        (216, 30, 3, 10),
        (288, -20, 1.5, 12),
    )
    w = [
        (i + 1, n, (0, 0, 0), arm(ang, el, 1.8), r * MM)
        for i, (ang, el, r, n) in enumerate(star)
    ]
    for t in (1, 4):
        deck(out, f"H2_w{t}s1", "holdout 5-star", w, t, 1, f=21.0)
    for f, half in ((7.0, 6.0), (28.0, 1.5)):
        w = step(17, 29, 2, 12, half)
        deck(out, f"H3_f{int(f)}_w2s1", "holdout step", w, 2, 1, f=f)
        deck(out, f"H3_f{int(f)}_w1s17", "holdout step", w, 1, 17, f=f)
    w = [
        (1, 18, (2.4, 0, 0), (0, 0, 0), 3 * MM),
        (2, 11, (0, 0, 0), arm(45, 0, 2.0), 0.5 * MM),
    ]
    deck(out, "H4_w1s18", "holdout bend", w, 1, 18)
    deck(out, "H4_w2s1", "holdout bend", w, 2, 1)
    w = [
        (1, 10, (0, -3, 0), (0, -1, 0), 1 * MM),
        (2, 10, (0, -1, 0), (0, 1, 0), 6 * MM),
        (3, 10, (0, 1, 0), (0, 3, 0), 1 * MM),
        (4, 8, (0, 1, 0), (0, 1, 1.2), 1 * MM),
    ]
    deck(out, "H5_w2s5", "holdout two nodes", w, 2, 5)
    deck(out, "H5_w4s1", "holdout two nodes", w, 4, 1)


if __name__ == "__main__":
    root = sys.argv[1]
    for fam in (e1, e2, e3, e4, e5, e6, e7):
        d = os.path.join(root, fam.__name__)
        os.makedirs(d, exist_ok=True)
        fam(d)
