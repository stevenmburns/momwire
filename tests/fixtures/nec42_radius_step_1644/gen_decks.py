"""The momwire#1644 gate decks: NEC-4 dialect, EX 0 voltage sources, 14 MHz
(14.15 MHz for the Yagi), each run through licensed NEC-4.2 as a black box
(README.md). Regenerate with

    python tests/fixtures/nec42_radius_step_1644/gen_decks.py tests/fixtures/nec42_radius_step_1644/decks

L_*   the stepped-dipole ladder: two collinear 3 m wires (wire 1 the thin
      one, z -3..0; wire 2 the thick one, z 0..3), n = 20/40/80/160 segments
      per half (x1..x8), at 1:2 (0.5/1 mm), 1:5 (1/5 mm) and 1:10 (1/10 mm).
      Feed `away`: wire 2 segment n/2, 1.5 m from the step. Feed `step`:
      wire 2 segment 1, beside it. `L_uni<r>_*` is the uniform-radius twin at
      the THICK radius with the same feed.
J_*   junction shapes: a T (two collinear 2 m arms and a perpendicular one,
      1 mm, fed on the perpendicular arm's segment beside the node), a
      symmetric Y (three 1 mm 2 m arms at 120 degrees, fed on arm 1 beside the
      node), an L (3 m wires of 1 mm and 5 mm at 90 degrees, fed at the 1 mm
      wire's middle).
Y_*   a tapered 20 m Yagi element (sections from the centre: 2.4 m of 12.7 mm
      radius, then 1.6 m of 11.1, 1.4 m of 9.525 and 0.9 m of 6.35 each side),
      centre fed, and a 3-element tapered Yagi (reflector x1.03 at -2.4 m,
      director x0.955 at +2.6 m, the driven element fed); four meshes each.
      `Yu*` is the uniform-radius twin (every section at 12.7 mm).
G_*   over ground: a horizontal 1:5 stepped dipole at 8 m (feed on the thick
      half's middle) over GN 1 and GN 2 (13, 0.005 S/m), and a grounded
      stepped vertical (2.5 m of 10 mm under 2.5 m of 2 mm) over GN 1, fed
      at its base segment.
"""

import math
import os
import sys

MM = 1e-3


def deck(out, name, cm, wires, tag, seg, f=14.0, gn=None):
    lines = [f"CM {cm}", "CE"]
    for t, n, a, b, r in wires:
        lines.append(
            f"GW {t} {n} {a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
            f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {r:.6g}"
        )
    lines.append("GE 1" if gn is not None else "GE 0")
    if gn == 1:
        lines.append("GN 1")
    if gn == 2:
        lines.append("GN 2 0 0 0 13 0.005")
    lines += [f"EX 0 {tag} {seg} 0 1.0 0.0", f"FR 0 1 0 0 {f} 0", "XQ 0", "EN"]
    with open(os.path.join(out, name + ".nec"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


SECTIONS = ((1.2, 12.7), (2.8, 11.1), (4.2, 9.525), (5.1, 6.35))
COUNTS = {1: (9, 6, 5, 4), 2: (17, 12, 10, 8), 4: (33, 24, 20, 16), 8: (65, 48, 40, 32)}


def element(x, scale, tag0, mult):
    c = COUNTS[mult]
    y_in = SECTIONS[0][0] * scale
    wires = [(tag0, c[0], (x, -y_in, 0), (x, y_in, 0), SECTIONS[0][1] * MM)]
    t, prev = tag0 + 1, y_in
    for (end, r), n in zip(SECTIONS[1:], c[1:]):
        e = end * scale
        wires.append((t, n, (x, prev, 0), (x, e, 0), r * MM))
        wires.append((t + 1, n, (x, -prev, 0), (x, -e, 0), r * MM))
        t, prev = t + 2, e
    return wires, (c[0] + 1) // 2


def uniform(ws):
    return [(t, n, a, b, SECTIONS[0][1] * MM) for (t, n, a, b, _r) in ws]


def main(out):
    for a1, a2 in ((0.5, 1), (1, 5), (1, 10)):
        for n in (20, 40, 80, 160):
            for feed, seg in (("away", n // 2), ("step", 1)):
                for lo, hi, lab in ((a1, a2, f"step{a1}-{a2}"), (a2, a2, f"uni{a2}")):
                    deck(
                        out,
                        f"L_{lab}_{feed}_n{n}",
                        f"{lab} {lo}/{hi} mm, {n}/{n} seg, feed w2 s{seg}",
                        [
                            (1, n, (0, 0, -3), (0, 0, 0), lo * MM),
                            (2, n, (0, 0, 0), (0, 0, 3), hi * MM),
                        ],
                        2,
                        seg,
                    )
    for n in (10, 20, 40, 80, 160):
        deck(
            out,
            f"J_T_n{n}",
            f"T junction 1 mm, {n} seg/arm, feed w3 s1",
            [
                (1, n, (-2, 0, 0), (0, 0, 0), MM),
                (2, n, (0, 0, 0), (2, 0, 0), MM),
                (3, n, (0, 0, 0), (0, 0, 2), MM),
            ],
            3,
            1,
        )
    for n in (10, 20, 40):
        arms = []
        for kk, ang in enumerate((90, 210, 330), start=1):
            end = (2 * math.cos(math.radians(ang)), 0, 2 * math.sin(math.radians(ang)))
            arms.append((kk, n, (0, 0, 0), end, MM))
        deck(
            out,
            f"J_Y_n{n}",
            f"Y, three 1 mm arms at 120 deg, {n} seg/arm, feed w1 s1",
            arms,
            1,
            1,
        )
    for n in (21, 41, 81):
        deck(
            out,
            f"J_L_n{n}",
            f"L 1 mm / 5 mm at 90 deg, {n} seg/wire, feed w1 middle",
            [(1, n, (3, 0, 0), (0, 0, 0), MM), (2, n, (0, 0, 0), (0, 0, 3), 5 * MM)],
            1,
            (n + 1) // 2,
        )
    for mult in (1, 2, 4, 8):
        w, mid = element(0.0, 1.0, 1, mult)
        deck(
            out,
            f"Y_elem_x{mult}",
            f"tapered 20 m element, mesh x{mult}",
            w,
            1,
            mid,
            f=14.15,
        )
        deck(
            out,
            f"Yu_elem_x{mult}",
            f"uniform 12.7 mm twin of the tapered element, mesh x{mult}",
            uniform(w),
            1,
            mid,
            f=14.15,
        )
        wr, _ = element(-2.4, 1.03, 1, mult)
        wd, mid = element(0.0, 1.0, 11, mult)
        wdd, _ = element(2.6, 0.955, 21, mult)
        deck(
            out,
            f"Y3_x{mult}",
            f"3-element tapered Yagi, mesh x{mult}",
            wr + wd + wdd,
            11,
            mid,
            f=14.15,
        )
        deck(
            out,
            f"Yu3_x{mult}",
            f"uniform 12.7 mm twin of the 3-element Yagi, mesh x{mult}",
            uniform(wr + wd + wdd),
            11,
            mid,
            f=14.15,
        )
    for n in (20, 40):
        for gn, lab in ((1, "pec"), (2, "somm")):
            deck(
                out,
                f"G_hdip_{lab}_n{n}",
                f"horizontal 1/5 mm step, {lab}, {n}/half",
                [
                    (1, n, (-3, 0, 8), (0, 0, 8), 1 * MM),
                    (2, n, (0, 0, 8), (3, 0, 8), 5 * MM),
                ],
                2,
                n // 2,
                gn=gn,
            )
        deck(
            out,
            f"G_vert_pec_n{n}",
            f"grounded stepped vertical 10/2 mm, pec, {n}/section",
            [
                (1, n, (0, 0, 0), (0, 0, 2.5), 10 * MM),
                (2, n, (0, 0, 2.5), (0, 0, 5.0), 2 * MM),
            ],
            1,
            1,
            gn=1,
        )


if __name__ == "__main__":
    main(sys.argv[1])
