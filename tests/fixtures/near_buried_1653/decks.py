"""Synthetic black-box decks for momwire#1653 (no user's deck): the shapes of
Dan's Hub / Slant style (a buried radial screen under a vertical) and a
simple buried dipole. Writes <stem>.nec5.nec (momwire + NEC-5 dialect) and
<stem>.nec42.nec (NEC-4.2 dialect), same geometry, segment-centre feeds."""

import math


def radials(n, L, depth, tag0, segs, slant=False):
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = L * math.cos(a), L * math.sin(a)
        z0 = 0.0 if slant else -depth
        out.append(f"GW {tag0 + i} {segs} 0.0 0.0 {z0} {x:.9f} {y:.9f} {-depth} 0.001")
    return out


NE = [
    "NE 0 20 1 1 0.5 0.0 1.0 1.0 0.0 0.0",
    "NE 0 20 1 1 0.5 0.0 0.01 1.0 0.0 0.0",
    "NE 0 20 1 1 0.5 0.0 -0.05 1.0 0.0 0.0",
    "NE 0 20 1 1 0.5 0.0 -0.5 1.0 0.0 0.0",
    "NE 0 20 1 1 20.5 0.0 1.0 10.0 0.0 0.0",
    "NE 0 20 1 1 20.5 0.0 -0.05 10.0 0.0 0.0",
    "NH 0 20 1 1 0.5 0.0 1.0 1.0 0.0 0.0",
    "NH 0 20 1 1 0.5 0.0 -0.05 1.0 0.0 0.0",
    "NE 1 1 1 7 300.0 0.0 10.0 0.0 0.0 25.0",
]
FR = "FR 0 1 0 0 7.0 0"


def write(stem, cm, gws, tag, seg):
    body5 = [
        f"CM {cm} (NEC-5)",
        "CE",
        *gws,
        "GE -1 0",
        "GN 0 0 0 0 13 0.005 1.0 0.0 NOFILE",
        f"EX 0 {tag} {seg} 0 1.0 0.0",
        FR,
        *NE,
        "EN",
        "",
    ]
    body4 = [
        f"CM {cm} (NEC-4.2)",
        "CE",
        *gws,
        "GE -1",
        "GN 2 0 0 0 13 0.005 NOFILE",
        f"EX 0 {tag} {seg} 0 1.0 0.0",
        FR,
        *NE,
        "EN",
        "",
    ]
    open(stem + ".nec5.nec", "w").write("\n".join(body5))
    open(stem + ".nec42.nec", "w").write("\n".join(body4))


write(
    "d1_buried_dipole",
    "buried horizontal dipole 5 m at -0.3 m",
    ["GW 1 11 -2.5 0.0 -0.3 2.5 0.0 -0.3 0.001"],
    1,
    6,
)
write(
    "d2_elev_screen",
    "10 m vertical 1 m up over 8 buried radials",
    ["GW 1 15 0.0 0.0 1.0 0.0 0.0 11.0 0.001"] + radials(8, 5.0, 0.15, 2, 10),
    1,
    1,
)
write(
    "d3_hub",
    "vertical from a buried hub, 8 radials",
    [
        "GW 1 2 0.0 0.0 -0.15 0.0 0.0 0.0 0.001",
        "GW 10 16 0.0 0.0 0.0 0.0 0.0 10.0 0.001",
    ]
    + radials(8, 5.0, 0.15, 2, 10),
    10,
    2,
)
write(
    "d4_slant",
    "vertical on the surface node, 8 radials slanting down",
    ["GW 1 16 0.0 0.0 0.0 0.0 0.0 10.0 0.001"]
    + radials(8, 5.0, 0.15, 2, 10, slant=True),
    1,
    2,
)
