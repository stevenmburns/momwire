"""Synthetic two-radius crossing decks for momwire#1630.

A fat vertical (default 19 mm radius, 5 m, 14 MHz) whose base meets 16 thin
buried radials (default 1 mm, 4 m, at -0.0254 m), over GN 2 13/0.005. Two
spellings of the node:

* ``hub``   - a drop from the base (0,0,0) down to a buried hub at depth d,
  the radials leaving the hub horizontally. The crossing node joins TWO
  members (vertical + drop); the hub is wholly below.
* ``slope`` - the radials meet the base AT z = 0: each leaves the node on a
  45-degree sloping leg down to depth d, then runs horizontally. The crossing
  node joins 17 members.

Refinement r multiplies every segment count (vertical 8r, radial 8r, drop and
slope legs r). The source is a segment source on the vertical (NEC-4.2 EX 0):
``bottom`` puts it on the vertical's first segment, next to the interface
junction; ``mid`` on segment 2r+1, whose lower end sits 1.25 m up at every r.

Writes NEC-4.2 dialect (GE -1, GN 2 ... NOFILE, EX 0 on a segment) and NEC-5
dialect (GE -1 0, GN 0 ... NOFILE, EX 0 at end 2 of a segment: for ``mid``
the NEC-5 source sits at the knot 1.25 m up, i.e. end 2 of segment 2r).
"""

from __future__ import annotations

import math

FREQ_MHZ = 14.0
H_V = 5.0
L_RAD = 4.0
DEPTH = 0.0254
N_RAD = 16


def _gw(tag, n, p, q, a):
    return (
        f"GW {tag} {n} {p[0]:.9E} {p[1]:.9E} {p[2]:.9E} "
        f"{q[0]:.9E} {q[1]:.9E} {q[2]:.9E} {a:.6E}"
    )


def wires(variant, r, a_v=0.019, a_b=0.001, n_rad=N_RAD):
    """[(tag, nseg, p, q, a)], the vertical first (tag 1)."""
    out = [(1, 8 * r, (0, 0, 0), (0, 0, H_V), a_v)]
    tag = 2
    if variant == "hub":
        hub = (0.0, 0.0, -DEPTH)
        out.append((tag, r, hub, (0, 0, 0), a_b))
        tag += 1
        for i in range(n_rad):
            phi = 2 * math.pi * i / n_rad
            far = (L_RAD * math.cos(phi), L_RAD * math.sin(phi), -DEPTH)
            out.append((tag, 8 * r, hub, far, a_b))
            tag += 1
    elif variant == "slope":
        for i in range(n_rad):
            phi = 2 * math.pi * i / n_rad
            c, s = math.cos(phi), math.sin(phi)
            foot = (DEPTH * c, DEPTH * s, -DEPTH)
            far = (L_RAD * c, L_RAD * s, -DEPTH)
            out.append((tag, r, (0, 0, 0), foot, a_b))
            tag += 1
            out.append((tag, 8 * r, foot, far, a_b))
            tag += 1
    else:
        raise ValueError(variant)
    return out


def feed_seg(feed, r):
    return 1 if feed == "bottom" else 2 * r + 1


def nec42(variant, r, feed="mid", **kw):
    lines = [f"CM momwire#1630 {variant} r={r} feed={feed} {kw}", "CE"]
    lines += [_gw(*w) for w in wires(variant, r, **kw)]
    lines += [
        "GE -1",
        "GN 2 0 0 0 13 0.005 NOFILE",
        f"EX 0 1 {feed_seg(feed, r)} 0 1.0 0.0",
        f"FR 0 1 0 0 {FREQ_MHZ} 0",
        "XQ 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


def nec5(variant, r, feed="mid", **kw):
    """NEC-5: the source at a knot. ``mid`` -> end 2 of segment 2r (1.25 m)."""
    if feed == "bottom":
        raise ValueError("NEC-5 refuses... use mid (no knot source at the node)")
    lines = [f"CM momwire#1630 {variant} r={r} feed={feed} {kw} (NEC-5)", "CE"]
    lines += [_gw(*w) for w in wires(variant, r, **kw)]
    lines += [
        "GE -1 0",
        "GN 0 0 0 0 13 0.005 1.0 0.0 NOFILE",
        f"EX 0 1 {2 * r} 2 1.0 0.0",
        f"FR 0 1 0 0 {FREQ_MHZ} 0",
        "XQ 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"
