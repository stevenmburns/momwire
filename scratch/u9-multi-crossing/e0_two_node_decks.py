"""U9 Amendment 8, step 0 (geometry only, no fill, no Z): write (c)'s two-node
soil-A decks as NEC card decks, one per (separation, rung, excitation), and
preflight each momwire deck's below/below grazing angle on the kwargs
antennaknobs' momwire engine builds.

  PYTHONPATH=<momwire src>:<antennaknobs src> python e0_two_node_decks.py --out F

Geometry is `test_crossing_serve_524.crossing_deck` (the rod 2 m down to the node,
the 10 m monopole up from it, radius 1 mm) twice, the second pair shifted d
along x: (c)'s deck, spelled as cards so both engines read the same bytes.

Rungs:
  r1     the (c) mesh (node grading level 1);
  far3   every edge count x 3 except each wire's node-adjacent edge (keeps
         theta_min; Amendment 4);
  node2  node grading level 2 (preflight only: expected refused by the floor);
  all3   every edge count x 3 (NEC-5 only: it has no grazing floor).

Excitations: `both` (EX on each monopole: momwire's 2-port input) and `p1` /
`p2` (one EX each: NEC-5's one-port runs). Every port is the segment centred at
z = 4.25 m on its monopole's long edge, so the feed does not move across rungs.
antennaknobs' AK#1464 vertex preflight is stubbed for construction only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import momwire
from momwire import bspline
from momwire.bspline import BSplineSolver

bspline.below_reach_refusal = lambda *a, **k: None  # AK#1464 stub (PLAN.md Amendment 3)
REAL_INIT = BSplineSolver.__init__
CAPTURED = []

HERE = Path(__file__).resolve().parent
DECK_DIR = HERE / "a8_decks"
SEPARATIONS = (3.0, 5.0, 8.0, 11.0)
RADIUS = 0.001
FEED_Z = 4.25
GRADES = {
    1: dict(
        below=([-2.0, -0.5, -0.1], [3, 2, 2]), above=([0.1, 0.5, 10.0], [2, 2, 19])
    ),
    2: dict(
        below=([-2.0, -0.5, -0.1, -0.025], [3, 2, 3, 2]),
        above=([0.025, 0.1, 0.5, 10.0], [2, 3, 2, 19]),
    ),
}


class Stop(Exception):
    pass


def capture(self, *a, **kw):
    CAPTURED.append(kw)
    raise Stop


from antennaknobs.cli import (  # noqa: E402
    _GROUND_UNSET,
    file_ground_default,
    make_engine_factory,
)
from antennaknobs.file_designs import builder_from_file  # noqa: E402


def edges(rung):
    """[(z0, z1, n, node_adjacent)] for one rod (bottom -> node) and one
    monopole (node -> top), with the rung's segment counts."""
    g = GRADES[2 if rung == "node2" else 1]
    bz = g["below"][0] + [0.0]
    az = [0.0] + g["above"][0]
    rod = [
        (bz[i], bz[i + 1], g["below"][1][i], i == len(bz) - 2)
        for i in range(len(bz) - 1)
    ]
    mono = [(az[i], az[i + 1], g["above"][1][i], i == 0) for i in range(len(az) - 1)]

    def scale(ws, far_only):
        return [
            (z0, z1, n * 3 if (not far_only or not adj) else n, adj)
            for z0, z1, n, adj in ws
        ]

    if rung in ("far3", "all3"):
        rod, mono = scale(rod, rung == "far3"), scale(mono, rung == "far3")
    return rod, mono


def deck_text(d, rung, excite):
    rod, mono = edges(rung)
    lines = [
        f"CM U9 amendment 8: two-node soil-A deck, d = {d:g} m, {rung}, EX {excite}",
        "CE",
    ]
    tag = 0
    ports = []
    for x in (0.0, d):
        for z0, z1, n, _adj in rod:
            tag += 1
            lines.append(
                f"GW {tag} {n} {x:.6E} 0.000000E+00 {z0:.6E} {x:.6E} 0.000000E+00 {z1:.6E} {RADIUS:.6E}"
            )
        for z0, z1, n, _adj in mono:
            tag += 1
            lines.append(
                f"GW {tag} {n} {x:.6E} 0.000000E+00 {z0:.6E} {x:.6E} 0.000000E+00 {z1:.6E} {RADIUS:.6E}"
            )
            if z0 <= FEED_Z < z1:
                h = (z1 - z0) / n
                k = (FEED_Z - z0) / h + 0.5
                assert abs(k - round(k)) < 1e-9, (d, rung, k)
                ports.append((tag, int(round(k))))
    lines.append("GE -1")
    lines.append("GN 2 0 0 0 1.300000E+01 5.000000E-03")
    for i, (t, s) in enumerate(ports, start=1):
        if excite == "both" or excite == f"p{i}":
            lines.append(f"EX 0 {t} {s} 0 1.000000E+00 0.000000E+00")
    lines.append("FR 0 1 0 0 7.000000E+00 0.000000E+00")
    lines += ["XQ 0", "EN"]
    return "\n".join(lines) + "\n", ports


def extents(kw):
    s = BSplineSolver(**kw)
    geom = s._build_geometry()
    b_idx = np.nonzero(s._below_segments(geom))[0]
    obs_b = s._buried_nodes(geom, b_idx)[0]
    d_b = float(kw.get("ground_z", 0.0)) - obs_b[:, 2]
    _r1, th = bspline._pair_extents_below(obs_b[:, 0], obs_b[:, 1], d_b)
    return dict(
        segments=int(len(geom["seg_l"])),
        crossing_junctions=len(s._crossing_junctions()),
        n_feeds=len(kw.get("feeds") or []),
        th_min_deg=math.degrees(th),
        refusal=s.buried_serve_refusal(),
    )


def preflight(path):
    CAPTURED.clear()
    BSplineSolver.__init__ = capture
    row = {}
    try:
        b = builder_from_file(str(path), refine=1)
        make_engine_factory("momwire", file_ground_default(_GROUND_UNSET, b))(
            b()
        ).impedance()
        row["capture"] = "NO STOP"
    except Stop:
        pass
    except Exception as exc:  # noqa: BLE001 - a record, not a handler
        row["capture"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    finally:
        BSplineSolver.__init__ = REAL_INIT
    if CAPTURED:
        try:
            row.update(extents(CAPTURED[0]))
        except Exception as exc:  # noqa: BLE001 - a record, not a handler
            row["extents"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    DECK_DIR.mkdir(exist_ok=True)
    rec = dict(momwire=momwire.__file__, decks=[])
    for d in SEPARATIONS:
        for rung in ("r1", "far3", "node2", "all3"):
            for excite in ("both", "p1", "p2"):
                text, ports = deck_text(d, rung, excite)
                path = DECK_DIR / f"two_node_d{d:g}_{rung}_{excite}.nec"
                path.write_text(text)
                row = dict(
                    d=d,
                    rung=rung,
                    excite=excite,
                    path=str(path.relative_to(HERE)),
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                    ports=ports,
                )
                if excite == "both" and rung != "all3":
                    row["preflight"] = preflight(path)
                rec["decks"].append(row)
                print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
