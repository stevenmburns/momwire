"""U9 (d) step 0 (geometry only, no fill, no Z): write the LPDA's one-node
spelling, and preflight every ladder rung of both decks.

  PYTHONPATH=<momwire src>:<antennaknobs src> python d0_lpda_decks.py --out F

The one-node spelling (Amendment 3):
- the fed element (x = 26.2128 m, GW 43-48) keeps its rise and its 24 radials,
  GW 861-976 (each element's screen is the 116-card block starting at its rise:
  49, 165, 281, 397, 513, 629, 745, 861);
- the other seven elements lose their rise and radials, and their node-adjacent
  wire (GW 6, 12, ..., 42; z 0.1524 -> 0) is shortened to end at z = 0.0762, so
  no element ends on the plane (a GE -1 free end there is refused), AND is split
  into TWO segments (Amendment 6): with one segment its free end left the TL
  port on segment 1 with no basis function, and native NEC-5 stopped on it;
- every other card (the elements, the TL network, the EX) is unchanged.

Each rung is built through antennaknobs' `builder_from_file(path, refine=r)`, the
U2 ladder's own path. The solver kwargs are captured before the solver runs,
and theta_min is read on the solver's own quadrature nodes. antennaknobs' AK#1464
vertex preflight (`momwire.bspline.below_reach_refusal`) is stubbed so the
engine can be constructed; it is not what this measures.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

import momwire
from momwire import bspline
from momwire.bspline import BSplineSolver

bspline.below_reach_refusal = lambda *a, **k: None  # AK#1464 stub (named in PLAN.md)
REAL_INIT = BSplineSolver.__init__
CAPTURED = []

HERE = Path(__file__).resolve().parent
FULL = (
    Path.home()
    / "antennas/nec-wild/community/cebik-w4rnl/models/LPDAs/nec"
    / "lpma3r5-4-6el86ft75o-buriedradials.nec"
)
ONE = HERE / "lpda_one_node.nec"
FED_SCREEN_FIRST_TAG = 861
FED_NODE_WIRE_TAG = 48
LIFT_Z = 0.0762
LIFT_SEGMENTS = 2  # Amendment 6
RUNGS = (1, 3, 9)


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


def fields(line):
    return [x for x in re.split(r"[,\s]+", line.strip()) if x]


def write_one_node():
    """The one-node spelling, card for card from the full deck."""
    out = []
    removed = lifted = 0
    gone = set()
    for line in FULL.read_text().splitlines():
        f = fields(line)
        if f and f[0] == "LD" and int(f[2]) in gone:
            continue
        if f and f[0] == "GW":
            tag = int(f[1])
            z1, z2 = float(f[5]), float(f[8])
            below = min(z1, z2) < 0.0
            if below and tag < FED_SCREEN_FIRST_TAG:
                # A rise or radial of an unfed element's screen. Screens are
                # told apart by TAG BLOCK, not by position: radials run 24 m and
                # cross the neighbouring nodes' x.
                removed += 1
                gone.add(tag)
                continue
            if not below and z2 == 0.0 and tag != FED_NODE_WIRE_TAG:
                f[8] = f"{LIFT_Z}"
                f[2] = str(LIFT_SEGMENTS)
                line = ",".join(f)
                lifted += 1
        out.append(line)
    assert (removed, lifted) == (7 * 116, 7), (removed, lifted)
    ONE.write_text("\n".join(out) + "\n")
    return removed, lifted


NODE_WIRE_TAGS = frozenset(range(6, 49, 6))
RISE_TAGS = frozenset(49 + 116 * k for k in range(8))


def write_far3(src, dst):
    """Every GW count x 3 except each element's node-adjacent wire (GW 6, 12,
    ..., 48, which also carry the TL network and the EX on segment 1) and each
    screen's rise (GW 49, 165, ..., 861): those set the shallowest below
    quadrature nodes, so keeping them keeps theta_min (Amendment 4's idea,
    applied to a deck)."""
    out, scaled = [], 0
    for line in src.read_text().splitlines():
        f = fields(line)
        if f and f[0] == "GW":
            tag = int(f[1])
            if tag not in NODE_WIRE_TAGS and tag not in RISE_TAGS:
                f[2] = str(3 * int(f[2]))
                line = ",".join(f)
                scaled += 1
        out.append(line)
    dst.write_text("\n".join(out) + "\n")
    return scaled


def extents(kw):
    s = BSplineSolver(**kw)
    geom = s._build_geometry()
    below = s._below_segments(geom)
    b_idx = np.nonzero(below)[0]
    obs_b = s._buried_nodes(geom, b_idx)[0]
    d_b = float(kw.get("ground_z", 0.0)) - obs_b[:, 2]
    r1, th = bspline._pair_extents_below(obs_b[:, 0], obs_b[:, 1], d_b)
    lam_m = 2 * math.pi / abs(s._buried_medium()[3])
    return dict(
        segments=int(len(geom["seg_l"])),
        below_nodes=int(obs_b.shape[0]),
        crossing_junctions=len(s._crossing_junctions()),
        th_min_deg=math.degrees(th),
        r1_max_lam_m=r1 / lam_m,
        refusal=s.buried_serve_refusal(),
    )


def preflight(path, r):
    CAPTURED.clear()
    BSplineSolver.__init__ = capture
    row = dict(deck=path.name, refine=r)
    try:
        b = builder_from_file(str(path), refine=r)
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
    removed, lifted = write_one_node()
    far_full = HERE / "lpda_full_far3.nec"
    far_one = HERE / "lpda_one_node_far3.nec"
    rec = dict(
        momwire=momwire.__file__,
        one_node=dict(removed_gw=removed, lifted_gw=lifted),
        far3_scaled_gw=dict(
            full=write_far3(FULL, far_full), one=write_far3(ONE, far_one)
        ),
    )
    print(json.dumps(rec), flush=True)
    rows = []
    for path, rungs in ((FULL, RUNGS), (far_full, (1,)), (ONE, RUNGS), (far_one, (1,))):
        for r in rungs:
            row = preflight(path, r)
            rows.append(row)
            print(json.dumps(row), flush=True)
    rec["rungs"] = rows
    args.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
