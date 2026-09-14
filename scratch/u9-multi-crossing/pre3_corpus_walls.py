"""Geometry-only walls for the three multi-node corpus decks, on the kwargs
antennaknobs' momwire engine builds (refine 1). No fill, no Z."""

import math
from collections import Counter

import numpy as np

import momwire
from momwire import _sommerfeld_below as sb
from momwire import bspline
from momwire.bspline import BSplineSolver

bspline.below_reach_refusal = lambda *a, **k: None  # past AK's vertex preflight
REAL_INIT = BSplineSolver.__init__
calls = []


class Stop(Exception):
    pass


def init(self, *a, **kw):
    calls.append(kw)
    raise Stop


from antennaknobs.cli import _GROUND_UNSET, file_ground_default, make_engine_factory  # noqa: E402
from antennaknobs.file_designs import builder_from_file  # noqa: E402

print(momwire.__file__)
ROOT = "/home/smburns/antennas/nec-wild/community/cebik-w4rnl/models/"
for rel in (
    "Phased-Arrays/nec/1r5-bc3elendfire-burrad.nec",
    "Phased-Arrays/nec/1r8-4el-endfire-burrad.nec",
    "LPDAs/nec/lpma3r5-4-6el86ft75o-buriedradials.nec",
):
    calls.clear()
    BSplineSolver.__init__ = init
    b = builder_from_file(ROOT + rel, refine=1)
    try:
        eng = make_engine_factory("momwire", file_ground_default(_GROUND_UNSET, b))
        eng(b()).impedance()
        print(rel, "NO STOP")
        continue
    except Stop:
        pass
    except Exception as exc:  # noqa: BLE001 - a record, not a handler
        print(rel, "CAPTURE RAISED", type(exc).__name__, str(exc)[:200])
        continue
    finally:
        BSplineSolver.__init__ = REAL_INIT
    kw = calls[0]
    try:
        s = BSplineSolver(**kw)
    except Exception as exc:  # noqa: BLE001 - a record, not a handler
        print(rel, "INIT RAISED", type(exc).__name__, str(exc)[:200])
        continue
    geom = s._build_geometry()
    below = s._below_segments(geom)
    a_idx, b_idx = np.nonzero(~below)[0], np.nonzero(below)[0]
    obs_b = s._buried_nodes(geom, b_idx)[0]
    d_b = float(kw.get("ground_z", 0.0)) - obs_b[:, 2]
    r1, th = bspline._pair_extents_below(obs_b[:, 0], obs_b[:, 1], d_b)
    _e, _em, k_p, k_m, _c2, _am = s._buried_medium()
    lam_m = 2 * math.pi / abs(k_m)
    media = s._wire_media()
    radii = np.asarray(s._radius_per_wire)
    by_side = {
        m: Counter(np.round(radii[[i for i, x in enumerate(media) if x == m]], 6))
        for m in set(media)
    }
    groups = s.junctions
    grounded = s._grounded_junctions()
    nodes = [j for j in grounded if len({media[w] for w, _e2 in groups[j]}) == 2]
    above_per = Counter(
        sum(1 for w, _e2 in groups[j] if media[w] != "below") for j in nodes
    )
    xy = sorted(
        {
            (
                round(
                    float(
                        np.asarray(s.wires_polylines[groups[j][0][0]])[
                            0 if groups[j][0][1] == "start" else -1
                        ][0]
                    ),
                    3,
                ),
                round(
                    float(
                        np.asarray(s.wires_polylines[groups[j][0][0]])[
                            0 if groups[j][0][1] == "start" else -1
                        ][1]
                    ),
                    3,
                ),
            )
            for j in nodes
        }
    )
    print(
        f"== {rel}\n  segs={len(geom['seg_l'])} below_nodes={obs_b.shape[0]} crossing_nodes={len(nodes)} above_members_per_node={dict(above_per)}"
    )
    print(f"  node xy={xy[:8]}")
    print(f"  radii by side={ {k: dict(v) for k, v in by_side.items()} }")
    print(
        f"  min below-node depth={d_b.min():.4e} m  th_min={math.degrees(th):.4f} deg (floor {sb._SOMM_BELOW_TH_MIN_DEG})  r1_max={r1:.2f} m = {r1 / lam_m:.2f} lam_m (lam_m {lam_m:.3f})"
    )
    print(f"  refusal on this main: {str(s.buried_serve_refusal())[:180]}")
