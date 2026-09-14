"""U9 fill-cost timing, step 0 (geometry only, no fill, no Z): which decks reach
the below/below LOW band (theta < 0.1 deg), on the kwargs antennaknobs' momwire
engine builds for each catalog design at its defaults, the 48-radial screen of
the #983 harness, and #935's 3 mm dipole.

  PYTHONPATH=<momwire src>:<antennaknobs src> python f0_timing_preflight.py --out F

antennaknobs' AK#1464 vertex preflight is stubbed (`below_reach_refusal`) only so
the solver can be constructed on decks it over-refuses; nothing is solved.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import momwire
from momwire import bspline
from momwire.bspline import BSplineSolver

bspline.below_reach_refusal = lambda *a, **k: None  # AK#1464 stub, construction only
REAL_INIT = BSplineSolver.__init__
CAPTURED = []
SOIL_A = ("finite", 13.0, 0.005)


class Stop(Exception):
    pass


def capture(self, *a, **kw):
    CAPTURED.append(kw)
    raise Stop


import antennaknobs.web.examples  # noqa: E402, F401 -- binds register_all first
from antennaknobs.builder import resolve_variant_params  # noqa: E402
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402

CATALOG = (
    ("specialty.buried_dipole", "default", {}),
    ("verticals.buried_radial_vertical", "default", {}),
    ("verticals.buried_radial_vertical", "bundle", {}),
    ("verticals.buried_radial_vertical", "detached", {}),
    ("verticals.elevated_buried_counterpoise", "default", {}),
    ("verticals.buried_radial_vertical", "default", {"n_radials": 48}),
)


def catalog_builder(design, variant, overrides):
    cls = importlib.import_module(f"antennaknobs.designs.{design}").Builder
    b = (
        cls(params=dict(resolve_variant_params(cls, variant)))
        if variant != "default"
        else cls()
    )
    for k, v in overrides.items():
        setattr(b, k, v)
    return b


def extents(kw):
    s = BSplineSolver(**kw)
    geom = s._build_geometry()
    b_idx = np.nonzero(s._below_segments(geom))[0]
    obs_b = s._buried_nodes(geom, b_idx)[0]
    d_b = float(kw.get("ground_z", 0.0)) - obs_b[:, 2]
    r1, th = bspline._pair_extents_below(obs_b[:, 0], obs_b[:, 1], d_b)
    lam_m = 2 * math.pi / abs(s._buried_medium()[3])
    return dict(
        segments=int(len(geom["seg_l"])),
        below_nodes=int(obs_b.shape[0]),
        th_min_deg=math.degrees(th),
        r1_max_m=r1,
        r1_max_lam_m=r1 / lam_m,
        reaches_low_band=math.degrees(th) < 0.1,
        under_old_floor=math.degrees(th) < 0.05,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for design, variant, overrides in CATALOG:
        CAPTURED.clear()
        BSplineSolver.__init__ = capture
        row = dict(design=design, variant=variant, overrides=overrides)
        try:
            MomwireEngine(
                catalog_builder(design, variant, overrides), ground=SOIL_A
            ).impedance()
            row["capture"] = "NO STOP"
        except Stop:
            pass
        except Exception as exc:  # noqa: BLE001 - a record, not a handler
            row["capture"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            BSplineSolver.__init__ = REAL_INIT
        if CAPTURED:
            try:
                row.update(extents(CAPTURED[0]))
            except Exception as exc:  # noqa: BLE001 - a record, not a handler
                row["extents"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        rows.append(row)
        print(json.dumps(row), flush=True)
    sys.path.insert(0, str(Path(momwire.__file__).resolve().parents[2] / "tests"))
    from test_grazing_band_lo_935 import _SHAPE_HALF, _SHAPE_RAD, _SHAPE_SOIL, _SHAPE_WL

    wire = np.array([[-_SHAPE_HALF, 0.0, -0.003], [_SHAPE_HALF, 0.0, -0.003]])
    kw = dict(
        wires=[wire],
        n_per_edge_per_wire=[[41]],
        wire_radius=_SHAPE_RAD,
        wavelength=_SHAPE_WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=0,
        feed_arclength=_SHAPE_HALF,
        ground_z=0.0,
        ground_eps=_SHAPE_SOIL,
        ground_model="sommerfeld",
    )
    row = dict(design="momwire test_grazing_band_lo_935._shape_z", variant="3 mm, n=41")
    row.update(extents(kw))
    rows.append(row)
    print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(dict(momwire=momwire.__file__, rows=rows), indent=1))


if __name__ == "__main__":
    main()
