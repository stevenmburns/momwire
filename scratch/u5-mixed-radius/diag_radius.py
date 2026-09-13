"""U5 diagnosis of step (b): where does the rise's radius go?

Step (b) found observer-side momwire's R insensitive to the buried rise's radius
and sensitive to the above wire's, against NEC-5 and the ln(ratio)/L physics.
Registered in MEASUREMENTS.md before this ran:

  d0        resolved settings on step (b)'s ratio-2 deck (L 0.30, r 1, observer):
            `_radius_per_wire` with each wire's z-range, per-row radii reaching
            the buried same-medium block builders, and the z-range of the
            crossing fill's a_idx
  d1        no crossing: a wholly buried two-wire rod, lower half's radius varied
  d2below   step (b)'s rod with the radius step 0.10 m below the node (node one radius)
  d2above   the rise and the fed wire stepped, the radiator above at 0.25 mm

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/u5-mixed-radius/diag_radius.py MODE [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

import b_rod_ladder as bl
import check2_continuity as c2
import numpy as np

from antennaknobs.designs.verticals.buried_radial_vertical import Builder
from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine
from antennaknobs.wire_catalog import Wire, WireSpec, graded_wire
from momwire import _crossing_fill
from momwire.bspline import BSplineSolver

SOIL_A = ("finite", 13.0, 0.005)
A0 = 0.00025


def _z_range(geom, idx):
    z = 0.5 * (geom["seg_l"][idx][:, 2] + geom["seg_r"][idx][:, 2])
    return float(z.min()), float(z.max())


def d0():
    a_top, a_rise = bl.CONFIGS["ratio2"]
    b = bl.build(0.30, 1, a_top, a_rise)
    log = {"J": [], "chunk": [], "wires": None, "a_idx_z": [], "b_idx_z": []}
    orig_j = BSplineSolver._build_J_blocks_subset
    orig_chunk = BSplineSolver._accumulate_Z_subset_chunked
    orig_ctx = BSplineSolver._crossing_context

    def j_blocks(self, geom, k, seg_idx, mirror_sources=False):
        a_row = self._seg_radius(geom)[seg_idx]
        log["J"].append(
            dict(
                mirror=bool(mirror_sources),
                n=int(len(seg_idx)),
                z=_z_range(geom, seg_idx),
                radii=sorted({round(float(x), 12) for x in a_row}),
            )
        )
        return orig_j(self, geom, k, seg_idx, mirror_sources)

    def chunk(self, Z, geom, k, seg_idx, *args, **kwargs):
        a_row = self._seg_radius(geom)[seg_idx]
        log["chunk"].append(
            dict(
                mirror=bool(kwargs.get("mirror_sources", False)),
                n=int(len(seg_idx)),
                z=_z_range(geom, seg_idx),
                radii=sorted({round(float(x), 12) for x in a_row}),
            )
        )
        return orig_chunk(self, Z, geom, k, seg_idx, *args, **kwargs)

    def ctx_log(self, geom, supp_seg, polys):
        log["wires"] = [
            dict(
                wire=w,
                radius=float(self._radius_per_wire[w]),
                z=(
                    float(np.asarray(p)[:, 2].min()),
                    float(np.asarray(p)[:, 2].max()),
                ),
            )
            for w, p in enumerate(self.wires_polylines)
        ]
        return orig_ctx(self, geom, supp_seg, polys)

    inner_cross = None

    def cross_log(ctx, a_idx, b_idx, A, B, **kw):
        geom = dict(seg_l=ctx.geom.seg_l, seg_r=ctx.geom.seg_r)
        log["a_idx_z"].append(_z_range(geom, a_idx))
        log["b_idx_z"].append(_z_range(geom, b_idx))
        return inner_cross(ctx, a_idx, b_idx, A, B, **kw)

    n5 = bl.nec5_segs(b)
    BSplineSolver._build_J_blocks_subset = j_blocks
    BSplineSolver._accumulate_Z_subset_chunked = chunk
    # ctx_log must see the solver before c2's ctx_min wraps it
    BSplineSolver._crossing_context = ctx_log
    c2._orig_ctx = ctx_log
    c2.install()
    inner_cross = _crossing_fill.cross_complete_block_split
    _crossing_fill.cross_complete_block_split = cross_log
    try:
        rec, _z = bl.momwire_rung(b, a_top, a_rise, "observer", n5)
    finally:
        c2.uninstall()
        BSplineSolver._build_J_blocks_subset = orig_j
        BSplineSolver._accumulate_Z_subset_chunked = orig_chunk
        BSplineSolver._crossing_context = orig_ctx
    out = dict(momwire=rec, **log)
    print(json.dumps(out, indent=1), flush=True)
    return out


class BuriedTwoRadiusRod(Builder):
    default_params = MappingProxyType(
        {
            **Builder.default_params,
            "rod_len": 0.60,
            "rod_depth": 0.20,
            "refine": 8,
            "a_low": A0,
            "a_high": A0,
        }
    )

    def build_wires(self):
        eps = 0.005
        top = -abs(self.rod_depth)
        bot = top - abs(self.rod_len)
        mid = 0.5 * (top + bot)
        lo, hi = (0.0, 0.0, mid - eps), (0.0, 0.0, mid + eps)
        h = 0.025 / int(self.refine)
        low = WireSpec(radius=float(self.a_low))
        high = WireSpec(radius=float(self.a_high))
        return [
            graded_wire((0.0, 0.0, bot), lo, toward="p1", rest_h=h, spec=low),
            Wire(lo, hi, ex=1 + 0j, spec=high),
            graded_wire(hi, (0.0, 0.0, top), toward="p0", rest_h=h, spec=high),
        ]


class SteppedCrossingRod(Builder):
    default_params = MappingProxyType(
        {
            **Builder.default_params,
            "uniform_r": 1,
            "step": "below",
            "a_step": A0,
            "a_top": A0,
        }
    )

    def build_wires(self):
        r = int(self.uniform_r)
        height = 0.25 * self.design_wavelength * self.length_factor
        node = (0.0, 0.0, 0.0)
        hub = (0.0, 0.0, -self.depth)
        top = WireSpec(radius=float(self.a_top))
        stp = WireSpec(radius=float(self.a_step))
        rest = 0.25 * self.design_wavelength / self.nominal_nsegs / r
        if self.step == "below":
            z_s = -0.10
            n_low = max(2, round((self.depth - 0.10) / (bl.REST_H / r)))
            return [
                Wire(hub, (0.0, 0.0, z_s), n_seg=n_low, spec=stp),
                graded_wire(
                    (0.0, 0.0, z_s),
                    node,
                    toward="p1",
                    per_panel=2 * r,
                    rest_h=bl.REST_H / r,
                    spec=top,
                ),
                Wire(
                    node,
                    (0.0, 0.0, bl.EPS),
                    n_seg=None if r == 1 else 2 * r,
                    ex=1 + 0j,
                    spec=top,
                ),
                graded_wire(
                    (0.0, 0.0, bl.EPS),
                    (0.0, 0.0, height),
                    toward="p0",
                    per_panel=2 * r,
                    rest_h=rest,
                    spec=top,
                ),
            ]
        return [
            graded_wire(
                hub, node, toward="p1", per_panel=2 * r, rest_h=bl.REST_H / r, spec=stp
            ),
            Wire(
                node,
                (0.0, 0.0, bl.EPS),
                n_seg=None if r == 1 else 2 * r,
                ex=1 + 0j,
                spec=stp,
            ),
            graded_wire(
                (0.0, 0.0, bl.EPS),
                (0.0, 0.0, height),
                toward="p0",
                per_panel=2 * r,
                rest_h=rest,
                spec=top,
            ),
        ]


@contextmanager
def node_radius_scope(a_node):
    """Hand the scope check uniform radii and give the crossing context the
    node's one radius; no rule patch (the node has one radius)."""
    orig_scope = c2._below_interface.crossing_junctions
    orig_ctx = BSplineSolver._crossing_context

    def ctx(self, geom, supp_seg, polys):
        return orig_ctx(self, geom, supp_seg, polys)._replace(a_wire=a_node)

    c2._below_interface.crossing_junctions = c2.scope
    BSplineSolver._crossing_context = ctx
    try:
        yield
    finally:
        c2._below_interface.crossing_junctions = orig_scope
        BSplineSolver._crossing_context = orig_ctx


def both(b, a_node=None):
    n5 = bl.nec5_segs(b)
    zn = complex(NEC5Engine(b, ground=SOIL_A).impedance()[0])
    with bl.even_parity():
        eng = MomwireEngine(b, ground=SOIL_A)
        segs, on_knot = bl.port_check(eng)
        if segs != n5 or not on_knot:
            raise RuntimeError(f"port not matched: {segs} vs {n5}, knot {on_knot}")
        if a_node is None:
            zm = complex(eng.impedance()[0])
        else:
            with node_radius_scope(a_node):
                zm = complex(eng.impedance()[0])
    return zm, zn, n5


def ladder(make, refines, a_node_of=None):
    rows = {}
    for r in refines:
        b = make(r)
        zm, zn, n5 = both(b, None if a_node_of is None else a_node_of)
        rows[r] = dict(momwire=str(zm), nec5=str(zn), segs=n5)
    lo, hi = refines
    rm = 2 * complex(rows[hi]["momwire"]).real - complex(rows[lo]["momwire"]).real
    rn = 2 * complex(rows[hi]["nec5"]).real - complex(rows[lo]["nec5"]).real
    return rows, rm, rn


def response_table(label, radii, runs):
    out = []
    base_m, base_n = runs[radii[0]][1], runs[radii[0]][2]
    for a in radii:
        _rows, rm, rn = runs[a]
        dm, dn = rm - base_m, rn - base_n
        ratio = dm / dn if abs(dn) > 1e-9 else float("nan")
        out.append(
            dict(
                a=a,
                R_inf_momwire=rm,
                R_inf_nec5=rn,
                dR_momwire=dm,
                dR_nec5=dn,
                ratio=ratio,
            )
        )
        print(
            f"{label} a={a * 1000:.4f} mm  R_inf momwire {rm:.4f}  NEC-5 {rn:.4f}  "
            f"response momwire {dm:+.3f}  NEC-5 {dn:+.3f}  ratio {ratio:.3f}",
            flush=True,
        )
    return out


def d1():
    radii = (A0, A0 / 2, A0 / 4)
    runs = {}
    for a in radii:

        def make(refine, a=a):
            b = BuriedTwoRadiusRod()
            b.nominal_nsegs = 42
            b.design_eps_r, b.design_sigma = 13.0, 0.005
            b.refine = refine
            b.a_low = a
            return b

        runs[a] = ladder(make, (8, 16))
    return dict(
        runs={str(a): runs[a][0] for a in radii},
        table=response_table("d1", radii, runs),
    )


def d2(step):
    radii = (A0, A0 / 2, A0 / 4) if step == "below" else (A0, A0 / 2)
    runs = {}
    for a in radii:

        def make(r, a=a):
            b = SteppedCrossingRod()
            b.nominal_nsegs = 42
            b.design_eps_r, b.design_sigma = 13.0, 0.005
            b.depth = 0.30
            b.uniform_r = r
            b.step = step
            b.a_step = a
            return b

        a_node = A0 if step == "below" else a
        runs[a] = ladder(make, (2, 4), a_node_of=a_node)
    return dict(
        runs={str(a): runs[a][0] for a in radii},
        table=response_table(f"d2{step}", radii, runs),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["d0", "d1", "d2below", "d2above"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        mode=args.mode,
        nec5_exe=exe,
        nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
    )
    print(meta, flush=True)
    if args.mode == "d0":
        result = d0()
    elif args.mode == "d1":
        result = d1()
    elif args.mode == "d2below":
        result = d2("below")
    else:
        result = d2("above")
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, result=result), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
