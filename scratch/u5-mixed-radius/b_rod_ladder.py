"""U5 step (b): the two-radius crossing rod against NEC-5.

#931's crossing rod (graded rise from the hub at -L to the node, a 50 mm fed
wire from the node up with its source at the centre, graded radiator to
lambda/4), soil A, 7.1 MHz. Radii per wire: rise a_B; fed wire and radiator a_A.
Uniform refinement r = 1, 2, 4 with the graded panel boundaries held
(`per_panel` = 2r, fed wire 2r segments, far panels / r), so the source region
refines with everything else.

momwire runs the observer-side rule of DERIVATION-MIXED-RADIUS.md through
check 2's harness patches, with its feed parity patched to even so both engines
share one mesh and one knot source. Asserted at every momwire rung:
`extended_kernel` False, momwire's segment count equal to NEC-5's, the feed on a
knot, the crossing fill entered once. The equal-radius control through the
patched path must reproduce the unpatched solver bit for bit at each L. Read at
every rung beside Z: the node KCL deficit and the slope ratio against 1/eps_t.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/u5-mixed-radius/b_rod_ladder.py CONFIG [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

import check2_continuity as c2
import numpy as np

import antennaknobs.engines.momwire as mwe
from antennaknobs.designs.verticals.buried_radial_vertical import Builder
from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine
from antennaknobs.wire_catalog import Wire, WireSpec, graded_wire

SOIL_A = ("finite", 13.0, 0.005)
REST_H = 0.025
EPS = 0.05
CONFIGS = {
    "control": (0.00025, 0.00025),
    "ratio2": (0.00025, 0.000125),
    "ratio4": (0.00025, 0.0000625),
    "ratio05": (0.000125, 0.00025),
}


class TwoRadiusRod(Builder):
    default_params = MappingProxyType(
        {
            **Builder.default_params,
            "uniform_r": 1,
            "a_top": 0.00025,
            "a_rise": 0.00025,
        }
    )

    def build_wires(self):
        r = int(self.uniform_r)
        height = 0.25 * self.design_wavelength * self.length_factor
        node = (0.0, 0.0, 0.0)
        hub = (0.0, 0.0, -self.depth)
        top = WireSpec(radius=float(self.a_top))
        rise = WireSpec(radius=float(self.a_rise))
        return [
            graded_wire(
                hub, node, toward="p1", per_panel=2 * r, rest_h=REST_H / r, spec=rise
            ),
            Wire(
                node,
                (0.0, 0.0, EPS),
                n_seg=None if r == 1 else 2 * r,
                ex=1 + 0j,
                spec=top,
            ),
            graded_wire(
                (0.0, 0.0, EPS),
                (0.0, 0.0, height),
                toward="p0",
                per_panel=2 * r,
                rest_h=0.25 * self.design_wavelength / self.nominal_nsegs / r,
                spec=top,
            ),
        ]


def build(L, r, a_top, a_rise, nn=42):
    b = TwoRadiusRod()
    b.nominal_nsegs = nn
    b.design_eps_r, b.design_sigma = 13.0, 0.005
    b.depth = L
    b.uniform_r = r
    b.a_top = a_top
    b.a_rise = a_rise
    return b


@contextmanager
def even_parity():
    orig = mwe._parity_for_solver
    mwe._parity_for_solver = lambda solver, solver_kwargs: "even"
    try:
        yield
    finally:
        mwe._parity_for_solver = orig


def nec5_segs(b):
    deck = NEC5Engine(b, ground=SOIL_A).deck([b.freq])
    return sum(int(ln.split()[2]) for ln in deck.splitlines() if ln.startswith("GW "))


def node_readout(sim, coeffs):
    """KCL deficit and slope ratio at the node, orientation-aware."""
    polys = [np.asarray(p, dtype=float) for p in sim.wires_polylines]
    s_arrays = [np.zeros(0) for _ in polys]
    below = above = None
    for w, p in enumerate(polys):
        z0, z1 = p[0][2], p[-1][2]
        length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        if abs(z1) < 1e-12 and z0 < 0:
            below = (w, length, +1.0)
        elif abs(z0) < 1e-12 and z1 < 0:
            below = (w, 0.0, -1.0)
        elif abs(z0) < 1e-12 and z1 > 0:
            above = (w, 0.0, +1.0)
        elif abs(z1) < 1e-12 and z0 > 0:
            above = (w, length, -1.0)
    if below is None or above is None:
        raise RuntimeError("could not find the two node members")
    for w, s, _sg in (below, above):
        s_arrays[w] = np.array([s])
    cur = sim.currents_at_knots(coeffs, s_array=s_arrays)
    slo = sim.current_slopes(coeffs, s_array=s_arrays)
    i_m = below[2] * complex(cur[below[0]][0])
    i_p = above[2] * complex(cur[above[0]][0])
    d_m = complex(slo[below[0]][0])
    d_p = complex(slo[above[0]][0])
    eps_t = sim._buried_medium()[0]
    ratio = d_p / d_m
    return dict(
        kcl_rel=abs(i_p - i_m) / abs(i_p),
        slope_ratio=str(ratio),
        slope_vs_agard=abs(ratio - 1 / eps_t) / abs(1 / eps_t),
    )


def port_check(eng):
    w, s_f, _v = eng._feeds[0]
    poly = np.asarray(eng._polylines[w], dtype=float)
    knots = mwe._polyline_knots(poly, eng._edge_segments[w])
    arcs = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))]
    )
    segs = sum(sum(e) for e in eng._edge_segments)
    return segs, bool(np.min(np.abs(arcs - s_f)) < 1e-12)


def momwire_rung(b, a_top, a_rise, rule, n5_segs):
    c2.STATE.update(rule=rule, a_A=a_top, a_B=a_rise, calls=0)
    c2.PENDING.clear()
    with even_parity():
        eng = MomwireEngine(b, ground=SOIL_A)
        segs, on_knot = port_check(eng)
        if segs != n5_segs or not on_knot:
            raise RuntimeError(
                f"port not matched: momwire {segs} vs NEC-5 {n5_segs}, knot {on_knot}"
            )
        sim, coeffs, z = eng._solved_excited(eng._wavelength_for(b.freq))
    if sim.extended_kernel:
        raise RuntimeError("extended_kernel resolved True")
    rec = dict(
        rule=rule, z=str(complex(z)), segs=segs, crossing_calls=c2.STATE["calls"]
    )
    rec.update(node_readout(sim, coeffs))
    return rec, complex(z)


def unpatched_z(b):
    with even_parity():
        eng = MomwireEngine(b, ground=SOIL_A)
        return complex(eng.impedance()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=sorted(CONFIGS))
    ap.add_argument("--lengths", type=float, nargs="+", default=[0.30, 0.60, 1.20])
    ap.add_argument("--refines", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    a_top, a_rise = CONFIGS[args.config]
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        config=args.config,
        a_top=a_top,
        a_rise=a_rise,
        nec5_exe=exe,
        nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
    )
    print(meta, flush=True)
    out = []
    for L in args.lengths:
        if args.config == "control":
            b = build(L, args.refines[0], a_top, a_rise)
            z_ref = unpatched_z(b)
            c2.install()
            try:
                _rec, z_pat = momwire_rung(b, a_top, a_rise, "observer", nec5_segs(b))
            finally:
                c2.uninstall()
            if z_pat != z_ref:
                raise SystemExit(
                    f"control at L={L}: patched {z_pat!r} != unpatched {z_ref!r}; stop"
                )
            print(f"L={L} control bit-identical to unpatched: {z_pat}", flush=True)
        for r in args.refines:
            b = build(L, r, a_top, a_rise)
            n5 = nec5_segs(b)
            zn = complex(NEC5Engine(b, ground=SOIL_A).impedance()[0])
            c2.install()
            try:
                rec, zm = momwire_rung(b, a_top, a_rise, "observer", n5)
                naive = None
                if r == 2 and args.config != "control":
                    naive, _zn = momwire_rung(b, a_top, a_rise, "single_B", n5)
            finally:
                c2.uninstall()
            if rec["crossing_calls"] != 1:
                raise RuntimeError(
                    f"crossing fill entered {rec['crossing_calls']} times"
                )
            row = dict(L=L, r=r, nec5=str(zn), nec5_segs=n5, momwire=rec, naive=naive)
            out.append(row)
            msg = (
                f"L={L:4.2f} r={r} segs={n5}  NEC-5 {zn:.6g}  momwire {zm:.6g}  "
                f"dZ {zn - zm:.4g}  kcl {rec['kcl_rel']:.2e}  slope-vs-AGARD "
                f"{rec['slope_vs_agard']:.2e}"
            )
            if naive is not None:
                zn_naive = complex(naive["z"])
                msg += f"  | naive single_B {zn_naive:.6g} dZ {zn - zn_naive:.4g} kcl {naive['kcl_rel']:.2e}"
            print(msg, flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
