"""momwire#1149 U1: the detached route's gates, measured through the REAL
constructor (no in-process bypass, unlike scoping probe 5).

Deck: `crossing_deck(1)` pulled apart - the above wire lifted by `gap`, the
buried wire lowered by `gap` and moved `dx` sideways; ports at above
arclength 4.5 and buried arclength 1.0; x`m` multiplies every edge count but
the one at each wire's plane end (probe 5's ladder).

Sections: (a) reciprocity ladder x1..x8 at the probe-5 geometry, (b) eps~ = 1
collapse against the same deck in free space, (c) razor vs bspline Z11 / Z22
/ Z12 ladder, (d) stand-off ladder gap 0.3 -> 0.01 m at dx = 0.5 and dx = 0,
(e) loading on the detached deck vs bspline, (f) the catalog
elevated_buried_counterpoise spelled in momwire kwargs vs bspline.

    python scratch/razor-buried-u1/probe_u1_gates.py [a b c d e f]
"""

import json
import math
import pathlib
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import BSplineSolver, RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402

warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).with_suffix(".json")
C0 = 299792458.0


def deck(m=1, gap=0.3, dx=0.5, ground=True, eps=(13.0, 0.005), **kw):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    below, above = d["wires"]
    d["wires"] = [below + np.array([dx, 0.0, -gap]), above + np.array([0.0, 0.0, gap])]
    d.pop("junctions")
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if ground:
        d["ground_eps"] = eps
    else:
        for k in ("ground_z", "ground_eps", "ground_model"):
            d.pop(k)
    d.update(kw)
    return d


def two_port(cls, d, **kw):
    t0 = time.perf_counter()
    s = cls(**d, **kw)
    Y = np.asarray(s.compute_y_matrix())
    dt = time.perf_counter() - t0
    Z = np.linalg.inv(Y)
    nonrec = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
    return Z, nonrec, dt, s


def c(z):
    return [float(z.real), float(z.imag)]


res = json.loads(OUT.read_text()) if OUT.exists() else {}
want = set(sys.argv[1:]) or set("abcdef")

if "a" in want:
    rows = []
    for m in (1, 2, 4, 8):
        Z, nr, dt, s = two_port(RazorSolver, deck(m), nec5_quadrature=True)
        assert s._detached
        rows.append(
            dict(x=m, nonrec=nr, t=dt, z11=c(Z[0, 0]), z22=c(Z[1, 1]), z12=c(Z[0, 1]))
        )
        print(f"(a) x{m} nonrec {nr:.3e}  {dt:.2f} s", flush=True)
    for r0, r1 in zip(rows, rows[1:]):
        print(f"    ratio x{r0['x']}->x{r1['x']}: {r0['nonrec'] / r1['nonrec']:.2f}")
    res["a_reciprocity"] = rows

if "b" in want:
    rows = []
    for m in (1, 2):
        for lane in (True, False):
            Zf, *_ = two_port(RazorSolver, deck(m, ground=False), nec5_quadrature=lane)
            Z1, *_ = two_port(
                RazorSolver, deck(m, eps=(1.0, 0.0)), nec5_quadrature=lane
            )
            rel = float(np.max(np.abs(Z1 - Zf)) / np.max(np.abs(Zf)))
            rows.append(dict(x=m, nec5=lane, rel=rel))
            print(f"(b) x{m} nec5={lane} eps~=1 vs free: {rel:.3e}", flush=True)
    res["b_collapse"] = rows

if "c" in want:
    rows = []
    for m in (1, 2, 4, 8):
        Zr, nr, tr, _ = two_port(RazorSolver, deck(m), nec5_quadrature=True)
        Zb, nb, tb, _ = two_port(BSplineSolver, deck(m))
        g = {
            k: float(abs(Zr[i, j] - Zb[i, j]))
            for k, (i, j) in dict(z11=(0, 0), z22=(1, 1), z12=(0, 1)).items()
        }
        rows.append(
            dict(
                x=m,
                razor=[c(Zr[0, 0]), c(Zr[1, 1]), c(Zr[0, 1])],
                bspline=[c(Zb[0, 0]), c(Zb[1, 1]), c(Zb[0, 1])],
                gap=g,
                tb=tb,
            )
        )
        print(
            f"(c) x{m} |dZ11| {g['z11']:.4f} |dZ22| {g['z22']:.4f} |dZ12| {g['z12']:.5f}"
            f"  bspline {tb:.1f} s",
            flush=True,
        )
    res["c_vs_bspline"] = rows

if "d" in want:
    rows = []
    for dx in (0.5, 0.0):
        for gap in (0.3, 0.1, 0.03, 0.01):
            row = dict(dx=dx, gap=gap, nonrec=[])
            try:
                for m in (1, 2, 4):
                    Z, nr, dt, _ = two_port(
                        RazorSolver, deck(m, gap=gap, dx=dx), nec5_quadrature=True
                    )
                    row["nonrec"].append(nr)
                row["razor_x4"] = [c(Z[0, 0]), c(Z[1, 1]), c(Z[0, 1])]
                Zb, *_ = two_port(BSplineSolver, deck(4, gap=gap, dx=dx))
                row["bspline_x4"] = [c(Zb[0, 0]), c(Zb[1, 1]), c(Zb[0, 1])]
                row["gap_x4"] = [
                    float(abs(Z[i, j] - Zb[i, j])) for i, j in ((0, 0), (1, 1), (0, 1))
                ]
            except Exception as e:  # noqa: BLE001 - probe records any refusal verbatim
                row["error"] = f"{type(e).__name__}: {e}"[:400]
            rows.append(row)
            print(
                f"(d) dx={dx} gap={gap}: "
                + (
                    row.get("error")
                    or f"nonrec {['%.2e' % v for v in row['nonrec']]} |dZ| x4 "
                    f"{['%.4f' % v for v in row['gap_x4']]}"
                ),
                flush=True,
            )
    res["d_standoff"] = rows

if "e" in want:
    rows = []
    for m in (1, 2, 4):
        out = {}
        for name, cls, kw in (
            ("razor", RazorSolver, {"nec5_quadrature": True}),
            ("bspline", BSplineSolver, {}),
        ):
            Z0, *_ = two_port(cls, deck(m), **kw)
            ZL, *_ = two_port(
                cls, deck(m, wire_conductivity=3.5e7, insulation_radius=None), **kw
            )
            out[name] = [c(ZL[0, 0] - Z0[0, 0]), c(ZL[1, 1] - Z0[1, 1])]
        rows.append(dict(x=m, **out))
        print(
            f"(e) x{m} loading shift razor {out['razor']} bspline {out['bspline']}",
            flush=True,
        )
    res["e_loading"] = rows

if "f" in want:
    # antennaknobs verticals.elevated_buried_counterpoise at its defaults,
    # spelled directly: 7.1 MHz, a lambda/4 radiator from base 0.5 m whose
    # first 0.05 m is the house eps-gap (fed at its middle, 0.025 m above the
    # free foot - the design's `Wire(..., ex=1)` idiom, so the feed stays at
    # the SAME place on every rung; a first draft fed the first knot, which
    # walks toward the open end as N grows and diverges by construction),
    # four radials 0.6 x lambda/4 at depth 0.15 m from a buried hub.
    lam = C0 / 7.1e6
    h = 0.25 * lam
    rad = 0.6 * h
    radiator = np.array([(0, 0, 0.5), (0, 0, 0.55), (0, 0, 0.5 + h)])
    radials = []
    for i in range(4):
        th = 2 * math.pi * i / 4
        cth = 0.0 if abs(math.cos(th)) < 1e-12 else math.cos(th)
        sth = 0.0 if abs(math.sin(th)) < 1e-12 else math.sin(th)
        radials.append(np.array([(0, 0, -0.15), (rad * cth, rad * sth, -0.15)]))
    rows = []
    for m in (1, 2, 4, 8):
        kw = dict(
            wires=[radiator] + radials,
            n_per_edge_per_wire=[[2 * m, 20 * m]] + [[6 * m]] * 4,
            feeds=[(0, 0.025, 1 + 0j)],
            wavelength=lam,
            wire_radius=0.001,
            ground_z=0.0,
            ground_eps=(13.0, 0.005),
            ground_model="sommerfeld",
        )
        t0 = time.perf_counter()
        s = RazorSolver(**kw, nec5_quadrature=True)
        assert s._detached
        zr = complex(s.compute_impedance()[0])
        tr = time.perf_counter() - t0
        t0 = time.perf_counter()
        zb = complex(BSplineSolver(**kw).compute_impedance()[0])
        tb = time.perf_counter() - t0
        rows.append(
            dict(x=m, razor=c(zr), bspline=c(zb), gap=abs(zr - zb), tr=tr, tb=tb)
        )
        print(
            f"(f) x{m} razor {zr:.3f} bspline {zb:.3f} |d| {abs(zr - zb):.3f}"
            f"  {tr:.1f} / {tb:.1f} s",
            flush=True,
        )
    res["f_catalog_counterpoise"] = rows

OUT.write_text(json.dumps(res, indent=1))
print("wrote", OUT)
