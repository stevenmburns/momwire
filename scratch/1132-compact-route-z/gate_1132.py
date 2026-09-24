"""momwire#1132 cross-build gate: dump every number the bit-identity and the
drift gates compare, from WHICHEVER momwire PYTHONPATH resolves.

  PYTHONPATH=<tree>/src python gate_1132.py --out <file>.npz [--radials 4 12 48]

Route decks (the new argument):  z_in, coeffs, and the route's Z rows R --
  on a build WITH `compact=` the compact fill (and, separately, the square
  rows= fill sliced, `rowsq_*`); on the base build the square rows= fill sliced.
Drift decks (never the new argument): full Z + z_in + coeffs of
  - the same radial screen solved DENSELY (buried, crossing, windowed cplx),
  - an elevated monopole over a detached buried radial (transmitted family),
  - a dipole over Sommerfeld soil, chunked windowed fill forced (swept_mem_mb=1),
  - the same dipole with the default dispatch,
  - a free-space dipole, chunked forced and default.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import warnings

import numpy as np

C0 = 299792458.0
GROUND = ("finite", 13.0, 0.005)


def radial_solver(n_radials, route, **kw):
    from antennaknobs.designs.verticals.buried_radial_vertical import Builder
    from antennaknobs.engines.momwire import MomwireEngine

    from momwire import BSplineSolver

    b = Builder()
    b.n_radials = n_radials
    extra = {"rotational_symmetry": True} if route else {}
    eng = MomwireEngine(
        b,
        solver=BSplineSolver,
        solver_kwargs={"degree": 2, **extra, **kw},
        ground=GROUND,
    )
    return eng._make_solver(wavelength=eng._wavelength_for(float(b.freq)))


def route_rows(s, compact):
    from momwire import _rotational_symmetry as RS

    geom = s._build_geometry()
    supp_seg, polys, _k, _wk, wbg = s._build_basis_polynomials(geom)
    rows = RS.observer_rows(s, geom)
    if compact:
        Zc, R = s._compute_Z_operator_buried(
            geom, supp_seg, polys, rows=rows, compact=True
        )
        return Zc, R
    Z = s._compute_Z_operator_buried(geom, supp_seg, polys, rows=rows)
    sectors, axial = RS.dof_groups(s, wbg, supp_seg.shape[0])
    R = np.union1d(sectors[0], axial)
    return np.ascontiguousarray(Z[R]), R


def full_Z(s):
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    return np.asarray(s._compute_Z_operator(geom, supp_seg, polys))


def solve(s):
    z, c = s.compute_impedance()
    return np.atleast_1d(np.asarray(z, dtype=np.complex128)), np.asarray(c)


def served_deck():
    from momwire.bspline import BSplineSolver

    n = 15
    arc = (round(0.4333 * n) - 0.5) / n * 10.0
    return BSplineSolver(
        wires=[
            np.array([(0.0, 0.0, 11.0), (0.0, 0.0, 1.0)]),
            np.array([(0.0, 0.0, -0.15), (5.0, 0.0, -0.15)]),
        ],
        n_per_edge_per_wire=[[n], [10]],
        feeds=[(0, arc, 1 + 0j)],
        wavelength=C0 / 7e6,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )


def dipole(ground, mem):
    from momwire.bspline import BSplineSolver

    kw = {}
    if ground:
        kw = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
    wl = C0 / 14.2e6
    L = 0.48 * wl
    return BSplineSolver(
        wires=[np.array([(0.0, -L / 2, 8.0), (0.0, L / 2, 8.0)])],
        n_per_edge_per_wire=[[301]],
        feeds=[(0, L / 2, 1 + 0j)],
        wavelength=wl,
        wire_radius=0.001,
        swept_mem_mb=mem,
        **kw,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--radials", type=int, nargs="+", default=[4, 12, 48])
    ap.add_argument("--skip-drift", action="store_true")
    args = ap.parse_args()
    import momwire
    from momwire import _rotational_symmetry as RS

    has_compact = (
        "compact"
        in inspect.signature(
            momwire.bspline.BSplineSolver._compute_Z_operator_buried
        ).parameters
    )
    print("momwire", momwire.__file__, "compact:", has_compact, file=sys.stderr)
    out = {}
    warnings.simplefilter("ignore")
    for n in args.radials:
        s = radial_solver(n, True)
        z, c = solve(s)
        out[f"route{n}_z"], out[f"route{n}_c"] = z, c
        Zr, R = route_rows(radial_solver(n, True), compact=has_compact)
        out[f"route{n}_Zrows"], out[f"route{n}_R"] = Zr, R
        if has_compact:
            Zq, Rq = route_rows(radial_solver(n, True), compact=False)
            out[f"rowsq{n}_Zrows"], out[f"rowsq{n}_R"] = Zq, Rq
            RS.COMPACT_Z = False
            zq, cq = solve(radial_solver(n, True))
            RS.COMPACT_Z = True
            out[f"rowsq{n}_z"], out[f"rowsq{n}_c"] = zq, cq
        print(f"route {n}: z={z[0]:.12g} rows={Zr.shape}", file=sys.stderr)
    # loaded route (wire conductivity): the loading remap is exercised
    s = radial_solver(12, True, wire_conductivity=1e6)
    out["loaded12_z"], out["loaded12_c"] = solve(s)
    Zr, R = route_rows(
        radial_solver(12, True, wire_conductivity=1e6), compact=has_compact
    )
    out["loaded12_Zrows"], out["loaded12_R"] = Zr, R
    if not args.skip_drift:
        s = radial_solver(12, False)
        out["dense12_Z"] = full_Z(s)
        out["dense12_z"], out["dense12_c"] = solve(radial_solver(12, False))
        s = served_deck()
        out["served_Z"] = full_Z(s)
        out["served_z"], out["served_c"] = solve(served_deck())
        for g in (True, False):
            for mem in (1, 256):
                tag = f"{'somm' if g else 'free'}_mem{mem}"
                out[f"{tag}_Z"] = full_Z(dipole(g, mem))
                out[f"{tag}_z"], out[f"{tag}_c"] = solve(dipole(g, mem))
    np.savez(args.out, **out)
    print("wrote", args.out, sorted(out), file=sys.stderr)


if __name__ == "__main__":
    main()
