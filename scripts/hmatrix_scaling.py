"""Scaling study: HMatrixSolver (hierarchical / ACA) vs dense BSplineSolver.

Sweeps a geometry at increasing segment count N and reports, for each N:

  * storage      — H-matrix complex scalars stored vs dense N^2, and MB
  * rank         — mean / max far-block ACA rank
  * far%         — fraction of the matrix area in admissible (far) blocks
  * accuracy     — |Z_hmat - Z_dense| / |Z_dense| (only while the dense
                   reference is still affordable, see --dense-max)
  * wall-clock   — dense assemble+solve vs H build+solve
  * rss          — peak process RSS (cumulative across the ladder when several
                   N run in one invocation; run one N per process for a clean
                   per-N figure)

Geometries (--geometry):

  wire    — fixed-length straight wire (4 wavelengths), mesh refined. The
            cluster tree is effectively 1-D: the most favourable case for
            ACA (tiny constant ranks).
  screen  — square panel of ~sqrt(N) parallel disconnected wires, segment
            length ~ wire spacing (an NEC-style wire-grid ground screen).
            2-D cluster structure: the honest stand-in for the large-N
            wire-grid workload motivating the FMM scoping (issue #168).

Run:
  python scripts/hmatrix_scaling.py                       # classic ladder to 4000
  python scripts/hmatrix_scaling.py --ns 16000 --geometry screen
  python scripts/hmatrix_scaling.py --ns 8000,16000,32000 --dense-max 8000
"""

import argparse
import resource
import time

import numpy as np

from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver

WAVELENGTH = 22.0
LEN_WL = 4.0  # straight-wire length in wavelengths (fixed as N grows)
SCREEN_WL = 2.0  # screen panel side in wavelengths


def _wire_geometry(nsegs):
    half = 0.5 * LEN_WL * WAVELENGTH
    wires = [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]
    return wires, [[nsegs]]


def _screen_geometry(nsegs):
    """~sqrt(N) parallel wires spanning a square panel, segment length ~
    spacing — a wire-grid screen without junctions (wires couple through
    the field, which is all the far-field compression sees)."""
    side = SCREEN_WL * WAVELENGTH
    m = max(2, int(round(np.sqrt(nsegs))))
    n_w = max(1, int(round(nsegs / m)))
    xs = np.linspace(-side / 2, side / 2, m)
    wires = [
        np.array([[x, 0.0, -side / 2], [x, 0.0, side / 2]]) for x in xs
    ]
    return wires, [[n_w]] * m


GEOMETRIES = {"wire": _wire_geometry, "screen": _screen_geometry}


def _dense_solve(sim):
    geom = sim._build_geometry()
    ss, po, kcl, wk, wbg = sim._build_basis_polynomials(geom)
    n = ss.shape[0]
    t0 = time.perf_counter()
    Z = sim._assemble_Z(sim._build_J_blocks(geom, sim.k), ss, po, geom)
    t_fill = time.perf_counter() - t0
    v_list = []
    for wi, arc, _ in sim.feeds:
        ak = geom["per_wire"][wi]["arc_at_knot"]
        sf = arc if arc is not None else ak[-1] / 2
        v_list.append(sim._build_source_vector(geom, wk, wbg, n, wi=wi, s_f=sf))
    v = v_list[0]
    t0 = time.perf_counter()
    c = sim._solve_with_kcl(Z, v, kcl)
    t_solve = time.perf_counter() - t0
    z = sim.feeds[0][2] / (v_list[0] @ c)
    return z, t_fill, t_solve, n


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ns",
        default="250,500,1000,2000,4000",
        help="comma-separated total segment counts",
    )
    ap.add_argument(
        "--geometry", choices=sorted(GEOMETRIES), default="wire"
    )
    ap.add_argument(
        "--dense-max",
        type=int,
        default=4000,
        help="skip the dense reference above this N (time/memory wall)",
    )
    ap.add_argument("--aca-tol", type=float, default=1e-5)
    ap.add_argument("--aca-eta", type=float, default=2.0)
    ap.add_argument("--degree", type=int, default=1)
    args = ap.parse_args()

    ns = [int(s) for s in args.ns.split(",")]
    print(
        f"geometry={args.geometry} degree={args.degree} "
        f"aca_tol={args.aca_tol} aca_eta={args.aca_eta}"
    )
    print(
        f"{'N':>7} {'n':>7} {'far%':>6} {'rank':>9} "
        f"{'store%':>7} {'H_MB':>8} {'relZ':>9} "
        f"{'Dfill':>8} {'Dslv':>8} {'Hbld':>8} {'Hslv':>8} {'it':>4} {'rssMB':>8}"
    )
    for nsegs in ns:
        wires, n_per_edge = GEOMETRIES[args.geometry](nsegs)

        if nsegs <= args.dense_max:
            dense = BSplineSolver(
                wires=wires,
                degree=args.degree,
                n_per_edge_per_wire=n_per_edge,
                wavelength=WAVELENGTH,
            )
            zd, t_dfill, t_dsolve, _ = _dense_solve(dense)
            del dense
        else:
            zd = t_dfill = t_dsolve = None

        hmat = HMatrixSolver(
            wires=wires,
            degree=args.degree,
            n_per_edge_per_wire=n_per_edge,
            wavelength=WAVELENGTH,
            aca_tol=args.aca_tol,
            aca_eta=args.aca_eta,
        )
        t0 = time.perf_counter()
        H = hmat.build_hmatrix()
        t_hbuild = time.perf_counter() - t0
        st = H.stats()
        part = hmat.build_partition()

        t0 = time.perf_counter()
        zh, _ = hmat.compute_impedance()
        t_hsolve = time.perf_counter() - t0

        relz = "" if zd is None else f"{abs(zh - zd) / abs(zd):>9.1e}"
        dfill = "" if t_dfill is None else f"{t_dfill:>7.2f}s"
        dslv = "" if t_dsolve is None else f"{t_dsolve:>7.2f}s"
        h_mb = st["storage"] * 16 / 1e6
        print(
            f"{nsegs:>7} {st['n']:>7} {part['stats']['far_frac'] * 100:>5.0f}% "
            f"{st['mean_rank']:>4.1f}/{st['max_rank']:>4d} "
            f"{st['compression'] * 100:>6.1f}% {h_mb:>7.1f} "
            f"{relz:>9} {dfill:>8} {dslv:>8} "
            f"{t_hbuild:>7.2f}s {t_hsolve:>7.2f}s "
            f"{max(hmat._last_solve_iters):>4d} {_rss_mb():>7.0f}",
            flush=True,
        )
        del hmat, H


if __name__ == "__main__":
    main()
