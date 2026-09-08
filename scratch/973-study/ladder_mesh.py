"""momwire#973: does the sampled-residual check hold across skyloop's mesh?

Six rungs of the loop's per-edge segment count, each solved three ways:
dense BSpline d=2 (the reference), H-matrix with the check DISABLED
(`somm_residual_tol=inf`, i.e. pre-#973 behaviour), and H-matrix as shipped.

Reports the probe residual, the ACA rank, whether the fallback fired, and the
wall cost of the check, so "the detector does not fire where it is not needed"
is a measurement rather than a claim.
"""

import time

import numpy as np

import momwire.hmatrix as HM
from momwire import BSplineSolver, HMatrixSolver

FEED_STUB = np.array([[-0.05, 0.0, 15.0], [0.05, 0.0, 15.0]])
LOOP = np.array(
    [
        [0.05, 0.0, 15.0],
        [13.806232, -23.826492, 15.0],
        [13.756232, -23.913095, 15.0],
        [-13.756232, -23.913095, 15.0],
        [-13.806232, -23.826492, 15.0],
        [-0.05, 0.0, 15.0],
    ]
)
RUNGS = (15, 29, 43, 59, 75, 91)

_probe = []
_orig = HM._sampled_residual


def _spy(*a, **kw):
    r = _orig(*a, **kw)
    _probe.append(r[0])
    return r


HM._sampled_residual = _spy


def kw_for(n_edge):
    return dict(
        wires=[FEED_STUB, LOOP],
        n_per_edge_per_wire=[[1], [n_edge, 1, n_edge, 1, n_edge]],
        feeds=[(0, 0.05, 1.0 + 0j)],
        junctions=[[(0, "start"), (1, "end")], [(0, "end"), (1, "start")]],
        wavelength=16.563119226519337,
        wire_radius=0.0005,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
        degree=2,
    )


def solve(cls, n_edge, **extra):
    _probe.clear()
    t = time.perf_counter()
    s = cls(**{**kw_for(n_edge), **extra})
    cur = np.asarray(s.compute_impedance()[1])
    return s, cur, time.perf_counter() - t, (max(_probe) if _probe else float("nan"))


def main():
    print(
        f"{'n/edge':>7s} {'bases':>6s} {'OFF dI':>10s} {'ON dI':>10s} "
        f"{'probe':>10s} {'rank':>5s} {'fb':>3s} {'OFF s':>7s} {'ON s':>7s}"
    )
    for ne in RUNGS:
        _s, ref, _t, _p = solve(BSplineSolver, ne)
        _so, coff, toff, _po = solve(HMatrixSolver, ne, somm_residual_tol=float("inf"))
        son, con, ton, pon = solve(HMatrixSolver, ne)
        nrm = np.linalg.norm(ref)
        eoff = np.linalg.norm(coff - ref) / nrm
        eon = np.linalg.norm(con - ref) / nrm
        print(
            f"{ne:7d} {ref.size:6d} {eoff:10.3e} {eon:10.3e} {pon:10.3e} "
            f"{son._last_somm_rank:5d} {'Y' if son._last_somm_fallback else 'n':>3s} "
            f"{toff:7.3f} {ton:7.3f}"
        )


main()
