"""Width-factor sweep harness for vtune profiling of the N=21 hentenna.

Mirrors what the UI does when the user nudges the "width_factor" slider:
each step rebuilds geometry, solves for the feedline impedance, and asks
the backend for per-wire knot currents. We do 10 steps of +0.0005 then
10 steps of -0.0005 about the params_50 default (0.1378).

Usage:
    .venv/bin/python scripts/vtune_hentenna_width_sweep.py --solver sin
    .venv/bin/python scripts/vtune_hentenna_width_sweep.py --solver pynec

Two solvers are kept symmetric: same N, same geometry topology, same
RHS (delta-gap V=1), one solve + one currents-at-knots call per step.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from scripts.compare_hentenna_solvers import (
    EPS_FEED,
    FREQ_MHZ,
    MID_HEIGHT_FACTOR,
    TOP_HEIGHT_FACTOR,
    WIRE_RADIUS,
    _hentenna_wires_and_junctions,  # noqa: F401 — re-used via private helper below
    _pysim_npe,
)

C_LIGHT = 299_792_458.0


def _hentenna_wires_with_width(width_factor: float):
    """Same geometry as scripts/compare_hentenna_solvers but parametric in
    width_factor (the only knob the sweep moves)."""
    wavelength = C_LIGHT / (FREQ_MHZ * 1e6)
    half_w = wavelength * width_factor / 2.0
    z_mid = wavelength * (MID_HEIGHT_FACTOR - TOP_HEIGHT_FACTOR)
    z_bot = -wavelength * TOP_HEIGHT_FACTOR

    A = (0.0, half_w, 0.0)
    B = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, EPS_FEED, z_mid)
    C = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E = (0.0, -half_w, z_bot)
    T = (0.0, -EPS_FEED, z_mid)

    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B], dtype=float),
        np.array([B, A, C, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E, F, B], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    return wires, junctions, wavelength


def _step_sin(n: int, width_factor: float):
    from pysim.sinusoidal import SinusoidalPySim

    wires, junctions, wavelength = _hentenna_wires_with_width(width_factor)
    sim = SinusoidalPySim(
        wires=wires,
        n_per_edge_per_wire=_pysim_npe(n, nfeed=3),
        feed_wire_index=0,
        feed_arclength=EPS_FEED,
        wavelength=wavelength,
        wire_radius=WIRE_RADIUS,
        nsegs=n,
        junctions=junctions,
    )
    z_in, alpha = sim.compute_impedance()
    currents = sim.currents_at_knots(alpha)
    return z_in, currents


def _step_tri(n: int, width_factor: float):
    from pysim.triangular import TriangularPySim

    wires, junctions, wavelength = _hentenna_wires_with_width(width_factor)
    sim = TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=_pysim_npe(n, nfeed=2),
        feed_wire_index=0,
        feed_arclength=EPS_FEED,
        wavelength=wavelength,
        wire_radius=WIRE_RADIUS,
        nsegs=n,
        junctions=junctions,
    )
    z_in, coeffs = sim.compute_impedance()
    currents = sim.currents_at_knots(coeffs)
    return z_in, currents


def _step_bspline(n: int, width_factor: float):
    from pysim.bspline import BSplinePySim

    wires, junctions, wavelength = _hentenna_wires_with_width(width_factor)
    sim = BSplinePySim(
        degree=2,
        wires=wires,
        n_per_edge_per_wire=_pysim_npe(n, nfeed=2),
        feed_wire_index=0,
        feed_arclength=EPS_FEED,
        wavelength=wavelength,
        wire_radius=WIRE_RADIUS,
        nsegs=n,
        junctions=junctions,
        use_singular_enrichment=False,
    )
    z_in, coeffs = sim.compute_impedance()
    currents = sim.currents_at_knots(coeffs)
    return z_in, currents


def _step_pynec(n: int, width_factor: float):
    from web import pynec_backend

    req = {
        "n_per_wire": n,
        "design_freq_mhz": FREQ_MHZ,
        "measurement_freq_mhz": FREQ_MHZ,
        "width_factor": width_factor,
        "top_height_factor": TOP_HEIGHT_FACTOR,
        "mid_height_factor": MID_HEIGHT_FACTOR,
        "wire_radius": WIRE_RADIUS,
        "ground": False,
    }
    out = pynec_backend.solve_hentenna(req)
    z = complex(out["z_in_re"], out["z_in_im"])
    return z, out["wires"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--solver", choices=["sin", "pynec", "tri", "bspline"], required=True
    )
    ap.add_argument("--n", type=int, default=21)
    ap.add_argument("--steps-up", type=int, default=10)
    ap.add_argument("--steps-down", type=int, default=10)
    ap.add_argument("--dwf", type=float, default=0.0005)
    ap.add_argument("--w0", type=float, default=0.1378)
    ap.add_argument(
        "--reps", type=int, default=1, help="how many times to repeat the up+down sweep"
    )
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="run a single throwaway step to amortize JIT/imports",
    )
    args = ap.parse_args()

    step = {
        "sin": _step_sin,
        "pynec": _step_pynec,
        "tri": _step_tri,
        "bspline": _step_bspline,
    }[args.solver]

    if args.warmup:
        step(args.n, args.w0)

    ws_up = [args.w0 + (i + 1) * args.dwf for i in range(args.steps_up)]
    ws_down = [
        args.w0 + args.steps_up * args.dwf - (i + 1) * args.dwf
        for i in range(args.steps_down)
    ]
    sweep = ws_up + ws_down

    t_total = 0.0
    for _ in range(args.reps):
        for w in sweep:
            t0 = time.perf_counter()
            z, _curs = step(args.n, w)
            t_total += time.perf_counter() - t0

    n_total = args.reps * len(sweep)
    mean_ms = t_total / n_total * 1e3
    print(
        f"solver={args.solver}  N={args.n}  reps={args.reps}  "
        f"steps_per_rep={len(sweep)}  total_steps={n_total}  "
        f"mean_step_ms={mean_ms:.1f}  total_s={t_total:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
