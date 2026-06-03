"""Per-n convergence sweep on the single-band hentenna across four solvers.

Re-evaluates NEXT_STEPS items 13/14 (pysim-vs-PyNEC arbitration) and item
15(a) (post-PR-#51 enrichment story) on the same geometry, all built
inline so the script is independent of `web.server` request plumbing.

Geometry: antenna_designer hentenna params_50, 28.47 MHz free space,
width_factor=0.1378, top_height_factor=0.5081, mid_height_factor=0.1094,
wire_radius=0.5 mm, eps_feed=0.05 m.

Five columns at each n:
  - tri  : TriangularPySim (tent basis, degree-1 polynomial)
  - b2   : BSplinePySim(degree=2)                        — d=2 polynomial
  - b2e  : BSplinePySim(degree=2, use_singular_enrichment=True)
  - sin  : SinusoidalPySim — NEC2's const+sin+cos three-term basis,
           from-scratch in pysim (PR #44). Expected to track PyNEC's
           super-log X drift since it's the same basis family.
  - pynec: NEC2 via PyNEC (item-13's reference column)

Three independent polynomial-family bases (tri/b2/b2e) vs two
independent three-term-basis implementations (sin/pynec). The
arbitration argument from PR #45 / item 14 is that the polynomial
trio agrees on Z, while the three-term pair tracks each other's drift.

Feed-segment parity:
  - pysim polynomial (tri / b2 / b2e): nfeed=2 (EVEN — interior knot at z=0).
  - pysim sinusoidal (sin):            nfeed=3 (ODD — delta-gap segment
                                       centred at z=0; same parity as PyNEC).
  - pynec:                             nfeed picked by the existing backend
                                       rule (odd, source segment centre at z=0).

After the table, the script reports for each solver:
  - n=161 anchor R + jX
  - X-convergence rate fit p in X(N) = X_inf + C/N^p using consecutive
    triples of points. Same Richardson-style estimator as in the
    test_bspline_d2_hentenna_singular_enrichment pinned test:
        p ≈ log(|ΔX_12 / ΔX_23|) / log(N2/N1)
    Reported for each consecutive triple (N1, N2, N3) so a slowly-converging
    rate vs a basis-limited one is visible at a glance.

Reasoning under PR #51's enrichment-orig sign fix (load-bearing for item
15(a)): if `b2e` actually accelerates convergence on the hentenna (as PR
#47 claimed all along), `b2e` should reach `b2`'s asymptote at smaller n
with a higher fitted p; if not, the enrichment basis may be irrelevant
on this geometry once the sign is right, in which case the UI default
slot-B should flip enrichment off.

Run from the project root (PyNEC column needs `web/` on the path):

    PYTHONPATH=. .venv/bin/python scripts/compare_hentenna_solvers.py
    PYTHONPATH=. .venv/bin/python scripts/compare_hentenna_solvers.py \\
        --n-list 15,21,41,81,161
    PYTHONPATH=. .venv/bin/python scripts/compare_hentenna_solvers.py \\
        --skip-pynec

Post-PR-#51 result (2026-06-03; nfeed=2 for tri/b2/b2e, nfeed=3 for
sin and pynec — the basis-specific source-segment-centering rules):

       n  |    tri         |    b2 (no enr)  |    b2e (enr)    |    sin          |    pynec
      15  | 43.20 + j37.13 | 43.07 + j38.85  | 42.86 + j40.07  | 45.61 − j 5.72  | 45.61 − j 5.77
      21  | 43.16 + j38.03 | 43.07 + j38.85  | 43.03 + j39.09  | 45.60 − j 4.55  | 45.60 − j 4.60
      41  | 43.13 + j38.65 | 43.06 + j38.84  | 43.06 + j38.86  | 45.46 − j 1.78  | 45.44 − j 1.84
      81  | 43.11 + j38.79 | 43.05 + j38.84  | 43.05 + j38.84  | 45.25 + j 1.71  | 45.24 + j 1.65
     161  | 43.11 + j38.82 | 43.05 + j38.84  | 43.05 + j38.84  | 44.98 + j 6.51  | 45.01 + j 6.54

The polynomial trio (tri, b2, b2e) **converges** to 43.05 + j38.84.
The three-term pair (sin, pynec) **drifts super-log on X** — sin
tracks pynec to ~0.05 Ω on R and ~0.07 Ω on X at every n, exactly as
predicted by item 14 (the two implementations of the same basis family
diverge together).

Per-solver X-convergence rates over the (41, 81, 161) triple:
  tri   → p ≈ 2.19   (basis-limited at degree-1's O(1/N²) theoretical rate)
  b2    → p ≈ 2.53   (already at convergence by n=15; rate is tail-noise)
  b2e   → p ≈ 3.23   (visibly faster tail, but the small-N transient is
                      worse than b2's — b2e starts +1.2 Ω off at n=15 where
                      b2 is already converged)
  sin   → p ≈ −0.46  (anti-convergent — fitted p < 0 every triple)
  pynec → p ≈ −0.50  (anti-convergent — same as sin)

The arbitration result is now visible in a single table: two
independent basis FAMILIES (polynomial vs three-term), each with two
or three independent code paths, with the polynomial side reaching
an asymptote that the three-term side cannot. The three-term basis
is the outlier on this geometry.

Decision for slot-B default: **flip enrichment OFF**. At the UI default
n=21, b2 alone gives 43.07 + j38.85 (within 0.02 Ω of asymptote) while
b2e gives 43.03 + j39.09 (0.26 Ω off X). The enrichment basis injects
its own early-N transient on this geometry; for the hentenna it costs
more than it buys at every n < 161.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pysim.triangular import TriangularPySim
from pysim.bspline import BSplinePySim
from pysim.sinusoidal import SinusoidalPySim


C_LIGHT = 299_792_458.0

# antenna_designer hentenna params_50 — pinned across every hentenna test
# / sweep in this repo. See web/server.py:_hentenna_geometry for the
# reference geometry.
FREQ_MHZ = 28.47
WIDTH_FACTOR = 0.1378
TOP_HEIGHT_FACTOR = 0.5081
MID_HEIGHT_FACTOR = 0.1094
EPS_FEED = 0.05
WIRE_RADIUS = 0.0005


def _hentenna_wires_and_junctions():
    """Build the five-wire / four-junction params_50 hentenna geometry.

    Identical to the inline construction in
    test_bspline_d2_hentenna_arbitrates_against_triangular, kept here so
    this script doesn't depend on the web-server geometry helpers (which
    drag in FastAPI imports).
    """
    wavelength = C_LIGHT / (FREQ_MHZ * 1e6)
    half_w = wavelength * WIDTH_FACTOR / 2.0
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
        np.array([T, S], dtype=float),  # 0: feed gap
        np.array([S, B], dtype=float),  # 1: cross-bar right
        np.array([B, A, C, D], dtype=float),  # 2: upper rectangle B→A→C→D
        np.array([T, D], dtype=float),  # 3: cross-bar left
        np.array([D, E, F, B], dtype=float),  # 4: lower rectangle D→E→F→B
    ]
    junctions = [
        [(0, "end"), (1, "start")],  # at S (K=2)
        [(0, "start"), (3, "start")],  # at T (K=2)
        [(1, "end"), (2, "start"), (4, "end")],  # at B (K=3)
        [(2, "end"), (3, "end"), (4, "start")],  # at D (K=3)
    ]
    return wires, junctions, wavelength


def _pysim_npe(n: int, nfeed: int) -> list[list[int]]:
    """Per-edge segment counts matching the test-fixture convention."""
    return [[nfeed], [n], [n, n, n], [n], [n, n, n]]


def solve_tri(n: int) -> tuple[complex, float]:
    wires, junctions, wavelength = _hentenna_wires_and_junctions()
    nfeed = 2
    sim = TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=_pysim_npe(n, nfeed),
        feed_wire_index=0,
        feed_arclength=EPS_FEED,
        wavelength=wavelength,
        wire_radius=WIRE_RADIUS,
        nsegs=n,
        junctions=junctions,
    )
    t0 = time.perf_counter()
    z, _ = sim.compute_impedance()
    return complex(z), (time.perf_counter() - t0) * 1e3


def solve_bspline(n: int, *, use_enrichment: bool) -> tuple[complex, float]:
    wires, junctions, wavelength = _hentenna_wires_and_junctions()
    nfeed = 2
    sim = BSplinePySim(
        degree=2,
        wires=wires,
        n_per_edge_per_wire=_pysim_npe(n, nfeed),
        feed_wire_index=0,
        feed_arclength=EPS_FEED,
        wavelength=wavelength,
        wire_radius=WIRE_RADIUS,
        nsegs=n,
        junctions=junctions,
        use_singular_enrichment=use_enrichment,
    )
    t0 = time.perf_counter()
    z, _ = sim.compute_impedance()
    return complex(z), (time.perf_counter() - t0) * 1e3


def solve_sinusoidal(n: int) -> tuple[complex, float]:
    """NEC2's three-term basis (const + sin + cos per segment), from-scratch
    re-implementation in pysim (PR #44). Uses ODD nfeed parity — the basis
    centres a delta-gap segment at z=0, matching PyNEC's convention. Expected
    to track PyNEC's super-log X drift since it's the same basis family."""
    wires, junctions, wavelength = _hentenna_wires_and_junctions()
    nfeed = 3
    sim = SinusoidalPySim(
        wires=wires,
        n_per_edge_per_wire=_pysim_npe(n, nfeed),
        feed_wire_index=0,
        feed_arclength=EPS_FEED,
        wavelength=wavelength,
        wire_radius=WIRE_RADIUS,
        nsegs=n,
        junctions=junctions,
    )
    t0 = time.perf_counter()
    z, _ = sim.compute_impedance()
    return complex(z), (time.perf_counter() - t0) * 1e3


def solve_pynec(n: int) -> tuple[complex, float]:
    """Run PyNEC via the existing web backend's hentenna builder. Imported
    lazily so the rest of the script runs even without the PyNEC build."""
    from web import pynec_backend

    req = {
        "n_per_wire": n,
        "design_freq_mhz": FREQ_MHZ,
        "measurement_freq_mhz": FREQ_MHZ,
        "width_factor": WIDTH_FACTOR,
        "top_height_factor": TOP_HEIGHT_FACTOR,
        "mid_height_factor": MID_HEIGHT_FACTOR,
        "wire_radius": WIRE_RADIUS,
        "ground": False,
    }
    t0 = time.perf_counter()
    out = pynec_backend.solve_hentenna(req)
    return complex(out["z_in_re"], out["z_in_im"]), (time.perf_counter() - t0) * 1e3


def _fit_x_rate(
    ns: list[int], xs: list[float]
) -> list[tuple[int, int, int, float, float]]:
    """Richardson-style three-point X convergence rate. Returns one (N1, N2,
    N3, p, X_inf_estimate) tuple per consecutive triple. Same estimator as
    the test_bspline_d2_hentenna_singular_enrichment pinned test.
    """
    out = []
    for i in range(len(ns) - 2):
        n1, n2, n3 = ns[i], ns[i + 1], ns[i + 2]
        d12 = xs[i] - xs[i + 1]
        d23 = xs[i + 1] - xs[i + 2]
        if d12 * d23 <= 0 or abs(d23) < 1e-9:
            # Either crossed zero (noise floor) or stalled. Mark rate as nan
            # so the caller can flag it without crashing.
            out.append((n1, n2, n3, float("nan"), float("nan")))
            continue
        p = float(np.log(abs(d12 / d23)) / np.log(n2 / n1))
        # Z_inf ≈ X3 − ΔX_23 · (1 / (r^p − 1)), where r = n3/n2. With the
        # same p the ratio gives the geometric-series limit.
        r = n3 / n2
        x_inf = xs[i + 2] - d23 / (r**p - 1.0)
        out.append((n1, n2, n3, p, float(x_inf)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-list",
        type=str,
        default="15,21,41,81,161",
        help="comma-separated per-non-feed-edge segment counts to sweep",
    )
    ap.add_argument(
        "--skip-pynec",
        action="store_true",
        help="omit the PyNEC column (e.g. running without the PyNEC build)",
    )
    args = ap.parse_args()

    ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    if not ns:
        ap.error("--n-list produced an empty list")

    wavelength = C_LIGHT / (FREQ_MHZ * 1e6)
    print(
        f"Hentenna params_50 @ {FREQ_MHZ:.2f} MHz  (λ = {wavelength:.4f} m)\n"
        f"  width = {WIDTH_FACTOR * wavelength:.4f} m  "
        f"top_height = {TOP_HEIGHT_FACTOR * wavelength:.4f} m  "
        f"mid_offset = {(MID_HEIGHT_FACTOR - TOP_HEIGHT_FACTOR) * wavelength:.4f} m\n"
        f"  wire_radius = {WIRE_RADIUS * 1e3:.2f} mm  feed_gap = {2 * EPS_FEED:.3f} m\n"
    )

    cols = ["tri", "b2", "b2e", "sin"]
    if not args.skip_pynec:
        cols.append("pynec")

    # Each column is { n -> (Z, solve_ms) }.
    results: dict[str, dict[int, tuple[complex, float]]] = {c: {} for c in cols}

    width_per_col = 22
    header = "  n  | " + " | ".join(f"{c:^{width_per_col}}" for c in cols)
    sep = " " * 5 + "+-" + "-+-".join("-" * width_per_col for _ in cols)
    print(header)
    print(sep)
    for n in ns:
        cells: list[str] = []
        for c in cols:
            if c == "tri":
                z, ms = solve_tri(n)
            elif c == "b2":
                z, ms = solve_bspline(n, use_enrichment=False)
            elif c == "b2e":
                z, ms = solve_bspline(n, use_enrichment=True)
            elif c == "sin":
                z, ms = solve_sinusoidal(n)
            elif c == "pynec":
                z, ms = solve_pynec(n)
            else:
                raise AssertionError(c)
            results[c][n] = (z, ms)
            cells.append(f"{z.real:+8.3f} {z.imag:+8.3f}j ({ms:5.0f}ms)")
        print(f" {n:>3} | " + " | ".join(f"{cell:^{width_per_col}}" for cell in cells))

    print()
    print("X-convergence rate fit p in X(N) = X_inf + C/N^p (Richardson 3-pt):")
    print()
    if len(ns) < 3:
        print("  (need at least 3 n-values for a rate fit; skipping)")
        return
    for c in cols:
        xs = [results[c][n][0].imag for n in ns]
        rs = [results[c][n][0].real for n in ns]
        print(f"  {c}:")
        # Headline: anchor + dX per decade-ish step
        print(
            f"    last n={ns[-1]}: {rs[-1]:+8.3f} + j{xs[-1]:+8.3f}    "
            f"dX over last step (n={ns[-2]}→{ns[-1]}): {xs[-1] - xs[-2]:+.4f} Ω"
        )
        # Per-triple p
        for n1, n2, n3, p, x_inf in _fit_x_rate(ns, xs):
            tag = f"({n1},{n2},{n3})"
            if np.isnan(p):
                print(f"    {tag:>13s}  p=nan (sign-flip or noise floor)")
            else:
                print(f"    {tag:>13s}  p={p:5.2f}   X_inf≈{x_inf:+8.4f}")
        print()


if __name__ == "__main__":
    main()
