"""Compare three independent solvers on the K-band fan dipole.

This is the third-reference validation called for in NEXT_STEPS.md item 8.
PR #36 surfaced a ~14 Ω real-part disagreement between pysim's
TriangularPySim and PyNEC (NEC2) at the 14.3 MHz design freq, with both
solvers converged to their own stable answers. pymininec (MININEC) is
this script's candidate arbitrator.

All three solvers see the same fan-dipole geometry — built by
`web.server._fandipole_geometry` and replayed wire-for-wire onto each
solver's API. The feed is a T→S gap shared by all K bands per side.
Pysim/pymininec use a 2-segment feed wire (delta-gap source on the
midpoint's interior knot); PyNEC's NEC2 backend uses a 1-segment feed
with the source on that single segment.

Run from the project venv (needs PyNEC and pymininec):

    PYTHONPATH=. .venv/bin/python scripts/compare_fandipole_solvers.py
    PYTHONPATH=. .venv/bin/python scripts/compare_fandipole_solvers.py --n-bands 1
    PYTHONPATH=. .venv/bin/python scripts/compare_fandipole_solvers.py --n-list 21,41,81

Findings as of the first run (n_bands=2, 14.3 MHz):
    - pysim: R≈60.4 Ω, X≈0, stable across N.
    - PyNEC: R≈46.5 Ω, X≈+0.4, stable across N.
    - pymininec: R near pysim (~60 Ω) at low N but the imaginary part runs
      away with N (X goes −1 → −7 → −30 Ω over N=21→81 at K=2; even more
      extreme at K=1). MININEC's algorithm has known weaknesses around
      polyline kinks and segment-length-disparate junctions — the fan
      dipole's S→A_i→B_i bent arm hits both, and removing the kink shrinks
      the drift dramatically. pymininec on a straight dipole at the same
      frequency agrees with PyNEC to ~0.3 Ω, so it's correct on its
      comfort zone but not a usable arbitrator here.

So MININEC alone doesn't settle the pysim/PyNEC disagreement on this
geometry. Item 8 stays open — NEC4 or a measurement are the next options.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import mininec.mininec as mn

from validation import momwire_backend as web_server
from validation import pynec_backend


def _make_req(
    n_per_wire: int,
    n_bands: int,
    design_freq_mhz: float,
    cone_radius_m: float = 0.12,
) -> dict:
    """Build a fan-dipole request dict matching the web UI's defaults so the
    three solvers all see the same geometry (other than per-solver
    segmentation, which is what we're sweeping)."""
    band_lengths_full = [10.2551, 5.2691, 4.0, 3.0, 2.5]
    band_lengths = band_lengths_full[:n_bands]
    return {
        "n_per_wire": n_per_wire,
        "n_bands": n_bands,
        "band_lengths_m": band_lengths,
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": design_freq_mhz,
        "wire_radius": 0.0005,
        "slope": 0.5,
        "cone_radius_m": cone_radius_m,
        "t0_factor": float(np.sqrt(2.0)),
        "ground": False,
        "height_m": 0.0,
    }


def solve_pysim(req: dict) -> tuple[complex, float]:
    """Run TriangularPySim on the fan dipole. Returns (Z, solve_ms)."""
    out = web_server._solve_fandipole(req)
    return complex(out["z_in_re"], out["z_in_im"]), float(out["solve_ms"])


def solve_pynec(req: dict) -> tuple[complex, float]:
    """Run PyNEC on the fan dipole. Returns (Z, solve_ms)."""
    out = pynec_backend.solve_fandipole(req)
    return complex(out["z_in_re"], out["z_in_im"]), float(out["solve_ms"])


def solve_pymininec(req: dict) -> tuple[complex, float]:
    """Run pymininec on the fan dipole. Returns (Z, solve_ms).

    Per-wire layout (matches pysim's `_fandipole_geometry`; the feed wire
    is split into 2 segments so its midpoint is an interior pulse rather
    than a K-way junction pulse):
      - wires 0,1: feed T→midpoint and midpoint→S, 1 segment each
      - per band: 2 wires for the +y arm (S→A_pos, A_pos→B_pos)
      - per band: 2 wires for the -y arm (T→A_neg, A_neg→B_neg)
    pymininec auto-detects shared endpoints (NEC-style fuzzy match), so the
    K-way junctions at S and T get stitched without any explicit directive.

    The 2-segment-feed trick matters: a 1-segment feed wire whose endpoints
    are both K-way junctions has no interior pulse of its own. MININEC
    instead creates K-1 bridging pulses per junction, and the standard
    `register_source` call can only excite one of them — biasing the
    solution toward whichever wire-pair the picked bridging pulse covers.
    Splitting into 2 segments gives the midpoint a clean interior pulse to
    host the delta-gap source.
    """
    g = web_server._fandipole_geometry(req)
    freq_mhz = float(req["measurement_freq_mhz"])
    wire_radius = float(req["wire_radius"])
    n_per_wire = int(req["n_per_wire"])
    n_bands = int(req["n_bands"])
    S, T = g["S"], g["T"]
    midp = tuple(0.5 * (t + s) for t, s in zip(T, S))

    wires = [mn.Wire(1, *T, *midp, wire_radius), mn.Wire(1, *midp, *S, wire_radius)]
    for i in range(n_bands):
        wires.append(mn.Wire(n_per_wire, *S, *g["A_pos"][i], wire_radius))
        wires.append(mn.Wire(n_per_wire, *g["A_pos"][i], *g["B_pos"][i], wire_radius))
    for i in range(n_bands):
        wires.append(mn.Wire(n_per_wire, *T, *g["A_neg"][i], wire_radius))
        wires.append(mn.Wire(n_per_wire, *g["A_neg"][i], *g["B_neg"][i], wire_radius))

    sim = mn.Mininec(f=freq_mhz, geo=wires)

    midp_arr = np.asarray(midp, dtype=float)
    feed_pulse = None
    for i in range(sim.pulses.pulse_idx):
        if np.linalg.norm(sim.pulses[i].point - midp_arr) < 1e-9:
            feed_pulse = i
            break
    if feed_pulse is None:
        raise RuntimeError("could not locate feed-midpoint pulse")

    exc = mn.Excitation(cvolt=1.0)
    sim.register_source(exc, pulse=feed_pulse)
    t0 = time.perf_counter()
    sim.compute()
    solve_ms = (time.perf_counter() - t0) * 1e3
    return complex(exc.impedance), solve_ms


def _fmt_z(z: complex) -> str:
    return f"{z.real:+8.3f} {z.imag:+8.3f}j"


def _print_geometry_summary(req: dict) -> None:
    g = web_server._fandipole_geometry(req)
    print(f"Geometry: n_bands={g['n_bands']}  freq={req['design_freq_mhz']:.3f} MHz")
    print(f"          band_lengths_m={list(g['band_lengths_m'])}")
    print(f"          slope={g['slope']}  cone_radius={g['cone_radius_m']} m")
    print(
        f"          wire_radius={req['wire_radius']} m  feed_gap={2 * g['feed_arclength']:.3f} m"
    )
    # Inter-arm angle at S (between the first two +y arms), useful when
    # cross-referencing PR #36's cone-angle sweep.
    if g["n_bands"] >= 2:
        a0 = np.array(g["A_pos"][0]) - np.array(g["S"])
        a1 = np.array(g["A_pos"][1]) - np.array(g["S"])
        cos_t = float(np.dot(a0, a1) / (np.linalg.norm(a0) * np.linalg.norm(a1)))
        ang_deg = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
        print(f"          inter-arm angle at S (bands 0,1): {ang_deg:.2f}°")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-bands", type=int, default=2, help="number of bands (1-5)")
    ap.add_argument(
        "--freq", type=float, default=14.3, help="design = measurement freq, MHz"
    )
    ap.add_argument(
        "--n-list",
        type=str,
        default="21,41,81",
        help="comma-separated per-wire segment counts to sweep",
    )
    ap.add_argument(
        "--pynec-max",
        type=int,
        default=81,
        help=(
            "skip PyNEC above this n_per_wire. NEXT_STEPS notes PyNEC trips "
            "its geometry-overlap check at N≥161 for K≥2 bands; default 81."
        ),
    )
    ap.add_argument(
        "--cone-radius",
        type=float,
        default=0.12,
        help="cone ring radius in m (default 0.12). Larger values reduce inter-arm proximity effects near the junction.",
    )
    args = ap.parse_args()
    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]

    _print_geometry_summary(
        _make_req(n_list[0], args.n_bands, args.freq, cone_radius_m=args.cone_radius)
    )

    header = (
        f"  {'N':>3}  | {'pysim (R + jX) Ω':>22} {'t_ms':>6} | "
        f"{'PyNEC (R + jX) Ω':>22} {'t_ms':>6} | "
        f"{'pymininec (R + jX) Ω':>22} {'t_ms':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for n in n_list:
        req = _make_req(n, args.n_bands, args.freq, cone_radius_m=args.cone_radius)
        z_pys, t_pys = solve_pysim(req)
        if n <= args.pynec_max:
            try:
                z_nec, t_nec = solve_pynec(req)
                nec_cell = f"{_fmt_z(z_nec):>22} {t_nec:>6.1f}"
            except Exception as e:  # noqa: BLE001
                nec_cell = f"{'FAIL: ' + type(e).__name__:>22} {'-':>6}"
        else:
            nec_cell = f"{'(skipped)':>22} {'-':>6}"
        try:
            z_mn, t_mn = solve_pymininec(req)
            mn_cell = f"{_fmt_z(z_mn):>22} {t_mn:>6.1f}"
        except Exception as e:  # noqa: BLE001
            mn_cell = f"{'FAIL: ' + type(e).__name__:>22} {'-':>6}"
        print(f"  {n:>3}  | {_fmt_z(z_pys):>22} {t_pys:>6.1f} | {nec_cell} | {mn_cell}")


if __name__ == "__main__":
    main()
