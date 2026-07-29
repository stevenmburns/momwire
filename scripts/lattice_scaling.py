"""Scaling study: lattice-FFT vs per-pair ArrayBlock on dipole lattices.

Builds Px × Py grids of identical vertical half-wave dipoles (0.7 λ spacing —
the docs/lattice-fft-plan.md driving example is 100x100) and reports build /
matvec / solve wall-clock, GMRES iterations, storage, and peak RSS for the
selected coupling representation.

Run:
  python scripts/lattice_scaling.py 8x8 16x16 32x32          # lattice-FFT (auto)
  python scripts/lattice_scaling.py --path pair 8x8 16x16    # per-pair baseline
  python scripts/lattice_scaling.py --nseg 15 100x100
"""

import argparse
import resource
import time

import numpy as np

from momwire.array_block import ArrayBlockSolver, reset_array_caches

WAVELENGTH = 22.0


def run(px, py, nseg, lattice_fft):
    reset_array_caches()
    half = 0.48 * WAVELENGTH / 2
    sp = 0.7 * WAVELENGTH
    wires = [
        np.array([[i * sp, j * sp, -half], [i * sp, j * sp, half]])
        for i in range(px)
        for j in range(py)
    ]
    P = len(wires)
    sim = ArrayBlockSolver(
        wires=wires,
        degree=1,
        n_per_edge_per_wire=[[nseg]] * P,
        wavelength=WAVELENGTH,
        feed_wire_index=P // 2,
        lattice_fft=lattice_fft,
    )
    t0 = time.perf_counter()
    H = sim.build_array_blocks()
    t_build = time.perf_counter() - t0
    st = H.stats()

    x = np.random.default_rng(0).standard_normal(st["n"]).astype(np.complex128)
    t0 = time.perf_counter()
    H.matvec(x)
    t_mv = time.perf_counter() - t0

    t0 = time.perf_counter()
    z, _ = sim.compute_impedance()
    t_solve = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(
        f"{px}x{py} P={P} n={st['n']} op={type(H).__name__} "
        f"fills={sim._last_n_coupling_aca} store={100 * st['compression']:.2f}% "
        f"build={t_build:.2f}s matvec={t_mv * 1e3:.1f}ms "
        f"solve={t_solve:.2f}s it={max(sim._last_solve_iters)} "
        f"Z={z:.1f} rss={rss:.0f}MB",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("grids", nargs="+", help="PxQ grid sizes, e.g. 16x16")
    ap.add_argument(
        "--path",
        choices=["fft", "pair"],
        default="fft",
        help="coupling representation (fft = lattice-FFT, pair = per-pair ACA)",
    )
    ap.add_argument("--nseg", type=int, default=15, help="segments per dipole")
    args = ap.parse_args()
    for g in args.grids:
        px, py = (int(v) for v in g.split("x"))
        run(px, py, args.nseg, lattice_fft=(args.path == "fft"))


if __name__ == "__main__":
    main()
