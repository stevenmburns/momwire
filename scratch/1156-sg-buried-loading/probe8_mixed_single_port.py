"""momwire#1156 probe 8: probe5's refined mixed deck with ONE feed at a time.

probe7 found SG's mixed-deck mutual coupling sign-inverted (pre-existing,
momwire#980 D2; its eps~ = 1 Y12 is minus free space's). A single-port Z is
exactly invariant under that inversion (it is D.G.D with D = diag(+-1) per
medium class, and loading is block-diagonal per wire), so these are the
decks the cross-trunk loading gate can read."""

import sys
import time

sys.path.insert(0, "scratch/1156-sg-buried-loading")

from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from probe5_mixed_refined import LOADS, TRUNKS, mixed, z  # noqa: E402

if __name__ == "__main__":
    ms = [int(x) for x in sys.argv[1:]] or [1, 2, 4]
    gaps = {}
    for fed in (0, 1):
        for m in ms:
            d0 = dict(mixed(m), feeds=[(fed, 2.5, 1 + 0j)])
            shifts = {}
            for cls in TRUNKS:
                t = time.time()
                z0 = z(cls, d0)[0]
                for lname, kw in LOADS.items():
                    shifts[(lname, cls)] = z(cls, dict(d0, **kw))[0] - z0
                print(
                    f"fed={fed} m={m} {cls.__name__:26s} bare {z0:.3f} "
                    f"({time.time() - t:.1f} s)",
                    flush=True,
                )
            for lname in LOADS:
                sg = shifts[(lname, SinusoidalGalerkinSolver)]
                row = f"    {lname:12s} SG {sg:.3f}"
                for cls in TRUNKS[1:]:
                    o = shifts[(lname, cls)]
                    g = abs(sg - o)
                    gaps.setdefault((fed, lname, cls.__name__[:5]), []).append(g)
                    row += f" | {cls.__name__[:5]} {o:.3f} gap {g:.3f}"
                print(row, flush=True)
    print("gap series:")
    for key, g in gaps.items():
        print(
            f"  fed={key[0]} {key[1]:12s} SG-{key[2]}: "
            + " -> ".join(f"{x:.3f}" for x in g)
        )
