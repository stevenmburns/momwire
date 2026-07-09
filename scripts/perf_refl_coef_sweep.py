"""Perf + memory check for the ground_eps swept path (plan Phase 2).

41-frequency compute_y_matrix_swept on the validation dipole at its captured
segmentation (45 segs) and at a 4x-boosted segmentation (~180 segs), for
free space / PEC image / ground_eps. Also times one k's image J fill vs the
Python weighted image assembly to settle the "C++ second-weight-table
extension?" question, and reports peak RSS.

Run:  .venv/bin/python scripts/perf_refl_coef_sweep.py
"""

import os
import resource
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"),
)

from fixtures_refl_coef_geoms import GEOMS  # noqa: E402

from momwire import BSplineSolver  # noqa: E402

N_FREQ = 41


def build(kw, boost, **ground):
    kw = dict(kw)
    if boost != 1:
        kw["n_per_edge_per_wire"] = [
            [n * boost for n in edges] for edges in kw["n_per_edge_per_wire"]
        ]
    return BSplineSolver(**kw, **ground)


def time_sweep(sim, k_arr):
    t0 = time.perf_counter()
    sim.compute_y_matrix_swept(k_arr)
    return time.perf_counter() - t0


def main():
    kw = GEOMS[("dipole", 0.2)]
    for boost in (1, 4):
        sim = build(kw, boost, ground_z=0.0, ground_eps=(10.0, 0.002))
        geom = sim._build_geometry()
        n_seg = geom["tangents"].shape[0]
        k0 = sim.k
        k_arr = np.linspace(0.95 * k0, 1.05 * k0, N_FREQ)

        t_free = time_sweep(build(kw, boost), k_arr)
        t_pec = time_sweep(build(kw, boost, ground_z=0.0), k_arr)
        t_eps = time_sweep(sim, k_arr)

        # single-k breakdown of the ground_eps extra work (best of 5 after a
        # warm-up call each — first-call einsum/OMP setup otherwise dominates)
        supp_seg, polys, *_ = sim._build_basis_polynomials(geom)

        def best_of(fn, n=5):
            fn()
            times = []
            for _ in range(n):
                t0 = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t0)
            return min(times)

        t_fill = best_of(lambda: sim._build_J_image_blocks(geom, k0))
        J_img = sim._build_J_image_blocks(geom, k0)
        t_asm = best_of(lambda: sim._image_Z_refl(J_img, supp_seg, polys, geom))
        td_img = sim._image_tangent_dot(geom["tangents"])
        t_asm_pec = best_of(
            lambda: sim._assemble_Z(J_img, supp_seg, polys, geom, td_all=td_img)
        )

        print(f"N_seg={n_seg}  ({N_FREQ}-freq sweep)")
        print(f"  free space : {t_free:7.2f} s")
        print(f"  PEC image  : {t_pec:7.2f} s")
        print(
            f"  ground_eps : {t_eps:7.2f} s   (+{100 * (t_eps / t_pec - 1):.1f}% vs PEC)"
        )
        print(f"  per-k image J fill        : {1e3 * t_fill:8.1f} ms")
        print(
            f"  per-k weighted image asm  : {1e3 * t_asm:8.1f} ms "
            f"({100 * t_asm / t_fill:.0f}% of fill)"
        )
        print(f"  per-k PEC asm (C++)       : {1e3 * t_asm_pec:8.1f} ms")

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak RSS: {rss_mb:.0f} MB")


if __name__ == "__main__":
    main()
