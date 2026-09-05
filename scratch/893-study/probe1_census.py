"""momwire#893 / #895 deliverable 1: the crossing serve's point set, measured.

Wraps `_near_interface.designed_tables` to record every (rho_eff, z, z') the
crossing fill asks for on two real decks, then reports: unique triples, the
distinct z' and (z, z') groups (the "nearly 2-D" claim), the rho count per
group, and the per-point walk cost bucketed by (rho, s) on a sample of the
unique set — single-threaded, numpy walk and C++ twin both.
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

import momwire._near_interface as ni  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
import test_ble_1937_838 as ble  # noqa: E402


def census(build, label):
    asked = []
    real = ni.designed_tables

    def spy(eps_t, k2, rho, z, zp, **kw):
        r, zz, zpz = np.broadcast_arrays(
            np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
        )
        asked.extend(zip(r.ravel().tolist(), zz.ravel().tolist(), zpz.ravel().tolist()))
        spy.eps_t, spy.k2 = eps_t, k2
        return real(eps_t, k2, rho, z, zp, **kw)

    ni.designed_tables = spy
    try:
        t0 = time.perf_counter()
        z_in, _ = BSplineSolver(**build).compute_impedance()
        wall = time.perf_counter() - t0
    finally:
        ni.designed_tables = real

    uniq = sorted(set(asked))
    zps = sorted({t[2] for t in uniq})
    groups = defaultdict(list)
    for r, z, zp in uniq:
        groups[(z, zp)].append(r)
    per_group = np.array([len(v) for v in groups.values()])
    print(f"\n== {label}: Z = {z_in:.4f}, wall {wall:.1f} s")
    print(
        f"   asked {len(asked)}, unique {len(uniq)} ({len(asked) / max(1, len(uniq)):.2f}x dup)"
    )
    print(f"   distinct z' {len(zps)}: {np.array(zps)}")
    print(f"   distinct z  {len({t[1] for t in uniq})}")
    print(
        f"   (z, z') groups {len(groups)}; rho per group min/median/max "
        f"{per_group.min()}/{int(np.median(per_group))}/{per_group.max()}"
    )
    rho = np.array([t[0] for t in uniq])
    s = np.array([t[1] - t[2] for t in uniq])
    # How much of the unique set lives in groups a shared-rho contour can
    # serve: fraction of points in (z, z') groups of size >= k.
    sizes = {g: len(v) for g, v in groups.items()}
    pt_size = np.array([sizes[(z, zp)] for _, z, zp in uniq])
    print(
        "   share of unique points in (z, z') groups of size >= k:  "
        + "  ".join(f"k={k}: {np.mean(pt_size >= k):.2f}" for k in (2, 4, 8, 32))
    )
    a_min = rho.min()
    on_axis = np.isclose(rho, a_min)
    print(
        f"   on-axis pairs (rho == a-fold {a_min:.4g}): {on_axis.sum()} of {len(uniq)} "
        f"= mast x rise; off-axis (mast x radials): {(~on_axis).sum()}"
    )
    # and the alternative grouping — shared z' and rho, varying z (a mast column)
    cols = defaultdict(int)
    for r, z, zp in uniq:
        cols[(r, zp)] += 1
    col_size = np.array([cols[(r, zp)] for r, _, zp in uniq])
    print(
        "   share in (rho, z') columns of size >= k:  "
        + "  ".join(f"k={k}: {np.mean(col_size >= k):.2f}" for k in (2, 4, 8, 32))
    )
    print(
        f"   rho range {rho.min():.4g}..{rho.max():.4g}, s range {s.min():.4g}..{s.max():.4g}"
    )
    return uniq, spy.eps_t, spy.k2


def cost_by_region(uniq, eps_t, k2, n_sample=240, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(uniq), size=min(n_sample, len(uniq)), replace=False)
    tri = np.asarray([uniq[i] for i in idx], float)
    rho, z, zp = tri.T
    s = z - zp
    # numpy walk, one point at a time
    t = np.empty(len(tri))
    for i, (r, zz, zpz) in enumerate(tri):
        t0 = time.perf_counter()
        ni.six_point(eps_t, k2, r, zz, zpz)
        t[i] = time.perf_counter() - t0
    # C++ twin, whole sample in one batch, single thread
    k_p = float(k2)
    k_m = ni.k_medium(complex(eps_t), k_p)
    t0 = time.perf_counter()
    ni._nia.near_interface_six_batch(
        k_p,
        k_m,
        np.ascontiguousarray(rho),
        np.ascontiguousarray(z),
        np.ascontiguousarray(zp),
        1e-10,
        ni._LAM_MULT,
        ni._ADAPT_DEPTH,
        ni._DETOUR,
        ni._GX,
        ni._GW,
    )
    t_cpp = (time.perf_counter() - t0) / len(tri)
    print(
        f"   sample {len(tri)}: numpy {1e3 * t.mean():.2f} ms/pt (median {1e3 * np.median(t):.2f}), "
        f"C++ {1e3 * t_cpp:.3f} ms/pt single-thread"
    )
    # region buckets on (rho, s) in decades
    kk = max(k_p, abs(k_m))
    for name, mask in (
        ("corner  s+rho < 0.02", s + rho < 0.02),
        ("near    0.02..0.5", (s + rho >= 0.02) & (s + rho < 0.5)),
        ("mid     0.5..5", (s + rho >= 0.5) & (s + rho < 5)),
        ("far     >= 5", s + rho >= 5),
    ):
        if mask.any():
            print(
                f"     {name:22s} n={mask.sum():4d}  numpy {1e3 * t[mask].mean():.2f} ms/pt  "
                f"kR ~ {kk * np.median(np.hypot(rho[mask], s[mask])):.2f}"
            )


if __name__ == "__main__":
    u, e, k = census(ble.ble_deck(4), "BLE 45 ft, N = 4")
    cost_by_region(u, e, k)
    u, e, k = census(ble.ble_deck(16), "BLE 45 ft, N = 16")
    cost_by_region(u, e, k, n_sample=120)
