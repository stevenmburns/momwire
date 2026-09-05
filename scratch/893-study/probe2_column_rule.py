"""momwire#895 deliverable 2: a FIXED composite rule per (rho, z') column.

The census (probe1) says every point of the crossing set sits in a column of
>= 32 mast heights sharing (rho, z'). On such a column `_core`'s only z
dependence is e^{-gamma_p z}, and the only rho dependence is the Bessel /
Hankel factor — so the walk's path, nodes, weights, path derivative and
Bessel factor are shared, and a point is one exponential per node plus a
(6 x K) . K product.

This probe builds that rule on the walk's OWN path (head detour, real-axis
mid, rotated rays), non-adaptively at a chosen resolution, converged for the
column's smallest s, and gates every point against `six_point` at relative
tolerance. It reports the node count K, the max relative error over the
column, and the per-point cost, on three real columns of the BLE N = 4 deck.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import hankel1, hankel2, jv

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(HERE))

import momwire._near_interface as ni  # noqa: E402
from momwire._sommerfeld_below import _GX, _GW, _DETOUR  # noqa: E402
from momwire._sommerfeld_transmitted import _gamma, k_medium  # noqa: E402
import test_ble_1937_838 as ble  # noqa: E402
from probe1_census import census  # noqa: E402

RAY = np.exp(1j * np.pi / 4.0)


def _gauss_nodes(edges, p):
    """Gauss nodes/weights on [e0,e1],[e1,e2]... each split into 2^p panels."""
    xs, ws = [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        sub = np.linspace(e0, e1, 2**p + 1)
        for a, b in zip(sub[:-1], sub[1:]):
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            xs.append(mid + half * _GX)
            ws.append(_GW * half)
    return np.concatenate(xs), np.concatenate(ws)


def column_rule(
    rho, k_p, k_m, s_min, p_head=1, p_mid=1, p_ray=1, lam_mult=ni._LAM_MULT
):
    """Nodes lam_k (complex) and weights w_k (path derivative and the Bessel
    factor folded in) for one (rho, z') column whose smallest s is s_min."""
    kk = max(k_p, abs(k_m))
    a_head = 1.1 * kk
    lam_top = lam_mult * kk
    if s_min > 0.0 and ni._FAR_PAIR_KILL / s_min < lam_top:
        lam_kill = ni._FAR_PAIR_KILL / s_min
        a_head = max(2.2 * k_p, min(a_head, lam_kill))
        lam_top = max(1.5 * a_head, lam_kill)
    # --- head: the first-quadrant detour, edges seeded as `_head` seeds them
    H = min(0.35 * a_head, _DETOUR / max(rho, 1e-12))
    H = max(H, 1e-6 * a_head)
    edges = {0.0, a_head}
    for i in range(1, 7):
        edges.add(a_head * i / 7.0)
    for mk in (k_p, abs(k_m.real)):
        for w in (0.0, -0.15, 0.15, -0.4, 0.4):
            v = mk * (1.0 + w)
            if 0.0 < v < a_head:
                edges.add(v)
    t, wt = _gauss_nodes(sorted(edges), p_head)
    lam_h = t + 1j * H * np.sin(np.pi * t / a_head)
    dl_h = 1.0 + 1j * H * (np.pi / a_head) * np.cos(np.pi * t / a_head)
    w_h = wt * dl_h * jv(0, lam_h * rho)
    # --- mid: real axis [a_head, lam_top]
    t, wt = _gauss_nodes([a_head, lam_top], p_mid)
    lam_m = t.astype(complex)
    w_m = wt * jv(0, lam_m * rho)
    # --- tail: rotated rays, geometric panels from the lam0 scale doubling
    # toward the decay scale, out to where e^{-t(s+rho)/sqrt2} is dead.
    scale = np.sqrt(2.0) / (s_min + rho)
    step = min(0.25 * scale, lam_top)
    t_edges = [0.0]
    while t_edges[-1] < 60.0 * scale:
        t_edges.append(t_edges[-1] + step)
        step *= 2.0
    tt, wtt = _gauss_nodes(t_edges, p_ray)
    if rho == 0.0:
        lam_t = lam_top + tt * RAY
        w_t = wtt * RAY
    else:
        up = lam_top + tt * RAY
        dn = lam_top + tt * np.conj(RAY)
        lam_t = np.concatenate([up, dn])
        w_t = np.concatenate(
            [
                wtt * RAY * 0.5 * hankel1(0, up * rho),
                wtt * np.conj(RAY) * 0.5 * hankel2(0, dn * rho),
            ]
        )
    lam = np.concatenate([lam_h, lam_m, lam_t])
    w = np.concatenate([w_h, w_m, w_t])
    return lam, w


def column_factors(lam, w, k_p, k_m):
    """The z-independent part of `_core` at every node, weights folded in:
    F (6, K) such that six(z, z') = F @ exp(g_m z' - g_p z)."""
    g_p = _gamma(lam, k_p)
    g_m = _gamma(lam, k_m)
    u = 2.0 * lam / (g_p + g_m)
    v = 2.0 * lam / (k_m * k_m * g_p + k_p * k_p * g_m)
    wv = (g_p - g_m) * v
    F = np.stack([u, v, wv, -g_p * wv, -g_p * g_m * v, g_m * wv]) * w
    return F, g_p, g_m


def six_column(F, g_p, g_m, zs, zp):
    zs = np.asarray(zs, float)
    with np.errstate(under="ignore"):
        E = np.exp(g_m[None, :] * zp - g_p[None, :] * zs[:, None])  # (nz, K)
    return E @ F.T  # (nz, 6)


def run_column(label, rho, zp, zs, eps_t, k2, res):
    k_p = float(k2)
    k_m = k_medium(complex(eps_t), k_p)
    zs = np.sort(np.asarray(zs, float))
    s_min = float(np.min(zs - zp))
    ref = np.array([ni.six_point(eps_t, k2, rho, z, zp) for z in zs])
    print(
        f"\n== {label}: rho {rho:.4g}, z' {zp:.4g}, {len(zs)} z in [{zs.min():.3g}, {zs.max():.3g}], s_min {s_min:.3g}"
    )
    for p_head, p_mid, p_ray in res:
        lam, w = column_rule(rho, k_p, k_m, s_min, p_head, p_mid, p_ray)
        F, g_p, g_m = column_factors(lam, w, k_p, k_m)
        t0 = time.perf_counter()
        for _ in range(5):
            got = six_column(F, g_p, g_m, zs, zp)
        dt = (time.perf_counter() - t0) / 5 / len(zs)
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
        worst = np.unravel_index(np.argmax(rel), rel.shape)
        print(
            f"   p=({p_head},{p_mid},{p_ray}) K={len(lam):5d}  max rel {rel.max():.2e} "
            f"(z={zs[worst[0]]:.3g}, {ni.KEYS[worst[1]]})  median rel {np.median(rel):.1e}  "
            f"{1e6 * dt:.1f} us/pt"
        )


if __name__ == "__main__":
    uniq, eps_t, k2 = census(ble.ble_deck(4), "BLE 45 ft, N = 4 (for the columns)")
    from collections import defaultdict

    cols = defaultdict(list)
    for r, z, zp in uniq:
        cols[(r, zp)].append(z)
    rhos = sorted({r for r, _ in cols})
    a_fold = rhos[0]
    depth = min(zp for _, zp in cols)
    picks = [
        ("on-axis, hub depth", a_fold, depth),
        ("on-axis, z' = 0", a_fold, 0.0),
        ("mid radial", min(rhos, key=lambda r: abs(r - 1.0)), depth),
        ("radial end", rhos[-1], depth),
    ]
    res = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    for label, rho, zp in picks:
        zs = cols.get((rho, zp))
        if not zs:
            print(f"\n== {label}: no column at rho {rho:.4g}, z' {zp:.4g}")
            continue
        run_column(label, rho, zp, zs, eps_t, k2, res)
