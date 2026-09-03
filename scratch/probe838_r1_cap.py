"""momwire#838 part 2, phase 0: what actually limits the below/below R1 cap.

`_SOMM_BELOW_R1_CAP_LAMBDA_M = 2.0`'s comment records a clean cliff -- the
far annulus R1 in [2, 4) lambda_m interpolates at 2.4e-3, 12x worse than
everything inside it -- and momwire#568 is carried forward as saying theta
density is NOT the lever there, with radial density or a lateral-wave
asymptotic branch as the candidates.

Measured here, isolating one axis at a time: **theta IS the lever, and the
steep band is the whole of it.** Radial density is not, and neither is the
sub-1 deg band.

  * R1 axis, at theta fixed ON a node, over [2, 4) lambda_m:
    dr = 0.05 lambda_m already reads 1.6e-5 (A/7), 4.9e-6 (B/7), 1.3e-5
    (C/21). Halving it buys 15x that nobody needs. Not the lever.

  * theta axis, at R1 fixed on a node at 3 lambda_m: the STEEP band
    (30-90 deg, dtheta 2.5) reads 4.6e-4 to 8.4e-4, and halving to 1.25
    buys ~12x.

  * Full 2-D over the annulus, which reproduces the recorded cliff:
    steep 2.5 reads 8.3e-4 / 1.5e-3 / 2.0e-3 on B/7 / A/7 / C/21 -- the
    2.4e-3 in the constant's comment. The grazing band at 1.0 deg also
    misses on C/21 (5.9e-4).

So the honest reading of the earlier note is narrower than it has been
carried: the constant's OWN comment already names dtheta 2.5 -> 1.25 as the
fix and rejects it on COST ("~34 % more nodes for the steep band alone"),
not on effectiveness. "theta density is not the lever" is true of part 1's
sub-1 deg FLOOR, where the panel cap binds; it is false of the cap.

    python scratch/probe838_r1_cap.py
"""

import time

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below
from momwire._sommerfeld import SommerfeldGrid, _SURF_KEYS

C0 = 299792458.0
EPS0 = 8.8541878128e-12
DECKS = [
    ("A/7", (13.0, 0.005), 7e6),
    ("B/7", (20.0, 0.03), 7e6),
    ("C/21", (5.0, 0.001), 21e6),
]
BAR = 2e-4  # what the inner domain is gated at today
L4 = SommerfeldGrid._lagrange4


def _deck(gr, f):
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    eps_t = _ground_refl.eps_tilde(gr, om, EPS0)
    return eps_t, k2, om, below.lambda_medium(eps_t, k2)


def surf(eps_t, k2, om, r, t):
    d = below.iv_surfaces_direct_below(eps_t, k2, r, t, rtol=1e-9, omega=om)
    return np.stack([d[k] for k in _SURF_KEYS])


def bicubic(r_nodes, th_nodes, vals, rq, tq):
    fr = (rq - r_nodes[0]) / (r_nodes[1] - r_nodes[0])
    ft = (tq - th_nodes[0]) / (th_nodes[1] - th_nodes[0])
    i0 = np.clip(np.floor(fr).astype(int) - 1, 0, len(r_nodes) - 4)
    j0 = np.clip(np.floor(ft).astype(int) - 1, 0, len(th_nodes) - 4)
    wr, wt = L4(fr - i0), L4(ft - j0)
    ii = i0[:, None] + np.arange(4)[None, :]
    jj = j0[:, None] + np.arange(4)[None, :]
    return np.einsum("snij,ni,nj->sn", vals[:, ii[:, :, None], jj[:, None, :]], wr, wt)


def annulus_err(eps_t, k2, om, lam_m, th_lo, th_hi, dth, drl=0.05):
    """Worst relative bicubic error over R1 in [2, 4) lambda_m."""
    dr = drl * lam_m
    r_nodes = np.arange(2.0 * lam_m - 2 * dr, 4.0 * lam_m + 3 * dr, dr)
    n_th = int(round((th_hi - th_lo) / dth)) + 1
    th_nodes = np.radians(th_lo + dth * np.arange(n_th))
    rr, tt = np.meshgrid(r_nodes, th_nodes, indexing="ij")
    vals = surf(eps_t, k2, om, rr.ravel(), tt.ravel()).reshape(
        len(_SURF_KEYS), len(r_nodes), n_th
    )
    rq = r_nodes[2:-3] + 0.5 * dr
    tq = 0.5 * (th_nodes[2:-2] + th_nodes[3:-1])
    RQ, TQ = np.meshgrid(rq, tq, indexing="ij")
    got = bicubic(r_nodes, th_nodes, vals, RQ.ravel(), TQ.ravel())
    ref = surf(eps_t, k2, om, RQ.ravel(), TQ.ravel())
    scale = np.abs(ref).max(axis=0)[None, :]
    return float((np.abs(got - ref) / scale).max()), len(r_nodes) * n_th


def main():
    print(f"far annulus R1 in [2, 4) lambda_m; bar {BAR:.0e}\n")
    print(
        f"{'deck':6s} {'band':>6} {'dtheta':>7} {'worst rel':>11} {'x bar':>7} {'nodes':>7}"
    )
    for name, gr, f in DECKS:
        eps_t, k2, om, lam_m = _deck(gr, f)
        for tag, lo, hi, dths in (
            ("steep", 30.0, 90.0, (2.5, 1.25, 1.0)),
            ("graze", 1.0, 30.0, (1.0, 0.5)),
        ):
            for dth in dths:
                e, n = annulus_err(eps_t, k2, om, lam_m, lo, hi, dth)
                print(
                    f"{name:6s} {tag:>6} {dth:7.3f} {e:11.3e} {BAR / e:7.1f} {n:7d}"
                    f"{'' if e < BAR else '   <- FAIL'}"
                )
        print()

    print("far-zone fill cost, and where it sits (A/7)")
    eps_t, k2, om, lam_m = _deck(*DECKS[0][1:])
    dr = 0.05 * lam_m
    n_r = len(np.arange(2.0 * lam_m - 2 * dr, 4.0 * lam_m + 3 * dr, dr))
    total = 0.0
    for tag, lo, hi, dth in (
        ("band", 0.1, 1.0, 0.225),
        ("graze", 1.0, 30.0, 0.5),
        ("steep", 30.0, 90.0, 1.0),
    ):
        n_th = int(round((hi - lo) / dth)) + 1
        th = lo + dth * np.arange(n_th)
        p = float(np.sum(6.4 / np.tan(np.radians(th)))) * n_r
        total += p
        print(f"  {tag:6s} dtheta {dth:5.3f}  {n_r * n_th:6d} nodes  {p:12.0f} panels")
    print(f"  TOTAL                              {total:12.0f} panels")
    t0 = time.perf_counter()
    below.SommerfeldGridBelow(eps_t, k2, 2.0 * lam_m, omega=om)
    print(
        f"  against today's whole grid: 101058 panels, "
        f"{time.perf_counter() - t0:.2f} s fill"
    )


if __name__ == "__main__":
    main()
