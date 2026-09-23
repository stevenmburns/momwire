"""momwire#1156 probe 13 (needs mpmath, which the test extra does not
install): the conditioning of SG's closed-form loading terms.

probe12: the loading term on the crossing deck moves 6.5e-11 relative for a
one-ulp change of the wavelength. Which half — the (pre-existing) series
overlap or the new charge Gram — and how does each compare with an
extended-precision (mpmath, 40 digits) evaluation of the SAME closed form
from the same float64 coefficients?"""

import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, "tests")

from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_sg_buried_loading_1156 import assembled, dipole, xing  # noqa: E402

mp.mp.dps = 40


def terms(s, seg_view, geom, k):
    starts = np.asarray(seg_view["starts"])
    left, right, m_of_pair = s._shared_segment_pairs(starts)
    k_arr = (
        np.asarray(seg_view["k_entry"])[left]
        if "k_entry" in seg_view
        else np.full(left.size, k)
    )
    sig = seg_view["sigma"].astype(np.complex128)
    P, Q, R = sig * seg_view["A"], seg_view["B"], sig * seg_view["C"]
    h = np.asarray(geom["seg_h"], dtype=float)[m_of_pair]
    worst = {"series": 0.0, "charge": 0.0}
    mag = {"series": 0.0, "charge": 0.0}
    for t in range(left.size):
        kk, hh = k_arr[t], h[t]
        pl, pr, ql, qr, rl, rr = (
            P[left[t]],
            P[right[t]],
            Q[left[t]],
            Q[right[t]],
            R[left[t]],
            R[right[t]],
        )
        w_pr = (2.0 / kk) * np.sin(0.5 * kk * hh)
        hs = np.sin(kk * hh) / (2.0 * kk)
        ser = (
            pl * pr * hh
            + (pl * rr + rl * pr) * w_pr
            + ql * qr * (0.5 * hh - hs)
            + rl * rr * (0.5 * hh + hs)
        )
        chg = kk * kk * (ql * qr * (0.5 * hh + hs) + rl * rr * (0.5 * hh - hs))
        K, H = mp.mpc(kk), mp.mpf(hh)
        c = [mp.mpc(v) for v in (pl, pr, ql, qr, rl, rr)]
        W = (2 / K) * mp.sin(K * H / 2)
        HS = mp.sin(K * H) / (2 * K)
        SER = (
            c[0] * c[1] * H
            + (c[0] * c[5] + c[4] * c[1]) * W
            + c[2] * c[3] * (H / 2 - HS)
            + c[4] * c[5] * (H / 2 + HS)
        )
        CHG = K * K * (c[2] * c[3] * (H / 2 + HS) + c[4] * c[5] * (H / 2 - HS))
        for name, lo, hi in (("series", ser, SER), ("charge", chg, CHG)):
            worst[name] = max(worst[name], float(abs(mp.mpc(lo) - hi)))
            mag[name] = max(mag[name], float(abs(hi)))
    return {n: (worst[n], mag[n], worst[n] / mag[n]) for n in worst}


for label, d in (("dipole", dipole(n=20)), ("crossing", xing())):
    s = SinusoidalGalerkinSolver(
        **d,
        insulation_radius=[1.8e-3] * len(d["wires"]),
        insulation_eps_r=[3.5] * len(d["wires"]),
    )
    G, sv, geom, k = assembled(s, True)
    for n, (err, m, rel) in terms(s, sv, geom, k).items():
        print(
            f"{label:9s} {n:7s} max abs err {err:.3e} of max |term| {m:.3e}: rel {rel:.3e}"
        )
