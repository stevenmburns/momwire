"""False-stop hunt for the production WA stop rule (branch tree).

    python sweep.py OUT.jsonl

Nodes: 5 soils x R1 in {0.02, 0.1, 0.5, 1.5, 3} lam_m x 24 theta geometric
0.03-60 deg (the study's 360-node sweep, widened to five soils = 600), plus
under-floor angles {0.0125, 0.01, 0.005, 0.001, 0} deg at the same R1s.

Per node:
  prod     the production C++ engine (`below_six_integrals_batch`, the WA rule
           as shipped)
  prod_np  the production numpy engine
  strict   numpy WA with a much stricter rule: rtol 1e-13, three agreements,
           at least 8 lattice sums (a false stop of the production rule shows
           as prod - strict >> 1e-11 of scale)
A node that refuses (cap) is recorded as such.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import mpref as R  # noqa: E402

from momwire import _sommerfeld_below as SB  # noqa: E402

THETAS = [float(x) for x in np.geomspace(0.03, 60.0, 24)] + [
    0.0125,
    0.01,
    0.005,
    0.001,
    0.0,
]
R1S = (0.02, 0.1, 0.5, 1.5, 3.0)
SOILS = ("dan", "salt1.8", "fresh1.8", "salt28", "fresh28")


def vec(v, ref):
    return float(np.max(np.abs(v - ref)) / max(float(np.max(np.abs(ref))), 1e-300))


def comp(v, ref):
    return float(np.max(np.abs(v - ref) / np.maximum(np.abs(ref), 1e-300)))


def strict(kp, km, rho, h):
    keep = (SB._WA_MIN_SUMS, SB._WA_AGREE)
    SB._WA_MIN_SUMS, SB._WA_AGREE = 8, 3
    try:

        def f(lam):
            return SB._integrand_six_below(lam, rho, h, kp, km)

        val, hp, tp, conv, accel = SB._run_contour(
            f,
            kp,
            km,
            rho,
            h,
            1e-13,
            SB._ADAPT_DEPTH,
            SB._DETOUR,
            SB._GX,
            SB._GW,
            wa_pow=SB._WA_POW,
        )
    finally:
        SB._WA_MIN_SUMS, SB._WA_AGREE = keep
    return val, tp, conv


def main():
    out = open(sys.argv[1], "w")
    worst = 0.0
    for s in SOILS:
        f_hz, et = R.soil(s)
        for r1 in R1S:
            kp, km, _, _ = R.node(s, r1, 1.0)
            rhos, hs, ths = [], [], []
            for th in THETAS:
                _, _, rho, h = R.node(s, r1, th)
                rhos.append(rho)
                hs.append(h)
                ths.append(th)
            vals, tp, hp, conv, accel, _ = SB._six_below_accel(
                kp, km, np.array(rhos), np.array(hs), 1e-11, False
            )
            for i, th in enumerate(ths):
                rho, h = rhos[i], hs[i]
                sv = SB._FORCE_NUMPY
                SB._FORCE_NUMPY = True
                try:

                    def f(lam, rho=rho, h=h, kp=kp, km=km):
                        return SB._integrand_six_below(lam, rho, h, kp, km)

                    vnp, _, tpn, convn, _ = SB._run_contour(
                        f,
                        kp,
                        km,
                        rho,
                        h,
                        1e-11,
                        SB._ADAPT_DEPTH,
                        SB._DETOUR,
                        SB._GX,
                        SB._GW,
                        wa_pow=SB._WA_POW,
                    )
                finally:
                    SB._FORCE_NUMPY = sv
                vs, tps, convs = strict(kp, km, rho, h)
                row = dict(
                    soil=s,
                    r1_wl=r1,
                    theta_deg=th,
                    prod_panels=int(tp[i]),
                    prod_conv=bool(conv[i]),
                    np_panels=int(tpn),
                    strict_panels=int(tps),
                    strict_conv=bool(convs),
                    prod_vs_strict=(vec(vals[i], vs), comp(vals[i], vs)),
                    np_vs_strict=(vec(vnp, vs), comp(vnp, vs)),
                    prod_vs_np=vec(vals[i], vnp),
                )
                worst = max(worst, row["prod_vs_strict"][0])
                out.write(json.dumps(row) + "\n")
                out.flush()
            print(s, r1, f"worst so far {worst:.2e}", flush=True)


if __name__ == "__main__":
    main()
