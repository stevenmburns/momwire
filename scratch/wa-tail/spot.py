"""Extended-precision spot-check of the below/below tail: today's engine,
the long-walk reference and WA against an mpmath reference.

    python spot.py [nproc] > spot.log      (writes spot.jsonl)

Routes per node:
  cpp       today's production engine (`_six_integrals_below`, C++ when built)
  numpy     today's numpy engine (`_run_contour`)
  long      the study's long-walk reference: numpy base at rtol 1e-12,
            400,000-panel cap
  long_cf   the same long walk with D2 in its cancellation-free form
            (diagnostic for WHY long walks drift; see `six_cf`)
  wa        the study's prototype (omega exponent alpha - 1/2, as studied)
  wa_b      the same with omega exponent alpha (the integrand's own power)
"""

from __future__ import annotations

import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import mpref as R  # noqa: E402
import study_methods as Mt  # noqa: E402

from momwire import _sommerfeld_below as SB  # noqa: E402
from momwire._sommerfeld import _bessel_j0_j1x, _gamma  # noqa: E402

SWEEP_TH = [float(x) for x in np.geomspace(0.03, 60.0, 24)]


def nodes():
    out = []
    for s in ("dan", "salt1.8", "fresh28"):
        for th in SWEEP_TH:
            if th <= 0.16:
                out.append((s, 0.02, th, "corner"))
    for s in ("dan", "salt1.8"):
        for th in (0.05, 0.1):
            out.append((s, 2.0, th, "r2-lossy-graze"))
    for r1 in (0.2, 1.0):
        for th in (0.05, 45.0):
            out.append(("dan", r1, th, "dan"))
    for s in ("salt1.8", "fresh1.8", "salt28", "fresh28"):
        for th in (0.05, 1.0, 45.0):
            out.append((s, 1.0, th, "soils"))
    for s in ("dan", "salt1.8", "fresh28"):
        out.append((s, 1.0, 0.005, "under-floor"))
    return out


def six_cf(lam, rho, h, k_p, k_m):
    """`_integrand_six_below` with D2 = 2 k2^2 (k1^2 - k2^2) /
    [(g1 + g2)(k1^2 g2 + k2^2 g1) g2 (k1^2 + k2^2)], algebraically equal to
    the engine's difference form but with no cancellation at large lam."""
    lam = np.asarray(lam, dtype=np.complex128)
    g1 = _gamma(lam, k_p)
    g2 = _gamma(lam, k_m)
    k1s, k2s = k_p * k_p, k_m * k_m
    S = k1s + k2s
    d1 = 2.0 / (g1 + g2) - 2.0 * k2s / (g2 * S)
    d2 = 2.0 * k2s * (k1s - k2s) / ((g1 + g2) * (k1s * g2 + k2s * g1) * g2 * S)
    e = np.exp(-g2 * h)
    x = lam * rho
    b0, b1x = _bessel_j0_j1x(x)
    l2 = lam * lam
    l3 = l2 * lam
    common = d2 * e
    return np.stack(
        [
            common * (b1x - b0) * l3,
            common * g2 * g2 * b0 * lam,
            -common * g2 * (b1x * x) * l2,
            -common * b1x * l3,
            common * b0 * lam,
            d1 * e * b0 * lam,
        ]
    )


def long_walk(kp, km, rho, h, f):
    val, hp, tp, conv, accel = SB._run_contour(
        f,
        kp,
        km,
        rho,
        h,
        1e-12,
        SB._ADAPT_DEPTH,
        SB._DETOUR,
        SB._GX,
        SB._GW,
        max_panels=400_000,
    )
    return val, tp, bool(conv)


def rel(v, ref):
    v, ref = np.asarray(v), np.asarray(ref)
    comp = float(np.max(np.abs(v - ref) / np.maximum(np.abs(ref), 1e-300)))
    vec = float(np.max(np.abs(v - ref)) / max(float(np.max(np.abs(ref))), 1e-300))
    return comp, vec


def one(nd):
    s, r1, th, tag = nd
    kp, km, rho, h = R.node(s, r1, th)
    f_hz, et = R.soil(s)
    t0 = time.time()
    ref = R.reference(kp, km, rho, h)
    t_ref = time.time() - t0
    refv = np.array(ref["value"])
    row = dict(
        soil=s,
        r1_wl=r1,
        theta_deg=th,
        tag=tag,
        rho=rho,
        h=h,
        ref_spread=ref["spread"],
        ref_panels=ref["panels"],
        ref_wall=t_ref,
        ref=[[z.real, z.imag] for z in refv],
    )

    def f6(lam):
        return SB._integrand_six_below(lam, rho, h, kp, km)

    def fcf(lam):
        return six_cf(lam, rho, h, kp, km)

    if th >= SB._SOMM_BELOW_TH_MIN_DEG:
        v = SB._six_integrals_below(et, kp, rho, h)
        row["cpp"] = rel(v, refv)
        sv = SB._FORCE_NUMPY
        SB._FORCE_NUMPY = True
        hl = SB.Health()
        v = SB._six_integrals_below(et, kp, rho, h, health=hl)
        SB._FORCE_NUMPY = sv
        row["numpy"] = rel(v, refv)
        row["numpy_tail_panels"] = hl.max_tail_panels
    v, tp, conv = long_walk(kp, km, rho, h, f6)
    row["long"] = rel(v, refv) + (tp, conv)
    v, tp, conv = long_walk(kp, km, rho, h, fcf)
    row["long_cf"] = rel(v, refv) + (tp, conv)
    for name, extra in (("wa", -0.5), ("wa_b", 0.0)):
        r = Mt.run(kp, km, rho, h, "wa", extra=extra)
        row[name] = rel(r["value"], refv) + (r["tail_panels"], r["converged"])
    return row


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    nds = nodes()
    # longest first: the corner's small theta
    nds.sort(
        key=lambda n: (
            -(n[1] / max(math.tan(math.radians(n[2])), 1e-9)) if n[1] < 0.1 else 0
        )
    )
    with Pool(nproc) as p, open(__file__.rsplit("/", 1)[0] + "/spot.jsonl", "w") as fh:
        for row in p.imap_unordered(one, nds):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"{row['soil']:8s} R1={row['r1_wl']:<5} th={row['theta_deg']:<8.4g} "
                f"spread={row['ref_spread']:.0e} "
                + " ".join(
                    f"{k}={row[k][0]:.1e}"
                    + (f"/{row[k][2]}" if len(row[k]) > 2 else "")
                    for k in ("cpp", "numpy", "long", "long_cf", "wa", "wa_b")
                    if k in row
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
