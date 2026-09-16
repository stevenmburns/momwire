"""P-E′ (amendment, registered before running): empymod on an R ladder.

`gate_empymod.py calib` found empymod unconverged at 50–200 λ₀ over real
soil for every Hankel transform at their default kernel sampling (O(1)
against the direct + Fresnel-image reference that momwire's readout
reproduces to 5e-11), and the qwe transform with a 6000-points-per-decade
kernel grid reaching the expected 1/(k_pR) residual at 20 λ₀. So the far
range is taken by extrapolation rather than by distance:

    rungs R = 20, 40, 80 λ₀ at 21 MHz, qwe / 6000 ppd;
    control: the above-soil source (reference = the image formula);
    subject: the buried HED (d = 0.15 m) and VED (d = 0.6 m).

Predictions: raw residual at each rung within 2× of 1/(k_pR) for θ ≤ 60°;
the (40, 80) Richardson pair within 3e-3 of the closed form on the subject
and within the same bar on the control (the control bounds empymod's own
error and is quoted beside every subject number).
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

import gate_empymod as ge
import proto_far as p

SETTINGS = dict(
    ht="qwe",
    htarg={
        "rtol": 1e-12,
        "atol": 1e-30,
        "nquad": 51,
        "maxint": 2000,
        "pts_per_dec": 6000,
    },
)
THETAS = (5.0, 20.0, 40.0, 60.0)
PHIS = (0.0, 30.0, 90.0)
RUNGS = (20, 40, 80)


def dirs():
    return [(math.radians(t), math.radians(f)) for t in THETAS for f in PHIS]


def fields(src, kind, freq, eps_r, sigma, k_p, R):
    obs = [tuple(R * ge.spherical(t, f)[0]) for t, f in dirs()]
    E = ge.e_spec(src, kind, obs, freq, eps_r, sigma, SETTINGS)
    g = np.exp(-1j * k_p * R) / R
    out = []
    for (t, f), e in zip(dirs(), E):
        _, th, ph = ge.spherical(t, f)
        out.append(np.array([e @ th, e @ ph]) / g)
    return np.array(out)


def main():
    freq = 21e6
    out, misses = {}, []
    for soil in ("A", "B", "C"):
        eps_r, sigma = p.SOILS[soil]
        eps_t, k_p, k_m, w = p.medium(freq, eps_r, sigma)
        lam0 = 2.0 * math.pi / k_p
        cases = [
            (
                "control-HED",
                "HED",
                (0.0, 0.0, 2.0),
                [
                    ge.closed_form_above("HED", 2.0, k_p, eps_t, w, t, f)
                    for t, f in dirs()
                ],
            ),
            (
                "control-VED",
                "VED",
                (0.0, 0.0, 2.0),
                [
                    ge.closed_form_above("VED", 2.0, k_p, eps_t, w, t, f)
                    for t, f in dirs()
                ],
            ),
            (
                "buried-HED",
                "HED",
                (0.0, 0.0, -0.15),
                [
                    ge.closed_form_buried("HED", 0.15, k_p, k_m, w, t, f)
                    for t, f in dirs()
                ],
            ),
            (
                "buried-VED",
                "VED",
                (0.0, 0.0, -0.6),
                [
                    ge.closed_form_buried("VED", 0.6, k_p, k_m, w, t, f)
                    for t, f in dirs()
                ],
            ),
        ]
        for name, kind, src, closed in cases:
            t0 = time.time()
            ref = np.array(closed)
            vals = {
                n: fields(src, kind, freq, eps_r, sigma, k_p, n * lam0) for n in RUNGS
            }
            raw = {
                n: (np.linalg.norm(vals[n] - ref, axis=1) / np.linalg.norm(ref, axis=1))
                for n in RUNGS
            }
            rich = 2.0 * vals[80] - vals[40]
            rel_r = np.linalg.norm(rich - ref, axis=1) / np.linalg.norm(ref, axis=1)
            key = f"{soil}/{name}"
            rec = {
                "raw_max": {n: float(raw[n].max()) for n in RUNGS},
                "raw_expect_1_over_kR": {n: 1.0 / (k_p * n * lam0) for n in RUNGS},
                "richardson_max": float(rel_r.max()),
                "richardson_rows": [
                    {"theta": math.degrees(t), "phi": math.degrees(f), "rel": float(r)}
                    for (t, f), r in zip(dirs(), rel_r)
                ],
                "secs": time.time() - t0,
            }
            out[key] = rec
            print(
                f"{key:14s} raw {[f'{raw[n].max():.1e}' for n in RUNGS]} (1/kR {[f'{v:.1e}' for v in rec['raw_expect_1_over_kR'].values()]})  Richardson {rel_r.max():.2e}  ({rec['secs']:.0f}s)",
                flush=True,
            )
            if rel_r.max() > 3e-3 or any(
                raw[n].max() > 2.0 / (k_p * n * lam0) for n in RUNGS
            ):
                misses.append(key)
    (p.HERE / "p_e_ladder.json").write_text(json.dumps(out, indent=1))
    print("P-E′", "HIT" if not misses else f"MISS {misses}")


if __name__ == "__main__":
    main()
