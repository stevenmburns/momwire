"""P-D′ (amendment, registered before running): Richardson in 1/R.

P-D measured an exact 1/R approach (halving ratios 2.000) with a θ-dependent
constant, so the registered absolute bar at 80 λ₀ was the wrong instrument.
The right one: extrapolate the numerical field in 1/R over the rung pair
(R, 2R) — 2·E(2R) − E(R) — and compare THAT with the closed form. Prediction:
rel < 1e-4 on the (40, 80) λ₀ pair for θ ≤ 70°, both kinds; the three-rung
second-order extrapolation is better still.
"""

from __future__ import annotations

import json
import math

import numpy as np

import proto_far as p


def main():
    eps_t, k_p, k_m, w = p.medium(7e6, *p.SOILS["A"])
    lam0 = 2.0 * math.pi / k_p
    c1 = -1j * w * p.MU0 / (4.0 * math.pi)
    rows, worst1, worst2 = [], 0.0, 0.0
    for kind, d in (("HED", 0.15), ("VED", 0.6)):
        phi = 0.6 if kind == "HED" else 0.0
        mid = np.array([[0.0, 0.0, -d]])
        mom = (
            np.array([[1.0, 0.0, 0.0]])
            if kind == "HED"
            else np.array([[0.0, 0.0, 1.0]])
        )
        for th_deg in (10.0, 30.0, 50.0, 70.0, 80.0):
            th = math.radians(th_deg)
            m_th, m_ph = p.transmitted_moments(
                mid, mom, k_p, k_m, np.array([th]), np.array([phi]), 0.0, form="fresnel"
            )
            closed = np.array([c1 * m_th[0, 0], c1 * m_ph[0, 0]])
            vals = {}
            for n in (20, 40, 80):
                e_th, e_ph, _ = p.numeric_far(
                    k_p, k_m, eps_t, n * lam0, th, phi, d, kind, w
                )
                vals[n] = np.array([e_th, e_ph])
            r1 = 2.0 * vals[80] - vals[40]
            # second order on 1/R with rungs 1:2:4 → (8·E80 − 6·E40 + E20)/3
            r2 = (8.0 * vals[80] - 6.0 * vals[40] + vals[20]) / 3.0
            rel1 = float(np.linalg.norm(r1 - closed) / np.linalg.norm(closed))
            rel2 = float(np.linalg.norm(r2 - closed) / np.linalg.norm(closed))
            raw = float(np.linalg.norm(vals[80] - closed) / np.linalg.norm(closed))
            rows.append(
                {
                    "kind": kind,
                    "theta": th_deg,
                    "raw80": raw,
                    "rich1": rel1,
                    "rich2": rel2,
                }
            )
            print(
                f"  {kind} θ={th_deg:4.0f}  raw@80λ₀ {raw:.2e}  Richardson-1 {rel1:.2e}  Richardson-2 {rel2:.2e}"
            )
            if th_deg <= 70.0:
                worst1, worst2 = max(worst1, rel1), max(worst2, rel2)
    ok = worst1 < 1e-4
    print(
        f"[{'HIT ' if ok else 'MISS'}] P-D′: worst Richardson-1 (θ≤70°) {worst1:.2e}, Richardson-2 {worst2:.2e}"
    )
    p.RESULTS["P-D'"] = {"ok": ok, "worst_rich1": worst1, "worst_rich2": worst2}
    (p.HERE / "p_d_richardson.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
