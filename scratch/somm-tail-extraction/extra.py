"""Follow-up experiments: the b scan, nodes under the current floor (to h = 0),
and a dense robustness sweep of WA against the tight reference.

    python extra.py
"""

from __future__ import annotations

import json
import math

import numpy as np

import methods as Mt
from study import rel

REF_PANELS = 400_000


def b_scan(fh):
    for s in ("dan", "salt1.8", "fresh28"):
        for r1, th in ((1.0, 0.1), (2.0, 0.05), (0.2, 1.0)):
            kp, km, rho, h = Mt.node(s, r1, th)
            ref = Mt.run(kp, km, rho, h, "base", rtol=1e-12, max_panels=REF_PANELS)
            for bf in (0.5, 1.0, 2.0, 4.0):
                for m in ("ext3", "ext3+wa"):
                    r = Mt.run(kp, km, rho, h, m, N=3, b_fac=bf)
                    fh.write(
                        json.dumps(
                            dict(
                                exp="b_scan",
                                soil=s,
                                r1_wl=r1,
                                theta_deg=th,
                                b_fac=bf,
                                method=m,
                                tail_panels=r["tail_panels"],
                                points=r["points"],
                                vs_ref=rel(r["value"], ref["value"]),
                            )
                        )
                        + "\n"
                    )
                    print(
                        "b",
                        s,
                        r1,
                        th,
                        bf,
                        m,
                        r["tail_panels"],
                        rel(r["value"], ref["value"]),
                    )


def under_floor(fh):
    # The current floor is 0.016667 deg; the engine refuses below it. The
    # reference here is ext3 on the plain quiet tail at rtol 1e-12 with a
    # 400k budget, which reaches h = 0 because its residual decays
    # algebraically (lambda^-3.5).
    for s in ("dan", "salt1.8", "fresh28"):
        for r1 in (0.2, 1.0, 2.0):
            for th in (0.01, 0.005, 0.001, 0.0):
                kp, km, rho, h = Mt.node(s, r1, th)
                ref = Mt.run(
                    kp, km, rho, h, "ext3", N=3, rtol=1e-12, max_panels=REF_PANELS
                )
                base = Mt.run(kp, km, rho, h, "base")
                row = dict(
                    exp="under_floor",
                    soil=s,
                    r1_wl=r1,
                    theta_deg=th,
                    ref_panels=ref["tail_panels"],
                    ref_conv=ref["converged"],
                    base_panels=base["tail_panels"],
                    base_conv=base["converged"],
                    base_vs_ref=rel(base["value"], ref["value"]),
                )
                for m, N in (("wa", 0), ("ext1+wa", 1), ("ext3+wa", 3)):
                    r = Mt.run(kp, km, rho, h, m, N=N)
                    row[m] = dict(
                        tail_panels=r["tail_panels"],
                        points=r["points"],
                        converged=r["converged"],
                        vs_ref=rel(r["value"], ref["value"]),
                    )
                fh.write(json.dumps(row) + "\n")
                print(
                    "floor",
                    s,
                    r1,
                    th,
                    row["ref_panels"],
                    row["base_panels"],
                    row["base_conv"],
                    {
                        m: (row[m]["tail_panels"], f"{row[m]['vs_ref'][0]:.0e}")
                        for m in ("wa", "ext1+wa", "ext3+wa")
                    },
                    flush=True,
                )


def sweep(fh):
    thetas = np.geomspace(0.03, 60.0, 24)
    for s in ("dan", "salt1.8", "fresh28"):
        for r1 in (0.02, 0.1, 0.5, 1.5, 3.0):
            for th in thetas:
                kp, km, rho, h = Mt.node(s, r1, float(th))
                ref = Mt.run(kp, km, rho, h, "base", rtol=1e-12, max_panels=REF_PANELS)
                base = Mt.run(kp, km, rho, h, "base")
                wa = Mt.run(kp, km, rho, h, "wa")
                fh.write(
                    json.dumps(
                        dict(
                            exp="sweep",
                            soil=s,
                            r1_wl=r1,
                            theta_deg=float(th),
                            km_h=abs(km) * h,
                            base_panels=base["tail_panels"],
                            base_points=base["points"],
                            base_vs_ref=rel(base["value"], ref["value"]),
                            wa_panels=wa["tail_panels"],
                            wa_points=wa["points"],
                            wa_vs_ref=rel(wa["value"], ref["value"]),
                            wa_accel=wa["accel"],
                        )
                    )
                    + "\n"
                )
            fh.flush()
            print("sweep", s, r1, flush=True)


if __name__ == "__main__":
    with open("extra.jsonl", "w") as fh:
        b_scan(fh)
        under_floor(fh)
        sweep(fh)
    print("done", math.pi)
