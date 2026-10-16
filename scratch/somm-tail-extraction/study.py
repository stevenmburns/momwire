"""The measurement grid: every method on every node, against two tight
references. Writes results.jsonl (one line per node x method).

    python study.py [out.jsonl]

Light by design (the laptop rule): one process, numpy only, no grid fill.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import methods as Mt

SOILS = ("dan", "salt1.8", "fresh28")
R1S = (0.2, 1.0, 2.0)
THETAS = (0.05, 0.1, 0.2, 0.5, 1.0, 10.0, 45.0, 80.0)
METHODS = (
    ("base", 0),
    ("wa", 0),
    ("ext1", 1),
    ("ext2", 2),
    ("ext3", 3),
    ("ext1+wa", 1),
    ("ext2+wa", 2),
    ("ext3+wa", 3),
)
REF_PANELS = 400_000


def rel(v, ref):
    v = np.asarray(v)
    ref = np.asarray(ref)
    comp = float(np.max(np.abs(v - ref) / np.maximum(np.abs(ref), 1e-300)))
    vec = float(np.max(np.abs(v - ref)) / max(float(np.max(np.abs(ref))), 1e-300))
    return comp, vec


def cplx(v):
    return [[float(x.real), float(x.imag)] for x in np.asarray(v)]


def main(out):
    with open(out, "w") as fh:
        for s in SOILS:
            for r1 in R1S:
                for th in THETAS:
                    kp, km, rho, h = Mt.node(s, r1, th)
                    ref = Mt.run(
                        kp, km, rho, h, "base", rtol=1e-12, max_panels=REF_PANELS
                    )
                    ref2 = Mt.run(
                        kp, km, rho, h, "ext3", N=3, rtol=1e-12, max_panels=REF_PANELS
                    )
                    rr = rel(ref2["value"], ref["value"])
                    rows = []
                    for m, N in METHODS:
                        r = Mt.run(kp, km, rho, h, m, N=N)
                        rows.append((m, N, r))
                    base = rows[0][2]
                    for m, N, r in rows:
                        line = dict(
                            soil=s,
                            r1_wl=r1,
                            theta_deg=th,
                            rho=rho,
                            h=h,
                            method=m,
                            N=N,
                            tail_panels=r["tail_panels"],
                            head_panels=r["head_panels"],
                            points=r["points"],
                            wall=r["wall"],
                            converged=r["converged"],
                            accel=r["accel"],
                            vs_base=rel(r["value"], base["value"]),
                            vs_ref=rel(r["value"], ref["value"]),
                            vs_ref2=rel(r["value"], ref2["value"]),
                            value=cplx(r["value"]),
                        )
                        fh.write(json.dumps(line) + "\n")
                    fh.write(
                        json.dumps(
                            dict(
                                soil=s,
                                r1_wl=r1,
                                theta_deg=th,
                                method="REF",
                                ref_panels=ref["tail_panels"],
                                ref_conv=ref["converged"],
                                ref2_panels=ref2["tail_panels"],
                                ref2_conv=ref2["converged"],
                                ref_vs_ref2=rr,
                                ref_wall=ref["wall"],
                                value=cplx(ref["value"]),
                            )
                        )
                        + "\n"
                    )
                    fh.flush()
                    print(
                        f"{s:8s} R1={r1:3.1f} th={th:5.2f} ref={ref['tail_panels']:6d} "
                        f"ref~ref2={rr[0]:.1e} base={base['tail_panels']:5d} "
                        + " ".join(
                            f"{m}:{r['tail_panels']}/{rel(r['value'], ref['value'])[0]:.0e}"
                            for m, _, r in rows[1:]
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
