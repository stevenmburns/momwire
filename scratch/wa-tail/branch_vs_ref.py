"""The production rule (this tree) against spot.jsonl / spot2.jsonl's mpmath
references: componentwise and vec-norm, both machines.

    python branch_vs_ref.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpref as R  # noqa: E402

from momwire import _sommerfeld_below as SB  # noqa: E402

HERE = Path(__file__).resolve().parent
rows = []
for f in ("spot.jsonl", "spot2.jsonl", "spot3.jsonl"):
    p = HERE / f
    if p.exists():
        rows += [json.loads(x) for x in p.open()]


def rel(v, ref):
    comp = float(np.max(np.abs(v - ref) / np.abs(ref)))
    vec = float(np.max(np.abs(v - ref)) / np.max(np.abs(ref)))
    return comp, vec


out = []
for r in rows:
    kp, km, rho, h = R.node(r["soil"], r["r1_wl"], r["theta_deg"])
    et = R.soil(r["soil"])[1]
    ref = np.array([complex(a, b) for a, b in r["ref"]])
    res = dict(soil=r["soil"], r1=r["r1_wl"], th=r["theta_deg"], tag=r["tag"])
    for name, fn in (("prod_cpp", False), ("prod_np", True)):
        if fn is False and not SB._HAVE_BELOW_FILLS_ACCEL:
            continue
        SB._FORCE_NUMPY = fn
        hl = SB.Health()
        try:
            v = SB._six_integrals_below(et, kp, rho, h, health=hl)
        except ValueError:
            # under the floor the engine refuses only a capped walk; a WA
            # node is served -- this is the numpy contour directly then
            def f6(lam, rho=rho, h=h, kp=kp, km=km):
                return SB._integrand_six_below(lam, rho, h, kp, km)

            v, _, tp, _, _ = SB._run_contour(
                f6,
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
        res[name] = rel(v, ref) + (hl.max_tail_panels,)
    SB._FORCE_NUMPY = False
    res["main_cpp"] = r.get("cpp")
    out.append(res)
    print(
        f"{r['soil']:8s} {r['r1_wl']:<5} {r['theta_deg']:<8.4g} "
        + " ".join(
            f"{k}={res[k][0]:.1e}/{res[k][1]:.1e}"
            for k in ("main_cpp", "prod_cpp", "prod_np")
            if res.get(k)
        )
    )
(HERE / "branch_vs_ref.json").write_text(json.dumps(out, indent=1))
worst = max(o["prod_cpp"][0] for o in out)
print("worst prod componentwise", worst, "vec", max(o["prod_cpp"][1] for o in out))
