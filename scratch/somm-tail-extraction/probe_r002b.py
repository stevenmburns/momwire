"""Mutual agreement of the short-walk routes at the R1 = 0.02 corner: WA alone,
ext3 on the plain quiet tail (no WA), ext1+wa, ext3+wa. Three of these share
no tail machinery, so their agreement referees the long-walk references.

    python probe_r002b.py
"""

import itertools

import methods as Mt
from study import rel

NODES = (
    ("dan", 0.03),
    ("salt1.8", 0.03),
    ("salt1.8", 0.0809),
    ("dan", 0.157),
    ("fresh28", 0.03),
)
for s, th in NODES:
    kp, km, rho, h = Mt.node(s, 0.02, th)
    runs = {
        "wa": Mt.run(kp, km, rho, h, "wa"),
        "ext3": Mt.run(kp, km, rho, h, "ext3", N=3),
        "ext1+wa": Mt.run(kp, km, rho, h, "ext1+wa", N=1),
        "ext3+wa": Mt.run(kp, km, rho, h, "ext3+wa", N=3),
    }
    worst = max(
        rel(a["value"], b["value"])[0]
        for a, b in itertools.combinations(runs.values(), 2)
    )
    print(
        s,
        th,
        {k: v["tail_panels"] for k, v in runs.items()},
        f"max pairwise comp {worst:.1e}",
        flush=True,
    )
