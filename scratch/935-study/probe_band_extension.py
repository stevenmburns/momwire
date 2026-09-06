"""momwire#935 part 2: extending the theta band BELOW 0.1 deg.

Reuses momwire#838's probe machinery deliberately rather than writing a new
scorer, because that probe already answered the design question for the band
above 0.1 deg and its methodology note is the one that matters here:

    scoring each lattice at its OWN cell midpoints compares different points
    and flatters the log band

so both candidates are scored against a COMMON set of off-node queries.

#935 proposes replacing the bottom rows with LOG-spaced ones. The comment at
`_SOMM_BELOW_DTH_BAND_DEG` records that exact comparison over [0.1, 1.0] and
uniform won by three to four orders at equal node count. This asks whether that
still holds in the NEW range, where the lateral wave's logarithm is steeper.

Cost currency is the same as #553 U2: ~6.4/tan(theta) tail panels per node, so
a node at 0.05 deg costs ~2x a node at 0.1 deg.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "scratch")
from probe838_grazing_band import (  # noqa: E402
    BAR,
    OM7,
    R1LS,
    SOILS,
    interp,
    lattice,
    panels,
    surf_at,
)

from momwire import _ground_refl  # noqa: E402
from momwire import _sommerfeld_below as below  # noqa: E402

# The direct evaluator REFUSES below ~0.058 deg on the shipped tail budget:
# _MAX_TAIL_PANELS = 4000, of which the worst SPEC soil already uses ~97 % at
# the 0.1 deg floor. So the reference cannot even be BUILT below the floor
# without lifting it — which is the same thing the constant's own comment did
# to measure its truncation ladder (it used 40000). Lifted here for the study
# only; production would have to move it, and that is part of the proposal
# rather than a detail.
below._MAX_TAIL_PANELS = 400000

EPS0 = 8.8541878128e-12
K7 = 2.0 * np.pi * 7e6 / 299792458.0


def score(lo, hi, kind, n, q):
    th_nodes, c_nodes = lattice(kind, n, lo=lo, hi=hi)
    worst = 0.0
    for ground in SOILS.values():
        eps_t = _ground_refl.eps_tilde(ground, OM7, EPS0)
        lam_m = below.lambda_medium(eps_t, K7)
        for r1l in R1LS:
            r1 = r1l * lam_m
            ref = surf_at(eps_t, r1, q)
            got = interp(
                c_nodes, surf_at(eps_t, r1, th_nodes), np.log(q) if kind == "log" else q
            )
            scale = np.abs(ref).max(axis=0)
            worst = max(worst, float((np.abs(got - ref) / scale[None, :]).max()))
    return worst, panels(th_nodes)


for lo, hi in ((0.05, 0.1), (0.03, 0.1)):
    # common off-node queries, interior enough for a full 4-node stencil
    q = np.exp(np.linspace(np.log(lo * 1.15), np.log(hi * 0.87), 13))
    print(
        f"\n=== band [{lo}, {hi}] deg   bar {BAR:.1e}   "
        f"{len(q)} common queries in [{q[0]:.4f}, {q[-1]:.4f}] ==="
    )
    print(
        f"{'kind':>8} {'nodes':>6} {'worst rel':>11} {'vs bar':>8} {'panels':>10} {'fill s':>8}"
    )
    for kind in ("uniform", "log"):
        for n in (4, 8, 16):
            t0 = time.perf_counter()
            worst, pan = score(lo, hi, kind, n, q)
            dt = time.perf_counter() - t0
            verdict = "PASS" if worst < BAR else "FAIL"
            print(f"{kind:>8} {n:6d} {worst:11.3e} {verdict:>8} {pan:10.0f} {dt:8.2f}")
