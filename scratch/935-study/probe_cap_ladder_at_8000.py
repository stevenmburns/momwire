"""momwire#935: re-measure the tail-cap ladder at `_MAX_TAIL_PANELS = 8000`.

The #841 gates pin a SERVED/REFUSED ladder that was measured at 4000. Raising
the cap moves it, and the gates have to be re-pinned to what is now true
rather than edited until they pass. Two questions:

  1. which theta rungs are served on soil A / 7 MHz / R1 = lambda_m (the deck
     the existing ladder gate uses), and
  2. what the WORST panel count over the whole SPEC matrix is at the new
     floor -- because the floor is chosen from the worst soil, not this one.
     Soil A at 7 MHz alone understated the old floor by 7 points (90 % vs the
     true 96.7 %), which is exactly the mistake this half is here to avoid.
"""

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below

C0 = 299792458.0
EPS0 = 8.8541878128e-12
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}


def deck(soil, f):
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    eps_t = _ground_refl.eps_tilde(SOILS[soil], om, EPS0)
    return eps_t, k2, om, below.lambda_medium(eps_t, k2)


def at(soil, f, th_deg, r1_over_lam=1.0):
    eps_t, k2, om, lam_m = deck(soil, f)
    h = below.Health()
    try:
        below.iv_surfaces_direct_below(
            eps_t,
            k2,
            np.array([r1_over_lam * lam_m]),
            np.radians([th_deg]),
            rtol=1e-9,
            omega=om,
            health=h,
        )
    except ValueError:
        return None, h
    return True, h


print(f"_MAX_TAIL_PANELS = {below._MAX_TAIL_PANELS}")
print(f"_SOMM_BELOW_TH_MIN_DEG = {below._SOMM_BELOW_TH_MIN_DEG}")

print("\n(1) soil A / 7 MHz / R1 = lambda_m -- the ladder gate's own deck")
for th in (0.12, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03, 0.023):
    ok, h = at("A", 7e6, th)
    panels = h.max_tail_panels
    print(f"  {th:6.3f} deg  {'SERVED ' if ok else 'REFUSED'}  panels={panels}")

print("\n(2) worst over the SPEC matrix at each candidate floor")
for th in (0.10, 0.06, 0.05, 0.04, 0.023):
    worst, where, refused = 0, None, []
    for soil in SOILS:
        for f in (7e6, 21e6):
            for r1l in (0.0, 0.02, 0.05, 0.2, 1.0, 2.0):
                ok, h = at(soil, f, th, r1l)
                p = h.max_tail_panels
                if p > worst:
                    worst, where = p, (soil, f / 1e6, r1l)
                if not ok:
                    refused.append((soil, f / 1e6, r1l))
    pct = 100.0 * worst / below._MAX_TAIL_PANELS
    tag = f"  {len(refused)} REFUSED" if refused else "  all converged"
    print(
        f"  {th:6.3f} deg  worst {worst}/{below._MAX_TAIL_PANELS} = {pct:.1f}%  at {where}{tag}"
    )
