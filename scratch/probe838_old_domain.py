"""momwire#838 part 1: capture / check `SommerfeldGridBelow.eval` in the OLD
domain (theta >= 1 deg), the bit-identity record for the sub-1 deg band.

The deal for part 1 was that extending the grid's domain downward must not
move a single cell that was already served. That holds by construction --
the two old theta bands keep their exact node sets and spacings and the new
band is a separate region -- but "by construction" is an argument, so this
takes the measurement.

Run it on the commit BEFORE the change to bank a baseline, and after to
compare:

    git stash && python scratch/probe838_old_domain.py before.npz
    git stash pop && python scratch/probe838_old_domain.py after.npz --against before.npz

The committed gate is `test_the_old_domain_is_unmoved`, which replays the
pre-band routing over the SAME grid's tables in process rather than
comparing against banked values -- a golden captured on one box would be
asserting bit-equality of the contour fill across toolchains, which is not
portable (macOS reads ~1.7e-12 from Linux here) and is not this unit's
claim. This script is the cross-COMMIT check that motivated it.

Sampled OFF-node deliberately: a sweep that only landed on lattice nodes
would pass even if the interpolation stencil changed underneath it. The
theta axis is dense in [1, 2] deg because the seam's own first cell is the
place a routing mistake shows up first -- that is exactly where dtheta =
0.25 was caught moving the answer by 1.2e-07.
"""

import sys

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below
from momwire._sommerfeld import _SURF_KEYS

C0 = 299792458.0
EPS0 = 8.8541878128e-12
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}
FREQS = (7e6, 21e6)

# The seam's first cell, the body, and both sides of the 30 deg split.
THETA_DEG = np.concatenate(
    [
        np.linspace(1.0, 1.999, 40),
        np.linspace(2.0, 29.9, 120),
        np.linspace(30.0, 89.9, 90),
    ]
)
R1_FRAC = np.linspace(0.02, 0.98, 37)


def capture():
    rec = {}
    for soil, ground in SOILS.items():
        for f in FREQS:
            k2 = 2.0 * np.pi * f / C0
            om = 2.0 * np.pi * f
            eps_t = _ground_refl.eps_tilde(ground, om, EPS0)
            lam_m = below.lambda_medium(eps_t, k2)
            r1_max = below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
            g = below.SommerfeldGridBelow(eps_t, k2, r1_max, omega=om)
            rr, tt = np.meshgrid(R1_FRAC * r1_max, np.radians(THETA_DEG), indexing="ij")
            out = g.eval(rr, tt)
            for k in _SURF_KEYS:
                rec[f"{soil}/{f:.0e}/{k}"] = out[k]
    return rec


def main(argv):
    rec = capture()
    np.savez(argv[0], **rec)
    print(f"{len(rec)} surfaces -> {argv[0]}")
    if len(argv) > 2 and argv[1] == "--against":
        ref = np.load(argv[2])
        same = moved = 0
        worst = 0.0
        for k in sorted(rec):
            if np.array_equal(rec[k], ref[k]):
                same += 1
            else:
                moved += 1
                worst = max(
                    worst,
                    float((np.abs(rec[k] - ref[k]) / np.abs(ref[k]).max()).max()),
                )
        print(f"bit-identical {same}   moved {moved}")
        if moved:
            print(f"worst relative move {worst:.3e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
