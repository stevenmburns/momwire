"""Every region of a below/below grid, filled, for three soils x both
spacings; saved for a main-vs-branch per-region comparison.

    python fullgrid.py OUT.npz
"""

import math
import os
import sys

import numpy as np

os.environ["MOMWIRE_SOMM_CACHE"] = "0"
os.environ["MOMWIRE_BELOW_ANCHORS"] = "off"

from momwire import _sommerfeld as sm  # noqa: E402
from momwire import _sommerfeld_below as sb  # noqa: E402

EPS0 = 8.8541878128e-12
CASES = {
    "dan6": (6.0078e6, complex(13.0, -14.96)),
    "salt1.8": (1.8e6, complex(81.0, -5.0 / (2 * math.pi * 1.8e6 * EPS0))),
    "fresh28": (28e6, complex(80.0, -0.001 / (2 * math.pi * 28e6 * EPS0))),
}
out = {}
for name, (f, et) in CASES.items():
    om = 2 * math.pi * f
    k2 = om / sm._C_LIGHT
    for sp_name, sp in (
        ("point", sb._SOMM_BELOW_SPACING_POINT),
        ("fill", sb._SOMM_BELOW_SPACING_FILL),
    ):
        sm._GRID_CACHE.clear()
        g = sb.get_grid_below(et, k2, 3.9 * sb._lambda_m(et, k2), om, spacing=sp)
        g._ensure_for(g.r1_max, g.th_min, 0.5 * math.pi)
        for i, reg in enumerate(g._regions):
            if reg["filled"]:
                out[f"{name}|{sp_name}|{i}"] = np.asarray(reg["vals"])
        print(name, sp_name, flush=True)
np.savez(sys.argv[1], **out)
