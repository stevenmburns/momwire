"""momwire#1156 probe 15: probe3's detached_hub "non-convergence" read with
ONE feed at a time. probe3 drove both of its ports at once; if the gap there
is SG's inverted mixed-deck coupling (probe7), a single port must agree."""

import sys

sys.path.insert(0, "tests")
sys.path.insert(0, "scratch/1156-sg-buried-loading")

from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from probe3_convergence import LOADS, per_wire, z  # noqa: E402
from test_razor_detached_1149 import detached_hub  # noqa: E402

for feed in ((3, 4.0, 1 + 0j), (0, 1.0, 1 + 0j)):
    for m in (1, 2):
        d0 = dict(detached_hub(m), wire_radius=1e-3, feeds=[feed])
        zb = {cls: z(cls, d0) for cls in (SinusoidalGalerkinSolver, BSplineSolver)}
        row = f"feed wire {feed[0]} m={m} bare SG {zb[SinusoidalGalerkinSolver]:.3f} bs {zb[BSplineSolver]:.3f}"
        for lname, kw in LOADS.items():
            sh = {
                cls: z(cls, dict(d0, **per_wire(d0, kw))) - zb[cls]
                for cls in (SinusoidalGalerkinSolver, BSplineSolver)
            }
            row += f" | {lname} gap {abs(sh[SinusoidalGalerkinSolver] - sh[BSplineSolver]):.3f}"
        print(row, flush=True)
