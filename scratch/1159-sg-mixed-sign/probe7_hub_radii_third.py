"""momwire#1159 probe 7: detached_hub with HUB_RADII (three radii meeting at
the buried hub) leaves a FLAT SG-bspline Y12 floor after the fix, where the
same deck at one radius converges. Razor as a third opinion on Y12, per m:
whose number moves away from the other two?"""

import sys
import warnings

import numpy as np

sys.path.insert(0, "tests")
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_razor_detached_1149 import detached_hub  # noqa: E402

for m in [int(x) for x in sys.argv[1:]] or [1, 2, 4]:
    d = detached_hub(m)
    y = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cls, kw in (
            (SinusoidalGalerkinSolver, {}),
            (BSplineSolver, {}),
            (RazorSolver, {"nec5_quadrature": True}),
        ):
            y[cls.__name__[:5]] = np.asarray(cls(**d, **kw).compute_y_matrix())[0, 1]
    rel = {
        f"{a}-{b}": abs(y[a] - y[b]) / abs(y[b])
        for a, b in (("Sinus", "BSpli"), ("Razor", "BSpli"), ("Sinus", "Razor"))
    }
    print(f"m={m}", {k: f"{v:.3e}" for k, v in rel.items()}, flush=True)
