"""momwire#1156 probe 12: where does probe11's one-ulp jump live? The
loaded crossing deck's G at lambda and at nextafter(lambda), split into the
fill (loading off) and the loading term (on - off)."""

import math
import sys

import numpy as np

sys.path.insert(0, "tests")

from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_sg_buried_loading_1156 import SIG_JACKET, assembled, per_wire, xing  # noqa: E402

d = per_wire(xing(), **SIG_JACKET)
lam = d["wavelength"]
out = {}
for tag, lw in (("lam", lam), ("lam+", np.nextafter(lam, math.inf))):
    s = SinusoidalGalerkinSolver(**dict(d, wavelength=lw))
    on, sv, geom, k = assembled(s, True)
    off, *_ = assembled(s, False)
    out[tag] = (on, off, s)
    print(
        tag,
        "k",
        repr(s.k),
        "omega",
        repr(s.omega),
        "cond(G_on)",
        f"{np.linalg.cond(on):.3e}",
        "cond(G_off)",
        f"{np.linalg.cond(off):.3e}",
    )
on0, off0, _ = out["lam"]
on1, off1, _ = out["lam+"]
dF = off1 - off0
dL = (on1 - off1) - (on0 - off0)
print(
    f"fill move    max {np.abs(dF).max():.3e} at {np.unravel_index(np.argmax(np.abs(dF)), dF.shape)}"
    f"  (|G| {np.abs(off0).max():.3e})"
)
print(f"loading move max {np.abs(dL).max():.3e}  (|L| {np.abs(on0 - off0).max():.3e})")
idx = np.argwhere(np.abs(dF) > 1e-9 * np.abs(off0).max())
print("fill entries moving > 1e-9 |G|:", idx.tolist()[:20])
