"""#813 step 3, probe 4: is the eps~ = 1 residual the trunk's QUADRATURE?

The decisive question. The assembly's residual concentrates on the junction
tent's row and diagonal at ~2.6e-5, which is the class momwire#813's half 1
already recorded for that row (5.3e-5, "quadrature: 4-point source Gauss vs
razor's 12 + statics"). If it IS quadrature it falls when the trunk's source
order rises and the bar is a quadrature bar; if it does not fall, the
formulation is wrong somewhere and no bar is honest.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill as CF  # noqa: E402
from probe1_blocks import setup  # noqa: E402
from probe3_residual import build  # noqa: E402


def main():
    f = setup(False)
    Z, jn = f["Z"], f["jn"]
    ctx = f["rs"]._crossing_context(f["geom"], ground_eps=(1.0, 0.0))
    near0, far0 = CF._NEAR_Q, CF._FAR_Q
    print(f"{'q':>4}  {'whole':>10}  {'(jn,jn)':>10}  {'tent row':>10}  {'rest':>10}")
    try:
        for q in (4, 6, 8, 12, 16, 24):
            CF._NEAR_Q = CF._FAR_Q = q
            M = build(f, ctx)
            d = np.abs(M - Z) / np.abs(Z).max()
            r = d[jn, :].copy()
            r[jn] = 0
            rest = d.copy()
            rest[jn, :] = 0
            rest[:, jn] = 0
            print(
                f"{q:>4}  {d.max():>10.3e}  {d[jn, jn]:>10.3e}  "
                f"{r.max():>10.3e}  {rest.max():>10.3e}"
            )
    finally:
        CF._NEAR_Q, CF._FAR_Q = near0, far0


if __name__ == "__main__":
    main()
