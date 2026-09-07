"""momwire#936 THE DISCRIMINATOR: add -(t_perp . rho_hat) dW/drho and look.

Laptop-control's rule: if adding the dropped term moves the 45 deg answer
toward the lean-0 residual by an amount tracking the 0.70 pp / 0.25 ohm drift,
the term is real (option 1). If it does not move, or moves away, the guard is
over-strict (option 0).

WHERE IT GOES. The by-parts boundary content lives in `_ends_and_corner`'s
source-end (SW) loop, which contracts the TEST axis against W carrying the
test tangent's z-component alone:

    _real_matvec_c(A["F"], wA_tz * te["W"])          wA_tz = w_A * tz_A

That `tz_A` is the substitution: a general tangent contributes its transverse
part too, through the transverse gradient of W, which is
-(t_perp . rho_hat) dW/drho. This patches exactly that one contraction and
leaves everything else alone, so whatever moves is attributable.

dW/drho is taken by CENTRED FINITE DIFFERENCE on the shipped table -- no new
kernel, which is the point: this is a diagnostic, not the build.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe9_op_lean_control as L  # noqa: E402

from momwire import _crossing_fill as CF  # noqa: E402

_FD = 1e-5


def patched_ends(ctx, A, B, eps_t, k_p, c1, gz, *, corner=True, memo=None):
    """`_ends_and_corner` with the transverse W contribution restored."""
    base = _ORIG(ctx, A, B, eps_t, k_p, c1, gz, corner=corner, memo=memo)
    txA, tyA, _tzA = A["t"].T
    wA = A["w"]
    extra = np.zeros_like(base)
    for pt, sign, fv in B["ends"]:
        ex = A["nodes"][:, 0] - pt[0]
        ey = A["nodes"][:, 1] - pt[1]
        rho_e = np.hypot(ex, ey)
        safe = np.maximum(rho_e, 1e-12)
        # d/drho of W by centred difference on the shipped table
        zA = A["nodes"][:, 2] - gz
        zB = np.full_like(rho_e, pt[2] - gz)
        wp = CF._tables(ctx, eps_t, k_p, rho_e + _FD, zA, zB, CF._CROSS_RTOL)["W"]
        wm = CF._tables(
            ctx, eps_t, k_p, np.maximum(rho_e - _FD, 1e-12), zA, zB, CF._CROSS_RTOL
        )["W"]
        dWdrho = (wp - wm) / (2 * _FD)
        # -(t_perp . rho_hat), rho_hat pointing from the end toward the node
        proj = -(txA * ex + tyA * ey) / safe
        col = CF._real_matvec_c(A["F"], wA * proj * dWdrho)
        nz = np.flatnonzero(fv)
        extra[:, nz] += np.outer(col, fv[nz]) * (-c1 * sign)
    return base + extra


_ORIG = CF._ends_and_corner
orig_tol = CF._TILT_TOL
CF._TILT_TOL = 1e9
try:
    print(
        f"{'lean':>6s} {'as shipped':>22s} {'+ correction':>22s} "
        f"{'NEC-5':>22s}   {'dR shipped':>11s} {'dR corrected':>13s}"
    )
    for lean in (0.0, 15.0, 30.0, 45.0):
        zn = L.nec5(lean)
        CF._ends_and_corner = _ORIG
        z0 = L.mw(lean)
        CF._ends_and_corner = patched_ends
        z1 = L.mw(lean)
        CF._ends_and_corner = _ORIG
        d0 = 100 * (z0.real - zn.real) / zn.real
        d1 = 100 * (z1.real - zn.real) / zn.real
        print(
            f"{lean:6.1f} {z0.real:10.3f}{z0.imag:+10.3f}j "
            f"{z1.real:10.3f}{z1.imag:+10.3f}j {zn.real:10.3f}{zn.imag:+10.3f}j"
            f"   {d0:+10.2f}% {d1:+12.2f}%"
        )
finally:
    CF._ends_and_corner = _ORIG
    CF._TILT_TOL = orig_tol
