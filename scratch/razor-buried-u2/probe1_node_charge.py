"""Probe 1 (momwire#1149 U2): is the crossing defect the node point charge
that the same-medium Sommerfeld REMAINDER carries and nothing else does?

Hypothesis H1. Razor's crossing assembly splits the crossing tent into two
half tents, one per medium. Each half's charge is spelled as its wing's LINE
charge only (q = dLambda/dtau) in the direct and image terms of its family,
and in both trunk cross blocks (Fd = 0 on a path axis kills SQ, the corner
is off). But the family's Sommerfeld remainder Q is the field of the half
tent's CURRENT, integrated through unit-moment fields, and a current that
stops at the node implies a point charge there. So Q alone carries the node
charge. The two families' node charges are opposite and would cancel only in
one convention; mixed, they leave the remainder potential of a unit node
charge on every row.

The correction applied here is the COMPLETE convention (bspline's): give each
family's direct+image its node charge, and give each cross block the other
half's node charge through the transmitted V. Per row m (T2 endpoints e with
razor's signs), per crossing column n:

  above rows: dZ = sum_e sign_e * ( -Qa * fam_a(e) - Qb * c1 V(e; z'=0) )
  below rows: dZ = sum_e sign_e * ( -Qb * fam_b(e) - Qa * c1 V(z=0; e) )

fam_a = (1 - C2) g_k(R) / (j w eps0),  fam_b = (1 - A_m) g_km(R) / (j w eps_m),
R = sqrt(|e - node|^2 + a^2), Qa/Qb the half tents' node proxies (Qa = -Qb).
At eps~ = 1, c1 V == fam and the correction is identically zero.

Usage: probe1_node_charge.py [on|off] [rungs...]
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import numpy as np
from test_crossing_serve_524 import crossing_deck

import momwire
from momwire import _medium_spec, _near_interface
from momwire import razor as _razor
from momwire._sommerfeld_below import _c1_moment
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_CROSSING = True


def node_charge_dZ(s, geom, k, omega):
    tents = s._crossing_tents(geom)
    n = geom["n_basis_total"]
    dZ = np.zeros((n, n), dtype=np.complex128)
    if not tents:
        return dZ
    knots = s._knot_points(geom)
    ctx = s._crossing_context(geom, k=k, omega=omega)
    eps_t, eps_m, k_p, k_m, c2, a_m = ctx.medium
    c1 = _c1_moment(omega, s.mu)
    a = float(ctx.a_wire)
    gz = float(ctx.ground_z)
    wr, wg = geom["wing_rise"], geom["wing_sigma"]
    axes = {
        side: s._crossing_path_axis(geom, tents, side)
        for side in (_medium_spec.ABOVE, _medium_spec.BELOW)
    }
    for col, jb in tents:
        ja = 1 - jb
        # the half tents' node proxies: minus the live wing's integrated q
        Qa = -float(wg[col, ja]) * (1.0 if wr[col, ja] else -1.0)
        Qb = -float(wg[col, jb]) * (1.0 if wr[col, jb] else -1.0)
        node = knots[col]
        for side, ax in axes.items():
            for pt, sign, e in ax["ends"]:
                m = int(np.flatnonzero(e)[0])
                d = pt - node
                R = float(np.sqrt(d @ d + a * a))
                rho_eff = float(np.sqrt(d[0] ** 2 + d[1] ** 2 + a * a))
                if side == _medium_spec.ABOVE:
                    fam = (1 - c2) * np.exp(-1j * k_p * R) / (4 * np.pi * R)
                    fam /= 1j * omega * s.eps
                    zt = max(float(pt[2] - gz), 0.0)
                    V = _near_interface.six_point(eps_t, k_p, rho_eff, zt, 0.0)[1]
                    dZ[m, col] += sign * (-Qa * fam - Qb * c1 * V)
                else:
                    fam = (1 - a_m) * np.exp(-1j * k_m * R) / (4 * np.pi * R)
                    fam /= 1j * omega * eps_m
                    zt = min(float(pt[2] - gz), 0.0)
                    V = _near_interface.six_point(eps_t, k_p, rho_eff, 0.0, zt)[1]
                    dZ[m, col] += sign * (-Qb * fam - Qa * c1 * V)
    return dZ


_orig = RazorSolver._assemble_Z_crossing
MODE = sys.argv[1] if len(sys.argv) > 1 else "on"


def patched(self, geom, k, omega, *, detached=False):
    Z = _orig(self, geom, k, omega, detached=detached)
    if MODE == "on" and not detached:
        Z = Z + node_charge_dZ(self, geom, k, omega)
    return Z


RazorSolver._assemble_Z_crossing = patched


def deck(m, medium):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if medium == "eps1":
        d["ground_eps"] = (1.0, 0.0)
    elif medium == "diel":
        d["ground_eps"] = (13.0, 0.0)
    elif medium == "diel80":
        d["ground_eps"] = (80.0, 0.0)
    return d


if __name__ == "__main__":
    rungs = [int(x) for x in sys.argv[2:]] or [1, 2, 4]
    for medium in ("eps1", "soilA", "diel", "diel80"):
        prev = None
        for m in rungs:
            out = {}
            for name, cls in (("razor", RazorSolver), ("bspline", BSplineSolver)):
                d = deck(m, medium)
                kw = {"nec5_quadrature": True} if cls is RazorSolver else {}
                if cls is RazorSolver:
                    d.pop("junctions")
                t0 = time.perf_counter()
                Y = np.asarray(cls(**d, **kw).compute_y_matrix())
                out[name] = (Y, time.perf_counter() - t0)
            Y, dt = out["razor"]
            Zoc = np.linalg.inv(Y)
            Zb = np.linalg.inv(out["bspline"][0])
            nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
            ratio = "" if prev is None else f" ({prev / nr:5.2f}x)"
            prev = nr
            print(
                f"{MODE:3s} {medium:6s} x{m:<2d} nonrec {nr:.3e}{ratio}  "
                f"Z11 {Zoc[0, 0]:.3f} (bs {Zb[0, 0]:.3f})  "
                f"Z22 {Zoc[1, 1]:.3f} (bs {Zb[1, 1]:.3f})  {dt:.2f}s",
                flush=True,
            )
