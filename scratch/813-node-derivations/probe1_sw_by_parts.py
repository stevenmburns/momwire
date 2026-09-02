"""momwire#813 derivation (b), by measurement (2026-09-03): is the SW end term
the trunk emits on the junction column from the below wing's in-plane end a
by-parts PARTNER of a current-current term (then razor's chopped rows keep
it), or a source-end charge term like the corner (then they drop it)?

The identity under test, on the below wing of the junction tent (basis jn,
F rising 0 -> 1 toward the node at u = h, tz_B = +-1), for a path-tested row
with vertical tangent tz_A:

    s_w1 + SW  =  c1 (F_A tz_A) [ sum_u w_u W F'_B  -  W(node) F_B(node) ]
               =  -c1 (F_A tz_A) sum_u w_u F_B tz_B (dW/dz')          (by parts)

So if `s_w1 + SW` reproduces the DIRECT form built from the dz'W table with
F (not F') on the source and no end term, SW is the by-parts remnant of the
vertical-current coupling and must stay. If instead `s_w1` alone matches, SW
is extra. At eps_tilde = 1, W = dz'W = 0 and the test is vacuous (that is why
the probe on #651 could not see this); soil A is where it bites.

    python scratch/813-node-derivations/probe1_sw_by_parts.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from test_buried_serve_553 import SOIL_A
from test_crossing_serve_524 import crossing_deck

from momwire import RazorSolver
from momwire import _crossing_fill as CF
from momwire._sommerfeld_below import _c1_moment

warnings.simplefilter("ignore")

deck = crossing_deck(1)
fs = {
    k: v for k, v in deck.items() if k not in ("ground_z", "ground_eps", "ground_model")
}
rs = RazorSolver(**fs, nec5_quadrature=False, n_qp_path=8)
geom = rs._build_geometry()
seg_off = np.asarray(geom["seg_offsets"])
bas_off = np.asarray(geom["basis_offsets"])
b_idx = np.arange(seg_off[0], seg_off[1])
jn = geom["n_basis_total"] - 1
rows_above = np.arange(bas_off[1], bas_off[2])
n_basis = geom["n_basis_total"]

for label, eps in (("eps_tilde = 1", (1.0, 0.0)), ("soil A", SOIL_A)):
    ctx = rs._crossing_context(geom, ground_eps=eps)
    A = CF.path_test_axis(n_basis, rs._path_test_rows(geom, rows_above))
    B = CF.axis_data(ctx, b_idx)
    full = CF.cross_complete_block(ctx, A, B, corner=False)
    B_noend = dict(B)
    B_noend["ends"] = [e for e in B["ends"] if abs(e[0][2]) > 1e-9]
    noend = CF.cross_complete_block(ctx, A, B_noend, corner=False)
    end_terms = full - noend  # SW (+ SQ, identically 0 on razor rows: Fd_A = 0)

    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = ctx.medium
    c1 = _c1_moment(ctx.omega, ctx.mu)
    dx = A["nodes"][:, 0][:, None] - B["nodes"][:, 0][None, :]
    dy = A["nodes"][:, 1][:, None] - B["nodes"][:, 1][None, :]
    rho = np.hypot(dx, dy)
    z = np.broadcast_to(A["nodes"][:, 2][:, None], rho.shape)
    zp = np.broadcast_to(B["nodes"][:, 2][None, :], rho.shape)
    tb = CF._tables(ctx, eps_t, k_p, rho, z, zp, CF._CROSS_RTOL)
    W, dzpW = tb["W"], tb["dzpW"]
    FA_w = A["F"] * A["w"]
    tzA = A["t"][:, 2]
    FB_w = B["F"] * B["w"]
    FdB_w = B["Fd"] * B["w"]
    tzB = B["t"][:, 2]
    s_w1 = c1 * ((FA_w * tzA) @ W @ FdB_w.T)
    direct = -c1 * ((FA_w * tzA) @ dzpW @ (FB_w * tzB).T)  # the by-parts prediction

    col = jn
    got = (s_w1 + end_terms)[rows_above, col]
    alone = s_w1[rows_above, col]
    ref = direct[rows_above, col]
    whole = full[rows_above, col]
    scale = max(np.abs(ref).max(), 1e-300)
    print(
        f"\n== {label}: W max {np.abs(W).max():.3e}, dz'W max {np.abs(dzpW).max():.3e}"
    )
    print(f"   junction column on the {rows_above.size} above razor rows")
    print(
        f"   |end terms| / |whole column|      = {np.abs(end_terms[rows_above, col]).max() / max(np.abs(whole).max(), 1e-300):.3e}"
    )
    print(
        f"   |s_w1 + SW  - direct| / |direct|  = {np.abs(got - ref).max() / scale:.3e}"
    )
    print(
        f"   |s_w1 alone - direct| / |direct|  = {np.abs(alone - ref).max() / scale:.3e}"
    )
    if scale > 1e-300:
        r = got / ref
        print(
            f"   elementwise ratio (s_w1+SW)/direct: {r.real.min():.6f} .. {r.real.max():.6f} (imag {np.abs(r.imag).max():.1e})"
        )
