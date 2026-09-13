"""U5 §8, identity check 3: one node potential at a two-radius crossing node.

DERIVATION-MIXED-RADIUS.md §8. Each crossing-node row holds POINT tests of the
potential at the node: the self completion's node row term and node corner,
the cross block's test-side end terms (BT and TW for the above node row; SQ and
SW, by transpose, for the below node row) and the cross corner. Everything else
in the fill is a line test. Harness-side spellings, no src change:

  obs       every term at its observer's radius; the node point tests of row
            na at a_A and of row nb at a_B (check 2's observer-side, piecewise)
  node_A    line tests as obs; both node rows' point tests at a_A
  node_B    the same at a_B
  single_B  everything at a_B (check 2's single_B, piecewise)
  VC        from any run: its node point tests removed from rows na and nb and
            a KCL multiplier row added for the crossing junction -- the single
            node potential the same-medium junctions carry (`kcl_A`)

Rods: check 2's momwire-native rod (`--rod check2`), and step (b)'s rod at
L = 0.30 m, r = 2, against the NEC-5 values banked in b_*.json (`--rod b`).

Run from the momwire repo root:
  python scratch/u5-mixed-radius/check3_node_potential.py --rod check2 [--out F]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import b_rod_ladder as bl
import check2_continuity as c2
import momwire
import numpy as np

from antennaknobs.engines.momwire import MomwireEngine
from momwire import _below_interface, _crossing_fill
from momwire._sommerfeld_transmitted import _c1_moment
from momwire.bspline import BSplineSolver

HERE = Path(__file__).resolve().parent
SPELLINGS = ("obs", "node_A", "node_B", "single_B")
STATE = {"line": None, "point": None, "a_min": None, "calls": 0}
CAP = {}

_orig_ctx = BSplineSolver._crossing_context
_orig_cross = _crossing_fill.cross_complete_block_split
_orig_selfc = _crossing_fill.self_completions
_orig_bnd = _crossing_fill._bnd_and_corner
_orig_solve = BSplineSolver._solve_with_kcl
_orig_scope = _below_interface.crossing_junctions


def radii(spelling, a_A, a_B):
    """((line above, line below), (point in row na, point in row nb))."""
    if spelling == "obs":
        return (a_A, a_B), (a_A, a_B)
    if spelling == "node_A":
        return (a_A, a_B), (a_A, a_A)
    if spelling == "node_B":
        return (a_A, a_B), (a_B, a_B)
    if spelling == "single_B":
        return (a_B, a_B), (a_B, a_B)
    raise ValueError(spelling)


def ctx_min(self, geom, supp_seg, polys):
    return _orig_ctx(self, geom, supp_seg, polys)._replace(a_wire=STATE["a_min"])


def pieces(ctx, a_idx, b_idx, A, B, x):
    """t_ab at radius x as (main, e_a, e_b, corner): the main sandwich; the
    above-end terms BT + TW (rows = above node tents); the below-end terms
    SQ + SW (columns = below node tents); the corner."""
    ctx_x = ctx._replace(a_wire=x)
    eps_t, _eps_m, k_p, _k_m, _c2w, _a_m = ctx.medium
    gz = float(ctx.ground_z)
    c1 = _c1_moment(ctx.omega, ctx.mu)
    memo = {}
    main = _crossing_fill._main_split(
        ctx_x, a_idx, b_idx, A, B, eps_t, k_p, c1, gz, memo
    )
    ends = _crossing_fill._ends_and_corner
    e_a = ends(ctx_x, A, dict(B, ends=[]), eps_t, k_p, c1, gz, memo=memo, corner=False)
    e_b = ends(ctx_x, dict(A, ends=[]), B, eps_t, k_p, c1, gz, memo=memo, corner=False)
    both = ends(ctx_x, A, B, eps_t, k_p, c1, gz, memo=memo, corner=False)
    full = ends(ctx_x, A, B, eps_t, k_p, c1, gz, memo=memo, corner=True)
    scale = max(float(np.abs(both).max()), 1e-300)
    if float(np.abs(e_a + e_b - both).max()) > 1e-12 * scale:
        raise RuntimeError("the end terms do not separate by axis")
    return main, e_a, e_b, full - both


def cross(ctx, a_idx, b_idx, A, B, **kw):
    STATE["calls"] += 1
    if kw.get("corner", True) is not True:
        raise RuntimeError("corner=False is not this harness's path")
    (l_a, l_b), (p_a, p_b) = STATE["line"], STATE["point"]
    cache = {}

    def at(x):
        if x not in cache:
            cache[x] = pieces(ctx, a_idx, b_idx, A, B, x)
        return cache[x]

    rows_a = at(l_a)[0] + at(p_a)[1] + at(l_a)[2] + at(p_a)[3]
    rows_b = at(l_b)[0] + at(l_b)[1] + at(p_b)[2] + at(p_b)[3]
    CAP["pt_a"] = at(p_a)[1] + at(p_a)[3]
    CAP["pt_b"] = at(p_b)[2] + at(p_b)[3]
    # the call site subtracts rows_a and rows_a.T; the self-completion return
    # adds this back so the net below-row subtraction is rows_b.T
    CAP["fix"] = rows_a.T - rows_b.T
    return rows_a


def selfc(ctx, ax_b, ax_a):
    _eps_t, eps_m, k_p, k_m, c2w, a_m = ctx.medium
    gz = float(ctx.ground_z)
    omega, eps0 = ctx.omega, ctx.eps
    (l_a, l_b), (p_a, p_b) = STATE["line"], STATE["point"]
    n = ax_b["n_basis"]
    total = np.zeros((n, n), dtype=np.complex128)
    point = np.zeros((n, n), dtype=np.complex128)
    for name, ax, k, wgt, eps, r_line, r_point in (
        ("nb", ax_b, k_m, a_m, eps_m, l_b, p_b),
        ("na", ax_a, k_p, c2w, eps0, l_a, p_a),
    ):
        if any(abs(e[0][2] - gz) > 1e-12 for e in ax["ends"]):
            raise RuntimeError("an end off the crossing node: not this harness's deck")
        beta_dir = 1.0 / (1j * omega * eps * 4 * np.pi)
        beta_img = wgt / (1j * omega * eps * 4 * np.pi)
        for beta, mirror in ((beta_dir, False), (-beta_img, True)):
            live, _row, col, _cor = _orig_bnd(ax, k, r_line, gz, mirror=mirror)
            live_p, row, _col, cor = _orig_bnd(ax, k, r_point, gz, mirror=mirror)
            if live.size == 0:
                continue
            if not np.array_equal(live, live_p) or live.size != 1:
                raise RuntimeError(f"node tents on {name}: {live}, {live_p}")
            CAP[name] = int(live[0])
            total[live, :] += beta * row
            total[:, live] += beta * col
            total[np.ix_(live, live)] += beta * cor
            point[live, :] += beta * row
            point[np.ix_(live, live)] += beta * cor
    CAP["self_pt"] = point
    return total + CAP["fix"]


def solve_capture(self, Z, v, kcl_A, overwrite=False):
    CAP["solves"] = CAP.get("solves", 0) + 1
    CAP["Z"] = np.array(Z, copy=True)
    CAP["v"] = np.array(v, copy=True)
    CAP["kcl"] = np.array(kcl_A, copy=True)
    return _orig_solve(self, Z, v, kcl_A, overwrite=overwrite)


def install():
    _below_interface.crossing_junctions = c2.scope
    BSplineSolver._crossing_context = ctx_min
    _crossing_fill.cross_complete_block_split = cross
    _crossing_fill.self_completions = selfc
    BSplineSolver._solve_with_kcl = solve_capture


def uninstall():
    _below_interface.crossing_junctions = _orig_scope
    BSplineSolver._crossing_context = _orig_ctx
    _crossing_fill.cross_complete_block_split = _orig_cross
    _crossing_fill.self_completions = _orig_selfc
    BSplineSolver._solve_with_kcl = _orig_solve


def vc_solve(s):
    """The V-constrained reference from the captured run."""
    Z = CAP["Z"]
    removal = CAP["pt_a"] + CAP["pt_b"].T - CAP["self_pt"]
    na, nb = CAP["na"], CAP["nb"]
    rows = set(np.flatnonzero(np.any(removal != 0, axis=1)).tolist())
    if not rows <= {na, nb}:
        raise RuntimeError(f"node point tests found outside rows {na}, {nb}: {rows}")
    u = np.zeros((1, Z.shape[0]))
    u[0, na], u[0, nb] = 1.0, -1.0
    kcl = np.vstack([CAP["kcl"].reshape(-1, Z.shape[0]), u])
    coeffs = _orig_solve(s, Z + removal, CAP["v"], kcl)
    return complex(1.0 / (CAP["v"] @ coeffs)), coeffs


def finish(spelling, s, z, coeffs, readout, **meta):
    if CAP.get("solves") != 1 or STATE["calls"] != 1:
        raise RuntimeError(
            f"solves {CAP.get('solves')}, crossing calls {STATE['calls']}"
        )
    z_formula = complex(1.0 / (CAP["v"] @ coeffs))
    if abs(z_formula - z) > 1e-10 * abs(z):
        raise RuntimeError(f"Z = 1/(v.c) does not hold: {z_formula} vs {z}")
    z_vc, c_vc = vc_solve(s)
    rec = dict(spelling=spelling, **meta, z=str(z), **readout(s, coeffs))
    rec["vc"] = dict(z=str(z_vc), **readout(s, c_vc))
    return rec


def run_check2_rod(a_A, a_B, spelling, refine):
    line, point = radii(spelling, a_A, a_B)
    STATE.update(line=line, point=point, a_min=min(a_A, a_B), calls=0)
    CAP.clear()
    s = BSplineSolver(**c2.deck(a_A, a_B, refine))
    z, coeffs = s.compute_impedance()
    return finish(
        spelling,
        s,
        complex(z),
        coeffs,
        c2.node_readout,
        a_A=a_A,
        a_B=a_B,
        refine=refine,
    )


def run_b_rod(a_top, a_rise, spelling, L=0.30, r=2):
    line, point = radii(spelling, a_top, a_rise)
    STATE.update(line=line, point=point, a_min=min(a_top, a_rise), calls=0)
    CAP.clear()
    b = bl.build(L, r, a_top, a_rise)
    with bl.even_parity():
        eng = MomwireEngine(b, ground=bl.SOIL_A)
        eng_segs, on_knot = bl.port_check(eng)
        if not on_knot:
            raise RuntimeError("feed off a knot")
        sim, coeffs, z = eng._solved_excited(eng._wavelength_for(b.freq))
    if sim.extended_kernel:
        raise RuntimeError("extended_kernel resolved True")
    return finish(
        spelling,
        sim,
        complex(z),
        coeffs,
        bl.node_readout,
        a_top=a_top,
        a_rise=a_rise,
        L=L,
        r=r,
        segs=eng_segs,
    )


def show(rec):
    vc = rec["vc"]
    return (
        f"{rec['spelling']:8s} z={complex(rec['z']):.8g} kcl={rec['kcl_rel']:.2e} "
        f"slope-vs-AGARD={rec['slope_vs_agard']:.3e} | VC z={complex(vc['z']):.8g} "
        f"kcl={vc['kcl_rel']:.2e} slope-vs-AGARD={vc['slope_vs_agard']:.3e}"
    )


def check2_rod():
    z0, _ = BSplineSolver(**c2.deck(1e-3, 1e-3)).compute_impedance()
    z0 = complex(z0)
    out = []
    install()
    try:
        for sp in SPELLINGS:
            rec = run_check2_rod(1e-3, 1e-3, sp, 1)
            rec["rel_to_unpatched"] = abs(complex(rec["z"]) - z0) / abs(z0)
            rec["vc_rel_to_split"] = abs(complex(rec["vc"]["z"]) - z0) / abs(z0)
            out.append(rec)
            print(
                f"equal radii {show(rec)}  rel {rec['rel_to_unpatched']:.1e}  "
                f"VC-vs-split {rec['vc_rel_to_split']:.1e}",
                flush=True,
            )
            if rec["rel_to_unpatched"] > 1e-9:
                raise SystemExit(f"C3.0a: {sp} is not the shipped fill; stop")
        for ratio in (2.0, 4.0):
            for refine in (1, 2):
                a_A, a_B = 1e-3, 1e-3 / ratio
                for sp in SPELLINGS:
                    rec = run_check2_rod(a_A, a_B, sp, refine)
                    out.append(rec)
                    print(f"a_A/a_B={ratio:g} refine={refine} {show(rec)}", flush=True)
    finally:
        uninstall()
    return out


def b_rod():
    banked = {
        cfg: {
            row["r"]: row
            for row in json.loads((HERE / f"b_{cfg}.json").read_text())["rows"]
            if abs(row["L"] - 0.30) < 1e-12
        }
        for cfg in bl.CONFIGS
    }
    out = []
    install()
    try:
        for cfg, (a_top, a_rise) in bl.CONFIGS.items():
            ref = banked[cfg][2]
            for sp in SPELLINGS:
                rec = run_b_rod(a_top, a_rise, sp)
                rec["config"] = cfg
                rec["nec5"] = ref["nec5"]
                rec["banked_observer_z"] = ref["momwire"]["z"]
                out.append(rec)
                print(
                    f"{cfg:8s} {show(rec)}  NEC-5 {complex(ref['nec5']):.8g}",
                    flush=True,
                )
    finally:
        uninstall()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rod", choices=("check2", "b"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    meta = dict(rod=args.rod, momwire_file=momwire.__file__)
    print(meta, flush=True)
    out = check2_rod() if args.rod == "check2" else b_rod()
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
