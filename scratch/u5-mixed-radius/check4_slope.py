"""U5 §9, check 4: the slope target at a two-radius crossing node.

DERIVATION-MIXED-RADIUS.md §9. Check 2's rod with step (b)'s radii at
r = 1, 2, 4, 8, under check 3's spellings `obs` and `node_B`, with
VC_keep(node_B) (the crossing junction's KCL multiplier, every point term kept)
and VC_keep(node_B) solved with the transposed matrix. Readouts per solve: the
node KCL deficit, the node slope ratio times eps~, and at d = 0.2, 0.1, 0.05,
0.025 m the pointwise ratio I'(+d)/I'(-d) and the secant ratio
[(I(+d) - I(0+))/d] / [(I(0-) - I(-d))/d], both times eps~.

Run from the momwire repo root:
  python scratch/u5-mixed-radius/check4_slope.py [--out F]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import check2_continuity as c2
import check3_node_potential as c3
import numpy as np

from momwire.bspline import BSplineSolver

CONFIGS = {
    "control": (0.25e-3, 0.25e-3),
    "ratio2": (0.25e-3, 0.125e-3),
    "ratio4": (0.25e-3, 0.0625e-3),
    "ratio05": (0.125e-3, 0.25e-3),
}
REFINES = (1, 2, 4, 8)
DS = (0.2, 0.1, 0.05, 0.025)


def readout(s, coeffs):
    ds = np.array(DS)
    at = [np.append(c2.L_BELOW - ds, c2.L_BELOW), np.append(ds, 0.0)]
    cur = s.currents_at_knots(coeffs, s_array=at)
    slo = s.current_slopes(coeffs, s_array=at)
    eps_t = complex(s._buried_medium()[0])
    below_i, above_i = np.asarray(cur[0], complex), np.asarray(cur[1], complex)
    below_d, above_d = np.asarray(slo[0], complex), np.asarray(slo[1], complex)
    i_m, i_p = below_i[-1], above_i[-1]
    rec = dict(
        kcl_rel=float(abs(i_p - i_m) / abs(i_p)),
        node=str(above_d[-1] / below_d[-1] * eps_t),
    )
    for k, d in enumerate(DS):
        r_p = above_d[k] / below_d[k] * eps_t
        r_s = ((above_i[k] - i_p) / d) / ((i_m - below_i[k]) / d) * eps_t
        rec[f"p{d:g}"] = str(r_p)
        rec[f"s{d:g}"] = str(r_s)
    return rec


def solve(a_A, a_B, spelling, refine):
    line, point = c3.radii(spelling, a_A, a_B)
    c3.STATE.update(line=line, point=point, a_min=min(a_A, a_B), calls=0)
    c3.CAP.clear()
    s = BSplineSolver(**c2.deck(a_A, a_B, refine))
    if s.degree < 2:
        raise RuntimeError(f"degree {s.degree}: the slope is not continuous")
    z, coeffs = s.compute_impedance()
    z = complex(z)
    if c3.CAP.get("solves") != 1 or c3.STATE["calls"] != 1:
        raise RuntimeError(f"solves {c3.CAP.get('solves')}, calls {c3.STATE['calls']}")
    if abs(1.0 / (c3.CAP["v"] @ coeffs) - z) > 1e-10 * abs(z):
        raise RuntimeError("Z = 1/(v.c) does not hold")
    return s, z, coeffs


def bordered(s, transpose):
    """The captured run with the crossing junction's KCL row, every point term
    kept; with `transpose`, the matrix transposed."""
    Z = c3.CAP["Z"]
    n = Z.shape[0]
    u = np.zeros((1, n))
    u[0, c3.CAP["na"]], u[0, c3.CAP["nb"]] = 1.0, -1.0
    kcl = np.vstack([c3.CAP["kcl"].reshape(-1, n), u])
    matrix = np.ascontiguousarray(Z.T) if transpose else Z
    coeffs = c3._orig_solve(s, matrix, c3.CAP["v"], kcl)
    return complex(1.0 / (c3.CAP["v"] @ coeffs)), coeffs


def dev(rec, key):
    return abs(complex(rec[key]) - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    a0 = CONFIGS["control"][0]
    z0 = {
        r: complex(BSplineSolver(**c2.deck(a0, a0, r)).compute_impedance()[0])
        for r in REFINES
    }
    meta = dict(momwire_file=c3.momwire.__file__, configs=CONFIGS, ds=DS)
    print(meta, flush=True)
    out = []
    c3.install()
    try:
        for cfg, (a_A, a_B) in CONFIGS.items():
            for r in REFINES:
                for sp in ("obs", "node_B"):
                    s, z, coeffs = solve(a_A, a_B, sp, r)
                    if cfg == "control" and abs(z - z0[r]) > 1e-9 * abs(z0[r]):
                        raise SystemExit(f"control r={r} {sp}: {z} vs {z0[r]}; stop")
                    rec = dict(
                        config=cfg,
                        a_A=a_A,
                        a_B=a_B,
                        refine=r,
                        spelling=sp,
                        z=str(z),
                        split=readout(s, coeffs),
                    )
                    msg = (
                        f"{cfg:8s} r={r} {sp:6s} z={z:.8g} "
                        f"kcl={rec['split']['kcl_rel']:.2e} "
                        f"|node-1|={dev(rec['split'], 'node'):.3e}"
                    )
                    if sp == "node_B":
                        z_k, c_k = bordered(s, transpose=False)
                        z_t, c_t = bordered(s, transpose=True)
                        r_k, r_t = readout(s, c_k), readout(s, c_t)
                        if r_k["kcl_rel"] > 1e-9 or r_t["kcl_rel"] > 1e-9:
                            raise RuntimeError(
                                f"KCL {r_k['kcl_rel']}, {r_t['kcl_rel']}: row mis-built"
                            )
                        if abs(z_t - z_k) > 1e-9 * abs(z_k):
                            raise RuntimeError(f"transposed Z {z_t} vs {z_k}")
                        rec["vc_keep"] = dict(z=str(z_k), **r_k)
                        rec["transposed"] = dict(z=str(z_t), **r_t)
                        msg += (
                            f" | VC_keep z={z_k:.8g} |node-1|={dev(r_k, 'node'):.3e} "
                            f"|p0.1-1|={dev(r_k, 'p0.1'):.3e} "
                            f"|s0.1-1|={dev(r_k, 's0.1'):.3e} "
                            f"| transposed p0.1 rel "
                            f"{abs(complex(r_t['p0.1']) - complex(r_k['p0.1'])) / abs(complex(r_k['p0.1'])):.1e}"
                        )
                    out.append(rec)
                    print(msg, flush=True)
    finally:
        c3.uninstall()
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
