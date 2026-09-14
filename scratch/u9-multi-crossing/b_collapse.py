"""U9 (b): the eps~ = 1 collapse on two crossing nodes (PLAN.md (b), Amendment 1).

Run against the worktree under test (`run_b.sh` does both, guards first):

  PYTHONPATH=<worktree>/src python b_collapse.py guards --out F
  PYTHONPATH=<worktree>/src python b_collapse.py run --spelling A --d 12 \\
      --mode cross --out F

Scratch patches only, no src edit:
- P1 lifts `crossing_junctions`' second-node refusal;
- P2 clamps the plan's theta_min to the floor and zeroes the below/below
  projection, at eps~ = 1 only, where the integrals are identically zero;
- P3 re-spells the corner loop: `same` / `cross` / `blind`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import momwire
from momwire import _below_interface, _crossing_fill, _near_interface
from momwire import _sommerfeld_below
from momwire import bspline
from momwire.bspline import BSplineSolver
from test_crossing_serve_524 import _COLLAPSE_N_QP, _GRADES, A_WIRE, WL7, crossing_deck

EPS1 = (1.0, 0.0)
FEED = 4.3333333333
TOP = 10.0
BAR_OHM = 0.05
FLOOR = float(np.radians(_sommerfeld_below._SOMM_BELOW_TH_MIN_DEG))
STATE = {"calls": 0, "cross_pairs": 0, "proj_calls": 0, "th_min_real": np.inf}
REAL_EAC = _crossing_fill._ends_and_corner
REAL_PROJ = _sommerfeld_below.remainder_field_proj_below


def _pair_points():
    g = _GRADES[1]
    below = np.array([(0.0, 0.0, z) for z in g["below"][0] + [0.0]])
    above = np.array([(0.0, 0.0, z) for z in [0.0] + g["above"][0]])
    return below, above, list(g["below"][1]), list(g["above"][1])


def crossing_two(spelling, d):
    """Pair 1 is `crossing_deck(1)` at x = 0; pair 2 at x = d, as spelled."""
    below, above, nb, na = _pair_points()
    sh = np.array([d, 0.0, 0.0])
    if spelling == "A":
        rod2, mono2, nb2, na2 = below + sh, above + sh, nb, na
        j2, f2 = [(2, "end"), (3, "start")], FEED
    else:
        rod2, mono2 = below[::-1] + sh, above[::-1] + sh
        nb2, na2 = nb[::-1], na[::-1]
        j2, f2 = [(2, "start"), (3, "end")], TOP - FEED
    return dict(
        wires=[below, above, rod2, mono2],
        n_per_edge_per_wire=[nb, na, nb2, na2],
        junctions=[[(0, "end"), (1, "start")], j2],
        feeds=[(1, FEED, 1 + 0j), (3, f2, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=EPS1,
        ground_model="sommerfeld",
        n_qp_pair=_COLLAPSE_N_QP,
    )


def truth_two(spelling, d):
    """Each pair as ONE free-space polyline in its above wire's direction."""
    below, above, nb, na = _pair_points()
    w1 = np.vstack([below, above[1:]])
    n1 = nb + na
    sh = np.array([d, 0.0, 0.0])
    if spelling == "A":
        w2, n2, f2 = w1 + sh, n1, 2.0 + FEED
    else:
        w2, n2, f2 = w1[::-1] + sh, n1[::-1], TOP - FEED
    return dict(
        wires=[w1, w2],
        n_per_edge_per_wire=[n1, n2],
        feeds=[(0, 2.0 + FEED, 1 + 0j), (1, f2, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        n_qp_pair=_COLLAPSE_N_QP,
    )


def patch_scope():
    """P1: the one `if len(crossing) > 1:` becomes `if False:`."""
    src = inspect.getsource(_below_interface.crossing_junctions)
    old = "    if len(crossing) > 1:\n"
    assert src.count(old) == 1, "P1's anchor moved"
    code = compile(
        src.replace(old, "    if False:\n"), _below_interface.__file__, "exec"
    )
    exec(code, _below_interface.__dict__)


def patch_floor():
    """P2, at eps~ = 1 only."""
    real_ext = bspline._pair_extents_below

    def ext(x, y, d_b, *a, **k):
        r1, th = real_ext(x, y, d_b, *a, **k)
        STATE["th_min_real"] = min(STATE["th_min_real"], th)
        return r1, max(th, FLOOR)

    def proj(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid):
        assert complex(k_m) == complex(k_p), "P2 zeroes the remainder at eps~ = 1 only"
        STATE["proj_calls"] += 1
        return np.zeros((np.asarray(obs).shape[0], np.asarray(src).shape[0]), complex)

    bspline._pair_extents_below = ext
    _sommerfeld_below.remainder_field_proj_below = proj


def patch_corner(mode):
    """P3: the corner loop over every in-plane end pair, in the shipped float
    expression, with the cross-node pairs spelled by `mode`."""

    def eac(
        ctx,
        A,
        B,
        eps_t,
        k_p,
        c1,
        gz,
        memo=None,
        *,
        corner=True,
        test_ends=True,
        source_ends=True,
    ):
        t_ab = REAL_EAC(
            ctx,
            A,
            B,
            eps_t,
            k_p,
            c1,
            gz,
            memo=memo,
            corner=False,
            test_ends=test_ends,
            source_ends=source_ends,
        )
        STATE["calls"] += 1
        if not corner:
            return t_ab
        a_wire = float(ctx.a_wire)
        v_at = {}
        for pt_a, sig_a, fv_a in A["ends"]:
            if abs(pt_a[2] - gz) > 1e-12:
                continue
            for pt_b, sig_b, fv_b in B["ends"]:
                if abs(pt_b[2] - gz) > 1e-12:
                    continue
                rho = float(np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]))
                same = rho < 1e-9
                if not same:
                    STATE["cross_pairs"] += 1
                    if mode == "same":
                        continue
                key = 0.0 if same else rho
                if key not in v_at:
                    r_eff = a_wire if same else float(np.hypot(rho, a_wire))
                    v_at[key] = complex(
                        _near_interface.six_point(
                            eps_t,
                            k_p,
                            r_eff,
                            0.0,
                            0.0,
                            rtol=_crossing_fill._CORNER_RTOL,
                        )[1]
                    )
                nza, nzb = np.flatnonzero(fv_a), np.flatnonzero(fv_b)
                if same or mode == "cross":
                    coef = -sig_a * sig_b * c1 * v_at[key]
                else:
                    coef = c1 * v_at[key]
                t_ab[np.ix_(nza, nzb)] += coef * np.outer(fv_a[nza], fv_b[nzb])
        return t_ab

    _crossing_fill._ends_and_corner = eac


def zmat(kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        y = np.asarray(BSplineSolver(**kw).compute_port_solution().y, dtype=complex)
    return np.linalg.inv(np.atleast_2d(y)), round(time.time() - t0, 1)


def enc(z):
    return [[float(v.real), float(v.imag)] for v in np.asarray(z).ravel()]


def guards():
    rec = dict(mode="guards", momwire=momwire.__file__)
    one = crossing_deck(1, ground_eps=EPS1, n_qp_pair=_COLLAPSE_N_QP)

    # G-b0: the real projection on the served single-node eps~ = 1 deck.
    seen = []

    def record(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid):
        out = REAL_PROJ(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid)
        want = (np.asarray(obs).shape[0], np.asarray(src).shape[0])
        seen.append((tuple(np.shape(out)), want, float(np.max(np.abs(out)))))
        return out

    _sommerfeld_below.remainder_field_proj_below = record
    z0, t0 = zmat(one)
    _sommerfeld_below.remainder_field_proj_below = REAL_PROJ
    rec["G_b0"] = dict(
        calls=len(seen),
        shapes_match=all(s == w for s, w, _m in seen),
        max_abs=max((m for _s, _w, m in seen), default=None),
        ok=bool(seen) and all(s == w and m == 0.0 for s, w, m in seen),
    )

    # G-b1: P3 cross alone, then P1 + P2 + P3, against the unpatched Z.
    patch_corner("cross")
    z1, _ = zmat(one)
    calls_p3 = STATE["calls"]
    patch_scope()
    patch_floor()
    z2, _ = zmat(one)
    rec["G_b1"] = dict(
        z_unpatched=enc(z0),
        p3_equal=bool(np.array_equal(z0, z1)),
        p123_equal=bool(np.array_equal(z0, z2)),
        p3_called=calls_p3,
        cross_pairs_on_one_node=STATE["cross_pairs"],
        seconds=t0,
    )
    rec["G_b1"]["ok"] = (
        rec["G_b1"]["p3_equal"]
        and rec["G_b1"]["p123_equal"]
        and calls_p3 > 0
        and STATE["cross_pairs"] == 0
    )

    # G-b2: k^2 V = e^{-jkR}/R at eps~ = 1.
    k = 2.0 * np.pi / WL7
    rows = []
    for d in (1.0, 12.0):
        R = float(np.hypot(d, A_WIRE))
        six = _near_interface.six_point(
            1.0, k, R, 0.0, 0.0, rtol=_crossing_fill._CORNER_RTOL
        )
        g = np.exp(-1j * k * R) / R
        rows.append(dict(d=d, rel=float(abs(k * k * six[1] - g) / abs(g))))
    rec["G_b2"] = dict(rows=rows, ok=all(r["rel"] <= 1e-8 for r in rows))
    rec["ok"] = rec["G_b0"]["ok"] and rec["G_b1"]["ok"] and rec["G_b2"]["ok"]
    return rec


def run(spelling, d, mode):
    patch_scope()
    patch_floor()
    patch_corner(mode)
    zx, tx = zmat(crossing_two(spelling, d))
    calls, pairs = STATE["calls"], STATE["cross_pairs"]
    zt, tt = zmat(truth_two(spelling, d))
    dz = np.abs(zx - zt)
    return dict(
        mode="run",
        momwire=momwire.__file__,
        spelling=spelling,
        d=d,
        corner_mode=mode,
        z_crossing=enc(zx),
        z_truth=enc(zt),
        abs_dz=[float(v) for v in dz.ravel()],
        max_abs_dz=float(dz.max()),
        abs_dz12=float(dz[0, 1]),
        reciprocity_crossing=float(abs(zx[0, 1] - zx[1, 0]) / abs(zx[0, 1])),
        reciprocity_truth=float(abs(zt[0, 1] - zt[1, 0]) / abs(zt[0, 1])),
        within_bar=bool(dz.max() <= BAR_OHM),
        G_b3=dict(
            eac_calls=calls,
            cross_pairs=pairs,
            per_call=pairs / calls if calls else None,
        ),
        P2=dict(
            th_min_real_deg=float(np.degrees(STATE["th_min_real"])),
            clamp_engaged=bool(STATE["th_min_real"] < FLOOR),
            proj_calls=STATE["proj_calls"],
        ),
        seconds=dict(crossing=tx, truth=tt),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("guards", "run"))
    ap.add_argument("--spelling", choices=("A", "B"))
    ap.add_argument("--d", type=float)
    ap.add_argument("--mode", choices=("same", "cross", "blind"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = guards() if args.what == "guards" else run(args.spelling, args.d, args.mode)
    args.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    if args.what == "guards" and not rec["ok"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
