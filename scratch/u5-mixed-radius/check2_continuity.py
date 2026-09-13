"""U5 (a), identity check 2: does current continuity through the node still
EMERGE at a two-radius crossing node, under each candidate corner rule?

The crossing fill imposes no continuity row: continuity emerges from each
block's field-form equivalence (momwire#524 phase 2), which the derivation says
survives only if each block keeps ONE radius (DERIVATION-MIXED-RADIUS.md §4).
Registered before this ran (§6): observer and source keep the node's KCL
deficit at the equal-radius level (~1e-7); the harmonic mean does not.

A momwire-native crossing rod (below arm 2 m, above arm 10 m, fed at 4.33 m on
the above arm, soil A, 7 MHz), graded toward the node as probe18's g2, with
per-wire radii [a_B, a_A]. Harness-side spellings, no src change:

  * the scope check is handed uniform radii so the refusal does not fire;
  * the crossing context carries min(a_A, a_B), which only sets the node grading;
  * `cross_complete_block_split` is evaluated at the rule's above-observer
    radius x1 (returned, so `Z -= t_ab` subtracts it) and at the
    below-observer radius x2; because the call site then subtracts t_ab(x1).T,
    the difference t_ab(x1).T - t_ab(x2).T is added back through the
    self-completion return, making the net Z -= t_ab(x2).T;
  * self completions are computed per family at that member's own radius.

Rules (x1, x2): observer (a_A, a_B), source (a_B, a_A), harmonic (a_h, a_h).
Readings, not verdicts: `single_A` and `single_B`, the whole fill at one radius
(what removing the refusal and reading `_radius_per_wire[0]` would do). At
a_A == a_B every rule must reproduce the unpatched solver bit for bit
(the plumbing check).

Run from the momwire repo root:
  python scratch/u5-mixed-radius/check2_continuity.py [--out F]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from momwire import _below_interface, _crossing_fill
from momwire.bspline import BSplineSolver

C0 = 299792458.0
WL7 = C0 / 7e6
SOIL_A = (13.0, 0.005)
BELOW = ([-2.0, -0.5, -0.1, -0.025], [3, 2, 3, 2])
ABOVE = ([0.025, 0.1, 0.5, 10.0], [2, 3, 2, 19])
L_BELOW = 2.0

STATE = {"rule": None, "a_A": None, "a_B": None, "calls": 0}


def deck(a_A, a_B, refine=1):
    below_pts = np.array([(0.0, 0.0, z) for z in [*BELOW[0], 0.0]])
    above_pts = np.array([(0.0, 0.0, z) for z in [0.0, *ABOVE[0]]])
    return dict(
        wires=[below_pts, above_pts],
        n_per_edge_per_wire=[
            [c * refine for c in BELOW[1]],
            [c * refine for c in ABOVE[1]],
        ],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(1, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=[a_B, a_A],
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def rule_radii(rule, a_A, a_B):
    if rule == "observer":
        return a_A, a_B
    if rule == "source":
        return a_B, a_A
    if rule == "harmonic":
        h = 2 * a_A * a_B / (a_A + a_B)
        return h, h
    if rule == "single_A":
        return a_A, a_A
    if rule == "single_B":
        return a_B, a_B
    raise ValueError(rule)


def self_radii(rule, a_A, a_B):
    """(above family, below family) self-completion radii."""
    if rule == "single_A":
        return a_A, a_A
    if rule == "single_B":
        return a_B, a_B
    return a_A, a_B


_orig_scope = _below_interface.crossing_junctions
_orig_ctx = BSplineSolver._crossing_context
_orig_cross = _crossing_fill.cross_complete_block_split
_orig_bnd = _crossing_fill._bnd_and_corner


def scope(media, groups, grounded, polylines, ground_z, radii):
    radii = np.asarray(radii, dtype=float)
    return _orig_scope(
        media, groups, grounded, polylines, ground_z, np.full_like(radii, radii.max())
    )


def ctx_min(self, geom, supp_seg, polys):
    ctx = _orig_ctx(self, geom, supp_seg, polys)
    return ctx._replace(a_wire=min(STATE["a_A"], STATE["a_B"]))


PENDING = {}


def cross(ctx, a_idx, b_idx, A, B, **kw):
    STATE["calls"] += 1
    x1, x2 = rule_radii(STATE["rule"], STATE["a_A"], STATE["a_B"])
    t1 = _orig_cross(ctx._replace(a_wire=x1), a_idx, b_idx, A, B, **kw)
    if x2 == x1:
        PENDING["fix"] = None
    else:
        t2 = _orig_cross(ctx._replace(a_wire=x2), a_idx, b_idx, A, B, **kw)
        PENDING["fix"] = t1.T - t2.T
    return t1


def self_completions(ctx, ax_b, ax_a):
    _eps_t, eps_m, k_p, k_m, c2, a_m = ctx.medium
    gz = float(ctx.ground_z)
    omega, eps0 = ctx.omega, ctx.eps
    r_above, r_below = self_radii(STATE["rule"], STATE["a_A"], STATE["a_B"])
    total = np.zeros((ax_b["n_basis"],) * 2, dtype=np.complex128)
    for ax, k, wgt, eps, a in (
        (ax_b, k_m, a_m, eps_m, r_below),
        (ax_a, k_p, c2, eps0, r_above),
    ):
        beta_dir = 1.0 / (1j * omega * eps * 4 * np.pi)
        beta_img = wgt / (1j * omega * eps * 4 * np.pi)
        for beta, mirror in ((beta_dir, False), (-beta_img, True)):
            live, row_term, col_term, corner = _orig_bnd(ax, k, a, gz, mirror=mirror)
            if live.size == 0:
                continue
            total[live, :] += beta * row_term
            total[:, live] += beta * col_term
            total[np.ix_(live, live)] += beta * corner
    if PENDING.get("fix") is not None:
        total += PENDING["fix"]
    return total


def install():
    _below_interface.crossing_junctions = scope
    BSplineSolver._crossing_context = ctx_min
    _crossing_fill.cross_complete_block_split = cross
    _crossing_fill.self_completions = self_completions


def uninstall():
    _below_interface.crossing_junctions = _orig_scope
    BSplineSolver._crossing_context = _orig_ctx
    _crossing_fill.cross_complete_block_split = _orig_cross
    _crossing_fill.self_completions = _orig_selfc


_orig_selfc = _crossing_fill.self_completions


def node_readout(s, coeffs):
    at = [np.array([L_BELOW]), np.array([0.0])]
    cur = s.currents_at_knots(coeffs, s_array=at)
    slo = s.current_slopes(coeffs, s_array=at)
    i_m, i_p = complex(cur[0][0]), complex(cur[1][0])
    d_m, d_p = complex(slo[0][0]), complex(slo[1][0])
    eps_t = s._buried_medium()[0]
    return dict(
        kcl_rel=abs(i_p - i_m) / abs(i_p),
        slope_ratio=str(d_p / d_m),
        slope_vs_agard=abs(d_p / d_m - 1 / eps_t) / abs(1 / eps_t),
    )


def solve(a_A, a_B, rule, refine):
    STATE.update(rule=rule, a_A=a_A, a_B=a_B, calls=0)
    PENDING.clear()
    s = BSplineSolver(**deck(a_A, a_B, refine))
    z, coeffs = s.compute_impedance()
    rec = dict(
        rule=rule,
        a_A=a_A,
        a_B=a_B,
        refine=refine,
        z=str(complex(z)),
        cross_calls=STATE["calls"],
        **node_readout(s, coeffs),
    )
    return rec, complex(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = []

    # plumbing: at equal radii every rule is the unpatched solver, bit for bit
    s0 = BSplineSolver(**deck(1e-3, 1e-3))
    z0, _ = s0.compute_impedance()
    install()
    try:
        for rule in ("observer", "source", "harmonic", "single_A", "single_B"):
            rec, z = solve(1e-3, 1e-3, rule, 1)
            rec["bit_identical_to_unpatched"] = z == complex(z0)
            out.append(rec)
            print(
                f"equal radii  {rule:9s} z={z:.10g} bit-identical={rec['bit_identical_to_unpatched']}"
                f"  kcl_rel={rec['kcl_rel']:.2e} cross_calls={rec['cross_calls']}",
                flush=True,
            )
        broken = [r["rule"] for r in out if not r["bit_identical_to_unpatched"]]
        if broken:
            raise SystemExit(
                f"equal-radius collapse is not bit-identical for {broken}: "
                "the harness spelling is not the shipped fill, stop"
            )
        for ratio in (2.0, 4.0):
            for refine in (1, 2):
                a_A, a_B = 1e-3, 1e-3 / ratio
                for rule in ("observer", "source", "harmonic", "single_A", "single_B"):
                    rec, z = solve(a_A, a_B, rule, refine)
                    out.append(rec)
                    print(
                        f"a_A/a_B={ratio:g} refine={refine} {rule:9s} z={z:.8g}  "
                        f"kcl_rel={rec['kcl_rel']:.2e}  slope-vs-AGARD={rec['slope_vs_agard']:.3e}  "
                        f"cross_calls={rec['cross_calls']}",
                        flush=True,
                    )
    finally:
        uninstall()
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
