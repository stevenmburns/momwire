"""momwire#1029 phase 1, gates S-A and S-B on the `rows=` parameter.

  # S-A: run in BOTH trees, then diff the two dumps
  PYTHONPATH=<tree>/src:<ak src> python .../s_ab_gates.py --dump out.json
  # S-B: the patched tree only
  PYTHONPATH=<patched>/src:<ak src> python .../s_ab_gates.py --sb --out sb.json

S-A — `rows=None` is bit-identical to the pre-change tree. The dump carries, per
deck, a sha256 of the assembled Z's bytes, of the solved coefficients' bytes, and
Z_in's exact hex. Bytes, not tolerances: rows=None must take today's path.

S-B — a restricted fill equals the corresponding rows of the full fill, and the
PLAN is untouched. Per PLAN-phase1.md Amendment 1:

  * the requested rows agree to the assembled tolerance;
  * rows OUTSIDE the request may be non-zero only where the crossing block
    writes them (it is filled in full by contract), so they are reported, not
    failed;
  * `q_factor` and every `plan_buried` field — r1_below, r1_above, r_cross_max,
    r_cross_min, zp_min, zp_max — are equal between rows=None and rows=subset.

The plan is captured by wrapping `_below_interface.plan_buried` for the duration
of each fill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feasibility import build_solver, sector_groups

DECKS = ("radials4", "radials12")


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def _parts(solver):
    """geom, basis and the pieces `compute_impedance` would use."""
    geom = solver._build_geometry()
    supp_seg, polys, kcl_A, wire_knots, wbg = solver._build_basis_polynomials(geom)
    return geom, supp_seg, polys, kcl_A, wire_knots, wbg


def _capture_plan(momwire_below):
    """Wrap `plan_buried` so the fill's plan and q_factor can be compared."""
    seen = {}
    original = momwire_below.plan_buried

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        # BY NAME: BuriedPlan carries a leading `below` field the call site's
        # unpacking does not show, so positional indices are off by one.
        seen["plan"] = {k: float(v) for k, v in out.plan.items() if np.isscalar(v)}
        # q_factor shows up as the node stacks' lengths: the base and raised
        # stacks differ exactly when it is > 1.
        seen["q_len_a"] = int(len(out.obs_a))
        seen["q_len_ax"] = int(len(out.obs_ax))
        seen["q_len_b"] = int(len(out.obs_b))
        seen["q_len_bx"] = int(len(out.obs_bx))
        return out

    momwire_below.plan_buried = spy
    return seen, (lambda: setattr(momwire_below, "plan_buried", original))


def fill(solver, rows=None):
    from momwire import _below_interface

    geom, supp_seg, polys, kcl_A, wire_knots, wbg = _parts(solver)
    seen, restore = _capture_plan(_below_interface)
    try:
        Z = _below_interface.compute_Z_operator_buried(
            geom,
            supp_seg,
            polys,
            f=_below_interface.BuriedFills(
                checkpoint=solver._checkpoint,
                nodes=solver._buried_nodes,
                wire_media=solver._wire_media,
                crossing_context=solver._crossing_context,
                assemble_Z=solver._assemble_Z,
                build_J_blocks_subset=solver._build_J_blocks_subset,
                accumulate_Z_subset_chunked=solver._accumulate_Z_subset_chunked,
                image_Z_weighted=solver._image_Z_weighted,
                image_tangent_dot=solver._image_tangent_dot,
                field_galerkin_block=solver._field_galerkin_block,
                apply_loading=solver._apply_loading,
            ),
            below_segments=solver._below_segments,
            buried_medium=solver._buried_medium,
            refuse_out_of_scope_fn=solver._refuse_buried_out_of_scope,
            crossing_junctions_fn=solver._crossing_junctions,
            serve_plan_fn=solver._buried_serve_plan,
            somm_grid_fn=solver._somm_grid,
            crossing_node_members_fn=solver._crossing_node_members,
            ground_z=solver.ground_z,
            eps=solver.eps,
            omega=solver.omega,
            mu=solver.mu,
            cancel_flag=solver._cancel_flag,
            chunked=solver._buried_chunked_serves,
            # `rows` is passed ONLY when set, so --dump calls the pre-change
            # signature unchanged and runs on the tree before the patch.
            **({} if rows is None else {"rows": rows}),
        )
    finally:
        restore()
    return Z, seen, (geom, supp_seg, polys, kcl_A, wire_knots, wbg)


def dump(n_radials):
    solver, z_engine, secs = build_solver(n_radials)
    Z, seen, parts = fill(solver)
    geom, supp_seg, polys, kcl_A, wire_knots, wbg = parts
    v, port_vectors, _vpf, all_voltages, kcl_con = solver._feed_drive_and_readout(
        geom, wire_knots, wbg, supp_seg.shape[0], kcl_A
    )
    coeffs = solver._solve_with_kcl(Z.copy(), v, kcl_con)
    z_in = complex(np.atleast_1d(all_voltages)[0] / (port_vectors[0] @ coeffs))
    return {
        "radials": n_radials,
        "n_basis": int(supp_seg.shape[0]),
        "sha_Z": _sha(Z),
        "sha_coeffs": _sha(coeffs),
        "z_in_hex": [z_in.real.hex(), z_in.imag.hex()],
        "z_in": [z_in.real, z_in.imag],
        "z_engine": [z_engine.real, z_engine.imag],
        "plan": seen.get("plan"),
        "secs": round(secs, 3),
    }


def sb(n_radials):
    solver, _z, _t = build_solver(n_radials, structure_only=True)
    geom = solver._build_geometry()
    _ss, _po, _kcl, _wk, wbg = _parts(solver)[1:]
    g = sector_groups(solver, geom, wbg)
    # the requested rows: sector 0's segments plus every axial segment
    seg_off = geom["seg_offsets"]
    per_wire = geom["per_wire"]

    def wire_segs(w):
        return np.arange(seg_off[w], seg_off[w] + per_wire[w]["n_total"])

    sector0 = g["wires"][0]
    axis_wires = [w for w in range(len(per_wire)) if w not in g["wires"]]
    rows = np.sort(
        np.concatenate([wire_segs(sector0), *[wire_segs(w) for w in axis_wires]])
    )

    # the call-shape log: what observer count each fill actually ran with
    calls = {"full": [], "rows": []}
    tag = {"which": "full"}
    for name in ("_build_J_blocks_subset", "_accumulate_Z_subset_chunked"):
        orig = getattr(solver, name)

        def make(orig=orig, name=name):
            def spy(*a, **kw):
                obs = kw.get("obs_idx")
                seg = a[2] if name == "_build_J_blocks_subset" else a[3]
                calls[tag["which"]].append(
                    [name, int(len(seg)), None if obs is None else int(len(obs))]
                )
                return orig(*a, **kw)

            return spy

        setattr(solver, name, make())

    Z_full, seen_full, _ = fill(solver)
    tag["which"] = "rows"
    Z_rows, seen_rows, _ = fill(solver, rows=rows)
    # the control: every pair-class fill writes nothing, so what survives is
    # exactly the crossing block and the self completions (Amendment 1).
    tag["which"] = "full"
    Z_empty, _se, _ = fill(solver, rows=np.array([], dtype=np.int64))

    supp_seg = _ss
    # basis rows whose support lies wholly inside the requested segments
    on_rows = np.zeros(int(geom["n_segs_total"]) + 1, dtype=bool)
    on_rows[rows] = True
    keep = np.array([bool(on_rows[s].all()) for s in supp_seg])
    kept = np.nonzero(keep)[0]
    other = np.nonzero(~keep)[0]

    d_kept = np.max(np.abs(Z_rows[kept] - Z_full[kept])) if kept.size else 0.0
    scale = np.max(np.abs(Z_full[kept])) if kept.size else 1.0
    out_nonzero = int(np.count_nonzero(np.any(Z_rows[other] != 0.0, axis=1)))
    out_max = float(np.max(np.abs(Z_rows[other]))) if other.size else 0.0
    # non-vacuity: the restriction must REMOVE pair-class content from the
    # non-requested rows, and what is left there must be exactly the control.
    removed = (
        float(np.max(np.abs(Z_rows[other] - Z_full[other]))) if other.size else 0.0
    )
    only_crossing = (
        float(np.max(np.abs(Z_rows[other] - Z_empty[other]))) if other.size else 0.0
    )
    pair_work_kept = (
        float(np.max(np.abs(Z_rows[kept] - Z_empty[kept]))) if kept.size else 0.0
    )
    narrowed = [c for c in calls["rows"] if c[2] is not None and c[2] < c[1]]
    return {
        "calls_full": calls["full"],
        "calls_rows": calls["rows"],
        "narrowed_calls": len(narrowed),
        "removed_from_other_rows": removed,
        "other_rows_minus_control": only_crossing,
        "pair_work_on_kept_rows": pair_work_kept,
        "S_B_not_vacuous": bool(removed > 0.0 and len(narrowed) > 0),
        "S_B_other_rows_are_control": bool(only_crossing == 0.0),
        "radials": n_radials,
        "n_basis": int(supp_seg.shape[0]),
        "n_rows_requested": int(rows.size),
        "kept_basis_rows": int(kept.size),
        "other_basis_rows": int(other.size),
        "max_abs_diff_kept": float(d_kept),
        "rel_diff_kept": float(d_kept / scale),
        "S_B_rows": bool(d_kept / scale <= 1e-12),
        "rows_outside_nonzero": out_nonzero,
        "rows_outside_max_abs": out_max,
        "plan_full": seen_full.get("plan"),
        "plan_rows": seen_rows.get("plan"),
        "S_B_plan": seen_full.get("plan") == seen_rows.get("plan")
        and seen_full.get("q_len_a") == seen_rows.get("q_len_a")
        and seen_full.get("q_len_ax") == seen_rows.get("q_len_ax"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--sb", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--radials", type=int, nargs="+", default=[4, 12])
    args = ap.parse_args()
    import momwire

    recs = {
        "_meta": {
            "momwire": str(Path(momwire.__file__).parent),
            "mode": "sb" if args.sb else "dump",
        },
        "decks": [],
    }
    for n in args.radials:
        rec = sb(n) if args.sb else dump(n)
        recs["decks"].append(rec)
        print(json.dumps(rec), flush=True)
    target = args.out or args.dump
    if target:
        target.write_text(json.dumps(recs, indent=1))
    if args.sb:
        ok = all(
            r["S_B_rows"]
            and r["S_B_plan"]
            and r["S_B_not_vacuous"]
            and r["S_B_other_rows_are_control"]
            for r in recs["decks"]
        )
        print("S-B:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
