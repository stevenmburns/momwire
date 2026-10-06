"""Regenerate the grazing-band interpolation fixtures (momwire#1359).

    python tests/fixtures/grazing_1359/regenerate.py [floor lo mid far]
    python tests/fixtures/grazing_1359/regenerate.py --verify-full floor

Run it when `test_grazing_interp_units_1359.py::test_the_fixture_is_current`
says the fixture is stale, i.e. when the grid code or constants the numbers
depend on have changed. It prints how far every stored number moved against
the fixture it replaces, so the diff a reviewer sees is a measurement and not
just a binary blob: a pure refactor reads 0, a numerical change reads its
size. Review that before committing.

Per sweep, for ONE deck (the sweep's measured worst on the six-deck sweep):

  1. build the production lattice with nothing integrated, and RECORD which
     nodes the production `_interp` reads to answer the sweep's queries;
  2. fill exactly those R1 rows with the production `_fill_region`, cold;
  3. interpolate from those nodes alone (the unit test's own path) and take
     the direct reference at the queries;
  4. check the sweep's bar, and write `<sweep>.npz`.

`--verify-full` also fills the touched regions WHOLE, the way the nightly
sweep does, and asserts that interpolating from the full grid gives the same
bits as interpolating from the stored nodes alone -- i.e. that the unit test
measures the same number as the integration test. Minutes, not seconds.

Cost without it: about two minutes, at one thread (the script pins it, so the
fill sample is sized for one xdist worker).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# One OpenMP thread, as on an xdist worker, so the fill sample's measured cost
# is the cost the default lane will pay. The VALUES do not depend on it: each
# node's contour is independent of every other. Set before momwire loads.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import grazing_fixture_1359 as gf  # noqa: E402
from momwire import _sommerfeld_below as below  # noqa: E402
from momwire._sommerfeld import _SURF_KEYS  # noqa: E402

# The deck each fixture is taken on: the sweep's WORST over soils A/B/C x
# 7/21 MHz, measured on the full sweep (momwire#1359, gesher, 2026-10-05):
#
#   sweep  worst deck   worst      runner-up         bar
#   floor  B, 7 MHz     7.99e-10   A 7 MHz 6.86e-10  3.9e-7
#   lo     C, 7 MHz     9.79e-10   B 7 MHz 7.94e-10  9.8e-8
#   mid    A, 21 MHz    1.77e-07   C 21 MHz 1.39e-7  1.8e-5
#   far    A, 21 MHz    5.05e-05   C 21 MHz 2.25e-5  1.5e-4
#
# (C, 7 MHz is the low band's worst, not the floor band's.) Re-measure with
# the nightly sweep before moving one: its failure message names the deck.
DECKS = {
    "floor": ("B", 7e6),
    "lo": ("C", 7e6),
    "mid": ("A", 21e6),
    "far": ("A", 21e6),
}

# The fill gate recomputes a few stored rows cold, in the default lane, so it
# is sized by COST: each stored row's restricted fill is timed here, each
# region offers its cheapest row, and regions are taken cheapest first while
# the total stays under this many seconds (at least one). The choice and its
# measured cost go in the header, so the gate's cost is fixed by the fixture.
FILL_SAMPLE_BUDGET_S = 1.5


def build(sweep, verify_full=False):
    soil, f = DECKS[sweep]
    spec = gf.SWEEPS[sweep]
    eps_t, k2, om, lam_m = gf._deck(soil, f)
    g = gf.lattice_grid(soil, f)
    queries = spec["queries"](g, lam_m)
    nodes = gf.stencil_nodes(g, queries)

    h = below.Health()
    g._health = h
    t0 = time.perf_counter()
    cheapest = {}
    for idx, ij in nodes.items():
        for i in sorted({i for i, _ in ij}):
            t = time.perf_counter()
            gf.fill_rows(g, idx, [i])
            dt = time.perf_counter() - t
            if idx not in cheapest or dt < cheapest[idx][1]:
                cheapest[idx] = (i, dt)
    t_fill = time.perf_counter() - t0
    fill_sample, cost = {}, 0.0
    for idx, (i, dt) in sorted(cheapest.items(), key=lambda kv: kv[1][1]):
        if fill_sample and cost + dt > FILL_SAMPLE_BUDGET_S:
            break
        fill_sample[str(idx)] = [i]
        cost += dt

    reg_ids, ii, jj, rr, tt, vals = [], [], [], [], [], []
    for idx, ij in sorted(nodes.items()):
        reg = g._regions[idx]
        for i, j in ij:
            reg_ids.append(idx)
            ii.append(i)
            jj.append(j)
            rr.append(reg["r_nodes"][i])
            tt.append(reg["th_nodes"][j])
            vals.append(reg["vals"][:, i, j])
    vals = np.stack(vals, axis=1)
    assert np.isfinite(vals).all(), "a stored node is not finite"

    q_r1 = np.concatenate([np.asarray(r, float) for r, _ in queries])
    q_th = np.concatenate([np.asarray(t, float) for _, t in queries])
    t0 = time.perf_counter()
    ref = below.iv_surfaces_direct_below(eps_t, k2, q_r1, q_th, rtol=gf.RTOL, omega=om)
    t_ref = time.perf_counter() - t0
    ref = np.stack([np.asarray(ref[k], complex) for k in _SURF_KEYS])

    header = gf.provenance(soil, f)
    header.update(
        {
            "sweep": sweep,
            "gate": spec["gate"],
            "metric": spec["metric"],
            "n_queries": int(q_r1.size),
            "n_nodes": int(vals.shape[1]),
            "fill_sample": fill_sample,
            "fill_sample_cost_s": round(cost, 3),
            "max_tail_panels": int(h.max_tail_panels),
            "nonconvergent": int(h.nonconvergent),
        }
    )
    arrays = {
        "query_r1": q_r1,
        "query_th": q_th,
        "ref": ref,
        "node_region": np.asarray(reg_ids, dtype=np.int64),
        "node_i": np.asarray(ii, dtype=np.int64),
        "node_j": np.asarray(jj, dtype=np.int64),
        "node_r": np.asarray(rr, float),
        "node_th": np.asarray(tt, float),
        "node_vals": vals,
    }

    # The unit test's own path: interpolate from the stored nodes alone.
    lg = gf.loaded_grid({**arrays, "header": header})
    got = lg.eval(q_r1, q_th)
    interp = np.stack([np.asarray(got[k], complex) for k in _SURF_KEYS])
    assert np.isfinite(interp).all(), "the stencil read a node it was not given"
    arrays["interp"] = interp
    err = gf.rel_error(sweep, interp, ref)
    s, n = np.unravel_index(np.argmax(err), err.shape)
    header["measured_worst"] = float(err[s, n])
    header["measured_worst_at"] = [
        float(q_r1[n] / lam_m),
        float(np.degrees(q_th[n])),
        _SURF_KEYS[s],
    ]
    assert err.max() < gf.bar(sweep), f"{sweep}: {err.max():.3e} over the bar"

    if verify_full:
        full = gf.lattice_grid(soil, f)
        t0 = time.perf_counter()
        for idx in nodes:
            full._fill_region(idx)
        t_full = time.perf_counter() - t0
        fg = full.eval(q_r1, q_th)
        fg = np.stack([np.asarray(fg[k], complex) for k in _SURF_KEYS])
        same = np.array_equal(fg, interp)
        print(
            f"  verify-full: whole-region fill {t_full:.1f} s; full-grid "
            f"interpolation {'IDENTICAL to' if same else 'DIFFERS from'} the "
            f"stored-node one (max |d| {np.abs(fg - interp).max():.3e})"
        )
        assert same

    return header, arrays, t_fill, t_ref


def moved(old, arrays):
    """Max relative movement of every stored number against the old fixture."""
    out = {}
    for k in ("node_vals", "interp", "ref"):
        a, b = arrays[k], old.get(k)
        if b is None or a.shape != b.shape:
            out[k] = "shape changed"
            continue
        out[k] = float(np.max(np.abs(a - b) / gf.node_scale(b)[None, :]))
    if not np.array_equal(arrays["node_region"], old["node_region"]) or not (
        np.array_equal(arrays["node_i"], old["node_i"])
        and np.array_equal(arrays["node_j"], old["node_j"])
    ):
        out["nodes"] = "the stencil now reads a different node set"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sweeps", nargs="*", default=list(gf.SWEEPS))
    ap.add_argument("--verify-full", action="store_true")
    args = ap.parse_args(argv)
    gf.FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for sweep in args.sweeps:
        print(f"{sweep}: deck {DECKS[sweep]}")
        header, arrays, t_fill, t_ref = build(sweep, args.verify_full)
        old = gf.load(sweep) if gf.path(sweep).exists() else None
        gf.save(sweep, header, arrays)
        print(
            f"  {header['n_nodes']} nodes filled in {t_fill:.1f} s, "
            f"{header['n_queries']} direct references in {t_ref:.1f} s; "
            f"worst {header['measured_worst']:.3e} at {header['measured_worst_at']} "
            f"(bar {gf.bar(sweep):.1e}); {gf.path(sweep).stat().st_size / 1024:.1f} KB"
        )
        if old is not None:
            print(f"  moved vs the previous fixture: {moved(old, arrays)}")


if __name__ == "__main__":
    main()
