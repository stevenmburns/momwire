"""momwire#1109 / #1029 phase 2, probe 0: WHERE the 8.4 GB at 150 radials is.

  PYTHONPATH=<momwire src>:<antennaknobs src> \\
      python scratch/1029-block-circulant/p2_memory_probe.py --radials 150 --mode route

One process, one solve (the cold pass; peak RSS is a high-water mark and the
question is attribution, not the warm second). Every phase of the buried fill
is wrapped: VmRSS / VmHWM at entry and exit, and tracemalloc's peak inside the
call with the peak reset at entry, so each record says how much the phase
ADDED on top of what was live when it started. Array-returning phases log the
result's shape and bytes. `_row_weights` and `_tables` are counted rather than
listed: calls, largest, and total bytes.
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as _sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feasibility import GROUND  # noqa: E402

RECORDS = []
COUNTS = {}


def _status():
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            out[line.split(":")[0]] = round(int(line.split()[1]) / 1024.0, 1)
    return out


def _bytes_of(x):
    """Bytes an array or a scipy sparse matrix holds (momwire#1109 put the
    axis samples and the row weights in CSR, and a CSR's cost is its three
    buffers, not its shape)."""
    if isinstance(x, np.ndarray):
        return int(x.nbytes)
    if _sp.issparse(x):
        return int(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes)
    return 0


def _nbytes(x):
    if isinstance(x, np.ndarray) or _sp.issparse(x):
        return {
            "shape": list(x.shape),
            "mb": round(_bytes_of(x) / 2**20, 3),
            "nnz": int(x.nnz) if _sp.issparse(x) else None,
        }
    if isinstance(x, tuple):
        return [_nbytes(e) for e in x if isinstance(e, np.ndarray) or _sp.issparse(e)]
    if isinstance(x, dict):
        big = {
            k: _nbytes(v)
            for k, v in x.items()
            if (isinstance(v, np.ndarray) or _sp.issparse(v)) and _bytes_of(v) >= 2**16
        }
        return big or None
    return None


def phase(name, fn):
    @functools.wraps(fn)
    def wrapped(*a, **kw):
        s0 = _status()
        cur0, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        dt = time.perf_counter() - t0
        _cur1, peak = tracemalloc.get_traced_memory()
        s1 = _status()
        rec = {
            "phase": name,
            "s": round(dt, 3),
            "rss_in_mb": s0["VmRSS"],
            "rss_out_mb": s1["VmRSS"],
            "hwm_out_mb": s1["VmHWM"],
            "traced_in_mb": round(cur0 / 2**20, 1),
            "traced_peak_in_call_mb": round(peak / 2**20, 1),
            "added_at_peak_mb": round((peak - cur0) / 2**20, 1),
            "result": _nbytes(out),
        }
        RECORDS.append(rec)
        print(json.dumps(rec), flush=True)
        return out

    return wrapped


def counted(name, fn, size_of):
    c = COUNTS.setdefault(
        name, {"calls": 0, "max_mb": 0.0, "total_mb": 0.0, "max_shape": None}
    )

    @functools.wraps(fn)
    def wrapped(*a, **kw):
        out = fn(*a, **kw)
        mb, shape = size_of(out)
        c["calls"] += 1
        c["total_mb"] += mb
        if mb > c["max_mb"]:
            c["max_mb"], c["max_shape"] = mb, shape
        return out

    return wrapped


def _rw_size(out):
    mb = sum(_bytes_of(x) for x in out) / 2**20
    return mb, list(out[0].shape)


def _tab_size(out):
    mb = sum(v.nbytes for v in out.values() if isinstance(v, np.ndarray)) / 2**20
    first = next(v for v in out.values() if isinstance(v, np.ndarray))
    return mb, list(first.shape)


BLOCKS = []


def _block_stats(fn):
    @functools.wraps(fn)
    def wrapped(tree_a, tree_b, eta):
        far, near = fn(tree_a, tree_b, eta)
        sz = lambda pairs: sorted(  # noqa: E731
            ((int(cs.indices.size), int(ct.indices.size)) for cs, ct in pairs),
            key=lambda t: -t[0] * t[1],
        )
        n_sz, f_sz = sz(near), sz(far)
        rec = {
            "block_tree": True,
            "n_near": len(near),
            "n_far": len(far),
            "near_pairs_segs": int(sum(a * b for a, b in n_sz)),
            "far_pairs_segs": int(sum(a * b for a, b in f_sz)),
            "near_top5_segs": n_sz[:5],
            "far_top5_segs": f_sz[:5],
        }
        BLOCKS.append(rec)
        print(json.dumps(rec), flush=True)
        return far, near

    return wrapped


def instrument():
    from momwire import _aca, _below_interface, _crossing_fill, bspline

    _aca.build_block_tree = _block_stats(_aca.build_block_tree)
    cf = _crossing_fill
    cf._row_weights = counted("_row_weights", cf._row_weights, _rw_size)
    cf._tables = counted("_tables", cf._tables, _tab_size)
    for nm in (
        "axis_data",
        "_main_split",
        "_ends_and_corner",
        "cross_complete_block_split",
        "self_completions",
    ):
        setattr(cf, nm, phase(f"crossing.{nm}", getattr(cf, nm)))
    _below_interface.compute_Z_operator_buried = phase(
        "buried.compute_Z_operator_buried", _below_interface.compute_Z_operator_buried
    )
    S = bspline.BSplineSolver
    for nm in (
        "_build_geometry",
        "_build_basis_polynomials",
        "_field_galerkin_block",
        "_accumulate_Z_subset_chunked",
        "_image_Z_weighted",
        "_assemble_Z",
        "_solve_with_kcl",
        "_rotational_solve",
    ):
        setattr(S, nm, phase(f"solver.{nm}", getattr(S, nm)))


def build(n_radials, route):
    from antennaknobs.designs.verticals.buried_radial_vertical import Builder
    from antennaknobs.engines.momwire import MomwireEngine

    from momwire import BSplineSolver

    b = Builder()
    b.n_radials = n_radials
    extra = {"rotational_symmetry": True} if route else {}
    eng = MomwireEngine(
        b, solver=BSplineSolver, solver_kwargs={"degree": 2, **extra}, ground=GROUND
    )
    return eng._make_solver(wavelength=eng._wavelength_for(float(b.freq)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, default=48)
    ap.add_argument("--mode", choices=("dense", "route"), default="route")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    tracemalloc.start(1)
    instrument()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = build(args.radials, args.mode == "route")
        t0 = time.perf_counter()
        res = s.compute_impedance()
        dt = time.perf_counter() - t0
    z = res[0] if isinstance(res, tuple) else res
    z = complex(np.atleast_1d(z).ravel()[0])
    summary = {
        "summary": True,
        "radials": args.radials,
        "mode": args.mode,
        "hostname": platform.node(),
        "t_cold_s": round(dt, 2),
        "z_in": [z.real, z.imag],
        "n_basis": int(s._last_n_basis) if hasattr(s, "_last_n_basis") else None,
        "peak": _status(),
        "counts": COUNTS,
        "block_trees": BLOCKS,
    }
    print(json.dumps(summary), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            for r in RECORDS:
                fh.write(json.dumps(r) + "\n")
            fh.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
