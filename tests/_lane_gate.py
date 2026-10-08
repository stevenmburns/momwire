"""The walk-vs-lanes gate, per platform (momwire#1371).

A lane kernel (`src/momwire/_lanes.h`) is gated against the scalar walk it
replaces. Where the build is GCC or clang (Linux, macOS) the gate is BIT
EQUALITY: those builds are -ffp-contract=off, every lane operation is one IEEE
operation in the walk's order, and the lanes' `fmadd` is the walk's
`mw_fma::fma`. Nothing here changes that.

On Windows it cannot be. The MSVC build is /fp:fast, which contracts and
reassociates the scalar walk on its own terms, and its `mw_fma::fma` is the
unfused a*b + c while the lanes' `fmadd` is fused. Since #1371 the lanes run
there too, so the gate on win32 is a DERIVED tolerance (Steve, 2026-10-03:
Windows needs no bit-compatibility; 2026-10-07: the fused kernels run there,
gated by it):

    max |lanes - walk|  <=  LANE_RTOL_WIN32 * max |walk|

over each compared array, NaN in the same places and compared nowhere else.

`MOMWIRE_LANE_GATE_LOG=<path>` appends one JSON line per win32 gate call (the
test, the measured ratio, the array size), and the suite's terminal summary
prints the maximum per test. That is how LANE_RTOL_WIN32 was derived and how
to re-derive it; see the value's comment.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

WIN32 = sys.platform == "win32"

# The variants that compile the lane layer's AVX2 backend: `_avx2`, and
# `_avx512` (momwire#1370), which defines __AVX2__/__FMA__ too and keeps the
# 4-wide backend because no build sets MW_LANES_ENABLE_AVX512. A gate that
# asserts "the lanes were built" on the AVX2 variant asserts it on both, or
# under `_avx512` it compares the walk with itself and passes vacuously.
VECTOR_VARIANTS = ("avx2", "avx512")

# DERIVED, not chosen (momwire#1371). Measured on windows-latest (MSVC 14.51,
# /O2 /arch:AVX2 /fp:fast, cp312 wheel; wheels.yml run 37651444139) with the
# log on: 501 win32 comparisons over the twelve lane modules. Worst per module:
#
#   1.16e-12  test_hmatrix_ek_block_1362    (an EK block vs the reduced one)
#   3.8e-14   test_offedge_lanes_1290       (a free-space solve's currents)
#   1.5e-14   test_swept_ek_lanes_ladder_1362
#   1.1e-14   test_ek_pair_order_ladder_1362, test_offedge_ek_lanes_1362
#   2.5e-15   test_weighted_windowed_lanes
#   1.0e-15   test_below_lanes_1290, test_somm_proj_lanes_1290,
#             test_windowed_lanes_1290
#   0         test_field_galerkin_lanes_1290, test_pair_extents_lanes_1290,
#             test_sheet_fixed_width_1290 (bit-equal even there)
#
# So ~5000 eps at worst: an H-matrix block's entries are sums of a few
# hundred quadrature terms each, normed by the block's largest entry, where
# /fp:fast's reassociation and the unfused walk each leave a few ulp per term.
# The gate is the worst case with 8.6x headroom, 1e-11: four orders above the
# typical kernel's 1e-15, and far below anything physical (the derived solver
# tolerances here are 1e-9 and up). The measurement is deterministic: two
# Windows runs gave the same lanes-vs-walk Z ratios to the last digit.
LANE_RTOL_WIN32 = 1e-11
# The "moved" threshold for the EK-factor controls on win32: a pair whose
# relative change exceeds it moved. Same run: the noise between the EK and
# reduced kernels on pairs that must NOT move is at most 9.3e-16, and the EK
# factor's smallest effect on a pair that must is 2.6e-8. 1e-10 sits five
# decades above the one and two and a half below the other.
MOVED_RTOL_WIN32 = 1e-10

_LOG_ENV = "MOMWIRE_LANE_GATE_LOG"


def bits(a) -> np.ndarray:
    """The array's float64 words as uint64: equality of these is bit equality."""
    return np.ascontiguousarray(a).view(np.uint64)


def rel_diff(got, want) -> float:
    """max |got - want| / max |want| over the finite entries; inf when the
    shapes differ or a NaN (or an infinity) is not matched exactly."""
    g = np.asarray(got)
    w = np.asarray(want)
    if g.shape != w.shape:
        return float("inf")
    g_nan = np.isnan(g)
    w_nan = np.isnan(w)
    if not np.array_equal(g_nan, w_nan):
        return float("inf")
    g = g[~w_nan]
    w = w[~w_nan]
    if not np.array_equal(np.isinf(g), np.isinf(w)) or not np.array_equal(
        g[np.isinf(w)], w[np.isinf(w)]
    ):
        return float("inf")
    fin = np.isfinite(w)
    g = g[fin]
    w = w[fin]
    if g.size == 0:
        return 0.0
    d = float(np.max(np.abs(g - w)))
    if d == 0.0:
        return 0.0
    scale = float(np.max(np.abs(w)))
    return d / scale if scale > 0.0 else float("inf")


def _record(rel: float, size: int, **extra) -> None:
    path = os.environ.get(_LOG_ENV)
    if not path:
        return
    test = os.environ.get("PYTEST_CURRENT_TEST", "?").rsplit(" (", 1)[0]
    row = {"test": test, "rel": rel, "size": int(size), **extra}
    # One file per process: appends from several xdist workers to one file
    # interleave on Windows.
    with open(f"{path}.{os.getpid()}", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def lanes_match(got, want) -> bool:
    """Bit equality off Windows; the derived tolerance on it."""
    if not WIN32:
        return np.array_equal(bits(got), bits(want))
    rel = rel_diff(got, want)
    _record(rel, np.asarray(want).size)
    return rel <= LANE_RTOL_WIN32


def lanes_match_values(got, want) -> bool:
    """A tuple of scalars (a kernel's side outputs): `==` off Windows, the
    derived tolerance on it."""
    if not WIN32:
        return tuple(got) == tuple(want)
    return lanes_match(np.asarray(got, dtype=float), np.asarray(want, dtype=float))


def moved_pairs(a, b, axis, expected=None) -> np.ndarray:
    """Which pairs differ between two kernels' moment arrays, reduced over
    `axis`. Off Windows: any bit differs. On win32 the two kernels' own
    arithmetic differs at the rounding level (/fp:fast), so a pair has moved
    when its relative difference (max over `axis` of |a - b| over max |b|)
    exceeds MOVED_RTOL_WIN32. With `expected` (the pairs that should move)
    the log records the largest ratio among the rest (the noise) and the
    smallest among the expected (the effect): the threshold sits between."""
    a = np.asarray(a)
    b = np.asarray(b)
    if not WIN32:
        return np.any(a != b, axis=axis)
    diff = np.max(np.abs(a - b), axis=axis)
    scale = np.max(np.abs(b), axis=axis)
    safe = np.where(scale > 0, scale, 1.0)
    rel = np.where(scale > 0, diff / safe, np.where(diff > 0, np.inf, 0.0))
    if expected is not None:
        e = np.asarray(expected, dtype=bool)
        noise = float(rel[~e].max()) if (~e).any() else 0.0
        effect = float(rel[e].min()) if e.any() else float("inf")
        _record(noise, rel.size, kind="moved", effect_min=effect)
    return rel > MOVED_RTOL_WIN32


def assert_lanes_match(got, want, what: str = "") -> None:
    if lanes_match(got, want):
        return
    rel = rel_diff(got, want)
    gate = f"<= {LANE_RTOL_WIN32:.1e} (win32)" if WIN32 else "bit equality"
    raise AssertionError(f"{what}: lanes vs walk {rel:.3e}, gate {gate}")


def report(terminalreporter) -> None:
    """Print the logged gate calls' maxima, worst first (the suite's
    terminal summary calls this; it prints nothing without the log)."""
    path = os.environ.get(_LOG_ENV)
    if not path:
        return
    files = glob.glob(glob.escape(path) + ".*")
    if not files:
        return
    worst: dict[str, float] = {}
    moved: dict[str, tuple[float, float]] = {}
    n = 0
    for name in files:
        with open(name, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                n += 1
                t = row["test"]
                if row.get("kind") == "moved":
                    noise, eff = moved.get(t, (0.0, float("inf")))
                    moved[t] = (max(noise, row["rel"]), min(eff, row["effect_min"]))
                else:
                    worst[t] = max(worst.get(t, 0.0), row["rel"])
    tr = terminalreporter
    tr.write_sep("-", f"lane gate (momwire#1371): {n} win32 calls")
    for test, rel in sorted(worst.items(), key=lambda kv: -kv[1]):
        tr.write_line(f"{rel:.3e}  {test}")
    tr.write_line(
        f"MAX {max(worst.values(), default=0.0):.3e}  gate {LANE_RTOL_WIN32:.1e}"
    )
    for test, (noise, eff) in sorted(moved.items()):
        tr.write_line(f"moved: noise {noise:.3e}  effect min {eff:.3e}  {test}")
