"""momwire#1168 U2: `_crossing_path_axis` computes `_testing_paths` and
`_knot_points` once per side, not once per basis row.

`_path_test_rows` used to call both on every invocation, and
`_crossing_path_axis` invoked it once per row in `range(n_basis_total)`:
O(N^2), 1,406 calls at hub_deck(16) x8 (the momwire#1168 audit). The fix
hoists both calls out of the row loop in `_crossing_path_axis` and threads
the results through `_path_test_rows` as optional `paths=`/`knot=`
parameters, left `None` (and so computed as before) for every other caller.

This pins it by counting, the way `test_crossing_main_sandwich_sparse.py`
pins `_sandwich_dense` residency: a regression back to the per-row spelling
fails here rather than surfacing only as a memory drift nobody reads.

The spy is scoped to the SEAM this unit changed -- calls to `_testing_paths`
/ `_knot_points` whose immediate caller is `_path_test_rows` or
`_crossing_path_axis`. Both methods have other, pre-existing callers
(`_assemble_Z_prepare`, `_below_remainder_th_min`, `_assemble_Z_crossing`
itself) that are already O(1) per their own geometry object and were never
part of the audit's per-row count; counting those in would conflate "at
most 2 times per geometry" (the audit's own words for the `_path_test_rows`
hotspot) with call sites the audit never named.
"""

from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402

_SEAM = {"_path_test_rows", "_crossing_path_axis"}


def _seam_spy_counts(build):
    """Solve `build` once through the real `RazorSolver` constructor,
    counting `_testing_paths` / `_knot_points` calls whose immediate caller
    is the U2 seam, and `_path_test_rows` calls unconditionally -- the
    positive control that the seam ran at all."""
    counts = {"_testing_paths": 0, "_knot_points": 0, "_path_test_rows": 0}
    orig_tp = RazorSolver._testing_paths
    orig_kp = RazorSolver._knot_points
    orig_ptr = RazorSolver._path_test_rows

    def tp(self, geom):
        if inspect.currentframe().f_back.f_code.co_name in _SEAM:
            counts["_testing_paths"] += 1
        return orig_tp(self, geom)

    def kp(self, geom):
        if inspect.currentframe().f_back.f_code.co_name in _SEAM:
            counts["_knot_points"] += 1
        return orig_kp(self, geom)

    def ptr(self, geom, rows, **kw):
        counts["_path_test_rows"] += 1
        return orig_ptr(self, geom, rows, **kw)

    RazorSolver._testing_paths = tp
    RazorSolver._knot_points = kp
    RazorSolver._path_test_rows = ptr
    try:
        d = {k: v for k, v in build.items() if k != "junctions"}
        s = RazorSolver(**d, nec5_quadrature=True)
        s.compute_impedance()
    finally:
        RazorSolver._testing_paths = orig_tp
        RazorSolver._knot_points = orig_kp
        RazorSolver._path_test_rows = orig_ptr
    return counts


def test_testing_paths_and_knot_points_computed_once_per_side_not_per_row():
    """A real crossing solve on `crossing_deck(1)`: `_testing_paths` and
    `_knot_points` are each called at most twice through the seam (one call
    per side, ABOVE/BELOW), never once per row. `_path_test_rows` itself is
    UNCHANGED by this unit -- still one call per basis row -- so its count
    being > 1 is the positive control: a deck that never reaches
    `_crossing_path_axis` (never builds a crossing geometry) cannot pass
    this test trivially by calling nothing."""
    counts = _seam_spy_counts(crossing_deck(1))

    assert counts["_path_test_rows"] > 1, (
        "seam not exercised -- _path_test_rows called "
        f"{counts['_path_test_rows']} time(s); crossing_deck(1) must drive "
        "RazorSolver through _crossing_path_axis's per-row loop"
    )
    assert counts["_testing_paths"] <= 2, (
        f"_testing_paths called {counts['_testing_paths']} times through "
        "the seam (momwire#1168 U2: expected at most 2, one per side)"
    )
    assert counts["_knot_points"] <= 2, (
        f"_knot_points called {counts['_knot_points']} times through the "
        "seam (momwire#1168 U2: expected at most 2, one per side)"
    )
