"""momwire#1290: the grid sheet's fixed-width cell sums.

`near_interface_grid_sheet` interpolates each row from its cell's p x p block
of tabulated nodes. With the table's width a runtime value GCC kept the
running inner sums in memory, and the cell sum ran at about a third of the
speed of the same arithmetic at a compile-time width; the AVX2 build now
takes a fixed-width sum for the two widths the families have (six kernels,
four point kernels), in registers, two independent inner sums at a time.
Every entry is the same chain of fused multiply-adds in the same order, so
the gate is uint64 equality against the width-generic loop, which stays
reachable as `generic=True`.

`bary` was split into its node test, divisions and normalisation in the same
change so the divisions vectorise; that moved no bit either (a division is
correctly rounded in any lane). Its gate is the on-node rows below, which
take the early return, and the production check at the PR: the kernel's
output on the three largest real calls of razor invl x32 equal as uint64 to
main's, and full Z equal on invl/buried x4/x8 and x16.

On Windows (MSVC, /fp:fast) the lanes run too since momwire#1371, and every
lanes-vs-walk comparison here takes the derived win32 tolerance instead of bit
equality (`assert_lanes_match`, tests/_lane_gate.py). Linux and macOS keep the
bit gates.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel
from momwire import _near_interface as ni
from momwire._near_interface import k_medium

from test_crossing_serve_524 import SOIL_A, WL7
from _lane_gate import assert_lanes_match

K_P = 2.0 * np.pi / WL7
EPS_T = complex(
    SOIL_A[0], -SOIL_A[1] / (2.0 * np.pi * (299792458.0 / WL7) * 8.8541878128e-12)
)
K_M = k_medium(EPS_T, K_P)

pytestmark = pytest.mark.skipif(
    not (
        ni._HAVE_PLANE_SHEET_ACCEL
        and ni._HAVE_POINT_COLUMNS_ACCEL
        and ni._use_column_accel()
        and hasattr(ni._nia, "grid_sheet_fixed_1290")
    ),
    reason="needs both C++ near-interface extensions at momwire#1290",
)


def _sheet(family, height, sub):
    if height:
        fixed = float(sub[0, 1])
        sheet = ni.PlaneSheet(
            K_P,
            K_M,
            fixed,
            ni._LAM_MULT,
            height=True,
            depth=ni._height_reach(float(-sub[:, 2].min())),
            family=family,
        )
    else:
        sheet = ni.PlaneSheet(K_P, K_M, float(sub[0, 2]), ni._LAM_MULT, family=family)
    sheet.cover(float(sub[:, 0].max()), float((sub[:, 1] - sub[:, 2]).max()))
    return sheet


def _rows(height, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    rho = np.hypot(rng.uniform(0, 6.0, n), 1e-3)
    if height:
        h = 0.7
        return np.ascontiguousarray(
            np.stack([rho, np.full(n, h), -rng.uniform(0.0, 1.5, n)], axis=1)
        )
    zp = -0.15
    z = np.concatenate([rng.uniform(0, 6.0, n // 2), 10 ** rng.uniform(-4, 0, n // 2)])
    return np.ascontiguousarray(np.stack([rho, z, np.full(n, zp)], axis=1))


def _on_node_rows(sheet, sub):
    """Rows whose rho, s or both sit exactly on a cell's Chebyshev node, the
    case `bary` returns early on."""
    rho_edges, s_edges, _co, _v = sheet.arrays()
    out = []
    for a in range(min(3, rho_edges.size - 1)):
        lo, hi = rho_edges[a], rho_edges[a + 1]
        for xn in sheet.x[::3]:
            rho = 0.5 * ((hi - lo) * xn + (lo + hi))
            for b in range(min(3, s_edges.size - 1)):
                slo, shi = s_edges[b], s_edges[b + 1]
                s_node = 0.5 * ((shi - slo) * sheet.x[1] + (slo + shi))
                s_off = 0.5 * (slo + shi) + 0.123 * (shi - slo) / 2
                for s in (s_node, s_off):
                    if sheet.height:
                        out.append((rho, sheet.zp, sheet.zp - s))
                    else:
                        out.append((rho, sheet.zp + s, sheet.zp))
    rows = np.asarray(out, dtype=np.float64)
    keep = rows[:, 2] <= 0.0 if sheet.height else rows[:, 1] >= 0.0
    return np.ascontiguousarray(np.concatenate([sub, rows[keep]]))


def _interp(sheet, sub, generic):
    rho_edges, s_edges, cell_off, vals = sheet.arrays()
    out = np.empty((sub.shape[0], sheet.width), dtype=np.complex128)
    ni._nia.near_interface_grid_sheet(
        sub, np.arange(sub.shape[0]), sheet.zp, sheet.height, rho_edges,
        s_edges, cell_off, sheet.x, sheet.bw, vals, out,
        ni._physical_cpu_count(), sheet._rpow_arr, generic=generic,
    )  # fmt: skip
    return out


def test_the_avx2_build_has_the_fixed_route():
    """The gate below compares two routes only where the fixed one is built;
    on the AVX2 variant it must be, or the comparison is the generic loop
    against itself."""
    if _accel.VARIANT == "avx2":
        assert ni._nia.grid_sheet_fixed_1290 is True


@pytest.mark.parametrize("family", ["six", "point"])
@pytest.mark.parametrize("height", [False, True], ids=["plane", "height"])
def test_the_fixed_width_sum_is_the_generic_loop_to_the_bit(family, height):
    sub = _rows(height)
    sheet = _sheet(family, height, sub)
    sub = _on_node_rows(sheet, sub)
    fast = _interp(sheet, sub, generic=False)
    ref = _interp(sheet, sub, generic=True)
    assert np.isfinite(ref).all()
    assert_lanes_match(fast, ref)
    # The production entry is the default route.
    via = np.empty_like(fast)
    sheet.interpolate(sub, np.arange(sub.shape[0]), via)
    assert np.array_equal(via.view(np.uint64), fast.view(np.uint64))


def test_a_row_on_a_node_reads_the_node():
    """On a node in both directions `bary` returns unit weights, and every
    fused step then adds an exact zero or the node itself: the row is the
    tabulated value times R^-pw, exactly."""
    sub = _rows(False, n=8)
    sheet = _sheet("six", False, sub)
    rho_edges, s_edges, cell_off, vals = sheet.arrays()
    p = sheet.x.size
    i, j = 4, 7
    rho = 0.5 * (
        (rho_edges[1] - rho_edges[0]) * sheet.x[i] + (rho_edges[0] + rho_edges[1])
    )
    s = 0.5 * ((s_edges[1] - s_edges[0]) * sheet.x[j] + (s_edges[0] + s_edges[1]))
    row = np.array([[rho, sheet.zp + s, sheet.zp]])
    # The weights are exact only if the node test fires on the round trip.
    xr = (2.0 * rho - (rho_edges[0] + rho_edges[1])) / (rho_edges[1] - rho_edges[0])
    xs = (2.0 * s - (s_edges[0] + s_edges[1])) / (s_edges[1] - s_edges[0])
    if not (abs(xr - sheet.x[i]) < 1e-15 and abs(xs - sheet.x[j]) < 1e-15):
        pytest.skip("the node does not round-trip through the cell map")
    node = vals[cell_off[0, 0] + i * p + j]
    inv_r = 1.0 / np.hypot(rho, s)
    pw = np.asarray(ni.KEY_RPOW)
    f = np.where(pw == 1, inv_r, inv_r * inv_r)
    want = np.empty(node.shape, dtype=np.complex128)
    want.real = node.real * f
    want.imag = node.imag * f
    for generic in (False, True):
        got = _interp(sheet, row, generic)[0]
        assert np.array_equal(got.view(np.uint64), want.view(np.uint64))
