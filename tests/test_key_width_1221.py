"""momwire#1221: the near-interface table layer takes its width from `KEYS`.

The array memo (#1168 U1) and the sheets (#1173 Design E, the strip grid of
phase 2 — plane and height sheets alike) were written
for exactly six kernels. #1220 stage 2 adds point-observer surfaces, so the
width is now data: `N_KEYS = len(KEYS)` everywhere a row of kernels is stored
or moved, and each column's power of R (`KEY_RPOW`) is handed to the C++
sheet rather than hard-coded there.

What these gates hold:

  * the widening moved nothing — a seven-column sheet whose extra column is a
    copy of column 0 answers the original six BIT-identically and the copy
    equal to its source, so the width is independent per column;
  * a memo holds whatever width `N_KEYS` says, and hands back the very floats;
  * a mismatched width is refused by name, not read out of bounds.

The six-wide route's bits against main were measured by script at the PR
(a plane and a height sheet's own interpolation, `designed_tables` through a
memo, and razor invl_deck(4/16, leaning) and bspline invl_deck(4) production
fills with the sheets on and off): bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _near_interface as ni
from momwire._near_interface import k_medium

from test_crossing_serve_524 import SOIL_A, WL7

K_P = 2.0 * np.pi / WL7
EPS_T = complex(
    SOIL_A[0], -SOIL_A[1] / (2.0 * np.pi * (299792458.0 / WL7) * 8.8541878128e-12)
)
K_M = k_medium(EPS_T, K_P)

needs_sheet = pytest.mark.skipif(
    not (ni._HAVE_PLANE_SHEET_ACCEL and ni._use_column_accel()),
    reason="the plane sheet needs the C++ near-interface extension",
)


def _rows(zp, n=120, rmax=4.0, seed=5):
    rng = np.random.default_rng(seed)
    rho = np.hypot(rng.uniform(0, rmax, n), 1e-3)
    z = np.concatenate([rng.uniform(0, rmax, n // 2), 10 ** rng.uniform(-4, 0, n // 2)])
    return np.ascontiguousarray(np.stack([rho, z, np.full_like(z, zp)], axis=1))


def test_the_width_is_the_key_tuple():
    assert ni.N_KEYS == len(ni.KEYS) == len(ni.KEY_RPOW)
    assert ni.KEY_RPOW == (1, 1, 1, 2, 2, 2)
    m = ni.TripleMemo()
    hit, block = m.lookup(np.zeros((3, 3)))
    assert block.shape == (3, ni.N_KEYS) and not hit.any()


def _sheet(zp, sub, height=False):
    kw = dict(height=True, depth=float(-sub[:, 2].min()) + 0.1) if height else {}
    sheet = ni.PlaneSheet(K_P, K_M, zp, ni._LAM_MULT, **kw)
    sheet.cover(float(sub[:, 0].max()), float((sub[:, 1] - sub[:, 2]).max()))
    return sheet


def _grid(sheet, sub, vals, out, rpow):
    rho_edges, s_edges, cell_off, _v = sheet.arrays()
    ni._nia.near_interface_grid_sheet(
        sub, np.arange(sub.shape[0]), sheet.zp, sheet.height, rho_edges,
        s_edges, cell_off, sheet.x, sheet.bw, vals, out,
        ni._physical_cpu_count(), rpow,
    )  # fmt: skip


def _height_rows(h, n=120, seed=7):
    rng = np.random.default_rng(seed)
    rho = np.hypot(rng.uniform(0, 4.0, n), 1e-3)
    return np.ascontiguousarray(
        np.stack([rho, np.full(n, h), -rng.uniform(0.0, 1.5, n)], axis=1)
    )


@needs_sheet
@pytest.mark.parametrize("kind", ["plane", "height"])
def test_a_wider_sheet_answers_the_six_columns_to_the_bit(kind):
    if kind == "plane":
        zp = -0.15
        sub = _rows(zp)
        sheet = _sheet(zp, sub)
    else:
        sub = _height_rows(0.7)
        sheet = _sheet(0.7, sub, height=True)
    six = np.empty((sub.shape[0], ni.N_KEYS), dtype=np.complex128)
    sheet.interpolate(sub, np.arange(sub.shape[0]), six)

    vals = sheet.arrays()[3]
    # A seventh column: a copy of column 3 (a derivative, R^2).
    wide_vals = np.ascontiguousarray(np.concatenate([vals, vals[:, 3:4]], axis=1))
    rpow = np.asarray(ni.KEY_RPOW + (2,), dtype=np.int64)
    seven = np.empty((sub.shape[0], ni.N_KEYS + 1), dtype=np.complex128)
    _grid(sheet, sub, wide_vals, seven, rpow)
    assert np.array_equal(seven[:, : ni.N_KEYS], six)
    assert np.array_equal(seven[:, ni.N_KEYS], six[:, 3])


@needs_sheet
def test_a_width_mismatch_is_refused():
    zp = -0.15
    sub = _rows(zp, n=8)
    sheet = _sheet(zp, sub)
    vals = sheet.arrays()[3]
    out = np.empty((sub.shape[0], ni.N_KEYS), dtype=np.complex128)
    short = np.asarray(ni.KEY_RPOW[:-1], dtype=np.int64)
    with pytest.raises(ValueError, match="len\\(rpow\\)"):
        _grid(sheet, sub, vals, out, short)


def test_a_memo_stores_the_width_it_is_given(monkeypatch):
    """`TripleMemo` sizes from `N_KEYS` at call time, so a wider key tuple
    needs no edit to it: the stored floats come back exactly."""
    monkeypatch.setattr(ni, "N_KEYS", ni.N_KEYS + 3)
    rng = np.random.default_rng(0)
    rows = rng.uniform(size=(5, 3))
    vals = rng.normal(size=(5, ni.N_KEYS)) + 1j * rng.normal(size=(5, ni.N_KEYS))
    m = ni.TripleMemo()
    m.insert(rows, vals)
    hit, block = m.lookup(rows[::-1])
    assert hit.all()
    assert np.array_equal(block, vals[::-1])
