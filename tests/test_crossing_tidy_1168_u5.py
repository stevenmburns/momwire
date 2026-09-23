"""momwire#1168 U5 — the bit-identical tidy of the buried crossing fill.

What the tidy renamed or moved, pinned where a rename could silently disarm a
switch or a shared helper could silently stop being shared.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from momwire import _crossing_fill as CF
from momwire.bspline import BSplineSolver
from test_crossing_serve_524 import crossing_deck


def _flag_in_fresh_process(env_value):
    env = {k: v for k, v in os.environ.items() if k != "MOMWIRE_CROSSING_FORCE_DENSE"}
    if env_value is not None:
        env["MOMWIRE_CROSSING_FORCE_DENSE"] = env_value
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from momwire import _crossing_fill as CF;"
            "print(CF._WHOLE_AXIS_NO_ACA, hasattr(CF, '_FORCE_DENSE'))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


@pytest.mark.parametrize("value, expect", [(None, "False"), ("1", "True")])
def test_the_legacy_env_var_still_sets_the_whole_axis_switch(value, expect):
    """The switch was renamed (`_FORCE_DENSE` -> `_WHOLE_AXIS_NO_ACA`); its
    environment variable kept the legacy name, so a bisect note that sets
    `MOMWIRE_CROSSING_FORCE_DENSE=1` still flips it, and the old attribute
    is gone rather than left as a dead twin."""
    assert _flag_in_fresh_process(value) == [expect, "False"]


def _crossing_axes():
    s = BSplineSolver(**crossing_deck(1))
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    ctx = s._crossing_context(geom, supp_seg, polys)
    so = np.asarray(geom["seg_offsets"])
    b_idx, a_idx = np.arange(so[0], so[1]), np.arange(so[1], so[2])
    return ctx, a_idx, b_idx, CF.axis_data(ctx, a_idx), CF.axis_data(ctx, b_idx)


def test_the_whole_axis_switch_routes_every_split_entry(monkeypatch):
    """With the switch on, the three split entries answer through the
    whole-axis fill and never partition: `_main_split` is not entered."""
    ctx, a_idx, b_idx, A, B = _crossing_axes()
    split_calls = []
    real_split = CF._main_split

    def spy(*a, **k):
        split_calls.append(1)
        return real_split(*a, **k)

    monkeypatch.setattr(CF, "_main_split", spy)
    monkeypatch.setattr(CF, "_WHOLE_AXIS_NO_ACA", True)
    whole = CF.cross_complete_block(ctx, A, B)
    assert np.array_equal(CF.cross_complete_block_split(ctx, a_idx, b_idx, A, B), whole)
    assert np.array_equal(
        CF.cross_complete_block_reversed_split(ctx, b_idx, a_idx, B, A),
        CF.cross_complete_block_reversed(ctx, B, A),
    )
    assert split_calls == []
    monkeypatch.setattr(CF, "_WHOLE_AXIS_NO_ACA", False)
    CF.cross_complete_block_split(ctx, a_idx, b_idx, A, B)
    assert split_calls == [1]


# ---------------------------------------------------------------------------
# The reversed ends on the forward helpers (U5 item 3).
#
# No served deck reaches a wrong-side end: the crossing scope admits exactly
# one above member and the rest below, and every U5 gate deck came back
# bit-identical through the new spelling, so none of them tripped it. The
# input is therefore built at the function level — a real crossing deck's
# axes with one in-plane END moved off the plane onto the wrong side.


def _moved_end(ax, dz):
    """`ax` with its in-plane ends moved `dz` along z."""
    out = dict(ax)
    out["ends"] = [
        (np.asarray(pt, dtype=float) + np.array([0.0, 0.0, dz]), sign, fv)
        if abs(pt[2]) < 1e-12
        else (pt, sign, fv)
        for pt, sign, fv in ax["ends"]
    ]
    assert any(abs(pt[2] - dz) < 1e-15 for pt, _s, _f in out["ends"])
    return out


def _call(fn, ctx, R, C):
    # Spelled out rather than through `_block_preamble`, so the test runs
    # unchanged against the tree before U5 (where it must fail).
    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = ctx.medium
    gz = float(ctx.ground_z)
    c1 = CF._c1_moment(ctx.omega, ctx.mu)
    memo = CF._near_interface.TripleMemo()
    return fn(ctx, R, C, eps_t, k_p, c1, gz, memo=memo)


WRONG = 10 * CF._PLANE_TOL


@pytest.mark.parametrize(
    "which", ["below end above the plane", "above end below the plane"]
)
def test_a_wrong_side_end_raises_in_the_reversed_spelling_as_in_the_forward(which):
    """`_ends_and_corner_reversed` used to clamp an end onto the plane with a
    bare `min(..., 0)` / `max(..., 0)` — silently, at any distance — where the
    forward `_ends_and_corner` refuses by name past `_PLANE_TOL` (the
    momwire#852 rule, `_on_plane_side`). Both spellings now refuse."""
    ctx, _a_idx, _b_idx, A, B = _crossing_axes()
    if which == "below end above the plane":
        B = _moved_end(B, +WRONG)
    else:
        A = _moved_end(A, -WRONG)
    with pytest.raises(ValueError, match="wrong side of ground_z"):
        _call(CF._ends_and_corner, ctx, A, B)
    with pytest.raises(ValueError, match="wrong side of ground_z"):
        _call(CF._ends_and_corner_reversed, ctx, B, A)


@pytest.mark.parametrize("dz_below, dz_above", [(+0.5e-12, 0.0), (0.0, -0.5e-12)])
def test_a_within_tolerance_end_is_the_plane_in_both_spellings(dz_below, dz_above):
    """Inside `_PLANE_TOL` the point IS the interface: snapped to exactly 0.0
    and served. The old clamp did the same, so the reversed block is unchanged
    there — bit for bit the block of the unmoved deck, in both spellings (an
    end's z enters only through the tables and the corner's in-plane test)."""
    ctx, _a_idx, _b_idx, A, B = _crossing_axes()
    A2 = _moved_end(A, dz_above) if dz_above else A
    B2 = _moved_end(B, dz_below) if dz_below else B
    assert np.array_equal(
        _call(CF._ends_and_corner, ctx, A2, B2), _call(CF._ends_and_corner, ctx, A, B)
    )
    assert np.array_equal(
        _call(CF._ends_and_corner_reversed, ctx, B2, A2),
        _call(CF._ends_and_corner_reversed, ctx, B, A),
    )


def test_the_reversed_ends_answer_rows_and_accumulate_like_the_forward():
    """The reversed ends now have the forward's #1029 answers: `rows=` is the
    pair of slices of the full block, and `out=` accumulates bit-identically
    to `out += <the full block>` (what `cross_complete_block_reversed` used
    to do, through an (n, n) transient)."""
    ctx, _a_idx, _b_idx, A, B = _crossing_axes()
    full = _call(CF._ends_and_corner_reversed, ctx, B, A)
    n = full.shape[0]
    assert full.shape == (n, n)
    R = np.flatnonzero(np.any(full != 0, axis=1))[::2]
    assert 0 < R.size < n
    eps_t, k_p, gz, c1, memo = CF._block_preamble(ctx)
    t_r, t_c = CF._ends_and_corner_reversed(
        ctx, B, A, eps_t, k_p, c1, gz, memo=memo, rows=R
    )
    assert np.array_equal(t_r, full[R, :])
    assert np.array_equal(t_c, full[:, R])
    rng = np.random.default_rng(1168)
    base = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    want = base.copy()
    want += full
    got = base.copy()
    eps_t, k_p, gz, c1, memo = CF._block_preamble(ctx)
    CF._ends_and_corner_reversed(ctx, B, A, eps_t, k_p, c1, gz, memo=memo, out=got)
    assert np.array_equal(got, want)
