"""Degree 3 reaches the C++ kernels — momwire#999 step 3.

#883 shipped `bspline-3` on the basis axis and it has run **pure numpy end to
end** ever since, because every degree switch in `_accel_bspline.cpp` stopped
at `case 2:` and two entry points carried an explicit `max_d > 2` guard. The
generated degree-3 closed forms were compiled into the shipped `.so` the whole
time with nothing able to call them.

Steps 1 and 2 removed the two silent-failure modes on that path (an
out-of-bounds binomial table, and a `p * 3 + q` flattening that collides at
degree 3). This step lifts the ceiling itself: 20 `case 3:` arms, 2 range
guards, and the two Python routing gates.

What this module pins:

  * **no degree switch is left behind.** A census, not a spot check — the
    twenty sites are found by the message they throw, which is the only
    honest count: six of them switch on `support_seg.shape(1) - 1` and never
    mention `max_d`, so a `grep max_d` census undercounts by six.

  * **the routing gates READ the binary.** `_BSPLINE_ACCEL_MAX_D` and
    `_OFFEDGE_BLOCK_ACCEL_MAX_D` were hand-written 2s. That is exactly the
    shape that drifted between #883 and #999 — the tables moved to degree 3
    and the constants did not, for a release. A constant that names a
    property of a compiled artefact belongs to the artefact.

  * **degree 3 agrees with the twin it used to BE.** Until this step the
    numpy path was not a reference for degree 3, it was the implementation.
    The same answers must survive the move.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from momwire import hmatrix as H
from momwire import _bspline_kernels as K
from momwire._bspline_static_moments import MAX_D

acc = pytest.importorskip("momwire._accelerators")

pytestmark = pytest.mark.skipif(
    not hasattr(acc, "BSPLINE_MOMENT_MAX_D"),
    reason="accelerator predates momwire#999",
)

N_DEGREE_SWITCHES = 20


def _source() -> str:
    import momwire

    return (pathlib.Path(momwire.__file__).parent / "_accel_bspline.cpp").read_text()


def test_no_degree_switch_still_stops_at_two():
    """Comments stripped: the file discusses the old ceiling in prose, and a
    raw search matches the discussion rather than the code (the shape that bit
    steps 1 and 2's tripwires, the #936 AST gate, and #988's)."""
    code = re.sub(r"//[^\n]*", "", _source())
    assert "max_d must be 1 or 2" not in code, "a degree switch was left at 2"
    n = code.count("max_d must be 1, 2 or 3")
    assert n == N_DEGREE_SWITCHES, f"expected {N_DEGREE_SWITCHES} switches, found {n}"


def test_the_range_guards_use_the_constant_not_a_literal():
    code = re.sub(r"//[^\n]*", "", _source())
    assert "max_d > 2" not in code
    assert code.count("max_d > BSPLINE_MOMENT_MAX_D") == 2


def test_the_routing_gates_read_the_binary():
    assert K._BSPLINE_ACCEL_MAX_D == acc.BSPLINE_MOMENT_MAX_D
    assert H._OFFEDGE_BLOCK_ACCEL_MAX_D == acc.BSPLINE_MOMENT_MAX_D
    assert acc.BSPLINE_MOMENT_MAX_D == MAX_D == K.MAX_D_SUPPORTED


@pytest.mark.parametrize(
    "module,name",
    [
        ("_bspline_kernels", "_BSPLINE_ACCEL_MAX_D"),
        ("hmatrix", "_OFFEDGE_BLOCK_ACCEL_MAX_D"),
    ],
)
def test_the_gates_are_not_restated_literals(module, name):
    """A `= 2` (or `= 3`) here is the drift this step removed. The value has to
    come from the extension, whose dispatch it describes."""
    import momwire

    src = (pathlib.Path(momwire.__file__).parent / f"{module}.py").read_text()
    code = re.sub(r"#[^\n]*", "", src)
    assert not re.search(rf"^{re.escape(name)}\s*=\s*\d+\s*$", code, re.M), (
        f"{name} is a literal again"
    )
    assert re.search(rf"{re.escape(name)}\s*=.*BSPLINE_MOMENT_MAX_D", code)


@pytest.mark.parametrize("N", (11, 41, 81))
def test_degree_three_matches_the_numpy_twin_it_used_to_be(N):
    """The house bar: max|diff| <= 1e-14 * max|table|, as in #808."""
    a, ends = 5e-4, np.arange(N + 1, dtype=float) * (4.0 / N)
    real = K._BSPLINE_ACCEL_MAX_D
    try:
        K._BSPLINE_ACCEL_MAX_D = real
        cxx = np.asarray(K._seg_seg_static_moments(ends, a, 3))
        K._BSPLINE_ACCEL_MAX_D = -1
        npy = np.asarray(K._seg_seg_static_moments(ends, a, 3))
    finally:
        K._BSPLINE_ACCEL_MAX_D = real
    assert cxx.shape == npy.shape == (4, 4, N, N)
    assert np.abs(cxx - npy).max() <= 1e-14 * np.abs(npy).max()


def test_the_table_entries_serve_degree_three():
    """These two carried the explicit `max_d out of range [0, 2]` guard, so
    they are the pair a case-list census would miss entirely."""
    h, a, N = 0.05, 5e-4, 9
    for fn, args in (
        (acc.seg_seg_static_moments_bspline_uniform_table, (h, a, N, 3)),
        (acc.seg_seg_static_moments_bspline_uniform, (h, a, N, 3)),
        (acc.seg_seg_static_moments_bspline_uniform_ek_table, (h, a, N, 3, a)),
        (acc.seg_seg_static_moments_bspline_uniform_ek, (h, a, N, 3, a)),
    ):
        out = np.asarray(fn(*args))
        assert out.shape[0] == out.shape[1] == 4
        assert np.isfinite(out).all()


def test_degree_four_still_refuses():
    """The ceiling moved; it did not vanish. Nothing is derived past MAX_D."""
    with pytest.raises(Exception):
        acc.seg_seg_static_moments_bspline_uniform_table(0.05, 5e-4, 9, MAX_D + 1)
    with pytest.raises(NotImplementedError):
        K._seg_seg_static_moments(np.arange(10, dtype=float), 5e-4, MAX_D + 1)
