"""The same-edge smooth-kernel moments at a complex k in C++ (momwire#1752 L4).

`_seg_seg_reg_moments_from_geometry` at an in-medium k used to be numpy's
`expm1_neg_jkR(k, R) / (4 pi R)` and its einsum. It now calls
`seg_seg_reg_moments_bspline_cplx`, the kernel the executables' buried fill
already ran (`_bspline_fill_core`'s `reg_moments_cplx`), so the exe audits
hold Python and the executables to one arithmetic. These tests hold the kernel
to the numpy reference it replaced, byte for byte, on the shapes the buried
fill hands it: one segment (the einsum iterator drops that axis), several,
uniform and graded arcs, each Gauss order and degree the fill uses, and both
lossy and nearly lossless soils.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _bspline_kernels as bk

pytestmark = pytest.mark.skipif(
    not bk._HAVE_BSPLINE_REG_CPLX_ACCEL,
    reason="the extension has no seg_seg_reg_moments_bspline_cplx",
)

# k_m in a buried medium: a lossy soil at 7 MHz and a nearly lossless one.
KS = (complex(0.5570054, -0.2965546), complex(0.94, -0.0012))


def _arc(n, graded):
    if not graded:
        return np.linspace(0.0, 0.042 * n, n + 1)
    h = 0.05 * (1.0 + 0.3 * np.sin(np.arange(n)))
    return np.concatenate([[0.0], np.cumsum(h)])


@pytest.mark.parametrize("k", KS, ids=["lossy", "low-loss"])
@pytest.mark.parametrize("n", [1, 2, 7, 33])
@pytest.mark.parametrize("n_qp", [4, 8])
@pytest.mark.parametrize("max_d", [1, 2, 3])
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_the_kernel_is_numpys_bytes(k, n, n_qp, max_d, graded, monkeypatch):
    geo = bk._seg_seg_reg_geometry(_arc(n, graded), 0.002, max_d, n_qp)
    got = bk._seg_seg_reg_moments_from_geometry(geo, k)
    monkeypatch.setattr(bk, "_HAVE_BSPLINE_REG_CPLX_ACCEL", False)
    want = bk._seg_seg_reg_moments_from_geometry(geo, k)
    assert got.shape == want.shape == (max_d + 1, max_d + 1, n, n)
    assert got.dtype == want.dtype
    assert got.tobytes() == want.tobytes()


def test_a_windowed_geometry_keeps_the_numpy_route(monkeypatch):
    """The kernel serves the square geometry only: a windowed one (an
    observer window, momwire#968) still runs numpy, and the call says so by
    never reaching the extension."""
    geo = bk._seg_seg_reg_geometry(_arc(9, False), 0.002, 2, 8, rows=slice(2, 5))

    def boom(*a, **kw):
        raise AssertionError("the windowed geometry reached the C++ kernel")

    monkeypatch.setattr(
        bk._acc, "seg_seg_reg_moments_bspline_cplx", boom, raising=False
    )
    out = bk._seg_seg_reg_moments_from_geometry(geo, KS[0])
    assert out.shape == (3, 3, 3, 9)
