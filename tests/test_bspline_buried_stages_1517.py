"""momwire#1517 step 4: bs2's C++ buried fill, stage by stage, against
`compute_Z_operator_buried`.

The C++ fill stops before the crossing serve (the cross blocks and the self
completions, step 4b-2) and the loading, so the Python side runs the same
fill with those two left out. What is held bit for bit is every block before
them:

- the direct and image blocks in both media (the subset windows, the
  above-ground extended kernel, the complex same-edge moments);
- the above/above remainder over `remainder_field_proj`;
- the below/below remainder over the C++-filled table.

This covers the crossing decks the printout cannot serve yet.
"""

import ctypes
import glob
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver, _accel, _crossing_fill

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
DECKS = [
    FIX / "eznec_split_1281" / "thr_n21.nec",
    FIX / "eznec_crossing_source_1282" / "dan_seg1end2.nec",
    FIX / "eznec_split_1281" / "dan_through.nec",
    FIX / "eznec_fr_step_1508" / "b01_buried_xq.nec",
    FIX / "eznec_phantom_1139" / "ground_rod_efhw_wa7ark.nec",
    FIX / "bspline_printout_buried" / "ab_bent_pair.nec",
]

acc = _accel.acc
pytestmark = pytest.mark.skipif(
    acc is None or not hasattr(acc, "bspline_buried_stages"),
    reason="needs the extension's buried stages hook",
)


def _blas():
    """numpy's bundled ILP64 OpenBLAS's (dgemm, dgemv, ddot), as the C++
    fill calls them for the image weights."""
    import numpy

    found = sorted(
        glob.glob(
            str(Path(numpy.__file__).parent.parent / "numpy.libs" / "*openblas64*")
        )
    )
    if not found:
        pytest.skip("no bundled OpenBLAS beside numpy (not a wheel install)")
    lib = ctypes.CDLL(found[0])

    def sym(*names):
        for n in names:
            if hasattr(lib, n):
                return ctypes.cast(getattr(lib, n), ctypes.c_void_p).value
        pytest.skip(f"no {names[0]} in numpy's OpenBLAS")

    return [
        sym("scipy_cblas_dgemm64_", "cblas_dgemm64_"),
        sym("scipy_cblas_dgemv64_", "cblas_dgemv64_"),
        sym("scipy_cblas_ddot64_", "cblas_ddot64_"),
    ]


def _without_crossing(monkeypatch):
    """`compute_Z_operator_buried` with the crossing serve contributing
    nothing: zero cross blocks, no self completions, no axes built."""

    def zeros(ctx, a_idx, b_idx, A, B, *, rows=None, **_kw):
        return np.zeros((A["n_basis"], B["n_basis"]), dtype=np.complex128)

    monkeypatch.setattr(_crossing_fill, "cross_complete_block_split", zeros)
    monkeypatch.setattr(
        _crossing_fill,
        "cross_complete_blocks_two_radius",
        lambda *a, **k: (zeros(*a, **k), zeros(*a, **k)),
    )
    monkeypatch.setattr(_crossing_fill, "self_completions", lambda *a, **k: None)
    monkeypatch.setattr(
        _crossing_fill, "self_completions_two_radius", lambda *a, **k: None
    )
    monkeypatch.setattr(
        _crossing_fill,
        "axis_data",
        lambda ctx, idx, **k: {"n_basis": ctx.basis.n_basis},
    )


def test_the_buried_blocks_before_the_crossing_serve_are_pythons(monkeypatch):
    blas = _blas()
    _without_crossing(monkeypatch)
    compared = 0
    for path in DECKS:
        raw = path.read_bytes()
        entries = acc.bspline_front_geometry(raw)
        stages = acc.bspline_buried_stages(raw, blas)
        # the hook answers per preparation, the geometry per structure: both
        # in deck order, the hook's list the shorter (no run-only entries)
        for entry, st in zip(entries, stages, strict=False):
            if not st:
                continue
            assert "stopped" not in st, (path, st.get("stopped"))
            s = BSplineSolver(**entry["kwargs"])
            s._apply_loading = lambda Z, **_k: Z
            g = s._build_geometry()
            supp, polys, *_ = s._build_basis_polynomials(g)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Zp = np.ascontiguousarray(s._compute_Z_operator(g, supp, polys))
            Zc = np.asarray(st["Z"], dtype=np.complex128).reshape(Zp.shape)
            assert np.array_equal(Zp.view(np.uint64), Zc.view(np.uint64)), path
            compared += 1
    # crossing decks (one- and two-radius, loaded), a rod, and a wholly-buried one
    assert compared >= len(DECKS), compared


def test_the_crossing_serve_is_pythons():
    """Step 4b: the whole buried Z of a crossing deck, the crossing serve
    included (the cross block's split fill, its ends and corner, the self
    completions), bit for bit against `compute_Z_operator_buried` with the
    loading off: one-radius crossing decks, one of them through the ACA."""
    blas = _blas()
    import numpy

    lib = ctypes.CDLL(
        sorted(
            glob.glob(
                str(Path(numpy.__file__).parent.parent / "numpy.libs" / "*openblas64*")
            )
        )[0]
    )

    def sym(*names):
        for n in names:
            if hasattr(lib, n):
                return ctypes.cast(getattr(lib, n), ctypes.c_void_p).value
        pytest.skip(f"no {names[0]} in numpy's OpenBLAS")

    blas += [
        sym("scipy_cblas_zgemm64_", "cblas_zgemm64_"),
        sym("scipy_cblas_zgemv64_", "cblas_zgemv64_"),
        sym("scipy_cblas_zdotu_sub64_", "cblas_zdotu_sub64_"),
        sym("scipy_cblas_zdotc_sub64_", "cblas_zdotc_sub64_"),
    ]
    compared = 0
    for path in (
        FIX / "eznec_split_1281" / "thr_n21.nec",
        FIX / "eznec_fr_step_1508" / "b01_buried_xq.nec",
        FIX / "bspline_printout_crossing" / "aca_vertical_screen.nec",
    ):
        raw = path.read_bytes()
        entries = acc.bspline_front_geometry(raw)
        stages = acc.bspline_buried_stages(raw, blas, True)
        for entry, st in zip(entries, stages, strict=False):
            if not st:
                continue
            assert "stopped" not in st, (path, st.get("stopped"))
            s = BSplineSolver(**entry["kwargs"])
            s._apply_loading = lambda Z, **_k: Z
            g = s._build_geometry()
            supp, polys, *_ = s._build_basis_polynomials(g)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Zp = np.ascontiguousarray(s._compute_Z_operator(g, supp, polys))
            Zc = np.asarray(st["Z"], dtype=np.complex128).reshape(Zp.shape)
            assert np.array_equal(Zp.view(np.uint64), Zc.view(np.uint64)), path
            compared += 1
    assert compared >= 3, compared
