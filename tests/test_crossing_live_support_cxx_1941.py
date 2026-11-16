"""momwire#1941 (buried memory arc, the executables' twin of U2): bs2's C++
crossing serve holds the cross block on its live support (`live_rows`, as
`_crossing_fill._live_support`) and folds it into Z there, never as an
(n_basis, n_basis) array.

Pinned against `compute_Z_operator_buried` bit for bit through the stages
hook: the whole buried Z, through the fold, at a one-radius node, at one
whose far blocks go through the ACA (rho formed per sample), and at a
two-radius node (two blocks on one support, `Z -= t_above; Z -= t_below.T`),
which the older stages test leaves out. Skipping the subtractions off the
support moves no bit only if the whole block was an exact +0 there; a live
entry the support missed throws (`Held`), so a Z that matches carries both.
The hook's block, rebuilt from the held one, has exact-zero rows and columns,
so the skip ran.

The block itself is not compared with a standalone `cross_complete_block_*`
call: that call rounds a few entries differently from the one inside the
fill (1e-14 of 4.5e4 on the two-radius deck, measured on the pre-#1941 C++
too), while the Z both fold into agrees to the bit.

The C++ fold has no slab knob to force, unlike `_fold_sub`: it subtracts in
place, one entry at a time, with no temporary whose size a slab would bound.
"""

from __future__ import annotations

import ctypes
import glob
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver, _accel
from momwire import _crossing_fill as cf

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
ONE_RADIUS = FIX / "eznec_split_1281" / "thr_n21.nec"
ACA = FIX / "bspline_printout_crossing" / "aca_vertical_screen.nec"
TWO_RADIUS = FIX / "sg_two_radius_1630" / "hub1_b025mm.nec5.nec"

acc = _accel.acc
pytestmark = [
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "bspline_buried_stages"),
        reason="needs the extension's buried stages hook",
    ),
    pytest.mark.filterwarnings("ignore::momwire._crossing_fill.CoarseCrossingNode"),
]


def _blas():
    """numpy's bundled ILP64 OpenBLAS's entry points, in the order the stages
    hook reads them: the image weights' three, then the ACA's four."""
    import numpy

    found = sorted(
        glob.glob(
            str(Path(numpy.__file__).parent.parent / "numpy.libs" / "*openblas64*")
        )
    )
    if not found:
        pytest.skip("no bundled OpenBLAS beside numpy (not a wheel install)")
    lib = ctypes.CDLL(found[0])

    def sym(name):
        for n in (f"scipy_cblas_{name}64_", f"cblas_{name}64_"):
            if hasattr(lib, n):
                return ctypes.cast(getattr(lib, n), ctypes.c_void_p).value
        pytest.skip(f"no {name} in numpy's OpenBLAS")

    return [
        sym(n)
        for n in ("dgemm", "dgemv", "ddot", "zgemm", "zgemv", "zdotu_sub", "zdotc_sub")
    ]


def _served(path):
    """(solver kwargs, the C++ stages) of the deck's one buried crossing run."""
    raw = path.read_bytes()
    entries = acc.bspline_front_geometry(raw)
    stages = acc.bspline_buried_stages(raw, _blas(), True)
    got = [(e, st) for e, st in zip(entries, stages, strict=False) if st]
    assert len(got) == 1, (path, len(got))
    entry, st = got[0]
    assert "stopped" not in st, (path, st.get("stopped"))
    return entry["kwargs"], st


def _python_z(kwargs, monkeypatch):
    """Python's buried Z with the loading off, and which serves folded it
    through the live support (`into=`)."""
    s = BSplineSolver(**kwargs)
    s._apply_loading = lambda Z, **_k: Z
    g = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(g)
    folds = []
    for name in ("cross_complete_block_split", "cross_complete_blocks_two_radius"):

        def spy(*a, _orig=getattr(cf, name), _name=name, **k):
            if k.get("into") is not None:
                folds.append(_name)
            return _orig(*a, **k)

        monkeypatch.setattr(cf, name, spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = np.ascontiguousarray(s._compute_Z_operator(g, supp, polys))
    return Z, folds


@pytest.mark.parametrize(
    ("path", "serve"),
    [
        (ONE_RADIUS, "cross_complete_block_split"),
        (ACA, "cross_complete_block_split"),
        (TWO_RADIUS, "cross_complete_blocks_two_radius"),
    ],
    ids=["one-radius", "aca", "two-radius"],
)
def test_the_held_block_folds_to_pythons_z(path, serve, monkeypatch):
    kwargs, st = _served(path)
    Zp, folds = _python_z(kwargs, monkeypatch)
    # the reference is Python's own live-support fold, not the old (n, n) one
    assert folds == [serve], folds
    n = Zp.shape[0]
    tc = np.asarray(st["t_ab"], dtype=np.complex128).reshape(n, n)
    # not vacuous: the block has rows and columns the fold never visits
    live_r, live_c = np.any(tc != 0, axis=1).sum(), np.any(tc != 0, axis=0).sum()
    assert 0 < live_r < n and 0 < live_c < n, (live_r, live_c, n)
    Zc = np.asarray(st["Z"], dtype=np.complex128).reshape(n, n)
    assert np.array_equal(Zp.view(np.uint64), Zc.view(np.uint64)), path
