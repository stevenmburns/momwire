"""The buried fill on the chunked route (momwire#915).

A deck with buried wires used to be filled through four DENSE (d+1, d+1,
N, N) moment tensors — direct and image, above and below — and refused when
one of them did not fit `swept_mem_mb`: the chunked fill+assemble route
(momwire#136) had no medium, because its two windowed C++ assemblers took a
`double eps`. That refusal capped the hosted app (64 MB) near 12 radials.

Now the windowed assemblers have complex-eps~ twins (the #910 pattern: a
COMPLEX_EPS template flag whose branch carries c = 1/(jωε̃); the real
instantiations unchanged and array_equal to the pre-change build), and
`_accumulate_Z_subset_chunked` streams each of the four terms window by
window into Z. It is the default whenever the twins are built — faster than
the dense route even where the tensor fits, because the zero-padded scatter
is gone — and the dense route stays as the REFERENCE every buried gate was
pinned on.

Gates:

- G-915-1  the windowed complex-eps twins over the full window equal the
           non-windowed complex-eps assemblers (#910, themselves gated
           against the numpy loop), plain and weighted, degrees 1 and 2.
- G-915-2  the chunked route equals the dense route to 1e-12 relative on
           the hub deck, the crossing deck and a wholly-below dipole, and
           the chunked accumulator is what ran.
- G-915-3  a budget the dense tensor does not fit no longer refuses; the
           refusal survives only when the twins are absent, and its
           message still names the budget.
- G-915-4  `_contiguous_runs` is the run decomposition it claims to be.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import momwire.bspline as _bs
from momwire.bspline import BSplineSolver, _contiguous_runs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_buried_serve_553 import buried_dipole  # noqa: E402
from test_crossing_serve_524 import crossing_deck, hub_deck  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:crossing node"),
    pytest.mark.skipif(
        not (
            _bs._HAVE_BSPLINE_WINDOWED_CPLX_EPS_ACCEL
            and _bs._HAVE_BSPLINE_W_WINDOWED_CPLX_EPS_ACCEL
        ),
        reason="the windowed complex-eps assemblers are not built",
    ),
]

OMEGA, EPS0, MU0 = 2 * np.pi * 7e6, 8.854e-12, 4e-7 * np.pi
EPS_M = EPS0 * (13.0 - 4.2j)


def _synthetic(d, seed=915, nb=40, ns=30):
    rng = np.random.default_rng(seed)
    NM = d + 1
    J = rng.normal(size=(NM, NM, ns, ns)) + 1j * rng.normal(size=(NM, NM, ns, ns))
    supp = rng.integers(0, ns, size=(nb, NM))
    polys = rng.normal(size=(nb, NM, NM))
    tan = rng.normal(size=(ns, 3))
    wA = rng.normal(size=(ns, ns)) + 1j * rng.normal(size=(ns, ns))
    wP = rng.normal(size=(ns, ns)) + 1j * rng.normal(size=(ns, ns))
    return J, supp, polys, tan, wA, wP


@pytest.mark.parametrize("d", [1, 2])
def test_g915_1_the_windowed_twins_equal_the_assemblers_over_the_full_window(d):
    J, supp, polys, tan, wA, wP = _synthetic(d)
    nb, ns = supp.shape[0], J.shape[2]
    all_n = np.arange(nb, dtype=np.int64)
    acc = _bs._acc
    # plain: the windowed kernel forms t_m . t_n itself; hand the assembler
    # the same table.
    ref = acc.assemble_Z_bspline_cplx_eps(
        J, supp, polys, tan @ tan.T, OMEGA, EPS_M, MU0, d
    )
    Z = np.zeros((nb, nb), dtype=np.complex128, order="F")
    # two observer windows, so the accumulation across windows is exercised
    for i0, i1 in ((0, 11), (11, ns)):
        acc.assemble_Z_bspline_windowed_cplx_eps(
            np.ascontiguousarray(J[:, :, i0:i1, :]),
            supp,
            polys,
            tan,
            all_n,
            all_n,
            i0,
            i1,
            0,
            ns,
            OMEGA,
            EPS_M,
            MU0,
            Z,
            0,
        )
    rel = np.abs(Z - ref).max() / np.abs(ref).max()
    assert rel < 1e-13, f"plain d={d}: {rel:.3e}"
    ref_w = acc.assemble_Z_bspline_weighted_cplx_eps(
        J, supp, polys, wA, wP, OMEGA, EPS_M, MU0, d
    )
    Z = np.zeros((nb, nb), dtype=np.complex128, order="F")
    for i0, i1 in ((0, 11), (11, ns)):
        acc.assemble_Z_bspline_weighted_windowed_cplx_eps(
            np.ascontiguousarray(J[:, :, i0:i1, :]),
            supp,
            polys,
            np.ascontiguousarray(wA[i0:i1]),
            np.ascontiguousarray(wP[i0:i1]),
            all_n,
            all_n,
            i0,
            i1,
            0,
            ns,
            OMEGA,
            EPS_M,
            MU0,
            complex(-1.0),
            Z,
            0,
        )
    rel = np.abs(Z + ref_w).max() / np.abs(ref_w).max()
    assert rel < 1e-13, f"weighted d={d}: {rel:.3e}"


def _dense_and_chunked(monkeypatch, make_solver):
    calls = {"n": 0}
    real = BSplineSolver._accumulate_Z_subset_chunked

    def counted(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(BSplineSolver, "_accumulate_Z_subset_chunked", counted)
    z_chunked, _ = make_solver().compute_impedance()
    n_chunked = calls["n"]
    monkeypatch.setattr(_bs, "_HAVE_BSPLINE_WINDOWED_CPLX_EPS_ACCEL", False)
    calls["n"] = 0
    z_dense, _ = make_solver().compute_impedance()
    return z_dense, z_chunked, n_chunked, calls["n"]


@pytest.mark.parametrize("name", ["hub", "crossing", "below"])
def test_g915_2_the_chunked_route_is_the_dense_route(
    monkeypatch, name, record_property
):
    if name == "hub":
        make = lambda: BSplineSolver(**hub_deck())  # noqa: E731
        expect_calls = 4
    elif name == "crossing":
        make = lambda: BSplineSolver(**crossing_deck(1))  # noqa: E731
        expect_calls = 4
    else:
        make = lambda: buried_dipole()[0]  # noqa: E731
        expect_calls = 2  # no above segments: direct below + image below
    z_dense, z_chunked, n_chunked, n_dense = _dense_and_chunked(monkeypatch, make)
    record_property(f"z_dense_{name}", f"{z_dense:.9f}")
    record_property(f"z_chunked_{name}", f"{z_chunked:.9f}")
    assert n_chunked == expect_calls and n_dense == 0, (n_chunked, n_dense)
    assert abs(z_chunked - z_dense) <= 1e-12 * abs(z_dense), (z_chunked, z_dense)


def test_g915_3_a_budget_the_tensor_does_not_fit_no_longer_refuses(monkeypatch):
    s = BSplineSolver(**hub_deck())
    s.swept_mem_mb = 0  # nothing fits: the dense route would refuse
    z, _ = s.compute_impedance()
    assert np.isfinite(z)
    monkeypatch.setattr(_bs, "_HAVE_BSPLINE_W_WINDOWED_CPLX_EPS_ACCEL", False)
    s = BSplineSolver(**hub_deck())
    s.swept_mem_mb = 0
    with pytest.raises(NotImplementedError, match="swept_mem_mb budget"):
        s.compute_impedance()


def test_g915_4_contiguous_runs():
    assert _contiguous_runs(np.array([], dtype=np.int64)) == []
    assert _contiguous_runs(np.array([3])) == [(3, 4)]
    assert _contiguous_runs(np.array([0, 1, 2, 5, 6, 9])) == [(0, 3), (5, 7), (9, 10)]
    assert _contiguous_runs(np.arange(4, 40)) == [(4, 40)]
