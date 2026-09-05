"""The buried assembly after the pair kernel (momwire#910).

After #906 the 12-radial buried screen's profile was einsums, the crossing
fill, the serve plan and the two assemblies. Three of those are this PR:

- the field-form Jc contraction is contracted pairwise rather than as one
  loop over every index (`optimize=True`);
- `_buried_serve_plan`'s below/below extents are a chunked reduction that
  never holds an (n_nodes, n_nodes) array;
- the two C++ assemblers gain complex-eps~ twins, so the buried fill's
  assemblies no longer fall to the numpy einsum loop.

Gates:

- G-910-1  the complex-eps assemblers match the numpy loop they replace
           (plain and weighted, degrees 1 and 2), and the buried hub solve
           reaches them.
- G-910-2  the chunked extents equal the all-pairs formulation to 1e-12,
           the chunk size does not matter, and the two refusals still fire.
- G-910-3  the pairwise Jc contraction equals the single-loop one to
           roundoff on the field block's real shapes.
- G-910-4  a free-space and a reflection-coefficient solve never reach a
           complex-eps entry.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import momwire.bspline as _bs
from momwire.bspline import BSplineSolver, _pair_extents_below

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_crossing_serve_524 import hub_deck  # noqa: E402

CPLX = "assemble_Z_bspline_cplx_eps"
CPLX_W = "assemble_Z_bspline_weighted_cplx_eps"
REAL = "assemble_Z_bspline"
REAL_W = "assemble_Z_bspline_weighted"

pytestmark = pytest.mark.skipif(
    not (
        _bs._HAVE_BSPLINE_ASSEMBLE_CPLX_EPS_ACCEL
        and _bs._HAVE_BSPLINE_ASSEMBLE_W_CPLX_EPS_ACCEL
    ),
    reason="the complex-eps assemblers are not built",
)


class _AccelSpy:
    def __init__(self, real, names):
        self._real = real
        self.counts = dict.fromkeys(names, 0)

    def __getattr__(self, name):
        target = getattr(self._real, name)
        if name not in self.counts:
            return target

        def counted(*args, **kwargs):
            self.counts[name] += 1
            return target(*args, **kwargs)

        return counted


@pytest.fixture
def spy(monkeypatch):
    s = _AccelSpy(_bs._acc, (CPLX, CPLX_W, REAL, REAL_W))
    monkeypatch.setattr(_bs, "_acc", s)
    return s


def _synthetic(d, seed=910, nb=40, ns=30):
    rng = np.random.default_rng(seed)
    NM = d + 1
    J = rng.normal(size=(NM, NM, ns, ns)) + 1j * rng.normal(size=(NM, NM, ns, ns))
    supp = rng.integers(0, ns, size=(nb, NM))
    polys = rng.normal(size=(nb, NM, NM))
    td = rng.normal(size=(ns, ns))
    wA = rng.normal(size=(ns, ns)) + 1j * rng.normal(size=(ns, ns))
    wP = rng.normal(size=(ns, ns)) + 1j * rng.normal(size=(ns, ns))
    return J, supp, polys, td, wA, wP


@pytest.mark.parametrize("d", [1, 2])
def test_g910_1_the_complex_eps_assemblers_match_the_numpy_loop(monkeypatch, d):
    J, supp, polys, td, wA, wP = _synthetic(d)
    eps = 8.854e-12 * (13.0 - 4.2j)
    solver = BSplineSolver.__new__(BSplineSolver)
    solver.degree = d
    solver.omega = 2 * np.pi * 7e6
    solver.mu = 4e-7 * np.pi
    solver.eps = 8.854e-12  # no token: the class default gives cancel_flag 0
    got = solver._assemble_Z(J, supp, polys, {"tangents": None}, td_all=td, eps=eps)
    got_w = solver._image_Z_weighted(J, supp, polys, wA, wP, eps=eps)
    monkeypatch.setattr(_bs, "_HAVE_BSPLINE_ASSEMBLE_CPLX_EPS_ACCEL", False)
    monkeypatch.setattr(_bs, "_HAVE_BSPLINE_ASSEMBLE_W_CPLX_EPS_ACCEL", False)
    ref = solver._assemble_Z(J, supp, polys, {"tangents": None}, td_all=td, eps=eps)
    ref_w = solver._image_Z_weighted(J, supp, polys, wA, wP, eps=eps)
    for name, a, b in (("plain", got, ref), ("weighted", got_w, ref_w)):
        rel = np.abs(a - b).max() / np.abs(b).max()
        assert rel < 1e-13, f"{name} d={d}: {rel:.3e}"


@pytest.mark.filterwarnings("ignore:crossing node")
def test_g910_1b_the_buried_hub_reaches_the_complex_eps_assemblers(spy, monkeypatch):
    """On the DENSE route (forced here by taking the windowed twin away —
    momwire#915 made the chunked route the default) the hub reaches both
    non-windowed complex-eps twins."""
    monkeypatch.setattr(_bs, "_HAVE_BSPLINE_WINDOWED_CPLX_EPS_ACCEL", False)
    z, _ = BSplineSolver(**hub_deck()).compute_impedance()
    assert spy.counts[CPLX] >= 1 and spy.counts[CPLX_W] >= 1, spy.counts
    assert np.isfinite(z)


def _all_pairs(x, y, d_b):
    rho = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    hh = d_b[:, None] + d_b[None, :]
    return float(np.max(np.hypot(rho, hh))), float(np.min(np.arctan2(hh, rho)))


def test_g910_2_the_chunked_extents_are_the_all_pairs_numbers(monkeypatch):
    # Pinned to the numpy path (momwire#914 unit 1). `_pair_extents_below`
    # now dispatches to a C++ twin when one is built, which ignores `rows`
    # entirely — so with the accelerator live all four chunk sizes below
    # would take the same C++ path and the chunking this gate exists to
    # test would go unexercised. The twin has its own gate in
    # test_plan_extents_914.py; this one stays the CHUNKED form's.
    monkeypatch.setattr(_bs, "_HAVE_PLAN_EXTENTS_ACCEL", False)
    rng = np.random.default_rng(2)
    n = 1500
    x = rng.uniform(-11, 11, n)
    y = rng.uniform(-11, 11, n)
    d_b = rng.uniform(0.005, 0.3, n)
    r1_ref, th_ref = _all_pairs(x, y, d_b)
    for rows in (1, 7, 256, 4096):
        r1, th = _pair_extents_below(x, y, d_b, rows=rows)
        assert abs(r1 - r1_ref) <= 1e-12 * r1_ref, (rows, r1, r1_ref)
        assert abs(th - th_ref) <= 1e-12 * th_ref, (rows, th, th_ref)
    # A node meeting itself has rho = 0 and hh > 0: the ratio is +inf there
    # and never the minimum, so no divide warning escapes.
    with np.errstate(divide="raise"):
        _pair_extents_below(x[:3], y[:3], d_b[:3])


def test_g910_2_the_chunked_form_is_reachable_at_all(monkeypatch):
    """The guard above is only meaningful if the flag it clears is the one
    the dispatch reads: a renamed flag would make `monkeypatch.setattr`
    silently create a new attribute and the test would pass while still
    running C++."""
    assert hasattr(_bs, "_HAVE_PLAN_EXTENTS_ACCEL")


@pytest.mark.filterwarnings("ignore:crossing node")
def test_g910_2b_the_refusals_still_fire():
    # Past the r1 cap: a radial far longer than the below-cap allows.
    long = hub_deck()
    long["wires"] = [
        np.array([(200.0 * dx, 200.0 * dy, -0.15), (0.0, 0.0, -0.15)])
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1))
    ] + long["wires"][4:]
    with pytest.raises(ValueError, match="R1"):
        BSplineSolver(**long).compute_impedance()
    # Grazing: a buried radial a hair under the plane.
    shallow = hub_deck(depth=1e-4)
    with pytest.raises(ValueError, match="grazing|floor|theta|angle"):
        BSplineSolver(**shallow).compute_impedance()


def test_g910_3_the_pairwise_jc_contraction_is_the_single_loop_one():
    rng = np.random.default_rng(3)
    d, q, chunk, n_src = 2, 6, 22, 60
    W_obs = rng.normal(size=(d + 1, chunk, q))
    W_src = rng.normal(size=(d + 1, n_src, q))
    fq = rng.normal(size=(chunk, q, n_src, q)) + 1j * rng.normal(
        size=(chunk, q, n_src, q)
    )
    ref = np.einsum("piq,iqjr,Pjr->pPij", W_obs, fq, W_src)
    got = np.einsum("piq,iqjr,Pjr->pPij", W_obs, fq, W_src, optimize=True)
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-13


def test_g910_4_free_space_and_refl_coef_never_reach_a_complex_eps_entry(spy):
    bent = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])],
        n_per_edge_per_wire=[[8, 8]],
        feeds=[(0, 0.5, 1 + 0j)],
    )
    BSplineSolver(**bent).compute_impedance()
    BSplineSolver(**bent, ground_z=-1.0, ground_eps=(10.0, 0.002)).compute_impedance()
    assert spy.counts[CPLX] == spy.counts[CPLX_W] == 0, spy.counts
    # The refl-coef ground assembles through the WINDOWED weighted kernel,
    # so only the plain real-eps entry is asserted reached here.
    assert spy.counts[REAL] >= 1, spy.counts
