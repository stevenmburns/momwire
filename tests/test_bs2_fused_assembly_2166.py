"""bs2's direct family assembled from the class values (momwire#2166).

With `pair_reuse` on, the chunked fill used to gather each observer window's
moments from the class values, (d+1)^2 complex a pair written whole, and run
the windowed assembly over it. `plan.assemble` fills each tile of Z from the
class values as it runs instead (`assemble_Z_bspline_windowed_tiles`,
`_bspline_core.h`), the shape razor2p's `assemble_rows` has (momwire#2144):

- a window assembled is the gathered window's sum, bit for bit, its near
  same-edge singles the fill's own moments (momwire#2063 step 3) or the
  class values;
- the fill runs it (counted by the kernel, not by the fill's report), and
  is the fill the gathered windows give, bit for bit;
- a reciprocal member copied untransposed in the fused source moves Z, and
  leaves the gathered route's Z alone: the fused path is the one that ran;
- every fill that read no plan before reads none now.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire.bspline as bs
from momwire.bspline import BSplineSolver

pytestmark = [
    pytest.mark.reach(4),
    pytest.mark.skipif(
        bs._acc is None or not hasattr(bs._acc, "bspline_reuse_window_counts"),
        reason="bs2's fused reuse assembly is not built",
    ),
]

LAM = 21.1
#: Whether this build's tiles run the lane kernel (else one source call fills
#: the window, and the windowed kernel's reference loop runs on it).
LANES = bool(getattr(bs._acc, "windowed_lanes_1290", False))
#: Z moves at the key's rounding (2^-40 of a pair's scale), far under the
#: answer's drift (`test_pair_reuse_chunked_2020.py`'s bound).
Z_DRIFT = 1e-11


@pytest.fixture(autouse=True)
def _fresh():
    acc = bs._acc
    acc.bspline_reuse_cache_clear()
    acc.bspline_reuse_set_small_deck_segs(0)
    acc.bspline_reuse_set_window_probe(0)
    yield
    acc.bspline_reuse_set_window_probe(0)
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.bspline_reuse_cache_clear()


def _yagi(D=8, n=21, radius=0.006, **kw):
    """Reflector, driver and D identical directors, 0.2 lambda apart: one
    radius, so reciprocal pairs share classes (code 1, the transpose)."""
    Ld = 0.47 * LAM
    elems = [(1.05 * Ld, n + 1), (Ld, n)] + [(0.95 * Ld, n - 1)] * D
    wires = [
        np.array([[0.2 * LAM * i, -L / 2, 0.0], [0.2 * LAM * i, L / 2, 0.0]])
        for i, (L, _n) in enumerate(elems)
    ]
    kw.setdefault("pair_reuse", True)
    # past the tensor's budget (203 segments: 6 MB): the chunked fill
    kw.setdefault("swept_mem_mb", 1)
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[m] for _L, m in elems],
        feeds=[(1, None, 1.0 + 0j)],
        wavelength=LAM,
        wire_radius=radius,
        **kw,
    )


def _fill(s, **kw):
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    return np.array(s._compute_Z_operator(geom, supp, polys, **kw))


def _moved(Z, ref):
    return float(np.max(np.abs(Z - ref)) / np.max(np.abs(ref)))


def _counts():
    return dict(bs._acc.bspline_reuse_window_counts())


def _ran(before, after):
    """Windows the fused assembly took, by the kernel's own count."""
    key = "tiled" if LANES else "whole"
    return after[key] - before[key]


class _Gathered:
    """A plan whose `assemble` is the route it replaced: `gather` the window,
    overwrite its near same-edge singles with `own`, `assemble_Z_bspline_windowed`
    (bs2's `_reuse_block` before momwire#2166 in what it computes)."""

    def __init__(self, plan, calls):
        self._plan = plan
        self._calls = calls

    def __getattr__(self, name):
        return getattr(self._plan, name)

    def assemble(
        self, values, r0, r1, own, lo, hi, supp, polys, tan, m_idx, n_idx, *rest
    ):
        omega, eps, mu, Z, cancel = rest
        d1 = values.shape[1]
        n = tan.shape[0]
        J = np.empty((d1, d1, r1 - r0, n), dtype=np.complex128)
        self._plan.gather(values, r0, r1, J)
        if own is not None:
            for i in range(r1 - r0):
                if hi[i] <= lo[i]:
                    continue
                near = self._plan.near_same_edge_mask(r0 + i, r0 + i + 1, lo[i], hi[i])[
                    0
                ]
                cols = lo[i] + np.flatnonzero(near)
                J[:, :, i, cols] = own[:, :, i, cols - lo[i]]
        bs._acc.assemble_Z_bspline_windowed(
            J, supp, polys, tan, m_idx, n_idx, r0, r1, 0, n, omega, eps, mu, Z, cancel
        )
        self._calls.append((r0, r1))


_REAL_REUSE_PLAN = BSplineSolver._reuse_plan


@pytest.fixture
def gathered(monkeypatch):
    """Route every plan the fill reads through `_Gathered`."""
    calls = []
    real = BSplineSolver._reuse_plan

    def wrap(self, geom):
        plan = real(self, geom)
        return None if plan is None else _Gathered(plan, calls)

    monkeypatch.setattr(BSplineSolver, "_reuse_plan", wrap)
    return calls


def _window_inputs(s):
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    plan = s._reuse_plan(geom)
    assert plan is not None, s.pair_reuse_declined
    ek = s._ek_spec(geom) if s.extended_kernel else None
    ladder = s._fill_ladder(s.k, geom["seg_l"], geom["seg_r"], ek)
    values = s._reuse_values(plan, geom, s.k, ladder, ek)
    supp = np.ascontiguousarray(supp, dtype=np.int64)
    polys = np.ascontiguousarray(polys, dtype=np.float64)
    tan = np.ascontiguousarray(geom["tangents"], dtype=np.float64)
    return geom, plan, values, supp, polys, tan, ladder, ek


@pytest.mark.parametrize("own", ["moments", "classes"])
def test_a_window_assembled_is_the_gathered_windows_sum_to_the_bit(own):
    """Rows 5..17 of the driver's edge, its near same-edge singles the fill's
    moments (or the class values), added into a Z already holding numbers:
    `plan.assemble` adds exactly what the gathered window adds. The wires
    are thicker than their segments are long (h < a), so the near singles
    have a band (the hairpin bridge's case)."""
    s = _yagi(radius=0.6)
    geom, plan, values, supp, polys, tan, ladder, ek = _window_inputs(s)
    sl = s._same_edge_slices(geom)[1]
    r0, r1 = sl.start + 5, sl.start + 17
    band_lo, band_hi = plan.same_edge_bands()
    lo = np.ascontiguousarray(band_lo[r0:r1])
    hi = np.ascontiguousarray(band_hi[r0:r1])
    b0, b1 = int(lo.min()), int(hi.max())
    assert sl.start <= b0 < b1 <= sl.stop
    own_J = None
    if own == "moments":
        own_J = bs._seg_seg_full_moments_offedge(
            geom["seg_l"][r0:r1],
            geom["seg_r"][r0:r1],
            geom["seg_l"][b0:b1],
            geom["seg_r"][b0:b1],
            s._seg_radius(geom)[r0:r1],
            s.k,
            s.degree,
            s.n_qp_pair,
            ek=bs._ek_slice(ek, rows=slice(r0, r1), cols=slice(b0, b1)),
            ladder=ladder,
        )
        band = np.zeros((3, 3, r1 - r0, int((hi - lo).max())), dtype=np.complex128)
        for i in range(r1 - r0):
            band[:, :, i, : hi[i] - lo[i]] = own_J[:, :, i, lo[i] - b0 : hi[i] - b0]
        own_J = band
    nb = supp.shape[0]
    m_idx = np.nonzero(((supp >= r0) & (supp < r1)).any(axis=1))[0].astype(np.int64)
    all_n = np.arange(nb, dtype=np.int64)
    rng = np.random.default_rng(2166)
    Z0 = np.asfortranarray(
        rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
    )
    args = (float(s.omega), float(s.eps), float(s.mu))
    ref = Z0.copy(order="F")
    calls = []
    _Gathered(plan, calls).assemble(
        values, r0, r1, own_J, lo, hi, supp, polys, tan, m_idx, all_n, *args, ref, 0
    )
    got = Z0.copy(order="F")
    before = _counts()
    plan.assemble(
        values, r0, r1, own_J, lo, hi, supp, polys, tan, m_idx, all_n, *args, got
    )
    assert _ran(before, _counts()) == 1
    assert got.tobytes() == ref.tobytes()
    assert not np.array_equal(got, Z0)


@pytest.mark.parametrize("swept_mem_mb", [1, 2])
@pytest.mark.parametrize("ek", [False, True])
def test_the_fill_is_the_gathered_fill_to_the_bit(ek, swept_mem_mb, monkeypatch):
    """The chunked fill with the plan: every observer window through the
    fused assembly (the kernel's count), and Z the gathered windows' to the
    bit; the windows are the sweep's, cut inside the edges."""
    kw = {"extended_kernel": ek, "swept_mem_mb": swept_mem_mb}
    before = _counts()
    Z = _fill(_yagi(**kw))
    pieces = _ran(before, _counts())
    calls = []
    real = BSplineSolver._reuse_plan

    def wrap(self, geom):
        plan = real(self, geom)
        return None if plan is None else _Gathered(plan, calls)

    with monkeypatch.context() as m:
        m.setattr(BSplineSolver, "_reuse_plan", wrap)
        ref = _fill(_yagi(**kw))
    assert pieces == len(calls) >= 2
    assert Z.tobytes() == ref.tobytes()
    assert (
        _moved(Z, _fill(_yagi(pair_reuse=False, offset_reuse=False, **kw))) <= Z_DRIFT
    )


def test_degree_one_is_the_gathered_fill_too(gathered):
    """The tiled assembler switches on the degree (`max_d` 1, 2 or 3): at
    d = 1 each entry has 2 x 2 wings and (p, q) up to 1."""
    ref = _fill(_yagi(degree=1))
    n_gathered = len(gathered)
    gathered.clear()
    with pytest.MonkeyPatch.context() as m:
        m.setattr(BSplineSolver, "_reuse_plan", _REAL_REUSE_PLAN)
        before = _counts()
        Z = _fill(_yagi(degree=1))
        assert _ran(before, _counts()) == n_gathered >= 2
    assert Z.tobytes() == ref.tobytes()


def test_a_member_copied_untransposed_moves_only_the_fused_fill(gathered):
    """The adversarial gate: the probe copies a reciprocal member's class
    value without its transpose, in the fused source only. The fused fill
    moves far past the key's rounding; the gathered route, under the same
    probe, is its own bits. So the fill's Z is the fused path's, and a fused
    path that dropped the transpose would be seen."""
    ref = _fill(_yagi())  # the gathered route (fixture)
    assert gathered
    bs._acc.bspline_reuse_set_window_probe(1)
    assert _fill(_yagi()).tobytes() == ref.tobytes()


def test_the_probe_reaches_the_fused_fill():
    ref = _fill(_yagi())
    bs._acc.bspline_reuse_set_window_probe(1)
    before = _counts()
    spoiled = _fill(_yagi())
    assert _ran(before, _counts()) > 0
    assert _moved(spoiled, ref) > 1e3 * Z_DRIFT


def test_reuse_off_assembles_nothing_from_a_plan():
    before = _counts()
    s = _yagi(pair_reuse=False)
    _fill(s)
    assert _counts() == before
    assert s.pair_reuse_served is None


def test_a_fill_of_chosen_rows_still_reads_no_plan():
    s = _yagi()
    before = _counts()
    _fill(s, rows=np.arange(22))  # the reflector's segments
    assert _counts() == before
    assert s.pair_reuse_declined == "a fill of chosen rows does not read the hint"


def test_a_small_deck_still_declines():
    bs._acc.bspline_reuse_set_small_deck_segs(-1)
    s = _yagi()  # 8 + 22 + 21 + 8 * 20 = 203 segments: two directors fewer is below
    s2 = _yagi(D=6)
    before = _counts()
    _fill(s2)
    assert _counts() == before
    assert s2.pair_reuse_declined == "too few segments to repay the plan (N < 200)"
    _fill(s)
    assert _ran(before, _counts()) > 0


def test_a_window_out_of_range_is_refused():
    s = _yagi()
    geom, plan, values, supp, polys, tan, *_ = _window_inputs(s)
    sl = s._same_edge_slices(geom)[1]
    nb = supp.shape[0]
    n = tan.shape[0]
    Z = np.zeros((nb, nb), dtype=complex)
    idx = np.arange(nb, dtype=np.int64)
    args = (float(s.omega), float(s.eps), float(s.mu), Z)
    with pytest.raises(ValueError, match="rows out of range"):
        plan.assemble(
            values, n - 2, n + 1, None, None, None, supp, polys, tan, idx, idx, *args
        )
    bad = np.zeros((3, 3, 4, 5), dtype=complex)
    lo = np.full(3, sl.start, dtype=np.int64)  # three rows' bands for four rows
    with pytest.raises(ValueError, match="own is"):
        plan.assemble(
            values,
            sl.start,
            sl.start + 4,
            bad,
            lo,
            lo + 5,
            supp,
            polys,
            tan,
            idx,
            idx,
            *args,
        )
    with pytest.raises(ValueError, match="values are"):
        plan.assemble(
            values[:-1],
            sl.start,
            sl.start + 4,
            None,
            None,
            None,
            supp,
            polys,
            tan,
            idx,
            idx,
            *args,
        )


def test_the_windows_are_the_sweeps():
    """`_reuse_block` takes the sweep's observer windows whole (momwire#2063
    step 3: no same-edge column is cut out, so no window is cut at the
    edges): one fused call per window."""
    s = _yagi()
    geom = s._build_geometry()
    n = int(geom["n_segs_total"])
    chunk = bs._schedule.mb_rows(s.swept_mem_mb, 9 * n * 16)
    before = _counts()
    _fill(s)
    assert _ran(before, _counts()) == len(range(0, n, chunk)) >= 2
