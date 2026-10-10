"""The exact-kernel correction's base moments, deduplicated (momwire#1421).

`_exact_kernel_base.BaseRows` returns the windows' base moments -- what the
free-space fill put on a coaxial group's pairs -- from one evaluation per
diagonal of each uniform run-pair block instead of one per pair. Gates:

* Z with the dedup equals Z without it (`_EXACT_KERNEL_BASE_DEDUP = False`,
  the per-pair `_exact_kernel_base_J`) to a derived tolerance, on #1409's
  and #1411's decks (EK off and on, degree 2 and 3, an end-port pair, a bent
  wire, a collinear junction), a reversed segment run, a piecewise-uniform
  wire, a graded wire and two parallel wires, at window heights down to one
  row, with the gather forced onto every block (`TOEPLITZ_MIN_PAIRS = 0`)
  and at the production threshold;
* the dedup ran, counted from the production seam: on a uniform run the
  kernel sees 2n - 1 pairs over all windows, not n²;
* pairs that share lengths and offset but differ in a per-segment field the
  moment depends on (an EK group label, a tube radius) keep their own
  values: the run boundary sees the field;
* the representative pairs' same-edge geometry is the square table's;
* memory: a graded wire, where nothing deduplicates, still peaks at one
  window.

TOLERANCE. A gathered moment is its diagonal representative's: the same
kernel on a pair translated along the axis, so the two differ by the
coordinates' roundoff -- |x|·eps over distances no smaller than the radius,
a relative ~|x|·eps/a per moment, 2.2e-16 · 0.5 / 0.004 ≈ 3e-14 on these
decks -- and by a length difference inside the run resolution `q` =
1e-11·h, which uniform meshes never use (their lengths agree to roundoff).
Measured max|ΔZ|/max|Z| on Haswell: <= 7.5e-15 over every deck here (the
cross-wire gathers of the end-port pair are the largest), 3e-15 on a
601-segment dipole, 1.2e-14 on a 3,000-segment 9 m wire (|x|/a = 4,500).
Gated at 1e-13, ten times inside Steve's 1e-12 bar for the exact-kernel
routes: a dropped run field reads 1.4e-11 (a 5e-7 radius step) to 1.4e-5
(an EK label), a mis-offset window or an ignored reversal 0.09 and up.
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire import _bspline_kernels as BK
from momwire import _exact_kernel_base as EKB
from momwire import bspline as _bspline

TOL_Z = 1e-13

L = 0.47
A = 0.00425
DIPOLE = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]


def _halves(w, *, flip_second=False):
    top = np.array([(0.0, 0.0, w / 2), (0.0, 0.0, L / 2)])
    return [
        np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, -w / 2)]),
        top[::-1] if flip_second else top,
    ]


def _graded(n, lo=0.004, hi=0.02):
    h = np.r_[np.geomspace(hi, lo, n // 2), np.geomspace(lo, hi, n - n // 2)]
    z = np.r_[0.0, np.cumsum(h)]
    return np.stack([np.zeros_like(z), np.zeros_like(z), z], -1)


_DECKS = {
    "dipole": dict(n=[[37]]),
    "dipole-segment-gap": dict(n=[[37]], feed_model="segment"),
    "end-port pair": dict(
        wires=_halves(L / 75),
        n=[[37], [37]],
        feeds=[(0, (L - L / 75) / 4, 0j)],
        junctions=[[(0, "end")], [(1, "start")]],
        junction_ports=[(0, 0j), (1, 0j)],
    ),
    "reversed second wire": dict(
        wires=_halves(L / 75, flip_second=True), n=[[37], [37]]
    ),
    "collinear junction + bend": dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.2)]),
            np.array([(0.0, 0.0, 0.2), (0.0, 0.0, 0.3), (0.1, 0.0, 0.4)]),
        ],
        n=[[19], [9, 13]],
        junctions=[[(0, "end"), (1, "start")]],
    ),
    "bent wire": dict(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.15, 0.0, 0.40)])],
        n=[[23, 17]],
    ),
    # three collinear edges, h = 0.01 / 0.02 / 0.01: Toeplitz blocks inside
    # each edge, between the two outer ones, and direct pairs between them
    "piecewise uniform": dict(
        wires=[
            np.array(
                [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.0, 0.3), (0.0, 0.0, 0.4)]
            )
        ],
        n=[[10, 10, 10]],
    ),
    "graded": dict(wires=[_graded(40)], n=[[1] * 40]),
    "two parallel wires": dict(
        wires=[DIPOLE[0], DIPOLE[0] + np.array([0.05, 0.0, 0.0])],
        n=[[31], [31]],
    ),
    "degree 3": dict(n=[[29]], degree=3),
}


def _solver(wires=DIPOLE, n=None, a=A, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=n,
            wavelength=1.0,
            wire_radius=a,
            exact_kernel=True,
            **kw,
        )


def _Z(s):
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array(s._compute_Z_operator(geom, supp_seg, polys))


def _rel(got, want):
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def _spy(monkeypatch):
    """The BaseRows the production seam builds, recorded."""
    made = []
    init = EKB.BaseRows.__init__

    def spy(self, **kw):
        init(self, **kw)
        made.append(self)

    monkeypatch.setattr(EKB.BaseRows, "__init__", spy)
    return made


def _dedup_vs_reference(monkeypatch, s, rows=None):
    """(rel ΔZ, the BaseRows the dedup route built)."""
    made = _spy(monkeypatch)
    ref_calls = []
    orig = BSplineSolver._exact_kernel_base_J

    def ref_spy(self, *a, **kw):
        ref_calls.append(1)
        return orig(self, *a, **kw)

    monkeypatch.setattr(BSplineSolver, "_exact_kernel_base_J", ref_spy)
    if rows is not None:
        monkeypatch.setattr(_bspline, "_EXACT_KERNEL_ROWS_OVERRIDE", rows)
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_BASE_DEDUP", False)
    Z0 = _Z(s)
    assert ref_calls and not made, "the reference route did not run"
    n_ref = len(ref_calls)
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_BASE_DEDUP", True)
    Z1 = _Z(s)
    assert made and len(ref_calls) == n_ref, "the dedup route did not run"
    return _rel(Z1, Z0), made


@pytest.mark.parametrize("rows", [None, 1, 5])
@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("deck", sorted(_DECKS))
def test_dedup_matches_the_per_pair_base(monkeypatch, deck, ek, rows):
    monkeypatch.setattr(EKB, "TOEPLITZ_MIN_PAIRS", 0)
    s = _solver(extended_kernel=ek, **_DECKS[deck])
    rel, made = _dedup_vs_reference(monkeypatch, s, rows)
    assert rel <= TOL_Z
    if deck != "graded":
        assert sum(b.toeplitz_blocks for b in made) > 0, "nothing was gathered"


@pytest.mark.parametrize("deck", ["end-port pair", "piecewise uniform", "degree 3"])
def test_dedup_at_the_production_threshold(monkeypatch, deck):
    s = _solver(extended_kernel=True, **_DECKS[deck])
    rel, _made = _dedup_vs_reference(monkeypatch, s, 3)
    assert rel <= TOL_Z


@pytest.mark.parametrize("ek", [False, True])
def test_uniform_run_kernel_work_is_linear(monkeypatch, ek):
    """From the production seam, at the production threshold: a uniform
    601-segment wire cut into 16-row windows hands its kernels 2n - 1 pairs,
    and the per-pair route n²."""
    n = 601
    s = _solver(n=[[n]], extended_kernel=ek)
    rel, made = _dedup_vs_reference(monkeypatch, s, 16)
    assert rel <= TOL_Z
    (base,) = made
    assert base.toeplitz_blocks >= n // 16
    assert base.evaluated == 2 * n - 1, base.evaluated


def test_two_wire_kernel_work_is_linear(monkeypatch):
    """Two collinear 301-segment wires: the same-edge blocks and the
    off-edge blocks between the wires all gather, O(n) pairs in all."""
    n = 301
    s = _solver(wires=_halves(L / 75), n=[[n], [n]], extended_kernel=True)
    # 43-row windows: seven to a wire, so no window holds a fragment too
    # short to gather
    rel, made = _dedup_vs_reference(monkeypatch, s, 43)
    assert rel <= TOL_Z
    (base,) = made
    # per wire 2n - 1 same-edge, per ordered wire pair 2n - 1 off-edge
    assert base.evaluated == 4 * (2 * n - 1), base.evaluated


# --------------------------------------------------------------------------
# collisions: the same lengths and offset, a different field
# --------------------------------------------------------------------------
def _relabelled(s, split):
    """EK labels that change mid-wire: segments from `split` on are their own
    group, so a pair across `split` is not extended and a pair on one side
    is. The fill reads the same labels (`_ek_spec`), so this is a mesh the
    base must reproduce, not a mismatch it is asked to absorb."""
    orig = s._ek_axis_labels

    def labels(geom, mirror):
        gi, gj = orig(geom, mirror)
        gi, gj = np.array(gi), np.array(gj)
        gi[split:] += 1
        gj[split:] += 1
        return gi, gj

    return labels


def test_ek_group_change_keeps_its_own_values(monkeypatch):
    monkeypatch.setattr(EKB, "TOEPLITZ_MIN_PAIRS", 0)
    s = _solver(extended_kernel=True, **_DECKS["end-port pair"])
    # wire 0 is segments 0..36: relabel from 20 on
    monkeypatch.setattr(s, "_ek_axis_labels", _relabelled(s, 20))
    rel, made = _dedup_vs_reference(monkeypatch, s, 5)
    (base,) = made
    assert rel <= TOL_Z
    assert len(base.runs) == 3, base.runs  # the label change split wire 0
    # the field matters: the same geometry, extended vs not, differs
    geom = s._build_geometry()
    (S, g_edges), *_ = s._exact_kernel_groups(geom)
    J = s._exact_kernel_base_J(geom, S, g_edges, s.k)
    # (19, 40) and (20, 41): one diagonal, offset 21 segments (both cross
    # the gap); only the second is a same-label pair
    assert abs(J[0, 0, 19, 40] - J[0, 0, 20, 41]) > 1e-6 * abs(J[0, 0, 19, 40])


def test_tube_radius_change_keeps_its_own_values(monkeypatch):
    """Two collinear wires whose radii differ by 1e-7: one exact-kernel group
    (the coaxial rule's tolerance is 1e-6), two observer radii and, under
    EK, two source-tube radii b_j (momwire#1368)."""
    monkeypatch.setattr(EKB, "TOEPLITZ_MIN_PAIRS", 0)
    deck = dict(_DECKS["end-port pair"])
    s = _solver(extended_kernel=True, a=[A, A * (1 + 1e-7)], **deck)
    rel, made = _dedup_vs_reference(monkeypatch, s, 5)
    (base,) = made
    assert len(base.runs) == 2
    assert rel <= TOL_Z


def test_radius_change_inside_an_edge_keeps_its_own_values(monkeypatch):
    """A radius is per wire, so in a real deck it changes only where an edge
    does, and the edge boundary alone would split the run. Here it changes
    mid-wire (by 5e-7, inside the coaxial rule's 1e-6 tolerance): the observer
    radius `a` and, under EK, the tube radius `b_j` must split the run on
    their own. The fill reads the same radii (`_seg_radius`)."""
    monkeypatch.setattr(EKB, "TOEPLITZ_MIN_PAIRS", 0)
    s = _solver(extended_kernel=True, **_DECKS["end-port pair"])
    orig = s._seg_radius

    def seg_radius(geom):
        a = np.array(orig(geom), dtype=np.float64)
        a[20:] *= 1 + 5e-7
        return a

    monkeypatch.setattr(s, "_seg_radius", seg_radius)
    rel, made = _dedup_vs_reference(monkeypatch, s, 5)
    (base,) = made
    assert rel <= TOL_Z
    assert len(base.runs) == 3, base.runs


# --------------------------------------------------------------------------
# the representative pairs' same-edge geometry
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ek", [None, _bspline._EK_SAME_EDGE])
def test_pair_geometry_is_the_square_tables(ek):
    arc = np.r_[0.0, np.cumsum(np.geomspace(0.01, 0.02, 23))]
    nodes = BK._seg_seg_reg_nodes(arc, 2, 4)
    sq = BK._seg_seg_reg_geometry(arc, A, 2, 4, ek=ek, rows=slice(5, 12))
    pg = BK._seg_seg_reg_geometry_pairs(
        nodes, A, 4, np.arange(5, 12), np.arange(23), ek=ek
    )
    assert np.array_equal(pg["R"], sq["R"])
    assert np.array_equal(pg["wu_row"], sq["wu_row"])
    assert np.array_equal(pg["wu_pow"], sq["wu_pow"])
    k = 2 * np.pi
    assert np.array_equal(
        BK._seg_seg_reg_moments_from_geometry(pg, k),
        BK._seg_seg_reg_moments_from_geometry(sq, k),
    )
    # a scattered subset is the square answer's entries
    rows, cols = np.array([3, 17, 9]), np.array([0, 22, 5, 6])
    sub = BK._seg_seg_reg_moments_from_geometry(
        BK._seg_seg_reg_geometry_pairs(nodes, A, 4, rows, cols, ek=ek), k
    )
    full = BK._seg_seg_reg_moments_from_geometry(
        BK._seg_seg_reg_geometry(arc, A, 2, 4, ek=ek), k
    )
    want = full[:, :, rows[:, None], cols[None, :]]
    assert np.max(np.abs(sub - want)) <= 1e-15 * np.max(np.abs(want))


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------
def _traced_correction(monkeypatch, s):
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    orig = BSplineSolver._add_exact_kernel_correction
    peak = []

    def traced(self, *args):
        tracemalloc.start()
        tracemalloc.reset_peak()
        at_entry = tracemalloc.get_traced_memory()[0]
        try:
            return orig(self, *args)
        finally:
            peak.append((tracemalloc.get_traced_memory()[1] - at_entry) / 2**20)
            tracemalloc.stop()

    monkeypatch.setattr(BSplineSolver, "_add_exact_kernel_correction", traced)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s._compute_Z_operator(geom, supp_seg, polys)
    assert len(peak) == 1, "the correction did not run"
    return peak[0]


# A graded 1,200-segment wire at a 16 MB budget: every segment its own run,
# so nothing gathers and every pair is recomputed directly, one call per
# window as before the dedup. Traced through the production seam. The bar is
# the uniform-wire memgate's (tests/test_exact_kernel_rows_1411.py).
@pytest.mark.memgate
def test_graded_wire_base_peak_is_one_window(monkeypatch):
    n = 1200
    s = _solver(wires=[_graded(n, 0.002, 0.02)], n=[[1] * n], a=0.002, swept_mem_mb=16)
    made = _spy(monkeypatch)
    peak = _traced_correction(monkeypatch, s)
    (base,) = made
    assert base.toeplitz_blocks == 0 and len(base.runs) == n
    assert base.evaluated >= n * (n - 1)  # nothing deduplicated
    assert peak <= 48


# The uniform 1,500-segment wire of the #1411 memgate, at the same 16 MB
# budget, with the base dedup and its cache: still one window.
@pytest.mark.memgate
def test_uniform_wire_base_peak_is_one_window(monkeypatch):
    s = _solver(
        wires=[np.array([(0.0, 0.0, -2.25), (0.0, 0.0, 2.25)])],
        n=[[1500]],
        swept_mem_mb=16,
    )
    made = _spy(monkeypatch)
    peak = _traced_correction(monkeypatch, s)
    (base,) = made
    assert base.evaluated == 2 * 1500 - 1  # O(n)
    assert peak <= 48
