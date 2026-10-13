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
  and at the production threshold, where a deck too small to gather is
  bit-identical;
* the dedup ran, counted from the production seam: on a uniform run the
  kernel sees 2n - 1 pairs over all windows, not n², also far from the
  origin, where the mesh's own lengths scatter most;
* runs whose pitches differ by more than coordinate roundoff do not gather
  with each other, however small the difference beside the group's longest
  segment (the mixed-pitch family), and a run is an arithmetic progression,
  not only a row of nearly equal lengths;
* pairs that share lengths and offset but differ in a per-segment field the
  moment depends on (an EK group label, a radius, a length inside an edge)
  keep their own values: the run boundary sees the field, and each test has
  a red twin that drops the field from `RUN_FIELDS` and must fail the gate;
* a diagonal sitting on a ladder threshold (16, the buried 2, a coarse tier)
  or a run on the phase guard's ceiling is evaluated pair by pair, through
  the production seam on the C++ and the numpy off-edge kernels, with a red
  twin (`TIE_REL = 0`);
* the representative pairs' same-edge geometry is the square table's;
* memory: a graded wire, where nothing deduplicates, still peaks at one
  window.

TOLERANCE (`_tol`). A gathered moment is its diagonal representative's: the
same kernel on two segments that are the pair's own, translated. On a mesh
`_build_geometry` made, the stored endpoints sit within e = 2.3·eps·X of an
exact arithmetic progression (X the largest |coordinate|; measured over 37
to 4,000 segments up to 34 m from the origin, see `_exact_kernel_base`), so
the representative differs from the pair in two ways:

* each segment's computed length by up to ~e, and a moment carries the
  Jacobian h_i·h_j and the local coordinates u/h: a relative 2·e/h;
* any separation the kernel reads by up to 4·e (four endpoints), and a
  moment's sensitivity to a shift δ of one segment against the other is at
  most δ/max(h, a) of the self moment: ∂/∂z of 1/R integrates to
  ln(4h/a)/4π over two touching segments against a self moment of
  2h·ln(2h/a)/4π for h >> a, and to 0.39·δ/a for h << a. A relative 4·e/h
  at most.

Together 6·e/h ≈ 14·eps·X/h_min in the worst alignment; the gate is
TOL_EPS = 16 of u = eps·X/h_min per deck, capped at Steve's 1e-12 bar for
the exact-kernel routes (the cap binds from X/h_min ≈ 280: the 601-segment
dipole, the mixed-pitch deck). It is the segment length and not the radius
that sets the scale. Measured max|ΔZ|/max|Z| on Haswell: at most 0.19 u on
the decks here (1.6e-15 on the reversed second wire with the gather forced;
6.0e-14 on the mixed-pitch deck, the largest reading in this file), 0.02 u
on a 3,000-segment 9 m wire (6.7e-15), and 0.5 u on two 100-segment wires
25 and 100 times finer than their radius (2.4e-14 and 4.6e-14, which are 13
and 51 times |x|·eps/a: a bound on the radius does not hold there). The red
twins all read above their deck's gate: 1.4e-12 and 2.0e-12 with the tie
rule off and 1.2e-12 with the phase-guard tie unseen against 5.0e-13,
1.4e-11 and 1.4e-5 with a field dropped against 1.3e-13, 4.4e-12 and
1.8e-11 on the mixed-pitch deck without the drift bound against 1e-12. Two
more red controls were run by hand on Haswell and are not in the suite,
since they need a source edit: a window-relative row offset in the gather
(43 tests fail, ΔZ 0.03 to 1.04) and an ignored reversal (6 fail, 0.16).
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

# The gate, in units of eps·X/h_min of the deck, and its cap (the module
# docstring's TOLERANCE).
TOL_EPS = 16
TOL_BAR = 1e-12


def _tol_of(seg_l, seg_r):
    X = max(np.max(np.abs(seg_l)), np.max(np.abs(seg_r)))
    h_min = np.min(np.linalg.norm(seg_r - seg_l, axis=1))
    return float(min(TOL_BAR, TOL_EPS * np.finfo(np.float64).eps * X / h_min))


def _tol(s):
    """The gate of solver `s`'s deck: 16·eps·X/h_min, at most 1e-12."""
    geom = s._build_geometry()
    return _tol_of(geom["seg_l"], geom["seg_r"])


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
    assert rel <= _tol(s)
    if deck != "graded":
        assert sum(b.toeplitz_blocks for b in made) > 0, "nothing was gathered"


@pytest.mark.parametrize("rows", [None, 3])
@pytest.mark.parametrize("deck", sorted(_DECKS))
def test_small_decks_are_bit_identical_at_the_production_threshold(
    monkeypatch, deck, rows
):
    """No block of these decks reaches the gather threshold, so every pair
    takes the per-pair route's own calls and Z is the reference's bit for
    bit (`rel` is max|ΔZ|, so 0.0 is array equality)."""
    s = _solver(extended_kernel=True, **_DECKS[deck])
    rel, made = _dedup_vs_reference(monkeypatch, s, rows)
    assert sum(b.toeplitz_blocks for b in made) == 0
    assert rel == 0.0


@pytest.mark.parametrize("ek", [False, True])
def test_uniform_run_kernel_work_is_linear(monkeypatch, ek):
    """From the production seam, at the production threshold: a uniform
    601-segment wire cut into 16-row windows hands its kernels 2n - 1 pairs,
    and the per-pair route n²."""
    n = 601
    s = _solver(n=[[n]], extended_kernel=ek)
    rel, made = _dedup_vs_reference(monkeypatch, s, 16)
    assert rel <= _tol(s)
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
    assert rel <= _tol(s)
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


def _drop_fields(monkeypatch, drop):
    """The red control: run boundaries blind to the `drop` fields."""
    if drop:
        kept = tuple(f for f in EKB.RUN_FIELDS if f not in drop)
        monkeypatch.setattr(EKB, "RUN_FIELDS", kept)


# Red with the label fields dropped: measured 1.4e-5 on Haswell.
@pytest.mark.parametrize("drop", [None, ("group_i", "group_j")])
def test_ek_group_change_keeps_its_own_values(monkeypatch, drop):
    monkeypatch.setattr(EKB, "TOEPLITZ_MIN_PAIRS", 0)
    _drop_fields(monkeypatch, drop)
    s = _solver(extended_kernel=True, **_DECKS["end-port pair"])
    # wire 0 is segments 0..36: relabel from 20 on
    monkeypatch.setattr(s, "_ek_axis_labels", _relabelled(s, 20))
    rel, made = _dedup_vs_reference(monkeypatch, s, 5)
    (base,) = made
    if drop:
        assert rel > _tol(s), "the gate cannot see a dropped label field"
        return
    assert rel <= _tol(s)
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
    assert rel <= _tol(s)


# Red with the radius fields dropped: measured 1.4e-11 on Haswell.
@pytest.mark.parametrize("drop", [None, ("a", "b_j")])
def test_radius_change_inside_an_edge_keeps_its_own_values(monkeypatch, drop):
    """A radius is per wire, so in a real deck it changes only where an edge
    does, and the edge boundary alone would split the run. Here it changes
    mid-wire (by 5e-7, inside the coaxial rule's 1e-6 tolerance): the observer
    radius `a` and, under EK, the tube radius `b_j` must split the run on
    their own. The fill reads the same radii (`_seg_radius`)."""
    monkeypatch.setattr(EKB, "TOEPLITZ_MIN_PAIRS", 0)
    _drop_fields(monkeypatch, drop)
    s = _solver(extended_kernel=True, **_DECKS["end-port pair"])
    orig = s._seg_radius

    def seg_radius(geom):
        a = np.array(orig(geom), dtype=np.float64)
        a[20:] *= 1 + 5e-7
        return a

    monkeypatch.setattr(s, "_seg_radius", seg_radius)
    rel, made = _dedup_vs_reference(monkeypatch, s, 5)
    (base,) = made
    if drop:
        assert rel > _tol(s), "the gate cannot see a dropped radius field"
        return
    assert rel <= _tol(s)
    assert len(base.runs) == 3, base.runs


# A length change INSIDE one edge. The deck front ends cannot make one: every
# edge is meshed `np.linspace(0, edge_len, n_e + 1)` (`_build_geometry`), so a
# length changes only where an edge does and the "edge" field already splits
# the run. BaseRows takes its geometry as arrays, so a two-pitch edge is
# built here directly and the gather is held against BaseRows' own per-pair
# route (every block below the threshold), which is `_exact_kernel_base_J`'s
# arithmetic. Red with "h" dropped from the run fields.
@pytest.mark.parametrize("drop", [None, ("h",)])
def test_length_change_inside_an_edge_keeps_its_own_values(monkeypatch, drop):
    _drop_fields(monkeypatch, drop)
    h = np.r_[np.full(30, 0.01), np.full(30, 0.013)]
    arc = np.r_[0.0, np.cumsum(h)]
    z = np.array([0.0, 0.0, 1.0])
    seg_l, seg_r = arc[:-1, None] * z, arc[1:, None] * z
    n = h.size

    def rows(min_pairs):
        monkeypatch.setattr(EKB, "TOEPLITZ_MIN_PAIRS", min_pairs)
        base = EKB.BaseRows(
            seg_l=seg_l,
            seg_r=seg_r,
            seg_a=np.full(n, A),
            tangents=np.tile(z, (n, 1)),
            S=np.arange(n),
            g_edges=[(slice(0, n), arc, A)],
            k=2 * np.pi,
            d=2,
            n_qp_pair=8,
            n_qp_same_edge=4,
            ek=None,
            ladder=_bspline.DEFAULT_PAIR_ORDER_LADDER,
            ek_se=None,
            n_segs_total=n,
        )
        out = np.concatenate(
            [base.rows(r0, min(n, r0 + 7)) for r0 in range(0, n, 7)], 2
        )
        return out, base

    want, ref = rows(10**12)
    got, base = rows(0)
    assert ref.toeplitz_blocks == 0 and base.toeplitz_blocks > 0
    rel = _rel(got, want)
    tol = _tol_of(seg_l, seg_r)
    if drop:
        assert rel > tol, "the gate cannot see a dropped length field"
        return
    assert len(base.runs) == 2
    assert rel <= tol


# --------------------------------------------------------------------------
# the drift bound: what "the same step" means
# --------------------------------------------------------------------------
# One wire of three edges, 20 x 5 cm, 200 x 1 mm and 200 x 1 mm·(1 + δ), thin
# (a = 0.3 mm), EK on, at the production threshold (the #1421 review's deck).
# A pair s steps along a diagonal of the block between the two fine edges
# sits s·δ·1 mm off its representative's separation, up to 199 steps. The
# first rule held lengths to 1e-11 of the group's LONGEST segment (5e-13 m
# here), so it gathered that block up to δ = 5e-10: measured ΔZ 4.4e-12 at
# δ = 1e-10 and 1.8e-11 at 4e-10 (6.0e-14 at δ = 0), outside the 1e-12 bar.
# Under the drift bound (ρ = 16·eps·X = 5e-15 m here) the block gathers only
# while 199·δ·1 mm <= ρ, δ <= 2.5e-14, where the two pitches differ by less
# than the coordinates resolve; past it the two fine edges still gather
# within themselves and their mutual pairs are evaluated one by one.
_MIXED_GATHERS = {0.0: True, 1e-14: True, 1e-12: False, 1e-10: False, 4e-10: False}


def _mixed_pitch(delta):
    z = np.array([0.0, 1.0, 1.2, 1.2 + 0.2 * (1 + delta)])
    wire = np.stack([np.zeros(4), np.zeros(4), z], -1)
    return _solver(wires=[wire], n=[[20, 200, 200]], a=0.0003, extended_kernel=True)


@pytest.mark.parametrize("delta", sorted(_MIXED_GATHERS))
def test_runs_of_different_pitch_do_not_gather(monkeypatch, delta):
    s = _mixed_pitch(delta)
    rel, made = _dedup_vs_reference(monkeypatch, s)
    assert rel <= _tol(s)
    (base,) = made
    assert len(base.runs) == 3
    fine = 2 * 200 * 200  # the pairs between the two fine edges
    if _MIXED_GATHERS[delta]:
        # both fine edges and both blocks between them
        assert base.toeplitz_blocks == 4
        assert base.evaluated < fine // 4, base.evaluated
    else:
        assert base.toeplitz_blocks == 2
        assert base.evaluated >= fine, base.evaluated


# Red: the positions rule switched off by a tolerance of 1e6·eps·X (3e-10 m
# here, wider than the first rule's 5e-13), so the fine edges gather across
# their pitch difference again. Measured 4.4e-12 and 1.8e-11 on Haswell.
@pytest.mark.parametrize("delta", [1e-10, 4e-10])
def test_drift_bound_is_load_bearing(monkeypatch, delta):
    monkeypatch.setattr(EKB, "RUN_TOL_EPS", 1e6)
    s = _mixed_pitch(delta)
    rel, made = _dedup_vs_reference(monkeypatch, s)
    assert sum(b.toeplitz_blocks for b in made) == 4
    assert rel > _tol(s), "the gate cannot see a block gathered across pitches"


def test_uniform_run_far_from_the_origin_is_one_run(monkeypatch):
    """An oblique 601-segment wire ~45 m from the origin: its lengths scatter
    by a relative ~1e-12 (coordinate roundoff, eps·X/h), which a relative
    length tolerance at roundoff scale would read as changes of pitch. Under
    the positions rule it is one run and the kernels see 2n - 1 pairs."""
    n = 601
    wire = np.array([(31.3, -17.7, 19.1), (31.7, -18.2, 19.4)])
    s = _solver(wires=[wire], n=[[n]], a=0.001, extended_kernel=True)
    geom = s._build_geometry()
    h = np.linalg.norm(geom["seg_r"] - geom["seg_l"], axis=1)
    assert np.max(np.abs(h - h[0])) > 1e-13 * h[0]  # the scatter is real
    rel, made = _dedup_vs_reference(monkeypatch, s, 16)
    (base,) = made
    assert len(base.runs) == 1
    assert base.evaluated == 2 * n - 1, base.evaluated
    assert rel <= _tol(s)


def test_a_run_is_an_arithmetic_progression():
    """`_progressions` on positions: lengths that all agree within ρ can
    still drift (here two pitches 8·eps·X apart, 300 segments each, 2,400
    eps·X of drift), and such a stretch is cut; a uniform stretch is not."""
    eps = np.finfo(np.float64).eps
    n = 600
    i = np.arange(n + 1.0)
    h0 = 1.0 / n
    z = np.array([0.0, 0.0, 1.0])

    def pieces(arc):
        rho = EKB.RUN_TOL_EPS * eps * arc[-1]
        return EKB._progressions(arc[:-1, None] * z, arc[1:, None] * z, 0, n, rho)

    assert pieces(np.linspace(0.0, 1.0, n + 1)) == [(0, n)]
    # the second half's pitch is 8·eps longer: inside ρ = 16·eps length by
    # length, 2,400·eps of drift over the stretch
    two = i * h0 + np.maximum(i - n // 2, 0.0) * (8 * eps)
    assert pieces(two) == [(0, n // 2), (n // 2, n)]
    # a pitch growing by eps/2 a segment is halved until each piece passes
    cut = pieces(i * h0 + 0.25 * eps * i * i)
    assert len(cut) > 8 and cut[0][0] == 0 and cut[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(cut[:-1], cut[1:]))


# The coaxial rule admits a perpendicular offset up to 1e-6·a, so an edge may
# be tilted inside it and the perpendicular offset then varies along a
# diagonal. Measured on Haswell with the positions rule off (two 100-segment
# wires meeting at a junction, the second tilted to 0.9e-6·a at its far end):
# no change in ΔZ on a = 1 mm, L = 0.3 m (1.9e-15 tilted or not), and 6e-16
# of max|Z| on a = 1 mm, L = 4 mm (segments 25 times finer than the radius);
# the offset enters R² as its square, 1e-12·a² at most. The drift bound
# holds it to ρ regardless: the tilted block does not gather.
def test_tilted_edge_inside_the_coaxial_rule_does_not_gather(monkeypatch):
    a, n = 0.001, 100
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.3)]),
        np.array([(0.0, 0.0, 0.3), (0.9e-6 * a, 0.0, 0.6)]),
    ]
    s = _solver(
        wires=wires,
        n=[[n], [n]],
        a=a,
        extended_kernel=True,
        junctions=[[(0, "end"), (1, "start")]],
    )
    rel, made = _dedup_vs_reference(monkeypatch, s)
    (base,) = made  # one coaxial group, both wires
    assert len(base.runs) == 2
    assert base.toeplitz_blocks == 2  # each wire with itself, not with the other
    assert base.evaluated >= 2 * n * n
    assert rel <= _tol(s)


# --------------------------------------------------------------------------
# ties: a branch keyed on pair geometry, sitting on its threshold
# --------------------------------------------------------------------------
# Collinear equal-length edges put every off-edge ladder ratio (centre
# distance over the longer length) on an integer, so the free-space ladder's
# 16 -- and the buried ladder's 2 -- is met EXACTLY and the kernel's `>=`
# goes either way by roundoff. At kL 0.49 the order-4 tier is far from the
# order-8 base, so a gathered tie diagonal moved Z by 2.0e-12 (one wire, two
# edges) and 1.4e-12 (two wires joined) before the fix (the #1421 review).
_W2 = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 3.5), (0.0, 0.0, 7.0)])]
_TIE_DECKS = {
    "two edges kL 0.49": dict(wires=_W2, n=[[70, 70]], wavelength=0.64),
    "two wires joined": dict(
        wires=[_W2[0][:2], _W2[0][1:]],
        n=[[70], [70]],
        wavelength=0.64,
        junctions=[[(0, "end"), (1, "start")]],
    ),
    # the buried ladder's thresholds on a free-space deck (its order 32)
    "buried ladder": dict(
        wires=_W2,
        n=[[70, 70]],
        wavelength=0.64,
        n_qp_pair=32,
        pair_order_ladder=_bspline.BURIED_PAIR_ORDER_LADDER,
    ),
    # a coarse tier at the 2 threshold, where order 2 is far from order 8
    "coarse tier at 2": dict(
        wires=_W2, n=[[70, 70]], wavelength=0.64, pair_order_ladder=((2.0, 2),)
    ),
    # |k|·h = 0.5 to roundoff (k = 1, h = 0.5 off an offset origin, so the
    # lengths differ in their last bits): the phase guard on its ceiling
    "phase guard ceiling": dict(
        wires=[np.array([(0.0, 0.0, 0.137), (0.0, 0.0, 35.137), (0.0, 0.0, 70.137)])],
        n=[[70, 70]],
        wavelength=2 * np.pi,
    ),
}


def _tie_solver(deck):
    kw = dict(_TIE_DECKS[deck])
    wl = kw.pop("wavelength")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=kw.pop("wires"),
            n_per_edge_per_wire=kw.pop("n"),
            wavelength=wl,
            wire_radius=0.004,
            exact_kernel=True,
            **kw,
        )


def _numpy_offedge(monkeypatch):
    for name in (
        "_HAVE_BSPLINE_ACCEL",
        "_HAVE_BSPLINE_OFFEDGE_TIERED_ACCEL",
        "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL",
        "_HAVE_BSPLINE_OFFEDGE_EK_TIERED_ACCEL",
    ):
        monkeypatch.setattr(BK, name, False)


@pytest.mark.parametrize("rows", [None, 70])
@pytest.mark.parametrize("route", ["accel", "numpy"])
@pytest.mark.parametrize("deck", sorted(_TIE_DECKS))
def test_tie_diagonals_are_evaluated_per_pair(monkeypatch, deck, route, rows):
    """Production threshold, through the production seam, on the C++ and the
    numpy off-edge kernels; default windows (the numpy route's are too short
    to gather at order 32) and 70-row windows, one per edge, which always
    gather (a tie diagonal sits at the end of edge 0, where a shorter window
    would leave too few rows to gather)."""
    if route == "numpy":
        _numpy_offedge(monkeypatch)
    elif not BK._HAVE_BSPLINE_OFFEDGE_TIERED_ACCEL:
        pytest.skip("the C++ off-edge kernels are not built")
    s = _tie_solver(deck)
    rel, made = _dedup_vs_reference(monkeypatch, s, rows)
    assert rel <= _tol(s)
    if deck == "phase guard ceiling":
        assert all(any(b._phase_tie) for b in made), "the guard tie was missed"
    if rows is None:
        return
    assert sum(b.toeplitz_blocks for b in made) > 0, "nothing was gathered"
    if deck != "phase guard ceiling":
        assert sum(b.tie_pairs for b in made) > 0, "no tie was found"


# Red without the tie rule (TIE_REL = 0), on the decks where the two tiers
# are far apart; measured in the PR.
@pytest.mark.parametrize("route", ["accel", "numpy"])
@pytest.mark.parametrize("deck", ["two edges kL 0.49", "two wires joined"])
def test_tie_rule_is_load_bearing(monkeypatch, deck, route):
    if route == "numpy":
        _numpy_offedge(monkeypatch)
    elif not BK._HAVE_BSPLINE_OFFEDGE_TIERED_ACCEL:
        pytest.skip("the C++ off-edge kernels are not built")
    monkeypatch.setattr(EKB, "TIE_REL", 0.0)
    s = _tie_solver(deck)
    rel, _made = _dedup_vs_reference(monkeypatch, s)
    assert rel > _tol(s), "the gate cannot see a gathered tie"


# Red with the phase-guard tie unseen: BaseRows reads a ceiling 1e-6 off the
# kernels' own, so no run is flagged while the kernels still split on 0.5.
# Measured 1.2e-12 on Haswell.
@pytest.mark.parametrize("route", ["accel", "numpy"])
def test_phase_guard_tie_is_load_bearing(monkeypatch, route):
    if route == "numpy":
        _numpy_offedge(monkeypatch)
    elif not BK._HAVE_BSPLINE_OFFEDGE_TIERED_ACCEL:
        pytest.skip("the C++ off-edge kernels are not built")
    monkeypatch.setattr(EKB, "_LADDER_PHASE_KL_CEILING", 0.5 * (1 + 1e-6))
    s = _tie_solver("phase guard ceiling")
    rel, made = _dedup_vs_reference(monkeypatch, s)
    assert not any(any(b._phase_tie) for b in made)
    assert rel > _tol(s), "the gate cannot see a gathered phase-guard tie"


# --------------------------------------------------------------------------
# the representative pairs' same-edge geometry
# --------------------------------------------------------------------------
# Exact equality on every platform, macOS arm64 included. The square table is
# C++ since momwire#1441 (`reg_distance_table`) and the pairs' is numpy, but
# both are the same four IEEE operations per element (subtract, square, add
# a², square root), each correctly rounded, with nothing to reorder and no
# fused multiply-add: the accelerator is built -ffp-contract=off on Linux and
# on darwin (setup.py). momwire#1441's own gate holds the same identity as
# uint64 on every lane (tests/test_same_edge_glue_1441.py, G-1441-0a-2). The
# win32 build (/fp:fast) may contract and is not held to it; no lane runs
# this file there.
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
