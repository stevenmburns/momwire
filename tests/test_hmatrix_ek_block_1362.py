"""The H-matrix's extended-kernel far-block assembler -- momwire#1362.

`bspline_assemble_offedge_block[_refl]_ek` served every ACA row and column of
an EK H-matrix solve with the pre-#1290 fused walk: per basis pair and wing
pair, a fresh 64-point moment quadrature. Measured on Haswell (free x16,
N 2816, EK on, 4 threads) it was 67 % of the solve (4.6 of 6.9 s, 12,738
calls, every one a single row or column). It now

* computes each SEGMENT pair's moments once into a table (phase A) and
  combines basis pairs from it (phase B). A segment pair sits under up to
  (d+1)^2 basis-pair wings, so the walk computed each moment ~2.9x on that
  solve's rows and columns. Jc is a pure function of the segment pair, so
  the combine reads the same doubles in the same order;
* runs phase A with the pre-#1362 per-pair arithmetic (stage 2 t-outer,
  which only reorders work across chains) and, in the AVX2 build, four
  pairs to a vector lane -- any four, gathered, so a column call (three
  source segments) fills lanes as well as a row call does;
* takes the pair-order ladder under EK through `_tiered` entries, with the
  per-pair phase guard answered in the kernel from per-segment masks, so a
  pair's order is the pair's alone (an ACA row and column cannot disagree).

What a single build can pin, this file pins:

* lanes == walk (`reference=True`) to the bit, through the solver's own
  evaluators, d = 1, 2, 3, orders 3, 4, 8 and 10, free / PEC image / refl;
* a row call, a column call and a dense call agree to the bit on the
  entries they share (one table, one combine);
* an all-ineligible EK block IS the reduced assembler's block, bit for bit;
  a 1 ppm EK radius moves it;
* one tier is the flat entry bit for bit; a ladder whose only tier is
  phase-limited is ignored, bit for bit, when every segment fails the guard;
* solver level: the EK H-matrix on the ladder is the flat EK H-matrix within
  Z_TOL, with the tiered entries counted and the bits moved, EK off far away
  and never reaching an EK entry; the lane solve is the reference solve to
  the bit.

The walk against the pre-#1362 build (two builds) is in the PR: 723
recorded row/column calls, d = 1..3, free / PEC / refl, bit-equal; every
EK-off H-matrix Z and the flat-ladder EK H-matrix Z bit-equal on 8 decks.

Ladder tolerance, DERIVED. EK H-matrix, ladder vs flat, |dZ_in| / |Z_in|,
Haswell, default aca_tol 1e-6 / solve_tol 1e-6:

    deck                          ladder vs flat   H-matrix vs dense
    free_array (N 164)            1.8e-15          2.5e-11
    junction, free (N 226)        2.7e-14          7.9e-9
    junction, PEC                 1.6e-13          8.7e-9
    junction, refl-coef           9.8e-14          7.7e-9
    junction, Sommerfeld          1.9e-13          1.3e-7
    free x4  (N 704)              1.6e-12          3.0e-7
    free x16 (N 2816)             4.4e-11          6.2e-8

The movement is the ladder's own (the dense EK fill moves 1.5e-15 on x4 and
2.4e-13 on x16) passed through ACA's pivoting and GMRES's stopping point:
at aca_tol 1e-12 / solve_tol 1e-13 the H-matrix's ladder movement falls to
1.6e-13 (x4) and 5.8e-13 (x16). It sits three orders or more under the
H-matrix's own distance from the dense fill. The gate below holds the test
decks (all <= 2e-13) to #1365's Z_TOL = 1e-11.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from momwire import _bspline_kernels as BK
from momwire import hmatrix as H

acc = BK._acc

pytestmark = pytest.mark.skipif(
    not (
        H._HAVE_OFFEDGE_BLOCK_EK_ACCEL
        and H._HAVE_OFFEDGE_BLOCK_REFL_EK_ACCEL
        and H._HAVE_OFFEDGE_BLOCK_EK_TIERED_ACCEL
        and H._HAVE_OFFEDGE_BLOCK_REFL_EK_TIERED_ACCEL
    ),
    reason="needs the C++ EK block assemblers",
)

Z_TOL = 1e-11
C0 = 299792458.0
ENTRIES = (
    "bspline_assemble_offedge_block_ek",
    "bspline_assemble_offedge_block_refl_ek",
    "bspline_assemble_offedge_block_ek_tiered",
    "bspline_assemble_offedge_block_refl_ek_tiered",
)
TIERED = ENTRIES[2:]


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _deck(ground=None, **kw):
    """Four parallel dipoles and a bent wire: coaxial groups per straight
    run, with lane groups that straddle them."""
    wires = [
        np.array([(-10.0, j * 8.5, 10.0), (10.0, j * 8.5, 10.0)]) for j in range(4)
    ]
    wires.append(np.array([(-10.0, -6.0, 9.0), (0.0, -6.0, 12.0), (10.0, -6.0, 9.0)]))
    d = dict(
        wires=wires,
        n_per_edge_per_wire=[[61]] * 4 + [[30, 31]],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=C0 / 7e6,
        wire_radius=0.001,
        extended_kernel=True,
    )
    if ground == "pec":
        d.update(ground_z=0.0)
    elif ground == "refl":
        d.update(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="refl-coef")
    d.update(kw)
    return d


def _block(n):
    """Observer bases on the first dipole; sources further along the same
    dipole (coaxial, so EK-eligible), on the third dipole and on the bent
    wire (ineligible, and far enough for the order-4 tier)."""
    J = np.concatenate(
        [np.arange(46, 58), np.arange(n // 2, n // 2 + 15), np.arange(n - 25, n - 2)]
    )
    return np.arange(5, 41, dtype=np.int64), J.astype(np.int64)


def _evaluators(solver, image):
    """(get_row, get_col, dense) for a block of the solver's own."""
    ctx = solver._context()
    I, J = _block(ctx["supp_seg"].shape[0])
    refl = image == "refl"
    return (
        solver._offedge_block_evaluators(
            ctx, I, J, solver.k, mirror_J=image is not None, refl=refl
        ),
        I,
        J,
    )


class _Route:
    """Pin every EK block entry to `reference` and count calls."""

    def __init__(self, monkeypatch, reference=None):
        self.n = dict.fromkeys(ENTRIES, 0)
        for name in ENTRIES:
            f = getattr(acc, name)
            if reference is not None:
                f = functools.partial(f, reference=reference)

            def counted(*a, _f=f, _name=name, **k):
                self.n[_name] += 1
                return _f(*a, **k)

            monkeypatch.setattr(acc, name, counted)


def _ground_for(image):
    return {None: None, "pec": "pec", "refl": "refl"}[image]


# --- kernel level, through the solver's evaluators --------------------------


@pytest.mark.parametrize("image", [None, "pec", "refl"])
@pytest.mark.parametrize(
    "degree, n_qp", [(1, 4), (2, 8), (3, 8), (2, 3), (2, 10)], ids=str
)
@pytest.mark.parametrize("ladder", [(), None], ids=["flat", "ladder"])
def test_lanes_are_the_walk_to_the_bit(monkeypatch, image, degree, n_qp, ladder):
    kw = dict(degree=degree, n_qp_pair=n_qp)
    if ladder is not None:
        kw["pair_order_ladder"] = ladder
    s = H.HMatrixSolver(**_deck(_ground_for(image), **kw))
    out = {}
    for reference in (True, False):
        with monkeypatch.context() as mp:
            route = _Route(mp, reference=reference)
            (get_row, get_col, dense), I, J = _evaluators(s, image)
            out[reference] = (get_row(3), get_col(7), dense())
            assert sum(route.n.values()) == 3, route.n
    for got, want in zip(out[False], out[True]):
        assert np.array_equal(_bits(got), _bits(want))


@pytest.mark.parametrize("image", [None, "pec", "refl"])
def test_rows_columns_and_the_dense_block_agree_to_the_bit(image):
    s = H.HMatrixSolver(**_deck(_ground_for(image)))
    (get_row, get_col, dense), I, J = _evaluators(s, image)
    Z = dense()
    for i in (0, 7, len(I) - 1):
        assert np.array_equal(_bits(get_row(i)), _bits(Z[i]))
    for j in (0, 11, len(J) - 1):
        assert np.array_equal(_bits(get_col(j)), _bits(Z[:, j]))


def _flat_args(s, I, J, image):
    """The flat EK entry's arguments for the block, as the evaluator builds
    them, with the EK labels returned separately."""
    ctx = s._context()
    supp, polys = ctx["supp_seg"], ctx["polys"]
    segI = np.unique(supp[I].ravel())
    segJ = np.unique(supp[J].ravel())
    mirror = (lambda p: s._image_positions(p)) if image else (lambda p: p)
    tJ = ctx["tangents"][segJ]
    if image:
        from momwire import _ground_mirror

        tJ = _ground_mirror.mirror_tangents(tJ)
    glt, glw = s._gl01()
    args = (
        np.searchsorted(segI, supp[I]).astype(np.int64),
        np.ascontiguousarray(polys[I]),
        ctx["seg_l"][segI],
        ctx["seg_r"][segI],
        ctx["tangents"][segI],
        np.searchsorted(segJ, supp[J]).astype(np.int64),
        np.ascontiguousarray(polys[J]),
        mirror(ctx["seg_l"][segJ]),
        mirror(ctx["seg_r"][segJ]),
        tJ,
        float(s.wire_radius) ** 2,
        s.k,
        s.omega,
        s.eps,
        s.mu,
        s.degree,
    )
    ek = s._ek_spec(ctx["geom"], mirror=image is not None)
    return args, (glt, glw), ek.group_i[segI], ek.group_j[segJ]


@pytest.mark.parametrize("image", [None, "pec"])
def test_an_all_ineligible_block_is_the_reduced_assembler(image):
    s = H.HMatrixSolver(**_deck(_ground_for(image)))
    I, J = _block(s._context()["supp_seg"].shape[0])
    args, gl, gi, gj = _flat_args(s, I, J, image)
    a = float(s.wire_radius)
    none_i = np.full_like(gi, -1)
    for reference in (True, False):
        ek = acc.bspline_assemble_offedge_block_ek(
            *args, *gl, none_i, gj, a, reference=reference
        )
        red = acc.bspline_assemble_offedge_block(*args, *gl)
        assert np.array_equal(_bits(ek), _bits(red))
    # Negative controls: with its labels the block moves — the image block
    # too, since momwire#1368 extends a horizontal wire against its own image
    # — and a 1 ppm EK radius moves it again.
    ek = acc.bspline_assemble_offedge_block_ek(*args, *gl, gi, gj, a)
    assert not np.array_equal(ek, red)
    ppm = acc.bspline_assemble_offedge_block_ek(*args, *gl, gi, gj, a * (1 + 1e-6))
    assert not np.array_equal(ppm, ek)


def test_one_tier_is_the_flat_entry_and_a_failed_guard_drops_the_limited_tier():
    s = H.HMatrixSolver(**_deck())
    I, J = _block(s._context()["supp_seg"].shape[0])
    args, gl, gi, gj = _flat_args(s, I, J, None)
    a = float(s.wire_radius)
    flat = acc.bspline_assemble_offedge_block_ek(*args, *gl, gi, gj, a)
    none = np.zeros(0, dtype=np.uint8)
    one = acc.bspline_assemble_offedge_block_ek_tiered(
        *args, *BK._ladder_arrays(8, ()), gi, gj, a, none, none, 8
    )
    assert np.array_equal(_bits(one), _bits(flat))
    lad = BK._ladder_arrays(8, ((16.0, 4),))
    tiered = acc.bspline_assemble_offedge_block_ek_tiered(
        *args, *lad, gi, gj, a, none, none, 8
    )
    assert not np.array_equal(tiered, flat)  # the order-4 tier serves pairs
    # Every segment failing the phase guard: the order-4 tier is phase-
    # limited (4 < 8), so no pair may take it -- the flat block exactly.
    fail_i = np.zeros(args[2].shape[0], dtype=np.uint8)
    fail_j = np.zeros(args[7].shape[0], dtype=np.uint8)
    guarded = acc.bspline_assemble_offedge_block_ek_tiered(
        *args, *lad, gi, gj, a, fail_i, fail_j, 8
    )
    assert np.array_equal(_bits(guarded), _bits(flat))
    # ... and the guard is per PAIR: failing only the source side does too.
    ok_i = np.ones_like(fail_i)
    half = acc.bspline_assemble_offedge_block_ek_tiered(
        *args, *lad, gi, gj, a, ok_i, fail_j, 8
    )
    assert np.array_equal(_bits(half), _bits(flat))


# --- solver level --------------------------------------------------------------


def _z(monkeypatch, deck, reference=None, **kw):
    with monkeypatch.context() as mp:
        route = _Route(mp, reference=reference)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, cur = H.HMatrixSolver(**{**deck, **kw}).compute_impedance()
    return complex(np.atleast_1d(z)[0]), np.asarray(cur), route.n


@pytest.mark.parametrize("ground", [None, "pec", "refl"])
def test_the_ek_hmatrix_on_the_ladder_is_the_flat_one_within_the_derived_tolerance(
    monkeypatch, ground
):
    deck = _deck(ground)
    z_lad, _, n_lad = _z(monkeypatch, deck)
    z_flat, _, n_flat = _z(monkeypatch, deck, pair_order_ladder=())
    rel = abs(z_lad - z_flat) / abs(z_flat)
    assert rel < Z_TOL, f"{ground}: EK H-matrix ladder vs flat {rel:.3e}"
    # Negative controls: the tiered entries served the far blocks and moved
    # the bits; the flat solve never reached them ...
    assert sum(n_lad[e] for e in TIERED) > 0, n_lad
    assert sum(n_flat[e] for e in TIERED) == 0, n_flat
    assert z_lad != z_flat
    # ... and EK is on: EK off sits far away and reaches no EK entry.
    z_off, _, n_off = _z(monkeypatch, deck, extended_kernel=False)
    assert abs(z_lad - z_off) / abs(z_off) > 1e3 * Z_TOL
    assert sum(n_off.values()) == 0, n_off


@pytest.mark.parametrize("ladder", [None, ()], ids=["ladder", "flat"])
def test_an_ek_hmatrix_solve_is_the_reference_walk_to_the_bit(monkeypatch, ladder):
    deck = _deck("pec")
    kw = {} if ladder is None else dict(pair_order_ladder=ladder)
    z_ref, c_ref, n_ref = _z(monkeypatch, deck, reference=True, **kw)
    z, c, n = _z(monkeypatch, deck, reference=False, **kw)
    assert n == n_ref and sum(n.values()) > 0
    assert _bits(np.array([z])).tolist() == _bits(np.array([z_ref])).tolist()
    assert np.array_equal(_bits(c), _bits(c_ref))


def test_a_deck_straddling_the_phase_guard(monkeypatch):
    """A coarse parasitic (L = 4.8 m, past lambda / (4 pi) ~ 3.4 m at 7 MHz)
    beside fine dipoles: its pairs may not take the phase-limited order-4
    tier, so the far blocks are handed per-segment masks, and the answer is
    still the flat solve within Z_TOL."""
    deck = _deck()
    deck["wires"] = deck["wires"] + [
        np.array([(-12.0, 60.0, 10.0), (12.0, 60.0, 10.0)])
    ]
    deck["n_per_edge_per_wire"] = deck["n_per_edge_per_wire"] + [[5]]
    masks = []
    f = acc.bspline_assemble_offedge_block_ek_tiered

    def spy(*a, **k):
        masks.append((a[23].size, a[24].size))
        return f(*a, **k)

    with monkeypatch.context() as mp:
        mp.setattr(acc, "bspline_assemble_offedge_block_ek_tiered", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, _ = H.HMatrixSolver(**deck).compute_impedance()
    assert masks and any(i or j for i, j in masks), "no call carried the guard"
    z_flat, _, _ = _z(monkeypatch, deck, pair_order_ladder=())
    assert abs(z - z_flat) / abs(z_flat) < Z_TOL
