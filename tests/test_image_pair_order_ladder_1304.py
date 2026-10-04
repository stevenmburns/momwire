"""The image term on the pair-order ladder (momwire#1304).

#906 put the direct off-edge term on a distance-adaptive quadrature ladder and
#907 extended it to free space, but left the above-ground IMAGE fills
(`_accumulate_Z_image_chunked` -- the chunked sweep and its #631 near-image
fixup -- and the dense `_build_J_image_blocks`) at the flat `n_qp_pair`, because
#906's study binned direct pair geometry only. #1304's study binned image
geometry the same way and found it inside #906's envelope, so those fills now
take the deck's ladder (`_image_fill_ladder`), with the flat fill kept as the
reference `image_pair_order_ladder=()`.

The tolerance, 1e-9 relative on Z and per pair, is G-906-3's contract: every
pair the ladder serves below the base order is within 1e-9 of the base order,
per moment index, normalised by that index's largest entry over the block (the
metric that gate uses; it is unit-free, which a metric mixing moment indices
of different powers of length is not), and the kL <= 0.5 guard exists to keep
that promise. The image term inherits the contract rather than needing a new
one: to the kernel an image pair is an ordinary pair whose source the fill has
already reflected, the selector bins on that source, and the guard reads
segment lengths, which a mirror does not change. Carried to Z it is not a
proved bound (the solve could in principle amplify a moment perturbation), so
Z is checked directly too; the study's worst Z movement was 1.4e-11 (PEC,
2-lambda wire at 0.1 lambda, kL 0.42), 70x inside it, and measured here at
kL 0.48 it is 2.4e-11 with the worst pair at 3.0e-10. Every gate also asserts
the order-4 tier was SERVED and Z MOVED, so none of them can pass with the
image ladder switched off (verified: all fail with `_image_fill_ladder`
returning `()` by default).

  G-1304-1  ladder vs reference Z <= 1e-9 on PEC, refl-coef and Sommerfeld, at
            the worst study geometry and at a near-ground height (0.01 lambda,
            where the #631 fixup engages).
  G-1304-2  the phase guard: kL 0.48 serves order 4 within 1e-9; kL 0.63 is
            the reference bit for bit.
  G-1304-3  every image pair the production fill served, recomputed at the base
            order: within 1e-9, and the base-order pairs bit-identical.
  G-1304-4  default construction reaches the TIERED accelerator from the image
            fill on both routes (chunked, dense fallback), the near-image fixup
            gets the sweep's ladder, and order-4 pairs were served.
  G-1304-5  resolution: default follows `_fill_ladder`, `()` is the reference,
            explicit ladders normalise, EK gets none, and a buried deck never
            reaches these fills.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import momwire._bspline_kernels as _bk
import momwire.bspline as _bs
from momwire.bspline import BSplineSolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_crossing_serve_524 import hub_deck  # noqa: E402
from test_pair_order_ladder_906 import (  # noqa: E402
    PLAIN,
    TIERED,
    _AccelSpy,
    _pair_orders,
)

pytestmark = pytest.mark.skipif(
    not (
        _bk._HAVE_BSPLINE_OFFEDGE_TIERED_ACCEL
        and _bs._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL
    ),
    reason="the tiered off-edge / weighted windowed accelerators are not built",
)

WL = 299792458.0 / 7e6
A = 0.001
SOIL = (13.0, 0.005)
TOL = 1e-9  # G-906-3's per-pair contract; see the module docstring
GROUNDS = {
    "pec": {},
    "refl": {"ground_eps": SOIL},
    "somm": {"ground_eps": SOIL, "ground_model": "sommerfeld"},
}


def _hlong(h_wl, n, ground="pec", **over):
    """A horizontal 2-lambda centre-fed wire at h: long enough that its own
    image pairs reach ratio 16 at kL up to the guard. n = 30 is kL 0.419, the
    study's worst case; 26 is kL 0.483, just under the guard; 20 is kL 0.628,
    over it."""
    L = 2.0 * WL
    h = h_wl * WL
    d = dict(
        wires=[np.array([(-L / 2, 0.0, h), (L / 2, 0.0, h)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, L / 2, 1 + 0j)],
        wavelength=WL,
        wire_radius=A,
        ground_z=0.0,
        **GROUNDS[ground],
    )
    d.update(over)
    return d


def _bent(h_wl, n_per_edge=30, ground="pec", **over):
    """A horizontal L (two 1-lambda edges) at h. At 0.01 lambda each edge sits
    within one segment of its own image, so the #631 near-image fixup replaces
    the self-edge image blocks wholesale -- which on a straight wire would
    leave the ladder nothing to do. The CROSS-edge image pairs are what reach
    order 4 here, and they are swept, not fixed up. kL 0.21."""
    h = h_wl * WL
    w = np.array([(-WL, 0.0, h), (0.0, 0.0, h), (0.0, WL, h)])
    d = dict(
        wires=[w],
        n_per_edge_per_wire=[[n_per_edge, n_per_edge]],
        feeds=[(0, 0.5 * WL, 1 + 0j)],
        wavelength=WL,
        wire_radius=A,
        ground_z=0.0,
        **GROUNDS[ground],
    )
    d.update(over)
    return d


def _solve_recording(deck, monkeypatch):
    """Production solve (`BSplineSolver(**deck).compute_impedance()`), with
    every IMAGE call of the off-edge kernel recorded: (args, ladder, J).

    An image call is recognised by its sources: the deck sits on or above
    ground_z = 0, so a call whose source endpoints are all at or below it,
    some strictly, is handed the mirror. The recorder wraps the name the fills look up, so it sees what
    production passes and nothing is built on the test's side."""
    real = _bs._seg_seg_full_moments_offedge
    calls = []

    def rec(sli, sri, slj, srj, a, k, max_d, n_qp, **kw):
        J = real(sli, sri, slj, srj, a, k, max_d, n_qp, **kw)
        zs = np.concatenate([np.asarray(slj)[:, 2], np.asarray(srj)[:, 2]])
        if np.all(zs <= 0.0) and np.any(zs < 0.0):
            calls.append(((sli, sri, slj, srj, a, k, max_d, n_qp), kw, J))
        return J

    monkeypatch.setattr(_bs, "_seg_seg_full_moments_offedge", rec)
    s = BSplineSolver(**deck)
    z, _ = s.compute_impedance()
    monkeypatch.setattr(_bs, "_seg_seg_full_moments_offedge", real)
    return s, complex(np.atleast_1d(z)[0]), calls


def _orders_served(calls):
    """{order: pairs} over the recorded image calls, from each call's own
    ladder through the kernel's selector + per-pair guard."""
    out = {}
    for (sli, sri, slj, srj, _a, k, _d, n_qp), kw, _J in calls:
        o = _pair_orders(sli, sri, slj, srj, k, n_qp, kw.get("ladder"))
        for v, c in zip(*np.unique(o, return_counts=True)):
            out[int(v)] = out.get(int(v), 0) + int(c)
    return out


CASES = [
    ("hlong-h0.1-kL0.42", lambda g: _hlong(0.1, 30, g)),
    ("bent-h0.01-kL0.21", lambda g: _bent(0.01, 30, g)),
]


@pytest.mark.parametrize("ground", list(GROUNDS))
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_g1304_1_ladder_vs_reference_z(monkeypatch, ground, case):
    deck = case[1](ground)
    _s, z_lad, calls = _solve_recording(deck, monkeypatch)
    _s_ref, z_ref, calls_ref = _solve_recording(
        {**deck, "image_pair_order_ladder": ()}, monkeypatch
    )
    served = _orders_served(calls)
    assert served.get(4, 0) > 0, f"no image pair served at order 4: {served}"
    assert _orders_served(calls_ref).keys() == {8}, "the reference must be flat"
    rel = abs(z_lad - z_ref) / abs(z_ref)
    assert z_lad != z_ref, "Z did not move: the image ladder did not run"
    assert rel <= TOL, f"|dZ|/|Z| = {rel:.3e} ({z_lad} vs {z_ref})"


def test_g1304_2_the_phase_guard_holds_on_image_pairs(monkeypatch):
    # kL 0.483: under the guard, order 4 is served and within the contract.
    _s, z_in, calls = _solve_recording(_hlong(0.1, 26), monkeypatch)
    _s, z_in_ref, _c = _solve_recording(
        _hlong(0.1, 26, image_pair_order_ladder=()), monkeypatch
    )
    assert _orders_served(calls).get(4, 0) > 0
    assert z_in != z_in_ref
    assert abs(z_in - z_in_ref) / abs(z_in_ref) <= TOL
    # kL 0.628: the guard drops the only sub-8 tier, so the laddered fill IS
    # the reference, bit for bit, though it was handed the ladder.
    _s, z_out, calls = _solve_recording(_hlong(0.1, 20), monkeypatch)
    _s, z_out_ref, _c = _solve_recording(
        _hlong(0.1, 20, image_pair_order_ladder=()), monkeypatch
    )
    assert {kw.get("ladder") for _a, kw, _J in calls} == {((16.0, 4),)}
    assert _orders_served(calls).keys() == {8}
    assert z_out == z_out_ref


@pytest.mark.parametrize(
    "deck",
    [_hlong(0.1, 26), _hlong(0.1, 30), _bent(0.01, 30)],
    ids=["hlong-kL0.48", "hlong-kL0.42", "bent-h0.01"],
)
def test_g1304_3_every_served_image_pair_is_within_1e_9(monkeypatch, deck):
    """Each image block the production fill computed, recomputed at the flat
    base order on the SAME arguments; G-906-3's metric (per moment index,
    normalised by the block's largest entry for that index)."""
    _s, _z, calls = _solve_recording(deck, monkeypatch)
    laddered = [c for c in calls if c[1].get("ladder")]
    assert laddered, "the production image fill passed no ladder"
    n4 = 0
    for args, kw, J in laddered:
        sli, sri, slj, srj, _a, k, max_d, n_qp = args
        base = _bk._seg_seg_full_moments_offedge(*args, ek=kw.get("ek"))
        nm = (max_d + 1) ** 2
        scale = np.abs(base).reshape(nm, -1).max(axis=1)[:, None, None]
        err = (np.abs(J - base).reshape(nm, *base.shape[2:]) / scale).max(axis=0)
        orders = _pair_orders(sli, sri, slj, srj, k, n_qp, kw["ladder"])
        assert err.max() <= TOL, f"worst image pair {err.max():.3e}"
        assert np.array_equal(
            err[orders == n_qp], np.zeros(int((orders == n_qp).sum()))
        )
        n4 += int((orders == 4).sum())
        if (orders == 4).any():
            assert err[orders == 4].max() > 0.0, "order-4 pairs identical to order 8"
    assert n4 > 0


@pytest.mark.parametrize("dense", [False, True], ids=["chunked", "dense"])
def test_g1304_4_production_reaches_the_tiered_image_fill(monkeypatch, dense):
    """Default construction, default solve: the image fill hands the tiered
    accelerator the deck's ladder, the near-image fixup gets the SAME ladder
    as the sweep (#921, image side), and order-4 pairs were served. `dense`
    takes the `_build_J_image_blocks` fallback by hiding the weighted
    windowed assembler, as a build without it would."""
    spy = _AccelSpy(_bk._acc, (TIERED, PLAIN))
    monkeypatch.setattr(_bk, "_acc", spy)
    if dense:
        monkeypatch.setattr(_bs, "_HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL", False)
    deck = _bent(0.01, 30)
    s, z, calls = _solve_recording(deck, monkeypatch)
    geom = s._build_geometry()
    assert s.pair_order_ladder == ((16.0, 4),)
    assert {kw.get("ladder") for _a, kw, _J in calls} == {s.pair_order_ladder}
    # chunked: one sweep window plus one fixup per near-image edge (both
    # edges qualify at 0.01 lambda); dense: the one whole-mesh call.
    n_near = len(s._near_image_edge_blocks(geom))
    assert n_near == 2
    assert len(calls) == (1 if dense else 1 + n_near), len(calls)
    assert _orders_served(calls).get(4, 0) > 0
    assert spy.counts[TIERED] >= len(calls), spy.counts
    if dense:
        monkeypatch.undo()
        _s, z_chunked, _c = _solve_recording(deck, monkeypatch)
        # Same per-pair arithmetic; the two routes differ in assembly order
        # (the chunked route's fixup adds a correction, the dense one
        # overwrites), so roundoff, not bits.
        assert abs(z - z_chunked) / abs(z_chunked) < 1e-11


@pytest.mark.filterwarnings("ignore:crossing node")
def test_g1304_5_resolution(monkeypatch):
    s = BSplineSolver(**_hlong(0.1, 30))
    geom = s._build_geometry()
    sl, sr = geom["seg_l"], geom["seg_r"]
    # default: the direct term's ladder, so one rule on both terms
    assert s._image_fill_ladder(s.k, sl, sr, None) == s._fill_ladder(s.k, sl, sr, None)
    assert s._image_fill_ladder(s.k, sl, sr, None) == ((16.0, 4),)
    # `()` is the reference; an explicit ladder normalises against the base
    ref = BSplineSolver(**_hlong(0.1, 30, image_pair_order_ladder=()))
    assert ref._image_fill_ladder(ref.k, sl, sr, None) == ()
    bur = BSplineSolver(**_hlong(0.1, 30, image_pair_order_ladder=((2, 8), (16, 4))))
    assert bur._image_fill_ladder(bur.k, sl, sr, None) == ((16.0, 4),)
    # the image ladder follows an explicit direct one unless told otherwise
    off = BSplineSolver(**_hlong(0.1, 30, pair_order_ladder=()))
    assert off._image_fill_ladder(off.k, sl, sr, None) == ()

    # EK: none, and the EK image fill still solves (the kernel refuses
    # ladder + EK, so a leak here would raise).
    vdeck = dict(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25 * WL)])],
        n_per_edge_per_wire=[[20]],
        feeds=[(0, 0.02 * WL, 1 + 0j)],
        wavelength=WL,
        wire_radius=A,
        ground_z=0.0,
        extended_kernel=True,
    )
    ek_s, _z, ek_calls = _solve_recording(vdeck, monkeypatch)
    assert ek_calls and {kw.get("ladder") for _a, kw, _J in ek_calls} == {()}

    # A buried deck's images ride the subset fill, never these two.
    counts = {"chunked": 0, "dense": 0}
    for name, key in (
        ("_accumulate_Z_image_chunked", "chunked"),
        ("_build_J_image_blocks", "dense"),
    ):
        orig = getattr(BSplineSolver, name)

        def wrapped(self, *a, _orig=orig, _key=key, **kw):
            counts[_key] += 1
            return _orig(self, *a, **kw)

        monkeypatch.setattr(BSplineSolver, name, wrapped)
    BSplineSolver(**hub_deck()).compute_impedance()
    assert counts == {"chunked": 0, "dense": 0}, counts
