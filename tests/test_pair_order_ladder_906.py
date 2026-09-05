"""The pair-order ladder: distance-adaptive off-edge quadrature (momwire#906).

Every off-edge pair used to pay one Gauss–Legendre grid, n_qp × n_qp, whether
the segments touch or sit a hundred lengths apart. The buried fill's order is
32, so the 654-segment radial screen spent 6.3 of its 11.3 s in two calls of
one kernel. The study on #906 binned each pair's lower-order error by its
CENTRE DISTANCE OVER THE LONGER SEGMENT and found order 8 at machine precision
beyond two lengths and order 4 at 3e-14 beyond sixteen, for real and complex
k alike; four ladders reproduced the uniform-32 Z to the printed digit.

#906 turned the ladder on for buried decks only and #907 extended it to free
space, where the base order is 8 and the sole tier is therefore ((16, 4),).
The wiring is half the change: #906 passed `ladder=` from the buried subset
path alone, so the free-space default was a dead letter until #907 plumbed
the dense and chunked fills too.

What these gates pin, in order:

- G-906-1  the C++ tiered kernel matches its numpy twin (real k, complex k,
           scalar and mixed radii) — the two implementations keep tracking.
- G-906-2  no ladder is the pre-#906 kernel BIT FOR BIT: `ladder=None`,
           `ladder=()` and a one-tier C++ ladder are array_equal to the plain
           entry, so every free-space fill is unchanged.
- G-906-3  every served pair is within 1e-9 of the base order, and the pairs
           the order-8 tier serves within 1e-11.
- G-906-4  the phase guard: a block whose longest segment passes kL = 0.5
           loses its sub-8 tiers, both in the helper and at the C++ call.
- G-906-5  a ladder is validated, not reordered, and refuses to combine with
           the extended kernel.
- G-906-6  resolution follows the deck: buried gets the buried ladder,
           free space gets none, explicit always wins, tiers at or above
           the base order are dropped.
- G-906-7  the buried hub's Z is unmoved by the default ladder, and the
           tiered accelerator is what served it.
- G-906-8  a bent free-space deck DOES reach a tiered entry, on both the
           dense and the chunked fill (momwire#907 inverted this gate); 8b
           the extended kernel still refuses a ladder and is bit-identical;
           8c one fill resolves ONE ladder, so the sweep and the same-edge
           correction it subtracts cannot land on different orders.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import momwire._bspline_kernels as _bk
import momwire.bspline as _bs
from momwire._bspline_kernels import (
    _EK,
    _gl01,
    _ladder_arrays,
    _ladder_for_block,
    _normalize_ladder,
    _pair_ratio,
    _seg_seg_full_moments_offedge,
)
from momwire.bspline import (
    BURIED_N_QP_PAIR,
    BURIED_PAIR_ORDER_LADDER,
    DEFAULT_PAIR_ORDER_LADDER,
    BSplineSolver,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_crossing_serve_524 import hub_deck  # noqa: E402

LADDER = ((2.0, 8), (16.0, 4))
K_REAL = 0.5
K_CPLX = 0.5 - 0.12j
TIERED = "seg_seg_full_moments_bspline_tiered"
TIERED_CPLX = "seg_seg_full_moments_bspline_cplx_tiered"
PLAIN = "seg_seg_full_moments_bspline"
PLAIN_CPLX = "seg_seg_full_moments_bspline_cplx"

pytestmark = pytest.mark.skipif(
    not (
        _bk._HAVE_BSPLINE_OFFEDGE_TIERED_ACCEL
        and _bk._HAVE_BSPLINE_OFFEDGE_CPLX_TIERED_ACCEL
    ),
    reason="the tiered off-edge accelerators are not built",
)


def _random_deck(n=60, seed=906):
    """Segments scattered in a box: ratios from ~1 to ~100 with no exact
    ties at a threshold (a tie is where two ulp-different selectors could
    legitimately disagree, and is measure zero here)."""
    rng = np.random.default_rng(seed)
    sl = rng.uniform(-3, 3, (n, 3))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    length = rng.uniform(0.02, 0.2, n)
    return sl, sl + d * length[:, None]


def _chain_deck(n=40, h=0.1, offset=0.15):
    """Two parallel chains: every ratio from 1.5 up, including the near pairs
    the base order exists for."""
    x = np.arange(n) * h
    sl = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    sr = sl + np.array([h, 0.0, 0.0])
    sl2 = sl + np.array([0.0, offset, 0.0])
    sr2 = sr + np.array([0.0, offset, 0.0])
    return np.vstack([sl, sl2]), np.vstack([sr, sr2])


class _AccelSpy:
    """Counting proxy over the accelerator module (the #270 pattern), which
    also keeps the last argument tuple of each counted symbol."""

    def __init__(self, real, names):
        self._real = real
        self.counts = dict.fromkeys(names, 0)
        self.last_args = {}

    def __getattr__(self, name):
        target = getattr(self._real, name)
        if name not in self.counts:
            return target

        def counted(*args, **kwargs):
            self.counts[name] += 1
            self.last_args[name] = args
            return target(*args, **kwargs)

        return counted


@pytest.fixture
def spy(monkeypatch):
    s = _AccelSpy(_bk._acc, (TIERED, TIERED_CPLX, PLAIN, PLAIN_CPLX))
    monkeypatch.setattr(_bk, "_acc", s)
    return s


def _numpy_tiered(monkeypatch, *args, **kwargs):
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_TIERED_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_CPLX_TIERED_ACCEL", False)
    out = _seg_seg_full_moments_offedge(*args, **kwargs)
    monkeypatch.undo()
    return out


@pytest.mark.parametrize("k", [K_REAL, K_CPLX], ids=["real", "cplx"])
@pytest.mark.parametrize("mixed", [False, True], ids=["scalar-a", "mixed-a"])
def test_g906_1_the_tiered_accelerator_matches_its_numpy_twin(
    monkeypatch, spy, k, mixed
):
    sl, sr = _random_deck()
    a = np.where(np.arange(len(sl)) < len(sl) // 2, 1e-3, 2e-3) if mixed else 1e-3
    got = _seg_seg_full_moments_offedge(sl, sr, sl, sr, a, k, 2, 32, ladder=LADDER)
    name = TIERED_CPLX if isinstance(k, complex) else TIERED
    assert spy.counts[name] == (2 if mixed else 1), spy.counts
    assert spy.counts[PLAIN] == spy.counts[PLAIN_CPLX] == 0, spy.counts
    ref = _numpy_tiered(monkeypatch, sl, sr, sl, sr, a, k, 2, 32, ladder=LADDER)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 1e-13, f"C++ tiered vs numpy tiered: {rel:.3e}"


@pytest.mark.parametrize("k", [K_REAL, K_CPLX], ids=["real", "cplx"])
def test_g906_2_no_ladder_is_bit_identical_to_the_plain_entry(spy, k):
    sl, sr = _random_deck()
    plain = _seg_seg_full_moments_offedge(sl, sr, sl, sr, 1e-3, k, 2, 8)
    assert np.array_equal(
        _seg_seg_full_moments_offedge(sl, sr, sl, sr, 1e-3, k, 2, 8, ladder=None), plain
    )
    assert np.array_equal(
        _seg_seg_full_moments_offedge(sl, sr, sl, sr, 1e-3, k, 2, 8, ladder=()), plain
    )
    plain_name = PLAIN_CPLX if isinstance(k, complex) else PLAIN
    assert (
        spy.counts[plain_name] == 3
        and spy.counts[TIERED] == spy.counts[TIERED_CPLX] == 0
    )
    # A one-tier ladder through the tiered entry itself is the same loop.
    tier_t, tier_w, tier_n_qp, tier_ratio = _ladder_arrays(8, ())
    fn = getattr(_bk._acc, TIERED_CPLX if isinstance(k, complex) else TIERED)
    one = fn(sl, sr, sl, sr, 1e-6, k, 2, tier_t, tier_w, tier_n_qp, tier_ratio)
    assert np.array_equal(one, plain)


@pytest.mark.parametrize("k", [K_REAL, K_CPLX], ids=["real", "cplx"])
def test_g906_3_every_served_pair_is_within_1e_9_of_the_base_order(k):
    sl, sr = _chain_deck()
    base = _seg_seg_full_moments_offedge(sl, sr, sl, sr, 1e-3, k, 2, 32)
    got = _seg_seg_full_moments_offedge(sl, sr, sl, sr, 1e-3, k, 2, 32, ladder=LADDER)
    scale = np.abs(base).reshape(9, -1).max(axis=1)[:, None, None]
    err = (np.abs(got - base).reshape(9, *base.shape[2:]) / scale).max(axis=0)
    ratio = _pair_ratio(sl, sr, sl, sr)
    assert err.max() < 1e-9, f"worst pair {err.max():.3e}"
    served_8 = (ratio >= 2.0) & (ratio < 16.0)
    assert err[served_8].max() < 1e-11, f"order-8 tier worst {err[served_8].max():.3e}"
    assert np.array_equal(err[ratio < 2.0], np.zeros(int((ratio < 2.0).sum())))


def test_g906_4_the_phase_guard_drops_the_sub_8_tiers(spy):
    sl, sr = _chain_deck(h=0.1)
    # kL = 0.05: both tiers survive.
    assert _ladder_for_block(LADDER, 0.5, sl, sr, sl, sr) == LADDER
    # kL = 0.6: the phase-limited order-4 tier goes, the order-8 tier stays.
    assert _ladder_for_block(LADDER, 6.0, sl, sr, sl, sr) == ((2.0, 8),)
    assert _ladder_for_block((), 6.0, sl, sr, sl, sr) == ()
    # ... and the C++ call sees the trimmed table.
    _seg_seg_full_moments_offedge(sl, sr, sl, sr, 1e-3, 6.0, 2, 32, ladder=LADDER)
    assert list(spy.last_args[TIERED][9]) == [32, 8]
    _seg_seg_full_moments_offedge(sl, sr, sl, sr, 1e-3, 0.5, 2, 32, ladder=LADDER)
    assert list(spy.last_args[TIERED][9]) == [32, 8, 4]


def test_g906_5_a_ladder_is_validated_not_reordered():
    assert _normalize_ladder(None, 32) == ()
    assert _normalize_ladder((), 32) == ()
    assert _normalize_ladder([(2, 8), (16, 4)], 32) == ((2.0, 8), (16.0, 4))
    with pytest.raises(ValueError, match="strictly ascend"):
        _normalize_ladder(((16, 8), (2, 4)), 32)
    with pytest.raises(ValueError, match="strictly ascend"):
        _normalize_ladder(((0.0, 8),), 32)
    with pytest.raises(ValueError, match="strictly descend"):
        _normalize_ladder(((2, 32),), 32)
    with pytest.raises(ValueError, match="strictly descend"):
        _normalize_ladder(((2, 4), (16, 8)), 32)
    with pytest.raises(ValueError, match="strictly descend"):
        _normalize_ladder(((2, 0),), 32)
    sl, sr = _chain_deck(n=6)
    ek = _EK(
        a=None,
        group_i=np.zeros(12, dtype=np.int64),
        group_j=np.zeros(12, dtype=np.int64),
    )
    with pytest.raises(NotImplementedError, match="plain off-edge kernel only"):
        _seg_seg_full_moments_offedge(
            sl, sr, sl, sr, 1e-3, K_REAL, 2, 32, ek=ek, ladder=LADDER
        )


def test_g906_6_resolution_follows_the_deck():
    buried = BSplineSolver(**hub_deck())
    assert buried.n_qp_pair == BURIED_N_QP_PAIR
    assert buried.pair_order_ladder == BURIED_PAIR_ORDER_LADDER == ((2.0, 8), (16.0, 4))
    assert BSplineSolver(**hub_deck(pair_order_ladder=())).pair_order_ladder == ()
    assert BSplineSolver(**hub_deck(pair_order_ladder=((3, 6),))).pair_order_ladder == (
        (3.0, 6),
    )
    # An explicit base order drops the tiers it no longer sits above.
    assert BSplineSolver(**hub_deck(n_qp_pair=8)).pair_order_ladder == ((16.0, 4),)
    assert BSplineSolver(**hub_deck(n_qp_pair=4)).pair_order_ladder == ()
    free = BSplineSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])],
        n_per_edge_per_wire=[[8, 8]],
        feeds=[(0, 0.5, 1 + 0j)],
    )
    # momwire#907: free space has ONE tier, because its base order 8 already
    # is the order-8 rung — the same tuple an explicit `n_qp_pair=8` leaves of
    # the buried ladder, just arrived at from the other side.
    assert free.pair_order_ladder == DEFAULT_PAIR_ORDER_LADDER == ((16.0, 4),)
    assert (
        BSplineSolver(
            wires=[np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])],
            n_per_edge_per_wire=[[8, 8]],
            feeds=[(0, 0.5, 1 + 0j)],
            pair_order_ladder=(),
        ).pair_order_ladder
        == ()
    )


@pytest.mark.filterwarnings("ignore:crossing node")
def test_g906_7_the_buried_hub_z_is_unmoved_by_the_default_ladder(spy, record_property):
    z_ladder, _ = BSplineSolver(**hub_deck()).compute_impedance()
    assert spy.counts[TIERED_CPLX] >= 2, spy.counts
    assert spy.counts[PLAIN_CPLX] == 0, spy.counts
    z_flat, _ = BSplineSolver(**hub_deck(pair_order_ladder=())).compute_impedance()
    record_property("z_ladder", f"{z_ladder:.9f}")
    record_property("z_flat", f"{z_flat:.9f}")
    assert abs(z_ladder - z_flat) < 1e-6, f"{z_ladder} vs {z_flat}"


def _free_bent_deck(n_per_edge=100, side=0.25, **over):
    """A bent free-space deck meshed finely enough to have far pairs AND to
    sit under the kL = 0.5 guard — the shape #907 turned the ladder on for."""
    w = np.array(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side, side, 0.0],
            [0.0, side, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    d = dict(
        wires=[w],
        n_per_edge_per_wire=[[n_per_edge] * 4],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=1.0,
        wire_radius=1e-4,
    )
    d.update(over)
    return d


@pytest.mark.parametrize("chunked", [False, True], ids=["dense", "chunked"])
def test_g906_8_free_space_now_reaches_a_tiered_entry(spy, chunked):
    """momwire#907 inverted this gate. It used to pin that free space never
    tiered; it now pins that it DOES, on both fills.

    The inversion is the point. #906 wired `ladder=` into the buried subset
    path only, so for a while flipping `DEFAULT_PAIR_ORDER_LADDER` changed
    nothing at all while this gate — in its old form — went on passing and
    said so. A gate that cannot tell "off" from "not plumbed" is the failure
    this one exists to prevent, so it asserts the tiered entry is REACHED
    rather than that some Z is unchanged.
    """
    deck = _free_bent_deck(**({"swept_mem_mb": 1} if chunked else {}))
    s = BSplineSolver(**deck)
    assert s.pair_order_ladder == ((16.0, 4),)
    s.compute_impedance()
    assert spy.counts[TIERED] >= 1, spy.counts
    assert spy.counts[TIERED_CPLX] == 0, spy.counts


def test_g906_8b_the_extended_kernel_still_refuses_a_ladder(spy):
    """The kernel raises on ladder+EK, and `_fill_ladder` is what keeps a
    free-space EK deck from ever meeting that refusal now that the default is
    no longer empty.

    The kernel's own comment justified the refusal with "the free-space default
    ladder is empty", which is exactly the assumption #907 flipped, so without
    the guard every extended-kernel deck raises.
    """
    deck = _free_bent_deck(n_per_edge=25)
    s = BSplineSolver(**deck, extended_kernel=True)
    geom = s._build_geometry()
    ek = s._ek_spec(geom)
    assert ek is not None
    # The deck still WISHES for the ladder ...
    assert s.pair_order_ladder == ((16.0, 4),)
    # ... and the fill declines to ask for it, so the refusal is never reached.
    assert s._fill_ladder(s.k, geom["seg_l"], geom["seg_r"], ek) == ()
    z_ek, _ = s.compute_impedance()
    # EK takes its own accelerator entry, so the plain counters stay at zero
    # here; the tiered ones are the check that matters.
    assert spy.counts[TIERED] == spy.counts[TIERED_CPLX] == 0, spy.counts
    z_ref, _ = BSplineSolver(
        **deck, extended_kernel=True, pair_order_ladder=()
    ).compute_impedance()
    assert z_ek == z_ref, f"{z_ek!r} vs {z_ref!r}"


def test_g906_8c_the_whole_fill_agrees_on_one_ladder(spy):
    """The chunked fill adds every pair in a sweep and subtracts same-edge
    blocks back, so both arms must have run the same quadrature.

    `_ladder_for_block` keys the guard on a block's LONGEST segment, so a
    per-window resolution lets the sweep (which spans the whole mesh) drop the
    order-4 tier while a correction window for one fine edge keeps it. This
    deck is built to straddle exactly that: a finely meshed radiator whose own
    edge is far under the ceiling, plus one 0.3-lambda segment that puts the
    mesh far over it. `_fill_ladder` resolves once against the whole mesh, so
    every window here must see the SAME empty ladder.
    """
    fine = np.array([[0.0, -0.25, 0.0], [0.0, 0.25, 0.0]])
    coarse = np.array([[0.35, 0.0, 0.0], [0.35, 0.3, 0.0]])
    deck = dict(
        wires=[fine, coarse],
        n_per_edge_per_wire=[[200], [1]],
        feeds=[(0, 0.25, 1 + 0j)],
        wavelength=1.0,
        wire_radius=1e-4,
        swept_mem_mb=1,
    )
    s = BSplineSolver(**deck)
    assert s.pair_order_ladder == ((16.0, 4),)
    k = s.k
    geom = s._build_geometry()
    seg_l, seg_r = geom["seg_l"], geom["seg_r"]
    # the mesh trips the guard, so the whole fill must run untiered ...
    assert s._fill_ladder(k, seg_l, seg_r, None) == ()
    # ... even though the fine wire's own edge, taken alone, would not.
    assert _ladder_for_block(
        ((16.0, 4),), k, seg_l[:200], seg_r[:200], seg_l[:200], seg_r[:200]
    ) == ((16.0, 4),)
    z, _ = BSplineSolver(**deck).compute_impedance()
    assert spy.counts[TIERED] == 0, spy.counts
    z_flat, _ = BSplineSolver(**deck, pair_order_ladder=()).compute_impedance()
    assert z == z_flat, f"{z!r} vs {z_flat!r}"


def test_g906_9_the_selector_is_the_one_the_study_binned_by():
    """`_pair_ratio` is the contract the C++ selector mirrors: centre
    distance over the LONGER segment. Pinned on hand-built pairs."""
    sl = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    sr = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 5.5, 0.0]])
    r = _pair_ratio(sl, sr, sl, sr)
    assert r[0, 0] == 0.0
    assert np.isclose(r[0, 1], 1.5 / 2.0)  # centres 0.5 and 2.0, longer is 2
    assert np.isclose(r[0, 2], np.hypot(0.5, 5.25) / 1.0)
    assert np.allclose(r, r.T)
    t, w = _gl01(4)
    assert np.isclose(w.sum(), 1.0) and (0 < t).all() and (t < 1).all()


# ----------------------------------------------------------------------
# G-921 — one fill, one ladder, on the BURIED path too
# ----------------------------------------------------------------------


def _buried_mixed_mesh_deck(**over):
    """A buried deck built to straddle the kL = 0.5 guard.

    A 60-segment radial at lambda/120 — fine enough that its own same-edge
    block holds pairs past ratio 16, and far under the ceiling on its own —
    plus ONE 0.3-lambda buried segment that puts any window spanning it well
    over. That is the whole bug in a deck: the sweep sees the coarse segment
    and the same-edge correction does not.
    """
    fine = np.array([[0.0, 0.0, -0.05], [0.5, 0.0, -0.05]])
    coarse = np.array([[0.0, 0.3, -0.05], [0.3, 0.3, -0.05]])
    d = dict(
        wires=[fine, coarse],
        n_per_edge_per_wire=[[60], [1]],
        feeds=[(0, 0.25, 1 + 0j)],
        wavelength=1.0,
        wire_radius=1e-4,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    d.update(over)
    return d


def _ladders_seen(deck):
    """Every `ladder=` a fill hands the off-edge kernel, with the effective
    (post-guard) ladder each block would resolve for itself."""
    s = BSplineSolver(**deck)
    seen = []
    real = _bs._seg_seg_full_moments_offedge

    def spy(sli, sri, slj, srj, a, k, max_d, n_qp, **kw):
        passed = _normalize_ladder(kw.get("ladder"), n_qp)
        seen.append(
            (
                passed,
                _ladder_for_block(passed, k, sli, sri, slj, srj),
                int((_pair_ratio(sli, sri, slj, srj) >= 16.0).sum()),
            )
        )
        return real(sli, sri, slj, srj, a, k, max_d, n_qp, **kw)

    _bs._seg_seg_full_moments_offedge = spy
    try:
        z, _ = s.compute_impedance()
    finally:
        _bs._seg_seg_full_moments_offedge = real
    return z, seen


@pytest.mark.filterwarnings("ignore:crossing node")
def test_g921_one_fill_resolves_one_ladder_on_the_buried_path():
    """momwire#921: the sweep and the same-edge correction it subtracts back
    must have run the SAME quadrature on every pair.

    `corr = (A_st + A_reg) - J_edge` cancels only if both arms agree, and
    before #907's `_fill_ladder` the buried path passed `pair_order_ladder`
    raw — so on this deck the sweep resolved to ((2, 8),) and the correction
    to ((2, 8), (16, 4)), leaving the difference in Z as an artefact of which
    window happened to span the coarse segment.
    """
    _z, seen = _ladders_seen(_buried_mixed_mesh_deck())
    assert seen, "no off-edge kernel call — the deck stopped exercising the fill"
    with_far = [(passed, eff) for passed, eff, far in seen if far]
    assert with_far, "no window holds a far pair — the gate would be vacuous"
    # One ladder handed out ...
    assert len({p for p, _ in with_far}) == 1, seen
    # ... and, because it was resolved against the whole subset, every window
    # agrees on what survives the guard. This is the assertion that fails on
    # the raw property.
    assert len({e for _, e in with_far}) == 1, seen


@pytest.mark.filterwarnings("ignore:crossing node")
def test_g921_the_gate_is_not_vacuous_a_uniform_buried_deck_does_tier():
    """The control for the gate above: strip the coarse segment and the same
    fill tiers, on every window. Without this, a build that silently stopped
    laddering buried decks would pass G-921 by agreeing on nothing."""
    _z, seen = _ladders_seen(
        _buried_mixed_mesh_deck(
            wires=[np.array([[0.0, 0.0, -0.05], [0.5, 0.0, -0.05]])]
        )
        | {"n_per_edge_per_wire": [[60]]}
    )
    with_far = [(passed, eff) for passed, eff, far in seen if far]
    assert with_far, "no window holds a far pair"
    assert len({e for _, e in with_far}) == 1, seen
    eff = with_far[0][1]
    assert eff == ((2.0, 8), (16.0, 4)), eff


@pytest.mark.filterwarnings("ignore:crossing node")
def test_g921_the_buried_hub_z_is_unchanged(record_property):
    """The fix must not move the deck #906 certified on."""
    z, _ = BSplineSolver(**hub_deck()).compute_impedance()
    record_property("z_hub", f"{z:.9f}")
    z_flat, _ = BSplineSolver(**hub_deck(pair_order_ladder=())).compute_impedance()
    assert abs(z - z_flat) < 1e-6, f"{z} vs {z_flat}"
