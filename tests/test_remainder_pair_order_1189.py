"""momwire#1189: the Sommerfeld remainder order is chosen per segment pair.

momwire#631 keyed the remainder's Gauss order to grazing height, but as ONE
order for the whole fill, so a single wire near the plane raised every pair.
On 4nec2's GndScreen (16 radials at 1.8 mm) that was q = 192 for all ~200k
segment pairs — 529 of a 547 s solve — for the sake of the few hundred pairs
whose near-image ridge actually needs it.

`_quadrature.remainder_qp_pairs` evaluates the same rule one pair at a time,
the dense fill runs the whole deck at the base order, and only the listed
pairs are raised afterwards. These gates read the mechanism (which orders ran,
on which pairs) as well as the answer.

The accuracy bar is momwire#647's REMAINDER_KEYING_AGREEMENT (1e-5 of |Z_in|):
the tightest bar the suite already holds two remainder-order plumbings to, and
three orders inside #631's own 1e-2 convergence gate.
"""

import warnings

import numpy as np
import pytest

from momwire import _quadrature
from momwire import bspline as _bs
from momwire.bspline import BSplineSolver

from test_sommerfeld_ground import _GRAZE_WL, _grazing_wire

C0 = 299792458.0

PAIR_AGREEMENT = 1e-5


def gnd_screen_deck(**override):
    """4nec2's `GndScreen.nec` (antennaknobs AK#1705's fixture), momwire-native.

    A 20 m vertical from 1.8 mm, a 30 m top wire, and 16 radials 25 m long at
    1.8 mm over `GN 2` soil (eps_r 14, sigma 0.005) at 1.8 MHz, with the
    deck's own `int(...)` segment counts: 20, 30 and 25. The deck's `LD 5`
    copper loading is left out; it does not touch the remainder fill.
    """
    radh, hgh, ln, wrad = 0.0018, 20.0, 50.0, 0.6e-3
    wires = [
        np.array([(0.0, 0.0, radh), (0.0, 0.0, hgh)]),
        np.array([(0.0, 0.0, hgh), (ln - (hgh + radh), 0.0, hgh)]),
    ]
    npe = [[20], [30]]
    for k in range(1, 17):
        a = np.deg2rad(k * 360.0 / 16)
        wires.append(
            np.array([(0.0, 0.0, radh), (25 * np.cos(a), 25 * np.sin(a), radh)])
        )
        npe.append([25])
    build = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(0, "start")] + [(w, "start") for w in range(2, 18)],
            [(0, "end"), (1, "start")],
        ],
        feeds=[(0, 0.5 * (hgh - radh) / 20, 1 + 0j)],
        wavelength=C0 / 1.8e6,
        wire_radius=wrad,
        ground_z=0.0,
        ground_eps=(14.0, 0.005),
        ground_model="sommerfeld",
    )
    build.update(override)
    return build


def _somm(hl, soil=(13.0, 0.005), n_seg=16):
    return _grazing_wire(
        hl * _GRAZE_WL, n_seg=n_seg, ground_eps=soil, ground_model="sommerfeld"
    )


def _solve(kw, **flags):
    saved = {k: getattr(_bs, k) for k in flags}
    try:
        for k, v in flags.items():
            setattr(_bs, k, v)
        s = BSplineSolver(**kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the near-ground advisory
            z, _ = s.compute_impedance()
        return z, s
    finally:
        for k, v in saved.items():
            setattr(_bs, k, v)


def _pairs(kw):
    s = BSplineSolver(**kw)
    g = s._build_geometry()
    return s, g, s._remainder_qp_pairs(g["seg_l"], g["seg_r"], s.ground_z)


def test_the_deck_wide_order_is_the_maximum_of_the_pair_orders():
    """Same rule, different reduction — so the scalar is the pair maximum,
    and a deck where the scalar is the base lists no pair at all."""
    for kw in (_somm(1.09e-4), _somm(1e-3), gnd_screen_deck()):
        s, g, (I, J, Q) = _pairs(kw)
        q_deck = s._remainder_qp(g["seg_l"], g["seg_r"], s.ground_z)
        assert q_deck > s.n_qp_sommerfeld
        assert int(Q.max()) == q_deck
        assert (Q > s.n_qp_sommerfeld).all()
        # Symmetric, as the filled Q must stay.
        n = g["seg_l"].shape[0]
        fwd = dict(zip((I * n + J).tolist(), Q.tolist(), strict=True))
        assert all(fwd.get(j * n + i) == q for i, j, q in zip(I, J, Q, strict=True))
    for kw in (_somm(1e-1), dict(gnd_screen_deck(), ground_z=-2.0)):
        s, g, (I, J, Q) = _pairs(kw)
        assert s._remainder_qp(g["seg_l"], g["seg_r"], s.ground_z) == 3
        assert I.size == 0


def test_gnd_screen_raises_only_its_near_plane_pairs():
    """The deck #1189 was filed on: the radial self pairs still get the cap,
    and the vertical and the top wire, far from the plane, get the base."""
    s, g, (I, J, Q) = _pairs(gnd_screen_deck())
    n = g["seg_l"].shape[0]
    assert n == 20 + 30 + 16 * 25
    vert = np.arange(0, 20)
    top = np.arange(20, 50)
    radials = np.arange(50, n)
    q = {(int(i), int(j)): int(v) for i, j, v in zip(I, J, Q, strict=True)}
    # Every radial segment's own pair is resolved at the cap, as before.
    assert all(q.get((r, r)) == _bs._REMAINDER_QP_CAP for r in radials)
    # The top wire at 20 m never pairs above the base with anything, and the
    # vertical only through the hub it shares with the radials.
    assert not np.isin(I, top).any()
    assert set(I[np.isin(I, vert)].tolist()) == {0}
    # What the saving is: under 2 % of the pairs run above the base, where
    # the deck-wide order ran all of them at 192.
    assert I.size < 0.02 * n * n, I.size


def test_the_fill_runs_the_orders_it_was_given(monkeypatch):
    """Spy on the fill: the census is what ran, and the high order is only
    ever evaluated on listed pairs."""
    kw = _somm(1.09e-4)
    seen = []
    orig = BSplineSolver._remainder_pair_moments

    def spy(self, geom, obs, src, q, grid):
        seen.extend((int(i), int(j), int(q)) for i in obs for j in src)
        return orig(self, geom, obs, src, q, grid)

    monkeypatch.setattr(BSplineSolver, "_remainder_pair_moments", spy)
    _z, s = _solve(kw)
    census = s._last_remainder_orders
    base, cap = s.n_qp_sommerfeld, _bs._REMAINDER_QP_CAP
    n = 16
    assert sum(census.values()) == n * n
    assert census[base] > 0 and max(census) > base, census
    _, _, (I, J, Q) = _pairs(kw)
    listed = {(int(i), int(j)): int(q) for i, j, q in zip(I, J, Q, strict=True)}
    high = [(i, j, q) for i, j, q in seen if q != base]
    assert sorted(high) == sorted((i, j, q) for (i, j), q in listed.items())
    assert max(q for _, _, q in high) <= cap
    # Every listed pair's base-order moments were taken back out, too.
    low = [(i, j) for i, j, q in seen if q == base]
    assert sorted(low) == sorted(listed)


@pytest.mark.slow
@pytest.mark.parametrize("hl", [1.09e-4, 1e-3])
@pytest.mark.parametrize("soil", [(81.0, 5.0), (10.0, 0.002), (2.5, 1e-5)])
def test_the_pair_orders_agree_with_the_deck_wide_order(hl, soil):
    """The accuracy gate, on #631's grazing wire and momwire#647's soils.
    (`test_negative_controls` holds one rung of it on the default lane.)"""
    kw = _somm(hl, soil)
    z_pair, s = _solve(kw)
    z_deck, _ = _solve(kw, _REMAINDER_PER_PAIR=False)
    assert len(s._last_remainder_orders) > 1  # the per-pair route really ran
    rel = abs(z_pair - z_deck) / abs(z_deck)
    assert rel < PAIR_AGREEMENT, f"{rel:.3e}"


def test_negative_controls():
    """(1) At the deck-wide order as the BASE, the per-pair fill lists no pair
    and is the deck-wide fill to the bit — the two routes differ only in
    which pairs run high. (2) Without the one-ring the pair rule alone misses
    the corners of the ridge and fails the bar. (3) Capping the near-plane
    pairs at the base order is the pre-#631 answer, far outside it."""
    kw = _somm(1.09e-4)
    z_deck, s_deck = _solve(kw, _REMAINDER_PER_PAIR=False)
    (q_deck,) = s_deck._last_remainder_orders
    z_pair, _ = _solve(kw)
    rel = abs(z_pair - z_deck) / abs(z_deck)
    assert rel < PAIR_AGREEMENT, f"{rel:.3e}"
    z_same, s_same = _solve(dict(kw, n_qp_sommerfeld=q_deck))
    assert s_same._last_remainder_orders == {q_deck: 16 * 16}
    assert z_same == z_deck

    z_bare, _ = _solve(kw, _REMAINDER_PAIR_DILATE=False)
    bare = abs(z_bare - z_deck) / abs(z_deck)
    assert bare > 100 * PAIR_AGREEMENT, f"one-ring inert? {bare:.3e}"

    z_low, s_low = _solve(kw, _REMAINDER_QP_CAP=3)
    assert s_low._last_remainder_orders == {3: 16 * 16}
    low = abs(z_low - z_deck) / abs(z_deck)
    assert low > 0.25, f"order-3 answer only {low:.1%} away"


def test_pair_moments_match_the_numpy_route(monkeypatch):
    """The fused kernel is driven through unit pseudo-bases to return raw
    moments; the numpy spelling of the same double sum must agree."""
    if _bs._acc is None or not hasattr(_bs._acc, "sommerfeld_remainder_bspline_Q"):
        pytest.skip("no accelerator")
    s = BSplineSolver(**_somm(1.09e-4))
    g = s._build_geometry()
    from momwire import _ground_refl, _sommerfeld

    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    grid = s._somm_grid(
        eps_t, _sommerfeld.max_image_distance(g["seg_l"], g["seg_r"], s.ground_z)
    )
    obs = np.array([4, 9], dtype=np.int64)
    src = np.array([3, 4, 5], dtype=np.int64)
    fused = s._remainder_pair_moments(g, obs, src, 24, grid)
    monkeypatch.setattr(_bs, "_acc", None)
    ref = s._remainder_pair_moments(g, obs, src, 24, grid)
    assert fused.shape == ref.shape == (2, 3, s.degree + 1, s.degree + 1)
    assert np.abs(fused - ref).max() <= 1e-10 * np.abs(ref).max()


def test_the_pair_rule_is_the_scalar_rule_on_arbitrary_points():
    """Rule-level: over random observer groups and segments near a plane the
    pair maximum is the scalar rule, including the cap and the clip."""
    rng = np.random.default_rng(1189)
    for _ in range(20):
        n = 12
        a = rng.uniform(-3, 3, (n, 3))
        b = a + rng.normal(0, 1.0, (n, 3))
        a[:, 2] = np.abs(a[:, 2]) * 0.05 + 1e-4
        b[:, 2] = np.abs(b[:, 2]) * 0.05 + 1e-4
        t = np.array([0.1, 0.5, 0.9])
        obs = a[:, None, :] + t[None, :, None] * (b - a)[:, None, :]
        scalar = _quadrature.remainder_qp(obs.reshape(-1, 3), a, b, 0.0, 3, 64, 1.0)
        I, J, Q = _quadrature.remainder_qp_pairs(obs, a, b, 0.0, 3, 64, 1.0)
        assert (int(Q.max()) if Q.size else 3) == scalar


# ---------------------------------------------------------------------------
# The padded-slot regression (found by momwire#1131)
# ---------------------------------------------------------------------------
#
# `supp_seg` pads each basis's unused wing slots with segment 0 and a zero
# polynomial. The correction's per-segment wing table read those slots as
# real, so segment 0's list grew with the basis count — 54 wide at 12
# radials, 198 at 48 where a real segment carries three — and the
# (pairs, width, width) einsum transient reached 7 GiB on a 48-radial surface
# screen.


def surface_screen_deck(n_radials, n_rad=10):
    """N6LF's 7.2 MHz vertical over `n_radials` radials at h = 2a (the
    surface convention), over Sommerfeld soil — momwire#1131's deck."""
    ft = 0.3048
    mast, radial, a = 33.5 * ft, 33.0 * ft, 0.51e-3
    h = 2.0 * a
    ang = 2.0 * np.pi * np.arange(n_radials) / n_radials
    wires = [
        np.array([(radial * np.cos(t), radial * np.sin(t), h), (0.0, 0.0, h)])
        for t in ang
    ]
    npe = [[n_rad] for _ in ang]
    m = len(wires)
    wires.append(np.array([(0.0, 0.0, z) for z in (mast + h, 0.5 + h, 0.05 + h, h)]))
    npe.append([19, 2, 3])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[[(i, "end") for i in range(n_radials)] + [(m, "end")]],
        feeds=[(m, mast - 0.05, 1 + 0j)],
        wavelength=C0 / 7.2e6,
        wire_radius=a,
        degree=2,
        ground_z=0.0,
        ground_eps=(30.0, 0.020),
        ground_model="sommerfeld",
    )


@pytest.mark.parametrize("n_radials", [12, 48])
def test_the_wing_table_holds_only_real_wings(n_radials):
    """The table is as wide as the most wings any segment REALLY carries —
    degree + 1 on a b-spline mesh, whatever the basis count — while the raw
    padded map, the regression's spelling, grows with the radials."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = BSplineSolver(**surface_screen_deck(n_radials))
    g = s._build_geometry()
    supp, polys = s._build_basis_polynomials(g)[:2]
    n_seg = g["seg_l"].shape[0]
    ent = BSplineSolver._segment_wing_table(supp, polys, n_seg)
    assert ent.shape[1] <= s.degree + 1, ent.shape
    # Negative control: the padded reading is the wide one.
    assert np.bincount(supp.ravel()).max() > 4 * n_radials
    # Nothing real was dropped: every wing with a nonzero polynomial is listed
    # exactly once, on its own segment.
    listed = np.sort(ent[ent >= 0])
    real = np.flatnonzero(np.any(polys != 0.0, axis=2).ravel())
    assert np.array_equal(listed, real)
    rows = np.nonzero(ent >= 0)[0]
    assert np.array_equal(supp.ravel()[ent[ent >= 0]], rows)


def test_the_pair_correction_stays_small_on_a_surface_screen(monkeypatch):
    """Memory bound on the correction itself (tracemalloc sees numpy's
    buffers) on the accelerated route every shipped build takes. Measured
    0.8 MiB on a 12-radial surface screen (Haswell); the (pairs, width,
    width) transient alone was 1021 x 54 x 54 x 16 B = 46 MiB before the fix,
    and 7 GiB at 48 radials."""
    import tracemalloc

    if _bs._acc is None or not hasattr(_bs._acc, "sommerfeld_remainder_bspline_Q"):
        pytest.skip("no accelerator: the numpy route's own transients dominate")

    peaks = []
    orig = BSplineSolver._remainder_pair_correction

    def traced(self, *a, **k):
        tracemalloc.start()
        try:
            return orig(self, *a, **k)
        finally:
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()

    monkeypatch.setattr(BSplineSolver, "_remainder_pair_correction", traced)
    _z, s = _solve(surface_screen_deck(12))
    assert len(peaks) == 1, "the per-pair correction did not run"
    assert s._last_pair_wing_width <= s.degree + 1
    assert peaks[0] < 200 * 2**20, f"correction peaked {peaks[0] / 2**20:.0f} MiB"
