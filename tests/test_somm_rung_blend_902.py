"""momwire#902 — the Sommerfeld master ladder serves a query BETWEEN two rungs
as the linear blend of the bracketing pair, filled on one shared lattice.

The symptom: inside EZNEC, a 20 m SWR sweep of `Bydipole1` stalled five
times for master fills (6.3 s of a 7.3 s sweep) — four crossings of the 1 %
nearest-rung ladder and one r1-bucket edge — against a licensed engine that
never stalls. The study behind the fix (the issue and the calibration note
at `_SOMM_EPS_IM_BUCKET`): on soil εr 20 / σ 0.0303 at 14 MHz, r1 = 1 λ,
max over every lattice node per surface,

    1 % nearest-rung snap, 0.5 % offset .... 2.0e-3
    10 % pair, blend at the midpoint ....... 5.8e-4
    20 % pair, blend at the midpoint ....... 2.2e-3

so a 10 % ladder with the blend is MORE accurate than the 1 % snap it
replaces and needs a tenth of the rungs. The gates here pin: the blend's
error against an exact fill on the same lattice (the oracle), the pair's
shared lattice, the cache's fill economy over a band sweep, the r1 slack,
and the pass-throughs (a rung exactly, the ladder off, nonstandard ε̃).
"""

import numpy as np
import pytest

from momwire import _sommerfeld as sm

C0 = 299792458.0
EPS0 = 8.8541878128e-12


def _eps(eps_r, sigma, f):
    return complex(eps_r, -sigma / (2.0 * np.pi * f * EPS0))


def _fill_on(grid, eps):
    """The four surfaces at `eps` on `grid`'s own lattice — the exact fill a
    blended table is measured against."""
    out = []
    for i, reg in enumerate(grid._regions):
        r = reg["r0"] + reg["dr"] * np.arange(reg["n_r"])
        th = reg["th0"] + reg["dth"] * np.arange(reg["n_th"])
        rr, tt = np.meshgrid(r, th, indexing="ij")
        if i == 1 and grid._r0_fill > 0.0:
            rr = rr.copy()
            rr[0, :] = grid._r0_fill
        s = sm.iv_surfaces_direct(
            eps, grid.k2, rr, tt, rtol=1e-6, omega=grid.omega, mu=grid.mu
        )
        out.append(np.stack([s[k] for k in sm._SURF_KEYS]))
    return out


def _err(tables, exact):
    return max(
        float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)
        for a, b in zip(tables, exact)
    )


def _clear():
    sm._GRID_CACHE.clear()
    sm._NORM_CACHE.clear()


@pytest.fixture(autouse=True)
def _fresh_caches():
    _clear()
    yield
    _clear()


def test_g902_1_rungs_bracket_and_weight():
    """`_somm_eps_rungs`: lo and hi are adjacent rungs of the geometric
    ladder with Im(eps) between them, t is the linear weight, and a value
    ON a rung, the ladder disabled, or a nonstandard ε̃ give one master."""
    step = 1.0 + sm._SOMM_EPS_IM_BUCKET
    e = complex(20.0, -38.4)
    lo, hi, t = sm._somm_eps_rungs(e)
    assert lo.real == hi.real == e.real
    assert -lo.imag <= -e.imag <= -hi.imag
    assert -hi.imag == pytest.approx(-lo.imag * step)
    assert (1.0 - t) * lo.imag + t * hi.imag == pytest.approx(e.imag)
    for n in (3, 38, -7):
        on = complex(20.0, -(step**n))
        assert sm._somm_eps_rungs(on) == (on, on, 0.0)
    for weird in (16.0 + 0.0j, 1.0 + 0.0j, 10.0 + 2.0j, -3.0 - 1.0j):
        assert sm._somm_eps_rungs(weird) == (weird, weird, 0.0)


def test_g902_2_the_blend_beats_the_snap_against_an_exact_fill():
    """The oracle: an exact fill at the query eps on the pair's lattice. The
    blend at the rung MIDPOINT (its worst case) must sit under 1e-3 and
    under the nearest-rung snap at the same offset — measured 5.8e-4 vs
    1.9e-2 on this soil."""
    f = 14.175e6
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    lam = 2.0 * np.pi / k2
    # The cheap soil (εr 10 / σ 0.002, |k1| small, fast rows) keeps this in
    # the fast lane; the Bydipole1-soil numbers are the slow gate below.
    lo, hi, _ = sm._somm_eps_rungs(_eps(10.0, 0.002, f))
    mid = complex(10.0, -0.5 * (-lo.imag - hi.imag))
    # 0.35 lambda: the smallest grid (four regions), which keeps this in the
    # fast lane; the study's numbers are at 1 lambda, the ratio is the same.
    g_lo = sm.SommerfeldGrid(lo, k2, 0.35 * lam, omega=om, lattice_eps=hi)
    g_hi = sm.SommerfeldGrid(hi, k2, 0.35 * lam, omega=om, lattice_eps=hi)
    exact = _fill_on(g_hi, mid)
    _, _, t = sm._somm_eps_rungs(mid)
    blend = g_lo.blend(g_hi, t, mid)
    e_blend = _err([r["vals"] for r in blend._regions], exact)
    e_snap = min(
        _err([r["vals"] for r in g_lo._regions], exact),
        _err([r["vals"] for r in g_hi._regions], exact),
    )
    assert e_blend < 1e-3, e_blend
    assert e_blend < 0.25 * e_snap, (e_blend, e_snap)
    assert blend.eps_t == mid and blend.lattice_eps == hi


def test_g902_3_a_pair_shares_the_demanding_rungs_lattice():
    """Both masters of a pair are filled on the lattice of the rung with the
    larger |Im| (larger |k1| keys the spacing down), so the blend is
    elementwise; a lattice keyed at the other rung would differ in node
    count, and `blend` refuses it rather than resampling."""
    f = 14.0e6
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    lam = 2.0 * np.pi / k2
    lo, hi, _ = sm._somm_eps_rungs(_eps(10.0, 0.002, f))
    a = sm.SommerfeldGrid(lo, k2, 0.35 * lam, omega=om, lattice_eps=hi)
    b = sm.SommerfeldGrid(hi, k2, 0.35 * lam, omega=om)
    assert [r["n_r"] for r in a._regions] == [r["n_r"] for r in b._regions]
    assert [r["n_th"] for r in a._regions] == [r["n_th"] for r in b._regions]
    # A different lattice (here: a different r1 bucket) is refused, not
    # resampled. On this cheap soil the two rungs' own lattices happen to
    # coincide; the lossy-soil case where they differ is the slow gate.
    other = sm.SommerfeldGrid(lo, k2, 0.5 * lam, omega=om, lattice_eps=hi)
    with pytest.raises(ValueError, match="not one lattice"):
        other.blend(b, 0.5, lo)


def test_g902_4_a_band_sweep_is_one_pair_and_a_second_sweep_is_free():
    """The fill economy the issue asked for: 50 points across 2.5 % (the
    20 m band) on the Bydipole1 soil fill at most one pair — two masters,
    or four if the band straddles a rung — all at the FIRST point, and a
    second sweep of the same band fills nothing. Before #902 this sweep
    filled five times, spread through the sweep."""
    fills = []
    orig = sm.SommerfeldGrid.__init__

    def counting(self, *a, **kw):
        fills.append((a[0], kw.get("lattice_eps")))
        orig(self, *a, **kw)

    sm.SommerfeldGrid.__init__ = counting
    try:
        # 28.35-29.06 MHz: the same 2.5 % span as the 20 m band, on the
        # cheap soil at the frequency the #159 reuse gate fills in the fast
        # lane (the fill cost is the slow grazing rows, which grow with loss)
        freqs = np.linspace(28.35e6, 29.06e6, 50)
        first_point_fills = None
        for i, f in enumerate(freqs):
            k2 = 2.0 * np.pi * f / C0
            sm.get_grid(
                _eps(20.0, 0.0303, f), k2, 0.98 * 2.0 * np.pi / k2, 2.0 * np.pi * f
            )
            if i == 0:
                first_point_fills = len(fills)
        n_sweep = len(fills)
        for f in freqs:
            k2 = 2.0 * np.pi * f / C0
            sm.get_grid(
                _eps(20.0, 0.0303, f), k2, 0.98 * 2.0 * np.pi / k2, 2.0 * np.pi * f
            )
    finally:
        sm.SommerfeldGrid.__init__ = orig
    assert first_point_fills == 2, fills
    assert n_sweep in (2, 4), fills
    assert len(fills) == n_sweep, "the second sweep filled"
    # every master of a pair carries the pair's lattice rung
    assert all(le is not None for _, le in fills)


def test_g902_5_r1_slack_keeps_a_sweep_inside_one_bucket():
    """Bydipole1's r1 sits at 0.98 λ at 14.0 MHz and crossed the 1.0 λ
    bucket edge at 14.307 MHz — one of the five stalls. The same shape at a
    cheaper edge: r1 = 0.40 λ sits just under the 1.25⁻⁴ = 0.4096 λ bucket
    and would cross it 2.5 % up the band; with the slack the whole band
    keys to the 0.512 λ bucket."""
    slack = sm._SOMM_R1_SWEEP_SLACK
    assert 0.0 < slack <= 0.1
    fills = []
    orig = sm.SommerfeldGrid.__init__

    def counting(self, *a, **kw):
        fills.append(a[2])
        orig(self, *a, **kw)

    sm.SommerfeldGrid.__init__ = counting
    try:
        rungs = set()
        for f in (28.35e6, 29.06e6):
            k2 = 2.0 * np.pi * f / C0
            lam = 2.0 * np.pi / k2
            e = _eps(10.0, 0.002, f)
            rungs.add(sm._somm_eps_rungs(e)[:2])
            sm.get_grid(e, k2, 0.40 * lam, 2.0 * np.pi * f)
    finally:
        sm.SommerfeldGrid.__init__ = orig
    # one r1 bucket across the band (the pairs may differ, the bucket not)
    assert len({round(r, 9) for r in fills}) == 1, fills
    assert fills[0] == pytest.approx(1.25**-3)
    # and without the slack the band WOULD have straddled the edge
    assert sm._somm_r1_bucket_wl(0.40) != sm._somm_r1_bucket_wl(0.40 * 1.025)


def test_g902_6_ladder_off_and_on_a_rung_fill_exactly(monkeypatch):
    """The two single-master paths: the ladder disabled (env 0) fills at the
    exact eps_t with no blend; a query ON a rung uses that rung's master
    alone — bit-identical to a fresh exact fill scaled the same way."""
    f = 14.0e6
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    lam = 2.0 * np.pi / k2
    e = _eps(10.0, 0.002, f)

    monkeypatch.setattr(sm, "_SOMM_EPS_IM_BUCKET", 0.0)
    g = sm.get_grid(e, k2, 0.35 * lam, om)
    assert g.eps_t == e and g.lattice_eps == e
    assert len(sm._NORM_CACHE) == 1 and next(iter(sm._NORM_CACHE))[1] == e
    monkeypatch.setattr(sm, "_SOMM_EPS_IM_BUCKET", 0.10)
    _clear()

    step = 1.1
    on = complex(10.0, -(step**3))
    g = sm.get_grid(on, k2, 0.35 * lam, om)
    assert g.eps_t == on and len(sm._NORM_CACHE) == 1
    fresh = sm.SommerfeldGrid(
        on, sm._K2_REF, next(iter(sm._NORM_CACHE))[2], omega=sm._K2_REF * sm._C_LIGHT
    ).scaled_to(k2, om, sm._MU0)
    for a, b in zip(g._regions, fresh._regions):
        assert np.array_equal(a["vals"], b["vals"])


@pytest.mark.slow
def test_g902_7_view_matches_direct_at_the_true_eps_across_the_band():
    """End to end: the blended, rescaled view against direct evaluation at
    the TRUE eps at random (R1, θ) over the band's three points — the grid
    bar the 1 % snap was held to, now held by the 10 % blend."""
    rng = np.random.default_rng(902)
    for f in (14.0e6, 14.175e6, 14.35e6):
        k2 = 2.0 * np.pi * f / C0
        om = 2.0 * np.pi * f
        lam = 2.0 * np.pi / k2
        e = _eps(20.0, 0.0303, f)
        v = sm.get_grid(e, k2, 0.9 * lam, om)
        r1 = rng.uniform(0.0, 0.9 * lam, 120)
        th = rng.uniform(0.0, np.pi / 2, 120)
        got = v.eval(r1, th)
        want = sm.iv_surfaces_direct(e, k2, r1, th, rtol=1e-8, omega=om)
        for kk in sm._SURF_KEYS:
            scale = np.abs(want[kk]).max()
            assert np.abs(got[kk] - want[kk]).max() < 2.5e-3 * scale, (f, kk)


@pytest.mark.slow
def test_g902_8_the_study_numbers_on_the_bydipole1_soil():
    """The #902 study's own row, as a gate: soil εr 20 / σ 0.0303 at
    14.175 MHz, r1 = 1 λ, the pair's blend at the midpoint against an exact
    fill on the pair's lattice — measured 5.8e-4 against the 1 % snap's
    2.0e-3 and the same-offset snap's 1.9e-2. Held at 1e-3 and at a quarter
    of the snap."""
    f = 14.175e6
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    lam = 2.0 * np.pi / k2
    lo, hi, _ = sm._somm_eps_rungs(_eps(20.0, 0.0303, f))
    mid = complex(20.0, -0.5 * (-lo.imag - hi.imag))
    g_lo = sm.SommerfeldGrid(lo, k2, 1.0 * lam, omega=om, lattice_eps=hi)
    g_hi = sm.SommerfeldGrid(hi, k2, 1.0 * lam, omega=om, lattice_eps=hi)
    exact = _fill_on(g_hi, mid)
    _, _, t = sm._somm_eps_rungs(mid)
    blend = g_lo.blend(g_hi, t, mid)
    e_blend = _err([r["vals"] for r in blend._regions], exact)
    e_snap = min(
        _err([r["vals"] for r in g_lo._regions], exact),
        _err([r["vals"] for r in g_hi._regions], exact),
    )
    assert e_blend < 1e-3, e_blend
    assert e_blend < 0.25 * e_snap, (e_blend, e_snap)
    # and on this soil the rungs' OWN lattices differ, which is why a pair
    # is filled on one lattice rather than blended across two
    own = sm.SommerfeldGrid(lo, k2, 1.0 * lam, omega=om)
    assert [r["n_r"] for r in own._regions] != [r["n_r"] for r in g_hi._regions]
    with pytest.raises(ValueError, match="not one lattice"):
        own.blend(g_hi, t, mid)
