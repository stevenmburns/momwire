"""Sommerfeld-integral engine tests (docs/sommerfeld-ground-plan.md Phase 1).

Layers, from machinery to physics:
  1. contour + branch machinery against a closed form (Sommerfeld
     identity: the module's own contours must reproduce e^{-jkR}/R);
  2. integrand internal identity (Laplacian of J0);
  3. cross-form agreement (Bessel vs Hankel contours at the same point —
     the strongest contour-invariance check, the paths share nothing);
  4. exact limits (free-space eps=1 -> 0, PEC scaling, R1 -> 0 limits);
  5. the NEC-2 theory manual's published surface extrema (figs 7-11).

Working units: wavelengths (k2 = 2*pi) unless a test says otherwise.
"""

import numpy as np
import pytest

from momwire import _sommerfeld as som

from oracle_sommerfeld_figs import FIG_FREQ_MHZ, FIG_ORACLES

K2 = 2.0 * np.pi
EPS_GROUND = 10.0 - 3.6j  # generic lossy ground for machinery tests
EPS0 = 8.8541878128e-12


# ---------------------------------------------------------------------------
# 1. Sommerfeld identity: contours reproduce the closed-form Green's function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rho,h",
    [(0.0, 0.1), (0.02, 0.1), (0.1, 0.5), (0.05, 2.0), (1e-5, 1e-4)],
)
def test_sommerfeld_identity_bessel_form(rho, h):
    num, exact = som.greens_free_space_check(K2, rho, h, "J")
    assert abs(num - exact) / abs(exact) < 1e-8


@pytest.mark.parametrize(
    "rho,h",
    [(0.1, 0.1), (0.5, 0.1), (1.0, 0.02), (0.3, 0.0), (0.05, 0.08), (1e-4, 0.0)],
)
def test_sommerfeld_identity_hankel_form(rho, h):
    num, exact = som.greens_free_space_check(K2, rho, h, "H")
    assert abs(num - exact) / abs(exact) < 1e-8


# ---------------------------------------------------------------------------
# 2. Integrand internal identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["J", "H"])
def test_integrand_laplacian_identity(form):
    """d2/drho2 + (1/rho) d/drho acting on J0(lam*rho) is -lam^2*J0, so
    integrands 0 + 3 + lam^2 * 4 must vanish pointwise."""
    k1 = K2 * np.sqrt(EPS_GROUND)
    lam = np.array([0.3 + 0.1j, 5.0 + 1.0j, 20.0 - 4.0j, 100.0 - 1.0j])
    six = som._integrand_six(lam, 0.3, 0.2, np.conj(k1), K2, form)
    resid = six[0] + six[3] + lam**2 * six[4]
    assert np.max(np.abs(resid)) < 1e-13 * np.max(np.abs(six[0]))


# ---------------------------------------------------------------------------
# 3. Cross-form agreement at identical points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r1", [0.02, 0.1, 0.5, 1.0])
def test_bessel_vs_hankel_same_point(r1):
    """Both forms converge in the band h/2 < rho < 2h; they share no
    contour geometry, so agreement validates both."""
    theta = np.arctan2(1.0, 1.0)  # rho == h
    rho = r1 * np.cos(theta)
    h = r1 * np.sin(theta)
    sj = som._six_integrals(EPS_GROUND, K2, rho, h, form="J")
    sh = som._six_integrals(EPS_GROUND, K2, rho, h, form="H")
    assert np.max(np.abs(sj - sh)) < 1e-7 * np.max(np.abs(sj))


# ---------------------------------------------------------------------------
# 3b. Fig-14 waypoint vs the k1 branch point (issue #161)
# ---------------------------------------------------------------------------
#
# The cross-form check above cannot reach this: the Bessel form needs
# rho ~< 2h, and the waypoint cap only bites once max(rho, h) > ~2.8
# wavelengths, which at rho ~ 2h means h > 1.4 and puts the k1 branch
# point back in the numerically-dead regime. The defect lived exactly in
# the gap — grazing (rho >> h), where only the Hankel form converges — so
# it is gated here against contours that differ only in their waypoint.

EPS_STRESS = 16.0 + 0.0j  # lossless: k1's branch point sits ON the real axis
EPS_DEFAULT = 10.0 - 1.26j
TH_GRAZE = np.radians(0.5)


def _six_hankel_with_waypoint(eps_t, k2, rho, h, d, rtol=1e-11):
    """The fig-14 contour with the caller dictating the waypoint `d`.

    Same role as `greens_free_space_check` one layer down: the value
    cannot depend on where the real-axis run turns into the descending
    tail, provided it turns down clear of the k1 branch point.
    """
    k1 = k2 * np.sqrt(complex(eps_t))
    if k1.imag > 0:
        k1 = np.conj(k1)

    def f(lam):
        return som._integrand_six(lam, rho, h, k1, k2, "H")

    r1 = np.hypot(rho, h)
    dir_right = (h - 1.0j * rho) / r1
    dir_left = (-h - 1.0j * rho) / r1
    a = -0.4j * k2
    b = (0.6 + 0.2j) * k2
    c = (1.02 + 0.2j) * k2
    qtol = min(rtol, 1e-11)
    total = som._adaptive_segment(f, a, b, qtol)
    total = total + som._adaptive_segment(f, b, c, qtol)
    total = total + som._adaptive_segment(f, c, d, qtol)
    ref = np.max(np.abs(total))
    p0 = 0.5 * max(abs(k1), k2)
    total = total + som._tail(f, d, dir_right, 0.2 * np.pi / max(rho, h), rtol, ref, p0)
    total = total - som._tail(f, a, dir_left, 0.2 * np.pi / max(rho, h), rtol, ref, p0)
    return total


@pytest.mark.parametrize(
    "eps_t,r1",
    [
        (EPS_STRESS, 2.7),  # below the old cap/branch-point crossing
        (EPS_STRESS, 2.9),  # just above it
        (EPS_STRESS, 3.5),
        (EPS_DEFAULT, 3.9),
        (EPS_DEFAULT, 4.2),
    ],
)
def test_waypoint_clearance_invariance(eps_t, r1):
    """Straddling the radius where the old max(rho, h)-keyed cap dropped
    the waypoint below k1.real (2.842 wavelengths on the stress ground at
    theta = 0.5 deg, 4.045 on the default one), the shipped contour must
    agree with ones that turn down further out. Contours that keep the
    clearance agree to ~1e-11 of scale among themselves; the capped one
    was off by up to 1.2e-1 (stress) / 1e-8 (default) before #161."""
    rho, h = r1 * np.cos(TH_GRAZE), r1 * np.sin(TH_GRAZE)
    k1r = (K2 * np.sqrt(complex(eps_t))).real
    k1i = -abs((K2 * np.sqrt(complex(eps_t))).imag)
    shipped = som._six_integrals(eps_t, K2, rho, h, 1e-9)
    scale = np.max(np.abs(shipped))
    for mult in (1.2, 1.5):
        ref = _six_hankel_with_waypoint(eps_t, K2, rho, h, mult * k1r + 0.99j * k1i)
        assert np.max(np.abs(shipped - ref)) < 1e-8 * scale, mult


@pytest.mark.parametrize(
    "eps_t,lo,hi",
    [(EPS_STRESS, 2.55, 3.15), (EPS_DEFAULT, 3.75, 4.35)],
)
def test_grazing_scan_has_no_isolated_jump(eps_t, lo, hi):
    """Issue #161's acceptance criterion: a zoomed second-difference scan
    across the old switch radius shows no isolated spike. Curvature varies
    smoothly over so narrow a window, so max/median is a jump detector;
    before the fix it read 8.6e6 (stress) and 2.4e3 (default), after, ~2.
    Runs through _six_integrals_batch, i.e. the C++ kernel when built."""
    r1 = np.linspace(lo, hi, 301)
    surf = som.iv_surfaces_direct(eps_t, K2, r1, np.full_like(r1, TH_GRAZE), rtol=1e-9)
    for kk in som._SURF_KEYS:
        d2 = np.abs(np.diff(surf[kk], 2))
        assert d2.max() < 10.0 * np.median(d2), (kk, d2.max(), np.median(d2))


# ---------------------------------------------------------------------------
# 4. Exact limits
# ---------------------------------------------------------------------------


def test_free_space_limit_is_zero():
    """eps_t = 1 kills D1 and D2 identically: no ground, no remainder."""
    out = som.iv_surfaces_direct(1.0 + 0.0j, K2, [0.05, 0.3], [0.4, 1.2])
    ref = som.iv_surfaces_direct(EPS_GROUND, K2, [0.05, 0.3], [0.4, 1.2])
    scale = max(np.max(np.abs(v)) for v in ref.values())
    assert max(np.max(np.abs(v)) for v in out.values()) < 1e-12 * scale


def test_pec_limit_scales_away():
    """|I| ~ O(1/sqrt(eps)) as eps -> infinity at fixed R1 > 0 (D1's
    leading term is 2/gamma_1 ~ 2/k1); check the trend and smallness."""
    ref = som.iv_surfaces_direct(EPS_GROUND, K2, [0.2], [0.7])
    scale = max(np.max(np.abs(v)) for v in ref.values())
    prev = None
    for mag in (1e4, 1e6, 1e8):
        out = som.iv_surfaces_direct(mag * (1 - 1e-3j), K2, [0.2], [0.7])
        size = max(np.max(np.abs(v)) for v in out.values()) / scale
        if prev is not None:
            assert size < 0.15 * prev  # ~1/sqrt(1e2) per decade-squared step
        prev = size
    assert size < 1e-3


def test_r1_zero_limits_continuity():
    """Direct evaluation at R1 = 1e-4 wavelengths lands on the analytic
    eqs 169-172 limits (relative to each surface's limit scale)."""
    th = np.array([0.05, 0.4, 1.0, np.pi / 2])
    lim = som._limits_r1_zero(EPS_GROUND, K2, th, K2 * som._C_LIGHT, som._MU0)
    near = som.iv_surfaces_direct(EPS_GROUND, K2, np.full_like(th, 1e-4), th)
    for kk in ("IrhoV", "IzV", "IrhoH", "IphiH"):
        dev = np.max(np.abs(near[kk] - lim[kk])) / np.max(np.abs(lim[kk]))
        assert dev < 5e-3, (kk, dev)


def test_iv_surfaces_r1_zero_uses_limits():
    """R1 = 0 entries route to the analytic limits (no integration)."""
    th = np.array([0.3, np.pi / 2])
    out = som.iv_surfaces_direct(EPS_GROUND, K2, np.zeros_like(th), th)
    lim = som._limits_r1_zero(EPS_GROUND, K2, th, K2 * som._C_LIGHT, som._MU0)
    for kk in ("IrhoV", "IzV", "IrhoH", "IphiH"):
        np.testing.assert_allclose(out[kk], lim[kk], rtol=1e-12)


def test_near_free_space_stability():
    """eps_t = 1 + delta exercises the near-singularity at k2 the manual
    warns about (virtual D poles when k1 -> k2); values must be small,
    finite, and roughly proportional to delta."""
    outs = {}
    for delta in (1e-2, 1e-3):
        out = som.iv_surfaces_direct(1.0 + delta - 1e-4j, K2, [0.3], [0.6])
        m = max(np.max(np.abs(v)) for v in out.values())
        assert np.isfinite(m)
        outs[delta] = m
    ratio = outs[1e-2] / outs[1e-3]
    assert 3.0 < ratio < 30.0  # ~linear in delta


# ---------------------------------------------------------------------------
# 5. Manual figure extrema (figs 7-11)
# ---------------------------------------------------------------------------

# Unit-moment conversion: the manual plots Idl = 1 A*wavelength at 10 MHz
# (pinned in oracle_sommerfeld_figs.py) -> multiply our A*m surfaces by
# lambda(10 MHz) in meters.
_LAMBDA_M = som._C_LIGHT / (FIG_FREQ_MHZ * 1e6)


@pytest.fixture(scope="module")
def fig_meshes():
    """One surface mesh per (eps_r, sigma) ground in the oracle table.
    31x13 samples R1 in [1e-3, 1] wavelengths x theta in [0, 90] deg —
    comparable to the manual's plot mesh."""
    r1, th = np.meshgrid(
        np.linspace(1e-3, 1.0, 31),
        np.radians(np.linspace(0.0, 90.0, 13)),
        indexing="ij",
    )
    omega = 2.0 * np.pi * FIG_FREQ_MHZ * 1e6
    meshes = {}
    for _comp, er, sig in FIG_ORACLES:
        if (er, sig) in meshes:
            continue
        eps_t = er - 1j * sig / (omega * EPS0)
        meshes[(er, sig)] = som.iv_surfaces_direct(
            eps_t, K2, r1, th, rtol=1e-7, omega=omega
        )
    return meshes


@pytest.mark.parametrize("comp,er,sig", sorted(FIG_ORACLES))
def test_manual_figure_extrema(fig_meshes, comp, er, sig):
    """Computed surface extrema against the NEC-2 manual's printed
    Max/Min (figs 7-11). Tolerance is dominated by mesh sampling of the
    small-theta ridge: 6% of the surface's own extremum scale for the
    lossy eps_r=4 figures, 15% for the oscillatory lossless fig 11."""
    rng = FIG_ORACLES[(comp, er, sig)]
    v = fig_meshes[(er, sig)][comp] * _LAMBDA_M
    tol = 0.15 if sig == 0.0 else 0.06
    for part, arr in (("re", v.real), ("im", v.imag)):
        lo, hi = rng[part]
        scale = max(abs(lo), abs(hi))
        assert abs(arr.min() - lo) < tol * scale, (part, "min", arr.min(), lo)
        assert abs(arr.max() - hi) < tol * scale, (part, "max", arr.max(), hi)


# ---------------------------------------------------------------------------
# 6. Interpolation grid (Phase 2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lossy_grid():
    return som.SommerfeldGrid(10.0 - 1.26j, K2, r1_max=1.2)


def test_grid_matches_direct(lossy_grid):
    """Interpolation vs direct evaluation on random points across all
    three regions: within 2e-3 of the global surface scale (NEC's own
    measured range is 1e-3..1e-4 on its coarser 1975-vintage grid)."""
    rng = np.random.default_rng(11)
    r1 = np.concatenate([rng.uniform(0.004, 0.19, 40), rng.uniform(0.21, 1.19, 60)])
    th = rng.uniform(0.0, np.pi / 2, r1.size)
    gi = lossy_grid.eval(r1, th)
    di = som.iv_surfaces_direct(10.0 - 1.26j, K2, r1, th, rtol=1e-8)
    for kk in som._SURF_KEYS:
        scale = np.max(np.abs(di[kk]))
        assert np.max(np.abs(gi[kk] - di[kk])) < 2e-3 * scale, kk


def test_grid_stress_lossless():
    """Zero-loss eps_r=16 (manual fig 11): the evanescent interface wave
    is the worst case for the grid; gate at 4e-3 of scale."""
    eps_t = 16.0 + 0.0j
    g = som.SommerfeldGrid(eps_t, K2, r1_max=1.2)
    rng = np.random.default_rng(3)
    r1 = rng.uniform(0.01, 1.19, 60)
    th = rng.uniform(0.0, np.pi / 2, r1.size)
    gi = g.eval(r1, th)
    di = som.iv_surfaces_direct(eps_t, K2, r1, th, rtol=1e-8)
    for kk in som._SURF_KEYS:
        scale = np.max(np.abs(di[kk]))
        assert np.max(np.abs(gi[kk] - di[kk])) < 4e-3 * scale, kk


def test_grid_stress_lossless_past_the_waypoint_switch():
    """The same stress ground sampled from 1.5 to 4.9 wavelengths.

    The test above stops at 1.2, and every other stress sample sits below
    it too — which left the whole region past R1 ~ 2.84, where the fig-14
    waypoint cap used to cross k1's branch point (issue #161), untested.
    Same 4e-3-of-scale bar; measured 1.4e-3 across the four surfaces."""
    eps_t = 16.0 + 0.0j
    g = som.SommerfeldGrid(eps_t, K2, r1_max=5.0)
    rng = np.random.default_rng(5)
    r1 = rng.uniform(1.5, 4.9, 40)
    th = rng.uniform(0.0, np.pi / 2, r1.size)
    gi = g.eval(r1, th)
    di = som.iv_surfaces_direct(eps_t, K2, r1, th, rtol=1e-8)
    for kk in som._SURF_KEYS:
        scale = np.max(np.abs(di[kk]))
        assert np.max(np.abs(gi[kk] - di[kk])) < 4e-3 * scale, kk


@pytest.fixture(scope="module")
def far_grid():
    """A grid past the near/far split (issue #159): lossy ground, 7 lambda."""
    return som.SommerfeldGrid(10.0 - 1.26j, K2, r1_max=7.0)


def test_grid_far_zone_layout(far_grid):
    """Past _SOMM_R1_NEAR_LAMBDA the grid adds two coarse far regions
    instead of keying the fine near spacings to the full extent (which grew
    the node count ~quadratically with geometry size — issue #159). The
    near tabulation stops at r_near and region 2's theta keying follows it."""
    assert som._SOMM_R1_NEAR_LAMBDA == pytest.approx(4.0)
    g = far_grid
    assert g.r_near == pytest.approx(4.0)  # lambda = 1 in these units
    assert len(g._regions) == 5
    near2, near3, far2, far3 = g._regions[1:]
    # Near regions tabulate to r_near (last row may pad one dr past it).
    for reg in (near2, near3):
        r_last = reg["r0"] + reg["dr"] * (reg["n_r"] - 1)
        assert g.r_near <= r_last < g.r_near + reg["dr"] + 1e-12
    # dth2 keys to r_near, not r1_max: 0.07/4 rad = 1.002 deg -> 21 columns.
    assert near2["n_th"] == 21
    # Far regions: [r_near, r1_max] on the coarse lattice.
    for reg, dth_deg in ((far2, som._SOMM_DTH_FAR_DEG), (far3, 10.0)):
        assert reg["r0"] == pytest.approx(g.r_near)
        assert reg["dr"] == pytest.approx(som._SOMM_DR_FAR_LAMBDA)
        assert reg["dth"] == pytest.approx(np.radians(dth_deg))
        r_last = reg["r0"] + reg["dr"] * (reg["n_r"] - 1)
        assert g.r1_max <= r_last < g.r1_max + reg["dr"] + 1e-12
    # The point of the split: far fewer nodes than the near-keyed layout
    # (~6.6k at 7 lambda) and the count now grows linearly with extent.
    assert sum(r["n_r"] * r["n_th"] for r in g._regions) < 3100


def test_grid_far_zone_accuracy(far_grid):
    """Random points across all five regions vs direct evaluation: the far
    zone holds the same 2e-3 global-scale bar as the near zone (the
    surfaces' lateral-wave fine structure has decayed out there — the
    calibration behind _SOMM_R1_NEAR_LAMBDA)."""
    rng = np.random.default_rng(17)
    r1 = np.concatenate([rng.uniform(0.004, 3.9, 40), rng.uniform(4.1, 6.99, 60)])
    th = rng.uniform(0.0, np.pi / 2, r1.size)
    gi = far_grid.eval(r1, th)
    di = som.iv_surfaces_direct(10.0 - 1.26j, K2, r1, th, rtol=1e-8)
    for kk in som._SURF_KEYS:
        scale = np.max(np.abs(di[kk]))
        assert np.max(np.abs(gi[kk] - di[kk])) < 2e-3 * scale, kk


def test_grid_far_zone_matches_near_keyed_layout(far_grid, monkeypatch):
    """The split changes the tabulation, not the answers: far-zone queries
    agree with the pre-#159 near-keyed layout (forced by raising the split
    past r1_max) to within the two interpolants' own error budgets."""
    monkeypatch.setattr(som, "_SOMM_R1_NEAR_LAMBDA", 100.0)
    ref = som.SommerfeldGrid(10.0 - 1.26j, K2, r1_max=7.0)
    assert len(ref._regions) == 3  # the old layout
    rng = np.random.default_rng(23)
    r1 = rng.uniform(4.05, 6.95, 80)
    th = rng.uniform(0.0, np.pi / 2, r1.size)
    a = far_grid.eval(r1, th)
    b = ref.eval(r1, th)
    for kk in som._SURF_KEYS:
        scale = np.max(np.abs(b[kk]))
        assert np.max(np.abs(a[kk] - b[kk])) < 2e-3 * scale, kk


def test_grid_small_extent_keeps_pre_split_layout():
    """r1_max at/below the split builds exactly the pre-#159 grid: three
    regions, theta keying to r1_max, no far tables."""
    g = som.SommerfeldGrid(10.0 - 1.26j, K2, r1_max=3.0)
    assert len(g._regions) == 3
    assert g.r_near == pytest.approx(g.r1_max)
    # dth2 keyed to r1_max: 0.07/3 rad = 1.337 deg -> ceil(20/1.337)+1 = 16.
    assert g._regions[1]["n_th"] == 16


def test_grid_r1_max_is_capped(monkeypatch):
    """A geometry that would size the grid to hundreds of wavelengths — the
    NEC TL-anchor idiom, or any large structure over real ground (issue
    #157) — is capped at _SOMM_R1_CAP_LAMBDA wavelengths so the fill stays
    bounded instead of doing millions of oscillatory integrals. lambda =
    2 pi / K2 = 1 in these units."""
    # The shipped default is the calibrated 15 lambda.
    assert som._SOMM_R1_CAP_LAMBDA == pytest.approx(15.0)
    # Exercise the mechanism at a small cap so the grids build cheaply.
    monkeypatch.setattr(som, "_SOMM_R1_CAP_LAMBDA", 3.0)
    som._GRID_CACHE.clear()
    som._NORM_CACHE.clear()
    lam = 2.0 * np.pi / K2
    cap = 3.0 * lam
    g = som.SommerfeldGrid(10.0 - 1.26j, K2, r1_max=1000.0 * lam)
    assert g.r1_max == pytest.approx(cap)
    # get_grid caps before its 1.25**n bucketing too, so an oversized request
    # keys to the capped grid rather than minting a distinct huge entry.
    gg = som.get_grid(10.0 - 1.26j, K2, 1000.0 * lam, omega=K2 * som._C_LIGHT)
    assert gg.r1_max == pytest.approx(cap)
    # A modest geometry well under the cap is untouched.
    small = som.SommerfeldGrid(10.0 - 1.26j, K2, r1_max=2.0 * lam)
    assert small.r1_max == pytest.approx(2.0 * lam)


def test_grid_r1_zero_and_bounds(lossy_grid):
    """R1 = 0 queries interpolate onto the analytic-limit row; a query
    beyond r1_max clamps to r1_max (the far-pair cap, issue #157) rather
    than raising; a negative R1 still raises; theta is clipped to [0, pi/2]."""
    th = np.array([0.2, 1.0])
    lim = som._limits_r1_zero(10.0 - 1.26j, K2, th, K2 * som._C_LIGHT, som._MU0)
    out = lossy_grid.eval(np.zeros_like(th), th)
    for kk in som._SURF_KEYS:
        np.testing.assert_allclose(out[kk], lim[kk], rtol=2e-3, atol=0)
    # Beyond r1_max: clamped, so a far query equals the r1_max edge (the C++
    # proj_one path clamps identically). Was a ValueError before #157.
    beyond = lossy_grid.eval([1.5], [0.3])
    at_cap = lossy_grid.eval([lossy_grid.r1_max], [0.3])
    for kk in som._SURF_KEYS:
        np.testing.assert_allclose(beyond[kk], at_cap[kk], rtol=1e-12)
    # A negative R1 is a genuine bug and still raises.
    with pytest.raises(ValueError):
        lossy_grid.eval([-0.1], [0.3])
    a = lossy_grid.eval([0.5], [np.pi / 2])
    b = lossy_grid.eval([0.5], [np.pi / 2 + 1e-12])
    for kk in som._SURF_KEYS:
        np.testing.assert_allclose(a[kk], b[kk], rtol=1e-9)
