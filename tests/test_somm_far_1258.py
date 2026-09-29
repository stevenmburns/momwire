"""momwire#1258: the Sommerfeld ground remainder past the grid's edge.

The interpolation grid stops at `r1_max` (at most 15 wavelengths, #157). Past
it the surfaces used to be served FROZEN at the edge, which near grazing
overstates the ground wave by a factor growing like R1 (56x at 1000 m on the
#1257 test bed). They are now continued by `_sommerfeld.far_surfaces` (saddle,
Sommerfeld pole and lateral wave), matched to the edge value.

The gates, each against something other than the code under test:

* an independent evaluation of the same integrals (`somm_far_oracle_1258`:
  the real lambda axis, Bessel form, brute force), itself held to
  `iv_surfaces_direct` inside the cap and to its own panel-doubling;
* the textbook geometric-optics limit (the plane-wave Fresnel coefficients);
* continuity at the edge, and the C++ kernels against the numpy body there;
* licensed NEC-5 (x13) printouts of decks we wrote, near the horizon at 20,
  50 and 1000 wavelengths (tests/fixtures/somm_far_1258/README.md).

Each value gate also shows the FROZEN edge failing it, so none of them can
pass by not reaching the code past the edge.
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path

import numpy as np
import pytest
import somm_far_oracle_1258 as oracle

from momwire import _sommerfeld as sm
from momwire.eznec._shell import render

C = sm._C_LIGHT
EPS0 = 8.8541878128e-12


def _ground(er, sigma, f_hz):
    k = 2.0 * np.pi * f_hz / C
    om = k * C
    return er - 1j * sigma / (om * EPS0), k, om


# The #1257 test bed's ground and three that stress other parts of the form:
# sea water (the pole next to the saddle), a dry low-loss soil (the lateral
# wave through the ground survives), a poor soil (a small |eps|).
GROUNDS = {
    "average-299.8MHz": (13.0, 0.005, 299.8e6),
    "sea-7MHz": (80.0, 5.0, 7e6),
    "dry-28MHz": (3.0, 1e-4, 28e6),
    "poor-14MHz": (5.0, 0.001, 14e6),
}
# The default lane's sample; the slow lane walks the full one.
THETAS_DEG = (0.2, 2.0, 10.0, 45.0)
RANGES_WL = (16.0, 1000.0)
THETAS_DEG_FULL = (
    0.05,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    45.0,
    60.0,
    80.0,
    89.0,
)
RANGES_WL_FULL = (15.5, 16.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 3000.0)


def _scale(om):
    """|C1 k2^2| = omega mu0 / 4 pi: the image field's own scale, the unit
    the surfaces are measured in."""
    return om * sm._MU0 / (4.0 * np.pi)


def _tolerance(r1_wl, cap_wl):
    """The past-edge envelope, in units of `_scale`: what the matched
    continuation carries forward decays like r1_max/R1 (the table's own
    error at the edge plus the saddle's second-order term), over a floor.
    Measured worst (the four grounds): 1.1e-3 at 16 wavelengths (dry),
    3.5e-4 at 50, 8.8e-5 at 200, 2.6e-5 at 1000 (sea)."""
    return 2.5e-3 * cap_wl / r1_wl + 5e-5


def test_the_oracle_is_the_direct_evaluator_where_that_is_sound():
    """Inside the cap `iv_surfaces_direct` is the reference; the real-axis
    oracle must reproduce it before it may judge anything past it. And its
    own answer must not move when the panels are doubled."""
    eps, k, om = _ground(*GROUNDS["average-299.8MHz"])
    lam = 2.0 * np.pi / k
    sc = _scale(om)
    for r_wl in (2.0, 5.0, 10.0):
        for th in (0.5, 5.0, 20.0, 45.0, 80.0):
            t = np.radians(th)
            o = oracle.surfaces(eps, k, r_wl * lam, t)
            d = sm.iv_surfaces_direct(eps, k, [r_wl * lam], [t], rtol=1e-10)
            err = max(abs(o[q] - d[q][0]) for q in oracle.KEYS) / sc
            assert err < 1e-8, (r_wl, th, err)
    t = np.radians(0.5)
    a = oracle.surfaces(eps, k, 1000.0 * lam, t)
    b = oracle.surfaces(eps, k, 1000.0 * lam, t, density=2.0)
    # Measured 4.4e-9: three decades under the tightest envelope below.
    assert max(abs(a[q] - b[q]) for q in oracle.KEYS) / sc < 1e-7


@functools.lru_cache(maxsize=None)
def _past_edge(name, full=False):
    """(served error, frozen-edge error) per (R1, theta), in `_scale` units,
    on the cached production grid for a geometry reaching 1000 wavelengths."""
    eps, k, om = _ground(*GROUNDS[name])
    lam = 2.0 * np.pi / k
    grid = sm.get_grid(eps, k, 1000.0 * lam, om)
    assert grid.r1_max == pytest.approx(sm._SOMM_R1_CAP_LAMBDA * lam)
    sc = _scale(om)
    out = {}
    for th in THETAS_DEG_FULL if full else THETAS_DEG:
        t = np.radians(th)
        edge = grid.eval([grid.r1_max], [t])
        for r_wl in RANGES_WL_FULL if full else RANGES_WL:
            o = oracle.surfaces(eps, k, r_wl * lam, t)
            got = grid.eval([r_wl * lam], [t])
            served = max(abs(o[q] - got[q][0]) for q in oracle.KEYS) / sc
            frozen = max(abs(o[q] - edge[q][0]) for q in oracle.KEYS) / sc
            out[(r_wl, th)] = (served, frozen)
    return out


def _check_past_edge(name, full):
    cap_wl = sm._SOMM_R1_CAP_LAMBDA
    res = _past_edge(name, full)
    for (r_wl, th), (served, _) in res.items():
        assert served <= _tolerance(r_wl, cap_wl), (name, r_wl, th, served)
    # The frozen edge fails the same envelope, by an order of magnitude or
    # more near grazing at range: the gate reaches the continuation.
    worst_frozen = max(
        fz / _tolerance(r_wl, cap_wl)
        for (r_wl, th), (_, fz) in res.items()
        if th <= 2.0 and r_wl >= 200.0
    )
    assert worst_frozen > 50.0, (name, worst_frozen)


@pytest.mark.parametrize("name", sorted(GROUNDS))
def test_past_the_edge_the_surfaces_are_the_oracles(name):
    _check_past_edge(name, full=False)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(GROUNDS))
def test_past_the_edge_the_surfaces_are_the_oracles_everywhere(name):
    """The same envelope over 13 angles and 10 ranges out to 3000 lambda."""
    _check_past_edge(name, full=True)


def test_the_continuation_is_continuous_at_the_edge():
    """No step in the served surfaces as R1 crosses r1_max (the edge is the
    interpolated table; the continuation is matched to it)."""
    eps, k, om = _ground(*GROUNDS["average-299.8MHz"])
    lam = 2.0 * np.pi / k
    grid = sm.get_grid(eps, k, 1000.0 * lam, om)
    th = np.radians(np.array([0.0, 0.05, 0.5, 2.0, 10.0, 30.0, 60.0, 89.0, 90.0]))
    edge = grid.r1_max
    inside = grid.eval(np.full(th.shape, edge * (1.0 - 1e-9)), th)
    outside = grid.eval(np.full(th.shape, edge * (1.0 + 1e-9)), th)
    sc = _scale(om)
    for q in sm._SURF_KEYS:
        assert np.all(np.isfinite(outside[q]))
        assert np.max(np.abs(outside[q] - inside[q])) / sc < 1e-9


def test_far_out_the_surfaces_are_the_fresnel_reflection():
    """R1 -> inf at a fixed angle: the remainder is the plane-wave reflected
    field less the C2 image, i.e. I_z^V -> C1 k2^2 cos^2(t) (R_TM - C2) and
    I_phi^H -> -C1 k2^2 (R_TE + C2), with the textbook Fresnel coefficients
    written from Delta = sqrt(eps - cos^2 t)."""
    eps, k, om = _ground(*GROUNDS["average-299.8MHz"])
    c1k = -1j * om * sm._MU0 / (4.0 * np.pi)
    c2 = (eps - 1.0) / (eps + 1.0)
    t = np.radians(np.array([3.0, 10.0, 30.0, 60.0, 85.0]))
    root = np.sqrt(eps - np.cos(t) ** 2)
    r_tm = (eps * np.sin(t) - root) / (eps * np.sin(t) + root)
    r_te = (np.sin(t) - root) / (np.sin(t) + root)
    far = sm.far_surfaces(eps, k, 1e7 * 2.0 * np.pi / k, t, omega=om)
    np.testing.assert_allclose(
        far["IzV"], c1k * np.cos(t) ** 2 * (r_tm - c2), atol=1e-4
    )
    np.testing.assert_allclose(far["IphiH"], -c1k * (r_te + c2), atol=1e-4)


def test_free_space_and_the_pec_limit():
    k = 2.0 * np.pi
    t = np.radians(np.array([0.0, 1.0, 45.0, 90.0]))
    zero = sm.far_surfaces(1.0 + 0.0j, k, 30.0, t)
    for q in sm._SURF_KEYS:
        assert np.all(zero[q] == 0.0)
    pec = sm.far_surfaces(1e16 + 0.0j, k, 30.0, t)
    for q in sm._SURF_KEYS:
        assert np.all(np.isfinite(pec[q]))
        assert np.max(np.abs(pec[q])) < 1e-5 * _scale(k * C)


@pytest.mark.skipif(sm._acc is None, reason="C++ accelerator not built")
@pytest.mark.parametrize(
    "eps_t", [13.0 - 0.3j, 3.0 - 0.01j, 81.0 - 12840j, 16.0 + 0.0j]
)
def test_the_cpp_kernels_continue_like_the_numpy_body(eps_t, monkeypatch):
    """A small grid, so most pairs of a 60-wavelength sample are past its
    edge: `proj_one`'s continuation (the table kernel and the owned-pair
    kernel) against `SommerfeldGrid.eval`'s."""
    k2 = 2.0 * np.pi
    grid = sm.SommerfeldGrid(eps_t, k2, 1.6)
    rng = np.random.default_rng(1258)

    def pts(n):
        return np.c_[rng.uniform(-30.0, 30.0, (n, 2)), rng.uniform(0.01, 2.0, n)]

    def tang(n):
        v = rng.standard_normal((n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    obs, t_obs, src, t_src = pts(41), tang(41), pts(53), tang(53)
    owner = rng.integers(0, 41, 53)
    fast = sm.remainder_field_proj(obs, t_obs, src, t_src, 0.0, k2, grid)
    fast_own = sm.remainder_field_proj_owned(
        obs, t_obs, src, t_src, owner, 0.0, k2, grid
    )
    monkeypatch.setattr(sm, "_acc", None)
    slow = sm.remainder_field_proj(obs, t_obs, src, t_src, 0.0, k2, grid)
    slow_own = sm.remainder_field_proj_owned(
        obs, t_obs, src, t_src, owner, 0.0, k2, grid
    )
    scale = np.abs(slow).max()
    assert np.abs(fast - slow).max() / scale < 1e-11
    assert np.abs(fast_own - slow_own).max() / scale < 1e-11
    r1 = np.hypot(
        np.hypot(obs[:, None, 0] - src[None, :, 0], obs[:, None, 1] - src[None, :, 1]),
        obs[:, None, 2] + src[None, :, 2],
    )
    assert (r1 > grid.r1_max).mean() > 0.9


# -- against the licensed engine -----------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "somm_far_1258"
DECKS = ("grazing-near", "grazing-far")


def _rows(text):
    lines = text.splitlines()
    i = next(i for i, ln in enumerate(lines) if "- - - NEAR " in ln)
    out = []
    for ln in lines[i + 5 :]:
        if not ln.strip():
            break
        f = [float(t) for t in ln.split()]
        e = np.array(
            [f[3 + 2 * c] * np.exp(1j * np.radians(f[4 + 2 * c])) for c in range(3)]
        )
        out.append((np.array(f[:3]), e))
    return out


@functools.lru_cache(maxsize=None)
def _pairs(name):
    deck = (FIXTURES / f"{name}.nec").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = _rows(render(deck, basis="razor-2p"))
    theirs = _rows((FIXTURES / f"{name}.out").read_text(encoding="latin-1"))
    out = []
    for (_, eo), (pt, et) in zip(ours, theirs, strict=True):
        r = float(np.linalg.norm(pt))
        elevation = float(np.degrees(np.arcsin(np.clip(pt[2] / r, -1.0, 1.0))))
        out.append((round(r), elevation, eo, et))
    return out


def _kept(rows):
    # The licensed engine's own near-zenith defect (tests/fixtures/
    # eznec_ne_spherical_1257/README.md): 1 to 8 degrees from the zenith its
    # near field departs from its far field -- 17 % at 20 m, 64 % at 50 m,
    # 1100x at 1000 m. The zenith itself is sound and stays in.
    return [row for row in rows if not 81.5 < row[1] < 90.0 - 1e-9]


@pytest.mark.parametrize("name", DECKS)
def test_the_near_field_past_the_table_is_the_licensed_engines(name):
    """razor-2p (NEC-5's formulation twin) against x13, per point. Within 15
    degrees of the horizon -- where the field is the ground wave and the
    frozen edge was wrong by up to 58x -- the vector difference is held to
    1 % of the licensed |E| at that point (measured worst 5.5e-3, most of
    it a steady ~0.3 degree phase offset between the engines at 1000 m that
    predates #1258). Higher up, where the pattern has nulls, it is held to
    1 % of the table's scale at that range, as #1257's envelope is."""
    rows = _kept(_pairs(name))
    scale = {}
    for r, _, _, et in rows:
        scale[r] = max(scale.get(r, 0.0), float(np.linalg.norm(et)))
    low = [row for row in rows if row[1] <= 15.0]
    assert len(low) >= 32
    for r, el, eo, et in low:
        rel = np.linalg.norm(eo - et) / np.linalg.norm(et)
        assert rel <= 1e-2, (r, el, rel)
    for r, el, eo, et in rows:
        assert np.linalg.norm(eo - et) / scale[r] <= 1e-2, (r, el)
