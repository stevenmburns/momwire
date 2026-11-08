"""The transmitted (below->above) grid blended from frequency anchors
(momwire#1847, the opt-in feature `transmitted_anchors`).

What is held:

- off, nothing changes: the solvers default to the exact grid, and
  `get_grid_below_above(anchors=False)` is the exact grid's own object;
- on, the blend is the exact fill ON THE SAME METRES LATTICE to the measured
  frequency-interpolation error (1e-7 of scale at 10 % / 8 points), and the
  exact grid's own answer to its spatial interpolation error;
- a sweep shares its anchors: three nearby frequencies fill 9, not 24;
- an anchor a request could not tabulate declines, and the exact grid serves;
- anchors round-trip through the disk store bit for bit, so a fresh process
  per point (AutoEZ's pattern) refills nothing;
- a solve's Z with the flag on agrees with the flag off.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from momwire import _features, _somm_disk_cache
from momwire import _sommerfeld as som
from momwire import _sommerfeld_transmitted as trans
from momwire import _transmitted_anchors as anchors
from momwire._constants import C_LIGHT
from momwire._ground_refl import eps_tilde
from momwire.bspline import BSplineSolver

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

EPS0 = 8.8541878128e-12
SOIL = (13.0, 0.005)
F0 = 7.1e6
# A small lattice keeps the eight anchor fills to a few seconds.
R_MIN, R_MAX, DEPTH = 0.2, 2.0, 0.15


def _req(f_hz, r_max=R_MAX, zp=(DEPTH, DEPTH), r_min=R_MIN):
    om = 2.0 * math.pi * f_hz
    return dict(
        eps_t=eps_tilde(SOIL, om, EPS0),
        k2=om / C_LIGHT,
        r_max=r_max,
        zp_min=zp[0],
        zp_max=zp[1],
        omega=om,
        r_min=r_min,
    )


@pytest.fixture(scope="module")
def filled(tmp_path_factory):
    """The anchors for F0, filled once with the disk store on."""
    d = tmp_path_factory.mktemp("somm")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MOMWIRE_SOMM_CACHE", "1")
        mp.setenv("MOMWIRE_SOMM_CACHE_DIR", str(d))
        anchors.reset()
        r = _req(F0)
        g = anchors.get_grid(mu=som._MU0, **r)
        stats = dict(anchors.STATS)
        yield g, r, stats, d


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_TRANSMITTED_ANCHORS", raising=False)
    assert not _features.enabled("transmitted_anchors")
    som._GRID_CACHE.clear()
    r = _req(F0)
    g = trans.get_grid_below_above(
        r["eps_t"],
        r["k2"],
        r["r_max"],
        r["zp_min"],
        r["zp_max"],
        r["omega"],
        r_min=r["r_min"],
    )
    key = next(k for k in som._GRID_CACHE if k[0] == "below-above")
    assert som._GRID_CACHE[key] is g
    som._GRID_CACHE.clear()


def test_the_blend_is_the_exact_fill_on_its_lattice(filled):
    g, r, stats, _ = filled
    assert stats["anchor_fills"] == anchors.ANCHOR_POINTS and stats["blends"] == 1
    assert g.eps_t == r["eps_t"] and g.omega == r["omega"]
    lat = anchors._Lattice(g.r_min, g.r_max, g.zp_nodes)
    exact = anchors._fill_anchor(lat, r["eps_t"], r["k2"], r["omega"], som._MU0, 1e-9)
    scale = np.abs(exact._vals).reshape(exact._vals.shape[:2] + (-1,)).max(-1)
    err = np.abs(g._vals - exact._vals).reshape(scale.shape + (-1,)).max(-1)
    assert float(np.max(err / scale)) < 1e-7


def test_the_blend_answers_like_the_exact_grid(filled):
    g, r, _, _ = filled
    exact = trans.TransmittedGrid(
        r["eps_t"],
        r["k2"],
        r["r_max"],
        r["zp_min"],
        r["zp_max"],
        omega=r["omega"],
        r_min=r["r_min"],
    )
    rng = np.random.default_rng(1847)
    R = np.exp(rng.uniform(math.log(R_MIN), math.log(R_MAX), 200))
    th = rng.uniform(max(g.th_min, exact.th_min), 0.5 * np.pi, 200)
    a = g.eval(R, th, -DEPTH)
    b = exact.eval(R, th, -DEPTH)
    for k in a:
        scale = float(np.max(np.abs(b[k])))
        assert float(np.max(np.abs(a[k] - b[k]))) < 2e-3 * scale, k


def test_a_sweep_shares_its_anchors(filled):
    base = anchors.STATS["anchor_fills"]
    for f in (F0 * 1.02, F0 * 1.05):
        assert anchors.get_grid(mu=som._MU0, **_req(f)) is not None
    # 7.1 -> 7.46 MHz crosses at most one anchor step.
    assert anchors.STATS["anchor_fills"] - base <= 1


def test_anchors_round_trip_through_the_disk(filled, monkeypatch):
    g, r, _, d = filled
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "1")
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(d))
    anchors._ANCHORS.clear()
    anchors._BLENDS.clear()
    fills = anchors.STATS["anchor_fills"]
    again = anchors.get_grid(mu=som._MU0, **r)
    assert anchors.STATS["anchor_fills"] == fills
    assert again is not g
    assert _somm_disk_cache.grid_digest(again) == _somm_disk_cache.grid_digest(g)


def test_an_anchor_past_the_cap_declines_and_the_exact_grid_serves():
    # 2 lambda_p is 84 m at 7.1 MHz but 58 m at the stencil's top anchor.
    r = _req(F0, r_max=70.0)
    assert anchors.get_grid(mu=som._MU0, **r) is None
    som._GRID_CACHE.clear()
    g = trans.get_grid_below_above(
        r["eps_t"],
        r["k2"],
        r["r_max"],
        r["zp_min"],
        r["zp_max"],
        r["omega"],
        r_min=r["r_min"],
        anchors=True,
    )
    assert any(v is g for k, v in som._GRID_CACHE.items() if k[0] == "below-above")
    som._GRID_CACHE.clear()


def test_a_lattice_whose_grazing_floor_is_above_the_exact_one_declines(monkeypatch):
    """The plan's grazing check reads the exact grid's floor; a lattice with a
    higher one must not serve, or an observer the plan passed would be
    refused from inside the fill."""
    r = _req(F0)
    exact_floor = trans.grid_extent(r["k2"], r["r_max"], r["zp_min"], r["r_min"])[2]
    monkeypatch.setattr(
        anchors._Lattice,
        "__init__",
        _with_floor(anchors._Lattice.__init__, exact_floor + 0.01),
    )
    anchors._BLENDS.clear()
    assert anchors.get_grid(mu=som._MU0, **r) is None


def _with_floor(init, th):
    def patched(self, *a, **k):
        init(self, *a, **k)
        self.th_min = th

    return patched


def test_a_multi_depth_ladder_is_shared_by_neighbouring_points():
    zp = (0.1, 1.0)
    lats = []
    for f in (F0, F0 * 1.03):
        r = _req(f, zp=zp)
        js, _ = anchors._stencil(f)
        eps_r, q = anchors._soil(r["eps_t"], r["omega"])
        lats.append(
            anchors._lattice(f, js, r["r_max"], *zp, r["r_min"], eps_r, q).key()
        )
    assert lats[0] == lats[1]
    assert lats[0][-1] in anchors._LADDER_BUCKETS


def test_the_soil_curve_is_one_family_across_a_sweep():
    keys = {
        anchors._soil(**{k: _req(f)[k] for k in ("eps_t", "omega")})
        for f in (5.3e6, 7.1e6, 8.9e6)
    }
    assert len(keys) == 1


def _served(transmitted_anchors, f_hz=F0):
    """An elevated 10 m monopole over a detached buried radial: the
    above/below pair class, so the transmitted grid is on the path."""
    wl = C_LIGHT / f_hz
    return BSplineSolver(
        wires=[
            np.array([(0.0, 0.0, 11.0), (0.0, 0.0, 1.0)]),
            np.array([(0.0, 0.0, -0.15), (5.0, 0.0, -0.15)]),
        ],
        n_per_edge_per_wire=[[15], [10]],
        feeds=[(0, (round(0.4333 * 15) - 0.5) / 15 * 10.0, 1 + 0j)],
        wavelength=wl,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL,
        ground_model="sommerfeld",
        transmitted_anchors=transmitted_anchors,
    )


# Slow lane at the 0.88.0 pre-tag check: 11.6 s median on the Linux test job.
@pytest.mark.slow
def test_z_with_anchors_agrees_with_the_exact_grid():
    anchors.reset()
    som._GRID_CACHE.clear()
    z_off = 1.0 / complex(_served(False).compute_port_solution().y.ravel()[0])
    som._GRID_CACHE.clear()
    s = _served(True)
    assert s._transmitted_anchors
    z_on = 1.0 / complex(s.compute_port_solution().y.ravel()[0])
    assert anchors.STATS["blends"] == 1
    assert abs(z_on - z_off) < 1e-8 * abs(z_off)
    som._GRID_CACHE.clear()
