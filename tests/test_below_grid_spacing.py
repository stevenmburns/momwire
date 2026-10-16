"""The below/below grid's two spacings: the point lattice (today's, per-point
calibrated) and the coarser fill lattice the matrix fills read.

A closed-form stand-in kernel keeps these cheap: they pin the lattice each
spacing builds, that the two never share a cache entry (in process, on the
anchor ladder or in the sweep count), that a coarse radial step never puts a
row at or under R1 = 0, and that every caller of `get_grid_below` names its
spacing. What the fill spacing does to Z is measured in the PR that chose
it, not here.
"""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from momwire import _sommerfeld as sm
from momwire import _sommerfeld_below as sb
from momwire._sommerfeld import _C_LIGHT, _MU0, _SURF_KEYS

EPS0 = 8.8541878128e-12
POINT = sb._SOMM_BELOW_SPACING_POINT
FILL = sb._SOMM_BELOW_SPACING_FILL


def _medium(f_hz, eps_r=13.0, sigma=0.005):
    omega = 2.0 * math.pi * f_hz
    return complex(eps_r, -sigma / (omega * EPS0)), omega / _C_LIGHT, omega


@pytest.fixture
def standin(monkeypatch):
    """No grids or history, and a cheap closed-form kernel."""
    monkeypatch.setattr(sm, "_GRID_CACHE", {})
    monkeypatch.setattr(sb, "_GRID_CACHE", sm._GRID_CACHE)
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "0")
    monkeypatch.delenv(sb.ENV_ANCHORS, raising=False)
    sb.reset_sweep_state()

    def fake(eps_t, k2, R1, theta, rtol=1e-9, omega=None, mu=_MU0, **_kw):
        R1b, thb = np.broadcast_arrays(np.asarray(R1, float), np.asarray(theta, float))
        assert np.all(R1b >= 0.0)
        g = (1.0 + R1b) * np.exp(1j * thb)
        return {key: omega * mu * (i + 1) * g for i, key in enumerate(_SURF_KEYS)}

    monkeypatch.setattr(sb, "iv_surfaces_direct_below", fake)
    yield
    sb.reset_sweep_state()


def _grid(spacing, r1_wl=2.3, f_hz=6.0e6):
    eps_t, k2, omega = _medium(f_hz)
    return sb.SommerfeldGridBelow(
        eps_t, k2, r1_wl * sb._lambda_m(eps_t, k2), omega=omega, spacing=spacing
    )


def _steps(grid):
    """(dr / lam_m, dtheta in degrees) per region."""
    return [(reg["dr"] / grid.lam_m, math.degrees(reg["dth"])) for reg in grid._regions]


def test_the_point_spacing_is_the_default_and_todays_lattice(standin):
    grid = _grid(None)
    assert grid.spacing == POINT == (1.0, 1.0, 1.0)
    steps = _steps(grid)
    inner, near = sb._ZONE_INNER, sb._ZONE_NEAR
    graze = sb.region_index(near, sb._N_BANDS - 2)
    steep = sb.region_index(near, sb._N_BANDS - 1)
    assert steps[sb.region_index(inner, 0)][0] == pytest.approx(
        sb._SOMM_BELOW_DR_NEAR_LAMBDA_M, rel=1e-12
    )
    assert steps[graze] == pytest.approx(
        (sb._SOMM_BELOW_DR_LAMBDA_M, sb._SOMM_BELOW_DTH_GRAZE_DEG), rel=1e-12
    )
    assert steps[steep][1] == pytest.approx(sb._SOMM_BELOW_DTH_STEEP_DEG, rel=1e-12)
    # The literal constants, bit for bit: the point lattice is not re-derived.
    assert grid._regions[graze]["dth"] == np.radians(sb._SOMM_BELOW_DTH_GRAZE_DEG)


def test_the_fill_spacing_coarsens_only_the_axes_it_names(standin):
    point, fill = _grid(POINT), _grid(FILL)
    s_dr, s_th, s_band = FILL
    for idx, ((dr_p, th_p), (dr_f, th_f)) in enumerate(
        zip(_steps(point), _steps(fill), strict=True)
    ):
        assert dr_f == pytest.approx(s_dr * dr_p, rel=1e-12)
        band = idx % sb._N_BANDS
        if band in (sb._N_BANDS - 2, sb._N_BANDS - 1):
            # Graze and steep: the nearest whole cell count to s_th times the
            # step, so the band still ends on its edge.
            assert th_f / th_p == pytest.approx(s_th, rel=0.25)
            reg = fill._regions[idx]
            top = math.degrees(reg["th0"] + reg["dth"] * (reg["n_th"] - 1))
            edge = 90.0 if band == sb._N_BANDS - 1 else sb._SOMM_TH_SPLIT_DEG
            assert top == pytest.approx(edge, abs=1e-9)
        elif band == sb._N_BANDS - 3:
            assert th_f == pytest.approx(s_band * th_p, rel=1e-12)
        else:
            assert th_f == th_p  # the floor bands keep their steps
    assert sum(r["n_r"] * r["n_th"] for r in fill._regions) < sum(
        r["n_r"] * r["n_th"] for r in point._regions
    )


@pytest.mark.parametrize("s_dr", [2.0, 3.0, 4.0])
def test_a_coarse_radial_step_never_puts_a_row_at_or_under_zero(standin, s_dr):
    grid = _grid((s_dr, 8.0, 1.0))
    for reg in grid._regions:
        if reg["r0"] > 0.0 or reg["r_nodes"][0] > 0.0:
            assert reg["r_nodes"][0] > 0.0
    # A zone with room keeps its centred stencil's lower rows.
    point = _grid(POINT)
    near = sb.region_index(sb._ZONE_NEAR, 0)
    assert (
        point._regions[near]["r_nodes"][0] < point.r_break - point._regions[near]["dr"]
    )


def test_the_two_spacings_never_share_a_cache_entry(standin):
    eps_t, k2, omega = _medium(6.0e6)
    r1 = 1.2 * sb._lambda_m(eps_t, k2)
    point = sb.get_grid_below(eps_t, k2, r1, omega)
    fill = sb.get_grid_below(eps_t, k2, r1, omega, spacing=FILL)
    assert point is not fill
    assert (point.spacing, fill.spacing) == (POINT, FILL)
    assert sb.get_grid_below(eps_t, k2, r1, omega, spacing=FILL) is fill
    assert sb.get_grid_below(eps_t, k2, r1, omega, spacing=POINT) is point


def test_anchors_and_the_sweep_count_carry_the_spacing(standin, monkeypatch):
    eps_t, _k2, _omega = _medium(6.0e6)
    p, f = sb._anchor_plan(eps_t), sb._anchor_plan(eps_t, FILL)
    assert p.rungs == f.rungs
    assert all(p.key(j) != f.key(j) for j in range(len(p.rungs)))
    assert p.family != f.family
    # A blend is built on its plan's lattice, from anchors on that lattice.
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    eps_t, k2, omega = _medium(6.0e6)
    blended = sb.get_grid_below(
        eps_t, k2, 1.2 * sb._lambda_m(eps_t, k2), omega, spacing=FILL
    )
    assert isinstance(blended, sb.SommerfeldGridBelowBlended)
    assert blended.spacing == FILL
    assert all(a.spacing == FILL for a in sb._ANCHOR_CACHE.values())


def test_every_caller_of_get_grid_below_names_its_spacing():
    """A new reader of the below/below table must choose: the fill spacing
    if it integrates the table, the point spacing if it reads one point."""
    src = Path(sb.__file__).parent
    calls = []
    for path in sorted(src.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", getattr(node.func, "id", None))
                == "get_grid_below"
            ):
                calls.append((path.name, node.lineno, {k.arg for k in node.keywords}))
    assert calls
    assert [c for c in calls if "spacing" not in c[2]] == []
