"""Lazy Sommerfeld grids: only the nodes an answer reads are filled (momwire#1512).

A master (`_sommerfeld._norm_master`) is its lattice with a node mask, and the
razor remainder windows, the near-field remainder and the below projection
take a CENSUS of their reads (the accelerator's own pair geometry and stencil
address) and fill those nodes before reading; every other reader still gets
whole grids. What this pins:

  * a node's value is the eager fill's bit for bit whether it is filled
    alone, in a partial lane group or in a large batch, above and below;
  * a solve on lazy grids is bit-identical to the same solve on eager ones
    (MOMWIRE_SOMM_LAZY=0), buried and above ground, and the near field (E, H)
    too -- and the check mode (MOMWIRE_SOMM_LAZY_CHECK: every lazy read must
    be finite, and a node not filled reads NaN) ran over every reader;
  * the store: what one run filled, the next reads; two runs' nodes are
    merged on write; an eager master file (no masks) still reads.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from momwire import RazorSolver, _field_point, _somm_disk_cache, _sommerfeld
from momwire import _sommerfeld_below as sb
from momwire._accel import acc
from test_crossing_serve_524 import SOIL_A, WL7, A_WIRE, hub_deck

pytestmark = pytest.mark.skipif(
    acc is None
    or not hasattr(acc, "remainder_field_proj_nodes")
    or not hasattr(acc, "remainder_field_proj_nodes_below"),
    reason="the lazy grids' census needs the accelerator",
)


@pytest.fixture
def cold():
    """Empty in-process grid caches for the test, the old ones restored."""
    saved = (dict(_sommerfeld._GRID_CACHE), dict(_sommerfeld._NORM_CACHE))
    _sommerfeld._GRID_CACHE.clear()
    _sommerfeld._NORM_CACHE.clear()
    yield
    _sommerfeld._GRID_CACHE.clear()
    _sommerfeld._NORM_CACHE.clear()
    _sommerfeld._GRID_CACHE.update(saved[0])
    _sommerfeld._NORM_CACHE.update(saved[1])


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _batches(nodes, rng, size):
    i = 0
    while i < len(nodes):
        b = size(rng)
        yield nodes[i : i + b]
        i += b


@pytest.mark.parametrize("size", ["alone", "lanes", "batch"])
def test_an_above_node_is_the_eager_fill_whatever_shares_its_call(size):
    pick = {
        "alone": lambda r: 1,
        "lanes": lambda r: 1 + r.randrange(7),
        "batch": lambda r: 1 + r.randrange(97),
    }[size]
    rng = random.Random(1512)
    kw = dict(omega=_sommerfeld._K2_REF * _sommerfeld._C_LIGHT, mu=_sommerfeld._MU0)
    eps = 13.0 - 9.7j
    full = _sommerfeld.SommerfeldGrid(
        eps, _sommerfeld._K2_REF, 0.512, lattice_eps=eps, **kw
    )
    g = _sommerfeld.SommerfeldGrid(
        eps, _sommerfeld._K2_REF, 0.512, lattice_eps=eps, lazy=True, **kw
    )
    nodes = [
        (r, i, j)
        for r, reg in enumerate(g._regions)
        for i in range(reg["n_r"])
        for j in range(reg["n_th"])
    ]
    rng.shuffle(nodes)
    for part in _batches(nodes, rng, pick):
        need = [np.zeros(reg["have"].shape, bool) for reg in g._regions]
        for r, i, j in part:
            need[r][i, j] = True
        g.fill_nodes(need)
    assert g.missing(_sommerfeld._all_nodes(g)) is None
    for a, b in zip(g._regions, full._regions, strict=True):
        assert np.array_equal(_bits(a["vals"]), _bits(b["vals"]))


def test_a_below_node_is_the_eager_fill_whatever_shares_its_call():
    rng = random.Random(1512)
    kp = 2.0 * np.pi / WL7
    eps = 13.0 - 12.84j
    lam_m = 2.0 * np.pi / abs(sb.k_medium(eps, kp))
    args = (eps, kp, 0.9 * lam_m)
    full = sb.SommerfeldGridBelow(*args)
    lazy = sb.SommerfeldGridBelow(*args, lazy=True)
    for idx in range(len(full._regions)):
        full._fill_region(idx)
        lazy._activate(idx)
    nodes = [
        (r, i, j)
        for r, reg in enumerate(lazy._regions)
        for i in range(reg["n_r"])
        for j in range(reg["n_th"])
    ]
    rng.shuffle(nodes)
    # Only a sample of the (expensive) nodes, in batches of 1 to 7 and the
    # floor band's shared columns among them; the rest whole at the end.
    for part in _batches(nodes[:160], rng, lambda r: 1 + r.randrange(7)):
        need = [np.zeros(reg["have"].shape, bool) for reg in lazy._regions]
        for r, i, j in part:
            need[r][i, j] = True
        lazy.fill_nodes(need)
    for idx in range(len(lazy._regions)):
        lazy._fill_region(idx)
    for a, b in zip(lazy._regions, full._regions, strict=True):
        assert a["filled"] and a["have"].all()
        assert np.array_equal(_bits(a["vals"]), _bits(b["vals"]))


def _dipole_over_ground():
    wire = np.array([(-5.0, 0.0, 6.0), (5.0, 0.0, 6.0)])
    return dict(
        wires=[wire],
        n_per_edge_per_wire=[[11]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


@pytest.mark.parametrize("deck", ["buried", "above"])
def test_a_solve_on_lazy_grids_is_the_eager_one_and_every_read_was_checked(
    monkeypatch, cold, deck
):
    build = hub_deck(n_radials=3) if deck == "buried" else _dipole_over_ground()
    monkeypatch.setenv("MOMWIRE_SOMM_LAZY", "0")
    z_eager = RazorSolver(**build, nec5_quadrature=True).compute_impedance()
    _sommerfeld._GRID_CACHE.clear()
    _sommerfeld._NORM_CACHE.clear()
    monkeypatch.setenv("MOMWIRE_SOMM_LAZY", "1")
    monkeypatch.setenv("MOMWIRE_SOMM_LAZY_CHECK", "1")
    monkeypatch.setattr(_sommerfeld, "LAZY_CHECKED", {})
    z_lazy = RazorSolver(**build, nec5_quadrature=True).compute_impedance()
    assert np.array_equal(
        _bits(np.atleast_1d(z_lazy[0])), _bits(np.atleast_1d(z_eager[0]))
    )
    checked = _sommerfeld.LAZY_CHECKED
    assert checked.get("the above remainder window", (0, 0))[1] > 0, checked
    if deck == "buried":
        assert checked.get("the below remainder projection", (0, 0))[1] > 0, checked
    # A lazy view, not a whole one, served the windows.
    lazy_views = [g for k, g in _sommerfeld._GRID_CACHE.items() if k[-1] == "lazy"]
    assert lazy_views
    assert any(not reg["have"].all() for g in lazy_views for reg in g._regions)


@pytest.mark.parametrize("magnetic", [False, True])
def test_the_near_field_on_a_lazy_grid_is_the_eager_one(monkeypatch, cold, magnetic):
    rng = np.random.default_rng(1512)
    pts = np.column_stack(
        [rng.uniform(-8, 8, 7), rng.uniform(-8, 8, 7), rng.uniform(1, 9, 7)]
    )
    mids = np.column_stack([np.linspace(-5, 5, 11), np.zeros(11), np.full(11, 6.0)])
    mom = rng.normal(size=(11, 3)) + 1j * rng.normal(size=(11, 3))
    k = 2.0 * np.pi / WL7
    f = (
        _field_point.reflected_h_field_at
        if magnetic
        else _field_point.reflected_field_at
    )
    args = (pts, mids, mom, 13.0 - 12.84j, 0.0, k, k * _sommerfeld._C_LIGHT)
    monkeypatch.setenv("MOMWIRE_SOMM_LAZY", "0")
    eager = f(*args)
    _sommerfeld._GRID_CACHE.clear()
    _sommerfeld._NORM_CACHE.clear()
    monkeypatch.setenv("MOMWIRE_SOMM_LAZY", "1")
    monkeypatch.setenv("MOMWIRE_SOMM_LAZY_CHECK", "1")
    monkeypatch.setattr(_sommerfeld, "LAZY_CHECKED", {})
    lazy = f(*args)
    assert np.array_equal(_bits(lazy), _bits(eager))
    assert _sommerfeld.LAZY_CHECKED, "the lazy near-field path did not run"


def test_the_store_keeps_and_merges_what_each_run_filled(monkeypatch, cold, tmp_path):
    """Run A fills some nodes of a master, run B (a fresh process: empty
    memory) others; the file holds both, and a run C reads every one of
    them from it without filling any."""
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "1")
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    eps, rb = 13.0 - 9.7j, 0.512
    nkey = ("above", eps, rb, eps)

    def master_with(sel):
        _sommerfeld._NORM_CACHE.clear()
        m = _sommerfeld._norm_master(eps, rb, eps)
        need = [np.zeros(reg["have"].shape, bool) for reg in m._regions]
        for r, i, j in sel:
            need[r][i, j] = True
        _sommerfeld._ensure_master(nkey, m, need)
        return m

    a = master_with([(0, 0, 0), (0, 1, 1)])
    b = master_with([(1, 2, 2), (2, 3, 3)])
    assert b._regions[0]["have"][0, 0]  # B started from A's file
    calls = []
    real = _sommerfeld.SommerfeldGrid.fill_nodes
    monkeypatch.setattr(
        _sommerfeld.SommerfeldGrid,
        "fill_nodes",
        lambda self, need, cancel_flag=0: (
            calls.append(1) or real(self, need, cancel_flag)
        ),
    )
    c = master_with([(0, 0, 0), (0, 1, 1), (1, 2, 2), (2, 3, 3)])
    assert not calls
    for g in (a, b):
        for ra, rc in zip(g._regions, c._regions, strict=True):
            m = ra["have"]
            assert np.array_equal(_bits(ra["vals"][:, m]), _bits(rc["vals"][:, m]))


def test_an_eager_master_file_still_reads(monkeypatch, cold, tmp_path):
    """A master stored whole (no masks, as the eager fill wrote them) serves
    every node of the lazy master, bit for bit."""
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "1")
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    eps, rb = 5.0 - 0.5j, 0.512
    kw = dict(omega=_sommerfeld._K2_REF * _sommerfeld._C_LIGHT, mu=_sommerfeld._MU0)
    whole = _sommerfeld.SommerfeldGrid(
        eps, _sommerfeld._K2_REF, rb, lattice_eps=eps, **kw
    )
    _somm_disk_cache.store(("above", eps, rb, eps), whole)
    m = _sommerfeld._norm_master(eps, rb, eps)
    assert m.missing(_sommerfeld._all_nodes(m)) is None
    for a, b in zip(m._regions, whole._regions, strict=True):
        assert np.array_equal(_bits(a["vals"]), _bits(b["vals"]))
