"""momwire#1131: the above-ground fill takes `rows=`, so the sector route
(momwire#1029) serves elevated and surface radial screens, not only buried
ones.

The deck is Severns' 2009 N6LF vertical in miniature (the shape of
`test_surface_radials_865.py`): a mast over N radials at height `h`, spelled
in plain momwire. The mesh is coarse on purpose -- every gate is the route or
the restricted fill against the dense one on the SAME deck.

What is gated:

* **the route against the dense solve** over free space, PEC, refl-coef and
  Sommerfeld, at momwire#1029's bars (`|dZ|/|Z| <= 1e-9`,
  `max|dc|/max|c| <= 1e-8`);
* **the fill to the bit** -- the `rows=` rows against the DENSE CHUNKED
  fill's rows as uint64, with `swept_mem_mb=1` so the observer chunks cut
  through bases; the square target's other rows exactly zero; the compact
  target equal to the square one; loading included;
* **rows=None is the old code** -- the restriction object is never built;
* **negative controls** -- another sector's rows are NOT these rows bit for
  bit, and one dropped sweep window moves Z_in;
* **the refusals that are about symmetry still refuse** on an above-ground
  deck, and a split basis is refused by name.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from momwire import _rotational_symmetry as RS
from momwire import bspline
from momwire._rotational_symmetry import RotationalSymmetryRefused
from momwire.bspline import BSplineSolver

FT = 0.3048
WL = 299792458.0 / 7.2e6
MAST = 33.5 * FT
RADIAL = 33.0 * FT
A = 0.51e-3
SOIL = (30.0, 0.020)
GROUNDS = {
    "free": dict(ground_z=None, ground_eps=None),
    "pec": dict(ground_z=0.0, ground_eps=None),
    "refl-coef": dict(ground_z=0.0, ground_eps=SOIL, ground_model="refl-coef"),
    "sommerfeld": dict(ground_z=0.0, ground_eps=SOIL, ground_model="sommerfeld"),
}
ELEVATED = 0.5
SURFACE = 2.0 * A  # the validity floor, where the near-image fixup engages


def screen(n=4, h=ELEVATED, radial=RADIAL, azimuths=None):
    lengths = [radial] * n if np.isscalar(radial) else list(radial)
    if azimuths is None:
        azimuths = [2.0 * math.pi * i / n for i in range(n)]
    wires = [
        np.array([(r * math.cos(t), r * math.sin(t), h), (0.0, 0.0, h)])
        for r, t in zip(lengths, azimuths)
    ]
    wires.append(np.array([(0.0, 0.0, z) for z in (MAST + h, 0.5 + h, h)]))
    return wires


def solver(n=4, *, ground="pec", h=ELEVATED, rot=True, wires=None, **kw):
    wires = screen(n, h) if wires is None else wires
    n_rad = len(wires) - 1
    kwargs = dict(
        wires=wires,
        n_per_edge_per_wire=[[5]] * n_rad + [[9, 2]],
        junctions=[[(i, "end") for i in range(n_rad)] + [(n_rad, "end")]],
        feeds=[(n_rad, MAST - 0.05, 1 + 0j)],
        wavelength=WL,
        wire_radius=A,
        degree=2,
        rotational_symmetry=rot,
        **GROUNDS[ground],
    )
    kwargs.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(**kwargs)


def solve(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z, c = s.compute_impedance()
    return complex(np.atleast_1d(z)[0]), c


def parts(s):
    geom = s._build_geometry()
    supp_seg, polys, _kcl, _knots, wbg = s._build_basis_polynomials(geom)
    return geom, supp_seg, polys, wbg


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def fills(s):
    """(dense chunked Z, square rows= Z, compact Z, R, rows) on one solver."""
    geom, supp_seg, polys, _wbg = parts(s)
    rows = RS.observer_rows(s, geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s._dense_tensor_fits_budget = lambda n_segs: False
        try:
            Z = np.asarray(s._compute_Z_operator(geom, supp_seg, polys))
        finally:
            del s._dense_tensor_fits_budget
        Z_sq = s._compute_Z_operator(geom, supp_seg, polys, rows=rows)
        Z_c, R = s._compute_Z_operator(geom, supp_seg, polys, rows=rows, compact=True)
    return Z, Z_sq, Z_c, R, rows


@pytest.fixture(scope="module")
def sommerfeld_dense():
    """The dense Sommerfeld answer, solved once per module. Most of its cost
    is the Sommerfeld grid, which is cached per process -- so this fixture
    also warms the grid every other Sommerfeld test here reads (same soil,
    same k, same extent), and the module is grouped onto one worker in
    `conftest._FIXTURE_GROUP_FILES` so that happens once."""
    return solve(solver(4, ground="sommerfeld", h=SURFACE, rot=False))


# ----------------------------------------------------------------------
# The route against the dense solve
# ----------------------------------------------------------------------


def _route_against(z_d, c_d, ground, h):
    route = solver(4, ground=ground, h=h)
    z_r, c_r = solve(route)
    assert route._rotational_map.n_sectors == 4
    assert abs(z_r - z_d) / abs(z_d) <= 1e-9, (z_r, z_d)
    assert np.max(np.abs(c_r - c_d)) / np.max(np.abs(c_d)) <= 1e-8


@pytest.mark.parametrize(
    "ground,h",
    [
        ("free", ELEVATED),
        ("pec", ELEVATED),
        ("pec", SURFACE),
        ("refl-coef", ELEVATED),
        ("refl-coef", SURFACE),
    ],
)
def test_the_route_is_the_dense_solve_above_ground(ground, h):
    z_d, c_d = solve(solver(4, ground=ground, h=h, rot=False))
    _route_against(z_d, c_d, ground, h)


def test_the_route_is_the_dense_solve_over_sommerfeld(sommerfeld_dense):
    _route_against(*sommerfeld_dense, "sommerfeld", SURFACE)


def test_the_route_actually_fills_restricted_rows(monkeypatch):
    """Not a dense solve in disguise: the route builds the restriction once,
    with sector 0's wires and the mast, and the fill it gets back is the
    compact one."""
    built = []
    orig = bspline._ObserverRows.__init__

    def spy(self, *a, **kw):
        orig(self, *a, **kw)
        built.append((self.basis_rows.size, kw["compact"]))

    monkeypatch.setattr(bspline._ObserverRows, "__init__", spy)
    s = solver(6, ground="refl-coef")
    solve(s)
    geom, supp_seg, _p, wbg = parts(s)
    sectors, axial = s._rotational_dof_groups(wbg, supp_seg.shape[0])
    assert built == [(sectors[0].size + axial.size, True)]


def test_the_swept_route_is_the_dense_sweep_above_ground():
    ks = np.array([0.98, 1.02]) * 2 * math.pi / WL
    z_d = solver(4, ground="pec", rot=False).compute_impedance_swept(ks)
    z_r = solver(4, ground="pec").compute_impedance_swept(ks)
    assert np.max(np.abs(z_r - z_d) / np.abs(z_d)) <= 1e-9


# ----------------------------------------------------------------------
# The fill, to the bit
# ----------------------------------------------------------------------


def _rows_fill_is_dense_chunked(ground, h, n=24, **kw):
    s = solver(n, ground=ground, h=h, **kw)
    if "swept_mem_mb" in kw:
        n_segs = s._build_geometry()["n_segs_total"]
        chunk = (1 << 20) // (9 * n_segs * 16)  # `_compute_Z_dense_chunked`'s
        assert 2 * chunk < n_segs  # several chunks, not one
    Z, Z_sq, Z_c, R, _rows = fills(s)
    off = np.ones(Z.shape[0], dtype=bool)
    off[R] = False
    assert np.array_equal(bits(Z_sq[R]), bits(Z[R]))
    assert not np.any(Z_sq[off])
    assert Z_c.shape == (R.size, Z.shape[1])
    assert np.array_equal(bits(Z_c), bits(Z[R]))
    return s


@pytest.mark.parametrize(
    "ground,h,kw",
    [
        ("free", ELEVATED, {}),
        ("pec", SURFACE, {}),
        ("refl-coef", SURFACE, {}),
        ("refl-coef", ELEVATED, {"wire_conductivity": 3.5e7}),
    ],
)
def test_the_rows_fill_is_the_dense_chunked_fill_row_for_row(ground, h, kw):
    """`swept_mem_mb=1` cuts the observer axis into many chunks, so bases
    straddle chunk boundaries and the restricted fill has to keep the dense
    chunking to stay bitwise."""
    _rows_fill_is_dense_chunked(ground, h, swept_mem_mb=1, **kw)


def test_the_rows_fill_is_the_dense_chunked_fill_over_sommerfeld(sommerfeld_dense):
    """The same, with the remainder block, at the SURFACE height where
    momwire#1189 raises some segment pairs' order -- so the restricted pair
    correction is exercised too (asserted, not assumed). Four radials and
    one chunk: the chunking is the sweeps' and is held above; what is new
    here is the remainder, which does not chunk by `swept_mem_mb`. The
    fixture warms the grid."""
    s = _rows_fill_is_dense_chunked("sommerfeld", SURFACE, n=4)
    orders = s._last_remainder_orders
    assert len(orders) > 1, orders


def test_the_surface_screen_exercises_the_near_image_fixup():
    """The surface convention is where `_near_image_edge_blocks` engages, and
    the bitwise gate above covers it only if it does."""
    s = solver(4, ground="pec", h=SURFACE)
    assert s._near_image_edge_blocks(s._build_geometry())


def test_the_sommerfeld_remainder_rows_are_the_square_blocks_rows(sommerfeld_dense):
    s = solver(4, ground="sommerfeld", h=SURFACE)
    geom, supp_seg, polys, _wbg = parts(s)
    rows = RS.observer_rows(s, geom)
    restrict = bspline._ObserverRows(
        rows, supp_seg, polys, geom["n_segs_total"], compact=True
    )
    eps_t = bspline._potential_ground.potential_ground_for(
        s, geom, s.k, s.omega
    ).eps_tilde
    Q = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)
    Q_R = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t, restrict=restrict)
    assert np.array_equal(bits(Q_R), bits(Q[restrict.basis_rows]))


def test_rows_none_never_builds_the_restriction(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("rows=None reached the restricted fill")

    monkeypatch.setattr(bspline, "_ObserverRows", boom)
    solve(solver(4, ground="refl-coef", rot=False))


def test_compact_without_rows_is_refused():
    s = solver(4, rot=False)
    geom, supp_seg, polys, _wbg = parts(s)
    with pytest.raises(ValueError, match="pass rows= as well"):
        s._compute_Z_operator(geom, supp_seg, polys, compact=True)


def test_the_sliced_fallback_is_the_dense_default_fill(monkeypatch):
    """Without the windowed assemblers the restricted fill is the dense fill
    sliced -- the dense DEFAULT one, tensor route included."""
    s = solver(4, ground="refl-coef")
    geom, supp_seg, polys, _wbg = parts(s)
    rows = RS.observer_rows(s, geom)
    Z = np.asarray(s._compute_Z_operator(geom, supp_seg, polys))
    monkeypatch.setattr(bspline, "_HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL", False)
    Z_c, R = s._compute_Z_operator(geom, supp_seg, polys, rows=rows, compact=True)
    Z_sq = s._compute_Z_operator(geom, supp_seg, polys, rows=rows)
    assert np.array_equal(bits(Z_c), bits(Z[R]))
    assert np.array_equal(bits(Z_sq[R]), bits(Z[R]))


def test_a_split_wire_is_refused_by_name():
    s = solver(4, rot=False)
    geom, supp_seg, polys, _wbg = parts(s)
    half = np.arange(0, geom["per_wire"][0]["n_total"] // 2)
    with pytest.raises(ValueError, match="Restrict by whole wires"):
        s._compute_Z_operator(geom, supp_seg, polys, rows=half)


# ----------------------------------------------------------------------
# Negative controls
# ----------------------------------------------------------------------


def test_negative_another_sectors_rows_are_not_these_rows():
    """The bitwise gate can fail: sector 1's rows are a rotation of sector
    0's, equal in every row SUM the route reads, and still not the same
    entries."""
    s = solver(4, ground="refl-coef")
    geom, supp_seg, polys, _wbg = parts(s)
    smap = s._rotational_map
    off, pw = geom["seg_offsets"], geom["per_wire"]
    s1 = np.sort(
        np.concatenate(
            [
                np.arange(off[w], off[w] + pw[w]["n_total"])
                for w in (*smap.sectors[1], *smap.axial)
            ]
        )
    )
    rows = RS.observer_rows(s, geom)
    Z0, _ = s._compute_Z_operator(geom, supp_seg, polys, rows=rows, compact=True)
    Z1, _ = s._compute_Z_operator(geom, supp_seg, polys, rows=s1, compact=True)
    assert Z0.shape == Z1.shape
    assert not np.array_equal(bits(Z0), bits(Z1))


def test_negative_a_dropped_window_moves_the_answer(monkeypatch):
    z_d, _ = solve(solver(4, ground="pec", rot=False))
    orig = bspline._ObserverRows.windows
    monkeypatch.setattr(
        bspline._ObserverRows, "windows", lambda self, i0, i1: orig(self, i0, i1)[1:]
    )
    z_bad, _ = solve(solver(4, ground="pec"))
    assert abs(z_bad - z_d) / abs(z_d) > 1e-2


# ----------------------------------------------------------------------
# What still refuses, on an above-ground deck
# ----------------------------------------------------------------------


def _refusal(**kw):
    with pytest.raises(RotationalSymmetryRefused) as exc:
        solver(**kw)
    return str(exc.value)


def test_a_longer_radial_still_refuses_above_ground():
    lengths = [RADIAL, RADIAL * 1.01, RADIAL, RADIAL]
    msg = _refusal(wires=screen(4, radial=lengths), ground="refl-coef")
    assert "sector 1's wire is +1.00 % longer" in msg


def test_a_radial_at_the_wrong_azimuth_still_refuses_above_ground():
    az = [0.0, 0.5 * math.pi, math.pi, math.radians(272.5)]
    msg = _refusal(wires=screen(4, azimuths=az), ground="pec")
    assert "sector 3 sits at 272.500 deg" in msg


def test_an_off_axis_feed_still_refuses_above_ground():
    msg = _refusal(ground="pec", feeds=[(0, 3.0, 1 + 0j)])
    assert "off the symmetry axis" in msg


def test_a_ground_that_is_not_axisymmetric_still_refuses_above_ground():
    class _Terrain(BSplineSolver):
        def _rotational_ground_kind(self):
            return "terrain"

    with pytest.raises(RotationalSymmetryRefused, match="'terrain' is not invariant"):
        _build(_Terrain)


def _build(cls):
    wires = screen(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(
            wires=wires,
            n_per_edge_per_wire=[[5]] * 4 + [[9, 2]],
            junctions=[[(i, "end") for i in range(4)] + [(4, "end")]],
            feeds=[(4, MAST - 0.05, 1 + 0j)],
            wavelength=WL,
            wire_radius=A,
            degree=2,
            rotational_symmetry=True,
            **GROUNDS["pec"],
        )


def test_singular_enrichment_still_refuses_above_ground():
    msg = _refusal(ground="pec", use_singular_enrichment=True)
    assert "singular enrichment" in msg
