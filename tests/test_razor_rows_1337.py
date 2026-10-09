"""Razor's observer restriction, above ground (momwire#1337 phase 2, PR 5):
`RazorSolver._compute_Z_operator(rows=, compact=)`, the call the sector route
makes, and the prerequisite of razor on that route (momwire#1251).

The deck is `test_route_above_ground_1131.py`'s radial screen in razor
spelling: a mast over N radials at height `h`, one junction at the mast
base. The restriction is radial 0 plus the mast -- whole wires -- so R is
their interior tents and the one junction tent joining them.

What is gated:

* **the fill to the bit** -- the restricted rows against the DENSE fill's
  rows as uint64, square and compact, over free space, PEC, refl-coef and
  Sommerfeld, on the NEC-5 path lane, the Gauss-Legendre lane and the numpy
  (no-accelerator) lane; the square target's other rows exactly zero;
  loading (distributed and lumped) included, through `loading_map`;
* **the restricted path really ran** -- a spy on the T1 token builder shows
  the restricted prepare built |R| rows' observers per source set where the
  dense one built n_basis, and `STATS` shows the prepares cut by
  `basis_windows` and fewer replays;
* **window independence at sub-window sizes 1 and 2** (proposal §5): the
  dense window forced to 1 and to 2 rows, so every requested row reaches
  the T1/T2 kernels in a 1- or 2-row window, still the dense default fill's
  row bit for bit;
* **the prepared fill binds its restriction** -- a replay under another R,
  or under none, or an unrestricted prepare replayed restricted, raises;
  the same R on the other target is served;
* **the restriction composes with the sweep** -- prepared once, replayed at
  two wavenumbers;
* **negative controls** -- radial 1's rows are not radial 0's bit for bit,
  and a fold that loses one window moves Z_in;
* **refusals** -- a split wire, and `compact` without `rows`. The buried
  fills this PR refused take `rows=` since PR 6
  (`test_razor_rows_buried_1337.py`).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from momwire import _schedule
from momwire import razor as _razor
from momwire.razor import RazorSolver

FT = 0.3048
WL = 299792458.0 / 7.2e6
MAST = 33.5 * FT
RADIAL = 33.0 * FT
A = 0.51e-3
SOIL = (30.0, 0.020)
GROUNDS = {
    "free": dict(ground_z=None),
    "pec": dict(ground_z=0.0),
    "refl-coef": dict(ground_z=0.0, ground_eps=SOIL, ground_model="refl-coef"),
    "sommerfeld": dict(ground_z=0.0, ground_eps=SOIL, ground_model="sommerfeld"),
}
LANES = {"nec5": {"nec5_quadrature": True}, "gauss-legendre": {}, "numpy": {}}
ELEVATED = 0.5


def screen(n=4, h=ELEVATED):
    wires = [
        np.array(
            [
                (
                    RADIAL * math.cos(2 * math.pi * i / n),
                    RADIAL * math.sin(2 * math.pi * i / n),
                    h,
                ),
                (0.0, 0.0, h),
            ]
        )
        for i in range(n)
    ]
    wires.append(np.array([(0.0, 0.0, h), (0.0, 0.0, MAST + h)]))
    return wires


def solver(n=4, *, ground="pec", lane="nec5", h=ELEVATED, n_rad=5, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RazorSolver(
            wires=screen(n, h),
            n_per_edge_per_wire=[[n_rad]] * n + [[11]],
            feeds=[(n, 1.0, 1 + 0j)],
            wavelength=WL,
            wire_radius=A,
            **LANES[lane],
            **GROUNDS[ground],
            **kw,
        )


@pytest.fixture
def lane_switch(monkeypatch):
    def apply(lane):
        monkeypatch.setattr(_razor, "_FORCE_NUMPY", lane == "numpy")

    return apply


def wire_rows(s, wires):
    off = s._build_geometry()["seg_offsets"]
    return np.concatenate([np.arange(off[w], off[w + 1]) for w in wires])


def sector0(s):
    return wire_rows(s, (0, len(s.wires_polylines) - 1))


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def fills(s, rows):
    geom = s._build_geometry()
    Z = quiet(s._compute_Z_operator, geom)
    Z_sq = quiet(s._compute_Z_operator, geom, rows=rows)
    Z_c, R = quiet(s._compute_Z_operator, geom, rows=rows, compact=True)
    return Z, Z_sq, Z_c, R


def assert_rows_are_dense(Z, Z_sq, Z_c, R):
    off = np.ones(Z.shape[0], dtype=bool)
    off[R] = False
    assert R.size and off.any()
    assert np.array_equal(bits(Z_sq[R]), bits(Z[R]))
    assert not np.any(Z_sq[off])
    assert Z_sq.flags.f_contiguous
    assert Z_c.shape == (R.size, Z.shape[1])
    assert np.array_equal(bits(Z_c), bits(Z[R]))


class _TokenSpy:
    """What every `_seg_moments_prepare` call was asked to build: one entry
    per call, its observer count. The T1 windows' tokens are the calls
    whose count is a multiple of `n_path` other than the centroid set."""

    def __init__(self, mp):
        self.n_obs = []
        inner = RazorSolver._seg_moments_prepare

        def spy(rs, obs, *a, **kw):
            self.n_obs.append(int(obs.shape[0]))
            return inner(rs, obs, *a, **kw)

        mp.setattr(RazorSolver, "_seg_moments_prepare", spy)


def n_path_of(s):
    pts, _t, _w = s._testing_paths(s._build_geometry())
    return int(pts.shape[1])


# ----------------------------------------------------------------------
# The fill, to the bit
# ----------------------------------------------------------------------


CASES = [
    pytest.param(
        g,
        lane,
        id=f"{lane}-{g}",
        # The Gauss-Legendre Sommerfeld fill is ~8 s: the push lane carries
        # it, and the NEC-5 lane covers the remainder rows on every PR.
        marks=[pytest.mark.slow]
        if (lane, g) == ("gauss-legendre", "sommerfeld")
        else [],
    )
    for g in GROUNDS
    for lane in LANES
    # The numpy lane's Sommerfeld fill is the slow one and adds no branch
    # the other two lanes do not carry; the accelerator lanes cover it.
    if not (lane == "numpy" and g == "sommerfeld")
]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("ground", "lane"), CASES)
def test_the_rows_fill_is_the_dense_fill_row_for_row(ground, lane, lane_switch):
    lane_switch(lane)
    assert _razor._use_razor_fill_accel() is (lane != "numpy")
    s = solver(ground=ground, lane=lane)
    assert_rows_are_dense(*fills(s, sector0(s)))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("ground", ["pec", "refl-coef"])
def test_loading_goes_through_the_loading_map(ground):
    """Distributed loading on every wire and a lumped load on a requested
    AND an unrequested wire: the stencil's unrequested entries are dropped,
    the requested ones land in the target's rows, bit for bit."""
    s = solver(
        ground=ground,
        wire_conductivity=3.5e7,
        lumped_loads=[(0, 2 * RADIAL / 5, 7.0 + 3.0j), (1, 2 * RADIAL / 5, 5.0 - 2.0j)],
    )
    Z, Z_sq, Z_c, R = fills(s, sector0(s))
    assert_rows_are_dense(Z, Z_sq, Z_c, R)
    # The loading is really in those rows: the unloaded fill differs there.
    bare = solver(ground=ground)
    assert not np.array_equal(
        Z[R], quiet(bare._compute_Z_operator, bare._build_geometry())[R]
    )


@pytest.mark.filterwarnings("ignore")
def test_the_full_restriction_is_the_dense_fill():
    s = solver(ground="refl-coef")
    geom = s._build_geometry()
    rows = np.arange(geom["n_segs_total"])
    Z = s._compute_Z_operator(geom)
    Z_c, R = s._compute_Z_operator(geom, rows=rows, compact=True)
    assert np.array_equal(R, np.arange(Z.shape[0]))
    assert np.array_equal(bits(Z_c), bits(Z))


# ----------------------------------------------------------------------
# The restricted path really ran
# ----------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("ground", ["free", "pec", "sommerfeld"])
def test_the_restricted_prepare_builds_only_requested_rows(ground, monkeypatch):
    s = solver(ground=ground)
    geom = s._build_geometry()
    n_path, n_seg = n_path_of(s), geom["n_segs_total"]
    n_basis = geom["n_basis_total"]
    n_sets = 1 if ground == "free" else 2
    rows = sector0(s)

    spy = _TokenSpy(monkeypatch)
    _schedule.reset_stats()
    s._compute_Z_operator(geom)
    dense_stats, dense_obs = dict(_schedule.STATS), spy.n_obs[:]
    spy.n_obs.clear()
    _schedule.reset_stats()
    _Z, R = s._compute_Z_operator(geom, rows=rows, compact=True)
    rest_stats, rest_obs = dict(_schedule.STATS), spy.n_obs[:]

    def t1_rows(obs):
        # The centroid sets are n_seg observers each; the rest are T1 windows.
        return sum(o for o in obs if o != n_seg) // n_path

    assert t1_rows(dense_obs) == n_sets * n_basis
    assert t1_rows(rest_obs) == n_sets * R.size
    assert R.size < n_basis
    assert dense_stats["basis_windows"] == 0
    # R is three runs here (radial 0's tents, the mast's, the junction
    # tent), each a sub-window of the one dense window, per source set.
    assert rest_stats["basis_windows"] == n_sets * 3
    assert rest_stats["prepares"] == dense_stats["prepares"] == n_sets
    assert rest_stats["replays"] == n_sets * 3
    assert dense_stats["replays"] == n_sets


@pytest.mark.filterwarnings("ignore")
def test_rows_none_never_builds_the_restriction(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("rows=None reached the restricted fill")

    monkeypatch.setattr(_razor, "_ObserverRows", boom)
    s = solver(ground="refl-coef")
    s._compute_Z_operator(s._build_geometry())
    s.compute_impedance()


# ----------------------------------------------------------------------
# Window independence at sub-window sizes 1 and 2 (proposal §5)
# ----------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize(
    ("ground", "lane"),
    [
        ("free", "nec5"),
        ("pec", "gauss-legendre"),
        ("refl-coef", "nec5"),
        ("sommerfeld", "nec5"),
        ("pec", "numpy"),
    ],
)
def test_one_and_two_row_windows_are_the_dense_rows(
    step, ground, lane, lane_switch, monkeypatch
):
    lane_switch(lane)
    s = solver(ground=ground, lane=lane)
    geom = s._build_geometry()
    Z = quiet(s._compute_Z_operator, geom)  # the default windows
    n_path, n_seg = n_path_of(s), geom["n_segs_total"]
    budget = step * n_path * geom["n_basis_total"]
    monkeypatch.setattr(_razor, "_CHUNK_ELEMS", budget)
    monkeypatch.setattr(_razor, "_WEIGHTED_CHUNK_ELEMS", budget)
    spy = _TokenSpy(monkeypatch)
    rows = sector0(s)
    Z_sq = quiet(s._compute_Z_operator, geom, rows=rows)
    Z_c, R = quiet(s._compute_Z_operator, geom, rows=rows, compact=True)
    assert_rows_are_dense(Z, Z_sq, Z_c, R)
    # Every T1 window reached the kernels at 1 (or 2) rows, and both sizes
    # occurred at step 2 (a run of odd length leaves a 1-row tail).
    sizes = {o // n_path for o in spy.n_obs if o != n_seg}
    assert sizes == ({1} if step == 1 else {1, 2})


# ----------------------------------------------------------------------
# The prepared fill binds its restriction
# ----------------------------------------------------------------------


def _prepared(s, rows, *, compact=True):
    geom = s._build_geometry()
    restrict = (
        None if rows is None else _razor._ObserverRows(rows, geom, compact=compact)
    )
    return geom, restrict, s._assemble_Z_prepare(geom, restrict=restrict)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("ground", ["free", "pec"])
def test_a_replay_under_another_restriction_raises(ground):
    s = solver(ground=ground)
    geom, r0, prep0 = _prepared(s, sector0(s))
    r1 = _razor._ObserverRows(wire_rows(s, (1, 4)), geom, compact=True)
    omega = s.c * s.k
    with pytest.raises(ValueError, match="binds the rows= it was prepared for"):
        s._assemble_Z_from_prepared(geom, prep0, s.k, omega, restrict=r1)
    with pytest.raises(ValueError, match="replayed under no restriction"):
        s._assemble_Z_from_prepared(geom, prep0, s.k, omega)
    _g, _r, dense = _prepared(s, None)
    with pytest.raises(ValueError, match="built for every row"):
        s._assemble_Z_from_prepared(geom, dense, s.k, omega, restrict=r0)
    # The same R on the other target is the same windows: served.
    sq = _razor._ObserverRows(sector0(s), geom, compact=False)
    Z_sq = s._assemble_Z_from_prepared(geom, prep0, s.k, omega, restrict=sq)
    Z_c = s._assemble_Z_from_prepared(geom, prep0, s.k, omega, restrict=r0)
    assert np.array_equal(bits(Z_sq[r0.basis_rows]), bits(Z_c))


@pytest.mark.filterwarnings("ignore")
def test_the_restriction_composes_with_the_sweep():
    """Restrict once at prepare, replay per k: each k's rows are that k's
    dense rows."""
    s = solver(ground="refl-coef")
    geom, restrict, prep = _prepared(s, sector0(s))
    dense = s._assemble_Z_prepare(geom)
    for kk in np.array([0.97, 1.03]) * 2 * math.pi / WL:
        Z = s._assemble_Z_from_prepared(geom, dense, kk, s.c * kk)
        Z_c = s._assemble_Z_from_prepared(geom, prep, kk, s.c * kk, restrict=restrict)
        assert np.array_equal(bits(Z_c), bits(Z[restrict.basis_rows]))


# ----------------------------------------------------------------------
# Negative controls
# ----------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
def test_negative_another_radials_rows_are_not_these_rows():
    """The bitwise gate can fail: radial 1's tents are a rotation of radial
    0's, the same shape and the same physics, and not the same entries."""
    s = solver(ground="refl-coef")
    geom = s._build_geometry()
    Z = s._compute_Z_operator(geom)
    Z0, R0 = s._compute_Z_operator(geom, rows=wire_rows(s, (0,)), compact=True)
    Z1, R1 = s._compute_Z_operator(geom, rows=wire_rows(s, (1,)), compact=True)
    assert Z0.shape == Z1.shape
    assert np.array_equal(bits(Z0), bits(Z[R0]))
    assert np.array_equal(bits(Z1), bits(Z[R1]))
    assert not np.array_equal(bits(Z0), bits(Z1))


def _z_in(s, Z):
    cols = s._port_columns(s._build_geometry())
    c = np.linalg.solve(Z, cols @ s._port_voltages())
    return complex(s._port_voltages()[0] / (cols.T @ c)[0])


@pytest.mark.filterwarnings("ignore")
def test_negative_a_fold_that_loses_a_window_moves_z_in(monkeypatch):
    """The restricted rows feed an answer, and a corruption of the
    restricted path reaches it: Z_in from the dense Z with R's rows replaced
    by the restricted fill is the dense Z_in EXACTLY, and dropping one
    window from the image prepare (radial 0's tents lose their image) moves
    it by over 1 %."""
    s = solver(ground="pec")
    geom = s._build_geometry()
    rows = sector0(s)
    Z = s._compute_Z_operator(geom)
    z_d = _z_in(s, Z)

    def hybrid():
        Z_c, R = s._compute_Z_operator(geom, rows=rows, compact=True)
        Z_h = Z.copy()
        Z_h[R] = Z_c
        return _z_in(s, Z_h)

    assert hybrid() == z_d

    orig = _schedule.prepare
    seen = []

    def lossy(n_rows, step, build, *, checkpoint=None, restrict=None):
        out = orig(n_rows, step, build, checkpoint=checkpoint, restrict=restrict)
        if restrict is not None:
            seen.append(len(out))
            if len(seen) == 2:  # the image source set's windows
                out = out[1:]
        return out

    monkeypatch.setattr(_schedule, "prepare", lossy)
    z_bad = hybrid()
    assert seen == [3, 3]
    assert abs(z_bad - z_d) / abs(z_d) > 1e-2, (z_bad, z_d)


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


def test_a_split_wire_is_refused_by_name():
    s = solver(ground="free")
    geom = s._build_geometry()
    with pytest.raises(ValueError, match="restricts by whole wires"):
        s._compute_Z_operator(geom, rows=np.arange(3))


def test_compact_without_rows_is_refused():
    s = solver(ground="free")
    with pytest.raises(ValueError, match="pass rows= as well"):
        s._compute_Z_operator(s._build_geometry(), compact=True)
