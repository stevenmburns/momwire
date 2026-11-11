"""momwire#1912: the sector route (`rotational_symmetry=True`) on the
point-matched `SinusoidalSolver`.

The route is BSplineSolver's (momwire#1029, #1897) on a basis with one
unknown per segment and one match point per segment: sector 0's segments
and the axial group's are the only observer rows filled, against every
source column, and the harmonic blocks and their solves are
`_rotational_symmetry`'s, with no constraint rows (this basis carries its
junction conditions in its coefficients). Every gate below is the route
against the dense solve of the SAME deck (`rotational_symmetry=False`), so
a coarse mesh weakens nothing.

The decks are #1897's — a mast inside a cage of off-axis verticals, with
and without a port on every vertical, members drawn the other way round,
and a fed vertical ringed by parasitic verticals — and #1029's radial
screen in the forms this family serves in one medium: elevated over every
ground, standing ON a Sommerfeld half-space (a ground contact on the axial
wire), with wire loading, and wholly buried. The last test is antennaknobs'
`verticals.elt_whip:sym4` (`slow`).
"""

from __future__ import annotations

import gzip
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import SinusoidalGalerkinSolver, SinusoidalSolver
from momwire import _rotational_symmetry as rs
from momwire._rotational_symmetry import RotationalSymmetryRefused

WL = 2.0
SOMM = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
GROUNDS = {
    "free": {},
    "pec": dict(ground_z=0.0),
    "refl-coef": dict(ground_z=0.0, ground_eps=(13.0, 0.005)),
    "sommerfeld": SOMM,
}
LIFT = 0.3
# The route's bar (the issue's acceptance): every Y / Z entry and every
# coefficient, relative to the array's largest entry.
TOL = 1e-9
N = 4


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def cage(n=N, *, lift=0.0):
    """#1897's cage: feed stub and mast on the axis (wires 0, 1), then n
    off-axis verticals (2 .. n+1), n top-ring chords, n spokes."""
    H, FOOT, R = 0.5, 0.05, 0.08
    z0, z1, z2 = lift, lift + FOOT, lift + H
    foot = [
        (R * math.cos(2 * math.pi * i / n), R * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    wires = [
        np.array([(0.0, 0.0, z0), (0.0, 0.0, z1)]),
        np.array([(0.0, 0.0, z1), (0.0, 0.0, z2)]),
    ]
    npe = [[3], [10]]
    for x, y in foot:
        wires.append(np.array([(x, y, z1), (x, y, z2)]))
        npe.append([8])
    for i in range(n):
        (ax, ay), (bx, by) = foot[i], foot[(i + 1) % n]
        wires.append(np.array([(ax, ay, z2), (bx, by, z2)]))
        npe.append([3])
    for x, y in foot:
        wires.append(np.array([(0.0, 0.0, z1), (x, y, z1)]))
        npe.append([3])
    return wires, npe


FEED = (0, 0.025, 1 + 0j)
SPAN = 0.45  # a cage vertical's length


def posts(n=N, v=0j, reversed_=()):
    """A port on every cage vertical, 0.2 m up: n rotated copies. A vertical
    drawn the other way names the same point from its other end."""
    return [(2 + i, SPAN - 0.2 if i in reversed_ else 0.2, v) for i in range(n)]


def reverse(wires, npe, idx):
    wires = [w.copy() for w in wires]
    npe = [list(e) for e in npe]
    for i in idx:
        wires[i] = wires[i][::-1].copy()
        npe[i] = npe[i][::-1]
    return wires, npe


def parasitic(n=N, *, lift=0.0):
    """#1897's parasitic ring: a fed centre vertical and n parasitic
    verticals at 0.25 wl, each standing on 4 short radials of its own, the
    radials turned with the element so the ring is C_n."""
    H, RR, ARM = 0.48, 0.5, 0.25
    w, npe = [], []
    centres = [(0.0, 0.0)] + [
        (RR * math.cos(2 * math.pi * i / n), RR * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    for k, (cx, cy) in enumerate(centres):
        w.append(np.array([(cx, cy, lift), (cx, cy, lift + 0.04)]))
        npe.append([3])
        w.append(np.array([(cx, cy, lift + 0.04), (cx, cy, lift + H)]))
        npe.append([12])
        for j in range(4):
            phi = (math.atan2(cy, cx) if k else 0.0) + 2 * math.pi * j / 4
            w.append(
                np.array(
                    [
                        (cx, cy, lift),
                        (cx + ARM * math.cos(phi), cy + ARM * math.sin(phi), lift),
                    ]
                )
            )
            npe.append([5])
    return w, npe, [(0, 0.02, 1 + 0j)]


def screen(n=N, *, z0=0.3, droop=0.0):
    """#1029's radial screen in one medium (the elevated form momwire#1131
    put on the route): a mast from `z0` up and n radials from its foot,
    drooping by `droop` m at the tip, fed at the mast foot."""
    arm, mast = 0.5, 0.48
    wires = [np.array([(0.0, 0.0, z0), (0.0, 0.0, z0 + mast)])]
    npe = [[12]]
    for i in range(n):
        th = 2 * math.pi * i / n
        tip = (arm * math.cos(th), arm * math.sin(th), z0 - droop)
        wires.append(np.array([(0.0, 0.0, z0), tip]))
        npe.append([7])
    return wires, npe, [(0, 0.02, 1 + 0j)]


def top_hat(n=N):
    """A vertical standing ON the ground (contact at its foot) under n
    sloping top-hat wires — the screen's grounded twin, so the contact
    correction's rows reach the route."""
    mast = 0.45
    wires = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, mast)])]
    npe = [[11]]
    for i in range(n):
        th = 2 * math.pi * i / n
        wires.append(
            np.array([(0.0, 0.0, mast), (0.3 * math.cos(th), 0.3 * math.sin(th), 0.3)])
        )
        npe.append([6])
    return wires, npe, [(0, 0.02, 1 + 0j)]


def buried(n=N):
    """A deck WHOLLY below the interface (one medium, momwire#1222): a short
    buried vertical over n buried radials."""
    d = 0.3
    wires = [np.array([(0.0, 0.0, -d), (0.0, 0.0, -0.05)])]
    npe = [[7]]
    for i in range(n):
        th = 2 * math.pi * i / n
        wires.append(
            np.array([(0.0, 0.0, -d), (0.4 * math.cos(th), 0.4 * math.sin(th), -d)])
        )
        npe.append([6])
    return wires, npe, [(0, 0.03, 1 + 0j)]


def screen_1029(n=N, nrad=6):
    """#1029's own deck: n radials to a hub 0.15 m down, the rise to the
    crossing node at z = 0, the mast above it, fed on the mast. Two media
    and a crossing junction ON the axis (`_assemble_Z_mixed`'s crossing
    rows), at the design's 40 m wavelength (`WL_1029`)."""
    depth, arm, mast = 0.15, 6.334, 10.556
    wires = []
    for i in range(n):
        th = 2 * math.pi * i / n
        wires.append(
            np.array(
                [(0.0, 0.0, -depth), (arm * math.cos(th), arm * math.sin(th), -depth)]
            )
        )
    wires.append(np.array([(0.0, 0.0, -depth), (0.0, 0.0, 0.0)]))
    wires.append(np.array([(0.0, 0.0, 0.0), (0.0, 0.0, mast)]))
    return wires, [[nrad]] * n + [[3], [8]], [(n + 1, 0.5 * mast / 8, 1 + 0j)]


WL_1029 = 42.2


def detached(n=8):
    """Two media and NO crossing: a mast above the interface over n buried
    radials it does not touch (the transmitted blocks of the mixed fill)."""
    wires = [np.array([(0.0, 0.0, 0.05), (0.0, 0.0, 0.5)])]
    npe = [[11]]
    for i in range(n):
        th = 2 * math.pi * i / n
        c, s_ = math.cos(th), math.sin(th)
        wires.append(
            np.array([(0.05 * c, 0.05 * s_, -0.08), (0.4 * c, 0.4 * s_, -0.08)])
        )
        npe.append([8])
    return wires, npe, [(0, 0.02, 1 + 0j)]


def build(wires, npe, feeds, *, rot=True, ground="free", cls=None, **kw):
    cls = SinusoidalSolver if cls is None else cls
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(
            wires=wires,
            n_per_edge_per_wire=npe,
            feeds=feeds,
            rotational_symmetry=rot,
            **GROUNDS[ground],
            **{"wavelength": WL, "wire_radius": 0.002, **kw},
        )


def pair(wires, npe, feeds, **kw):
    return build(wires, npe, feeds, rot=True, **kw), build(
        wires, npe, feeds, rot=False, **kw
    )


def loaded_zin(Y, z_load):
    Zm = np.linalg.inv(Y)
    n = Y.shape[0] - 1
    return complex(
        Zm[0, 0]
        - Zm[0, 1:] @ np.linalg.solve(Zm[1:, 1:] + z_load * np.eye(n), Zm[1:, 0])
    )


def assert_impedance(s, d):
    s._rotational_copy_spread = None
    zs, cs = s.compute_impedance()
    # set by the harmonic-0 block: the route ran, not a dense fallback
    assert s._rotational_copy_spread is not None
    zd, cd = d.compute_impedance()
    assert rel(zs, zd) <= TOL, rel(zs, zd)
    assert rel(cs, cd) <= TOL, rel(cs, cd)
    return zs, cs, cd


def assert_ports(s, d):
    s._rotational_copy_spread = None
    ps, pd = s.compute_port_solution(), d.compute_port_solution()
    assert s._rotational_copy_spread is not None
    assert ps.y.shape == pd.y.shape
    assert rel(ps.y, pd.y) <= TOL, rel(ps.y, pd.y)
    assert rel(ps.coeffs, pd.coeffs) <= TOL, rel(ps.coeffs, pd.coeffs)
    assert np.array_equal(s.compute_y_matrix(), ps.y)
    return ps, pd


# ----------------------------------------------------------------------
# The reading and the roster
# ----------------------------------------------------------------------


def test_the_segment_groups_partition_the_unknowns_by_orbit():
    """One unknown per segment: sector 0's segments, the N-1 images at the
    same positions, and the axial rest — a partition, equal sector sizes,
    and a reversed member listed from its far end."""
    w, npe = reverse(*cage(), [3])
    s = build(w, npe, [FEED])
    geom = s._build_geometry()
    n = int(geom["n_segs"])
    sectors, axial = rs.segment_groups(s, geom["wire_first"], n)
    assert len(sectors) == N and {x.size for x in sectors} == {8 + 3 + 3}
    joined = np.sort(np.concatenate([*sectors, axial]))
    assert np.array_equal(joined, np.arange(n))
    assert axial.size == 3 + 10
    first = np.asarray(geom["wire_first"])
    # sector 1's vertical (wire 3) runs backwards: its first listed segment
    # is the wire's LAST one
    assert sectors[1][0] == first[3] + 7 and sectors[1][7] == first[3]
    D = rs.segment_signs(s, geom["wire_first"], n)
    assert set(np.flatnonzero(D < 0)) == set(range(first[3], first[3] + 8))
    assert rs.segment_signs(build(*cage(), [FEED]), geom["wire_first"], n) is None


def test_the_roster_carries_sector_on_the_point_matched_row_only():
    assert SinusoidalSolver.capabilities.axes["solve_strategy"] == ("dense", "sector")
    assert SinusoidalGalerkinSolver.capabilities.axes["solve_strategy"] == ("dense",)


# ----------------------------------------------------------------------
# Against the dense solve
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ground", ["free", "sommerfeld"])
@pytest.mark.parametrize("n", [4, 6])
def test_the_cage_matches_the_dense_solve(n, ground):
    lift = LIFT if ground != "free" else 0.0
    s, d = pair(*cage(n, lift=lift), [FEED], ground=ground)
    assert s._rotational_map.n_sectors == n
    assert_impedance(s, d)


@pytest.mark.parametrize("ground", ["free", "sommerfeld"])
def test_loaded_posts_take_every_harmonic_through_the_port_solution(ground):
    lift = LIFT if ground != "free" else 0.0
    s, d = pair(*cage(lift=lift), [FEED, *posts()], ground=ground)
    geom = s._build_geometry()
    sectors, _ax = rs.segment_groups(s, geom["wire_first"], int(geom["n_segs"]))
    B = np.zeros((int(geom["n_segs"]), 1 + N), dtype=complex)
    for j, fi in enumerate(geom["feed_segs"]):
        B[fi, j] = 1.0
    assert rs.harmonics_needed(B, np.zeros((0, B.shape[0])), sectors) == (0, 1, 2, 3)
    ps, pd = assert_ports(s, d)
    omega = 2 * math.pi * 299792458.0 / WL
    z_load = 1 / (1j * omega * 1e-12)
    zr, zd = loaded_zin(ps.y, z_load), loaded_zin(pd.y, z_load)
    assert abs(zr - zd) / abs(zd) <= TOL


@pytest.mark.parametrize("ground", ["free", "sommerfeld"])
def test_members_drawn_the_other_way_are_signed_and_solved(ground):
    """Sector 1's vertical and sector 2's chord and spoke reversed: the
    route signs their segments' bases and rows (D Z D) and still matches,
    through compute_impedance and through every port."""
    lift = LIFT if ground != "free" else 0.0
    w, npe = reverse(*cage(lift=lift), [3, 2 + N + 2, 2 + 2 * N + 2])
    s, d = pair(w, npe, [FEED], ground=ground)
    assert s._rotational_map.reversed == (
        (False, False, False),
        (True, False, False),
        (False, True, True),
        (False, False, False),
    )
    assert_impedance(s, d)
    s, d = pair(w, npe, [FEED, *posts(reversed_=(1,))], ground=ground)
    assert_ports(s, d)


@pytest.mark.parametrize("ground", ["free", "sommerfeld"])
def test_the_parasitic_ring_of_off_axis_verticals_matches(ground):
    lift = LIFT if ground != "free" else 0.0
    s, d = pair(*parasitic(lift=lift), ground=ground)
    assert s._rotational_map.n_sectors == N
    assert_impedance(s, d)


@pytest.mark.parametrize("ground", ["free", "pec", "refl-coef", "sommerfeld"])
def test_the_elevated_radial_screen_matches_over_every_ground(ground):
    s, d = pair(*screen(droop=0.05), ground=ground)
    assert s._rotational_map.axial == (0,)
    assert_impedance(s, d)


def test_the_screen_on_six_radials_with_lossy_jacketed_wire():
    """Wire loading enters the point-matched rows (`_apply_loading`'s
    compact form): finite conductivity and a lossy jacket on every wire."""
    w, npe, feeds = screen(6)
    kw = dict(
        wire_conductivity=3.5e7,
        insulation_radius=0.003,
        insulation_eps_r=3.0,
        insulation_loss_tangent=0.01,
    )
    s, d = pair(w, npe, feeds, ground="sommerfeld", **kw)
    assert s._loading_active
    assert_impedance(s, d)


def test_a_vertical_standing_on_the_ground_takes_the_contact_rows():
    """The axial wire's foot lies in a Sommerfeld plane, so #282's contact
    correction runs on the route's (axial) rows."""
    s, d = pair(*top_hat(), ground="sommerfeld")
    geom = s._build_geometry()
    assert s._contact_nodes(geom)
    assert_impedance(s, d)


def test_a_wholly_buried_screen_matches():
    s, d = pair(*buried(), ground="sommerfeld")
    assert_impedance(s, d)


@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_1029_buried_screen_with_its_crossing_node_matches(n):
    w, npe, feeds = screen_1029(n)
    s, d = pair(w, npe, feeds, ground="sommerfeld", wavelength=WL_1029)
    assert s._is_crossing(s._build_geometry())
    assert s._rotational_map.axial == (n, n + 1)
    assert_impedance(s, d)


def test_a_detached_two_media_deck_matches():
    s, d = pair(*detached(), ground="sommerfeld")
    geom = s._build_geometry()
    assert s._is_mixed(geom) and not s._is_crossing(geom)
    assert int(geom["n_segs"]) >= 60
    assert_impedance(s, d)


def test_the_pattern_reads_the_route_coefficients_as_the_dense_ones():
    """`element_currents` and the far-field moments are read off the
    coefficients, which the route hands back in the dense ordering."""
    s, d = pair(*cage(lift=LIFT), [FEED], ground="sommerfeld")
    _zs, cs, cd = assert_impedance(s, d)
    ms, md = s.element_currents(cs), d.element_currents(cd)
    assert np.array_equal(ms[0], md[0])
    assert rel(ms[1], md[1]) <= TOL


def test_the_swept_entries_loop_the_per_k_route_bit_for_bit():
    s, d = pair(*cage(), [FEED, *posts()])
    ks = s.k * np.array([0.95, 1.05])
    swept_y = list(s.compute_y_matrix_swept(ks))
    k0 = s.k
    for kk, y in zip(ks, swept_y, strict=True):
        s._set_k(float(kk))
        assert np.array_equal(y, s.compute_y_matrix())
    s._set_k(k0)
    assert s.k == k0
    dense_y = list(d.compute_y_matrix_swept(ks))
    assert max(rel(a, b) for a, b in zip(swept_y, dense_y, strict=True)) <= TOL

    s, d = pair(*cage(), [FEED])
    z_sw = s.compute_impedance_swept(ks)
    assert z_sw.shape == (2,) and s.k == k0
    for kk, z in zip(ks, z_sw, strict=True):
        s._set_k(float(kk))
        assert z == s.compute_impedance()[0]
    s._set_k(k0)
    assert rel(z_sw, d.compute_impedance_swept(ks)) <= TOL


def test_a_driven_orbit_is_harmonic_zero_and_a_phased_one_refuses_the_drive():
    s, d = pair(*cage(), posts(v=1 + 0j))
    zs, _ = s.compute_impedance()
    zd, _ = d.compute_impedance()
    assert rel(zs, zd) <= TOL
    phased = [(w, a, complex(1.0, 0.1 * i)) for i, (w, a, _v) in enumerate(posts())]
    s, d = pair(*cage(), phased)
    with pytest.raises(RotationalSymmetryRefused) as exc:
        s.compute_impedance()
    assert "the drive is not rotation-invariant" in str(exc.value)
    assert rel(s.compute_y_matrix(), d.compute_y_matrix()) <= TOL


# ----------------------------------------------------------------------
# The probes that would catch a wrong map
# ----------------------------------------------------------------------


def _reversed_ports():
    w, npe = reverse(*cage(), [3, 2 + N + 2, 2 + 2 * N + 2])
    return w, npe, [FEED, *posts(reversed_=(1,))]


def test_dropping_the_reversed_segment_sign_is_caught(monkeypatch):
    """The reversed-member gate has teeth: with `segment_signs` answering
    None (no member signed) the route is far from the dense oracle, both
    through every harmonic and through harmonic 0 alone."""
    w, npe, feeds = _reversed_ports()
    d = build(w, npe, feeds, rot=False)
    y_dense = d.compute_y_matrix()
    z_dense = build(w, npe, [FEED], rot=False).compute_impedance()[0]
    assert rel(build(w, npe, feeds).compute_y_matrix(), y_dense) <= TOL
    monkeypatch.setattr(rs, "segment_signs", lambda *a, **k: None)
    assert rel(build(w, npe, feeds).compute_y_matrix(), y_dense) > 1e-3
    s = build(w, npe, [FEED])
    assert abs(s.compute_impedance()[0] - z_dense) / abs(z_dense) > 1e-3


def test_an_orbit_map_off_by_one_is_caught(monkeypatch):
    """Two wrong maps, each pinned to the dense oracle: every sector after
    the first shifted one segment along (the POSITION pairing broken), and
    sectors 1 and 2 swapped (a labelling that is no rotation of C4, which
    only the harmonics past 0 can see)."""
    w, npe, feeds = cage()[0], cage()[1], [FEED, *posts()]
    y_dense = build(w, npe, feeds, rot=False).compute_y_matrix()
    real = rs.segment_groups

    def shifted(*a, **k):
        sectors, axial = real(*a, **k)
        return [sectors[0]] + [np.roll(x, 1) for x in sectors[1:]], axial

    monkeypatch.setattr(rs, "segment_groups", shifted)
    assert rel(build(w, npe, feeds).compute_y_matrix(), y_dense) > 1e-3

    def swapped(*a, **k):
        sectors, axial = real(*a, **k)
        return [sectors[0], sectors[2], sectors[1], sectors[3]], axial

    monkeypatch.setattr(rs, "segment_groups", swapped)
    assert rel(build(w, npe, feeds).compute_y_matrix(), y_dense) > 1e-3


# ----------------------------------------------------------------------
# What stays refused, and the flag off
# ----------------------------------------------------------------------


def test_the_bspline_refusals_read_the_same_here():
    """The deck-shape conditions are `sector_map`'s, shared, so a lone
    off-axis port is refused in the same words as on BSplineSolver."""
    with pytest.raises(RotationalSymmetryRefused) as exc:
        build(*cage(), [(2, 0.2, 1 + 0j)])
    msg = str(exc.value)
    assert "port 'feed 0' sits at (0.080, 0.000, 0.250)" in msg
    assert "its rotated images are not all ports" in msg
    assert msg.endswith("Drop rotational_symmetry=True to solve this deck densely.")
    w, npe, feeds = screen()
    w[2] = np.array([(0.0, 0.0, 0.3), (0.0, 0.55, 0.3)])  # one longer radial
    with pytest.raises(RotationalSymmetryRefused):
        build(w, npe, feeds)


@pytest.mark.parametrize(
    "case,needle",
    [
        ("galerkin", "SinusoidalGalerkinSolver does not serve the sector route"),
        ("nec42", "the NEC-4 junction rule"),
    ],
)
def test_the_family_scope_refuses_by_name_before_any_fill(case, needle, monkeypatch):
    def _no(self, *a, **kw):
        raise AssertionError("filled while refusing")

    monkeypatch.setattr(SinusoidalSolver, "_assemble_Z", _no)
    if case == "galerkin":
        w, npe = cage()
        kw = dict(cls=SinusoidalGalerkinSolver)
        feeds = [FEED]
    else:
        w, npe = cage()
        kw = dict(junction_rule="nec42")
        feeds = [FEED]
    with pytest.raises(RotationalSymmetryRefused) as exc:
        build(w, npe, feeds, **kw)
    assert needle in str(exc.value)
    assert str(exc.value).endswith(
        "Drop rotational_symmetry=True to solve this deck densely."
    )


def test_the_flag_off_runs_no_route_code(monkeypatch):
    def _no(*a, **k):
        raise AssertionError("route code ran with the flag off")

    for name in ("segment_groups", "segment_signs", "sector_map", "solve_ports"):
        monkeypatch.setattr(rs, name, _no)
    d = build(*cage(), [FEED, *posts()], rot=False)
    assert d._rotational_map is None and d.rotational_symmetry is False
    d.compute_impedance()
    d.compute_y_matrix()
    d.compute_impedance_swept(d.k * np.array([0.9, 1.1]))


@pytest.mark.parametrize("deck", ["crossing", "detached"])
def test_the_compact_mixed_fill_rows_are_the_dense_rows(deck):
    """`_assemble_Z_mixed(obs_rows=)` against the square mixed fill's rows:
    the class bands, the transmitted blocks and the crossing rows each
    restricted to the compact observers, the same numbers row for row."""
    if deck == "crossing":
        w, npe, feeds = screen_1029(16)
        d = build(w, npe, feeds, rot=False, ground="sommerfeld", wavelength=WL_1029)
    else:
        d = build(*detached(), rot=False, ground="sommerfeld")
    geom = d._build_geometry()
    n = int(geom["n_segs"])
    assert n >= 60
    G, _v = d._assemble_Z(geom, d.k)
    first = np.asarray(geom["wire_first"])
    rows = np.union1d(np.arange(first[1], first[2]), np.arange(first[-2], n))
    Gc, _v = d._assemble_Z(geom, d.k, obs_rows=rows)
    assert Gc.shape == (rows.size, n)
    assert np.array_equal(Gc, G[rows])


def test_the_compact_fill_rows_are_the_dense_rows():
    """`_assemble_Z(obs_rows=)` against the square fill's rows, on a deck
    whose fill takes the sparse reduction, a ground, a contact and loading:
    the same numbers row for row."""
    w, npe, feeds = top_hat(6)
    npe[0] = [24]
    d = build(w, npe, feeds, rot=False, ground="sommerfeld", wire_conductivity=5e6)
    geom = d._build_geometry()
    n = int(geom["n_segs"])
    assert n >= 60  # the sparse-M regime, where a row's bits are its own
    G, _v = d._assemble_Z(geom, d.k)
    rows = np.array(sorted(set(range(0, 11)) | set(range(30, 41))), dtype=np.int64)
    Gc, _v = d._assemble_Z(geom, d.k, obs_rows=rows)
    assert Gc.shape == (rows.size, n)
    assert np.array_equal(Gc, G[rows])


# ----------------------------------------------------------------------
# The case that asked for it: antennaknobs' verticals.elt_whip:sym4
# ----------------------------------------------------------------------

SYM4 = Path(__file__).parent / "fixtures" / "elt_whip_sym4_1897.json.gz"


@pytest.mark.slow
def test_sym4_on_the_point_matched_route_matches_its_dense_solve():
    """3,799 wires, C4 about the whip, 1,758 sector wires drawn backwards,
    four loaded posts: a 5-port solve through all four harmonics, free
    space, 406 MHz. The route against this family's own dense solve."""
    doc = json.load(gzip.open(SYM4, "rt"))

    def mk(rot):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SinusoidalSolver(
                wires=[np.array(w) for w in doc["wires"]],
                n_per_edge_per_wire=doc["n_per_edge_per_wire"],
                wire_radius=doc["wire_radius"],
                feeds=[(w, a, complex(*v)) for w, a, v in doc["feeds"]],
                junctions=[[(w, e) for w, e in j] for j in doc["junctions"]],
                wavelength=doc["wavelength"],
                rotational_symmetry=rot,
            )

    s = mk(True)
    m = s._rotational_map
    assert m.n_sectors == 4 and sum(map(sum, m.reversed)) == 1758
    ys = s.compute_y_matrix()
    yd = mk(False).compute_y_matrix()
    assert rel(ys, yd) <= TOL


@pytest.mark.parametrize("ground", ["free", "pec"])
def test_pair_reuse_rides_the_compact_fill(ground):
    """`pair_reuse` (momwire#1911) serves the route's bands as it serves the
    square fill's: the route with it on against the dense solve with it
    off, on a deck it does not decline."""
    w, npe, feeds = screen(6)
    s = build(w, npe, feeds, ground=ground, pair_reuse=True)
    d = build(w, npe, feeds, rot=False, ground=ground)
    zs, cs = s.compute_impedance()
    assert s.pair_reuse_declined is None and s.pair_reuse_census
    zd, cd = d.compute_impedance()
    assert rel(zs, zd) <= TOL and rel(cs, cd) <= TOL
