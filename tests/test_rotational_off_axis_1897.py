"""momwire#1897: the sector route serves any C_N structure, not only a radial
screen — vertical wires off the axis, ports that come as N rotated copies,
junctions off the axis, and wires drawn either way round.

The deck is a mast inside a cage of N verticals, closed by a top ring and
joined to the mast by N spokes: every vertical but the mast stands off the
axis, and every ring and spoke junction is off it too. As in
`test_rotational_symmetry_1029.py`, every gate is the route against the
dense solve on the SAME deck, so a coarse mesh weakens nothing.

The last test is the case that asked for this: antennaknobs'
`verticals.elt_whip:sym4` (3,799 wires, four loaded posts), held to the
dense answer recorded for it. It is `slow`.
"""

from __future__ import annotations

import gzip
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire import _rotational_symmetry as rs
from momwire._rotational_symmetry import RotationalSymmetryRefused

WL = 2.0
H = 0.5  # cage top
FOOT = 0.05  # the spokes' height, where the feed stub meets the mast
R = 0.08  # cage radius
N = 4
SOMM = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
LIFT = 0.3  # over ground, the whole cage is raised this far
# The route's bars, phase 1's G1 (`test_rotational_symmetry_1029.py`).
Z_TOL = 1e-9
C_TOL = 1e-8


def cage(n=N, *, move=0.0, lift=0.0):
    """`(wires, n_per_edge)`: feed stub and mast on the axis (wires 0, 1),
    then n verticals (2 .. n+1), n top-ring chords, n spokes. `move` shifts
    vertical 1's foot, top and so its chords and spoke, by that many metres
    in x."""
    foot = []
    for i in range(n):
        th = 2 * math.pi * i / n
        foot.append((R * math.cos(th) + (move if i == 1 else 0.0), R * math.sin(th)))
    z0, z1, z2 = lift, lift + FOOT, lift + H
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


FEED = (0, 0.5 * FOOT, 1 + 0j)


def build(*, rot=True, ground=None, feeds=None, wires=None, npe=None):
    if wires is None:
        wires, npe = cage(lift=LIFT if ground else 0.0)
    kw = SOMM if ground else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            degree=2,
            wavelength=WL,
            wire_radius=0.002,
            feeds=[FEED] if feeds is None else feeds,
            rotational_symmetry=rot,
            **kw,
        )


def screen():
    """momwire#1029's deck: four buried radials, the rise, the mast."""
    depth, arm, mast = 0.15, 6.334, 10.556
    wires = [
        np.array([(0.0, 0.0, -depth), (x, y, -depth)])
        for x, y in ((arm, 0.0), (0.0, arm), (-arm, 0.0), (0.0, -arm))
    ]
    wires.append(np.array([(0.0, 0.0, -depth), (0.0, 0.0, 0.0)]))
    wires.append(np.array([(0.0, 0.0, 0.0), (0.0, 0.0, mast)]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=[[6]] * 4 + [[3], [8]],
            degree=2,
            wavelength=42.2,
            wire_radius=0.001,
            feeds=[(5, 0.5 * mast / 8, 1 + 0j)],
            rotational_symmetry=True,
            **SOMM,
        )


def pair(**kw):
    return build(rot=True, **kw), build(rot=False, **kw)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def posts(v=0j):
    """A port on every cage vertical, at one height: N rotated copies."""
    return [(2 + i, 0.2, v) for i in range(N)]


def loaded_zin(Y, z_load):
    """Port 0's input impedance with `z_load` across every other port — the
    port algebra a consumer stamps a load with."""
    Zm = np.linalg.inv(Y)
    n = Y.shape[0] - 1
    return complex(
        Zm[0, 0]
        - Zm[0, 1:] @ np.linalg.solve(Zm[1:, 1:] + z_load * np.eye(n), Zm[1:, 0])
    )


# ----------------------------------------------------------------------
# The reading
# ----------------------------------------------------------------------


def test_the_cage_reads_as_n_sectors_with_its_verticals_in_them():
    """The mast's line is the axis; each off-axis vertical is an ordinary
    sector wire, in an orbit of N with its ring chord and its spoke."""
    m = build()._rotational_map
    assert m.axis == (0.0, 0.0)
    assert m.n_sectors == N
    assert m.axial == (0, 1)
    assert m.sectors[0] == (2, 2 + N, 2 + 2 * N)
    assert sorted(w for sec in m.sectors for w in sec) == list(range(2, 2 + 3 * N))
    assert m.reversed == ()


def test_the_screen_reading_is_unchanged():
    """A deck whose vertical wires share ONE line never reaches the new
    reading: the radial screen of momwire#1029 maps as it always did."""
    m = screen()._rotational_map
    assert (m.n_sectors, m.sectors, m.axial, m.reversed) == (
        4,
        ((0,), (1,), (2,), (3,)),
        (4, 5),
        (),
    )


# ----------------------------------------------------------------------
# Against the dense solve
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ground", [None, "sommerfeld"], ids=["free", "sommerfeld"])
def test_the_cage_matches_the_dense_solve(ground):
    """compute_impedance: harmonic 0 only. The ring and spoke junctions are
    off the axis, so their KCL rows reach the solve as orbit sums."""
    s, d = pair(ground=ground)
    zs, cs = s.compute_impedance()
    zd, cd = d.compute_impedance()
    assert abs(zs - zd) / abs(zd) <= Z_TOL
    assert rel(cs, cd) <= C_TOL


@pytest.mark.parametrize("ground", [None, "sommerfeld"], ids=["free", "sommerfeld"])
def test_loaded_posts_are_served_through_the_port_solution(ground):
    """N 0 V ports off the axis — the shape of a deck's loaded posts — are a
    1 + N port solve: every harmonic, from the one sector fill. Y, the
    columns, and the loaded input impedance all match the dense solve."""
    feeds = [FEED, *posts()]
    s, d = pair(ground=ground, feeds=feeds)
    ps, pd = s.compute_port_solution(), d.compute_port_solution()
    assert ps.y.shape == pd.y.shape == (1 + N, 1 + N)
    assert rel(ps.y, pd.y) <= Z_TOL
    assert rel(ps.coeffs, pd.coeffs) <= C_TOL
    omega = 2 * math.pi * 299792458.0 / WL
    z_load = 1 / (1j * omega * 1e-12)
    zr, zd = loaded_zin(ps.y, z_load), loaded_zin(pd.y, z_load)
    assert abs(zr - zd) / abs(zd) <= Z_TOL
    assert np.array_equal(s.compute_y_matrix(), ps.y)


def test_a_driven_orbit_is_harmonic_zero_and_a_phased_one_is_every_harmonic():
    """N feeds off the axis at one voltage are a rotation-invariant drive:
    compute_impedance serves it. Phased, the drive needs every harmonic, so
    compute_impedance still refuses it (by the momwire#1029 sentence) while
    the port solution, which drives each port alone, serves it."""
    s, d = pair(feeds=posts(1 + 0j))
    zs, _ = s.compute_impedance()
    zd, _ = d.compute_impedance()
    assert rel(zs, zd) <= Z_TOL
    phased = [(w, a, complex(1.0, 0.1 * i)) for i, (w, a, _v) in enumerate(posts())]
    s, d = pair(feeds=phased)
    with pytest.raises(RotationalSymmetryRefused) as exc:
        s.compute_impedance()
    assert "the drive is not rotation-invariant" in str(exc.value)
    assert rel(s.compute_y_matrix(), d.compute_y_matrix()) <= Z_TOL


def test_the_swept_port_entries_loop_the_per_k_route_bit_for_bit():
    s, d = pair(feeds=[FEED, *posts()])
    ks = s.k * np.array([0.95, 1.05])
    swept = list(s.compute_y_matrix_swept(ks))
    k0 = s.k
    for kk, y in zip(ks, swept, strict=True):
        s._set_k(float(kk))
        assert np.array_equal(y, s.compute_y_matrix())
    s._set_k(k0)
    dense = list(d.compute_y_matrix_swept(ks))
    assert max(rel(a, b) for a, b in zip(swept, dense, strict=True)) <= Z_TOL


@pytest.mark.parametrize("ground", [None, "sommerfeld"], ids=["free", "sommerfeld"])
def test_members_drawn_the_other_way_are_read_and_solved(ground):
    """A rotated copy that runs backwards is the same wire (momwire#1897:
    1,746 of sym4's 3,796 off-axis wires). Reverse sector 1's vertical and
    sector 2's chord and spoke; the route signs those bases and still
    matches the dense solve, through compute_impedance and the ports."""
    wires, npe = cage(lift=LIFT if ground else 0.0)
    for i in (3, 2 + N + 2, 2 + 2 * N + 2):
        wires[i] = wires[i][::-1].copy()
        npe[i] = npe[i][::-1]
    s, d = pair(ground=ground, wires=wires, npe=npe)
    assert s._rotational_map.reversed == (
        (False, False, False),
        (True, False, False),
        (False, True, True),
        (False, False, False),
    )
    zs, cs = s.compute_impedance()
    zd, cd = d.compute_impedance()
    assert abs(zs - zd) / abs(zd) <= Z_TOL
    assert rel(cs, cd) <= C_TOL
    # the post on the reversed vertical is measured from its other end
    span = H - FOOT
    feeds = [FEED] + [(2 + i, span - 0.2 if i == 1 else 0.2, 0j) for i in range(N)]
    s, d = pair(ground=ground, wires=wires, npe=npe, feeds=feeds)
    ps, pd = s.compute_port_solution(), d.compute_port_solution()
    assert rel(ps.y, pd.y) <= Z_TOL
    assert rel(ps.coeffs, pd.coeffs) <= C_TOL


# ----------------------------------------------------------------------
# What stays refused
# ----------------------------------------------------------------------


def test_a_lone_off_axis_port_is_refused_by_name():
    with pytest.raises(RotationalSymmetryRefused) as exc:
        build(feeds=[(2, 0.2, 1 + 0j)])
    msg = str(exc.value)
    assert "port 'feed 0' sits at (0.080, 0.000, 0.250)" in msg
    assert "its rotated images are not all ports" in msg
    assert "this route serves the axis-symmetric drive only" in msg


def test_a_cage_that_is_not_c_n_is_refused_naming_a_wire_it_strands():
    """Vertical 1 moved 3 mm: no line is an axis any more. The sentence is
    read against the line nearest the centroid (the mast's) and the N that
    strands the fewest wires (4), and names the first stranded one — the
    vertical whose image was the moved one."""
    wires, npe = cage(move=0.003)
    with pytest.raises(RotationalSymmetryRefused) as exc:
        build(wires=wires, npe=npe)
    msg = str(exc.value)
    assert msg.startswith(
        "rotational symmetry: under rotation by 2*pi/4 about the axis "
        "(0.000, 0.000), 7 of the 12 wires off it have no rotated copy; the "
        "first is wire 2, which stands on a vertical line at (0.080, 0.000)."
    )
    assert "a vertical one included, must be one of N rotated copies" in msg
    assert msg.endswith("Drop rotational_symmetry=True to solve this deck densely.")


def test_the_refusals_still_cost_no_fill(monkeypatch):
    calls = []

    def _no(self, *a, **kw):
        calls.append(1)
        raise AssertionError("filled while refusing")

    monkeypatch.setattr(BSplineSolver, "_compute_Z_operator", _no)
    for kw in (
        dict(feeds=[(2, 0.2, 1 + 0j)]),
        dict(wires=cage(move=0.003)[0], npe=cage()[1]),
    ):
        with pytest.raises(RotationalSymmetryRefused):
            build(**kw)
    assert calls == []


# ----------------------------------------------------------------------
# The constraint reduction
# ----------------------------------------------------------------------


def test_orbit_constraints_hands_an_invariant_matrix_back_as_is():
    """The byte-identity of every deck served before rests on this: with
    every junction on the axis the route's constraint matrix is the same
    object, so nothing downstream of it moved."""
    s = screen()
    geom = s._build_geometry()
    supp_seg, _p, kcl_A, _k, wbg = s._build_basis_polynomials(geom)
    sectors, _ax = s._rotational_dof_groups(wbg, supp_seg.shape[0])
    assert rs.orbit_constraints(kcl_A, sectors) is kcl_A


def test_off_axis_junctions_reduce_to_one_invariant_row_per_orbit():
    s = build()
    geom = s._build_geometry()
    supp_seg, _p, kcl_A, _k, wbg = s._build_basis_polynomials(geom)
    sectors, _ax = s._rotational_dof_groups(wbg, supp_seg.shape[0])
    red = rs.orbit_constraints(kcl_A, sectors)
    # on the axis: the stub/mast/spokes node and the mast/... none else;
    # off it: N ring-top nodes and N spoke feet, one orbit each
    on_axis = int(np.sum(rs._invariant_rows(kcl_A, sectors)))
    assert red.shape[0] == on_axis + (kcl_A.shape[0] - on_axis) // N
    assert bool(np.all(rs._invariant_rows(red, sectors)))


# ----------------------------------------------------------------------
# The case that asked for it: antennaknobs' verticals.elt_whip:sym4
# ----------------------------------------------------------------------

SYM4 = Path(__file__).parent / "fixtures" / "elt_whip_sym4_1897.json.gz"
# The dense solve of the same kwargs, momwire 0.89.1 on Skylake (avx2):
# 52.6966 - j2.6800 ohm, the number momwire#1897 quotes.
SYM4_DENSE = complex(52.69660490031302, -2.6800239511717168)


@pytest.mark.slow
def test_sym4_solves_on_the_route_and_matches_the_dense_answer():
    """3,799 wires, exactly C4v about the whip; a 12-wire cage of off-axis
    verticals, four spokes and four posts each loaded with 0.9 pF. The
    posts are ports, so this is a 5-port solve through all four harmonics;
    1,758 of the sector wires run backwards against their sector-0 copy.
    Free space, 406 MHz."""
    doc = json.load(gzip.open(SYM4, "rt"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = BSplineSolver(
            wires=[np.array(w) for w in doc["wires"]],
            n_per_edge_per_wire=doc["n_per_edge_per_wire"],
            wire_radius=doc["wire_radius"],
            feeds=[(w, a, complex(*v)) for w, a, v in doc["feeds"]],
            junctions=[[(w, e) for w, e in j] for j in doc["junctions"]],
            wavelength=doc["wavelength"],
            ground_z=None,
            exact_kernel="auto",
            rotational_symmetry=True,
        )
    m = s._rotational_map
    assert m.n_sectors == 4
    assert m.axial == (0, 5, 10)
    assert sum(map(sum, m.reversed)) == 1758
    Y = s.compute_y_matrix()
    omega = 2 * math.pi * 299792458.0 / doc["wavelength"]
    z = loaded_zin(Y, 1 / (1j * omega * doc["post_c_farad"]))
    assert abs(z - SYM4_DENSE) / abs(SYM4_DENSE) <= Z_TOL, z
    assert f"{z.real:.4f} {z.imag:.4f}" == "52.6966 -2.6800"
