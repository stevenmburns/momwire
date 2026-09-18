"""momwire#1029 phase 1: the opt-in sector (block-circulant) route.

The deck is `antennaknobs`' `verticals.buried_radial_vertical` in miniature —
N radials to a buried hub, one rise to the crossing node at z = 0, one mast
above it — spelled here in plain momwire so the gate needs no consumer
installed (the momwire#1092 rule). The mesh is coarse on purpose: what is
under test is the ROUTE, and every gate below is the route against the dense
solve on the SAME deck, so a coarse mesh weakens nothing.

What is gated:

* the check reads the screen as N sectors about the axis (`sector_map`);
* **G2** — the six registered refusals of `PLAN-phase1.md` §3, one deck each,
  each raising AT CONSTRUCTION and naming its own condition;
* **G1 at 4 radials** — Z_in and the currents against the dense solve
  (`|dZ|/|Z| <= 1e-9`, `max|dc|/max|c| <= 1e-8`). The 48- and 150-radial
  rungs are scratch records, not tests: they cost minutes;
* **G4's half that is about code** — the flag off runs no route code, and
  every entry point but `compute_impedance` refuses rather than quietly
  handing back the dense answer.

Phase 2b adds two more:

* **the ROSTER** — `solve_strategy` carries "sector" beside "dense", dense
  first, and the accelerated subclasses do not gain it;
* **the SWEPT entry** — `compute_impedance_swept` serves the route instead of
  refusing, returning the dense path's own contract, and every refusal still
  lands before a fill.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from momwire import ArrayBlockSolver, BSplineSolver, HMatrixSolver, axes_for
from momwire._capabilities import AXIS_VALUES
from momwire._rotational_symmetry import RotationalSymmetryRefused

WL = 42.2  # 40 m, the design's own band
RADIAL = 6.334
DEPTH = 0.15
MAST = 10.556
SOIL = (13.0, 0.005)
GROUND = dict(ground_z=0.0, ground_eps=SOIL, ground_model="sommerfeld")


def screen(n_radials=4, radial=RADIAL, azimuths=None, tilt=0.0):
    """The wires: N radials from the hub, the rise, the mast.

    `radial` may be a per-radial sequence (one arm longer), `azimuths` may
    replace the equal spacing, and `tilt` leans the mast off z by that many
    metres at its top.
    """
    lengths = [radial] * n_radials if np.isscalar(radial) else list(radial)
    if azimuths is None:
        azimuths = [2 * math.pi * i / n_radials for i in range(n_radials)]
    wires = []
    for r, th in zip(lengths, azimuths):
        c, s = math.cos(th), math.sin(th)
        wires.append(
            np.array(
                [
                    (0.0, 0.0, -DEPTH),
                    (
                        r * (0.0 if abs(c) < 1e-15 else c),
                        r * (0.0 if abs(s) < 1e-15 else s),
                        -DEPTH,
                    ),
                ]
            )
        )
    wires.append(np.array([(0.0, 0.0, -DEPTH), (0.0, 0.0, 0.0)]))  # the rise
    wires.append(np.array([(0.0, 0.0, 0.0), (tilt, 0.0, MAST)]))  # the mast
    return wires


def solver(n_radials=4, *, rotational_symmetry=True, wires=None, **kw):
    wires = screen(n_radials) if wires is None else wires
    n_w = len(wires)
    npe = [[6]] * (n_w - 2) + [[3], [8]]
    kwargs = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        degree=2,
        wavelength=WL,
        wire_radius=0.001,
        feeds=[(n_w - 1, 0.5 * MAST / 8, 1 + 0j)],
        rotational_symmetry=rotational_symmetry,
        **GROUND,
    )
    kwargs.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(**kwargs)


def refusal(**kw):
    """The refusal a deck raises at construction, as text."""
    with pytest.raises(RotationalSymmetryRefused) as exc:
        solver(**kw)
    return str(exc.value)


# ----------------------------------------------------------------------
# The check
# ----------------------------------------------------------------------


def test_the_check_reads_the_screen_as_n_sectors_about_the_axis():
    s = solver(4)
    smap = s._rotational_map
    assert smap.axis == (0.0, 0.0)
    assert smap.n_sectors == 4
    # one wire per sector, and the same wire of each sector at position 0
    assert smap.sectors == ((0,), (1,), (2,), (3,))
    # the rise and the mast are the rotation's fixed wires
    assert smap.axial == (4, 5)
    assert smap.tol == pytest.approx(1e-9 * MAST)


def test_the_sector_labelling_is_free_but_the_position_is_not():
    """Any transversal of the orbits is a fundamental domain: the route sums
    a row over a source dof's whole orbit, so relabelling cannot move a
    number. What pairs a row with its images is the POSITION, and S-0's
    bijection is what makes position meaningful — so this asserts the dof
    groups partition the basis exactly, with equal-size sectors."""
    s = solver(6)
    geom = s._build_geometry()
    supp_seg, _polys, _kcl, _knots, wbg = s._build_basis_polynomials(geom)
    n_b = supp_seg.shape[0]
    sectors, axial = s._rotational_dof_groups(wbg, n_b)
    assert len({len(x) for x in sectors}) == 1
    joined = np.concatenate([*sectors, axial])
    assert sorted(joined.tolist()) == list(range(n_b))
    assert len(sectors) * len(sectors[0]) + len(axial) == n_b


def test_the_requested_rows_are_whole_wires():
    """The fill restriction is by SEGMENT, and it stands for a dof row set
    only because no basis straddles two wires (gate S-0's S0a). So the
    requested segments are exactly sector 0's wires plus the axial ones."""
    s = solver(4)
    geom = s._build_geometry()
    rows = s._rotational_rows(geom)
    off = geom["seg_offsets"]
    want = []
    for w in (0, 4, 5):
        want.extend(range(off[w], off[w] + geom["per_wire"][w]["n_total"]))
    assert rows.tolist() == sorted(want)


# ----------------------------------------------------------------------
# G2 — the six refusals, one deck each, each by name
# ----------------------------------------------------------------------


def test_g2_1_a_longer_radial_names_the_sector_and_the_percentage():
    lengths = [RADIAL, RADIAL, RADIAL * 1.01, RADIAL]
    msg = refusal(wires=screen(4, radial=lengths))
    assert msg.startswith("rotational symmetry: sector 2's wire is +1.00 % longer")
    assert f"{RADIAL * 1.01:.3f} m against {RADIAL:.3f} m" in msg
    assert "map onto the next under rotation by 2*pi/N" in msg
    assert "Drop rotational_symmetry=True" in msg


def test_g2_2_a_radial_at_the_wrong_azimuth_names_the_angle_it_wanted():
    az = [0.0, 0.5 * math.pi, math.pi, math.radians(272.5)]
    msg = refusal(wires=screen(4, azimuths=az))
    assert "sector 3 sits at 272.500 deg, where 4 sectors require 270.000 deg" in msg
    assert "tolerance 1e-09 x 10.6 m" in msg


def test_g2_3_an_off_axis_port_names_where_it_sits():
    msg = refusal(feeds=[(0, 3.0, 1 + 0j)])
    assert "port 'feed 0' sits at (3.000, 0.000, -0.150)" in msg
    assert "off the symmetry axis (0.000, 0.000)" in msg
    assert "this route serves the axis-symmetric drive only" in msg


def test_g2_4_a_thicker_radial_names_both_radii():
    a = [0.001] * 6
    a[1] = 0.0008
    msg = refusal(wire_radius=a)
    assert "sector 1's conductor radius is 0.800 mm against sector 0's 1.000 mm" in msg
    assert "the same radius, conductivity and jacket" in msg


def test_g2_5_a_ground_that_is_not_axisymmetric_names_the_model():
    """The seam, not a hypothetical. Every ground `BSplineSolver` can express
    is invariant under rotation about a vertical axis (the next test asserts
    that), so the only way to reach this sentence today is a family that
    answers with a name the whitelist does not carry — which is exactly what
    a terrain or two-media ground would do."""

    class _TerrainSolver(BSplineSolver):
        def _rotational_ground_kind(self):
            return "terrain"

    with pytest.raises(RotationalSymmetryRefused) as exc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wires = screen(4)
            _TerrainSolver(
                wires=wires,
                n_per_edge_per_wire=[[6]] * 4 + [[3], [8]],
                degree=2,
                wavelength=WL,
                wire_radius=0.001,
                feeds=[(5, 0.5 * MAST / 8, 1 + 0j)],
                rotational_symmetry=True,
                **GROUND,
            )
    msg = str(exc.value)
    assert "the ground model 'terrain' is not invariant under rotation" in msg
    assert "Sommerfeld half-space qualify" in msg


def test_g2_5_every_ground_this_family_can_express_qualifies():
    """The other half of the seam: the whitelist is not a way of refusing
    grounds this solver actually serves."""
    from momwire._rotational_symmetry import AXISYMMETRIC_GROUNDS

    assert solver(4)._rotational_ground_kind() == "sommerfeld"
    for kw, want in (
        (dict(ground_z=None, ground_eps=None, ground_model="refl-coef"), "free"),
        (dict(ground_z=0.0, ground_eps=None, ground_model="refl-coef"), "pec"),
        (dict(ground_z=0.0, ground_eps=SOIL, ground_model="refl-coef"), "refl-coef"),
    ):
        probe = BSplineSolver.__new__(BSplineSolver)
        for k, v in kw.items():
            setattr(probe, k, v)
        assert probe._rotational_ground_kind() == want
        assert want in AXISYMMETRIC_GROUNDS


def test_g2_6_a_tilted_mast_names_the_wire_and_its_angle():
    """The mast is still the mast: it reaches the axial group across the
    degree-2 node junction, and then fails the group's own rule. Without that
    growth it would read as a seventh sector and refuse with the wrong
    sentence."""
    msg = refusal(wires=screen(4, tilt=0.04))
    assert "the axial group's wires are not parallel to z (wire 5 runs" in msg
    assert "deg off)" in msg
    assert "The symmetry axis must be normal to the interface" in msg
    angle = math.degrees(math.atan2(0.04, MAST))
    assert f"{angle:.3f} deg off" in msg


# ----------------------------------------------------------------------
# Scope: what the route says it does not serve
# ----------------------------------------------------------------------


def test_one_radial_is_not_a_screen():
    msg = refusal(wires=screen(1))
    assert "the route needs N >= 2 sectors" in msg


def test_an_unburied_deck_is_out_of_scope():
    above = [
        np.array([(0.0, 0.0, 1.0), (6.0, 0.0, 1.0)]),
        np.array([(0.0, 0.0, 1.0), (-6.0, 0.0, 1.0)]),
        np.array([(0.0, 0.0, 1.0), (0.0, 0.0, 11.0)]),
    ]
    with pytest.raises(RotationalSymmetryRefused) as exc:
        BSplineSolver(
            wires=above,
            n_per_edge_per_wire=[[6], [6], [8]],
            degree=2,
            wavelength=WL,
            wire_radius=0.001,
            feeds=[(2, 0.5, 1 + 0j)],
            rotational_symmetry=True,
            **GROUND,
        )
    assert "has no wire below the interface" in str(exc.value)


def test_singular_enrichment_is_out_of_scope():
    msg = refusal(use_singular_enrichment=True)
    assert "singular enrichment adds a block that is not sector-structured" in msg


@pytest.mark.parametrize(
    "entry", ["compute_y_matrix", "compute_port_solution", "compute_impedance_swept"]
)
def test_every_other_entry_point_refuses_rather_than_answering_densely(entry):
    s = solver(4)
    args = ([np.array([s.k])],) if entry == "compute_impedance_swept" else ()
    with pytest.raises(RotationalSymmetryRefused) as exc:
        getattr(s, entry)(*(a for a in args))
    assert entry in str(exc.value)
    assert "compute_impedance only" in str(exc.value)


def test_a_drive_that_is_not_rotation_invariant_is_refused():
    """The route builds harmonic 0 and nothing else, so a right-hand side
    with content anywhere else has no block to go through. The two the route
    actually sees are exactly invariant; this drives the guard directly."""
    s = solver(4)
    geom = s._build_geometry()
    supp_seg, polys, kcl_A, _knots, wbg = s._build_basis_polynomials(geom)
    n_b = supp_seg.shape[0]
    sectors, axial = s._rotational_dof_groups(wbg, n_b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = s._compute_Z_operator_buried(
            geom, supp_seg, polys, rows=s._rotational_rows(geom)
        )
    v = np.zeros(n_b, dtype=np.complex128)
    v[sectors[1][0]] = 1.0  # one sector driven, the others not
    with pytest.raises(RotationalSymmetryRefused) as exc:
        s._rotational_solve(Z, v, kcl_A[:0], sectors, axial)
    assert "the drive is not rotation-invariant" in str(exc.value)


# ----------------------------------------------------------------------
# G1 at 4 radials, and G4's code half
# ----------------------------------------------------------------------


def test_the_flag_off_runs_no_route_code():
    s = solver(4, rotational_symmetry=False)
    assert s.rotational_symmetry is False
    assert s._rotational_map is None
    # and the other entry points are untouched
    s._rotational_route_serves_one_drive("compute_y_matrix")


def test_g1_the_route_matches_the_dense_solve_at_four_radials():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_d, c_d = solver(4, rotational_symmetry=False).compute_impedance()
        route = solver(4)
        z_r, c_r = route.compute_impedance()
    z_d = complex(np.atleast_1d(z_d)[0])
    z_r = complex(np.atleast_1d(z_r)[0])
    assert abs(z_r - z_d) / abs(z_d) <= 1e-9
    assert np.max(np.abs(c_r - c_d)) / np.max(np.abs(c_d)) <= 1e-8
    # the N axis-against-sector copies agree, which is the free check on the
    # sector assignment PLAN-phase1 §4 registers (never bit-identical)
    assert route._rotational_copy_spread <= 1e-12


def test_g1_the_route_still_matches_with_more_sectors():
    """Six sectors, so N does not divide into the four-fold case's answer by
    accident and the DFT's bookkeeping is exercised at another N."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_d, c_d = solver(6, rotational_symmetry=False).compute_impedance()
        z_r, c_r = solver(6).compute_impedance()
    z_d = complex(np.atleast_1d(z_d)[0])
    z_r = complex(np.atleast_1d(z_r)[0])
    assert abs(z_r - z_d) / abs(z_d) <= 1e-9
    assert np.max(np.abs(c_r - c_d)) / np.max(np.abs(c_d)) <= 1e-8


# ----------------------------------------------------------------------
# The roster (momwire#1029 phase 2b). The route is opt-in through a
# constructor kwarg, so nothing downstream can discover it by name: a
# consumer offering the flag has to read it off the capability row, which
# is what these gate.
# ----------------------------------------------------------------------


def test_the_row_declares_the_sector_route_dense_first():
    """`solve_strategy` gains "sector" beside "dense", in that order.

    DEFAULT FIRST is the row's convention (`tests/test_feed_model_row.py`):
    the first declared value must be what the constructor picks when nothing
    is passed, so a consumer reading the row for "what does this solver do by
    default" gets the right answer. Measured against the constructor rather
    than asserted about the literal.
    """
    assert BSplineSolver.capabilities.axes["solve_strategy"] == ("dense", "sector")
    assert solver(4, rotational_symmetry=False)._rotational_map is None
    assert solver(4)._rotational_map is not None


def test_the_sector_value_is_in_the_vocabulary_and_reaches_axes_for():
    """`AXIS_VALUES` is the written-down spelling and `axes_for` is the one
    place a consumer reads a row's axes from — a value declared on the row
    but missing from either is a value nothing downstream can render."""
    assert "sector" in AXIS_VALUES["solve_strategy"]
    assert axes_for(BSplineSolver.capabilities)["solve_strategy"] == frozenset(
        {"dense", "sector"}
    )


def test_the_accelerated_rows_do_not_gain_the_sector_route():
    """Both subclasses REPLACE the cell rather than extending it, and both
    declare `buried=False` — the route fills through the buried mixed-medium
    path, which is exactly what their own row refuses. A row that inherited
    "sector" here would publish a cell neither class reaches."""
    for cls, want in (
        (HMatrixSolver, ("aca",)),
        (ArrayBlockSolver, ("element-block",)),
    ):
        assert cls.capabilities.axes["solve_strategy"] == want
        assert cls.capabilities.buried is False
        assert cls.capabilities.refusal("buried") is not None
