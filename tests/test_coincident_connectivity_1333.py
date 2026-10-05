"""Coinciding is not enough to be a duplicate: a copy must also be JOINED like
its twin — momwire#1333, narrowing #1042.

#1042 merges a wire listed twice. Two coincident wires whose ends sit in
DIFFERENT junctions (each rise of a fan joined to its own radial alone) are
not a wire listed twice: they are what the caller wrote, so they are kept and
solved as written, exactly as before #1042, and a family that cannot solve
coincident conductors says so by name.

Coincident wires whose ends share ONE junction at each end are still merged.
That is antennaknobs' `buried_radial_vertical` `bundle` variant: its walk
joins every rise and every radial at one hub junction, so its rises are
connected exactly alike and stay merged (see the #1333 report: the as-written
bundle converges in quadrature to the merged answer).

The deck is momwire#524's fan in free space with each rise split off its
radial: radial k ends at the hub, rise k starts there, and an explicit
two-member junction joins only those two.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire._wire_spec import DuplicateWire, drop_duplicated_wires  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_crossing_serve_524 import A_WIRE, WL7, _radial_dirs  # noqa: E402

N = 4
DEPTH = 0.15

# The split fan's Z before #1042 (momwire 3ec54c79, the commit before PR
# #1328), measured on Skylake. This change reproduces both bit for bit there;
# the gate allows rounding so it holds on every platform's BLAS.
PRE_1042 = {
    BSplineSolver: 32.94864673320206 - 337.2324201391893j,
    SinusoidalSolver: 32.75416025899061 - 342.05210557941166j,
}


def fan(shared=False):
    """N radials at depth, N coincident rises to the node, a monopole above.

    `shared=False`: rise k joined to radial k alone (N hub junctions).
    `shared=True`: one hub junction for every radial and rise, the way
    antennaknobs' polyline walk joins the `bundle` variant's coincident ends.
    """
    dirs = _radial_dirs(N)
    hub = (0.0, 0.0, -DEPTH)
    wires = [np.array([(5.0 * dx, 5.0 * dy, -DEPTH), hub]) for dx, dy in dirs]
    wires += [np.array([hub, (0.0, 0.0, 0.0)]) for _ in dirs]
    mono = len(wires)
    wires.append(np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)]))
    if shared:
        hubs = [[(k, "end") for k in range(N)] + [(N + k, "start") for k in range(N)]]
    else:
        hubs = [[(k, "end"), (N + k, "start")] for k in range(N)]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[10]] * N + [[2]] * N + [[15]],
        junctions=hubs + [[(N + k, "end") for k in range(N)] + [(mono, "end")]],
        feeds=[(mono, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
    )


def one_rise():
    """`fan(shared=True)` with rises 1..N-1 deleted by hand."""
    d = fan(shared=True)
    d["wires"] = d["wires"][: N + 1] + d["wires"][-1:]
    d["n_per_edge_per_wire"] = [[10]] * N + [[2]] + [[15]]
    d["junctions"] = [
        [(k, "end") for k in range(N)] + [(N, "start")],
        [(N, "end"), (N + 1, "end")],
    ]
    d["feeds"] = [(N + 1, 4.3333333333, 1 + 0j)]
    return d


def _build(cls, **deck):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = cls(**deck)
    return s, [w for w in caught if issubclass(w.category, DuplicateWire)]


@pytest.mark.parametrize("cls", list(PRE_1042), ids=lambda c: c.__name__)
def test_rises_joined_to_their_own_radials_solve_as_written(cls):
    s, found = _build(cls, **fan())
    assert found == [] and s._wire_dedup is None
    assert len(s.wires_polylines) == 2 * N + 1
    z, _coeffs = s.compute_impedance()
    assert complex(z) == pytest.approx(PRE_1042[cls], rel=1e-9, abs=0)


def test_the_galerkin_family_refuses_them_by_name():
    with pytest.raises(ValueError) as exc:
        SinusoidalGalerkinSolver(**fan())
    msg = str(exc.value)
    assert "wires 4 and 5 coincide" in msg and "joined to different wires" in msg
    assert "cannot solve coincident conductors" in msg and "momwire#1333" in msg


def test_razor_refuses_them_at_the_solve_by_its_bundle_sentence():
    s, found = _build(RazorSolver, **fan())
    assert found == [] and s._wire_dedup is None
    with pytest.raises(ValueError, match="geometrically COINCIDENT segments"):
        s.compute_impedance()


@pytest.mark.parametrize(
    "cls",
    [BSplineSolver, SinusoidalSolver, SinusoidalGalerkinSolver],
    ids=lambda c: c.__name__,
)
def test_rises_sharing_one_hub_are_still_merged(cls):
    """antennaknobs' `bundle` shape: every rise in one hub junction and one
    node junction, i.e. connected exactly alike. Solved as the deck with the
    extra rises deleted, input for input."""
    s, found = _build(cls, **fan(shared=True))
    assert len(found) == 1 and "3 wire(s) listed twice" in str(found[0].message)
    assert s._wire_dedup.removed == ((5, 4, False), (6, 4, False), (7, 4, False))
    ref, none = _build(cls, **one_rise())
    assert none == []
    z, coeffs = s.compute_impedance()
    z_ref, coeffs_ref = ref.compute_impedance()
    assert z == z_ref
    assert np.array_equal(coeffs, coeffs_ref)


def test_each_copy_merges_into_the_first_wire_joined_like_it():
    """Four copies of one path, two at each of two hub junctions: each pair
    merges within itself, and the two survivors are kept as written."""
    p, q = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    wires = [np.array([p, q]) for _ in range(4)]
    wires += [
        np.array([(1.0, 0.0, 0.0), p]),
        np.array([(-1.0, 0.0, 0.0), p]),
        np.array([q, (0.0, 0.0, 2.0)]),
    ]
    junctions = [
        [(4, "end"), (0, "start"), (1, "start")],
        [(5, "end"), (2, "start"), (3, "start")],
        [(0, "end"), (1, "end"), (2, "end"), (3, "end"), (6, "start")],
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kept, _npe, _pw, _sites, out_junctions, dedup = drop_duplicated_wires(
            "X", wires, None, junctions=junctions
        )
    assert dedup.removed == ((1, 0, False), (3, 2, False))
    assert dedup.kept == (0, 2, 4, 5, 6)
    assert len(kept) == 5
    assert out_junctions == [
        [(2, "end"), (0, "start")],
        [(3, "end"), (1, "start")],
        [(0, "end"), (1, "end"), (4, "start")],
    ]
    assert len([w for w in caught if issubclass(w.category, DuplicateWire)]) == 1


def test_a_reversed_copy_is_compared_end_for_end():
    """A copy run backwards is joined like its twin when its END sits where
    the twin's START does; free ends match free ends."""
    p, q = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    wires = [np.array([p, q]), np.array([q, p]), np.array([(1.0, 0.0, 0.0), p])]
    same = [[(2, "end"), (0, "start"), (1, "end")]]
    *_rest, dedup = drop_duplicated_wires("X", wires, None, junctions=same)
    assert dedup.removed == ((1, 0, True),)
    *_rest, dedup = drop_duplicated_wires(
        "X", wires, None, junctions=[[(2, "end"), (0, "start")]]
    )
    assert dedup is None  # wire 1's end is free where wire 0's start is joined
