"""momwire#667: the deck seams adopt the crossing serve.

The native API has served current across the ground interface since
momwire#524 phase 2, through a declared crossing junction: a wire wholly
below the plane ENDING in it, sharing that node with a wire starting there
and rising above it. Both deck seams refused every crossing deck, including
that spelling — the EZNEC seam because its per-wire check read an end in the
plane as "touching both sides", the NEC-2 portal because the chaining fused
the two cards' degree-2 node into one polyline with points on both sides,
which the solver refuses by name.

Now: the chaining ends a polyline at a degree-2 node in the plane between
media and declares it a junction; a straight card wire that crosses the
plane mid-span is split where its line meets the plane (exact for a straight
wire) with its element count shared by length; and the EZNEC seam passes the
split spelling to the serve-time check that already knew how to read it.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire.deck._polylines import split_at_plane, to_polylines
from momwire.deck.model import DeckModel, DeckWire
from momwire.portal._portal import main as portal_main

SOIL = (13.0, 0.005)
F_MHZ = 7.1
WL = 299.792458 / F_MHZ
RADIUS = 0.001

# The geometry: a 0.3 m rise from a buried hub to the surface node, a 10.556 m
# radiator above it, two 6.334 m radials at the hub. Counts chosen so that
# the ONE-wire spelling (24 elements over 10.856 m) splits at the plane into
# the same 1 + 23 the two-wire spelling below is written with — same knots,
# same matrix, so the two decks must agree to the solver's own roundoff.
RISE, MAST, RADIAL = 0.3, 10.556, 6.334
N_RISE, N_MAST, N_RADIAL = 1, 23, 6


def _nec2(single: bool, ge: str = "GE 1") -> str:
    if single:
        wires = f"GW 1 {N_RISE + N_MAST} 0. 0. -{RISE} 0. 0. {MAST} {RADIUS}\n"
        feed = f"EX 0 1 {N_RISE + 1} 0 1. 0.\n"  # first segment above the plane
    else:
        wires = (
            f"GW 1 {N_RISE} 0. 0. -{RISE} 0. 0. 0. {RADIUS}\n"
            f"GW 2 {N_MAST} 0. 0. 0. 0. 0. {MAST} {RADIUS}\n"
        )
        feed = "EX 0 2 1 0 1. 0.\n"
    return (
        "CE crossing\n"
        + wires
        + f"GW 3 {N_RADIAL} 0. 0. -{RISE} {RADIAL} 0. -{RISE} {RADIUS}\n"
        + f"GW 4 {N_RADIAL} 0. 0. -{RISE} -{RADIAL} 0. -{RISE} {RADIUS}\n"
        + f"{ge}\n"
        + f"GN 2 0 0 0 {SOIL[0]} {SOIL[1]}\n"
        + f"FR 0 1 0 0 {F_MHZ} 0.\n"
        + feed
        + "XQ 0\nEN\n"
    )


def _portal(deck: str) -> str:
    out = io.StringIO()
    code = portal_main([], io.StringIO(deck), out, io.StringIO())
    assert code == 0, out.getvalue()[-600:]
    return out.getvalue()


_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+" + r"([-+.\dE]+)\s+" * 8 + r"[-+.\dE]+", re.M)


def _printed_z(printout: str) -> complex:
    assert "ERROR" not in printout, printout[-800:]
    i = printout.index("ANTENNA INPUT PARAMETERS")
    m = _ROW.search(printout, i)
    assert m, printout[i : i + 800]
    return complex(float(m.group(7)), float(m.group(8)))


def _native() -> complex:
    below = np.array([(0.0, 0.0, -RISE), (0.0, 0.0, 0.0)])
    above = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, MAST)])
    r1 = np.array([(0.0, 0.0, -RISE), (RADIAL, 0.0, -RISE)])
    r2 = np.array([(0.0, 0.0, -RISE), (-RADIAL, 0.0, -RISE)])
    s = BSplineSolver(
        wires=[below, above, r1, r2],
        n_per_edge_per_wire=[[N_RISE], [N_MAST], [N_RADIAL], [N_RADIAL]],
        junctions=[
            [(0, "end"), (1, "start")],
            [(0, "start"), (2, "start"), (3, "start")],
        ],
        feeds=[(1, 0.5 * MAST / N_MAST, 1 + 0j)],
        wavelength=WL,
        wire_radius=RADIUS,
        ground_z=0.0,
        ground_eps=SOIL,
        ground_model="sommerfeld",
    )
    z, _ = s.compute_impedance()
    return complex(z)


# --- the model layer ---------------------------------------------------------


def _model(vertices, elements, *, ground=("finite", 13.0, 0.005), node_gaps=()):
    return DeckModel(
        wires=(
            DeckWire(
                vertices=tuple(vertices), radius=RADIUS, edge_elements=tuple(elements)
            ),
        ),
        ground=ground,
        ground_z=0.0,
        node_gaps=tuple(node_gaps),
    )


def test_split_at_plane_leaves_free_space_and_non_crossing_models_untouched():
    m = _model([(0, 0, -1.0), (0, 0, 2.0)], [6], ground=None)
    assert split_at_plane(m) is m
    m = _model([(0, 0, 0.5), (0, 0, 2.0)], [6])
    assert split_at_plane(m) is m
    m = _model([(0, 0, -1.0), (0, 0, 0.0)], [6])  # ends IN the plane: not a crossing
    assert split_at_plane(m) is m


def test_split_at_plane_inserts_the_exact_crossing_and_shares_the_count():
    m = _model([(0, 0, -1.0), (0, 0, 3.0)], [8])
    out = split_at_plane(m)
    w = out.wires[0]
    assert w.vertices == ((0, 0, -1.0), (0.0, 0.0, 0.0), (0, 0, 3.0))
    assert w.edge_elements == (2, 6)  # 8 * 0.25 = 2 below, the rest above
    # a slanted wire: the crossing is where the LINE meets the plane
    m = _model([(0, 0, -1.0), (4.0, 0, 1.0)], [1])
    w = split_at_plane(m).wires[0]
    assert w.vertices[1] == pytest.approx((2.0, 0.0, 0.0))
    assert w.edge_elements == (1, 1)  # one element cannot be shared: two


def test_split_at_plane_shifts_node_gaps_after_the_inserted_vertex():
    m = _model(
        [(0, 0, -1.0), (0, 0, 3.0), (0, 0, 5.0)],
        [8, 4],
        node_gaps=[(0, 0, 1 + 0j), (0, 1, 1 + 0j), (0, 2, 1 + 0j)],
    )
    out = split_at_plane(m)
    assert out.node_gaps == ((0, 0, 1 + 0j), (0, 2, 1 + 0j), (0, 3, 1 + 0j))


def test_the_plane_node_ends_the_polylines_and_is_a_junction():
    m = DeckModel(
        wires=(
            DeckWire(
                vertices=((0, 0, -1.0), (0, 0, 0.0)), radius=RADIUS, edge_elements=(4,)
            ),
            DeckWire(
                vertices=((0, 0, 0.0), (0, 0, 5.0)), radius=RADIUS, edge_elements=(10,)
            ),
        ),
        ground=("finite", 13.0, 0.005),
        ground_z=0.0,
    )
    mesh = to_polylines(m, ())
    assert len(mesh.polylines) == 2
    assert mesh.junctions == (((0, "end"), (1, "start")),)
    # the same two cards in free space chain into ONE polyline, as before
    free = to_polylines(DeckModel(wires=m.wires, ground=None), ())
    assert len(free.polylines) == 1 and free.junctions == ()


# --- the NEC-2 portal --------------------------------------------------------


@pytest.mark.slow
def test_portal_serves_the_split_spelling_and_matches_the_native_api(record_property):
    z = _printed_z(_portal(_nec2(single=False)))
    ref = _native()
    record_property("z_portal_split", f"{z:.6f}")
    record_property("z_native", f"{ref:.6f}")
    # The printout carries five significant digits per component, so the
    # comparison is to the print's own precision and no tighter.
    assert abs(z - ref) <= 1e-4 * abs(ref)


@pytest.mark.slow
def test_portal_splits_one_wire_at_the_plane_and_matches_the_split_spelling(
    record_property,
):
    z_one = _printed_z(_portal(_nec2(single=True)))
    z_two = _printed_z(_portal(_nec2(single=False)))
    record_property("z_portal_single", f"{z_one:.6f}")
    assert abs(z_one - z_two) <= 1e-6 * abs(z_two) + 1e-6


@pytest.mark.slow
def test_portal_ge_minus_one_still_wants_the_interpolated_card():
    """Unchanged: the portal serves the ground contact only under GE 1, and
    says so by name — the split adds nothing under GE -1."""
    out = io.StringIO()
    portal_main([], io.StringIO(_nec2(single=False, ge="GE -1")), out, io.StringIO())
    assert "write GE 1" in out.getvalue()
