"""momwire#1139: EZNEC's virtual wire is not geometry, and stops acting like it.

The readout half (momwire#1136) moved a gyrator-driven source's ROW to the
node it drives. This is the geometry half: the parked wire stops being a
polyline at all, so it no longer puts a radius in the crossing serve's
one-radius-per-side census (antennaknobs plan U5) and no longer sizes the
Sommerfeld grid. Its addressed nodes become circuit nodes with no momwire
port, which is what antennaknobs' ``PortVirtual`` already made of them.

The line is drawn at the MESH and not at :func:`~momwire.eznec._serve.
structure_of`: the four STRUCTURE SPECIFICATION counts and the current and
charge tables are the ENGINE's, which solves the phantom as real geometry and
prints a row per element, so the rows stay and carry an exact zero.

Six gates, and the first four fail with the production change reverted — the
revert lever is ``_phantom_tags`` answering nothing, which is where this
seam read the idiom before #1139 and reads it now:

1. the radius census, on WA7ARK's ground-rod deck;
2. the Sommerfeld grid extent, on the same deck;
3. the phantom carries no current and no charge, over the whole corpus;
4. an addressed phantom node reaches no momwire port;
5. the deck that motivated the issue SERVES, and its driving point is the
   one its transformer asks for;
6. what must NOT move: the counts and the addresses are still the deck's own.

A seventh gate at the bottom belongs to a DIFFERENT defect, found while
measuring gate 5 and named there.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver, _sommerfeld
from momwire.deck._nec5 import parse_nec5
from momwire.deck._solver import basis_entry
from momwire.eznec import _serve, serve

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_phantom_1139"
CORPUS = Path(__file__).parent / "fixtures" / "eznec"

# WA7ARK's 40 m choked-coax EFHW over a ground rod, as EZNEC wrote it. Its
# GW 7 is the phantom: a 2-segment wire 0.145 m long parked at (4192.9,
# 4192.9, 4234.8), carrying the deck's `EX 4` and one end of the `NT`
# transformer that feeds the antenna at tag 1 node 95.
WA7ARK = "ground_rod_efhw_wa7ark.nec"

# The radius GW 7 contributes and the antenna's own three, in metres. The
# phantom's is not the largest or the smallest of the four, so no rule about
# extremes can stand in for recognising it.
PHANTOM_RADIUS = 0.0041929
ANTENNA_RADII = (0.0008138, 0.002794, 0.0079375)


def wa7ark_deck():
    return parse_nec5((FIXTURES / WA7ARK).read_bytes().decode("latin-1"))


def built(deck):
    """``(mesh, solver)`` for a deck, the way :func:`serve` builds them."""
    solver_class, basis_kwargs = basis_entry(_serve.BASIS)
    structure = _serve.structure_of(deck)
    mesh = _serve.build_mesh(
        deck,
        structure,
        solver_class=solver_class,
        crossing=bool(_serve._crossing_nodes(deck)),
    )
    wavelength = _serve.SPEED_OF_LIGHT_MHZ_M / deck.frequency_mhz
    with warnings.catch_warnings():
        # The deck's own advisories (a near-ground conductor, a coarse
        # crossing node) are momwire#865's and momwire#696's subject, not
        # this file's.
        warnings.simplefilter("ignore")
        solver = _serve._solver_for(
            deck,
            mesh,
            wavelength,
            _serve._medium(deck.ground, wavelength),
            solver_class,
            basis_kwargs,
        )
    return mesh, solver


def disarm(monkeypatch):
    """The production change reverted, and nothing else with it.

    Every consequence in this file follows from one function answering, so
    making it answer nothing is the whole revert: the phantom becomes a
    polyline again, its radius rejoins the census, its position rejoins the
    grid and its nodes become wire ports.
    """
    monkeypatch.setattr(_serve, "_phantom_tags", lambda deck, wavelength: frozenset())


# The 23 committed captures that park a phantom, and which tag it is —
# WRITTEN DOWN rather than asked of ``_phantom_tags``, because a gate that
# took its own subject from the detector would pass by finding nothing the
# moment the detector went quiet, which is exactly the revert.  The census
# that keeps this table honest is the detector's own
# (``test_eznec_gyrator_1134.py`` gate 5); the assertion below only checks the
# two agree here.
PHANTOM_DECKS = {
    "0000": 3,
    "0001": 5,
    "0002": 5,
    "0003": 5,
    "0004": 5,
    "0005": 5,
    "0006": 5,
    "0007": 5,
    "0008": 5,
    "0009": 5,
    "0012": 4,
    "0014": 4,
    "0016": 4,
    "0017": 4,
    "0018": 4,
    "0023": 5,
    "0024": 5,
    "0025": 5,
    "0026": 3,
    "0027": 3,
    "0028": 6,
    "0120": 3,
    "0121": 3,
}


def phantom_corpus():
    """Every committed capture that parks a phantom, with its tag."""
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    by_id = {entry["id"]: entry["deck"] for entry in manifest["captures"]}
    for cid, tag in PHANTOM_DECKS.items():
        yield cid, parse_nec5((CORPUS / by_id[cid]).read_bytes().decode("latin-1")), tag


# --------------------------------------------------------------------------
# gate 1 - the radius census
# --------------------------------------------------------------------------


def test_the_phantom_s_radius_is_not_in_the_census_the_solver_is_handed():
    """The first harm the issue names, as the number the solver sees.

    ``_below_interface.crossing_side_radii`` refuses a crossing deck whose
    ABOVE wires carry more than one radius, and falls back to that whole-side
    rule whenever a deck has other than exactly one crossing node
    (momwire#1140).  WA7ARK's deck has three authored above radii and the
    third is GW 7's, which is 100 lambda away and part of no antenna.  So the
    census is gated as a SET rather than as a refusal: the refusal depends on
    which node the deck happens to have, and the wrong radius being in the
    list at all does not.
    """
    mesh, solver = built(wa7ark_deck())
    radii = {piece.radius for piece in mesh.pieces}
    assert radii == set(ANTENNA_RADII)
    assert PHANTOM_RADIUS not in radii
    # And the same list as the solver actually holds it, not only as the mesh
    # spells it: `_solver_for` collapses a uniform census to one scalar, so
    # asking the solver is asking what the fill will use.
    assert set(np.atleast_1d(solver.wire_radius)) == set(ANTENNA_RADII)


def test_the_phantom_s_radius_IS_in_the_census_with_the_change_reverted(monkeypatch):
    """Gate 1's own control: the third radius, and where it comes from."""
    disarm(monkeypatch)
    mesh, _solver = built(wa7ark_deck())
    radii = {piece.radius for piece in mesh.pieces}
    assert radii == set(ANTENNA_RADII) | {PHANTOM_RADIUS}
    assert next(p.tag for p in mesh.pieces if p.radius == PHANTOM_RADIUS) == 7


# --------------------------------------------------------------------------
# gate 2 - the Sommerfeld grid extent
# --------------------------------------------------------------------------


def test_the_phantom_s_position_does_not_size_the_sommerfeld_grid():
    """The second harm, as the quantity that actually sizes the grid.

    :func:`~momwire._sommerfeld.max_image_distance` is the one shared
    computation four fill sites hand to ``_somm_grid`` (momwire#331): the
    largest observer-to-image distance over the segment endpoints, and
    therefore the radius every Sommerfeld interpolation table is built out to.

    The bar is ONE WAVELENGTH, which is not a tuning: the deck's antenna is
    19.85 m of wire and a rod over a 41.93 m wavelength, so a grid sized to
    the structure cannot reach a wavelength, and one sized to a wire parked at
    4,192 m cannot come near it.  Measured 25.047 m against 8478.300 m, which
    is 0.60 lambda against 202 lambda and leaves two decades on either side of
    the line.
    """
    deck = wa7ark_deck()
    _mesh, solver = built(deck)
    geom = solver._build_geometry()
    r1 = _sommerfeld.max_image_distance(geom["seg_l"], geom["seg_r"], solver.ground_z)
    assert r1 < _serve.SPEED_OF_LIGHT_MHZ_M / deck.frequency_mhz


def test_the_phantom_s_position_DOES_size_it_with_the_change_reverted(monkeypatch):
    """Gate 2's control: 200 wavelengths of grid for a 0.6-wavelength antenna."""
    disarm(monkeypatch)
    deck = wa7ark_deck()
    _mesh, solver = built(deck)
    geom = solver._build_geometry()
    r1 = _sommerfeld.max_image_distance(geom["seg_l"], geom["seg_r"], solver.ground_z)
    assert r1 > 100.0 * _serve.SPEED_OF_LIGHT_MHZ_M / deck.frequency_mhz


# --------------------------------------------------------------------------
# gate 3 - the phantom carries nothing, over the whole corpus
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_no_phantom_in_the_corpus_carries_a_current_or_a_charge():
    """23 of the 80 committed decks park one, and none of them scatters.

    Gated as EXACT zero rather than as a small number, because that is the
    difference the fix makes: the phantom used to sit in the solve and come
    back with ~1e-9 A of coupling dust at 173 lambda, and it is now not in
    the solve at all.  A bar would pass on either.

    The rows are still WRITTEN - one per deck element, the engine's own count
    - which is the other half of the statement and gate 6's subject.
    """
    for cid, deck, tag in phantom_corpus():
        data = serve(deck)
        phantom_rows = [row for row in data.currents if row.tag == tag]
        assert phantom_rows, cid
        for row in phantom_rows:
            assert (row.real, row.imag, row.magnitude) == (0.0, 0.0, 0.0), cid
        for row in [row for row in data.charges if row.tag == tag]:
            assert row.magnitude == 0.0, cid
        # The antenna is still answering, so a zero everywhere is not what
        # this gate is passing on.
        assert any(row.magnitude > 0.0 for row in data.currents if row.tag != tag)


@pytest.mark.integration
def test_the_phantom_DOES_carry_dust_with_the_change_reverted(monkeypatch):
    """Gate 3's control, on the smallest deck that parks one.

    0012 is 15 elements of free space, so it costs nothing to solve twice and
    its phantom's three rows are the whole difference.
    """
    disarm(monkeypatch)
    deck = parse_nec5(
        (CORPUS / "decks" / "0012_network-connection-test.nec")
        .read_bytes()
        .decode("latin-1")
    )
    data = serve(deck)
    assert any(row.magnitude > 0.0 for row in data.currents if row.tag == 4)


# --------------------------------------------------------------------------
# gate 4 - an addressed phantom node reaches no momwire port
# --------------------------------------------------------------------------


def test_a_phantom_node_is_a_deck_port_with_no_momwire_port_under_it():
    """What "not geometry" means at the seam, spelled in the mesh.

    The address still becomes a SITE - it carries the deck's ``LD`` pin, it
    can carry a drive and it prints its rows - and it reaches no column of
    the solver's port admittance, so ``T`` has an empty row there and the
    composed ``T^T Y T`` a zero row and a zero column.  That is the whole of
    the physics claim: the antenna's admittance at a node the antenna is not
    at is zero, not small.
    """
    deck = wa7ark_deck()
    mesh, _solver = built(deck)
    (virtual,) = [site for site in mesh.sites if site.spelling == "virtual"]
    assert (virtual.at.tag, virtual.at.node) == (7, 1)
    assert virtual.piece == _serve._NO_PIECE
    assert virtual.column < 0
    assert virtual.weight == 0.0
    assert virtual not in mesh.feeds and virtual not in mesh.gaps

    t = _serve._transform(mesh)
    assert not t[:, virtual.index].any()
    # One column of `T` per site, one ROW per momwire port - and the virtual
    # site added none, so the two counts differ by exactly the phantom.
    assert t.shape == (mesh.n_columns, len(mesh.sites))
    assert mesh.n_columns == len(mesh.sites) - 1


def test_a_phantom_node_IS_a_wire_port_with_the_change_reverted(monkeypatch):
    """Gate 4's control: the same address, on a polyline, with a column."""
    disarm(monkeypatch)
    mesh, _solver = built(wa7ark_deck())
    assert not [site for site in mesh.sites if site.spelling == "virtual"]
    site = next(site for site in mesh.sites if site.at.tag == 7)
    assert site.column >= 0
    assert mesh.pieces[site.piece].tag == 7


# --------------------------------------------------------------------------
# gate 5 - the deck the issue was filed for
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_wa7arks_ground_rod_deck_serves_and_reports_its_transformer():
    """The headline, and the one number a user would check.

    The deck feeds a 40 m EFHW through an ``NT`` transformer: ``Y11 = 10``,
    ``Y22 = .1234568``, so ``Y11/Y22 = 81`` and the turns ratio is 9.  The
    antenna side is read at tag 1 node 95 - the ``NT``'s real end, and a
    ``STRUCTURE EXCITATION DATA`` row - and the ``EX 4`` reports the SOURCE
    side, which is where the operator put it and which the readout half
    deliberately leaves alone (a transformer is not a gyrator: momwire#1134,
    "the source-side impedance is precisely what the operator asked for").

    So the gate is the RATIO, not either number: 81 relates the two sides of
    the card, and it holds whatever the antenna does.  Measured 4088.780 -
    13.247j on the antenna side against 50.5716 - 0.1635j on the source side,
    a ratio of 80.85 - which is 81 less the transformer's own 0.19 % of
    reactive division, and comfortably inside the 2 % below.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = serve(wa7ark_deck())

    (antenna,) = [row for row in data.network_excitation if row.tag == 1]
    source = data.sources[0]
    assert antenna.segment == 95
    assert antenna.impedance.real == pytest.approx(4088.78, rel=1e-4)
    assert source.impedance.real == pytest.approx(50.5716, rel=1e-4)
    assert abs(antenna.impedance / source.impedance) == pytest.approx(81.0, rel=0.02)


# --------------------------------------------------------------------------
# gate 6 - what must NOT move
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_the_printout_still_counts_and_addresses_the_phantom():
    """The other half of gate 3, and the reason the line is drawn at the mesh.

    NEC solves the phantom as real geometry: its elements are in the four
    STRUCTURE SPECIFICATION counts, its segments are the addresses the
    ``NETWORK DATA`` and ``ANTENNA INPUT PARAMETERS`` tables print, and there
    is a current and a charge row for each.  A printout that dropped any of
    that would stop being the engine's printout, whatever it did with the
    physics - so :func:`~momwire.eznec._serve.structure_of` still sees every
    ``GW`` and only :func:`~momwire.eznec._serve.build_mesh` does not.

    Every assertion below holds on BOTH sides of the fix, which is the point
    of them; the one line that does not is the first, which checks the pinned
    table against the detector and so goes quiet with it.
    """
    for cid, deck, tag in phantom_corpus():
        structure = _serve.structure_of(deck)
        # The detector and the table above still name the same wire.
        assert _serve._phantom_tags(
            deck, _serve.SPEED_OF_LIGHT_MHZ_M / deck.frequency_mhz
        ) == frozenset({tag}), cid
        assert tag in {w.tag for w in structure.wires}, cid
        assert structure.wire_element_count == sum(
            w.segment_count for w in deck.wires
        ), cid
        data = serve(deck)
        assert len(data.currents) == structure.wire_element_count, cid
        assert data.wire_element_count == structure.wire_element_count, cid
        # Every card that addressed the phantom still prints its segment,
        # under the phantom's own tag and the deck's own numbering.
        addressed = {row.tag for row in data.network_excitation}
        addressed |= {row.tag_from for row in data.networks}
        addressed |= {row.tag_to for row in data.networks}
        assert tag in addressed, cid


# --------------------------------------------------------------------------
# NOT this issue - the defect that actually blocked WA7ARK's deck
# --------------------------------------------------------------------------


def test_a_delta_gap_at_a_wire_end_lands_on_the_knot_vector():
    """A feed AT a wire's far end, where the two arclengths disagree by ulps.

    This is a SEPARATE defect from the one above and it is what really stopped
    WA7ARK's deck: momwire#1139 reports the deck dying on ``ValueError: Out of
    bounds w/ x = [19.84542]`` with the phantom present and solving without
    it, and reads that as the grid extent.  It is not - the deck dies at the
    same arclength with the phantom parked 5 m from the antenna instead of
    4,192 m, and the measurement it is read off removed the ``NT`` along with
    the wire, which removed the feed that was failing.

    The mechanism: an address at a wire's end asks for a gap at the
    polyline's straight-line LENGTH, and the solver's knot vector carries that
    same length ACCUMULATED over the mesh.  95 steps of 19.84542/95 sum to
    19.845419999999997, so the request sits 3.6e-15 past the last knot and
    ``BSpline.design_matrix`` refuses it.  Nothing about the deck is wrong and
    no mesh is too coarse; the arithmetic simply rounds the other way.

    Gated on the smallest reproduction rather than on the 210-element deck:
    a 2 m dipole at 10 m, six elements a side, fed at the junction, where
    1.0 accumulates to 0.9999999999999999.  Without the clip in
    ``_build_source_vector`` this raises rather than answering 7.81 -
    949.25j.
    """
    left = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    right = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    solver = BSplineSolver(
        wires=[left, right],
        n_per_edge_per_wire=[[6], [6]],
        feeds=[(0, 1.0, 1 + 0j)],
        junctions=[[(0, "end"), (1, "start")]],
        wire_radius=1e-3,
        wavelength=10.0,
    )
    arc = solver._build_geometry()["per_wire"][0]["arc_at_knot"]
    assert float(arc[-1]) < 1.0, "the reproduction stopped reproducing"

    z = 1.0 / solver.compute_port_solution().y[0, 0]
    assert z.real == pytest.approx(7.8125, rel=1e-3)
    assert z.imag == pytest.approx(-949.25, rel=1e-3)
