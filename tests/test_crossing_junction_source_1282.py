"""momwire#1282: a series source ON a crossing junction in the ground plane.

Dan AC6LA's deck (``tests/fixtures/eznec_crossing_source_1282/``, README
there): a vertical ending at z = 0, a one-segment buried connector from that
node down to a radial hub, and the source at the node. NEC-5 answers the
node named through the vertical (``1,-1``) and through the connector
(``6,-1``) with the same 70.787 - j0.696 ohm, so the source is the
junction's single through-current and the tag carries no physics.

The seam read the node as GROUNDED (#151: shorted to the plane through the
image) and refused it. A crossing node is not shorted to anything — one
member is in the soil — and each basis now drives its through-current:

* razor carries one tent across the node, so a delta gap on the knot is the
  through-current already;
* the B-spline and sinusoidal-Galerkin families leave the two members' end
  bases independent at a crossing node, so a gap at one member's end is a
  source between the node and that member only. Measured before the fix:
  bspline 71.123 + 1.201j via ``1,-1`` against 70.877 + 0.659j via
  ``6,-1``; SG 71.224 + 1.249j against 70.969 + 0.788j. Their port is the
  node gap, served with the junction's continuity imposed while the gap is
  there, which makes the two spellings one port.

The spelling gate below is what fails if that continuity is dropped (a
relative difference of 5.8e-3 to 8.4e-3, measured, six orders above the bar)
or the exemption is (a refusal).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver, SinusoidalGalerkinSolver, SinusoidalSolver
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _printout, _serve, _shell
from momwire.eznec._resident import _CODEC as _RESIDENT_CODEC
from momwire.razor import RazorSolver

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_crossing_source_1282"

KNOT_BASES = ("razor-nec5", "razor-2p", "bspline", "bspline-d1", "sinusoidal-galerkin")

# Each basis's own answer on the two placements, measured on Haswell at
# f95a912 (`scratch` probe through `_serve.serve`). Pinned to 1e-3 ohm: four
# orders under the ~0.5 ohm the one-member port was wrong by, and far above
# the cross-machine BLAS/FMA drift this repo sees (~1e-9 relative).
PINNED = {
    "razor-nec5": (70.8068660130 - 0.2415288500j, 71.4387098395 + 0.4515876592j),
    "razor-2p": (70.8068660130 - 0.2415288500j, 71.4387098395 + 0.4515876592j),
    "bspline": (71.1232015671 + 1.2012528422j, 71.7044595259 + 2.1861961542j),
    "bspline-d1": (71.0012455387 + 0.9516474150j, 71.6041366040 + 1.6922649323j),
    # momwire#1354: the mixed-potential fill. The direct fill read
    # 71.2220705882 + 1.2490278872j / 71.7861410477 + 2.2249137096j; the new
    # fill sits 0.15 ohm closer to NEC-5 on both placements (1.84 / 2.12 ohm
    # against 1.99 / 2.26) and 0.01 ohm from bspline's own answer.
    "sinusoidal-galerkin": (
        71.1117686390 + 1.1134179730j,
        71.6934518427 + 2.0954101627j,
    ),
}
PIN_OHM = 1e-3

# Moving the source one segment up the vertical shifts NEC-5's Z by
# -0.628 - 0.689j. razor is NEC-5's formulation twin and reproduces that shift
# to 6 mOhm (measured 5.5); the Galerkin bases differ from NEC-5 in the
# near-node reactance at this mesh (the node is unresolved, `CoarseCrossingNode`
# says so: 258 mm against a 25 mm bar) and land within 0.30 ohm (measured
# 0.300 bspline, 0.057 bspline-d1, 0.294 SG).
SHIFT_OHM = {"razor-nec5": 0.02, "razor-2p": 0.02}
SHIFT_OHM_GALERKIN = 0.35

# Spelling invariance: the two addresses solve one port on every basis, so
# they agree to the solve's roundoff.
SPELLING_REL = 1e-9

_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+\d\s+" + r"([-+.\dE]+)\s+" * 8, re.M)


def _deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


def _nec5_z(name: str) -> complex:
    """NEC-5's printed input impedance (black-box number off its printout)."""
    text = (FIXTURES / f"{name}.out").read_text()
    i = text.index("ANTENNA INPUT PARAMETERS")
    m = _ROW.search(text, i)
    assert m, text[i : i + 600]
    return complex(float(m.group(7)), float(m.group(8)))


def _serve_deck(name: str, basis: str):
    return _serve.serve(parse_nec5(_deck(name)), basis=basis)


# --------------------------------------------------------------------------
# the gates: every knot basis serves both spellings, as one port
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("basis", KNOT_BASES)
def test_both_spellings_of_the_junction_source_are_one_port(basis, record_property):
    via1 = _serve_deck("dan_end1", basis)
    via6 = _serve_deck("dan_end1_via6", basis)
    z1 = via1.sources[0].impedance
    z6 = via6.sources[0].impedance
    rel = abs(z1 - z6) / abs(z1)
    record_property("z_via1", f"{z1:.10f}")
    record_property("spelling_rel", f"{rel:.3e}")
    assert rel <= SPELLING_REL, (z1, z6)

    # The two cards drive the node in opposite directions along their own
    # wires, so every current flips sign and nothing else moves — exactly
    # what NEC-5 prints for the pair. The vertical's first element carries
    # +1.41e-2 A under `1,-1` in NEC-5's printout, and so must it here.
    i1 = np.array([complex(r.real, r.imag) for r in via1.currents])
    i6 = np.array([complex(r.real, r.imag) for r in via6.currents])
    np.testing.assert_allclose(i6, -i1, rtol=1e-8, atol=1e-14)
    first = next(r for r in via1.currents if r.element == 1)
    assert first.real > 0.0


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("basis", KNOT_BASES)
def test_the_junction_source_against_nec5_and_against_the_moved_source(
    basis, record_property
):
    z_end1 = _serve_deck("dan_end1", basis).sources[0].impedance
    z_moved = _serve_deck("dan_seg1end2", basis).sources[0].impedance
    n5_end1, n5_moved = _nec5_z("dan_end1"), _nec5_z("dan_seg1end2")
    assert _nec5_z("dan_end1_via6") == n5_end1

    record_property("z_end1", f"{z_end1:.6f}")
    record_property("offset_from_nec5", f"{z_end1 - n5_end1:.6f}")
    record_property("shift", f"{z_end1 - z_moved:.6f}")

    pinned_end1, pinned_moved = PINNED[basis]
    assert abs(z_end1 - pinned_end1) <= PIN_OHM, z_end1
    assert abs(z_moved - pinned_moved) <= PIN_OHM, z_moved

    # The junction source sits no farther from NEC-5 than this basis's own
    # already-served placement one segment up: the offset is the basis's,
    # not the port's (razor 0.455 vs 0.460 ohm, bspline 1.93 vs 2.21).
    assert abs(z_end1 - n5_end1) <= abs(z_moved - n5_moved)

    # And moving the source moves Z the way it moves NEC-5's.
    bound = SHIFT_OHM.get(basis, SHIFT_OHM_GALERKIN)
    assert abs((z_end1 - z_moved) - (n5_end1 - n5_moved)) <= bound


# --------------------------------------------------------------------------
# what the seam builds for the node
# --------------------------------------------------------------------------


def _site(text: str, solver_class: type, tag: int = 1, node: int = 0):
    deck = parse_nec5(text)
    mesh = _serve.build_mesh(
        deck,
        _serve.structure_of(deck),
        solver_class=solver_class,
        crossing=bool(_serve._crossing_nodes(deck)),
    )
    return mesh, next(s for s in mesh.sites if s.at.tag == tag and s.at.node == node)


@pytest.mark.parametrize(
    ("solver_class", "spelling"),
    [
        (BSplineSolver, "node"),
        (SinusoidalGalerkinSolver, "node"),
        (RazorSolver, "gap"),
    ],
)
def test_the_crossing_node_is_a_junction_site_not_a_contact(solver_class, spelling):
    mesh, site = _site(_deck("dan_end1"), solver_class)
    assert site.crossing and not site.contact
    assert site.spelling == spelling
    (group,) = [g for g in mesh.junctions if (site.piece, site.end) in g]
    assert len(group) == 2


# --------------------------------------------------------------------------
# adversarial: what the exemption must NOT reach
# --------------------------------------------------------------------------

_TWO_IN_THE_PLANE = (
    "CM two wires meeting in the plane, nothing buried\n"
    "CE\n"
    "GW 1,10,0.,0.,0.,0.,0.,5.,1.E-3\n"
    "GW 2,10,0.,0.,0.,3.,0.,4.,1.E-3\n"
    "GE 1,-1\n"
    "FR 0,1,0,0,14.\n"
    "{ground}\n"
    "EX 0,1,-1,0,1.,0.\n"
    "RP 0,1,1,1000,90.,0.,0.,0.,0.\n"
    "EN\n"
)


@pytest.mark.parametrize("ground", ["GN 1", "GN 0,0,0,0,13.,.005"])
@pytest.mark.parametrize("basis", ["bspline", "razor-nec5", "sinusoidal-galerkin"])
def test_a_grounded_junction_with_nothing_buried_keeps_its_refusal(ground, basis):
    """Two wires meeting IN the plane over a PEC or a finite ground with no
    buried member: the node is #151's grounded junction, a short to the plane
    through the image, and a series source there is still refused — never
    served as a crossing gap."""
    with pytest.raises(_serve.ServeRefusal, match="several wires meet IN the ground"):
        _serve.serve(parse_nec5(_TWO_IN_THE_PLANE.format(ground=ground)), basis=basis)


def test_a_lone_wire_ending_in_the_plane_is_still_a_ground_contact():
    """Nothing buried and nothing else at the node: today's contact feed."""
    text = (
        "CM lone contact\nCE\n"
        "GW 1,10,0.,0.,0.,0.,0.,5.,1.E-3\n"
        "GE 1,-1\nFR 0,1,0,0,14.\nGN 0,0,0,0,13.,.005\n"
        "EX 0,1,-1,0,1.,0.\nRP 0,1,1,1000,90.,0.,0.,0.,0.\nEN\n"
    )
    for solver_class in (BSplineSolver, RazorSolver):
        _mesh, site = _site(text, solver_class)
        assert site.contact and not site.crossing and site.spelling == "gap"


def test_a_crossing_node_of_three_wires_refuses_naming_why():
    """A second buried wire ending on the node: K = 3, and the source would
    sit between the named wire and the other two together."""
    text = _deck("dan_end1").replace(
        "GE -1,-1", "GW 7,2,0.,0.,0.,.5,0.,-.0254,1.02616E-3\nGE -1,-1"
    )
    with pytest.raises(_serve.ServeRefusal) as caught:
        _serve.serve(parse_nec5(text), basis="bspline")
    message = str(caught.value)
    assert "momwire#1282" in message
    assert "3 wires meet (1 above the ground, 2 buried)" in message
    assert "IN the ground plane; a series source at a grounded" not in message


_PEC_POLYLINES = [
    np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)]),
    np.array([(0.0, 0.0, 0.0), (3.0, 0.0, 4.0)]),
]


def test_bspline_keeps_the_grounded_node_gap_refusal_over_pec():
    solver = BSplineSolver(
        wires=_PEC_POLYLINES,
        n_per_edge_per_wire=[[10], [10]],
        feeds=[],
        junctions=[[(0, "start"), (1, "start")]],
        node_gaps=[(0, "start", 1.0)],
        wavelength=299.792458 / 14.0,
        wire_radius=1e-3,
        ground_z=0.0,
    )
    assert solver._node_gap_crossing_junctions() == frozenset()
    with pytest.raises(ValueError, match="junction 0 is grounded"):
        solver.compute_port_solution()


def test_sg_keeps_the_grounded_node_port_refusal_over_pec():
    solver = SinusoidalGalerkinSolver(
        wires=_PEC_POLYLINES,
        n_per_edge_per_wire=[[10], [10]],
        feeds=[],
        junctions=[[(0, "start"), (1, "start")]],
        node_gaps=[(0, "start", 1.0)],
        wavelength=299.792458 / 14.0,
        wire_radius=1e-3,
        ground_z=0.0,
    )
    assert solver._node_port_crossing_junctions() == frozenset()
    with pytest.raises(ValueError, match="both grounded and a node port"):
        solver.compute_port_solution()


# --------------------------------------------------------------------------
# the point-matched sinusoidal family: refused, and the refusal is printed
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("name", ["dan_end1", "dan_end1_via6"])
def test_sinusoidal_refuses_the_junction_source_naming_this_issue(name, tmp_path):
    out = tmp_path / "NEC5.OUT"
    _shell.run(FIXTURES / f"{name}.nec", out, basis="sinusoidal")
    lines = out.read_bytes().decode(_shell._CODEC).splitlines()
    errors = [line for line in lines if "NEC ERROR" in line]
    assert len(errors) == 1, errors
    assert "INTERNAL ERROR" not in errors[0]
    assert "momwire#1282" in errors[0]
    # Refused by design, not pending (Steve, 2026-10-02): a segment-centre
    # family's only spelling of this source straddles the interface.
    assert "by design" in errors[0]
    assert "grounded junction" not in errors[0]


@pytest.mark.integration
def test_a_refusal_with_an_em_dash_reaches_the_printout(tmp_path):
    """Dan's WORKING deck on the sinusoidal engine: its knot-feed refusal
    carries an em dash, which crashed the latin-1 writer into "INTERNAL
    ERROR ... UnicodeEncodeError" instead of printing the sentence."""
    assert "\u2014" in SinusoidalSolver.capabilities.refusal("knot_feeds")
    out = tmp_path / "NEC5.OUT"
    _shell.run(FIXTURES / "dan_seg1end2.nec", out, basis="sinusoidal")
    lines = out.read_bytes().decode(_shell._CODEC).splitlines()
    errors = [line for line in lines if "NEC ERROR" in line]
    assert len(errors) == 1, errors
    assert "INTERNAL ERROR" not in errors[0]
    assert "knot feeds are not served by SinusoidalSolver" in errors[0]
    assert "instead - the match points" in errors[0]


# --------------------------------------------------------------------------
# the codec boundary itself
# --------------------------------------------------------------------------


def test_any_refusal_text_is_written_transliterated(tmp_path):
    reason = (
        "a \u2014 b\u2014c, 50 \u03a9 at \u03bb/4, x \u2264 1 \u2212 \u03b5 \u2192 ok"
    )
    out = tmp_path / "NEC5.OUT"
    _shell.write_printout(out, _printout.render_refusal(None, reason))
    data = out.read_bytes()
    line = next(
        raw for raw in data.decode("latin-1").splitlines() if "NEC ERROR" in raw
    )
    assert line == (
        " ***** NEC ERROR - a - b - c, 50 ohm at lambda/4, x <= 1 - eps -> ok"
    )
    assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")


def test_latin1_text_is_untouched_and_the_resident_uses_the_same_handler():
    text = "caf\u00e9 \u00b2 \u00d7 \u00b5 plain"
    assert _shell.transliterate(text) == text
    encoded = "x \u2014 y".encode(_RESIDENT_CODEC, errors=_shell._PRINTOUT_ERRORS)
    assert encoded == b"x - y"
