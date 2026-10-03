"""momwire#1300: several insertion objects at one junction of K >= 3 wires.

Licensed NEC-5 (a black box: only its printed ANTENNA INPUT PARAMETERS are
read) puts an object in the branch of the wire its card NAMES.  On a half-wave
dipole with a 0.15 m stub off its centre (a K = 3 T), driven on wire 1 at the
junction: 50 ohm named through the stub moves Z by -2.0 - j0.8 ohm, through
the far dipole arm by 45.7 - j31.8, and through the source's own wire by
exactly +50.  At K = 2 the two addresses of one node are one cut and NEC-5
puts two objects there in series; that case was already served, shares one
port at the EZNEC seam, and is not this module's.

Each address at a K >= 3 node is therefore its own node gap on its own member
(``_wire_spec.normalize_node_gaps``), the EZNEC seam opens a port for each
(``_serve._assign_columns``), and every node-gap family builds the ports as
independent columns: razor's pair-tent rows, the B-spline row's sigma-signed
one-hots, sinusoidal-Galerkin's branch cut vectors.

The decks are the investigation's (antennaknobs ``scratch/i1300-junction-
series/gen_decks.py``), rebuilt here card for card by :func:`_deck`.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve
from momwire.razor import RazorSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

# --------------------------------------------------------------------------
# the decks
# --------------------------------------------------------------------------

_GROUNDS = {
    # name: (GE card, GN card, height of the horizontal structure in m)
    "fs": ("GE 0,-1", "GN -1", 0.0),
    "pec": ("GE 1,-1", "GN 1", 0.25),
    "som": ("GE 1,-1", "GN 0,0,0,0,13.,.005,1.,0.", 0.25),
}

# name -> (loads as (tag, node, R, X), the EX address).  Node -1 is a tag's
# end 1 and node N its end 2, EZNEC's NEC-5 addressing; the junction is wire
# 1's end 2 and wires 2 and 3's end 1.  The node numbers are for the x1 mesh
# and scale with it (`_deck`).
_CASES = {
    "ex1_ld50w2": (((2, -1, 50.0, 0.0),), (1, 10)),
    "ex1_ld50w3": (((3, -1, 50.0, 0.0),), (1, 10)),
    "2ld": (((2, -1, 50.0, 0.0), (3, -1, 0.0, 100.0)), (1, 5)),
    # sign and order checks: the source named through the far arm or the
    # stub, and three objects at one node
    "ex2_ld50w1": (((1, 10, 50.0, 0.0),), (2, -1)),
    "ex3_ld50w1": (((1, 10, 50.0, 0.0),), (3, -1)),
    "ex1_ld50w2_ldjm100w3": (((2, -1, 50.0, 0.0), (3, -1, 0.0, -100.0)), (1, 10)),
}


def _deck(case: str, ground: str, refine: int = 1) -> str:
    """The K = 3 T: a lambda/2 dipole at 299.7925 MHz as two GWs meeting at
    its centre, plus a 0.15 m stub along +x from that node; ``refine``
    multiplies every wire's segment count (and the interior node numbers)."""
    ge, gn, h = _GROUNDS[ground]
    loads, (ex_tag, ex_node) = _CASES[case]
    n1, n3 = 10 * refine, 6 * refine

    def node(n: int) -> int:
        return n if n < 0 else n * refine

    lines = [
        f"CM i1300 k3 {case} {ground} x{refine}",
        "CM ! Written by EZNEC/Pro+ v. 7.0 in NEC-5 format.",
        "CE",
        f"GW 1,{n1},0.,-.25,{h},0.,0.,{h},.0005",
        f"GW 2,{n1},0.,0.,{h},0.,.25,{h},.0005",
        f"GW 3,{n3},0.,0.,{h},.15,0.,{h},.0005",
        ge,
        *(f"LD 4,{t},{node(n)},0,{r},{x}" for t, n, r, x in loads),
        "FR 0,1,0,0,299.7925",
        gn,
        f"EX 0,{ex_tag},{node(ex_node)},0,1.,0.",
        "PQ 0",
        "XQ 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


def _z(case: str, ground: str, basis: str, refine: int = 1) -> complex:
    run = _serve.serve(parse_nec5(_deck(case, ground, refine)), basis=basis)
    return run.sources[0].impedance


# Licensed NEC-5, black box, 2026-10-03, scratch/i1300-junction-series
# (antennaknobs): the printed input impedance of each deck, to the digits
# NEC-5 prints.  Every one of them was refused by this seam before #1300.
NEC5_X1 = {
    ("ex1_ld50w2", "fs"): 133.12 - 0.78251j,
    ("ex1_ld50w2", "pec"): 163.81 + 18.854j,
    ("ex1_ld50w2", "som"): 148.20 + 9.532j,
    ("ex1_ld50w3", "fs"): 85.395 + 30.126j,
    ("ex1_ld50w3", "pec"): 112.31 + 57.736j,
    ("ex1_ld50w3", "som"): 98.486 + 44.341j,
    ("2ld", "fs"): 195.17 - 74.933j,
    ("2ld", "pec"): 243.19 - 75.988j,
    ("2ld", "som"): 219.40 - 73.790j,
    ("ex2_ld50w1", "fs"): 133.12 - 0.78251j,
    ("ex3_ld50w1", "fs"): 14.499 - 214.28j,
    ("ex1_ld50w2_ldjm100w3", "fs"): 135.41 + 11.313j,
}

# The source alone (no load), same provenance.
NEC5_CTRL_FS = 87.412 + 30.964j

# The same decks on the x4 mesh: the fs rows of ex1_ld50w2 and 2ld have the
# provenance above; the other seven are the same licensed NEC-5 binary, a
# black box, run on 2026-10-03 for #1300 on decks `_deck` writes (whose x1
# twins reproduce the twelve x1 numbers above to every printed digit).
NEC5_X4 = {
    ("ex1_ld50w2", "fs"): 137.45 + 3.0901j,
    ("ex1_ld50w2", "pec"): 170.03 + 22.279j,
    ("ex1_ld50w2", "som"): 153.43 + 13.242j,
    ("ex1_ld50w3", "fs"): 89.012 + 35.918j,
    ("ex1_ld50w3", "pec"): 117.89 + 63.766j,
    ("ex1_ld50w3", "som"): 103.02 + 50.300j,
    ("2ld", "fs"): 198.68 - 73.780j,
    ("2ld", "pec"): 246.63 - 78.147j,
    ("2ld", "som"): 223.02 - 74.188j,
}

# razor is NEC-5's formulation twin (momwire#316): measured 0.000-0.006 ohm
# off on every row above, which is NEC-5's own print rounding (5 digits).
RAZOR_OHM = 0.05


# --------------------------------------------------------------------------
# the gates: razor against NEC-5
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("case", "ground"), sorted(NEC5_X1))
def test_razor_reproduces_nec5_with_objects_on_several_wires(case, ground):
    z = _z(case, ground, "razor-2p")
    assert abs(z - NEC5_X1[case, ground]) <= RAZOR_OHM, z


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore")
def test_the_far_arm_load_is_not_in_series_and_the_mirror_agrees():
    """The two shapes a seam that lost the branch would answer instead.

    In SERIES with the source (one cut, the K = 2 reading) the far-arm load
    would add exactly 50 ohm to the source's own-wire-only Z; NEC-5 prints
    45.7 - j31.8 instead.  And the mirror deck — source through the far arm,
    load through wire 1 — is the same circuit by symmetry of the T and
    prints the same number."""
    z = _z("ex1_ld50w2", "fs", "razor-2p")
    z_mirror = _z("ex2_ld50w1", "fs", "razor-2p")
    assert abs(z - z_mirror) <= 1e-6 * abs(z)
    assert abs(z - (NEC5_CTRL_FS + 50.0)) > 30.0


# --------------------------------------------------------------------------
# the Galerkin families: same ports, their usual coarse-mesh offset, and it
# shrinks with the mesh
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("basis", ("bspline", "bspline-d1", "sinusoidal-galerkin"))
@pytest.mark.parametrize(("case", "ground"), sorted(NEC5_X4))
def test_galerkin_bases_converge_toward_nec5(basis, case, ground):
    err1 = abs(_z(case, ground, basis) - NEC5_X1[case, ground])
    err4 = abs(_z(case, ground, basis, refine=4) - NEC5_X4[case, ground])
    assert err4 < 0.5 * err1, (err1, err4)


# --------------------------------------------------------------------------
# the core: K >= 3 takes one gap per member, K = 2 one gap, nobody twice
# --------------------------------------------------------------------------


def _star(k: int, nseg: int = 6):
    """``k`` wires leaving one node by their starts, coplanar and spread."""
    node = np.zeros(3)
    wires = [
        np.array([node, np.array([np.cos(t), np.sin(t), 0.0]) * 2.0])
        for t in (2 * np.pi * i / k for i in range(k))
    ]
    return wires, [[nseg]] * k


def _solver(cls, k, gaps):
    wires, npe = _star(k)
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[],
        node_gaps=gaps,
        wire_radius=1e-3,
        wavelength=20.0,
    )
    if cls is not RazorSolver:
        kw["junctions"] = [[(w, "start") for w in range(k)]]
    return cls(**kw)


NODE_GAP_FAMILIES = (RazorSolver, BSplineSolver, SinusoidalGalerkinSolver)


@pytest.mark.parametrize("cls", NODE_GAP_FAMILIES)
def test_every_member_of_a_three_wire_node_takes_its_own_gap(cls):
    """Three gaps at K = 3: one column each, and the K-port Y is what three
    branch EMFs at one node must be.

    Symmetric (each column is its own drive and readout), its diagonal is
    each gap's single-port admittance (a node gap adds no basis, so the
    ports superpose), and every row sums to zero: the three currents into
    the node close on each other (KCL), so one EMF added to all three
    branches moves nothing."""
    gaps = [(w, "start", 1.0 + 0j) for w in range(3)]
    y = _solver(cls, 3, gaps).compute_port_solution().y
    assert y.shape == (3, 3)
    scale = np.max(np.abs(y))
    np.testing.assert_allclose(y, y.T, atol=1e-10 * scale)
    np.testing.assert_allclose(y.sum(axis=1), 0.0, atol=1e-9 * scale)
    for p in range(3):
        alone = _solver(cls, 3, [gaps[p]]).compute_port_solution().y
        assert y[p, p] == pytest.approx(alone[0, 0], rel=1e-10)


@pytest.mark.parametrize("cls", NODE_GAP_FAMILIES)
def test_one_member_twice_and_a_second_gap_at_two_wires_still_refuse(cls):
    with pytest.raises(ValueError, match="listed twice"):
        _solver(cls, 3, [(1, "start", 1.0 + 0j), (1, "start", 0j)])
    with pytest.raises(ValueError, match="one series gap per two-wire junction"):
        _solver(cls, 2, [(0, "start", 1.0 + 0j), (1, "start", 0j)])
    # and two members of a K = 4 node are two cuts
    s = _solver(cls, 4, [(0, "start", 1.0 + 0j), (2, "start", 0j)])
    assert [(w, e) for w, e, _v in s.node_gaps] == [(0, "start"), (2, "start")]


def test_sg_node_ports_share_a_junction_only_as_branch_cuts():
    """sinusoidal-Galerkin's own spelling, `node_ports`, by bipartition.

    A port with ONE member on one side cuts that member's branch; several of
    those share a junction when the branches differ.  ``(1, 2)`` at K = 3
    is member 0's branch spelt from the other side, so it collides with
    ``(0,)``.  At K = 4 a two-and-two split is no branch at all, and is
    still a junction's only port."""
    wires, npe = _star(4)
    common = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[],
        wire_radius=1e-3,
        wavelength=20.0,
    )
    k3 = dict(common, wires=wires[:3], n_per_edge_per_wire=npe[:3])
    k3["junctions"] = [[(w, "start") for w in range(3)]]
    s = SinusoidalGalerkinSolver(**k3, node_ports=[(0, (0,), 1.0), (0, (0, 2), 0.0)])
    assert [side for _j, side, _v in s.node_ports] == [(0,), (0, 2)]
    with pytest.raises(ValueError, match="member 0's branch is already cut"):
        SinusoidalGalerkinSolver(**k3, node_ports=[(0, (0,)), (0, (1, 2))])
    k4 = dict(common, junctions=[[(w, "start") for w in range(4)]])
    with pytest.raises(ValueError, match="cuts ONE member's branch"):
        SinusoidalGalerkinSolver(**k4, node_ports=[(0, (0, 1)), (0, (2,))])
    SinusoidalGalerkinSolver(**k4, node_ports=[(0, (0, 1))])
