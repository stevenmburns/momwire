"""The NEC-5 seam's structure half for razor-2p in C++ (`_nec5_razor_seam_core`,
unit 2b of the standalone executable) is the Python seam's, deck by deck.

Each hand-written deck below first asserts where the PYTHON seam lands (so a
case that stopped exercising its rung would say so), then requires the C++
seam to agree with it in full through `scripts/core_split/seam_compare.py`:
the route, the refusal sentence or the escaping exception word for word, and
for every preparation the keyword arguments handed `RazorSolver` (bytes and
Python types), the mesh, the structure counts, the medium, the sites, and the
solved coefficients X and Y byte for byte. A deck the C++ seam does not cover
must decline BY NAME, at a rung no earlier answer of the Python seam depends
on, and those are pinned too.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPLIT = ROOT / "scripts" / "core_split"


def _module(name):
    spec = importlib.util.spec_from_file_location(
        f"core_split_{name}", SPLIT / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tools():
    from momwire import _accel

    acc = _accel.acc
    if acc is None or not hasattr(acc, "nec5_razor_seam"):
        pytest.skip("no accelerator with the C++ seam loaded")
    sys.path.insert(0, str(SPLIT))
    try:
        fc = _module("front_compare")
    finally:
        sys.path.remove(str(SPLIT))
    return acc, fc, _module("seam_compare")


HEAD = "CM unit 2b\nCE\n"
FREE_TAIL = (
    "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,6,0,1.,0.\nRP 0,1,1,1000,90.,0.,0.,0.\nEN\n"
)
DIPOLE = "GW 1,11,0.,-2.5,5.,0.,2.5,5.,.001\n"
MONO = "GW 1,10,0.,0.,0.,0.,0.,2.5,.002\n"
PEC = "GE 1,-1\nGN 1\nFR 0,1,0,0,30.\n"
GD = "GE 1,-1\nGD 0,0,0,0,13.,.005,1.,0.\nFR 0,1,0,0,30.\n"
RP = "RP 0,1,1,1000,90.,0.,0.,0.\n"

# (deck, the route, a piece of the sentence / the decline reason)
CASES = {
    # --- served ---------------------------------------------------------
    "free_dipole": (HEAD + DIPOLE + FREE_TAIL, "served", ""),
    "free_dipole_xq": (
        HEAD + DIPOLE + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,6,0,1.,0.\nXQ\nEN\n",
        "served",
        "",
    ),
    "pec_monopole_contact": (
        HEAD + MONO + PEC + "EX 0,1,-1,0,1.,0.\n" + RP + "EN\n",
        "served",
        "",
    ),
    "pec_without_ge_flag": (
        HEAD + MONO + "GE 0,-1\nGN 1\nFR 0,1,0,0,30.\nEX 0,1,5,0,1.,0.\n" + RP + "EN\n",
        "served",
        "",
    ),
    "gd_monopole": (
        HEAD + MONO + GD + "EX 0,1,-1,0,1.,0.\n" + RP + "EN\n",
        "served",
        "",
    ),
    "gd_negative_sigma": (
        HEAD
        + MONO
        + "GE 1,-1\nGD 0,0,0,0,13.,-12.84,1.,0.\nFR 0,1,0,0,7.\nEX 0,1,-1,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "inverted_v_apex_junction_k2": (
        HEAD
        + "GW 1,7,-3.,0.,4.,0.,0.,6.,.001\nGW 2,7,0.,0.,6.,3.,0.,4.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,25.\nEX 0,1,7,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "k2_junction_named_through_the_far_side": (
        HEAD
        + "GW 1,7,-3.,0.,4.,0.,0.,6.,.001\nGW 2,7,0.,0.,6.,3.,0.,4.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,25.\nEX 0,2,-1,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "three_wire_apex_node_gap": (
        HEAD
        + "GW 1,6,-3.,0.,4.,0.,0.,6.,.001\nGW 2,6,0.,0.,6.,3.,0.,4.,.001\n"
        + "GW 3,6,0.,0.,6.,0.,0.,9.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,25.\nEX 0,3,-1,0,1.,0.\nEX 0,1,3,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "per_wire_radii": (
        HEAD
        + "GW 1,9,0.,-2.5,5.,0.,2.5,5.,.001\nGW 2,9,1.,-2.6,5.,1.,2.6,5.,.0015\n"
        + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,5,0,1.,0.\nEX 0,2,5,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "lumped_load_is_a_site": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nLD 4,1,3,0,50.,25.\nEX 0,1,6,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "probes_beside_a_current_drive": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nLD 4,1,3,0,50.,25.\nEX 4,1,6,0,1.,0.\n"
        + "EX 0,1,3,0,1.e-10,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "load_at_a_free_end": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nLD 4,1,11,0,50.,0.\nEX 0,1,6,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "addresses a FREE end of wire 1",
    ),
    "multi_run_shared": (
        HEAD
        + "GW 1,21,0.,-5.2,10.,0.,5.2,10.,.001\nGW 2,21,4.282,-5.2,10.,4.282,5.2,10.,.001\n"
        + "GE 0\nFR 0,1,0,0,14.\nEX 0,1,11,0,1.\nEX 0,2,11,0,1.e-10\nXQ\n"
        + "EX 0,1,11,0,1.e-10\nEX 0,2,11,0,1.\nXQ\nEN\n",
        "served",
        "",
    ),
    "multi_run_moved_sites": (
        HEAD
        + "GW 1,21,0.,-5.2,10.,0.,5.2,10.,.001\nGW 2,21,4.282,-5.2,10.,4.282,5.2,10.,.001\n"
        + "GE 0\nFR 0,1,0,0,14.\nEX 0,1,11,0,1.\nXQ\nEX 0,2,10,0,1.\nXQ\nEN\n",
        "served",
        "",
    ),
    "nx_chain_then_refusal": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,6,0,1.,0.\nXQ\nNX\n"
        + HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nXQ\nEN\n",
        "refused",
        "this deck carries no EX card",
    ),
    # --- refused, word for word -----------------------------------------
    "no_ex": (
        HEAD + DIPOLE + "GE 0,-1\nFR 0,1,0,0,30.\n" + RP + "EN\n",
        "refused",
        "no EX",
    ),
    "no_fr": (
        HEAD + DIPOLE + "GE 0,-1\nEX 0,1,6,0,1.,0.\n" + RP + "EN\n",
        "refused",
        "no FR",
    ),
    "fr_zero": (
        HEAD + DIPOLE + "GE 0,-1\nFR 0,1,0,0,0.\nEX 0,1,6,0,1.,0.\n" + RP + "EN\n",
        "refused",
        "no FR",
    ),
    "no_request": (
        HEAD + DIPOLE + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,6,0,1.,0.\nEN\n",
        "refused",
        "asks for nothing",
    ),
    "rp_negative_range": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,6,0,1.,0.\nRP 0,1,1,1000,90.,0.,0.,0.,-5.\nEN\n",
        "refused",
        "negative range",
    ),
    "duplicate_ex": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,6,0,1.,0.\nEX 0,1,6,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "two EX cards address 1,6",
    ),
    "mixed_drive_generator": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nEX 4,1,6,0,1.,0.\nEX 0,1,3,0,2.5,1.\n"
        + RP
        + "EN\n",
        "refused",
        "which is a generator rather than a probe",
    ),
    "mixed_drive_unloaded_probe": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nEX 4,1,6,0,1.,0.\nEX 0,1,3,0,1.e-10,0.\n"
        + RP
        + "EN\n",
        "refused",
        "no LD card loads",
    ),
    "buried_under_gn1": (
        HEAD
        + "GW 1,10,0.,0.,1.,0.,0.,3.,.002\nGW 2,5,0.,0.,-1.,2.,0.,-1.,.002\n"
        + PEC
        + "EX 0,1,5,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "under a GN 1 card",
    ),
    "buried_under_gd": (
        HEAD
        + "GW 1,10,0.,0.,1.,0.,0.,3.,.002\nGW 2,5,0.,0.,-.75,2.,0.,-1.25,.002\n"
        + GD
        + "EX 0,1,5,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "under a GD card",
    ),
    "card_crossing_the_plane": (
        HEAD
        + "GW 1,10,0.,0.,2.,0.,0.,-.37,.002\n"
        + PEC
        + "EX 0,1,3,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "(min z = -0.37 m)",
    ),
    "in_plane_wire": (
        HEAD
        + MONO
        + "GW 2,4,0.,0.,0.,2.,0.,0.,.002\n"
        + PEC
        + "EX 0,1,5,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "lies in the ground plane",
    ),
    "free_end": (
        HEAD + DIPOLE + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,-1,0,1.,0.\n" + RP + "EN\n",
        "refused",
        "addresses a FREE end of wire 1",
    ),
    "interior_node_where_a_wire_lands": (
        HEAD
        + "GW 1,10,0.,-2.5,5.,0.,2.5,5.,.001\nGW 2,4,0.,0.,5.,0.,0.,7.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,5,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "addresses a node INSIDE wire 1 where 1 more element ends meet",
    ),
    "grounded_junction": (
        HEAD
        + MONO
        + "GW 2,4,0.,0.,0.,1.,0.,1.,.002\n"
        + PEC
        + "EX 0,1,-1,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "several wires meet IN the ground plane",
    ),
    "two_drives_one_cut": (
        HEAD
        + "GW 1,7,-3.,0.,4.,0.,0.,6.,.001\nGW 2,7,0.,0.,6.,3.,0.,4.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,25.\nEX 0,1,7,0,1.,0.\nEX 0,2,-1,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "are the two sides of ONE cut",
    ),
    "inert_one_segment_wire": (
        HEAD
        + "GW 1,8,0.,-2.,0.,0.,2.,0.,.001\nGW 2,1,0.5,-0.25,0.,0.5,0.25,0.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,4,0,1.,0.\nXQ\nEN\n",
        "refused",
        "is one segment long and meets no other wire",
    ),
    "reader_refusal": (HEAD + DIPOLE + "ZZ 1\n" + FREE_TAIL, "refused", ""),
    # --- raised: what escapes the Python seam escapes here too -----------
    "reader_index_error": (
        HEAD + DIPOLE + ", ,\n" + FREE_TAIL,
        "raised",
        "only separators",
    ),
    "constructor_value_error": (
        HEAD + "GW 1,11,0.,-2.5,5.,0.,2.5,5.,0.\n" + FREE_TAIL,
        "raised",
        "wire_radius entries must be positive and finite, got [0]",
    ),
    # --- declined by name -----------------------------------------------
    # TL / NT and phantom wires are the structure half's since the networks
    # stage: the card ends are sites, a phantom's nodes virtual sites.
    "serves_tl": (
        HEAD
        + DIPOLE
        + "GW 2,3,0.,10.,5.,0.,10.3,5.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,30.\nTL 1,6,2,2,50.,1.,0.,0.,0.,0.\nEX 0,2,2,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "serves_nt_gyrator_on_a_phantom": (
        HEAD
        + DIPOLE
        + "GW 2,3,0.,10000.,0.,0.,10000.05,0.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,30.\nNT 2,2,1,6,0.,0.,0.,1.,0.,0.\nEX 0,2,2,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "refuses_an_interleaved_network_table": (
        HEAD
        + DIPOLE
        + "GW 2,3,0.,10.,5.,0.,10.3,5.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,30.\nNT 1,2,2,2,0.,0.,0.,1.,0.,0.\n"
        + "TL 1,6,2,2,50.,1.,0.,0.,0.,0.\nEX 0,1,6,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "writes an NT card before a TL card",
    ),
    "refuses_a_zero_length_line": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nTL 1,6,1,6,50.,0.,0.,0.,0.,0.\nEX 0,1,6,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "writes no length",
    ),
    "serves_sommerfeld": (
        HEAD
        + MONO
        + "GE 1,-1\nGN 0,0,0,0,13.,.005,1.,0.\nFR 0,1,0,0,30.\nEX 0,1,-1,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "serves_sommerfeld_gn2": (
        HEAD
        + DIPOLE
        + "GE 1,-1\nGN 2,0,0,0,5.,.001\nFR 0,1,0,0,30.\nEX 0,1,6,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "refuses_a_near_field_at_a_sommerfeld_contact": (
        HEAD
        + MONO
        + "GE 1,-1\nGN 0,0,0,0,13.,.005,1.,0.\nFR 0,1,0,0,30.\nEX 0,1,-1,0,1.,0.\n"
        + "NE 0,1,1,1,0.,0.,0.,0.,0.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "stands on this deck's finite ground",
    ),
    "refuses_an_in_plane_wire_over_sommerfeld": (
        HEAD
        + MONO
        + "GW 2,5,1.,0.,0.,2.,0.,0.,.002\n"
        + "GE 1,-1\nGN 2,0,0,0,13.,.005,1.,0.\nFR 0,1,0,0,30.\nEX 0,1,-1,0,1.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "lies in the ground plane",
    ),
    "declines_a_buried_fill_to_the_run_half": (
        HEAD
        + "GW 1,10,0.,-2.5,-0.5,0.,2.5,-0.5,.002\n"
        + "GE 1,-1\nGN 0,0,0,0,13.,.005,1.,0.\nFR 0,1,0,0,30.\nEX 0,1,5,0,1.,0.\n"
        + RP
        + "EN\n",
        # The seam's structure half serves a buried wire since momwire#1512
        # B1; its front leaves the buried fill to the run half, which fills
        # the below/below table (B5).
        "declined",
        "front:buried-fill",
    ),
    "refuses_a_near_field_below_gd": (
        HEAD
        + MONO
        + GD
        + "EX 0,1,-1,0,1.,0.\nNE 0,1,1,1,1.,0.,-1.,0.,0.,0.\n"
        + RP
        + "EN\n",
        "refused",
        "which is below this deck's GD ground",
    ),
    "declines_a_near_field_below_sommerfeld": (
        HEAD
        + MONO
        + "GE 1,-1\nGN 0,0,0,0,13.,.005,1.,0.\nFR 0,1,0,0,30.\nEX 0,1,-1,0,1.,0.\n"
        + "NE 0,1,1,1,1.,0.,-1.,0.,0.,0.\n"
        + RP
        + "EN\n",
        # momwire#1653: the Python seam's transmitted readout serves it.
        "declined",
        "near-field-below",
    ),
    "declines_a_buried_decks_near_field": (
        HEAD
        + "GW 1,10,0.,-2.5,-0.5,0.,2.5,-0.5,.002\n"
        + "GE 1,-1\nGN 0,0,0,0,13.,.005,1.,0.\nFR 0,1,0,0,30.\nEX 0,1,5,0,1.,0.\n"
        + "NE 0,1,1,1,1.,0.,1.,0.,0.,0.\n"
        + RP
        + "EN\n",
        "declined",
        "near-field-buried",
    ),
    "serves_wire_loading": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nLD 5,0,1,11,5.8E+7,1.\nEX 0,1,6,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "serves_per_metre_rlc": (
        HEAD
        + DIPOLE
        + "GE 0,-1\nFR 0,1,0,0,30.\nLD 2,1,1,11,0.5,2.E-7,0.\nEX 0,1,6,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "serves_the_exact_kernel": (
        HEAD + "GW 1,11,0.,-2.5,5.,0.,2.5,5.,.2\n" + FREE_TAIL,
        "served",
        "",
    ),
    "serves_the_exact_kernel_across_a_radius_step": (
        HEAD
        + "GW 1,6,0.,-2.5,5.,0.,0.,5.,.2\nGW 2,5,0.,0.,5.,0.,2.5,5.,.25\n"
        + FREE_TAIL,
        "served",
        "",
    ),
    "serves_a_phantom_wire": (
        HEAD
        + DIPOLE
        + "GW 2,3,0.,10000.,0.,0.,10000.05,0.,.001\n"
        + "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,6,0,1.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
    "serves_a_near_field_over_gd": (
        HEAD
        + MONO
        + GD
        + "EX 0,1,-1,0,1.,0.\nNE 0,1,1,1,1.,0.,1.,0.,0.,0.\n"
        + RP
        + "EN\n",
        "served",
        "",
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_cxx_seam_is_the_python_seam(tools, name):
    acc, fc, sc = tools
    text, route, words = CASES[name]
    row, bad = sc.compare(text, acc, fc)
    assert bad == [], bad
    assert row["route"] == route, row
    py = sc.python_seam(text)
    if route == "declined":
        assert row["reason"] == words
    else:
        # Where the Python seam lands is what this case is about.
        assert py["outcome"] == route, py
        assert words in py["message"], py["message"]
    if route == "served":
        assert row.get("solves", 0) >= 1


def test_the_multi_run_deck_shares_its_preparation(tools):
    acc, _fc, _sc = tools
    shared = acc.nec5_razor_seam(CASES["multi_run_shared"][0].encode("latin-1"))
    assert [r["kind"] for r in shared["structures"][0]["runs"]] == [
        "prepared",
        "shared",
    ]
    moved = acc.nec5_razor_seam(CASES["multi_run_moved_sites"][0].encode("latin-1"))
    assert [r["kind"] for r in moved["structures"][0]["runs"]] == [
        "prepared",
        "prepared",
    ]


def test_the_constants_are_the_seams(tools):
    from momwire.eznec import _serve

    acc, _fc, _sc = tools
    k = acc.nec5_seam_constants()
    assert k["speed_of_light_mhz_m"] == _serve.SPEED_OF_LIGHT_MHZ_M
    assert k["node_eps"] == _serve._NODE_EPS
    assert k["probe_volts"] == _serve._PROBE_VOLTS
    assert k["epsc_conductivity_factor"] == _serve.EPSC_CONDUCTIVITY_FACTOR


# Default-lane budget is 5 s; CI median 19 s Linux, 16 s Windows, 9.5 s macOS (momwire 0.82.0 release prep).
@pytest.mark.slow
def test_every_fixture_deck_the_cxx_seam_answers_is_answered_alike(tools):
    """momwire's own fixture decks, field for field (the corpus gate in
    miniature: `seam_compare.py` runs the same comparison over the step-0
    corpus)."""
    acc, fc, sc = tools
    decks = [
        p
        for p in sorted((ROOT / "tests" / "fixtures").rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]
    routes = {}
    for path in decks:
        text = path.read_bytes().decode("latin-1")
        cx = acc.nec5_razor_seam(text.encode("latin-1"))
        routes[cx["route"]] = routes.get(cx["route"], 0) + 1
        if cx["route"] == "declined":
            continue
        row, bad = sc.compare(text, acc, fc)
        assert bad == [], (path, bad)
    assert routes.get("served", 0) >= 20, routes
