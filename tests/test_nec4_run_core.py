"""The C++ NEC-4.2 run half prints the Python slot's printout for the
sinusoidal basis, byte for byte.

`src/momwire/_nec4_run_core.cpp` (with `_sinusoidal_run_core.cpp`) copies
`eznec/_nec4.py`'s `solve` / `_run` and the `portal.DeckSolver` readouts
they call (momwire#1607 B4a). `scripts/core_split/nec4_run_compare.py`
holds the C++ printout equal to `_nec4.render(text, basis="sinusoidal")`
with only the two measured-time lines (FILL= ... SEC. and RUN TIME)
masked, by exact pattern and with the same count on both sides, over:
- the NEC-4.2 fixtures;
- hand-written free-space decks for each branch the run half takes: an EX 0
  and an EX 6 source, two sources, LD 0 / 1 / 4 port loads, an FR sweep, two
  execute cards with and without a new FR, PQ charges with free ends and a
  junction, RP with and without a range, an XNDA average-gain trailer, a GM
  copy, a GS scale, a GA arc, and an NX chain;
- a perfect ground (GN 1): a monopole, and RP with a range and rows below
  the horizon;
- wire loading: LD 5 (with its loss in the power budget), on one wire
  through a sweep, and LD 2 / LD 3 per metre;
- a Sommerfeld ground above the plane: RP with a range, the lower medium's
  line and rows below the horizon, PQ, GN 3 with LD 5;
- a monopole on a Sommerfeld ground (the contact correction), with PQ and
  RP;
- a GH helix (its wire-table row), alone and tapered beside a wire;
- a phased drive (current sources among several): an EX 6 beside an EX 0,
  and two EX 6 with a port load;
- NE / NH tables: free space (rectangular, and spherical after an RP), a
  perfect ground, and a Sommerfeld ground above the plane and beside a
  monopole's contact;
- networks: a TL (straight, and crossed with shunts and its length taken
  between the segment centres), an NT beside a port load, a TL under a
  current source and under a phased drive, and a TL read after a run;
- GD's far-field cliff over GN 1, GN 2 and GN 3: an edge and a step
  splitting the reflection points, with a range, an average, further tables,
  a monopole on the ground, an empty second medium, and GD in free space
  (its block printed, no number moved);
- a wire listed twice (momwire#1042): merged on the dense and the sparse
  route, reversed, three times, made by GM, as a T's arm, over GN 1 and
  GN 2, under LD 5, with PQ / RP / NE; refused for a feed on the pair,
  differing segment counts, radii, and LD 2 / LD 5 on the copy alone;
- a wholly-buried deck (momwire#1607 B6a): a dipole over GN 2 with PQ and
  RP, GN 3 with LD 5, an FR sweep under EX 6, a buried hub, two radii with a
  port load; the refusals in Python's words (a buried wire over GN 1, and
  the crossing scope: two radii, a hub, an oblique node, two nodes, two
  nodes too close);
- the declines: a near-field point on a wire's finite-ground contact or in
  the soil, a near field over a buried wire, a crossing node inside the
  scope (B6b), wires on both sides with no crossing.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the reader copies CPython 3.12's float rules"
)


def _compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_nec4_run_compare",
        ROOT / "scripts" / "core_split" / "nec4_run_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _compare_module()


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "sinusoidal_nec4_printout"):
        pytest.skip("no compiled accelerator with the NEC-4.2 run half")
    try:
        rc.blas_addresses()
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    return _accel.acc


@pytest.fixture(autouse=True)
def _below_anchors_off(monkeypatch):
    # The below/below grid's anchors (momwire#1471) follow the frequencies a
    # process has seen; the gates hold both copies to the exact grid.
    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")


@pytest.fixture(autouse=True)
def _one_blas_thread():
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


def _decks(root):
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]


@pytest.mark.slow  # median over the 5 s default-lane budget at 0.84.0
def test_the_nec42_fixtures_print_the_same(acc):
    counts, mismatches = rc.compare(_decks(FIXTURES / "eznec_nec42"))
    assert not mismatches, mismatches[:2]
    assert counts["printed"] >= 30, counts
    assert counts["masked_lines"] >= 2 * 30, counts


# momwire#1644: the NEC-4 dialect's radius-step junction (the junction solve's
# per-end basis, the NEC-4 kernel, the end caps and their images), on the
# fixture decks small enough for the default lane.
_STEP_1644 = sorted(
    p
    for p in (FIXTURES / "nec42_radius_step_1644" / "decks").glob("*.nec")
    if not any(t in p.stem for t in ("_n80", "_n160", "_x4", "_x8"))
)


@pytest.mark.parametrize("rule", [None, "nec2", "nec42"])
def test_the_radius_step_decks_have_the_same_printout(rule, acc, monkeypatch):
    # both junction rules (momwire#1644): the dialect's default, and each
    # spelled through MOMWIRE_JUNCTION_RULE, which both copies read
    if rule is None:
        monkeypatch.delenv("MOMWIRE_JUNCTION_RULE", raising=False)
    else:
        monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", rule)
    counts, mismatches = rc.compare(_STEP_1644)
    assert not mismatches, mismatches[:2]
    assert counts["printed"] >= 40, counts


FR = "FR 0 1 0 0 299.7925\n"


def deck(geometry: str, body: str, ge: str = "GE 0") -> str:
    return "CM run\nCE\n" + geometry + ge + "\n" + body + "EN\n"


DIPOLE = "GW 1 11 0 0 -.25 0 0 .25 .001\n"
TWO = DIPOLE + "GW 2 11 .2 0 -.25 .2 0 .25 .001\n"
TEE = "GW 1 10 0 0 -.25 0 0 .25 .001\nGW 2 5 0 0 0 .3 0 0 .001\n"
RP = "RP 0 7 4 1000 0 0 30 90\n"
GD_DIPOLE = "GW 1 11 0 -.25 .6 0 .25 .6 .001\n"
DUP_DIPOLE = "GW 1 41 0 0 -.25 0 0 .25 .001\n"
BURIED_DIPOLE = "GW 1 21 -5 0 -.5 5 0 -.5 .001\n"
BFR = "FR 0 1 0 0 7.15\n"
DUP_PARASITE = "GW 2 31 .2 0 -.23 .2 0 .23 .001\n"
GD_RP = "RP 0 19 4 1000 0 0 5 45\n"

CASES = [
    ("ex0", deck(DIPOLE, FR + "EX 0 1 6 0 1\nXQ\n"), "printed"),
    ("ex6", deck(DIPOLE, FR + "EX 6 1 6 0 1 0\nXQ\n"), "printed"),
    ("two_sources", deck(TWO, FR + "EX 0 1 6 0 1\nEX 0 2 6 0 0 1\nXQ\n"), "printed"),
    (
        "loads",
        deck(
            TWO,
            "LD 0 1 3 3 10 1e-7 1e-11\nLD 1 2 4 4 50 1e-6 0\nLD 4 2 8 8 25 -30\n"
            + FR
            + "EX 0 1 6 0 1\nXQ\n",
        ),
        "printed",
    ),
    ("sweep", deck(DIPOLE, "FR 0 3 0 0 250 50\nEX 0 1 6 0 1\nXQ\n"), "printed"),
    (
        "two_runs",
        deck(TWO, FR + "EX 0 1 6 0 1\nXQ\nEX 0 2 6 0 1\nXQ\n"),
        "printed",
    ),
    (
        "two_runs_new_fr",
        deck(DIPOLE, FR + "EX 0 1 6 0 1\nXQ\nFR 0 1 0 0 280\nXQ\n"),
        "printed",
    ),
    ("charges", deck(TEE, "PQ 0\n" + FR + "EX 0 1 5 0 1\nXQ\n"), "printed"),
    ("rp", deck(DIPOLE, FR + "EX 0 1 6 0 1\n" + RP), "printed"),
    (
        "rp_range",
        deck(DIPOLE, FR + "EX 0 1 6 0 1\nRP 0 7 4 1000 0 0 30 90 1000\n"),
        "printed",
    ),
    (
        "rp_average",
        deck(DIPOLE, FR + "EX 0 1 6 0 1\nRP 0 7 4 1001 0 0 30 90\n"),
        "printed",
    ),
    (
        "gm_copy",
        deck(DIPOLE + "GM 1 1 0 0 0 .2 0 0 1\n", FR + "EX 0 1 6 0 1\n" + RP),
        "printed",
    ),
    ("gs_scale", deck(DIPOLE + "GS 0 0 2\n", FR + "EX 0 1 6 0 1\nXQ\n"), "printed"),
    ("ga_arc", deck("GA 1 12 .2 0 180 .002\n", FR + "EX 0 1 6 0 1\nXQ\n"), "printed"),
    (
        "nx_chain",
        "CM a\nCE\n"
        + DIPOLE
        + "GE 0\n"
        + FR
        + "EX 0 1 6 0 1\nXQ\nNX\nCM b\nCE\n"
        + DIPOLE
        + "GE 0\nFR 0 1 0 0 280\nEX 0 1 6 0 1\nXQ\nEN\n",
        "printed",
    ),
    (
        "monopole_on_pec",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n", "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n", "GE 1"
        ),
        "printed",
    ),
    (
        "pec_rp_range_below_horizon",
        deck(
            "GW 1 11 0 0 .5 0 0 1 .001\n",
            "GN 1\n" + FR + "EX 0 1 6 0 1\nPQ 0\nRP 0 10 2 1001 0 0 20 90 1000\n",
        ),
        "printed",
    ),
    (
        "ld5_copper",
        deck(DIPOLE, "LD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 6 0 1\nPQ 0\n" + RP),
        "printed",
    ),
    (
        "ld5_one_wire_sweep",
        deck(TWO, "LD 5 2 0 0 3.5e7\nFR 0 3 0 0 250 50\nEX 0 1 6 0 1\nXQ\n"),
        "printed",
    ),
    (
        "ld2_ld3_per_metre",
        deck(
            TWO,
            "LD 2 1 0 0 10 1e-7 0\nLD 3 2 0 0 1e4 1e-6 0\n" + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "printed",
    ),
    (
        "gn2_rp_range_below_horizon",
        deck(
            "GW 1 11 0 0 .3 0 0 .8 .001\n",
            "GN 2 0 0 0 13 .005\n"
            + FR
            + "EX 0 1 6 0 1\nPQ 0\nRP 0 10 2 1001 0 0 20 90 1000\n",
        ),
        "printed",
    ),
    (
        "gn3_ld5_horizontal",
        deck(
            "GW 1 21 -.5 0 .3 .5 0 .3 .001\n",
            "GN 3 0 0 0 5 .02\nLD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 11 0 1\n" + RP,
        ),
        "printed",
    ),
    (
        "monopole_on_gn2_contact",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n",
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 1 0 1\nPQ 0\n" + RP,
            "GE 1",
        ),
        "printed",
    ),
    (
        "gd_under_xq",
        "CM run\nCE\nGW 1,11,0.,-.25,10.,0.,.25,10.,.0005\nGE 1,-1\n"
        "FR 0,1,0,0,299.7925\nGN 1,0,0,0,0.,0.\nEX 0,1,6,0,1.414214,0.\n"
        "GD 2,0,0,0,13.,.005,0.,0.\nXQ 0\nEN\n",
        "printed",
    ),
    # GD's second medium (B5d): the cliff's edge and step put some
    # reflection points short of the edge and some beyond it
    (
        "gd_cliff_over_gn1",
        deck(
            GD_DIPOLE,
            "GN 1\n" + FR + "EX 0 1 6 0 1\nGD 2 0 0 0 5 .001 .8 .1\n" + GD_RP,
            "GE 1",
        ),
        "printed",
    ),
    (
        "gd_cliff_over_gn1_range",
        deck(
            GD_DIPOLE,
            "GN 1\n" + FR + "EX 0 1 6 0 1\nGD 2 0 0 0 5 .001 .8 -.2\n"
            "RP 0 19 4 1000 0 0 5 45 1000\n",
            "GE 1",
        ),
        "printed",
    ),
    (
        "gd_cliff_over_gn2",
        deck(
            GD_DIPOLE,
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 6 0 1\nGD 2 0 0 0 5 .001 .8 .1\n"
            "RP 0 37 4 1000 0 0 5 45\n",
            "GE 1",
        ),
        "printed",
    ),
    (
        "gd_over_gn3_average",
        deck(
            GD_DIPOLE,
            "GN 3 0 0 0 13 .005\n" + FR + "EX 0 1 6 0 1\nGD 2 0 0 0 20 .02 0 0\n"
            "RP 0 19 2 1001 0 0 5 90\n",
            "GE 1",
        ),
        "printed",
    ),
    (
        "gd_further_tables_over_gn2",
        deck(
            GD_DIPOLE,
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 6 0 1\nGD 2 0 0 0 5 .001 .8 .1\n"
            "RP 0 10 1 1000 0 0 10 0\nNE 0 1 1 2 .1 0 .1 0 0 .1\nRP 0 5 1 1000 0 0 20 0\n",
            "GE 1",
        ),
        "printed",
    ),
    (
        "gd_in_free_space",
        deck(GD_DIPOLE, FR + "EX 0 1 6 0 1\nGD 2 0 0 0 5 .001 .8 .1\n" + RP),
        "printed",
    ),
    (
        "gd_monopole_on_gn1",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n",
            "GN 1\n" + FR + "EX 0 1 1 0 1\nGD 2 0 0 0 5 .001 .3 .05\n" + GD_RP,
            "GE 1",
        ),
        "printed",
    ),
    (
        "gd_empty_second_medium",
        deck(
            GD_DIPOLE,
            "GN 1\n"
            + FR
            + "EX 0 1 6 0 1\nGD 2 0 0 0 0 0 0 0\nRP 0 7 2 1000 0 0 15 90\n",
            "GE 1",
        ),
        "printed",
    ),
    # momwire#1042: a wire listed twice is solved once (the dropped copy
    # reads its twin's current in the tables and none in RP / NE), or refused
    # in the constructor's words
    (
        "duplicate_sparse_ld5_pq_rp",
        deck(
            DUP_DIPOLE + DUP_PARASITE + DUP_PARASITE.replace("GW 2", "GW 3"),
            "LD 5 0 0 0 3.5e7\n"
            + FR
            + "EX 0 1 21 0 1\nPQ 0\nRP 0 7 4 1000 0 0 30 90\n",
        ),
        "printed",
    ),
    (
        "duplicate_reversed_sparse_ne",
        deck(
            DUP_DIPOLE + DUP_PARASITE + "GW 3 31 .2 0 .23 .2 0 -.23 .001\n",
            FR + "EX 0 1 21 0 1\nNE 0 1 3 1 .3 -.1 0 0 .1 0\n",
        ),
        "printed",
    ),
    (
        "duplicate_three_times",
        deck(
            DIPOLE
            + "GW 2 9 .2 0 -.23 .2 0 .23 .001\nGW 3 9 .2 0 -.23 .2 0 .23 .001\n"
            + "GW 4 9 .2 0 .23 .2 0 -.23 .001\n",
            FR + "EX 0 1 6 0 1\nPQ 0\nXQ\n",
        ),
        "printed",
    ),
    (
        "duplicate_by_gm",
        deck(
            DIPOLE + "GW 2 9 .2 0 -.23 .2 0 .23 .001\nGM 0 0 0 0 0 0 0 0 2\n",
            FR + "EX 0 1 6 0 1\nRP 0 7 1 1000 0 0 15 0\n",
        ),
        "printed",
    ),
    (
        "duplicate_tee_arm",
        deck(
            DIPOLE + "GW 2 5 0 0 .25 .2 0 .25 .001\nGW 3 5 0 0 .25 .2 0 .25 .001\n",
            FR + "EX 0 1 6 0 1\nPQ 0\nXQ\n",
        ),
        "printed",
    ),
    (
        "duplicate_monopoles_on_gn1",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\nGW 2 10 .3 0 0 .3 0 .23 .001\n"
            "GW 3 10 .3 0 0 .3 0 .23 .001\n",
            "GN 1\n" + FR + "EX 0 1 1 0 1\nRP 0 10 4 1000 0 0 10 90\n",
            "GE 1",
        ),
        "printed",
    ),
    (
        "duplicate_reversed_over_gn2",
        deck(
            "GW 1 10 0 0 .1 0 0 .35 .001\nGW 2 10 .3 0 .1 .3 0 .33 .001\n"
            "GW 3 10 .3 0 .33 .3 0 .1 .001\n",
            "GN 2 0 0 0 13 .005\n"
            + FR
            + "EX 0 1 5 0 1\nPQ 0\nRP 0 10 4 1000 0 0 10 90\n",
            "GE 1",
        ),
        "printed",
    ),
    (
        "duplicate_fed_refused",
        deck(DIPOLE + DIPOLE.replace("GW 1", "GW 2"), FR + "EX 0 1 6 0 1\nXQ\n"),
        "refused",
    ),
    (
        "duplicate_counts_differ_refused",
        deck(
            DIPOLE + "GW 2 9 .2 0 -.23 .2 0 .23 .001\nGW 3 7 .2 0 .23 .2 0 -.23 .001\n",
            FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "refused",
    ),
    (
        "duplicate_radius_differs_refused",
        deck(
            DIPOLE + "GW 2 9 .2 0 -.23 .2 0 .23 .001\nGW 3 9 .2 0 -.23 .2 0 .23 .002\n",
            FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "refused",
    ),
    (
        "duplicate_ld2_on_the_copy_refused",
        deck(
            DIPOLE + "GW 2 9 .2 0 -.23 .2 0 .23 .001\nGW 3 9 .2 0 -.23 .2 0 .23 .001\n",
            "LD 2 3 0 0 10 1e-7 0\n" + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "refused",
    ),
    (
        "duplicate_ld5_on_the_copy_refused",
        deck(
            DIPOLE + "GW 2 9 .2 0 -.23 .2 0 .23 .001\nGW 3 9 .2 0 -.23 .2 0 .23 .001\n",
            "LD 5 3 0 0 1e7\n" + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "refused",
    ),
    # momwire#1607 B6a: a wholly-buried deck, solved at k_m (anchors off, as
    # the exe audit runs: the below grid's anchors follow process history)
    (
        "buried_dipole_gn2_pq_rp",
        deck(
            BURIED_DIPOLE,
            "GN 2 0 0 0 13 .005\n"
            + BFR
            + "EX 0 1 11 0 1\nPQ 0\nRP 0 10 4 1000 0 0 10 90\n",
            "GE -1",
        ),
        "printed",
    ),
    (
        "buried_dipole_gn3_ld5",
        deck(
            "GW 1 15 -3 0 -.3 3 0 -.3 .002\n",
            "GN 3 0 0 0 20 .01\nLD 5 0 0 0 5.8e7\n" + BFR + "EX 0 1 8 0 1\nXQ\n",
            "GE -1",
        ),
        "printed",
    ),
    (
        "buried_sweep_ex6",
        deck(
            "GW 1 11 -3 0 -.3 3 0 -.3 .001\n",
            "GN 2 0 0 0 13 .005\nFR 0 3 0 0 7 0.5\nEX 6 1 6 0 1 0\nXQ\n",
            "GE -1",
        ),
        "printed",
    ),
    (
        "buried_hub_of_three",
        deck(
            "GW 1 10 0 0 -.3 3 0 -.3 .001\nGW 2 10 0 0 -.3 -3 0 -.3 .001\nGW 3 10 0 0 -.3 0 3 -.3 .001\n",
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 1 1 0 1\nPQ 0\nXQ\n",
            "GE -1",
        ),
        "printed",
    ),
    (
        "buried_two_radii_port_load",
        deck(
            "GW 1 11 -3 0 -.3 3 0 -.3 .001\nGW 2 11 -3 1 -.3 3 1 -.3 .002\n",
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 1 6 0 1\nLD 4 2 6 6 50 0\nXQ\n",
            "GE -1",
        ),
        "printed",
    ),
    (
        "buried_over_pec_refused",
        deck(
            "GW 1 11 -3 0 -.3 3 0 -.3 .001\n",
            "GN 1\n" + BFR + "EX 0 1 6 0 1\nXQ\n",
            "GE -1",
        ),
        "refused",
    ),
    (
        "crossing_two_radii_refused",
        deck(
            "GW 1 10 0 0 0 0 0 5 .002\nGW 2 10 0 0 0 3 0 -.3 .001\n",
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "refused",
    ),
    (
        "crossing_hub_refused",
        deck(
            "GW 1 10 0 0 0 0 0 5 .001\nGW 2 10 0 0 0 3 0 -.3 .001\nGW 3 10 0 0 0 -3 0 -.3 .001\n",
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "refused",
    ),
    (
        "crossing_oblique_refused",
        deck(
            "GW 1 10 0 0 0 3 0 5 .001\nGW 2 10 0 0 0 3 0 -.3 .001\n",
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "refused",
    ),
    (
        "crossing_two_nodes_refused",
        deck(
            "GW 1 10 0 0 0 0 0 5 .001\nGW 2 10 0 0 0 3 0 -.3 .001\n"
            "GW 3 10 5 0 0 5 0 5 .001\nGW 4 10 5 0 0 8 0 -.3 .001\n",
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "refused",
    ),
    (
        "crossing_close_nodes_refused",
        deck(
            "GW 1 10 0 0 0 0 0 5 .001\nGW 2 10 0 0 0 3 0 -.3 .001\n"
            "GW 3 10 .5 0 0 .5 0 5 .001\nGW 4 10 .5 0 0 3 2 -.3 .001\n",
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "refused",
    ),
    (
        "crossing_declines",
        deck(
            "GW 1 10 0 0 0 0 0 5 .001\nGW 2 5 0 0 0 0 0 -1 .001\n",
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "declined",
    ),
    (
        "above_and_below_declines",
        deck(
            "GW 1 11 -3 0 -.3 3 0 -.3 .001\nGW 2 5 5 0 .5 5 0 2 .001\n",
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 2 3 0 1\nXQ\n",
            "GE -1",
        ),
        "declined",
    ),
    (
        "buried_near_field_declines",
        deck(
            BURIED_DIPOLE,
            "GN 2 0 0 0 13 .005\n" + BFR + "EX 0 1 11 0 1\nNE 0 1 1 1 0 0 1 0 0 0\n",
            "GE -1",
        ),
        "declined",
    ),
    ("tl", deck(TWO, FR + "TL 1 6 2 6 50 .1\nEX 0 1 6 0 1\nXQ\n"), "printed"),
    (
        "tl_crossed_shunts_zero_length",
        deck(TWO, FR + "TL 1 6 2 6 -75 0 0 .01 .002 -.003\nEX 0 1 6 0 1\nPQ 0\nXQ\n"),
        "printed",
    ),
    (
        "nt_with_a_port_load",
        deck(
            TWO,
            FR
            + "LD 4 2 3 3 20 -10\nNT 1 6 2 6 0 -.02 0 .01 0 -.02\nEX 0 1 6 0 1\n"
            + RP,
        ),
        "printed",
    ),
    (
        "tl_current_source",
        deck(TWO, FR + "TL 1 6 2 6 50 .2\nEX 6 1 6 0 1 0\nXQ\n"),
        "printed",
    ),
    (
        "tl_phased_two_current_sources",
        deck(
            TWO + "GW 3 11 .4 0 -.25 .4 0 .25 .001\n",
            FR + "TL 1 6 3 6 50 .25\nEX 6 1 6 0 1 0\nEX 6 2 6 0 0 1\nXQ\n",
        ),
        "printed",
    ),
    (
        "tl_read_after_the_first_run",
        deck(TWO, FR + "EX 0 1 6 0 1\nXQ\nTL 1 6 2 6 50 .1\nXQ\n"),
        "printed",
    ),
    (
        "ne_free",
        deck(DIPOLE, FR + "EX 0 1 6 0 1\nNE 0 2 3 2 .1 -.1 -.2 .05 .1 .2\n"),
        "printed",
    ),
    ("nh_free", deck(TWO, FR + "EX 0 1 6 0 1\nNH 0 1 1 3 .1 0 0 0 0 .1\n"), "printed"),
    (
        "ne_spherical_after_rp",
        deck(DIPOLE, FR + "EX 0 1 6 0 1\n" + RP + "NE 1 2 4 3 1 0 0 .5 30 45\n"),
        "printed",
    ),
    (
        "ne_over_pec_monopole",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n",
            "GN 1\n" + FR + "EX 0 1 1 0 1\nNE 0 3 1 2 .05 0 0 .05 0 .1\n",
            "GE 1",
        ),
        "printed",
    ),
    (
        "nh_over_gn2",
        deck(
            "GW 1 11 0 0 .3 0 0 .8 .001\n",
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 6 0 1\nNH 0 2 1 2 .2 0 .1 .1 0 .2\n",
        ),
        "printed",
    ),
    (
        "ne_over_gn3_off_the_contact",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n",
            "GN 3 0 0 0 5 .02\n" + FR + "EX 0 1 1 0 1\nNE 0 2 1 2 .05 0 0 .05 0 .1\n",
            "GE 1",
        ),
        "printed",
    ),
    (
        "ne_on_the_gn2_contact_declines",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n",
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 1 0 1\nNE 0 1 1 1 0 0 0 0 0 0\n",
            "GE 1",
        ),
        "declined",
    ),
    (
        "ne_in_the_soil_declines",
        deck(
            "GW 1 11 0 0 .3 0 0 .8 .001\n",
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 6 0 1\nNE 0 1 1 2 .2 0 -.1 0 0 .2\n",
        ),
        "declined",
    ),
    (
        "helix",
        deck("GH 1 40 3 .3 .05 .05 .001 .001\n", FR + "EX 0 1 1 0 1\nPQ 0\n" + RP),
        "printed",
    ),
    (
        "catenary",
        deck(
            "CW 1 21 -5 0 10 5 0 10 .001 3 10.6 0\n", FR + "EX 0 1 11 0 1\nPQ 0\n" + RP
        ),
        "printed",
    ),
    (
        "catenary_between_wires",
        deck(
            "GW 1 5 -5 0 0 -5 0 10 .001\nCW 2 10 -5 0 10 5 0 10 .001 1 5 8\n"
            "GW 3 5 5 0 10 5 0 0 .001\n",
            FR + "EX 0 1 1 0 1\nXQ\n",
        ),
        "printed",
    ),
    (
        "helix_tapered_beside_a_wire",
        deck(
            "GH 1 30 -2.5 .4 .08 .03 .002 .001\nGW 2 11 .3 0 0 .3 0 .5 .001\n",
            FR + "EX 0 2 6 0 1\nXQ\n",
        ),
        "printed",
    ),
    (
        "phased_current_and_voltage",
        deck(TWO, FR + "EX 6 1 6 0 1 0\nEX 0 2 6 0 0 1\n" + RP),
        "printed",
    ),
    (
        "phased_two_current_sources_loaded",
        deck(
            TWO,
            FR + "LD 4 1 6 6 10 5\nEX 6 1 6 0 1 0\nEX 6 2 6 0 0 -1\nPQ 0\nXQ\n",
        ),
        "printed",
    ),
]


@pytest.mark.parametrize("name, text, expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_prints_the_same(name, text, expect, acc):
    # "refused": printed, and the printout is the reference's refusal
    ok, cxx = rc.cxx_printout(text, acc)
    assert ("declined" if not ok else "printed") == expect.replace(
        "refused", "printed"
    ), cxx[:400]
    if not ok:
        return
    a, na = rc.mask(rc.py_printout(text))
    b, nb = rc.mask(cxx)
    assert rc.first_difference(a, b) is None, rc.first_difference(a, b)
    if expect == "refused":
        assert "NEC ERROR - " in cxx and na == nb == 0, (na, nb)
    else:
        assert na == nb and na >= 2, (na, nb)


def test_the_masks_hide_only_the_two_timing_lines():
    """The mask is an exact line pattern: a FILL or RUN TIME line with
    anything else on it is not masked, and nothing else is."""
    text = (
        "                        FILL=    0.012 SEC.,  FACTOR=    0.000 SEC.\n"
        " RUN TIME =     0.250\n"
        "                        FILL=    0.012 SEC.,  FACTOR=    0.000 SEC. x\n"
        " RUN TIME =     0.250 SEC. (WALL CLOCK)\n"
        "  1.0000E+00  0.012\n"
    )
    masked, n = rc.mask(text)
    assert n == 2
    lines = masked.split("\n")
    assert lines[2:5] == text.split("\n")[2:5]
