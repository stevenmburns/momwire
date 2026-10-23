"""The C++ NEC-4.2 seam front is the Python one, structure for structure.

`src/momwire/_nec4_seam_core.cpp` copies what the NEC-4.2 slot does with a
deck before the solve (momwire#1607 A2). That is `eznec/_nec4.py`'s
`render`:
- the basis gate and the NX chain split;
- `portal._portal.parse_deck`'s view of the cards;
- `_further_tables` and `_served_shape`.

Python stays the reference. `scripts/core_split/nec4_seam_compare.py` drives
the slot's real control flow with the solve stubbed out, and these tests hold
the C++ dump equal to it on:
- every NEC deck in `tests/fixtures`, for both served bases;
- hand-written decks, one or more per refusal `_served_shape` and
  `_served_geometry` name, and per path through a chain. Each case first
  checks where the PYTHON front lands (served, the named refusal, or an
  exception that escapes it), so that a case cannot silently stop exercising
  its branch.
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
        "core_split_nec4_seam_compare",
        ROOT / "scripts" / "core_split" / "nec4_seam_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _compare_module()


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "nec4_seam_dump"):
        pytest.skip("no compiled accelerator with the NEC-4.2 seam front")
    return _accel.acc


FIXTURE_DECKS = [
    p
    for p in sorted(FIXTURES.rglob("*"))
    if p.is_file() and p.suffix in (".nec", ".deck")
]


@pytest.mark.parametrize("basis", ["sinusoidal", "bspline"])
def test_every_fixture_deck_reaches_the_same_seam(basis, acc):
    counts, mismatches = nc.compare(FIXTURE_DECKS, basis)
    assert not mismatches, mismatches[:3]
    assert counts["served"] >= 200 and counts["refused"] >= 100, counts
    assert counts["declined"] <= 3, counts


# --- hand-written decks -----------------------------------------------------

GW = "GW 1 11 0 0 -.25 0 0 .25 .001\n"
FR = "FR 0 1 0 0 299.7925\n"
EX = "EX 0 1 6 0 1 0\n"


def deck(geometry: str = GW, body: str = FR + EX + "XQ\n", ge: str = "GE 0") -> str:
    return "CM a dipole\nCE\n" + geometry + ge + "\n" + body + "EN\n"


SERVED = "served"

CASES: list[tuple[str, str, str]] = [
    # (name, deck, "served" / a fragment of the refusal / "raised:<Type>")
    ("dipole", deck(), SERVED),
    ("dipole_rp", deck(body=FR + EX + "RP 0 1 361 1000 90 0 0 1\n"), SERVED),
    ("ex6_current_source", deck(body=FR + "EX 6 1 6 0 1.414214 0\nXQ\n"), SERVED),
    (
        "gn_nofile_trailer",
        deck(body="GN 2 0 0 0 13 .005 NOFILE\n" + FR + EX + "XQ\n"),
        SERVED,
    ),
    (
        "gn1_gd2",
        deck(body="GN 1\nGD 2 0 0 0 13 .005 10 -1\n" + FR + EX + "XQ\n"),
        SERVED,
    ),
    (
        "gd1",
        deck(body="GN 1\nGD 1 0 0 0 13 .005 10 -1\n" + FR + EX + "XQ\n"),
        "GD 1 is not",
    ),
    (
        "gn0_refl",
        deck(body="GN 0 0 0 0 13 .005\n" + FR + EX + "XQ\n"),
        "the GN 0 reflection-coefficient ground",
    ),
    (
        "ld_cleared",
        deck(body="LD 4 1 2 2 50 0\nLD -1\nLD 4 1 3 3 20 5\n" + FR + EX + "XQ\n"),
        SERVED,
    ),
    ("gm_then_gs", deck(GW + "GM 0 0 0 0 30 0 0 .1 0\nGS 0 0 .5\n"), SERVED),
    # GX (momwire#1651): served among the wire cards and beside a GM, and
    # the symmetry's cell and flag reach the printout; GR still refuses.
    (
        "gx_geometry",
        deck(
            "GW 1 5 0 .1 -.25 0 .1 .25 .001\nGX 10 010\n",
            body=FR + "EX 0 1 3 0 1\nXQ\n",
        ),
        SERVED,
    ),
    (
        "gx_three_planes_then_gm",
        deck(
            "GW 1 5 .1 .1 .05 .1 .3 .3 .001\nGX 100 111\nGM 0 0 0 0 0 0 0 1 0\n",
            body=FR + "EX 0 101 3 0 1\nXQ\n",
        ),
        SERVED,
    ),
    (
        "gx_then_gw",
        deck(
            "GW 1 5 .1 0 -.25 .1 0 .25 .001\nGX 10 100\nGW 3 5 0 .2 -.25 0 .2 .25 .001\n",
            body=FR + "EX 0 3 3 0 1\nXQ\n",
        ),
        SERVED,
    ),
    (
        "gx_after_gs",
        deck(
            "GW 1 5 .1 0 -.25 .1 0 .25 .001\nGS 0 0 .5\nGX 10 100\n",
            body=FR + "EX 0 1 3 0 1\nXQ\n",
        ),
        "a GX card after a GS card",
    ),
    (
        "gr_geometry",
        deck(
            "GW 1 5 .1 0 -.25 .1 0 .25 .001\nGR 10 4\n",
            body=FR + "EX 0 1 3 0 1\nXQ\n",
        ),
        "the GR geometry card",
    ),
    ("gs_ranged", deck(GW + "GS 1 1 .5\n"), "GS scaling a range of tags"),
    (
        "second_gm",
        deck(GW + "GM 0 0 0 0 10 0 0 0 0\nGM 0 0 0 0 10 0 0 0 0\n"),
        "a second GM card",
    ),
    (
        "gw_after_gm",
        deck(GW + "GM 0 0 0 0 10 0 0 0 0\nGW 2 3 1 1 0 1 1 .5 .001\n"),
        "a GW card after a GS or GM card",
    ),
    (
        "gm_after_gs",
        deck(GW + "GS 0 0 .5\nGM 0 0 0 0 10 0 0 0 0\n"),
        "a GM card after a GS card",
    ),
    # GA and GH (momwire#1592): served where a GW may stand.
    ("ga_arc", deck("GA 1 12 2 0 180 .001\n"), SERVED),
    ("gh_helix", deck("GH 1 24 1 .3 .15 .15 .002 .002\n"), SERVED),
    (
        "ga_gw_gm",
        deck("GA 1 12 2 0 180 .001\nGW 2 9 -2 0 0 2 0 0 .001\nGM 0 0 0 0 0 0 0 5 0\n"),
        SERVED,
    ),
    (
        "ga_after_gm",
        deck(GW + "GM 0 0 0 0 10 0 0 0 0\nGA 2 4 1 0 90 .001\n"),
        "a GA card after a GS or GM card",
    ),
    (
        "gh_after_gs",
        deck(GW + "GS 0 0 .5\nGH 2 8 1 .3 .1 .1 .001 .001\n"),
        "a GH card after a GS or GM card",
    ),
    (
        "is_card",
        deck(body="IS 0 1 0 0 3.5 0 .002\n" + FR + EX + "XQ\n"),
        "the IS card is not",
    ),
    ("mp_card", deck(body="MP 4 32\n" + FR + EX + "XQ\n"), "the MP card is not"),
    ("pt_card", deck(body="PT -1\n" + FR + EX + "XQ\n"), "the PT card is not"),
    ("no_execute_card", deck(body=FR + EX), "a deck with no execute card"),
    (
        "execute_before_ge",
        "CM x\nCE\n" + GW + "XQ\nGE 0\n" + FR + EX + "XQ\nEN\n",
        "execute cards against",
    ),
    (
        "source_only_reruns",
        deck(body=FR + EX + "XQ\nEX 0 1 5 0 1\nXQ\nEX 0 1 7 0 2\nRP 0 1 1 1000\n"),
        SERVED,
    ),
    (
        "new_frequency_run",
        deck(body=FR + EX + "XQ\nFR 0 3 0 0 300 1\nRP 0 1 1 1000\n"),
        SERVED,
    ),
    (
        "sweep_then_run",
        deck(body="FR 0 3 0 0 300 1\n" + EX + "XQ\nEX 0 1 5 0 1\nXQ\n"),
        "several execute cards over a frequency sweep",
    ),
    (
        "ground_between_runs",
        deck(body=FR + EX + "XQ\nGN 1\nXQ\n"),
        "a later execute card after a card that refills the matrix",
    ),
    (
        "load_between_runs",
        deck(body=FR + EX + "XQ\nLD 4 1 2 2 50 0\nFR 0 1 0 0 300\nXQ\n"),
        "a later execute card after a card that refills the matrix",
    ),
    ("pq_unrestricted", deck(body="PQ 0\n" + FR + EX + "XQ\n"), SERVED),
    (
        "pq_restricted",
        deck(body="PQ 0 1 2 3\n" + FR + EX + "XQ\n"),
        "a PQ card restricted to a segment range",
    ),
    ("pq_off_restricted", deck(body="PQ -1 1 2 3\n" + FR + EX + "XQ\n"), SERVED),
    ("rp_xnda_1005", deck(body=FR + EX + "RP 0 1 1 1005\n"), "RP with XNDA 1005"),
    ("rp_xnda_1012", deck(body=FR + EX + "RP 0 1 1 1012\n"), SERVED),
    ("rp_xnda_1013", deck(body=FR + EX + "RP 0 1 1 1013\n"), SERVED),
    ("rp_xnda_1014", deck(body=FR + EX + "RP 0 1 1 1014\n"), "RP with XNDA 1014"),
    (
        "rp_negative_range",
        deck(body=FR + EX + "RP 0 1 1 1000 0 0 0 0 -5\n"),
        "RP at a negative range",
    ),
    (
        "rp_cliff_mode",
        deck(body="GN 1\nGD 2 0 0 0 13 .005 10 -1\n" + FR + EX + "RP 2 1 1 1000\n"),
        "RP mode 2",
    ),
    ("ne_spherical", deck(body=FR + EX + "NE 1 2 3 4 1 0 0 .1 10 10\n"), SERVED),
    (
        "further_tables",
        deck(body=FR + EX + "XQ\nRP 0 1 1 1000\nNE 0 1 1 1\nXQ\nRP 0 1 1 1001\n"),
        SERVED,
    ),
    (
        "further_bad_xnda",
        deck(body=FR + EX + "XQ\nRP 0 1 1 1005\n"),
        "RP with XNDA 1005",
    ),
    (
        "further_after_sweep",
        deck(body="FR 0 3 0 0 300 1\n" + EX + "XQ\nRP 0 1 1 1000\n"),
        "a further RP card after a frequency sweep's execute card",
    ),
    (
        "further_after_gd",
        deck(body="GN 1\n" + FR + EX + "XQ\nGD 2 0 0 0 13 .005 10 -1\nRP 0 1 1 1000\n"),
        "a further RP card after a GD card read since the run it follows",
    ),
    (
        "network_cards",
        deck(
            GW + "GW 2 11 .25 0 -.25 .25 0 .25 .001\n",
            body="TL 1 6 2 6 50 .5\nNT 1 5 2 5 0 .01 0 0 0 .02\n" + FR + EX + "XQ\n",
        ),
        SERVED,
    ),
    ("reader_refusal", deck(body=FR + "EX 4 1 6 0 1\nXQ\n"), "EX 4 is not served"),
    (
        "separators_after_ge_raise",
        deck(body=FR + EX + ", ,\nXQ\n"),
        "raised:IndexError",
    ),
    (
        "chain_both_served",
        "CM a\nCE\n"
        + GW
        + "GE 0\n"
        + FR
        + EX
        + "XQ\nNX\nCM b\nCE\n"
        + GW
        + "GE 0\nFR 0 1 0 0 150\n"
        + EX
        + "XQ\nEN\n",
        SERVED,
    ),
    (
        "chain_second_refused",
        "CM a\nCE\n"
        + GW
        + "GE 0\n"
        + FR
        + EX
        + "XQ\nNX\nCM b\nCE\n"
        + GW
        + "GE 0\nGN 0 0 0 0 13 .005\n"
        + FR
        + EX
        + "XQ\nNX\nCM c\nCE\n"
        + GW
        + "GE 0\n"
        + FR
        + EX
        + "XQ\nEN\n",
        SERVED,
    ),
    (
        "chain_first_refused",
        "CM a\nCE\n"
        + GW
        + "GE 0\n"
        + FR
        + "EX 4 1 6 0 1\nXQ\nNX\nCM b\nCE\n"
        + GW
        + "GE 0\n"
        + FR
        + EX
        + "XQ\nEN\n",
        "EX 4 is not served by this engine's nec4 dialect",
    ),
    ("chain_open_end", "CM a\nCE\n" + GW + "GE 0\n" + FR + EX + "XQ\nNX\n", SERVED),
    ("text_after_en", deck() + "garbage ' \xe9 after EN\n", SERVED),
]


def _python_lands(text: str, basis: str) -> str:
    """Where the Python front lands on the FIRST structure."""
    dump = nc.py_seam(text, basis)
    kind = nc.classify(dump)
    if kind == "raised":
        line = next(x for x in dump.splitlines() if " ! " in x)
        return "raised:" + line.split(" ! ", 1)[1].split(" ", 1)[0]
    if kind == "refused":
        line = next(x for x in dump.splitlines() if " = refused " in x)
        return bytes.fromhex(line.rsplit(" ", 1)[1]).decode("utf-8")
    return kind


@pytest.mark.parametrize("basis", ["sinusoidal", "bspline"])
@pytest.mark.parametrize("name, text, expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_reaches_the_same_seam(name, text, expect, basis, acc):
    lands = _python_lands(text, basis)
    if expect in (SERVED,) or expect.startswith("raised:"):
        assert lands == expect, lands
    else:
        assert expect in lands, lands
    diff = nc.dc.first_difference(
        nc.py_seam(text, basis), nc.cxx_seam(text, basis, acc)
    )
    assert diff is None, diff


def test_a_chain_stops_at_its_first_refusal(acc):
    """Not vacuous: the second chained case refuses its SECOND structure and
    the dump says so, while the third structure is never reached."""
    text = next(c[1] for c in CASES if c[0] == "chain_second_refused")
    dump = nc.cxx_seam(text, "sinusoidal", acc)
    assert "seam.structures : list 3" in dump
    assert "seam.structures[1] = refused" in dump
    assert "seam.structures[2]" not in dump


def test_a_basis_this_copy_does_not_serve_is_declined(acc):
    assert " ? declined " in nc.cxx_seam(deck(), "razor-2p", acc)
