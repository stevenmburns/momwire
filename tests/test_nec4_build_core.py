"""The C++ NEC-4.2 build is the Python one, operating point for operating point.

`src/momwire/_nec4_build_core.cpp` copies what the NEC-4.2 slot does
between its seam front and the fill (momwire#1607 A2b). That is
`portal.DeckSolver`'s construction and every `at(freq, ek, environment)` its
runs ask for:
- `prepare_mesh`'s union port set and `to_polylines`' chaining;
- `build_solver`'s constructor arguments, the extended kernel default and
  its advisories, and the port plan;
- DeckSolver's NEC-facing tables, and each fill's sign congruence.

Python stays the reference (`scripts/core_split/nec4_build_compare.py`, which
records the solver class's constructor arguments instead of filling). These
tests hold the C++ dump equal to it over:
- every NEC deck in `tests/fixtures`, for both served bases;
- hand-written decks for the shapes the corpus reaches rarely or never:
  junctions of two, three and four wires, closed loops (the cycle cut), a
  radius step, wires crossing the ground plane, buried and partly buried
  structures (the kernel's fallback and partial advisories), wire loading,
  networks, current sources, several operating points, and the build's own
  refusals. Each case first checks where the PYTHON build lands.
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
        "core_split_nec4_build_compare",
        ROOT / "scripts" / "core_split" / "nec4_build_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _compare_module()


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "nec4_build_dump"):
        pytest.skip("no compiled accelerator with the NEC-4.2 build")
    return _accel.acc


FIXTURE_DECKS = [
    p
    for p in sorted(FIXTURES.rglob("*"))
    if p.is_file() and p.suffix in (".nec", ".deck")
]


@pytest.mark.parametrize("basis", ["sinusoidal", "bspline", "sinusoidal-galerkin"])
def test_every_fixture_deck_builds_the_same(basis, acc):
    counts, mismatches = nc.compare(FIXTURE_DECKS, basis)
    assert not mismatches, mismatches[:3]
    assert counts["built"] >= 200, counts
    assert counts["declined"] <= 3, counts


# --- hand-written decks -----------------------------------------------------

GW = "GW 1 11 0 0 -.25 0 0 .25 .001\n"
FR = "FR 0 1 0 0 299.7925\n"
EX = "EX 0 1 6 0 1 0\n"


def deck(geometry: str = GW, body: str = FR + EX + "XQ\n", ge: str = "GE 0") -> str:
    return "CM build\nCE\n" + geometry + ge + "\n" + body + "EN\n"


SQUARE = (
    "GW 1 4 0 0 0 .2 0 0 .001\nGW 2 4 .2 0 0 .2 .2 0 .001\n"
    "GW 3 4 .2 .2 0 0 .2 0 .001\nGW 4 4 0 .2 0 0 0 0 .001\n"
)
BURIED = "GW 1 11 0 0 -1 0 0 -.5 .001\n"
CROSSING = "GW 1 7 0 0 0 0 0 1 .001\nGW 2 5 0 0 0 0 0 -.6 .001\n"
GN2 = "GN 2 0 0 0 13 .005\n"

CASES: list[tuple[str, str, str]] = [
    ("dipole", deck(), "built"),
    (
        "v_junction",
        deck(
            "GW 1 5 0 0 0 .2 0 .2 .001\nGW 2 5 0 0 0 -.2 0 .2 .001\n",
            body=FR + "EX 0 1 1 0 1\nXQ\n",
        ),
        "built",
    ),
    (
        "t_junction",
        deck(GW + "GW 2 5 0 0 0 .3 0 0 .001\n", body=FR + EX + "XQ\n"),
        "built",
    ),
    (
        "cross_junction",
        deck(GW + "GW 2 5 0 0 0 .3 0 0 .001\nGW 3 5 0 0 0 -.3 0 0 .001\n"),
        "built",
    ),
    ("square_loop", deck(SQUARE, body=FR + "EX 0 1 2 0 1\nXQ\n"), "built"),
    (
        "parasitic_loop",
        deck(GW + SQUARE.replace("GW 1", "GW 5").replace(" 0 0 0 .2", " .5 0 0 .7")),
        "built",
    ),
    (
        "radius_step",
        deck(
            "GW 1 5 0 0 -.25 0 0 0 .001\nGW 2 5 0 0 0 0 0 .25 .002\n",
            body=FR + "EX 0 1 5 0 1\nXQ\n",
        ),
        "built",
    ),
    (
        "load_and_feed_one_segment",
        deck(body="LD 4 1 6 6 50 0\n" + FR + EX + "XQ\n"),
        "built",
    ),
    (
        "loads_per_run",
        deck(body="LD 4 1 3 3 50 0\n" + FR + EX + "XQ\nEX 0 1 5 0 1\nXQ\n"),
        "built",
    ),
    ("ld5_conductivity", deck(body="LD 5 0 0 0 5.8e7\n" + FR + EX + "XQ\n"), "built"),
    ("ld2_distributed", deck(body="LD 2 1 0 0 2 1e-7\n" + FR + EX + "XQ\n"), "built"),
    (
        "networks",
        deck(
            GW + "GW 2 11 .25 0 -.25 .25 0 .25 .001\n",
            body="TL 1 6 2 6 50 .5\nNT 1 4 2 8 0 .01 0 0 0 .02\n" + FR + EX + "XQ\n",
        ),
        "built",
    ),
    (
        "network_on_the_feed",
        deck(
            GW + "GW 2 11 .25 0 -.25 .25 0 .25 .001\n",
            body="TL 1 6 2 6 -50 .5\n" + FR + EX + "XQ\n",
        ),
        "built",
    ),
    (
        "current_sources",
        deck(
            GW + "GW 2 11 .25 0 -.25 .25 0 .25 .001\n",
            body=FR + "EX 6 1 6 0 1 0\nEX 6 2 6 0 0 1\nXQ\n",
        ),
        "built",
    ),
    (
        "pec_ground",
        deck(
            "GW 1 11 0 0 .1 0 0 .6 .001\n", body="GN 1\n" + FR + EX + "XQ\n", ge="GE 1"
        ),
        "built",
    ),
    (
        "sommerfeld_ground",
        deck("GW 1 11 0 0 .1 0 0 .6 .001\n", body=GN2 + FR + EX + "XQ\n", ge="GE 1"),
        "built",
    ),
    (
        "crossing_plane_split",
        deck("GW 1 12 0 0 -.3 0 0 .7 .001\n", body=GN2 + FR + EX + "XQ\n", ge="GE -1"),
        "built",
    ),
    (
        "crossing_junction",
        deck(CROSSING, body=GN2 + FR + "EX 0 1 2 0 1\nXQ\n", ge="GE -1"),
        "built",
    ),
    ("wholly_buried", deck(BURIED, body=GN2 + FR + EX + "XQ\n", ge="GE -1"), "built"),
    ("sweep", deck(body="FR 0 4 0 0 280 10\n" + EX + "XQ\n"), "built"),
    (
        "new_frequency_run",
        deck(body=FR + EX + "XQ\nFR 0 2 0 0 150 1\nRP 0 1 1 1000\n"),
        "built",
    ),
    (
        "source_reruns_share_a_fill",
        deck(body=FR + EX + "XQ\nEX 0 1 4 0 1\nXQ\n"),
        "built",
    ),
    (
        "same_segment_two_addresses",
        deck(body=FR + EX + "XQ\nEX 0 0 6 0 1\nXQ\n"),
        "feed 1 shares its position with feed 0",
    ),
    (
        "zero_frequency",
        deck(body="FR 0 1 0 0 0\n" + EX + "XQ\n"),
        "frequency must be > 0 MHz, got 0.0",
    ),
    (
        "sweep_through_zero",
        deck(body="FR 0 3 0 0 10 -10\n" + EX + "XQ\n"),
        "frequency must be > 0 MHz, got 0.0",
    ),
    ("seam_refusal", deck(body="GN 0 0 0 0 13 .005\n" + FR + EX + "XQ\n"), "seam"),
    ("chain", deck() + "NX\n" + deck(body="FR 0 1 0 0 150\n" + EX + "XQ\n"), "built"),
]


def _python_lands(text: str, basis: str) -> str:
    dump = nc.py_build(text, basis)
    kind = nc.classify(dump)
    if kind == "refused":
        line = next(
            x
            for x in dump.splitlines()
            if x.startswith("build.structures[0] = refused")
        )
        return bytes.fromhex(line.rsplit(" ", 1)[1]).decode("utf-8")
    return kind


@pytest.mark.parametrize("basis", ["sinusoidal", "bspline", "sinusoidal-galerkin"])
@pytest.mark.parametrize("name, text, expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_builds_the_same(name, text, expect, basis, acc):
    lands = _python_lands(text, basis)
    if expect in ("built", "seam"):
        assert lands == expect, lands
    else:
        assert expect in lands, lands
    diff = nc.dc.first_difference(
        nc.py_build(text, basis), nc.cxx_build(text, basis, acc)
    )
    assert diff is None, diff


@pytest.mark.parametrize(
    "name, basis, fragment",
    [
        (
            "wholly_buried",
            "sinusoidal",
            "cannot serve here, so it solved with the reduced kernel",
        ),
        (
            "wholly_buried",
            "bspline",
            "every wire of this deck is below the ground plane",
        ),
        (
            "crossing_junction",
            "bspline",
            "1 of 2 wires below the plane kept the reduced kernel",
        ),
    ],
)
def test_the_kernel_advisories_are_reached(name, basis, fragment, acc):
    """Not vacuous: the buried cases above do reach the extended kernel's
    fallback and partial advisories, in the C++ dump as in Python's."""
    text = next(c[1] for c in CASES if c[0] == name)
    dump = nc.cxx_build(text, basis, acc)
    hexed = fragment.encode("utf-8").hex()
    assert any(
        ".advisories[0] = str " in line and hexed in line for line in dump.splitlines()
    )


def test_a_sweep_reaches_several_operating_points(acc):
    text = next(c[1] for c in CASES if c[0] == "sweep")
    assert "build.structures[0].points : list 4" in nc.cxx_build(
        text, "sinusoidal", acc
    )
