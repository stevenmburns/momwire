"""The C++ sinusoidal front half is `SinusoidalSolver`'s, to the bit.

`src/momwire/_sinusoidal_front_core.cpp` copies what the solver derives from
its constructor arguments before any fill (momwire#1607 B2): the constants,
the junction normalisation, `_build_geometry` and `_basis_coefs(geom, k)`.
`scripts/core_split/sinusoidal_front_compare.py` feeds the REAL solver the
arguments the NEC-4.2 build records at every operating point of a deck, and
these tests hold the C++ dump equal to it over:
- the NEC-4.2 fixtures and the whole fixture tree;
- hand-written decks for each shape `_build_geometry` and `_basis_coefs`
  branch on: in-wire neighbours only, V / T / cross junctions (N- and N+
  with both sigma signs), a closed loop, a radius step (the per-segment
  `a_const`), a free end at each side, a wire end on a perfect ground and a
  grounded junction (`ground_minus` / `ground_plus` and their P-sum atoms),
  the Sommerfeld ground above the plane, an FR sweep (one point per
  frequency), and the segment-in-the-plane refusal;
- the declines: a buried wire (momwire#1607 B6) and duplicated wires.
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
        "core_split_sinusoidal_front_compare",
        ROOT / "scripts" / "core_split" / "sinusoidal_front_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _compare_module()


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "sinusoidal_front_dump"):
        pytest.skip("no compiled accelerator with the sinusoidal front half")
    return _accel.acc


def _decks(root):
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]


def test_the_nec42_fixtures_have_the_same_front(acc):
    counts, mismatches = nc.compare(_decks(FIXTURES / "eznec_nec42"))
    assert not mismatches, mismatches[:2]
    assert counts["fronts"] >= 50, counts


def test_every_fixture_deck_has_the_same_front(acc):
    counts, mismatches = nc.compare(_decks(FIXTURES))
    assert not mismatches, mismatches[:2]
    assert counts["fronts"] >= 200, counts


@pytest.mark.parametrize("rule", [None, "nec2", "nec42"])
def test_the_radius_step_decks_have_the_same_front(rule, acc, monkeypatch):
    # momwire#1644: the per-end basis under both junction rules
    if rule is None:
        monkeypatch.delenv("MOMWIRE_JUNCTION_RULE", raising=False)
    else:
        monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", rule)
    counts, mismatches = nc.compare(_decks(FIXTURES / "nec42_radius_step_1644"))
    assert not mismatches, mismatches[:2]
    assert counts["fronts"] >= 80, counts


FR = "FR 0 1 0 0 299.7925\n"


def deck(geometry: str, body: str, ge: str = "GE 0") -> str:
    return "CM front\nCE\n" + geometry + ge + "\n" + body + "EN\n"


DIPOLE = "GW 1 11 0 0 -.25 0 0 .25 .001\n"
# An even count puts a segment boundary at the origin, where NEC connects.
EVEN = "GW 1 10 0 0 -.25 0 0 .25 .001\n"
SQUARE = (
    "GW 1 4 0 0 0 .2 0 0 .001\nGW 2 4 .2 0 0 .2 .2 0 .001\n"
    "GW 3 4 .2 .2 0 0 .2 0 .001\nGW 4 4 0 .2 0 0 0 0 .001\n"
)

CASES = [
    ("dipole", deck(DIPOLE, FR + "EX 0 1 6 0 1\nXQ\n"), "fronts"),
    (
        "v_junction",
        deck(
            "GW 1 5 0 0 0 .2 0 .2 .001\nGW 2 5 0 0 0 -.2 0 .2 .001\n",
            FR + "EX 0 1 1 0 1\nXQ\n",
        ),
        "fronts",
    ),
    (
        "t_junction",
        deck(EVEN + "GW 2 5 0 0 0 .3 0 0 .001\n", FR + "EX 0 1 5 0 1\nXQ\n"),
        "fronts",
    ),
    (
        "cross_junction",
        deck(
            EVEN + "GW 2 5 0 0 0 .3 0 0 .001\nGW 3 5 0 0 0 -.3 0 0 .001\n",
            FR + "EX 0 1 5 0 1\nXQ\n",
        ),
        "fronts",
    ),
    (
        "end_to_end",
        deck(
            "GW 1 5 0 0 -.25 0 0 0 .001\nGW 2 5 0 0 .25 0 0 0 .001\n",
            FR + "EX 0 1 5 0 1\nXQ\n",
        ),
        "fronts",
    ),
    ("square_loop", deck(SQUARE, FR + "EX 0 1 2 0 1\nXQ\n"), "fronts"),
    (
        "radius_step",
        deck(
            "GW 1 5 0 0 -.25 0 0 0 .001\nGW 2 5 0 0 0 0 0 .25 .002\n",
            FR + "EX 0 1 5 0 1\nXQ\n",
        ),
        "fronts",
    ),
    (
        "single_segment_wire",
        deck(DIPOLE + "GW 2 1 .3 0 -.05 .3 0 .05 .001\n", FR + "EX 0 1 6 0 1\nXQ\n"),
        "fronts",
    ),
    (
        "monopole_on_pec",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n",
            "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n",
            ge="GE 1",
        ),
        "fronts",
    ),
    (
        "grounded_junction",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\nGW 2 5 0 0 0 .2 0 .1 .001\nGW 3 5 0 0 0 -.2 0 .1 .001\n",
            "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n",
            ge="GE 1",
        ),
        "fronts",
    ),
    (
        "sommerfeld_above",
        deck(
            "GW 1 11 0 0 .1 0 0 .6 .001\n",
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "fronts",
    ),
    ("sweep", deck(DIPOLE, "FR 0 4 0 0 250 25\nEX 0 1 6 0 1\nXQ\n"), "fronts"),
    (
        "loaded",
        deck(DIPOLE, "LD 4 1 3 3 50 0\nLD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fronts",
    ),
    (
        "buried_declines",
        deck(
            "GW 1 11 0 0 -1 0 0 -.5 .001\n",
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 6 0 1\nXQ\n",
            ge="GE -1",
        ),
        "declined",
    ),
    (
        "duplicate_declines",
        deck(DIPOLE + "GW 2 11 0 0 -.25 0 0 .25 .001\n", FR + "EX 0 1 6 0 1\nXQ\n"),
        "declined",
    ),
]


@pytest.mark.parametrize("name, text, expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_has_the_same_front(name, text, expect, acc):
    cxx = nc.cxx_front(text, acc)
    assert nc.classify(cxx) == expect, cxx[:500]
    if expect == "declined":
        return
    diff = nc.dc.first_difference(nc.py_front(text), cxx)
    assert diff is None, diff


def _values(dump: str, name: str) -> list[str]:
    """The first point's values under geometry/basis key `name`, as dumped."""
    lines = dump.splitlines()
    hexed = name.encode().hex()
    for line in lines:
        if line.startswith("front.structures[0].points[0].") and line.endswith(
            f".key = str {hexed}"
        ):
            prefix = line[: -len(f".key = str {hexed}")] + ".value["
            return [x.rsplit(" ", 1)[1] for x in lines if x.startswith(prefix)]
    raise AssertionError(f"no key {name!r}")


def test_the_branches_are_reached(acc):
    """Not vacuous: both sigma signs, a ground flag on each side, a grounded
    junction and several operating points appear in the C++ dumps."""
    dumps = {name: nc.cxx_front(text, acc) for name, text, _e in CASES}
    # (a V and two collinear wires chain into one polyline at a degree-2
    # node; the signs come from the three- and four-way junctions)
    signs = [
        v
        for name in ("t_junction", "cross_junction")
        for key in ("nm_sigma", "np_sigma")
        for v in _values(dumps[name], key)
    ]
    assert "-1" in signs and "1" in signs
    assert "True" in _values(dumps["monopole_on_pec"], "ground_minus")
    assert _values(dumps["grounded_junction"], "grounded_junctions")
    assert "True" in _values(dumps["grounded_junction"], "ground_minus")
    assert ".points : list 4" in dumps["sweep"]


def test_a_segment_in_the_plane_is_refused_as_python_refuses(acc):
    text = deck(
        "GW 1 5 0 0 .5 0 0 .1 .001\nGW 2 5 0 0 .1 .3 0 .1 .001\nGW 3 4 .3 0 .1 .3 0 0 .001\n"
        "GW 4 4 .3 0 0 .6 0 0 .001\n",
        "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n",
        ge="GE 1",
    )
    cxx = nc.cxx_front(text, acc)
    py = nc.py_front(text)
    assert nc.dc.first_difference(py, cxx) is None
    if " ! ValueError " not in cxx:
        pytest.skip("the deck reader or build refuses this geometry before the front")
