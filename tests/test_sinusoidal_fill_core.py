"""The C++ sinusoidal free-space fill and solve is `SinusoidalSolver`'s, to
the bit.

`src/momwire/_sinusoidal_fill_core.cpp` copies `compute_port_solution` in
free space (momwire#1607 B3): `_assemble_Z` (the field tensors from B1's
kernels, numpy's zgemm reduction below `_DENSE_ASSEMBLY_THRESHOLD` and
scipy's CSR tile sums above it), `scipy.linalg.solve(G, B)` and the feed
currents into Y. `scripts/core_split/sinusoidal_fill_compare.py` runs the
REAL solver at every operating point the NEC-4.2 build records, and these
tests hold the C++ dump (Z, the coefficients, Y) equal to it over:
- the NEC-4.2 fixtures;
- hand-written decks on each side of the dense threshold (59 / 60
  segments), a ragged last reduction tile (65), a radius step on both
  routes, junctions and a loop, two sources, LD 4 loads as ports, an FR
  sweep, a one-segment structure (scipy's scalar route), and the folded
  cos-1 fill (numpy's route) below and above numpy's temporary elision and
  across a radius step;
- a perfect ground (GN 1): a monopole, a dipole above it on the sparse
  route, and the folded cos-1 fill with its image above the elision size;
- wire loading: LD 5 on every wire and on one, LD 2 / LD 3 per metre, and
  LD 5 over a perfect ground on the cos-1 route;
- a Sommerfeld ground (GN 2 / GN 3) above the plane: a vertical and a
  horizontal wire on the sparse route;
- a wire end on a Sommerfeld ground (the contact correction): a vertical
  monopole (its extended end's delta) and a slanted wire on the sparse route.
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
        "core_split_sinusoidal_fill_compare",
        ROOT / "scripts" / "core_split" / "sinusoidal_fill_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _compare_module()


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "sinusoidal_fill_dump"):
        pytest.skip("no compiled accelerator with the sinusoidal fill")
    try:
        nc.blas_addresses()
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    return _accel.acc


@pytest.fixture(autouse=True)
def _one_blas_thread():
    """One BLAS thread on both sides: OpenBLAS's LU is deterministic at a
    given width, not across widths."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


@pytest.fixture(autouse=True)
def _below_anchors_off(monkeypatch):
    # the below grid's anchors follow process history; the gate is the exact grid
    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")


def _decks(root):
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]


@pytest.mark.slow  # median over the 5 s default-lane budget at 0.84.2
def test_the_nec42_fixtures_have_the_same_fill(acc):
    counts, mismatches = nc.compare(_decks(FIXTURES / "eznec_nec42"))
    assert not mismatches, mismatches[:2]
    assert counts["fills"] >= 30, counts


# momwire#1644: the NEC-4 dialect's radius-step junction (the junction solve's
# per-end basis, the NEC-4 kernel, the end caps and their images), on the
# fixture decks small enough for the default lane.
_STEP_1644 = sorted(
    p
    for p in (FIXTURES / "nec42_radius_step_1644" / "decks").glob("*.nec")
    if not any(t in p.stem for t in ("_n80", "_n160", "_x4", "_x8"))
)


@pytest.mark.parametrize("rule", [None, "nec2", "nec42"])
def test_the_radius_step_decks_have_the_same_fill(rule, acc, monkeypatch):
    # both junction rules (momwire#1644): the dialect's default, and each
    # spelled through MOMWIRE_JUNCTION_RULE, which both copies read
    if rule is None:
        monkeypatch.delenv("MOMWIRE_JUNCTION_RULE", raising=False)
    else:
        monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", rule)
    counts, mismatches = nc.compare(_STEP_1644)
    assert not mismatches, mismatches[:2]
    assert counts["fills"] >= 40, counts


FR = "FR 0 1 0 0 299.7925\n"
LF = "FR 0 1 0 0 .001\n"  # k·h ≈ 1e-6: the folded cos-1 shape


def deck(geometry: str, body: str, ge: str = "GE 0") -> str:
    return "CM fill\nCE\n" + geometry + ge + "\n" + body + "EN\n"


def dipole(n: int, half: float = 0.25, r: float = 0.001) -> str:
    return f"GW 1 {n} 0 0 {-half} 0 0 {half} {r}\n"


EVEN = "GW 1 10 0 0 -.25 0 0 .25 .001\n"
SQUARE = (
    "GW 1 4 0 0 0 .2 0 0 .001\nGW 2 4 .2 0 0 .2 .2 0 .001\n"
    "GW 3 4 .2 .2 0 0 .2 0 .001\nGW 4 4 0 .2 0 0 0 0 .001\n"
)

CASES = [
    ("dipole_dense", deck(dipole(11), FR + "EX 0 1 6 0 1\nXQ\n"), "fills"),
    ("dense_59", deck(dipole(59), FR + "EX 0 1 30 0 1\nXQ\n"), "fills"),
    ("sparse_60", deck(dipole(60), FR + "EX 0 1 30 0 1\nXQ\n"), "fills"),
    ("ragged_tile_65", deck(dipole(65), FR + "EX 0 1 33 0 1\nXQ\n"), "fills"),
    ("sparse_161", deck(dipole(161, 0.5), FR + "EX 0 1 81 0 1\nXQ\n"), "fills"),
    (
        "radius_step_dense",
        deck(
            "GW 1 5 0 0 -.25 0 0 0 .001\nGW 2 5 0 0 0 0 0 .25 .002\n",
            FR + "EX 0 1 5 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "radius_step_sparse",
        deck(
            "GW 1 40 0 0 -.25 0 0 0 .001\nGW 2 30 0 0 0 0 0 .25 .002\n"
            "GW 3 20 .1 0 -.25 .1 0 .25 .001\n",
            FR + "EX 0 1 40 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "t_junction",
        deck(EVEN + "GW 2 5 0 0 0 .3 0 0 .001\n", FR + "EX 0 1 5 0 1\nXQ\n"),
        "fills",
    ),
    ("square_loop", deck(SQUARE, FR + "EX 0 1 2 0 1\nXQ\n"), "fills"),
    (
        "two_sources",
        deck(
            dipole(11) + "GW 2 11 .2 0 -.25 .2 0 .25 .001\n",
            FR + "EX 0 1 6 0 1\nEX 0 2 6 0 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "ld4_ports",
        deck(dipole(11), "LD 4 1 3 3 50 0\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    ("sweep", deck(dipole(11), "FR 0 4 0 0 250 25\nEX 0 1 6 0 1\nXQ\n"), "fills"),
    ("one_segment", deck(dipole(1, 0.05), FR + "EX 0 1 1 0 1\nXQ\n"), "fills"),
    # The folded cos-1 fill (min k·h below 1.5e-3): numpy's route, element by
    # element, on each side of numpy's temporary elision (M·N = 16384).
    ("cos_minus_one", deck(dipole(11), LF + "EX 0 1 6 0 1\nXQ\n"), "fills"),
    ("cos_minus_one_elided", deck(dipole(131), LF + "EX 0 1 66 0 1\nXQ\n"), "fills"),
    (
        "cos_minus_one_radius_step",
        deck(
            "GW 1 6 0 0 -.25 0 0 0 .001\nGW 2 6 0 0 0 0 0 .25 .02\n"
            "GW 3 5 0 0 0 .3 0 0 .001\n",
            LF + "EX 0 1 6 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "monopole_on_pec",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n",
            "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n",
            ge="GE 1",
        ),
        "fills",
    ),
    (
        "dipole_over_pec_sparse",
        deck(
            dipole(81, 0.5).replace("0 0 -0.5 0 0 0.5", "0 0 1 0 0 2"),
            "GN 1\n" + FR + "EX 0 1 41 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "pec_cos_minus_one_elided",
        deck(
            dipole(131).replace("0 0 -0.25 0 0 0.25", "0 0 .1 0 0 .6"),
            "GN 1\n" + LF + "EX 0 1 66 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "dipole_over_gn2",
        deck(
            "GW 1 11 0 0 .1 0 0 .6 .001\n",
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "horizontal_over_gn3_sparse",
        deck(
            "GW 1 61 -.5 0 .3 .5 0 .3 .001\n",
            "GN 3 0 0 0 5 .02\n" + FR + "EX 0 1 31 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "monopole_on_gn2_contact",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n",
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 1 0 1\nXQ\n",
            ge="GE 1",
        ),
        "fills",
    ),
    (
        "slant_on_gn3_contact_sparse",
        deck(
            "GW 1 70 0 0 0 .3 0 .6 .001\n",
            "GN 3 0 0 0 5 .02\n" + FR + "EX 0 1 1 0 1\nXQ\n",
            ge="GE 1",
        ),
        "fills",
    ),
    (
        "ld5_copper",
        deck(dipole(11), "LD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    (
        "ld5_one_wire_sparse",
        deck(
            dipole(61, 0.5) + "GW 2 11 .3 0 -.25 .3 0 .25 .001\n",
            "LD 5 2 0 0 3.5e7\n" + FR + "EX 0 1 31 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "ld2_series_per_metre",
        deck(dipole(11), "LD 2 1 0 0 10 1e-7 0\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    (
        "ld3_parallel_per_metre",
        deck(dipole(11), "LD 3 1 0 0 1e4 1e-6 0\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    (
        "ld5_over_pec_cos_minus_one",
        deck(
            dipole(31).replace("0 0 -0.25 0 0 0.25", "0 0 .1 0 0 .6"),
            "GN 1\nLD 5 0 0 0 5.8e7\n" + LF + "EX 0 1 16 0 1\nXQ\n",
        ),
        "fills",
    ),
    # momwire#1042: a parasite listed twice leaves a (segment, basis) pair
    # repeated; the dense M keeps the last value, scipy's CSC sums the two,
    # and the loading's `G[rows, cols] -= v` keeps the last difference
    (
        "duplicate_dense_ld5",
        deck(
            dipole(11)
            + "GW 2 9 .2 0 -.23 .2 0 .23 .001\nGW 3 9 .2 0 -.23 .2 0 .23 .001\n",
            "LD 5 0 0 0 3.5e7\n" + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "fills",
    ),
    # momwire#1607 B6a: a wholly-buried deck at k_m, dense and sparse
    (
        "buried_dense",
        deck(
            dipole(21, 5.0).replace("0 0 -5.0 0 0 5.0", "-5 0 -.5 5 0 -.5"),
            "GN 2 0 0 0 13 .005\nFR 0 1 0 0 7.15\nEX 0 1 11 0 1\nXQ\n",
            "GE -1",
        ),
        "fills",
    ),
    (
        "buried_sparse_ld5",
        deck(
            "GW 1 71 -5 0 -.5 5 0 -.5 .001\n",
            "GN 3 0 0 0 20 .01\nLD 5 0 0 0 5.8e7\nFR 0 1 0 0 7.15\nEX 0 1 36 0 1\nXQ\n",
            "GE -1",
        ),
        "fills",
    ),
    # momwire#1607 B6b: a crossing node's class blocks and point rows
    (
        "crossing_dense",
        deck(
            "GW 1 10 0 0 0 0 0 5 .001\nGW 2 5 0 0 0 0 0 -1 .001\n",
            "GN 2 0 0 0 13 .005\nFR 0 1 0 0 7.15\nEX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "fills",
    ),
    # momwire#1607: the folded cos-1 shape on a crossing deck (kΔ under
    # `_WELL_SCALED_KD`), dense and sparse
    (
        "crossing_folded_dense",
        deck(
            "GW 1 10 0 0 0 0 0 1 .001\nGW 2 5 0 0 0 0 0 -.5 .001\n",
            "GN 2 0 0 0 13 .005\nFR 0 1 0 0 0.5\nEX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "fills",
    ),
    (
        "crossing_folded_sparse",
        deck(
            "GW 1 40 0 0 0 0 0 4 .001\nGW 2 10 0 0 0 0 0 -.5 .001\nGW 3 30 0 0 -.5 8 0 -.5 .001\n",
            "GN 2 0 0 0 13 .005\nFR 0 1 0 0 0.5\nEX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "fills",
    ),
    (
        "crossing_sparse_ld5",
        deck(
            "GW 1 30 0 0 0 0 0 10 .001\nGW 2 10 0 0 0 0 0 -.5 .001\nGW 3 30 0 0 -.5 8 0 -.5 .001\n",
            "GN 3 0 0 0 20 .01\nLD 5 0 0 0 5.8e7\nFR 0 1 0 0 7.15\nEX 0 1 1 0 1\nXQ\n",
            "GE -1",
        ),
        "fills",
    ),
    (
        "duplicate_sparse_ld5",
        deck(
            dipole(41)
            + "GW 2 31 .2 0 -.23 .2 0 .23 .001\nGW 3 31 .2 0 .23 .2 0 -.23 .001\n",
            "LD 5 0 0 0 3.5e7\n" + FR + "EX 0 1 21 0 1\nXQ\n",
        ),
        "fills",
    ),
]


@pytest.mark.parametrize("name, text, expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_has_the_same_fill(name, text, expect, acc):
    cxx = nc.cxx_fill(text, acc)
    assert nc.classify(cxx) == expect, cxx[:500]
    if expect == "declined":
        return
    diff = nc.dc.first_difference(nc.py_fill(text), cxx)
    assert diff is None, diff


def test_the_routes_are_reached(acc):
    """Not vacuous: whole dumps below the threshold, row digests above it,
    several ports and several points."""
    dumps = {name: nc.cxx_fill(text, acc) for name, text, _e in CASES}
    assert ".G : shape 59 59" in dumps["dense_59"]
    assert ".G[0][0] = complex " in dumps["dense_59"]
    assert ".G : shape 65 65" in dumps["ragged_tile_65"]
    assert ".G[64] = digest " in dumps["ragged_tile_65"]
    assert ".y : shape 2 2" in dumps["two_sources"]
    assert ".points : list 4" in dumps["sweep"]
    assert ".G : shape 1 1" in dumps["one_segment"]
    assert ".G : shape 131 131" in dumps["cos_minus_one_elided"]
