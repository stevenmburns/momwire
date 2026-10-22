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
- the declines: a ground, a network, an NE table, a GH helix.
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


def test_the_nec42_fixtures_print_the_same(acc):
    counts, mismatches = rc.compare(_decks(FIXTURES / "eznec_nec42"))
    assert not mismatches, mismatches[:2]
    assert counts["printed"] >= 30, counts
    assert counts["masked_lines"] >= 2 * 30, counts


FR = "FR 0 1 0 0 299.7925\n"


def deck(geometry: str, body: str, ge: str = "GE 0") -> str:
    return "CM run\nCE\n" + geometry + ge + "\n" + body + "EN\n"


DIPOLE = "GW 1 11 0 0 -.25 0 0 .25 .001\n"
TWO = DIPOLE + "GW 2 11 .2 0 -.25 .2 0 .25 .001\n"
TEE = "GW 1 10 0 0 -.25 0 0 .25 .001\nGW 2 5 0 0 0 .3 0 0 .001\n"
RP = "RP 0 7 4 1000 0 0 30 90\n"

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
        "ground_declines",
        deck(
            "GW 1 10 0 0 0 0 0 .25 .001\n", "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n", "GE 1"
        ),
        "declined",
    ),
    (
        "network_declines",
        deck(TWO, FR + "TL 1 6 2 6 50 .1\nEX 0 1 6 0 1\nXQ\n"),
        "declined",
    ),
    (
        "near_field_declines",
        deck(DIPOLE, FR + "EX 0 1 6 0 1\nNE 0 1 1 3 .1 0 0 0 0 .1\n"),
        "declined",
    ),
    (
        "helix_declines",
        deck("GH 1 20 2 .5 .05 .05 .05 .05 .001\n", FR + "EX 0 1 1 0 1\nXQ\n"),
        "declined",
    ),
]


@pytest.mark.parametrize("name, text, expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_prints_the_same(name, text, expect, acc):
    ok, cxx = rc.cxx_printout(text, acc)
    assert ("printed" if ok else "declined") == expect, cxx[:400]
    if not ok:
        return
    a, na = rc.mask(rc.py_printout(text))
    b, nb = rc.mask(cxx)
    assert rc.first_difference(a, b) is None, rc.first_difference(a, b)
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
