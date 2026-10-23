"""The C++ nec2 / nec4 deck reader is the Python one, field for field.

`src/momwire/_nec4_deck_core.cpp` copies momwire's reader for the
segment-centre dialects (`deck/_nec2.py`, `deck/_nec2_geometry.py` and the
NEC-4.2 slot's deltas in `deck/_nec4.py`) so that a Python-free executable
reads a deck exactly as `momwire-nec42-sinusoidal` / `momwire-nec42` do
(momwire#1607 A1). Python stays the reference; these tests hold the copy to
it:

* every NEC deck in `tests/fixtures`, in both dialects;
* hand-written decks, one or more per card and per refusal -- each case first
  checks that the PYTHON reader lands where the case says (parsed, or the
  named refusal), so a case cannot silently stop exercising its branch, then
  that the C++ dump is byte-equal;
* the copies of Python's float `repr` and `math.dist`, pinned on random
  inputs.

The comparison format is `scripts/core_split/deck_compare.py`'s, through
`scripts/core_split/nec4_deck_compare.py`. A deck the C++ declines (a
non-finite field, sizes past what it materialises) is never compared; the
tests below say which ones decline.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import errno
import os
import random
import struct
import sys

import pytest

from momwire.deck import parse

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

# `math.dist` and float `sum()` are CPython 3.12's algorithms in the copy.
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the reader copies CPython 3.12's float rules"
)


def _compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_nec4_deck_compare",
        ROOT / "scripts" / "core_split" / "nec4_deck_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _compare_module()


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "nec4_deck_dump"):
        pytest.skip("no compiled accelerator with the nec2 / nec4 deck reader")
    return _accel.acc


def _same(text: str, dialect: str, acc) -> str:
    cxx = nc.cxx_parse(text, dialect, acc)
    kind = nc.classify(cxx)
    if kind != "declined":
        diff = nc.dc.first_difference(nc.py_parse(text, dialect), cxx)
        assert diff is None, f"{dialect}:\n{diff}"
    return kind


# --- the fixture decks ------------------------------------------------------

FIXTURE_DECKS = [
    p
    for p in sorted(FIXTURES.rglob("*"))
    if p.is_file() and p.suffix in (".nec", ".deck")
]


def test_the_fixture_decks_are_there():
    assert len(FIXTURE_DECKS) >= 400


@pytest.mark.parametrize("dialect", ["nec2", "nec4"])
def test_every_fixture_deck_reads_the_same(dialect, acc):
    """One test per dialect over the whole fixture tree (400+ decks); the
    counts say both outcomes are exercised, and that nothing but the three
    deliberately non-finite / oversized decks declines."""
    counts = {"parsed": 0, "refused": 0, "declined": 0}
    for path in FIXTURE_DECKS:
        counts[_same(path.read_bytes().decode("latin-1"), dialect, acc)] += 1
    assert counts["parsed"] >= 200 and counts["refused"] >= 100, counts
    assert counts["declined"] <= 3, counts


# --- hand-written decks -----------------------------------------------------

DIPOLE = "GW 1 11 0 0 -.25 0 0 .25 .001\n"
TAIL = "FR 0 1 0 0 299.7925\nEX 0 1 6 0 1 0\nXQ\nEN\n"
NEC4_TAIL = "FR 0 1 0 0 299.7925\nEX 0 1 6 0 1 0\nXQ\nEN\n"


def deck(
    geometry: str = DIPOLE, *cards: str, ge: str = "GE 0", tail: str = TAIL
) -> str:
    return geometry + ge + "\n" + "".join(c + "\n" for c in cards) + tail


# Python's OverflowError text for `float ** int` is (errno, strerror): the C
# library's words, which differ between glibc, Apple's libc and the UCRT.
_ERANGE = f"({errno.ERANGE}, {os.strerror(errno.ERANGE)!r})"

# A two-wire structure that a GX in Y=0 mirrors (cell = wires 1-2).
HALF = "GW 1 5 0 .1 -.25 0 .1 .25 .001\nGW 2 3 0 .1 .25 0 .3 .25 .001\n"

CASES: list[tuple[str, tuple[str, ...], str, str]] = [
    # (name, dialects, deck, "parsed" or a fragment of the Python refusal)
    ("dipole", ("nec2", "nec4"), deck(), "parsed"),
    (
        "no_fr_default_frequency",
        ("nec2", "nec4"),
        DIPOLE + "GE 0\nEX 0 1 6 0 1\nXQ\nEN\n",
        "parsed",
    ),
    (
        "fr_additive_sweep",
        ("nec2",),
        deck(DIPOLE, tail="FR 0 7 0 0 14 .35\nEX 0 1 6 0 1\nXQ\nEN\n"),
        "parsed",
    ),
    (
        "fr_multiplicative",
        ("nec2",),
        deck(DIPOLE, tail="FR 1 5 0 0 3 1.1\nEX 0 1 6 0 1\nXQ\nEN\n"),
        "parsed",
    ),
    (
        "fr_multiplicative_overflow",
        ("nec2",),
        deck(DIPOLE, tail="FR 1 400 0 0 1 10\nEX 0 1 6 0 1\nXQ\nEN\n"),
        _ERANGE,
    ),
    (
        "fr_multiplicative_underflow",
        ("nec2",),
        deck(DIPOLE, tail="FR 1 400 0 0 1 .1\nEX 0 1 6 0 1\nXQ\nEN\n"),
        "parsed",
    ),
    (
        "gm_rotate_replicate",
        ("nec2", "nec4"),
        deck(
            "GW 1 4 .1 0 0 .1 0 .2 .001\nGM 1 3 0 0 90 0 0 .05 0\n",
            tail="FR 0 1 0 0 300\nEX 0 1 2 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gm_move_from_tag",
        ("nec2",),
        deck(DIPOLE + "GW 2 5 .3 0 -.1 .3 0 .1 .001\nGM 0 0 10 20 30 .1 .2 .3 2\n"),
        "parsed",
    ),
    (
        "gm_missing_tag",
        ("nec2",),
        deck(DIPOLE + "GM 0 0 0 0 0 0 0 .5 7\n"),
        "no wire has tag 7",
    ),
    (
        "gm_negative_repeats",
        ("nec2",),
        deck(DIPOLE + "GM 1 -3 0 0 30 0 0 0 0\n"),
        "parsed",
    ),
    ("gs_scale", ("nec2", "nec4"), deck(DIPOLE + "GS 0 0 .5\n"), "parsed"),
    (
        "gs_ranged",
        ("nec2",),
        deck(DIPOLE + "GW 2 5 .3 0 -.1 .3 0 .1 .001\nGS 2 2 2.5\n"),
        "parsed",
    ),
    (
        "gs_zero",
        ("nec2",),
        deck(DIPOLE + "GS 0 0 0\n"),
        "scale factor must be > 0, got 0.0",
    ),
    ("gs_negative", ("nec2",), deck(DIPOLE + "GS 0 0 -2.5e-7\n"), "got -2.5e-07"),
    (
        "gx_mirror",
        ("nec2",),
        deck(HALF + "GX 10 010\n", tail="FR 0 1 0 0 300\nEX 0 1 3 0 1\nXQ\nEN\n"),
        "parsed",
    ),
    (
        "gx_three_planes",
        ("nec2",),
        deck(
            "GW 1 3 .1 .1 .1 .2 .2 .3 .001\nGX 1 111\n",
            tail="FR 0 1 0 0 300\nEX 0 1 2 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gx_negative_code",
        ("nec2",),
        deck(
            "GW 1 3 .1 .1 .1 .2 .2 .3 .001\nGX 1 -1\n",
            tail="FR 0 1 0 0 300\nEX 0 1 2 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gx_no_plane",
        ("nec2",),
        deck(
            HALF + "GX 10 000\n",
            "LD 4 1 1 1 50 0",
            tail="FR 0 1 0 0 300\nEX 0 1 3 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gx_in_plane",
        ("nec2",),
        deck(DIPOLE + "GX 1 100\n"),
        "lies in or crosses the X=0",
    ),
    (
        "gx_cell_load_replicates",
        ("nec2",),
        deck(
            HALF + "GX 10 010\n",
            "LD 4 1 3 3 50 5",
            tail="FR 0 1 0 0 300\nEX 0 1 3 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gx_load_outside_cell",
        ("nec2",),
        deck(
            HALF + "GX 10 010\n",
            "LD 4 11 3 3 50 5",
            tail="FR 0 1 0 0 300\nEX 0 1 3 0 1\nXQ\nEN\n",
        ),
        "outside the GX/GR symmetric cell",
    ),
    (
        "gx_ld5_cell_wire",
        ("nec2",),
        deck(
            HALF + "GX 10 010\n",
            "LD 5 2 0 0 5.8e7",
            tail="FR 0 1 0 0 300\nEX 0 1 3 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gx_destroyed_by_gw",
        ("nec2",),
        deck(
            HALF + "GX 10 010\nGW 30 3 1 1 1 1 1 1.5 .001\n",
            "LD 4 11 3 3 50 5",
            tail="FR 0 1 0 0 300\nEX 0 1 3 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gr_restatement",
        ("nec2",),
        deck(
            "GW 1 5 .2 0 0 .2 0 .5 .001\nGR 1 4\n",
            "LD 4 1 2 2 30 0",
            "LD 4 2 2 2 30 0",
            tail="FR 0 1 0 0 300\nEX 0 1 1 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gr_disagreeing_copy",
        ("nec2",),
        deck(
            "GW 1 5 .2 0 0 .2 0 .5 .001\nGR 1 4\n",
            "LD 4 1 2 2 30 0",
            "LD 4 2 2 2 31 0",
            tail="FR 0 1 0 0 300\nEX 0 1 1 0 1\nXQ\nEN\n",
        ),
        "outside the GX/GR symmetric cell",
    ),
    (
        "gr_zero_copies",
        ("nec2",),
        deck(DIPOLE + "GR 1 0\n"),
        "structure count must be >= 1, got 0",
    ),
    (
        "gr_under_gm",
        ("nec2",),
        deck(
            "GW 1 5 .2 0 0 .2 0 .5 .001\nGR 1 3\nGM 0 0 0 0 15 0 0 0 0\n",
            "LD 4 1 2 2 30 0",
            tail="FR 0 1 0 0 300\nEX 0 1 1 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "junction_cut",
        ("nec2", "nec4"),
        deck(
            DIPOLE + "GW 2 4 0 0 .0227272727 .3 0 .0227272727 .001\n",
            tail="FR 0 1 0 0 299.7925\nEX 0 2 1 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "snapped_ends",
        ("nec2", "nec4"),
        deck(
            "GW 1 5 0 0 0 0 0 .25 .001\nGW 2 5 1e-7 0 .25 .2 0 .4 .001\nGW 3 5 0 0 .2500002 -.2 0 .4 .001\n",
            tail="FR 0 1 0 0 299.7925\nEX 0 1 1 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gw_zero_segments",
        ("nec2",),
        deck("GW 1 0 0 0 -.25 0 0 .25 .001\n"),
        "segment count must be >= 1, got 0",
    ),
    ("gw_zero_radius", ("nec2",), deck("GW 1 3 0 0 -.25 0 0 .25 0\n"), "tapered wire"),
    (
        "ex_type_1_nec2",
        ("nec2",),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 1 1 6 0 1\nXQ\nEN\n"),
        "EX 0 only",
    ),
    (
        "ex_zero_volts",
        ("nec2", "nec4"),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 0 0\nXQ\nEN\n"),
        "parsed",
    ),
    (
        "ex_out_of_range",
        ("nec2", "nec4"),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 99 0 1\nXQ\nEN\n"),
        "segment 99 is out of range for tag 1",
    ),
    (
        "ex_tag_zero",
        ("nec2",),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 0 0 4 0 1\nXQ\nEN\n"),
        "parsed",
    ),
    (
        "ex_current_source",
        ("nec4",),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 6 1 6 0 1.414214 0\nXQ\nEN\n"),
        "parsed",
    ),
    (
        "ex_current_zero",
        ("nec4",),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 6 1 6 0 0 0\nXQ\nEN\n"),
        "sets a current of zero",
    ),
    (
        "ex_both_kinds_one_port",
        ("nec4",),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nEX 6 1 6 0 1\nXQ\nEN\n"),
        "an earlier EX 0 drove",
    ),
    (
        "ex_phased_two_ports",
        ("nec4",),
        deck(
            DIPOLE + "GW 2 11 .25 0 -.25 .25 0 .25 .001\n",
            tail="FR 0 1 0 0 300\nEX 6 1 6 0 1 0\nEX 6 2 6 0 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "ex_type_5_nec4",
        ("nec4",),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 5 1 6 0 1\nXQ\nEN\n"),
        "EX 0 (voltage source) and EX 6",
    ),
    (
        "ex_retention_and_union",
        ("nec2", "nec4"),
        deck(
            DIPOLE,
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nEX 0 1 5 0 2\nEX 0 1 7 0 3\nXQ\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "no_ex",
        ("nec2", "nec4"),
        DIPOLE + "GE 0\nFR 0 1 0 0 300\nXQ\nEN\n",
        "no EX card",
    ),
    ("ge_nec4_two", ("nec4",), deck(ge="GE 2"), "is not a NEC-4.2 ground flag"),
    ("ge_nec4_second_field", ("nec4",), deck(ge="GE 1 -2"), "second field of -2"),
    (
        "ge_minus_one_contact",
        ("nec2", "nec4"),
        deck(
            "GW 1 5 0 0 0 0 0 .5 .001\n",
            "GN 1",
            ge="GE -1",
            tail="FR 0 1 0 0 300\nEX 0 1 1 0 1\nXQ\nEN\n",
        ),
        "GE -1 declares the ground plane",
    ),
    (
        "ge_minus_one_free_space",
        ("nec2", "nec4"),
        deck(
            "GW 1 5 0 0 0 0 0 .5 .001\n",
            ge="GE -1",
            tail="FR 0 1 0 0 300\nEX 0 1 1 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "ge_minus_one_crossing",
        ("nec4",),
        deck(
            "GW 1 5 0 0 0 0 0 .5 .001\nGW 2 3 0 0 0 0 0 -.3 .001\n",
            "GN 2 0 0 0 13 .005",
            ge="GE -1",
            tail="FR 0 1 0 0 30\nEX 0 1 1 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "ge_one_contact",
        ("nec2", "nec4"),
        deck(
            "GW 1 5 0 0 0 0 0 .5 .001\n",
            "GN 1",
            ge="GE 1",
            tail="FR 0 1 0 0 300\nEX 0 1 1 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "gn_trailer_nec2",
        ("nec2",),
        deck(DIPOLE, "GN 2 0 0 0 13 .005 NOFILE"),
        "trailing token 'NOFILE'",
    ),
    (
        "gn_trailer_quote_nec2",
        ("nec2",),
        deck(DIPOLE, "GN 2 0 0 0 13 .005 it's"),
        'trailing token "it\'s"',
    ),
    (
        "gn_trailer_control_nec2",
        ("nec2",),
        deck(DIPOLE, "GN 2 0 0 0 13 .005 a\x01b\xe9\xad"),
        "trailing token",
    ),
    ("gn_trailer_nec4", ("nec4",), deck(DIPOLE, "GN 2 0 0 0 13 .005 NOFILE"), "parsed"),
    ("gn3_nec4", ("nec4",), deck(DIPOLE, "GN 3 0 0 0 13 .005"), "parsed"),
    (
        "gn3_nec2",
        ("nec2",),
        deck(DIPOLE, "GN 3 0 0 0 13 .005"),
        "GN type 3 is not supported",
    ),
    (
        "gn3_radials_nec4",
        ("nec4",),
        deck(DIPOLE, "GN 3 4 0 0 13 .005"),
        "GN 3 with a 4-wire radial",
    ),
    (
        "gn2_zero_eps_nec4",
        ("nec4",),
        deck(DIPOLE, "GN 2"),
        "needs a positive relative permittivity",
    ),
    (
        "gn3_zero_eps_nec4",
        ("nec4",),
        deck(DIPOLE, "GN 3 0 0 0 -1 .005"),
        "GN 3 needs a positive",
    ),
    (
        "gn0_radials",
        ("nec2", "nec4"),
        deck(DIPOLE, "GN 0 8 0 0 13 .005 1 .01"),
        "8-wire radial",
    ),
    ("gn4", ("nec2", "nec4"), deck(DIPOLE, "GN 4"), "GN type 4 is not supported"),
    ("gn_pec_radials", ("nec2",), deck(DIPOLE, "GN 1 4 0 0 0 0 1 2 3 4"), "parsed"),
    (
        "gn1_gd",
        ("nec2", "nec4"),
        deck(DIPOLE, "GN 1", "GD 2 0 0 0 13 .005 10 -1"),
        "parsed",
    ),
    (
        "rp_cliff_no_medium",
        ("nec2", "nec4"),
        deck(
            DIPOLE,
            "GN 1",
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nRP 2 1 1 1000 90 0 0 0\nEN\n",
        ),
        "no second medium stated",
    ),
    (
        "rp_cliff_free_space",
        ("nec2",),
        deck(
            DIPOLE,
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nRP 3 3 4 1000 0 0 30 90 100\nEN\n",
        ),
        "parsed",
    ),
    (
        "rp_mode_1",
        ("nec2", "nec4"),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nRP 1 1 1\nEN\n"),
        "RP mode 1 is not supported",
    ),
    (
        "ne_rectangular",
        ("nec2", "nec4"),
        deck(
            DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nNE 0 2 3 4 0 0 0 .1 .1 .1\nEN\n"
        ),
        "parsed",
    ),
    (
        "nh_spherical",
        ("nec2",),
        deck(
            DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nNH 1 2 3 4 1 0 0 .1 10 10\nEN\n"
        ),
        "(spherical)",
    ),
    (
        "nh_spherical_nec4",
        ("nec4",),
        deck(
            DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nNH 1 2 3 4 1 0 0 .1 10 10\nEN\n"
        ),
        "parsed",
    ),
    (
        "ne_coordinates_2_nec4",
        ("nec4",),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nNE 2 1 1 1\nEN\n"),
        "coordinate system 2 is not one",
    ),
    (
        "ne_over_gn0",
        ("nec2", "nec4"),
        deck(
            DIPOLE,
            "GN 0 0 0 0 13 .005",
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nNE 0 1 1 1\nEN\n",
        ),
        "reflection-coefficient ground",
    ),
    (
        "unarmed_rp_after_xq",
        ("nec2", "nec4"),
        deck(
            DIPOLE,
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nRP 0 1 1 1000\nNE 0 1 1 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "unarmed_rp_mode_1",
        ("nec2",),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nRP 1 1 1\nEN\n"),
        "parsed",
    ),
    (
        "unarmed_rp_mode_1_nec4",
        ("nec4",),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nRP 1 1 1\nEN\n"),
        "RP mode 1 is not supported",
    ),
    (
        "gn_between_runs",
        ("nec2", "nec4"),
        deck(DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nGN 1\nXQ\nEK\nXQ\nEN\n"),
        "parsed",
    ),
    ("ek_minus_one", ("nec2", "nec4"), deck(DIPOLE, "EK -1"), "parsed"),
    ("ek_on", ("nec2",), deck(DIPOLE, "EK"), "parsed"),
    (
        "ld_lumped_kinds",
        ("nec2", "nec4"),
        deck(
            DIPOLE,
            "LD 0 1 2 2 10 1e-6 1e-12",
            "LD 1 1 3 3 10 1e-6 1e-12",
            "LD 4 1 4 4 50 -20",
        ),
        "parsed",
    ),
    ("ld_zero_noop", ("nec2",), deck(DIPOLE, "LD 4 1 4 4 0 0"), "parsed"),
    (
        "ld_whole_tag_too_wide",
        ("nec2",),
        deck(DIPOLE, "LD 4 1 0 3 50 0"),
        "LD over 11 segments",
    ),
    ("ld_range_reversed", ("nec2",), deck(DIPOLE, "LD 4 1 5 2 50 0"), "parsed"),
    # segments -3..0 all read as 1: the range loads segment 1 five times
    (
        "ld_negative_first",
        ("nec2",),
        deck(DIPOLE, "LD 4 1 -3 2 50 0"),
        "already carries a load",
    ),
    (
        "ld_double",
        ("nec2", "nec4"),
        deck(DIPOLE, "LD 4 1 4 4 50 0", "LD 0 1 4 4 5"),
        "already carries a load",
    ),
    (
        "ld_clear_and_restate",
        ("nec2", "nec4"),
        deck(
            DIPOLE,
            "LD 4 1 4 4 50 0",
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nLD -1\nXQ\nLD 4 1 4 4 50 0\n"
            "XQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "ld_type_7",
        ("nec2", "nec4"),
        deck(DIPOLE, "LD 7 1 1 1 1"),
        "LD type 7 is not supported",
    ),
    (
        "ld5_whole_and_wire",
        ("nec2", "nec4"),
        deck(
            DIPOLE + "GW 2 5 .3 0 -.1 .3 0 .1 .001\n",
            "LD 5 0 0 0 5.8e7",
            "LD 5 2 0 0 3.5e7",
        ),
        "parsed",
    ),
    (
        "ld5_partial",
        ("nec2",),
        deck(DIPOLE, "LD 5 1 2 3 5.8e7"),
        "partial-wire segment range",
    ),
    (
        "ld5_mu",
        ("nec2", "nec4"),
        deck(DIPOLE, "LD 5 0 0 0 5.8e7 2"),
        "relative permeability of 2",
    ),
    (
        "ld5_changes_between_runs",
        ("nec2",),
        deck(
            DIPOLE,
            "LD 5 0 0 0 5.8e7",
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nLD 5 0 0 0 1e7\nXQ\nEN\n",
        ),
        "changes between execute cards",
    ),
    (
        "ld5_after_last_run",
        ("nec2",),
        deck(
            DIPOLE,
            "LD 5 0 0 0 5.8e7",
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nLD 5 0 0 0 1e7\nEN\n",
        ),
        "parsed",
    ),
    (
        "ld2_series_with_c_nec2",
        ("nec2",),
        deck(DIPOLE, "LD 2 1 0 0 1 1e-7 1e-9"),
        "parsed",
    ),
    (
        "ld2_with_c_nec4",
        ("nec4",),
        deck(DIPOLE, "LD 2 1 0 0 1 1e-7 1e-9"),
        "capacitance of 1e-09",
    ),
    (
        "ld3_whole_structure",
        ("nec2", "nec4"),
        deck(DIPOLE, "LD 3 0 0 0 100 1e-6"),
        "parsed",
    ),
    (
        "ld2_negative",
        ("nec2", "nec4"),
        deck(DIPOLE, "LD 2 1 0 0 1 -1e-7"),
        "negative per-unit-length inductance",
    ),
    (
        "ld3_partial",
        ("nec2",),
        deck(DIPOLE, "LD 3 1 2 2 100"),
        "partial-wire segment range",
    ),
    ("is_jacket", ("nec2", "nec4"), deck(DIPOLE, "IS 0 1 0 0 3.5 0 .002"), "parsed"),
    (
        "is_with_ld5",
        ("nec2",),
        deck(DIPOLE, "LD 5 1 0 0 5.8e7", "IS 0 1 0 0 3.5 0 .002"),
        "parsed",
    ),
    ("is_vacuum_noop", ("nec2",), deck(DIPOLE, "IS 0 1 0 0 1 0 .002"), "parsed"),
    (
        "is_sheath",
        ("nec2",),
        deck(DIPOLE, "IS 0 1 0 0 3.5 1e-3 .002"),
        "conductive sheath",
    ),
    (
        "is_partial",
        ("nec2",),
        deck(DIPOLE, "IS 0 1 2 3 3.5 0 .002"),
        "partial-wire segment range",
    ),
    ("is_thin", ("nec2",), deck(DIPOLE, "IS 0 1 0 0 3.5 0 .0005"), "does not exceed"),
    (
        "is_after_xq",
        ("nec2",),
        deck(
            DIPOLE, tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nIS 0 1 0 0 3.5 0 .002\nEN\n"
        ),
        "IS after an execute",
    ),
    (
        "is_under_symmetry",
        ("nec2",),
        deck(
            HALF + "GX 10 010\n",
            "IS 0 1 0 0 3.5 0 .002",
            tail="FR 0 1 0 0 300\nEX 0 1 3 0 1\nXQ\nEN\n",
        ),
        "IS while a GX/GR symmetric cell",
    ),
    (
        "tl_nt",
        ("nec2", "nec4"),
        deck(
            DIPOLE + "GW 2 11 .25 0 -.25 .25 0 .25 .001\n",
            "TL 1 6 2 6 -50 .5 0 0 0 0",
            "PT -1",
            "NT 1 6 2 6 0 .01 0 0 0 .02",
        ),
        "parsed",
    ),
    (
        "tl_destroyed",
        ("nec2",),
        deck(
            DIPOLE + "GW 2 11 .25 0 -.25 .25 0 .25 .001\n",
            "TL 1 6 2 6 50 .5",
            "LD 4 1 2 2 5",
            "TL 1 7 2 7 50 .5",
        ),
        "with an interposed LD card",
    ),
    (
        "tl_segment_zero",
        ("nec2",),
        deck(DIPOLE, "TL 1 0 1 6 50"),
        "addresses segment 0 of tag 1",
    ),
    (
        "nt_tag_zero_segment_minus",
        ("nec2",),
        deck(DIPOLE, "NT 0 -2 1 6 1"),
        "of the structure",
    ),
    (
        "tl_zero_z0",
        ("nec2", "nec4"),
        deck(DIPOLE, "TL 1 2 1 6 0 1"),
        "zero characteristic impedance",
    ),
    (
        "tl_after_run",
        ("nec2",),
        deck(
            DIPOLE + "GW 2 11 .25 0 -.25 .25 0 .25 .001\n",
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nTL 1 6 2 6 50 .5\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "pt_forms",
        ("nec2", "nec4"),
        deck(
            DIPOLE,
            tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nPT 0 1 3 5\nXQ\nPT -1\nEX 0 1 5 0 1\nXQ\n"
            "PT -2\nEX 0 1 4 0 1\nXQ\nPT 0 0 0 0\nEX 0 1 3 0 1\nXQ\nEN\n",
        ),
        "parsed",
    ),
    (
        "pt_out_of_range",
        ("nec2",),
        deck(DIPOLE, "PT 0 1 9 20"),
        "segment 12 is out of range",
    ),
    (
        "mp_fraction",
        ("nec2", "nec4"),
        deck(DIPOLE, "MP 2.5 1"),
        "MP field 1 must be an integer, not 2.5",
    ),
    ("mp_huge", ("nec2",), deck(DIPOLE, "MP 1 1e20"), "parsed"),
    ("pq_card", ("nec2", "nec4"), deck(DIPOLE, "PQ 0 1 2 3"), "parsed"),
    (
        "comment_directives",
        ("nec2", "nec4"),
        "CM model QQ 1 then ff 3\nCE end FF 2.7\n" + deck(),
        "parsed",
    ),
    ("comment_directive_bad_float", ("nec2",), "CM QQ x FF\nCE\n" + deck(), "parsed"),
    ("comment_directive_nan", ("nec2",), "CM FF nan\nCE\n" + deck(), "parsed"),
    (
        "comment_directive_inf",
        ("nec2",),
        "CM FF -inf\nCE\n" + deck(),
        "cannot convert float infinity",
    ),
    (
        "comment_latin1",
        ("nec2", "nec4"),
        "CM caf\xe9 \xb5H \xff\nCE\n" + deck(),
        "parsed",
    ),
    (
        "refused_by_name_sy",
        ("nec2", "nec4"),
        "SY L=1.5\n" + deck(),
        "SY (4nec2 symbolic variables)",
    ),
    # GC and GF stay refused by name in both dialects (momwire#1651).
    (
        "refused_by_name_gc",
        ("nec2", "nec4"),
        deck(DIPOLE + "GC 0 0 1 .001 .002\n"),
        "GC (tapered wire continuation)",
    ),
    (
        "refused_by_name_gf",
        ("nec2", "nec4"),
        "GF 0\n" + deck(),
        "GF (numerical Green's function)",
    ),
    # The nec2 dialect's GA (NEC-2's arc, as the nec4 one) and GH (NEC-2's
    # helix, nec2c's `helix`: spacing, length, four radii, wire radius),
    # momwire#1651: every branch of the vertex rule, and each refusal.
    ("nec2_ga_arc", ("nec2",), deck(DIPOLE + "GA 2 9 12.5 -45 225 .002\n"), "parsed"),
    ("nec2_ga_falling", ("nec2",), deck(DIPOLE + "GA 2 6 1 90 0 .001\n"), "parsed"),
    (
        "nec2_ga_full_loop",
        ("nec2",),
        deck("GA 1 12 1 0 360 .002\n"),
        "parsed",
    ),
    (
        "ga_full_loop_nec4",
        ("nec4",),
        deck("GA 1 12 1 0 360 .002\n"),
        "sweeps 360 degrees, which NEC-4.2 stops on",
    ),
    (
        "ga_just_under_360_nec4",
        ("nec4",),
        deck("GA 1 12 1 0 359.999999 .002\n"),
        "parsed",
    ),
    (
        "ga_over_360",
        ("nec2", "nec4"),
        deck("GA 1 12 1 -90 270.00001 .002\n"),
        "degrees, which NEC-",
    ),
    (
        "ga_no_wire_nec2",
        ("nec2",),
        deck(DIPOLE + "GA 2 4 1 0 90 0\n"),
        "nec2 dialect has no",
    ),
    (
        "nec2_gh_constant",
        ("nec2",),
        deck(DIPOLE + "GH 2 20 .1 .5 .05 .05 .05 .05 .001\n"),
        "parsed",
    ),
    (
        "nec2_gh_b_default",
        ("nec2",),
        deck(DIPOLE + "GH 2 16 .2 .4 .06 0 .06 0 .001\n"),
        "parsed",
    ),
    (
        "nec2_gh_left_hand",
        ("nec2",),
        deck(DIPOLE + "GH 2 16 .2 -.4 .06 .06 .06 .06 .001\n"),
        "parsed",
    ),
    (
        "nec2_gh_taper",
        ("nec2",),
        deck(DIPOLE + "GH 2 24 .1 .6 .05 .05 .1 .1 .001\n"),
        "parsed",
    ),
    (
        "nec2_gh_elliptic",
        ("nec2",),
        deck(DIPOLE + "GH 2 20 .1 .5 .06 .04 .06 .04 .001\n"),
        "parsed",
    ),
    (
        "nec2_gh_taper_b2_zero",
        ("nec2",),
        deck(DIPOLE + "GH 2 20 .1 .5 .05 .05 .1 0 .001\n"),
        "parsed",
    ),
    (
        "nec2_gh_taper_from_axis",
        ("nec2",),
        deck(DIPOLE + "GH 2 20 .1 .5 0 0 .1 .1 .001\n"),
        "parsed",
    ),
    (
        "nec2_gh_one_segment",
        ("nec2",),
        deck(DIPOLE + "GH 2 1 .25 .1 .1 .1 .1 .1 .001\n"),
        "parsed",
    ),
    (
        "nec2_gh_then_gx",
        ("nec2",),
        deck("GH 1 12 1 .5 .3 .3 .3 .3 .002\nGX 10 100\n"),
        "parsed",
    ),
    (
        "nec2_gh_no_segments",
        ("nec2",),
        deck(DIPOLE + "GH 2 0 .1 .5 .05 .05 .05 .05 .001\n"),
        "GH segment count",
    ),
    (
        "nec2_gh_no_spacing",
        ("nec2",),
        deck(DIPOLE + "GH 2 8 0 .5 .05 .05 .05 .05 .001\n"),
        "nonzero spacing",
    ),
    (
        "nec2_gh_no_length",
        ("nec2",),
        deck(DIPOLE + "GH 2 8 .1 0 .05 .05 .05 .05 .001\n"),
        "nonzero total length",
    ),
    (
        "nec2_gh_negative_radius",
        ("nec2",),
        deck(DIPOLE + "GH 2 8 .1 .5 .05 -.05 .05 .05 .001\n"),
        "non-negative helix radii",
    ),
    (
        "nec2_gh_no_radius",
        ("nec2",),
        deck(DIPOLE + "GH 2 8 .1 .5 0 .05 0 .05 .001\n"),
        "non-negative helix radii",
    ),
    (
        "nec2_gh_no_wire",
        ("nec2",),
        deck(DIPOLE + "GH 2 8 .1 .5 .05 .05 .05 .05 0\n"),
        "positive wire radius (F7)",
    ),
    (
        "nec2_gh_nec4_layout",
        ("nec2",),
        deck(DIPOLE + "GH 2 8 1 .3 .1 .1 .001 .001\n"),
        "positive wire radius (F7)",
    ),
    # The nec4 dialect's GA and GH (momwire#1592): the arc, the helix, its
    # log-spiral taper, a flat spiral, negative turns, one segment, and each
    # refusal, all field for field and bit for bit.
    ("ga_arc", ("nec4",), deck(DIPOLE + "GA 2 9 12.5 -45 225 .002\n"), "parsed"),
    ("ga_falling", ("nec4",), deck(DIPOLE + "GA 2 6 1 90 0 .001\n"), "parsed"),
    ("ga_one_segment", ("nec4",), deck(DIPOLE + "GA 2 1 1 0 90 .001\n"), "parsed"),
    (
        "ga_closed_by_gw",
        ("nec4",),
        deck("GA 1 12 2 0 180 .001\nGW 2 9 -2 0 0 2 0 0 .001\n"),
        "parsed",
    ),
    (
        "gh_helix",
        ("nec4",),
        deck(DIPOLE + "GH 2 24 1 .3 .15 .15 .002 .002\n"),
        "parsed",
    ),
    (
        "gh_log_spiral",
        ("nec4",),
        deck(DIPOLE + "GH 2 24 2 -.3 .15 .1 .002 .0012\n"),
        "parsed",
    ),
    ("gh_flat", ("nec4",), deck(DIPOLE + "GH 2 30 2 0 .05 .3 .002 .004\n"), "parsed"),
    (
        "gh_neg_turns",
        ("nec4",),
        deck(DIPOLE + "GH 2 30 -1.5 .5 .1 .1 .002 .002\n"),
        "parsed",
    ),
    (
        "gh_one_segment",
        ("nec4",),
        deck(DIPOLE + "GH 2 1 .25 .1 .1 .2 .001 .003\n"),
        "parsed",
    ),
    (
        "gh_then_gm_gs",
        ("nec4",),
        deck("GH 1 20 2 400 100 100 2 2\nGM 0 1 0 0 90 0 0 0 0\nGS 0 0 .001\n"),
        "parsed",
    ),
    (
        "ga_no_segments",
        ("nec4",),
        deck(DIPOLE + "GA 2 0 1 0 90 .001\n"),
        "GA segment count",
    ),
    (
        "ga_no_radius",
        ("nec4",),
        deck(DIPOLE + "GA 2 4 0 0 90 .001\n"),
        "positive arc radius",
    ),
    (
        "ga_no_length",
        ("nec4",),
        deck(DIPOLE + "GA 2 4 1 30 30 .001\n"),
        "has no length",
    ),
    (
        "ga_no_wire",
        ("nec4",),
        deck(DIPOLE + "GA 2 4 1 0 90 0\n"),
        "positive wire radius",
    ),
    (
        "gh_no_segments",
        ("nec4",),
        deck(DIPOLE + "GH 2 0 1 .3 .1 .1 .001 .001\n"),
        "GH segment",
    ),
    (
        "gh_no_turns",
        ("nec4",),
        deck(DIPOLE + "GH 2 8 0 .3 .1 .1 .001 .001\n"),
        "zero turns",
    ),
    (
        "gh_no_radius",
        ("nec4",),
        deck(DIPOLE + "GH 2 8 1 .3 .1 0 .001 .001\n"),
        "helix radius",
    ),
    (
        "gh_nec2_fields",
        ("nec4",),
        deck(DIPOLE + "GH 2 8 .1 .3 .15 .15 0 0 .002\n"),
        "not NEC-2's",
    ),
    (
        "ex4_current_element",
        ("nec4",),
        deck(DIPOLE, tail="FR 0 1 0 0 299.7925\nEX 4 1 6 0 1 0\nXQ\nEN\n"),
        "write EX 6 with that tag and segment",
    ),
    (
        "unrecognised_card",
        ("nec2", "nec4"),
        deck(DIPOLE, "ZZ 1 2"),
        "unrecognised NEC card 'ZZ'",
    ),
    (
        "fused_mnemonic",
        ("nec2", "nec4"),
        DIPOLE + "GE0\nFR0,1,0,0,300\nEX0,1,6,0,1\nXQ\nEN\n",
        "parsed",
    ),
    (
        "separators_only",
        ("nec2",),
        DIPOLE + ", ,\n" + "GE 0\n" + TAIL,
        "only separators",
    ),
    (
        "non_numeric_field",
        ("nec2",),
        deck(DIPOLE, "LD 4 1 2 2 5O"),
        "NON-NUMERICAL CHARACTER",
    ),
    (
        "geometry_after_ge",
        ("nec2",),
        deck(DIPOLE, "LD 4 1 2 2 50", "GW 2 5 .3 0 -.1 .3 0 .1 .001", "LD 4 2 2 2 50"),
        "parsed",
    ),
    ("after_terminator", ("nec2",), deck() + "garbage after EN ' \xe9\n", "parsed"),
    (
        "nx_ends_the_deck",
        ("nec2", "nec4"),
        deck(tail="FR 0 1 0 0 300\nEX 0 1 6 0 1\nXQ\nNX\nGW 9 1\n"),
        "parsed",
    ),
]


@pytest.mark.parametrize(
    "name, dialects, text, expect",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_a_hand_written_deck_reads_the_same(name, dialects, text, expect, acc):
    for dialect in dialects:
        try:
            parse(text, dialect=dialect)
        except Exception as exc:  # noqa: BLE001 — the case pins the outcome
            assert expect != "parsed", f"{dialect}: Python refused: {exc}"
            assert expect in str(exc), f"{dialect}: Python said {exc}"
        else:
            assert expect == "parsed", f"{dialect}: Python parsed it"
        kind = _same(text, dialect, acc)
        assert kind != "declined", f"{dialect}: the C++ declined"


@pytest.mark.parametrize(
    "text",
    [
        deck(DIPOLE, "LD 5 0 0 0 nan"),
        deck("GW 1 3 0 0 -inf 0 0 .25 .001\n"),
        deck("GW 1 1e30 0 0 -.25 0 0 .25 .001\n"),
        deck("GW 1 3 0 0 -1e308 0 0 1e308 .001\n"),
        deck(DIPOLE, tail="FR 0 1e9 0 0 1 1\nEX 0 1 6 0 1\nXQ\nEN\n"),
    ],
    ids=[
        "nan_field",
        "inf_field",
        "huge_segment_count",
        "overflowing_length",
        "huge_sweep",
    ],
)
def test_what_the_reader_declines(text, acc):
    """The declines are a decline in both dialects, never a misread."""
    for dialect in ("nec2", "nec4"):
        assert nc.classify(nc.cxx_parse(text, dialect, acc)) == "declined"


# --- Python's float rules ----------------------------------------------------


def _rand_double(rng: random.Random) -> float:
    kind = rng.random()
    if kind < 0.3:
        return rng.uniform(-10, 10)
    if kind < 0.6:
        return rng.choice([-1, 1]) * 10 ** rng.uniform(-320, 308)
    while True:
        x = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
        if math.isfinite(x):
            return x


def test_float_repr_follows_python(acc):
    rng = random.Random(1607)
    values = [
        0.0,
        -0.0,
        1.0,
        0.1,
        2.5,
        1e16,
        1e15,
        123456789012345678.0,
        1e-4,
        1e-5,
        5e-324,
        1.7976931348623157e308,
        100.0,
        -2.5e-07,
    ]
    values += [_rand_double(rng) for _ in range(20000)]
    for x in values:
        assert acc.nec4_float_probe("repr", [x]) == repr(x), x


def test_dist_follows_python(acc):
    rng = random.Random(1951)
    for _ in range(20000):
        a = [
            _rand_double(rng) if rng.random() < 0.2 else rng.uniform(-50, 50)
            for _ in range(3)
        ]
        b = [x + rng.uniform(-1, 1) * 10 ** rng.uniform(-12, 2) for x in a]
        got = acc.nec4_float_probe("dist", a + b)
        want = math.dist(a, b)
        assert got == want or (math.isnan(got) and math.isnan(want)), (a, b)
