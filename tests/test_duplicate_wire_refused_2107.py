"""A wire listed twice on the NEC-4.2 slot's sinusoidal, bspline and
bspline-d1 routes: refused by name (momwire#2107).

The NEC-4.2 slot on the sinusoidal basis (`momwire-nec42-sinusoidal`), and
since Steve's 11-22 decision on the bs2 and bs1 bases too, which print the
same rounding residue (bs2: 43.95 ohm for the reversed parasite where the
deck without its copy prints 65.22; bs1: 40.08 against 58.58). #1042's merge does not reach this route whole: the nec4 dialect's mesh cuts
each wire into pieces, the copy merges piece by piece, and the pieces at its
ends, joined differently from their twins', stay in place, so Z is singular
and the printout reads rounding residue (duplicate_monopoles_on_gn1 printed
27.95 ohm where the deck without its copy prints 38.94). Licensed NEC-4.2
stops on every such deck at its geometry check (black box, 11-22), so the
slot refuses it, before the build, in the deck's own GW tags: any count of
copies, either direction. A GM that moves without copying is no duplicate
and prints. The Python slot and the C++ run half say the same sentence.

The slot's other bases are not widened here: pulse refuses the copy in
its own words, and arrayblock / hmatrix are pinned as printing, so a
widening is a deliberate edit.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMALL = ROOT / "tests" / "fixtures" / "razor_printout_small"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the reader copies CPython 3.12's float rules"
)

FR = "FR 0 1 0 0 299.7925\n"
DIPOLE = "GW 1 11 0 0 -.25 0 0 .25 .001\n"
PARASITE = "GW 2 9 .2 0 -.23 .2 0 .23 .001\n"
REVERSED = "GW 3 9 .2 0 .23 .2 0 -.23 .001\n"
DRIVE = FR + "EX 0 1 6 0 1\nPQ 0\nXQ\n"


def deck(geometry: str, body: str = DRIVE, ge: str = "GE 0") -> str:
    return "CM run\nCE\n" + geometry + ge + "\n" + body + "EN\n"


def sentence(named: str, copies: str) -> str:
    return (
        f"NEC ERROR - the deck lists a wire more than once: the GW wires of "
        f"{named} coincide, end for end in either direction, so each set is "
        "one conductor written more than once and the current has no unique "
        "split between the copies. This engine's NEC-4.2 slot does not merge "
        "them, and NEC-4.2 refuses such a deck too: delete the "
        f"{copies} and run the deck again - momwire#2107"
    )


# (deck, the sentence's tags, the copies it names)
DUPLICATES = {
    "same_direction": (
        deck(DIPOLE + PARASITE + PARASITE.replace("GW 2", "GW 3")),
        "tags 2 and 3",
        "copy (tag 3)",
    ),
    "reversed": (deck(DIPOLE + PARASITE + REVERSED), "tags 2 and 3", "copy (tag 3)"),
    "three_times": (
        deck(
            DIPOLE
            + PARASITE
            + PARASITE.replace("GW 2", "GW 3")
            + REVERSED.replace("GW 3", "GW 4")
        ),
        "tags 2, 3 and 4",
        "copies (tags 3 and 4)",
    ),
    # one card written twice, so both copies carry one tag
    "same_tag_twice": (
        deck(DIPOLE + PARASITE + PARASITE),
        "tag 2",
        "copy (tag 2)",
    ),
}
# GM with no copies (NRPT 0) moves the parasite and copies nothing: 20
# segments, served by licensed NEC-4.2 as it is here
BY_GM = deck(DIPOLE + PARASITE + "GM 0 0 0 0 0 0 0 0 2\n")


def _render(text: str, basis: str) -> str:
    from momwire.dropin.nec5 import _nec4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _nec4.render(text, basis=basis)


# the bases the slot refuses a duplicate on (`_nec4._REFUSES_DUPLICATES`),
# each by its canonical name and, for bs2 and bs1, its long spelling
REFUSING = ["sinusoidal", "bs2", "bspline", "bs1", "bspline-d1"]


@pytest.mark.parametrize("basis", REFUSING)
@pytest.mark.parametrize("name", list(DUPLICATES))
def test_the_nec42_slot_refuses_a_wire_listed_twice(name, basis):
    text, named, copies = DUPLICATES[name]
    out = _render(text, basis)
    assert sentence(named, copies) in out, out[-700:]
    assert "CURRENTS AND LOCATION" not in out


@pytest.mark.parametrize("basis", REFUSING)
def test_a_gm_that_copies_nothing_still_prints(basis):
    out = _render(BY_GM, basis)
    assert "NEC ERROR" not in out, out[-700:]
    assert "ANTENNA INPUT PARAMETERS" in out


@pytest.mark.parametrize("basis", ["arrayblock", "hmatrix"])
def test_the_other_bases_are_not_widened(basis):
    # Steve's 11-22 scope: sinusoidal, bs2 and bs1
    out = _render(DUPLICATES["reversed"][0], basis)
    assert "momwire#2107" not in out
    assert "ANTENNA INPUT PARAMETERS" in out


def test_the_fixture_deck_the_issue_names_is_refused():
    out = _render((SMALL / "dup_reversed_rp_ne_pq.nec").read_text(), "sinusoidal")
    assert sentence("tags 2 and 3", "copy (tag 3)") in out, out[-700:]


# -- the C++ run half (the executable's route) -------------------------------


def _run_compare():
    spec = importlib.util.spec_from_file_location(
        "core_split_nec4_run_compare_2107",
        ROOT / "scripts" / "core_split" / "nec4_run_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the bases with a C++ NEC-4.2 run half (bs1 has none: it builds in Python)
CXX_BASES = ["sinusoidal", "bs2"]


@pytest.fixture(scope="module", params=CXX_BASES)
def cxx(request):
    from momwire import _accel

    rc = _run_compare()
    basis = request.param
    if _accel.acc is None or not hasattr(_accel.acc, rc.BINDINGS[basis]):
        pytest.skip(f"no compiled accelerator with the {basis} NEC-4.2 run half")
    try:
        rc.blas_addresses(basis)
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    return rc, _accel.acc, basis


@pytest.mark.parametrize("name", list(DUPLICATES))
def test_the_cxx_run_half_refuses_it_in_the_same_words(name, cxx):
    rc, acc, basis = cxx
    text, named, copies = DUPLICATES[name]
    ok, out = rc.cxx_printout(text, acc, basis=basis)
    assert ok, out[:400]
    assert sentence(named, copies) in out, out[-700:]
    a, _ = rc.mask(rc.py_printout(text, basis))
    b, _ = rc.mask(out)
    assert rc.first_difference(a, b) is None, rc.first_difference(a, b)


def test_the_cxx_run_half_prints_the_gm_deck(cxx):
    rc, acc, basis = cxx
    ok, out = rc.cxx_printout(BY_GM, acc, basis=basis)
    assert ok, out[:400]
    assert "NEC ERROR" not in out
