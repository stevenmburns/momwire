"""The C++ NEC-4.2 run half prints the Python slot's printout for the
B-spline basis (bs2, the slot's default), byte for byte, and solves the same
arrays to the bit.

`src/momwire/_bspline_nec4_run_core.cpp` answers the NEC-4.2 run half
(`_nec4_run_core.cpp`) for `basis="bspline"` (momwire#1607 Phase C1): bs2's
own front, fill and readouts on the slot's build.
`scripts/core_split/nec4_run_compare.py --basis bspline --arrays` holds the
C++ printout equal to `_nec4.render(text, basis="bspline")` with only the two
measured-time lines masked, and each solve's operator Z, port columns B,
coefficients X and port admittances Y equal to the ones `BSplineSolver` hands
its port solve and returns, over:
- every hand-written deck of `tests/test_nec4_run_core.py` (RP, NE / NH,
  TL / NT, phased sources, PQ, GH, GD, FR sweeps, loads, a wire listed twice,
  buried and crossing wires, ...): each one the C++ prints is compared, and
  one it declines is counted;
- the NEC-4.2 fixtures (slow).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
BASIS = "bspline"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the reader copies CPython 3.12's float rules"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load(
    "core_split_nec4_run_compare",
    ROOT / "scripts" / "core_split" / "nec4_run_compare.py",
)
sin_cases = _load("test_nec4_run_core_cases", ROOT / "tests" / "test_nec4_run_core.py")


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "bspline_nec4_printout"):
        pytest.skip("no compiled accelerator with the B-spline run half")
    try:
        rc.blas_addresses(BASIS)
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    return _accel.acc


@pytest.fixture(autouse=True)
def _one_blas_thread(monkeypatch):
    from threadpoolctl import threadpool_limits

    # the below grid's anchors depend on the frequencies a process has seen
    # (momwire#1471): the exact grid on both sides
    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")
    with threadpool_limits(limits=1):
        yield


def _decks(root):
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]


def _same(text, acc):
    """'declined', or 'printed' once the two masked printouts and every
    solve's arrays are equal."""
    ok, cxx, cxx_solves = rc.cxx_arrays(text, acc)
    if not ok:
        return "declined"
    py, py_solves = rc.py_arrays(text)
    a, na = rc.mask(py)
    b, nb = rc.mask(cxx)
    assert rc.first_difference(a, b) is None, rc.first_difference(a, b)
    assert na == nb, (na, nb)
    diff = rc.arrays_difference(py_solves, cxx_solves)
    assert diff is None, diff
    return "printed"


CASES = sin_cases.CASES


@pytest.mark.parametrize("name, text, _expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_prints_the_same(name, text, _expect, acc):
    _same(text, acc)


def test_most_hand_written_decks_print(acc):
    """Not vacuous: most hand-written decks print rather than decline."""
    printed = sum(_same(text, acc) == "printed" for _n, text, _e in CASES)
    assert printed >= 60, printed


@pytest.mark.slow  # every NEC-4.2 fixture, printed and solved by both copies
def test_the_nec42_fixtures_print_the_same(acc):
    counts, mismatches = rc.compare(
        _decks(FIXTURES / "eznec_nec42"), basis=BASIS, arrays=True
    )
    assert not mismatches, mismatches[:2]
    assert counts["printed"] >= 110, counts
    assert counts["solves"] >= counts["printed"], counts
