"""The C++ NEC-4.2 run half prints the Python slot's printout for the
Sinusoidal-Galerkin basis, byte for byte.

`src/momwire/_sinusoidal_galerkin_run_core.cpp` answers the NEC-4.2 run
half (`_nec4_run_core.cpp`) for `basis="sinusoidal-galerkin"` (momwire#1718
S3): its own solve (the Galerkin fill and port solve) and the sinusoidal
basis' readouts, which SinusoidalGalerkinSolver inherits.
`scripts/core_split/nec4_run_compare.py --basis sinusoidal-galerkin` holds
the C++ printout equal to `_nec4.render(text, basis="sinusoidal-galerkin")`
with only the two measured-time lines masked, over:
- the NEC-4.2 fixtures;
- every hand-written deck of `tests/test_nec4_run_core.py` (RP, NE / NH,
  TL / NT, phased sources, PQ, GH, GD, FR sweeps, several execute cards, a
  wire listed twice, ...): each one the Galerkin copy prints is compared,
  and one it declines (a ground or route a later unit serves) is counted;
- the dialect default's fallback to the reduced kernel at a radius-step
  junction, with its advisory (SinusoidalGalerkinSolver's declared refusal;
  the point-matched basis has none);
- a wire end on the ground plane (momwire#1718 S7, the direct-field fill):
  `test_sinusoidal_galerkin_fill_core.py`'s contact decks, and the
  radius-step ones that reach it on the reduced kernel.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
BASIS = "sinusoidal-galerkin"

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
fill_cases = _load(
    "test_sg_fill_core_cases", ROOT / "tests" / "test_sinusoidal_galerkin_fill_core.py"
)


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(
        _accel.acc, "sinusoidal_galerkin_nec4_printout"
    ):
        pytest.skip("no compiled accelerator with the Sinusoidal-Galerkin run half")
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


def _same(text, acc):
    """'declined', or 'printed' once the two masked printouts are equal."""
    ok, cxx = rc.cxx_printout(text, acc, basis=BASIS)
    if not ok:
        return "declined"
    a, na = rc.mask(rc.py_printout(text, BASIS))
    b, nb = rc.mask(cxx)
    assert rc.first_difference(a, b) is None, rc.first_difference(a, b)
    assert na == nb, (na, nb)
    return "printed"


@pytest.mark.slow  # every NEC-4.2 fixture, printed by both copies
def test_the_nec42_fixtures_print_the_same(acc):
    counts, mismatches = rc.compare(_decks(FIXTURES / "eznec_nec42"), basis=BASIS)
    assert not mismatches, mismatches[:2]
    assert counts["printed"] >= 80, counts
    assert counts["masked_lines"] >= 2 * 80, counts


CASES = sin_cases.CASES


@pytest.mark.parametrize("name, text, _expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_prints_the_same(name, text, _expect, acc):
    _same(text, acc)


def test_most_hand_written_decks_print(acc):
    """Not vacuous: the free-space and GN 1 cases print rather than
    decline."""
    printed = sum(_same(text, acc) == "printed" for _n, text, _e in CASES)
    assert printed >= 40, printed


STEP = (
    "CM step\nCE\nGW 1 5 0 0 -.25 0 0 0 .001\nGW 2 5 0 0 0 0 0 .25 .002\nGE 0\n"
    "FR 0 1 0 0 299.7925\nEX 0 1 5 0 1\nXQ\nEN\n"
)


def test_a_radius_step_junction_falls_back_to_the_reduced_kernel(acc):
    """The dialect's default extended kernel, which this basis declares it
    cannot serve at a radius-step junction: the build solves with the
    reduced kernel and says why (the point-matched basis has no such
    refusal), and the printout is the reference's."""
    assert _same(STEP, acc) == "printed"
    dump = acc.nec4_build_dump(STEP.encode("latin-1"), BASIS)
    assert "points[0].extended_kernel = bool False" in dump
    assert "radius step at a junction".encode().hex() in dump
    sin = acc.nec4_build_dump(STEP.encode("latin-1"), "sinusoidal")
    assert "points[0].extended_kernel = bool True" in sin


CONTACT = fill_cases.CONTACT_CASES + fill_cases.CONTACT_STEP_CASES


@pytest.mark.parametrize("name, text", CONTACT, ids=[c[0] for c in CONTACT])
def test_a_wire_end_on_the_ground_prints_the_same(name, text, acc):
    """The direct-field fill under the run half: printed, not declined."""
    assert _same(text, acc) == "printed"
