"""One GH deck and one GA deck through every reader, each held to its own
dialect's reference (momwire#1651).

The rule: a geometry card is read by the dialect's own rules, everywhere.

* the nec2 dialect (``momwire-nec2c``, the SimNEC portal) reads NEC-2's
  card, and its segments are nec2c's (``tests/fixtures/nec2_ga_gh_1651``);
* the nec4 dialect (``momwire-nec42*``, Python) reads NEC-4.2's card, and its
  segments are the licensed NEC-4.2's (``tests/fixtures/eznec_nec42``,
  black-box printouts of the reference build);
* the C++ NEC-4.2 route that ``momwire-nec42-sinusoidal``'s standalone
  executable compiles reads NEC-4.2's card too, and prints the same
  segments; the executable itself joins when
  ``MOMWIRE_NEC42_SIN_STANDALONE_EXE`` names one.

The two decks are the issue's own, byte for byte the same cards in both
fixture trees: ``GH 1 12 0.5 1.5 0.2 0.2 0.2 0.2 0.002`` (n11 / p83), three
turns to nec2c and half a turn to NEC-4.2; and a 180-degree arc moved up
10 m (n01 / p78), which both engines build alike.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
NEC2C = FIXTURES / "nec2_ga_gh_1651"
NEC42 = FIXTURES / "eznec_nec42"
EXE = os.environ.get("MOMWIRE_NEC42_SIN_STANDALONE_EXE")

# (card, nec2c fixture, NEC-4.2 probe)
CASES = {
    "GH": ("n11_gh_issue_layout", "p83_gh_nec2_layout"),
    "GA": ("n01_ga_issue_gm", "p78_ga_issue_arc_gm"),
}

# nec2c prints four decimals, NEC-4.2 five.
_NEC2C_TOL = 1.01e-4
_NEC42_TOL = 6e-6


def _cards(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln[:2] not in ("CM", "CE")]


def test_the_two_trees_hold_the_same_decks():
    for nec2c, nec42 in CASES.values():
        a = (NEC2C / f"{nec2c}.nec").read_text()
        b = (NEC42 / "probes" / f"{nec42}.nec").read_text()
        assert _cards(a) == _cards(b)


# --------------------------------------------------------------------------
# reading SEGMENTATION DATA: (centre, length, alpha, beta, radius, tag)


def _nec2c_rows(text: str) -> list[tuple]:
    lines = text.split("\n")
    at = next(i for i, ln in enumerate(lines) if "SEGMENTATION DATA" in ln)
    head = next(i for i in range(at, len(lines)) if lines[i].startswith("   No:"))
    rows = []
    for line in lines[head + 1 :]:
        if not line.strip():
            break
        v = line.split()
        rows.append((*(float(x) for x in v[1:8]), int(v[-1])))
    return rows


def _nec42_rows(text: str) -> list[tuple]:
    lines = text.split("\n")
    at = next(i for i, ln in enumerate(lines) if ln.startswith("  NO.       X"))
    edges = (6, 16, 26, 36, 46, 57, 67, 77)
    rows = []
    for line in lines[at + 1 :]:
        if not line.strip():
            break
        rows.append(
            (
                *(float(line[a:b]) for a, b in zip(edges, edges[1:])),
                int(line.split()[-1]),
            )
        )
    return rows


def _same(ours: list[tuple], theirs: list[tuple], tol: float) -> None:
    assert len(ours) == len(theirs)
    for k, (a, b) in enumerate(zip(ours, theirs, strict=True), start=1):
        assert a[-1] == b[-1], (k, "tag")
        # BETA (index 5) has no value on a vertical segment.
        fields = range(7) if abs(b[4]) < 89.999 else (0, 1, 2, 3, 4, 6)
        for f in fields:
            assert a[f] == pytest.approx(b[f], abs=tol), (k, f, a, b)


def _helix_turns(rows: list[tuple]) -> float:
    """The turns a helix's segment centres sweep about Z, end to end, plus
    one segment's share."""
    angles = [math.atan2(r[1], r[0]) for r in rows]
    steps = [
        (b - a + math.pi) % (2 * math.pi) - math.pi
        for a, b in zip(angles, angles[1:], strict=False)
    ]
    return sum(steps) * len(rows) / (len(rows) - 1) / (2 * math.pi)


# --------------------------------------------------------------------------
# the paths


def _nec2_python(name: str) -> list[tuple]:
    from momwire.portal._portal import file_printout

    return _nec2c_rows(file_printout((NEC2C / f"{name}.nec").read_text()))


def _nec42_python(name: str) -> list[tuple]:
    from momwire.eznec._shell import render

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = render(
            (NEC42 / "probes" / f"{name}.nec").read_text(),
            basis="sinusoidal",
            dialect="nec4",
        )
    assert "NEC ERROR" not in text
    return _nec42_rows(text)


def _run_compare():
    spec = importlib.util.spec_from_file_location(
        "core_split_nec4_run_compare_1651",
        ROOT / "scripts" / "core_split" / "nec4_run_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nec42_cxx(name: str) -> list[tuple]:
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "sinusoidal_nec4_printout"):
        pytest.skip("no compiled accelerator with the NEC-4.2 run half")
    rc = _run_compare()
    try:
        blas = rc.blas_addresses()
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        text = (NEC42 / "probes" / f"{name}.nec").read_text()
        ok, printout = rc.cxx_printout(text, blas=blas)
    assert ok, printout[-400:]
    return _nec42_rows(printout)


def _nec42_exe(name: str, tmp_path: Path) -> list[tuple]:
    out = tmp_path / f"{name}.out"
    subprocess.run(
        [EXE, str(NEC42 / "probes" / f"{name}.nec"), str(out)],
        check=True,
        capture_output=True,
    )
    return _nec42_rows(out.read_bytes().decode("latin-1"))


def _licensed(name: str) -> list[tuple]:
    return _nec42_rows((NEC42 / "printouts" / f"{name}.out").read_text())


# --------------------------------------------------------------------------
# the gates


@pytest.mark.parametrize("card", sorted(CASES))
def test_nec2_mode_builds_nec2c_s_segments(card):
    nec2c, _ = CASES[card]
    want = _nec2c_rows((NEC2C / f"{nec2c}.out").read_text())
    _same(_nec2_python(nec2c), want, _NEC2C_TOL)


@pytest.mark.parametrize("card", sorted(CASES))
def test_the_nec42_python_path_builds_nec42_s_segments(card):
    _, nec42 = CASES[card]
    _same(_nec42_python(nec42), _licensed(nec42), _NEC42_TOL)


@pytest.mark.parametrize("card", sorted(CASES))
def test_the_nec42_cxx_route_builds_nec42_s_segments(card):
    _, nec42 = CASES[card]
    _same(_nec42_cxx(nec42), _licensed(nec42), _NEC42_TOL)


@pytest.mark.skipif(
    not EXE, reason="MOMWIRE_NEC42_SIN_STANDALONE_EXE does not name a built executable"
)
@pytest.mark.parametrize("card", sorted(CASES))
def test_the_nec42_executable_builds_nec42_s_segments(card, tmp_path):
    _, nec42 = CASES[card]
    _same(_nec42_exe(nec42, tmp_path), _licensed(nec42), _NEC42_TOL)


def test_the_one_gh_card_is_two_helices():
    """The dialects disagree on GH, and each reference says so: three turns
    in nec2c, half a turn in NEC-4.2, from the same card."""
    nec2c, nec42 = CASES["GH"]
    assert _helix_turns(_nec2c_rows((NEC2C / f"{nec2c}.out").read_text())) == (
        pytest.approx(3.0, abs=1e-3)
    )
    assert _helix_turns(_licensed(nec42)) == pytest.approx(0.5, abs=1e-3)


def test_the_one_ga_card_is_one_arc():
    """GA is the same arc to both engines: nine 0.69459 m chords in the XZ
    plane, the first centred at 10 degrees, 10 m up."""
    nec2c, nec42 = CASES["GA"]
    a = _nec2c_rows((NEC2C / f"{nec2c}.out").read_text())
    b = _licensed(nec42)
    _same(a, b, _NEC2C_TOL)
    assert all(r[3] == pytest.approx(0.69459, abs=6e-6) for r in b)
    assert math.degrees(math.atan2(b[0][2] - 10.0, b[0][0])) == pytest.approx(10.0)
