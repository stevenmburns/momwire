"""``RP 0`` with XNDA N >= 1 on the NEC-4.2 slot: the NORMALIZED GAIN table.

The 11-17 census found user decks refused on the NEC-4.2 slot for asking for
the table ("RP with XNDA ... is not served by this engine's NEC-4
printout"), which the NEC-5 seam has served since momwire#1961. The layout
was read black-box off licensed NEC-4.2 for synthetic decks
(``tests/fixtures/nec42_rp_normalized_gain/``, README there): it is the
NEC-5 printout's table to the byte, and so is the gap after it. Each rule is
pinned on the engine's printouts, so the evidence is in the tree, and on the
slot's printouts of the same decks; the C++ printout core and the C++ run
half are held to the Python printout byte for byte.
"""

from __future__ import annotations

import gzip
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from momwire.deck._cards import parse_card
from momwire.dropin.nec5 import _nec4, _printout
from momwire.dropin.nec5._shell import render

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "nec42_rp_normalized_gain"
DECKS = FIXTURES / "decks"
HEADING = "- - - - NORMALIZED GAIN - - - -"
KINDS = {
    "MAJOR AXIS GAIN": 1,
    "MINOR AXIS GAIN": 2,
    "VERTICAL GAIN": 3,
    "HORIZONTAL GAIN": 4,
    "TOTAL GAIN": 5,
}
BASES = ("bspline", "sinusoidal")
# The slot's cells against licensed NEC-4.2's: a cell is a difference of two
# gains, and momwire's bases are not NEC-4.2's. MEASURED over all 4159
# compared cells of the 66 decks (2026-11-17, Rocketlake), linear minor axes
# aside: worst 0.03 dB on bspline and on sinusoidal alike (free1101_gnor0p5,
# a cell 13.5 dB down). The bar is that with headroom, far below anything the
# layout decides (a column split, a sign, which row is the peak).
CELL_DB = 0.1


def _stems() -> list[str]:
    return sorted(p.stem for p in DECKS.glob("*.nec"))


def _deck(stem: str) -> str:
    return (DECKS / f"{stem}.nec").read_bytes().decode("latin-1")


def _engine(stem: str) -> str:
    path = FIXTURES / "nec42" / f"{stem}.out"
    if path.exists():
        data = path.read_bytes()
    else:
        data = gzip.decompress((FIXTURES / "nec42" / f"{stem}.out.gz").read_bytes())
    return data.decode("latin-1").replace("\r\n", "\n")


# A GN 2 / GN 3 card that NAMES its table file (every ground deck here writes
# SimNEC's NOFILE) prints the line saying the file was not read, and this
# engine prints it too; the lines about the table the licensed engine then
# computes are its own file handling and leave the reference
# (`test_eznec_nec42_printout_1295`'s `_NAMED_CACHE_BLOCKS`).
_NAMED_CACHE_BLOCKS = (
    (
        re.compile(
            r"(\n\n GNDINO: [^\n]*\n) WILL COMPUTE SOMMERFELD-GROUND TABLES\n"
            r"\n Time to generate Sommerfeld ground tables =[^\n]*\n"
        ),
        r"\1",
    ),
    (
        re.compile(
            r"(\n\n GNDINO3: [^\n]*)\n Will compute Sommerfeld-ground tables(?=\n)"
        ),
        r"\1",
    ),
)


def _reference(stem: str) -> str:
    text = _engine(stem)
    for block, kept in _NAMED_CACHE_BLOCKS:
        text = block.sub(kept, text)
    return text


_RENDERED: dict[tuple[str, str], str] = {}


def _momwire(stem: str, basis: str = "bspline") -> str:
    key = (stem, basis)
    if key not in _RENDERED:
        text = render(_deck(stem), basis=basis, dialect="nec4")
        _RENDERED[key] = text.replace("\r\n", "\n")
    return _RENDERED[key]


def _tables(text: str) -> list[dict]:
    """Every NORMALIZED GAIN table in a printout: its lines, its kind, its
    factor, and its cells in the order the columns hold them."""
    lines = text.split("\n")
    out = []
    for k, line in enumerate(lines):
        if HEADING not in line:
            continue
        label = lines[k + 2].strip()
        factor = float(lines[k + 3].split("=")[1].split()[0])
        rows = []
        j = k + 8
        while j < len(lines) and lines[j].strip():
            rows.append(lines[j])
            j += 1
        groups: list[list[tuple[float, ...]]] = [[], [], []]
        for row in rows:
            values = [float(v) for v in row.split()]
            assert len(values) % 3 == 0, row
            for g in range(len(values) // 3):
                groups[g].append(tuple(values[3 * g : 3 * g + 3]))
        out.append(
            {
                "lines": lines[k:j],
                "kind": KINDS[label],
                "factor": factor,
                "rows": rows,
                "groups": groups,
                "cells": groups[0] + groups[1] + groups[2],
                "before": lines[:k],
                "after": lines[j:],
            }
        )
    return out


def _requests(stem: str) -> list[int]:
    """The N digit of each RP card, once per frequency of the deck's FR."""
    cards = [parse_card(line) for line in _deck(stem).splitlines()]
    cards = [c for c in cards if c is not None]
    kinds = [c.i(3) // 100 % 10 for c in cards if c.mnemonic == "RP"]
    (fr,) = [c for c in cards if c.mnemonic == "FR"]
    return kinds * max(fr.i(1), 1)


def _leading_blanks(lines: list[str]) -> int:
    return next(k for k, line in enumerate(lines) if line.strip())


# ---------------------------------------------------------------------------
# the fixtures, and what the licensed engine prints
# ---------------------------------------------------------------------------


def test_the_fixtures_are_the_generators_decks(tmp_path):
    stems = _stems()
    assert len(stems) == 66, len(stems)
    for stem in stems:
        assert (FIXTURES / "nec42" / f"{stem}.out").exists() or (
            FIXTURES / "nec42" / f"{stem}.out.gz"
        ).exists(), stem
    subprocess.run(
        [sys.executable, str(FIXTURES / "gen_decks.py"), str(tmp_path)], check=True
    )
    assert sorted(p.name for p in tmp_path.glob("*.nec")) == sorted(
        p.name for p in DECKS.glob("*.nec")
    )
    for path in tmp_path.glob("*.nec"):
        assert path.read_bytes() == (DECKS / path.name).read_bytes(), path.name


@pytest.mark.parametrize("stem", _stems())
def test_the_engine_prints_one_table_per_card_with_n(stem):
    text = _engine(stem)
    assert "NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)" in text
    assert [t["kind"] for t in _tables(text)] == [n for n in _requests(stem) if n]


@pytest.mark.parametrize("stem", _stems())
def test_the_engine_prints_the_nec5_layout(stem):
    """The heading, the name, the factor line, the column headings and the
    column split are the NEC-5 printout's (`_printout._normalized`), and the
    blank run after the table is four lines."""
    for table in _tables(_engine(stem)):
        lines = table["lines"]
        assert lines[0] == _printout._NORMALIZED_HEADING
        assert lines[1] == "" and lines[4] == ""
        assert lines[2] == _printout._NORMALIZED_KINDS[table["kind"]].rjust(53)
        assert lines[3] == " " * 38 + (
            f"NORMALIZATION FACTOR ={table['factor']:9.2f} DB"
        )
        assert tuple(lines[5:8]) == _printout._NORMALIZED_COLUMNS
        n = len(table["cells"])
        assert [len(g) for g in table["groups"]] == [
            n // 3 + (g < n % 3) for g in range(3)
        ]
        for row in table["rows"]:
            cells = [tuple(row.split()[k : k + 3]) for k in range(0, 9, 3)]
            assert row == _printout._NORMALIZED_SEPARATOR.join(
                _printout._NORMALIZED_CELL % tuple(float(v) for v in cell)
                for cell in cells
                if cell
            )
        assert _leading_blanks(table["before"][::-1]) == 4
        assert _leading_blanks(table["after"]) == 4


def test_the_engine_has_no_cap_on_the_cell_count():
    (table,) = _tables(_engine("free1500_big"))
    assert [len(g) for g in table["groups"]] == [901, 900, 900]


@pytest.mark.parametrize("stem", ["free1501", "free1511"])
def test_a2_prints_the_table_after_the_trailer_with_no_rows(stem):
    a1 = _tables(_engine(stem))
    a2 = _tables(_engine(stem[:-1] + "2"))
    assert [t["lines"] for t in a1] == [t["lines"] for t in a2]
    # the trailer, its blank and its radiated-power line, then four blanks
    assert a2[0]["before"][-7].startswith("   AVERAGE POWER GAIN=")
    assert a2[0]["before"][-5].startswith("   POWER RADIATED ASSUMING")


def test_gnor_is_the_factor_in_db():
    for n in (1, 5):
        base = _tables(_engine(f"free1{n}01"))[0]
        for tag, gnor in (("gnor10", 10.0), ("gnor0p5", 0.5), ("gnorm3", -3.0)):
            table = _tables(_engine(f"free1{n}01_{tag}"))[0]
            assert table["factor"] == gnor
            for b, c in zip(base["cells"], table["cells"], strict=True):
                assert abs((c[2] - b[2]) - (base["factor"] - gnor)) <= 0.0101


# ---------------------------------------------------------------------------
# the slot's printout
# ---------------------------------------------------------------------------


def test_the_census_refusal_is_gone():
    """The census's refusal, on a deck of the same shape (the sentence the
    slot printed for every N >= 1 before)."""
    text = _momwire("free1501")
    assert "NEC ERROR" not in text
    assert "is not served by this engine's NEC-4 printout" not in text
    (table,) = _tables(text)
    assert table["kind"] == 5


@pytest.mark.parametrize("xnda", [1600, 1701, 1014, 1904, 2501])
def test_an_xnda_no_capture_shows_still_refuses(xnda):
    text = _deck("free1501").replace(" 1501 ", f" {xnda} ")
    with pytest.raises(_nec4.Nec4Refusal, match=f"RP with XNDA {xnda}"):
        _nec4.solve(text, basis="bspline")


# A few decks on the default lane, the whole set under `integration`.
_QUICK = (
    "free1101",
    "free1501_gnor10",
    "free1513",
    "small4",
    "free_tworp_mixed2",
    "cut1001",
    "phicut1502",
)


def _served_layout(stem: str, basis: str) -> None:
    # `test_eznec_nec42_printout_1295`'s column skeleton: every literal
    # character equal, every number in the same field and format
    spec = importlib.util.spec_from_file_location(
        "nec42_printout_1295", ROOT / "tests" / "test_eznec_nec42_printout_1295.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ours = mod._unstamped(_momwire(stem, basis))
    theirs = mod._unstamped(_reference(stem))
    assert "NEC ERROR" not in "\n".join(ours)
    assert len(ours) == len(theirs), (len(ours), len(theirs))
    for number, (a, b) in enumerate(zip(ours, theirs, strict=True), start=1):
        assert mod.skeleton(a) == mod.skeleton(b), (number, a, b)


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("stem", _QUICK)
def test_the_slot_prints_the_licensed_columns_line_for_line(stem, basis):
    _served_layout(stem, basis)


@pytest.mark.integration
@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("stem", [s for s in _stems() if s not in _QUICK])
def test_every_deck_prints_the_licensed_columns_line_for_line(stem, basis):
    _served_layout(stem, basis)


def _linear_minor(stem: str, table: dict) -> bool:
    """A minor-axis table of a linearly polarised structure: the engine
    prints round-off there and momwire the null (`rp_normalized_gain`'s
    README), so its cells are not compared."""
    return table["kind"] == 2 and not stem.startswith("xdip")


@pytest.mark.integration
@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("stem", _stems())
def test_the_cells_are_the_engines_within_the_bar(stem, basis):
    engine, mine = _tables(_engine(stem)), _tables(_momwire(stem, basis))
    assert [t["kind"] for t in engine] == [t["kind"] for t in mine]
    for e, m in zip(engine, mine, strict=True):
        assert e["lines"][:3] == m["lines"][:3] and e["lines"][4:8] == m["lines"][4:8]
        if _linear_minor(stem, e):
            assert m["factor"] == -999.99
            continue
        assert abs(e["factor"] - m["factor"]) <= CELL_DB, stem
        for ec, mc in zip(e["cells"], m["cells"], strict=True):
            assert ec[:2] == mc[:2]
            if ec[2] < -900 or mc[2] < -900:
                assert ec[2] < -900 and mc[2] < -900, (stem, ec, mc)
                continue
            assert abs(ec[2] - mc[2]) <= CELL_DB, (stem, ec, mc)


# ---------------------------------------------------------------------------
# the C++ printout core and run half print Python's bytes
# ---------------------------------------------------------------------------


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "nec4_render_printout"):
        pytest.skip("no compiled accelerator with the NEC-4.2 printout writer")
    return _accel.acc


def _quick_paths() -> list[Path]:
    return [DECKS / f"{stem}.nec" for stem in _QUICK]


def _all_paths() -> list[Path]:
    return sorted(DECKS.glob("*.nec"))


def _printout_core(paths: list[Path], basis: str) -> None:
    nc = _load(
        "core_split_nec4_printout_compare",
        ROOT / "scripts" / "core_split" / "nec4_printout_compare.py",
    )
    counts, mismatches = nc.compare(paths, basis)
    assert not mismatches, mismatches[:2]
    assert counts["printout"] == len(paths), counts


@pytest.mark.parametrize("basis", BASES)
def test_the_printout_core_prints_the_same(basis, acc):
    _printout_core(_quick_paths(), basis)


@pytest.mark.slow
@pytest.mark.parametrize("basis", BASES)
def test_the_printout_core_prints_every_deck_the_same(basis, acc):
    _printout_core(_all_paths(), basis)


def _run_half(paths: list[Path], basis: str, monkeypatch) -> dict:
    rc = _load(
        "core_split_nec4_run_compare",
        ROOT / "scripts" / "core_split" / "nec4_run_compare.py",
    )
    try:
        rc.blas_addresses(basis)
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    from threadpoolctl import threadpool_limits

    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")
    with threadpool_limits(limits=1):
        counts, mismatches = rc.compare(paths, basis=basis)
    assert not mismatches, mismatches[:2]
    assert counts["py_raised"] == 0, counts
    return counts


@pytest.mark.parametrize("basis", BASES)
def test_the_run_half_prints_the_same(basis, acc, monkeypatch):
    counts = _run_half(_quick_paths(), basis, monkeypatch)
    assert counts["printed"] == len(_QUICK), counts


@pytest.mark.slow
@pytest.mark.parametrize("basis", BASES)
def test_the_run_half_prints_every_deck_the_same(basis, acc, monkeypatch):
    counts = _run_half(_all_paths(), basis, monkeypatch)
    assert counts["printed"] == len(_all_paths()), counts
