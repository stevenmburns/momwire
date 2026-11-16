"""``RP 0`` with XNDA N >= 1: the NORMALIZED GAIN table.

A census of every deck at hand (2026-11-15) found NEC-5 and NEC-4.2 user
decks refused for asking for one (the W8IO ELT whip's ``RP ... 1500`` among
them). The table's printout was read black-box off the licensed x13 engine
for synthetic decks (``tests/fixtures/rp_normalized_gain/``, whose README
lists what each deck probes and what the captures show). Each rule is pinned
twice where it can be: on the engine's captures, so the evidence is in the
tree, and on momwire's printouts of the same decks.
"""

from __future__ import annotations

import gzip
import math
import re
from pathlib import Path

import pytest

from momwire.deck._nec5 import DeckError, parse_nec5
from momwire.dropin.nec5 import _serve, _shell

FIXTURES = Path(__file__).parent / "fixtures" / "rp_normalized_gain"
HEADING = "- - - - NORMALIZED GAIN - - - -"
KINDS = {
    "MAJOR AXIS GAIN": 1,
    "MINOR AXIS GAIN": 2,
    "VERTICAL GAIN": 3,
    "HORIZONTAL GAIN": 4,
    "TOTAL GAIN": 5,
}
# momwire's gains against x13's on these decks: the two solvers differ in
# their last digits as everywhere (momwire's B-spline and razor bases against
# NEC-5's own), and a cell is a difference of two such gains. MEASURED over
# all 8082 compared cells on bspline and razor-2p (2026-11-15): worst 0.04 dB
# (bspline, free1101_gnor10, a cell 24 dB down) and 0.01 dB (razor-2p). The
# bar is that with headroom, far below anything the table's layout decides
# (a column split, a sign, which row is the peak).
CELL_DB = 0.1


def _decks() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.nec"))


def _engine(stem: str) -> str:
    path = FIXTURES / f"{stem}.nec5.out"
    if not path.exists():
        data = gzip.decompress((FIXTURES / f"{stem}.nec5.out.gz").read_bytes())
    else:
        data = path.read_bytes()
    return data.decode("latin-1").replace("\r\n", "\n")


def _deck(stem: str) -> str:
    return (FIXTURES / f"{stem}.nec").read_bytes().decode("latin-1")


_RENDERED: dict[tuple[str, str], str] = {}


def _momwire(stem: str, basis: str = "bspline") -> str:
    key = (stem, basis)
    if key not in _RENDERED:
        _RENDERED[key] = _shell.render(_deck(stem), basis=basis).replace("\r\n", "\n")
    return _RENDERED[key]


def _cell_groups(line: str) -> list[tuple[float, float, float]]:
    values = [float(v) for v in line.split()]
    assert len(values) % 3 == 0, line
    return [tuple(values[k : k + 3]) for k in range(0, len(values), 3)]


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
        groups: list[list[tuple[float, float, float]]] = [[], [], []]
        for row in rows:
            for g, cell in enumerate(_cell_groups(row)):
                groups[g].append(cell)
        out.append(
            {
                "start": k,
                "lines": lines[k:j],
                "kind": KINDS[label],
                "factor": factor,
                "rows": rows,
                "groups": groups,
                "cells": groups[0] + groups[1] + groups[2],
                "after": lines[j:],
            }
        )
    return out


_ROW = re.compile(r"^\s*-?\d+\.\d\d\s+-?\d+\.\d\d\s+-?\d+\.\d\d\s")


def _patterns(text: str) -> list[list[tuple[float, ...]]]:
    """Each RADIATION PATTERNS block's rows: theta, phi, vert, hor, total."""
    lines = text.split("\n")
    out = []
    for k, line in enumerate(lines):
        if "RADIATION PATTERNS" not in line:
            continue
        rows = []
        for row in lines[k + 1 :]:
            if HEADING in row or "INPUT LINE" in row or "FREQUENCY" in row:
                break
            if _ROW.match(row) and len(row.split()) >= 11:
                rows.append(tuple(float(v) for v in row.split()[:5]))
        out.append(rows)
    return out


def _gnor(stem: str) -> float:
    card = next(line for line in _deck(stem).splitlines() if line.startswith("RP"))
    fields = card.split()[1:]
    return float(fields[9]) if len(fields) > 9 else 0.0


def _xnda_tables(stem: str) -> list[int]:
    """The N digit of each RP card in the deck, once per frequency."""
    text = _deck(stem)
    kinds = [
        int(line.split()[4]) // 100 % 10
        for line in text.splitlines()
        if line.startswith("RP")
    ]
    fr = next(line for line in text.splitlines() if line.startswith("FR"))
    return kinds * max(int(fr.split()[2]), 1)


def test_the_fixtures_are_there():
    stems = _decks()
    assert len(stems) == 54, len(stems)
    for stem in stems:
        assert (FIXTURES / f"{stem}.nec5.out").exists() or (
            FIXTURES / f"{stem}.nec5.out.gz"
        ).exists(), stem


@pytest.mark.parametrize("stem", _decks())
def test_the_engine_prints_one_table_per_card_with_n(stem):
    tables = _tables(_engine(stem))
    assert [t["kind"] for t in tables] == [n for n in _xnda_tables(stem) if n], stem


@pytest.mark.parametrize("stem", _decks())
def test_the_engine_splits_the_cells_into_three_columns(stem):
    for table in _tables(_engine(stem)):
        n = len(table["cells"])
        assert [len(g) for g in table["groups"]] == [
            n // 3 + (g < n % 3) for g in range(3)
        ], stem


def test_the_engine_has_no_cap_on_the_cell_count():
    (table,) = _tables(_engine("free1500_big"))
    assert [len(g) for g in table["groups"]] == [901, 900, 900]


def _column(n: int) -> int | None:
    return {3: 2, 4: 3, 5: 4}.get(n)


@pytest.mark.parametrize("source", ["engine", "bspline"])
@pytest.mark.parametrize("stem", _decks())
def test_each_cell_is_its_rows_gain_less_the_factor(stem, source):
    """N = 3 / 4 / 5 read the VERT. / HOR. / TOTAL column; the factor is
    GNOR when nonzero, else the largest of them. Each printed value is
    rounded once, so a cell is the rounded gain less the rounded factor
    within two half-units."""
    text = _engine(stem) if source == "engine" else _momwire(stem, source)
    tables = _tables(text)
    patterns = [
        p for p, n in zip(_patterns(text), _xnda_tables(stem), strict=True) if n
    ]
    assert len(tables) == len(patterns)
    gnor = _gnor(stem)
    for table, rows in zip(tables, patterns, strict=True):
        if "A = 2" in stem or stem.endswith(("1502", "1512")):
            continue  # A = 2 prints no rows to read the gains from
        assert len(table["cells"]) == len(rows)
        column = _column(table["kind"])
        if column is None:
            gains = None
        else:
            gains = [row[column] for row in rows]
        for cell, row in zip(table["cells"], rows, strict=True):
            assert cell[:2] == row[:2]
        if gains is None:
            continue
        factor = gnor if gnor else max(gains)
        assert abs(table["factor"] - factor) <= 0.0051, stem
        for cell, gain in zip(table["cells"], gains, strict=True):
            assert abs(cell[2] - (gain - table["factor"])) <= 0.0101, (stem, cell, gain)


@pytest.mark.parametrize("basis", ["bspline", "razor-2p"])
@pytest.mark.parametrize("stem", _decks())
def test_momwire_lays_the_table_out_as_the_engine(stem, basis):
    """Every table line for line: the heading, the name, the factor line's
    text, the column headers and the angles are the engine's characters, the
    cells within CELL_DB, and the blank run after the table is the engine's.
    The minor-axis tables of a LINEAR structure are the one exception (the
    engine prints round-off there, see the last test)."""
    engine = _tables(_engine(stem))
    mine = _tables(_momwire(stem, basis))
    assert len(engine) == len(mine)
    for e, m in zip(engine, mine, strict=True):
        assert e["kind"] == m["kind"]
        assert e["lines"][:3] == m["lines"][:3]
        assert e["lines"][3].split("=")[0] == m["lines"][3].split("=")[0]
        assert e["lines"][4:8] == m["lines"][4:8]
        assert [len(g) for g in e["groups"]] == [len(g) for g in m["groups"]]
        assert [len(r) for r in e["rows"]] == [len(r) for r in m["rows"]]
        blank = lambda after: next(k for k, line in enumerate(after) if line.strip())  # noqa: E731
        assert blank(e["after"]) == blank(m["after"]) == 4
        if e["kind"] == 2 and not stem.startswith("xdip"):
            continue
        assert abs(e["factor"] - m["factor"]) <= CELL_DB, stem
        for ec, mc in zip(e["cells"], m["cells"], strict=True):
            assert ec[:2] == mc[:2]
            if ec[2] < -900 or mc[2] < -900:
                assert abs(ec[2] - mc[2]) <= CELL_DB + 0.2, (stem, ec, mc)
                continue
            assert abs(ec[2] - mc[2]) <= CELL_DB, (stem, ec, mc)


def _block(stem: str, basis: str = "razor-2p") -> list:
    data = _serve.serve(parse_nec5(_deck(stem)), basis=basis)
    return data.patterns


@pytest.mark.parametrize(
    "stem", ["free1101", "free1201", "free1301", "free1401", "free1511", "xdip1201"]
)
def test_momwire_cells_are_the_gains_less_the_factor_exactly(stem):
    (block,) = _block(stem)
    kind = block.normalized_kind
    assert kind == int(stem[-3]) if stem.startswith("free") else kind == 2
    db = [cell[2] + block.normalization_db for cell in block.normalized]
    assert block.normalization_db == max(db)
    assert len(block.normalized) == len(block.rows)
    for row, cell in zip(block.rows, block.normalized, strict=True):
        assert (row.theta_deg, row.phi_deg) == cell[:2]
        if kind == 3:
            assert cell[2] == row.vert_db - block.normalization_db
        elif kind == 4:
            assert cell[2] == row.hor_db - block.normalization_db
        elif kind == 5:
            assert cell[2] == row.total_db - block.normalization_db
        else:
            total = 10 ** (row.total_db / 10)
            ar2 = row.axial_ratio**2
            share = 1 / (1 + ar2) if kind == 1 else ar2 / (1 + ar2)
            expect = (
                10 * math.log10(total * share) if total * share > 1e-20 else -999.99
            )
            assert abs(cell[2] + block.normalization_db - expect) <= 1e-9


@pytest.mark.parametrize("source", ["engine", "bspline"])
def test_a2_prints_the_table_without_the_rows(source):
    get = _engine if source == "engine" else _momwire
    a1, a2 = _tables(get("free1501")), _tables(get("free1502"))
    assert [t["lines"] for t in a1] == [t["lines"] for t in a2]
    assert not _patterns(get("free1502"))[0]
    a0 = _tables(get("free1500"))
    assert [t["lines"] for t in a0] == [t["lines"] for t in a1]


@pytest.mark.parametrize("source", ["engine", "bspline"])
def test_d1_moves_the_factor_and_not_the_cells(source):
    get = _engine if source == "engine" else _momwire
    for n in (1, 5):
        d0, d1 = _tables(get(f"free1{n}01"))[0], _tables(get(f"free1{n}11"))[0]
        assert d0["cells"] == d1["cells"]
        assert d1["factor"] - d0["factor"] == pytest.approx(1.34, abs=0.011)


@pytest.mark.parametrize("source", ["engine", "bspline"])
def test_gnor_is_the_factor_in_db(source):
    get = _engine if source == "engine" else _momwire
    base = _tables(get("free1501"))[0]
    for tag, gnor in (("10", 10.0), ("0p5", 0.5), ("m3", -3.0)):
        table = _tables(get(f"free1501_gnor{tag}"))[0]
        assert table["factor"] == gnor
        for b, c in zip(base["cells"], table["cells"], strict=True):
            assert abs((c[2] - b[2]) - (base["factor"] - gnor)) <= 0.0101


def test_a_null_column_normalises_to_zero_everywhere():
    """zdip1401: a z-directed dipole has no horizontal gain anywhere, so
    the factor is the null and every cell prints 0.00, in both."""
    for text in (_engine("zdip1401"), _momwire("zdip1401")):
        (table,) = _tables(text)
        assert table["factor"] == -999.99
        assert {c[2] for c in table["cells"]} == {0.0}


def test_a_linear_structures_minor_axis_is_round_off_in_the_engine():
    """The one place momwire does not print the engine's digits, and why:
    on the tilted dipole every row is LINEAR (axial ratio 0.00000) and the
    engine's minor-axis gains are round-off (a factor of -164.43 dB, cells
    from -5.35 to -835.56 that move with the arithmetic). momwire prints a
    linear row's minor axis as the null, so the table is the floor
    normalised to itself."""
    (engine,) = _tables(_engine("free1201"))
    assert engine["factor"] < -100
    (mine,) = _tables(_momwire("free1201"))
    assert mine["factor"] == -999.99
    assert {c[2] for c in mine["cells"]} == {0.0}


@pytest.mark.parametrize("xnda", [1600, 1701, 1813, 1902])
def test_an_n_nec_does_not_define_refuses_by_digit(xnda):
    text = _deck("free1501").replace(" 1501 ", f" {xnda} ")
    with pytest.raises(DeckError) as excinfo:
        parse_nec5(text)
    message = str(excinfo.value)
    assert f"normalization N = {xnda // 100 % 10}, which NEC does not define" in message
    assert "(N = 0 to 5)" in message


@pytest.mark.parametrize("xnda", [1100, 1213, 1302, 1411, 1500, 1513])
def test_every_n_from_one_to_five_parses(xnda):
    text = _deck("free1501").replace(" 1501 ", f" {xnda} ")
    (request,) = parse_nec5(text).requests[:1]
    assert request.xnda == xnda
