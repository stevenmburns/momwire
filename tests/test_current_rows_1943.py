"""momwire#1943 lever 3: the CURRENTS AND LOCATION (and CHARGE DENSITIES)
tables are formatted as columns, line for line what the row formatter prints.

The sym4 ELT whip prints 4,182 rows at each of 13 frequencies; a typed record
plus an f-string per row cost more than the readout behind them. The gates:
`fmt_current_rows` equals `fmt_current_row` row by row, over values that
exercise every formatting edge (zeros of both signs, NaN, infinities, the
widths' overflow), and over a real deck's tables through `_run_records`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from momwire.dropin.nec2c import _portal

FIXTURES = Path(__file__).parent / "fixtures" / "nec_portal"


def _row_by_row(numbers, tags, centres, lengths, values):
    return [
        _portal.fmt_current_row(int(n), int(t), c, float(length), complex(v))
        for n, t, c, length, v in zip(numbers, tags, centres, lengths, values)
    ]


def test_the_column_form_is_the_row_form_line_for_line():
    rng = np.random.default_rng(1943)
    n = 4000
    numbers = np.arange(1, n + 1, dtype=np.int64)
    numbers[-3:] = [99999, 123456, 7]  # past the %5d width, which still prints
    tags = rng.integers(0, 20000, n)
    centres = rng.normal(scale=10.0 ** rng.integers(-6, 6, (n, 1)), size=(n, 3))
    lengths = np.abs(rng.normal(scale=1e-2, size=n))
    re = rng.normal(scale=10.0 ** rng.integers(-30, 30, n))
    im = rng.normal(scale=10.0 ** rng.integers(-30, 30, n))
    values = re + 1j * im
    special = [
        0j,
        complex(-0.0, 0.0),
        complex(0.0, -0.0),
        complex(-0.0, -0.0),
        complex(np.nan, 0.0),
        complex(0.0, np.nan),
        complex(np.inf, 1.0),
        complex(-np.inf, -np.inf),
        complex(1e-320, -1e-320),  # subnormal
        complex(-1.0, 0.0),  # phase 180
        complex(-1.0, -0.0),  # phase -180
    ]
    values[: len(special)] = special
    centres[0] = [-0.0, np.nan, np.inf]
    lengths[1] = -0.0
    got = _portal.fmt_current_rows(numbers, tags, centres, lengths, values)
    want = _row_by_row(numbers, tags, centres, lengths, values)
    assert got == want


def test_an_empty_table_prints_nothing():
    empty = np.zeros(0)
    assert (
        _portal.fmt_current_rows(
            empty.astype(int), empty.astype(int), empty, empty, empty
        )
        == []
    )


@pytest.mark.parametrize(
    "name", ["dipole_pq_charges", "dipole_pq_segment_range", "apex_pq_reversed_walk"]
)
def test_a_decks_tables_print_as_their_rows(name):
    """Through `_run_records`: each table's `lines()` is its own `rows()`
    formatted one at a time, and the rows are the segments' own geometry over
    the wavelength."""
    path = FIXTURES / f"{name}.deck"
    if not path.exists():
        pytest.fail(f"fixture {name} moved: the gate would measure nothing")
    deck = _portal.parse_deck(path.read_text())
    solver = _portal.DeckSolver(deck, basis="sinusoidal")
    group = next(g for g in deck.groups if g is not None)
    records = _portal._run_records(solver, group, group.freqs_mhz[0], 0)
    tables = [records.current_rows]
    if records.charge_rows is not None:
        tables.append(records.charge_rows)
    assert len(tables) == 2  # every deck above prints charges: both measured
    wl = records.result.wavelength
    for table in tables:
        rows = table.rows()
        assert len(rows) == len(table) > 0
        assert table.lines() == [
            _portal.fmt_current_row(r.number, r.tag, r.centre_wl, r.length_wl, r.value)
            for r in rows
        ]
        for r in rows:
            seg = solver.segments[r.number - 1]
            assert r.tag == seg.tag
            assert (seg.centre / wl).tobytes() == np.asarray(r.centre_wl).tobytes()
            assert r.length_wl == float(_portal.norm3(seg.direction)) / wl
