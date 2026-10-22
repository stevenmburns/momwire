"""The C++ NEC-4.2 printout writer is the Python one, byte for byte.

`src/momwire/_nec4_printout_core.cpp` copies `eznec/_nec4_printout.py`'s
`render_nec4_printout` and the `_printout` sections it shares, plus the
slot's refusal printout (momwire#1607 A3). It is a formatter, so these tests
hand it the very `Nec4Printout` Python's own solve built
(`scripts/core_split/nec4_printout_compare.py`), and hold the text equal on:
- the NEC-4.2 slot's own fixtures (the EZNEC captures and the probes),
  solved for both served bases;
- every fixture deck the slot refuses before any solve, with no Python in
  the C++ path (the seam or the build refuses, and the C++ writes it);
- hand-built `Nec4Printout` records for the shapes no fixture prints: the GS
  and GM rows, a further-table report, a named ground file (GN 2 and GN 3),
  the FAR FIELD GROUND block, a range pattern with its lower-medium line and
  a three-digit exponent, a directive pattern, mixed TL/NT network runs,
  every loading-row kind, and NaN, infinity and negative zero in the cells;
- the whole fixture tree, solved, in the slow lane.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

from momwire.eznec import _nec4_printout as n4p
from momwire.eznec import _printout as p

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the reader copies CPython 3.12's float rules"
)


def _compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_nec4_printout_compare",
        ROOT / "scripts" / "core_split" / "nec4_printout_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _compare_module()


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "nec4_render_printout"):
        pytest.skip("no compiled accelerator with the NEC-4.2 printout writer")
    return _accel.acc


def _decks(root: pathlib.Path) -> list[pathlib.Path]:
    return [
        x
        for x in sorted(root.rglob("*"))
        if x.is_file() and x.suffix in (".nec", ".deck")
    ]


@pytest.mark.parametrize("basis", ["sinusoidal", "bspline"])
def test_the_nec42_fixtures_print_the_same(basis, acc):
    counts, mismatches = nc.compare(_decks(FIXTURES / "eznec_nec42"), basis)
    assert not mismatches, mismatches[:2]
    assert counts["printout"] >= 50, counts


@pytest.mark.parametrize("basis", ["sinusoidal", "bspline"])
def test_every_refusal_before_a_solve_prints_the_same(basis, acc):
    counts, mismatches = nc.compare(_decks(FIXTURES), basis, refusals_only=True)
    assert not mismatches, mismatches[:2]
    assert counts["refusal"] >= 150, counts


@pytest.mark.slow
@pytest.mark.parametrize("basis", ["sinusoidal", "bspline"])
def test_every_fixture_deck_prints_the_same(basis, acc):
    counts, mismatches = nc.compare(_decks(FIXTURES), basis)
    assert not mismatches, mismatches[:2]
    assert counts["printout"] >= 200 and counts["refusal"] >= 150, counts


# --- hand-built records ------------------------------------------------------

DECK = (
    "CM a hand-built record\nCM caf\xe9\nCE\nGW 1,5,0.,0.,-.25,0.,0.,.25,.001\nGE 0\n"
    "GN 2,0,0,0,13.,.005 NOFILE\nEK 0\nFR 0,1,0,0,30.\nEX 0,1,3,0,1.\nRP 0,1,1,1000,90.,0.,0.,0.\n"
    "XQ\nNE 0,1,1,1,1.,0.,0.\nEN\n"
)


def _port(tag=1, seg=3, v=1 + 0j, i=0.01 - 0.002j):
    z = v / i
    return p.PortRow(tag, seg, 1, v, i, z, 1 / z, 0.5 * (v * i.conjugate()).real)


def _value(n, end=False, value=complex(1e-3, -2e-3)):
    return n4p.value_row(n, 1, (0.0, -0.1 * n, 0.25), 0.02, value, end=end)


def _pattern(**kw):
    rows = (
        p.PatternRow(
            90.0, 0.0, 2.15, -999.99, 2.15, 0.0, 90.0, "LINEAR", 1.5, -12.3, 0.0, 0.0
        ),
        p.PatternRow(
            -0.0,
            360.0,
            float("nan"),
            -999.99,
            -999.99,
            1e-300,
            -45.0,
            "",
            1e-310,
            180.0,
            2.0,
            -180.0,
        ),
    )
    return p.PatternBlock(rows=rows, **kw)


def _preamble(**kw):
    base = dict(
        frequency_mhz=30.0,
        wavelength_m=9.993,
        environment=p.ENVIRONMENT_FREE_SPACE,
        ground=None,
        loads=(),
        fill_seconds=0.0,
        factor_seconds=0.0,
    )
    base.update(kw)
    return n4p.Preamble(**base)


def _run(**kw):
    base = dict(
        echo_through=None,
        preamble=_preamble(),
        networks=(),
        network_excitation=(),
        sources=(_port(),),
        currents=(_value(1), _value(2)),
        charges=(_value(1, end=True), _value(2), _value(2, end=True)),
        power=p.PowerBudget(1.0, 0.98, 0.02, 98.0),
        near_fields=(),
        far_field_ground=None,
        patterns=(),
    )
    base.update(kw)
    return n4p.Nec4Run(**base)


def _printout(runs, **kw):
    base = dict(
        wires=(n4p.WireRow(1, (0.0, 0.0, -0.25), (0.0, 0.0, 0.25), 0.001, 5, 1, 5, 1),),
        ge_flag=0,
        total_segments=5,
        junctions=(),
        segments=tuple(
            n4p.SegmentRow(
                k,
                (0.0, 0.0, -0.2 + 0.1 * (k - 1)),
                0.1,
                90.0,
                0.0,
                0.001,
                k - 1,
                k + 1,
                1,
            )
            for k in range(1, 6)
        ),
        runs=tuple(runs),
        run_seconds=0.123,
    )
    base.update(kw)
    return n4p.Nec4Printout(**base)


GM = n4p.MoveRow(1, 3, (0.0, 0.0, 90.0), (0.1, -0.2, 1e5), 2, 10)
GS = n4p.ScaleRow(0.0254)
LOADS = (
    p.LoadRow(1, 3, 3, resistance=50.0, reactance=-12.5),
    p.LoadRow(3, 0, 0, resistance=50.0),
    p.LoadRow(
        1, 2, 2, resistance=10.0, inductance=1e-6, capacitance=0.0, kind="SERIES"
    ),
    p.LoadRow(
        1, 2, 2, resistance=0.0, inductance=1e-6, capacitance=1e-12, kind="PARALLEL"
    ),
    p.LoadRow(0, 1, 5, conductivity=5.8e7, kind="WIRE"),
    p.LoadRow(1, 1, 5, inductance=1e-7, kind="SERIES (PER METER)"),
    p.LoadRow(
        0, 1, 5, resistance=2.0, capacitance=float("nan"), kind="PARALLEL (PER METER)"
    ),
)
NETWORKS = (
    p.LineRow(1, 3, 2, 8, 50.0, 0.25, 0j, complex(0, 1e-3), crossed=True),
    p.LineRow(1, 3, 2, 8, 75.0, 0.0, 0j, 0j),
    p.NetworkRow(1, 2, 2, 9, complex(1e-3, -2e-3), complex(0, 5e-4), complex(-1e-3, 0)),
)
NEAR = p.NearFieldBlock(
    rows=(
        p.NearFieldRow(
            (1.0, 0.0, 2.0), (6.6673e2, float("inf"), 0.0), (12.5, -0.0, 180.0)
        ),
    ),
    magnetic=False,
)
NEAR_H = p.NearFieldBlock(rows=NEAR.rows, magnetic=True)
SOMMERFELD = p.GroundMedium(13.0, 0.005, complex(13.0, -2.997))

RECORDS = {
    "plain": _printout([_run(echo_through=4)]),
    "gm_gs_rows": _printout(
        [_run(echo_through=4)],
        wires=(
            n4p.WireRow(
                1, (1e6, -2e-9, 0.0), (-1234567.0, 0.5, 0.25), 0.001, 5, 1, 5, 1
            ),
            GM,
            GS,
        ),
    ),
    "ground_plane_junctions": _printout(
        [_run(echo_through=4)], ge_flag=1, junctions=((-1, 6, -11), (5, 10, 15, -20))
    ),
    "ge_minus_one": _printout([_run(echo_through=4)], ge_flag=-1),
    "loads_networks": _printout(
        [
            _run(
                echo_through=4,
                preamble=_preamble(loads=LOADS),
                networks=NETWORKS,
                network_excitation=(_port(2, 8), _port(2, 9, v=0j + 1e-300)),
                power=p.PowerBudget(1.0, 0.9, 0.0, 90.0, network_loss=0.1),
            )
        ]
    ),
    "sommerfeld_named_file": _printout(
        [
            _run(
                echo_through=4,
                preamble=_preamble(
                    environment=p.ENVIRONMENT_FINITE_GROUND,
                    ground=SOMMERFELD,
                    ground_file="NOFILE",
                ),
            )
        ]
    ),
    "gn3_named_file": _printout(
        [
            _run(
                echo_through=4,
                preamble=_preamble(
                    environment="FINITE GROUND.  SOMMERFELD SOLUTION GN3",
                    ground=SOMMERFELD,
                    ground_file="caf\xe9.NEC",
                ),
            )
        ]
    ),
    "perfect_ground_far_ground_patterns": _printout(
        [
            _run(
                echo_through=5,
                preamble=_preamble(environment=p.ENVIRONMENT_PERFECT_GROUND),
                near_fields=(NEAR, NEAR_H),
                far_field_ground=n4p.FarFieldGround(
                    "CIRCLE CLIFF", 10.0, -1.0, 13.0, 0.005
                ),
                patterns=(
                    _pattern(
                        average_power_gain=1.234,
                        solid_angle_pi=2.0,
                        power_radiated_4pi=0.98,
                    ),
                    _pattern(
                        range_m=1000.0,
                        range_phase_deg=-12.5,
                        lower_range_mag=1.75281e-107,
                        lower_range_phase_deg=33.0,
                    ),
                    _pattern(directive=True),
                ),
            )
        ]
    ),
    "sweep_and_reports": _printout(
        [
            _run(echo_through=4),
            _run(echo_through=None),
            _run(echo_through=5, preamble=None, currents=None, charges=None),
            n4p.Nec4Report(echo_through=7, near_fields=(NEAR,)),
            n4p.Nec4Report(
                echo_through=2,
                patterns=(_pattern(),),
                far_field_ground=n4p.FarFieldGround("CIRCLE CLIFF", 1.0, 2.0, 3.0, 4.0),
            ),
            n4p.Nec4Report(echo_through=99),
        ]
    ),
    "odd_cells": _printout(
        [
            _run(
                echo_through=-3,
                sources=(
                    _port(
                        v=complex(float("nan"), -0.0), i=complex(float("inf"), 1e-320)
                    ),
                ),
                power=p.PowerBudget(float("inf"), float("-inf"), -0.0, float("nan")),
            )
        ],
        run_seconds=1e9,
    ),
}


@pytest.mark.parametrize("closing", [True, False], ids=["closing", "chained"])
@pytest.mark.parametrize("name", list(RECORDS))
def test_a_hand_built_record_prints_the_same(name, closing, acc):
    data = RECORDS[name]
    version, variant, marker = nc.stamp()
    python = n4p.render_nec4_printout(DECK, data, basis="sinusoidal", closing=closing)
    cxx = acc.nec4_render_printout(
        DECK.encode("latin-1"), data, "sinusoidal", version, variant, marker, closing
    )
    diff = nc.first_difference(python, cxx)
    assert diff is None, diff


def test_the_odd_cells_reach_the_odd_spellings(acc):
    """Not vacuous: NaN, infinity and a three-digit exponent do print."""
    version, variant, marker = nc.stamp()
    data = RECORDS["odd_cells"]
    cxx = acc.nec4_render_printout(
        DECK.encode("latin-1"), data, "sinusoidal", version, variant, marker, True
    )
    assert "NAN" in cxx and "INF" in cxx
    rng = RECORDS["perfect_ground_far_ground_patterns"]
    cxx = acc.nec4_render_printout(
        DECK.encode("latin-1"), rng, "sinusoidal", version, variant, marker, True
    )
    assert "1.75281-107" in cxx
