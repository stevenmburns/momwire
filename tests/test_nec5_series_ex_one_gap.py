"""Several voltage EX cards on one gap: generators in series, as NEC-5 serves them.

SimNEC extracts an N-port Y matrix by running one XQ per port, one EX card at
1 V and the rest at 1e-10 V probes, so two ports that sit at one point (two
series loads at one node) write two EX cards for one segment. The NEC-5 seams
refused every such deck except EZNEC's repeated load probe beside an EX 4
(momwire#1787). They now serve it the way the licensed NEC-5 does, measured
black box on the invented decks in ``tests/fixtures/nec5_series_ex/`` (README
there):

* the gap is driven by the SUM of the cards' volts and every card carries the
  one gap current;
* ANTENNA INPUT PARAMETERS prints one row per card, in card order, each with
  its own voltage, that shared current, and its own V/I, I/V and power;
* a voltage and a current source on one segment is an error in NEC-5, and two
  set currents stay refused here (the sentence names the EX 4);
* at a TL/NT connection point NEC-5 does NOT answer the repeat as a series
  sum (two 0.5 V cards print a different current from one 1 V card), so that
  shape is refused by name.

The tolerance against the oracle is each basis's own: razor2p is NEC-5's
formulation twin (0.1 %, as tests/test_eznec_simnec_nec5_1230.py holds it);
bspline differs by basis, so each repeated deck may disagree with NEC-5 no
more than the same deck written with one summed card does, which is what
says the repeat itself adds nothing. The byte gate between the Python and C++
seams is the slow lane's corpus sweep in tests/test_razor_core_standalone.py,
which reads these fixtures like every other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.dropin.nec5._serve import serve
from momwire.dropin.nec5._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "nec5_series_ex"
TWIN = "razor2p"
BASES = (TWIN, "bs2")
TWIN_TOLERANCE = 1e-3

# deck -> the one-summed-card deck it must answer as
SERVED = {
    "two_halves": "one_card",
    "three_cards": "one_card",
    "loaded_two": "loaded_one",
    "second_wire_pair": None,
    "probe_pair": None,
    "four_port": None,
}


def deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


def oracle(name: str) -> list[str]:
    return (FIXTURES / f"{name}.out").read_text(encoding="latin-1").splitlines()


def _blocks(lines: list[str]) -> list[list[str]]:
    """Every ANTENNA INPUT PARAMETERS block's rows, one list per XQ."""
    out = []
    for i, line in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" not in line:
            continue
        rows = []
        for row in lines[i + 4 :]:
            if not row.strip():
                break
            rows.append(row)
        out.append(rows)
    return out


def _cells(row: str) -> list[float]:
    return [float(x) for x in row[12:].split()]


def _complex(row: str, k: int) -> complex:
    c = _cells(row)
    return complex(c[2 * k], c[2 * k + 1])


def _rel(a: complex, b: complex) -> float:
    return abs(a - b) / abs(b)


def _ex_cards_per_run(text: str) -> list[int]:
    counts, n = [], 0
    for line in text.splitlines():
        if line.startswith("EX "):
            n += 1
        elif line.startswith("XQ"):
            counts.append(n)
            n = 0
    return counts


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("name", sorted(SERVED))
def test_a_repeat_prints_one_row_per_card_in_card_order(name, basis):
    rendered = render(deck(name), basis=basis)
    assert "NEC ERROR" not in rendered
    ours, theirs = _blocks(rendered.splitlines()), _blocks(oracle(name))
    assert [len(b) for b in ours] == [len(b) for b in theirs]
    assert [len(b) for b in ours] == _ex_cards_per_run(deck(name))
    for mine, licensed in zip(ours, theirs, strict=True):
        for a, b in zip(mine, licensed, strict=True):
            # TAG, SEG., the end digit and each card's OWN set voltage
            assert a[:36] == b[:36]


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("name", sorted(SERVED))
def test_every_card_on_a_gap_carries_the_one_gap_current(name, basis):
    """Rows that share an address share the current to the last byte, and
    each row's impedance and power are its own volts over that current."""
    for block in _blocks(render(deck(name), basis=basis).splitlines()):
        by_address: dict[str, list[str]] = {}
        for row in block:
            by_address.setdefault(row[:12], []).append(row)
        for rows in by_address.values():
            assert len({row[36:60] for row in rows}) == 1
            for row in rows:
                v, i = _complex(row, 0), _complex(row, 1)
                assert _rel(_complex(row, 2), v / i) < 1e-4
                power = _cells(row)[8]
                assert power == pytest.approx(0.5 * (v * i.conjugate()).real, rel=1e-4)


@pytest.mark.parametrize("name", sorted(SERVED))
def test_razor_2p_lands_on_the_licensed_rows(name):
    """razor2p is NEC-5's twin: every row's current and impedance to 0.1 %."""
    ours = _blocks(render(deck(name), basis=TWIN).splitlines())
    for mine, licensed in zip(ours, _blocks(oracle(name)), strict=True):
        for a, b in zip(mine, licensed, strict=True):
            assert _rel(_complex(a, 1), _complex(b, 1)) < TWIN_TOLERANCE, (a, b)
            assert _rel(_complex(a, 2), _complex(b, 2)) < TWIN_TOLERANCE, (a, b)


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("name", [n for n, one in SERVED.items() if one])
def test_cards_summing_to_one_volt_answer_as_the_one_volt_card(name, basis):
    """The physics: generators in series drive the gap with their sum.

    The gap current is the one-card deck's to every printed digit, and below
    the print to 1e-12 relative. Measured, it is the same BITS on every deck
    here (relative difference 0.0, both bases): the volts are chosen to sum
    exactly in binary, and the single-source path (one solve, scaled) and the
    multi-drive path (the loaded admittance applied to the summed volts) land
    on identical bits on these meshes. That second half is an observation,
    not algebra, so the gate is round-off rather than bytes."""
    one = SERVED[name]
    (rep,) = _blocks(render(deck(name), basis=basis).splitlines())
    (ref,) = _blocks(render(deck(one), basis=basis).splitlines())
    for row in rep:
        assert row[36:60] == ref[0][36:60]
    (single,) = serve(parse_nec5(deck(one)), basis=basis).sources
    for row in serve(parse_nec5(deck(name)), basis=basis).sources:
        assert abs(row.current - single.current) <= 1e-12 * abs(single.current)
    # the oracle holds the same identity on its own print
    (lic,) = _blocks(oracle(name))
    (lic_one,) = _blocks(oracle(one))
    assert {row[36:60] for row in lic} == {lic_one[0][36:60]}


@pytest.mark.parametrize("name", [n for n, one in SERVED.items() if one])
def test_bspline_disagrees_with_nec5_no_more_than_the_one_card_deck(name):
    """bspline is not NEC-5's basis; what the repeat must not do is ADD to the
    disagreement the same drive written as one card already has."""
    one = SERVED[name]
    (rep,) = _blocks(render(deck(name), basis="bs2").splitlines())
    (ref,) = _blocks(render(deck(one), basis="bs2").splitlines())
    (lic_one,) = _blocks(oracle(one))
    control = _rel(_complex(ref[0], 1), _complex(lic_one[0], 1))
    for a, b in zip(rep, _blocks(oracle(name))[0], strict=True):
        assert _rel(_complex(a, 1), _complex(b, 1)) <= control * (1 + 1e-9) + 1e-12


def _refusal(text: str, basis: str) -> str:
    printout = render(text, basis=basis)
    assert " ***** NEC ERROR - " in printout, "this deck was SERVED"
    return printout.rsplit(" ***** NEC ERROR - ", 1)[1].strip()


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize(
    ("name", "words"),
    [
        (
            "network_two_halves",
            "two EX 0 cards address 1,11, which is also an end of the TL card "
            "between 1,11 and 2,11, so it is a network CONNECTION POINT",
        ),
        (
            "nt_two_halves",
            "two EX 0 cards address 1,11, which is also an end of the NT card "
            "between 1,11 and 2,11, so it is a network CONNECTION POINT",
        ),
        # NEC-5 errors on this one; here the mixed-drive sentence comes first,
        # because a 1 V card beside an EX 4 is a generator, not a probe
        (
            "voltage_and_current",
            "this deck carries 2 EX cards writing BOTH kinds",
        ),
        (
            "two_currents",
            "two EX cards address 1,11 and at least one of them is an EX 4",
        ),
    ],
)
def test_the_repeats_that_stay_refused_say_why(name, words, basis):
    assert _refusal(deck(name), basis).startswith(words)


def test_the_network_refusal_rests_on_the_oracle():
    """What the network sentence quotes: at a TL end NEC-5's two 0.5 V cards
    print neither the one 1 V card's current nor half of it."""
    (two,) = _blocks(oracle("network_two_halves"))
    (one,) = _blocks(oracle("network_one_card"))
    i_two, i_one = _complex(two[0], 1), _complex(one[0], 1)
    assert _rel(i_two, i_one) > 0.1
    assert _rel(i_two, 0.5 * i_one) > 0.1
    assert "both a voltage source and a current source" in "\n".join(
        oracle("voltage_and_current")
    )
