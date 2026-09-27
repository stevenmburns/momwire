"""Multi-run NEC-5 decks — momwire#1237.

SimNEC measures an N-port by driving each port in turn at 1 V, with the
others at 1e-10 V, one ``XQ`` block per port, in ONE deck. The seam serves
each block as its own single-run deck (``split_runs``) and prints them in the
licensed engine's multi-run layout (``render_multi_printout``).

``tests/fixtures/eznec_multi_run_1237/`` (README there) holds the decks and
the licensed printouts. The gates:

  * the LAYOUT is the licensed one line for line, with the numbers masked:
    free space with RP (an RP executes, and the XQ after it echoes with the
    next group), without RP (each XQ executes), and over a Sommerfeld ground
    (the range form's LOWER MEDIUM line);
  * the per-run port currents, i.e. the N-port admittance matrix, are the
    licensed engine's on razor-nec5, its formulation twin;
  * a later block that changes anything but its sources and requests is
    refused by name, and a single-run deck never takes this path.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from momwire.eznec import _serve
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_multi_run_1237"
DECKS = ("two-port", "two-port-xq", "two-port-ground")
TWIN = "razor-nec5"
_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?(E?[-+]\d+)?")


def _deck(name):
    return (FIXTURES / f"{name}.nec").read_text()


def _oracle(name):
    lines = (FIXTURES / f"{name}.out").read_text(encoding="latin-1").splitlines()
    # The licensed engine's own ground-table cache chatter (README): a blank,
    # then two lines, which momwire does not print.
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip() and i + 1 < len(lines) and "GMPINO" in lines[i + 1]:
            i += 3
            continue
        out.append(lines[i])
        i += 1
    return out


def _mask(line):
    return " ".join("#" if _NUMBER.fullmatch(t) else t for t in line.split())


def _from_structure(lines):
    i = next(i for i, ln in enumerate(lines) if "STRUCTURE SPECIFICATION" in ln)
    return [_mask(ln) for ln in lines[i:]]


def _port_currents(lines):
    runs = []
    for block in "\n".join(lines).split("ANTENNA INPUT PARAMETERS")[1:]:
        rows = [
            ln.split()
            for ln in block.splitlines()
            if re.match(r"^\s+\d+\s+\d+\s+1\s", ln)
        ]
        runs.append([complex(float(r[5]), float(r[6])) for r in rows])
    return np.array(runs)


@pytest.mark.parametrize("name", DECKS)
def test_the_layout_is_the_licensed_one(name):
    ours = _from_structure(render(_deck(name), basis=TWIN).splitlines())
    theirs = _from_structure(_oracle(name))
    assert ours == theirs


@pytest.mark.parametrize("name", DECKS)
def test_the_n_port_is_the_licensed_one(name):
    ours = _port_currents(render(_deck(name), basis=TWIN).splitlines())
    theirs = _port_currents(_oracle(name))
    assert ours.shape == theirs.shape == (2, 2)
    assert np.max(np.abs(ours - theirs)) <= 2e-3 * np.max(np.abs(theirs))


def test_each_block_replaces_the_sources():
    """Two runs of two sources each; an accumulating reader would print four
    rows in the second run."""
    ours = _port_currents(render(_deck("two-port"), basis=TWIN).splitlines())
    assert ours.shape[1] == 2


def test_the_split_is_one_single_run_deck_per_block():
    runs = _serve.split_runs(_deck("two-port"))
    assert len(runs) == 2
    for i, text in enumerate(runs):
        cards = [ln[:2] for ln in text.splitlines()]
        assert cards.count("XQ") == 1 and cards.count("EX") == 2
        assert cards[-1] == "EN"
        drive = [ln for ln in text.splitlines() if ln.startswith("EX")][i]
        assert drive.split()[5] == "1.000000e+00"


def test_a_single_run_deck_takes_the_old_path():
    """At most one XQ, as in every EZNEC deck: `split_runs` answers None and
    `render` is exactly what it was."""
    assert _serve.split_runs(_deck("two-port").split("XQ\n", 1)[0] + "XQ\nEN\n") is None
    one_field_ge = Path(__file__).parent / "fixtures" / "eznec_simnec_1230"
    assert _serve.split_runs((one_field_ge / "dipole-ex5.nec").read_text()) is None


@pytest.mark.parametrize("card", ["FR 0 1 0 0 15.0 1", "LD 5 1 1 21 5.8e7"])
def test_a_later_block_that_changes_more_than_its_sources_is_refused(card):
    text = _deck("two-port").replace(
        "EX 0 1 11 0 1.000000e-10\n", card + "\nEX 0 1 11 0 1.000000e-10\n", 1
    )
    out = render(text, basis=TWIN)
    assert "NEC ERROR" in out
    assert "XQ block 2 carries " + card[:2] in out
