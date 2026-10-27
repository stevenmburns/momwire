"""momwire#1314: EZNEC's parallel-connected loads, saved as ``.nec`` unpinned.

EZNEC writes a "parallel connected load" as a ONE-port ``NT``: ``Y11`` only,
port 2 on its virtual wire with ``Y12 = Y22 = 0``. The decks EZNEC hands an
engine also pin each of those virtual ends with ``LD 4 … 1.E+10`` (corpus
0017), but a model saved as ``.nec`` from EZNEC's File menu carries no pins
(Dan AC6LA, antennaknobs#1880). Unpinned, the card stamps nothing at port 2
and a virtual site has no antenna admittance (momwire#1139), so its KCL row
is identically zero and the seam raised ``SingularNetworkError``.

The seam now holds such a site at zero volts. That is exact rather than a
stand-in for the pin: nothing reads the node's voltage, so the unpinned deck
and the pinned one are the same circuit. The gates are 0017 with its two pin
cards deleted, against 0017 as written and against licensed NEC-5's printout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.eznec import serve

DECK = (
    Path(__file__).parent
    / "fixtures"
    / "eznec"
    / "decks"
    / "0017_network-connection-test.nec"
)
PINS = ("LD 4,4,1,0,1.E+10,0.", "LD 4,4,2,0,1.E+10,0.")

# 0017's ANTENNA INPUT PARAMETERS row, licensed NEC-5 (razor-2p's own basis).
NEC5 = 1.9534e02 - 5.7458e01j


def _pinned() -> str:
    return DECK.read_text()


def _unpinned() -> str:
    text = _pinned()
    for pin in PINS:
        assert text.count(pin) == 1, pin
    lines = [line for line in text.splitlines() if line.strip() not in PINS]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
def test_the_unpinned_deck_is_the_pinned_circuit(basis):
    pinned = serve(parse_nec5(_pinned()), basis=basis)
    unpinned = serve(parse_nec5(_unpinned()), basis=basis)
    z_pinned = complex(pinned.sources[0].impedance)
    z_unpinned = complex(unpinned.sources[0].impedance)
    assert z_unpinned == pytest.approx(z_pinned, rel=1e-12, abs=0)


def test_the_unpinned_deck_answers_nec5():
    z = complex(serve(parse_nec5(_unpinned()), basis="razor-2p").sources[0].impedance)
    assert abs(z - NEC5) < 5e-3, z


def test_a_held_site_reads_zero_volts():
    """The virtual ends are held, so the connection-point rows print an exact
    zero there rather than the solve's round-off."""
    run = serve(parse_nec5(_unpinned()), basis="razor-2p")
    virtual = [row for row in run.network_excitation if row.tag == 4]
    assert len(virtual) == 2, run.network_excitation
    for row in virtual:
        assert complex(row.voltage) == 0
        assert np.isfinite(complex(row.current))
