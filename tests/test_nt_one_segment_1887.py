"""An NT card with both ends on one segment sums its four Y entries (momwire#1887).

A two-port whose two terminals are one node injects ``I1 + I2 =
(y11 + y12 + y21 + y22) V`` into that node. Licensed NEC-5 and NEC-4.2, run
as black boxes (Haswell ``exe/nec5-ripplefix`` 4f27b568 and
``exe/nec42-release`` 5fc0d508) on a 10 m dipole, ``GW 1 21``, 14 MHz, fed
on segment 11, with the card on segment 6, both print exactly that:

=======================================  =================  ==================
deck                                     NEC-5 Z            NEC-4.2 Z
=======================================  =================  ==================
no card                                  63.339-j62.266     64.3546-j55.7531
``NT 1 6 1 6`` y11=.01+.005j, y12=.002j,  89.348-j86.459     88.2427-j77.2813
y22=.003
``LD 4 1 6 6`` Z = 1/(y11+2 y12+y22)     89.348-j86.459     88.2427-j77.2813
``LD 4`` Z = 1/(y11+y22) (no y12)        98.511-j82.809     96.4912-j73.7628
``NT 1 6 1 6`` y11=y22=-y12=.01+.02j     28.059-j563.12     31.7281-j566.125
``LD 4 1 6 6 1E9 0`` (the gap open)      28.059-j563.12     31.7285-j566.125
=======================================  =================  ==================

The summed card equals the series ``LD 4`` of the summed admittance to every
printed digit on both, with rc=0 and no warning; a card whose entries sum to
zero leaves the gap open. Before the fix the C++ run half declined the card
and the Python reducer's buffered fancy-index ``+=`` kept one of the four
entries and dropped the rest, silently.

Here: the stamp itself (sums on a repeated index, the old bits on distinct
ones), the reducer's ``Admittance`` and ``TouchstoneTwoPort`` with a repeated
port, and the two decks through the Python route of both dialects, each
against its ``LD 4`` equivalent. The C++ route of the same decks is held to
Python's bytes by ``test_razor_core_standalone`` (``razor_printout_net``,
both dialects' no-Python drivers) and ``test_nec4_run_core``'s ``CASES``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire.networks import (
    C_LIGHT,
    Admittance,
    Driven,
    Network,
    NetworkReducer,
    PortOnWire,
    TouchstoneTwoPort,
)
from momwire.networks._reducer import _stamp_block

NET_DECKS = Path(__file__).resolve().parent / "fixtures" / "razor_printout_net"
SUMMED = NET_DECKS / "net_nt_one_segment.nec"
ZERO_SUM = NET_DECKS / "net_nt_one_segment_zero_sum.nec"

Y11, Y12, Y22 = 0.01 + 0.005j, 0.002j, 0.003 + 0j
YB = np.array([[Y11, Y12], [Y12, Y22]])
Y0 = 0.01 + 0.02j
YZ = np.array([[Y0, -Y0], [-Y0, Y0]])

WL = C_LIGHT / 14e6


# --- the stamp --------------------------------------------------------------


def _g(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))


def test_a_repeated_index_takes_all_four_entries_in_c_order():
    g = _g(4, 1)
    want = g.copy()
    for v in (Y11, Y12, Y12, Y22):  # np.add.at's order, which the C++ copies
        want[2, 2] = want[2, 2] + v
    _stamp_block(g, [2, 2], YB)
    assert g.tobytes() == want.tobytes()


def test_distinct_indices_keep_the_buffered_stamps_bits():
    rng = np.random.default_rng(2)
    for idxs in ([0, 3], [3, 0], [1, 2, 4], [4]):
        n = len(idxs)
        yb = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        g = _g(5, 3)
        old = g.copy()
        old[np.ix_(idxs, idxs)] += yb
        _stamp_block(g, idxs, yb)
        assert g.tobytes() == old.tobytes(), idxs


def test_a_port_named_twice_in_a_three_port_lands_every_entry():
    yb = np.arange(1, 10).reshape(3, 3) * (1 + 1j)
    g = np.zeros((3, 3), complex)
    _stamp_block(g, [0, 2, 0], yb)
    assert g[0, 0] == yb[0, 0] + yb[0, 2] + yb[2, 0] + yb[2, 2]
    assert g[0, 2] == yb[0, 1] + yb[2, 1]
    assert g[2, 0] == yb[1, 0] + yb[1, 2]
    assert g[2, 2] == yb[1, 1]


# --- the reducer ------------------------------------------------------------


class FixedY:
    """A frequency-flat `AdmittanceData` (the protocol's whole contract)."""

    def __init__(self, y):
        self._y = np.asarray(y, dtype=np.complex128)

    @property
    def nports(self):
        return self._y.shape[0]

    def y_at(self, f_hz):
        return self._y


def _antenna_y():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    return 0.004 * (a + a.T) / 2.0 + np.eye(2) * (0.02 + 0.008j)


def _feed_z(branch):
    net = Network(
        ports={"feed": PortOnWire("feed"), "n": PortOnWire("n")},
        branches=[branch],
        sources=[Driven(port="feed")],
    )
    reducer = NetworkReducer(net, {"feed": 0, "n": 1}, 2)
    return reducer.driven_impedance(_antenna_y(), WL)[0]


@pytest.mark.parametrize(
    "two_port",
    [
        lambda y: Admittance(ports=("n", "n"), y=tuple(map(tuple, y))),
        lambda y: TouchstoneTwoPort("n", "n", FixedY(y)),
    ],
    ids=["admittance", "touchstone"],
)
@pytest.mark.parametrize("y", [YB, YZ], ids=["asymmetric", "zero_sum"])
def test_a_two_port_on_one_port_is_its_summed_one_port(two_port, y):
    one_port = Admittance(ports=("n",), y=((complex(y.sum()),),))
    assert _feed_z(two_port(y)) == pytest.approx(_feed_z(one_port), rel=1e-13)


def test_the_dropped_reading_is_not_what_the_reducer_prints():
    # not vacuous: one entry alone, the old answer, is a different antenna
    one_entry = Admittance(ports=("n",), y=((Y22,),))
    summed = Admittance(ports=("n", "n"), y=tuple(map(tuple, YB)))
    assert _feed_z(summed) != pytest.approx(_feed_z(one_entry), rel=1e-3)


# --- the decks, Python route --------------------------------------------------

_E = r"-?\d\.\d+E[+-]\d\d"


def _zin(printout: str) -> tuple[str, str]:
    """The printed (R, X) of the one ANTENNA INPUT PARAMETERS row."""
    lines = printout.splitlines()
    (head,) = [i for i, ln in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in ln]
    for ln in lines[head + 1 :]:
        nums = re.findall(_E, ln)
        if len(nums) >= 6:
            return nums[4], nums[5]
    raise AssertionError("no input-parameters row")


def _render(dialect: str, basis: str, text: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dialect == "nec5":
            from momwire.dropin.nec5 import _shell

            return _shell.render(text, basis=basis)
        from momwire.dropin.nec5 import _nec4

        return _nec4.render(text, basis=basis)


def _as_ld4(dialect: str, deck: Path, z: complex) -> str:
    """The deck with its NT replaced by a series LD 4 of `z` on segment 6, in
    the dialect's spelling (the NEC-5 card's fourth field is LDTAGT, an end
    code, where NEC-4.2's is the last segment of a range)."""
    text = deck.read_text()
    (nt,) = [ln for ln in text.splitlines() if ln.startswith("NT ")]
    last = "0" if dialect == "nec5" else "6"
    return text.replace(nt, f"LD 4 1 6 {last} {z.real:.10E} {z.imag:.10E}")


ROUTES = [
    ("nec5", "bspline"),
    ("nec5", "razor-2p"),
    ("nec42", "sinusoidal"),
    ("nec42", "bspline"),
]


@pytest.mark.parametrize("dialect, basis", ROUTES, ids=[f"{d}-{b}" for d, b in ROUTES])
def test_the_summed_card_prints_as_the_load_of_its_sum(dialect, basis):
    nt = _render(dialect, basis, SUMMED.read_text())
    ld = _render(dialect, basis, _as_ld4(dialect, SUMMED, 1 / (Y11 + 2 * Y12 + Y22)))
    assert "NEC ERROR" not in nt and "NEC ERROR" not in ld
    assert _zin(nt) == _zin(ld)
    # and not the reading that drops y12 (the black box's falsifier)
    assert _zin(nt) != _zin(
        _render(dialect, basis, _as_ld4(dialect, SUMMED, 1 / (Y11 + Y22)))
    )


@pytest.mark.parametrize("dialect, basis", ROUTES, ids=[f"{d}-{b}" for d, b in ROUTES])
def test_a_zero_sum_card_leaves_the_gap_open(dialect, basis):
    nt = _render(dialect, basis, ZERO_SUM.read_text())
    ld = _render(dialect, basis, _as_ld4(dialect, ZERO_SUM, 1e9 + 0j))
    assert "NEC ERROR" not in nt and "NEC ERROR" not in ld
    r, x = map(float, _zin(nt))
    r0, x0 = map(float, _zin(ld))
    # 1e9 ohm is open to the printed digits of X and to 1e-4 of R (NEC-5's
    # own printouts of the pair differ there too)
    assert (r, x) == pytest.approx((r0, x0), rel=1e-4)


# The black boxes' printed Z for the two decks (the module docstring's table).
# nec5-razor-2p prints the no-card dipole within 3e-5 of NEC-5 and
# nec42-sinusoidal within 2e-4 of NEC-4.2, so the tolerances are the basis
# gap, an order below what dropping y12 moves (about 10 %).
REFERENCE = [
    ("nec5", "razor-2p", SUMMED, 89.348 - 86.459j, 1e-4),
    ("nec5", "razor-2p", ZERO_SUM, 28.059 - 563.12j, 1e-4),
    ("nec42", "sinusoidal", SUMMED, 88.2427 - 77.2813j, 2e-3),
    ("nec42", "sinusoidal", ZERO_SUM, 31.7281 - 566.125j, 2e-3),
]


@pytest.mark.parametrize(
    "dialect, basis, deck, z_ref, rel",
    REFERENCE,
    ids=[f"{d}-{b}-{p.stem}" for d, b, p, _z, _r in REFERENCE],
)
def test_the_decks_print_what_the_licensed_engines_print(
    dialect, basis, deck, z_ref, rel
):
    r, x = map(float, _zin(_render(dialect, basis, deck.read_text())))
    assert r == pytest.approx(z_ref.real, rel=rel)
    assert x == pytest.approx(z_ref.imag, rel=rel)
