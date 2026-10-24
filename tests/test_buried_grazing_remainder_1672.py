"""momwire#1672: the buried remainder near the surface.

A buried wire within a segment of the surface sees its own mirror there at
d + d', and the below/below Sommerfeld remainder carries a spike of that
width. Both bases integrated it at one flat order, so radials leaving a base
at z = 0 and sloping a few centimetres into the soil converged tens of ohms
slowly in X. razor-2p now keys the order to grazing height
(`_remainder_qp`, as above ground), and bs2 raises the listed pairs to their
own order (`_below_pair_correction`), both only as far as the grid's
grazing floor serves.

The decks under ``tests/fixtures/buried_grazing_1672`` are a 5 m vertical
(a = 30 mm) fed at its base over four 5 m radials (a = 2 mm) leaving the
base node, at 14 MHz over GN 2 (eps 13, sigma 0.005): ``shallow`` slopes to
5 cm, ``deep`` to 0.5 m; ``_s40`` / ``_s200`` are 40 and 200 segments per
wavelength. The licensed NEC-5 (nec5cl-simd-d0c1383, a black box,
2026-10-24) reads them as below; momwire main (26785ef6) read razor-2p
60.54 - j23.83 and bs2 65.96 - j17.39 on ``shallow_s40``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from momwire import _below_interface
from momwire.bspline import BSplineSolver
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "buried_grazing_1672"

# licensed NEC-5 d0c1383, the ANTENNA INPUT PARAMETERS row
NEC5 = {
    "shallow_s40": complex(67.534, -7.1501),
    "shallow_s200": complex(68.271, -4.5686),
    "deep_s40": complex(66.227, -6.4838),
}


def _z(text: str) -> complex:
    lines = text.splitlines()
    i = next(k for k, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line)
    for line in lines[i + 1 :]:
        v = re.findall(r"-?\d\.\d+E[+-]\d+", line)
        if len(v) >= 6:
            return complex(float(v[4]), float(v[5]))
    raise AssertionError("no impedance row")


def _deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


@pytest.mark.parametrize("name", sorted(NEC5))
def test_razor_reads_the_shallow_radials_as_nec5_does(name):
    """razor-2p is NEC-5's own basis: with the remainder resolved it lands
    on the licensed engine at a coarse mesh too (main was 7 + j17 ohm off on
    shallow_s40)."""
    z = _z(render(_deck(name), basis="razor-2p"))
    assert abs(z - NEC5[name]) < 0.5, (name, z)


def test_bs2_converges_on_the_shallow_radials():
    """bs2 converges to its own value; what #1672 fixes is how fast. Main
    moved 9.4 ohm in X between 40 and 200 segments per wavelength."""
    z40 = _z(render(_deck("shallow_s40"), basis="bspline"))
    z200 = _z(render(_deck("shallow_s200"), basis="bspline"))
    assert abs(z40 - z200) < 2.5, (z40, z200)


def _radials(depth: float, n: int):
    seg_l, seg_r = [], []
    for x, y in [(5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)]:
        for k in range(n):
            a, b = k / n, (k + 1) / n
            seg_l.append((a * x, a * y, -a * depth))
            seg_r.append((b * x, b * y, -b * depth))
    return np.array(seg_l), np.array(seg_r)


def test_bs2_lists_the_pairs_near_the_surface_only():
    """The listing rule is the above-ground one (#631/#1189) below the
    plane: radials sloping to 5 cm list pairs, and level radials a metre
    down list none, so their fill is the base order's, bit for bit."""
    solver = BSplineSolver.__new__(BSplineSolver)
    I, _J, Q = solver._below_remainder_pairs(*_radials(0.05, 9), 0.0, 6)
    assert I.size and Q.max() > 6
    sl, sr = _radials(0.0, 9)
    sl[:, 2] -= 1.0
    sr[:, 2] -= 1.0
    I, J, Q = solver._below_remainder_pairs(sl, sr, 0.0, 6)
    assert I.size == J.size == Q.size == 0


def test_a_halved_list_reaches_the_base_fill():
    pairs = (np.array([0, 1]), np.array([1, 0]), np.array([192, 13]))
    I, J, Q = _below_interface._halved(pairs, 6)
    assert I.tolist() == [0] and Q.tolist() == [96]
    assert _below_interface._halved((I, J, np.array([12])), 6) is None
