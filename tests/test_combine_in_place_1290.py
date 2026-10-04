"""Razor's lossy-medium T1/T2 combine, spelled in place (momwire#1290).

Over a lossy medium `c_Phi` is not purely imaginary, so the fused T2 kernel
hands T2 back and the window's rows are `c_A * t1 - t2 / c_Phi`. That
expression built three (rows, n_basis) temporaries and copied the last into
Z's window. It is now three ufuncs, each writing into an operand it alone
owns: the same multiply, divide and subtract on the same operands, so the
same floats. Gated here as uint64 on whole buried and inverted-L fills
against the expression (`_COMBINE_IN_PLACE = False`), with a count that the
combine ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import razor as _razor
from momwire.razor import RazorSolver

from test_crossing_serve_524 import hub_deck, invl_deck

DECKS = {
    "buried": lambda: hub_deck(n_radials=4),
    "invl": lambda: invl_deck(n_radials=4),
}


def _z(deck):
    s = RazorSolver(**DECKS[deck](), nec5_quadrature=True)
    return s._assemble_Z(s._build_geometry(), s.k)


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_the_in_place_combine_is_the_expression(deck, monkeypatch):
    monkeypatch.setattr(_razor, "_COMBINE_IN_PLACE", False)
    ref = _z(deck)
    monkeypatch.setattr(_razor, "_COMBINE_IN_PLACE", True)
    monkeypatch.setattr(_razor, "_COMBINE_COUNT", [0])
    got = _z(deck)
    if not _razor._COMBINE_COUNT[0]:
        pytest.skip("this build takes another combine (no razor_t2_rows)")
    assert np.array_equal(
        np.asarray(got).view(np.uint64), np.asarray(ref).view(np.uint64)
    )
