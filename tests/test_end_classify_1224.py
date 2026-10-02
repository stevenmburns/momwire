"""The crossing end loops' classification, batched (momwire#1224).

`_classify_ends` decides each end's `_fast_end_desc` from 2-D arrays
(`_EndArgs.batch`) instead of one end at a time. It must name the same
product rows for every end -- the descs are what the end loops gather V and
W by -- and count the same on-node ends. These rows run razor's crossing
decks with every classification checked against the one-end reference as
it happens, and pin Z with the batches on and off.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill as cf
from momwire.razor import RazorSolver

from test_crossing_serve_524 import crossing_deck, hub_deck, invl_deck
from test_factorize_1224 import _solve

DECKS = {
    "hub4": lambda: hub_deck(n_radials=4),
    "invl4": lambda: invl_deck(n_radials=4),
    "invl4-lean": lambda: invl_deck(n_radials=4, lean=True),
    "crossing1": lambda: crossing_deck(1),
}


def _same_desc(a, b):
    if a is None or b is None:
        return a is None and b is None
    if a[0] != b[0] or len(a) != len(b):
        return False
    return all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
        for x, y in zip(a[1:], b[1:])
    )


def _make(deck):
    d = DECKS[deck]()
    if deck == "crossing1":
        d = {k: v for k, v in d.items() if k != "junctions"}
    return RazorSolver(**d, nec5_quadrature=True)


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_every_batched_desc_is_the_one_end_desc(deck, monkeypatch):
    real = cf._classify_ends
    seen = {"ends": 0, "fast": 0, "on_node": 0, "batched": 0}

    def checked(fast, a_wire, ends, args):
        descs, on_node = real(fast, a_wire, ends, args)
        seen["batched"] += isinstance(args, cf._EndArgs) and bool(ends)
        with monkeypatch.context() as mp:
            mp.setattr(cf, "_END_CLASSIFY_BATCHED", False)
            ref, ref_on = real(fast, a_wire, ends, args)
        assert len(descs) == len(ref) == len(ends)
        assert all(_same_desc(a, b) for a, b in zip(descs, ref))
        assert on_node == ref_on
        seen["ends"] += len(ends)
        seen["fast"] += sum(d is not None for d in descs)
        seen["on_node"] += sum(on_node)
        return descs, on_node

    monkeypatch.setattr(cf, "_classify_ends", checked)
    _solve(lambda: _make(deck), ("_assemble_Z",))
    assert seen["batched"] > 0 and seen["fast"] > 0, seen


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_razor_z_does_not_move(deck, monkeypatch):
    monkeypatch.setattr(cf, "_END_CLASSIFY_BATCHED", False)
    z_ref, i_ref = _solve(lambda: _make(deck), ("_assemble_Z",))
    monkeypatch.setattr(cf, "_END_CLASSIFY_BATCHED", True)
    z_new, i_new = _solve(lambda: _make(deck), ("_assemble_Z",))
    assert np.array_equal(z_new, z_ref)
    assert np.array_equal(i_new, i_ref)


def test_a_small_batch_budget_moves_nothing(monkeypatch):
    """Several batches per loop: the descs are per end, so the cut is free."""
    z_ref, i_ref = _solve(lambda: _make("invl4"), ("_assemble_Z",))
    monkeypatch.setattr(cf, "_END_CLASSIFY_PAIRS", 64)
    z_new, i_new = _solve(lambda: _make("invl4"), ("_assemble_Z",))
    assert np.array_equal(z_new, z_ref)
    assert np.array_equal(i_new, i_ref)
