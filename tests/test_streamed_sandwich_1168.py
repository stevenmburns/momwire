"""The main sandwich contracted chunk by chunk — momwire#1168.

Once #1169 chunked the whole-axis main sandwich's kernel tables, the fill's
process peak was the six whole (|rA|, |iB|) left products `_sandwich_dense`
assembled from those chunks before contracting them: 169 MB at razor
hub_deck(16) x8, alive beside everything else. `_streamed_sandwich` contracts
each basis row of the below axis at the first chunk that completes its
pattern, and keeps only the left-product columns a pending row still reads.

The reference is in-process: `_MAIN_STREAMED = False` is the assembled
products and one contraction, the pre-#1168 code verbatim. The gate is Z at
`np.array_equal`, at a budget small enough to cut every block into dozens of
chunks. The negative control, `_STREAMED_WHOLE_ROWS = False`, contracts each
chunk's columns for every row and adds the partial sums — the naive chunked
contraction — and the same comparison must FAIL, which is what says the gate
can see a reassociated running sum.

Measured 2026-09-23 against origin/main b9630a4 by script, Z to the bit: razor
hub_deck(16) x2 and x4 and crossing_deck(1), at the shipped budget and at
20,000 bytes (2,597 / 9,031 / 55 chunks). The negative control at 20,000
bytes moves 9,519 / 186 / 178 entries (at most 2.8e-19 of max|Z|).
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill as cf

from test_crossing_serve_524 import crossing_deck, hub_deck
from test_razor_detached_1149 import detached
from test_triple_memo_1168 import _razor

_TINY = 20_000


class _Counts:
    """Chunks served, streamed calls and contractions made during a solve."""

    def __init__(self, mp):
        self.chunks = self.streamed = self.contractions = 0
        chunked, streamed, combine = (
            cf._chunked_tables,
            cf._streamed_sandwich,
            cf._combine,
        )

        def spy_chunked(*a):
            for item in chunked(*a):
                self.chunks += 1
                yield item

        def spy_streamed(*a, **k):
            self.streamed += 1
            return streamed(*a, **k)

        def spy_combine(*a):
            self.contractions += 1
            return combine(*a)

        mp.setattr(cf, "_chunked_tables", spy_chunked)
        mp.setattr(cf, "_streamed_sandwich", spy_streamed)
        mp.setattr(cf, "_combine", spy_combine)


def _z(make, monkeypatch, *, budget=_TINY, streamed=True, whole_rows=True):
    with monkeypatch.context() as mp:
        if budget is not None:
            mp.setattr(cf, "_MAIN_CHUNK_BYTES", budget)
        mp.setattr(cf, "_MAIN_STREAMED", streamed)
        mp.setattr(cf, "_STREAMED_WHOLE_ROWS", whole_rows)
        counts = _Counts(mp)
        s = make()
        # The matrix `compute_impedance` would factor, read where it is
        # filled: the solve now factors it in place (momwire#1173), so there
        # is no stashed Z to read afterwards.
        return s._assemble_Z(s._build_geometry(), s.k), counts


def _hub(x):
    d = hub_deck(n_radials=16)
    npe = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [[m * x for m in e] for e in npe[:16]] + [
        npe[16],
        [npe[17][0] * x],
    ]
    return lambda: _razor(d)


FAST = {
    "crossing1": lambda: _razor(crossing_deck(1)),
    "detached": lambda: _razor(detached()),
}


@pytest.mark.parametrize("case", sorted(FAST))
def test_streamed_is_the_assembled_contraction_bit_for_bit(case, monkeypatch):
    """Z to the bit against the assembled products, with the chunks proved
    to have formed and more than one contraction per streamed block (the
    reference makes exactly one)."""
    ref, rc = _z(FAST[case], monkeypatch, streamed=False)
    got, gc = _z(FAST[case], monkeypatch)
    assert rc.streamed == 0 and gc.streamed == 2  # forward + reversed
    assert gc.chunks == rc.chunks > 20, (gc.chunks, rc.chunks)
    assert gc.contractions > rc.contractions, "every row waited for the end"
    assert np.array_equal(got, ref)


def test_negative_control_per_chunk_partial_sums_move_z(monkeypatch):
    """The gate above can fail: summing each chunk's partial contraction
    reassociates the rows that straddle a chunk boundary, and Z moves."""
    make = FAST["crossing1"]
    ref, _ = _z(make, monkeypatch, streamed=False)
    bad, counts = _z(make, monkeypatch, whole_rows=False)
    assert counts.streamed == 2 and counts.contractions == counts.chunks
    assert not np.array_equal(bad, ref)
    # ...and only at rounding: the control reassociates, it computes no other
    # quantity.
    assert np.abs(bad - ref).max() <= 1e-15 * np.abs(ref).max()


@pytest.mark.slow
@pytest.mark.parametrize(
    "x, budget", [(2, _TINY), (4, None), (4, _TINY)], ids=["x2-tiny", "x4", "x4-tiny"]
)
def test_streamed_hub16_bit_for_bit(x, budget, monkeypatch):
    """razor hub_deck(16): x4 at the shipped budget is the smallest hub
    that chunks unforced (2 chunks per block); the tiny budget cuts x2 and
    x4 into thousands."""
    ref, _ = _z(_hub(x), monkeypatch, budget=budget, streamed=False)
    got, counts = _z(_hub(x), monkeypatch, budget=budget)
    assert counts.streamed == 2 and counts.chunks > 2
    assert np.array_equal(got, ref)
