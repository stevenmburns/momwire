"""The end loops' kernel tables, a span of ends per call — momwire#1168 U4.

`_ends_and_corner` and `_ends_and_corner_reversed` made one `_tables` call per
end: 2,823 calls at razor hub_deck(16) x8, nearly all memo hits. `_end_tables`
serves a span of consecutive ends per call instead, each end's rows labelled
with its position so that `designed_tables` groups the column routes' fresh
rows by (label, exact rho) — the columns the per-end calls built. Without the
labels, fresh rows of two ends at one rho share a column whose smaller s_min
widens the rule, and Z moves (the audit: 12 of 1.96 M entries at x8).

The reference is in-process: `_END_BATCH_PAIRS = 1` makes every span one end,
which passes no labels and IS the per-end call. The integrated rows compare Z
at `np.array_equal`; the negative control turns the labels off and requires
the same comparison to FAIL, which is what says the gate can see the defect
it exists for.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill as cf
from momwire import _near_interface as ni

from test_crossing_serve_524 import crossing_deck
from test_triple_memo_1168 import CASES, _razor, _solve


@pytest.fixture(autouse=True)
def _grid_route(monkeypatch):
    """This file's subject is the GRID route (`_chunked_tables` and the
    lookup path); since momwire#1173 design B a deck whose nodes factorise
    takes the product route instead, which never calls it.
    `test_product_grid_1173` gates the product route against this one."""
    monkeypatch.setattr(cf, "_PRODUCT_TABLES", False)


# ----------------------------------------------------------------------
# Unit rows: the label semantics on designed_tables itself
# ----------------------------------------------------------------------

_EPS, _K = 4.0 - 0.5j, 2 * np.pi / 20
# Three calls. Call 1 re-asks one of call 0's rows (0.3, 0.9, -0.4) and adds
# fresh rows at call 0's rho 0.3 with a far smaller s = z - z', so a column
# merged across the two calls takes call 1's s_min and changes call 0's rule.
_CALLS = [
    ([0.3, 0.3, 0.7], [0.9, 1.2, 0.5], [-0.4, -0.6, -0.3]),
    ([0.3, 0.3, 0.7, 0.3], [0.01, 0.9, 0.02, 0.05], [-0.01, -0.4, 0.0, -0.02]),
    ([0.7, 1.1], [0.5, 0.2], [-0.3, -0.1]),
]


def _sequence_and_batch(labelled):
    """The calls made one by one on one memo, and made as one call on a fresh
    memo (labels = call index when `labelled`): both answers and both memos."""
    calls = [tuple(np.asarray(c, dtype=float) for c in call) for call in _CALLS]
    memo_seq = ni.TripleMemo()
    seq = [ni.designed_tables(_EPS, _K, *c, memo=memo_seq) for c in calls]
    seq = {key: np.concatenate([s[key] for s in seq]) for key in ni.KEYS}
    cat = [np.concatenate([c[i] for c in calls]) for i in range(3)]
    labels = np.concatenate([np.full(c[0].size, j) for j, c in enumerate(calls)])
    memo_one = ni.TripleMemo()
    one = ni.designed_tables(
        _EPS, _K, *cat, memo=memo_one, group_labels=labels if labelled else None
    )
    return seq, one, memo_seq, memo_one


def _same(seq, one, memo_seq, memo_one):
    return (
        all(np.array_equal(seq[key], one[key]) for key in ni.KEYS)
        and memo_one.keys() == memo_seq.keys()
        and np.array_equal(np.array(memo_one.values()), np.array(memo_seq.values()))
    )


@pytest.mark.parametrize("route", ["twin", "numpy"])
def test_labelled_call_is_the_call_sequence_to_the_bit(route, monkeypatch):
    """Values AND memo contents (keys, values, insertion order) equal the
    sequence's, on both column routes; unlabelled, the same batch differs —
    so the deck above does exercise a merged column."""
    if route == "twin" and not ni._use_column_accel():
        pytest.skip("the C++ column twin is not built")
    monkeypatch.setattr(ni, "_FORCE_NUMPY", route == "numpy")
    assert ni._use_column_route()
    assert _same(*_sequence_and_batch(labelled=True))
    assert not _same(*_sequence_and_batch(labelled=False))


def test_labels_need_a_memo_and_integers():
    """Without a memo a call sequence dedups nothing across its calls, which
    one call cannot reproduce; a float label is not a call index."""
    with pytest.raises(ValueError, match="memo"):
        ni.designed_tables(_EPS, _K, 0.3, 0.2, -0.1, group_labels=0)
    with pytest.raises(TypeError, match="integers"):
        ni.designed_tables(
            _EPS, _K, 0.3, 0.2, -0.1, memo=ni.TripleMemo(), group_labels=0.0
        )


def test_without_a_memo_every_span_is_one_end():
    assert cf._end_groups(5, 10, None) == [(i, i + 1) for i in range(5)]
    assert cf._end_groups(5, 10, ni.TripleMemo()) == [(0, 5)]


# ----------------------------------------------------------------------
# Integrated rows: Z through the real constructors
# ----------------------------------------------------------------------


class _EndCalls:
    """Counts `_tables` calls made from inside the two end routines, the
    labelled ones among them, the ends they served and the spans
    `_end_groups` planned for them."""

    def __init__(self, monkeypatch):
        self.calls = self.labelled = self.ends = self.spans = 0
        depth = [0]
        tables = cf._tables

        def spy(*a, **k):
            if depth[0]:
                self.calls += 1
                self.labelled += k.get("group_labels") is not None
            return tables(*a, **k)

        monkeypatch.setattr(cf, "_tables", spy)

        for name in ("_ends_and_corner", "_ends_and_corner_reversed"):
            real = getattr(cf, name)

            def wrapped(
                ctx, A, B, *a, _real=real, _fwd=name == "_ends_and_corner", **k
            ):
                loops = []
                if not _fwd or k.get("test_ends", True):
                    loops.append((len(A["ends"]), B["nodes"].shape[0]))
                if not _fwd or k.get("source_ends", True):
                    loops.append((len(B["ends"]), A["nodes"].shape[0]))
                for n_ends, n_nodes in loops:
                    self.ends += n_ends
                    self.spans += len(cf._end_groups(n_ends, n_nodes, k.get("memo")))
                depth[0] += 1
                try:
                    return _real(ctx, A, B, *a, **k)
                finally:
                    depth[0] -= 1

            monkeypatch.setattr(cf, name, wrapped)


def _z(make, monkeypatch, *, budget=None, labels=True):
    """Z of a fresh solve at `_END_BATCH_PAIRS = budget` (default: the
    shipped budget), with its end-call counts."""
    with monkeypatch.context() as mp:
        if budget is not None:
            mp.setattr(cf, "_END_BATCH_PAIRS", budget)
        mp.setattr(cf, "_END_LABELS", labels)
        counts = _EndCalls(mp)
        z, _currents = _solve(make)
    return z, counts


def test_one_call_per_span_not_per_end(monkeypatch):
    """The count proof, cheap enough for every PR: on crossing_deck(1) the
    end loops make one `_tables` call per planned span — 4 calls for 62 ends
    at the default budget, labelled wherever a span holds more than one end —
    and 62 at spans of one, none labelled."""
    make = lambda: _razor(crossing_deck(1))  # noqa: E731
    _z0, per_end = _z(make, monkeypatch, budget=1)
    assert per_end.calls == per_end.spans == per_end.ends == 62
    assert per_end.labelled == 0
    _z1, spans = _z(make, monkeypatch)
    assert spans.ends == 62
    assert spans.calls == spans.spans == 4
    assert spans.labelled == 2


@pytest.mark.slow
@pytest.mark.parametrize(
    "case, budget",
    [(case, None) for case in sorted(CASES)] + [("razor-hub16-x2", 3000)],
)
def test_spans_are_bit_identical_to_per_end_calls(case, budget, monkeypatch):
    """Z to the bit against spans of one end, at the shipped budget on every
    trunk and at a 3,000-pair budget (spans of a few ends, 154 calls) on razor.

    Counts at x2, per end -> default budget: razor hub16 723 -> 11 calls,
    bspline hub16 19 -> 2, SG hub16 36 -> 2; razor hub16 x4 1,423 -> 33. The
    same Z equals origin/main 003f247's, by script, at budgets 1, 3,000, 64 k
    and unbounded, on the twin and the numpy column routes and the point
    route."""
    z_ref, per_end = _z(CASES[case], monkeypatch, budget=1)
    assert per_end.calls == per_end.ends and per_end.labelled == 0
    z_new, spans = _z(CASES[case], monkeypatch, budget=budget)
    assert spans.calls == spans.spans < spans.ends, "the spans never formed"
    assert spans.labelled > 0, "no labelled call: the labels were never tested"
    assert np.array_equal(z_new, z_ref)


@pytest.mark.slow
def test_negative_control_unlabelled_spans_move_z(monkeypatch):
    """The gate above can fail: with the labels off, the same spans move Z.

    Measured 2026-09-23 against origin/main, labels off at the default
    budget: bspline hub16 x2 2,729 entries (1.1e-18 of max|Z|), SG hub16 x2
    370, razor hub16 x4 3, razor hub16 x2 1; crossing_deck(1) none. bspline
    is the row because it moves the most entries at the lowest cost."""
    case = "bspline-hub16-x2"
    z_ref, _ = _z(CASES[case], monkeypatch, budget=1)
    z_off, spans = _z(CASES[case], monkeypatch, labels=False)
    assert spans.calls == spans.spans < spans.ends and spans.labelled == 0
    assert not np.array_equal(z_off, z_ref)
