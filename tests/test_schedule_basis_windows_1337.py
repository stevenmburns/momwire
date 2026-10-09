"""The basis-axis half of the observer restriction (momwire#1337 phase 2,
PR 4): `ObserverRows.basis_windows`, `target`, `key`, and
`_schedule.prepare(restrict=)`.

Razor's observer rows are basis tents, with the junction through-currents
numbered after the interior tents on the same axis, so its `rows=` (PR 5)
cuts its prepared T1 windows on the BASIS axis where bspline's chunks cut
segment rows. What is gated here, with no solve:

* **the walk is the hand-written loop** -- for every dense window, the
  maximal runs of requested rows inside it, in order; on an empty
  restriction, the full one (== the dense windows), runs that cross window
  boundaries, ragged windows, and runs that span the interior/junction seam;
* **a restricted prepare builds only requested rows, at the dense
  boundaries**, with one checkpoint per DENSE window, and is the
  unrestricted prepare on the full restriction;
* **`target` is `row_of`'s contiguous image** on the compact target and the
  identity on the square one, and refuses a range not wholly requested;
* **`key` binds R and only R**, so a prepared fill can refuse a replay under
  another restriction;
* **a malformed R is refused, not normalised** (unsorted, duplicate, out of
  range).
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _schedule


@pytest.fixture
def stats():
    _schedule.reset_stats()
    yield _schedule.STATS
    _schedule.reset_stats()


def _restrict(basis_rows, n_basis, *, compact=True):
    # The segment side is a stand-in: the basis-axis methods never read it.
    # `basis_rows_of` is how a formulation names R, so R goes in through it.
    R = np.asarray(basis_rows, dtype=np.int64)
    return _schedule.ObserverRows(
        np.arange(1), 1, n_basis, lambda _seg: R, compact=compact
    )


def _hand_windows(R, i0, i1):
    """The loop `basis_windows` replaces: walk the dense window a row at a
    time and close a sub-window at every gap."""
    held = set(int(r) for r in R)
    out, start = [], None
    for m in range(i0, i1):
        if m in held and start is None:
            start = m
        if m not in held and start is not None:
            out.append((start, m))
            start = None
    if start is not None:
        out.append((start, i1))
    return out


def _dense(n, step):
    return [(lo, min(lo + step, n)) for lo in range(0, n, step)]


# n_interior = 30 interior tents, then 7 junction rows [30, 37).
N_INT, N_BASIS = 30, 37
CASES = {
    "empty": [],
    "full": list(range(N_BASIS)),
    "several_runs": [*range(2, 9), *range(11, 12), *range(14, 23), 29],
    "interior_and_its_junctions": [*range(5, 12), 31, 32, 35],
    "across_the_seam": [*range(26, 34)],
    "junctions_only": [30, 31, 33, 36],
    "singletons": [0, 2, 4, 36],
}


@pytest.mark.parametrize("step", [1, 2, 4, 5, 7, 36, 37, 100])
@pytest.mark.parametrize("case", sorted(CASES))
def test_basis_windows_is_the_hand_loop(case, step, stats):
    R = CASES[case]
    restrict = _restrict(R, N_BASIS)
    n = 0
    for i0, i1 in _dense(N_BASIS, step):
        got = restrict.basis_windows(i0, i1)
        assert got == _hand_windows(R, i0, i1), (i0, i1)
        n += len(got)
    assert stats["basis_windows"] == n
    covered = [
        m
        for a0, a1 in (
            w for d in _dense(N_BASIS, step) for w in restrict.basis_windows(*d)
        )
        for m in range(a0, a1)
    ]
    assert covered == sorted(R)


def test_the_full_restriction_is_the_dense_windows():
    restrict = _restrict(range(N_BASIS), N_BASIS)
    for step in (1, 3, 8, 37):
        for i0, i1 in _dense(N_BASIS, step):
            assert restrict.basis_windows(i0, i1) == [(i0, i1)]


def test_ragged_windows_any_partition():
    # Windows need not be a fixed step: any partition of the axis, including
    # empty windows and ones past the last row, gives the hand loop's runs.
    R = CASES["several_runs"] + CASES["junctions_only"]
    R = sorted(R)
    restrict = _restrict(R, N_BASIS)
    cuts = [0, 1, 1, 3, 10, 11, 13, 22, 23, 30, 31, 35, 37]
    for i0, i1 in zip(cuts, cuts[1:]):
        assert restrict.basis_windows(i0, i1) == _hand_windows(R, i0, i1)
    assert restrict.basis_windows(37, 50) == []


@pytest.mark.parametrize("step", [1, 3, 6, 37])
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_restricted_prepare_builds_only_requested_rows(case, step, stats):
    R = CASES[case]
    restrict = _restrict(R, N_BASIS)
    built, ticks = [], []

    def build(lo, hi):
        built.append((lo, hi))
        return (hi - lo, f"{lo}:{hi}")

    got = _schedule.prepare(
        N_BASIS, step, build, checkpoint=lambda: ticks.append(1), restrict=restrict
    )
    want = [
        (a0, a1, a1 - a0, f"{a0}:{a1}")
        for i0, i1 in _dense(N_BASIS, step)
        for a0, a1 in _hand_windows(R, i0, i1)
    ]
    assert got == want
    assert built == [w[:2] for w in want]
    # One checkpoint per DENSE window, requested rows or not.
    assert len(ticks) == len(_dense(N_BASIS, step)) == stats["chunks"]
    assert stats["prepares"] == 1
    assert stats["basis_windows"] == len(want)
    # Replay walks exactly the built windows.
    assert list(_schedule.replay(got)) == got
    assert stats["replays"] == len(want)


def test_the_full_restriction_prepares_the_unrestricted_windows(stats):
    def build(lo, hi):
        return (lo * 100 + hi,)

    dense = _schedule.prepare(N_BASIS, 4, build)
    assert stats["basis_windows"] == 0  # unrestricted never asks
    full = _schedule.prepare(N_BASIS, 4, build, restrict=_restrict(range(37), 37))
    assert full == dense


def test_prepare_refuses_a_restriction_on_another_axis():
    with pytest.raises(ValueError, match="37 basis rows cannot cut a 36-row"):
        _schedule.prepare(36, 4, lambda lo, hi: (), restrict=_restrict([1], 37))


@pytest.mark.parametrize("case", sorted(set(CASES) - {"empty"}))
def test_target_is_row_of_on_compact_and_identity_on_square(case):
    R = np.asarray(CASES[case], dtype=np.int64)
    compact = _restrict(R, N_BASIS, compact=True)
    square = _restrict(R, N_BASIS, compact=False)
    for i0, i1 in _dense(N_BASIS, 4):
        for a0, a1 in compact.basis_windows(i0, i1):
            t0, t1 = compact.target(a0, a1)
            assert np.array_equal(np.arange(t0, t1), compact.row_of[a0:a1])
            assert np.array_equal(R[t0:t1], np.arange(a0, a1))
            assert square.target(a0, a1) == (a0, a1)


@pytest.mark.parametrize("compact", [True, False])
def test_target_refuses_an_unrequested_row(compact):
    restrict = _restrict([2, 3, 4, 31], N_BASIS, compact=compact)
    assert restrict.target(2, 5) == ((0, 3) if compact else (2, 5))
    for lo, hi in [(1, 3), (2, 6), (4, 32), (5, 5), (31, 38), (-1, 2)]:
        with pytest.raises(ValueError, match="not all requested"):
            restrict.target(lo, hi)


def test_key_binds_the_rows_and_only_the_rows():
    a = _restrict([1, 2, 31], N_BASIS, compact=True)
    assert a.key == _restrict([1, 2, 31], N_BASIS, compact=False).key
    assert a.key != _restrict([1, 2, 32], N_BASIS).key
    assert a.key != _restrict([1, 2], N_BASIS).key
    assert a.key != _restrict([1, 2, 31], N_BASIS + 1).key
    assert _restrict([], N_BASIS).key != _restrict([], N_BASIS + 1).key
    hash(a.key)
    assert _schedule.restriction_key(None) is None
    assert _schedule.restriction_key(a) == a.key
    # Equal R from different array spellings is the same key.
    b = _schedule.ObserverRows(
        np.arange(1),
        1,
        N_BASIS,
        lambda _s: np.array([1, 2, 31], np.int32),
        compact=True,
    )
    assert b.key == a.key


@pytest.mark.parametrize(
    "R, match",
    [
        ([3, 2, 5], "sorted 1-D array of distinct rows"),
        ([2, 2, 5], "sorted 1-D array of distinct rows"),
        ([[1, 2]], "sorted 1-D array of distinct rows"),
        ([-1, 2], r"outside \[0, 37\)"),
        ([2, 37], r"outside \[0, 37\)"),
    ],
)
def test_a_malformed_basis_row_set_is_refused(R, match):
    with pytest.raises(ValueError, match=match):
        _restrict(R, N_BASIS)
