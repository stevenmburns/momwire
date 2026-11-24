"""The sinusoidal pair-reuse census folds its bands into one table.

A deck whose rows repeat (the ELT whip: 0.92 M classes, 5.0 M band classes)
used to hold every band's classes until one merge, a census five times its
plan, and declined on the 256 MB budget of the time (momwire#2019). The
census now holds each class once (`_sin_pair_reuse`, THE CENSUS). Here in
miniature: a curtain of equal parallel wires, whose every band of rows sees
most of the deck's classes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import _sin_pair_reuse as pr
from momwire import sinusoidal
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 4: through the drop-in
# on Linux and Windows).
pytestmark = pytest.mark.reach(4)

LAM = 10.7
WIRES, SEGS = 10, 41  # 410 segments: past SMALL_DECK_SEGS_FREE

# The miniature's numbers at 8 workers: 168,100 pairs, 463 classes, 2,990
# near pairs; a plan whose fill is 0.90 MB beside a 0.64 MB class map. The
# census that held every band's classes counted 2.58 MB (8.1 k band classes
# at 240 B, bands of 16 rows), the fold 1.26 MB.
BIG = 1 << 40


@pytest.fixture(autouse=True)
def _whips_bands(monkeypatch):
    """The band height the whip had at 8 threads, on any machine: the census's
    bands, not the pool that runs them, decide how often a class repeats."""
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)
    monkeypatch.setattr(pr, "_workers", lambda: 8)
    pr.clear_shared_plans()
    yield
    pr.clear_shared_plans()


def _curtain():
    wires = [
        np.array([[0.1 * LAM * i, 0.0, 0.0], [0.1 * LAM * i, 0.0, 0.5 * LAM]])
        for i in range(WIRES)
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[SEGS]] * WIRES,
        feed_wire_index=WIRES // 2,
        wavelength=LAM,
        wire_radius=2e-3,
    )


def _fill(pair_reuse=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SinusoidalSolver(**_curtain(), pair_reuse=pair_reuse)
        geom = s._build_geometry()
        with s._operating_medium(geom) as medium:
            G, _ = s._assemble_Z(geom, s.k, s._medium_eta(medium))
    return G, s


def test_a_deck_whose_rows_repeat_reuses(monkeypatch):
    """The whip's deck in miniature, served with the classes an allowance
    that never binds finds (its bands follow the allowance: the same
    plan)."""
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", BIG)
    lifted, s = _fill(pair_reuse=True)
    assert s.pair_reuse_declined is None
    pr.clear_shared_plans()
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", None)
    on, s = _fill(pair_reuse=True)
    assert s.pair_reuse_declined is None
    pairs, classes, near = s.pair_reuse_census["direct"]
    assert (pairs, near) == (168_100, 2_990)
    assert classes < pairs // 300
    report = s.fill_report(wires=False)["reuse"]
    assert report["state"] == "served"
    # what the census holds a class: test_the_census_holds_its_classes_once
    assert report["census_peak_bytes"] is not None
    assert on.tobytes() == lifted.tobytes()


def test_the_census_holds_its_classes_once(monkeypatch):
    """What the census counts grows with the deck's classes: beside the map
    and the bands in flight (`BAND_SHARE` of the plan's allowance), a few
    hundred bytes a class, where holding each band's would be ~240 B per
    class per band that sees it."""
    _, s = _fill(pair_reuse=True)
    plan = next(iter(s._sin_reuse_plans.values()))[1]
    bands = pr.BAND_SHARE * pr.plan_allowance(plan.n_pairs)
    held = plan.census_peak_bytes - plan.cls.nbytes - bands
    assert held < 600 * plan.n_classes + 8 * plan.n_near + (256 << 10)


class _Counting:
    """The extension, with the free-space kernel's (rows x sources) counted."""

    def __init__(self, inner):
        self._inner = inner
        self.pairs = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def sinusoidal_field_tensor(self, obs_c, obs_t, src_c, *a):
        self.pairs += obs_c.shape[0] * src_c.shape[0]
        return self._inner.sinusoidal_field_tensor(obs_c, obs_t, src_c, *a)


@pytest.mark.skipif(acc is None, reason="needs the C++ field-tensor kernel")
def test_the_served_fill_evaluates_a_fraction_of_the_pairs(monkeypatch):
    """Adversarial: the kernel itself is counted, so a fill that declined,
    or one that reports reuse and evaluates every pair anyway, fails."""
    counting = _Counting(acc)
    monkeypatch.setattr(sinusoidal, "_acc", counting)
    _, s = _fill(pair_reuse=True)
    report = s.fill_report(wires=False)["reuse"]
    assert report["state"] == "served"
    assert counting.pairs == report["kernel_evals"]
    # 463 classes and 2,990 near pairs, these evaluated in rectangles of
    # rows by their union of sources: 9,489 kernel pairs of 168,100.
    assert counting.pairs < report["pairs"] // 10


def test_off_is_the_shipped_fill_bit_for_bit(monkeypatch):
    base, _ = _fill()
    off, s = _fill(pair_reuse=False)
    assert s.pair_reuse_declined is None and s.pair_reuse_census == {}
    assert off.tobytes() == base.tobytes()


# IntGroups (the census's tables): first appearance over every add's rows in turn.


def _reference(chunks):
    """First-appearance ids over the concatenation, by a dict of tuples."""
    seen, out = {}, []
    for cols in chunks:
        for row in zip(*(c.tolist() for c in cols)):
            out.append(seen.setdefault(row, len(seen)))
    return out


def _chunks(width, sizes, hi, seed):
    rng = np.random.default_rng(seed)
    return [
        [rng.integers(-hi, hi, size=n, dtype=np.int64) for _ in range(width)]
        for n in sizes
    ]


@pytest.mark.skipif(
    acc is None or not getattr(acc, "int_groups", False), reason="needs IntGroups"
)
@pytest.mark.parametrize("width", [1, 3, 5, 6, 8])
def test_groups_are_numbered_across_adds(width):
    chunks = _chunks(width, [0, 7, 500, 1, 3000, 64], 3, seed=width)
    table = acc.IntGroups(width)
    got = []
    for cols in chunks:
        got += np.asarray(table.add(cols)).tolist()
    assert got == _reference(chunks)
    assert len(table) == len(set(got))


@pytest.mark.skipif(
    acc is None or not getattr(acc, "int_groups", False), reason="needs IntGroups"
)
def test_the_kernel_table_grows_with_its_groups_not_its_rows():
    table = acc.IntGroups(5)
    cols = [np.zeros(200_000, np.int64) for _ in range(5)]
    cols[0][:] = np.arange(200_000) % 1000
    for _ in range(5):
        table.add(cols)
    assert len(table) == 1000
    assert table.nbytes() < 1 << 20
    with pytest.raises(RuntimeError):
        acc.IntGroups(9)
    with pytest.raises(RuntimeError):
        table.add(cols[:4])
