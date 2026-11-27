"""bs2's census sees the same-edge pairs (momwire#2063 step 3).

The fill without reuse computes every pair of one edge by the same-edge
routine (the analytic static moments plus the regularised part on the edge's
arc) and every other pair by the off-edge kernel. The census now keys the
same-edge pairs too, APART (`Key::Apart`): a class of them is evaluated by
the same-edge routine at its representative, and the key's routine bit keeps
the two routines' classes disjoint, so every pair takes the value of the
routine its own fill uses. `offset_reuse`'s inexact mechanisms (the
free-space offset block, the image offset block and the remainder's offset
fold, each spreading one row of an edge's moments by index) are gone; the
census serves those blocks.

The gates, and why each is not green by construction:

* the routine count reads each pair's class off the plan and checks it
  against the fill's own choice (same edge, direct block): a plan whose key
  mixed the routines would fail it, and the routine probe (the key without
  its routine bit) is shown to fail it;
* the probe's mixed classes move Z by far more than the drift on a deck
  built for it (one straight wire split into two collinear edges, so pairs
  across the split are congruent to same-edge pairs);
* the #1971 hairpin holds the registry's drift; its bridge is one edge of
  21 segments of 95 um on a 1 mm radius, every pair of it within a radius,
  and a near rule in segment lengths alone (the probe) moves it ~2.5e-8.
  It is the deck that sets the drift: its tensor route assembles Z tile by
  tile from the class values, in the windowed assembler's sum order, which
  alone moves its Zin 7e-10 (cond(Z) ~7e9);
* every agreement first shows the census served the same-edge pairs (their
  classes counted, fewer than their pairs).
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel, _features
from momwire import bspline as bspline_mod
from momwire.bspline import BSplineSolver

from _reuse_seam import bs2

acc = _accel.acc
pytestmark = [
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "bspline_reuse_set_routine_probe"),
        reason="needs the C++ accelerator",
    ),
    pytest.mark.filterwarnings("ignore"),
]

DRIFT = _features.family("pair_reuse", "bspline").drift
LAM = 14.0


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("MOMWIRE_X_PAIR_REUSE", "MOMWIRE_X_OFFSET_REUSE"):
        monkeypatch.delenv(k, raising=False)
    acc.bspline_reuse_set_routine_probe(0)
    acc.bspline_reuse_set_same_edge_near_lengths(-1)
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.pair_reuse_cap_compression(-1)
    yield
    acc.bspline_reuse_set_routine_probe(0)
    acc.bspline_reuse_set_same_edge_near_lengths(-1)
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.pair_reuse_cap_compression(-1)


def collinear(n=60):
    """One straight wire as two collinear edges of equal segment length (a
    polyline point at its middle), beside a parallel element: a pair across
    the middle point is congruent to a same-edge pair at the same offset."""
    L = 0.48 * LAM
    return {
        "wires": [
            np.array([[0.0, -L / 2, 5.0], [0.0, 0.0, 5.0], [0.0, L / 2, 5.0]]),
            np.array([[0.2 * LAM, -L / 2, 5.0], [0.2 * LAM, L / 2, 5.0]]),
        ],
        "n_per_edge_per_wire": [[n, n], [2 * n]],
        "feeds": [(0, None, 1.0 + 0j)],
        "wavelength": LAM,
        "wire_radius": 0.004,
    }


def bent_yagi(n=40):
    """Five elements, the odd ones bent into a V (two edges, a kink)."""
    Ld = 0.47 * LAM
    wires, nper = [], []
    for i in range(5):
        x, L = 0.2 * LAM * i, Ld * (1.05 - 0.05 * i)
        if i % 2:
            wires.append(np.array([[x, -L / 2, 4.3], [x, 0.0, 4.0], [x, L / 2, 4.3]]))
            nper.append([n // 2, n // 2])
        else:
            wires.append(np.array([[x, -L / 2, 4.0], [x, L / 2, 4.0]]))
            nper.append([n])
    return {
        "wires": wires,
        "n_per_edge_per_wire": nper,
        "feeds": [(1, None, 1.0 + 0j)],
        "wavelength": LAM,
        "wire_radius": 0.006,
    }


def hairpin(x0=3.3):
    """razor2p's #1971 hairpin: two 10 m wires 2 mm apart joined by a 2 mm
    bridge, 21 segments each, so the bridge's h = 95 um is under the 1 mm
    radius and cond(Z) is ~7e9; translated off the origin."""
    ws = [
        [[0, -5, 8], [0, 5, 8]],
        [[0, 5, 8], [0.002, 5, 8]],
        [[0.002, 5, 8], [0.002, -5, 8]],
    ]
    return {
        "wires": [np.array(w, dtype=float) + [x0, 0.0, 0.0] for w in ws],
        "n_per_edge_per_wire": [[21], [21], [21]],
        "feeds": [(0, 5.238095238095238, 1 + 0j)],
        "wire_radius": 0.001,
        "wavelength": 20.0,
    }


def zin(deck, **kw):
    s = bs2(**deck, **kw)
    return complex(np.ravel(s.compute_impedance()[0])[0]), s


def routine_census(s):
    """Per class, how many of its pairs the fill computes by the same-edge
    routine and how many by the off-edge kernel, against whether the plan
    evaluates the class by the same-edge routine: (mixed classes, classes
    whose routine is not their pairs', the plan)."""
    geom = s._build_geometry()
    plan = s._reuse_plan(geom)
    assert plan is not None, s.pair_reuse_declined
    n = int(geom["n_segs_total"])
    edge = np.full(n, -1)
    for e, sl in enumerate(s._same_edge_slices(geom)):
        edge[sl] = e
    idx = np.arange(n, dtype=np.int64)
    cls = plan.classes_of(0, idx, idx)
    same = edge[:, None] == edge[None, :]
    apart = np.zeros(plan.classes, dtype=bool)
    apart[plan.apart()[0]] = True
    n_same = np.bincount(cls[same], minlength=plan.classes)
    n_other = np.bincount(cls[~same], minlength=plan.classes)
    mixed = int(np.sum((n_same > 0) & (n_other > 0)))
    wrong = int(np.sum(apart & (n_other > 0)) + np.sum(~apart & (n_same > 0)))
    return mixed, wrong, plan


@pytest.mark.parametrize("make", [collinear, bent_yagi], ids=["collinear", "bent"])
def test_every_pair_takes_its_own_fills_routine(make):
    """No class mixes the routines, and each class is evaluated by the
    routine its pairs' own fill uses; the same-edge pairs are served (fewer
    classes than pairs) and on against off holds the drift."""
    deck = make()
    off, _ = zin(deck)
    on, s = zin(deck, pair_reuse=True)
    mixed, wrong, plan = routine_census(s)
    assert (mixed, wrong) == (0, 0)
    st = plan.stats()
    assert 0 < st["same_edge_classes"] < st["same_edge_pairs"] // 2, st
    assert abs(on - off) / abs(off) <= DRIFT


def test_without_the_routine_bit_classes_mix_and_z_moves():
    """The adversarial gate: keyed with the general key and no routine bit,
    pairs across the collinear split share the same-edge pairs' classes.
    At the shipped pair order (8) the two routines agree on such a far pair
    to ~1e-15 of Z (the probe moves Zin ~2e-14), so the deck is built with
    the off-edge rule at order 2, where they differ by far more than the
    drift: the right key holds it there, the probe does not."""
    deck = collinear()
    off, _ = zin(deck, n_qp_pair=2)
    on, s = zin(deck, n_qp_pair=2, pair_reuse=True)
    assert routine_census(s)[:2] == (0, 0)
    assert abs(on - off) / abs(off) <= DRIFT
    acc.bspline_reuse_set_routine_probe(1)
    wrong_z, s = zin(deck, n_qp_pair=2, pair_reuse=True)
    mixed, wrong, _plan = routine_census(s)
    assert mixed > 0 and wrong > 0, (mixed, wrong)
    assert abs(wrong_z - off) / abs(off) > 1e-8


def test_hairpin_holds_the_drift():
    """The #1971 hairpin on bs2: the bridge's pairs are all within a radius,
    so they are near singles of the routine, their own fills' bits; the
    legs' same-edge pairs past four radii are shared. The census's small-deck
    and cap rules are lifted (63 segments). On moves 5.0e-10 free, of which
    the tensor route's sum order (the windowed assembler's, not
    `assemble_Z`'s) is 7.0e-10 alone: the registry's drift is this deck's."""
    acc.bspline_reuse_set_small_deck_segs(0)
    acc.pair_reuse_cap_compression(1)
    deck = hairpin()
    off, _ = zin(deck)
    on, s = zin(deck, pair_reuse=True)
    st = s._reuse_plan(s._build_geometry()).stats()
    assert 0 < st["same_edge_classes"] < st["same_edge_pairs"], st
    good = abs(on - off) / abs(off)
    assert good <= DRIFT, good
    # The near rule in segment lengths alone shares the bridge's pairs,
    # which differ from their own fills by their arcs' rounding.
    acc.bspline_reuse_set_same_edge_near_lengths(0)
    bad_z, _ = zin(deck, pair_reuse=True)
    assert abs(bad_z - off) / abs(off) > 1e-8


def test_the_chunked_fill_keeps_its_near_singles_and_holds_the_drift(monkeypatch):
    """Past the tensor's budget the plan's window takes each near same-edge
    single from the sweep and the fixup corrects it, so on against off holds
    at the chunked route too (the route read the plan: `pair_reuse_served`)."""
    monkeypatch.setattr(
        BSplineSolver, "_dense_tensor_fits_budget", lambda self, n: False
    )
    deck = bent_yagi()
    off, _ = zin(deck)
    on, s = zin(deck, pair_reuse=True)
    assert s.pair_reuse_served["route"] == "chunked"
    assert abs(on - off) / abs(off) <= DRIFT


def test_a_sweep_that_reads_the_plan_hoists_no_prep(monkeypatch):
    """A swept fill with a served plan builds no same-edge prep (no fill
    reads it), and agrees with its per-k fills."""
    calls = []
    orig = BSplineSolver._same_edge_prep
    monkeypatch.setattr(
        BSplineSolver,
        "_same_edge_prep",
        lambda self, geom: calls.append(1) or orig(self, geom),
    )
    deck = bent_yagi()
    s = bs2(**deck, pair_reuse=True)
    ks = 2 * np.pi / np.array([LAM * 1.01, LAM, LAM * 0.99])
    z = s.compute_impedance_swept(ks)
    assert calls == []
    for i, k in enumerate(ks):
        t = bs2(**{**deck, "wavelength": 2 * np.pi / k}, pair_reuse=True)
        zi = complex(np.ravel(t.compute_impedance()[0])[0])
        assert abs(complex(np.ravel(z[i])[0]) - zi) / abs(zi) < 1e-12


def test_the_toeplitz_form_alone_serves_no_pair():
    """The same-edge Toeplitz form (what was `offset_reuse`) held on without
    the census (`_reuse_seam`) copies no pair: what is left of it
    is the regularised moments' Toeplitz form inside the same-edge kernel."""
    deck = bent_yagi()
    off, _ = zin(deck)
    alone, s = zin(deck, toeplitz=True)
    assert s.pair_reuse_served is None
    assert (
        abs(alone - off) / abs(off) <= _features.family("pair_reuse", "bspline").drift
    )
    assert not hasattr(bspline_mod, "FREE_OFFSET_EDGES")
