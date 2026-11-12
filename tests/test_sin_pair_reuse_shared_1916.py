"""`pair_reuse`'s plans outlive their solver (momwire#1916 part C).

The portal builds one SinusoidalSolver per frequency, so a per-instance
cache paid the census at every FR point. The plan holds no k and is keyed on
the geometry's bytes, so the process keeps it (`_sin_pair_reuse.
shared_plan`): a second solver over the same wires finds the first one's
classes, and its Z is the Z of a solver that built its own, bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _sin_pair_reuse as pr
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        acc is None or not getattr(acc, "pair_reuse_reduce_1916", False),
        reason="accelerator None or without pair_reuse_reduce_1916",
    ),
]

LAM = 10.7

# Three parallel dipoles in free space, three FR points: a sweep the portal
# answers with one solver per frequency.
DECK = """CM shared plans
CE
GW 1 41 0 -2.5 10 0 2.5 10 .001
GW 2 41 1.5 -2.5 10 1.5 2.5 10 .001
GW 3 41 3.0 -2.5 10 3.0 2.5 10 .001
GE 0
EX 0 2 21 0 1 0
FR 0 3 0 0 28 0.5
XQ
EN
"""


@pytest.fixture
def builds(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)
    calls = []
    orig = pr.build_plan

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(pr, "build_plan", counting)
    return calls


def _deck(dx=1.5):
    wires = [np.array([[dx * i, -2.5, 10.0], [dx * i, 2.5, 10.0]]) for i in range(3)]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[41]] * 3,
        feed_wire_index=1,
        wavelength=LAM,
        wire_radius=1e-3,
    )


def _z(wavelength=LAM, dx=1.5):
    deck = {**_deck(dx), "wavelength": wavelength}
    s = SinusoidalSolver(**deck, pair_reuse=True)
    geom = s._build_geometry()
    Z, _ = s._assemble_Z(geom, s.k)
    assert s.pair_reuse_declined is None
    return Z


def test_a_second_solver_finds_the_first_ones_classes(builds):
    _z(LAM)
    assert len(builds) == 1
    shared = _z(0.97 * LAM)
    assert len(builds) == 1
    pr.clear_shared_plans()
    own = _z(0.97 * LAM)
    assert len(builds) == 2
    assert shared.tobytes() == own.tobytes()


def test_other_wires_are_another_plan(builds):
    _z(dx=1.5)
    _z(dx=1.6)
    assert len(builds) == 2
    _z(dx=1.5)
    assert len(builds) == 2


def test_the_budget_keeps_the_newest(builds, monkeypatch):
    monkeypatch.setattr(pr, "SHARED_PLAN_BYTES", 1)
    _z(dx=1.5)
    _z(dx=1.6)
    _z(dx=1.6)
    assert len(builds) == 2
    _z(dx=1.5)
    assert len(builds) == 3


def test_the_portal_pays_one_census_for_a_sweep(builds, monkeypatch):
    """The production seam: the NEC-4.2 slot's sinusoidal launcher, which
    builds a solver per FR point. Before the shared cache, three."""
    from momwire.dropin.nec5._shell import render

    monkeypatch.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    shared = render(DECK, basis="sinusoidal", dialect="nec4")
    assert len(builds) == 1
    # The same run with every solver building its own: three censuses, and
    # the same printout.
    monkeypatch.setattr(pr, "shared_plan", lambda key, build: build())
    own = render(DECK, basis="sinusoidal", dialect="nec4")
    assert len(builds) == 4

    def results(text):
        # The printout's own clocks are the one thing two runs differ in.
        return [
            ln for ln in text.splitlines() if "SEC." not in ln and "RUN TIME" not in ln
        ]

    assert results(shared) == results(own)
