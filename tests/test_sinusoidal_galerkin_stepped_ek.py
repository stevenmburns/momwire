"""The sg-ek stepped-radius refusal (momwire#398 D2).

The taper-readiness study measured `SinusoidalGalerkinSolver(extended_kernel
=True)` DIVERGING on a radius step at a junction — not merely inaccurate:
the extrapolated continuum limit on momwire#435's two-wire 10:1 step deck is
`7.110 - 483.925j` against NEC-5's `132.560 - 11.921j`, with the residual
GROWING every rung refined (23.2 -> 285.8 ohm) and a 286 ohm dX spread down
the ladder. That is materially worse than the reduced `sg` row's ~20 ohm
walk-away momwire#435 already documents (a formulation gap, not a
divergence).

**The maintainer's decision (D2): exclude now, gate the exclusion.**
`SinusoidalGalerkinSolver.__init__` refuses `extended_kernel=True` when any
junction's members carry different radii, at CONSTRUCTION time — following
the house idiom `BSplineSolver` uses for its three `use_singular_enrichment`
combination refusals (momwire#396's `_OUT_OF_SCOPE`-style dict, here
`_EK_STEPPED_RADIUS_JUNCTION_REFUSAL` reused by both the `__init__` raise
and `capabilities.refusals["extended_kernel+stepped_radius_junction"]`).

This file gates two things:

1. the refusal actually fires, on the exact deck class the divergence was
   measured on;
2. the refusal is SCOPED to the mixed-radius case — `extended_kernel=True`
   on a UNIFORM-radius deck (with or without a junction) is completely
   untouched, pinned bit-identical against a value computed before this
   unit's change landed (`test_sinusoidal_galerkin.py` and
   `test_extended_kernel_galerkin.py` already gate this solver's ordinary
   behaviour; this file adds the one new construction-time branch only).
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import SinusoidalGalerkinSolver
from momwire.sinusoidal_galerkin import _EK_STEPPED_RADIUS_JUNCTION_REFUSAL


def _two_wire(radii, junction_end="end-start"):
    wires = [
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        np.array([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    ]
    junctions = [[(0, "end"), (1, "start")]]
    return dict(
        wires=wires,
        wire_radius=radii,
        junctions=junctions,
        nsegs=10,
        wavelength=20.0,
        feed_wire_index=0,
    )


# --------------------------------------------------------------------------
# 1. the refusal fires on a stepped-radius junction under EK
# --------------------------------------------------------------------------
def test_ek_refuses_a_radius_step_at_a_junction():
    kw = _two_wire([1.0e-2, 1.0262e-3])
    with pytest.raises(NotImplementedError, match="radius step at a junction"):
        SinusoidalGalerkinSolver(extended_kernel=True, **kw)


def test_the_refusal_cites_the_measured_divergence():
    """House style (momwire#396's precedent): the refusal text carries the
    evidence, not just the verdict, so a reader sees WHY without having to
    find the study."""
    kw = _two_wire([1.0e-2, 1.0262e-3])
    with pytest.raises(NotImplementedError) as exc_info:
        SinusoidalGalerkinSolver(extended_kernel=True, **kw)
    msg = str(exc_info.value)
    assert "7.110" in msg and "483.925" in msg
    assert "132.560" in msg and "11.921" in msg
    assert "momwire#398" in msg or "398" in msg


def test_capabilities_declares_the_refusal():
    reason = SinusoidalGalerkinSolver.capabilities.refusal(
        "extended_kernel", "stepped_radius_junction"
    )
    assert reason == _EK_STEPPED_RADIUS_JUNCTION_REFUSAL


def test_refusal_is_construction_time_only():
    """The raise happens inside `__init__`, before any fill or solve — no
    partial state, no wasted matrix assembly. Cheap to prove: the object
    never comes into existence, so there is nothing to inspect but the
    exception itself (already gated above); this test instead confirms the
    reduced sibling on the SAME deck does not raise, i.e. the refusal is not
    a blanket rejection of the deck."""
    kw = _two_wire([1.0e-2, 1.0262e-3])
    SinusoidalGalerkinSolver(extended_kernel=False, **kw)  # does not raise


# --------------------------------------------------------------------------
# 2. uniform radius is untouched
# --------------------------------------------------------------------------
def test_ek_on_uniform_radius_junction_is_not_refused():
    kw = _two_wire([1.0e-2, 1.0e-2])
    SinusoidalGalerkinSolver(extended_kernel=True, **kw)  # does not raise


def test_ek_on_uniform_radius_answer_is_unchanged_from_the_pre_unit_value():
    """Pinned once against this exact solve, before momwire#398 unit 1's
    `__init__` check landed — the check is a pure early-return gate with no
    other code path touched, so a uniform-radius solve must be unchanged.

    This is a NO-CODE-PATH-TOUCHED claim, not a bit claim, and the bar is
    rel=1e-10 on purpose (momwire#451): the original rel=1e-12 pin was
    bit-identical on the authoring machine but 3.795e-12 off on part of the
    GitHub runner fleet — same commit, different CPU's FMA contraction — so
    it randomly reddened unrelated PRs. 1e-10 is still ~26x wider than the
    measured fleet spread and ~9 orders tighter than any change that could
    mean anything physically. Do not re-tighten it: cross-build bit equality
    is not a property this tree has (the momwire#249 lesson).
    """
    kw = _two_wire([1.0e-2, 1.0e-2])
    z, _ = SinusoidalGalerkinSolver(extended_kernel=True, **kw).compute_impedance()
    assert z == pytest.approx(
        209.4439774792844 + 79.14170016358155j, abs=0.0, rel=1e-10
    )


def test_ek_on_a_junction_with_no_radius_step_but_a_bend_is_not_refused():
    """The refusal keys on RADIUS, not on "is a junction present" — a
    junction whose members bend but share one `a` must solve exactly as it
    did before this unit."""
    wires = [
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        np.array([[5.0, 0.0, 0.0], [5.0, 5.0, 0.0]]),
    ]
    junctions = [[(0, "end"), (1, "start")]]
    SinusoidalGalerkinSolver(
        wires=wires,
        wire_radius=1.0e-2,
        junctions=junctions,
        nsegs=10,
        wavelength=20.0,
        feed_wire_index=0,
        extended_kernel=True,
    )  # does not raise


def test_no_junctions_at_all_is_not_refused_regardless_of_radius():
    """A single uniform wire (no junctions declared) must never hit the new
    check — `self.junctions` is empty, so the loop body never runs."""
    wires = [np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])]
    SinusoidalGalerkinSolver(
        wires=wires,
        wire_radius=1.0e-2,
        nsegs=10,
        wavelength=20.0,
        feed_wire_index=0,
        extended_kernel=True,
    )  # does not raise
