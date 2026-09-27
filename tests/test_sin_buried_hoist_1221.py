"""momwire#1221 (#1220 stage 0): SG's testing-agnostic buried code lives on
`SinusoidalSolver`.

The hoist is a pure refactor, so its real gate is bit-identity and that was
measured by script at the PR (every matrix handed to the solve, Y, the
coefficients and the knot currents, raw bits, on SG's wholly-buried, mixed and
crossing decks and on both families' above-ground decks). What these tests pin
is the STRUCTURE the refactor exists for, so a later edit cannot quietly undo
it:

  * the point-matched lane refuses a crossing junction BY NAME (it served
    no buried deck at stage 0; since #1222 it serves wholly-buried and
    detached ones);
  * each hoisted method is the base's, not re-grown on the subclass;
  * both Sommerfeld remainder families replay through ONE loop, and a solve
    really goes through it (counted, not assumed);
  * the transmitted tensor's source shape is an explicit choice.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import SinusoidalGalerkinSolver, SinusoidalSolver

C0 = 299792458.0
WL7 = C0 / 7e6
SOIL_A = (13.0, 0.005)
GROUND = dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="sommerfeld")

HOISTED = (
    "_somm_remainder_below_prepare",
    "_replay_somm_remainder_below",
    "_replay_remainder",
    "_lower_medium",
    "_grounded_junction_ends",
    "_wire_media",
    "_below_segments",
    "_medium_eta",
    "_fill_medium",
    "_operating_medium",
    "_is_mixed",
    "_class_geom",
    "_stitch_basis_coefs",
    "_mixed_serve_plan",
    "_transmitted_tensor",
)


def _buried_dipole(cls, n=11, depth=0.5):
    return cls(
        wires=[np.array([(-2.5, 0.0, -depth), (2.5, 0.0, -depth)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        **GROUND,
    )


def test_the_point_matched_lane_serves_buried_and_crossing():
    """Stage 0 kept the refusal; stage 1 (#1222) serves wholly-buried and
    detached decks, and stage 2 (#1223) the crossing JUNCTION: one two-member
    node per deck, with the multi-node and hub decks refused by name
    (`tests/test_sin_crossing_serve_1223.py` has the gates)."""
    assert SinusoidalSolver.capabilities.buried
    refusals = SinusoidalSolver.capabilities.refusals
    assert "buried+crossing_junction" not in refusals
    assert "buried+crossing_multi_node" in refusals
    assert "buried+crossing_hub" in refusals
    s = SinusoidalSolver(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
            np.array([(0.0, 0.0, 0.0), (2.0, 0.0, -0.5)]),
        ],
        n_per_edge_per_wire=[[5], [5]],
        feeds=[(0, 1.0, 1 + 0j)],
        junctions=[[(0, "start"), (1, "start")]],
        wavelength=WL7,
        wire_radius=1e-3,
        **GROUND,
    )
    assert np.isfinite(complex(s.compute_impedance()[0]))


@pytest.mark.parametrize("name", HOISTED)
def test_the_hoisted_method_is_the_base_class_one(name):
    assert name in vars(SinusoidalSolver), f"{name} is not on SinusoidalSolver"
    # By identity, not by `vars()`: a spy restored with a plain assignment
    # (`SG.x = orig`, the try/finally shape older tests use) leaves the BASE
    # function in the subclass's own dict, which is not a re-grown method.
    own = vars(SinusoidalGalerkinSolver).get(name, vars(SinusoidalSolver)[name])
    assert own is vars(SinusoidalSolver)[name], (
        f"{name} was re-grown on SinusoidalGalerkinSolver"
    )


def test_both_remainder_families_replay_through_one_loop(monkeypatch):
    """A wholly-buried SG solve reaches the BELOW replay, an above-ground
    point-matched solve the ABOVE one, and both arrive at `_replay_remainder`
    — counted per family, so a replay that grew its own loop again fails."""
    seen = []
    real = SinusoidalSolver._replay_remainder

    def spy(self, prepared, *a, **kw):
        seen.append("below" if "k_m" in prepared else "above")
        return real(self, prepared, *a, **kw)

    monkeypatch.setattr(SinusoidalSolver, "_replay_remainder", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _buried_dipole(SinusoidalGalerkinSolver).compute_impedance()
        SinusoidalSolver(
            wires=[np.array([(-2.5, 0.0, 1.0), (2.5, 0.0, 1.0)])],
            n_per_edge_per_wire=[[11]],
            feeds=[(0, 2.5, 1 + 0j)],
            wavelength=WL7,
            wire_radius=1e-3,
            **GROUND,
        ).compute_impedance()
    assert "below" in seen and "above" in seen, seen


def test_the_transmitted_tensor_names_its_source_shape():
    """No default: the third shape must be the one every other block of the
    same fill is written in (#606), so omitting it is a TypeError and a
    misspelling a ValueError — both before any grid is built."""
    s = _buried_dipole(SinusoidalGalerkinSolver)
    geom = s._build_geometry()
    args = (geom, None, None, None, None, False, None, None)
    with pytest.raises(TypeError):
        s._transmitted_tensor(*args)
    with pytest.raises(ValueError, match="cos_shape"):
        s._transmitted_tensor(*args, cos_shape="cos-2")
