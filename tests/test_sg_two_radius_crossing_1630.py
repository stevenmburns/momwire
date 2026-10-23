"""The TWO-RADIUS crossing node on the sinusoidal pair (momwire#1630).

`SinusoidalGalerkinSolver` serves a crossing deck whose above wires share one
radius and whose below wires share another, on bspline's rule (antennaknobs
plan U5, momwire#1140; `tests/test_crossing_two_radius_u5.py`), ported rather
than re-derived: the two-radius cross blocks
(`_crossing_fill.cross_complete_blocks_two_radius`, no longer transposes of
each other), the two-radius completions on the mixed-potential fill, and the
crossing node's continuity row on the solve (`_crossing_continuity_rows`, as
bspline keeps the junction's KCL row).

`SinusoidalSolver` (point-matched) refuses the node by MEASUREMENT, and its
`fill="direct"` sibling on SG does too. The numbers are in
`scratch/1630-two-radius/` and in the refusals' own comments.

The instrument is the U5 one: the RESPONSE of Z to a radius change, which
cancels the node convention's own gap. Two decks carry it here — check 2's
momwire-native rod (`crossing_deck(2)`, against bspline) and a synthetic
1-radial buried hub at 14 MHz (`tests/fixtures/sg_two_radius_1630/`, against
the licensed NEC-5 printouts committed beside it).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import _crossing_fill
from momwire import sinusoidal as sin_mod
from momwire import sinusoidal_galerkin as sg_mod
from momwire.bspline import BSplineSolver
from momwire.deck import build_solver, parse
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

from test_crossing_serve_524 import crossing_deck, fan_rise_deck

A = 0.25e-3
FIX = Path(__file__).resolve().parent / "fixtures" / "sg_two_radius_1630"
_NUM = r"[-+]?\d\.\d+E[-+]\d+"


def _z(out):
    return complex(out[0] if isinstance(out, tuple) else out)


def _rod(cls, a_above, a_below, **kw):
    return cls(**crossing_deck(2, wire_radius=[a_below, a_above]), **kw)


def _printout_z(stem):
    """The ANTENNA INPUT PARAMETERS row's impedance (both engines print the
    six E-format numbers V, I, Z in that order)."""
    text = (FIX / f"{stem}.out").read_text(errors="replace")
    i = text.index("ANTENNA INPUT PARAMETERS")
    for line in text[i:].splitlines()[1:12]:
        nums = re.findall(_NUM, line)
        if len(nums) >= 8:
            return complex(float(nums[4]), float(nums[5]))
    raise AssertionError(f"no impedance row in {stem}.out")


def _deck_z(stem, basis="sinusoidal-galerkin"):
    model = parse((FIX / f"{stem}.nec42.nec").read_text(), dialect="nec4")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _z(build_solver(model, basis=basis).solver.compute_impedance())


@pytest.fixture(scope="module")
def rod_responses():
    """R for the rod at (above, below) radii, SG and bspline."""
    cfg = {
        "equal": (A, A),
        "rise2": (A, A / 2),
        "rise4": (A, A / 4),
        "top2": (A / 2, A),
    }
    return {
        cls.__name__: {
            k: _z(_rod(cls, *ab).compute_impedance()) for k, ab in cfg.items()
        }
        for cls in (SinusoidalGalerkinSolver, BSplineSolver)
    }


def test_g1_equal_radii_take_the_shipped_path(monkeypatch):
    """One radius: the two-radius blocks and completions are never entered and
    no continuity row is added, exactly as before momwire#1630."""

    def entered(*_a, **_k):
        raise AssertionError("the two-radius fill was entered at one radius")

    monkeypatch.setattr(_crossing_fill, "cross_complete_blocks_two_radius", entered)
    monkeypatch.setattr(_crossing_fill, "self_completions_two_radius", entered)
    s = _rod(SinusoidalGalerkinSolver, A, A)
    assert s._two_radius_crossing() is None
    geom = s._build_geometry()
    assert s._crossing_continuity_rows(geom, None, s.k) is None
    assert np.isfinite(_z(s.compute_impedance()))


def test_g2_a_two_radius_node_is_served_with_its_continuity_closed():
    s = _rod(SinusoidalGalerkinSolver, A, A / 2)
    assert s._two_radius_crossing() == (A, A / 2)
    z, alpha = s.compute_impedance()
    assert np.isfinite(complex(z))
    # The row the solve imposed, rebuilt on the solve's own view
    # (`compute_impedance`'s spelling): the net inflow over the node's
    # members vanishes on the solution.
    geom = s._build_geometry()
    with s._operating_medium(geom) as medium:
        _G, seg_view = s._assemble_Z_ported(geom, s.k, s._medium_eta(medium))
        C = s._crossing_continuity_rows(geom, seg_view, s.k)
    assert C is not None and C.shape[0] == 1
    inflow = C @ np.asarray(alpha).reshape(-1)
    through = np.max(np.abs(np.asarray(alpha)))
    assert abs(inflow[0]) <= 1e-9 * through, (inflow, through)


def test_g3_the_z_guard_reads_the_physics(rod_responses):
    """bspline's U5 guard on SG: thinning the BURIED member raises R by
    several ohms, thinning the ABOVE member by the same factor barely moves
    it, and the rise's response is logarithmic in the radius ratio — and SG's
    responses are bspline's to 0.05 ohm (measured 0.005: +7.862 / +15.627 /
    +0.493 against +7.861 / +15.624 / +0.492)."""
    sg = rod_responses["SinusoidalGalerkinSolver"]
    bs = rod_responses["BSplineSolver"]
    r0 = sg["equal"].real
    rise2 = sg["rise2"].real - r0
    rise4 = sg["rise4"].real - r0
    top2 = sg["top2"].real - r0
    assert rise2 >= 3.0, rise2
    assert abs(top2) <= 0.25 * rise2, (top2, rise2)
    assert 1.7 <= rise4 / rise2 <= 2.3, (rise4, rise2)
    for k in ("rise2", "rise4", "top2"):
        d_sg = sg[k] - sg["equal"]
        d_bs = bs[k] - bs["equal"]
        assert abs(d_sg - d_bs) <= 0.05, (k, d_sg, d_bs)


@pytest.mark.parametrize("stem", ["hub1_v19mm", "hub1_b025mm"])
def test_g4_the_radius_response_tracks_nec5(stem):
    """The 1-radial hub (fixtures): SG's response to the radius change against
    the licensed NEC-5 printouts' — a 19 mm vertical, or 0.25 mm buried wires,
    each minus the 1 mm control. Measured at r = 2: -7.50+8.65j against
    NEC-5's -6.50+8.43j, and +20.37-7.43j against +19.88-6.95j; the feeds sit
    h/2 apart (segment centre against knot), which is most of what is left.
    The bound is 1.5 ohm, against responses of 11 and 21 ohm, and the
    point-matched lane's -54.2 / -2.6 under the rule it refuses."""
    d_sg = _deck_z(stem) - _deck_z("hub1_all1mm")
    d_n5 = _printout_z(f"{stem}.nec5") - _printout_z("hub1_all1mm.nec5")
    assert abs(d_sg - d_n5) <= 1.5, (d_sg, d_n5)


def test_g5_nec42_cannot_adjudicate_a_two_radius_node():
    """Why NEC-4.2 is not this node's reference (fixtures, numbers only): with
    one radius it agrees with NEC-5 to a few ohms, and a 10 % radius step at
    the crossing node moves its X by tens of ohms where NEC-5 — over the whole
    1 -> 19 mm range — moves it by 8."""
    one = _printout_z("hub1_all1mm.nec42")
    assert abs(one - _printout_z("hub1_all1mm.nec5")) <= 5.0
    step = _printout_z("hub1_v1p1mm.nec42") - one
    assert abs(step.imag) >= 40.0, step
    n5_whole = _printout_z("hub1_v19mm.nec5") - _printout_z("hub1_all1mm.nec5")
    assert abs(n5_whole.imag) <= 10.0, n5_whole


def test_g6_a_radius_spread_within_a_side_is_still_refused():
    s = SinusoidalGalerkinSolver(
        **fan_rise_deck(n_radials=2, wire_radius=[A / 2, A / 4, A])
    )
    with pytest.raises(NotImplementedError, match="differ within the below wires"):
        s.compute_impedance()


def test_g7_the_point_matched_lane_refuses_the_node_by_name():
    s = _rod(SinusoidalSolver, A, A / 2)
    with pytest.raises(NotImplementedError) as err:
        s.compute_impedance()
    assert str(err.value).endswith(sin_mod._CROSSING_TWO_RADIUS_REFUSAL)
    assert (
        SinusoidalSolver.capabilities.refusal("buried", "crossing_two_radius")
        == sin_mod._CROSSING_TWO_RADIUS_REFUSAL
    )
    # One radius stays served.
    assert np.isfinite(_z(_rod(SinusoidalSolver, A, A).compute_impedance()))


def test_g8_the_direct_fill_refuses_the_node_by_name():
    s = _rod(SinusoidalGalerkinSolver, A, A / 2, fill="direct")
    with pytest.raises(NotImplementedError) as err:
        s.compute_impedance()
    assert str(err.value) == sg_mod._CROSSING_TWO_RADIUS_DIRECT_FILL_REFUSAL
    assert (
        SinusoidalGalerkinSolver.capabilities.refusal(
            "buried", "crossing_two_radius_direct_fill"
        )
        == sg_mod._CROSSING_TWO_RADIUS_DIRECT_FILL_REFUSAL
    )


def test_g9_thin_legs_under_a_fat_mast_are_not_coincident():
    """The coincident-members check judges each pair by ITS radii: two 1 mm
    legs leaving the node 30 degrees apart under a 19 mm mast are 13 mm apart
    at their feet, not coincident — the deck's largest radius used to say
    they were. Two members on one path (0.5 mm apart) are still refused."""
    legs = []
    for phi in (0.0, np.pi / 6):
        c, sn = np.cos(phi), np.sin(phi)
        legs.append(np.array([(0.0, 0.0, 0.0), (0.0254 * c, 0.0254 * sn, -0.0254)]))
    mast = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)])
    build = dict(
        wires=[mast, *legs],
        n_per_edge_per_wire=[[10], [1], [1]],
        junctions=[[(0, "start"), (1, "start"), (2, "start")]],
        feeds=[(0, 1.25, 1 + 0j)],
        wavelength=299.792458 / 14.0,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    s = SinusoidalGalerkinSolver(**build, wire_radius=[0.019, 0.001, 0.001])
    assert s._crossing_junction_indices() == (0,)
    near = legs[0] + np.array([(0.0, 0.0, 0.0), (0.0, 0.0005, 0.0)])
    same = dict(build, wires=[mast, legs[0], near])
    with pytest.raises(NotImplementedError, match="COINCIDENT"):
        SinusoidalGalerkinSolver(
            **same, wire_radius=[0.019, 0.001, 0.001]
        )._crossing_junction_indices()


def test_g10_the_extended_kernel_at_a_two_radius_node_is_refused_by_name():
    """momwire#1626 x momwire#1630: the partial extended kernel and the
    two-radius node do not compose. The node is a radius step at a junction,
    which SG refuses under the extended kernel by name at construction
    (`extended_kernel+stepped_radius_junction`, measured divergent at
    momwire#398), and a NEC-4.2 deck, whose dialect defaults to the extended
    kernel, falls back to the reduced one with that sentence in its advisory:
    the hub with a 19 mm vertical reads exactly its reduced solve."""
    from momwire.deck import ExtendedKernelDefault

    with pytest.raises(NotImplementedError) as err:
        _rod(SinusoidalGalerkinSolver, A, A / 2, extended_kernel=True)
    assert str(err.value) == sg_mod._EK_STEPPED_RADIUS_JUNCTION_REFUSAL
    model = parse((FIX / "hub1_v19mm.nec42.nec").read_text(), dialect="nec4")
    with pytest.warns(ExtendedKernelDefault) as seen:
        built = build_solver(model, basis="sinusoidal-galerkin")
    assert built.extended_kernel is False
    assert any(
        sg_mod._EK_STEPPED_RADIUS_JUNCTION_REFUSAL in str(w.message) for w in seen
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_default = _z(built.solver.compute_impedance())
        z_reduced = _z(
            build_solver(
                model, basis="sinusoidal-galerkin", extended_kernel=False
            ).solver.compute_impedance()
        )
    assert z_default == z_reduced
