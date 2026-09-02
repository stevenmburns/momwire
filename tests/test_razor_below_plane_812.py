"""RazorSolver's below-plane fill — momwire#812, unit 1 of the razor buried arc.

A deck whose wires lie WHOLLY below the interface is filled in the lower-
medium family: the razor-blade fill at `k_m = k₂·√ε̃`, Φ under `ε_m = ε₀·ε̃`,
the image weighted by `A_m = image_coefficient_below(ε̃)` through the windows,
and the below remainder in place of the above one. It is the composing
ground's fold with the medium's numbers in it, which is why the whole unit is
one ground object (`_potential_ground.BelowMediumGround`, whose `remainder()`
is `RemainderBelow`) and one assembly method that calls
`_assemble_Z_source_block` twice with `eps` and the ground swapped.

It lands BEHIND the public refusal: razor's `buried` capability cell stays
False until unit 3 (momwire#814) flips it, and every test here reaches the
fill through the module switch `_SERVE_BELOW_PLANE`. Mixed above/below decks
are unit 2's (momwire#813).

What this module gates, measured 2026-09-02 on this box:

  * **the ε̃ = 1 collapse is exact.** `A_m = 0` kills the image, the below
    surfaces vanish and `k_m = k₂` leaves the direct block, so a wholly-below
    wire at ε̃ = 1 IS the free-space wire: 2.2e-20 (vertical) and 1.1e-19
    (horizontal) relative — the same machine-precision claim
    `test_gu5_3_a_fully_buried_deck_collapses_to_free_space` makes for
    bspline.
  * **at soil the two formulations converge together.** Razor and bspline
    on the same buried dipole differ by the razor-vs-Galerkin walk and
    nothing else: the gap falls monotonically down the mesh ladder,
    8.83 → 6.10 → 3.93 → 2.58 Ω for N = 11 → 81 vertical (1.86e-2 → 5.7e-3
    relative), the same shape horizontal. Not equality at one mesh, which
    would be the wrong claim between a path-tested and a Galerkin-tested
    basis (razor.py's module docstring on the O(1/N) walk).
  * **both lanes serve it, and the fused path is the numpy path.** The
    two-point (NEC-5) and Gauss-Legendre lanes sit 0.01 Ω apart at N=41; the
    fused moments kernel at complex k (momwire#796) plus the fused weighted
    assembly through #806's `CONSTANT_MIRROR` window rule agree with the
    numpy closure to 4e-16 / 5e-15, with no kernel work — the rule seam was
    built for exactly this.
  * **nothing above the plane moves.** The #762 protocol on free-space, PEC,
    EK-over-PEC and refl-coef decks: bit-identical across this change (run
    by hand for the PR; here the structural half — the default refusal is
    unchanged and the switch is off).
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import BSplineSolver, RazorSolver, _potential_ground, _sommerfeld_below
from momwire import razor as _razor

C0 = 299792458.0
WL7 = C0 / 7.0e6
SOIL_A = (13.0, 0.005)


def deck(n, *, eps=SOIL_A, free=False, depth=0.15, length=1.0, vertical=True, **kw):
    """The phase-0 buried dipole (`test_buried_serve_553.buried_dipole`),
    as kwargs so either solver can take it."""
    if vertical:
        pts = np.array([(0.0, 0.0, -(depth + length)), (0.0, 0.0, -depth)])
    else:
        pts = np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    fed = (n + 1) // 2
    arc = (fed - 0.5) / n * length
    ground = (
        {} if free else dict(ground_z=0.0, ground_eps=eps, ground_model="sommerfeld")
    )
    return dict(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, arc, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **ground,
        **kw,
    )


@pytest.fixture
def serve_below(monkeypatch):
    monkeypatch.setattr(_razor, "_SERVE_BELOW_PLANE", True)


def _zin(cls, **kw):
    z, _ = cls(**kw).compute_impedance()
    return complex(z)


# ----------------------------------------------------------------------
# the refusal, and the unit's own boundary
# ----------------------------------------------------------------------


def test_the_public_refusal_is_unchanged():
    """Off by default: the capability cell is still False (unit 3 flips it),
    and the sentence is the one the matrix carries."""
    with pytest.raises(ValueError, match="wholly below the ground plane"):
        RazorSolver(**deck(11))


def test_a_mixed_deck_is_unit_two(serve_below):
    """An above wire beside a below one is the crossing block on razor rows
    (momwire#813), refused by name here rather than filled wrong."""
    kw = deck(11)
    kw["wires"] = [kw["wires"][0], np.array([(0.0, 0.0, 0.5), (0.0, 0.0, 1.5)])]
    kw["n_per_edge_per_wire"] = [[11], [11]]
    with pytest.raises(ValueError, match="momwire#813"):
        RazorSolver(**kw)


def test_the_extended_kernel_is_declined_below_the_plane(serve_below):
    with pytest.raises(ValueError, match="extended kernel"):
        RazorSolver(**deck(11), extended_kernel=True)


# ----------------------------------------------------------------------
# the ground object
# ----------------------------------------------------------------------


def test_the_below_medium_ground_is_the_composing_ground_with_the_mediums_numbers(
    serve_below,
):
    s = RazorSolver(**deck(11))
    geom = s._build_geometry()
    eps_t = complex(13.0, -0.005 / (s.omega * s.eps))
    g = _potential_ground.BelowMediumGround(s, geom, s.k, s.omega, eps_tilde=eps_t)
    assert g.mode == "compose" and g.below is True
    assert g.image_coefficient == _sommerfeld_below.image_coefficient_below(eps_t)
    # A_m is the NEGATIVE of C₂ — measured, not derived (`image_coefficient_below`).
    assert g.image_coefficient == -((eps_t - 1.0) / (eps_t + 1.0))
    assert g.k_m == _sommerfeld_below.k_medium(eps_t, s.k) and g.k_m.imag <= 0.0
    assert g.eps_m == s.eps * eps_t
    rule = g.fused_window_rule()
    assert rule.kind == _potential_ground.WINDOW_RULE_CONSTANT_MIRROR
    assert rule.coefficient == g.image_coefficient
    assert isinstance(g.remainder(), _potential_ground.RemainderBelow)
    # The base class is not below, and says so.
    assert _potential_ground.PotentialGround.below is False


# ----------------------------------------------------------------------
# the ε̃ = 1 collapse — exact
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vertical", [True, False])
def test_a_wholly_below_deck_collapses_to_free_space_at_eps_one(serve_below, vertical):
    zf = _zin(RazorSolver, **deck(11, free=True, vertical=vertical))
    z1 = _zin(RazorSolver, **deck(11, eps=(1.0, 0.0), vertical=vertical))
    rel = abs(z1 - zf) / abs(zf)
    assert rel < 1e-12, f"eps_t -> 1 did not collapse: {rel:.3e}"


# ----------------------------------------------------------------------
# soil: the two formulations converge together
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vertical", [True, False])
def test_razor_and_bspline_converge_together_at_soil(
    serve_below, vertical, record_property
):
    ladder = (11, 21, 41, 81)
    gaps = []
    for n in ladder:
        zr = _zin(RazorSolver, **deck(n, vertical=vertical))
        zb = _zin(BSplineSolver, **deck(n, vertical=vertical))
        gaps.append(abs(zr - zb))
        record_property(f"{'v' if vertical else 'h'}_N{n}", f"{zr:.4f} vs {zb:.4f}")
    # Monotone down the ladder, and by more than a third over 8x in N
    # (measured 0.29 vertical, 0.29 horizontal). The walk is O(1/N)-class,
    # so an equality bar at one mesh would be the wrong claim.
    assert all(b < a for a, b in zip(gaps, gaps[1:])), gaps
    assert gaps[-1] / gaps[0] < 0.4, gaps
    assert gaps[-1] / abs(zb) < 1e-2, gaps


# ----------------------------------------------------------------------
# both lanes, and the fused path is the numpy path
# ----------------------------------------------------------------------


def test_the_two_quadrature_lanes_agree_to_the_lane_gap(serve_below):
    z2 = _zin(RazorSolver, **deck(41), nec5_quadrature=True)
    zg = _zin(RazorSolver, **deck(41), nec5_quadrature=False)
    assert abs(z2 - zg) < 0.1, (z2, zg)  # measured 0.0105 Ω


@pytest.mark.parametrize("lane", [False, True])
def test_the_fused_path_agrees_with_numpy_at_complex_k(serve_below, monkeypatch, lane):
    """The moments kernel at complex k (#796) and the weighted assembler
    through the CONSTANT_MIRROR rule (#806) against the numpy closure, on
    the below family. Not bitwise; the reduction orders differ (#742)."""
    if not (_razor._HAVE_RAZOR_FILL_ACCEL and _razor._HAVE_RAZOR_CPLX_ACCEL):
        pytest.skip("razor accelerators not built")
    monkeypatch.setattr(_razor, "_FORCE_NUMPY", True)
    zn = _zin(RazorSolver, **deck(41), nec5_quadrature=lane)
    monkeypatch.setattr(_razor, "_FORCE_NUMPY", False)
    za = _zin(RazorSolver, **deck(41), nec5_quadrature=lane)
    rel = abs(zn - za) / abs(zn)
    assert rel < 1e-11, f"lane nec5={lane}: {rel:.3e}"  # measured 4e-16 / 5e-15
