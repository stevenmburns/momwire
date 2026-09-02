"""The outer testing-path order, derived from the mesh — momwire#800.

`RazorSolver.n_qp_path` was the constant 32. #754's re-derivation (PR #795)
kept it, and what that sweep actually found is that 32 is the COARSE-mesh
answer: on the binding deck — a 90° corner — the order the outer integral
needs falls as the mesh refines, and every rung #754's own timing table
quotes sits where half of it would do.

What this module gates:

  * **the calibration, row by row.** The derivation is a pure function, so
    it is checked directly against the #795 table rather than through a
    solve: every N ≤ 120 row of the corner takes 32 and every N ≥ 240 row
    takes 16, and the easy decks — which converge at q ≈ 3–8 — are served
    rather than driving the threshold.

  * **that the threshold is not tight.** The corner's last coarse row and
    its first fine one are a factor of two apart in `k·h`, and the switch
    sits at their geometric mean. Both margins are asserted, so a later
    nudge to `PATH_ORDER_KH_SWITCH` that eats one of them fails here rather
    than on someone's deck.

  * **that an explicit integer is untouched.** The #762 standard: a caller
    passing `n_qp_path=32` gets exactly what it got before #800, and that is
    checked on the assembled Z rather than on the attribute.

  * **that `nec5_quadrature` ignores it**, which is the pre-existing contract
    and the one place the knob has never applied.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import RazorSolver
from momwire.razor import (
    PATH_ORDER_COARSE,
    PATH_ORDER_FINE,
    PATH_ORDER_KH_SWITCH,
    derive_n_qp_path,
)

LAMBDA = 22.0
L_DIPOLE = 0.962 * LAMBDA / 2
A_THIN = 5e-4


def _bent(n):
    arm = L_DIPOLE / 2
    return dict(
        wires=[[(0.0, 0.0, -arm), (0.0, 0.0, 0.0), (arm, 0.0, 0.0)]],
        n_per_edge_per_wire=[[n // 2, n // 2]],
        nsegs=n,
        wire_radius=A_THIN,
        feed_arclength=0.25 * L_DIPOLE,
    )


def _straight(n):
    h = L_DIPOLE / 2
    return dict(
        wires=[[(0.0, 0.0, -h), (0.0, 0.0, h)]],
        nsegs=n,
        wire_radius=A_THIN,
        feed_arclength=0.25 * L_DIPOLE,
    )


def _contact(n):
    return dict(
        wires=[[(0.0, 0.0, 0.0), (0.0, 0.0, L_DIPOLE / 2)]],
        nsegs=n,
        wire_radius=A_THIN,
        feed_arclength=0.1 * (L_DIPOLE / 2),
        ground_z=0.0,
    )


def _ek(n):
    h = L_DIPOLE / 2
    return dict(
        wires=[[(0.0, 0.0, -h), (0.0, 0.0, h)]],
        nsegs=n,
        wire_radius=L_DIPOLE / n / 4.0,
        feed_arclength=0.25 * L_DIPOLE,
        extended_kernel=True,
    )


# The #795 table's verdict per row: what the corner NEEDS, and what the
# other classes need, by #754's own 2x-margin rule against a q=128
# reference. The corner is the only deck that ever asks for 32.
CALIBRATION = [
    # deck fn,   N,    required by #795,  what derive must return
    (_bent, 30, 32, PATH_ORDER_COARSE),
    (_bent, 60, 32, PATH_ORDER_COARSE),
    (_bent, 120, 32, PATH_ORDER_COARSE),
    (_bent, 240, 16, PATH_ORDER_FINE),
    (_bent, 400, 16, PATH_ORDER_FINE),
    (_straight, 30, 16, PATH_ORDER_COARSE),
    (_straight, 120, 8, PATH_ORDER_COARSE),
    (_straight, 240, 8, PATH_ORDER_FINE),
    (_straight, 400, 8, PATH_ORDER_FINE),
    (_contact, 30, 8, PATH_ORDER_COARSE),
    (_contact, 120, 8, PATH_ORDER_FINE),
    (_contact, 240, 8, PATH_ORDER_FINE),
    (_ek, 30, 8, PATH_ORDER_COARSE),
    (_ek, 400, 8, PATH_ORDER_FINE),
]


@pytest.mark.parametrize(
    "fn,n,required,expected",
    CALIBRATION,
    ids=[f"{fn.__name__.strip('_')}-{n}" for fn, n, _, _ in CALIBRATION],
)
def test_the_derived_order_serves_every_calibration_row(fn, n, required, expected):
    """Derived ≥ required, and equal to the table's expectation.

    Two assertions rather than one on purpose. The second pins the rule; the
    first pins the thing the rule is FOR — an order below what #795 measured
    is a wrong answer, not a different tuning, and it should fail as that.
    """
    s = RazorSolver(wavelength=LAMBDA, **fn(n))
    assert s.n_qp_path >= required, f"{fn.__name__} N={n}: under-served"
    assert s.n_qp_path == expected


def test_the_switch_keeps_its_margin_on_both_sides():
    """The corner's own gap is a factor of two; the switch is its geometric
    mean, so neither side is tight. A later nudge that eats a margin fails
    here rather than in the field."""
    coarse = RazorSolver(wavelength=LAMBDA, **_bent(120))
    fine = RazorSolver(wavelength=LAMBDA, **_bent(240))
    kh_coarse = coarse.k * max(np.asarray(coarse._build_geometry()["seg_h"]))
    kh_fine = fine.k * max(np.asarray(fine._build_geometry()["seg_h"]))
    assert kh_fine < PATH_ORDER_KH_SWITCH < kh_coarse
    assert kh_coarse / PATH_ORDER_KH_SWITCH > 1.25
    assert PATH_ORDER_KH_SWITCH / kh_fine > 1.25


def test_the_derivation_is_a_function_of_geometry_not_of_segment_count():
    """Two decks at the same electrical segment length get the same order
    whatever their N. `contact`'s arm is half `straight`'s, so its N=240
    sorts with `straight`'s N=120 — by k·h, not by the number in the deck."""
    a = derive_n_qp_path(
        2 * np.pi / LAMBDA, [np.array([[0.0, 0, 0], [1.0, 0, 0]])], [[10]]
    )
    b = derive_n_qp_path(
        2 * np.pi / LAMBDA, [np.array([[0.0, 0, 0], [2.0, 0, 0]])], [[20]]
    )
    assert a == b


def test_an_explicit_order_is_what_it_always_was():
    """The #762 standard, on the assembled Z rather than on the attribute:
    a caller that passes an integer must not be able to tell #800 happened.
    """
    kw = _bent(60)
    s = RazorSolver(wavelength=LAMBDA, n_qp_path=32, **kw)
    assert s.n_qp_path == 32
    z_explicit = s._assemble_Z(s._build_geometry(), s.k)
    # The derived answer at this mesh is also 32, so the two must agree to
    # the BIT — this is the branch that proves the derivation returns the
    # same code path and not merely the same number.
    d = RazorSolver(wavelength=LAMBDA, **kw)
    assert d.n_qp_path == 32
    z_derived = d._assemble_Z(d._build_geometry(), d.k)
    assert np.array_equal(z_explicit, z_derived)


def test_an_explicit_order_overrides_the_derivation_where_they_differ():
    """At N=240 derive says 16; an explicit 32 must still be 32, and must
    give the pre-#800 answer."""
    kw = _bent(240)
    assert RazorSolver(wavelength=LAMBDA, **kw).n_qp_path == PATH_ORDER_FINE
    assert RazorSolver(wavelength=LAMBDA, n_qp_path=32, **kw).n_qp_path == 32


def test_nec5_quadrature_ignores_the_knob_derived_or_not():
    """The knob has never applied under NEC-5's identified rule, and #800
    does not change that: one node per wing either way."""
    kw = _bent(240)
    a = RazorSolver(wavelength=LAMBDA, nec5_quadrature=True, **kw)
    b = RazorSolver(wavelength=LAMBDA, nec5_quadrature=True, n_qp_path=32, **kw)
    assert a._path_nodes_per_wing() == b._path_nodes_per_wing() == 1
    za = a._assemble_Z(a._build_geometry(), a.k)
    zb = b._assemble_Z(b._build_geometry(), b.k)
    assert np.array_equal(za, zb)


def test_a_bad_explicit_order_is_still_refused():
    with pytest.raises(ValueError, match="must be >= 1"):
        RazorSolver(wavelength=LAMBDA, n_qp_path=0, **_straight(30))
