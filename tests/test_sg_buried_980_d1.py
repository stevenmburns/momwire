"""momwire#980 D1: the fully-buried serve on SinusoidalGalerkinSolver.

Every segment below the interface, so the deck has exactly ONE pair class and
the medium is a property of the solve rather than of a pair. Three terms, all
in the lower medium: the direct fill at k_m/eta_m through step E's complex-k
far fill, the image at k_m weighted A_m, and the below-family Sommerfeld
remainder.

The gates are ordered so that a failure localises. `eps_tilde -> 1` comes
first because the collapse falls out of the construction — k_m -> k_p,
A_m -> 0, remainder -> 0 — so a miss there is one of three terms rather than
"the buried fill is wrong somewhere".

Measured 2026-09-09, 1 m dipole, soil A, 7 MHz:

    eps_t -> 1 collapse onto free space            7.5e-15
    deep burial (1.5 m) vs the infinite medium     2.05e-04
    shallow    (0.15 m) vs the infinite medium     7.87e-03
    SG vs bspline on the golden ladder             <= 5.6e-06
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import (
    SinusoidalGalerkinSolver,
    _ground_refl,
    _medium_spec,
    _sommerfeld_below,
)

SOIL_A = (13.0, 0.005)
C0 = 299792458.0
WL7 = C0 / 7e6

# The NEC-5 rows for the depth ladder. 0.15 m is the banked golden
# (`golden_buried_currents_nec5`); 1 m and 2 m were captured 2026-09-09 on the
# same recipe — impedance only, no currents table — and the d=0.15 controls of
# that capture reproduced the banked values to the printed digit.
NEC5 = {
    ("bvd1", 0.15): 335.29 - 324.34j,
    ("bvd1", 1.0): 333.36 - 321.28j,
    ("bvd1", 2.0): 333.37 - 321.06j,
    ("bhd1", 0.15): 346.32 - 336.38j,
    ("bhd1", 1.0): 333.62 - 321.31j,
    ("bhd1", 2.0): 333.42 - 321.12j,
}


def sg_dipole(n=41, length=1.0, depth=0.15, vertical=True, eps=SOIL_A, free=False):
    """The phase-0 buried dipole on this trunk — `buried_dipole`'s twin."""
    pts = (
        np.array([(0.0, 0.0, -(depth + length)), (0.0, 0.0, -depth)])
        if vertical
        else np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    )
    fed = (n + 1) // 2
    arc = (fed - 0.5) / n * length
    ground = (
        {} if free else dict(ground_z=0.0, ground_eps=eps, ground_model="sommerfeld")
    )
    return SinusoidalGalerkinSolver(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, arc, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **ground,
    )


def z_of(solver):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(solver.compute_impedance()[0])


def infinite_medium(depth=1.5, n=41, vertical=True):
    """The step-A oracle: free-space geometry driven at k_m/eta_m."""
    s = sg_dipole(n=n, depth=depth, vertical=vertical, free=True)
    eps_t = _ground_refl.eps_tilde(SOIL_A, s.omega, s.eps)
    s.k = _sommerfeld_below.k_medium(eps_t, s.k)
    s.eta = np.sqrt(s.mu / (s.eps * eps_t))
    return z_of(s)


# ----------------------------------------------------------------------
# Gate 1 — eps_tilde -> 1 collapses onto free space
# ----------------------------------------------------------------------


def test_the_ground_block_actually_runs_in_the_collapse():
    """The collapse's precondition, asserted before the collapse is believed.

    At eps_tilde = 1 the image coefficient is exactly zero and the remainder
    vanishes, so a buried fill that never built a ground block AT ALL would
    pass the collapse gate below for the wrong reason — the most plausible
    way for this whole file to be vacuous.
    """
    seen = {"fold": 0, "prepare": 0, "replay": 0, "coef": []}
    fold = SinusoidalGalerkinSolver._fold_ground_block
    prep = SinusoidalGalerkinSolver._somm_remainder_below_prepare
    replay = SinusoidalGalerkinSolver._replay_somm_remainder_below

    def f(self, geom, k, ctx, contribs, fg):
        seen["fold"] += 1
        seen["coef"].append(complex(fg.image_coefficient))
        return fold(self, geom, k, ctx, contribs, fg)

    def p(self, *a, **kw):
        seen["prepare"] += 1
        return prep(self, *a, **kw)

    def r(self, *a, **kw):
        seen["replay"] += 1
        return replay(self, *a, **kw)

    SinusoidalGalerkinSolver._fold_ground_block = f
    SinusoidalGalerkinSolver._somm_remainder_below_prepare = p
    SinusoidalGalerkinSolver._replay_somm_remainder_below = r
    try:
        z_of(sg_dipole(depth=1.5, eps=(1.0, 0.0)))
    finally:
        SinusoidalGalerkinSolver._fold_ground_block = fold
        SinusoidalGalerkinSolver._somm_remainder_below_prepare = prep
        SinusoidalGalerkinSolver._replay_somm_remainder_below = replay
    assert seen["fold"] == 1, "no ground block was folded"
    assert seen["prepare"] == 1 and seen["replay"] == 1, "the below remainder never ran"
    assert seen["coef"] == [0j], (
        f"A_m should be exactly 0 at eps_t=1, got {seen['coef']}"
    )


def test_eps_tilde_one_collapses_onto_free_space():
    """k_m -> k_p, A_m -> 0, remainder -> 0, so the buried fill IS free space."""
    z_free = z_of(sg_dipole(depth=1.5, free=True))
    z_collapsed = z_of(sg_dipole(depth=1.5, eps=(1.0, 0.0)))
    rel = abs(z_collapsed - z_free) / abs(z_free)
    # bspline's own collapse floor on the elevated deck is 5e-5; this trunk
    # measured 7.5e-15, so the gate is set two decades above the measurement
    # rather than at the neighbour's number.
    assert rel < 1e-12, f"{rel:.3e}"


# ----------------------------------------------------------------------
# Gate 2 — deep burial IS the infinite-medium solve
# ----------------------------------------------------------------------


def test_deep_burial_approaches_the_infinite_medium_solve():
    """As the source sinks, image and remainder both go away.

    Stated as a RATIO between two depths rather than as one tolerance: the
    absolute gap at any single depth is a real physical interface term, and a
    gate on it would be pinning the interface rather than the limit.
    """
    z_inf = infinite_medium(depth=1.5)
    deep = abs(z_of(sg_dipole(depth=1.5)) - z_inf) / abs(z_inf)
    shallow = abs(z_of(sg_dipole(depth=0.15)) - z_inf) / abs(z_inf)
    assert deep < shallow, f"deep {deep:.3e} not closer than shallow {shallow:.3e}"
    assert deep < 1e-3, f"deep burial should be near the limit, got {deep:.3e}"
    # And the approach must be substantial, not a rounding difference.
    assert shallow / deep > 10.0, f"only {shallow / deep:.1f}x closer"


# ----------------------------------------------------------------------
# Gate 3 — the depth ladder, each solver against its own convergence
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("deck", "depth"), sorted(NEC5))
def test_depth_ladder_against_bspline_and_nec5(deck, depth):
    """SG vs bspline at the cross-basis floor, and the SAME offset band
    against NEC-5 that bspline sits in.

    NOT "SG within 0.2 % of NEC-5": at N=11 both momwire bases carry a
    constant ~1.75 % mesh/basis offset from NEC-5 that is not an interface
    term (it is flat from 0.15 m to 2 m), so a tight absolute gate would be
    pinning the mesh, and the "bspline tracks NEC-5 to 0.14-0.18 %" figure
    does not reproduce at this N.
    """
    from test_buried_serve_553 import buried_dipole

    vertical = deck == "bvd1"
    z_sg = z_of(sg_dipole(n=11, depth=depth, vertical=vertical))
    b, _fed = buried_dipole(n=11, depth=depth, vertical=vertical)
    z_bs = z_of(b)
    z_n5 = NEC5[(deck, depth)]

    # The two bases must agree far better than either agrees with NEC-5.
    cross = abs(z_sg - z_bs) / abs(z_bs)
    assert cross < 1e-4, f"SG vs bspline {cross:.3e}"

    off_sg = (abs(z_sg) - abs(z_n5)) / abs(z_n5)
    off_bs = (abs(z_bs) - abs(z_n5)) / abs(z_n5)
    assert abs(off_sg - off_bs) < 1e-3, (
        f"SG's offset from NEC-5 ({off_sg:.4%}) differs from bspline's "
        f"({off_bs:.4%}) by more than the cross-basis floor"
    )


def test_the_shallow_horizontal_row_is_its_own_row():
    """`bhd1` at 0.15 m sits 1.94 % off NEC-5 where every other row sits at
    1.746 +- 0.005 %, and that is the row with by far the largest interface
    term (-4.1 % from 0.15 m to 1 m, against -0.75 % vertical).

    Pinned separately so a band wide enough to admit it cannot hide a real
    interface error on the deck that would show one first — which is the term
    this serve's below-remainder owns.
    """
    from test_buried_serve_553 import buried_dipole

    z_sg = z_of(sg_dipole(n=11, depth=0.15, vertical=False))
    b, _fed = buried_dipole(n=11, depth=0.15, vertical=False)
    off_sg = (abs(z_sg) - abs(NEC5[("bhd1", 0.15)])) / abs(NEC5[("bhd1", 0.15)])
    off_bs = (abs(z_of(b)) - abs(NEC5[("bhd1", 0.15)])) / abs(NEC5[("bhd1", 0.15)])
    assert 0.018 < off_sg < 0.021, f"{off_sg:.4%} left the recorded row"
    assert abs(off_sg - off_bs) < 1e-3


# ----------------------------------------------------------------------
# Gate 4 — the complex twin is what fills this, and the scope refusals
# ----------------------------------------------------------------------


def test_the_complex_twin_is_the_buried_fill():
    """Step E's gate, carried forward.

    A fill that silently routed the medium to numpy would pass every
    agreement gate above, because numpy is the oracle those gates compare to.
    """
    seen = {"n": 0, "cplx": 0}
    real = SinusoidalGalerkinSolver._far_fill_accel

    def spy(self, k, ctx, src_c, src_t, **kw):
        seen["n"] += 1
        if np.iscomplexobj(k) or np.iscomplexobj(self.eta):
            seen["cplx"] += 1
        return real(self, k, ctx, src_c, src_t, **kw)

    SinusoidalGalerkinSolver._far_fill_accel = spy
    try:
        z_of(sg_dipole(n=161, depth=1.5))
    finally:
        SinusoidalGalerkinSolver._far_fill_accel = real
    assert seen["n"] > 0, "the accelerated far fill never ran"
    assert seen["cplx"] == seen["n"], "a buried fill ran at a REAL k"


def test_the_operating_point_is_scoped_and_restores():
    s = sg_dipole(depth=1.5)
    geom = s._build_geometry()
    k_before, eta_before = s.k, s.eta
    with s._operating_medium(geom) as medium:
        assert np.iscomplexobj(s.k) and s.k.imag <= 0.0
        assert s.k == medium.k_m
        assert s.eta == np.sqrt(s.mu / medium.eps_m)
        assert medium.a_m == -medium.c2  # measured, not derived (phase 0)
    assert s.k == k_before and s.eta == eta_before


def test_a_mixed_deck_is_refused_by_name():
    """Above AND below is D2's three pair classes, not this serve's one."""
    s = SinusoidalGalerkinSolver(
        wires=[
            np.array([(0.0, 0.0, -2.0), (0.0, 0.0, -1.0)]),
            np.array([(5.0, 0.0, 1.0), (5.0, 0.0, 2.0)]),
        ],
        n_per_edge_per_wire=[[11], [11]],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    with pytest.raises(NotImplementedError, match="BOTH above and below"):
        s.compute_impedance()


@pytest.mark.parametrize(
    ("ground", "expected"),
    [
        (dict(ground_z=0.0), _medium_spec.BURIED_PEC_REFUSAL),
        (
            dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="refl-coef"),
            _medium_spec.BURIED_REFL_REFUSAL,
        ),
    ],
)
def test_a_ground_with_no_lower_medium_refuses_with_its_own_reason(ground, expected):
    """The reason is "this ground has no half-space", not "this family has no
    in-medium kernel" — which stopped being true at D1.

    Lifting `_build_geometry`'s blanket refusal is what makes these decks
    reach `_medium_spec`, where the accurate sentence lives.
    """
    s = SinusoidalGalerkinSolver(
        wires=[np.array([(0.0, 0.0, -1.15), (0.0, 0.0, -0.15)])],
        n_per_edge_per_wire=[[11]],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **ground,
    )
    with pytest.raises(ValueError) as exc:
        s.compute_impedance()
    assert str(exc.value).endswith(expected)


def test_the_extended_kernel_is_refused_by_bsplines_own_name():
    """Same deck, same sentence, whichever trunk a user reaches it through."""
    from momwire._below_interface import BURIED_EXTENDED_KERNEL_REFUSAL

    s = sg_dipole(n=11, depth=1.5)
    s.extended_kernel = True
    with pytest.raises(NotImplementedError) as exc:
        s.compute_impedance()
    assert str(exc.value) == BURIED_EXTENDED_KERNEL_REFUSAL
