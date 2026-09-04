"""The other half of the coated-wire pair: the equivalent radius (#865).

A dielectric-jacketed wire is TWO things at once in the quasi-static reading
(Popovic & Nesic, IEE Proc. 131 pt. H no. 3, 153-158, 1984): the kernel sees a
larger radius

    a' = a * (b/a) ** ((eps_r - 1) / eps_r),        a < a' < b

and a distributed series inductance L = (mu0/2pi) * ln(a'/a) puts back what
enlarging the radius would otherwise remove. momwire has carried L since
momwire#131 — King's `(1 - 1/eps_r) * ln(b/a)` form, which is the SAME number
by the identity `ln(a'/a) = ((eps_r - 1)/eps_r) * ln(b/a)` — and has never
carried the radius half.

## Why it matters, and the sign

The two halves do different jobs, and momwire had the wrong one for the
problem in front of it. Measured on the Severns surface deck at N = 16,
h = 1.6 mm, against a measurement of 56.1 + 6.2j:

    bare a           46.62 + 17.96j     dR -9.48   dX +11.76
    a + L (before)   49.33 + 18.83j     dR -6.77   dX +12.63
    a' + L (now)     50.76 + 14.76j     dR -5.34   dX  +8.56

**L raises R the right way and pushes X the WRONG way; the equivalent radius
pulls X down.** The surface-radial residual is R-low / X-high (momwire#865),
so the missing half was the one that fixes the half the inductance cannot.

That is also why the earlier reading of this — that momwire's jacket model had
the wrong sign outright — was wrong twice over: the measurement behind it had
jacketed the deck's bare aluminium MAST (a scalar `insulation_radius` covers
every wire), and once that was removed the model was directionally right on R
and merely missing its partner on X.

## What keeps the CONDUCTOR's radius

`a'` is a kernel radius and nothing else. The metal keeps `a`:

* skin-effect internal impedance (`wire_conductivity`) — a fact about the
  conductor, not about the charge distribution around it;
* the jacket's own L, which is defined as `ln(b/a)` and would be a different
  number evaluated at `a'`;
* the low-stand-off validity floor (momwire#865), whose h/a >= 2 came from
  MESH stability on a bare deck. Measuring the stand-off against a' would
  tighten the floor for coated wires on the strength of a quasi-static charge
  radius, which is not what that evidence measured.

So the spec layer carries both: `_radius_per_wire` is the kernel radius and
`_conductor_radius_per_wire` is the metal's. Put in `configure_loading` rather
than in four constructors so the formulations cannot disagree about what a
coated wire's radius means.
"""

import numpy as np
import pytest

from momwire._wire_loading import equivalent_radius, insulation_inductance
from momwire.bspline import BSplineSolver

MU0 = 4.0e-7 * np.pi
A = 0.51e-3
B = 0.9e-3
EPS_R = 3.0


def _tiny(*, ins=None, eps=None, radius=A, cond=None):
    kw = {}
    if ins is not None:
        kw = dict(insulation_radius=ins, insulation_eps_r=eps)
    if cond is not None:
        kw["wire_conductivity"] = cond
    return BSplineSolver(
        wires=[np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 2.0)])],
        nsegs=11,
        wavelength=41.64,
        wire_radius=radius,
        feed_wire_index=0,
        feed_arclength=2.0,
        **kw,
    )


# ---------------------------------------------------------------------------
# The algebra
# ---------------------------------------------------------------------------


def test_the_equivalent_radius_is_between_the_conductor_and_the_jacket():
    """`a < a' < b` is the whole shape of the formula, and it is what makes it
    a radius rather than a fudge: the charge sees more than the metal and less
    than the jacket's outside."""
    a_eq = equivalent_radius(A, B, EPS_R)
    assert A < a_eq < B
    assert a_eq == pytest.approx(A * (B / A) ** ((EPS_R - 1.0) / EPS_R))


def test_the_inductance_is_the_same_number_written_two_ways():
    """King's `(1 - 1/eps_r)*ln(b/a)` and Popovic's `(mu0/2pi)*ln(a'/a)` are
    one identity, which is how we know momwire already had this half."""
    a_eq = equivalent_radius(A, B, EPS_R)
    assert insulation_inductance(A, B, EPS_R) == pytest.approx(
        MU0 / (2.0 * np.pi) * np.log(a_eq / A), rel=1e-14
    )


@pytest.mark.parametrize("eps", [1.0, 1.5, 3.0, 10.0, 80.0])
def test_a_perfect_dielectric_of_eps_1_is_no_jacket_at_all(eps):
    """eps_r = 1 is vacuum: a' must collapse to a, and the exponent must carry
    that. A formula that only worked at eps_r = 3 would pass everything else
    in this file."""
    a_eq = equivalent_radius(A, B, eps)
    if eps == 1.0:
        assert a_eq == pytest.approx(A)
    else:
        assert A < a_eq < B
    # Monotone in eps_r: more dielectric, more of the jacket the charge sees.
    assert a_eq <= equivalent_radius(A, B, eps * 1.5) + 1e-18


# ---------------------------------------------------------------------------
# The split: what the kernel sees vs what the metal keeps
# ---------------------------------------------------------------------------


def test_the_kernel_takes_a_prime_and_the_conductor_keeps_a():
    s = _tiny(ins=B, eps=EPS_R)
    assert s._radius_per_wire[0] == pytest.approx(equivalent_radius(A, B, EPS_R))
    assert s._conductor_radius_per_wire[0] == pytest.approx(A)


def test_an_uncoated_deck_is_untouched():
    """The change must be invisible without a jacket — every existing deck."""
    s = _tiny()
    assert s._radius_per_wire[0] == pytest.approx(A)
    assert s._conductor_radius_per_wire[0] == pytest.approx(A)
    assert s._uniform_radius == pytest.approx(A)


def test_a_jacket_on_some_wires_drops_the_uniform_fast_path():
    """The scalar fast path hands ONE radius to the single-`a` C++ kernels. A
    jacket on some wires breaks the uniformity even though the conductors
    agreed, and a stale `_uniform_radius` would quietly mis-fill the deck."""
    s = BSplineSolver(
        wires=[
            np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 2.0)]),
            np.array([(5.0, 0.0, -2.0), (5.0, 0.0, 2.0)]),
        ],
        n_per_edge_per_wire=[[8], [8]],
        wavelength=41.64,
        wire_radius=A,
        feeds=[(0, 2.0, 1 + 0j)],
        insulation_radius=[B, np.nan],
        insulation_eps_r=[EPS_R, np.nan],
    )
    assert s._uniform_radius is None
    assert s._radius_per_wire[0] > s._radius_per_wire[1]
    assert s._conductor_radius_per_wire[0] == pytest.approx(
        s._conductor_radius_per_wire[1]
    )


def test_the_loading_is_evaluated_at_the_metal_not_the_jacket():
    """Both loading terms must read the CONDUCTOR radius.

    Checked against the closed forms rather than by eyeballing an impedance:
    the skin-effect term is a fact about the metal, and the jacket's own L is
    defined as ln(b/a) — evaluated at a' it would be ln(b/a'), a different and
    smaller number. Reading `_radius_per_wire` here (the kernel radius) is the
    single most likely way to get this wrong, so it is pinned directly.
    """
    from momwire._wire_loading import series_impedance_per_wire, wire_internal_impedance

    omega = 2.0 * np.pi * 7.2e6
    s = _tiny(ins=B, eps=EPS_R, cond=5.8e7)
    got = series_impedance_per_wire(
        omega,
        s._conductor_radius_per_wire,
        s.wire_conductivity,
        s.insulation_radius,
        s.insulation_eps_r,
    )[0]
    expected = wire_internal_impedance(omega, A, 5.8e7) + 1j * omega * (
        insulation_inductance(A, B, EPS_R)
    )
    assert got == pytest.approx(expected, rel=1e-12)

    # ...and the same read at the KERNEL radius is a different number, which
    # is what makes the assertion above load-bearing rather than a tautology.
    at_kernel = series_impedance_per_wire(
        omega,
        s._radius_per_wire,
        s.wire_conductivity,
        s.insulation_radius,
        s.insulation_eps_r,
    )[0]
    assert abs(at_kernel - got) > 1e-3 * abs(got)


def test_the_bare_limit_reproduces_the_uncoated_deck():
    """b -> a is no jacket: a' -> a and L -> 0, so the coated deck must
    collapse onto the bare one. This is the gate that a future re-derivation
    of `equivalent_radius` cannot quietly move."""
    bare = _tiny().compute_impedance()[0]
    limit = _tiny(ins=A * (1.0 + 1e-12), eps=EPS_R).compute_impedance()[0]
    assert abs(limit - bare) < 1e-6, (limit, bare)


# ---------------------------------------------------------------------------
# The sign, on the deck the change is for
# ---------------------------------------------------------------------------

FT = 0.3048
SEVERNS_MEASURED_N16 = 56.1 + 6.2j


def _severns(n, h, *, jacket, equivalent):
    """The Severns surface deck. `equivalent=False` emulates the pre-#865
    model by spelling the jacket's L by hand at the conductor radius while
    holding the kernel radius at a."""
    ang = 2.0 * np.pi * np.arange(n) / n
    wires = [
        np.array([(33.0 * FT * np.cos(t), 33.0 * FT * np.sin(t), h), (0.0, 0.0, h)])
        for t in ang
    ]
    n_per_edge = [[10] for _ in ang]
    mast = len(wires)
    wires.append(
        np.array([(0.0, 0.0, z) for z in (33.5 * FT + h, 0.5 + h, 0.05 + h, h)])
    )
    n_per_edge.append([19, 2, 3])
    kw = {}
    if jacket:
        kw = dict(
            insulation_radius=[B] * n + [np.nan],
            insulation_eps_r=[EPS_R] * n + [np.nan],
        )
    solver = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        junctions=[[(i, "end") for i in range(n)] + [(mast, "end")]],
        feeds=[(mast, 33.5 * FT - 0.05, 1 + 0j)],
        wavelength=299792458.0 / 7.2e6,
        wire_radius=A,
        ground_z=0.0,
        ground_eps=(30.0, 0.020),
        ground_model="sommerfeld",
        **kw,
    )
    if jacket and not equivalent:
        # Put the kernel radius back to the metal: the pre-#865 model.
        solver._radius_per_wire = np.array(solver._conductor_radius_per_wire)
        solver._uniform_radius = float(solver._radius_per_wire[0])
    return solver


@pytest.mark.slow
@pytest.mark.crossgate
def test_the_equivalent_radius_closes_the_reactance_and_does_not_worsen_R():
    """The reason this change exists, gated as a SIGN rather than a value.

    Values move with soil, grass and mesh (momwire#865's anchor states all
    three); the direction is the claim. The `equivalent=False` arm IS the
    mutation "force a' back to a", run on every invocation rather than by
    hand.

    N = 8 rather than the anchor's 16: the claim is directional and N = 8
    carries it in 35 s against 9 minutes, which matters on a push lane that
    already carries BLE. What is NOT gated here is the more specific
    observation that L ALONE pushes X the wrong way — that is N-dependent and
    measured both ways (at N = 16, h = 1.6 mm: bare dX +11.76, +L +12.63, so
    L worsens it; at N = 8 the same three run +32.22, +29.13, +5.71, so L
    helps slightly). Pinning an N-dependent sign at one N would be pinning the
    deck, not the physics.
    """
    h = 1.6e-3
    measured = 85.5 + 8.0j  # Severns Table 1, N = 8
    bare = _severns(8, h, jacket=False, equivalent=False).compute_impedance()[0]
    old = _severns(8, h, jacket=True, equivalent=False).compute_impedance()[0]
    new = _severns(8, h, jacket=True, equivalent=True).compute_impedance()[0]

    # THE CLAIM: the equivalent radius closes X...
    assert abs(new.imag - measured.imag) < abs(old.imag - measured.imag)
    assert abs(new.imag - measured.imag) < abs(bare.imag - measured.imag)
    # ...and does not worsen R against bare.
    assert abs(new.real - measured.real) <= abs(bare.real - measured.real)

    # Vacuity: the two arms must actually differ, or "closes X" is comparing
    # a deck with itself.
    assert abs(new - old) > 1.0, (new, old)
