"""NEC's extended thin-wire kernel on the sinusoidal family (momwire#233).

Every momwire solver before #233 answered NEC's EK-OFF question: the source
current is a filament on the wire axis and the conductor's girth survives only
as the a² regularization of ρ. That is NEC's *reduced* kernel, and it is what
`SinusoidalSolver` still computes by default. `extended_kernel=True` switches
the sinusoidal collocation fill to NEC's EXTENDED thin-wire kernel — Eqs 84-98
of the LLNL theory manual (Burke & Poggio Part I, pp. 24-28), where the source
current is a uniform tube of surface current at ρ' = a and the kernel is its
circumferential average expanded to O(a²).

The two kernels agree to O(a²/R²), so the option is invisible on ordinary HF
wire (Δ/a in the hundreds) and worth tens of percent on fat conductors. The
exposure class is fat conductors, not short segments: a 2 m boom at spwl=120
puts ½″ tube at Δ/a ≈ 2.7 and 1″ tube at Δ/a ≈ 1.4.

ORACLE
------
All impedances below come from the SimNEC-bundled nec2c binary
(`nec2c.ae6ty/bin/nec2c-ubuntu-x86`, VERSION 5b4az.ae6ty.1.23), run once per
rung with and without the `EK` card and pinned here so CI needs no binary. The
deck is the #233 investigation's — a free-space dipole, L = 5 m, NS = 41,
30 MHz, fed at the centre segment, radius swept to walk Δ/a:

    CM dipole L=5m NS=41 a=<A> EK=<on|off>
    CE
    GW 1 41 0. 0. -2.5 0. 0. 2.5 <A>
    GE 0
    EK                      <- present only on the EK-ON runs
    FR 0 1 0 0 30. 0.
    EX 0 1 21 0 1. 0.
    XQ
    EN
"""

import numpy as np
import pytest

from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

# nec2c's CVEL, so momwire and the oracle mesh the same electrical length.
LAM = 299.792458 / 30.0
NS = 41
LEN = 5.0

# radius -> (Δ/a, nec2c EK-OFF, nec2c EK-ON)
LADDER = {
    0.001: (121.951, 80.146 + 46.432j, 80.146 + 46.430j),
    0.0048780: (25.000, 83.291 + 48.053j, 83.281 + 48.016j),
    0.005: (24.390, 83.364 + 48.092j, 83.353 + 48.053j),
    0.02: (6.0976, 90.040 + 50.737j, 89.774 + 50.190j),
    0.05: (2.4390, 101.010 + 50.114j, 98.660 + 48.113j),
    0.1: (1.2195, 123.470 + 34.444j, 111.020 + 37.528j),
    0.15: (0.81301, 130.460 - 30.093j, 122.240 + 18.377j),
    0.3: (0.40650, 0.40269 - 7.5778j, 37.990 - 56.910j),
}


def _dipole(radius, **kw):
    return SinusoidalSolver(
        wires=[np.array([[0.0, 0.0, -LEN / 2], [0.0, 0.0, LEN / 2]])],
        n_per_edge_per_wire=[[NS]],
        wavelength=LAM,
        wire_radius=radius,
        nsegs=NS,
        feed_arclength=LEN / 2,
        **kw,
    )


def _rel(z, ref):
    return abs(z - ref) / abs(ref)


# ----------------------------------------------------------------------
# Gate 1 — the Δ/a ladder against nec2c EK-ON
# ----------------------------------------------------------------------

# Measured on this box, EK-ON vs the nec2c EK-ON column (see the module report
# in momwire#233 for the full table):
#
#   Δ/a     EK-off vs off   EK-on vs on   what EK moves in nec2c
#   24.390      0.82%           0.82%            0.04%
#    6.098      1.23%           1.63%            0.59%
#    2.439      0.19%           1.63%            2.74%
#    1.220      0.55%           0.25%           10.0%
#    0.813      1.07%           0.24%           36.7%
#    0.407      0.06%           0.62%          817%
#
# The two middle rungs sit above the 1.5% the issue hoped for. That residual is
# NOT the kernel: it is momwire's own discretization gap against NEC-2's fill,
# and it converges away under refinement at FIXED Δ/a — at Δ/a = 2.439 the
# EK-ON error runs 1.63% (NS=41) -> 1.01% (NS=81) -> 0.60% (NS=161), while the
# ratio of momwire's EK correction to nec2c's stays pinned at 1.34 throughout.
# The kernel itself is exact to machine precision in the a -> 0 limit
# (`test_extended_kernel_zero_radius_limit_is_the_reduced_kernel`) and carries
# the hard rungs, where EK is worth 10-800%, to a quarter of a percent.
EK_ON_TOL = 0.017


@pytest.mark.parametrize("radius", list(LADDER))
def test_extended_kernel_ladder_matches_nec2_ek_on(radius):
    z, _ = _dipole(radius, extended_kernel=True).compute_impedance()
    da, _, z_on = LADDER[radius]
    assert _rel(z, z_on) < EK_ON_TOL, f"Δ/a={da}: {z} vs nec2c EK-ON {z_on}"


@pytest.mark.parametrize("radius", [0.1, 0.15, 0.3])
def test_extended_kernel_carries_the_rungs_the_reduced_kernel_cannot(radius):
    """Below Δ/a ≈ 1.25 the reduced kernel is not merely imprecise — it answers
    a different question, by 10% at Δ/a = 1.22 and by two orders of magnitude
    at 0.41. EK has to close essentially all of that, not just improve on it.
    """
    da, _, z_on = LADDER[radius]
    z_red, _ = _dipole(radius).compute_impedance()
    z_ext, _ = _dipole(radius, extended_kernel=True).compute_impedance()
    assert _rel(z_ext, z_on) < 0.01, f"Δ/a={da}: EK-on {z_ext} vs {z_on}"
    # And the reduced kernel really was that far off — this is the signal.
    assert _rel(z_red, z_on) > 10 * _rel(z_ext, z_on)


@pytest.mark.parametrize("radius", list(LADDER))
def test_reduced_kernel_still_answers_the_ek_off_question(radius):
    """The default must keep tracking nec2c WITHOUT the EK card. This is the
    control for the ladder above: it is what makes the EK column a measurement
    of the kernel rather than of momwire's basis.
    """
    z, _ = _dipole(radius).compute_impedance()
    da, z_off, _ = LADDER[radius]
    assert _rel(z, z_off) < 0.013, f"Δ/a={da}: {z} vs nec2c EK-OFF {z_off}"


# ----------------------------------------------------------------------
# Gate 2 — the extended kernel must vanish into the reduced one
# ----------------------------------------------------------------------


@pytest.mark.parametrize("radius", [0.0048780, 0.001])
def test_extended_kernel_is_a_no_op_for_ordinary_wire(radius):
    """At Δ/a ≥ 25 — i.e. every ordinary HF wire deck, which lives in the
    hundreds — turning EK on must not move the answer. nec2c's own EK shift
    here is 0.04% and below.
    """
    z_red, _ = _dipole(radius).compute_impedance()
    z_ext, _ = _dipole(radius, extended_kernel=True).compute_impedance()
    assert _rel(z_ext, z_red) < 1e-3, f"{z_ext} vs {z_red}"


# ----------------------------------------------------------------------
# The kernel identity: a -> 0 must recover the reduced kernel exactly
# ----------------------------------------------------------------------


def test_extended_kernel_zero_radius_limit_is_the_reduced_kernel():
    """momwire's reduced kernel is NEC's EKSC (nec2-1.2.1.2.f:3124-3166)
    rearranged into per-endpoint brackets, and EKSCX (f.3170-3234) is EKSC with
    GX swapped for GXX per end. So with the O(a²) terms switched off — a_src =
    0, which zeroes T1, T2 and (kB/2)² — BOTH extended arms must reproduce the
    reduced tables to machine precision, on every one of the six shape tables.

    This is the test that pins the algebraic equivalence the whole
    substitution rests on; if the endpoint sign convention, the CON prefactor
    or the z1/z2 assignment were off, this would not be at 1e-15.
    """
    sim = SinusoidalSolver(
        wires=[
            np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]]),
            np.array([[1.0, 0.0, -1.0], [1.0, 0.7, 1.3]]),  # skew: E_ρ is live
        ],
        n_per_edge_per_wire=[[21], [7]],
        wavelength=LAM,
        wire_radius=0.05,
        nsegs=21,
        feed_arclength=2.5,
    )
    geom = sim._build_geometry()
    kw = dict(
        obs_c=geom["seg_centers"][:, None, :],
        obs_t=geom["seg_tangents"][:, None, :],
        a=0.05,
        src_c=geom["seg_centers"][None, :, :],
        src_t=geom["seg_tangents"][None, :, :],
        src_hh=(0.5 * geom["seg_h"])[None, :],
    )
    zeros = np.zeros(geom["n_segs"], dtype=np.int8)
    reduced = sim._field_components_bcast(sim.k, **kw)
    gxx = sim._field_components_bcast(sim.k, **kw, ek=(0.0, zeros, zeros))
    gx = sim._field_components_bcast(sim.k, **kw, ek=(0.0, zeros + 2, zeros + 2))
    for key in ("Ez_const", "Ez_sin", "Ez_cos", "Erho_const", "Erho_sin", "Erho_cos"):
        scale = np.max(np.abs(reduced[key]))
        for name, table in (("GXX", gxx), ("GX", gx)):
            err = np.max(np.abs(reduced[key] - table[key])) / scale
            assert err < 1e-13, f"{key} via {name}: {err:g}"


# ----------------------------------------------------------------------
# Gate 3 — per-end neighbour gating, NEC's IND1/IND2
# ----------------------------------------------------------------------
#
# The gating is mandated by nec2-1.2.1.2.f:2019-2053 (and repeated verbatim at
# 6197-6224 and 7217-7244). Its physical reason is the theory manual's own,
# p. 28: on a straight wire the end contributions of adjacent segments cancel
# exactly, because the NEC basis is C¹-continuous there — "Special treatment of
# bends in wires is required when the extended thin-wire kernel is used ...
# while there is not complete cancellation, there may be partial cancellation
# of large end contributions." So the extended kernel is applied at an end only
# where the current genuinely continues straight into an identical conductor.
#
#   IND = 1  free end (f.2032 / f.2049)                     -> extended (GXX)
#   IND = 0  collinear + equal-radius two-segment junction
#            (f.2040-2043), or a perpendicular ground
#            contact (f.2026-2029)                          -> extended (GXX)
#   IND = 2  bend, radius step, multi-way junction          -> reduced  (GX)
#
# EKSCX routes on exactly that: `IF( INX1.EQ.2) GOTO 3` at f.3193.


def test_gating_extends_both_ends_along_a_straight_wire():
    sim = _dipole(0.05, extended_kernel=True)
    ind1, ind2 = sim._ek_gating(sim._build_geometry())
    # Free wire ends get code 1, every interior end code 0 — nothing reduced.
    assert ind1[0] == 1 and ind2[-1] == 1
    assert (ind1[1:] == 0).all() and (ind2[:-1] == 0).all()


def test_gating_differs_per_end_at_a_bend():
    """A single polyline with a vertex. The two segments meeting at the vertex
    are collinear with their in-edge neighbours but NOT with each other, so one
    end of each is extended and the other reduced — the per-end decision NEC's
    IND1/IND2 pair exists to express.
    """
    sim = SinusoidalSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.2], [3.6, 1.1, 1.2]])],
        n_per_edge_per_wire=[[6, 5]],
        wavelength=LAM,
        wire_radius=0.01,
        nsegs=11,
        feed_arclength=1.0,
        extended_kernel=True,
    )
    ind1, ind2 = sim._ek_gating(sim._build_geometry())
    # Segment 5 is the last of edge 1, segment 6 the first of edge 2.
    assert ind2[5] == 2, "bend end of the incoming segment must fall back to GX"
    assert ind1[6] == 2, "bend end of the outgoing segment must fall back to GX"
    # Their OTHER ends are ordinary in-edge junctions and stay extended: the
    # decision is per end, not per segment.
    assert ind1[5] == 0 and ind2[6] == 0
    # And the free wire ends are still code 1.
    assert ind1[0] == 1 and ind2[-1] == 1


def test_gating_rejects_a_radius_step():
    """Two collinear wires of different radii joined end to end. Collinearity
    holds, so ONLY NEC's |BI(IPR)/B - 1| > 1e-6 test at f.2042-2043 can reject
    this end — which is the point of pinning it separately from the bend.
    """
    sim = SinusoidalSolver(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
        ],
        n_per_edge_per_wire=[[5], [5]],
        wavelength=LAM,
        wire_radius=[0.02, 0.005],
        nsegs=10,
        feed_arclength=1.0,
        junctions=[[(0, "end"), (1, "start")]],
        extended_kernel=True,
    )
    ind1, ind2 = sim._ek_gating(sim._build_geometry())
    assert ind2[4] == 2 and ind1[5] == 2, "radius step must fall back to GX"
    assert ind1[4] == 0 and ind2[5] == 0, "the far ends are unaffected"

    # Same geometry, one radius: now collinear AND equal, so it extends.
    same = SinusoidalSolver(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
        ],
        n_per_edge_per_wire=[[5], [5]],
        wavelength=LAM,
        wire_radius=0.02,
        nsegs=10,
        feed_arclength=1.0,
        junctions=[[(0, "end"), (1, "start")]],
        extended_kernel=True,
    )
    j1, j2 = same._ek_gating(same._build_geometry())
    assert j2[4] == 0 and j1[5] == 0


def test_gating_rejects_a_multiway_junction():
    """Three wires at one node. NEC threads 3+ segments onto a circular ICON
    chain, so its reciprocal test (f.2025 `IF(-ICON1(IPR).NE.J)` / f.2031
    `IF(ICON2(IPR).NE.J)`) fails for every member; momwire sees the same thing
    as a neighbour count of 2 rather than 1.
    """
    sim = SinusoidalSolver(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
            np.array([[0.0, 0.0, 2.0], [1.5, 0.0, 2.0]]),
        ],
        n_per_edge_per_wire=[[5], [5], [4]],
        wavelength=LAM,
        wire_radius=0.02,
        nsegs=14,
        feed_arclength=1.0,
        junctions=[[(0, "end"), (1, "start"), (2, "start")]],
        extended_kernel=True,
    )
    geom = sim._build_geometry()
    ind1, ind2 = sim._ek_gating(geom)
    assert geom["np_count"][4] == 2, "fixture must really be a 3-way junction"
    assert ind2[4] == 2 and ind1[5] == 2 and ind1[10] == 2


def test_gating_is_load_bearing():
    """Applying the extended kernel unconditionally is a DIFFERENT model, not a
    rounding difference — which is why #233 matches NEC's gating exactly rather
    than shipping an ungated variant.
    """

    class Ungated(SinusoidalSolver):
        def _ek_gating(self, geom):
            n = geom["n_segs"]
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)

    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.2], [3.6, 1.1, 1.2]])],
        n_per_edge_per_wire=[[6, 5]],
        wavelength=LAM,
        wire_radius=0.08,
        nsegs=11,
        feed_arclength=1.0,
        extended_kernel=True,
    )
    z_gated, _ = SinusoidalSolver(**kw).compute_impedance()
    z_ungated, _ = Ungated(**kw).compute_impedance()
    assert _rel(z_ungated, z_gated) > 1e-3, f"{z_ungated} vs {z_gated}"


# ----------------------------------------------------------------------
# Gate 4 — disabled-path armor
# ----------------------------------------------------------------------

# Impedances of a representative geometry set, captured on `main` (i.e. with
# the #233 diff stashed) and reproduced BIT-FOR-BIT by this branch's default.
# Hex floats so the comparison is exact rather than repr-rounded.
MAIN_SIDE = {
    name: (float.fromhex(re), float.fromhex(im))
    for name, (re, im) in {
        "dipole": ("0x1.6651bc56a91bfp+6", "0x1.8c83037a303dcp+5"),
        "vee": ("0x1.1d2c092cabf47p+6", "-0x1.51256e4c55fedp+7"),
        "tee_radii": ("0x1.3aa72a4355b88p+8", "0x1.e96509b03d599p+7"),
        "ground_pec": ("0x1.604543e8acef3p+4", "-0x1.b9e0c6df28707p+8"),
        "ground_refl": ("0x1.2a4bd618bebcdp+4", "-0x1.bd2c7e67eadb0p+8"),
        "galerkin": ("0x1.656e033053765p+6", "0x1.86e6e8c152246p+5"),
    }.items()
}

_DIPOLE_W = [np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])]
_VERT_W = [np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 3.0]])]

ARMOR_CASES = {
    "dipole": (
        SinusoidalSolver,
        dict(wires=_DIPOLE_W, n_per_edge_per_wire=[[41]], wire_radius=0.02, nsegs=41),
    ),
    "vee": (
        SinusoidalSolver,
        dict(
            wires=[np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.2], [3.6, 1.1, 1.2]])],
            n_per_edge_per_wire=[[13, 11]],
            wire_radius=0.01,
            nsegs=24,
            feed_arclength=1.0,
        ),
    ),
    "tee_radii": (
        SinusoidalSolver,
        dict(
            wires=[
                np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 6.0]]),
                np.array([[0.0, 0.0, 6.0], [2.4, 0.0, 6.0]]),
            ],
            n_per_edge_per_wire=[[11], [9]],
            wire_radius=[0.02, 0.005],
            nsegs=20,
            feed_arclength=1.0,
            junctions=[[(0, "end"), (1, "start")]],
        ),
    ),
    "ground_pec": (
        SinusoidalSolver,
        dict(
            wires=_VERT_W,
            n_per_edge_per_wire=[[21]],
            wire_radius=0.01,
            nsegs=21,
            ground_z=0.0,
            feed_arclength=1.25,
        ),
    ),
    "ground_refl": (
        SinusoidalSolver,
        dict(
            wires=_VERT_W,
            n_per_edge_per_wire=[[21]],
            wire_radius=0.01,
            nsegs=21,
            ground_z=0.0,
            ground_eps=(13.0, 0.005),
            feed_arclength=1.25,
        ),
    ),
    "galerkin": (
        SinusoidalGalerkinSolver,
        dict(wires=_DIPOLE_W, n_per_edge_per_wire=[[41]], wire_radius=0.02, nsegs=41),
    ),
}


def _armor_solver(name, **extra):
    cls, kw = ARMOR_CASES[name]
    kw = dict(kw)
    kw.setdefault("feed_arclength", 2.5)
    return cls(wavelength=9.993, **kw, **extra)


@pytest.mark.parametrize("name", list(ARMOR_CASES))
@pytest.mark.parametrize("explicit", [False, True])
def test_extended_kernel_off_is_bit_identical_to_main(name, explicit):
    """The option must be a true no-op when off, on free space, a bend, a
    mixed-radius junction, both ground models and the Galerkin fill alike —
    and identical whether it is defaulted or passed explicitly.
    """
    z, _ = _armor_solver(name).compute_impedance()
    if explicit:
        # Defaulted vs explicit MUST be bit-identical — same machine, same
        # run, so exact equality is the honest gate here.
        z_x, _ = _armor_solver(name, extended_kernel=False).compute_impedance()
        assert z.real == z_x.real and z.imag == z_x.imag, name
    # Against the pre-#233 captures the gate is the house cross-machine
    # margin, NOT bit equality: the pinned values are one dev box's
    # reduction order, and CI runners land 1 ulp away (the main run on
    # 31352791540 failed exact comparison at the 16th digit — same policy
    # call as the 1e-10 comments in test_momwire.py). The true bit-identity
    # claim was established pre-merge against the actual pre-#233 code on
    # one machine (PR #244's gate 4); THIS pin is drift armor.
    re, im = MAIN_SIDE[name]
    ref = complex(re, im)
    assert abs(z - ref) <= 1e-12 * abs(ref), f"{name}: {z!r} vs {ref!r}"


@pytest.mark.parametrize("name", list(ARMOR_CASES))
def test_extended_kernel_code_is_never_entered_when_off(name, monkeypatch):
    """Numerical identity is necessary but not sufficient — a feature path that
    happened to compute the same answer would pass it. Count entries into every
    #233 code object and require zero.
    """
    calls = []
    for attr in ("_ek_gating", "_ek_end_gxx", "_ek_end_gx", "_extended_kernel_fields"):
        original = getattr(SinusoidalSolver, attr)

        def trip(*a, _attr=attr, _o=original, **kw):
            calls.append(_attr)
            return _o(*a, **kw)

        monkeypatch.setattr(SinusoidalSolver, attr, trip)

    _armor_solver(name).compute_impedance()
    assert calls == [], f"{name} entered EK code with extended_kernel=False: {calls}"


# ----------------------------------------------------------------------
# Gate 7 — Galerkin refuses rather than quietly serving the reduced kernel
# ----------------------------------------------------------------------


def test_galerkin_refuses_the_extended_kernel():
    with pytest.raises(NotImplementedError, match="extended_kernel"):
        SinusoidalGalerkinSolver(
            wires=_DIPOLE_W,
            n_per_edge_per_wire=[[21]],
            wavelength=LAM,
            wire_radius=0.05,
            nsegs=21,
            feed_arclength=2.5,
            extended_kernel=True,
        )


def test_galerkin_still_accepts_the_reduced_kernel_explicitly():
    sim = SinusoidalGalerkinSolver(
        wires=_DIPOLE_W,
        n_per_edge_per_wire=[[21]],
        wavelength=LAM,
        wire_radius=0.05,
        nsegs=21,
        feed_arclength=2.5,
        extended_kernel=False,
    )
    assert sim.extended_kernel is False


# ----------------------------------------------------------------------
# Gate 8 — the C++ extended-kernel field tensor (momwire#245)
# ----------------------------------------------------------------------
#
# `_accelerators.sinusoidal_field_tensor_ek` is a transcription of
# `_ek_end_gxx` / `_ek_end_gx` / `_extended_kernel_fields` into C++. The numpy
# trio stays the reference implementation and the oracle, so the gate is the
# house accelerator-agreement one: the same returned (Phi_const, Phi_sin,
# Phi_cos) to ACCEL_AGREEMENT = 1e-13 relative — the tolerance test_momwire's
# reduced-kernel field-tensor test and test_sinusoidal_galerkin's own
# ACCEL_AGREEMENT both use. Measured on the free-space builds it sits at
# 4e-16 - 2.7e-15, the two paths differing only in scalar-vs-ufunc rounding.
#
# THE IMAGE BUILDS NEED A CONTROL, not a looser number. A mirrored source sits
# on the far side of the plane from the observer, and once kR grows the
# endpoint brackets of Eqs 76-79 cancel down to the radiation term — so BOTH
# kernels lose digits there, the long-standing reduced-kernel accelerator
# included. On this battery the reduced kernel's own C++-vs-numpy agreement
# runs to 1.3e-13 on the image blocks (7.6e-15 - 1.3e-13 across the fixtures),
# and it climbs to 2.3e-11 at a 20 m mirror plane. Gating EK at a flat 1e-13
# there would be measuring the geometry, not the transcription. So the rule
# below is: 1e-13, OR — where the reduced kernel cannot hold 1e-13 either —
# within `_CONDITIONING_HEADROOM` of what the reduced kernel achieves on the
# SAME geometry, which is the only honest control for the conditioning. The
# extended kernel measures 0.3x - 2.5x the reduced kernel's error on this
# battery and 4.2x - 6.1x at the far mirror planes; the transcription bug this
# gate actually caught in development sat at 3600x.
#
# For the gate to be decisive the battery has to reach every branch the C++
# can take:
#
#   IND = 1   free wire end                     -> GXX
#   IND = 0   collinear equal-radius junction   -> GXX
#   IND = 0   perpendicular ground contact      -> GXX  (NEC's f.2026-2029 arm,
#                                                        unreached before #245)
#   IND = 2   bend / radius step / 3-way node   -> GX
#   want_swapped True / False                   -> EKSCX's two IRA arms
#   uniform radius / per-observer-run dispatch  -> the two call shapes
#   free-space sources / mirrored PEC image     -> both builds
#
# `test_cpp_ek_battery_is_decisive` pins that coverage, so a fixture cannot
# quietly stop exercising an arm and leave the assertions looking green.

ACCEL_AGREEMENT = 1e-13
_CONDITIONING_HEADROOM = 20.0

# The image build of a fixture that carries no ground of its own runs against a
# plane just under it — clear of the wire by far more than the 1e-6·L contact
# tolerance, so the geometry and the gating are unchanged, and at the
# source-observer separations a real PEC ground produces.
_SPARE_GROUND_MARGIN = 0.3

EK_BATTERY = {
    # (a) single free wire: both ends IND=1, every interior end IND=0.
    "free_wire": dict(
        wires=[np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])],
        n_per_edge_per_wire=[[21]],
        wire_radius=0.05,
        nsegs=21,
        feed_arclength=2.5,
    ),
    # (c) bent wire: the two segments at the vertex take IND=2 on one end each.
    "bent_wire": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.2], [3.6, 1.1, 1.2]])],
        n_per_edge_per_wire=[[6, 5]],
        wire_radius=0.08,
        nsegs=11,
        feed_arclength=1.0,
    ),
    # (b) multi-wire junction: three members at one node, IND=2 on all three.
    "three_way": dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
            np.array([[0.0, 0.0, 2.0], [1.5, 0.0, 2.0]]),
        ],
        n_per_edge_per_wire=[[5], [5], [4]],
        wire_radius=0.02,
        nsegs=14,
        feed_arclength=1.0,
        junctions=[[(0, "end"), (1, "start"), (2, "start")]],
    ),
    # (d) mixed radii, collinear: the thin wire's segment centres lie INSIDE
    # the fat wire's radius, which is what sets EKSCX's IRA (want_swapped).
    "radius_step": dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
        ],
        n_per_edge_per_wire=[[5], [5]],
        wire_radius=[0.02, 0.005],
        nsegs=10,
        feed_arclength=1.0,
        junctions=[[(0, "end"), (1, "start")]],
    ),
    # (d) the same radius step with a SKEW wire parked beside it. Necessary,
    # not decorative: EKSCX's IRA arm rewrites the ρ-flavoured slots (G2, G2P,
    # G3) and leaves the z-flavoured ones alone, so on a purely collinear deck
    # the ρ-projection factor is identically zero and the whole arm cancels out
    # of Φ — a C++ build that ignored `want_swapped` passed the rest of this
    # battery. Here the arm is worth 19% of the tensor.
    "radius_step_skew": dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
            np.array([[1.1, 0.6, 0.3], [2.3, 1.9, 1.4]]),
        ],
        n_per_edge_per_wire=[[5], [5], [4]],
        wire_radius=[0.02, 0.005, 0.005],
        nsegs=14,
        feed_arclength=1.0,
        junctions=[[(0, "end"), (1, "start")]],
    ),
    # (d/f) mixed radii, skew: two observer-radius runs, but no observer falls
    # inside a source conductor — the IRA=0 half of the mixed-radius case.
    "skew_tee": dict(
        wires=[
            np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 6.0]]),
            np.array([[0.0, 0.0, 6.0], [2.4, 0.7, 6.0]]),
        ],
        n_per_edge_per_wire=[[11], [9]],
        wire_radius=[0.02, 0.005],
        nsegs=20,
        feed_arclength=1.0,
        junctions=[[(0, "end"), (1, "start")]],
    ),
    # (e) a wire STANDING IN the plane: the ground-contacting end is
    # perpendicular to it, so it takes NEC's IND=0 ICON==J arm, which no other
    # fixture here reaches.
    "grounded_monopole": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])],
        n_per_edge_per_wire=[[15]],
        wire_radius=0.03,
        nsegs=15,
        feed_arclength=0.1,
        ground_z=0.0,
    ),
    # (e+f) ground contact AND mixed radii at once: an inverted L whose
    # per-run dispatch also has to serve the mirrored image sources.
    "grounded_ell_radii": dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [1.6, 0.0, 2.0]]),
        ],
        n_per_edge_per_wire=[[9], [7]],
        wire_radius=[0.03, 0.008],
        nsegs=16,
        feed_arclength=0.15,
        ground_z=0.0,
        junctions=[[(0, "end"), (1, "start")]],
    ),
}


def _ek_kwargs(name, image=False):
    kw = dict(EK_BATTERY[name])
    if image and "ground_z" not in kw:
        low = min(float(np.asarray(w)[:, 2].min()) for w in kw["wires"])
        kw["ground_z"] = low - _SPARE_GROUND_MARGIN
    return kw


def _ek_solver(name, image=False):
    return SinusoidalSolver(
        wavelength=LAM, extended_kernel=True, **_ek_kwargs(name, image=image)
    )


def _accel_vs_numpy(sim, image):
    """Worst max-elementwise relative delta between the C++ field tensor and
    its numpy reference, over the three returned tensors, on one build. Works
    for either kernel — `sim.extended_kernel` picks which accelerator flag is
    switched off to get the reference."""
    import momwire.sinusoidal as sin_mod

    geom = sim._build_geometry()
    build = sim._field_tensor_image if image else sim._field_tensor
    cpp = build(geom, sim.k)
    flag = "_HAVE_FIELD_TENSOR_EK" if sim.extended_kernel else "_HAVE_FIELD_TENSOR"
    saved = getattr(sin_mod, flag)
    setattr(sin_mod, flag, False)
    try:
        ref = build(geom, sim.k)
    finally:
        setattr(sin_mod, flag, saved)
    worst, worst_label = 0.0, ""
    for label, a_cpp, a_ref in zip(("Phi_const", "Phi_sin", "Phi_cos"), cpp, ref):
        rel = np.max(np.abs(a_cpp - a_ref)) / max(np.max(np.abs(a_ref)), 1e-30)
        if rel > worst:
            worst, worst_label = rel, label
    return worst, worst_label


@pytest.mark.parametrize("name", list(EK_BATTERY))
@pytest.mark.parametrize("image", [False, True], ids=["free-space", "pec-image"])
def test_cpp_ek_field_tensor_matches_numpy(name, image):
    """Every gating branch, on both builds the PEC-image dispatch uses.

    The image build rides the same `_field_tensor` closure with mirrored
    sources, so it is accelerated identically: the (N,) gating and radius
    tables are indexed by SOURCE segment and the mirror reorders nothing —
    which is also what NEC does, EFLD passing one IND1/IND2 pair through both
    passes of its KSYMP loop (nec2-1.2.1.2.f:2914-2971).

    See the section header for why the reduced kernel is the control on the
    ill-conditioned image blocks rather than a looser constant.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK:
        pytest.skip("C++ accelerator not built")

    kw = _ek_kwargs(name, image=image)
    ek, label = _accel_vs_numpy(
        SinusoidalSolver(wavelength=LAM, extended_kernel=True, **kw), image
    )
    if ek < ACCEL_AGREEMENT:
        return
    reduced, _ = _accel_vs_numpy(
        SinusoidalSolver(wavelength=LAM, extended_kernel=False, **kw), image
    )
    assert ek < _CONDITIONING_HEADROOM * reduced, (
        f"{name} {'image' if image else 'free'} {label}: EK {ek:.3e} misses "
        f"{ACCEL_AGREEMENT:g} and is {ek / max(reduced, 1e-300):.1f}x the "
        f"reduced kernel's own {reduced:.3e} on the same geometry"
    )


@pytest.mark.parametrize("ground_z", [-6.0, -20.0])
def test_cpp_ek_far_image_conditioning_tracks_the_reduced_kernel(ground_z):
    """The same control, pushed to mirror planes far past anything the battery
    reaches: at 6 m and 20 m the reduced kernel is already at 3.1e-12 / 2.3e-11
    and no absolute tolerance means anything. The extended kernel measures 4.2x
    and 6.1x of it. This is what makes the conditioning claim in the section
    header a test rather than a comment.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK:
        pytest.skip("C++ accelerator not built")

    def _worst(extended):
        sim = SinusoidalSolver(
            wavelength=LAM,
            wires=_DIPOLE_W,
            n_per_edge_per_wire=[[21]],
            wire_radius=0.05,
            nsegs=21,
            feed_arclength=2.5,
            ground_z=ground_z,
            extended_kernel=extended,
        )
        return _accel_vs_numpy(sim, image=True)[0]

    reduced = _worst(False)
    extended = _worst(True)
    assert extended < _CONDITIONING_HEADROOM * reduced, (
        f"EK {extended:.3e} vs reduced {reduced:.3e}"
    )


def test_cpp_ek_ira_arm_is_load_bearing():
    """The IRA (`want_swapped`) arm must be worth catching. It rewrites only
    the ρ-flavoured slots (G2, G2P, G3), so on a collinear deck the ρ-projection
    factor is zero and the arm cancels out of Φ entirely — a C++ build that
    ignored the flag passed every other assertion in this section. This pins a
    fixture where it does not cancel, so `test_cpp_ek_field_tensor_matches_numpy`
    is actually testing something on that branch.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK:
        pytest.skip("C++ accelerator not built")

    class ForcedIra0(SinusoidalSolver):
        def _ek_any_swap(self, geom, src_c, src_t):
            return False

    kw = _ek_kwargs("radius_step_skew")
    sim = _ek_solver("radius_step_skew")
    geom = sim._build_geometry()
    assert sim._ek_any_swap(geom, geom["seg_centers"], geom["seg_tangents"])
    on = sim._field_tensor(geom, sim.k)
    off = ForcedIra0(wavelength=LAM, extended_kernel=True, **kw)._field_tensor(
        geom, sim.k
    )
    moved = max(np.max(np.abs(a - b)) / np.max(np.abs(a)) for a, b in zip(on, off))
    assert moved > 1e-3, f"IRA arm moves the tensor by only {moved:.3e} here"


def test_cpp_ek_battery_is_decisive():
    """The battery must really hit every arm — otherwise the 1e-13 gate above
    is only pinning the arms it happens to reach. Assert the coverage rather
    than trusting the fixtures to stay as written.
    """
    codes, swaps, uniform = set(), set(), set()
    multirun = False
    ground_contact_extends = False
    for name in EK_BATTERY:
        sim = _ek_solver(name)
        geom = sim._build_geometry()
        ind1, ind2 = sim._ek_gating(geom)
        codes |= set(ind1.tolist()) | set(ind2.tolist())
        uniform.add(sim._uniform_radius is not None)
        multirun = multirun or len(sim._radius_runs(geom)) > 1
        swaps.add(sim._ek_any_swap(geom, geom["seg_centers"], geom["seg_tangents"]))
        if sim.ground_z is None:
            continue
        at_plane = np.flatnonzero(geom["ground_minus"])
        if at_plane.size:
            # NEC's ICON == J arm: an end lying in the plane, on a segment
            # perpendicular to it, extends rather than reduces.
            ground_contact_extends = ground_contact_extends or bool(
                (ind1[at_plane] == 0).all()
            )
        # The image build resolves its own IRA, on the MIRRORED sources.
        src_c, src_t = sim._image_source_centers_tangents(geom)
        swaps.add(sim._ek_any_swap(geom, src_c, src_t))
    assert codes == {0, 1, 2}, codes
    assert swaps == {False, True}, "EKSCX's IRA arm is not covered both ways"
    assert uniform == {False, True}, "both radius dispatch shapes are needed"
    assert multirun, "no fixture produces more than one observer-radius run"
    assert ground_contact_extends, "the perpendicular ground-contact IND=0 arm is unhit"


@pytest.mark.parametrize("name", ["three_way", "radius_step", "grounded_ell_radii"])
def test_cpp_ek_impedance_matches_the_numpy_path(name, monkeypatch):
    """Kernel agreement is not the whole dispatch. Solving end to end both ways
    on junction geometries pins the gating/radius plumbing, the per-run
    stitching and the image-block routing too, not just the arithmetic inside
    one call.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK:
        pytest.skip("C++ accelerator not built")

    z_cpp, _ = _ek_solver(name).compute_impedance()
    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_EK", False)
    z_ref, _ = _ek_solver(name).compute_impedance()
    assert _rel(z_cpp, z_ref) < 1e-12, f"{name}: {z_cpp} vs {z_ref}"


def test_cpp_ek_y_matrix_matches_the_numpy_path(monkeypatch):
    """Same claim one level up: the multiport solve, where every basis column
    rides the same fill."""
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK:
        pytest.skip("C++ accelerator not built")

    def _y():
        sim = _ek_solver("three_way")
        return sim.compute_y_matrix()

    y_cpp = _y()
    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_EK", False)
    y_ref = _y()
    y_cpp = np.asarray(y_cpp[0] if isinstance(y_cpp, tuple) else y_cpp)
    y_ref = np.asarray(y_ref[0] if isinstance(y_ref, tuple) else y_ref)
    rel = np.max(np.abs(y_cpp - y_ref)) / np.max(np.abs(y_ref))
    assert rel < 1e-12, rel


def test_cpp_ek_path_does_not_re_enter_the_numpy_kernel(monkeypatch):
    """The mirror of the off-path armor. With the accelerator present, an EK-ON
    solve must go through C++ and never touch `_extended_kernel_fields` — the
    ~8.5x and the (N, N, n_qp) temporaries #245 exists to remove. Gating still
    runs in Python: it is a connectivity walk, not arithmetic.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK:
        pytest.skip("C++ accelerator not built")

    calls = []
    for attr in ("_ek_end_gxx", "_ek_end_gx", "_extended_kernel_fields"):
        original = getattr(SinusoidalSolver, attr)

        def trip(*a, _attr=attr, _o=original, **kw):
            calls.append(_attr)
            return _o(*a, **kw)

        monkeypatch.setattr(SinusoidalSolver, attr, trip)

    _ek_solver("grounded_ell_radii").compute_impedance()
    assert calls == [], f"EK-ON solve fell back to the numpy kernel: {calls}"
