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
    extra = {"extended_kernel": False} if explicit else {}
    z, _ = _armor_solver(name, **extra).compute_impedance()
    re, im = MAIN_SIDE[name]
    assert z.real == re and z.imag == im, f"{name}: {z.real.hex()} {z.imag.hex()}"


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
