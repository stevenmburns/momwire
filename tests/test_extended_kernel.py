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
import platform
import sys

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


@pytest.mark.integration
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

# The pre-#233 `main` capture that used to be pinned here — six geometries as
# hex floats, gated at 1e-12 relative — is GONE (momwire#483). It was drift
# armor, not the claim this section is named for, and it could not survive at
# a bit-level bar: swapping ONLY the fill's arithmetic path on ONE box moves
# these impedances by up to 2.6e-12 relative, so which case sits nearest the
# 1e-12 line is a lottery over the host's libm. Measured here (Haswell,
# glibc 2.35), relative to that capture:
#
#                        libmvec+FMA   libmvec, FMA masked   pure-numpy fill
#   vee                     1.00e-12              6.04e-14          2.12e-13
#   tee_radii               1.02e-13              8.31e-13          2.56e-12
#
# The three paths are all supported: the C++ accelerator's SIMD sincos binds
# `_ZGVdN4v_sin`, an IFUNC glibc resolves from the host's hwcaps (masking FMA
# with GLIBC_TUNABLES picks a different kernel for the same symbol), and
# `_accel.py` documents the numpy fallback as a deliberate pure-Python
# install. `vee` was not special — it was merely the case the local libm
# pushed over the line first.
#
# Nothing is lost by dropping it. The claim is structural and needs no
# literal: `test_extended_kernel_code_is_never_entered_when_off` proves the
# #233 code objects are never entered with the flag off, and the test below
# proves the flag changes not one bit of the answer, defaulted vs explicit,
# against a run-it-twice control. Portable coverage of the EK-OFF numbers
# themselves is Gate 1's `test_reduced_kernel_still_answers_the_ek_off_
# question` (nec2c's EK-OFF column, 1.3%).

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
    # `explicit` is the treatment/control axis. TREATMENT (True): passing the
    # flag off must be bit-identical to leaving it defaulted — that is the
    # no-op claim. CONTROL (False): the SAME defaulted construction, built and
    # solved a second time, must be bit-identical too — without it a green
    # treatment could be read as "nothing here reproduces to the bit anyway",
    # and run-to-run bit-drift is a live failure mode in this repo (#403,
    # #464). Same machine, same run, so exact equality is the honest gate for
    # both, and neither needs a recorded value to compare against.
    off = dict(extended_kernel=False) if explicit else {}
    z_x, _ = _armor_solver(name, **off).compute_impedance()
    # momwire#809: the two sides' fills measured BIT-IDENTICAL, so this
    # `==` is structural, not a solve-downstream lottery ticket.
    assert z.real == z_x.real and z.imag == z_x.imag, name


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
# Gate 7 — the Galerkin family's own contract
# ----------------------------------------------------------------------
#
# momwire#233 shipped the extended kernel on the point-matched solver and had
# `SinusoidalGalerkinSolver` refuse it outright; momwire#246 narrowed that
# refusal to one combination (Sommerfeld) and momwire#287 removed the last of
# it. What is left here is that the two solvers agree the kernel is served at
# all — the Galerkin fill's own gate battery, including every ground, is
# `tests/test_extended_kernel_galerkin.py`.


def _galerkin(**kw):
    return SinusoidalGalerkinSolver(
        wires=_DIPOLE_W,
        n_per_edge_per_wire=[[21]],
        wavelength=LAM,
        wire_radius=0.05,
        nsegs=21,
        feed_arclength=2.5,
        **kw,
    )


def test_galerkin_accepts_the_extended_kernel():
    """It solves, and the kernel is not a no-op on this Δ/a ≈ 2.4 wire."""
    z_red, _ = _galerkin().compute_impedance()
    z_ext, _ = _galerkin(extended_kernel=True).compute_impedance()
    assert abs(z_ext - z_red) > 1e-3 * abs(z_red)


def test_galerkin_serves_the_extended_kernel_under_sommerfeld_ground():
    """The combination momwire#246 refused and momwire#287 opened. The
    Sommerfeld ground-wave remainder is still the reduced-kernel evaluator
    under an extended image, which is a deliberate and measured mixture —
    `test_extended_kernel_galerkin.py`'s G-S1 is the measurement, and the
    class docstring carries the table."""
    ground = dict(ground_z=-3.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
    z_red, _ = _galerkin(**ground).compute_impedance()
    z_ext, _ = _galerkin(extended_kernel=True, **ground).compute_impedance()
    assert abs(z_ext - z_red) > 1e-3 * abs(z_red)


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
#   IRA = 1 / 0 pairs in the SAME fill          -> EKSCX's two IRA arms, which
#                                                  are chosen per pair (#258)
#   uniform radius / per-observer-run dispatch  -> the two call shapes
#   free-space sources / mirrored PEC image     -> both builds
#
# `test_cpp_ek_battery_is_decisive` pins that coverage, so a fixture cannot
# quietly stop exercising an arm and leave the assertions looking green.

ACCEL_AGREEMENT = 1e-13
_CONDITIONING_HEADROOM = 20.0

# The ratio needs an absolute floor: on macOS ARM (PR #529's new CI lane) the
# reduced kernel's own C++-vs-numpy worst can land an order tighter than on
# x86, and a 26x ratio between two agreements that are BOTH under 3e-10 is
# not a conditioning failure. Genuine blowups are orders above this floor.
_CONDITIONING_FLOOR = 1e-9

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
    # the fat wire's radius, which is what sets EKSCX's IRA.
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
    # of Φ — a C++ build that ignored the arm passed the rest of this battery.
    # Here the arm is worth 19% of the tensor. 25 of its 196 pairs swap and 171
    # do not, which is also what makes it the fixture where a GLOBAL arm and a
    # per-pair one part company (momwire#258, Gate 10 below).
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
    # (d) the deciding deck of momwire#258: the same shape as
    # `radius_step_skew` with the step made FAT (Δ/a = 1.33, where EK is worth
    # 13.5% to nec2c) and the skew member brought in close, so the 171 pairs
    # that must NOT take the IRA==1 arm carry real weight. Gate 10 anchors it
    # against nec2c; here it puts a mixed-IRA fill through the cross-backend
    # gates, which `radius_step_skew` alone does only weakly.
    "radius_step_skew_fat": dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
            np.array([[0.6, 0.35, 0.2], [2.1, 1.9, 2.4]]),
        ],
        n_per_edge_per_wire=[[5], [5], [6]],
        wire_radius=[0.30, 0.02, 0.02],
        nsegs=16,
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


def _swap_mask(sim, geom, src_c, src_t):
    """EKSCX's IRA per pair, from the literal definition (f.3186-3192): the
    (M, N) `rho_eval < src_a`, with `rho_eval` the OBSERVER-radius-regularized
    radial distance to the source axis and `src_a` the SOURCE conductor's
    radius. Both backends resolve this inline, per pair, since momwire#258 —
    the numpy path in `_extended_kernel_fields`, the C++ path in stage A of
    `sinusoidal_field_tensor_ek` — so the tests keep their own copy rather
    than asserting a spelling against itself.
    """
    obs_c = geom["seg_centers"]
    a = sim._seg_radius(geom)
    rvec = obs_c[:, None, :] - src_c[None, :, :]
    z = np.einsum("...d,...d->...", rvec, src_t[None, :, :])
    rho = np.linalg.norm(rvec - z[..., None] * src_t[None, :, :], axis=-1)
    rho_eval = np.sqrt(rho * rho + a[:, None] ** 2)
    return rho_eval < a


class _GlobalIra(SinusoidalSolver):
    """The pre-momwire#258 solver: EKSCX's IRA collapsed to ONE arm for the
    whole fill, `np.any(rhx < src_a)` over the entire (M, N) grid, so a single
    observation point inside a source conductor put every pair on the IRA==1
    formula — evaluated with the unswapped (rh, b) of the pairs that did not
    swap. Only the numpy kernel can be forced this way; the accelerated path
    has no such switch left, which is the point. Tests using it turn the
    accelerator off and compare numpy against numpy.
    """

    @staticmethod
    def _ek_end_gxx(k, zz, rh, b, want_swapped):
        return SinusoidalSolver._ek_end_gxx(k, zz, rh, b, bool(np.any(want_swapped)))


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
    assert extended < max(_CONDITIONING_HEADROOM * reduced, _CONDITIONING_FLOOR), (
        f"EK {extended:.3e} vs reduced {reduced:.3e}"
    )


def test_cpp_ek_ira_arm_is_load_bearing(monkeypatch):
    """The IRA arm must be worth catching. It rewrites only the ρ-flavoured
    slots (G2, G2P, G3), so on a collinear deck the ρ-projection factor is zero
    and the arm cancels out of Φ entirely — a C++ build that ignored it passed
    every other assertion in this section. This pins a fixture where it does
    not cancel, so `test_cpp_ek_field_tensor_matches_numpy` is actually testing
    something on that branch.

    The control is `_GlobalIra` — the pre-momwire#258 fill, which put every
    pair on the arm `np.any` chose. Forcing the arm OFF everywhere would prove
    nothing on this fixture: the pairs that DO swap here are the collinear
    ones, so the 25 of them are exactly the ones the ρ-projection cannot carry.
    It is the 171 pairs the global arm dragged onto IRA==1 that move Φ. Only
    the numpy kernel can still be forced this way (the accelerated path
    resolves IRA per pair in stage A and takes no argument for it), which is
    enough: the C++ path is pinned to this numpy path at 1e-13 here.
    """
    import momwire.sinusoidal as sin_mod

    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_EK", False)

    kw = _ek_kwargs("radius_step_skew")
    sim = _ek_solver("radius_step_skew")
    geom = sim._build_geometry()
    swap = _swap_mask(sim, geom, geom["seg_centers"], geom["seg_tangents"])
    assert swap.any() and not swap.all(), "fixture must mix both IRA arms"
    on = sim._field_tensor(geom, sim.k)
    off = _GlobalIra(wavelength=LAM, extended_kernel=True, **kw)._field_tensor(
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
    mixed_ira = False
    for name in EK_BATTERY:
        sim = _ek_solver(name)
        geom = sim._build_geometry()
        ind1, ind2 = sim._ek_gating(geom)
        codes |= set(ind1.tolist()) | set(ind2.tolist())
        uniform.add(sim._uniform_radius is not None)
        multirun = multirun or len(sim._radius_runs(geom)) > 1
        free = _swap_mask(sim, geom, geom["seg_centers"], geom["seg_tangents"])
        swaps |= set(np.unique(free).tolist())
        # A single fill carrying BOTH arms at once is the case momwire#258
        # exists for; a battery of all-swap and no-swap decks would leave the
        # per-pair selection itself untested.
        mixed_ira = mixed_ira or bool(free.any() and not free.all())
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
        swaps |= set(np.unique(_swap_mask(sim, geom, src_c, src_t)).tolist())
    assert codes == {0, 1, 2}, codes
    assert swaps == {False, True}, "EKSCX's IRA arm is not covered both ways"
    assert mixed_ira, "no fixture puts both IRA arms in the SAME fill"
    assert uniform == {False, True}, "both radius dispatch shapes are needed"
    assert multirun, "no fixture produces more than one observer-radius run"
    assert ground_contact_extends, "the perpendicular ground-contact IND=0 arm is unhit"


# Impedance-level agreement, per fixture. 1e-12 is the gate; the three named
# below need 5e-12 and it is conditioning, not dispatch. They are driven near
# an antiresonance (skew_tee answers 449 + 331j, the monopole is a quarter-wave
# stub over PEC), so the matrix solve multiplies the fill's 1e-15 reassociation
# delta by a large condition number on the way to Z. Measured across the
# battery: 2.8e-14 (radius_step), 2.8e-14 (radius_step_skew), 5.1e-14
# (radius_step_skew_fat), 8.1e-14 (three_way), 9.5e-14 (bent_wire), 9.0e-13
# (grounded_ell_radii), 1.2e-12 (free_wire), 1.9e-12 (grounded_monopole),
# 5.0e-13 (skew_tee). A dispatch
# fault — wrong gating table, unstitched radius run, image block on the wrong
# kernel — lands at 1e-2, not here.
#
# `free_wire` MOVED UP at momwire#799, 9.3e-13 -> 1.2e-12, and the reason is
# worth writing down because it is the shape this number always had. What
# these bars measure is a FILL gap times a solve amplification, and #799 did
# not touch the fill gap: `_accel_vs_numpy` reads 1.316e-15 on this fixture
# both before and after, bit for bit. What moved is the amplification — the
# rewrite re-rolled the last bits of the basis coefficients (`P_minus_atom`
# and the interior Q's now carry their exact spellings), so the same 1e-15
# lands on Z through a slightly different path. Isolated by reverting the
# coefficient spellings alone, which puts free_wire back at 9.297e-13 with
# every kernel change still in. A bar with 7 % headroom over a
# condition-number product was going to move on the next such change whatever
# it was; 5e-12 is the headroom the other two in this class already carry.
_Z_AGREEMENT = {name: 1e-12 for name in EK_BATTERY}
_Z_AGREEMENT["skew_tee"] = 5e-12
_Z_AGREEMENT["grounded_monopole"] = 5e-12
_Z_AGREEMENT["free_wire"] = 5e-12


@pytest.mark.parametrize("name", list(EK_BATTERY))
def test_cpp_ek_impedance_matches_the_numpy_path(name, monkeypatch):
    """Kernel agreement is not the whole dispatch. Solving end to end both ways
    pins the gating/radius plumbing, the per-run stitching and the image-block
    routing too, not just the arithmetic inside one call. The junction
    geometries — `three_way`, `radius_step`, `grounded_ell_radii` — are the
    ones that exercise all three at once, and they sit at 2e-14 - 2e-13.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK:
        pytest.skip("C++ accelerator not built")

    z_cpp, _ = _ek_solver(name).compute_impedance()
    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_EK", False)
    z_ref, _ = _ek_solver(name).compute_impedance()
    rel = _rel(z_cpp, z_ref)
    assert rel < _Z_AGREEMENT[name], f"{name}: {rel:.3e} — {z_cpp} vs {z_ref}"


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


def test_ira_is_resolved_per_build_from_the_sources_it_is_handed():
    """The IRA mask belongs to the BUILD, not the geometry: it is a statement
    about where the observers sit relative to the SOURCE conductors, and the
    PEC image moves the sources. The deck below is a horizontal radius step 3 m
    up, so the thin wire's own centres sit on the fat stub's axis (every
    fat-source pair swaps) while the mirror images sit 6 m below it (none do).

    Neither backend is told which: both recompute the mask from the source
    centres/tangents they were handed — the numpy path in
    `_extended_kernel_fields`, the C++ path in stage A — which is exactly what
    lets `_field_tensor` serve both passes of NEC's KSYMP loop from one call
    shape. `test_cpp_ek_field_tensor_matches_numpy[...][pec-image]` pins that
    they agree; this pins that there is something to agree about.
    """
    sim = SinusoidalSolver(
        wavelength=LAM,
        wires=[
            np.array([[0.0, 0.0, 3.0], [2.0, 0.0, 3.0]]),
            np.array([[2.0, 0.0, 3.0], [4.0, 0.0, 3.0]]),
        ],
        n_per_edge_per_wire=[[5], [5]],
        wire_radius=[0.02, 0.005],
        nsegs=10,
        feed_arclength=1.0,
        junctions=[[(0, "end"), (1, "start")]],
        ground_z=0.0,
        extended_kernel=True,
    )
    geom = sim._build_geometry()
    free = _swap_mask(sim, geom, geom["seg_centers"], geom["seg_tangents"])
    image = _swap_mask(sim, geom, *sim._image_source_centers_tangents(geom))
    assert free.any() and not free.all(), "free-space build must mix both arms"
    assert not image.any(), "image build must reach its own (all-IRA=0) answer"


# ----------------------------------------------------------------------
# Gate 9 — the ground-contact branch of the extended-kernel gating
# (momwire#247, follow-up from #233/#244)
# ----------------------------------------------------------------------
#
# `_ek_gating`'s ground branch (f.2026-2029: NEC's ICON==J self-connection,
# CABJ²+SABJ² <= 1e-8) maps a PERPENDICULAR ground contact to IND=0 — the
# image continues the wire straight through, so the O(a²) expansion is as
# legitimate there as at an ordinary collinear junction. Gates 1-3 above
# never exercise it: nothing in the ladder or the bend/junction fixtures
# touches ground_z under extended_kernel=True. This section is the
# oracle-anchored complement to Gate 8's own ground-contacting monopole
# fixtures above (#245) — that side pins ind1[0]==0 against the C++ path;
# this pins the SAME code against nec2c.
#
# GEOMETRY: a vertical quarter-wave-ish monopole, base at the origin, tip
# at z=H=λ/4 (30 MHz, nec2c's CVEL — same convention as the ladder above),
# fed at the base segment, PEC ground at z=0. Thick wire — radius 0.09
# against a segment length of H/21 ≈ 0.119 m puts Δ/a ≈ 1.32, the same
# "EK matters" band the ladder's Δ/a ≈ 1.22 rung walks.
#
#     CM monopole EK=<on|off> ground_z=0.0 (momwire#247)
#     CE
#     GW 1 21 0. 0. 0. 0. 0. 2.4982704833333336 0.09
#     GE 1
#     GN 1 0 0 0 0 0
#     EK                      <- present only on the EK-ON run
#     FR 0 1 0 0 30. 0.
#     EX 0 1 1 0 1. 0.
#     XQ
#     EN
#
# GE vs GN (the issue's open question): they are independent cards, and
# BOTH are required to model a wire touching a perfect ground. GN 1 alone
# turns on the field image (KSYMP=2) for every segment, but does NOT mark a
# touching end as topologically continuous with it — that bookkeeping is
# GE's first field (NEC's GPFLAG, consumed by `conect(ignd)` in
# geometry.c), and without it the base segment's ground-plane end is still
# treated as an ordinary FREE end by the current expansion, sitting right
# on the conducting surface. That combination (GE 0 + GN 1) is a real,
# distinct nec2c deck, not a hypothetical failure mode — solving it on this
# geometry gives 51.6+12.0j, 6% low on R and half the reactance of the
# correct answer below. GE 1 alone (no GN card) leaves nec2c in free space
# (KSYMP defaults to 1 absent a GN card — main.c's GN handler is what sets
# KSYMP=2), so the ground flag's connectivity bookkeeping fires but nothing
# is being connected TO: 6.0-395j, indistinguishable from GE 0 with no GN
# at all (5.7-381j, both just free-space monopoles with an unphysical open
# base at the feed). Only GE 1 + GN 1 together reproduce momwire's
# `ground_z=0.0` PEC model (measured below); that is the deck this section
# captures.
#
# CAPTURED (nec2c-ubuntu-x86, VERSION 5b4az.ae6ty.1.23, this box):
#   EK-off  55.066 + 24.336j
#   EK-on   52.035 + 22.223j
#
# MEASURED on this box against those two:
#   momwire EK-off  54.900 + 24.162j   rel 0.40%
#   momwire EK-on   52.075 + 22.292j   rel 0.14%
# Gates below sit at ~2.5x those margins.
#
# DIRECTION / IMAGE-FILL: momwire's own EK-on shift (-2.825-1.870j) and
# nec2c's (-3.031-2.113j) agree in sign on both the real and imaginary
# parts, magnitude ratio 0.92 — comfortably inside a 2x band. This also
# answers the issue's image-fill question: `_field_components` builds one
# (src_a, ind1, ind2) EK table per SOURCE segment and reuses it unchanged
# for the mirrored image source (sinusoidal.py:1624-1638), matching NEC's
# EFLD, which passes one IND1/IND2 pair through both KSYMP passes. As a
# counterfactual, forcing the image-side fill back to the reduced kernel
# while leaving the real fill extended — giving the SAME continuous
# ground-junction current two different kernels on its two halves — was
# tried and breaks badly: 59.806-151.173j, sign-flipped reactance at 7x the
# correct magnitude. The gated, both-sides-extended choice the production
# code makes is the one that tracks nec2c; the split-kernel alternative is
# not just less accurate, it is a different, wrong model.

MONO_H = LAM / 4
MONO_NS = 21
MONO_A = 0.09
MONO_FEED = (MONO_H / MONO_NS) / 2  # base-segment centre

MONO_EK_OFF_TOL = 0.01
MONO_EK_ON_TOL = 0.005
MONO_SHIFT_RATIO_LO = 0.5
MONO_SHIFT_RATIO_HI = 2.0

# nec2c oracle, the GE 1 + GN 1 deck above.
MONO_NEC2C_OFF = complex(55.066, 24.336)
MONO_NEC2C_ON = complex(52.035, 22.223)


def _grounded_monopole(radius=MONO_A, z0=0.0, **kw):
    return SinusoidalSolver(
        wires=[np.array([[0.0, 0.0, z0], [0.0, 0.0, z0 + MONO_H]])],
        n_per_edge_per_wire=[[MONO_NS]],
        wavelength=LAM,
        wire_radius=radius,
        nsegs=MONO_NS,
        ground_z=0.0,
        feed_arclength=MONO_FEED,
        **kw,
    )


def test_gating_extends_the_ground_contact_end():
    """The branch under test: a segment perpendicular to the plane, one end
    genuinely touching it (`ground_minus`), gates IND=0 — extended, same as
    an ordinary collinear junction — while the free tip stays IND=1."""
    sim = _grounded_monopole(extended_kernel=True)
    geom = sim._build_geometry()
    assert geom["ground_minus"][0], "fixture must really touch the plane"
    ind1, ind2 = sim._ek_gating(geom)
    assert ind1[0] == 0, "ground contact end must extend, not fall back"
    assert ind2[-1] == 1, "free tip is unaffected"


def test_gating_ignores_a_grounded_wire_that_does_not_touch():
    """Same wire, lifted clear of the plane. Ground PRESENCE alone must not
    trip the branch — only genuine contact does, which is why f.2026-2029
    keys on ICON==J (a self-connection at zero clearance) and not on
    ground_z being set at all."""
    sim = _grounded_monopole(z0=0.5, extended_kernel=True)
    geom = sim._build_geometry()
    assert not geom["ground_minus"].any(), "fixture must not touch the plane"
    ind1, ind2 = sim._ek_gating(geom)
    assert ind1[0] == 1 and ind2[-1] == 1, "both ends free, nothing to gate"


def test_monopole_ek_off_matches_nec2c():
    z, _ = _grounded_monopole().compute_impedance()
    assert _rel(z, MONO_NEC2C_OFF) < MONO_EK_OFF_TOL, f"{z} vs {MONO_NEC2C_OFF}"


def test_monopole_ek_on_matches_nec2c():
    z, _ = _grounded_monopole(extended_kernel=True).compute_impedance()
    assert _rel(z, MONO_NEC2C_ON) < MONO_EK_ON_TOL, f"{z} vs {MONO_NEC2C_ON}"


def test_monopole_ek_shift_direction_matches_nec2c():
    """The physics-direction pin: turning EK on must move momwire's answer
    the same way it moves nec2c's, by a comparable amount — not merely land
    both endpoints inside separate tolerance windows independently."""
    z_off, _ = _grounded_monopole().compute_impedance()
    z_on, _ = _grounded_monopole(extended_kernel=True).compute_impedance()
    mom_shift = z_on - z_off
    nec_shift = MONO_NEC2C_ON - MONO_NEC2C_OFF
    assert mom_shift.real * nec_shift.real > 0, f"{mom_shift} vs {nec_shift}"
    assert mom_shift.imag * nec_shift.imag > 0, f"{mom_shift} vs {nec_shift}"
    ratio = abs(mom_shift) / abs(nec_shift)
    assert MONO_SHIFT_RATIO_LO < ratio < MONO_SHIFT_RATIO_HI, ratio


# ----------------------------------------------------------------------
# Gate 10 — the C++ extended-kernel Fresnel image tensor (momwire#259)
# ----------------------------------------------------------------------
#
# #245 gave the point-matched fill an EKSCX entry point, but left one block
# behind by design: `_field_tensor_image_refl`, the `ground_eps` (NEC IPERF=0)
# image, stayed reduced-kernel-only, so `extended_kernel=True` over finite
# ground took the pure-numpy `_field_components` path. #259 closes it with
# `sinusoidal_field_tensor_ek_refl` — EKSCX's E_z/E_ρ tables under the SAME
# Fresnel dyad tail the reduced refl kernel already applied.
#
# The claim under test is that the marriage is exactly that and nothing more:
# the dyad's four geometric factors (td, rho_proj, t_n·p̂, ρ̂·p̂) are EFLD's,
# computed before NEC picks between EKSC and EKSCX — in particular ρ̂ and
# rho_proj ride the UNSWAPPED radial distance RHX, never EKSCX's ordered RH.
# Get that wrong and the tensor is still plausible; the battery below is what
# says so.
#
# Gate shape is Gate 8's, for the same reason: an image block is a mirrored
# source on the far side of a plane, both kernels shed digits as kR grows, so
# the honest control on an ill-conditioned geometry is the REDUCED refl
# kernel's own C++-vs-numpy agreement on the same deck, not a looser constant.
# 1e-13, OR within `_CONDITIONING_HEADROOM` of the reduced kernel. Measured on
# this battery the extended kernel runs 0.19x - 3.2x of the reduced one and
# 2.5x / 9.6x at the 6 m and 20 m mirror planes of the control below — where
# the reduced kernel is itself at 2.7e-12 and 2.3e-11 and an absolute
# tolerance would only be measuring the geometry.
#
# Two ground constants, because ε̃ sets the whole tail: average soil, where
# ρ_v passes through the Brewster null and ρ_v + ρ_h is O(1) — the p̂
# correction is fully live — and sea water, three orders of magnitude more
# conductive, where the dyad is close to the PEC one it must degenerate into.

# (eps_r, sigma) — average soil and sea water.
REFL_EPS = {
    "avg_soil": (13.0, 0.005),
    "sea_water": (81.0, 5.0),
}

# momwire#282 stage 1 (2026-08-18) — read this before the fixtures.
#
# Three decks here (`monopole_contact`, `grounded_ell_radii`,
# `grounded_radius_step_skew`) stood IN the plane, which is ground CONTACT
# over a finite ground and is refused under `ground_model="refl-coef"` now
# (`docs/design/contact-over-finite-ground.md` §3.6: that row sat 27 Ω from
# the Sommerfeld answer on the same deck). Each keeps its NAME — the PEC
# battery above uses the same names for the same geometries, and the
# parallel is what makes the two readable side by side — but the PLANE has
# moved 1 cm below the wires' feet. What that costs and what it does not:
#
# * KEPT: the degenerate specular geometry, which is the reason these
#   fixtures exist. A 1 cm clearance against a 0.17 m segment leaves the
#   image an order of magnitude closer than one segment, so rho_axis is
#   still ~0 on the collinear decks, ρ̂·p̂ is still a-regularized noise, and
#   the IRA arm on the image build still fires with both swaps present.
# * LOST: NEC's ICON == J ground-contact arm of IND = 0, which no deck can
#   reach through the Fresnel image kernel any more, because reaching it
#   needs a contact and a refl-coef ground at once. IND = 0 ITSELF is still
#   covered, by its other arm (a collinear equal-radius two-segment
#   junction), and the contact arm is still covered by the PEC battery above
#   and by the Sommerfeld decks. `test_cpp_ek_refl_battery_is_decisive` says
#   all of that in place rather than quietly dropping an assertion.
# * LOST: momwire#292's contact-charge bracket on this battery, which is why
#   `test_cpp_ek_refl_path_does_not_re_enter_the_numpy_kernel` now expects
#   NO `_ek_end_gxx` call at all rather than exactly one.
REFL_EK_BATTERY = {
    # Vertical monopole GRAZING the plane: the specular geometry's degenerate
    # limit — the image is the collinear continuation, so rho_axis = 0 and
    # ρ̂·p̂ is pure a-regularized noise the dyad must not amplify.
    "monopole_contact": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])],
        n_per_edge_per_wire=[[15]],
        wire_radius=0.03,
        nsegs=15,
        feed_arclength=0.1,
        ground_z=-0.01,
    ),
    # The same monopole LIFTED clear of the plane: both ends IND = 1 now, and
    # the image sits a real distance away.
    "monopole_lifted": dict(
        wires=[np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 2.9]])],
        n_per_edge_per_wire=[[15]],
        wire_radius=0.03,
        nsegs=15,
        feed_arclength=1.25,
        ground_z=0.0,
    ),
    # Horizontal wire at height — the case the vertical fixtures cannot reach:
    # t_m·p̂ and t_n·p̂ are O(1), so the −(ρ_v+ρ_h)(t_m·p̂)(E·p̂) half of the dyad
    # carries real weight instead of cancelling.
    "horizontal_at_height": dict(
        wires=[np.array([[-2.5, 0.0, 1.5], [2.5, 0.0, 1.5]])],
        n_per_edge_per_wire=[[21]],
        wire_radius=0.05,
        nsegs=21,
        feed_arclength=2.5,
        ground_z=0.0,
    ),
    # Bent wire above the plane: IND = 2 (GX) on one end of each vertex
    # segment, and a p̂ that rotates pair by pair.
    "bent_above": dict(
        wires=[np.array([[0.0, 0.0, 0.3], [2.0, 0.0, 1.5], [3.6, 1.1, 1.5]])],
        n_per_edge_per_wire=[[6, 5]],
        wire_radius=0.08,
        nsegs=11,
        feed_arclength=1.0,
        ground_z=0.0,
    ),
    # A grazing base AND mixed radii: two observer-radius runs, so the (M, N)
    # specular tables have to slice on their observer axis in step with the
    # obs arrays and the EK tables have to NOT slice (they are source-indexed).
    "grounded_ell_radii": dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [1.6, 0.0, 2.0]]),
        ],
        n_per_edge_per_wire=[[9], [7]],
        wire_radius=[0.03, 0.008],
        nsegs=16,
        feed_arclength=0.15,
        ground_z=-0.01,
        junctions=[[(0, "end"), (1, "start")]],
    ),
    # EKSCX's IRA arm on the IMAGE build. A grazing fat/thin collinear stack
    # puts the thin wire's centres on the fat wire's axis, so the MIRRORED fat
    # sources swap too (rho_eval = 0.005 < src_a = 0.02); the skew wire beside
    # it is what stops the arm from cancelling out of Φ, exactly as in Gate 8's
    # `radius_step_skew`. `test_cpp_ek_refl_ira_arm_is_load_bearing` pins that.
    "grounded_radius_step_skew": dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
            np.array([[1.1, 0.6, 0.3], [2.3, 1.9, 1.4]]),
        ],
        n_per_edge_per_wire=[[5], [5], [4]],
        wire_radius=[0.02, 0.005, 0.005],
        nsegs=14,
        feed_arclength=1.0,
        ground_z=-0.01,
        junctions=[[(0, "end"), (1, "start")]],
    ),
    # A FAR mirror plane — the conditioning regime, in the battery rather than
    # only in the dedicated control below, so the routine per-fixture gate has
    # to survive it too.
    "dipole_far_plane": dict(
        wires=_DIPOLE_W,
        n_per_edge_per_wire=[[21]],
        wire_radius=0.05,
        nsegs=21,
        feed_arclength=2.5,
        ground_z=-6.0,
    ),
}


def _refl_solver(name, eps_name="avg_soil", **extra):
    return SinusoidalSolver(
        wavelength=LAM,
        ground_eps=REFL_EPS[eps_name],
        **REFL_EK_BATTERY[name],
        **extra,
    )


def _refl_accel_vs_numpy(sim):
    """Worst max-elementwise relative delta between the fused C++ Fresnel image
    tensor and its numpy reference, over the three returned tensors.
    `sim.extended_kernel` picks which accelerator flag is switched off to get
    the reference — the EK one leaves the EK-ON solve on the numpy
    `_field_components` + `_project_weighted` path, the reduced one leaves an
    EK-OFF solve there."""
    import momwire.sinusoidal as sin_mod

    geom = sim._build_geometry()
    cpp = sim._field_tensor_image_refl(geom, sim.k)
    flag = (
        "_HAVE_FIELD_TENSOR_EK_REFL"
        if sim.extended_kernel
        else "_HAVE_FIELD_TENSOR_REFL"
    )
    saved = getattr(sin_mod, flag)
    setattr(sin_mod, flag, False)
    try:
        ref = sim._field_tensor_image_refl(geom, sim.k)
    finally:
        setattr(sin_mod, flag, saved)
    worst, worst_label = 0.0, ""
    for label, a_cpp, a_ref in zip(("Phi_const", "Phi_sin", "Phi_cos"), cpp, ref):
        rel = np.max(np.abs(a_cpp - a_ref)) / max(np.max(np.abs(a_ref)), 1e-30)
        if rel > worst:
            worst, worst_label = rel, label
    return worst, worst_label


@pytest.mark.parametrize("name", list(REFL_EK_BATTERY))
@pytest.mark.parametrize("eps_name", list(REFL_EPS))
def test_cpp_ek_refl_field_tensor_matches_numpy(name, eps_name):
    """Every gating branch, both IRA arms, both radius dispatch shapes, both
    ground constants — against the numpy EK + Fresnel-dyad reference.

    See the section header for why the reduced refl kernel is the control on
    the ill-conditioned image blocks rather than a looser constant.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK_REFL:
        pytest.skip("C++ accelerator not built")

    ek, label = _refl_accel_vs_numpy(_refl_solver(name, eps_name, extended_kernel=True))
    if ek < ACCEL_AGREEMENT:
        return
    reduced, _ = _refl_accel_vs_numpy(
        _refl_solver(name, eps_name, extended_kernel=False)
    )
    assert ek < _CONDITIONING_HEADROOM * reduced, (
        f"{name}/{eps_name} {label}: EK {ek:.3e} misses {ACCEL_AGREEMENT:g} and "
        f"is {ek / max(reduced, 1e-300):.1f}x the reduced refl kernel's own "
        f"{reduced:.3e} on the same geometry"
    )


@pytest.mark.parametrize("ground_z", [-6.0, -20.0])
def test_cpp_ek_refl_far_image_conditioning_tracks_the_reduced_kernel(ground_z):
    """Gate 8's conditioning control, on the Fresnel tail. At 6 m and 20 m the
    reduced refl kernel is already well past 1e-13 and no absolute tolerance
    means anything; what the extended one owes is to stay within
    `_CONDITIONING_HEADROOM` of it on the same deck.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK_REFL:
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
            ground_eps=REFL_EPS["avg_soil"],
            extended_kernel=extended,
        )
        return _refl_accel_vs_numpy(sim)[0]

    reduced = _worst(False)
    extended = _worst(True)
    assert extended < max(_CONDITIONING_HEADROOM * reduced, _CONDITIONING_FLOOR), (
        f"EK {extended:.3e} vs reduced {reduced:.3e}"
    )


def test_cpp_ek_refl_ira_arm_is_load_bearing(monkeypatch):
    """The IRA arm on the IMAGE build must be worth getting right. It rewrites
    only the ρ-flavoured slots, so on a collinear deck the whole arm cancels
    out of Φ — the defect Gate 8 caught, here with the Fresnel tail on top,
    which pulls E_ρ in through a SECOND route (ρ̂·p̂) and could have masked it
    differently. The control is `_GlobalIra`, the pre-momwire#258 fill, for the
    reason spelled out in Gate 8's twin; only the numpy kernel can still be
    forced that way, and the fused Fresnel kernel is pinned to that numpy path
    at 1e-13 above.
    """
    import momwire.sinusoidal as sin_mod

    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_EK_REFL", False)

    sim = _refl_solver("grounded_radius_step_skew", extended_kernel=True)
    geom = sim._build_geometry()
    src_c, src_t = sim._image_source_centers_tangents(geom)
    swap = _swap_mask(sim, geom, src_c, src_t)
    assert swap.any() and not swap.all(), "fixture must mix both arms on the image"
    on = sim._field_tensor_image_refl(geom, sim.k)
    off = _GlobalIra(
        wavelength=LAM,
        ground_eps=REFL_EPS["avg_soil"],
        extended_kernel=True,
        **REFL_EK_BATTERY["grounded_radius_step_skew"],
    )._field_tensor_image_refl(geom, sim.k)
    moved = max(np.max(np.abs(a - b)) / np.max(np.abs(a)) for a, b in zip(on, off))
    assert moved > 1e-3, f"IRA arm moves the Fresnel image tensor by {moved:.3e}"


def test_cpp_ek_refl_battery_is_decisive():
    """The battery must reach every arm the C++ can take on the image build,
    asserted rather than trusted to the fixtures staying as written."""
    codes, swaps, uniform = set(), set(), set()
    multirun = False
    horizontal = False
    for name in REFL_EK_BATTERY:
        sim = _refl_solver(name, extended_kernel=True)
        geom = sim._build_geometry()
        ind1, ind2 = sim._ek_gating(geom)
        codes |= set(ind1.tolist()) | set(ind2.tolist())
        uniform.add(sim._uniform_radius is not None)
        multirun = multirun or len(sim._radius_runs(geom)) > 1
        # The Fresnel image block resolves its IRA on the MIRRORED sources.
        src_c, src_t = sim._image_source_centers_tangents(geom)
        swaps |= set(np.unique(_swap_mask(sim, geom, src_c, src_t)).tolist())
        # No `ground_minus` end can exist on this battery any more — see the
        # withdrawal note above `REFL_EK_BATTERY` — and the assertion below
        # states that as a fact rather than letting it pass unnoticed.
        assert not geom["ground_minus"].any(), (
            f"{name} contacts the plane: ground contact under refl-coef is "
            "refused (momwire#282 stage 1), so this deck cannot be built"
        )
        # A wire with a real horizontal run: t·p̂ is what makes the second half
        # of the dyad load-bearing, and every vertical fixture zeroes it.
        horizontal = horizontal or bool(
            np.max(np.abs(geom["seg_tangents"][:, :2])) > 0.5
        )
    # All three codes are still reached. What momwire#282 stage 1 removed is
    # narrower than a code: IND = 0 has two arms (`_ek_gating`'s docstring),
    # the collinear equal-radius two-segment junction and the perpendicular
    # GROUND CONTACT, and only the second is out of this kernel's reach now,
    # because reaching it needs a contact and a reflection-coefficient ground
    # at once. The collinear arm carries IND = 0 here, as it always did on
    # every multi-segment straight wire; the contact arm is still exercised
    # by the PEC battery above and by the Sommerfeld decks.
    assert codes == {0, 1, 2}, codes
    assert swaps == {False, True}, "EKSCX's IRA arm is uncovered on the image build"
    assert uniform == {False, True}, "both radius dispatch shapes are needed"
    assert multirun, "no fixture produces more than one observer-radius run"
    assert horizontal, "no fixture gives the p̂ half of the dyad any weight"


# Impedance-level agreement over finite ground, per fixture. 1e-12 is the gate.
# Measured across the battery: 3.8e-15 (bent_above), 5.4e-15
# (monopole_lifted), 7.2e-15 (dipole_far_plane), 1.5e-14
# (grounded_radius_step_skew), 6.6e-14 (grounded_ell_radii), 1.5e-12
# (horizontal_at_height), 1.7e-12 (monopole_contact).
#
# Two fixtures need 5e-12, and in both it is the metric, not dispatch.
# `horizontal_at_height` is the battery's worst-conditioned image block by a
# wide margin (the REDUCED refl kernel's own C++-vs-numpy delta on that deck
# is 3.5e-12 — larger than the extended kernel's 3.3e-12), so the matrix
# solve is multiplying a fill delta that has nothing to do with #259 by the
# deck's condition number. `monopole_contact` moved from 8.6e-13 to 1.7e-12
# under momwire#292 with the ABSOLUTE delta unchanged at 6.9e-11 Ω: the twin
# takes that deck's EK-on |Z| from 79.7 to 40.5 (28.56+74.42j, still walking
# with the reduced end-charge bracket, to 33.28+23.19j), so the same error
# is divided by half the number. cond(G) is 424 against the EK-off path's
# 400, and EK-OFF on this deck is bit-identical between the two paths.
#
# A dispatch fault — image block on the wrong kernel, unsliced specular
# table, gating dropped, IRA dropped — lands at 1e-2, not here.
_REFL_Z_AGREEMENT = {name: 1e-12 for name in REFL_EK_BATTERY}
_REFL_Z_AGREEMENT["horizontal_at_height"] = 5e-12
_REFL_Z_AGREEMENT["monopole_contact"] = 5e-12


@pytest.mark.parametrize("name", list(REFL_EK_BATTERY))
def test_cpp_ek_refl_impedance_matches_the_numpy_path(name, monkeypatch):
    """Kernel agreement is not the whole dispatch: solving end to end both ways
    pins the gating/radius plumbing, the per-run stitching of the sliced
    specular tables, and the fact that the free-space block still rides #245's
    kernel while only the image block changed."""
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK_REFL:
        pytest.skip("C++ accelerator not built")

    z_cpp, _ = _refl_solver(name, extended_kernel=True).compute_impedance()
    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_EK_REFL", False)
    z_ref, _ = _refl_solver(name, extended_kernel=True).compute_impedance()
    rel = _rel(z_cpp, z_ref)
    assert rel < _REFL_Z_AGREEMENT[name], f"{name}: {rel:.3e} — {z_cpp} vs {z_ref}"


def test_cpp_ek_refl_path_does_not_re_enter_the_numpy_kernel(monkeypatch):
    """The whole point of #259: an EK-ON finite-ground solve must never touch
    `_extended_kernel_fields`. Before it, the free-space block went to C++ and
    the Fresnel image block silently did not — the residue #245 left.

    The list must now be EMPTY. It used to hold exactly one `_ek_end_gxx`:
    momwire#292's contact-charge twin called it once per contact node per
    fill, and `grounded_ell_radii` had exactly one contact node. momwire#282
    stage 1 (2026-08-18) refused ground contact under
    `ground_model="refl-coef"` and the fixture grazes the plane instead, so
    there is no contact node and no bracket to build — which makes this the
    STRICTER statement of the two: any call at all is now a fault. A fill
    that fell back would show up as `_extended_kernel_fields` in the list.
    """
    import inspect

    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK_REFL:
        pytest.skip("C++ accelerator not built")

    calls = []
    for attr in ("_ek_end_gxx", "_ek_end_gx", "_extended_kernel_fields"):
        original = getattr(SinusoidalSolver, attr)

        def trip(*a, _attr=attr, _o=original, **kw):
            calls.append(_attr)
            return _o(*a, **kw)

        if isinstance(inspect.getattr_static(SinusoidalSolver, attr), staticmethod):
            # The end routines are staticmethods; a bare function in their
            # slot would be handed `self` and blow up before it counted.
            trip = staticmethod(trip)
        monkeypatch.setattr(SinusoidalSolver, attr, trip)

    _refl_solver("grounded_ell_radii", extended_kernel=True).compute_impedance()
    assert calls == [], f"EK-ON refl solve fell back to the numpy kernel: {calls}"


def test_ek_refl_kernel_is_never_entered_when_ek_is_off(monkeypatch):
    """The mirror claim, and the armor #259 owes the reduced path: with EK off
    the Fresnel image block must still go through `sinusoidal_field_tensor_refl`
    and never through the new entry point, whichever geometry it is."""
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK_REFL:
        pytest.skip("C++ accelerator not built")

    seen = []
    original = sin_mod._acc.sinusoidal_field_tensor_ek_refl

    def trip(*a, **kw):
        seen.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(sin_mod._acc, "sinusoidal_field_tensor_ek_refl", trip)
    for name in REFL_EK_BATTERY:
        _refl_solver(name, extended_kernel=False).compute_impedance()
    assert seen == [], "an EK-OFF refl solve entered the extended-kernel kernel"


# Four corners of each of the three EK-OFF Fresnel image tensors on the deck
# below. Hex floats so the comparison is exact.
#
# RE-ANCHORED 2026-08-18 by momwire#282 stage 1, and the provenance is worth
# spelling out because an exact pin is worth exactly its provenance. The
# original capture was made on the pre-#259 build (`git stash` on the #259
# diff) with the deck's two wires STANDING IN the plane, and every build
# since reproduced it bit for bit. That deck is ground contact over a
# reflection-coefficient ground and is refused now, so the plane moved 1 cm
# below the wires' feet — which changes the image geometry and therefore the
# tensor, unavoidably: the tensor is a function of the mirror plane's
# position.
#
# The chain was not broken, it was re-forged, and here is the link. Both
# decks were run through a SHADOW checkout of the branch point (the last
# commit before this change) and through this branch: the shadow reproduces
# the historical literals exactly on the old deck, and the shadow and this
# branch agree bit for bit on the new one. So these numbers are still the
# pre-#259 kernel's output — the kernel did not move, the deck did — and the
# gate keeps doing what it was written to do.
_EK_OFF_REFL_PIN = [
    (
        "0x1.ef751dcf79f6bp+7",
        "0x1.e0a30f96357efp+10",
        "0x1.2298bc3a95de5p-2",
        "0x1.539ef9dd8199ap-1",
    ),
    (
        "-0x1.e2cefbbfd104ap+2",
        "-0x1.d74efa4723db1p+5",
        "-0x1.cd8d6e3471e19p-14",
        "-0x1.6b9474b566f39p-10",
    ),
    (
        "0x1.eeeda4c3f96afp+7",
        "0x1.e01f716c67d70p+10",
        "0x1.225c38afeb2e0p-2",
        "0x1.53582f9cbb9a9p-1",
    ),
]


@pytest.mark.skipif(
    not (sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")),
    reason="the pin is a bit-capture of the x86-64/GCC contraction; other FP "
    "environments (macOS ARM clang, PR #529) contract the same source "
    "differently and the byte claim is meaningless there — the relative "
    "gates in this file carry the physics on every platform",
)
def test_ek_off_refl_tensor_is_unmoved_by_the_new_kernel():
    """Byte-level armor for the reduced Fresnel tail. #259 edits the file the
    reduced kernel lives in, so pin its output exactly rather than relatively:
    EK-OFF, C++ refl kernel on, against the pre-#259 capture.

    This is the gate that decided the shape of the C++ change. Factoring the
    dyad tail into a helper the two kernels share — the obvious spelling —
    moved this tensor by 3.6e-15 and a grounded Z by 7.1e-14, because the
    factors then reach the multiply-adds through a struct and GCC contracts
    them differently. Both numbers pass every relative tolerance in this file,
    which is exactly why the pin is exact and why the reduced kernel keeps an
    open-coded copy of the tail instead.

    The deck grazes the plane by 1 cm rather than standing in it since
    momwire#282 stage 1 (2026-08-18); `_EK_OFF_REFL_PIN` above carries the
    re-anchoring and the shadow-checkout chain that keeps it honest.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_REFL:
        pytest.skip("C++ refl kernel not built")

    sim = SinusoidalSolver(
        # 9.993 exactly, not LAM: this deck is the capture deck, reproduced
        # verbatim so the pin below is literally the pre-#259 output.
        wavelength=9.993,
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0], [1.6, 0.0, 2.0]]),
        ],
        n_per_edge_per_wire=[[9], [7]],
        wire_radius=[0.03, 0.008],
        nsegs=16,
        feed_arclength=0.15,
        ground_z=-0.01,
        ground_eps=(4.0, 0.2),
        junctions=[[(0, "end"), (1, "start")]],
        extended_kernel=False,
    )
    geom = sim._build_geometry()
    phi = sim._field_tensor_image_refl(geom, sim.k)
    got = [
        (
            float(t[0, 0].real).hex(),
            float(t[0, 0].imag).hex(),
            float(t[-1, 3].real).hex(),
            float(t[-1, 3].imag).hex(),
        )
        for t in phi
    ]
    assert got == [tuple(row) for row in _EK_OFF_REFL_PIN], (
        f"reduced refl kernel moved: {got}"
    )


def test_sommerfeld_ground_under_ek_rides_the_pec_kernel(monkeypatch):
    """The scope claim #259 rests on, asserted rather than reasoned about.

    Sommerfeld ground needs no Fresnel kernel of its own: `_assemble_Z`
    decomposes it (theory manual eqs 136-147) into the exact PEC image scaled
    by the scalar C₂ plus a smooth interpolated remainder, so its image block
    is `_field_tensor_image` — #245's kernel with mirrored sources — and the
    remainder is a grid-dyad term with its own kernels that the thin-wire
    kernel choice does not reach at all. So an EK-ON Sommerfeld solve must
    call `sinusoidal_field_tensor_ek` twice (free space + image), never the
    refl entry point, and never fall back to `_extended_kernel_fields`.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR_EK_REFL:
        pytest.skip("C++ accelerator not built")

    seen = []
    for kernel in ("sinusoidal_field_tensor_ek", "sinusoidal_field_tensor_ek_refl"):
        original = getattr(sin_mod._acc, kernel)

        def trip(*a, _name=kernel, _o=original, **kw):
            seen.append(_name)
            return _o(*a, **kw)

        monkeypatch.setattr(sin_mod._acc, kernel, trip)

    numpy_entries = []
    original_fields = SinusoidalSolver._extended_kernel_fields

    def trip_fields(*a, **kw):
        numpy_entries.append(1)
        return original_fields(*a, **kw)

    monkeypatch.setattr(SinusoidalSolver, "_extended_kernel_fields", trip_fields)

    SinusoidalSolver(
        wavelength=LAM,
        wires=[np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 2.5]])],
        n_per_edge_per_wire=[[9]],
        wire_radius=0.02,
        nsegs=9,
        feed_arclength=1.0,
        ground_z=0.0,
        ground_eps=REFL_EPS["avg_soil"],
        ground_model="sommerfeld",
        extended_kernel=True,
    ).compute_impedance()

    assert seen == ["sinusoidal_field_tensor_ek"] * 2, seen
    assert numpy_entries == [], "Sommerfeld + EK fell back to the numpy kernel"


def test_ek_refl_kernel_aborts_as_solve_aborted():
    """The new kernel polls `cancel_flag` like every other fused fill, so its
    abort must reach callers as the shared `SolveAborted` — which means being
    listed in `_accel._CANCELLABLE_KERNELS`. Python-level checkpoints are
    neutralized so the ONLY thing that can observe the tripped token is the
    C++ poll (the Phase-2 pattern of `test_cancel.py`).

    #245 left `sinusoidal_field_tensor_ek` off that tuple, so this covers the
    plain EK entry point too.
    """
    import momwire
    from momwire import CancelToken, SolveAborted

    if not momwire.accelerated:
        pytest.skip("C++ accelerator not built")

    for ground_eps in (None, REFL_EPS["avg_soil"]):
        token = CancelToken()
        token.cancel()
        sim = SinusoidalSolver(
            wavelength=LAM,
            wires=[np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 2.9]])],
            n_per_edge_per_wire=[[41]],
            wire_radius=0.02,
            nsegs=41,
            feed_arclength=1.25,
            ground_z=0.0,
            ground_eps=ground_eps,
            extended_kernel=True,
            cancel=token,
        )
        sim._checkpoint = lambda: None
        with pytest.raises(SolveAborted):
            sim.compute_impedance()


# ----------------------------------------------------------------------
# Gate 10 — EKSCX's IRA arm is chosen PER PAIR (momwire#258)
# ----------------------------------------------------------------------
#
# f.3186-3192 sets IRA inside EKSCX, once per SEGMENT PAIR, from that pair's
# own `RHX < BX`. momwire#245 could not: `_extended_kernel_fields` reduced the
# test to one `np.any` over the whole (M, N) grid before calling `_ek_end_gxx`,
# so a single observation point inside a source conductor put EVERY pair on the
# IRA==1 formula — evaluated with the unswapped (rh, b) of the pairs that did
# not swap, which is not the physics for those pairs. The accelerated kernel
# took a `want_swapped` scalar precisely to reproduce that faithfully rather
# than repair it silently. #258 repaired it on both sides; the scalar is gone
# from the C++ signature and the arm now rides the same per-pair comparison
# that orders (rh, b), which is the only spelling in which the two cannot
# disagree at a knife-edge pair.
#
# The defect was MASKED, which is why it survived #245's battery: the IRA arm
# rewrites only the ρ-flavoured slots (G2, G2P, G3), and the swap needs an
# observer inside a source conductor — which on the reachable decks means a
# stepped-radius COLLINEAR run, where the ρ-projection factor is identically
# zero and both arms give the same Φ. `test_collinear_radius_step_is_unmoved`
# below pins that masking as the real physics it is. It stops masking the
# moment a skew member joins the deck.
#
# ORACLE (nec2c 1.3.1, /usr/bin/nec2c, run once and pinned here so CI needs no
# binary). A fat stepped-radius stack — Δ/a = 1.33 on the fat wire, where EK is
# worth 13.5% to nec2c — with a skew member close enough that the 171 pairs
# which must NOT take the IRA==1 arm carry real weight:
#
#     CM radius step + skew: momwire#258 deciding deck
#     CE
#     GW 1 5 0. 0. 0. 0. 0. 2.0 0.30
#     GW 2 5 0. 0. 2.0 0. 0. 4.0 0.02
#     GW 3 6 0.6 0.35 0.2 2.1 1.9 2.4 0.02
#     GE 0
#     EK                      <- present only on the EK-ON run
#     FR 0 1 0 0 30. 0.
#     EX 0 1 3 0 1. 0.
#     XQ
#     EN
#
# Measured on this box:
#
#   nec2c        EK-off  38.655 - 35.438j     EK-on  41.287 - 42.147j
#   momwire      EK-off  38.079 - 35.892j     1.40% off the oracle
#   momwire per-pair EK-on 40.979 - 42.286j   0.57% off the oracle
#   momwire GLOBAL   EK-on 42.464 - 46.178j   7.12% off the oracle
#
# So the oracle does discriminate: the per-pair answer sits inside Gate 1's
# 1.7% bar and the global one misses it by 4x. On `radius_step_skew` the same
# repair moves the fill by 17.7% and Z by 0.26% — enough to matter, not enough
# for nec2c to arbitrate through the discretization gap, which is why the fat
# deck exists.
NEC2C_FAT_STEP_EK_OFF = 38.655 - 35.438j
NEC2C_FAT_STEP_EK_ON = 41.287 - 42.147j


def _fat_step_solver(**extra):
    return SinusoidalSolver(
        wavelength=LAM, **EK_BATTERY["radius_step_skew_fat"], **extra
    )


def test_per_pair_ira_matches_nec2c_where_the_global_arm_does_not(monkeypatch):
    """The deciding deck, both ways, against nec2c EK-on. This is the whole of
    momwire#258 in one assertion: the per-pair arm has to land inside the same
    bar Gate 1's ladder holds, AND the pre-#258 global arm has to miss it —
    otherwise the repair is unfalsifiable here and the fixture is decorative.
    """
    z_off, _ = _fat_step_solver(extended_kernel=False).compute_impedance()
    assert _rel(z_off, NEC2C_FAT_STEP_EK_OFF) < EK_ON_TOL, (
        f"EK-off control drifted: {z_off} vs {NEC2C_FAT_STEP_EK_OFF}"
    )
    z_on, _ = _fat_step_solver(extended_kernel=True).compute_impedance()
    per_pair = _rel(z_on, NEC2C_FAT_STEP_EK_ON)
    assert per_pair < EK_ON_TOL, f"per-pair IRA: {per_pair:.3%} — {z_on}"
    # The pre-#258 answer, reachable only through the numpy kernel now — the
    # accelerator has no argument left to force.
    import momwire.sinusoidal as sin_mod

    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_EK", False)
    z_global, _ = _GlobalIra(
        wavelength=LAM, extended_kernel=True, **EK_BATTERY["radius_step_skew_fat"]
    ).compute_impedance()
    glob = _rel(z_global, NEC2C_FAT_STEP_EK_ON)
    assert glob > EK_ON_TOL, (
        f"the global IRA arm answers {z_global} ({glob:.3%} off nec2c) — the "
        "deck no longer discriminates, so this gate proves nothing"
    )
    assert glob > 4.0 * per_pair, (
        f"per-pair {per_pair:.3%} vs global {glob:.3%}: the margin collapsed"
    )


def test_collinear_radius_step_is_unmoved_by_the_per_pair_arm():
    """The masking is real physics, so pin it. `radius_step` swaps on 25 of its
    100 pairs and every member is collinear, so the ρ-projection factor is
    identically zero and the ρ-flavoured slots the IRA arm rewrites cannot
    reach Φ. Per-pair and global must therefore agree BIT FOR BIT there — the
    reason the defect went unnoticed through #245, and the reason a fixture
    like this one can never be the gate.
    """
    import momwire.sinusoidal as sin_mod

    kw = EK_BATTERY["radius_step"]
    sim = SinusoidalSolver(wavelength=LAM, extended_kernel=True, **kw)
    geom = sim._build_geometry()
    swap = _swap_mask(sim, geom, geom["seg_centers"], geom["seg_tangents"])
    assert swap.any() and not swap.all(), "fixture must mix both arms"

    saved = sin_mod._HAVE_FIELD_TENSOR_EK
    sin_mod._HAVE_FIELD_TENSOR_EK = False
    try:
        per_pair = sim._assemble_Z(geom, sim.k)[0]
        g = _GlobalIra(wavelength=LAM, extended_kernel=True, **kw)
        gg = g._build_geometry()
        glob = g._assemble_Z(gg, g.k)[0]
    finally:
        sin_mod._HAVE_FIELD_TENSOR_EK = saved
    assert np.array_equal(np.asarray(per_pair), np.asarray(glob)), (
        "the collinear stepped-radius fill is supposed to be blind to the arm"
    )


@pytest.mark.parametrize("radius", [0.001, 0.05, 0.3])
def test_uniform_radius_never_reaches_the_ira_arm(radius):
    """The #233 ladder is untouched by all of this, and not by luck: with one
    radius everywhere, rho_eval = sqrt(rho_axis² + a²) >= a = src_a in IEEE
    (the sqrt of a correctly rounded square is exact) and EKSCX's test is
    strict. So the mask is empty, `_extended_kernel_fields` passes the scalar
    `False`, and the fill is the pre-#258 spelling operation for operation.
    """
    sim = _dipole(radius, extended_kernel=True)
    geom = sim._build_geometry()
    swap = _swap_mask(sim, geom, geom["seg_centers"], geom["seg_tangents"])
    assert not swap.any(), f"uniform radius {radius} swapped {swap.sum()} pairs"


def test_accelerator_declares_the_per_pair_ira_capability():
    """The EK entry points lost an argument in momwire#258, and a STALE
    extension still exports both symbols under the old arity — so `hasattr`
    alone would hand the new caller a TypeError instead of the numpy fallback
    the guards exist to give. `sinusoidal.py` requires the `ek_ira_per_pair`
    attribute before it claims either accelerator; if a rebuild drops the
    attribute the EK paths must go quiet, not wrong.
    """
    import momwire.sinusoidal as sin_mod
    from momwire._accel import acc

    if acc is None:
        pytest.skip("C++ accelerator not built")
    assert getattr(acc, "ek_ira_per_pair", False) is True
    assert sin_mod._HAVE_FIELD_TENSOR_EK and sin_mod._HAVE_FIELD_TENSOR_EK_REFL
