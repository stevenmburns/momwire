"""RazorSolver's per-wire radii (momwire#147) — the last cell of the row.

`wire_radius` takes the siblings' spelling: a scalar for every wire, or one
radius per wire. The reduced kernel's a² is regularised with the SOURCE
segment's radius (`RazorSolver._seg_moments_prepare`), so a junction whose
two arms have different radii needs no special case — the tent is two wings
on two segments, and each wing's moments were already built against its own
source column.

The gates, strongest first:

1. **a uniform model is bit-frozen.** `wire_radius=r` and
   `wire_radius=[r, r]` must be BIT-IDENTICAL, on every ground state and
   both quadrature lanes, matrix and swept — the scalar fast path is what
   keeps every pre-#147 answer unmoved.

2. **the NEC-5 twin lane, on decks that carry per-`GW` radii natively.**
   A `GW` card has its own radius, so a mixed-radius model is native on
   both sides and the comparison is like-for-like with no translation step
   for a convention to hide in. Two geometries — a thin driven dipole
   beside a FAT parasitic (different radii on wires that never touch) and
   one dipole whose inner half is fat and outer half thin, meeting AT the
   fed knot (different radii at a JUNCTION, where the perpendicular
   distance vanishes and a² is the whole of it) — each in free space and
   over `GN 1`, at momwire#398 unit 2's sharp bar. Measured worst
   |ΔZ| = 0.0586 Ω against a 0.20 Ω bar, offsets constant to 0.0211 Ω
   against 0.05 Ω.

3. **cross-formulation**, in the shape this formulation can honestly make.
   On the junction deck the ground contributes nothing to the
   razor-vs-BSpline gap (measured 0.063 Ω against a 0.25 Ω bar). On the
   parasitic deck the absolute number is NOT the instrument, because the
   two Galerkin references disagree with each other by 0.23 Ω over the
   same refl-coef ground whatever the radii are; what is gated there is
   the claim the capability actually makes — that a mixed radius moves the
   gap no more than swapping to a single radius does, i.e. the mixed
   deck's number is BRACKETED by its two uniform controls.

4. **the junction convention**, read structurally (each wing carries its
   own segment's radius) and gated on a convergence ladder against
   `BSplineSolver`.

**A finding about the references, on the record.** On the fat/thin STEP
both sinusoidal-family solvers are unconverged and walk AWAY: at N = 192
`SinusoidalGalerkinSolver` reads 113.8 Ω and `SinusoidalSolver` 100.8 Ω
where `BSplineSolver` reads 69.23, razor 69.06 and the NEC-5 binary 68.88 —
while the SAME deck with a uniform radius has all of them inside 0.1 Ω.
That is a property of those solvers' mixed-radius junction, not of this
unit, and it is why the sinusoidal family is recorded rather than gated on
that deck (momwire#435).
"""

import numpy as np
import pytest

from momwire import RazorSolver
from momwire.bspline import BSplineSolver
from golden_razor_mixed_radius_nec5 import (
    DIP_LEN,
    FAT,
    FREQ_MHZ,
    HI,
    MIXED_LADDERS,
    REFL_LEN,
    SPACING,
    THIN,
)

WL = 299792458.0 / (FREQ_MHZ * 1e6)
LANES = ({}, {"nec5_quadrature": True})
LANE_IDS = ("gl", "n5q")

# The refl-coef ground's validity window (momwire#151): 0.25 lambda up.
H_REFL = 5.35
EPS = 13.0 - 0.03j

# The junction spec the Galerkin references need and razor detects for
# itself (it takes no `junctions=` kwarg — that is `_OUT_OF_SCOPE`).
STEP_JUNCTION = [[(0, "end"), (1, "start")]]


# ----------------------------------------------------------------------
# decks
# ----------------------------------------------------------------------
def parasitic_wires(h):
    return [
        np.array([[0.0, -DIP_LEN / 2, h], [0.0, DIP_LEN / 2, h]]),
        np.array([[SPACING, -REFL_LEN / 2, h], [SPACING, REFL_LEN / 2, h]]),
    ]


def stepped_wires(h):
    """Fat inner half meeting thin outer half at the fed knot."""
    return [
        np.array([[0.0, -DIP_LEN / 2, h], [0.0, 0.0, h]]),
        np.array([[0.0, 0.0, h], [0.0, DIP_LEN / 2, h]]),
    ]


# The feed is named from wire 1's START rather than wire 0's END: they are
# the same knot, and every formulation here accepts that spelling (a
# B-spline's parameter domain does not contain its own upper endpoint).
STEP_FEED = dict(feed_wire_index=1, feed_arclength=0.0)

DECKS = {
    "parasitic": (parasitic_wires, [THIN, FAT], {}),
    "stepped": (stepped_wires, [FAT, THIN], STEP_FEED),
}


def _razor(deck, n, h, radii=None, *, ground=None, **kw):
    wires, default_radii, feed = DECKS[deck]
    g = dict(ground or {})
    return RazorSolver(
        wires=wires(h),
        n_per_edge_per_wire=[[n // 2], [n // 2]],
        wire_radius=default_radii if radii is None else radii,
        wavelength=WL,
        **feed,
        **g,
        **kw,
    )


def _reference(cls, deck, n, h, radii=None, *, ground=None):
    wires, default_radii, feed = DECKS[deck]
    extra = dict(junctions=STEP_JUNCTION) if deck == "stepped" else {}
    z, _ = cls(
        wires=wires(h),
        n_per_edge_per_wire=[[n // 2], [n // 2]],
        wire_radius=default_radii if radii is None else radii,
        wavelength=WL,
        **feed,
        **extra,
        **dict(ground or {}),
    ).compute_impedance()
    return complex(z)


# ----------------------------------------------------------------------
# 1. a uniform model is bit-frozen
# ----------------------------------------------------------------------

GROUND_STATES = {
    "free": dict(wires=[np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])]),
    "pec": dict(wires=[np.array([[0.0, 0.0, 5.5], [0.0, 10.0, 5.5]])], ground_z=0.0),
    "contact": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])],
        ground_z=0.0,
        feed_arclength=0.0,
    ),
    "refl": dict(
        wires=[np.array([[0.0, 0.0, 5.5], [0.0, 10.0, 5.5]])],
        ground_z=0.0,
        ground_eps=EPS,
    ),
    "sommerfeld": dict(
        wires=[np.array([[0.0, 0.0, 5.5], [0.0, 10.0, 5.5]])],
        ground_z=0.0,
        ground_eps=EPS,
        ground_model="sommerfeld",
    ),
}


@pytest.mark.parametrize("state", sorted(GROUND_STATES))
@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_uniform_sequence_is_bit_identical_to_the_scalar(state, lane):
    """The scalar fast path, on every ground state and both lanes. Nothing
    about a pre-#147 answer may move because the spelling changed."""
    kw = dict(GROUND_STATES[state], nsegs=24, wavelength=22.0, **lane)
    z_s, c_s = RazorSolver(wire_radius=1.0e-3, **kw).compute_impedance()
    z_a, c_a = RazorSolver(wire_radius=[1.0e-3], **kw).compute_impedance()
    # momwire#809: the two sides' fills measured BIT-IDENTICAL, so this
    # `==` is structural, not a solve-downstream lottery ticket.
    assert z_a == z_s
    np.testing.assert_array_equal(c_a, c_s)

    ks = 2 * np.pi / np.array([22.0 * 0.98, 22.0 * 1.03])
    zs_s = RazorSolver(wire_radius=1.0e-3, **kw).compute_impedance_swept(ks)
    zs_a = RazorSolver(wire_radius=[1.0e-3], **kw).compute_impedance_swept(ks)
    np.testing.assert_array_equal(zs_a, zs_s)


def test_a_uniform_model_keeps_the_scalar_kernel_argument():
    """Structurally, not just numerically: a uniform model hands the kernel
    one float, so the historical code path is the one that runs."""
    sim = RazorSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])],
        nsegs=8,
        wire_radius=[1.0e-3],
        wavelength=22.0,
    )
    geom = sim._build_geometry()
    assert sim._uniform_radius == 1.0e-3
    assert isinstance(sim._kernel_radius(geom), float)

    mixed = _razor("parasitic", 8, 0.0)
    gm = mixed._build_geometry()
    assert mixed._uniform_radius is None
    a = mixed._kernel_radius(gm)
    assert a.shape == (gm["seg_h"].size,)


# ----------------------------------------------------------------------
# 2. the NEC-5 twin lane on mixed-radius decks
# ----------------------------------------------------------------------


def _ladder_key(deck, ground):
    return f"{deck}-{ground}"


@pytest.mark.slow
@pytest.mark.parametrize("deck", sorted(DECKS))
@pytest.mark.parametrize("ground", ("free", "pec"))
def test_mixed_radius_matches_the_binary_at_the_sharp_bar(deck, ground):
    """momwire#398 unit 2's bar, on decks whose radii differ per `GW`."""
    rows = MIXED_LADDERS[_ladder_key(deck, ground)]
    h = 0.0 if ground == "free" else HI
    offsets = []
    for n, z_nec5, z_n5q_golden, _z_gl in rows:
        z, _ = _razor(
            deck,
            n,
            h,
            ground=dict(ground_z=0.0) if ground == "pec" else None,
            nec5_quadrature=True,
        ).compute_impedance()
        z = complex(z)
        # the golden's own razor column must still be reproducible, or the
        # captured NEC-5 numbers are being compared against a moved solver
        assert abs(z - z_n5q_golden) < 1e-6, f"N={n}: {z} vs {z_n5q_golden}"
        d = z - z_nec5
        bar = max(0.20, 0.0025 * abs(z_nec5))
        assert abs(d) <= bar, f"N={n}: |{d}| = {abs(d):.4f} > {bar:.4f}"
        offsets.append(d)
    spread_r = max(o.real for o in offsets) - min(o.real for o in offsets)
    spread_x = max(o.imag for o in offsets) - min(o.imag for o in offsets)
    assert spread_r <= 0.05 and spread_x <= 0.05, (spread_r, spread_x)


# ----------------------------------------------------------------------
# 3. cross-formulation
# ----------------------------------------------------------------------

N_CROSS = 192
CROSS_BAR = 0.25


@pytest.mark.slow
@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_the_radii_add_no_cross_formulation_gap_at_a_junction(lane):
    """Units 4/5's difference-of-columns, on the fat/thin junction deck:
    the razor-vs-BSpline gap over the refl-coef ground must be the
    free-space gap, which is razor's own O(1/N) walk and nothing else.

    `SinusoidalGalerkinSolver` is deliberately not a reference here — see
    this module's docstring for the measured reason (it is the unconverged
    party on a radius step, by 45 Ω)."""
    gaps = {}
    for tag, ground in (("free", None), ("refl", dict(ground_z=0.0, ground_eps=EPS))):
        z_r, _ = _razor(
            "stepped", N_CROSS, H_REFL, ground=ground, **lane
        ).compute_impedance()
        z_b = _reference(BSplineSolver, "stepped", N_CROSS, H_REFL, ground=ground)
        gaps[tag] = complex(z_r) - z_b
    assert abs(gaps["refl"] - gaps["free"]) <= CROSS_BAR, gaps


@pytest.mark.slow
def test_a_mixed_radius_costs_no_more_gap_than_a_single_one():
    """The parasitic deck's absolute difference-of-columns is a property of
    the REFERENCES on this geometry (the two Galerkin solvers disagree with
    each other by 0.23 Ω over the same ground, at any radius), so what is
    gated is the claim per-wire radii actually make: the mixed model sits
    between its own two uniform controls."""

    def doc(radii):
        gaps = []
        for ground in (None, dict(ground_z=0.0, ground_eps=EPS)):
            z_r, _ = _razor(
                "parasitic", N_CROSS, H_REFL, radii, ground=ground
            ).compute_impedance()
            z_b = _reference(
                BSplineSolver, "parasitic", N_CROSS, H_REFL, radii, ground=ground
            )
            gaps.append(complex(z_r) - z_b)
        return abs(gaps[1] - gaps[0])

    thin, fat = doc([THIN, THIN]), doc([FAT, FAT])
    mixed = doc([THIN, FAT])
    lo, hi = min(thin, fat), max(thin, fat)
    assert lo <= mixed <= hi, (thin, mixed, fat)
    # ...and the whole family is in the same class as the single-radius
    # decks units 4/5 gated, rather than an order out.
    assert hi <= 0.30, (thin, mixed, fat)


# ----------------------------------------------------------------------
# 4. the junction convention
# ----------------------------------------------------------------------


def test_each_junction_wing_carries_its_own_segments_radius():
    """The convention, read off the arrays rather than inferred from an
    answer: the junction tent's two wings sit on segments of different
    wires, and `_seg_radius` gives each its own wire's radius. Nothing in
    the fill special-cases the junction — this IS the special case."""
    sim = _razor("stepped", 12, 0.0)
    geom = sim._build_geometry()
    seg_a = sim._seg_radius(geom)
    assert seg_a.shape == (geom["seg_h"].size,)
    assert np.array_equal(np.unique(seg_a), np.unique([THIN, FAT]))

    # the basis whose two wings straddle the junction
    wing_seg = geom["wing_seg"]
    straddling = [
        b
        for b in range(wing_seg.shape[0])
        if seg_a[wing_seg[b, 0]] != seg_a[wing_seg[b, 1]]
    ]
    assert len(straddling) == 1, straddling
    b = straddling[0]
    assert {seg_a[wing_seg[b, 0]], seg_a[wing_seg[b, 1]]} == {FAT, THIN}


@pytest.mark.slow
def test_the_junction_deck_converges_onto_the_galerkin_reference():
    """Razor's own O(1/N) walk, on a deck with a radius step at the
    junction: the gap to `BSplineSolver` must shrink monotonically, which
    is what says the step is modelled rather than merely tolerated."""
    gaps = []
    for n in (48, 96, 192):
        z_r, _ = _razor("stepped", n, H_REFL).compute_impedance()
        z_b = _reference(BSplineSolver, "stepped", n, H_REFL)
        gaps.append(abs(complex(z_r) - z_b))
    assert gaps[0] > gaps[1] > gaps[2], gaps
    assert gaps[-1] < 1.1, gaps


# ----------------------------------------------------------------------
# 5. every other consumer of the radius
# ----------------------------------------------------------------------


def test_loading_evaluates_at_each_wires_own_radius():
    """`_radius_per_wire` is a real per-wire array now, so the skin-effect
    model sees the fat wire's radius on the fat wire: the fatter conductor
    must show the smaller per-unit-length loss resistance."""
    from momwire import _wire_loading

    sim = _razor("parasitic", 12, 0.0, wire_conductivity=5.8e7)
    zw = _wire_loading.loading_for(sim, sim.omega).z_wire
    assert zw.shape == (2,)
    # wire 0 is THIN, wire 1 is FAT
    assert zw[1].real < zw[0].real
    for w, rad in enumerate([THIN, FAT]):
        assert zw[w] == complex(
            _wire_loading.wire_internal_impedance(sim.omega, rad, 5.8e7)
        )
    z, coeffs = sim.compute_impedance()
    p_tot, p_wire = sim.wire_loss_power(coeffs)
    assert p_tot > 0.0 and p_wire.shape == (2,)
    assert p_tot == pytest.approx(p_wire.sum())


def test_insulation_validation_uses_each_wires_own_radius():
    """A jacket that clears the thin wire but not the fat one is refused,
    naming the conductor radius it failed against."""
    with pytest.raises(ValueError, match="must exceed the conductor"):
        _razor("parasitic", 8, 0.0, insulation_radius=5.0e-3, insulation_eps_r=3.0)
    _razor("parasitic", 8, 0.0, insulation_radius=2.0e-2, insulation_eps_r=3.0)


def test_the_radius_column_is_chunk_invariant(monkeypatch):
    """The fill chunks the OBSERVER axis and the radius column is indexed by
    SOURCE, so every chunk sees it whole: shrinking the chunk budget until
    the fill is split many ways must not move one bit. (The observer-radius
    convention would have to be sliced with the chunk — one of the two
    reasons the source reading is taken.)"""
    import momwire.razor as mod

    z_ref, c_ref = _razor("stepped", 24, 0.0).compute_impedance()
    monkeypatch.setattr(mod, "_CHUNK_ELEMS", 500)
    z_chunked, c_chunked = _razor("stepped", 24, 0.0).compute_impedance()
    # momwire#809: the two sides' fills measured BIT-IDENTICAL, so this
    # `==` is structural, not a solve-downstream lottery ticket.
    assert z_chunked == z_ref
    np.testing.assert_array_equal(c_chunked, c_ref)


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_mixed_radius_swept_matches_the_per_k_solves(lane):
    """The radius rides the k-independent prepare half (it is geometry), so
    a sweep must reproduce the single solves exactly as a uniform one does."""
    ks = 2 * np.pi / np.array([WL * 0.97, WL, WL * 1.04])
    swept = _razor("parasitic", 16, 0.0, **lane).compute_impedance_swept(ks)
    for i, kk in enumerate(ks):
        sim = _razor("parasitic", 16, 0.0, **lane)
        sim.wavelength = 2 * np.pi / kk
        sim.k = float(kk)
        sim.omega = sim.k * sim.c
        z_i, _ = sim.compute_impedance()
        assert swept[i] == pytest.approx(complex(z_i), rel=1e-12)


@pytest.mark.parametrize("state", ("pec", "refl", "sommerfeld"))
def test_the_radii_reach_the_image_block_too(state):
    """An image segment keeps its own segment's radius, so swapping the two
    wires' radii must move a GROUNDED answer as well as a free-space one —
    otherwise the ground fill would be quietly using one radius."""
    ground = {
        "pec": dict(ground_z=0.0),
        "refl": dict(ground_z=0.0, ground_eps=EPS),
        "sommerfeld": dict(ground_z=0.0, ground_eps=EPS, ground_model="sommerfeld"),
    }[state]
    z_a, _ = _razor(
        "parasitic", 12, H_REFL, [THIN, FAT], ground=ground
    ).compute_impedance()
    z_b, _ = _razor(
        "parasitic", 12, H_REFL, [FAT, THIN], ground=ground
    ).compute_impedance()
    assert abs(complex(z_a) - complex(z_b)) > 1.0
