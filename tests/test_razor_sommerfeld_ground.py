"""RazorSolver's Sommerfeld ground (momwire#398 unit 5).

`docs/design/solver-architecture.md` §6.5. Units 2 and 4 gave `RazorSolver`
the two FOLDING grounds by consuming `PotentialGround`; this unit gives it
the COMPOSING one — `mode == "compose"`, the third row of the §2.2 table —
the same way, with this module still writing no ground physics of its own:

    Z = Z_free − (C₂·image + Q)

**NEC-5 is not the oracle here either, and for the same reason unit 4
recorded**: NEC-5's finite ground is Michalski and carries a limit offset
of its own, so comparing against it would measure that offset plus this
formulation's testing rule — the exact confound `RazorSolver` exists to
avoid. The binary is not run in this file and no golden module moves.

What the composing ground adds to unit 4's claim, in four parts:

1. **the two limits bracket it.** ε̃ → 1 must give back free space
   BIT-FOR-BIT (C₂ = 0 and the six λ-integrals are identically zero, so the
   ground layer collapses arithmetically rather than approximately), and
   ε̃ → ∞ must climb back onto the PEC image at the coefficient's own
   O(ε̃^{−1/2}) rate — the same rate unit 4 measured, because it is C₂'s
   rate and not the remainder's;
2. **the remainder is doing the work, and the number says so.** At 0.25 λ
   the two finite grounds agree to ~1.2 Ω, which is why this unit gates a
   LOW deck as well: 0.04 λ up, refl-coef and Sommerfeld differ by 22.8 Ω,
   and razor's split matches `BSplineSolver`'s to 0.03 Ω. A dead Q term
   would show up as a split near the refl-coef answer, not as a tolerance
   failure somewhere else;
3. **the cross-formulation bar is unit 4's**: the ground must not widen the
   razor-vs-Galerkin gap razor's own O(1/N) walk already opens in free
   space. Measured worst |Δgap| = 0.047 Ω against a 0.25 Ω bar, on three
   decks × two lanes × three reference formulations;
4. **the schedule is stricter than unit 4's.** Unit 4's weights are
   ω-dependent; the Sommerfeld grid is k-dependent in its lattice, its
   extent bucketing and its normalisation, so NOTHING of it may cross the
   prepare/replay boundary. `test_the_grid_is_built_at_every_solved_
   wavenumber` watches `_sommerfeld.get_grid` itself rather than inferring
   it from the answer, and the swept gate proves it end to end.

Measured (2026-08-17), N = 192, |Z_razor − Z_reference|, free space vs the
Sommerfeld ground, GL lane / NEC-5 lane:

    deck          reference       free space      sommerfeld
    dipole@0.25   BSplineSolver   0.910 / 0.928   0.949 / 0.968
                  Sinusoidal      0.817 / 0.836   0.853 / 0.871
                  SinusoidalGal   0.899 / 0.917   0.937 / 0.955
    invvee@0.3    BSplineSolver   0.490 / 0.712   0.491 / 0.709
                  Sinusoidal     12.674 /12.905  12.637 /12.867
                  SinusoidalGal   0.433 / 0.655   0.435 / 0.652
    dipole@0.04   BSplineSolver   0.910 / 0.928   0.864 / 0.882
                  Sinusoidal      0.817 / 0.836   0.776 / 0.793
                  SinusoidalGal   0.899 / 0.917   0.853 / 0.870

Every Sommerfeld column is its free-space column to within 0.047 Ω. The
inverted-V's 12.6 Ω against `SinusoidalSolver` is that solver's FREE-SPACE
behaviour on a bent three-anchor wire, unchanged by any ground — the same
loud case unit 4 recorded, and the same reason the gate is written on the
difference of the two columns.
"""

import numpy as np
import pytest

from momwire import (
    BSplineSolver,
    RazorSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
    _potential_ground,
    _sommerfeld,
)

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)
EPS = 13 - 0.03j
LOSSY = (13.0, 0.005)


def _dipole(height_wl):
    """A half-wave dipole at `height_wl` λ."""
    L, h = 0.478 * WL, height_wl * WL
    return dict(
        wires=[np.array([[0.0, -L / 2, h], [0.0, L / 2, h]])],
        split=lambda n: [[n]],
        radius=1.0262e-3,
    )


def _invvee():
    """An inverted-V whose centre of mass sits at 0.3 λ: a bend, a junction-
    free three-anchor wire, and two wings at different heights."""
    c = 0.19 * WL / np.sqrt(2.0)
    a = 0.3 * WL + c / 2
    return dict(
        wires=[np.array([[-c, 0.0, a - c], [0.0, 0.0, a], [c, 0.0, a - c]])],
        split=lambda n: [[n // 2, n // 2]],
        radius=1.0e-3,
    )


# The low deck is the unit's own addition to unit 4's pair, and it is not a
# stress test: the refl-coef ground's validity window is 0.1-0.5 λ
# (momwire#151) and the Sommerfeld ground has none, so 0.04 λ is where the
# two grounds are genuinely different physics rather than the same physics
# to three figures.
GEOMS = {
    "dipole@0.25": _dipole(0.25),
    "invvee@0.3": _invvee(),
    "dipole@0.04": _dipole(0.04),
}
SOMM = {"ground_z": 0.0, "ground_eps": EPS, "ground_model": "sommerfeld"}
GROUNDS = {
    "free": {},
    "refl": {"ground_z": 0.0, "ground_eps": EPS},
    "somm": SOMM,
}
REFERENCES = {
    "bspline": (BSplineSolver, {"degree": 2}),
    "sinusoidal": (SinusoidalSolver, {}),
    "sin-galerkin": (SinusoidalGalerkinSolver, {}),
}
LADDER = (24, 48, 96, 192)


def _solve(cls, gname, n, **extra):
    g = GEOMS[gname]
    z, _ = cls(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](n),
        wire_radius=g["radius"],
        wavelength=WL,
        **extra,
    ).compute_impedance()
    return complex(z)


def _Z(sim):
    return sim._assemble_Z(sim._build_geometry(), sim.k)


@pytest.fixture(scope="module")
def ladders():
    """Every gated point: three decks × three grounds × two razor lanes ×
    three reference formulations × the N ladder. Measured ~75 s, which is
    why every test that reads it is `slow`."""
    out = {}
    for gname in GEOMS:
        for ground, gkw in GROUNDS.items():
            for n in LADDER:
                out[(gname, ground, "razor-GL", n)] = _solve(
                    RazorSolver, gname, n, **gkw
                )
                out[(gname, ground, "razor-NEC5", n)] = _solve(
                    RazorSolver, gname, n, nec5_quadrature=True, **gkw
                )
                for rname, (cls, extra) in REFERENCES.items():
                    out[(gname, ground, rname, n)] = _solve(
                        cls, gname, n, **extra, **gkw
                    )
    return out


# --------------------------------------------------------------------------
# 1. the two limits, and the remainder in between
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lane", [False, True])
def test_the_epsilon_one_limit_is_free_space_bit_for_bit(lane):
    """ε̃ = 1 is not a ground, and the fill must produce that arithmetically.

    This is §6's gate (b) applied to the composing row: with ε̃ = 1 the
    image coefficient C₂ = (ε̃−1)/(ε̃+1) is exactly 0 and `_six_integrals`
    short-circuits the six λ-integrals to exact zeros, so BOTH halves of
    the composition are exactly zero and `Z_free − 0` is `Z_free` in
    float64. `allclose` would pass on a fill that computed a tiny non-zero
    ground and subtracted it; `array_equal` is what says the collapse is
    structural.
    """
    g = GEOMS["dipole@0.25"]
    base = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](12),
        wire_radius=g["radius"],
        wavelength=WL,
        nec5_quadrature=lane,
    )
    free = _Z(RazorSolver(**base))
    unity = _Z(
        RazorSolver(**base, ground_z=0.0, ground_eps=1.0, ground_model="sommerfeld")
    )
    assert np.array_equal(free, unity)


@pytest.mark.parametrize("lane", [False, True])
@pytest.mark.parametrize("loss", ["lossless", "lossy"])
def test_pec_limit_decays_smoothly_to_the_pec_image(lane, loss):
    """|Z(ε̃) − Z_PEC| falls monotonically as ε̃ grows, at C₂'s own rate.

    The composing ground reaches the PEC image from both terms at once:
    C₂ = (ε̃−1)/(ε̃+1) → 1 and the remainder → 0. The measured rate is
    O(ε̃^{−1/2}) — the same rate unit 4 measured on the Fresnel ground,
    because it is the coefficients' rate rather than the remainder's, and
    seeing it here is the evidence that BOTH terms are converging rather
    than one of them cancelling the other's error.

    Measured |Z(ε̃) − Z_PEC| at ε̃ = 10, 10², 10³, 10⁴, 10⁶ on the
    dipole@0.25 deck at N = 48, GL lane, lossless: 16.625, 6.3687, 2.1477,
    0.69348, 0.069959 Ω. Single-decade ratios 2.61, 2.97, 3.10 — climbing
    toward √10 from below — and the final two-decade step 9.91 ≈ √100. The
    NEC-5 lane tracks it to four figures and the lossy row differs only at
    ε̃ = 10 (15.646 for 16.625), where conduction is not yet negligible.
    """
    kw = dict(ground_z=0.0, nec5_quadrature=lane)
    z_pec = _solve(RazorSolver, "dipole@0.25", 48, **kw)
    seq = []
    for e in (1e1, 1e2, 1e3, 1e4, 1e6):
        eps = complex(e) if loss == "lossless" else (e, 0.005)
        z = _solve(
            RazorSolver,
            "dipole@0.25",
            48,
            ground_eps=eps,
            ground_model="sommerfeld",
            **kw,
        )
        seq.append(abs(z - z_pec))

    for a, b in zip(seq, seq[1:]):
        assert b < a, f"|Z(ε̃) − Z_PEC| is not monotone: {seq}"
    # The measured ε̃ = 1e6 point, at the level it was measured, +25 %.
    assert seq[-1] < 8.8e-2, seq
    assert seq[-1] > 1e-3, "suspiciously exact — is the ground reaching Z at all?"
    for a, b in zip(seq[:3], seq[1:4]):
        assert 2.3 < a / b < 3.4, f"decay rate is not O(ε̃^-1/2): {seq}"
    assert 8.5 < seq[-2] / seq[-1] < 11.5, seq


def test_the_remainder_is_worth_22_ohms_where_the_grounds_differ():
    """The positive gate on Q, at the height that can carry it.

    A composing ground whose remainder never reached Z would still pass a
    PEC-limit test (Q → 0 there), still pass an ε̃ → 1 test (Q = 0 there),
    and at 0.25 λ would sit only ~1.2 Ω from the reflection-coefficient
    answer — inside the noise a cross-formulation bar can absorb. So the
    gate is at 0.04 λ, BELOW the refl-coef ground's 0.1-0.5 λ validity
    window (momwire#151), where the two finite grounds are genuinely
    different physics.

    Measured at N = 48: refl-coef 48.918 − 21.346j, Sommerfeld
    71.112 − 16.087j — a 22.81 Ω split, of which the R shift is 22.2 Ω.
    And the split is not razor's own: `BSplineSolver(degree=2)` on the same
    deck splits by 22.84 Ω, i.e. the two formulations agree about what the
    ground did to within 0.03 Ω. At 0.25 λ the same comparison gives 1.211
    vs 1.218 Ω (0.006 Ω apart), which is the same statement made where
    there is little to say.
    """
    for deck, expect, bar in (("dipole@0.04", 22.81, 5.0), ("dipole@0.25", 1.21, 0.5)):
        z_refl = _solve(RazorSolver, deck, 48, ground_z=0.0, ground_eps=EPS)
        z_somm = _solve(RazorSolver, deck, 48, **SOMM)
        split = abs(z_refl - z_somm)
        assert split > bar, (
            f"{deck}: the two finite grounds differ by only {split:.4f} Ω — "
            "the remainder term is not reaching Z"
        )
        b_refl = _solve(BSplineSolver, deck, 48, degree=2, ground_z=0.0, ground_eps=EPS)
        b_somm = _solve(BSplineSolver, deck, 48, degree=2, **SOMM)
        assert abs(split - abs(b_refl - b_somm)) < 0.15, (
            f"{deck}: razor's refl→somm split is {split:.4f} Ω where the "
            f"Galerkin reference's is {abs(b_refl - b_somm):.4f} Ω"
        )
        assert abs(split - expect) < 0.25 * max(expect, 1.0)


# --------------------------------------------------------------------------
# 2. cross-formulation agreement
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("gname", list(GEOMS))
@pytest.mark.parametrize("lane", ["razor-GL", "razor-NEC5"])
@pytest.mark.parametrize("rname", list(REFERENCES))
def test_the_ground_adds_no_cross_formulation_gap(ladders, gname, lane, rname):
    """Unit 4's central claim, re-made for the composing ground.

    The question a ground unit answers is not "is the gap small" but "did
    the ground widen it". It did not: every Sommerfeld delta is its own
    free-space delta to within 0.047 Ω (measured worst, dipole@0.04 / NEC-5
    lane / sin-galerkin), against the same 0.25 Ω bar unit 4 used — and
    that is with the low deck in the set, where the ground itself moves the
    answer by 22 Ω.
    """
    n = LADDER[-1]
    d_free = abs(ladders[(gname, "free", lane, n)] - ladders[(gname, "free", rname, n)])
    d_somm = abs(ladders[(gname, "somm", lane, n)] - ladders[(gname, "somm", rname, n)])
    assert abs(d_somm - d_free) < 0.25, (
        f"{gname} {lane} vs {rname}: the ground moved the cross-formulation "
        f"gap from {d_free:.4f} to {d_somm:.4f} Ω"
    )


@pytest.mark.slow
@pytest.mark.parametrize("gname", list(GEOMS))
@pytest.mark.parametrize("lane", ["razor-GL", "razor-NEC5"])
def test_converged_deltas_against_the_galerkin_siblings(ladders, gname, lane):
    """The absolute numbers, pinned at the level they were measured.

    `BSplineSolver(degree=2)` and `SinusoidalGalerkinSolver` converge on
    the same answer as each other on all three decks, so they are the two
    an absolute bar can be written against. Measured worst |ΔZ| at N = 192
    over the Sommerfeld ground: 0.968 Ω (dipole@0.25, NEC-5 lane, bspline);
    best 0.435 Ω (inverted-V, GL lane, sin-galerkin). Bar 1.35 Ω, unit 4's
    — a regression guard, not a re-derivation.
    """
    n = LADDER[-1]
    z = ladders[(gname, "somm", lane, n)]
    for rname in ("bspline", "sin-galerkin"):
        d = abs(z - ladders[(gname, "somm", rname, n)])
        assert d < 1.35, f"{gname} {lane} vs {rname}: |ΔZ| = {d:.4f} Ω at N={n}"


@pytest.mark.slow
@pytest.mark.parametrize("gname", ["dipole@0.25", "dipole@0.04"])
@pytest.mark.parametrize("lane", ["razor-GL", "razor-NEC5"])
def test_the_dipole_ladders_walk_toward_the_galerkin_answer(ladders, gname, lane):
    """Both straight decks' Sommerfeld ladders close on `BSplineSolver`'s
    N = 192 answer monotonically, at the O(1/N) rate.

    Measured |Z_razor(N) − Z_bspline(192)|, GL lane: 5.207, 2.882, 1.650,
    0.949 Ω at 0.25 λ (ratios 1.81, 1.75, 1.74) and 4.692, 2.615, 1.501,
    0.864 at 0.04 λ (1.79, 1.74, 1.74) — the same walk, at the height where
    the ground contributes 22 Ω, which is the sharpest evidence available
    that the remainder converges with the mesh like everything else in the
    row rather than sitting on it as a constant.

    Not asserted on the inverted-V, for unit 4's recorded reason: there
    razor and the reference walk in opposite directions in R and X, so a
    monotone bar would pin a coincidence.
    """
    ref = ladders[(gname, "somm", "bspline", LADDER[-1])]
    seq = [abs(ladders[(gname, "somm", lane, n)] - ref) for n in LADDER]
    for a, b in zip(seq, seq[1:]):
        assert b < a, f"the ladder is not closing: {seq}"
    for a, b in zip(seq[:-1], seq[1:-1]):
        assert 1.5 < a / b < 2.5, f"not the O(1/N) rate: {seq}"


# --------------------------------------------------------------------------
# 3. the schedule: the grid is k-dependent, and it lives past the boundary
# --------------------------------------------------------------------------
def test_swept_sommerfeld_matches_the_per_k_solves():
    """`compute_impedance_swept` over a Sommerfeld ground == solving each k
    alone, bit for bit.

    Unit 4's swept gate had to sweep a `(eps_r, sigma)` ground to make ε̃
    move with ω; this one would fail on a fixed ε̃ too, because the
    interpolation grid is keyed to the WAVELENGTH — a build that hoisted
    the grid into `_assemble_Z_prepare` alongside the mirrored moments
    would return the sweep's nominal-k grid at every rung. The tuple ground
    is kept anyway, so both hazards are covered by one gate.
    """
    ks = 2 * np.pi / np.array([WL * 0.95, WL, WL * 1.05])
    g = GEOMS["dipole@0.25"]
    kw = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](16),
        wire_radius=g["radius"],
        ground_z=0.0,
        ground_eps=LOSSY,
        ground_model="sommerfeld",
    )
    swept = RazorSolver(wavelength=WL, **kw).compute_impedance_swept(ks)
    for i, k in enumerate(ks):
        one, _ = RazorSolver(wavelength=2 * np.pi / float(k), **kw).compute_impedance()
        assert swept[i] == one


def test_the_grid_is_built_at_every_solved_wavenumber(monkeypatch):
    """The schedule rule watched directly, not inferred from the answer.

    `_sommerfeld.get_grid` must be called once per SOLVED wavenumber, at
    that wavenumber — never once per fill, never at the constructor's k.
    The swept gate above would catch a frozen grid through the numbers;
    this catches it through the schedule, which is where it would be
    introduced (the prepare half is exactly where a mirrored-moment cache
    legitimately lives, and the grid is the one neighbour that may not join
    it).
    """
    seen = []
    real = _sommerfeld.get_grid

    def spy(eps_t, k2, r1_max, **kw):
        seen.append(float(k2))
        return real(eps_t, k2, r1_max, **kw)

    monkeypatch.setattr(_sommerfeld, "get_grid", spy)

    g = GEOMS["dipole@0.25"]
    kw = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](10),
        wire_radius=g["radius"],
        wavelength=WL,
        **SOMM,
    )
    ks = 2 * np.pi / np.array([WL * 0.97, WL, WL * 1.03])
    RazorSolver(**kw).compute_impedance_swept(ks)
    assert seen == [float(k) for k in ks], (
        f"grids built at {seen}, expected one per solved k at {list(ks)}"
    )

    seen.clear()
    RazorSolver(**kw).compute_impedance()
    assert seen == [2 * np.pi / WL]


def test_one_remainder_per_solved_wavenumber_and_never_the_galerkin_block(monkeypatch):
    """The call counts, and the one call that must never happen.

    One `Remainder` per solved wavenumber (it is built from the per-k
    ground object, which is what makes ε̃ and the grid per-k), one
    `field_windows` producer from it — the T1 observer set, since Q has no
    second observer set the way the fold's charge term does — and ZERO
    `evaluate` calls, ever. `evaluate` returns a B-spline Galerkin block;
    if razor ever reached it, the matrix would be wrong in a way no
    tolerance-based gate would name.
    """
    calls = {"remainder": 0, "windows": 0, "evaluate": 0}
    real_rem = _potential_ground.PotentialGround.remainder
    real_fw = _potential_ground.Remainder.field_windows
    real_ev = _potential_ground.Remainder.evaluate

    def spy_rem(self):
        out = real_rem(self)
        calls["remainder"] += out is not None
        return out

    def spy_fw(self, *a, **kw):
        calls["windows"] += 1
        return real_fw(self, *a, **kw)

    def spy_ev(self, *a, **kw):
        calls["evaluate"] += 1
        return real_ev(self, *a, **kw)

    monkeypatch.setattr(_potential_ground.PotentialGround, "remainder", spy_rem)
    monkeypatch.setattr(_potential_ground.Remainder, "field_windows", spy_fw)
    monkeypatch.setattr(_potential_ground.Remainder, "evaluate", spy_ev)

    g = GEOMS["dipole@0.25"]
    base = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](10),
        wire_radius=g["radius"],
        wavelength=WL,
    )
    ks = 2 * np.pi / np.array([WL * 0.98, WL, WL * 1.02])

    for ground, n_k, n_rem in (
        ({"ground_z": 0.0}, 1, 0),
        ({"ground_z": 0.0, "ground_eps": EPS}, 1, 0),
        (SOMM, 1, 1),
        (SOMM, 3, 3),
    ):
        for key in calls:
            calls[key] = 0
        sim = RazorSolver(**base, **ground)
        if n_k == 1:
            sim.compute_impedance()
        else:
            sim.compute_impedance_swept(ks)
        assert calls["remainder"] == n_rem, f"{ground}: {calls['remainder']} remainders"
        assert calls["windows"] == n_rem, f"{ground}: {calls['windows']} producers"
        assert calls["evaluate"] == 0, (
            "razor called the Galerkin convenience — its rows are path "
            "integrals, and that block is a different testing rule"
        )


def test_the_answer_does_not_depend_on_the_chunk_schedule():
    """Schedule invariance (architecture doc §3.1), on the composing fill.

    The remainder rides T1's observer windows, so `_WEIGHTED_CHUNK_ELEMS`
    slices it too — and a composing chunk holds `n_qp_sommerfeld` field
    samples per (observer, source segment) pair on top of the weight
    window. The chunk size is a memory decision and nothing else: the
    remainder is elementwise on the observer axis, so every schedule must
    give the identical matrix, bit for bit. Anything that leaked across a
    chunk boundary — a grid sized per chunk, most plausibly — would show up
    here first.
    """
    from momwire import razor as razor_mod

    g = GEOMS["dipole@0.04"]
    kw = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](20),
        wire_radius=g["radius"],
        wavelength=WL,
        **SOMM,
    )
    ref = _Z(RazorSolver(**kw))
    for elems in (1_000, 20_000, 40_000_000):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(razor_mod, "_WEIGHTED_CHUNK_ELEMS", elems)
            got = _Z(RazorSolver(**kw))
        assert np.array_equal(got, ref), f"chunk budget {elems} moved the matrix"


def test_the_quadrature_order_is_converged():
    """`n_qp_sommerfeld` is converged at its default, on the hardest deck.

    The remainder field is smooth on the scale of a segment by
    construction — its singular C₂-image part has been removed — so the
    default order 3 is not a compromise. Measured on the 0.04 λ deck at
    N = 48, where the term is worth 22 Ω: |Z(3) − Z(8)| = 3.6e-6 Ω, and
    even order 2 sits 4.3e-6 Ω away. Gated an order of magnitude looser
    than measured, since this is a convergence claim rather than a bit
    pin.
    """
    z3 = _solve(RazorSolver, "dipole@0.04", 48, n_qp_sommerfeld=3, **SOMM)
    z8 = _solve(RazorSolver, "dipole@0.04", 48, n_qp_sommerfeld=8, **SOMM)
    assert abs(z3 - z8) < 1e-4, f"|Z(3) − Z(8)| = {abs(z3 - z8):.3e} Ω"
    # ...and the knob is real: it must reach the fill at all.
    z1 = _solve(RazorSolver, "dipole@0.04", 48, n_qp_sommerfeld=1, **SOMM)
    assert z1 != z3


# --------------------------------------------------------------------------
# 4. the fill reads the object, not the strings
# --------------------------------------------------------------------------
def test_the_razor_fill_follows_the_object_not_the_strings(monkeypatch):
    """The structural row, in both directions, across the mode boundary.

    A solver configured for the reflection-coefficient ground, handed a
    COMPOSING `PotentialGround`, must fill the composed matrix — bit for
    bit, since the two reach the identical kernels by the identical
    schedule — and a solver configured for Sommerfeld, handed the folding
    one, must fill the folded matrix. This is a sharper swap than unit 4's:
    the two objects differ in `mode`, so a fill that read `ground_model`
    anywhere would produce a matrix with the wrong NUMBER OF TERMS rather
    than merely the wrong weights.
    """
    g = GEOMS["dipole@0.04"]
    base = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](14),
        wire_radius=g["radius"],
        wavelength=WL,
        ground_z=0.0,
        ground_eps=EPS,
    )
    refl_sim = RazorSolver(**base)
    somm_sim = RazorSolver(**base, ground_model="sommerfeld")
    Z_refl, Z_somm = _Z(refl_sim), _Z(somm_sim)
    assert not np.array_equal(Z_refl, Z_somm)

    def _swap(target, ground):
        monkeypatch.setattr(
            _potential_ground, "potential_ground_for", lambda *a, **kw: ground
        )
        out = _Z(target)
        monkeypatch.undo()
        return out

    from momwire import _ground_refl

    def _ground(sim, mode):
        # ω exactly as the k-dependent half spells it: `_assemble_Z` hands
        # `_assemble_Z_from_prepared` `self.c * k`, and c·(2π/λ) is not
        # bit-identical to `self.omega` = 2π·(c/λ). It makes no difference
        # to ε̃ (complex `ground_eps` ignores ω) but it does to the
        # Sommerfeld grid, whose values carry an ω·μ normalisation — a
        # 1-ULP ω moves Q in the last bit, which is exactly the resolution
        # this array_equal swap is written at.
        omega = sim.c * sim.k
        eps_t = _ground_refl.eps_tilde(EPS, omega, sim.eps)
        return _potential_ground.PotentialGround(
            sim,
            sim._build_geometry(),
            sim.k,
            omega,
            mode=mode,
            eps_tilde=eps_t,
            image_coefficient=(
                (eps_t - 1.0) / (eps_t + 1.0) if mode == "compose" else 1.0
            ),
            phi_mode=None if mode == "compose" else "normal",
        )

    assert np.array_equal(_swap(refl_sim, _ground(refl_sim, "compose")), Z_somm), (
        "the refl-coef solver ignored the composing ground it was handed — "
        "it is reading ground_model again"
    )
    assert np.array_equal(_swap(somm_sim, _ground(somm_sim, "fold")), Z_refl), (
        "the Sommerfeld solver ignored the folding ground it was handed"
    )


def test_the_consumer_never_applies_the_image_coefficient_itself():
    """The C₂-through-the-windows contract (`PotentialGround.image_
    coefficient`), on the ground that actually has a coefficient.

    Unit 4 could only pin this on a ground whose coefficient is 1, where a
    double application is invisible. Here C₂ ≠ 1 and reaches Z through the
    constant `(w_A, w_Φ)` windows — so the pin is a ground that LIES about
    its coefficient while its weights stay honest: report 2·C₂ from
    `image_coefficient`, leave `weight_windows` and `remainder` alone, and
    require the matrix not to move. A fill that multiplied by the attribute
    itself would double the exact-image half and nothing else, which is the
    exact bug `PotentialGround`'s docstring exists to prevent.
    """
    g = GEOMS["dipole@0.04"]
    kw = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](12),
        wire_radius=g["radius"],
        wavelength=WL,
        **SOMM,
    )
    sim = RazorSolver(**kw)
    ref = _Z(sim)

    class _CoefficientLiar:
        """Everything the real ground says, except the coefficient."""

        def __init__(self, inner):
            self._inner = inner
            self.image_coefficient = 2.0 * inner.image_coefficient

        def __getattr__(self, name):
            return getattr(self._inner, name)

    real = _potential_ground.potential_ground_for
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            _potential_ground,
            "potential_ground_for",
            lambda *a, **kwargs: _CoefficientLiar(real(*a, **kwargs)),
        )
        got = _Z(RazorSolver(**kw))
    assert np.array_equal(got, ref)


# --------------------------------------------------------------------------
# 5. the constructor's own contract, and parity with BSplineSolver
# --------------------------------------------------------------------------
def test_sommerfeld_requires_ground_eps_exactly_as_bspline_does():
    """Same condition, same wording, same exception type in both solvers:
    the exact ground is the exact ground OF something."""
    g = GEOMS["dipole@0.25"]
    for cls, extra in (
        (RazorSolver, {"n_per_edge_per_wire": g["split"](8)}),
        (BSplineSolver, {"nsegs": 8}),
    ):
        with pytest.raises(ValueError, match="requires ground_eps"):
            cls(
                wires=g["wires"],
                wire_radius=g["radius"],
                wavelength=WL,
                ground_z=0.0,
                ground_model="sommerfeld",
                **extra,
            )


def test_phi_mode_is_accepted_and_unread_exactly_as_bspline_does():
    """`ground_phi_mode` is a refl-coef knob, and the composing ground has
    no analogue of it — the Sommerfeld image coefficient is exact.

    `BSplineSolver` accepts the option and ignores it under
    `ground_model="sommerfeld"`; a solver that refused it instead, or
    (worse) let it move the answer, would be a second dialect of the shared
    option. Bit-identity across all four modes is the strong form of
    "ignored", and it comes for free from the object: the factory sets
    `phi_mode=None` for a composing ground, so there is nothing for the
    fill to read.
    """
    from momwire import _ground_refl

    g = GEOMS["dipole@0.04"]
    kw = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](12),
        wire_radius=g["radius"],
        wavelength=WL,
        **SOMM,
    )
    ref = None
    for mode in _ground_refl.PHI_MODES:
        sim = RazorSolver(**kw, ground_phi_mode=mode)
        assert (
            _potential_ground.potential_ground_for(
                sim, sim._build_geometry(), sim.k, sim.omega
            ).phi_mode
            is None
        )
        got = _Z(sim)
        if ref is None:
            ref = got
        assert np.array_equal(got, ref), f"{mode} moved the composed matrix"

    for bad in ("nope", "rho_h", None, "Normal"):
        with pytest.raises(ValueError, match="ground_phi_mode"):
            RazorSolver(**kw, ground_phi_mode=bad)
        with pytest.raises(ValueError, match="ground_phi_mode"):
            BSplineSolver(wires=g["wires"], nsegs=8, wavelength=WL, ground_phi_mode=bad)
