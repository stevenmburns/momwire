"""RazorSolver's reflection-coefficient ground (momwire#398 unit 4).

`docs/design/solver-architecture.md` §6.4. Unit 2 gave `RazorSolver` the PEC
image by consuming `PotentialGround`; this unit gives it the finite one the
same way — `ground_eps` for wires standing CLEAR of the plane, weighted per
(observer, source segment) pair by `PotentialGround.weight_windows`, with
this module still writing no reflection coefficient of its own.

**NEC-5 is not the oracle here, and that is a statement about the physics
rather than about effort.** NEC-5's finite ground is Michalski, which
carries a limit offset of its own; comparing against it would measure that
offset plus this formulation's testing rule, which is exactly the confound
`RazorSolver` exists to avoid (see the `ground_eps` docstring, and the PEC
lane in `tests/test_razor_pec_ground.py` for what an oracle-gated ground
looks like when there IS a shared exact object to agree about). The binary
is not run in this file, and no golden module moves.

What replaces it is a CROSS-FORMULATION claim, in three parts:

1. **the PEC limit** — ε̃ → ∞ collapses the Fresnel dyad and the Φ weight
   back onto the PEC image, smoothly and at the rate the coefficients
   themselves decay (ρ_v − 1 = O(ε̃^{-1/2}), so a decade of ε̃ buys √10);
2. **agreement with momwire's other formulations** — a half-wave dipole at
   0.25 λ and an inverted-V at 0.3 λ, both in the refl-coef ground's
   0.1–0.5 λ validity window (momwire#151), against `BSplineSolver`,
   `SinusoidalSolver` and `SinusoidalGalerkinSolver` on convergence
   ladders to N = 192. The sharp form of that claim is the one this
   formulation can actually make: **the ground contributes nothing to the
   cross-formulation gap.** Razor's disagreement with a Galerkin solver
   over `ground_eps` is the same number as its disagreement in free space
   — its own O(1/N) razor-blade walk, the thing the twin exists to
   exhibit — and `test_the_ground_adds_no_cross_formulation_gap` pins that
   difference-of-differences rather than the gap itself;
3. **the schedule** — the weights are ω-dependent, so they are built in
   the k-dependent half of the fill and rebuilt at every wavenumber of a
   sweep, which `test_swept_refl_coef_matches_the_per_k_solves` proves by
   sweeping a `(eps_r, sigma)` ground whose ε̃ MOVES with ω.

Measured (2026-08-17), |Z_razor − Z_reference| at N = 192, GL lane / NEC-5
lane, on the dipole@0.25λ deck with ε̃ = 13 − 0.03j:

    reference        free space        PEC          refl-coef
    BSplineSolver   0.910 / 0.928  0.984 / 1.002  0.951 / 0.970
    Sinusoidal      0.817 / 0.836  0.884 / 0.903  0.947 / 0.965
    SinusoidalGal   0.899 / 0.917  0.971 / 0.989  1.031 / 1.050

and on the inverted-V@0.3λ deck:

    BSplineSolver   0.490 / 0.712  0.489 / 0.703  0.488 / 0.706
    Sinusoidal     12.674 /12.905 12.583 /12.813 12.615 /12.845
    SinusoidalGal   0.433 / 0.655  0.434 / 0.646  0.427 / 0.642

Every refl-coef column is its free-space column to within 0.133 Ω (worst,
dipole/GL/SinusoidalGalerkin), which is the number the gate carries. The
inverted-V's 12.6 Ω against `SinusoidalSolver` is a FREE-SPACE property of
that solver on a bent three-anchor wire, unchanged by any ground — which is
precisely why the gate is on the difference of the two columns and not on
either column alone.

That 12.6 Ω is now explained, and it is not ours (momwire#421). It is NEC-2's
own: measured against `nec2c 1.3` on this geometry, our point-matched row
tracks nec2c to 0.088-0.096 Ω at every included angle from 180° down to 60°
— the same residual a straight wire shows — while nec2c and the other three
formulations separate by the full 12.6 Ω. Nor does it converge away: at 90°
the sibling formulations land within 0.27 Ω of each other by N=768 while the
NEC-2 scheme turns around after N≈96 and walks the other way, ours with it.
`SinusoidalGalerkinSolver` is the same basis on the same mesh and stays with
the siblings, so the discriminator is the testing rule, exactly as #573 found
around a loop. `tests/test_sinusoidal_bend_nec2_twin.py` pins it.
"""

import numpy as np
import pytest

from momwire import (
    BSplineSolver,
    RazorSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
    _potential_ground,
)

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)
EPS = 13 - 0.03j
LOSSY = (13.0, 0.005)


def _dipole():
    """A half-wave dipole at 0.25 λ — mid-window for the refl-coef ground's
    0.1–0.5 λ validity band (momwire#151)."""
    L, h = 0.478 * WL, 0.25 * WL
    return dict(
        wires=[np.array([[0.0, -L / 2, h], [0.0, L / 2, h]])],
        split=lambda n: [[n]],
        radius=1.0262e-3,
    )


def _invvee():
    """An inverted-V whose centre of mass sits at 0.3 λ: a bend, a junction-
    free three-anchor wire, and two wings at different heights, so every
    pair in the fill sees its own specular angle."""
    c = 0.19 * WL / np.sqrt(2.0)
    a = 0.3 * WL + c / 2
    return dict(
        wires=[np.array([[-c, 0.0, a - c], [0.0, 0.0, a], [c, 0.0, a - c]])],
        split=lambda n: [[n // 2, n // 2]],
        radius=1.0e-3,
    )


GEOMS = {"dipole@0.25": _dipole(), "invvee@0.3": _invvee()}
GROUNDS = {
    "free": {},
    "pec": {"ground_z": 0.0},
    "refl": {"ground_z": 0.0, "ground_eps": EPS},
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


@pytest.fixture(scope="module")
def ladders():
    """Every gated point: two decks × three grounds × two razor lanes ×
    three reference formulations × the N ladder. Measured ~70 s, which is
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
# 1. the PEC limit
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lane", [False, True])
@pytest.mark.parametrize("loss", ["lossless", "lossy"])
def test_pec_limit_decays_smoothly_to_the_pec_image(lane, loss):
    """|Z(ε̃) − Z_PEC| falls monotonically as ε̃ grows, at the coefficients'
    own rate.

    ρ_v = (ε̃cosθ − √(ε̃−sin²θ))/(ε̃cosθ + √(ε̃−sin²θ)) → 1 − 2/(√ε̃ cosθ)
    and "normal"'s w_Φ = (√ε̃−1)/(√ε̃+1) → 1 − 2/√ε̃, so the residual is
    O(ε̃^{-1/2}) and a DECADE of ε̃ buys a factor √10 ≈ 3.16, not 10. That
    rate is the evidence the limit is being approached by the physics
    rather than by a tolerance.

    Measured |Z(ε̃) − Z_PEC| at ε̃ = 10, 10², 10³, 10⁴, 10⁶ on the
    dipole@0.25 deck at N = 48, GL lane, lossless: 15.318, 5.876, 1.991,
    0.6441, 0.06503 Ω. The three single-decade ratios are 2.61, 2.95, 3.09
    — climbing toward √10 from below, exactly as the expansion says — and
    the last step spans two decades, so its ratio is 9.90 ≈ 10 = √100.
    The lossy row (13, 0.005) differs only at ε̃ = 10, where the conduction
    term is not yet negligible: 14.462 in place of 15.318. The NEC-5 lane
    tracks the GL one to four figures at every rung. Pinned at the ε̃ = 10⁶
    end: 6.50e-2 Ω on an |Z| ≈ 82 Ω deck, i.e. 7.9e-4 relative.

    The lossy row uses (eps_r, sigma) rather than a complex ε̃, so ε̃ is
    ω-dependent — which is the arm of the fill this whole unit had to move
    into the k-dependent half.
    """
    kw = dict(ground_z=0.0, nec5_quadrature=lane)
    z_pec = _solve(RazorSolver, "dipole@0.25", 48, **kw)
    seq = []
    for e in (1e1, 1e2, 1e3, 1e4, 1e6):
        eps = complex(e) if loss == "lossless" else (e, 0.005)
        z = _solve(RazorSolver, "dipole@0.25", 48, ground_eps=eps, **kw)
        seq.append(abs(z - z_pec))

    for a, b in zip(seq, seq[1:]):
        assert b < a, f"|Z(ε̃) − Z_PEC| is not monotone: {seq}"
    # The measured ε̃ = 1e6 point, at the level it was measured, +25 %.
    assert seq[-1] < 8.2e-2, seq
    assert seq[-1] > 1e-3, "suspiciously exact — is the weighting reaching Z at all?"
    # ...the single-decade ratios are √10-ish, not decades...
    for a, b in zip(seq[:3], seq[1:4]):
        assert 2.3 < a / b < 3.4, f"decay rate is not O(ε̃^-1/2): {seq}"
    # ...and the two-decade step at the end is its square.
    assert 8.5 < seq[-2] / seq[-1] < 11.5, seq


def test_the_finite_ground_actually_moves_the_answer():
    """The cheap belt-and-braces the limit test cannot give: a realistic
    ground is far from both PEC and free space, on both lanes. Measured on
    the dipole@0.25 deck at N = 48: |Z_refl − Z_PEC| = 13.7 Ω,
    |Z_refl − Z_free| = 16.2 Ω."""
    for lane in (False, True):
        kw = dict(nec5_quadrature=lane)
        z_free = _solve(RazorSolver, "dipole@0.25", 48, **kw)
        z_pec = _solve(RazorSolver, "dipole@0.25", 48, ground_z=0.0, **kw)
        z_ref = _solve(
            RazorSolver, "dipole@0.25", 48, ground_z=0.0, ground_eps=EPS, **kw
        )
        assert abs(z_ref - z_pec) > 5.0
        assert abs(z_ref - z_free) > 5.0


# --------------------------------------------------------------------------
# 2. cross-formulation agreement
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("gname", list(GEOMS))
@pytest.mark.parametrize("lane", ["razor-GL", "razor-NEC5"])
@pytest.mark.parametrize("rname", list(REFERENCES))
def test_the_ground_adds_no_cross_formulation_gap(ladders, gname, lane, rname):
    """The unit's central claim, in its sharpest form.

    Razor disagrees with every Galerkin/point-matched sibling at a finite
    mesh — that is its O(1/N) razor-blade walk, and the module docstring's
    table shows it is 0.4–0.9 Ω at N = 192 in FREE SPACE, before any
    ground exists. The question a ground unit has to answer is not "is the
    gap small" but "did the ground widen it". It did not: every
    refl-coef delta is its own free-space delta to within 0.133 Ω
    (measured worst, dipole/GL/sin-galerkin), against a 0.25 Ω bar.

    The comparison is deliberately delta-to-delta rather than absolute,
    because the absolute number is dominated by whatever the reference's
    own free-space behaviour on that deck is — the inverted-V's 12.6 Ω
    against `SinusoidalSolver` being the loud case, present identically in
    all three ground columns.
    """
    n = LADDER[-1]
    d_free = abs(ladders[(gname, "free", lane, n)] - ladders[(gname, "free", rname, n)])
    d_refl = abs(ladders[(gname, "refl", lane, n)] - ladders[(gname, "refl", rname, n)])
    assert abs(d_refl - d_free) < 0.25, (
        f"{gname} {lane} vs {rname}: the ground moved the cross-formulation "
        f"gap from {d_free:.4f} to {d_refl:.4f} Ω"
    )


@pytest.mark.slow
@pytest.mark.parametrize("gname", list(GEOMS))
@pytest.mark.parametrize("lane", ["razor-GL", "razor-NEC5"])
def test_converged_deltas_against_the_galerkin_siblings(ladders, gname, lane):
    """The absolute numbers, pinned at the level they were measured.

    `BSplineSolver(degree=2)` and `SinusoidalGalerkinSolver` are the two
    references that converge on the same answer as each other on both
    decks, so they are the ones an absolute bar can be written against.
    Measured worst |ΔZ| at N = 192 over the refl-coef ground: 1.050 Ω
    (dipole, NEC-5 lane, sin-galerkin); best 0.427 Ω (inverted-V, GL lane,
    sin-galerkin). Bar 1.35 Ω — a regression guard, not a re-derivation.
    """
    n = LADDER[-1]
    z = ladders[(gname, "refl", lane, n)]
    for rname in ("bspline", "sin-galerkin"):
        d = abs(z - ladders[(gname, "refl", rname, n)])
        assert d < 1.35, f"{gname} {lane} vs {rname}: |ΔZ| = {d:.4f} Ω at N={n}"


@pytest.mark.slow
@pytest.mark.parametrize("lane", ["razor-GL", "razor-NEC5"])
def test_the_dipole_ladder_walks_toward_the_galerkin_answer(ladders, lane):
    """The straight deck's refl-coef ladder closes on `BSplineSolver`'s
    N = 192 answer monotonically, at the O(1/N) rate.

    Measured |Z_razor(N) − Z_bspline(192)|, GL lane: 5.217, 2.887, 1.653,
    0.951 Ω — successive ratios 1.81, 1.75, 1.74, i.e. halving per mesh
    doubling to within the reference's own residual walk. NEC-5 lane:
    6.540, 3.208, 1.730, 0.970 (2.04, 1.85, 1.78).

    Not asserted on the inverted-V, and the reason is on the record: there
    razor and the reference walk in OPPOSITE directions in R and cross
    (NEC-5 lane: 3.012, 1.105, 0.179, 0.706), so a monotone bar would be
    pinning a coincidence. `test_the_ground_adds_no_cross_formulation_gap`
    is the claim that survives on both decks.
    """
    ref = ladders[("dipole@0.25", "refl", "bspline", LADDER[-1])]
    seq = [abs(ladders[("dipole@0.25", "refl", lane, n)] - ref) for n in LADDER]
    for a, b in zip(seq, seq[1:]):
        assert b < a, f"the ladder is not closing: {seq}"
    for a, b in zip(seq[:-1], seq[1:-1]):
        assert 1.5 < a / b < 2.5, f"not the O(1/N) rate: {seq}"


# --------------------------------------------------------------------------
# 3. the schedule: what is ω-dependent, and where it is built
# --------------------------------------------------------------------------
def test_swept_refl_coef_matches_the_per_k_solves():
    """`compute_impedance_swept` over a refl-coef ground == solving each k
    alone, bit for bit.

    The ω-boundary gate, and the reason the ground is (eps_r, sigma) here
    rather than a complex ε̃: the tuple form folds as
    ε̃ = εr − j·σ/(ω·ε₀), so ε̃ MOVES with the wavenumber and the Fresnel
    weights move with it. A build that hoisted the weights into
    `_assemble_Z_prepare` alongside the mirrored moments — which is what
    the PEC image legitimately does, and what unit 2's docstring predicted
    a refl-coef unit could not — would return the sweep's nominal-ε̃ answer
    at every k and fail here at two rungs out of three.
    """
    ks = 2 * np.pi / np.array([WL * 0.95, WL, WL * 1.05])
    g = GEOMS["dipole@0.25"]
    kw = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](16),
        wire_radius=g["radius"],
        ground_z=0.0,
        ground_eps=LOSSY,
    )
    swept = RazorSolver(wavelength=WL, **kw).compute_impedance_swept(ks)
    for i, k in enumerate(ks):
        one, _ = RazorSolver(wavelength=2 * np.pi / float(k), **kw).compute_impedance()
        assert swept[i] == one


def test_one_ground_object_per_solved_wavenumber(monkeypatch):
    """The positive half of the ω-boundary: the call counts.

    One ground per SOLVED WAVENUMBER in the fill (that construction is the
    per-k ε̃ hoist), plus the one the k-independent prepare half builds for
    the mirror map — so a single solve builds two and a three-point sweep
    builds four, not six. Two weight-window producers per grounded fill,
    one per term (T1's over the path points, T2's over the centroids), and
    NONE at all over PEC, where the object's `eps_tilde is None` sends the
    fill down the unweighted kernel instead.
    """
    calls = {"built": 0, "windows": 0, "tables": 0}
    real_factory = _potential_ground.potential_ground_for
    real_windows = _potential_ground.PotentialGround.weight_windows
    real_tables = _potential_ground.PotentialGround.weight_tables

    def spy_factory(*a, **kw):
        calls["built"] += 1
        return real_factory(*a, **kw)

    def spy_windows(self, *a, **kw):
        calls["windows"] += 1
        return real_windows(self, *a, **kw)

    def spy_tables(self):
        calls["tables"] += 1
        return real_tables(self)

    monkeypatch.setattr(_potential_ground, "potential_ground_for", spy_factory)
    monkeypatch.setattr(
        _potential_ground.PotentialGround, "weight_windows", spy_windows
    )
    monkeypatch.setattr(_potential_ground.PotentialGround, "weight_tables", spy_tables)

    g = GEOMS["dipole@0.25"]
    base = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](12),
        wire_radius=g["radius"],
        wavelength=WL,
    )
    ks = 2 * np.pi / np.array([WL * 0.98, WL, WL * 1.02])

    for ground, n_k, windows in (
        ({}, 1, 0),
        ({"ground_z": 0.0}, 1, 0),
        ({"ground_z": 0.0, "ground_eps": EPS}, 1, 2),
        ({"ground_z": 0.0, "ground_eps": EPS}, 3, 6),
    ):
        for key in calls:
            calls[key] = 0
        sim = RazorSolver(**base, **ground)
        if n_k == 1:
            sim.compute_impedance()
        else:
            sim.compute_impedance_swept(ks)
        # The prepare half asks once per FILL (free space is where it gets
        # `None` back, which is the answer that keeps the ground layer
        # structurally absent); the k-dependent half asks once per solved
        # wavenumber, and only when there is an image at all.
        expected_built = 1 if not ground else 1 + n_k
        assert calls["built"] == expected_built, (
            f"{ground} over {n_k} k: built {calls['built']} grounds, "
            f"expected {expected_built}"
        )
        assert calls["windows"] == windows, f"{ground}: {calls['windows']} producers"
        assert calls["tables"] == 0, (
            "razor asked for whole-geometry weight TABLES — it must never "
            "materialise anything (N, N) in the weights"
        )


def test_the_answer_does_not_depend_on_the_chunk_schedule():
    """Schedule invariance (architecture doc §3.1), on the weighted fill.

    `_WEIGHTED_CHUNK_ELEMS` shrinks the row chunk so a weight window's
    transient stays in the same memory class as an unweighted moment
    window's. It is a scheduling number and nothing else: the weighted fill
    is elementwise on the observer axis, so every chunk size must give the
    identical matrix, bit for bit — which is also what says the specular
    geometry is being drawn per observation POINT rather than per chunk.
    """
    from momwire import razor as razor_mod

    g = GEOMS["dipole@0.25"]
    kw = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](20),
        wire_radius=g["radius"],
        wavelength=WL,
        ground_z=0.0,
        ground_eps=EPS,
    )
    sim = RazorSolver(**kw)
    ref = sim._assemble_Z(sim._build_geometry(), sim.k)
    for elems in (1_000, 20_000, 40_000_000):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(razor_mod, "_WEIGHTED_CHUNK_ELEMS", elems)
            alt = RazorSolver(**kw)
            got = alt._assemble_Z(alt._build_geometry(), alt.k)
        assert np.array_equal(got, ref), f"chunk budget {elems} moved the matrix"


# --------------------------------------------------------------------------
# 4. the fill reads the object, not the strings
# --------------------------------------------------------------------------
def test_the_razor_fill_follows_the_object_not_the_strings(monkeypatch):
    """The structural row, in both directions.

    A solver configured for the refl-coef ground, handed a PEC
    `PotentialGround`, must fill the PEC matrix — bit for bit, since the
    two reach the identical kernels by the identical schedule. And a
    solver configured for PEC, handed a refl-coef ground, must fill the
    weighted one. If the fill branched on `ground_eps` instead of on the
    object, exactly one of those two would move and the other would not.

    Both decks stand clear of the plane, so nothing downstream (the
    grounded-end tent of unit 3) can mask the swap.
    """
    g = GEOMS["dipole@0.25"]
    base = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](14),
        wire_radius=g["radius"],
        wavelength=WL,
        ground_z=0.0,
    )

    def _Z(sim):
        return sim._assemble_Z(sim._build_geometry(), sim.k)

    pec_sim, refl_sim = RazorSolver(**base), RazorSolver(**base, ground_eps=EPS)
    Z_pec, Z_refl = _Z(pec_sim), _Z(refl_sim)
    assert not np.array_equal(Z_pec, Z_refl)

    def _swap(target, ground):
        monkeypatch.setattr(
            _potential_ground, "potential_ground_for", lambda *a, **kw: ground
        )
        out = _Z(target)
        monkeypatch.undo()
        return out

    as_pec = _potential_ground.PotentialGround(
        refl_sim,
        refl_sim._build_geometry(),
        refl_sim.k,
        refl_sim.omega,
        mode="fold",
        eps_tilde=None,
        image_coefficient=1.0,
    )
    assert np.array_equal(_swap(refl_sim, as_pec), Z_pec), (
        "the refl-coef solver ignored the PEC ground it was handed — it is "
        "reading ground_eps again"
    )

    from momwire import _ground_refl

    as_refl = _potential_ground.PotentialGround(
        pec_sim,
        pec_sim._build_geometry(),
        pec_sim.k,
        pec_sim.omega,
        mode="fold",
        eps_tilde=_ground_refl.eps_tilde(EPS, pec_sim.omega, pec_sim.eps),
        image_coefficient=1.0,
        phi_mode="normal",
    )
    assert np.array_equal(_swap(pec_sim, as_refl), Z_refl), (
        "the PEC solver ignored the refl-coef ground it was handed"
    )


def test_the_consumer_never_applies_the_image_coefficient_itself():
    """The C₂-through-tables contract (`PotentialGround.image_coefficient`).

    On this trunk the coefficient enters through the weights, and a
    consumer that also multiplied by it would be applying it twice. The
    folding grounds both carry coefficient 1, where a double application
    is invisible — so this pins it the only way it can be pinned from
    outside: hand the fill a ground whose coefficient is 2 and whose
    weights are untouched, and require the matrix not to move.
    """
    g = GEOMS["dipole@0.25"]
    sim = RazorSolver(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](12),
        wire_radius=g["radius"],
        wavelength=WL,
        ground_z=0.0,
        ground_eps=EPS,
    )
    geom = sim._build_geometry()
    ref = sim._assemble_Z(geom, sim.k)

    from momwire import _ground_refl

    lying = _potential_ground.PotentialGround(
        sim,
        geom,
        sim.k,
        sim.omega,
        mode="fold",
        eps_tilde=_ground_refl.eps_tilde(EPS, sim.omega, sim.eps),
        image_coefficient=2.0,
        phi_mode=sim.ground_phi_mode,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_potential_ground, "potential_ground_for", lambda *a, **kw: lying)
        alt = RazorSolver(
            wires=g["wires"],
            n_per_edge_per_wire=g["split"](12),
            wire_radius=g["radius"],
            wavelength=WL,
            ground_z=0.0,
            ground_eps=EPS,
        )
        got = alt._assemble_Z(alt._build_geometry(), alt.k)
    assert np.array_equal(got, ref)


# --------------------------------------------------------------------------
# 5. ground_phi_mode parity with BSplineSolver
# --------------------------------------------------------------------------
def test_phi_mode_accepts_and_refuses_exactly_what_bspline_does():
    """Same four values, same wording of the refusal, same default.

    `ground_phi_mode` is the potential trunk's own knob — the field form
    never separates the charge term, so it has no analogue there — and a
    solver that served three of the four, or spelled the refusal
    differently, would be a second dialect of the same option.
    """
    from momwire import _ground_refl

    g = GEOMS["dipole@0.25"]
    base = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](8),
        wire_radius=g["radius"],
        wavelength=WL,
    )
    assert RazorSolver(**base).ground_phi_mode == "normal"
    assert (
        BSplineSolver(wires=g["wires"], nsegs=8, wavelength=WL).ground_phi_mode
        == "normal"
    )

    for mode in _ground_refl.PHI_MODES:
        sim = RazorSolver(**base, ground_z=0.0, ground_eps=EPS, ground_phi_mode=mode)
        assert sim.ground_phi_mode == mode
        z, _ = sim.compute_impedance()
        assert np.isfinite(z.real) and np.isfinite(z.imag)

    for bad in ("nope", "rho_h", None, "Normal"):
        with pytest.raises(ValueError, match="ground_phi_mode"):
            RazorSolver(**base, ground_z=0.0, ground_eps=EPS, ground_phi_mode=bad)
        with pytest.raises(ValueError, match="ground_phi_mode"):
            BSplineSolver(wires=g["wires"], nsegs=8, ground_phi_mode=bad)


def test_the_phi_modes_are_physically_different_weightings():
    """...and they must MOVE the answer, or the T2 weighting is not landing.

    They are four different models of the image charge, so four different
    matrices. Measured at N = 16 on the dipole@0.25 deck, ε̃ = 13 − 0.005j
    folded from (13, 0.005): normal 77.319 − 14.093j, rho_v 76.830 −
    14.680j, image 79.503 − 15.683j, blend 78.167 − 15.181j — spread 2.67 Ω
    in R and 1.59 Ω in X, with "normal" (the default, and the mode the
    Phase-1 evaluation picked against the PyNEC gn 0 goldens) between
    "rho_v" and "image" exactly as `phi_term_weights` describes.

    Every mode must also collapse to the same PEC image in the ε̃ → ∞
    limit, since all four → 1 there — which is what says the spread is the
    modelling choice and not a bug in one branch.
    """
    from momwire import _ground_refl

    kw = dict(ground_z=0.0)
    zs = {}
    for mode in _ground_refl.PHI_MODES:
        zs[mode] = _solve(
            RazorSolver, "dipole@0.25", 16, ground_eps=LOSSY, ground_phi_mode=mode, **kw
        )
    for a, b in ((x, y) for x in zs for y in zs if x < y):
        assert abs(zs[a] - zs[b]) > 0.2, f"{a} and {b} give the same answer: {zs}"
    assert max(z.real for z in zs.values()) - min(z.real for z in zs.values()) > 2.0

    z_pec = _solve(RazorSolver, "dipole@0.25", 16, **kw)
    for mode in _ground_refl.PHI_MODES:
        z_lim = _solve(
            RazorSolver,
            "dipole@0.25",
            16,
            ground_eps=1e12 + 0j,
            ground_phi_mode=mode,
            **kw,
        )
        assert abs(z_lim - z_pec) < 1e-3, f"{mode} does not reach the PEC limit"


# --------------------------------------------------------------------------
# 6. the constructor's own contract
# --------------------------------------------------------------------------
def test_ground_eps_requires_ground_z():
    g = GEOMS["dipole@0.25"]
    with pytest.raises(ValueError, match="ground_eps requires ground_z"):
        RazorSolver(
            wires=g["wires"],
            n_per_edge_per_wire=g["split"](8),
            wire_radius=g["radius"],
            wavelength=WL,
            ground_eps=EPS,
        )


def test_the_default_ground_kwargs_are_inert():
    """Passing `ground_eps=None` / the default phi mode / the default
    model must not perturb one float of a free-space or PEC solve — the
    same standard `extended_kernel=False` holds itself to."""
    g = GEOMS["dipole@0.25"]
    base = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](10),
        wire_radius=g["radius"],
        wavelength=WL,
    )
    for extra in ({}, {"ground_z": 0.0}):
        a, _ = RazorSolver(**base, **extra).compute_impedance()
        b, _ = RazorSolver(
            **base,
            **extra,
            ground_eps=None,
            ground_phi_mode="normal",
            ground_model="refl-coef",
        ).compute_impedance()
        assert a == b
        # ...and the phi mode is unread without a finite ground.
        c, _ = RazorSolver(**base, **extra, ground_phi_mode="image").compute_impedance()
        assert a == c


def test_complex_eps_matches_the_tuple_spec():
    """(eps_r, sigma) folds to ε̃ = εr − j·σ/(ω·ε₀) in the shared layer, so
    handing that ε̃ directly must give the identical solve.

    **At the solver's own ω, which is `c·k` and not `self.omega`.** Those two
    are mathematically the same number and differ by one ulp, and
    `_assemble_Z` deliberately uses the first — `_assemble_Z_from_prepared`'s
    call site says why, in as many words: reading the stashed `self.omega`
    "would cost this method the branch point's bit-for-bit answer over a
    1-ULP omega drift". Folding the tuple at `self.omega` here therefore
    handed the second solve an ε̃ one ulp from the one the first solve built
    for itself, and this test's `==` was passing on the 3.6e-19 that
    difference makes in the fill rounding away in a 9x9 LU.

    It stopped rounding away on the macOS lane at momwire#799, one ulp out in
    Zin, on a fill that had moved for unrelated reasons. Nothing about the
    spec folding was wrong; the ω this test folded at was. Measured before
    the fix: ε̃ at `self.omega` is -0x1.9adc08dcb3cb4p+2 against the solver's
    own -0x1.9adc08dcb3cb6p+2.
    """
    g = GEOMS["dipole@0.25"]
    sim = RazorSolver(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](10),
        wire_radius=g["radius"],
        wavelength=WL,
        ground_z=0.0,
        ground_eps=LOSSY,
    )
    from momwire import _ground_refl

    eps_t = _ground_refl.eps_tilde(LOSSY, sim.c * sim.k, sim.eps)
    z_a, _ = sim.compute_impedance()
    z_b = _solve(RazorSolver, "dipole@0.25", 10, ground_z=0.0, ground_eps=eps_t)
    # momwire#809: fills measured BIT-IDENTICAL -- the tuple spec folds to
    # the same eps_tilde the complex spec passes, so one matrix is built
    # twice. Structural.
    assert z_a == z_b
