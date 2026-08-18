"""`PulseSolver`'s grounds, and what the shared layer did and did not give.

momwire#416's probe row implements NO ground physics. Everything below
either checks that the physics it inherited is right (the mirrored-twin
oracle, agreement with `BSplineSolver` over a reflection-coefficient
ground) or pins a FINDING about the interface it inherited it through —
which surface of `PotentialGround` a dense point-matched fill can actually
consume, and which two are shaped for their first consumer.

**momwire#430** closed the one surface that was still a refusal: #398 unit
5 landed `Remainder.field_windows(observers, sources, n_moment=…)`, and
`n_moment = 1` IS the signature this module's `_OUT_OF_SCOPE["ground_model"]`
used to say did not exist. Section 5 below is the capability that replaced
the refusal, and `docs/design/pulse-probe.md` carries the dated correction.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, PulseSolver, _ground_refl, _potential_ground

WAVELENGTH = 299.792458 / 14.0
DIP_LEN = 0.5 * WAVELENGTH * 0.95
DIP_RAD = 0.02
DIP_H = 5.0
GROUND_EPS = 13.0 - 0.03j
SOMM = dict(ground_z=0.0, ground_eps=GROUND_EPS, ground_model="sommerfeld")


def _horizontal(z):
    return np.array([[0.0, 0.0, z], [0.0, DIP_LEN, z]])


def _centred(z):
    return np.array([[0.0, -DIP_LEN / 2, z], [0.0, DIP_LEN / 2, z]])


# --------------------------------------------------------------------------
# 1. PEC: the exact mirrored-twin oracle
# --------------------------------------------------------------------------


def test_pec_equals_explicit_image_twin():
    """A dipole at h over PEC == that dipole + its mirror at −h, driven −1 V.

    Image theory written as two solves this row can express, and the
    agreement is not a tolerance story. In real/image block form mirror
    symmetry makes the image wire's self-block equal the real wire's (A)
    and the two cross-blocks equal (B) — for THIS scheme exactly, because
    every ingredient of a row is invariant under the reflection M: h is
    unchanged, t_m·t_n is unchanged (M is orthogonal), and every distance
    in the kernel is unchanged. So the twin system is [[A, B], [B, A]]
    against [+e_f, −e_f], its solution is exactly antisymmetric x = [c, −c],
    and c satisfies (A − B)c = e_f — where A − B is, entry for entry, the
    grounded solver's matrix. Only the LU's roundoff (a 2N system against
    an N one) separates the two drive-point currents.

    That makes this the sharpest possible statement that the image sign is
    right: a wrong sign on either potential term, or a mirror that flipped
    the tangent the other way, moves the grounded answer and leaves the
    twin alone. Measured 1.8e-14 relative at N = 24.
    """
    n = 24
    kw = dict(nsegs=n, wire_radius=DIP_RAD, wavelength=WAVELENGTH)

    z_ground, _ = PulseSolver(
        wires=[_horizontal(DIP_H)], **kw, ground_z=0.0
    ).compute_impedance()
    z_twin, _ = PulseSolver(
        wires=[_horizontal(DIP_H), _horizontal(-DIP_H)],
        **kw,
        feeds=[(0, None, 1.0 + 0j), (1, None, -1.0 + 0j)],
    ).compute_impedance()
    z_free, _ = PulseSolver(wires=[_horizontal(DIP_H)], **kw).compute_impedance()

    assert abs(z_ground - z_twin[0]) / abs(z_ground) < 1e-11
    # ...and the plane is doing something: free space is 24 Ω away here, so
    # the agreement above is not two paths both quietly ignoring it.
    assert abs(z_ground - z_free) > 8.0


def test_pec_image_is_a_mirror_not_a_translation():
    """`ground_z` enters as a REFLECTION, not as a fixed offset: raising the
    antenna weakens the image, and the same clearance over a plane moved
    with the wire is the same problem."""
    kw = dict(nsegs=16, wire_radius=DIP_RAD, wavelength=WAVELENGTH)

    def z_at(h, **extra):
        z, _ = PulseSolver(wires=[_horizontal(h)], **kw, **extra).compute_impedance()
        return z

    z_free = z_at(5.0)
    near, far = z_at(5.0, ground_z=0.0), z_at(12.0, ground_z=0.0)
    assert abs(far - z_free) < abs(near - z_free)
    shifted = z_at(8.0, ground_z=3.0)
    assert abs(shifted - near) / abs(near) < 1e-12


# --------------------------------------------------------------------------
# 2. refl-coef: agreement with the in-house reference
# --------------------------------------------------------------------------


def test_refl_coef_agrees_with_bspline_at_convergence():
    """Same dipole at 0.25 λ over ε = 13 − 0.03j, against `BSplineSolver`.

    Measured 2026-08-18, reference `BSplineSolver(degree=2)` at 200
    segments = 81.535 + 16.195j Ω:

    |    N | Δ/a  | Z (Ω)              | |Z − Z_ref| |
    |------|------|--------------------|-------------|
    |  256 |  2.0 |  81.252 + 11.657j  |     4.55    |
    |  512 |  1.0 |  82.643 + 16.266j  |     1.11    |

    The 1.11 Ω residue is not all this row's: the reference is still
    walking at 200 segments (0.45 Ω between its own 200- and 400-segment
    answers over this ground), so a chunk of the gap belongs to the bar.
    Pinned at 2 Ω absolute plus a 3× improvement across the rung — the
    honest reading of a measurement whose reference carries a half-ohm of
    its own.
    """
    wires = [_centred(0.25 * WAVELENGTH)]
    ground = dict(ground_z=0.0, ground_eps=GROUND_EPS)
    ref, _ = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[200]],
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        degree=2,
        **ground,
    ).compute_impedance()
    ref = complex(np.atleast_1d(ref)[0])

    errs = []
    for n in (256, 512):
        z, _ = PulseSolver(
            wires=wires,
            nsegs=n,
            wire_radius=DIP_RAD,
            wavelength=WAVELENGTH,
            **ground,
        ).compute_impedance()
        errs.append(abs(complex(z) - ref))
    assert errs[1] < errs[0] / 3.0, f"refl-coef ladder stalled: {errs}"
    assert errs[1] < 2.0, f"N=512 is {errs[1]:.3g} Ω from the reference"


# --------------------------------------------------------------------------
# 3. the structural row: the fill reads the OBJECT
# --------------------------------------------------------------------------


def _pulse(n=20, h=1.7, **ground):
    return PulseSolver(
        wires=[_centred(h)],
        nsegs=n,
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        **ground,
    )


def _Z(sim):
    return sim._assemble_Z(sim._build_geometry(), sim.k)


def test_the_fill_follows_the_ground_object_not_the_strings(monkeypatch):
    """Hand a refl-coef-configured solver a PEC `PotentialGround` and it
    fills the PEC matrix, bit for bit.

    The idiom is `tests/test_potential_ground.py`'s, and it means the same
    thing here: if `_assemble_Z` branched on `ground_eps` / `ground_model`
    it would be unmoved by the swap. It branches on nothing at all — PEC
    and refl-coef reach the identical code through `weight_windows()`,
    which is why the swapped result is bit-equal rather than merely close.
    """
    z_pec = _Z(_pulse(ground_z=0.0))
    z_refl = _Z(_pulse(ground_z=0.0, ground_eps=GROUND_EPS))
    assert not np.array_equal(z_pec, z_refl)

    sim = _pulse(ground_z=0.0, ground_eps=GROUND_EPS)
    geom = sim._build_geometry()
    pec_ground = _potential_ground.PotentialGround(
        sim, geom, sim.k, sim.omega, mode="fold", eps_tilde=None, image_coefficient=1.0
    )
    monkeypatch.setattr(
        _potential_ground, "potential_ground_for", lambda *a, **kw: pec_ground
    )
    assert np.array_equal(_Z(sim), z_pec), (
        "the fill ignored the ground object it was handed — it is reading "
        "ground_eps/ground_model again"
    )


def test_free_space_builds_no_ground_object_at_all():
    """`potential_ground_for` returning None is structural absence, not a
    skipped branch: the free-space fill must never touch a mirror.
    """
    calls = {"n": 0}
    real = _potential_ground.PotentialGround.image_geometry

    def counted(self):
        calls["n"] += 1
        return real(self)

    _potential_ground.PotentialGround.image_geometry = counted
    try:
        _Z(_pulse())
        assert calls["n"] == 0
        _Z(_pulse(ground_z=0.0))
        assert calls["n"] == 1, "exactly one mirror per grounded fill"
    finally:
        _potential_ground.PotentialGround.image_geometry = real


# --------------------------------------------------------------------------
# 4. the findings: which PotentialGround surfaces a second consumer can use
# --------------------------------------------------------------------------


def test_weight_tables_are_served_on_a_non_bspline_solver():
    """momwire#416's first interface finding, now closed — the positive
    replacement for `test_weight_tables_is_not_consumable_by_a_non_bspline_
    solver`, which momwire#429 unit 1 deleted rather than loosened.

    `PotentialGround.weight_tables()` is documented as the whole-geometry
    `(w_A, w_Φ)` pair, and until unit 1 its refl-coef branch did not
    compute it: it called back into `BSplineSolver._image_refl_prep` /
    `._image_refl_weights`, the caching SCHEDULE of its first consumer, so
    a second consumer got an `AttributeError` instead of a weight table.
    The chain lives in `_potential_ground` now and the cache reaches the
    object through `weight_tables(prep=…)`, so a solver that has no cache
    to offer simply omits it.

    The claim is not merely "it returns something": the tables must be the
    full-width `weight_windows` rectangle, entry for entry, because that
    is the surface this row's fill actually consumes and the two must not
    be allowed to drift into two different reflection coefficients.
    """
    sim = _pulse(ground_z=0.0, ground_eps=GROUND_EPS)
    geom = sim._build_geometry()
    ground = _potential_ground.potential_ground_for(sim, geom, sim.k, sim.omega)
    n = geom["h_per_seg"].size

    w_A, w_Phi = ground.weight_tables()
    assert w_A.shape == (n, n) and w_Phi.shape == (n, n)
    assert w_A.dtype == np.complex128 and w_Phi.dtype == np.complex128

    win_A, win_Phi = ground.weight_windows()(0, n)
    assert np.array_equal(w_A, win_A)
    assert np.array_equal(w_Phi, win_Phi)


def test_pec_weight_window_is_the_mirror_tangent_dot_and_a_unit_phi():
    """The other half of why one code path serves both grounds: over PEC the
    window IS `(t_m · M t_n, 1)`, so the "unweighted" kernel razor needs is
    the weighted one with a trivial pair here. The image sign rides in w_A
    plus the fill's single global minus, and this row applies
    `image_coefficient` nowhere — it is 1.0 over a folding ground and
    already inside the tables regardless.
    """
    sim = _pulse(ground_z=0.0)
    geom = sim._build_geometry()
    ground = _potential_ground.potential_ground_for(sim, geom, sim.k, sim.omega)
    n = geom["h_per_seg"].size
    w_A, w_Phi = ground.weight_windows()(0, n)

    t = geom["tangents"]
    expected = t @ ground.image_geometry().mirror_tangents(t).T
    assert np.array_equal(w_A, expected.astype(np.complex128))
    assert np.array_equal(w_Phi, np.ones_like(w_A))
    assert ground.mode == "fold" and ground.image_coefficient == 1.0
    assert ground.remainder() is None


# --------------------------------------------------------------------------
# 5. Sommerfeld (momwire#430): the capability that replaced the refusal
# --------------------------------------------------------------------------


def test_sommerfeld_is_no_longer_refused():
    """momwire#430's headline: the refusal is gone, and the capability row
    says so — `PulseSolver` is a THIRD consumer of `Remainder.field_windows`
    (razor's tent, `n_moment=2`, was the second), with `n_moment=1` being
    exactly the shape momwire#416's probe asked for and #398 unit 5 shipped.
    """
    c = PulseSolver.capabilities
    assert "sommerfeld" in c.grounds
    assert c.refusal("sommerfeld") is None
    assert "sommerfeld" not in c.refusals

    z, _ = _pulse(n=8, **SOMM).compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)


def test_sommerfeld_requires_ground_eps_exactly_as_bspline_does():
    """Same condition, same wording, same exception type as `BSplineSolver`
    — the exact ground is the exact ground OF something."""
    for cls, extra in (
        (PulseSolver, {"nsegs": 8}),
        (BSplineSolver, {"nsegs": 8}),
    ):
        with pytest.raises(ValueError, match="requires ground_eps"):
            cls(
                wires=[_centred(1.7)],
                wire_radius=DIP_RAD,
                wavelength=WAVELENGTH,
                ground_z=0.0,
                ground_model="sommerfeld",
                **extra,
            )


def test_ground_model_rejects_anything_else():
    with pytest.raises(ValueError, match="ground_model"):
        _pulse(n=8, ground_z=0.0, ground_model="nope")


def test_the_epsilon_one_limit_is_free_space_bit_for_bit():
    """ε̃ = 1 is not a ground, and the fill must produce that arithmetically.

    With ε̃ = 1 the image coefficient C₂ = (ε̃−1)/(ε̃+1) is exactly 0, so
    both the exact-image half (through the constant `weight_windows` pair)
    and the remainder Q (`_sommerfeld`'s six λ-integrals collapse to exact
    zeros at this limit) vanish structurally, and `Z_free − 0` is `Z_free`
    in float64. `array_equal`, not `allclose`, is what says the collapse is
    structural rather than merely small.
    """
    sim_free = _pulse(n=12)
    sim_unity = _pulse(n=12, ground_z=0.0, ground_eps=1.0, ground_model="sommerfeld")
    assert np.array_equal(_Z(sim_free), _Z(sim_unity))


def test_pec_limit_decays_smoothly_to_the_pec_image():
    """|Z(ε̃) − Z_PEC| falls monotonically as ε̃ grows, at C₂'s own rate.

    Measured |Z(ε̃) − Z_PEC| at ε̃ = 10, 10², 10³, 10⁴, 10⁶ on the dipole@
    0.25λ deck, N = 64: 14.677, 5.626, 1.898, 0.613, 0.0618 Ω. Single-decade
    ratios 2.61, 2.96, 3.10 — climbing toward √10 — and the final two-decade
    step 9.91 ≈ √100, the same rate `tests/test_razor_sommerfeld_ground.py`
    measured on a different formulation entirely: this is C₂'s rate, not
    the remainder's, so both terms are converging together.
    """
    n = 64
    h = 0.25 * WAVELENGTH
    z_pec, _ = _pulse(n=n, h=h, ground_z=0.0).compute_impedance()
    seq = []
    for e in (1e1, 1e2, 1e3, 1e4, 1e6):
        z, _ = _pulse(
            n=n, h=h, ground_z=0.0, ground_eps=complex(e), ground_model="sommerfeld"
        ).compute_impedance()
        seq.append(abs(complex(z) - complex(z_pec)))

    for a, b in zip(seq, seq[1:]):
        assert b < a, f"|Z(ε̃) − Z_PEC| is not monotone: {seq}"
    assert seq[-1] < 8.0e-2, seq
    assert seq[-1] > 1e-3, "suspiciously exact — is the ground reaching Z at all?"
    for a, b in zip(seq[:3], seq[1:4]):
        assert 2.3 < a / b < 3.4, f"decay rate is not O(ε̃^-1/2): {seq}"
    assert 8.5 < seq[-2] / seq[-1] < 11.5, seq


@pytest.mark.slow
def test_the_remainder_is_worth_tens_of_ohms_at_low_height():
    """The positive gate on Q — a dead remainder would show up here first.

    At 0.25 λ the two finite grounds sit close enough (~1 Ω, momwire#398
    unit 5's own finding) that a Q term which never reached Z could still
    pass every other gate in this file. The gate that CANNOT be faked that
    way is 0.04 λ, below the refl-coef ground's 0.1-0.5 λ validity window
    (momwire#151): there the two grounds are genuinely different physics.

    Measured refl-vs-Sommerfeld split on the dipole@0.04λ deck, this row's
    own Δ/a ≈ 1 ladder (`tests/test_pulse.py`'s LADDER convention):

    | N   | Δ/a | split (Ω) |
    |-----|-----|-----------|
    | 64  | 7.9 | 20.366    |
    | 128 | 4.0 | 21.982    |
    | 256 | 2.0 | 22.932    |
    | 512 | 1.0 | 23.167    |

    `BSplineSolver(degree=2)` splits by 23.126 Ω at N = 512 over the same
    deck — 0.042 Ω apart, well under the 1 Ω this basis's own slower
    convergence would excuse (razor agreed with `BSplineSolver` to 0.03 Ω;
    this scheme's floor is Δ/a ≈ 1, not Δ/λ, so N = 512 is as far as this
    ladder is honestly allowed to walk, per momwire#248 / the module
    docstring's §5.3 finding).
    """
    h = 0.04 * WAVELENGTH
    ladder = (64, 128, 256, 512)
    seq = []
    for n in ladder:
        z_refl, _ = _pulse(
            n=n, h=h, ground_z=0.0, ground_eps=GROUND_EPS
        ).compute_impedance()
        z_somm, _ = _pulse(n=n, h=h, **SOMM).compute_impedance()
        seq.append(abs(complex(z_refl) - complex(z_somm)))

    assert all(s > 15.0 for s in seq), f"split is not 'tens of ohms': {seq}"

    b_refl, _ = BSplineSolver(
        wires=[_centred(h)],
        nsegs=512,
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        degree=2,
        ground_z=0.0,
        ground_eps=GROUND_EPS,
    ).compute_impedance()
    b_somm, _ = BSplineSolver(
        wires=[_centred(h)],
        nsegs=512,
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        degree=2,
        **SOMM,
    ).compute_impedance()
    b_split = abs(complex(b_refl) - complex(b_somm))
    assert abs(seq[-1] - b_split) < 0.1, (
        f"N=512 split is {seq[-1]:.4f} Ω where BSplineSolver's is {b_split:.4f} Ω"
    )

    ref = [abs(s - b_split) for s in seq]
    for a, b in zip(ref, ref[1:]):
        assert b < a, f"the ladder is not closing on the reference split: {ref}"


@pytest.mark.slow
def test_the_ground_adds_no_cross_formulation_gap_at_quarter_wave():
    """The ground must not widen the pulse-vs-Galerkin gap this formulation's
    own slow convergence already opens in free space.

    Measured |Z_pulse − Z_bspline(400)| at 0.25 λ: free space 4.729 / 0.290
    Ω at N = 256 / 512; over the Sommerfeld ground 4.818 / 0.712 Ω — widened
    by 0.089 / 0.422 Ω. Pinned at 0.6 Ω, the honest reading of a comparison
    whose free-space half is itself only converged to ~0.3 Ω at this row's
    own Δ/a ≈ 1 floor.
    """
    ref_kw = dict(
        wires=[_centred(0.25 * WAVELENGTH)],
        nsegs=400,
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        degree=2,
    )
    b_free, _ = BSplineSolver(**ref_kw).compute_impedance()
    b_somm, _ = BSplineSolver(**ref_kw, **SOMM).compute_impedance()
    b_free, b_somm = complex(b_free), complex(b_somm)

    for n in (256, 512):
        p_free, _ = _pulse(n=n, h=0.25 * WAVELENGTH).compute_impedance()
        p_somm, _ = _pulse(n=n, h=0.25 * WAVELENGTH, **SOMM).compute_impedance()
        d_free = abs(complex(p_free) - b_free)
        d_somm = abs(complex(p_somm) - b_somm)
        assert abs(d_somm - d_free) < 0.6, (
            f"N={n}: the ground moved the cross-formulation gap from "
            f"{d_free:.4f} to {d_somm:.4f} Ω"
        )


def test_the_fill_follows_the_composing_ground_object_not_the_strings(monkeypatch):
    """The structural row, sharper than the folding-ground version above: a
    COMPOSING `PotentialGround` handed to a refl-coef-configured solver
    must fill the composed matrix, and a folding one handed to a
    Sommerfeld-configured solver must fill the folded matrix — bit for bit,
    both directions, since `_assemble_Z` reads the object's `mode` and
    nothing else.
    """
    refl_sim = _pulse(n=14, ground_z=0.0, ground_eps=GROUND_EPS)
    somm_sim = _pulse(n=14, **SOMM)
    Z_refl, Z_somm = _Z(refl_sim), _Z(somm_sim)
    assert not np.array_equal(Z_refl, Z_somm)

    def _ground(sim, mode):
        omega = sim.c * sim.k
        eps_t = _ground_refl.eps_tilde(GROUND_EPS, omega, sim.eps)
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

    def _swap(target, ground):
        monkeypatch.setattr(
            _potential_ground, "potential_ground_for", lambda *a, **kw: ground
        )
        out = _Z(target)
        monkeypatch.undo()
        return out

    assert np.array_equal(_swap(refl_sim, _ground(refl_sim, "compose")), Z_somm), (
        "the refl-coef solver ignored the composing ground it was handed"
    )
    assert np.array_equal(_swap(somm_sim, _ground(somm_sim, "fold")), Z_refl), (
        "the Sommerfeld solver ignored the folding ground it was handed"
    )


def test_the_consumer_never_applies_the_image_coefficient_itself():
    """The C₂-through-the-windows contract, on the ground that actually has
    a non-trivial coefficient (unlike PEC's, which is 1 and hides a double
    application). A ground that LIES about `image_coefficient` while its
    `weight_windows` / `remainder` stay honest must not move the matrix —
    the razor-blade unit-5 idiom, replayed here.
    """
    sim = _pulse(n=12, **SOMM)
    ref = _Z(sim)

    class _CoefficientLiar:
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
        got = _Z(_pulse(n=12, **SOMM))
    assert np.array_equal(got, ref)


def test_ground_phi_mode_is_accepted_and_unread_under_sommerfeld():
    """`ground_phi_mode` is a refl-coef knob with no Sommerfeld analogue —
    the composing ground's image coefficient is exact. Bit-identity across
    every mode, and `phi_mode is None` on the object itself, is the strong
    form of "ignored" — it comes for free because the factory never passes
    it into a composing `PotentialGround`.
    """
    ref = None
    for mode in _ground_refl.PHI_MODES:
        sim = _pulse(n=12, **SOMM, ground_phi_mode=mode)
        ground = _potential_ground.potential_ground_for(
            sim, sim._build_geometry(), sim.k, sim.omega
        )
        assert ground.phi_mode is None
        got = _Z(sim)
        if ref is None:
            ref = got
        assert np.array_equal(got, ref), f"{mode} moved the composed matrix"
