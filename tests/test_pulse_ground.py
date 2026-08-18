"""`PulseSolver`'s grounds, and what the shared layer did and did not give.

momwire#416's probe row implements NO ground physics. Everything below
either checks that the physics it inherited is right (the mirrored-twin
oracle, agreement with `BSplineSolver` over a reflection-coefficient
ground) or pins a FINDING about the interface it inherited it through —
which surface of `PotentialGround` a dense point-matched fill can actually
consume, and which two are shaped for their first consumer.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, PulseSolver, _potential_ground

WAVELENGTH = 299.792458 / 14.0
DIP_LEN = 0.5 * WAVELENGTH * 0.95
DIP_RAD = 0.02
DIP_H = 5.0
GROUND_EPS = 13.0 - 0.03j


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


def _pulse(**ground):
    return PulseSolver(
        wires=[_centred(1.7)],
        nsegs=20,
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


def test_sommerfeld_refusal_names_the_signature_that_blocked_it():
    """momwire#416's primary deliverable as a refusal message rather than a
    note: the blocker is `Remainder.evaluate(supp_seg, polys)`, and the
    refusal has to say WHICH signature would have served, because that is
    what momwire#398 unit 5 needs from a second consumer.

    The pulse basis can spell the B-spline description's first half
    (degree 0) and not its second: what comes back is a finished GALERKIN
    block, and a point-matched consumer needs the remainder FIELD at
    chosen observation points. Both halves must stay named in the message.
    """
    reason = PulseSolver.capabilities.refusal("sommerfeld")
    assert reason is not None
    for phrase in ("Remainder.evaluate", "supp_seg", "GALERKIN", "obs_points"):
        assert phrase in reason, f"the refusal stopped naming {phrase!r}"

    with pytest.raises(NotImplementedError, match="Remainder.evaluate"):
        PulseSolver(
            wires=[_centred(1.7)],
            nsegs=8,
            wavelength=WAVELENGTH,
            ground_z=0.0,
            ground_eps=GROUND_EPS,
            ground_model="sommerfeld",
        )
