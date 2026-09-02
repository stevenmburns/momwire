"""RazorSolver's ground CONTACT: the grounded-end tent (momwire#398 unit 3).

Unit 2 gave `RazorSolver` the PEC image and refused ground contact, because
the tent basis zeroes the current at a free wire end and a grounded end
instead needs its image to CONTINUE it. This file is the evidence that the
continuation is the right one, that it is exact rather than approximate, and
that the geometries which are not contact stay refused.

The basis, in one sentence (`razor.py`'s module docstring has the physics):
a wire end in the plane carries the junction tent between that wire and its
own image — monopole plus image IS a dipole — of which only the real wing is
spelled, because the fold `Z = Z_free − Z_image` already evaluates every
basis against the mirrored sources.

Four gates, in the order they answer "is this the physics":

1. **the analytic image, contact edition** — a grounded radiator over PEC is
   EXACTLY half of that radiator plus its mirror image in free space, on
   razor's own discretization, because the grounded model is the symmetric
   reduction of the mirror model: same rows, same test functions, the
   contact tent playing the mirror model's centre tent. Not a tolerance
   claim (tests 1-4, four geometry classes, both quadrature lanes,
   agreement 6e-16..2e-13 relative).
2. **the external oracle, sharp lane** — the two contact ladders against the
   licensed NEC-5 binary's printed impedances, at the maintainer's approved
   bar (tests 5 and 6). The binary is NOT run here;
   `tests/golden_razor_contact_nec5.py` carries its printouts, captured by
   `scripts/capture_razor_contact_nec5_lane.py`.
3. **the refusals** — contact is a wire END in the plane and nothing else
   (test 8); PEC and Sommerfeld serve it, refl-coef is refused (test 9,
   momwire#282).
4. **free space is not a ground** — a wire at z = 0 with `ground_z=None` has
   no plane to touch, and nothing about it changes (test 10).

**What the NEC-5 lane measures, and why it is gated against the MIRROR
deck.** Razor's grounded answer is its own mirror model halved, exactly
(gate 1). NEC-5's grounded answer is NOT its own mirror model halved: on the
monopole ladder the binary's two decks differ by −0.133+0.285j ohm at N=24,
decaying to −0.097+0.074j at N=96, and on the inverted-L by +0.102+0.064j
decaying to +0.010+0.026j. That difference is a grounded-end discretization
difference between the two codes which vanishes as the mesh refines — both
ladders are still walking at N=96, this formulation's O(1/N) walk. Against
the binary's MIRROR deck razor holds the twin property sharply (a constant
+0.001..0.003 + 0.019j ohm, spread <= 0.0004 ohm down each ladder, the same
kind of fixed kernel nuance unit 2's clearance lane measured); against the
binary's CONTACT deck the residual decays instead of holding, and misses the
constancy half of the bar (and, at the two coarsest monopole rungs, the
per-rung half). Both columns are recorded in the golden module; test 6 gates
the mirror column and test 7 pins the contact column's measured deltas so
the finding cannot drift silently.
"""

import numpy as np
import pytest

from momwire import RazorSolver
from golden_razor_contact_nec5 import CONTACT_LADDERS

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)
RAD = 1.0262e-3

MONO_LEN = 5.35
INVL_A = MONO_LEN / 2.0
LANES = ({}, {"nec5_quadrature": True})
LANE_IDS = ("gl", "n5q")


def _z(wires, npe, *, feed, ground=None, **mode):
    kw = dict(wire_radius=RAD, wavelength=WL, **mode)
    if ground is not None:
        kw["ground_z"] = ground
    z, coeffs = RazorSolver(
        wires=wires, n_per_edge_per_wire=npe, feed_arclength=feed, **kw
    ).compute_impedance()
    return complex(z), coeffs


# --------------------------------------------------------------------------
# 1-4. the analytic image, contact edition: grounded == mirror model / 2
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
@pytest.mark.parametrize("n", (8, 24, 48))
def test_monopole_is_exactly_half_its_dipole(n, lane):
    """A base-fed monopole over PEC == half the 2L dipole in free space.

    The correctness statement for the whole unit, and it is linear algebra
    rather than a tolerance: mesh the monopole with N segments and the
    dipole with 2N placed symmetrically, and the dipole's symmetric-mode
    reduction IS the grounded matrix. Every dipole tent above the plane
    pairs with its mirror below (the fold's two source blocks), and the
    dipole's CENTRE tent is its own mirror — which is the grounded tent,
    real wing plus image wing.

    The rows reduce with it. The dipole's centre row runs
    centroid(below) -> plane -> centroid(above); the total field of a system
    that is its own image obeys E(M·r) = −M·E(r), so the two halves of that
    path contribute the identical number and the grounded row is exactly
    half of it. Halving the row is what makes V the voltage of the BASE gap
    rather than of the dipole's whole gap, i.e. what makes the answer
    Z_dipole/2 instead of Z_dipole — so this test also pins the feed
    convention, not just the fill.
    """
    z_mono, _ = _z(
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])],
        [[n]],
        feed=0.0,
        ground=0.0,
        **lane,
    )
    z_dip, _ = _z(
        [np.array([[0.0, 0.0, -MONO_LEN], [0.0, 0.0, MONO_LEN]])],
        [[2 * n]],
        feed=MONO_LEN,
        **lane,
    )
    assert abs(z_mono - z_dip / 2) / abs(z_mono) < 1e-11, f"{z_mono} vs {z_dip / 2}"
    # ...and the plane is doing the work: the same wire in free space, fed
    # at the same knot, is nowhere near (the free monopole is a half-length
    # dipole fed off its end, hundreds of ohms away).
    z_free, _ = _z(
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])], [[n]], feed=0.0, **lane
    )
    assert abs(z_free - z_mono) > 100.0


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_bent_contact_is_exactly_half_its_mirror_model(lane):
    """An inverted-L over PEC == half the "Z" (top, doubled riser, top).

    Same identity as the monopole's, on a wire whose contact segment is
    followed by a BEND and a second edge with its own segment length: the
    grounded tent's wing sits on the riser's first segment, and the mirror
    model's centre tent spans two riser segments across the plane. A
    contact treatment that assumed a straight wire, or that leaked the
    bend's tangent into the image wing, would move this and leave the
    straight monopole alone.
    """
    nv, nh = 12, 9
    z_g, _ = _z(
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, INVL_A], [INVL_A, 0.0, INVL_A]])],
        [[nv, nh]],
        feed=0.0,
        ground=0.0,
        **lane,
    )
    z_m, _ = _z(
        [
            np.array(
                [
                    [INVL_A, 0.0, -INVL_A],
                    [0.0, 0.0, -INVL_A],
                    [0.0, 0.0, INVL_A],
                    [INVL_A, 0.0, INVL_A],
                ]
            )
        ],
        [[nh, 2 * nv, nh]],
        feed=INVL_A * 2,
        **lane,
    )
    assert abs(z_g - z_m / 2) / abs(z_g) < 1e-11, f"{z_g} vs {z_m / 2}"


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_both_ends_grounded_is_exactly_half_the_closed_loop(lane):
    """An inverted-U with BOTH feet in the plane == half the closed loop.

    Two grounded ends on one wire, at two different points, each with its
    own tent — a supported geometry, not a refused one — and the mirror
    model is then a closed rectangular loop, whose own start/end junction
    tent plays the fed foot's grounded tent while its far side's interior
    knot plays the other. Driving one foot drives the loop's junction tent,
    so this also checks that a grounded tent composes with the ordinary
    junction machinery on the same wire.
    """
    nv, nh, a, d = 12, 9, 4.0, 5.0
    z_g, _ = _z(
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, a], [d, 0.0, a], [d, 0.0, 0.0]])],
        [[nv, nh, nv]],
        feed=0.0,
        ground=0.0,
        **lane,
    )
    z_m, _ = _z(
        [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, a],
                    [d, 0.0, a],
                    [d, 0.0, -a],
                    [0.0, 0.0, -a],
                    [0.0, 0.0, 0.0],
                ]
            )
        ],
        [[nv, nh, 2 * nv, nh, nv]],
        feed=0.0,
        **lane,
    )
    assert abs(z_g - z_m / 2) / abs(z_g) < 1e-11, f"{z_g} vs {z_m / 2}"


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_grounded_junction_of_two_arms_matches_its_four_wire_twin(lane):
    """K wire ends meeting IN the plane get K tents, one per end.

    The plane is one more branch at the point, so a grounded junction has no
    through-path to distinguish: K real ends carry K independent currents
    and whatever does not cancel flows into the ground. The oracle is the
    explicit four-wire twin — both arms plus both images, in free space,
    driven antisymmetrically the way `test_razor_pec_ground.py`'s clearance
    twin is — solved here at an INTERIOR knot so the twin's own K=4 junction
    never has to be fed (that is refused, and rightly).

    The wrong twin sign is 15 % away, so this is a sharp statement about the
    grounded junction and not about two large numbers being similar.
    """
    arms = [
        np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 3.0]]),
        np.array([[0.0, 0.0, 0.0], [-3.0, 0.0, 4.0]]),
    ]
    mirror = np.array([1.0, 1.0, -1.0])
    n = 8
    arc = float(np.linalg.norm(arms[0][1])) * 0.5

    kw = dict(wire_radius=RAD, wavelength=WL, nsegs=n, **lane)
    z_g, _ = RazorSolver(
        wires=arms, ground_z=0.0, feeds=[(0, arc, 1.0 + 0j)], **kw
    ).compute_impedance()
    twin = arms + [a * mirror for a in arms]
    z_t, _ = RazorSolver(
        wires=twin, feeds=[(0, arc, 1.0 + 0j), (2, arc, -1.0 + 0j)], **kw
    ).compute_impedance()
    assert abs(z_g - z_t[0]) / abs(z_g) < 1e-11, f"{z_g} vs {z_t[0]}"

    z_wrong, _ = RazorSolver(
        wires=twin, feeds=[(0, arc, 1.0 + 0j), (2, arc, 1.0 + 0j)], **kw
    ).compute_impedance()
    assert abs(z_g - z_wrong[0]) / abs(z_g) > 0.1


# --------------------------------------------------------------------------
# 5-7. the NEC-5 lane
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sharp_lane():
    """Every gated point, solved once: both contact ladders in the
    `nec5_quadrature=True` lane (two path nodes per row, so ~0.3 s total)."""
    geoms = {
        "monopole": (
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])],
            lambda n: [[n]],
        ),
        "invl": (
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, INVL_A], [INVL_A, 0.0, INVL_A]])],
            lambda n: [[n // 2, n // 2]],
        ),
    }
    return {
        name: {
            n: _z(wires, split(n), feed=0.0, ground=0.0, nec5_quadrature=True)[0]
            for n, *_ in CONTACT_LADDERS[name]
        }
        for name, (wires, split) in geoms.items()
    }


@pytest.mark.parametrize("name", sorted(CONTACT_LADDERS))
def test_sharp_lane_matches_the_nec5_mirror_deck_at_every_rung(name, sharp_lane):
    """|Z_razor − Z_NEC-5(mirror)/2| <= max(0.20, 0.25 % |Z|) at every rung.

    The external half of gate 1: razor's grounded answer is its own mirror
    model halved by construction, and this says the BINARY's mirror model
    halved is the same number to a fixed kernel nuance. Measured worst case
    0.0190 ohm against a 0.20 ohm bar — a 10.5x margin, and the residual is
    the same +0.019j-dominated constant on both geometries.
    """
    solved = sharp_lane[name]
    for n, _z_contact, z_mirror, z_recorded, _z_gl in CONTACT_LADDERS[name]:
        z = solved[n]
        # The recorded razor column must still be what razor computes —
        # otherwise a regression could hide behind a stale fixture.
        assert abs(z - z_recorded) < 5e-6, f"{name} N={n}: razor moved"
        bar = max(0.20, 0.0025 * abs(z_mirror))
        assert abs(z - z_mirror) <= bar, (
            f"{name} N={n}: |dZ| = {abs(z - z_mirror):.4f} > {bar:.4f} ohm "
            f"(razor {z:.4f} vs NEC-5 mirror/2 {z_mirror:.4f})"
        )


@pytest.mark.parametrize("name", sorted(CONTACT_LADDERS))
def test_sharp_lane_offset_is_constant_down_each_ladder(name, sharp_lane):
    """The (razor − NEC-5 mirror/2) offset varies by <= 0.05 ohm per ladder.

    The half of the bar that actually says "twin": a residual CONSTANT in N
    is a fixed kernel/quadrature nuance shared by both codes, whereas a
    residual that walks is a discretization disagreement. Measured spreads
    are (0.0004, 0.0002) for the monopole and (0.0003, 0.0003) for the
    inverted-L — two orders inside the bar, and tighter than any clearance
    geometry unit 2 measured.
    """
    solved = sharp_lane[name]
    res = [solved[n] - z_mirror for n, _c, z_mirror, *_ in CONTACT_LADDERS[name]]
    spread_r = max(r.real for r in res) - min(r.real for r in res)
    spread_x = max(r.imag for r in res) - min(r.imag for r in res)
    assert spread_r <= 0.05, f"{name}: dR spread {spread_r:.4f}"
    assert spread_x <= 0.05, f"{name}: dX spread {spread_x:.4f}"


@pytest.mark.parametrize("name", sorted(CONTACT_LADDERS))
def test_the_contact_deck_residual_decays_instead_of_holding(name, sharp_lane):
    """NEC-5's own CONTACT deck: recorded, and pinned as DECAYING.

    Not the gate — the module docstring explains why — but not a free
    variable either. The finding is that the (razor − NEC-5 contact)
    residual shrinks monotonically down each ladder instead of holding
    constant, which is what a discretization difference that vanishes with
    the mesh looks like, and it is the same shape on both geometries. If a
    future change made razor track NEC-5's contact deck at fixed offset
    instead, that would be a real (and welcome) result — and this test would
    fail, which is the point of pinning it.
    """
    solved = sharp_lane[name]
    res = [abs(solved[n] - z_contact) for n, z_contact, *_ in CONTACT_LADDERS[name]]
    assert res == sorted(res, reverse=True), res
    assert res[-1] < 0.5 * res[0], res
    # ...and it is small in absolute terms throughout: the two codes are
    # already inside 0.30 ohm on a ~40 ohm feedpoint at the coarsest rung.
    assert max(res) < 0.30


# --------------------------------------------------------------------------
# 8-9. what is NOT ground contact
# --------------------------------------------------------------------------
def test_geometries_that_are_not_a_grounded_end_are_refused():
    """Contact is a wire END in the plane; the other three are refused.

    Two `ValueError`s — the interface-crossing wire now says the SHARED
    `_medium_spec` sentence (momwire#651), the in-plane edge says
    `BSplineSolver`'s wording — and one `NotImplementedError` (real physics
    that would need a second unknown at a knot that already carries a tent).
    """
    kw = dict(wire_radius=RAD, wavelength=WL, nsegs=6, ground_z=0.0)
    with pytest.raises(ValueError, match="crosses the ground interface"):
        RazorSolver(wires=[np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])], **kw)
    with pytest.raises(ValueError, match="edge lying in the ground plane"):
        RazorSolver(wires=[np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])], **kw)
    with pytest.raises(NotImplementedError, match="interior anchor"):
        RazorSolver(
            wires=[np.array([[-3.0, 0.0, 3.0], [0.0, 0.0, 0.0], [3.0, 0.0, 3.0]])],
            n_per_edge_per_wire=[[3, 3]],
            wire_radius=RAD,
            wavelength=WL,
            ground_z=0.0,
        )


def test_contact_over_a_finite_ground_is_refused_citing_282():
    """Contact makes the finite ground stricter rather than looser.

    Since momwire#398 unit 4 the reflection-coefficient ground IS served —
    for a wire standing clear of the plane. A grounded END over it is not,
    and the reason is the one #282 names: the fold hard-codes image
    coefficient 1, i.e. PEC, and the grounded tent's lower wing IS that
    image. Weighting the image block cannot repair a basis function that
    is wrong, so the refusal is a geometry one now — read off the wire
    ends, after the ground kwargs are accepted — and it must name both the
    contact and the issue.

    It fires when only SOME wire touches, which is the case a refusal keyed
    on "is there any ground contact" would be tempted to miss: the second
    deck below is an elevated dipole (perfectly serviceable over
    `ground_eps` on its own) standing beside one grounded vertical, and the
    message must point at the vertical by name.
    """
    mono = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])
    clear = np.array([[2.0, -2.0, 4.0], [2.0, 2.0, 4.0]])
    kw = dict(wire_radius=RAD, wavelength=WL, ground_z=0.0, ground_eps=13.0 - 0.4j)

    for wires, npe, name in (
        ([mono], [[8]], "wire 0 start"),
        ([clear, mono], [[8], [8]], "wire 1 start"),
        ([mono, clear], [[8], [8]], "wire 0 start"),
    ):
        with pytest.raises(NotImplementedError, match="momwire#282") as exc:
            RazorSolver(wires=wires, n_per_edge_per_wire=npe, **kw)
        msg = str(exc.value)
        assert name in msg
        assert "CONTACT" in msg or "contact" in msg

    # The same decks over the PEC plane are served, so the refusal is about
    # the finite ground and not about the geometry.
    for wires, npe in (([mono], [[8]]), ([clear, mono], [[8], [8]])):
        z, _ = RazorSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            wire_radius=RAD,
            wavelength=WL,
            ground_z=0.0,
            feed_arclength=0.0,
        ).compute_impedance()
        assert np.isfinite(z.real) and np.isfinite(z.imag)


def test_the_composing_ground_serves_contact_and_refl_coef_still_does_not():
    """This test's claim has now moved THREE times, and each move is a
    measurement rather than a change of mind.

    momwire#398 unit 4 read "`ground_model='sommerfeld'` is refused on its
    own terms — the composition — whether or not anything touches the
    plane". Unit 5 served the composition, and what was left was a contact
    refusal shared with the folding finite ground: the composing ground's
    image coefficient is C₂, not 1, so a grounded tent's lower wing is no
    more its own exact image over Sommerfeld than over Fresnel.

    momwire#624 measured that argument instead of reasoning about it, and
    it does not hold up as a reason to REFUSE. The study's §4.3 diagnosis
    — that the T2 drop discards (1 − w_Φ)·M0(plane) and restoring it is
    the fix — was run behind a flag under §5.5 and the restored term made
    the binary comparison worse at full strength and failed the stubbed
    ladder at every scale. What the row actually does over a finite ground
    is what D1 already accepted from `BSplineSolver`: a residual that is
    bounded and saturating, 0.005 Ω from the binary's own printed shift on
    sea water against bspline's 0.201, and 3.384 against 3.309 on poor
    soil. `test_razor_contact_finite_ground.py` is that measurement as a
    gate.

    What stays refused is D3's row, which is a statement about the MODEL
    and not about this solver: refl-coef at zero clearance is wrong in the
    reference implementation too (stock nec2c prints 175 − 779j Ω on the
    same monopole). So the pair below is the whole of the new boundary —
    same geometry, same soil, two ground models, one served and one not.
    """
    mono = [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])]
    clear = [np.array([[2.0, -2.0, 4.0], [2.0, 2.0, 4.0]])]
    somm = dict(
        wire_radius=RAD,
        wavelength=WL,
        ground_z=0.0,
        ground_eps=13.0 - 0.4j,
        ground_model="sommerfeld",
    )

    z, _ = RazorSolver(
        wires=mono, n_per_edge_per_wire=[[8]], **somm
    ).compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert z.real > 0.0  # a passive antenna, not a sign-flipped fold

    # The same wire under refl-coef is still refused, by D3, and the
    # sentence is `_ground_spec`'s own rather than a copy kept in razor.
    with pytest.raises(NotImplementedError, match="momwire#282") as exc:
        RazorSolver(
            wires=mono,
            n_per_edge_per_wire=[[8]],
            **{**somm, "ground_model": "refl-coef"},
        )
    assert "wire 0 start" in str(exc.value)

    # ...and the deck that does NOT touch is served under both, so that
    # refusal is the geometry's and not the ground model's alone.
    for model in ("sommerfeld", "refl-coef"):
        z, _ = RazorSolver(
            wires=clear, n_per_edge_per_wire=[[8]], **{**somm, "ground_model": model}
        ).compute_impedance()
        assert np.isfinite(z.real) and np.isfinite(z.imag)


# --------------------------------------------------------------------------
# 10-12. contact composes with the rest of razor
# --------------------------------------------------------------------------
def test_free_space_at_z_zero_is_not_contact():
    """`ground_z=None` has no plane to touch, and nothing changed.

    The whole razor free-space corpus lives at z = 0 — a wire there must
    keep being a free wire with both end currents pinned to zero, not
    silently acquire a grounded end. Pinned on the answer AND on the basis
    count, which is the thing that would move first.
    """
    wires = [np.array([[0.0, 0.0, 0.0], [0.0, 10.18946, 0.0]])]
    sim = RazorSolver(
        wires=wires, nsegs=24, wire_radius=RAD, wavelength=WL, nec5_quadrature=True
    )
    geom = sim._build_geometry()
    assert geom["junctions"] == []
    assert geom["n_basis_total"] == 23
    assert geom["grounded_bases"].size == 0
    # The ladder value `tests/test_razor_nec5_twin.py` and
    # `tests/test_razor_pec_ground.py` both already pin for this deck.
    z, _ = sim.compute_impedance()
    assert abs(z - (66.911 - 34.179j)) < 0.05


def test_the_grounded_end_carries_current_and_the_free_end_does_not():
    """The readout, at both ends of one monopole.

    N segments over the plane give N unknowns — N−1 interior tents plus the
    one grounded tent — where the same wire in free space gives N−1. The
    solved current at the base knot is the fed current itself (V / Z), which
    is the coefficient the feed row drove; the current at the top is exactly
    zero, the free end's boundary condition, untouched by the ground.
    """
    n = 20
    sim = RazorSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])],
        nsegs=n,
        wire_radius=RAD,
        wavelength=WL,
        ground_z=0.0,
        feed_arclength=0.0,
    )
    geom = sim._build_geometry()
    assert geom["n_basis_total"] == n
    assert geom["grounded_bases"].tolist() == [n - 1]
    (junction,) = geom["junctions"]
    assert junction["grounded"] and junction["ends"] == [(0, "start")]

    z, coeffs = sim.compute_impedance()
    (knots,) = sim.currents_at_knots(coeffs)
    assert knots.shape == (n + 1,)
    assert knots[-1] == 0.0
    assert abs(knots[0] - 1.0 / z) < 1e-12 * abs(knots[0])
    # ...and the profile is the quarter-wave taper, base to open tip: the
    # last interior knot carries under a tenth of the base current.
    assert abs(knots[-2]) < 0.1 * abs(knots[0])


def test_swept_contact_matches_the_per_k_solves():
    """`compute_impedance_swept` over a contact deck == solving each k alone.

    The grounded tent lives in the k-independent prepare half exactly as the
    interior tents do (it is wing bookkeeping, not fill arithmetic), so a
    sweep must reproduce the single solves bit for bit.
    """
    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])],
        nsegs=16,
        wire_radius=RAD,
        ground_z=0.0,
        feed_arclength=0.0,
    )
    ks = 2 * np.pi / np.array([WL * 0.95, WL, WL * 1.05])
    swept = RazorSolver(wavelength=WL, **kw).compute_impedance_swept(ks)
    for i, k in enumerate(ks):
        one, _ = RazorSolver(wavelength=2 * np.pi / float(k), **kw).compute_impedance()
        assert swept[i] == one
