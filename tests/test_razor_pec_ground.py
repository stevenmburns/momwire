"""RazorSolver's PEC ground: the razor-grounds pilot's first capability.

momwire#398 unit 2, `docs/design/solver-architecture.md` §6. `RazorSolver`
shipped implementing ZERO ground methods; this file is the evidence that it
now serves the PEC image, that the image is the right one, and that the
free-space path is untouched by its arrival.

Three gates, in the order they answer "is this the physics":

1. **the analytic image** — a dipole at height h over PEC is, exactly, the
   same dipole plus its mirrored twin driven with the opposite voltage. Not
   a tolerance claim: on razor's own discretization the two systems are the
   same linear algebra, so they agree to the LU's roundoff (test 1).
2. **the external oracle, sharp lane** — the four ladder geometries over
   `GN 1` against the licensed NEC-5 binary's printed impedances, at the
   maintainer's approved bar (tests 2 and 3). The binary is NOT run here;
   `tests/golden_razor_pec_nec5.py` carries its printouts, captured by
   `scripts/capture_razor_pec_nec5_lane.py`, which is the same pattern
   `tests/test_razor_nec5_twin.py` uses for the free-space ladder.
3. **the off switch** — `ground_z=None` must not merely skip the ground, it
   must be structurally absent (test 5), the standard §6 gate (b) sets.

The bar in tests 2/3 is the maintainer's decision of 2026-08-17 (momwire#398
thread), and it applies to the `nec5_quadrature=True` lane only:

    |Z_razor - Z_NEC-5| <= max(0.20 ohm, 0.25 % of |Z|) at every rung
    N >= 24, AND the (razor - NEC-5) offset constant down each ladder to
    within 0.05 ohm.

The production (default Gauss-Legendre) lane and every finite ground are
DEFERRED and deliberately ungated here; the golden module records the GL
deltas so the deferred bar's starting point is on file rather than in a
comment.
"""

import numpy as np
import pytest

from momwire import RazorSolver
from golden_razor_pec_nec5 import PEC_LADDERS

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)

# The capture script's geometries, re-spelled here as momwire polylines only
# (the NEC decks live in the script, with the binary that consumes them).
DIP_LEN, DIP_RAD, FAT_RAD, DIP_H = 10.18946, 1.0262e-3, 1.0e-2, 5.0
INVVEE_C = 5.35 / np.sqrt(2.0)
INVVEE_APEX_H = 10.0
LOOP_SIDE, LOOP_H, LOOP_RAD = 6.05, 5.0, 2.0e-3


def _dipole(radius):
    pl = np.array([[0.0, 0.0, DIP_H], [0.0, DIP_LEN, DIP_H]])
    return dict(wires=[pl], wire_radius=radius, split=lambda n: [[n]], feed={})


def _invvee():
    a = INVVEE_APEX_H
    pl = np.array(
        [[-INVVEE_C, 0.0, a - INVVEE_C], [0.0, 0.0, a], [INVVEE_C, 0.0, a - INVVEE_C]]
    )
    return dict(
        wires=[pl],
        wire_radius=1.0e-3,
        split=lambda n: [[n // 2, n // 2]],
        feed={},
    )


def _loop():
    s, h = LOOP_SIDE, LOOP_H
    pl = np.array([[0.0, 0.0, h], [s, 0.0, h], [s, s, h], [0.0, s, h], [0.0, 0.0, h]])
    return dict(
        wires=[pl],
        wire_radius=LOOP_RAD,
        split=lambda n: [[n // 4] * 4],
        feed={"feed_arclength": LOOP_SIDE / 2},
    )


GEOMS = {
    "dipole": _dipole(DIP_RAD),
    "fat-dipole": _dipole(FAT_RAD),
    "invvee": _invvee(),
    "loop": _loop(),
}


def _razor_pec(name, n, **mode):
    g = GEOMS[name]
    z, _ = RazorSolver(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](n),
        wire_radius=g["wire_radius"],
        wavelength=WL,
        ground_z=0.0,
        **g["feed"],
        **mode,
    ).compute_impedance()
    return complex(z)


@pytest.fixture(scope="module")
def sharp_lane():
    """Every gated point, solved once: the four geometries' ladders in the
    `nec5_quadrature=True` lane. Measured ~0.7 s in total (the whole lane is
    two path-quadrature nodes per row), so nothing here is `slow`."""
    return {
        name: {n: _razor_pec(name, n, nec5_quadrature=True) for n, *_ in rows}
        for name, rows in PEC_LADDERS.items()
    }


# --------------------------------------------------------------------------
# 1. the analytic image: PEC == the explicit mirrored twin, exactly
# --------------------------------------------------------------------------
def test_pec_equals_explicit_image_twin():
    """A dipole at h over PEC == that dipole + its mirror at -h, driven -1 V.

    This is image theory written as two solves razor can actually express,
    and the agreement is not a tolerance story. Write the twin's matrix in
    real/image block form: mirror symmetry makes the image wire's self-block
    equal the real wire's (A) and the two cross-blocks equal (B), so the
    system is [[A, B], [B, A]] against [+e_f, -e_f]. Its solution is exactly
    antisymmetric, x = [c, -c], and c satisfies (A - B) c = e_f — and A - B
    is, entry for entry, the grounded solver's matrix: same rows, same test
    functions, sources mirrored, one global minus. So the drive-point
    currents are the same number computed two ways, and only the LU's
    roundoff (on a 2N system vs an N one) separates them.

    That makes this the sharpest possible statement of "the image sign is
    right": a wrong sign on either potential term, or a mirror that flipped
    the tangent the other way, moves the grounded answer and leaves the twin
    alone.
    """
    n = 24
    real = np.array([[0.0, 0.0, DIP_H], [0.0, DIP_LEN, DIP_H]])
    twin = np.array([[0.0, 0.0, -DIP_H], [0.0, DIP_LEN, -DIP_H]])

    z_ground, _ = RazorSolver(
        wires=[real],
        nsegs=n,
        wire_radius=DIP_RAD,
        wavelength=WL,
        ground_z=0.0,
    ).compute_impedance()
    z_twin, _ = RazorSolver(
        wires=[real, twin],
        nsegs=n,
        wire_radius=DIP_RAD,
        wavelength=WL,
        feeds=[(0, None, 1.0 + 0j), (1, None, -1.0 + 0j)],
    ).compute_impedance()

    assert abs(z_ground - z_twin[0]) / abs(z_ground) < 1e-11
    # ...and the ground is doing something: free space is 8 ohms away in R
    # and 30 in X at this height, so the agreement above is not two paths
    # both quietly ignoring the plane.
    z_free, _ = RazorSolver(
        wires=[real], nsegs=n, wire_radius=DIP_RAD, wavelength=WL
    ).compute_impedance()
    assert abs(z_ground - z_free) > 8.0


def test_pec_image_is_a_mirror_not_a_translation():
    """Raising the antenna weakens the image; the plane's height is read.

    A cheap independent check that `ground_z` enters as a REFLECTION and not
    as some fixed offset: the same dipole at two heights over the same plane
    gives two different impedances, and the higher one is closer to free
    space (the image is further away).
    """
    kw = dict(nsegs=16, wire_radius=DIP_RAD, wavelength=WL)

    def z_at(h, **extra):
        wires = [np.array([[0.0, 0.0, h], [0.0, DIP_LEN, h]])]
        z, _ = RazorSolver(wires=wires, **kw, **extra).compute_impedance()
        return z

    z_free = z_at(5.0)
    near = z_at(5.0, ground_z=0.0)
    far = z_at(12.0, ground_z=0.0)
    assert abs(far - z_free) < abs(near - z_free)
    # The same clearance over a plane moved with the wire is the same
    # problem — a mirror at z=3 under a wire at 8 is the wire at 5 over 0.
    shifted = z_at(8.0, ground_z=3.0)
    assert abs(shifted - near) / abs(near) < 1e-12


# --------------------------------------------------------------------------
# 2 + 3. the sharp lane vs NEC-5's printed impedances
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(PEC_LADDERS))
def test_sharp_lane_agrees_with_nec5_at_every_rung(name, sharp_lane):
    """|Z_razor - Z_NEC-5| <= max(0.20, 0.25 % |Z|) ohm, every rung N >= 24.

    Measured worst case is the fat dipole at N=96, 0.078 ohm against a
    0.211 ohm bar; the thin dipole and the inverted-V sit at 0.036-0.038
    ohm against 0.20, and the loop at 0.070-0.076 against 0.85. Every
    geometry clears the bar by at least 2.7x.
    """
    solved = sharp_lane[name]
    for n, z_nec5, z_recorded, _z_gl in PEC_LADDERS[name]:
        z = solved[n]
        # The recorded razor column must still be what razor computes —
        # otherwise a regression could hide behind a stale fixture.
        assert abs(z - z_recorded) < 5e-6, f"{name} N={n}: razor moved"
        bar = max(0.20, 0.0025 * abs(z_nec5))
        assert abs(z - z_nec5) <= bar, (
            f"{name} N={n}: |dZ| = {abs(z - z_nec5):.4f} > {bar:.4f} ohm "
            f"(razor {z:.4f} vs NEC-5 {z_nec5:.4f})"
        )


@pytest.mark.parametrize("name", sorted(PEC_LADDERS))
def test_sharp_lane_offset_is_constant_down_each_ladder(name, sharp_lane):
    """The (razor - NEC-5) offset varies by <= 0.05 ohm down each ladder.

    The half of the bar that actually says "twin": a residual that is
    CONSTANT in N is a fixed kernel/quadrature nuance shared by both codes,
    whereas a residual that walks is a discretization disagreement, which is
    what a wrong ground would produce. Measured spreads: dipole (0.0009,
    0.0004), inverted-V (0.0004, 0.0002), loop (0.0064, 0.0061),
    fat dipole (0.0136, 0.0398) — the fat dipole is the tight one, and it is
    tight because its 1 cm radius makes the reduced kernel's own a-dependent
    nuance the largest term in the residual.
    """
    solved = sharp_lane[name]
    res = [solved[n] - z_nec5 for n, z_nec5, *_ in PEC_LADDERS[name]]
    spread_r = max(r.real for r in res) - min(r.real for r in res)
    spread_x = max(r.imag for r in res) - min(r.imag for r in res)
    assert spread_r <= 0.05, f"{name}: dR spread {spread_r:.4f}"
    assert spread_x <= 0.05, f"{name}: dX spread {spread_x:.4f}"


def test_ground_changes_the_answer_on_every_gated_geometry(sharp_lane):
    """None of the four geometries would pass the bar in free space.

    Guards the whole lane against the one failure mode that would make it
    vacuous — a ground that quietly did nothing on some geometry while the
    fixtures happened to sit near the free-space answer.
    """
    for name, rows in PEC_LADDERS.items():
        g = GEOMS[name]
        n = rows[0][0]
        z_free, _ = RazorSolver(
            wires=g["wires"],
            n_per_edge_per_wire=g["split"](n),
            wire_radius=g["wire_radius"],
            wavelength=WL,
            nec5_quadrature=True,
            **g["feed"],
        ).compute_impedance()
        assert abs(complex(z_free) - rows[0][1]) > max(0.20, 0.0025 * abs(rows[0][1]))


# --------------------------------------------------------------------------
# 4. the ground composes with the rest of razor
# --------------------------------------------------------------------------
def test_swept_pec_matches_the_per_k_solves():
    """`compute_impedance_swept` over a PEC ground == solving each k alone.

    The schedule gate for unit 2's cache decision: the mirrored static
    moments are cached in the prepare half and replayed at every wavenumber,
    exactly as the real ones are, so a sweep must reproduce the single
    solves bit for bit. If the image half ever became k-dependent (which a
    reflection-coefficient ground's weights WOULD be), this is the test that
    would catch it being replayed anyway.
    """
    wires = [np.array([[0.0, 0.0, DIP_H], [0.0, DIP_LEN, DIP_H]])]
    ks = 2 * np.pi / np.array([WL * 0.95, WL, WL * 1.05])
    kw = dict(wires=wires, nsegs=16, wire_radius=DIP_RAD, ground_z=0.0)

    swept = RazorSolver(wavelength=WL, **kw).compute_impedance_swept(ks)
    for i, k in enumerate(ks):
        one, _ = RazorSolver(wavelength=2 * np.pi / float(k), **kw).compute_impedance()
        assert swept[i] == one


def test_pec_ground_on_a_junctioned_model():
    """The image composes with junction tents, not just interior ones.

    The inverted-V is fed at a K=2 junction, so its feed row and its feed
    basis both span two wires; the image has to mirror both wings of that
    tent and keep their signs. A wrong sign on one wing would show up as a
    finite but wrong impedance, which the NEC-5 lane above already prices —
    this asserts the cheaper structural fact that the junction path runs at
    all and that the ground moved it.
    """
    g = GEOMS["invvee"]
    kw = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["split"](24),
        wire_radius=g["wire_radius"],
        wavelength=WL,
        nec5_quadrature=True,
    )
    z_g, coeffs = RazorSolver(**kw, ground_z=0.0).compute_impedance()
    z_f, _ = RazorSolver(**kw).compute_impedance()
    assert np.all(np.isfinite(coeffs))
    assert abs(z_g - z_f) > 1.0


# --------------------------------------------------------------------------
# 5. the off switch: free space is structurally ground-free
# --------------------------------------------------------------------------
def test_free_space_prepare_carries_no_image():
    """`ground_z=None` leaves the ground layer ABSENT, not skipped.

    §6 gate (b): the off path must be a structural no-op, so the prepared
    fill state carries no image entry at all — no mirrored moment cache, no
    mirrored tangent table, nothing for a later `if` to test per chunk.
    Pinning the state rather than the answer is the point: an implementation
    that built the image and then multiplied it by zero would give identical
    impedances and fail here, correctly.
    """
    wires = [np.array([[0.0, 0.0, DIP_H], [0.0, DIP_LEN, DIP_H]])]
    kw = dict(wires=wires, nsegs=12, wire_radius=DIP_RAD, wavelength=WL)

    free = RazorSolver(**kw)
    prepared = free._assemble_Z_prepare(free._build_geometry())
    assert prepared["image"] is None
    assert free.ground_z is None

    grounded = RazorSolver(**kw, ground_z=0.0)
    g_prep = grounded._assemble_Z_prepare(grounded._build_geometry())
    img = g_prep["image"]
    assert img is not None
    # Same observers, same row windows: only the sources moved.
    assert [c[:3] for c in img["t1_row_chunks"]] == [
        c[:3] for c in g_prep["t1_row_chunks"]
    ]
    assert img["td_a"].shape == g_prep["td_a"].shape
    # ...and the mirrored tangent table is the real one with z flipped.
    assert np.array_equal(
        img["td_a"] * np.array([[1.0], [1.0], [-1.0]]), g_prep["td_a"]
    )


def test_free_space_answers_are_unchanged_by_the_ground_layer():
    """The free-space ladder `tests/test_razor_nec5_twin.py` pins, re-read.

    Cheap belt-and-braces on top of the structural test: if the ground
    layer's arrival had perturbed the free-space fill at all, these two
    long-standing values would move.
    """
    wire = [np.array([[0.0, 0.0, 0.0], [0.0, DIP_LEN, 0.0]])]
    for n, expect in ((24, 66.911 - 34.179j), (48, 67.401 - 31.019j)):
        z, _ = RazorSolver(
            wires=wire,
            nsegs=n,
            wire_radius=DIP_RAD,
            wavelength=WL,
            nec5_quadrature=True,
        ).compute_impedance()
        assert abs(z - expect) < 0.05
