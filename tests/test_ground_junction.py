"""Ground-junction end condition (#151).

A wire end lying in an active ground plane is electrically connected to
its own image (NEC's "connected to ground"): the end current is a real
degree of freedom completed by the image, not a free end pinned to zero.
Before #151 a ground-mounted quarter-wave monopole read ~33 -10600j on
every basis — the pinched end acted as a tiny series capacitor — while
NEC-family kernels give ~34 -20j.

Cross-basis agreement is the oracle here (sinusoidal follows nec2c's tbf
ground path; the B-spline family keeps the value-1 end basis — two
independent formulations), plus a physical window that the old free-end
pathology misses by two orders of magnitude. PyNEC-sourced goldens for
this geometry: 33.6 -20.4j (monopole), 17.9 -35.7j (45-deg slant),
36.7 -19.5j (3-wire grounded junction) — see momwire#151.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, HMatrixSolver, SinusoidalSolver

WL = 299.792458 / 7.15  # 7.15 MHz
H = 0.238 * WL  # near-resonant monopole height
FEED = [(0, 0.02 * H, 1.0 + 0j)]
S45 = 0.7071067811865476


def _mono_wires():
    return [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, H)])]


def _solvers(wires, feeds, nsegs=21, **kw):
    base = dict(
        wires=wires,
        wavelength=WL,
        wire_radius=0.001,
        ground_z=0.0,
        feeds=feeds,
        **kw,
    )
    return {
        "sin": SinusoidalSolver(nsegs=nsegs, **base),
        "bs1": BSplineSolver(degree=1, nsegs=nsegs, **base),
        "bs2": BSplineSolver(degree=2, nsegs=nsegs, **base),
        "hmat": HMatrixSolver(nsegs=nsegs, **base),
    }


def _z(solver):
    z, _ = solver.compute_impedance()
    return complex(z)


def _assert_cluster(zs, r_lo, r_hi, x_lo, x_hi, rel=0.03):
    """Every solver inside the physical window, and all mutually close."""
    for name, z in zs.items():
        assert r_lo < z.real < r_hi, f"{name}: R={z.real}"
        assert x_lo < z.imag < x_hi, f"{name}: X={z.imag}"
    ref = zs["sin"]
    for name, z in zs.items():
        assert abs(z - ref) / abs(ref) < rel, f"{name}: {z} vs sin {ref}"


def test_monopole_pec():
    zs = {k: _z(s) for k, s in _solvers(_mono_wires(), FEED).items()}
    # PyNEC golden 33.6 -20.4j; the pre-#151 pathology was X ~ -10^4.
    _assert_cluster(zs, 25.0, 45.0, -45.0, 0.0)


def test_monopole_pec_via_tensor_route_fallback(monkeypatch):
    """#273: the same PyNEC golden window as `test_monopole_pec`, forced onto
    `_build_J_image_blocks` (the no-accelerator PEC image fallback) instead
    of the chunked accumulator every accelerated build silently prefers —
    `_compute_Z_operator` picks the tensor route whenever
    `_HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL` is False, independent of deck
    size, so this box otherwise never exercises it through a physics gate."""
    import momwire.bspline as bmod

    if not bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("weighted windowed Z assembly accelerator not built")
    monkeypatch.setattr(bmod, "_HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL", False)
    s = BSplineSolver(
        wires=_mono_wires(),
        degree=2,
        nsegs=21,
        wavelength=WL,
        wire_radius=0.001,
        ground_z=0.0,
        feeds=FEED,
    )
    z = _z(s)
    assert 25.0 < z.real < 45.0
    assert -45.0 < z.imag < 0.0


def test_monopole_base_current_is_maximal():
    for k, s in _solvers(_mono_wires(), FEED).items():
        z, coeffs = s.compute_impedance()
        knots = s.currents_at_knots(coeffs)
        mag = np.abs(np.asarray(knots[0]))
        assert mag[0] == pytest.approx(mag.max(), rel=0.05), k
        # Tip still a free end: current vanishes there.
        assert mag[-1] < 0.02 * mag[0], k


def test_slant_45deg_pec():
    wires = [np.array([(0.0, 0.0, 0.0), (H * S45, 0.0, H * S45)])]
    zs = {k: _z(s) for k, s in _solvers(wires, FEED).items()}
    # PyNEC golden 17.9 -35.7j (image continuation is not collinear).
    _assert_cluster(zs, 12.0, 25.0, -50.0, -20.0)


def test_three_wire_junction_at_ground():
    """Vertical + two up-slanted wires meeting at the grounded base. Per
    NEC's conect(), each member is ground-connected independently (no
    inter-wire junction bookkeeping); coupling flows through the images."""
    L = 0.6 * H
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, H)]),
        np.array([(0.0, 0.0, 0.0), (L * S45, 0.0, L * S45)]),
        np.array([(0.0, 0.0, 0.0), (-L * S45, 0.0, L * S45)]),
    ]
    junctions = [[(0, "start"), (1, "start"), (2, "start")]]
    zs = {k: _z(s) for k, s in _solvers(wires, FEED, junctions=junctions).items()}
    # PyNEC golden 36.7 -19.5j.
    _assert_cluster(zs, 28.0, 46.0, -40.0, 0.0)


def test_refl_coef_ground_contact_solves():
    s = SinusoidalSolver(
        wires=_mono_wires(),
        nsegs=21,
        wavelength=WL,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        feeds=FEED,
    )
    z = _z(s)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert z.real > 0.0


def test_sommerfeld_ground_contact_allowed():
    """Touching the plane no longer raises; the solve stays finite.
    (Sinusoidal reproduces NEC-2's gn2 contact answer — PyNEC parity 0.3%
    on this geometry; see momwire#151.)"""
    for cls, kw in (
        (SinusoidalSolver, {}),
        (BSplineSolver, {"degree": 2}),
        (HMatrixSolver, {}),
    ):
        s = cls(
            wires=_mono_wires(),
            nsegs=21,
            wavelength=WL,
            wire_radius=0.001,
            ground_z=0.0,
            ground_eps=(13.0, 0.005),
            ground_model="sommerfeld",
            feeds=FEED,
            **kw,
        )
        z = _z(s)
        assert np.isfinite(z.real) and np.isfinite(z.imag), cls.__name__
        assert z.real > 0.0, cls.__name__


def test_below_plane_rejected_bspline():
    wires = [np.array([(0.0, 0.0, -0.5), (0.0, 0.0, H)])]
    s = BSplineSolver(
        wires=wires,
        degree=2,
        nsegs=21,
        wavelength=WL,
        wire_radius=0.001,
        ground_z=0.0,
        feeds=FEED,
    )
    with pytest.raises(ValueError, match="below the ground plane"):
        s.compute_impedance()


def test_in_plane_wire_rejected():
    wires = [np.array([(0.0, 0.0, 0.0), (H, 0.0, 0.0)])]
    for cls, kw in ((SinusoidalSolver, {}), (BSplineSolver, {"degree": 2})):
        s = cls(
            wires=wires,
            nsegs=21,
            wavelength=WL,
            wire_radius=0.001,
            ground_z=0.0,
            feeds=FEED,
            **kw,
        )
        with pytest.raises(ValueError, match="ground plane"):
            s.compute_impedance()


def test_elevated_end_stays_free():
    """A wire comfortably above the plane keeps free ends: its end
    current vanishes and Z is the strongly capacitive short-vertical
    value — the ground-junction path must not fire on clearance."""
    wires = [np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 3.0 + H)])]
    s = SinusoidalSolver(
        wires=wires,
        nsegs=21,
        wavelength=WL,
        wire_radius=0.001,
        ground_z=0.0,
        feeds=FEED,
    )
    z, alpha = s.compute_impedance()
    assert z.imag < -5000.0
    mag = np.abs(np.asarray(s.currents_at_knots(alpha)[0]))
    assert mag[0] < 0.05 * mag.max()


# ----------------------------------------------------------------------
# The contact-node charge over a FINITE ground (momwire#282)
# ----------------------------------------------------------------------
#
# #151's junction is exact only for a PEC plane, where the image current
# equals the wire current at the node and the two end charges cancel. Over
# a finite ground the image carries ρ of it, leaving (1−ρ)·I/jω sitting ON
# the plane — a point charge whose potential at the nearest collocation
# point grows like 1/Δ, so |Z| walked away under refinement instead of
# settling. #282 subtracts that charge's field: it is double-counting, not
# physics (ρ < 1 already says the earth takes the current the plane does
# not reflect), and it is what the mixed-potential solvers never had.

_282_LAM = 299.792458 / 30.0
_282_H = _282_LAM / 4
_282_NS = (11, 21, 41, 61, 81)
_282_GROUNDS = {
    "refl-coef soil": dict(ground_eps=(13.0, 0.005)),
    "refl-coef lossless": dict(ground_eps=(13.0, 0.0)),
    "sommerfeld soil": dict(ground_eps=(13.0, 0.005), ground_model="sommerfeld"),
}


def _282_z(cls, ns, **ground):
    """The issue's own deck: a base-fed quarter-wave vertical whose base
    LIES IN the plane, a = 0.02 (Δ/a from 11 down to 1.5 over the ladder)."""
    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, _282_H]])],
        n_per_edge_per_wire=[[ns]],
        nsegs=ns,
        wire_radius=0.02,
        feed_arclength=(_282_H / ns) * 0.5,
        ground_z=0.0,
        wavelength=_282_LAM,
    )
    if cls is BSplineSolver:
        kw.update(degree=2, feed_model="segment")
    kw.update(ground)
    return complex(cls(**kw).compute_impedance()[0])


@pytest.mark.parametrize("ground", list(_282_GROUNDS))
def test_282_contact_over_finite_ground_no_longer_diverges(ground):
    """|Z| settles instead of walking. Before #282 this ladder ran
    66.81−392.21j → 96.13−984.75j (refl-coef soil, spread 1.49) with the
    reactance tracking 1/Δ; now the whole excursion is gone and what is
    left is a monotone approach.
    """
    z = [_282_z(SinusoidalSolver, n, **_282_GROUNDS[ground]) for n in _282_NS]
    spread = abs(z[-1] - z[0]) / abs(z[0])
    assert spread < 0.30, f"{ground}: spread {spread:.3f} over NS={_282_NS}: {z}"
    # Monotone approach to the fine-mesh end, not an excursion.
    errs = [abs(v - z[-1]) for v in z[:-1]]
    assert all(b < a for a, b in zip(errs, errs[1:])), f"{ground}: not monotone: {z}"
    # The old failure mode was a huge negative reactance; the new answer sits
    # in the same quadrant as the cross-basis reference at every mesh.
    assert all(v.imag > 0 for v in z), f"{ground}: reactance went capacitive: {z}"


def test_282_leaves_the_pec_contact_fill_bit_identical():
    """ρ = 1 is the PEC path, and the correction is identically zero there:
    the #151/#247 PEC contact decks must produce the SAME fill, bit for bit,
    not merely the same answer."""
    common = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, _282_H]])],
        n_per_edge_per_wire=[[21]],
        nsegs=21,
        wire_radius=0.02,
        feed_arclength=(_282_H / 21) * 0.5,
        ground_z=0.0,
        wavelength=_282_LAM,
    )
    for ek in (False, True):
        s = SinusoidalSolver(**common, extended_kernel=ek)
        geom = s._build_geometry()
        G, seg_view = s._assemble_Z(geom, s.k)
        G2 = G.copy()
        s._contact_charge_correction(G2, geom, s.k, seg_view)
        assert np.array_equal(G, G2), f"PEC contact fill moved (extended_kernel={ek})"


def test_282_leaves_an_elevated_wire_untouched():
    """No contact, no correction: a wire clear of the plane over the same
    finite ground keeps its exact fill (#153's elevated parity numbers ride
    on this)."""
    s = SinusoidalSolver(
        wires=[np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0 + _282_H]])],
        n_per_edge_per_wire=[[21]],
        nsegs=21,
        wire_radius=0.02,
        feed_arclength=(_282_H / 21) * 0.5,
        ground_z=0.0,
        wavelength=_282_LAM,
        ground_eps=(13.0, 0.005),
    )
    geom = s._build_geometry()
    G, seg_view = s._assemble_Z(geom, s.k)
    G2 = G.copy()
    s._contact_charge_correction(G2, geom, s.k, seg_view)
    assert np.array_equal(G, G2)


def test_282_correction_vanishes_in_the_pec_limit():
    """ε̃ → ∞ drives the correction to zero continuously, so the finite-ground
    path meets the PEC path rather than jumping to it."""
    z_pec = _282_z(SinusoidalSolver, 21)
    z_big = _282_z(SinusoidalSolver, 21, ground_eps=(1e12, 0.0))
    assert abs(z_big - z_pec) < 1e-3 * abs(z_pec), f"{z_big} vs PEC {z_pec}"


def test_282_brings_the_ground_shift_onto_the_cross_basis_reference():
    """The finite-ground SHIFT — Z(ground) − Z(PEC), the quantity feed-model
    differences cancel out of — now tracks the b-spline family instead of
    being 20x it. Measured at NS = 41: δ_bspline = −0.302−0.168j,
    δ_sinusoidal = −0.298−0.091j (was 1.312−20.232j).
    """
    ns = 41
    for ground in ("refl-coef soil", "sommerfeld soil"):
        kw = _282_GROUNDS[ground]
        d_s = (_282_z(SinusoidalSolver, ns, **kw) - _282_z(SinusoidalSolver, ns)) / abs(
            _282_z(SinusoidalSolver, ns)
        )
        d_b = (_282_z(BSplineSolver, ns, **kw) - _282_z(BSplineSolver, ns)) / abs(
            _282_z(BSplineSolver, ns)
        )
        mismatch = abs(d_s - d_b) / abs(d_b)
        assert mismatch < 0.30, (
            f"{ground}: shift mismatch {mismatch:.3f} (sin {d_s}, bsp {d_b})"
        )


# ----------------------------------------------------------------------
# The same node charge, under the EXTENDED kernel (momwire#292)
# ----------------------------------------------------------------------
#
# #282 wrote the subtraction on the REDUCED kernel's end-charge bracket —
# the isotropic (1+jkr₀)e^{−jkr₀}d/r₀³ field of a point charge on the axis.
# An EK-on fill carries EKSCX's bracket instead: the same charge smeared
# over the source tube's circumference, whose field is Eq 89's gradient and
# is O(a²/R²) ≈ 10% different at these meshes. Subtracting the wrong one of
# the two leaves that difference in the fill as a term that does NOT shrink
# under refinement, and the EK-on contact answer kept walking (spread 1.56
# over NS = 11 → 41 where EK-off had come down to 0.13). #292 builds the
# subtraction from the same per-end GXX quantities the fill uses.

_292_EK_GROUNDS = {
    "refl-coef soil": dict(ground_eps=(13.0, 0.005)),
    "sommerfeld soil": dict(ground_eps=(13.0, 0.005), ground_model="sommerfeld"),
}


def _292_z(cls, ns, ek, **ground):
    return _282_z(cls, ns, extended_kernel=ek, **ground)


@pytest.mark.parametrize("ground", list(_292_EK_GROUNDS))
def test_292_ek_contact_over_finite_ground_no_longer_diverges(ground):
    """The EK-ON ladder settles, and settles no worse than the EK-OFF one.

    Before #292 this ran 32.30+38.50j → 21.28+116.34j over NS = 11/21/41
    (spread 1.56) under refl-coef soil and 0.42 under Sommerfeld; it now
    tracks the EK-off ladder rung for rung.
    """
    kw = _292_EK_GROUNDS[ground]
    on = [_292_z(SinusoidalSolver, n, True, **kw) for n in _282_NS]
    off = [_292_z(SinusoidalSolver, n, False, **kw) for n in _282_NS]
    spread_on = abs(on[-1] - on[0]) / abs(on[0])
    spread_off = abs(off[-1] - off[0]) / abs(off[0])
    assert spread_on < 0.30, f"{ground}: EK-on spread {spread_on:.3f}: {on}"
    assert spread_on <= 1.5 * spread_off, (
        f"{ground}: EK-on spread {spread_on:.3f} against EK-off's "
        f"{spread_off:.3f} — the EKSCX end-charge bracket is not being used"
    )
    errs = [abs(v - on[-1]) for v in on[:-1]]
    assert all(b < a for a, b in zip(errs, errs[1:])), f"{ground}: not monotone: {on}"
    assert all(v.imag > 0 for v in on), f"{ground}: reactance went capacitive: {on}"
    # The kernel switch is a small perturbation of the answer, not a new one.
    for a, b in zip(on, off):
        assert abs(a - b) < 0.10 * abs(b), f"{ground}: EK-on {a} vs EK-off {b}"


@pytest.mark.parametrize("ground", list(_292_EK_GROUNDS))
def test_292_the_galerkin_contact_fill_gets_the_same_twin(ground):
    """The Galerkin solver carries the correction through its own test
    integration and scores eligibility with #246's per-PAIR rule, so it
    needs the twin too — and needed it more: its EK-on contact answer was
    58.79−236.52j at NS = 11 against 33.13+23.50j EK-off, stably wrong
    rather than walking.

    The reflection-coefficient ground used to be the whole of this test,
    because `SinusoidalGalerkinSolver` refused EK under Sommerfeld outright.
    momwire#287 lifted that refusal, so the C2-weighted Sommerfeld branch of
    `_contact_charge_ek_delta` is now reachable from this fill for the first
    time — and it lands in the same place: |on − off|/|off| is 0.004 / 0.005 /
    0.005 over NS = 11 / 21 / 41 under soil against refl-coef's 0.005 / 0.005 /
    0.008, and the ladder converges harder than the refl-coef one does
    (spread 0.039 against 0.128), the finite ground being a smaller
    perturbation of the PEC answer under a Sommerfeld evaluation than under
    the Φ-approximation #153 bounded to 0.1-0.5λ.
    """
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

    kw = _292_EK_GROUNDS[ground]
    z_on = []
    for ns in (11, 21, 41):
        on = _292_z(SinusoidalGalerkinSolver, ns, True, **kw)
        off = _292_z(SinusoidalGalerkinSolver, ns, False, **kw)
        assert abs(on - off) < 0.10 * abs(off), f"{ground} NS={ns}: {on} vs {off}"
        z_on.append(on)
    # The EK-on ladder settles rather than walking — the #292 failure mode was
    # an O(a²/R²) term that refinement could not remove.
    spread = abs(z_on[-1] - z_on[0]) / abs(z_on[0])
    assert spread < 0.30, f"{ground}: EK-on spread {spread:.3f}: {z_on}"


def test_292_leaves_an_elevated_ek_wire_untouched():
    """The twin is gated on a CONTACT exactly as #282's term is: a wire
    clear of the plane under EK keeps its exact fill."""
    s = SinusoidalSolver(
        wires=[np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0 + _282_H]])],
        n_per_edge_per_wire=[[21]],
        nsegs=21,
        wire_radius=0.02,
        feed_arclength=(_282_H / 21) * 0.5,
        ground_z=0.0,
        wavelength=_282_LAM,
        ground_eps=(13.0, 0.005),
        extended_kernel=True,
    )
    geom = s._build_geometry()
    G, seg_view = s._assemble_Z(geom, s.k)
    G2 = G.copy()
    s._contact_charge_correction(G2, geom, s.k, seg_view)
    assert np.array_equal(G, G2)


def _292_contact_solver(top, ek=True):
    """A contacting wire from the origin to `top`, over average soil."""
    return SinusoidalSolver(
        wires=[np.array([[0.0, 0.0, 0.0], top])],
        n_per_edge_per_wire=[[21]],
        nsegs=21,
        wire_radius=0.02,
        feed_arclength=(_282_H / 21) * 0.5,
        ground_z=0.0,
        wavelength=_282_LAM,
        ground_eps=(13.0, 0.005),
        extended_kernel=ek,
    )


def test_292_follows_the_fills_own_gating_at_the_contact_end():
    """The bracket is chosen by the same IND code the fill routed the end
    with, not by a global EK flag.

    NEC extends a ground contact only when the segment is PERPENDICULAR to
    the plane (CABJ² + SABJ² ≤ 1e-8); a slanted contact is IND = 2 and its
    end goes to the reduced GX, where #282's subtraction is already the
    right one. So the slanted deck must take no twin at all — and its EK-on
    fill must be bit-identical to what it was before #292 existed, which is
    what `_contact_ek_masks` returning None means.
    """
    vertical = _292_contact_solver(np.array([0.0, 0.0, _282_H]))
    geom = vertical._build_geometry()
    assert vertical._contact_ek_masks(geom, 0, -1.0, np.arange(21)) == (True, True)

    slanted = _292_contact_solver(np.array([_282_H * 0.6, 0.0, _282_H * 0.8]))
    geom = slanted._build_geometry()
    assert slanted._contact_ek_masks(geom, 0, -1.0, np.arange(21)) is None

    off = _292_contact_solver(np.array([0.0, 0.0, _282_H]), ek=False)
    assert off._contact_ek_masks(off._build_geometry(), 0, -1.0, np.arange(21)) is None


def test_292_gxx_returns_the_gradient_of_the_extended_kernel():
    """The derivation the twin rests on, checked numerically.

    `_contact_charge_end_gradient` spends NEC's GXX outputs as
    ∇G_ext = (G1P, G3) — the z- and ρ-derivatives of Eq 89's
    circumferentially averaged kernel, which is what makes them the field of
    the charge sitting at that end. The ρ half is the one that can be got
    wrong: T1 carries an explicit ρ² as well as the chain rule through R,
    which is exactly why GXX's IRA == 0 arm returns (G3 + GZP_T)·RH rather
    than G3·RH.
    """
    k = 2.0 * np.pi / _282_LAM
    gxx = SinusoidalSolver._ek_end_gxx

    def g1(z, rho, a):
        return gxx(k, np.asarray(z), np.asarray(rho), np.asarray(a), False)[0]

    for a, rho, z in ((0.02, 0.02, 0.15), (0.05, 0.31, 0.7), (0.08, 2.0, -1.3)):
        h = 1e-6 * max(rho, abs(z), 1.0)
        _g1, g1p, _g2, _g2p, g3, _gzp = gxx(
            k, np.asarray(z), np.asarray(rho), np.asarray(a), False
        )
        dz = (g1(z + h, rho, a) - g1(z - h, rho, a)) / (2.0 * h)
        dr = (g1(z, rho + h, a) - g1(z, rho - h, a)) / (2.0 * h)
        assert abs(g1p - dz) < 1e-6 * abs(dz), f"G1P at ({a},{rho},{z})"
        assert abs(g3 - dr) < 1e-6 * abs(dr), f"G3 at ({a},{rho},{z})"


def test_292_the_two_brackets_differ_by_the_expected_o_a2_amount():
    """The δ the fix removes, sized: on the #282 deck at NS = 21 the
    extended and reduced end-charge brackets differ by ~10% of the charge
    at the collocation points nearest the contact, and by more as the
    observer approaches it. That is the O(a²/R²) of Eq 89 and nothing
    else — it is NOT a small residual that refinement would wash out, which
    is why leaving it in made the ladder walk."""
    s = _292_contact_solver(np.array([0.0, 0.0, _282_H]))
    geom = s._build_geometry()
    obs_c, obs_t = geom["seg_centers"], geom["seg_tangents"]
    a_obs = s._seg_radius(geom)
    (i, sgn, node) = s._contact_nodes(geom)[0]
    red = s._contact_charge_kernel(geom, s.k, node, obs_c, obs_t, a_obs)
    delta = s._contact_charge_ek_delta(
        geom, s.k, i, sgn, node, obs_c, obs_t, a_obs, True, True
    )
    rel = np.abs(delta) / np.abs(red)
    assert 0.05 < rel[0] < 0.30, f"nearest-observer bracket delta {rel[0]:.4f}"
    # It falls off with distance — the correction is local to the node —
    # but never becomes the rounding-level term a converged fill could carry.
    assert rel[-1] < rel[0], f"delta does not decay along the wire: {rel}"


def _292_delta_at_tilted_observers(radius):
    """‖`_contact_charge_ek_delta`‖ on synthetic OFF-AXIS observers whose
    tangents tilt into the radial direction, and the radial share of it."""
    lam, H = 10.0, 2.0
    s = SinusoidalSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, H]])],
        n_per_edge_per_wire=[[21]],
        nsegs=21,
        wire_radius=radius,
        feed_arclength=(H / 21) * 0.5,
        ground_z=0.0,
        wavelength=lam,
        ground_eps=(13.0, 0.005),
        extended_kernel=True,
    )
    geom = s._build_geometry()
    ((i, sgn, node),) = s._contact_nodes(geom)
    obs_c = np.array([[0.3, 0.0, 0.4], [0.0, 0.25, 0.9], [0.2, 0.2, 0.1]])
    obs_t = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.2]])
    obs_t = obs_t / np.linalg.norm(obs_t, axis=1, keepdims=True)
    a_obs = np.full(3, radius)
    d = s._contact_charge_ek_delta(
        geom, s.k, i, sgn, node, obs_c, obs_t, a_obs, True, True
    )
    obs_t_ax = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    d_ax = s._contact_charge_ek_delta(
        geom, s.k, i, sgn, node, obs_c, obs_t_ax, a_obs, True, True
    )
    return np.abs(d).max(), np.abs(d - d_ax).max()


def test_292_the_delta_collapses_with_the_tube_radially_included():
    """δE = E_ext − E_red must vanish as O(a²) — WITH the radial half
    exercised.

    The ladder decks are vertical monopoles whose observers are coaxial with
    the contact, so `E_ρ` projects onto nothing there: a sign error in the
    radial component of `_contact_charge_end_gradient` sails through every
    ladder and bit-identity gate (measured: flipping `con·G3`'s sign leaves
    the whole g16/292 selection green). This gate uses off-axis observers
    with tilted tangents, checks the radial share is material (~56% of the
    delta at a = 0.02), and pins the a → 0 collapse the derivation demands:
    shrinking both radii 10× shrinks ‖δ‖ ~100× (measured ratio 0.0101).
    With the radial sign flipped the small-a delta converges to −2× the
    reduced radial field instead of zero and the ratio sits at ~1.0 — two
    orders off the pin.
    """
    big, big_radial = _292_delta_at_tilted_observers(0.02)
    small, _ = _292_delta_at_tilted_observers(0.002)
    assert big_radial > 0.3 * big, (
        f"radial share {big_radial:.3e} of {big:.3e} — observers no longer "
        "exercise the radial component; the collapse below proves nothing"
    )
    ratio = small / big
    assert ratio < 0.03, f"a->0 collapse ratio {ratio:.4f}, expected ~(1/10)^2"


# ----------------------------------------------------------------------
# What is left after #282: the contact solve has no mesh limit (#291)
# ----------------------------------------------------------------------


def test_291_contact_over_finite_ground_still_diverges_slowly():
    """#291 reframed with measurements: the post-#282 sinusoidal contact
    solve over a finite ground has NO mesh limit — it walks as ~N^0.4
    (increment magnitude grows ~1.3x per mesh doubling, measured out to
    NS = 1281), on BOTH sinusoidal schemes, with or without the
    Sommerfeld remainder, and under refl-coef too. The issue's
    "half-strength ground shift" is this walk sampled at NS = 81.

    Localization (2026-08-12, the #291 thread): an independent
    fine-quadrature implementation of the no-end-charge C2-image model
    reproduces the code's fill to machine precision at fine mesh — the
    #282/#292 subtraction class is COMPLETE, and the walk is intrinsic
    to collocating/Galerkin-testing the piecewise-sinusoid basis at the
    contact node, where the interface current is not representable. A
    real fix is a contact-region scheme change (e.g. a singular contact
    atom), not another correction column — and it must make THIS test
    fail loudly: the increments below stop growing and the total drift
    collapses to the b-spline reference's.
    """
    lam = 299.792458 / 30.0
    h = lam / 4

    def z(cls, ns, **over):
        kw = dict(
            wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h]])],
            n_per_edge_per_wire=[[ns]],
            nsegs=ns,
            wire_radius=0.0005,
            feed_arclength=(h / ns) * 0.5,
            ground_z=0.0,
            wavelength=lam,
            ground_eps=(13.0, 0.005),
            ground_model="sommerfeld",
        )
        kw.update(over)
        return complex(cls(**kw).compute_impedance()[0])

    # The mixed-potential reference barely moves over the same span.
    drift_b = abs(
        z(BSplineSolver, 321, degree=2, feed_model="segment")
        - z(BSplineSolver, 81, degree=2, feed_model="segment")
    )
    assert drift_b < 0.5, f"bspline reference moved {drift_b:.3f}"

    zs = [z(SinusoidalSolver, ns) for ns in (81, 161, 321, 641)]
    incs = [abs(b - a) for a, b in zip(zs, zs[1:])]
    # Measured 2026-08-12: 0.612, 0.832, 1.129 — growing ~1.36x per rung.
    total = abs(zs[-1] - zs[0])
    assert total > 1.5, (
        f"contact walk vanished (total drift {total:.3f} over NS 81->641) — "
        "if the sinusoidal contact now converges, celebrate: re-derive this "
        "test and the 0.2<gap<1.0 gate around the agreement (#291)"
    )
    assert all(b > 1.1 * a for a, b in zip(incs, incs[1:])), (
        f"the walk stopped accelerating ({incs}) — the divergence class "
        "changed; re-measure #291"
    )
