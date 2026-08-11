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
