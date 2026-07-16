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
