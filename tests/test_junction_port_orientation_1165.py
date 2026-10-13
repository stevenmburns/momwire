"""What a junction port's voltage and current mean -- momwire#1165.

The issue: one junction port on an L of two wires reads Re(Z) = -26.9 ohm,
identically on SinusoidalGalerkinSolver and BSplineSolver. A passive
antenna's input resistance cannot be negative, so is that a sign convention
or a defect?

Adjudicated by power balance, and it is neither. A junction port has ONE
terminal: it puts current into the wires at a node and nothing in the model
carries that current back, so one such port alone is not a circuit one-port
and its Re(Z) is not a radiation resistance. On the issue's deck the wire
currents radiate 5.94e-6 W while 0.5 Re(V I*) is -1.14e-4 W: the two are not
equal in size either, so no sign makes them agree. Close the current with a
lumped source between TWO junction ports on facing wire ends and everything
is ordinary: Z_AB = Z_AA + Z_BB - Z_AB - Z_BA is passive, its input power is
the radiated power to the size of the gap, and it tends to a delta gap's
impedance as the gap closes.

One defect did turn up on the way. Sin-Galerkin's junction port was reversed
against bspline's -- both V and I, so the port's own Y and every
junction-to-junction term were unaffected, but the mutual admittance between
a junction port and a gap feed had the opposite sign, and the wire currents
of a junction-port drive were minus the physical ones. Fixed here; the tests
below hold the two families to one orientation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

FAMILIES = [BSplineSolver, SinusoidalGalerkinSolver]
ETA = 376.730313668
WL7 = 299792458.0 / 7e6
ELL = [
    np.array([(0.0, 0.0, 1.0), (0.0, 0.0, 6.0)]),
    np.array([(0.0, 0.0, 6.0), (4.0, 0.0, 6.0)]),
]


def _solve(cls, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cls(wire_radius=1e-3, **kw)
        return s, s.compute_port_solution()


def _ell(cls, **ports):
    return _solve(
        cls,
        wires=ELL,
        n_per_edge_per_wire=[[10], [8]],
        junctions=[[(0, "end"), (1, "start")]],
        wavelength=WL7,
        **ports,
    )


def _gap_pair(cls, half, gap, wavelength, n=20):
    """Two collinear arms with facing ends `gap` apart, each facing end a
    one-member junction carrying a junction port. Port 0 is the lower arm."""
    wires = [
        np.array([(0.0, 0.0, -half), (0.0, 0.0, -0.5 * gap)]),
        np.array([(0.0, 0.0, 0.5 * gap), (0.0, 0.0, half)]),
    ]
    s, sol = _solve(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[n], [n]],
        junctions=[[(0, "end")], [(1, "start")]],
        wavelength=wavelength,
        feeds=[],
        junction_ports=[(0, 0.5 + 0j), (1, -0.5 + 0j)],
    )
    return s, sol, wires


def _radiated(s, coeffs, wires, k, ns=400, n_theta=64, n_phi=8):
    """Far-field power of the wire currents alone, by direct integration:
    (eta k^2 / 32 pi^2) * integral over the sphere of |N_perp|^2,
    N = sum I(l) t exp(jk r_hat.r) dl."""
    arcs, pts, tan, wts = [], [], [], []
    for poly in wires:
        length = np.linalg.norm(poly[1] - poly[0])
        t = (poly[1] - poly[0]) / length
        sm = (np.arange(ns) + 0.5) * length / ns
        arcs.append(sm)
        pts.append(poly[0] + sm[:, None] * t)
        tan.append(np.tile(t, (ns, 1)))
        wts.append(np.full(ns, length / ns))
    cur = np.concatenate([np.asarray(x) for x in s.currents_at_knots(coeffs, arcs)])
    pts, tan, wts = np.concatenate(pts), np.concatenate(tan), np.concatenate(wts)
    ct, wt = np.polynomial.legendre.leggauss(n_theta)
    ph = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    total = 0.0
    for c, w in zip(ct, wt):
        st = np.sqrt(1.0 - c * c)
        rh = np.stack([st * np.cos(ph), st * np.sin(ph), np.full(n_phi, c)], 1)
        n_vec = (np.exp(1j * k * rh @ pts.T) * (cur * wts)[None, :]) @ tan
        perp = np.sum(np.abs(n_vec) ** 2, 1) - np.abs(np.sum(n_vec * rh, 1)) ** 2
        total += w * perp.sum() * (2.0 * np.pi / n_phi)
    return ETA * k * k / (32.0 * np.pi**2) * total


# --------------------------------------------------------------------------
# the issue's number, and why it is not a resistance
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", FAMILIES)
def test_one_junction_port_reads_the_issues_impedance(cls):
    """Both families, the same number: a one-terminal port's Z."""
    _s, sol = _ell(cls, feeds=[], junction_ports=[(0, 1 + 0j)])
    z = 1.0 / sol.y[0, 0]
    assert z == pytest.approx(-26.887 - 342.33j, rel=1e-4)


@pytest.mark.parametrize("cls", FAMILIES)
def test_its_power_is_not_the_radiated_power_under_either_sign(cls):
    """0.5 Re(V I*) is -1.14e-4 W; the wire currents radiate 5.94e-6 W. A
    reversed reference direction would make them equal and opposite."""
    s, sol = _ell(cls, feeds=[], junction_ports=[(0, 1 + 0j)])
    p_in = 0.5 * sol.y[0, 0].real
    p_rad = _radiated(s, sol.coeffs[:, 0], ELL, 2.0 * np.pi / WL7, n_phi=48)
    assert p_in == pytest.approx(-1.1401e-4, rel=1e-3)
    assert p_rad == pytest.approx(5.940e-6, rel=1e-3)
    assert abs(p_in) > 10.0 * p_rad


# --------------------------------------------------------------------------
# the orientation: V is the node's potential, I goes into the wires
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", FAMILIES)
def test_a_positive_node_charges_its_arm(cls):
    """Two short arms are a capacitor. The arm held at +V must be charging:
    the current INTO it leads V by a quarter cycle, and that current is the
    port current."""
    s, sol, _w = _gap_pair(cls, half=0.3, gap=0.01, wavelength=10.0, n=12)
    v = np.array([0.5, -0.5])
    ends = s.currents_at_knots(sol.coeffs @ v)
    into_lower = -ends[0][-1]  # the lower arm runs TOWARD its node
    into_upper = ends[1][0]
    i_port = np.asarray(sol.y) @ v
    assert into_lower.imag > 0.0 > into_upper.imag
    assert into_lower == pytest.approx(i_port[0], rel=1e-9)
    assert into_upper == pytest.approx(i_port[1], rel=1e-9)


@pytest.mark.parametrize("cls", FAMILIES)
def test_the_wire_currents_are_the_port_currents(cls):
    """A gap feed and a junction port together: in each column the current
    leaving the node along the wires is that column's junction-port current,
    and the current at the feed is its feed current."""
    s, sol = _ell(cls, feeds=[(0, 2.5, 1 + 0j)], junction_ports=[(0, 1 + 0j)])
    y = np.asarray(sol.y)
    for col in (0, 1):
        kn = s.currents_at_knots(sol.coeffs[:, col])
        out_of_node = -kn[0][-1] + kn[1][0]
        assert out_of_node == pytest.approx(y[1, col], rel=1e-6)
        assert kn[0][5] == pytest.approx(y[0, col], rel=1e-6)


def test_the_two_families_agree_on_a_feed_and_a_junction_port():
    """The mutual admittance carries the orientation. Sin-Galerkin read
    -6.85e-5 + 9.44e-4j S where bspline reads +6.85e-5 - 9.44e-4j."""
    ys = [
        np.asarray(
            _ell(cls, feeds=[(0, 2.5, 1 + 0j)], junction_ports=[(0, 1 + 0j)])[1].y
        )
        for cls in FAMILIES
    ]
    assert ys[0][0, 1] == pytest.approx(6.8508e-5 - 9.4389e-4j, rel=1e-4)
    assert np.allclose(ys[1], ys[0], rtol=1e-4, atol=0.0)
    for y in ys:
        assert y[0, 1] == pytest.approx(y[1, 0], rel=1e-6)


# --------------------------------------------------------------------------
# closed by a lumped source, the pair is an ordinary port
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", FAMILIES)
def test_a_pair_across_a_gap_is_passive_and_balances_power(cls):
    """A half-wave-ish dipole (2 x 2.4 m at 10 m). Each port's own Re(Z) is
    negative; the pair's is not, its input power is the radiated power to
    the size of the gap (the lumped source's own length is not in the
    model), and it closes on the delta gap's 70.41 - 9.59j ohm."""
    k = 2.0 * np.pi / 10.0
    drive = np.array([1.0, -1.0])  # 1 A out of port 0, back into port 1
    z_pair = {}
    for gap, tol in ((0.2, 0.08), (0.05, 0.02), (0.01, 0.005)):
        s, sol, wires = _gap_pair(cls, half=2.4, gap=gap, wavelength=10.0)
        z = np.linalg.inv(np.asarray(sol.y))
        assert z[0, 0].real < 0.0 and z[1, 1].real < 0.0
        z_pair[gap] = z[0, 0] + z[1, 1] - z[0, 1] - z[1, 0]
        v = z @ drive
        p_in = 0.5 * np.real(np.vdot(drive, v))
        assert p_in == pytest.approx(0.5 * z_pair[gap].real)
        p_rad = _radiated(s, sol.coeffs @ v, wires, k)
        assert p_in > 0.0
        assert p_in == pytest.approx(p_rad, rel=tol)
    _s, sol = _solve(
        cls,
        wires=[np.array([(0.0, 0.0, -2.4), (0.0, 0.0, 2.4)])],
        n_per_edge_per_wire=[[40]],
        wavelength=10.0,
        feeds=[(0, 2.4, 1 + 0j)],
    )
    z_gap = 1.0 / sol.y[0, 0]
    assert z_gap == pytest.approx(70.41 - 9.59j, abs=0.02)
    # the resistance closes on the delta gap's as the gap does
    err = [abs(z_pair[g].real - z_gap.real) for g in (0.2, 0.05, 0.01)]
    assert err[0] > err[1] > err[2] and err[2] < 1.0
