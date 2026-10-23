"""Near fields with buried wires, and at points in the soil (momwire#1653
unit 1: #524 phase 3, #570's NE/NH half, #1257).

The readout contract and the four (source side, observer side) regimes are
`momwire._near_buried`'s docstring. The gates here are the ones the issue
names, each measuring the readout rather than the solve:

  G1  continuity across z = 0: tangential E, normal D = e*E_z, all of H,
      at points approaching the plane from both sides, falling with the
      approach distance as a field with no jump must;
  G2  reciprocity: <E_A, J_B> = <E_B, J_A> for a buried and an above probe;
  G3  convergence under mesh refinement;
  G4  the far-zone limit: the near readout at range tends to the served
      transmitted far-zone factors as 1/R;
  G5  the above/above regime is untouched;
  G6  the black boxes: NEC-5 (exe/nec5-ripplefix) printouts of decks whose
      solves agree with razor-2p's (tests/fixtures/near_buried_1653/README.txt).

Two identities ride along: e = 1 collapses the transmitted field onto the
free-space element sum in both directions, and the mixed-potential
transmitted field agrees with the independent field-form tables
(`_sommerfeld_transmitted.transmitted_field_below_to_above`).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from momwire import RazorSolver, _near_buried, _near_readout
from momwire import _sommerfeld_transmitted
from momwire._far_readout import ETA0, _element_fields, transmitted_moments
from momwire._sommerfeld_below import k_medium
from momwire.bspline import BSplineSolver
from momwire.eznec._shell import render

C0 = 299792458.0
F7 = 7e6
WL = C0 / F7
K = 2.0 * math.pi / WL
SOIL = (13.0, 0.005)
SUB = 8
FIXTURES = Path(__file__).parent / "fixtures" / "near_buried_1653"

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _solve(cls, wires, npe, feed, junctions=None):
    kw = {} if junctions is None else {"junctions": junctions}
    s = cls(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[feed],
        wavelength=WL,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL,
        ground_model="sommerfeld",
        **kw,
    )
    return s, s.compute_port_solution().coeffs[:, 0]


def _field(s, c, pts, magnetic=False, sub=SUB):
    el = s.element_currents(c, subdiv=sub)
    return _near_readout.near_field_at(
        np.asarray(pts, dtype=float),
        el,
        K,
        0.001,
        magnetic,
        _near_readout.near_ground(s),
        s.omega,
        counts=_near_buried.element_counts(s),
        subdiv=sub,
    )


def _buried_dipole(cls=BSplineSolver, n=11):
    return _solve(
        cls, [np.array([(-2.5, 0.0, -0.3), (2.5, 0.0, -0.3)])], [[n]], (0, 2.5, 1 + 0j)
    )


def _above_dipole(cls=BSplineSolver):
    return _solve(
        cls, [np.array([(-5.0, 0.0, 1.0), (5.0, 0.0, 1.0)])], [[11]], (0, 5.0, 1 + 0j)
    )


def _hub(cls=BSplineSolver):
    """A 10 m vertical rising from the z = 0 crossing node over a buried
    drop to a hub 0.15 m down and four 5 m radials from the hub."""
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]),
        np.array([(0.0, 0.0, -0.15), (0.0, 0.0, 0.0)]),
    ]
    npe = [[16], [2]]
    junctions = [[(0, "start"), (1, "end")], [(1, "start")]]
    for i, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        wires.append(np.array([(0.0, 0.0, -0.15), (5.0 * dx, 5.0 * dy, -0.15)]))
        npe.append([10])
        junctions[1].append((2 + i, "start"))
    return _solve(cls, wires, npe, (0, 1.5 * 10.0 / 16.0, 1 + 0j), junctions)


# --------------------------------------------------------------------------
# identities
# --------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["below", "above"])
def test_eps_one_collapses_the_transmitted_field_onto_free_space(side):
    """At e = 1 there is no interface: the WHOLE transmitted field, in both
    directions, is the free-space mixed-potential element sum, charges and
    all. E to the kernels' contour tolerance, H to its curl stencil."""
    z = (-0.3, -0.6) if side == "below" else (0.3, 0.9)
    s = BSplineSolver(
        wires=[np.array([(-2.5, 0.0, z[0]), (2.5, -0.5, z[1])])],
        n_per_edge_per_wire=[[11]],
        feeds=[(0, 2.5, 1 + 0j)],
        wavelength=WL,
        wire_radius=0.001,
    )
    c = s.compute_port_solution().coeffs[:, 0]
    el = s.element_currents(c, subdiv=SUB)
    ground = _near_readout.NearGround("compose", 0.0, 1.0 + 0j, 0j)
    # observers on the OTHER side, so every pair is a crossing pair
    sgn = 1.0 if side == "below" else -1.0
    pts = np.array([(0.7, 0.4, 0.2), (3, 1, 0.01), (10, 3, 2), (1, -1, 1e-4)])
    pts[:, 2] *= sgn
    for magnetic, tol in ((False, 2e-9), (True, 6e-5)):
        got = _near_readout.near_field_at(
            pts,
            el,
            K,
            0.001,
            magnetic,
            ground,
            s.omega,
            counts=_near_buried.element_counts(s),
            subdiv=SUB,
        )
        want = _element_fields(pts, el, K, 0.001, magnetic)
        rel = np.max(np.abs(got - want), axis=1) / np.max(np.abs(want), axis=1)
        assert float(np.max(rel)) < tol, (magnetic, rel)


def test_the_mixed_potential_transmitted_field_is_the_field_form_tables():
    """Two independent spellings of the same field: the designed kernels'
    mixed-potential sum (charges at the knots, direct contour values) and
    the transmitted family's field-form TABLES over a z' ladder (moments
    only, interpolated)."""
    s, c = _buried_dipole()
    g = _near_readout.near_ground(s)
    el = s.element_currents(c, subdiv=SUB)
    pts = np.array([(4.0, 1.0, 0.5), (6, -2, 1.5), (1, 3, 0.2), (10, 0, 2), (0, 5, 3)])
    src = _near_buried.Sources(el, [el[0].shape[0]], [True])
    a = _near_buried.transmitted_field_at(
        pts, src, g.eps_tilde, 0.0, K, s.omega, 0.001, observers_above=True
    )
    grid = _sommerfeld_transmitted.get_grid_below_above(
        g.eps_tilde, K, 30.0, -0.3, -0.3, s.omega
    )
    b = _sommerfeld_transmitted.transmitted_field_below_to_above(
        pts, el[0], el[1], 0.0, K, k_medium(g.eps_tilde, K), grid
    )
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)
    assert float(np.max(rel)) < 3e-4, rel  # measured 9.5e-5


# --------------------------------------------------------------------------
# G1 continuity across z = 0
# --------------------------------------------------------------------------


def _jumps(s, c, xy, d):
    eps = _near_readout.near_ground(s).eps_tilde
    pts = np.array([(x, y, sgn * d) for x, y in xy for sgn in (1.0, -1.0)])
    e = _field(s, c, pts)
    h = _field(s, c, pts, magnetic=True)
    tan, nrm, mag = [], [], []
    for i in range(len(xy)):
        ea, eb, ha, hb = e[2 * i], e[2 * i + 1], h[2 * i], h[2 * i + 1]
        scale = np.linalg.norm(ea)
        tan.append(np.linalg.norm(ea[:2] - eb[:2]) / scale)
        nrm.append(abs(ea[2] - eps * eb[2]) / scale)
        mag.append(np.linalg.norm(ha - hb) / np.linalg.norm(ha))
    return max(tan), max(nrm), max(mag)


_XY = [(0.7, 0.4), (3.0, 1.0), (0.0, 2.0)]


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver], ids=["bs2", "razor-2p"])
@pytest.mark.parametrize("source", ["below", "above"])
def test_g1_the_field_is_continuous_across_the_interface(cls, source, record_property):
    """Tangential E, normal D and H match across z = 0, and the mismatch
    falls with the approach distance d (it is the field's own gradient
    times 2d, not a jump). Measured at d = 1e-5 m, relative to |E| / |H|:
    tangential E <= 4e-5, normal D <= 3e-4, H <= 5e-4 (the H floor is the
    remainders' curl stencil)."""
    s, c = (_buried_dipole if source == "below" else _above_dipole)(cls)
    far = _jumps(s, c, _XY, 1e-3)
    near = _jumps(s, c, _XY, 1e-5)
    record_property(
        "jumps_d1e-5", f"tanE {near[0]:.2e} Dn {near[1]:.2e} H {near[2]:.2e}"
    )
    assert near[0] < 2e-4 and near[1] < 1e-3 and near[2] < 2e-3, near
    # tangential E and normal D fall ~linearly with d: 100x closer, >= 20x
    # smaller
    assert near[0] < far[0] / 20.0 and near[1] < far[1] / 5.0, (far, near)


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver], ids=["bs2", "razor-2p"])
def test_g1_continuity_on_a_crossing_deck(cls, record_property):
    """Both source sides at once: a vertical rising from the crossing node
    over a buried hub and radials. Every observer sees all four regimes'
    worth of sources."""
    s, c = _hub(cls)
    near = _jumps(s, c, [(2.0, 0.7), (0.6, 3.1), (7.0, -2.0)], 1e-5)
    record_property(
        "jumps_d1e-5", f"tanE {near[0]:.2e} Dn {near[1]:.2e} H {near[2]:.2e}"
    )
    assert near[0] < 2e-4 and near[1] < 1e-3 and near[2] < 2e-3, near


# --------------------------------------------------------------------------
# G2 reciprocity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver], ids=["bs2", "razor-2p"])
def test_g2_reaction_between_a_buried_and_an_above_probe_is_reciprocal(
    cls, record_property
):
    """<E_A, J_B> (the above dipole's field on the buried one's current) is
    <E_B, J_A> (the reverse): the downward transmitted regime against the
    upward one, through the readout, read at subdiv 4. On a 10 m / 5 m pair
    it measured 3.8e-6 / 9.6e-7 / 2.4e-7 at subdiv 4 / 8 / 16 (bs2; razor-2p
    alike), falling as subdiv^-2: it is the readout's own midpoint
    quadrature, not a regime mismatch."""
    a_wire = np.array([(-3.0, 1.0, 1.0), (3.0, 1.0, 1.5)])
    b_wire = np.array([(-2.5, 0.0, -0.3), (2.5, 0.5, -0.4)])
    sa, ca = _solve(cls, [a_wire], [[11]], (0, 3.0, 1 + 0j))
    sb, cb = _solve(cls, [b_wire], [[11]], (0, 2.5, 1 + 0j))
    ea = sa.element_currents(ca, subdiv=4)
    eb = sb.element_currents(cb, subdiv=4)
    r_ab = np.sum(_field(sa, ca, eb[0], sub=4) * eb[1])
    r_ba = np.sum(_field(sb, cb, ea[0], sub=4) * ea[1])
    rel = abs(r_ab - r_ba) / abs(r_ab)
    record_property("reciprocity_rel", f"{rel:.3e}")
    # measured 7.4e-5 (bs2) / 7.0e-5 (razor-2p) on this closer pair
    assert rel < 2e-4, (r_ab, r_ba)


# --------------------------------------------------------------------------
# G3 convergence
# --------------------------------------------------------------------------


def test_g3_the_readout_converges_under_resampling(record_property):
    """The readout's own refinement: the crossing deck's solved current
    resampled ever finer (`element_currents(subdiv=...)`), read at points
    on both sides down to 0.1 m from a buried radial. Second order, as a
    midpoint rule should be: measured E 5.4e-3 -> 8.1e-4 -> 2.0e-4 and H
    2.7e-2 -> 5.3e-3 -> 1.3e-3 over subdiv 4 -> 8 -> 16 -> 32 (worst
    point, the one 0.1 m from a radial)."""
    s, c = _hub()
    pts = [(1.0, 0.5, 0.2), (3.0, 1.0, -0.1), (0.5, 0.3, -0.6), (8.0, 2.0, 0.5)]
    pts += [(1.0, 0.0, 0.0), (0.3, 0.1, -0.25)]
    prev, steps = None, []
    for sub in (4, 8, 16, 32):
        e = _field(s, c, pts, sub=sub)
        h = _field(s, c, pts, magnetic=True, sub=sub)
        if prev is not None:
            steps.append(
                tuple(
                    float(
                        np.max(
                            np.linalg.norm(x - y, axis=1) / np.linalg.norm(x, axis=1)
                        )
                    )
                    for x, y in ((e, prev[0]), (h, prev[1]))
                )
            )
        prev = (e, h)
    record_property("resampling_steps", repr(steps))
    for a, b in zip(steps, steps[1:]):
        assert b[0] < a[0] / 3.0 and b[1] < a[1] / 3.0, steps
    assert steps[-1][0] < 1e-3 and steps[-1][1] < 5e-3, steps


@pytest.mark.slow
def test_g3_the_buried_near_field_converges_under_mesh_refinement(record_property):
    """bs2's buried dipole re-meshed: the near field moves by LESS than the
    driving-point impedance does at every doubling, and both fall. The
    impedance's own convergence is the delta-gap's (logarithmic, X moves
    1.3, 0.8, 0.5 ohm per doubling), so it sets the pace and the readout
    adds nothing slower. Measured field / Z steps: 8.3e-4 / 1.3e-2,
    5.1e-4 / 7.8e-3 over n = 21 -> 41 -> 81."""
    pts = [(1.0, 0.5, 0.2), (3.0, 1.0, -0.1), (0.5, 0.3, -0.6), (8.0, 2.0, 0.5)]
    pts += [(1.0, 0.0, 0.0)]
    prev, steps = None, []
    for n in (21, 41, 81):
        s, c = _buried_dipole(n=n)
        z = 1.0 / complex(np.asarray(s.compute_port_solution().y).ravel()[0])
        e = _field(s, c, pts)
        if prev is not None:
            de = float(
                np.max(np.linalg.norm(e - prev[0], axis=1) / np.linalg.norm(e, axis=1))
            )
            steps.append((de, abs(z - prev[1]) / abs(z)))
        prev = (e, z)
    record_property("mesh_steps", repr(steps))
    assert all(de < dz for de, dz in steps), steps
    assert steps[1][0] < steps[0][0] and steps[1][1] < steps[0][1], steps


# --------------------------------------------------------------------------
# G4 the far-zone limit
# --------------------------------------------------------------------------


def test_g4_the_near_field_tends_to_the_transmitted_far_field(record_property):
    """A tilted buried dipole read at R over theta 20/45/70 x phi 0/30/90
    deg, against `transmitted_moments` (the served pattern's factors) on
    the SAME element sampling. Measured worst 2.4e-2 at 23 wl, 5.9e-3 at
    93 wl, 1.5e-3 at 374 wl: 1/R, which is the near-field correction
    dying away, and nothing left over."""
    s, c = _solve(
        BSplineSolver,
        [np.array([(-2.5, 0.3, -0.3), (2.5, -0.2, -0.5)])],
        [[11]],
        (0, 2.5, 1 + 0j),
    )
    g = _near_readout.near_ground(s)
    el = s.element_currents(c, subdiv=SUB)
    th = np.radians([20.0, 45.0, 70.0])
    ph = np.radians([0.0, 30.0, 90.0])
    m_t, m_p = transmitted_moments(
        el[0], el[1], K, k_medium(g.eps_tilde, K), th, ph, 0.0
    )
    worst = {}
    for r in (1000.0, 4000.0):
        pts, far = [], []
        for i, t in enumerate(th):
            for j, p in enumerate(ph):
                pts.append(
                    r
                    * np.array(
                        [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]
                    )
                )
                t_hat = np.array(
                    [np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)]
                )
                p_hat = np.array([-np.sin(p), np.cos(p), 0.0])
                pref = -1j * ETA0 * K / (4.0 * math.pi) * np.exp(-1j * K * r) / r
                far.append(pref * (m_t[i, j] * t_hat + m_p[i, j] * p_hat))
        near = _field(s, c, pts)
        far = np.array(far)
        worst[r] = float(
            np.max(np.linalg.norm(near - far, axis=1) / np.linalg.norm(far, axis=1))
        )
    record_property("far_zone", " ".join(f"{r:g}:{w:.2e}" for r, w in worst.items()))
    assert worst[4000.0] < 1e-2, worst
    assert 3.0 < worst[1000.0] / worst[4000.0] < 5.0, worst


# --------------------------------------------------------------------------
# G5 above/above untouched
# --------------------------------------------------------------------------


def test_g5_a_deck_wholly_above_never_enters_the_buried_composition(monkeypatch):
    """The four-regime composition is reached only by a source element or an
    observer below the plane; an above-ground deck and grid take the code
    path they always took (its numbers are pinned bit for bit by the golden
    printouts and tests/test_field_point.py)."""

    def boom(
        *args, **kwargs
    ):  # pragma: no cover - the assertion is that it is not called
        raise AssertionError("the buried composition ran on an above-ground deck")

    monkeypatch.setattr(_near_readout, "_near_field_buried", boom)
    s, c = _above_dipole()
    e = _field(s, c, [(1.0, 0.5, 0.2), (3.0, 1.0, 0.0), (8.0, 2.0, 5.0)])
    assert np.all(np.isfinite(e))


# --------------------------------------------------------------------------
# G6 the black boxes
# --------------------------------------------------------------------------

_ROW = re.compile(r"\s+-?\d\.\d+E[+-]\d+")


def _tables(text):
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        if "NEAR ELECTRIC FIELDS" in lines[i] or "NEAR MAGNETIC FIELDS" in lines[i]:
            j = i + 1
            while j < len(lines) and not _ROW.match(lines[j]):
                j += 1
            rows = []
            while j < len(lines) and _ROW.match(lines[j]):
                v = [float(t) for t in lines[j].split()]
                rows.append(
                    (
                        v[:3],
                        [
                            v[3 + 2 * m] * np.exp(1j * math.radians(v[4 + 2 * m]))
                            for m in range(3)
                        ],
                    )
                )
                j += 1
            out.append((np.array([r[0] for r in rows]), np.array([r[1] for r in rows])))
            i = j
        else:
            i += 1
    return out


@pytest.mark.parametrize("stem", ["d1_buried_dipole", "d2_elev_screen"])
def test_g6_razor_2p_reads_the_nec5_reference_near_fields(stem, record_property):
    """Decks whose razor-2p solve agrees with NEC-5's (Z to 0.03 %): every
    NE/NH table - 1 cm above the surface, 5 cm and 50 cm into it, out to
    210 m and at R = 300 m - agrees with exe/nec5-ripplefix's printout.
    Measured per-table median 1.8e-4 .. 8.6e-4 relative per point, and at
    R = 300 m (the last table) 1.9e-3 / 3.6e-3: that one is the ABOVE
    wire's reflected remainder at range (d2's vertical; the unchanged
    above/above regime), whose bar is the grazing band's 1e-2."""
    ours = _tables(
        render((FIXTURES / f"{stem}.nec5.nec").read_text(), basis="razor-2p")
    )
    ref = _tables((FIXTURES / f"{stem}.nec5.ripplefix.out").read_text(errors="replace"))
    assert len(ours) == len(ref) == 9
    medians = []
    for i, ((pa, fa), (pb, fb)) in enumerate(zip(ours, ref, strict=True)):
        np.testing.assert_allclose(pa, pb, atol=1e-3)
        per_point = np.linalg.norm(fa - fb, axis=1) / np.linalg.norm(fb, axis=1)
        medians.append(float(np.median(per_point)) / (5.0 if i == 8 else 1.0))
        scale = np.max(np.linalg.norm(fb, axis=1))
        assert float(np.max(np.linalg.norm(fa - fb, axis=1))) / scale < 2e-2
    record_property("medians", " ".join(f"{m:.1e}" for m in medians))
    assert max(medians) < 2e-3, medians
