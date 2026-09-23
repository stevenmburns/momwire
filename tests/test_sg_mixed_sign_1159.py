"""momwire#1159: `SinusoidalGalerkinSolver` on a deck with wires on both sides
of the interface — the cross-medium coupling's SIGN, the k a buried entry is
driven and read at, and the port solution of a wholly-buried deck.

Three defects, one family: each was a place where SG's mixed/buried route
used a spelling that is right only somewhere else.

* **The transmitted pair class was subtracted.** The projected field tables
  `t_m . D . t_n` enter THIS trunk's G with a plus — the within-class
  remainder arrives through `_fold_ground_block`'s `free - (c2*img - rem)`,
  and the crossing block is `G + t_ab + t_ab.T` — while bspline, whose Z has
  the opposite global sign, subtracts them (`_sub_field_galerkin`). The copied
  minus negated both cross quadrants: D*G*D with D = diag(I_above, -I_below).
  Every single-port Z is EXACTLY invariant under that, which is why every D2
  gate (one feed, the eps~ = 1 collapse on Z) was blind to it; a multi-port
  Y, a both-ports-driven Z, the unfed medium's current and the far field are
  not.
* **A mixed deck's drive and port readout wrote a buried entry's shapes at
  air's k** (`_entry_k`): a mixed deck solves outside `_operating_medium`, so
  the k those were handed is k_p. Invisible at a segment-centre gap; live at a
  knot gap, which is where every deck below feeds.
* **The current readout rebuilt the basis in air** (`_readout_view`): a
  buried entry read with k_p's coefficients, and a crossing deck's node wings
  dropped outright.

Plus the port solution on a wholly-buried deck, which never entered
`_operating_medium` and so refused the deck through the above-ground family.

The gates are reference-free where the physics allows: at eps~ = 1 a mixed
or crossing deck IS free space, so SG's own free-space solve on the same wires
is the answer. Elsewhere bspline is the second opinion, gated on CONVERGENCE
(the gap has to fall under refinement), never on agreement at one mesh.
Every red control reproduces the pre-fix arithmetic by patching one seam,
bit for bit, so a control that passed would mean the gate measured nothing.

Measured 2026-09-23, soil A, 7 MHz (scratch/1159-sg-mixed-sign):

    eps~ = 1 mixed, Y rel per entry    diag 2.0e-8, off-diag 6.4e-6 (pre-fix 2.000)
    eps~ = 1 mixed, currents (2 ports) 3.0e-7 per wire          (pre-fix 8.6e-2)
    eps~ = 1 crossing, Y / currents    3.9e-6 / 1.1e-4 (node knot; pre-fix 1.2e-2)
    mixed(m) vs bspline, m = 1/2/4     Y12 1.1e-4 / 1.6e-5 / 2.7e-6
                                       Y11 4.9e-4 / 7.2e-5 / 1.1e-5 (was 3.7e-2 / 2.0e-2 / 1.1e-2)
    mixed(m) far field, above port     8.1e-6 / 1.4e-6 / 2.0e-7  (sign alone: 2.0e-1)
    wholly buried vs bspline, m = 1/2  Y12 1.4e-4 / 2.6e-5, currents 1.1e-4 / 1.7e-5
                                       (pre-fix readout: 0.96, flat)
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

from momwire import BSplineSolver, SinusoidalGalerkinSolver
from momwire import sinusoidal_galerkin as _sg
from momwire._far_readout import Ground, _far_moments
from momwire.sinusoidal import SinusoidalSolver

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_razor_detached_1149 import detached, detached_hub  # noqa: E402

C0 = 299792458.0
WL7 = C0 / 7e6
SOIL_A = (13.0, 0.005)
EPS_ONE = (1.0, 0.0)
GROUND_KEYS = ("ground_z", "ground_eps", "ground_model")


def mixed(m=1, eps=SOIL_A):
    """A buried centre-fed wire (5 m at depth 0.5) beside a centre-fed wire
    1 m above the plane, refined EVERYWHERE by m. Both gaps sit on a KNOT
    (feed_xi = +-h/2), which is what makes the drive's k observable."""
    return dict(
        wires=[
            np.array([(-2.5, 0.0, -0.5), (2.5, 0.0, -0.5)]),
            np.array([(-2.5, 0.0, 1.0), (2.5, 0.0, 1.0)]),
        ],
        n_per_edge_per_wire=[[10 * m], [10 * m]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        ground_z=0.0,
        ground_eps=eps,
        ground_model="sommerfeld",
    )


def crossing(eps=EPS_ONE, n=15):
    """`test_sg_crossing_980_d3`'s deck with a second port on the buried
    wire: the rise and the buried leg joined at a node IN the plane."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
            np.array([(0.0, 0.0, 0.0), (2.0, 0.0, -0.5)]),
        ],
        n_per_edge_per_wire=[[n], [n]],
        feeds=[(0, 1.0, 1 + 0j), (1, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        junctions=[[(0, "start"), (1, "start")]],
        ground_z=0.0,
        ground_eps=eps,
        ground_model="sommerfeld",
    )


def buried(m=1, eps=SOIL_A):
    """Two parallel WHOLLY-buried wires, one centre port each."""
    return dict(
        wires=[
            np.array([(-2.5, 0.0, -0.5), (2.5, 0.0, -0.5)]),
            np.array([(-2.5, 1.0, -1.0), (2.5, 1.0, -1.0)]),
        ],
        n_per_edge_per_wire=[[10 * m + 1], [10 * m + 1]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        ground_z=0.0,
        ground_eps=eps,
        ground_model="sommerfeld",
    )


def free_space(d):
    return {k: v for k, v in d.items() if k not in GROUND_KEYS}


def solve(cls, d, drive=None):
    """(Y, per-wire knot currents) with `drive` volts per port (all 1 V by
    default), through the public port solution and `currents_at_knots`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cls(**d)
        sol = s.compute_port_solution()
        v = np.ones(sol.coeffs.shape[1]) if drive is None else np.asarray(drive)
        cur = s.currents_at_knots(sol.coeffs @ v)
    return np.asarray(sol.y), [np.asarray(c) for c in cur]


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))


def far_pattern(cls, d, drive=None):
    """The EZNEC serve's own route: `_far_moments` over `element_currents`,
    upper hemisphere, three phi cuts. Complex (M_theta, M_phi) stacked."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cls(**d)
        sol = s.compute_port_solution()
        v = np.ones(sol.coeffs.shape[1]) if drive is None else np.asarray(drive)
        mid, moment, _n, _d = s.element_currents(sol.coeffs @ v)
    eps_r, sigma = d["ground_eps"]
    mt, mp = _far_moments(
        mid,
        moment,
        2 * np.pi / d["wavelength"],
        np.radians(np.linspace(0.0, 89.0, 90)),
        np.radians([0.0, 45.0, 90.0]),
        Ground("sommerfeld", eps_r, sigma),
        0.0,
        C0 / d["wavelength"],
    )
    return np.stack([mt, mp])


def _decays(gaps, bar=3.0):
    ratios = [a / b for a, b in zip(gaps, gaps[1:])]
    assert all(r >= bar for r in ratios), (gaps, ratios)


# --- the pre-fix arithmetic, one seam each ---------------------------------


def _negate_transmitted(monkeypatch):
    """The pre-fix fill: negating the tensor turns `+= r` into `-= r`, the
    subtraction the trunk shipped with, bit for bit."""
    orig = SinusoidalGalerkinSolver._transmitted_tensor
    calls = []

    def neg(self, *a, **k):
        calls.append(1)
        return -orig(self, *a, **k)

    monkeypatch.setattr(SinusoidalGalerkinSolver, "_transmitted_tensor", neg)
    return calls


def _air_readout(monkeypatch):
    """The pre-fix readout: the inherited `_basis_coefs(geom, self.k)`."""
    monkeypatch.setattr(
        SinusoidalGalerkinSolver, "_readout_view", SinusoidalSolver._readout_view
    )


def _air_drive(monkeypatch):
    """The pre-fix drive and port readout: every site handed `self.k`, which
    on a mixed deck is k_p."""
    monkeypatch.setattr(_sg, "_entry_k", lambda seg_view, s, e, k: k)


# ----------------------------------------------------------------------
# eps~ = 1: the mixed deck IS free space — full Y, every wire's current
# ----------------------------------------------------------------------


def test_eps_one_mixed_full_y_matrix_collapses():
    """Every entry of Y, not Z11. The off-diagonal floor is the transmitted
    GRID's own accuracy (bspline carries the same 7.5e-5 on Y12 in the
    issue's table); the diagonal is rounding."""
    y1, _ = solve(SinusoidalGalerkinSolver, mixed(2, EPS_ONE))
    yf, _ = solve(SinusoidalGalerkinSolver, free_space(mixed(2)))
    err = np.abs(y1 - yf) / np.abs(yf)
    assert err.max() < 1e-4, err
    assert np.diag(err).max() < 1e-6, err


def test_eps_one_mixed_currents_collapse_with_both_ports_driven():
    """Both ports driven, so every wire's current carries the cross term."""
    _, c1 = solve(SinusoidalGalerkinSolver, mixed(2, EPS_ONE))
    _, cf = solve(SinusoidalGalerkinSolver, free_space(mixed(2)))
    for a, b in zip(c1, cf):
        assert rel(a, b) < 1e-5


def test_red_control_the_pre_fix_fill_fails_the_collapse_by_sign(monkeypatch):
    """The subtracted transmitted block: Y11 untouched, Y12 exactly minus the
    free-space one — which is what an inversion is and a tolerance is not.
    Counted, so the control cannot pass on a route that skipped the block."""
    yf, cf = solve(SinusoidalGalerkinSolver, free_space(mixed(2)))
    calls = _negate_transmitted(monkeypatch)
    y1, c1 = solve(SinusoidalGalerkinSolver, mixed(2, EPS_ONE))
    assert len(calls) == 2  # both directions filled
    assert abs(y1[0, 0] - yf[0, 0]) / abs(yf[0, 0]) < 1e-6
    assert abs(y1[0, 1] - yf[0, 1]) / abs(yf[0, 1]) > 1.99
    assert abs(y1[0, 1] + yf[0, 1]) / abs(yf[0, 1]) < 1e-4
    assert min(rel(a, b) for a, b in zip(c1, cf)) > 1e-2


def test_eps_one_crossing_full_y_and_currents_collapse():
    """A crossing deck has no transmitted grid (its cross pair is
    `_crossing_fill`'s, added with a plus since D3), so its Y was never
    inverted — but its READOUT dropped the node wings. The current floor is
    the node knot, where the crossing deck's wings and the free deck's
    ordinary junction are different bases (2.5e-4 there, ~5e-6 elsewhere)."""
    y1, c1 = solve(SinusoidalGalerkinSolver, crossing(EPS_ONE))
    yf, cf = solve(SinusoidalGalerkinSolver, free_space(crossing()))
    assert (np.abs(y1 - yf) / np.abs(yf)).max() < 1e-4
    for a, b in zip(c1, cf):
        assert rel(a, b) < 1e-3


def test_the_crossing_deck_builds_no_transmitted_grid(monkeypatch):
    calls = _negate_transmitted(monkeypatch)
    solve(SinusoidalGalerkinSolver, crossing(EPS_ONE))
    assert calls == []


def test_red_control_the_air_readout_drops_the_crossing_wings(monkeypatch):
    _, cf = solve(SinusoidalGalerkinSolver, free_space(crossing()))
    _air_readout(monkeypatch)
    _, c1 = solve(SinusoidalGalerkinSolver, crossing(EPS_ONE))
    assert max(rel(a, b) for a, b in zip(c1, cf)) > 5e-3


# ----------------------------------------------------------------------
# soil: SG converges on bspline, two ports driven
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_ladder():
    out = []
    for m in (1, 2, 4):
        d = mixed(m)
        out.append(
            (
                solve(SinusoidalGalerkinSolver, d),
                solve(BSplineSolver, d),
                solve(SinusoidalGalerkinSolver, d, drive=[0.0, 1.0]),
                solve(BSplineSolver, d, drive=[0.0, 1.0]),
            )
        )
    return out


def test_mixed_y_converges_on_bspline(mixed_ladder):
    for i, j in ((0, 1), (0, 0), (1, 1)):
        gaps = [
            abs(sg[0][i, j] - bs[0][i, j]) / abs(bs[0][i, j])
            for sg, bs, _, _ in mixed_ladder
        ]
        _decays(gaps)
        assert gaps[-1] < 2e-5, (i, j, gaps)


def test_mixed_currents_converge_on_bspline_with_both_ports_driven(mixed_ladder):
    for w in range(2):
        gaps = [rel(sg[1][w], bs[1][w]) for sg, bs, _, _ in mixed_ladder]
        _decays(gaps)
        assert gaps[-1] < 1e-5, (w, gaps)


def test_the_unfed_buried_wire_carries_bspline_s_current(mixed_ladder):
    """Only the ABOVE port driven: the buried wire's current is induced
    entirely through the cross term, so its sign is the defect's sign."""
    gaps = [rel(sg[1][0], bs[1][0]) for _, _, sg, bs in mixed_ladder]
    _decays(gaps)
    assert gaps[-1] < 1e-5, gaps


def test_red_controls_in_soil(monkeypatch):
    """Each pre-fix seam alone, on the unfed buried wire at m = 2: the sign
    inverts it, the air readout reads a different function, the air drive
    moves Y11 by the percent-level gap the D2 gates had been living with."""
    d = mixed(2)
    y_bs, c_bs = solve(BSplineSolver, d, drive=[0.0, 1.0])
    with monkeypatch.context() as mp:
        _negate_transmitted(mp)
        _, c = solve(SinusoidalGalerkinSolver, d, drive=[0.0, 1.0])
        assert rel(-c[0], c_bs[0]) < 1e-3  # inverted, not merely off
    with monkeypatch.context() as mp:
        _air_readout(mp)
        _, c = solve(SinusoidalGalerkinSolver, d, drive=[0.0, 1.0])
        assert rel(c[0], c_bs[0]) > 0.5
    with monkeypatch.context() as mp:
        _air_drive(mp)
        y, _ = solve(SinusoidalGalerkinSolver, d)
        assert abs(y[0, 0] - y_bs[0, 0]) / abs(y_bs[0, 0]) > 1e-2


@pytest.mark.slow
def test_detached_1149_deck_converges_on_bspline():
    """razor#1149's `detached` deck (graded, near-plane ends fixed under m):
    Y12 and both wires' currents, two ports driven. Y12 2.3e-4 -> 3.5e-6,
    buried wire 9.3e-4 -> 7.3e-5. Slow for its Sommerfeld grids (~2 min),
    not its mesh."""
    rows = [
        (
            solve(SinusoidalGalerkinSolver, detached(m)),
            solve(BSplineSolver, detached(m)),
        )
        for m in (1, 2)
    ]
    gaps = [abs(sg[0][0, 1] - bs[0][0, 1]) / abs(bs[0][0, 1]) for sg, bs in rows]
    _decays(gaps)
    for w in range(2):
        _decays([rel(sg[1][w], bs[1][w]) for sg, bs in rows])


@pytest.mark.slow
def test_detached_hub_one_radius_converges_on_bspline():
    """`detached_hub` at ONE radius: Y12 and every wire's current halve per
    doubling (3.7e-4 / 2.0e-4 / 9.3e-5 on Y12). With its three-radius
    `HUB_RADII` the Y12 gap sits FLAT at 1.9e-3 after this fix — a separate
    question about SG at a mixed-radius junction (NOTES.md), not gated here."""
    rows = []
    for m in (1, 2, 4):
        d = detached_hub(m, radii=(0.001,) * 4)
        rows.append((solve(SinusoidalGalerkinSolver, d), solve(BSplineSolver, d)))
    _decays([abs(sg[0][0, 1] - bs[0][0, 1]) / abs(bs[0][0, 1]) for sg, bs in rows], 1.7)
    for w in range(4):
        _decays([rel(sg[1][w], bs[1][w]) for sg, bs in rows], 1.7)


# ----------------------------------------------------------------------
# the far field of a detached deck
# ----------------------------------------------------------------------


def test_far_field_of_the_elevated_wire_over_a_buried_one(monkeypatch):
    """Above port driven: the buried wire is a parasitic, and its induced
    current's sign is the whole defect. After the fix SG's pattern sits at
    bspline's to 1.4e-6; the sign alone moved it by 20 %."""
    d = mixed(2)
    bs = far_pattern(BSplineSolver, d, drive=[0.0, 1.0])
    sg = far_pattern(SinusoidalGalerkinSolver, d, drive=[0.0, 1.0])
    assert rel(sg, bs) < 1e-4
    _negate_transmitted(monkeypatch)
    assert rel(far_pattern(SinusoidalGalerkinSolver, d, drive=[0.0, 1.0]), bs) > 0.1


def test_far_field_converges_within_the_y_agreement(mixed_ladder):
    """Both ports driven: the pattern gap falls with m and never exceeds the
    worst Y-entry gap at the same mesh — a pattern cannot agree worse than the
    currents that radiate it are driven."""
    gaps = []
    for m, (sg, bs, _, _) in zip((1, 2, 4), mixed_ladder):
        g = rel(
            far_pattern(SinusoidalGalerkinSolver, mixed(m)),
            far_pattern(BSplineSolver, mixed(m)),
        )
        y_gap = (np.abs(sg[0] - bs[0]) / np.abs(bs[0])).max()
        assert g <= 2.0 * y_gap, (m, g, y_gap)
        gaps.append(g)
    _decays(gaps)


# ----------------------------------------------------------------------
# wholly buried: the port solution enters the medium
# ----------------------------------------------------------------------


def test_wholly_buried_port_solution_runs_and_inverts_to_compute_impedance():
    d = buried(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SinusoidalGalerkinSolver(**d)
        k_before = s.k
        y = np.asarray(s.compute_y_matrix())
        z = np.asarray(s.compute_impedance()[0])
        y_swept = np.asarray(s.compute_y_matrix_swept([s.k]))[0]
        k_after = s.k
    assert k_after == k_before  # the medium was restored
    np.testing.assert_allclose(1.0 / y.sum(axis=1), z, rtol=1e-12)
    np.testing.assert_allclose(y_swept, y, rtol=1e-13)


def test_wholly_buried_converges_on_bspline():
    rows = [
        (solve(SinusoidalGalerkinSolver, buried(m)), solve(BSplineSolver, buried(m)))
        for m in (1, 2)
    ]
    for i, j in ((0, 1), (0, 0)):
        _decays([abs(sg[0][i, j] - bs[0][i, j]) / abs(bs[0][i, j]) for sg, bs in rows])
    for w in range(2):
        gaps = [rel(sg[1][w], bs[1][w]) for sg, bs in rows]
        _decays(gaps)
        assert gaps[-1] < 1e-4, gaps


def test_red_control_the_air_readout_on_a_wholly_buried_deck(monkeypatch):
    _, c_bs = solve(BSplineSolver, buried(1))
    _air_readout(monkeypatch)
    _, c = solve(SinusoidalGalerkinSolver, buried(1))
    assert min(rel(a, b) for a, b in zip(c, c_bs)) > 0.5


# ----------------------------------------------------------------------
# a deck in air reads exactly as it did
# ----------------------------------------------------------------------


def test_a_deck_in_air_reads_through_the_inherited_view():
    """No `k_entry` off the buried routes, so the readout overrides take the
    inherited bodies and above-ground currents are the shipped arithmetic."""
    d = mixed(1)
    d["wires"] = [w + np.array([0.0, 0.0, 2.0]) for w in d["wires"]]
    for deck in (d, free_space(d)):
        s = SinusoidalGalerkinSolver(**deck)
        geom = s._build_geometry()
        view = s._readout_view(geom)
        base = s._basis_coefs(geom, s.k)
        assert "k_entry" not in view
        for key in ("starts", "jbasis", "sigma", "A", "B", "C", "AC"):
            assert np.array_equal(view[key], base[key])
