"""`HarringtonSolver` — the 1967 pulse row with the dual-cell charge.

The gates here are the ones momwire#557 set, and they come in two kinds.

Structural: the matrix IS Harrington (103) with cell-averaged ψ, checked
against a brute-force fill that shares nothing with the solver, on a deck
whose junction forces the branched cell.

Behavioural: the thing the charge model was changed FOR. The parent row's
error is governed by Δ/a — `test_pulse.py::test_the_scheme_is_governed_by
_delta_over_a_not_delta_over_lambda` measures that — and this row's must be
governed by Δ/λ instead. That test is the inverse of the parent's and it is
the regression that would catch a relapse to point charges, so it is
written to fail loudly rather than to pass quietly.

Ground behaviour rides `tests/test_pulse_ground.py`'s contract unchanged —
the charge model is a source-side substitution and the image fold never
sees it — with the PEC image twin repeated here because it is the one gate
that would catch a cell built on the real geometry when the image asked
for a mirrored one.
"""

import numpy as np
import pytest

from momwire import HarringtonSolver, PulseSolver

# IMPORTED, not copied. The whole claim of this pair is that the two rows
# differ in ONE ingredient, so they have to be measured on one specimen and
# against one reference — and a byte-copy of test_pulse.py's deck leaves
# that premise enforced by eyeball, which is exactly how a paired instrument
# quietly stops being one. The house already imports across test modules
# (test_eznec_networks, test_razor_production_lane); this is the case that
# most needs it.
from test_pulse import (
    DIP_RAD,
    DIP_LEN,
    WAVELENGTH,
    _dipole,
    _z_reference,
)

DIP_H = 1.7


def _horizontal(z):
    return _dipole(z)


def _z(cls, n, radius=DIP_RAD, **kw):
    z, _ = cls(
        wires=[_dipole()],
        nsegs=n,
        wire_radius=radius,
        wavelength=WAVELENGTH,
        **kw,
    ).compute_impedance()
    return complex(np.atleast_1d(z)[0])


# --------------------------------------------------------------------------
# 1. the matrix is Harrington (103)
# --------------------------------------------------------------------------


def _reference_fill(sim):
    """Harrington (103) written out entry by entry, sharing nothing.

    Every kernel evaluation is a fresh `exp(-jkR)/(4πR)`; the vector term's
    segment integral and every charge cell's integral are flat 64-point
    Gauss-Legendre rules with no static/remainder split; and the node map
    is rebuilt by brute-force O(n²) grouping of knot COORDINATES rather
    than by the solver's union-find over structural indices. The deck is
    fat enough (h/a ≈ 2) that the flat rule is converged on the self term,
    so this is an independent statement of the formula.
    """
    geom = sim._build_geometry()
    seg_l, seg_r = geom["seg_l"], geom["seg_r"]
    tang, h = geom["tangents"], geom["h_per_seg"]
    n, a2 = h.size, sim.wire_radius**2
    k, omega = sim.k, sim.omega
    cent = 0.5 * (seg_l + seg_r)
    xg, wg = np.polynomial.legendre.leggauss(64)

    def g(p, q):
        R = np.sqrt(np.sum((p - q) ** 2) + a2)
        return np.exp(-1j * k * R) / (4.0 * np.pi * R)

    # Brute-force nodes: every segment end, grouped by coordinate.
    node_pts, l_node, r_node = [], [], []
    for arr, out in ((seg_l, l_node), (seg_r, r_node)):
        for j in range(n):
            for idx, q in enumerate(node_pts):
                if float(np.linalg.norm(arr[j] - q)) <= 1e-9:
                    out.append(idx)
                    break
            else:
                node_pts.append(arr[j])
                out.append(len(node_pts) - 1)

    # Each node's cell: the half-segment of every incident segment end.
    cells = [[] for _ in node_pts]
    for j in range(n):
        cells[l_node[j]].append((seg_l[j], tang[j], h[j] / 2))
        cells[r_node[j]].append((cent[j], tang[j], h[j] / 2))

    def psi(p, node):
        """(1/L) ∫ g over every piece of `node`'s cell."""
        total, length = 0.0 + 0j, 0.0
        for start, t, hh in cells[node]:
            tau = 0.5 * hh * (1.0 + xg)
            total += float(0.5 * hh) * sum(
                wg[i] * g(p, start + tau[i] * t) for i in range(64)
            )
            length += hh
        return total / length

    Z = np.empty((n, n), dtype=np.complex128)
    for m in range(n):
        for j in range(n):
            tau = 0.5 * h[j] * (1.0 + xg)
            M0 = float(0.5 * h[j]) * sum(
                wg[i] * g(cent[m], seg_l[j] + tau[i] * tang[j]) for i in range(64)
            )
            # (103): the four cell-to-endpoint terms.
            ddg = (
                psi(seg_r[m], r_node[j])
                - psi(seg_r[m], l_node[j])
                - psi(seg_l[m], r_node[j])
                + psi(seg_l[m], l_node[j])
            )
            Z[m, j] = 1j * omega * sim.mu * h[m] * float(
                tang[m] @ tang[j]
            ) * M0 + ddg / (1j * omega * sim.eps)
    return Z


def test_the_matrix_is_harringtons_103_with_cell_averaged_psi():
    """`_assemble_Z` == the brute-force (103) fill, on a junction deck.

    Three wires meeting at one node, so the star cell is exercised: the
    node's charge spreads over THREE half-segments, its length is their
    sum, and every basis touching it pours into the same average. A cell
    built per-branch instead of per-node would pass every free-space
    convergence gate in this file and fail here.
    """
    p = np.array([0.0, 0.0, 0.0])
    sim = HarringtonSolver(
        wires=[
            np.array([p, p + [0.0, 1.3, 0.0]]),
            np.array([p, p + [0.9, 0.0, 0.0]]),
            np.array([p, p + [0.0, 0.0, -1.1]]),
        ],
        n_per_edge_per_wire=[[5], [4], [4]],
        wire_radius=0.05,
        wavelength=WAVELENGTH,
        feeds=[(0, 0.65, 1.0)],
    )
    got = sim._assemble_Z(sim._build_geometry(), sim.k)
    want = _reference_fill(sim)
    assert np.abs(got - want).max() < 1e-9 * np.abs(want).max()


def test_a_junction_node_is_one_cell_over_every_incident_half_segment():
    """The node map, read directly: K branches in, one cell, length Σh/2."""
    p = np.array([0.0, 0.0, 0.0])
    sim = HarringtonSolver(
        wires=[
            np.array([p, p + [0.0, 1.0, 0.0]]),
            np.array([p, p + [1.0, 0.0, 0.0]]),
            np.array([p, p + [0.0, 0.0, 1.0]]),
        ],
        n_per_edge_per_wire=[[4], [4], [4]],
        wire_radius=0.02,
        wavelength=WAVELENGTH,
    )
    geom = sim._build_geometry()
    left, _right, pieces, cell_of_piece = sim._node_map(geom)
    # every wire's arc-0 end is the shared node
    shared = {int(left[0]), int(left[4]), int(left[8])}
    assert len(shared) == 1, "three coincident wire ends must be ONE node"
    node = shared.pop()
    length = pieces["h_per_seg"][cell_of_piece == node].sum()
    assert (cell_of_piece == node).sum() == 3, "one half-segment per branch"
    assert length == pytest.approx(3 * (0.25 / 2), rel=1e-12)


def test_a_free_end_cell_is_half_length_so_the_charge_stays_on_the_wire():
    """The terminal cell is clipped to the conductor, not overhanging.

    Worth a steady ~11% on the ladder (momwire#557), and it is Harrington's
    own Fig. 6 footnote — "the extra 1/2 interval at each wire end".
    """
    sim = HarringtonSolver(
        wires=[_dipole()], nsegs=10, wire_radius=DIP_RAD, wavelength=WAVELENGTH
    )
    geom = sim._build_geometry()
    left, right, pieces, cell_of_piece = sim._node_map(geom)
    h = geom["h_per_seg"][0]
    for tip in (int(left[0]), int(right[-1])):
        assert (cell_of_piece == tip).sum() == 1
        assert pieces["h_per_seg"][cell_of_piece == tip].sum() == pytest.approx(h / 2)
    interior = int(right[0])
    assert (cell_of_piece == interior).sum() == 2
    assert pieces["h_per_seg"][cell_of_piece == interior].sum() == pytest.approx(h)


# --------------------------------------------------------------------------
# 2. what the charge model was changed for
# --------------------------------------------------------------------------

LADDER = (16, 32, 64, 128, 256)


def test_free_space_ladder_converges_at_the_classical_first_order_rate():
    """N = 16…256 against the converged `BSplineSolver`, measured 2026-08-22:

    |   N | Δ/a  | Z (Ω)             | |Z − Z_ref| | ratio |
    |-----|------|-------------------|-------------|-------|
    |  16 | 31.8 |  84.83 +  43.84j  |   44.47     |       |
    |  32 | 15.9 |  77.55 +  19.57j  |   19.13     | 2.32  |
    |  64 |  7.9 |  74.48 +   8.72j  |    7.86     | 2.43  |
    | 128 |  4.0 |  73.20 +   3.96j  |    2.93     | 2.68  |
    | 256 |  2.0 |  72.75 +   1.96j  |    0.89     | 3.29  |
    | ref |      |  72.45 +   1.13j  |             |       |

    Every rung better than half the one before — the classical O(1/N)
    pulse-basis rate `docs/pulse_basis_d0_nodal_charge.md` predicted the
    staggered cell would restore. The ratio IMPROVES down the ladder
    (2.32 → 3.29) because ψ here is the exact segment integral rather than
    Harrington's own two-term Appendix approximation, so the scheme's own
    O(h) term retires against a reference that no longer moves.

    Pinned at 2.0× per rung and 3 Ω at the last — room for the
    allocator/BLAS variance the house sees on CI, none for a regression.
    """
    ref = _z_reference()
    errs = [abs(_z(HarringtonSolver, n) - ref) for n in LADDER]
    for lo, hi, n_lo, n_hi in zip(errs[1:], errs[:-1], LADDER[1:], LADDER[:-1]):
        assert lo < hi / 2.0, f"N={n_hi}→{n_lo}: {hi:.4g} → {lo:.4g} is not halving"
    assert errs[-1] < 3.0, f"N={LADDER[-1]}: {errs[-1]:.4g} Ω from the reference"


def test_the_governor_is_delta_over_lambda_not_delta_over_a():
    """The inverse of the parent row's test, and the point of this class.

    `PulseSolver`'s error is set by Δ/a — segment length in wire radii —
    because its charge is two point charges observed at their own location
    with only the reduced kernel's a² regularizing them. Give the charge a
    cell and that dependence goes away: at MATCHED N, a wire and a wire ten
    times thinner must land within a small factor of each other, even
    though their Δ/a differ by 10×.

    Measured 2026-08-22 at N = 64, against each radius's own converged
    reference:

    | radius  | L/a  | Δ/a  | this row | `PulseSolver` |
    |---------|------|------|----------|---------------|
    | 20 mm   |  509 |  7.9 |   7.86 Ω |    157.27 Ω   |
    | 2 mm    | 5086 | 79.5 |  16.61 Ω |   2850.68 Ω   |
    | ratio   |      |      |  2.11×   |     18.1×     |

    A 10× thinner wire costs this row 2.11× and the parent row 18.1×. That
    residual 2.11 is the log(Δ/a) still living in the cell's own self
    term — a logarithm, not a power — and nothing more.

    The gate is written against the PARENT's ratio on the same pair rather
    than against an absolute number, so it calibrates itself: whatever the
    decks drift to, this row must be several times less radius-sensitive
    than point charges are. A relapse would put the two ratios together and
    fail here first.
    """
    n = 64
    fat_ref, thin_ref = _z_reference(radius=0.02), _z_reference(radius=0.002)
    fat = abs(_z(HarringtonSolver, n, radius=0.02) - fat_ref)
    thin = abs(_z(HarringtonSolver, n, radius=0.002) - thin_ref)
    nodal_fat = abs(_z(PulseSolver, n, radius=0.02) - fat_ref)
    nodal_thin = abs(_z(PulseSolver, n, radius=0.002) - thin_ref)

    assert thin < 4.0 * fat, (
        f"error tracks the radius, not the mesh: {fat:.4g} Ω at Δ/a=7.9 vs "
        f"{thin:.4g} Ω at Δ/a=79.5 — the charge has collapsed to a point"
    )
    assert (thin / fat) < (nodal_thin / nodal_fat) / 3.0, (
        f"radius sensitivity {thin / fat:.3g}× is not clearly better than the "
        f"nodal row's {nodal_thin / nodal_fat:.3g}×"
    )


def test_the_dual_cell_is_worth_orders_of_magnitude_over_the_nodal_row():
    """The pair as an instrument: one ingredient differs, and it is worth this.

    Same basis, same testing, same kernel, same feed, same mesh — only the
    charge's SUPPORT differs, so the gap is attributable to it alone. On
    the thin deck at N = 64 (Δ/a ≈ 80): 16.61 Ω against 2850.68 Ω, a factor
    of 172. Pinned at 20× to state the direction without pinning the size.
    """
    n, radius = 64, 0.002
    ref = _z_reference(radius=radius)
    dual = abs(_z(HarringtonSolver, n, radius=radius) - ref)
    nodal = abs(_z(PulseSolver, n, radius=radius) - ref)
    assert dual < nodal / 20.0, f"dual {dual:.4g} Ω vs nodal {nodal:.4g} Ω"


def test_a_colinear_split_is_the_same_antenna_to_roundoff():
    """One wire of 2N == two wires of N meeting at the midpoint.

    A STRONGER statement here than on the parent row, and the reason the
    cell is centred on the node. The split introduces a junction where the
    charge cell now spans both wires' half-segments; had each wire
    truncated its own cell there instead, the two spellings would differ by
    ~10% and the difference would GROW with refinement (momwire#557's
    rejected variant), permanently making a split a different antenna.
    Measured 7.6e-14 relative — LU roundoff, not agreement.
    """
    p0, p1 = _dipole()[0], _dipole()[1]
    mid = 0.5 * (p0 + p1)
    n = 48
    kw = dict(wire_radius=DIP_RAD, wavelength=WAVELENGTH, feeds=[(0, DIP_LEN / 4, 1j)])
    z1, c1 = HarringtonSolver(
        wires=[np.array([p0, p1])], n_per_edge_per_wire=[[2 * n]], **kw
    ).compute_impedance()
    z2, c2 = HarringtonSolver(
        wires=[np.array([p0, mid]), np.array([mid, p1])],
        n_per_edge_per_wire=[[n], [n]],
        **kw,
    ).compute_impedance()
    assert abs(z1 - z2) / abs(z1) < 1e-11
    assert np.abs(c1 - c2).max() < 1e-11 * np.abs(c1).max()


# --------------------------------------------------------------------------
# 3. the ground rides the parent's contract
# --------------------------------------------------------------------------


def test_pec_equals_explicit_image_twin():
    """A dipole at h over PEC == that dipole + its mirror, driven −1 V.

    The parent's gate, repeated because this row is the one that could
    fail it: the charge cells have to be rebuilt on the MIRRORED source
    geometry while the node grouping stays the real geometry's. A cell
    built from the real wire and used for the image would leave the twin
    alone and move the grounded answer.
    """
    n = 24
    kw = dict(nsegs=n, wire_radius=DIP_RAD, wavelength=WAVELENGTH)
    z_ground, _ = HarringtonSolver(
        wires=[_horizontal(DIP_H)], **kw, ground_z=0.0
    ).compute_impedance()
    z_twin, _ = HarringtonSolver(
        wires=[_horizontal(DIP_H), _horizontal(-DIP_H)],
        **kw,
        feeds=[(0, None, 1.0 + 0j), (1, None, -1.0 + 0j)],
    ).compute_impedance()
    z_free = _z(HarringtonSolver, n)

    assert abs(z_ground - z_twin[0]) / abs(z_ground) < 1e-11
    # ...and the plane is doing something, so the agreement above is not two
    # paths both quietly ignoring it.
    assert abs(z_ground - z_free) > 8.0


def test_refl_coef_and_sommerfeld_both_run_and_move_the_answer():
    """Both finite grounds serve on this row, and neither is free space.

    The ground physics is entirely the parent's — `_charge_stencil` is a
    source-side substitution and the image fold never sees which charge
    model produced it — so this is a wiring check, not a physics one.
    """
    kw = dict(nsegs=20, wire_radius=DIP_RAD, wavelength=WAVELENGTH)
    free = _z(HarringtonSolver, 20)
    out = []
    for model in ("refl-coef", "sommerfeld"):
        z, _ = HarringtonSolver(
            wires=[_horizontal(DIP_H)],
            **kw,
            ground_z=0.0,
            ground_eps=13.0 - 0.5j,
            ground_model=model,
        ).compute_impedance()
        assert np.isfinite(complex(z).real) and np.isfinite(complex(z).imag)
        assert abs(complex(z) - free) > 1.0
        out.append(complex(z))
    assert abs(out[0] - out[1]) > 0.1, "sommerfeld must differ from refl-coef"


def test_junctions_is_accepted_and_not_forwarded_to_the_parent():
    """momwire#590 step 3b: this row now takes a junction spec.

    It used to refuse one, on the grounds that coincident ends are found from
    the geometry so there is nothing to declare. That is true of AGREEING with
    the geometry and false of disagreeing with it -- a caller wanting two
    coincident ends left apart had no way to say so.

    The parent (PulseSolver) still refuses `junctions=` for its own and
    still-valid reason: its basis has no junction unknown to constrain. So the
    spec must be intercepted here, never forwarded.
    """
    s = HarringtonSolver(
        wires=[_dipole()],
        nsegs=8,
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        junctions=[],
    )
    assert s._declared_junctions == []

    with pytest.raises(NotImplementedError, match="no junction"):
        PulseSolver(
            wires=[_dipole()],
            nsegs=8,
            wire_radius=DIP_RAD,
            wavelength=WAVELENGTH,
            junctions=[],
        )


# --------------------------------------------------------------------------
# 4. the node map's two rules, both of which used to be claims
# --------------------------------------------------------------------------


def test_nearly_coincident_ends_are_refused_not_silently_disconnected():
    """Ends inside the deck layer's "same point" grid but outside this
    formulation's joining tolerance REFUSE.

    They used to be answered — as two separate charge cells, with the
    junction current having nowhere to cross. Measured before the refusal:
    a dipole split with a 5e-8 m gap read 137.2 + 4.7j against 131.7 − 4.1j
    unsplit at N = 96, a 7.9% error that GROWS with refinement.

    Since momwire#590 step 3b a caller CAN say otherwise -- the refusal now
    names `junctions=` as the second route, and the test below takes it.

    Nothing is served silently wrong: 5e-8 m is a thousand times finer than
    the grid `deck/_polylines` fuses endpoints onto, so a transformed or
    hand-assembled model lands in this window without doing anything odd.
    """
    p0 = np.array([0.0, -DIP_LEN / 2, 0.0])
    mid = np.array([0.0, 0.0, 0.0])
    gap = np.array([0.0, 5e-8, 0.0])
    p1 = np.array([0.0, DIP_LEN / 2, 0.0])
    with pytest.raises(ValueError, match="Make the two ends exactly equal"):
        HarringtonSolver(
            wires=[np.array([p0, mid]), np.array([mid + gap, p1])],
            n_per_edge_per_wire=[[24], [24]],
            wire_radius=DIP_RAD,
            wavelength=WAVELENGTH,
        ).compute_impedance()


def test_the_near_coincident_refusal_now_names_a_route_that_works():
    """The refusal names two remedies, so the two have to agree.

    Before momwire#590 step 3b it named one -- "make the two ends exactly
    equal" -- because `junctions=` was refused here. It now offers the spec as
    well, and a refusal that named a route which did not work, or worked
    differently, would be worse than the one-route version it replaced.

    Same geometry, same feed, both remedies applied: closing the 5e-8 m gap,
    versus leaving it and declaring the two ends one node.
    """
    p0 = np.array([0.0, -DIP_LEN / 2, 0.0])
    mid = np.array([0.0, 0.0, 0.0])
    gap = np.array([0.0, 5e-8, 0.0])
    p1 = np.array([0.0, DIP_LEN / 2, 0.0])
    kw = dict(
        n_per_edge_per_wire=[[24], [24]],
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        feed_wire_index=0,
        feed_arclength=DIP_LEN / 4,
    )
    # Remedy 1: exactly equal ends, joined by the tolerance.
    closed, _ = HarringtonSolver(
        wires=[np.array([p0, mid]), np.array([mid, p1])], **kw
    ).compute_impedance()
    # Remedy 2: the gap stays, the caller declares the node.
    declared, _ = HarringtonSolver(
        wires=[np.array([p0, mid]), np.array([mid + gap, p1])],
        junctions=[[(0, "end"), (1, "start")]],
        **kw,
    ).compute_impedance()
    rel = abs(complex(declared) - complex(closed)) / abs(complex(closed))
    assert rel < 1e-4, f"{declared} vs {closed} — {rel:.2e}"


def test_a_chained_tolerance_is_refused_so_the_grouping_rule_cannot_diverge():
    """The two node-map rules cannot disagree with razor, and here is why.

    `_node_map` groups by first match against a group REPRESENTATIVE, which
    is `razor._find_junctions`' rule and is deliberately non-transitive: with
    A~B and B~C but A≁C, razor gives B's group and a separate C where a
    transitive union-find merges all three. Holding both rules in one tree
    would mean two solvers disagreeing about the same deck's CONNECTIVITY,
    so this class walks razor's algorithm rather than something that
    resembles it.

    But the divergence is not merely avoided, it is UNREACHABLE — and that
    is worth a test rather than a comment. Chaining needs |AB| and |BC| both
    within `_JUNCTION_TOL`, so |AC| <= 2·_JUNCTION_TOL by the triangle
    inequality, which lands strictly inside the near-coincident window
    (_JUNCTION_TOL, _NEAR_COINCIDENT_TOL]. Every configuration in which the
    two rules could differ is therefore refused before either rule runs.

    So the refusal above is not only about junction current having nowhere
    to cross; it also closes the one gap where this solver's connectivity
    could have differed from its sibling's.
    """
    step = 6e-10  # A~B and B~C at 6e-10; A~C at 1.2e-9 — a chained tolerance
    a = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Make the two ends exactly equal"):
        HarringtonSolver(
            wires=[
                np.array([a, a + [0.0, 1.0, 0.0]]),
                np.array([a + [step, 0.0, 0.0], a + [step, 1.0, 0.0]]),
                np.array([a + [2 * step, 0.0, 0.0], a + [2 * step, 1.0, 0.0]]),
            ],
            n_per_edge_per_wire=[[4], [4], [4]],
            wire_radius=1e-3,
            wavelength=WAVELENGTH,
        ).compute_impedance()
