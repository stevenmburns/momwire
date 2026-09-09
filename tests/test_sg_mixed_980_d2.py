"""momwire#980 D2: the mixed-deck serve on SinusoidalGalerkinSolver.

Wires both above and below the interface, none ending in the plane, no
junction spanning media (that is D3). Three pair classes: above x above at
k_p, below x below at k_m (D1's serve), and the transmitted pair in both
directions.

The fill is SUBSET-COMPUTE, FULL-WIDTH PRESENT — bspline's shape. Each
class's block lands in a full-shaped array and the entries belonging to
another pair class stay ZERO, so correctness rests on "the untouched entries
are zero" rather than on a renumbering being right.

The heavy gates carry `slow`: their cost is the SOMMERFELD GRID, not the
mesh — trimming a 21-segment deck to 9 moved the far-apart gate from 32.14 s
to 32.00 s — so they belong in the push lane rather than being shrunk into
the PR one, where they would only get faster by testing less.

Measured 2026-09-09, soil A, 7 MHz:

    far-apart diagonal blocks vs the single-class solves   4.5e-22 / 9.7e-17
    eps_tilde -> 1, mixed deck vs free space               see the gate below
    transmitted reciprocity                                see the gate below
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import SinusoidalGalerkinSolver

SOIL_A = (13.0, 0.005)
C0 = 299792458.0
WL7 = C0 / 7e6
GROUND = dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="sommerfeld")


def mk(wires, npe, feeds=None, eps=SOIL_A, free=False):
    ground = {} if free else dict(GROUND, ground_eps=eps)
    return SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=feeds or [(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **ground,
    )


def z_of(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(s.compute_impedance()[0])


def G_of(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = s._build_geometry()
        if s._is_mixed(geom):
            return s._assemble_Z(geom, s.k)[0]
        with s._operating_medium(geom):
            return s._assemble_Z(geom, s.k)[0]


ABOVE = np.array([(0.0, 0.0, 0.25), (0.0, 0.0, 1.25)])
BELOW_FAR = np.array([(40.0, 0.0, -1.15), (40.0, 0.0, -0.15)])
BELOW_NEAR = np.array([(0.0, 0.0, -1.15), (0.0, 0.0, -0.15)])


# ----------------------------------------------------------------------
# The view is a view
# ----------------------------------------------------------------------


# `slow` because this is GRID-bound, not mesh-bound: trimming the deck from
# 21 segments to 9 moved it 32.14 s -> 32.00 s. Do not "optimise" it into the
# PR lane by shrinking the deck — that only makes it test less. Same for the
# three gates below.
@pytest.mark.slow
def test_far_apart_diagonal_blocks_reproduce_the_single_class_solves():
    """The (C) shape's whole correctness argument, as a measurement.

    A class's diagonal quadrant must be what that class alone would have
    assembled: the other medium's wires contribute to the OFF-diagonal
    quadrants and nowhere else. A leak between classes is the only way past
    this, which is what makes it worth running.

    **Not bit equality, and that is structural rather than sloppy.** Under
    subset-compute/full-width-present the fill's reduction runs over a
    different number of source columns than the single-class solve, so the
    summation order differs. Tightening this gate to `array_equal` would be
    unsatisfiable without subset-computing BOTH axes — which is the
    seam-heavy route this shape exists to avoid — so it is deliberately a
    float-noise gate. Measured 4.5e-22 (above) and 9.7e-17 (below).
    """
    Gm = G_of(mk([ABOVE, BELOW_FAR], [[9], [9]]))
    Ga = G_of(mk([ABOVE], [[9]]))
    Gb = G_of(mk([BELOW_FAR], [[9]]))
    na, nb = Ga.shape[0], Gb.shape[0]
    assert Gm.shape[0] == na + nb

    qa = Gm[:na, :na]
    qb = Gm[na : na + nb, na : na + nb]
    rel_a = np.abs(qa - Ga).max() / np.abs(Ga).max()
    rel_b = np.abs(qb - Gb).max() / np.abs(Gb).max()
    assert rel_a < 1e-12, f"above quadrant {rel_a:.3e}"
    assert rel_b < 1e-12, f"below quadrant {rel_b:.3e}"

    # And the off-diagonal must be POPULATED — a mixed assembly whose cross
    # blocks were all zero would pass the two gates above trivially.
    assert np.abs(Gm[:na, na:]).max() > 0.0
    assert np.abs(Gm[na:, :na]).max() > 0.0


def test_the_buried_block_is_filled_at_the_medium_eta():
    """The finding that the gate above caught, pinned directly.

    `self.eta` is read OFF THE SOLVER by the fill, so a mixed deck must set
    it per class. Filling the buried block at k_m with air's eta is wrong by
    |eta_0/eta_m| = 4.27x at soil A / 7 MHz and the deck still solves,
    returning a plausible number — so this asserts the context manager
    actually moves eta, and restores it.
    """
    s = mk([ABOVE, BELOW_FAR], [[9], [9]])
    geom = s._build_geometry()
    medium = s._fill_medium(geom)
    eta_air = s.eta
    with s._at_class_eta(medium, True):
        eta_m = s.eta
    assert s.eta == eta_air
    assert np.iscomplexobj(eta_m) or eta_m != eta_air
    ratio = abs(eta_air / eta_m)
    assert 4.0 < ratio < 4.6, f"|eta_0/eta_m| = {ratio:.3f}, expected ~4.27"
    with s._at_class_eta(medium, False):
        assert s.eta == eta_air  # the above class keeps air


# ----------------------------------------------------------------------
# eps_tilde -> 1
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_eps_tilde_one_collapses_the_mixed_deck_onto_free_space():
    """Every class collapses at once: k_m -> k_p, A_m -> 0, both remainders
    and the transmitted block -> the free-space coupling. So a miss localises
    to one term rather than to "the mixed fill"."""
    z_free = z_of(mk([ABOVE, BELOW_NEAR], [[9], [9]], free=True))
    z_coll = z_of(mk([ABOVE, BELOW_NEAR], [[9], [9]], eps=(1.0, 0.0)))
    rel = abs(z_coll - z_free) / abs(z_free)
    assert rel < 1e-9, f"{rel:.3e}"


@pytest.mark.slow
def test_close_cross_pairs_collapse_at_the_grid_floor():
    """The licence for taking NO near correction on the cross block.

    An above segment 5 cm over the plane and a buried one 5 cm under it is
    the close cross pair the near correction would have existed for. The
    transmitted block has no closed form to correct against, so what has to
    hold is that at eps_tilde -> 1 it reproduces the free-space coupling
    between those same bases to the GRID's own interpolation floor —
    `_below_interface.n_qp_buried_field`'s table puts that at ~6.8e-4 on a
    3+2 segment deck and ~1.0e-5 at 15+10 (bspline's measurement, q=6).

    If this misses, the near zone is under-resolved and the lever is q, not
    a correction.
    """
    above = np.array([(0.0, 0.0, 0.05), (0.6, 0.0, 0.05)])
    below = np.array([(0.0, 0.0, -0.05), (0.6, 0.0, -0.05)])
    z_free = z_of(mk([above, below], [[9], [9]], free=True))
    z_coll = z_of(mk([above, below], [[9], [9]], eps=(1.0, 0.0)))
    rel = abs(z_coll - z_free) / abs(z_free)
    # The grid floor, not machine epsilon — this is the number the decision
    # rests on, so it is stated rather than padded out of sight.
    assert rel < 5e-3, f"close cross pairs collapse to {rel:.3e}"


# ----------------------------------------------------------------------
# Reciprocity of the two transmitted directions
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_the_transmitted_directions_are_reciprocal():
    """below->above against above->below on the same deck.

    The two are filled by different evaluators over different observer sets,
    so their agreement is a genuine cross-check of the pair rather than an
    identity. Reciprocity is asserted on the ASSEMBLED matrix, which is where
    it has to hold: G must be symmetric across the two cross quadrants.
    """
    s = mk([ABOVE, BELOW_NEAR], [[9], [9]])
    G = G_of(s)
    ga = G_of(mk([ABOVE], [[9]]))
    na = ga.shape[0]
    upper = G[:na, na:]
    lower = G[na:, :na]
    scale = max(np.abs(upper).max(), np.abs(lower).max())
    assert scale > 0.0, "the cross quadrants are empty; nothing was tested"
    rel = np.abs(upper - lower.T).max() / scale
    assert rel < 1e-6, f"transmitted reciprocity {rel:.3e}"


# ----------------------------------------------------------------------
# Scope: what a mixed deck still refuses
# ----------------------------------------------------------------------


def test_wire_loading_on_a_mixed_deck_refuses_by_name():
    """`_apply_loading` is applied at ONE k and the classes load at two."""
    s = mk([ABOVE, BELOW_NEAR], [[11], [11]])
    s._loading_active = True
    with pytest.raises(NotImplementedError, match="single wavenumber"):
        s.compute_impedance()


def test_the_extended_kernel_on_a_mixed_deck_refuses_by_bsplines_name():
    """The mixed route does not enter `_operating_medium`, so it raises the
    scope refusals itself — otherwise an EK mixed deck slips through."""
    from momwire._below_interface import BURIED_EXTENDED_KERNEL_REFUSAL

    s = mk([ABOVE, BELOW_NEAR], [[11], [11]])
    s.extended_kernel = True
    with pytest.raises(NotImplementedError) as exc:
        s.compute_impedance()
    assert str(exc.value) == BURIED_EXTENDED_KERNEL_REFUSAL


def test_a_deck_past_the_transmitted_domain_refuses_with_its_numbers():
    """`serve_plan`'s cost law, reached before any grid is filled.

    2 km is 46.7 free-space wavelengths against the 2 the transmitted family
    is tabulated to, and the refusal names both.
    """
    far = np.array([(2000.0, 0.0, -1.15), (2000.0, 0.0, -0.15)])
    with pytest.raises(ValueError, match="transmitted family is tabulated"):
        z_of(mk([ABOVE, far], [[11], [11]]))


def _connected_screen(tip=(0.0, 0.0, 10.0), n_mast=15):
    """Four buried radials running straight to a node IN the plane, one mast
    above it: the connected screen in the DIRECT spelling (far end to node),
    which shares no geometry between members. COARSE on purpose
    (momwire#1000): the catalog's node-graded mast trips the transmitted
    grid's cost law and refuses by accident; this mesh does not, and before
    the by-name refusal it SOLVED, to 13258−15205j where bspline reads
    57.0−20.3j — the mast open at its base, the screen absent from the
    answer. (The N-rises spelling of the same screen blows up to 1e30 on
    this family for a different reason, coincident rise segments, which is
    its own issue and not what this gate is about.)"""
    wires = [
        np.array([(5.0 * np.cos(a), 5.0 * np.sin(a), -0.15), (0.0, 0.0, 0.0)])
        for a in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    ]
    wires.append(np.array([(0.0, 0.0, 0.0), tip]))
    return SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=[[12]] * 4 + [[n_mast]],
        junctions=[[(i, "end") for i in range(4)] + [(4, "start")]],
        feeds=[(4, 0.25, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **GROUND,
    )


def test_a_crossing_junction_refuses_by_name():
    """momwire#1000: the one buried class D2 leaves to D3 must REFUSE, with
    the sentence its capability row declares — a consumer reads the row
    before the solve (antennaknobs#1286), so the two must be one string."""
    from momwire.sinusoidal_galerkin import _CROSSING_JUNCTION_REFUSAL

    s = _connected_screen()
    with pytest.raises(NotImplementedError) as exc:
        s.compute_impedance()
    assert str(exc.value) == _CROSSING_JUNCTION_REFUSAL
    assert (
        SinusoidalGalerkinSolver.capabilities.refusal("buried", "crossing_junction")
        == _CROSSING_JUNCTION_REFUSAL
    )


def test_the_crossing_refusal_precedes_the_cost_law():
    """A sloped mast puts a quadrature node within 0.01 deg of a radial, which
    is the deck the transmitted grid's cost law refuses. The junction is the
    reason and must be the sentence; the panelling is not."""
    from momwire.sinusoidal_galerkin import _CROSSING_JUNCTION_REFUSAL

    s = _connected_screen(tip=(10.0, 0.0, 0.5))
    with pytest.raises(NotImplementedError) as exc:
        s.compute_impedance()
    assert str(exc.value) == _CROSSING_JUNCTION_REFUSAL


def test_a_junction_wholly_below_the_plane_is_not_a_crossing():
    """The buried hub — radials joined at depth, nothing reaching the plane —
    is D1's class and stays served; the refusal keys on media, not on
    junctions existing."""
    wires = [
        np.array([(5.0 * np.cos(a), 5.0 * np.sin(a), -0.15), (0.0, 0.0, -0.15)])
        for a in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    ]
    s = SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=[[7]] * 4,
        junctions=[[(i, "end") for i in range(4)]],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **GROUND,
    )
    assert np.isfinite(z_of(s))


# ----------------------------------------------------------------------
# The radial effect: SG against BOTH bspline and NEC-5
# ----------------------------------------------------------------------

# The elevated-detached deck of the buried-radials case study: 21 m vertical
# dipole, lower tip 0.25 m up, detached 5 m radials 0.15 m down, 1 mm radius,
# soil 13/0.005, 7 MHz. NEC-5's printed column is the 2026-09-09 re-capture
# under the DOCUMENTED ground card `GE -1,0`; the earlier 08-28 capture used
# the transposed `GE 1,-1` (momwire#929 / antennaknobs#1025) and its radial
# effect disagreed with momwire's by 3-5x. Under the documented card the two
# engines agree to ~0.01 ohm, so this is a genuine three-way gate and SG must
# land on BOTH — landing between them would be a finding, not a pass.
_NEC5_RADIAL_EFFECT = {1: -0.080 + 0.238j, 4: -0.448 + 0.727j}
_NV, _NR = 66, 30  # the x3 rung


def _radial_deck(n_radials):
    wires = [np.array([(0.0, 0.0, 0.25), (0.0, 0.0, 21.25)])]
    npe = [[_NV]]
    for i in range(n_radials):
        a = 2 * np.pi * i / max(n_radials, 1)
        wires.append(
            np.array([(0.0, 0.0, -0.15), (5.0 * np.cos(a), 5.0 * np.sin(a), -0.15)])
        )
        npe.append([_NR])
    fed = (_NV + 1) // 2
    return wires, npe, [(0, (fed - 0.5) / _NV * 21.0, 1 + 0j)]


@pytest.mark.slow
@pytest.mark.parametrize("n_radials", [1, 4])
def test_the_radial_effect_matches_bspline_and_nec5(n_radials):
    """The quantity gated is deck MINUS the no-radial reference.

    Absolute Z carries the feed-convention offset the case study documents,
    so it is the EFFECT of the radials — the thing the buried serve exists to
    predict — that has to agree, and it must agree with both engines at once.
    """
    from momwire import BSplineSolver

    def z(cls, n, **kw):
        wires, npe, feeds = _radial_deck(n)
        s = cls(
            wires=wires,
            n_per_edge_per_wire=npe,
            feeds=feeds,
            wavelength=WL7,
            wire_radius=0.001,
            **GROUND,
            **kw,
        )
        return z_of(s)

    d_sg = z(SinusoidalGalerkinSolver, n_radials) - z(SinusoidalGalerkinSolver, 0)
    d_bs = z(BSplineSolver, n_radials, degree=2) - z(BSplineSolver, 0, degree=2)
    d_n5 = _NEC5_RADIAL_EFFECT[n_radials]

    # SG against bspline: two bases, so the cross-basis floor, not machine eps.
    assert abs(d_sg - d_bs) < 5e-3, f"SG {d_sg} vs bspline {d_bs}"
    # SG against NEC-5: a different ENGINE, printed to 3 decimals.
    assert abs(d_sg - d_n5) < 2e-2, f"SG {d_sg} vs NEC-5 {d_n5}"
    # And the effect must be real, not a rounding difference either side of
    # zero — a serve that ignored the radials entirely would pass a gate
    # stated only as a tolerance.
    assert abs(d_sg) > 0.1
