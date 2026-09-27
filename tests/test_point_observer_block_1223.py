"""The cross block at point observers — momwire#1223 U4.

The point-matched crossing node tests the transmitted field at each
segment's midpoint, so its cross rows are t̂·E at a point:
`_crossing_fill.point_observer_block`, forward (observers above) and
reversed (observers below). Nothing is integrated by parts on the observer
side; the observer gradient comes from the U3a point keys.

Two gates, each checking a different thing:

  * ε̃ = 1, where the interface is transparent: the block must be the
    solver's OWN exact free-space field of the same bases at the same
    points (`_field_tensor` @ M over the source axis's segments). Ratio 1,
    agreement at the axis quadrature (measured ≤ 4.2e-6). This fixes every
    sign and the scale in absolute terms. At ε̃ = 1 W ≡ 0 and the
    transmitted kernels are the free-space ones, so it cannot see the
    interface terms.
  * soil, where it can: integrating the point rows against an interior
    Galerkin test basis (support off the axis ends, so no test-end or corner
    term) must reproduce `cross_complete_block`'s row (forward) and
    `cross_complete_block_reversed`'s (reversed). That is an identity, with
    the same nodes on both sides, so at a resolved test-axis rule it is exact:
    ~3e-15 at q = 12, even at εr 80 / σ = 5 S/m. The reversed block is
    derived by reciprocity alone and meets the independently built
    Galerkin reversed block there.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sin_crossing_recipe_1223 import DECKS, WL7, _eps_tilde, _solver

from momwire import _crossing_fill as cf
from momwire.sinusoidal import SinusoidalSolver

SOILS = [(4.0, 1e-3), (13.0, 0.005), (80.0, 5.0)]


def _crossing(deck, eps, n):
    s = _solver(deck, eps, n=n)
    geom = s._build_geometry()
    below = np.asarray(geom["seg_centers"])[:, 2] < s.ground_z
    med = cf.buried_medium(s.ground_eps, s.omega, s.eps, s.k)
    view = s._basis_coefs(geom, np.where(below, med.k_m, med.k_p))
    ctx = s._crossing_context(geom, view, med)
    return geom, below, ctx


def _free_field(deck, n, geom):
    """`_field_tensor` @ M for the same geometry in free space: G[obs, basis]
    split by SOURCE segment, as `ref(obs, srcsegs)`."""
    wires, junctions = DECKS[deck]
    f = SinusoidalSolver(
        wires=list(wires),
        n_per_edge_per_wire=[[n], [n]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        junctions=junctions,
    )
    fg = f._build_geometry()
    assert np.allclose(fg["seg_centers"], geom["seg_centers"])
    view = f._basis_coefs(fg, f.k)
    phi = f._field_tensor(fg, f.k, cos_shape="cos")
    n_segs = int(fg["n_segs"])
    starts = view["starts"]
    rows = np.repeat(np.arange(n_segs), starts[1:] - starts[:-1])
    cols = view["jbasis"]
    sig = view["sigma"]
    ms = []
    for c in (sig * view["A"], view["B"], sig * view["C"]):
        m = np.zeros((n_segs, n_segs), dtype=complex)
        m[rows, cols] = c
        ms.append(m)

    def ref(obs, srcsegs):
        return sum(phi[i][np.ix_(obs, srcsegs)] @ ms[i][srcsegs] for i in range(3))

    return ref


@pytest.mark.parametrize("n", [9, 15])
@pytest.mark.parametrize("deck", sorted(DECKS))
def test_at_eps_one_the_block_is_the_free_space_field(deck, n):
    geom, below, ctx = _crossing(deck, (1.0, 0.0), n)
    ref = _free_field(deck, n, geom)
    centres = np.asarray(geom["seg_centers"])
    tangents = np.asarray(geom["seg_tangents"])
    a_idx, b_idx = np.nonzero(~below)[0], np.nonzero(below)[0]
    for obs, src, above in ((a_idx, b_idx, True), (b_idx, a_idx, False)):
        ax = cf.axis_data(ctx, src)
        got = cf.point_observer_block(
            ctx, centres[obs], tangents[obs], ax, observers_above=above
        )
        want = ref(obs, src)
        assert np.max(np.abs(got - want)) <= 2e-5 * np.max(np.abs(want))


@pytest.mark.parametrize("soil", SOILS, ids=[f"er{e:g}-s{s:g}" for e, s in SOILS])
@pytest.mark.parametrize("deck", sorted(DECKS))
def test_integrated_point_rows_are_the_galerkin_rows(deck, soil):
    geom, below, ctx = _crossing(deck, _eps_tilde(*soil), 9)
    a_idx, b_idx = np.nonzero(~below)[0], np.nonzero(below)[0]
    A = cf.axis_data(ctx, a_idx, q=12)
    B = cf.axis_data(ctx, b_idx, q=12)
    for test, src, above, galerkin in (
        (A, B, True, cf.cross_complete_block(ctx, A, B)),
        (B, A, False, cf.cross_complete_block_reversed(ctx, B, A)),
    ):
        point = cf.point_observer_block(
            ctx, test["nodes"], test["t"], src, observers_above=above
        )
        F = test["F_csr"].toarray()
        integ = (F * np.asarray(test["w"])[None, :]) @ point
        end_rows = set()
        for _pt, _sign, fv in test["ends"]:
            end_rows |= set(np.flatnonzero(fv).tolist())
        interior = [
            j for j in range(F.shape[0]) if np.any(F[j] != 0) and j not in end_rows
        ]
        assert len(interior) >= 3
        g = np.asarray(galerkin)[interior]
        assert np.max(np.abs(integ[interior] - g)) <= 1e-12 * np.max(np.abs(g))


def test_the_point_tables_fold_the_radius_like_the_six():
    """`point_radius_tables` evaluates at ρ_eff = hypot(ρ, a), as
    `radius_tables` does, so the chain factor Δh/ρ_eff is consistent."""
    from momwire import _near_interface as ni

    eps = _eps_tilde(13.0, 0.005)
    k = 2 * np.pi / WL7
    rho = np.array([[0.0, 0.3]])
    got = ni.point_radius_tables(eps, k, rho, 0.2, -0.1, 0.001)
    want = ni.point_keys_columns(eps, k, float(np.hypot(0.3, 0.001)), [0.2], -0.1)[0]
    for i, key in enumerate(ni.POINT_KEYS):
        assert got[key][0, 1] == pytest.approx(want[i], rel=1e-15)
    assert got["gRhoV"][0, 0] != 0  # ρ_eff = a > 0: not the axis zero
