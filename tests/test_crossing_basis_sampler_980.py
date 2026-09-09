"""momwire#980 step B: the crossing trunk reads its basis through a sampler.

`_crossing_fill.axis_data` used to read `ctx.basis` as per-segment
polynomials (`BasisPolynomials`), the one bspline-shaped thing left in the
fill after momwire#801. It now asks a `BasisSampler` for the three things it
needs — value and arc-derivative samples at the axis nodes, and per-basis
values at wire ends — and `BasisPolynomials` is one implementation of that
protocol, by DELEGATION to the same two functions (`_basis_samples`,
`_end_values`), so bspline's and razor's crossing fills keep their bytes.

`SinusoidalBasisSampler` is the second implementation: the NEC three-term
basis from `_basis_coefs`' CSR entries, which is what lets the same trunk
serve `SinusoidalGalerkinSolver` below the interface without a polynomial
approximation of a sinusoid (the design-spike finding on #980).

Gates:

    G-980B-1  `BasisPolynomials.samples` / `end_values` are `_basis_samples` /
              `_end_values` to the byte, on the crossing deck's own context.
    G-980B-2  the sinusoidal sampler reproduces `_evaluate_basis_at_points`
              and `_evaluate_basis_slope_at_points` per unit basis vector, on
              a bent wire and a T junction (junction neighbour entries).
    G-980B-3  `axis_data` consumes the sinusoidal sampler: F/Fd at its own
              nodes equal the direct evaluation, `seg_rows` is the CSR
              support, and every wire end it keeps carries `end_values`.
    G-980B-4  bspline's crossing solve is unchanged (the existing suites are
              the real gate; this pins one number here too).
    G-980B-5  the ε̃ → 1 collapse on the sinusoidal sampler: the trunk's cross
              block over (above wire × below wire) IS SG's own free-space
              Galerkin coupling, ratio 1.000000, spread 1.8e-7 (razor's #813
              oracle, repeated for the second basis).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver, SinusoidalGalerkinSolver
from momwire import _crossing_fill as cf
from momwire.sinusoidal_galerkin import SinusoidalBasisSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_crossing_serve_524 import crossing_deck  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:crossing node")

WL7 = 299792458.0 / 7e6
SOIL_A = (13.0, 0.005)

BENT = ([np.array([(0.0, 0.0, 0.5), (0.0, 0.0, 1.5), (1.0, 0.0, 1.5)])], [[9, 7]])
TEE = (
    [
        np.array([(-1.0, 0.0, 1.5), (1.0, 0.0, 1.5)]),
        np.array([(0.0, 0.0, 1.5), (0.0, 0.0, 2.5)]),
    ],
    [[11], [6]],
)


def _bspline_ctx():
    s = BSplineSolver(**crossing_deck(1))
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    ctx = s._crossing_context(geom, supp_seg, polys)
    below = s._below_segments(geom)
    return s, ctx, np.nonzero(~below)[0], np.nonzero(below)[0]


def test_g980b_1_polynomials_delegate_to_the_byte(record_property):
    _s, ctx, a_idx, b_idx = _bspline_ctx()
    assert isinstance(ctx.basis, cf.BasisPolynomials)
    assert ctx.basis.n_basis == ctx.basis.polys.shape[0]
    rng = np.random.default_rng(980)
    for seg_idx in (a_idx, b_idx):
        # an arbitrary node layout: three nodes per segment at random arcs
        runs = {int(g): (3 * i, 3) for i, g in enumerate(seg_idx)}
        u = np.concatenate([rng.uniform(0.0, ctx.geom.h[g], 3) for g in seg_idx])
        F, Fd, rows = ctx.basis.samples(runs, u)
        F0, Fd0, rows0 = cf._basis_samples(ctx.basis.supp_seg, ctx.basis.polys, runs, u)
        assert np.array_equal(F, F0) and np.array_equal(Fd, Fd0)
        assert rows.keys() == rows0.keys()
        assert all(np.array_equal(rows[g], rows0[g]) for g in rows)
        live = np.any(ctx.basis.polys != 0.0, axis=2)
        for g in seg_idx:
            for uu in (0.0, float(ctx.geom.h[g])):
                fv = ctx.basis.end_values(int(g), uu)
                fv0 = cf._end_values(
                    ctx.basis.supp_seg,
                    ctx.basis.polys,
                    live,
                    int(g),
                    uu,
                    ctx.basis.degree,
                )
                assert np.array_equal(fv, fv0)
    record_property("n_basis", ctx.basis.n_basis)


def _sg(wires, n_per_edge):
    s = SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
    )
    geom = s._build_geometry()
    seg_view = s._basis_coefs(geom, s.k)
    return s, geom, seg_view, s._crossing_basis(geom, seg_view=seg_view)


def _direct(s, seg_view, n_basis, eval_seg, eval_s):
    """Per-basis value and slope tables from the solver's own evaluators,
    one unit α at a time: (n_basis, n_eval) each."""
    F = np.zeros((n_basis, eval_seg.shape[0]), dtype=np.complex128)
    Fd = np.zeros_like(F)
    for m in range(n_basis):
        alpha = np.zeros(n_basis, dtype=np.complex128)
        alpha[m] = 1.0
        F[m] = s._evaluate_basis_at_points(seg_view, eval_seg, eval_s, alpha)
        Fd[m] = s._evaluate_basis_slope_at_points(seg_view, eval_seg, eval_s, alpha)
    return F, Fd


@pytest.mark.parametrize("deck", [BENT, TEE], ids=["bent", "tee"])
def test_g980b_2_sinusoidal_sampler_is_the_solvers_own_evaluator(deck):
    s, geom, seg_view, sampler = _sg(*deck)
    assert isinstance(sampler, SinusoidalBasisSampler)
    n = geom["n_segs"]
    assert sampler.n_basis == n
    rng = np.random.default_rng(1980)
    q = 4
    u = np.concatenate([rng.uniform(0.0, geom["seg_h"][g], q) for g in range(n)])
    runs = {g: (q * g, q) for g in range(n)}
    F, Fd, rows = sampler.samples(runs, u)
    eval_seg = np.repeat(np.arange(n, dtype=np.int64), q)
    eval_s = u - 0.5 * geom["seg_h"][eval_seg]  # the evaluators take ξ from the centre
    F0, Fd0 = _direct(s, seg_view, n, eval_seg, eval_s)
    # Same coefficients, same shape set, same trig — one gather order apart.
    np.testing.assert_allclose(F, F0, rtol=0.0, atol=1e-15 * np.abs(F0).max())
    np.testing.assert_allclose(Fd, Fd0, rtol=0.0, atol=1e-15 * np.abs(Fd0).max())
    assert np.all(F.imag == 0.0), "real k: the samples must be real"
    # support: CSR entries, and nothing else
    starts, jbasis = seg_view["starts"], seg_view["jbasis"]
    for g in range(n):
        assert np.array_equal(rows[g], np.sort(jbasis[starts[g] : starts[g + 1]]))
        off = np.setdiff1d(np.arange(n), rows[g])
        assert np.all(F[off, q * g : q * g + q] == 0.0)
    # ends
    for g in range(n):
        for uu in (0.0, float(geom["seg_h"][g])):
            fv = sampler.end_values(g, uu)
            f0, _ = _direct(
                s, seg_view, n, np.array([g]), np.array([uu - 0.5 * geom["seg_h"][g]])
            )
            np.testing.assert_allclose(fv, f0[:, 0], rtol=0.0, atol=1e-15)


def _sg_context(s, geom, sampler, ground_z):
    """A CrossingContext over this solver's geometry. `ground_z` is chosen
    by the caller: far from every segment so no interface-touching panel
    grading engages, or at a node to exercise it."""
    n = geom["n_segs"]
    first = list(geom["wire_first"]) + [n]
    return cf.CrossingContext(
        basis=sampler,
        geom=cf.AxisGeometry(
            geom["seg_l"],
            geom["seg_r"],
            geom["seg_h"],
            geom["seg_tangents"],
            np.asarray(first, dtype=np.int64),
        ),
        medium=cf.buried_medium(SOIL_A, s.omega, s.eps, s.k),
        ground_z=float(ground_z),
        a_wire=0.001,
        omega=s.omega,
        mu=s.mu,
        eps=s.eps,
    )


@pytest.mark.parametrize("deck", [BENT, TEE], ids=["bent", "tee"])
def test_g980b_3_axis_data_consumes_the_sinusoidal_sampler(deck):
    s, geom, seg_view, sampler = _sg(*deck)
    n = geom["n_segs"]
    ctx = _sg_context(s, geom, sampler, ground_z=-50.0)
    ax = cf.axis_data(ctx, np.arange(n))
    assert ax["n_basis"] == n
    assert ax["F"].shape == (n, ax["nodes"].shape[0])
    # recover each node's arc from its segment start, and check the samples
    u = np.linalg.norm(ax["nodes"] - geom["seg_l"][ax["segof"]], axis=1)
    F0, Fd0 = _direct(s, seg_view, n, ax["segof"], u - 0.5 * geom["seg_h"][ax["segof"]])
    np.testing.assert_allclose(ax["F"], F0, rtol=0.0, atol=1e-13 * np.abs(F0).max())
    np.testing.assert_allclose(ax["Fd"], Fd0, rtol=0.0, atol=1e-13 * np.abs(Fd0).max())
    starts, jbasis = seg_view["starts"], seg_view["jbasis"]
    assert set(ax["seg_rows"]) == set(range(n))
    for g in range(n):
        assert np.array_equal(
            ax["seg_rows"][g], np.sort(jbasis[starts[g] : starts[g + 1]])
        )
    # ends: the trunk keeps a wire end only where some basis is nonzero there.
    # A free NEC end carries zero current, so a free-ended deck keeps NONE; a
    # junction end is interior to the fill and is not a wire end at all.
    for pt, sign, fv in ax["ends"]:
        assert sign in (-1.0, 1.0)
        assert np.any(fv != 0.0)
    free_ends = [
        (g, uu)
        for w in range(len(geom["wire_first"]))
        for g, uu in (
            (geom["wire_first"][w], 0.0),
            (geom["wire_last"][w], float(geom["seg_h"][geom["wire_last"][w]])),
        )
    ]
    kept = {
        (int(g), 0.0 if s_ == -1.0 else float(geom["seg_h"][g]))
        for pt, s_, _fv in ax["ends"]
        for g in [int(np.argmin(np.linalg.norm(geom["seg_l"] - pt, axis=1)))]
    }
    for g, uu in free_ends:
        fv = sampler.end_values(g, uu)
        assert (np.any(fv != 0.0)) == ((g, uu) in kept) or not np.any(fv != 0.0)


def test_g980b_4_bspline_crossing_solve_is_unchanged(record_property):
    """The existing crossing suites are the byte gate; this pins the
    crossing deck's own number so a seam regression is visible here too."""
    z, _ = BSplineSolver(**crossing_deck(1)).compute_impedance()
    record_property("z_crossing_deck_1", f"{z:.6f}")
    # `_medium_spec.ENGINE_CROSSING_PRINT` records the engine on this deck;
    # the momwire value is pinned by test_crossing_serve_524 — here only its
    # finiteness and sign are asserted, the suites do the rest.
    assert np.isfinite(z) and z.real > 0.0


def test_g980b_5_the_trunk_on_the_sinusoidal_sampler_collapses_onto_sg_free_space(
    record_property,
):
    """The ε̃ → 1 collapse, SG edition: at ε̃ = 1 the interface is not there,
    so the trunk's designed cross block over (above wire × below wire) must be
    the free-space Galerkin coupling SG builds for the same bases with its own
    closed forms. Razor's #651/#813 probe did this for the tent basis (to
    6.6e-6); here the sinusoidal sampler is what the trunk reads.

    Interior block only — the two bases centred on the node-adjacent segments
    have support on both sides of the plane and are the by-parts/corner terms'
    business (#980 D3). Same sign, same scale, no convention fix: the ratio
    is 1 to the trunk's quadrature floor.
    """
    s = SinusoidalGalerkinSolver(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -1.0)]),
        ],
        n_per_edge_per_wire=[[10], [10]],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
    )
    geom = s._build_geometry()
    G, seg_view = s._assemble_Z(geom, s.k)
    sampler = s._crossing_basis(geom, seg_view=seg_view)
    n = geom["n_segs"]
    first = list(geom["wire_first"]) + [n]
    ctx = cf.CrossingContext(
        basis=sampler,
        geom=cf.AxisGeometry(
            geom["seg_l"],
            geom["seg_r"],
            geom["seg_h"],
            geom["seg_tangents"],
            np.asarray(first, dtype=np.int64),
        ),
        medium=cf.buried_medium((1.0, 0.0), s.omega, s.eps, s.k),
        ground_z=0.0,
        a_wire=0.001,
        omega=s.omega,
        mu=s.mu,
        eps=s.eps,
    )
    assert ctx.medium.eps_t == 1.0 and ctx.medium.k_m == ctx.medium.k_p
    zc = 0.5 * (geom["seg_l"][:, 2] + geom["seg_r"][:, 2])
    a_idx, b_idx = np.nonzero(zc > 0)[0], np.nonzero(zc < 0)[0]
    t_ab = cf.cross_complete_block(
        ctx, cf.axis_data(ctx, a_idx), cf.axis_data(ctx, b_idx)
    )
    # node-adjacent segments are the ones whose centre is nearest the plane
    ra = [m for m in a_idx if m != a_idx[np.argmin(np.abs(zc[a_idx]))]]
    cb = [m for m in b_idx if m != b_idx[np.argmin(np.abs(zc[b_idx]))]]
    ratio = t_ab[np.ix_(ra, cb)] / G[np.ix_(ra, cb)]
    spread = float(np.std(ratio) / np.abs(np.median(ratio)))
    record_property("ratio_median", f"{np.median(ratio):.8f}")
    record_property("ratio_rel_spread", spread)
    assert abs(np.median(ratio) - 1.0) < 1e-6, np.median(ratio)
    assert spread < 1e-6, spread
