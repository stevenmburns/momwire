"""BSplineSolver ground_model="sommerfeld" tests (plan Phases 3-4).

Golden gates use the nec2c-captured gn 2 values ("finite" key) — NOT
PyNEC's, whose Sommerfeld solve is order-dependent and breaks below
0.1 wavelength (see the capture script's docstring). Measured residuals
at capture (2026-07-06), full 39-case matrix, |Z_somm - Z_gn2|:
dipole max 2.36 ohm (across 0.02-0.5 wl, all three grounds), inverted_l
max 2.74, yagi max 0.98 — i.e. the bspline-vs-NEC cross-solver floor at
every height, where the refl-coef model is ~22 ohm off at 0.05 wl and
>130 ohm off at 0.02 wl. Gates are set ~1.3x above those measurements.

The golden gates run at BOTH bspline degrees (parametrized). Degree 2 is
the constructor default and was the only degree pinned here originally;
degree 1 is the batched-swept basis the antennaknobs web sweep serves and
was otherwise unpinned against the physical oracle on Sommerfeld ground.
Its measured residuals sit in the same band (dipole 2.41, inverted_l 2.27,
yagi 1.54 — the last a touch above degree 2's 0.98 on that off-resonance
geometry, so yagi carries a per-degree gate).
"""

import sys

import numpy as np
import pytest

from momwire import ArrayBlockSolver, BSplineSolver, HMatrixSolver
from momwire import _ground_refl, _sommerfeld

from fixtures_refl_coef_geoms import GEOMS
from golden_refl_coef_ground import GOLDEN


def _solver(name, frac, cls=BSplineSolver, **overrides):
    kw = dict(GEOMS[(name, frac)])
    kw["ground_z"] = 0.0
    kw.update(overrides)
    return cls(**kw)


def _solve(name, frac, **overrides):
    z, _ = _solver(name, frac, **overrides).compute_impedance()
    return z


SOMM = {"ground_eps": (10.0, 0.002), "ground_model": "sommerfeld"}


# ---------------------------------------------------------------------------
# Constructor contract
# ---------------------------------------------------------------------------


def test_ground_model_validation():
    kw = dict(GEOMS[("dipole", 0.2)])
    with pytest.raises(ValueError, match="ground_model"):
        BSplineSolver(**kw, ground_z=0.0, ground_eps=(10, 0.002), ground_model="nope")
    with pytest.raises(ValueError, match="requires ground_eps"):
        BSplineSolver(**kw, ground_z=0.0, ground_model="sommerfeld")
    with pytest.raises(ValueError, match="requires ground_z"):
        BSplineSolver(**kw, ground_eps=(10, 0.002), ground_model="sommerfeld")


@pytest.mark.slow
def test_wires_below_ground_are_SERVED_over_sommerfeld():
    """momwire#553 U5 moved this one from a refusal to an answer.

    Since momwire#151 a wire under the plane was rejected at geometry build
    under EVERY ground model, because there was no lower medium anywhere in
    the tree. There is one now — but only under `sommerfeld`, which is the
    only ground that carries an eps_tilde and therefore a k_m. So the
    Sommerfeld row serves the deck (through the below/below direct, image and
    remainder blocks) and the other rows keep the refusal, with a sentence
    that now names WHICH card has no lower medium instead of naming the
    geometry.
    """
    kw = dict(GEOMS[("dipole", 0.2)])
    z_top = max(p[2] for wire in kw["wires"] for p in wire)
    s = BSplineSolver(**kw, ground_z=z_top + 1.0, **SOMM)
    z, _ = s.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert z.real > 0.0


@pytest.mark.parametrize(
    "ground,needle",
    [
        ({}, "PERFECTLY CONDUCTING"),
        (
            {"ground_eps": SOMM["ground_eps"], "ground_model": "refl-coef"},
            "plane-wave boundary condition",
        ),
    ],
    ids=["pec", "refl-coef"],
)
def test_wires_below_a_ground_with_no_lower_medium_are_still_rejected(ground, needle):
    kw = dict(GEOMS[("dipole", 0.2)])
    z_top = max(p[2] for wire in kw["wires"] for p in wire)
    s = BSplineSolver(**kw, ground_z=z_top + 1.0, **ground)
    with pytest.raises(ValueError) as exc:
        s.compute_impedance()
    assert needle in str(exc.value)
    assert "ground_model='sommerfeld'" in str(exc.value)


def test_default_model_is_refl_coef():
    """ground_model defaults to refl-coef: explicit and default solves are
    bit-identical, so v0.5.0 behavior is unchanged."""
    z_default = _solve("dipole", 0.2, ground_eps=(10.0, 0.002))
    z_explicit = _solve(
        "dipole", 0.2, ground_eps=(10.0, 0.002), ground_model="refl-coef"
    )
    # momwire#809: the two sides' fills measured BIT-IDENTICAL, so this
    # `==` is structural, not a solve-downstream lottery ticket.
    assert z_default == z_explicit


# ---------------------------------------------------------------------------
# Far-pair grid cap (issue #157)
# ---------------------------------------------------------------------------


def test_remote_wire_stays_bounded_and_irrelevant(monkeypatch):
    """A 1-segment wire parked ~150 wavelengths from a dipole — the NEC
    TL-anchor idiom, and any large structure over real ground (issue #157) —
    used to size the Sommerfeld grid to hundreds of wavelengths and hang the
    fill. With the r1_max cap the solve completes, and the electrically
    irrelevant far wire leaves the driven impedance essentially unchanged.
    (Completion itself is the no-hang guard: pre-cap this did not return.)

    The cap is knocked down to a few wavelengths here so the test builds a
    small grid quickly; the mechanism (far geometry -> capped grid -> bounded
    fill) is identical at the production default, which the grid unit test
    pins."""
    monkeypatch.setattr(_sommerfeld, "_SOMM_R1_CAP_LAMBDA", 4.0)
    _sommerfeld._GRID_CACHE.clear()  # don't reuse a grid built at another cap
    _sommerfeld._NORM_CACHE.clear()
    base = dict(GEOMS[("dipole", 0.2)])
    lam = base["wavelength"]
    z_ctrl, _ = BSplineSolver(**base, ground_z=0.0, **SOMM).compute_impedance()

    h = base["wires"][0][0][2]  # dipole height above the plane
    d = 150.0 * lam
    anchored = dict(base)
    anchored["wires"] = base["wires"] + [[[d, d, h], [d, d + 0.01, h]]]
    anchored["n_per_edge_per_wire"] = base["n_per_edge_per_wire"] + [[1]]
    z_anc, _ = BSplineSolver(**anchored, ground_z=0.0, **SOMM).compute_impedance()

    assert abs(z_anc - z_ctrl) < 0.1  # 150-lambda wire couples negligibly


# ---------------------------------------------------------------------------
# Exact limits
# ---------------------------------------------------------------------------


def test_free_space_limit():
    """eps -> 1: C2 = 0 and the remainder integrands vanish identically,
    so the grounded solve reproduces the no-ground solve."""
    kw = dict(GEOMS[("dipole", 0.2)])
    z_free, _ = BSplineSolver(**kw).compute_impedance()
    z_g = _solve("dipole", 0.2, ground_eps=1.0 + 0.0j, ground_model="sommerfeld")
    assert abs(z_g - z_free) / abs(z_free) < 1e-9


def test_pec_limit_collapses_to_image():
    """eps -> 1e16: C2 -> 1 and the remainder scales away (~1/sqrt(eps)),
    reproducing the PEC-image solve."""
    z_pec = _solve("dipole", 0.1)  # ground_z set, no ground_eps -> PEC image
    z_g = _solve("dipole", 0.1, ground_eps=1e16 + 0.0j, ground_model="sommerfeld")
    assert abs(z_g - z_pec) / abs(z_pec) < 1e-5


def test_tuple_and_complex_eps_equivalent():
    s = _solver("dipole", 0.2, **SOMM)
    eps_c = _ground_refl.eps_tilde((10.0, 0.002), s.omega, s.eps)
    z_t = _solve("dipole", 0.2, **SOMM)
    z_c = _solve("dipole", 0.2, ground_eps=eps_c, ground_model="sommerfeld")
    assert abs(z_t - z_c) / abs(z_t) < 1e-12


# ---------------------------------------------------------------------------
# Numerics contracts
# ---------------------------------------------------------------------------


def test_swept_matches_single_k():
    s = _solver("dipole", 0.2, **SOMM)
    k0 = s.k
    ks = np.array([0.97 * k0, k0, 1.03 * k0])
    z_swept = s.compute_impedance_swept(ks)
    z_single = _solve("dipole", 0.2, **SOMM)
    assert abs(z_swept[1] - z_single) / abs(z_single) < 1e-10
    # the flanking entries used per-k eps(omega) and grids: they must
    # differ from the center (guards a frozen-omega bug)
    assert abs(z_swept[0] - z_single) > 1e-3


def test_y_matrix_consistent_with_impedance():
    s = _solver("dipole", 0.2, **SOMM)
    y = s.compute_y_matrix()
    z = _solve("dipole", 0.2, **SOMM)
    assert abs(1.0 / y[0, 0] - z) / abs(z) < 1e-9


def test_quadrature_order_converged():
    """The remainder kernel is smooth (image point below the plane):
    n_qp_sommerfeld=3 vs 5 must agree far inside the physics residual."""
    z3 = _solve("dipole", 0.05, **SOMM, n_qp_sommerfeld=3)
    z5 = _solve("dipole", 0.05, **SOMM, n_qp_sommerfeld=5)
    assert abs(z3 - z5) / abs(z3) < 1e-3


def test_remainder_block_symmetric():
    """Reciprocity: the half-space dyad is symmetric, so Q must be too
    (up to grid-interpolation noise)."""
    s = _solver("dipole", 0.05, **SOMM)
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    q = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)
    asym = np.max(np.abs(q - q.T)) / np.max(np.abs(q))
    assert asym < 5e-3


def test_fast_solvers_match_dense():
    """HMatrix/ArrayBlock run sommerfeld on the FAST path since the
    sommerfeld-everywhere work (C2-scaled PEC-image blocks + one global
    low-rank remainder term) — results must match the dense BSplineSolver
    at ACA/GMRES tolerance, and the global remainder must actually be
    low rank (measured 8-17 on this dipole; ~50 is the plan's
    reconsider-the-architecture trigger)."""
    z_dense = _solve("dipole", 0.1, **SOMM)
    for cls in (HMatrixSolver, ArrayBlockSolver):
        s = _solver("dipole", 0.1, cls=cls, **SOMM)
        z_fast, _ = s.compute_impedance()
        assert abs(z_fast - z_dense) / abs(z_dense) < 1e-3, cls.__name__
        assert s._last_somm_rank < 50, cls.__name__


def test_rect_remainder_matches_dense_block():
    """The fast solvers' rectangular remainder sampler on the full index
    range reproduces the dense Galerkin remainder block — same dyad
    algebra, same shared grid, different plumbing."""
    s = _solver("dipole", 0.05, cls=HMatrixSolver, **SOMM)
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    q_dense = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)
    idx = np.arange(supp_seg.shape[0])
    q_rect = s._zblock_sommerfeld_remainder(idx, idx, eps_t=eps_t)
    assert np.max(np.abs(q_rect - q_dense)) / np.max(np.abs(q_dense)) < 1e-10


# ---------------------------------------------------------------------------
# Golden gn 2 gates (nec2c oracle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    "frac,ground",
    [
        (0.02, (10.0, 0.002)),
        (0.02, (3.0, 0.001)),
        (0.05, (10.0, 0.002)),
        (0.2, (13.0, 0.005)),
        (0.5, (10.0, 0.002)),
    ],
)
def test_dipole_tracks_gn2(frac, ground, degree):
    """Both bspline degrees track the nec2c gn 2 oracle. Measured max
    residual across the full dipole matrix: 2.36 ohm (degree 2), 2.41
    (degree 1) — gate at 3.0 for both. The degree-1 gate matters because
    it is the batched-swept default nowhere else pinned against the
    physical oracle on Sommerfeld ground."""
    gn2 = GOLDEN[("dipole", frac, *ground)]["finite"]
    z = _solve(
        "dipole", frac, degree=degree, ground_eps=ground, ground_model="sommerfeld"
    )
    assert abs(z - gn2) < 3.0


def test_dipole_tracks_gn2_via_tensor_route_fallback(monkeypatch):
    """#273: the same nec2c gn2 gate as `test_dipole_tracks_gn2` at
    frac=0.2, ground (13.0, 0.005), degree 2 — forced onto the
    no-accelerator tensor-route image fill (`_build_J_image_blocks` feeding
    `_ground_finite_Z`'s sommerfeld branch: the constant-C2 weighted image
    plus `_Z_sommerfeld_remainder`) instead of the chunked accumulator this
    box's accelerated build always takes. Before this test, Sommerfeld's
    fallback route had zero physics-oracle coverage — only the generic
    route-equality pin in test_momwire.py touched it at all."""
    import momwire.bspline as bmod

    if not bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("weighted windowed Z assembly accelerator not built")
    monkeypatch.setattr(bmod, "_HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL", False)
    gn2 = GOLDEN[("dipole", 0.2, 13.0, 0.005)]["finite"]
    z = _solve(
        "dipole", 0.2, degree=2, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    )
    assert abs(z - gn2) < 3.0


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("frac", [0.02, 0.2])
def test_inverted_l_tracks_gn2(frac, degree):
    """Junction + vertical-current geometry, both degrees; measured max
    2.74 ohm (degree 2), 2.27 (degree 1) — gate at 3.5."""
    gn2 = GOLDEN[("inverted_l", frac, 10.0, 0.002)]["finite"]
    z = _solve(
        "inverted_l",
        frac,
        degree=degree,
        ground_eps=(10.0, 0.002),
        ground_model="sommerfeld",
    )
    assert abs(z - gn2) < 3.5


@pytest.mark.parametrize("degree,tol", [(1, 2.0), (2, 1.5)])
def test_yagi_tracks_gn2_large_r1(degree, tol):
    """>1.2-wavelength boom: image-ray distances past NEC's 1-wavelength
    grid edge exercise the geometry-sized grid. Measured max 0.98 ohm
    (degree 2, gate 1.5); the lower-order degree-1 basis sits at 1.54 on
    this off-resonance |Z|~52 geometry, gated at 2.0."""
    gn2 = GOLDEN[("yagi", 0.2, 10.0, 0.002)]["finite"]
    z = _solve(
        "yagi", 0.2, degree=degree, ground_eps=(10.0, 0.002), ground_model="sommerfeld"
    )
    assert abs(z - gn2) < tol


def test_beats_refl_coef_below_010():
    """The point of the exercise: at 0.02 wl the refl-coef model is
    >20 ohm from gn 2; sommerfeld must recover >80% of that gap."""
    gn2 = GOLDEN[("dipole", 0.02, 10.0, 0.002)]["finite"]
    z_somm = _solve("dipole", 0.02, **SOMM)
    z_refl = _solve("dipole", 0.02, ground_eps=(10.0, 0.002))
    assert abs(z_somm - gn2) < 0.2 * abs(z_refl - gn2)
    assert abs(z_refl - gn2) > 20.0  # the gap being closed is real


# ---------------------------------------------------------------------------
# Module-level grid cache (perf plan Phase 1)
# ---------------------------------------------------------------------------


def test_somm_grid_shared_across_solver_instances():
    """The engine wrapper builds a fresh solver per impedance() call; the
    grid must survive that (identical key -> identical object)."""
    from momwire import _sommerfeld as sm

    sm._GRID_CACHE.clear()
    s1 = _solver("dipole", 0.05, **SOMM)
    s2 = _solver("dipole", 0.05, **SOMM)
    z1, _ = s1.compute_impedance()
    assert len(sm._GRID_CACHE) == 1
    grid = next(iter(sm._GRID_CACHE.values()))
    z2, _ = s2.compute_impedance()
    assert len(sm._GRID_CACHE) == 1
    assert next(iter(sm._GRID_CACHE.values())) is grid
    # momwire#809: the two sides' fills measured BIT-IDENTICAL, so this
    # `==` is structural, not a solve-downstream lottery ticket.
    assert z1 == z2


def test_somm_r1_bucket_rounds_up_and_reuses():
    from momwire import _sommerfeld as sm

    k = 2.0 * np.pi / 20.0  # 20 m wavelength
    lam = 20.0
    for r1 in (0.3 * lam, 1.7 * lam, 6.0 * lam):
        b = sm._somm_r1_bucket(r1, k)
        assert b >= r1  # never tabulate short of the geometry
        assert b <= 1.25 * r1 * (1 + 1e-9)  # bounded overshoot
        # nearby radii (a knob-turn) land in the same bucket
        assert sm._somm_r1_bucket(0.99 * b, k) == b
    # tiny radii share one floor bucket (the first 1.25^n step >= 0.1 wl)
    b_tiny = sm._somm_r1_bucket(1e-6, k)
    assert 0.1 * lam <= b_tiny <= 0.13 * lam
    assert sm._somm_r1_bucket(0.05 * lam, k) == b_tiny


def test_somm_eps_bucket_ladder():
    """Im(eps_t) rounds to the nearest rung of the geometric ladder (worst
    offset half a step); Re is exact; nonstandard values pass through."""
    from momwire import _sommerfeld as sm

    step = 1.0 + sm._SOMM_EPS_IM_BUCKET
    assert sm._SOMM_EPS_IM_BUCKET == pytest.approx(0.01)  # the shipped default
    e = 10.0 - 1.26j
    b = sm._somm_eps_bucket(e)
    assert b.real == e.real
    assert abs(b.imag / e.imag - 1.0) <= (step - 1.0) / 2 * (1 + 1e-9)
    # nearby frequencies (a band sweep tick) land on the same rung
    assert sm._somm_eps_bucket(complex(e.real, e.imag * 1.002)) == b
    # pass-throughs: lossless, free space, nonpassive, nonphysical Re
    for weird in (16.0 + 0.0j, 1.0 + 0.0j, 10.0 + 2.0j, -3.0 - 1.0j):
        assert sm._somm_eps_bucket(weird) == weird


def test_somm_scaled_view_matches_direct_fill():
    """The frequency-reuse scaling law: a master rescaled by `scaled_to`
    must reproduce a from-scratch fill at the target (k2, omega) — the
    lattice is lambda-proportional and S = omega*mu*G(eps; R1/lam, theta),
    so agreement is at quadrature/rounding level, not interpolation level."""
    from momwire import _sommerfeld as sm

    eps = 10.0 - 1.26j
    k_a, k_b = 2.0 * np.pi / 20.0, 2.0 * np.pi / 11.0  # 20 m -> 11 m
    master = sm.SommerfeldGrid(eps, k_a, r1_max=1.2 * 20.0, omega=k_a * sm._C_LIGHT)
    view = master.scaled_to(k_b, k_b * sm._C_LIGHT, sm._MU0)
    direct = sm.SommerfeldGrid(eps, k_b, r1_max=1.2 * 11.0, omega=k_b * sm._C_LIGHT)
    assert view.r1_max == pytest.approx(direct.r1_max)
    assert len(view._regions) == len(direct._regions)
    rng = np.random.default_rng(31)
    r1 = rng.uniform(0.0, 1.19 * 11.0, 150)
    th = rng.uniform(0.0, np.pi / 2, 150)
    a = view.eval(r1, th)
    b = direct.eval(r1, th)
    for kk in sm._SURF_KEYS:
        scale = np.abs(b[kk]).max()
        assert np.abs(a[kk] - b[kk]).max() < 1e-6 * scale, kk


def test_somm_grid_frequency_reuse_one_fill_per_rung():
    """A band sweep pays one fill per eps-ladder rung, not one per
    frequency — and the bucketed views still track the true-eps surfaces
    within the grid accuracy bar (issue #159 phase 2)."""
    from momwire import _sommerfeld as sm

    eps0_im = 0.002 / (2.0 * np.pi * 28.4e6 * 8.8541878128e-12)  # sigma/(w*eps0)
    fills = []
    orig = sm.SommerfeldGrid.__init__

    def counting(self, *a, **kw):
        fills.append(a)
        orig(self, *a, **kw)

    sm._GRID_CACHE.clear()
    sm._NORM_CACHE.clear()
    try:
        sm.SommerfeldGrid.__init__ = counting
        views = []
        for fmhz in np.linspace(28.35, 28.45, 7):  # ~0.35% span: one rung
            w = 2.0 * np.pi * fmhz * 1e6
            k2 = w / sm._C_LIGHT
            eps = 10.0 - 1j * 0.002 / (w * 8.8541878128e-12)
            views.append(sm.get_grid(eps, k2, 15.0, omega=w))
    finally:
        sm.SommerfeldGrid.__init__ = orig
        sm._GRID_CACHE.clear()
        sm._NORM_CACHE.clear()
    assert len(fills) == 1  # every sweep point shared one master fill
    assert len({id(v) for v in views}) == len(views)  # but distinct views
    # normalized master: filled at the reference wavenumber, bucketed eps
    eps_m, k_m = fills[0][0], fills[0][1]
    assert k_m == pytest.approx(sm._K2_REF)
    assert abs(eps_m.imag / -eps0_im - 1.0) < sm._SOMM_EPS_IM_BUCKET
    # end-to-end accuracy at the sweep edge (largest bucket offset): view
    # vs direct evaluation at the TRUE eps holds the grid bar
    w = 2.0 * np.pi * 28.45e6
    k2 = w / sm._C_LIGHT
    eps_true = 10.0 - 1j * 0.002 / (w * 8.8541878128e-12)
    v = views[-1]
    rng = np.random.default_rng(41)
    r1 = rng.uniform(0.0, 14.0, 200)
    th = rng.uniform(0.0, np.pi / 2, 200)
    got = v.eval(r1, th)
    want = sm.iv_surfaces_direct(eps_true, k2, r1, th, rtol=1e-8, omega=w)
    for kk in sm._SURF_KEYS:
        scale = np.abs(want[kk]).max()
        assert np.abs(got[kk] - want[kk]).max() < 2.5e-3 * scale, kk


def test_somm_grid_cache_bounded():
    from momwire import _sommerfeld as sm

    sm._GRID_CACHE.clear()
    try:
        for i in range(sm._GRID_CACHE_MAX + 5):
            sm._GRID_CACHE[("sentinel", i)] = None
            sm._evict_fifo(sm._GRID_CACHE, sm._GRID_CACHE_MAX)
        assert len(sm._GRID_CACHE) <= sm._GRID_CACHE_MAX
    finally:
        sm._GRID_CACHE.clear()


# ---------------------------------------------------------------------------
# Phase 4b: the fused C++ Galerkin remainder kernel must match the numpy
# assembly path (which itself is gated against nec2c gn 2 above).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2])
def test_fused_Q_kernel_matches_numpy_galerkin(degree, monkeypatch):
    import momwire.bspline as bs

    if bs._acc is None or not hasattr(bs._acc, "sommerfeld_remainder_bspline_Q"):
        pytest.skip("fused sommerfeld kernel unavailable")

    z_fused = _solve("yagi", 0.2, degree=degree, **SOMM)

    # Hide only the fused kernel, keeping the per-pair proj kernel and the
    # assemblers, so the fallback path is proj-kernel + numpy Galerkin einsum.
    real = bs._acc

    class _NoFused:
        def __getattr__(self, name):
            if name == "sommerfeld_remainder_bspline_Q":
                raise AttributeError(name)
            return getattr(real, name)

    monkeypatch.setattr(bs, "_acc", _NoFused())
    z_np = _solve("yagi", 0.2, degree=degree, **SOMM)

    assert np.abs(z_fused - z_np).max() / np.abs(z_np).max() < 1e-11


@pytest.mark.parametrize("cls", [HMatrixSolver, ArrayBlockSolver])
def test_fused_Q_rect_kernel_matches_numpy_fast_solvers(cls, monkeypatch):
    """The ACA sampler's rectangular fused kernel (obs != src) must match
    the numpy Galerkin path for both fast solvers — same rectangular kernel
    the dense block uses, exercised through the low-rank remainder term."""
    import momwire.hmatrix as hm

    if hm._acc is None or not hasattr(hm._acc, "sommerfeld_remainder_bspline_Q"):
        pytest.skip("fused sommerfeld kernel unavailable")

    z_fused = _solve("yagi", 0.2, cls=cls, **SOMM)

    real = hm._acc

    class _NoFused:
        def __getattr__(self, name):
            if name == "sommerfeld_remainder_bspline_Q":
                raise AttributeError(name)
            return getattr(real, name)

    monkeypatch.setattr(hm, "_acc", _NoFused())
    z_np = _solve("yagi", 0.2, cls=cls, **SOMM)

    assert np.abs(z_fused - z_np).max() / np.abs(z_np).max() < 1e-10


# ---------------------------------------------------------------------------
# `_sommerfeld.max_image_distance` (issue #331): the r1_max endpoint scan
# used to be pasted verbatim at 4 call sites, each materializing a (2N, 2N)
# float64 distance matrix just to read off its max. The shared, row-chunked
# helper must be BIT-IDENTICAL to that deleted expression (not merely
# close) — a max over row bands is exact, not an approximation, so any
# drift here means the chunking broke something.
# ---------------------------------------------------------------------------


def _naive_r1_max(seg_l, seg_r, ground_z):
    """The exact expression `max_image_distance` replaced at all 4 sites —
    kept here, independent of the production helper, as the reference the
    helper is checked against."""
    ex = np.concatenate([seg_l, seg_r])
    dxe = ex[:, 0][:, None] - ex[:, 0][None, :]
    dye = ex[:, 1][:, None] - ex[:, 1][None, :]
    hze = (ex[:, 2][:, None] - ground_z) + (ex[:, 2][None, :] - ground_z)
    return float(np.sqrt(dxe * dxe + dye * dye + hze * hze).max()) * 1.001


def test_max_image_distance_matches_naive_on_random_cloud():
    """(a) A 500-endpoint (250-segment) random cloud, default chunking —
    exact equality, not allclose."""
    rng = np.random.default_rng(331)
    seg_l = rng.uniform(-40.0, 40.0, (250, 3))
    seg_r = rng.uniform(-40.0, 40.0, (250, 3))
    ground_z = 1.7
    got = _sommerfeld.max_image_distance(seg_l, seg_r, ground_z)
    want = _naive_r1_max(seg_l, seg_r, ground_z)
    assert got == want


def test_max_image_distance_finds_max_in_final_partial_chunk():
    """(b) The winning pair's ROW sits in the final, PARTIAL row-chunk —
    genuinely exercises the tail, not just the interior. 5 segments (10
    endpoints) with an explicit `chunk_rows=3` gives row bands
    [0:3), [3:6), [6:9), [9:10) — the last one width-1. The extreme point
    (endpoint index 9, i.e. `seg_r[4]`) sits only in that final chunk: if
    a chunking bug dropped the tail (e.g. `range(0, n - chunk, chunk)`),
    this pair — and the correct r1_max — would go missing.
    """
    seg_l = np.zeros((5, 3))
    seg_r = np.zeros((5, 3))
    seg_r[4] = [0.0, 0.0, 100.0]  # ex[9] — last row, alone in its chunk
    ground_z = 0.0

    got = _sommerfeld.max_image_distance(seg_l, seg_r, ground_z, chunk_rows=3)
    want = _naive_r1_max(seg_l, seg_r, ground_z)
    assert got == want
    # Guard the guard: the extreme pair really is row 9 pairing with
    # itself (hze = 200), not something the interior chunks would also see.
    assert want == pytest.approx(200.0 * 1.001)


@pytest.mark.parametrize("n_seg", [1, 2])
def test_max_image_distance_degenerate_clouds(n_seg):
    """(c) N=1 and N=2 degenerate segment clouds (2 and 4 endpoints)."""
    rng = np.random.default_rng(100 + n_seg)
    seg_l = rng.uniform(-5.0, 5.0, (n_seg, 3))
    seg_r = rng.uniform(-5.0, 5.0, (n_seg, 3))
    ground_z = -0.3
    got = _sommerfeld.max_image_distance(seg_l, seg_r, ground_z)
    want = _naive_r1_max(seg_l, seg_r, ground_z)
    assert got == want


@pytest.mark.memgate
def test_max_image_distance_no_n_squared_transient():
    """Tracemalloc gate: the helper's peak transient must stay a few MB
    regardless of endpoint count, never scaling with the retired
    `(2N, 2N)` full-matrix expression.

    Arithmetic for the threshold, at n_seg = 2500 (5000 endpoints, so
    N=2500 basis-scale — the issue's own N=4700 example peaked ~4 GB on
    the deleted expression):

      * default chunking picks `rows = max(1, 4_000_000 // (16 * 5000))
        = 50`, so each `(rows, n)` float64 buffer (dxe/dye/hze, their
        squares, the running sum, and the sqrt output) is capped at
        `50 * 5000 * 8 = 2.0 MB`.
      * measured peak (this test, tracemalloc): ~10.1 MB — a handful of
        those 2 MB buffers alive at once during the elementwise chain,
        stable across N (500 to 20,000 endpoints all measured 0.5-10.2
        MB; it does NOT grow with N, since chunk_rows shrinks to hold
        `rows * n` ~constant).
      * the retired full-matrix expression at this same N would need
        `8 * (2N)**2 = 8 * 5000**2 = 200 MB` for ONE real array, and the
        issue measured ~10-12x that (multiple materialized temporaries)
        at its N=4700 case — ~4 GB.

    25 MB sits ~2.5x above the measured peak and ~8x below the single
    undersized (2N, 2N) array this replaces, let alone the full ~2 GB+
    transient the old code built at this N.
    """
    import tracemalloc

    rng = np.random.default_rng(331)
    n_seg = 2500
    seg_l = rng.uniform(-30.0, 30.0, (n_seg, 3))
    seg_r = rng.uniform(-30.0, 30.0, (n_seg, 3))

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        r1_max = _sommerfeld.max_image_distance(seg_l, seg_r, 0.5)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert r1_max > 0.0
    assert peak < 25_000_000, (
        f"max_image_distance peaked {peak / 1e6:.2f} MB at "
        f"{2 * n_seg} endpoints — an N-squared transient is back "
        f"(8 * (2N)**2 = {8 * (2 * n_seg) ** 2 / 1e6:.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Observer banding of the fused remainder kernel (issue #343). `Jf`, the
# kernel's internal (d+1, d+1, nsI, nsJ) moment tensor, used to be allocated
# whole: 144 N^2 bytes at degree 2, ~9x the dense Z block it feeds, ~10 GB at
# the 8,320-basis straight wire. It is now filled and consumed one observer
# BAND at a time, with the band height set from a byte budget.
# ---------------------------------------------------------------------------


def _remainder_inputs(name="dipole", frac=0.2, **overrides):
    """(solver, geom, supp_seg, polys, eps_t) for a direct
    `_Z_sommerfeld_remainder` call."""
    s = _solver(name, frac, **SOMM, **overrides)
    geom = s._build_geometry()
    supp_seg, polys, *_rest = s._build_basis_polynomials(geom)
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    return s, geom, supp_seg, polys, eps_t


class _BandBudget:
    """`_acc` proxy pinning the fused kernel's `max_jf_bytes`."""

    def __init__(self, real, nbytes):
        self._real = real
        self._nbytes = nbytes

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if name != "sommerfeld_remainder_bspline_Q":
            return attr

        def wrapped(*args, **kwargs):
            kwargs["max_jf_bytes"] = self._nbytes
            return attr(*args, **kwargs)

        return wrapped


# 1 byte forces band == 1: every basis's (d+1)-segment support straddles a
# band boundary, so the straddle handling is exercised on EVERY row. The
# larger budgets walk the band height up through partial straddling.
@pytest.mark.parametrize("budget", [1, 4096, 1 << 16, 1 << 20])
def test_fused_Q_banding_is_exact(budget, monkeypatch):
    """Banding the observer axis must not move Q at all.

    Q[m,n] is a plain sum over the (a, b) wing pairs of the two bases, and
    a band partitions ONLY the observer wing axis `a` — each wing lives in
    exactly one band. Each band seeds its accumulator from the running
    Q[m,n] and adds its in-band pairs in increasing (a, b), so with the
    non-decreasing support maps this solver builds the summation ORDER is
    the unbanded one too: the result is bit-identical, not merely close.
    The gate is stated at 1e-12 relative (house reassociation tolerance)
    but bit-equality is asserted as well, since that is what is measured.
    """
    import momwire.bspline as bs

    if bs._acc is None or not hasattr(bs._acc, "sommerfeld_remainder_bspline_Q"):
        pytest.skip("fused sommerfeld kernel unavailable")

    s, geom, supp_seg, polys, eps_t = _remainder_inputs()
    real = bs._acc

    # Reference: one band over all observer segments == the pre-#343 kernel.
    monkeypatch.setattr(bs, "_acc", _BandBudget(real, 1 << 40))
    q_ref = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)

    monkeypatch.setattr(bs, "_acc", _BandBudget(real, budget))
    q_banded = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)

    rel = np.abs(q_banded - q_ref).max() / np.abs(q_ref).max()
    assert rel <= 1e-12, f"banded Q drifted {rel:.3e} at max_jf_bytes={budget}"
    assert np.array_equal(q_banded, q_ref), (
        f"banding at max_jf_bytes={budget} is no longer bit-identical "
        "(order-preserving accumulation broken?)"
    )


# The remainder's transient lives in a C++ `std::vector`, which tracemalloc
# CANNOT see (that is the #343 lesson — the N-squared gates above all use
# tracemalloc and would have stayed green through this bug). The gate must
# therefore read real process memory, so it runs the solve in a CHILD and
# reads that child's /proc VmHWM.
_REMAINDER_RSS_CHILD = r"""
import gc, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from momwire import BSplineSolver, _ground_refl, _sommerfeld

n_seg = int(sys.argv[2])


def vm(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) / 1024.0


s = BSplineSolver(
    wires=[[[0.0, -5.0, 2.2], [0.0, 5.0, 2.2]]],
    n_per_edge_per_wire=[[n_seg]],
    feeds=[(0, 5.0, 1 + 0j)],
    wavelength=20.0,
    wire_radius=0.0005,
    ground_z=0.0,
    ground_eps=(10.0, 0.002),
    ground_model="sommerfeld",
)
geom = s._build_geometry()
supp_seg, polys, *_rest = s._build_basis_polynomials(geom)
eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
# Build the interpolation grid in an EARLIER phase so its footprint is part
# of the baseline and not attributed to the remainder assembly.
s._somm_grid(
    eps_t, _sommerfeld.max_image_distance(geom["seg_l"], geom["seg_r"], s.ground_z)
)
gc.collect()
rss0 = vm("VmRSS:")
Q = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)
print(vm("VmHWM:") - rss0, polys.shape[0], geom["seg_l"].shape[0])
"""


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="reads the child's /proc VmHWM — linux memory plumbing, no "
    "/proc on macOS (PR #529); the banding it gates is platform-free",
)
def test_sommerfeld_remainder_transient_is_slab_bounded():
    """The remainder phase must stay inside the kernel's band-slab bound.

    Straight wire, 1,200 segments / 1,200 bases, degree 2 (d+1 = 3),
    sommerfeld ground. Arithmetic:

      * pre-#343 Jf = 16 * (d+1)^2 * N^2 = 144 * 1200^2 = 198 MiB, plus the
        23 MiB Q it assembles into -> a ~220 MiB remainder phase (measured
        878 MiB at N = 2400, where the formula predicts 879).
      * banded: the slab is capped at MAX_JF_SLAB_BYTES = 64 MiB (band =
        64 MiB / (16 * 9 * N) = 194 observer segments here), so the phase is
        64 + 23 = 87 MiB plus interpreter noise. Measured: 88 MiB.

    The threshold is the 87 MiB bound + 25 MiB headroom = 112 MiB, which
    sits well under the 220 MiB the unbanded kernel needs, so the gate
    genuinely discriminates. N is kept at 1,200 to hold the child under
    ~1.5 s wall.

    tracemalloc is deliberately NOT used: the tensor is a C++ std::vector
    and is invisible to it. This runs a child process and reads its
    /proc VmHWM.
    """
    import subprocess
    from pathlib import Path

    import momwire

    src_root = str(Path(momwire.__file__).resolve().parent.parent)
    n_seg = 1200
    out = subprocess.run(
        [sys.executable, "-c", _REMAINDER_RSS_CHILD, src_root, str(n_seg)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    phase_mb, n_basis, n_segs = out.stdout.split()
    phase_mb, n_basis, n_segs = float(phase_mb), int(n_basis), int(n_segs)

    d1 = 3
    slab_mb = 64.0  # MAX_JF_SLAB_BYTES in _accelerators.cpp
    q_mb = 16.0 * n_basis * n_basis / 2**20
    dense_mb = 16.0 * d1 * d1 * n_segs * n_segs / 2**20 + q_mb
    bound_mb = slab_mb + q_mb + 25.0

    assert bound_mb < 0.7 * dense_mb, (
        "gate no longer discriminates: bound "
        f"{bound_mb:.0f} MiB vs unbanded {dense_mb:.0f} MiB"
    )
    assert phase_mb < bound_mb, (
        f"sommerfeld remainder peaked {phase_mb:.0f} MiB at N = {n_segs} "
        f"(bound {bound_mb:.0f} MiB = {slab_mb:.0f} slab + {q_mb:.0f} Q + 25 "
        f"headroom) — the full (d+1)^2 N^2 moment tensor is back "
        f"({dense_mb:.0f} MiB)"
    )


# ---------------------------------------------------------------------------
# The grazing near image (momwire#631)
# ---------------------------------------------------------------------------
#
# bspline's grazing error was TWO under-resolved quadratures, and the cross
# that named it showed neither half is sufficient alone: on a 39.6 m radial at
# h/lambda = 1.09e-4, n = 16, over `GN 0` against the licensed binary, keying
# the image order alone left 152 %, the remainder order alone 306 %, and both
# 0.62 %. The first half needs no order at all — a horizontal edge's own image
# is the SAME arc translated by -2h, so its block is the same-edge kernel at
# a_eff = sqrt(a^2 + 4h^2), exactly. These gates are all reference-free; CI has
# no binary, and every external reference this arc produced is a whole-deck
# impedance.

_GRAZE_L = 39.624  # m, capture 0033's radial
_GRAZE_WL = 163.6422  # m, 1.832 MHz
_GRAZE_A = 0.001294  # m


def _grazing_wire(h, n_seg=16, **overrides):
    """One horizontal wire at height `h`, fed at its centre."""
    kw = dict(
        wires=[[[0.0, 0.0, h], [_GRAZE_L, 0.0, h]]],
        n_per_edge_per_wire=[[n_seg]],
        feeds=[(0, _GRAZE_L / 2.0, 1.0 + 0j)],
        junctions=None,
        wire_radius=_GRAZE_A,
        wavelength=_GRAZE_WL,
        ground_z=0.0,
    )
    kw.update(overrides)
    return kw


def test_the_near_image_block_is_the_same_edge_kernel_at_a_eff(monkeypatch):
    """The identity the whole fix rests on, checked against brute force.

    `J_static_moment` integrates 1/sqrt((s-s')^2 + a^2) and
    `_seg_seg_reg_geometry` builds R the same way, so in both `a` is nothing
    but the constant perpendicular offset between the two arcs. For a
    horizontal edge and its mirror that offset is exactly 2h. If that reading
    is right the closed form must reproduce a high-order off-edge evaluation
    of the same block; if it is wrong — a sign, a factor of two, the arc
    running backwards — this is where it shows.

    The brute-force reference needs order 256, which only the numpy fallback
    can do (the C++ kernel refuses n_qp > 8 for scratch-buffer size), so the
    accelerators are switched off for the reference alone.
    """
    from momwire import _bspline_kernels as bk

    h = 1.09e-4 * _GRAZE_WL
    s = BSplineSolver(**_grazing_wire(h))
    geom = s._build_geometry()
    blocks = s._near_image_edge_blocks(geom)
    assert len(blocks) == 1, f"expected one horizontal edge, got {len(blocks)}"
    sl, arc, a_eff = blocks[0]
    assert a_eff == pytest.approx(np.hypot(_GRAZE_A, 2.0 * h), rel=1e-12)

    proposed = s._near_image_analytic_block(arc, a_eff, s.k)

    monkeypatch.setattr(bk, "_HAVE_BSPLINE_ACCEL", False)
    monkeypatch.setattr(bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    seg_l, seg_r = geom["seg_l"], geom["seg_r"]
    truth = bk._seg_seg_full_moments_offedge(
        seg_l[sl],
        seg_r[sl],
        s._image_positions(seg_l[sl]),
        s._image_positions(seg_r[sl]),
        _GRAZE_A,
        s.k,
        s.degree,
        256,
    )
    rel = np.abs(proposed - truth).max() / np.abs(truth).max()
    assert rel < 1e-4, f"closed form is not the image block: {rel:.3e}"

    # ...and this is the thing the shipped order could NOT do: the same block
    # at n_qp_pair = 4 is off by O(1). Measured 1.6 here; gated at 0.5 because
    # the claim is "the default rule has lost the block", not that number.
    shipped = bk._seg_seg_full_moments_offedge(
        seg_l[sl],
        seg_r[sl],
        s._image_positions(seg_l[sl]),
        s._image_positions(seg_r[sl]),
        _GRAZE_A,
        s.k,
        s.degree,
        4,
    )
    lost = np.abs(shipped - truth).max() / np.abs(truth).max()
    assert lost > 0.5, f"off-edge order 4 only {lost:.3e} out — gate is inert"


def test_an_ordinary_deck_has_no_near_image_block():
    """The property that lets this ship without moving a shipped gate.

    Nothing is claimed unless a horizontal edge's image has come closer than
    half a segment, so an ordinary deck keeps exactly the arithmetic it had —
    both the near-image fixup and the remainder keying are no-ops on it.
    """
    s = BSplineSolver(**dict(GEOMS[("dipole", 0.2)], ground_z=0.0))
    geom = s._build_geometry()
    assert s._near_image_edge_blocks(geom) == []
    seg_l, seg_r = geom["seg_l"], geom["seg_r"]
    assert s._remainder_qp(seg_l, seg_r, 0.0) == s.n_qp_sommerfeld


def test_a_vertical_grazing_wire_is_not_claimed():
    """Scope guard: only HORIZONTAL edges reduce to this kernel.

    A vertical wire's image is collinear with it rather than parallel and
    offset, so `a_eff` would be meaningless there — and experiment 3 of the
    #510 arc measured that a lone grazing vertical is already exact, so
    claiming it could only do harm. A tilted edge is neither case.
    """
    h = 1.09e-4 * _GRAZE_WL
    for wire in (
        [[0.0, 0.0, h], [0.0, 0.0, h + 20.0]],  # vertical
        [[0.0, 0.0, h], [_GRAZE_L, 0.0, h + 5.0]],  # tilted
    ):
        s = BSplineSolver(**_grazing_wire(h, wires=[wire]))
        assert s._near_image_edge_blocks(s._build_geometry()) == [], wire


def test_a_grazing_wire_is_insensitive_to_the_pair_order():
    """The image half's physics gate, and it needs no binary.

    With the block computed in closed form the answer must stop caring about
    `n_qp_pair` — that knob no longer touches these pairs. Before the fix the
    same deck moved 195 % -> 80 % of |Z| between orders 4 and 8 against the
    binary; the two momwire answers differed by more than half of |Z|.
    """
    kw = _grazing_wire(1.09e-4 * _GRAZE_WL)
    z4, _ = BSplineSolver(**kw, n_qp_pair=4).compute_impedance()
    z8, _ = BSplineSolver(**kw, n_qp_pair=8).compute_impedance()
    rel = abs(z4 - z8) / abs(z8)
    assert rel < 1e-3, f"still order-sensitive over PEC: {rel:.3%}"


def test_a_grazing_wire_converges_under_mesh_refinement():
    """#631's own symptom, gated directly: the answer must SETTLE.

    The issue was filed on "bspline's finite-ground answer gets worse with
    refinement" — 13.97 -> 83.60 % over N = 5..25 against the binary. CI has
    no binary, but divergence is visible without one: successive refinements
    must move the answer less, not more.
    """
    somm = dict(ground_eps=(13.0, 0.005), ground_model="sommerfeld")
    zs = [
        BSplineSolver(
            **_grazing_wire(1.09e-4 * _GRAZE_WL, n_seg=n), **somm
        ).compute_impedance()[0]
        for n in (8, 16, 32)
    ]
    step1 = abs(zs[1] - zs[0]) / abs(zs[1])
    step2 = abs(zs[2] - zs[1]) / abs(zs[2])
    assert step2 < step1, (
        f"not converging: {step1:.3%} then {step2:.3%} ({zs[0]:.3f}, "
        f"{zs[1]:.3f}, {zs[2]:.3f})"
    )
    assert step2 < 0.05, f"refinement step still {step2:.3%} of |Z|"


def test_the_grazing_remainder_order_is_keyed_and_matters():
    """The remainder half, the same self-consistency razor's #510 gate uses.

    If the keying picks a sufficient order, forcing the floor to the cap by
    hand must not move the answer; and capped back to the pre-#631 default the
    same deck must be a different answer entirely, or the keying is inert.
    The cap is read at CALL time precisely so this gate can move it — as a
    default argument it would bind at import and this would silently compare
    two identical solves.
    """
    from momwire import bspline as _bs

    somm = dict(ground_eps=(13.0, 0.005), ground_model="sommerfeld")
    kw = _grazing_wire(1.09e-4 * _GRAZE_WL, **somm)
    z_keyed, _ = BSplineSolver(**kw).compute_impedance()
    z_forced, _ = BSplineSolver(**kw, n_qp_sommerfeld=192).compute_impedance()
    rel = abs(z_keyed - z_forced) / abs(z_forced)
    assert rel < 1e-2, f"grazing remainder not converged: {rel:.3%}"

    saved = _bs._REMAINDER_QP_CAP
    try:
        _bs._REMAINDER_QP_CAP = 3
        z_flat, _ = BSplineSolver(**kw).compute_impedance()
    finally:
        _bs._REMAINDER_QP_CAP = saved
    moved = abs(z_flat - z_forced) / abs(z_forced)
    assert moved > 0.25, f"order-3 answer only {moved:.1%} away — keying inert?"


def test_the_fast_solvers_do_not_have_the_near_image_fix_yet():
    """The scope line of momwire#631, written down so it cannot rot.

    `HMatrixSolver` and `ArrayBlockSolver` reach the image by their own
    routes — `_zblock_image_refl` for near blocks and a fused ACA twin for far
    ones — neither of which is the dense/chunked/swept fill the near-image
    correction lives in. They were already wrong on a grazing deck before
    #631; what changed is that `BSplineSolver` is now right, so the two
    disagree instead of agreeing wrongly.

    `test_fast_solvers_match_dense` keeps them honest on an ordinary deck at
    ACA/GMRES tolerance, and that is where they are supported. This gate is
    the other half of that sentence: at grazing they are NOT, by a margin
    nobody could mistake for tolerance (measured 194 % over PEC and 407 % over
    `GN 0`), and if it ever closes the fixer should come back and turn this
    into a real agreement pin — the same bargain the eznec seam's grazing
    decks were held to.
    """
    kw = _grazing_wire(1.09e-4 * _GRAZE_WL)
    z_dense, _ = BSplineSolver(**kw).compute_impedance()
    for cls in (HMatrixSolver, ArrayBlockSolver):
        z_fast, _ = cls(**kw).compute_impedance()
        rel = abs(z_fast - z_dense) / abs(z_dense)
        assert rel > 0.5, (
            f"{cls.__name__} now agrees with the dense route at grazing "
            f"({rel:.2%}) — momwire#631's near-image correction has reached "
            f"it, so pin the agreement instead of this divergence"
        )
