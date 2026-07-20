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


def test_wires_below_ground_rejected():
    kw = dict(GEOMS[("dipole", 0.2)])
    z_top = max(p[2] for wire in kw["wires"] for p in wire)
    s = BSplineSolver(**kw, ground_z=z_top + 1.0, **SOMM)
    # Rejected at geometry build since #151 (every ground model).
    with pytest.raises(ValueError, match="below the ground plane"):
        s.compute_impedance()


def test_default_model_is_refl_coef():
    """ground_model defaults to refl-coef: explicit and default solves are
    bit-identical, so v0.5.0 behavior is unchanged."""
    z_default = _solve("dipole", 0.2, ground_eps=(10.0, 0.002))
    z_explicit = _solve(
        "dipole", 0.2, ground_eps=(10.0, 0.002), ground_model="refl-coef"
    )
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
