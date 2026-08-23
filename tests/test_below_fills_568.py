"""The below/below fills on the C++ contour engine (momwire#568 unit 2).

U1 landed the engine and gated it against scipy, against the Sommerfeld
identity's closed form, and against the numpy engine on a SYNTHETIC integrand.
U2 puts the real one on it: `_integrand_six_below`, the `_six_integrals_below`
driver with its coarse self-convergence twin, the per-point loop of
`iv_surfaces_direct_below` (this is where the OpenMP region lives), and the
projected table `remainder_field_proj_below`.

The numpy spellings stay, unchanged, as the references. Every gate here runs
BOTH machines in one process — `below._FORCE_NUMPY` flips the dispatch — and
compares them componentwise and relatively.

What this file gates, and why each gate exists:

* **Value parity** on the six integrals, the four surfaces and the projected
  table, over the SPEC cells and off-SPEC stressors. This is the whole
  correctness claim: the shipped ladders in `test_sommerfeld_below.py` are
  measured against oracles and are ~4 decades looser than the C++/numpy spread,
  so they would not notice a defect this file is sized to catch.
* **The paths that are NOT quadrature.** The ε̃ = 1 short circuit must return
  EXACT zeros and bump the same health counter on both machines; the (ρ, h)
  domain raise must raise with the same words; the R₁ = 0 rows must stay on the
  analytic limit path (they are closed form and cheap, and no contour is run
  for them at all).
* **Health parity.** `gu2_4` asserts `health.nonconvergent == 0`, so the
  counters are contract, not decoration — including `worst_selfconv_at`, which
  has to land on the same node.
* **The index-2 sign**, through the C++ integrand, because that one line IS the
  ±=− story (see the mutation note above `test_g5686_*`).
* **The ±=+ family keeps its bytes.** U2 added a projection TWIN rather than
  widening `remainder_field_proj_batch`; the gate that this was really a twin
  is that the ±=+ kernel still reproduces its own numpy body.

TOLERANCES, NEVER BYTE EQUALITY. The two machines take the same adaptive
decisions but are not bit-twins: the Gauss dot product reduces in a different
order, complex division is libgcc's rather than numpy's, and the Bessel pair is
a different algorithm entirely (U1's G-568-1).
"""

import numpy as np
import pytest

from momwire import _sommerfeld as som
from momwire import _sommerfeld_below as below

import golden_below_below_524 as gold

accel = pytest.importorskip("momwire._accelerators")

if not getattr(accel, "below_fills_568", False):
    pytest.skip(
        "the accelerator predates momwire#568 unit 2 (stale .so?)",
        allow_module_level=True,
    )

EPS0 = 8.8541878128e-12
C0 = 299792458.0
MU0 = som._MU0
KEYS = som._SURF_KEYS
GROUND_Z = 0.0


def eps_tilde(eps_r, sigma, f_hz):
    """ε̃ = ε_r − jσ/(ωε₀), as `test_sommerfeld_below` writes it."""
    return eps_r - 1j * sigma / (2.0 * np.pi * f_hz * EPS0)


def medium(cell):
    """(ε̃, k_p, ω, k_m, λ_m) for a golden cell key like "A/7MHz"."""
    sname, fname = cell.split("/")
    er, sg = gold.SOILS[sname]
    f = float(fname[:-3]) * 1e6
    om = 2.0 * np.pi * f
    et = eps_tilde(er, sg, f)
    kp = om / C0
    km = below.k_medium(et, kp)
    return et, kp, om, km, 2.0 * np.pi / abs(km)


SPEC_CELLS = sorted(gold.SURFACES)


class force_numpy:
    """Run a block on the numpy reference, restoring the dispatch after.

    A context manager rather than a fixture because every gate here needs BOTH
    machines inside ONE test — a fixture that pinned the module flag for a
    whole test could only ever measure one of them.
    """

    def __enter__(self):
        self._was = below._FORCE_NUMPY
        below._FORCE_NUMPY = True
        assert not below._use_below_accel()

    def __exit__(self, *exc):
        below._FORCE_NUMPY = self._was
        return False


@pytest.fixture(autouse=True)
def _accel_on():
    """Every test starts on the accelerated path, whatever the last one did."""
    was = below._FORCE_NUMPY
    below._FORCE_NUMPY = False
    assert below._use_below_accel(), "the dispatch must be live for these gates"
    yield
    below._FORCE_NUMPY = was


def cwise(got, ref):
    """Worst COMPONENTWISE relative difference. Not of-family: below the
    interface the six integrands span decades at one (ρ, h), and grading a
    small component against its largest neighbour would hide a factor of ten
    in it."""
    got = np.asarray(got)
    ref = np.asarray(ref)
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))


# ======================================================================
# G-568-6 — the six integrals, C++ against numpy
# ======================================================================
#
# MEASURED worst componentwise relative, per SPEC cell, over the geometry list
# below at rtol 1e-9:
#
#   A/7  9.98e-13   A/21 2.64e-13   B/7  2.79e-12
#   B/21 8.48e-13   C/7  4.41e-13   C/21 1.19e-13
#
# and over the off-SPEC stressors: sea water 2.21e-13, near-lossless rock
# 5.83e-14, ε̃ ≈ 1 (where D₁/D₂ are differences of nearly equal numbers)
# 2.62e-12, 160 m 1.20e-12. Grand worst 2.79e-12.
#
# Pinned at 1e-9 — ~360x over the realized worst, and the same pin U1's G-568-3
# carries. It is deliberately NOT set at the measurement: these last digits
# move by a factor of a few between builds (GCC decides where the FMAs land),
# and a cross-build parity pin must never be the thing someone re-tolerances in
# a hurry. The gates it protects sit far above it anyway — G-U2-1 at 1e-7
# against an independent oracle, the grid ladders at 1e-3/2e-3.

G_568_6_TOL = 1e-9

# (ρ/λ_m, h/λ_m). The last two are the product geometry: deep grazing at
# working range, where the remainder is many times direct+image and where the
# tail runs longest.
PARITY_GEOMS = (
    (0.3, 0.2),
    (1.0, 0.05),
    (2.0, 0.5),
    (0.02, 0.01),
    (0.001, 0.5),
    (1.5, 0.017),
    (1.9, 0.0035),
)


@pytest.mark.parametrize("cell", SPEC_CELLS)
def test_g5686_the_six_integrals_agree_over_the_spec_matrix(cell, record_property):
    et, kp, om, km, lam_m = medium(cell)
    worst = 0.0
    for rho_wl, h_wl in PARITY_GEOMS:
        rho, h = rho_wl * lam_m, h_wl * lam_m
        got = below._six_integrals_below(et, kp, rho, h, 1e-9)
        with force_numpy():
            ref = below._six_integrals_below(et, kp, rho, h, 1e-9)
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_6_TOL, f"{cell}: {worst:.3e}"


@pytest.mark.parametrize(
    "eps_r,sigma,f_hz",
    [
        (81.0, 5.0, 7.0e6),  # sea water: |k_m| ~ 95 k_p, the tail's stressor
        (3.0, 1e-5, 21.0e6),  # near-lossless dry rock
        (1.05, 1e-6, 14.0e6),  # ε̃ just off 1: the D-pair nearly cancels
        (13.0, 0.005, 1.8e6),  # 160 m, well off the SPEC frequencies
    ],
)
def test_g5686_the_six_integrals_agree_off_spec(eps_r, sigma, f_hz, record_property):
    """Off the SPEC matrix on purpose. Sea water pushes |k_m| to ~95 k_p, which
    is where the head's `a` and the tail's e-folding cap part company; ε̃ ≈ 1 is
    where D₁/D₂ are differences of nearly equal numbers and a port that
    regrouped them would show up first."""
    et = eps_tilde(eps_r, sigma, f_hz)
    kp = 2.0 * np.pi * f_hz / C0
    lam_m = 2.0 * np.pi / abs(below.k_medium(et, kp))
    worst = 0.0
    for rho_wl, h_wl in PARITY_GEOMS:
        rho, h = rho_wl * lam_m, h_wl * lam_m
        got = below._six_integrals_below(et, kp, rho, h, 1e-9)
        with force_numpy():
            ref = below._six_integrals_below(et, kp, rho, h, 1e-9)
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_6_TOL, f"({eps_r}, {sigma}, {f_hz:g}): {worst:.3e}"


def test_g5686_the_coarse_self_convergence_machine_is_the_same_call(record_property):
    """`selfconv=True` re-runs the whole contour on Gauss-16, rtol ×100 and a
    shallower detour. Both rules, both depths and both detours are ARGUMENTS to
    the C++ entry point for exactly this reason: the fine machine and the coarse
    one are one call, not two ports. The spread itself is compared, not just the
    value — it is what `Health.worst_selfconv` reports.

    MEASURED: values 2.79e-12, the two machines' self-convergence ESTIMATES
    agreeing to 2.47e-4 relative. The estimate is a fine-vs-coarse DIFFERENCE,
    so it inherits the coarse machine's own noise divided by the ~4e-9 spread
    it is estimating; a few parts in ten thousand is what two independent
    evaluations of that difference should agree to, and pinning at the
    measurement would be pinning that noise. 1e-2 is the bar, and it still
    catches the failure that matters — a coarse machine wired to the wrong
    rule, depth or detour moves the estimate by decades.
    """
    et, kp, om, km, lam_m = medium("B/7MHz")
    worst_val = worst_spread = 0.0
    for rho_wl, h_wl in PARITY_GEOMS:
        rho, h = rho_wl * lam_m, h_wl * lam_m
        hc = below.Health()
        got = below._six_integrals_below(et, kp, rho, h, 1e-9, health=hc, selfconv=True)
        with force_numpy():
            hn = below.Health()
            ref = below._six_integrals_below(
                et, kp, rho, h, 1e-9, health=hn, selfconv=True
            )
        worst_val = max(worst_val, cwise(got, ref))
        # The spread is an ESTIMATE five decades above the realized error, so
        # the two machines' estimates need only agree relatively, not exactly.
        worst_spread = max(
            worst_spread,
            abs(hc.worst_selfconv - hn.worst_selfconv) / max(hn.worst_selfconv, 1e-300),
        )
        assert hc.worst_selfconv_at == hn.worst_selfconv_at
    record_property("worst_rel", float(worst_val))
    record_property("worst_selfconv_spread_rel", float(worst_spread))
    assert worst_val < G_568_6_TOL, f"value: {worst_val:.3e}"
    assert worst_spread < 1e-2, f"self-convergence estimate: {worst_spread:.3e}"


# ---------------------------------------------------------------------
# MUTATION PAIR — the index-2 sign flip in `mw568_below::SixBelow`.
#
# `out[2] = -(common * g_m * (b1x * x) * l2)`. The leading minus is the ±=−
# family: h = |z + z′| with z + z′ < 0 below the interface, so
# ∂/∂z e^{−γ_m|z+z′|} = +γ_m·e where the ±=+ case gets −γ₂·e, and with
# ∂J₀/∂ρ = −λJ₁ the product lands at the NEGATIVE of `_sommerfeld.
# _integrand_six`'s index 2. Index 2 alone carries a single z-derivative;
# nothing else in the stack changes under the swap.
#
# VERIFIED BY HAND ONCE (2026-08-23, this box). Dropping that minus sign —
# `out[2] = (common * g_m * (b1x * x) * l2)` — and rebuilding turns 20 of this
# file's 37 gates red, and with them the shipped ladder:
#
#   test_g5686_the_index_2_sign_is_load_bearing   both cells, the direct
#                                                 assertion below
#   test_g5686_the_six_integrals_agree_*          all ten, worst rel exactly
#                                                 2.0 — index 2 IS its own
#                                                 negative
#   test_g5688_the_four_surfaces_agree            all six cells, plus the
#                                                 shape/dtype gate
#   test_gu2_1_surfaces_vs_the_phase0_prototype   RED on every SPEC cell at
#     [tests/test_sommerfeld_below.py]            worst rel 4.5e-1 to 6.0e-1
#                                                 of scale (that gate's metric
#                                                 is of-family, hence not 2.0)
#                                                 against the COMMITTED phase-0
#                                                 goldens — I_ρ^V is
#                                                 C₁(k_p²/k_m²)P·∂²V/∂ρ∂z and
#                                                 nothing else, so it inverts
#
# The last line is the one that matters: an O(1) failure against an INDEPENDENT
# oracle, not merely against the numpy twin. That is what makes the flip
# load-bearing physics rather than a shared convention.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cell", ["A/7MHz", "B/21MHz"])
def test_g5686_the_index_2_sign_is_load_bearing(cell, record_property):
    """The C++ integrand, read against the ±=+ stack built on the SAME swapped
    kernel pair.

    `_sommerfeld._integrand_six(λ, ρ, h, k_p, k_m, "J")` calls
    `_d12(λ, k_p, k_m)` — the identical D-pair — and differs from
    `_integrand_six_below` in exactly one place: the sign of index 2. So
    running the C++ engine's own contour against a numpy contour over that
    stack must reproduce five components and INVERT the sixth. A port that
    dropped the minus would agree on all six, and one that inverted the wrong
    component would disagree on two.
    """
    et, kp, om, km, lam_m = medium(cell)
    rho, h = 1.0 * lam_m, 0.05 * lam_m

    def f_plus(lam):
        return som._integrand_six(lam, rho, h, kp, km, "J")

    ref, _hp, _tp, _conv, _acc = below._run_contour(
        f_plus,
        kp,
        km,
        rho,
        h,
        1e-11,
        below._ADAPT_DEPTH,
        below._DETOUR,
        below._GX,
        below._GW,
    )
    got = below._six_integrals_below(et, kp, rho, h, 1e-11)

    flipped = np.array(ref, dtype=np.complex128)
    flipped[2] = -flipped[2]
    worst = cwise(got, flipped)
    record_property("worst_rel_against_the_flipped_stack", float(worst))
    assert worst < G_568_6_TOL, f"{cell}: {worst:.3e}"
    # ... and index 2 is not a numerical zero, so "agrees with its own
    # negative" cannot be how this passes.
    rel2 = abs(got[2] - ref[2]) / max(abs(ref[2]), 1e-300)
    record_property("index2_vs_unflipped", float(rel2))
    assert abs(ref[2]) > 1e-6 * float(np.max(np.abs(ref)))
    assert rel2 > 1.0, f"index 2 did not flip: {rel2:.3e}"


# ======================================================================
# G-568-7 — the paths that are not quadrature
# ======================================================================


def test_g5687_eps_one_short_circuits_identically_on_both_paths():
    """ε̃ = 1 makes D₁ = D₂ = 0 identically, so the answer is EXACTLY zero and
    no contour is run at all. Phase 0 measured what happens without the short
    circuit (1164 s instead of 0.4 s, a relative tail test chasing ulp noise),
    so this is not a micro-optimization — and the C++ path must not quietly
    acquire its own version of it. The counter bump is per NODE, which is what
    the per-point numpy spelling gives."""
    k2 = 2.0 * np.pi
    for use_numpy in (True, False):
        ctx = force_numpy() if use_numpy else None
        if ctx:
            ctx.__enter__()
        try:
            h1 = below.Health()
            six = below._six_integrals_below(1.0, k2, 0.3, 0.2, health=h1)
            assert np.count_nonzero(six) == 0
            assert six.dtype == np.complex128 and six.shape == (6,)
            assert h1.zero_kernel == 1 and h1.evaluations == 0

            h2 = below.Health()
            surf = below.iv_surfaces_direct_below(
                1.0,
                k2,
                np.array([0.0, 0.3, 1.0]),
                np.radians([20.0, 45.0, 70.0]),
                health=h2,
            )
            for k in KEYS:
                assert np.count_nonzero(surf[k][1:]) == 0
            # Two nonzero-R1 nodes, so two bumps — the R1 = 0 row never
            # reaches the contour driver at all.
            assert h2.zero_kernel == 2, h2.as_dict()
        finally:
            if ctx:
                ctx.__exit__()


@pytest.mark.parametrize("rho,h", [(-1.0, 0.2), (0.3, -0.2), (0.0, 0.0)])
def test_g5687_the_domain_raise_says_the_same_thing_on_both_paths(rho, h):
    """(ρ, h) ≥ 0 and R₁ > 0, refused by name. The C++ entry point does not
    carry this check: it is exact, it is cheap, and its words are contract."""
    msgs = []
    for use_numpy in (True, False):
        ctx = force_numpy() if use_numpy else None
        if ctx:
            ctx.__enter__()
        try:
            with pytest.raises(ValueError, match="need rho, h >= 0 and R1 > 0") as e:
                below._six_integrals_below(complex(13.0, -12.8), 2.0 * np.pi, rho, h)
            msgs.append(str(e.value))
        finally:
            if ctx:
                ctx.__exit__()
    assert msgs[0] == msgs[1], msgs


def test_g5687_the_r1_zero_rows_stay_on_the_analytic_limit():
    """R₁ = 0 is a closed form (`_limits_r1_zero_below`), not a contour, and it
    stays numpy on both paths — the brief's instruction and the cheaper answer.
    Gated as an EXACT match against the limit function so a future "just send
    everything to C++" cannot pass."""
    et, kp, om, km, lam_m = medium("A/7MHz")
    th = np.radians(np.array([1.0, 20.0, 55.0, 90.0]))
    r1 = np.zeros_like(th)
    lim = below._limits_r1_zero_below(et, kp, th, om, MU0)
    for use_numpy in (True, False):
        ctx = force_numpy() if use_numpy else None
        if ctx:
            ctx.__enter__()
        try:
            hh = below.Health()
            got = below.iv_surfaces_direct_below(
                et, kp, r1, th, rtol=1e-9, omega=om, health=hh
            )
            for k in KEYS:
                assert np.array_equal(got[k], lim[k]), k
            assert hh.evaluations == 0, hh.as_dict()
        finally:
            if ctx:
                ctx.__exit__()


# ======================================================================
# G-568-8 — the four surfaces and the health counters
# ======================================================================
#
# MEASURED over the mixed R1 = 0 / R1 > 0 lattice below, all six SPEC cells:
# worst per-surface relative 2.16e-14 (B/7; the other five read 4.3e-15 to
# 1.0e-14), and on the health side every single counter IDENTICAL —
# evaluations 18, nonconvergent 0, accelerated 0, zero_kernel 0, max_tail_panels
# (98/103/107) and max_head_panels (15/16) equal cell for cell,
# `worst_selfconv_at` the same node, `worst_selfconv` agreeing to 4e-6 relative
# or better.
#
# The surfaces are pinned at 1e-9 (as G-568-6). The panel counts get a band
# rather than equality even though the measurement is exact — they are the
# outcome of floating-point comparisons that differ in the last bits, and a
# last-bit tie-break on another box must not turn this red. U1's G-568-4
# carries the same 20 %.

G_568_8_TOL = 1e-9
G_568_8_PANEL_BAND = 0.20


@pytest.mark.parametrize("cell", SPEC_CELLS)
def test_g5688_the_four_surfaces_agree(cell, record_property):
    """`iv_surfaces_direct_below` end to end — the per-point loop that carries
    the OpenMP region, plus the R₁ = 0 analytic row mixed into the same call so
    the two branches' bookkeeping cannot drift."""
    et, kp, om, km, lam_m = medium(cell)
    rng = np.random.default_rng(568)
    r1 = np.concatenate([[0.0], rng.uniform(0.01, 2.0, 18) * lam_m])
    th = np.concatenate([[np.radians(45.0)], np.radians(rng.uniform(1.0, 90.0, 18))])

    hc = below.Health()
    got = below.iv_surfaces_direct_below(
        et, kp, r1, th, rtol=1e-9, omega=om, health=hc, selfconv=True
    )
    with force_numpy():
        hn = below.Health()
        ref = below.iv_surfaces_direct_below(
            et, kp, r1, th, rtol=1e-9, omega=om, health=hn, selfconv=True
        )

    worst = max(cwise(got[k], ref[k]) for k in KEYS)
    record_property("worst_rel", float(worst))
    record_property("health_cpp", repr(hc.as_dict()))
    record_property("health_numpy", repr(hn.as_dict()))
    assert worst < G_568_8_TOL, f"{cell}: {worst:.3e}"

    assert hc.evaluations == hn.evaluations == 18
    assert hc.nonconvergent == hn.nonconvergent == 0
    assert hc.accelerated == hn.accelerated
    assert hc.zero_kernel == hn.zero_kernel == 0
    assert hc.worst_selfconv_at == hn.worst_selfconv_at
    for a, b, what in (
        (hc.max_tail_panels, hn.max_tail_panels, "tail"),
        (hc.max_head_panels, hn.max_head_panels, "head"),
    ):
        assert abs(a - b) <= max(1, G_568_8_PANEL_BAND * b), f"{what}: {a} vs {b}"


def test_g5688_the_shapes_and_dtypes_survive_the_dispatch():
    """Broadcasting is part of the contract: `iv_surfaces_direct_below` returns
    arrays shaped like the broadcast (R₁, θ), and a batch entry point is exactly
    where a ravel can leak out."""
    et, kp, om, km, lam_m = medium("C/7MHz")
    r1 = np.array([[0.2], [0.7]]) * lam_m
    th = np.radians(np.array([10.0, 50.0, 80.0]))
    got = below.iv_surfaces_direct_below(et, kp, r1, th, rtol=1e-9, omega=om)
    with force_numpy():
        ref = below.iv_surfaces_direct_below(et, kp, r1, th, rtol=1e-9, omega=om)
    for k in KEYS:
        assert got[k].shape == (2, 3) == ref[k].shape
        assert got[k].dtype == np.complex128
    assert max(cwise(got[k], ref[k]) for k in KEYS) < G_568_8_TOL


# ======================================================================
# G-568-9 — the projected table
# ======================================================================
#
# MEASURED worst relative over the pair sets below, on a real (soil A, 7 MHz)
# tabulation at the full 2 lambda_m cap: 6.1e-15, and 6.2e-16 on the
# coincident-pair branch. The projection is a fixed 4x4 stencil and a handful
# of multiplies, so the two spellings differ only in reduction order and in
# `std::exp` versus `np.exp`; there is no adaptive decision here to diverge.
#
# Pinned at 1e-11 — ~1600x over the realized worst, and still eight decades
# under the grid's own 2e-3 interpolation bar, so this gate can only ever fire
# on a real port defect and never on the tabulation's accuracy.

G_568_9_TOL = 1e-11


@pytest.fixture(scope="module")
def proj_grid():
    """A real tabulation at the full cap on the reference soil. Module-scoped:
    the fill is the expensive part of this file and every projection gate wants
    the same one."""
    et, kp, om, km, lam_m = medium("A/7MHz")
    grid = below.SommerfeldGridBelow(
        et, kp, below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m, rtol=1e-9, omega=om
    )
    return grid, (et, kp, om, km, lam_m)


def _pair_set(seed, m, s, depth_lo=0.1, depth_hi=1.2, span=4.0):
    rng = np.random.default_rng(seed)
    obs = np.c_[
        rng.uniform(-span, span, m),
        rng.uniform(-span, span, m),
        -rng.uniform(depth_lo, depth_hi, m),
    ]
    src = np.c_[
        rng.uniform(-span, span, s),
        rng.uniform(-span, span, s),
        -rng.uniform(depth_lo, depth_hi, s),
    ]
    t_obs = rng.normal(size=(m, 3))
    t_src = rng.normal(size=(s, 3))
    # A purely vertical source tangent (th_src == 0) and a purely horizontal
    # one: the two ends of the tangent decomposition, including the ux/uy
    # fallback the rho -> 0 branch reads.
    t_src[0] = [0.0, 0.0, 1.0]
    t_src[-1] = [0.6, 0.8, 0.0]
    return obs, t_obs, src, t_src


@pytest.mark.parametrize("seed,m,s", [(1, 9, 7), (2, 3, 11), (3, 12, 1)])
def test_g5689_the_projected_table_agrees(proj_grid, seed, m, s, record_property):
    grid, (et, kp, om, km, lam_m) = proj_grid
    obs, t_obs, src, t_src = _pair_set(seed, m, s)
    got = below.remainder_field_proj_below(
        obs, t_obs, src, t_src, GROUND_Z, kp, km, grid
    )
    with force_numpy():
        ref = below.remainder_field_proj_below(
            obs, t_obs, src, t_src, GROUND_Z, kp, km, grid
        )
    assert got.shape == (m, s) == ref.shape
    worst = cwise(got, ref)
    record_property("worst_rel", float(worst))
    assert worst < G_568_9_TOL, f"seed {seed}: {worst:.3e}"


def test_g5689_the_coincident_horizontal_pair_takes_the_same_branch(
    proj_grid, record_property
):
    """ρ → 0 degenerates the incidence azimuth and both spellings fall back to
    the SOURCE horizontal direction. Two source points stacked directly under
    the observer, one vertical tangent and one horizontal, so the fallback is
    read on both."""
    grid, (et, kp, om, km, lam_m) = proj_grid
    obs = np.array([[0.0, 0.0, -0.4], [0.0, 0.0, -0.9]])
    src = np.array([[0.0, 0.0, -0.2], [0.0, 0.0, -0.7]])
    t_obs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t_src = np.array([[0.0, 0.0, 1.0], [0.6, 0.8, 0.0]])
    got = below.remainder_field_proj_below(
        obs, t_obs, src, t_src, GROUND_Z, kp, km, grid
    )
    with force_numpy():
        ref = below.remainder_field_proj_below(
            obs, t_obs, src, t_src, GROUND_Z, kp, km, grid
        )
    worst = cwise(got, ref)
    record_property("worst_rel", float(worst))
    assert worst < G_568_9_TOL, f"{worst:.3e}"


def test_g5689_a_duck_typed_surface_source_stays_on_numpy(proj_grid, monkeypatch):
    """The twin flattens a real tabulation's region tables. Anything else that
    can `eval(R1, theta)` — the suite's direct-surface stand-in — has no
    `_regions`, takes the numpy body, and must keep doing so: the three-way
    grid → direct → goldens gates rest on the direct side NOT sharing the
    grid side's interpolation code."""

    class DirectSurfaces:
        regime = "below"

        def __init__(self, et, kp, om, r1_max):
            self.et, self.kp, self.om, self.r1_max = et, kp, om, r1_max

        def eval(self, R1, theta):
            return below.iv_surfaces_direct_below(
                self.et, self.kp, R1, theta, rtol=1e-9, omega=self.om
            )

    grid, (et, kp, om, km, lam_m) = proj_grid
    surf = DirectSurfaces(et, kp, om, grid.r1_max)
    assert getattr(surf, "_regions", None) is None
    obs = np.array([[2.0, 0.5, -0.3]])
    src = np.array([[0.0, 0.0, -0.15]])
    t_obs = np.array([[1.0, 0.0, 0.0]])
    t_src = np.array([[0.0, 0.0, 1.0]])

    # A spy, not an inference: comparing the two dispatches' NUMBERS would not
    # prove this, because the surfaces underneath a duck-typed source come from
    # `iv_surfaces_direct_below`, which has a C++ path of its own.
    def boom(*a, **k):
        raise AssertionError("the twin was handed a duck-typed surface source")

    monkeypatch.setattr(below._acc, "remainder_field_proj_batch_below", boom)
    got = below.remainder_field_proj_below(
        obs, t_obs, src, t_src, GROUND_Z, kp, km, surf
    )
    assert got.shape == (1, 1) and np.all(np.isfinite(got))
    # ... and the real tabulation DOES reach it, so the spy is live.
    with pytest.raises(AssertionError, match="duck-typed"):
        below.remainder_field_proj_below(obs, t_obs, src, t_src, GROUND_Z, kp, km, grid)


@pytest.mark.parametrize(
    "what,match",
    [
        ("past_cap", "past the tabulation"),
        ("grazing", "grazing floor"),
        ("wrong_side", "strictly below"),
        ("above_grid", "SommerfeldGridBelow"),
    ],
)
def test_g5689_the_refusals_survive_the_dispatch(proj_grid, what, match):
    """The three domain refusals plus the wrong-family one, through the C++
    projection. None of that prose is transcribed into C++: the kernel reports
    the query's extremes and `SommerfeldGridBelow.eval` raises them, so the two
    paths cannot disagree about which geometries are served."""
    grid, (et, kp, om, km, lam_m) = proj_grid
    t1 = np.array([[1.0, 0.0, 0.0]])
    src = np.array([[0.0, 0.0, -0.2]])
    g = grid
    if what == "past_cap":
        obs = np.array([[5.0 * grid.r1_max, 0.0, -0.2]])
    elif what == "grazing":
        # theta well under the 1 deg floor, R1 comfortably inside the cap: two
        # 5 cm-deep points 10 m apart, which is the product geometry gone one
        # step too shallow.
        obs = np.array([[0.5 * grid.r1_max, 0.0, -0.05]])
        src = np.array([[0.0, 0.0, -0.05]])
        assert np.degrees(np.arctan2(0.1, 0.5 * grid.r1_max)) < 1.0
    elif what == "wrong_side":
        obs = np.array([[1.0, 0.0, +0.4]])
    else:
        obs = np.array([[1.0, 0.0, -0.4]])
        g = som.get_grid(et, kp, 1.0, om)

    msgs = []
    for use_numpy in (True, False):
        ctx = force_numpy() if use_numpy else None
        if ctx:
            ctx.__enter__()
        try:
            with pytest.raises(ValueError, match=match) as e:
                below.remainder_field_proj_below(obs, t1, src, t1, GROUND_Z, kp, km, g)
            msgs.append(str(e.value))
        finally:
            if ctx:
                ctx.__exit__()
    assert msgs[0] == msgs[1], msgs


def test_g5689_an_in_domain_query_at_the_edges_is_served_not_refused(proj_grid):
    """The complement of the refusals: R₁ exactly at the cap and θ exactly at
    the grazing floor are IN the domain, and the extremes the kernel reports
    must not push them out of it by a rounding step."""
    grid, (et, kp, om, km, lam_m) = proj_grid
    th_min = grid.th_min
    r1 = grid.r1_max
    hh = r1 * np.sin(th_min)
    rho = r1 * np.cos(th_min)
    obs = np.array([[rho, 0.0, -0.5 * hh]])
    src = np.array([[0.0, 0.0, -0.5 * hh]])
    t1 = np.array([[1.0, 0.0, 0.0]])
    got = below.remainder_field_proj_below(obs, t1, src, t1, GROUND_Z, kp, km, grid)
    with force_numpy():
        ref = below.remainder_field_proj_below(obs, t1, src, t1, GROUND_Z, kp, km, grid)
    assert np.all(np.isfinite(got))
    assert cwise(got, ref) < G_568_9_TOL


# ======================================================================
# G-568-10 — the ±=+ family keeps its bytes
# ======================================================================
#
# The standing arc-wide trap: U2 touches the below family only. The projection
# was served by a TWIN (`remainder_field_proj_batch_below`) precisely so that
# `proj_one` — ~90 % of an above/above Sommerfeld solve — did not have to grow
# a branch for a carrier it never uses. This is the gate that the twin really
# was a twin.
#
# MEASURED, ±=+ C++ kernel against its own numpy body over the pair set below:
# 1.89e-15 on the projection and 4.13e-12 on the six integrals. Neither is
# zero and neither should be — `SommerfeldGrid.eval`'s einsum reduces the 4x4
# stencil in a different order from `proj_one`'s two nested loops, which is the
# spread that pair has always had. What matters is that they are the SAME
# numbers U2 found: nothing in this unit touched `proj_one`, `_six_integrals`
# or the fig-13/fig-14 contour, and these two gates are what says so from
# inside the U2 build.
#
# Pinned at 1e-12 / 1e-9 — three decades over each, and far under any spread a
# real disturbance of those paths would produce.


def test_g56810_the_above_family_projection_did_not_move(record_property):
    et = complex(13.0, -12.839)
    k2 = 2.0 * np.pi
    om = k2 * C0
    grid = som.get_grid(et, k2, 1.0, om)
    obs, t_obs, src, t_src = _pair_set(11, 6, 5, depth_lo=0.05, depth_hi=0.5, span=0.8)
    obs[:, 2] = np.abs(obs[:, 2])  # the ±=+ family lives ABOVE the interface
    src[:, 2] = np.abs(src[:, 2])
    got = som.remainder_field_proj(obs, t_obs, src, t_src, GROUND_Z, k2, grid)

    # The numpy body, reached by hiding the accelerator from the ±=+ dispatch.
    saved = som._acc
    try:
        som._acc = None
        ref = som.remainder_field_proj(obs, t_obs, src, t_src, GROUND_Z, k2, grid)
    finally:
        som._acc = saved
    worst = cwise(got, ref)
    record_property("worst_rel", float(worst))
    assert worst < 1e-12, f"the ±=+ projection moved: {worst:.3e}"


def test_g56810_the_above_family_six_integrals_did_not_move(record_property):
    """`_six_integrals` and its C++ batch carry every fig-13/fig-14 landmark
    that assumes a REAL decay-medium k. U2 shares `_gamma`, `_d12` and
    `_bessel_j0_j1x` with them and must not have disturbed either."""
    et = complex(20.0, -77.036)
    k2 = 0.14671
    rho = np.array([0.3, 1.0, 2.5])
    h = np.array([0.2, 0.05, 0.9])
    got = som._six_integrals_batch(et, k2, rho, h, rtol=1e-9)
    ref = np.stack([som._six_integrals(et, k2, rho[i], h[i], 1e-9) for i in range(3)])
    worst = cwise(got, ref)
    record_property("worst_rel", float(worst))
    assert worst < 1e-9, f"the ±=+ six integrals moved: {worst:.3e}"
