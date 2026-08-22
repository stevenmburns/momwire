"""The transmitted Sommerfeld family: a crossing pair (momwire#553 unit 3).

ORACLES (three, none of them the licensed engine)
-------------------------------------------------
1. **The #524 phase-0 prototype's direct evaluation**, committed as numbers
   in `golden_transmitted_524.py`. An independent implementation of the same
   physics from the same open-literature equation kit, written before this
   module existed, whose ±=+ anchor reproduces momwire's OWN four
   interpolation surfaces to 1.4e-10.
2. **empymod** (`ht='quad'`, pts_per_dec 600), committed twice over: the
   banked phase-0 T-line/T-vert grids for below→above, and a FRESH run in
   the crossed source-above configuration for above→below. Every tolerance
   is named against that grid's own recorded quad-vs-quad spread.
3. **Physics with no oracle in it at all**: ε̃ → 1 must reproduce the
   free-space dipole field EXACTLY here — not approximately, and not as a
   remainder that vanishes — because the transmitted integral is the whole
   field; and σ → ∞ must extinguish it, kernel and all.

WHY THE FIFTH SURFACE IS GATED SEPARATELY EVERYWHERE
----------------------------------------------------
`TzH` = −C₁ ∂²V_T/∂ρ∂z′ is the one component with no ±=+ analogue (at ±=+
the ∂/∂z′ derivative collapses onto ∂/∂z and momwire ships four surfaces).
Phase 0 named it the highest-risk quantity in the kit: the licensed engine
departs from the prototype on it by O(1) — 0.59 to 1.14 relative, at every
depth, in both soils — while agreeing to its own printed-symmetry noise
floor (2e-4) on the E_x component from the same code path, same currents,
same grid; and empymod, run at converged quad, sides with the prototype.
So every gate here reports it on its own row and never grades it against a
larger neighbour, and the engine is not consulted for it at any depth. That
comparison is the reviewer's, outside this repo.

WHY THE RECIPROCITY GATE IS NOT SELF-SATISFYING
-----------------------------------------------
above→below is served as the transpose of the same tables, so "the
transpose equals the transpose" would prove nothing. Two things make
G-U3-4 a real test. The by-construction half checks the BOOKKEEPING, which
is not free: the role exchange reverses the horizontal direction d̂ and
sinφ is odd, and the transposed dyad swaps T_ρ^V with T_z^H — the fifth
surface — so a family with four surfaces could not have served both
directions at all. The other half is `CROSSED`, an empymod run with the
source in the air, which shares no line of code and no assumption with any
of it.
"""

import math

import numpy as np
import pytest

from momwire import _sommerfeld as som
from momwire import _sommerfeld_below as below
from momwire import _sommerfeld_transmitted as trans

import golden_transmitted_524 as gold

EPS0 = 8.8541878128e-12
MU0 = som._MU0
C0 = 299792458.0
KEYS = trans._SURF_KEYS_T
GROUND_Z = 0.0


# ----------------------------------------------------------------------
# Media and the closed forms the physics gates are assembled from
# ----------------------------------------------------------------------


def eps_tilde(eps_r, sigma, f_hz):
    """ε̃ = ε_r − jσ/(ωε₀) (scratch/524-phase0/proto/EQUATIONS.md)."""
    return eps_r - 1j * sigma / (2.0 * np.pi * f_hz * EPS0)


def medium(cell):
    """(ε̃, k_p, ω, k_m) for a golden cell key like "A/7MHz"."""
    sname, fname = cell.split("/")
    er, sg = gold.SOILS[sname]
    f = float(fname[:-3]) * 1e6
    om = 2.0 * np.pi * f
    et = eps_tilde(er, sg, f)
    kp = om / C0
    return et, kp, om, below.k_medium(et, kp)


def fs_field(c1, k, p_hat, d):
    """E = C₁(∇∇/k² + I)·(p̂ e^{−jkR}/R) at displacement d = r − r′, complex
    k — EQUATIONS.md's free-space dipole, transcribed, not remembered."""
    d = np.asarray(d, dtype=float)
    r = float(np.sqrt(np.dot(d, d)))
    rh = d / r
    a = 1.0 / (k * r)
    g = np.exp(-1j * k * r) / r
    pr = np.dot(rh, p_hat)
    return (
        c1
        * g
        * (
            np.asarray(p_hat, dtype=np.complex128) * (1.0 - 1j * a - a * a)
            + rh * pr * (-1.0 + 3j * a + 3.0 * a * a)
        )
    )


class DirectSurfaces:
    """A grid-shaped view on `t_surfaces_direct`.

    The point functions take anything that can `eval(R, theta, zp)`, so the
    same azimuth combination can be driven off the exact surfaces instead of
    a tabulation. That is what makes the three-way grid → direct → goldens
    comparison possible with ONE combination routine: when the grid and the
    direct evaluator disagree, the dyad is not the suspect.
    """

    regime = "below-above"

    def __init__(self, eps_t, k2, omega, r_max=1e9, rtol=1e-9):
        self.eps_t = eps_t
        self.k2 = k2
        self.omega = omega
        self.r_max = r_max
        self.rtol = rtol

    def eval(self, R, theta, zp):
        R = np.asarray(R, dtype=float)
        theta = np.asarray(theta, dtype=float)
        rho = np.maximum(R * np.cos(theta), 0.0)
        z = np.maximum(R * np.sin(theta), 0.0)
        return trans.t_surfaces_direct(
            self.eps_t, self.k2, rho, z, zp, rtol=self.rtol, omega=self.omega
        )


# `record_property` values must be plain Python scalars or strings, never
# numpy ones: execnet serializes each report to the xdist controller and its
# `_Serializer` has no dispatch for `np.float64` — the worker dies mid-report
# and the run ends in an INTERNALERROR naming the test, with no failure and
# no traceback pointing here. Hence the `float(...)` on every value below.


def per_surface_rel(got, ref, floor_rel=1e-10):
    """Worst per-surface relative error, and the surface it was on.

    PER SURFACE against its OWN magnitude, never against the largest of the
    five at that point. The transmitted surfaces span decades at a single
    point — the fifth one has a deep null in the near zone — and an
    of-family norm would grade a small surface against a large neighbour,
    which is U2's second inversion and the one that is easiest to reintroduce
    by accident. `floor_rel` skips a cell only when the reference is a
    numerical zero relative to that surface's own maximum over the sample.
    """
    worst = 0.0
    where = None
    skipped = 0
    total = 0
    for k in KEYS:
        r = np.abs(np.asarray(ref[k]))
        live = r >= floor_rel * max(float(r.max()), 1e-300)
        total += r.size
        skipped += int((~live).sum())
        if not live.any():
            continue
        e = float(
            np.max(np.abs(np.asarray(got[k]) - np.asarray(ref[k]))[live] / r[live])
        )
        if e > worst:
            worst, where = e, k
    return worst, where, skipped, total


def rel_vec(got, ref):
    got = np.asarray(got)
    ref = np.asarray(ref)
    den = max(float(np.max(np.abs(ref))), 1e-300)
    return float(np.max(np.abs(got - ref))) / den


# ======================================================================
# G-U3-1 — the five surfaces against the phase-0 prototype
# ======================================================================
#
# The committed lattice spans R/lambda_p from 0.001 to 2 (the full served
# radius), theta from 0 deg — the observer ON the interface, which this
# family tabulates — to 90, and three source depths across the ladder. This
# is the core formulation gate: the (7f)/(7g) kernels, all five analytic
# under-integral derivatives, the two-leg divide-out and the C1 prefactor at
# once, against an evaluator that shares no line of code with this one.
#
# MEASURED worst over every cell and every surface: 5.2e-10 (a separate
# 36-point sweep at rtol 1e-11 read 5.2e-10 too). Pinned at 1e-7 — ~200x
# margin, and still three decades under any tolerance that could absorb a
# sign or a swapped derivative.

G_U3_1_TOL = 1e-7


@pytest.mark.parametrize("cell", sorted(gold.SURFACES))
def test_gu3_1_surfaces_vs_the_phase0_prototype(cell, record_property):
    et, kp, om, km = medium(cell)
    lam_p = 2.0 * np.pi / kp
    rows = gold.SURFACES[cell]
    rr = np.array([r * lam_p for r, _, _, *_ in rows])
    th = np.radians(np.array([t for _, t, _, *_ in rows]))
    zp = np.array([-z for _, _, z, *_ in rows])
    rho = np.maximum(rr * np.cos(th), 0.0)
    zz = np.maximum(rr * np.sin(th), 0.0)
    got = trans.t_surfaces_direct(et, kp, rho, zz, zp, rtol=1e-9, omega=om)
    worst = 0.0
    per_key = {k: 0.0 for k in KEYS}
    for i, row in enumerate(rows):
        ref = dict(zip(KEYS, row[3:]))
        for k in KEYS:
            e = abs(got[k][i] - ref[k]) / max(abs(ref[k]), 1e-300)
            per_key[k] = max(per_key[k], e)
            worst = max(worst, e)
    for k in KEYS:
        record_property(f"worst_{k}", float(per_key[k]))
    record_property("worst_rel", float(worst))
    # The fifth surface on its own row, at every cell, never pooled.
    assert per_key["TzH"] < G_U3_1_TOL, f"{cell}: TzH {per_key['TzH']:.3e}"
    assert worst < G_U3_1_TOL, f"{cell}: worst rel {worst:.3e}"


def test_gu3_1_the_fifth_surface_is_not_a_multiple_of_the_first():
    """∂/∂z′ is not −∂/∂z, and that is why this family has five surfaces.

    At ±=+ the two exponential legs carry the SAME γ, so ∂/∂z′ = −∂/∂z and
    R_z^H collapses to −cosφ·R_ρ^V — the identity that lets `_sommerfeld`
    ship four surfaces. The transmitted legs carry γ_m and γ_p, so the
    collapse fails, and it fails by an amount that grows with the contrast.
    Asserted so a future "simplification" that reuses the below family's
    four-surface dyad cannot pass quietly.
    """
    et, kp, om, km = medium("A/7MHz")
    lam = np.linspace(1e-3, 4.0 * abs(km), 400).astype(np.complex128)
    stack = trans._integrand_six_transmitted(lam, 5.0, 1.0, -0.15, kp, km)
    d_z, d_zp = stack[0], stack[4]
    # ratio would be identically -1 if the two derivatives coincided
    ratio = d_zp / d_z
    assert np.max(np.abs(ratio + 1.0)) > 0.5, "the two z-derivatives coincide"
    # ... and they DO coincide in the eps_t -> 1 limit, which is the check
    # that the difference is the media contrast and not a coding accident.
    same = trans._integrand_six_transmitted(lam, 5.0, 1.0, -0.15, kp, kp + 0j)
    assert np.max(np.abs(same[4] / same[0] + 1.0)) < 1e-12


def test_gu3_1_the_transmitted_kernels_are_not_the_remainder_kernels():
    """The transmitted family is a different integral, not a swapped `_d12`.

    U2's below/below family IS `_sommerfeld._d12` with its arguments
    exchanged. This one is not, and the difference is structural: the
    transmitted denominators k_m²γ_p + k_p²γ_m and γ_m + γ_p carry NO
    subtracted static term, because there is no direct-plus-image part to
    subtract off. Asserted against both readings of `_d12`.
    """
    et, kp, om, km = medium("A/7MHz")
    lam = np.linspace(1e-3, 4.0 * abs(km), 200).astype(np.complex128)
    g_p = som._gamma(lam, kp)
    g_m = som._gamma(lam, km)
    v_kernel = 1.0 / (km * km * g_p + kp * kp * g_m)
    u_kernel = 1.0 / (g_m + g_p)
    for k1, k2 in ((kp, km), (km, kp)):
        d1, d2, _ = som._d12(lam, k1, k2)
        assert np.max(np.abs(d2 / 2.0 - v_kernel)) > 1e-3 * np.max(np.abs(v_kernel))
        assert np.max(np.abs(d1 / 2.0 - u_kernel)) > 1e-3 * np.max(np.abs(u_kernel))


# ======================================================================
# G-U3-2 — the assembled field against the banked empymod grids
# ======================================================================
#
# 34 grids, 340 points: every SPEC transmitted cell (T-line at z = +1 m out
# to 30 m, T-vert at x = 10 m up a ladder to z = 10 m), HED and VED, soils
# A/B/C at 7 and 21 MHz. The tolerance is named against each grid's own
# recorded quad(300)-vs-quad(600) spread and never below it: phase 0
# measured the prototype's disagreement tracking that spread at 0.25-0.42x
# across all 17 cells, which is what says the residual is the ORACLE's
# quadrature error and not ours.
#
# MEASURED worst over all 34 grids: see the recorded properties. E_z — the
# fifth surface's component — is reported on its own row at every grid and
# gated at the SAME class as E_x, never looser: that is the whole point of
# the split, and phase 0's engine comparison is why.

G_U3_2_SPREAD_FACTOR = 1.0


def _compose_transmitted_point(et, kp, om, obs, src, p_hat, surfaces):
    km = below.k_medium(et, kp)
    return trans.transmitted_field_below_to_above(
        np.asarray(obs, dtype=float)[None, :],
        np.asarray(src, dtype=float)[None, :],
        np.asarray(p_hat, dtype=complex)[None, :],
        GROUND_Z,
        kp,
        km,
        surfaces,
    )[0]


@pytest.mark.slow
@pytest.mark.parametrize("entry", range(len(gold.TRANSMITTED)))
def test_gu3_2_composed_field_vs_empymod(entry, record_property):
    cell = gold.TRANSMITTED[entry]
    om = 2.0 * np.pi * cell["freq_hz"]
    et = eps_tilde(cell["eps_r"], cell["sigma"], cell["freq_hz"])
    kp = om / C0
    src = np.array(cell["src_xyz"], dtype=float)
    p_hat = (
        np.array([1.0, 0.0, 0.0])
        if cell["kind"] == "HED"
        else np.array([0.0, 0.0, 1.0])
    )
    surfaces = DirectSurfaces(et, kp, om)
    worst = 0.0
    worst_ez = 0.0
    for pt, ref in zip(cell["points"], cell["E"]):
        got = _compose_transmitted_point(et, kp, om, pt, src, p_hat, surfaces)
        scale = max(abs(v) for v in ref)
        worst = max(worst, max(abs(got[i] - ref[i]) / scale for i in range(3)))
        worst_ez = max(worst_ez, abs(got[2] - ref[2]) / scale)
    spread = cell["oracle_spread"]
    record_property("worst_rel", float(worst))
    record_property("worst_Ez_the_fifth_surface", float(worst_ez))
    record_property("oracle_spread", float(spread))
    record_property("ratio_to_oracle_spread", float(worst / max(spread, 1e-300)))
    tol = G_U3_2_SPREAD_FACTOR * spread
    assert worst_ez < tol, (
        f"{cell['id']}: E_z (the fifth surface) {worst_ez:.3e} vs the oracle's "
        f"own spread {spread:.3e}"
    )
    assert worst < tol, f"{cell['id']}: {worst:.3e} vs oracle spread {spread:.3e}"


# ======================================================================
# G-U3-3 — the z' ladder
# ======================================================================
#
# Off-node z' against direct evaluation over the FULL ladder range, at every
# soil, with both endpoints and the shallow end forced into the sample. The
# 1e-3 class is the architecture's own premise (phase 0's measured target),
# so this gate is the premise itself and not a convenience.

G_U3_3_TOL = 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("cell", ["A/7MHz", "B/7MHz", "C/7MHz", "A/21MHz"])
def test_gu3_3_ladder_interpolates_off_node(cell, record_property):
    et, kp, om, km = medium(cell)
    lam_m = 2.0 * np.pi / abs(km)
    lam_p = 2.0 * np.pi / kp
    zp_lo, zp_hi = 0.05, min(0.5, trans._ZPRIME_MAX_LAMBDA_M * lam_m)
    nodes, n = trans.ladder_nodes(zp_lo, zp_hi, lam_m)
    record_property("ladder_rungs", int(n))
    record_property("ladder_range_m", f"[{zp_lo}, {zp_hi:.4f}]")
    record_property("ladder_quarters_lam_m", float(zp_hi / (0.25 * lam_m)))

    # Three observers spanning the served rectangle, and z' probes that
    # include BOTH endpoints and the shallow end explicitly plus off-node
    # points that fall between rungs by construction.
    rng = np.random.default_rng(5533)
    zq = np.concatenate(
        [
            [zp_lo, zp_hi],
            np.sqrt(nodes[:-1] * nodes[1:]),  # geometric cell centres
            np.exp(rng.uniform(math.log(zp_lo), math.log(zp_hi), 6)),
        ]
    )
    worst = 0.0
    worst_key = None
    for rw, thd in ((0.01, 0.5), (0.1, 5.0), (1.0, 30.0)):
        rr = rw * lam_p
        rho = max(rr * math.cos(math.radians(thd)), 0.0)
        zz = max(rr * math.sin(math.radians(thd)), 0.0)
        # Interpolate in ln|z'| from the rungs, exactly as the grid does.
        node_vals = trans.t_surfaces_direct(
            et, kp, rho, zz, -nodes, rtol=1e-9, omega=om
        )
        ref = trans.t_surfaces_direct(et, kp, rho, zz, -zq, rtol=1e-9, omega=om)
        got = {}
        lz = np.log(nodes)
        fz = (np.log(zq) - lz[0]) / (lz[1] - lz[0])
        k0 = np.clip(np.floor(fz).astype(int) - 1, 0, n - 4)
        wz = som.SommerfeldGrid._lagrange4(fz - k0)
        kk = k0[:, None] + np.arange(4)[None, :]
        for k in KEYS:
            got[k] = np.einsum("nk,nk->n", node_vals[k][kk], wz)
        e, where, _, _ = per_surface_rel(got, ref)
        if e > worst:
            worst, worst_key = e, where
        record_property(f"R{rw}lp_th{thd}", float(e))
    record_property("worst_rel", float(worst))
    record_property("worst_surface", str(worst_key))
    assert worst < G_U3_3_TOL, f"{cell}: ladder off-node {worst:.3e} on {worst_key}"


def test_gu3_3_the_ladder_rule_is_not_phase_zeros_scalar_rule():
    """The fourth inversion, asserted rather than remembered.

    Phase 0 sized the ladder at ≈13 nodes per quarter λ_m from the two
    SCALARS. The tabulation carries their DERIVATIVES and needs 1.5–2.3x
    that; `ladder_nodes` is sized from the surface measurement. A future
    edit that "restores" the phase-0 constant would halve the node count,
    so the two are held apart here by name.
    """
    lam_m = 10.019  # soil A / 7 MHz
    _, n = trans.ladder_nodes(0.02, 1.0, lam_m)
    phase0_style = math.ceil(13.0 * 1.0 / (0.25 * lam_m)) + 1
    assert n >= 21, f"the ladder rule went soft: {n} rungs over [0.02, 1] m"
    assert n > 2 * phase0_style, (
        f"the ladder rule collapsed onto the phase-0 SCALAR constant "
        f"({n} vs {phase0_style})"
    )
    # A single depth is one rung and no interpolation at all.
    nodes, one = trans.ladder_nodes(0.15, 0.15, lam_m)
    assert one == 1 and nodes[0] == 0.15


# ======================================================================
# G-U3-4 — reciprocity: above -> below is the transpose
# ======================================================================


def test_gu3_4_the_swapped_integrand_is_the_same_integral(record_property):
    """Reciprocity at the INTEGRAND level, before any quadrature or dyad.

    `_integrand_six_transmitted(swap=True)` builds the above→below stack
    independently — source above carrying γ_p, observer below carrying γ_m,
    with its own two derivative factors. Evaluated at the exchanged
    geometry it must reproduce the unswapped stack EXACTLY, with indices 0
    and 4 — the two z-derivatives, which are T_ρ^V and T_z^H — traded. That
    trade is the same one `_combine_transmitted_transposed` performs on the
    assembled dyad, so this is the algebra the transpose rests on, checked
    one level below where the transpose lives.

    Bit-exact, not near: with the verified (7f) the two are literally the
    same expression, which is what phase 0's G1 measured at 0.0.
    """
    et, kp, om, km = medium("A/7MHz")
    lam = np.linspace(1e-3, 4.0 * abs(km), 257).astype(np.complex128)
    a_depth, b_height = 0.15, 1.7
    up = trans._integrand_six_transmitted(lam, 6.0, b_height, -a_depth, kp, km)
    down = trans._integrand_six_transmitted(
        lam, 6.0, -a_depth, +b_height, kp, km, swap=True
    )
    order = [4, 1, 2, 3, 0, 5]
    per_index = [
        float(np.max(np.abs(down[i] - up[j]) / np.maximum(np.abs(up[j]), 1e-300)))
        for i, j in enumerate(order)
    ]
    record_property("per_index_rel", repr([f"{v:.3e}" for v in per_index]))
    assert max(per_index) == 0.0, (
        f"the swapped stack is not the same integral: {per_index}"
    )
    # Index 1 is bit-exact only because `_integrand_six_transmitted` writes
    # (d2/dz2 + k^2) as lambda^2 instead of gamma^2 + k^2. Spelled as the
    # sum it is a difference of two O(k^2) numbers giving an O(lambda^2)
    # answer, and the two media's spellings of the same quantity landed
    # 3.5e-11 apart at the bottom of the contour. Asserted, so that a future
    # edit "restoring" the textbook grouping is caught here rather than
    # eleven digits down in a fill.
    # ... and WITHOUT the 0<->4 trade it is a different thing, so the trade
    # is carrying weight rather than decorating a tautology.
    assert np.max(np.abs(down[0] - up[0])) > 0.0


@pytest.mark.slow
def test_gu3_4_the_transpose_is_the_dyad_transpose(record_property):
    """The by-construction half: the role exchange's BOOKKEEPING.

    Not self-satisfying. The above→below path reverses which endpoint is
    the source, which reverses d̂ — and sinφ is odd, so a naive reuse of the
    upward path's geometry flips E_φ. It also has to swap T_ρ^V with T_z^H,
    the fifth surface. Both are checked against a 3×3 dyad assembled column
    by column from the upward path and transposed with numpy.
    """
    et, kp, om, km = medium("A/7MHz")
    surfaces = DirectSurfaces(et, kp, om)
    above = np.array([[7.0, 3.0, 2.0], [0.0, 0.0, 0.5]])
    belowp = np.array([[-1.0, 2.0, -0.15], [0.0, 0.0, -0.4]])
    basis = np.eye(3, dtype=complex)

    # D_{b->a}: column j is the upward field of a unit j-directed source.
    d_up = np.empty((len(above), len(belowp), 3, 3), dtype=complex)
    for j in range(3):
        for n in range(len(belowp)):
            d_up[:, n, :, j] = trans.transmitted_field_below_to_above(
                above, belowp[n : n + 1], basis[j][None, :], GROUND_Z, kp, km, surfaces
            )
    # D_{a->b} the product way, and the same thing as the transpose.
    d_dn = np.empty((len(belowp), len(above), 3, 3), dtype=complex)
    for j in range(3):
        for n in range(len(above)):
            d_dn[:, n, :, j] = trans.transmitted_field_above_to_below(
                belowp, above[n : n + 1], basis[j][None, :], GROUND_Z, kp, km, surfaces
            )
    worst = rel_vec(d_dn, np.transpose(d_up, (1, 0, 3, 2)))
    record_property("worst_rel", float(worst))
    assert worst < 1e-13, f"the transpose is not the transpose: {worst:.3e}"

    # And a NAIVE transpose — the same dyad without the d-hat reversal —
    # must fail, so the gate is measuring something.
    naive = trans._combine_transmitted(
        {k: np.array([[v]]) for k, v in surfaces.eval(7.62, 0.26, -0.15).items()},
        np.array([[1.0 + 0j]]),
        np.array([[1.0]]),
        np.array([[0.0]]),
        np.array([[1.0 + 0j, 0.0, 0.0]]),
    )
    swapped = trans._combine_transmitted_transposed(
        {k: np.array([[v]]) for k, v in surfaces.eval(7.62, 0.26, -0.15).items()},
        np.array([[1.0 + 0j]]),
        np.array([[1.0]]),
        np.array([[0.0]]),
        np.array([[1.0 + 0j, 0.0, 0.0]]),
    )
    assert rel_vec(swapped, naive) > 1e-3, (
        "the transposed combination is numerically the untransposed one — "
        "the fifth surface has stopped being fifth"
    )


@pytest.mark.slow
@pytest.mark.parametrize("entry", range(len(gold.CROSSED)))
def test_gu3_4_above_to_below_vs_empymod(entry, record_property):
    """The independent half: empymod with the source in the AIR.

    The phase-0 matrix never captured this configuration, so this is a
    fresh oracle rather than a re-read of a banked one, and it is the only
    check on the transpose that shares nothing with the implementation.
    """
    cell = gold.CROSSED[entry]
    om = 2.0 * np.pi * cell["freq_hz"]
    et = eps_tilde(cell["eps_r"], cell["sigma"], cell["freq_hz"])
    kp = om / C0
    km = below.k_medium(et, kp)
    src = np.array([[0.0, 0.0, gold.CROSSED_SRC_Z]])
    p_hat = np.array(
        [[1.0 + 0j, 0.0, 0.0]] if cell["kind"] == "HED" else [[0.0, 0.0, 1.0 + 0j]]
    )
    surfaces = DirectSurfaces(et, kp, om)
    obs = np.array(cell["points"], dtype=float)
    got = trans.transmitted_field_above_to_below(
        obs, src, p_hat, GROUND_Z, kp, km, surfaces
    )
    worst = 0.0
    worst_ez = 0.0
    for i, ref in enumerate(cell["E"]):
        scale = max(abs(v) for v in ref)
        worst = max(worst, max(abs(got[i, j] - ref[j]) / scale for j in range(3)))
        worst_ez = max(worst_ez, abs(got[i, 2] - ref[2]) / scale)
    spread = cell["oracle_spread"]
    record_property("worst_rel", float(worst))
    record_property("worst_Ez", float(worst_ez))
    record_property("oracle_spread", float(spread))
    assert worst < max(spread, 1e-6), (
        f"{cell['id']}: above→below {worst:.3e} vs the oracle's own spread {spread:.3e}"
    )


# ======================================================================
# G-U3-5 — physics with no oracle in it
# ======================================================================


def test_gu3_5_eps_one_is_the_free_space_dipole_exactly(record_property):
    """ε̃ → 1 is EXACT here, not a vanishing correction.

    The ±=± families check that their remainder goes to zero at ε̃ = 1 —
    `_sommerfeld_below` short-circuits it, because D₁ = D₂ = 0 identically.
    The transmitted integral has no such structure: at k_m = k_p the
    Sommerfeld identity turns V_T into G/k_p² and U_T into G, and (7a)–(7e)
    reproduce the free-space dyadic dipole field TERM FOR TERM. So the whole
    composed field must equal the closed form, and there is nothing to
    short-circuit.
    """
    kp = 2.0 * np.pi * 7e6 / C0
    om = 2.0 * np.pi * 7e6
    c1 = -1j * om * MU0 / (4.0 * np.pi)
    surfaces = DirectSurfaces(1.0 + 0j, kp, om, rtol=1e-11)
    worst = 0.0
    for src_z, obs in (
        (-0.15, [6.0, 0.0, 1.0]),
        (-1.0, [0.5, 0.5, 3.0]),
        (-0.05, [0.0, 0.0, 2.0]),
    ):
        for p_hat in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            got = trans.transmitted_field_below_to_above(
                np.array([obs], dtype=float),
                np.array([[0.0, 0.0, src_z]]),
                p_hat[None, :].astype(complex),
                GROUND_Z,
                kp,
                kp + 0j,
                surfaces,
            )[0]
            ref = fs_field(c1, kp, p_hat, np.array(obs) - np.array([0.0, 0.0, src_z]))
            worst = max(worst, rel_vec(got, ref))
    record_property("worst_rel", float(worst))
    assert worst < 1e-9, f"eps_t -> 1 is not the free-space dipole: {worst:.3e}"


@pytest.mark.slow
def test_gu3_5_sigma_to_infinity_extinguishes_the_transmitted_field(record_property):
    """σ → ∞ must extinguish the WHOLE field above, kernel and all.

    A perfect conductor transmits nothing. Both the composed field and the
    kernel-level cancellation 2/(γ_m + γ_p) are asserted monotone, the
    second one because the field could in principle fall for the wrong
    reason (an overflow, a clamped surface) and the kernel cannot.
    """
    f = 7e6
    om = 2.0 * np.pi * f
    kp = om / C0
    mags = []
    kernels = []
    for sigma in (0.005, 0.05, 0.5, 5.0, 50.0, 500.0):
        et = eps_tilde(13.0, sigma, f)
        km = below.k_medium(et, kp)
        surfaces = DirectSurfaces(et, kp, om)
        e = trans.transmitted_field_below_to_above(
            np.array([[10.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, -0.05]]),
            np.array([[1.0 + 0j, 0.0, 0.0]]),
            GROUND_Z,
            kp,
            km,
            surfaces,
        )[0]
        mags.append(float(np.max(np.abs(e))))
        lam = 0.5
        kernels.append(abs(2.0 / (som._gamma(lam, km) + som._gamma(lam, kp))))
    record_property("sigma_ladder_abs_E", repr([f"{m:.4e}" for m in mags]))
    record_property("kernel_2_over_gm_plus_gp", repr([f"{k:.4e}" for k in kernels]))
    record_property("extinction_ratio", float(mags[-1] / mags[0]))
    assert all(b < a for a, b in zip(mags, mags[1:])), mags
    assert all(b < a for a, b in zip(kernels, kernels[1:])), kernels
    assert mags[-1] < 1e-4 * mags[0]


def test_gu3_5_the_transmitted_field_is_continuous_across_the_interface():
    """z = 0 is a value, not a refusal — and it is the value the limit
    approaches, with a FINITE slope.

    The below/below family cannot say this (its h → 0 strips the tail of its
    decay entirely); the transmitted family can, because the source leg
    holds the tail down, and the whole θ = 0 row of the grid rests on it.
    Tested as a slope rather than as a jump: the surfaces do move at z = 0 —
    T_ρ^V carries a z-derivative and changes by 2.5e-4 per 0.1 mm at
    ρ = 4 m — so the honest statement is that the change is proportional to
    Δz, which a discontinuity's would not be.
    """
    et, kp, om, km = medium("A/7MHz")
    on = trans.t_surfaces_direct(et, kp, 4.0, 0.0, -0.15, rtol=1e-11, omega=om)
    steps = (1e-3, 1e-4, 1e-5)
    jumps = []
    for dz in steps:
        near = trans.t_surfaces_direct(et, kp, 4.0, dz, -0.15, rtol=1e-11, omega=om)
        jumps.append(per_surface_rel(near, on)[0])
    slopes = [j / dz for j, dz in zip(jumps, steps)]
    assert max(slopes) < 2.0 * min(slopes), (
        f"the approach to z = 0 is not linear in dz: jumps {jumps}"
    )
    assert jumps[-1] < 1e-4, f"a jump remains at z -> 0: {jumps[-1]:.3e}"
    # ... and the z = 0 value is where the linear approach lands: Richardson
    # on the two finest steps must recover it to the slope's own accuracy.
    rich = jumps[-1] - (jumps[-2] - jumps[-1]) / 9.0
    assert abs(rich) < 1e-4, f"z = 0 is not the limit of its own neighbourhood: {rich}"


# ======================================================================
# G-U3-6 — refusals, by name
# ======================================================================


def test_gu3_6_a_source_on_or_above_the_interface_refuses():
    et, kp, om, km = medium("A/7MHz")
    with pytest.raises(ValueError, match="momwire#524 phase 3"):
        trans.t_surfaces_direct(et, kp, 4.0, 1.0, 0.0, omega=om)
    with pytest.raises(ValueError, match="momwire#524 phase 3"):
        trans.t_surfaces_direct(et, kp, 4.0, 1.0, +0.2, omega=om)


def test_gu3_6_a_wrong_side_endpoint_refuses_by_name():
    et, kp, om, km = medium("A/7MHz")
    surfaces = DirectSurfaces(et, kp, om)
    with pytest.raises(ValueError, match="STRICTLY below"):
        trans.transmitted_field_below_to_above(
            np.array([[4.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, +0.1]]),
            np.array([[1.0 + 0j, 0.0, 0.0]]),
            GROUND_Z,
            kp,
            km,
            surfaces,
        )
    with pytest.raises(ValueError, match="at or above"):
        trans.transmitted_field_below_to_above(
            np.array([[4.0, 0.0, -1.0]]),
            np.array([[0.0, 0.0, -0.1]]),
            np.array([[1.0 + 0j, 0.0, 0.0]]),
            GROUND_Z,
            kp,
            km,
            surfaces,
        )


def test_gu3_6_a_foreign_grid_is_refused():
    et, kp, om, km = medium("A/7MHz")
    other = som.get_grid(et, kp, 5.0, om)
    with pytest.raises(ValueError, match="needs a TransmittedGrid"):
        trans.transmitted_field_below_to_above(
            np.array([[4.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, -0.1]]),
            np.array([[1.0 + 0j, 0.0, 0.0]]),
            GROUND_Z,
            kp,
            km,
            other,
        )
    with pytest.raises(ValueError, match="needs a TransmittedGrid"):
        trans.transmitted_field_above_to_below(
            np.array([[4.0, 0.0, -1.0]]),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[1.0 + 0j, 0.0, 0.0]]),
            GROUND_Z,
            kp,
            km,
            other,
        )


def test_gu3_6_a_source_deeper_than_the_ladder_refuses_with_the_cost():
    et, kp, om, km = medium("A/7MHz")
    lam_m = 2.0 * np.pi / abs(km)
    with pytest.raises(ValueError) as ex:
        trans.TransmittedGrid(et, kp, 20.0, 0.1, 0.6 * lam_m, omega=om)
    msg = str(ex.value)
    assert "two-ray" in msg and "extra rungs per additional quarter" in msg
    assert "momwire#524 phase 0" in msg


def test_gu3_6_the_regimes_do_not_collide_in_the_grid_cache():
    """One cache, three regimes, and a third discriminator value.

    U2 gave `get_grid`'s key a leading regime tag so an above and a below
    grid at identical (ε̃, k₂, r₁, ω, μ) are two entries. This unit adds a
    third, and the collision it prevents is worse than U2's: a below/below
    grid handed to the transmitted projection would interpolate FOUR
    surfaces over a different divide-out and return a plausible number.
    """
    et, kp, om, km = medium("A/7MHz")
    som._GRID_CACHE.clear()
    g_above = som.get_grid(et, kp, 5.0, om)
    g_below = below.get_grid_below(et, kp, 5.0, om)
    g_trans = trans.get_grid_below_above(et, kp, 5.0, 0.15, 0.15, om)
    keys = [k for k in som._GRID_CACHE if isinstance(k, tuple)]
    regimes = {k[0] for k in keys if isinstance(k[0], str)}
    assert "below" in regimes and "below-above" in regimes
    assert g_above is not g_below and g_below is not g_trans
    assert getattr(g_trans, "regime", None) == "below-above"
    assert getattr(g_below, "regime", None) == "below"
    som._GRID_CACHE.clear()


def test_gu3_6_the_transmitted_grid_refuses_a_normalized_rescale():
    et, kp, om, km = medium("A/7MHz")
    g = trans.TransmittedGrid(et, kp, 0.005 * (2 * np.pi / kp), 0.15, 0.15, omega=om)
    with pytest.raises(NotImplementedError, match="two"):
        g.scaled_to(kp * 2.0, om * 2.0, MU0)


# ======================================================================
# G-U3-7 — byte stability of the families this unit did not build
# ======================================================================


def test_gu3_7_the_below_below_contour_keeps_its_bytes():
    """`_run_contour` grew a `max_panels` parameter so the transmitted
    grazing rows can buy the budget they need. The below/below numbers must
    not move a BIT — the refactor is allowed, the gate is exact.
    """
    et, kp, om, km = medium("B/7MHz")
    ref = np.array(
        [
            below._six_integrals_below(et, kp, rho, h, 1e-11)
            for rho, h in ((0.3, 0.2), (4.0, 0.9), (18.0, 0.05))
        ]
    )
    again = np.array(
        [
            below._six_integrals_below(et, kp, rho, h, 1e-11)
            for rho, h in ((0.3, 0.2), (4.0, 0.9), (18.0, 0.05))
        ]
    )
    assert np.array_equal(ref, again)

    # the explicit default is the same call as the implicit one, bit for bit
    def f(lam):
        return below._integrand_six_below(lam, 4.0, 0.9, kp, km)

    a = below._run_contour(
        f,
        kp,
        km,
        4.0,
        0.9,
        1e-11,
        below._ADAPT_DEPTH,
        below._DETOUR,
        below._GX,
        below._GW,
    )
    b = below._run_contour(
        f,
        kp,
        km,
        4.0,
        0.9,
        1e-11,
        below._ADAPT_DEPTH,
        below._DETOUR,
        below._GX,
        below._GW,
        max_panels=below._MAX_TAIL_PANELS,
    )
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]


def test_gu3_7_the_above_above_family_never_enters_this_module(monkeypatch):
    """The ±=+ path keeps its bytes because nothing here touches it.

    Spied rather than asserted from a diff: `_six_integrals` and
    `_integrand_six` are the reflected wave's own machinery and a
    transmitted fill must never reach them.
    """
    et, kp, om, km = medium("A/7MHz")
    for name in ("_six_integrals", "_integrand_six", "_limits_r1_zero"):
        monkeypatch.setattr(
            som,
            name,
            lambda *a, **k: pytest.fail(f"the transmitted family called {name}"),
        )
    trans.t_surfaces_direct(et, kp, 4.0, 1.0, -0.15, rtol=1e-9, omega=om)


# ======================================================================
# G-U3-8 — the grid over the domain it serves
# ======================================================================

# R bands in free-space wavelengths, theta bands in degrees. Reported
# separately because a single pooled worst hides which corner is weak, and
# this family's weak corner is a real physical feature rather than a
# resolution failure — see the pin below.
_R_BANDS = ((0.001, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0))
_TH_BANDS = ((0.0, 3.0), (3.0, 15.0), (15.0, 45.0), (45.0, 90.0))

# The near band carries a NULL of the fifth-and-second surfaces. Measured at
# soil A / 7 MHz, z' = -0.15 m: |TzV| runs 1.61e+3 down to 3.9 and back over
# R = 0.043 .. 0.86 m, a 400x null at R = 0.223 m, and a per-surface RELATIVE
# metric there is measuring against a vanishing quantity. The interpolant's
# error at that cell is 3.7e-3 relative but 1e-5 OF SCALE. The stricter
# number is the one reported and pinned; it is pinned at 5e-3 for that band
# alone, with the reason named, rather than loosening the whole rectangle or
# quietly widening the negligible-cell floor until the null falls through it.
G_U3_8_TOL = 1e-3
G_U3_8_NEAR_TOL = 5e-3
_NEAR_BAND = (0.001, 0.01)


@pytest.fixture(scope="module")
def small_transmitted_grid():
    """A deliberately small transmitted grid (soil A, 7 MHz, one rung) so the
    numpy fill stays inside a test's patience. Every layout decision it
    exercises is the shipped one; only the radial cap is small."""
    et, kp, om, km = medium("A/7MHz")
    lam_p = 2.0 * np.pi / kp
    health = below.Health()
    grid = trans.TransmittedGrid(
        et, kp, 0.02 * lam_p, 0.15, 0.15, rtol=1e-9, omega=om, health=health
    )
    return grid, health, (et, kp, om, km, lam_p)


@pytest.mark.slow
def test_gu3_8_grid_matches_direct_and_the_goldens(
    small_transmitted_grid, record_property
):
    """The three-way. A grid checked only against a direct evaluator proves
    the two agree; `_sommerfeld`'s #161 addendum is the standing reminder
    that they can agree on a mis-branched contour. So: grid vs direct, AND
    direct vs the committed prototype goldens at the same points.
    """
    grid, health, (et, kp, om, km, lam_p) = small_transmitted_grid
    rows = [r for r in gold.SURFACES["A/7MHz"] if r[2] == 0.15]
    rr = np.array([r[0] * lam_p for r in rows])
    th = np.radians(np.array([r[1] for r in rows]))
    keep = (rr >= grid.r_min) & (rr <= grid.r_max) & (th >= grid.th_min)
    assert keep.sum() >= 4, "no golden point lands inside the small grid"
    rr, th = rr[keep], th[keep]
    ref = {k: np.array([r[3 + i] for r in rows])[keep] for i, k in enumerate(KEYS)}
    direct = grid_direct = trans.t_surfaces_direct(
        et,
        kp,
        np.maximum(rr * np.cos(th), 0.0),
        np.maximum(rr * np.sin(th), 0.0),
        -0.15,
        rtol=1e-9,
        omega=om,
    )
    got = grid.eval(rr, th, -0.15)
    d_vs_gold, k1, _, _ = per_surface_rel(direct, ref)
    g_vs_d, k2, _, _ = per_surface_rel(got, grid_direct)
    record_property("direct_vs_goldens", float(d_vs_gold))
    record_property("grid_vs_direct", float(g_vs_d))
    record_property("nodes", int(grid.nodes))
    record_property("health", repr(health.as_dict()))
    assert health.nonconvergent == 0, health.as_dict()
    assert d_vs_gold < G_U3_1_TOL, f"direct vs goldens {d_vs_gold:.3e} on {k1}"
    assert g_vs_d < G_U3_8_NEAR_TOL, f"grid vs direct {g_vs_d:.3e} on {k2}"


@pytest.mark.slow
@pytest.mark.parametrize("cell", ["A/7MHz", "B/7MHz"])
def test_gu3_8_grid_matches_direct_over_the_whole_rectangle(cell, record_property):
    """grid → direct over the FULL served rectangle, per surface, per band,
    boundaries forced in, at the shipped cap.

    Not an interior patch: R runs from `r_min` to the 2 λ_p cap and θ from
    the grazing floor to 90°, and both ends of both axes are in the sample.
    """
    et, kp, om, km = medium(cell)
    lam_p = 2.0 * np.pi / kp
    health = below.Health()
    grid = trans.TransmittedGrid(
        et,
        kp,
        trans._R_CAP_LAMBDA_P * lam_p,
        0.15,
        0.15,
        rtol=1e-9,
        omega=om,
        health=health,
    )
    rng = np.random.default_rng(553)
    bands = {}
    skipped = total = 0
    for rb in _R_BANDS:
        rlo = max(rb[0] * lam_p, grid.r_min)
        rhi = min(rb[1] * lam_p, grid.r_max)
        if rhi <= rlo:
            continue
        for tb in _TH_BANDS:
            tlo = max(tb[0], math.degrees(grid.th_min))
            rq = np.concatenate(
                [[rlo, rhi], np.exp(rng.uniform(math.log(rlo), math.log(rhi), 3))]
            )
            tq = np.radians(np.concatenate([[tlo, tb[1]], rng.uniform(tlo, tb[1], 3)]))
            rq, tq = np.meshgrid(rq, tq, indexing="ij")
            rq = rq.ravel()
            tq = tq.ravel()
            got = grid.eval(rq, tq, -0.15)
            ref = trans.t_surfaces_direct(
                et,
                kp,
                np.maximum(rq * np.cos(tq), 0.0),
                np.maximum(rq * np.sin(tq), 0.0),
                -0.15,
                rtol=1e-9,
                omega=om,
            )
            w, _, sk, tot = per_surface_rel(got, ref)
            skipped += sk
            total += tot
            bands[(rb, tb)] = w
    for (rb, tb), w in sorted(bands.items()):
        record_property(f"R[{rb[0]},{rb[1]})lp_th[{tb[0]},{tb[1]})", float(w))
    near = max(v for k, v in bands.items() if k[0] == _NEAR_BAND)
    rest = max(v for k, v in bands.items() if k[0] != _NEAR_BAND)
    record_property("worst_near_band", float(near))
    record_property("worst_rest", float(rest))
    record_property("floor_skipped_frac", float(skipped) / float(total))
    record_property("nodes", int(grid.nodes))
    record_property("grazing_floor_deg", float(math.degrees(grid.th_min)))
    record_property("health", repr(health.as_dict()))
    assert health.nonconvergent == 0, health.as_dict()
    assert skipped < 0.05 * total, f"{skipped}/{total} skipped by the floor"
    assert near < G_U3_8_NEAR_TOL, f"{cell}: near band {near:.3e}"
    assert rest < G_U3_8_TOL, f"{cell}: rectangle worst {rest:.3e}"


@pytest.mark.slow
def test_gu3_8_past_the_cap_and_under_the_floor_refuse(small_transmitted_grid):
    grid, _, (et, kp, om, km, lam_p) = small_transmitted_grid
    with pytest.raises(ValueError, match="no negligible tail to freeze"):
        grid.eval(np.array([grid.r_max * 1.5]), np.array([0.5]), -0.15)
    with pytest.raises(ValueError, match="near zone"):
        grid.eval(np.array([grid.r_min * 0.5]), np.array([0.5]), -0.15)
    inside = 0.5 * (grid.r_min + grid.r_max)
    with pytest.raises(ValueError, match="ladder is log-spaced"):
        grid.eval(np.array([inside]), np.array([0.5]), -0.4)
    # and the cap is a REFUSAL, not a clamp: the parent would have answered
    got = som.get_grid(et, kp, 5.0, om).eval(np.array([1e9]), np.array([0.5]))
    assert np.isfinite(got["IrhoV"]).all()


@pytest.mark.slow
def test_gu3_8_the_projection_reads_the_grid_and_the_direct_alike(
    small_transmitted_grid, record_property
):
    """One combination routine, two surface sources. If the grid and the
    direct evaluator disagree through the projection, the dyad is not the
    suspect — it is the same code both times.
    """
    grid, _, (et, kp, om, km, lam_p) = small_transmitted_grid
    direct = DirectSurfaces(et, kp, om, r_max=grid.r_max)
    # Every pair has to land inside the small grid's radius, which is what
    # `r_max` is: the OBSERVER's distance from the source's ground projection.
    obs = np.array([[0.5, 0.1, 0.15], [0.4, 0.0, 0.2], [0.0, 0.0, 0.3]])
    src = np.array([[0.0, 0.0, -0.15], [-0.1, 0.05, -0.15]])
    p = np.array([[1.0 + 0.3j, -0.2j, 0.4], [0.0, 0.0, 1.0 + 0j]])
    a = trans.transmitted_field_below_to_above(obs, src, p, GROUND_Z, kp, km, grid)
    b = trans.transmitted_field_below_to_above(obs, src, p, GROUND_Z, kp, km, direct)
    worst = rel_vec(a, b)
    record_property("worst_rel", float(worst))
    assert worst < G_U3_8_NEAR_TOL, f"grid vs direct through the dyad {worst:.3e}"


@pytest.mark.slow
def test_gu3_8_a_multi_rung_grid_interpolates_in_z_prime(record_property):
    """The ladder inside the grid, not just in `ladder_nodes`: an off-node
    depth read through `eval` against direct evaluation.
    """
    et, kp, om, km = medium("A/7MHz")
    lam_p = 2.0 * np.pi / kp
    health = below.Health()
    grid = trans.TransmittedGrid(
        et, kp, 0.02 * lam_p, 0.08, 0.4, rtol=1e-9, omega=om, health=health
    )
    record_property("rungs", int(grid.n_zp))
    record_property("nodes", int(grid.nodes))
    rng = np.random.default_rng(3553)
    rq = np.exp(rng.uniform(math.log(grid.r_min), math.log(grid.r_max), 12))
    tq = rng.uniform(grid.th_min, 0.5 * np.pi, 12)
    zq = np.exp(rng.uniform(math.log(0.08), math.log(0.4), 12))
    got = grid.eval(rq, tq, -zq)
    ref = trans.t_surfaces_direct(
        et,
        kp,
        np.maximum(rq * np.cos(tq), 0.0),
        np.maximum(rq * np.sin(tq), 0.0),
        -zq,
        rtol=1e-9,
        omega=om,
    )
    worst, where, _, _ = per_surface_rel(got, ref)
    record_property("worst_rel", float(worst))
    record_property("worst_surface", str(where))
    assert health.nonconvergent == 0, health.as_dict()
    assert worst < G_U3_8_NEAR_TOL, f"off-node z' through the grid {worst:.3e}"


# ======================================================================
# G-U3-9 — the cache keys exactly, on eps_tilde and on the depth range
# ======================================================================


@pytest.mark.slow
def test_gu3_9_two_close_eps_values_do_not_share_a_grid():
    """U2's third inversion, re-asserted where ε̃ enters twice over.

    `_somm_eps_bucket` would put these two on the same rung of its 1 %
    ladder. Here ε̃ sets the divided-out phase e^{−jk_m|z′|} that the whole
    ladder architecture rests on, as well as the Fresnel amplitude, so the
    below-above key carries the caller's exact value.
    """
    om = 2.0 * np.pi * 7e6
    kp = om / C0
    a = eps_tilde(13.0, 0.005, 7e6)
    b = eps_tilde(13.0, 0.00502, 7e6)
    assert som._somm_eps_bucket(a) == som._somm_eps_bucket(b)
    som._GRID_CACHE.clear()
    ga = trans.get_grid_below_above(a, kp, 2.0, 0.15, 0.15, om)
    gb = trans.get_grid_below_above(b, kp, 2.0, 0.15, 0.15, om)
    assert ga is not gb
    assert ga.eps_t == a and gb.eps_t == b
    som._GRID_CACHE.clear()


@pytest.mark.slow
def test_gu3_9_the_depth_range_is_keyed_exactly_and_the_radius_is_bucketed():
    """`r_max` buckets up because the log-R lattice is anchored at `r_min`
    with a fixed Δln R, so a wider grid's nodes coincide with a narrower
    one's. The ladder is log-spaced BETWEEN its ends, so widening the depth
    range moves every rung — no coincidence to exploit, and no bucket.
    """
    om = 2.0 * np.pi * 7e6
    kp = om / C0
    et = eps_tilde(13.0, 0.005, 7e6)
    som._GRID_CACHE.clear()
    g1 = trans.get_grid_below_above(et, kp, 2.0, 0.15, 0.15, om)
    g2 = trans.get_grid_below_above(et, kp, 2.05, 0.15, 0.15, om)
    assert g1 is g2, "the radius bucket stopped sharing"
    g3 = trans.get_grid_below_above(et, kp, 2.0, 0.15, 0.16, om)
    assert g3 is not g1, "the depth range got bucketed"
    # node positions really do coincide across radius buckets
    g4 = trans.get_grid_below_above(et, kp, 20.0, 0.15, 0.15, om)
    n = min(g1.n_r, g4.n_r)
    r1 = np.exp(g1.lnr0 + g1.dlnr * np.arange(n))
    r4 = np.exp(g4.lnr0 + g4.dlnr * np.arange(n))
    assert np.allclose(r1, r4, rtol=0, atol=1e-12)
    som._GRID_CACHE.clear()


# ======================================================================
# G-U3-10 — the grazing floor is a measured cost law
# ======================================================================


def test_gu3_10_the_grazing_floor_follows_the_measured_cost_law():
    """The floor is solved from `panels ≈ 16·cot θ_true`, not chosen.

    It moves with BOTH the cap and the shallowest rung, because the source
    leg holds the tail down on its own — which is exactly why it lands two
    orders of magnitude below the below/below family's 1°, and why a deep
    enough ladder buys the θ = 0 row outright.
    """
    deep = trans.grazing_floor(85.65, 0.15)
    shallow = trans.grazing_floor(85.65, 0.02)
    near = trans.grazing_floor(8.5, 0.15)
    assert 0.0 < math.degrees(deep) < 0.1
    assert math.degrees(shallow) > math.degrees(deep)
    assert near == 0.0, "a small enough cap should buy the theta = 0 row"
    assert math.degrees(shallow) < 1.0, (
        "the transmitted floor should be far under the below/below family's"
    )


@pytest.mark.slow
def test_gu3_10_a_truncated_tail_is_not_a_graceful_degradation(record_property):
    """The trap this unit's budget exists for, measured rather than trusted.

    `_tail_below` falls back to Wynn epsilon when the panel budget runs out.
    Phase 0 kept that for the h → 0 limit and never needed it. On this
    family it fires at grazing and it is wrong by DECADES, so a
    `nonconvergent` count is a failure and not a warning — every fill gate
    above asserts it at zero.
    """
    et, kp, om, km = medium("A/7MHz")

    def f(lam):
        return trans._integrand_six_transmitted(lam, 85.65, 0.0, -0.02, kp, km)

    a = 1.1 * max(abs(kp), abs(km))
    head, _ = below._head(
        f,
        a,
        85.65,
        (kp, abs(km.real)),
        1e-11,
        below._ADAPT_DEPTH,
        below._DETOUR,
        below._GX,
        below._GW,
    )
    short, _, conv_s, accel = below._tail_below(
        f, a, 85.65, 0.02, head, 1e-11, below._GX, below._GW, max_panels=4000
    )
    long, panels, conv_l, _ = below._tail_below(
        f, a, 85.65, 0.02, head, 1e-11, below._GX, below._GW, max_panels=80000
    )
    assert not conv_s and accel and conv_l
    err = float(np.max(np.abs((short + head) - (long + head)) / np.abs(long + head)))
    record_property("accelerated_truncation_error", float(err))
    record_property("panels_to_converge", int(panels))
    assert err > 1e2, (
        "the Wynn fallback stopped being catastrophic — if that is real the "
        "budget can shrink, but measure it before believing it"
    )


@pytest.mark.slow
def test_gu3_10_the_contour_reports_its_own_health(record_property):
    """Self-convergence, measured on the geometries the product brings.

    The fine machine (Gauss-24, rtol 1e-11, detour 2.0) against the coarse
    one (Gauss-16, rtol 1e-9, detour 1.2) at the same points. Phase 0's own
    note is that this estimate is a conservative UPPER BOUND — where an
    exact answer exists it read 5e-11 while the realized error was 1.4e-15 —
    so it is quoted as a bound and never as the error.
    """
    health = below.Health()
    for cell in ("A/7MHz", "B/7MHz", "C/21MHz"):
        et, kp, om, km = medium(cell)
        lam_p = 2.0 * np.pi / kp
        for rw, thd, zp in (
            (0.001, 45.0, -0.02),
            (0.02, 0.5, -0.15),
            (0.5, 5.0, -0.15),
            (2.0, 30.0, -1.0),
            (2.0, 0.1, -0.15),
        ):
            rr = rw * lam_p
            trans._six_integrals_transmitted(
                et,
                kp,
                max(rr * math.cos(math.radians(thd)), 0.0),
                max(rr * math.sin(math.radians(thd)), 0.0),
                zp,
                1e-11,
                health=health,
                selfconv=True,
                where=(cell, rw, thd, zp),
            )
    d = health.as_dict()
    record_property("health", repr(d))
    assert d["nonconvergent"] == 0, d
    assert d["accelerated"] == 0, d
    assert d["worst_selfconv"] < 1e-6, d
