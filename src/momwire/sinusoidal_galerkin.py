"""Richmond-style piecewise-sinusoidal **Galerkin** solver (momwire#182).

`SinusoidalGalerkinSolver` shares everything geometric and basis-related with
`SinusoidalSolver` — the same three-term sinusoidal basis, the same N⁻/N⁺
neighbour tables that bake junction continuity / KCL into the basis-function
shapes — and differs in ONE place: how the EFIE is tested.

`SinusoidalSolver` collocates: it point-samples ŝ·E at each segment centre
(one test row per segment). This solver instead forms each test row by
**integrating** the field against the corresponding basis function, by Gauss
quadrature along the test segment(s). That makes it the missing
basis × testing cell (sinusoidal basis + Galerkin testing) beside the
point-matched sinusoidal solver and the B-spline Galerkin family.

Fill scheme (issue #182, chosen empirically in M1 — "let G1 decide"): reuse
the existing closed-form Eqs 76-79 field evaluators for the SOURCE integral
(analytic over each source segment) and Gauss-quadrature only the TEST
integral. M1 established that this is a structurally correct Galerkin fill:
‖Z − Zᵀ‖/‖Z‖ is exact on the diagonal and at roundoff for well-separated
pairs.

**M2 — the near-singular test integral.** M1 used one uniform Gauss rule for
every pair, which is the wrong tool for the pairs that touch. When the test
segment shares a node with the source segment (or IS the source segment), the
integrand carries a spike of width ~`a` (the thin-wire radius) located ~`a`
inside the shared endpoint: the Eqs 78-79 endpoint terms go as
Δz·(1+jkr₀)·G₀/r₀² with r₀ = √(a² + Δz²), which peaks at |Δz| ≈ a/√2 with
height ~1/a². A uniform rule cannot see a feature of relative width a/h until
it has ~h/a nodes — which is exactly why M1 needed `n_qp_test` ≈ 128 to reach
the G1 gate, and paid that cost on every one of the N² pairs.

M2 splits the quadrature instead of refining it globally:

* **Far pairs** keep one shared uniform rule (`n_qp_test`, default 8). Their
  integrands vary on the scale of the pair separation, so 8 nodes is already
  converged to roundoff.
* **Near pairs** — self, node-sharing neighbours, and any pair whose segments
  approach within `near_factor` of their combined half-lengths — are re-integrated
  on a composite rule whose panels shrink dyadically toward BOTH endpoints of
  the test segment, down to relative width `a/h` (`_graded_endpoint_rule`).
  Geometric grading plus a fixed per-panel Gauss order is the standard
  exponentially-convergent treatment for an endpoint-localized feature, and
  the panel count is *derived* from the geometry's own a/h — not tuned.

Because the near set is O(N) pairs rather than O(N²), the correction is
evaluated through `_field_components_bcast`, which pairs (P, G) quadrature
points against (P, 1) sources directly instead of forming the N² table those
pairs live in. Net effect: the G1 symmetry gate is met at the DEFAULT
quadrature, at ~1/16 the fill cost of M1's converged setting.

**M3 — the variational payoff, measured.** At 11-21 segments this solver's
driving-point impedance is closer to the fine-mesh answer than the
point-matched solver's, by 1.01× (k3_star) to 2.97× (k2_junction) worst-case
over a family of six references. Two things worth knowing when reading those
numbers, both pinned by tests in `tests/test_sinusoidal_galerkin.py`:

* The win is a **reactance** win. In R the point-matched solver actually
  converges faster on three of the four validation geometries; the net |Z|
  verdict is a reactance gain partly given back on resistance. That is the
  useful direction — the sin↔bs2 questions this solver exists to arbitrate are
  reactance-dominated.
* There is no "converged impedance" to measure against in an absolute sense:
  the delta-gap width IS the segment length, so refining the mesh shrinks the
  source, and X drifts logarithmically without limit while R settles. Every
  error figure above is relative to a *chosen* fine-mesh reference, and the
  gate is the verdict that survives the whole spread of defensible choices.

**M4 — ground models, by reuse.** All three of the point-matched solver's
grounds are wired through the same test quadrature, with NO new field
kernels: each one is the existing per-ground source-field evaluator called
with the Gauss points along the test segment as its observers instead of the
segment centres.

* **PEC image** — the mirrored-source build (`_image_source_centers_tangents`)
  through the same Eqs 76-79 evaluator, plain tangential projection.
* **Reflection-coefficient** (`ground_eps`) — the same mirrored build with
  NEC's Fresnel field dyad applied before the projection. The dyad's specular
  tables (cos θ, p̂) are per (observer, source) *pair*, so they are rebuilt at
  the quadrature points via `_ground_refl.specular_ray_tables_bcast` rather
  than reused from the segment-centre cache.
* **Sommerfeld** (`ground_model="sommerfeld"`) — NEC's C2·(PEC image) plus the
  smooth interpolated remainder, the latter from the point-matched solver's
  `_field_tensor_sommerfeld_remainder` with its observer set overridden to the
  quadrature points.

The image blocks get the SAME graded near-pair treatment as the free-space
block, selected against the mirrored source geometry — which matters exactly
when a wire touches or nearly touches the plane, because then a segment and
its own image share the endpoint that carries M2's width-`a` spike.

Still deferred: wire loading and junction ports (M5), and the C++ accelerator
— all raise rather than return plausible wrong numbers.
"""

import numpy as np
import scipy.linalg
import scipy.spatial.distance

from . import _ground_refl
from .sinusoidal import SinusoidalSolver

# Pairs are corrected in blocks so the (P, G, n_qp_const) source-quadrature
# scratch inside the field kernel stays bounded regardless of model size.
_PAIR_BLOCK = 512


def _graded_endpoint_rule(eps, n_per_panel, leggauss):
    """Composite Gauss rule on [-1, 1] with panels graded toward BOTH ends.

    Panel widths double away from each endpoint — eps, 2·eps, 4·eps, … — so a
    feature of relative width `eps` sitting at an endpoint is resolved by a
    panel of its own size, and the smooth interior is covered by a handful of
    large panels. Returns (nodes, weights) with Σw = 2.

    `eps` is the feature width relative to the segment half-length (a/h for
    the thin-wire endpoint spike); `leggauss` is the solver's cached
    Gauss-Legendre factory.
    """
    eps = float(np.clip(eps, 1e-9, 0.5))
    # Distances from the endpoint at which panels break. Stop at the midpoint
    # so the largest panel is [-0.5, 0] and the grading ratio stays ~2.
    d = eps * 2.0 ** np.arange(64)
    d = d[d <= 0.5]
    left = np.concatenate(([-1.0], -1.0 + d, [0.0]))
    edges = np.unique(np.concatenate([left, -left[::-1]]))

    gx, gw = leggauss(n_per_panel)
    lo, hi = edges[:-1], edges[1:]
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    x = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    w = (half[:, None] * gw[None, :]).ravel()
    return x, w


def _plain_projection(cm, m_idx, n_idx):
    """Tangential projection of the Eqs 76-79 component tables — NEC's rule
    E_t = (t_obs·t_src)·E_z + ((ρ⃗·t_obs)/ρ')·E_ρ, the same expression
    `SinusoidalSolver._field_tensor`'s numpy path uses.

    Serves the free-space block and the PEC image block; the (test-segment,
    source-segment) index arrays are the projector-protocol signature (see
    `SinusoidalGalerkinSolver._tested_contribs`) and are unused here — only
    the Fresnel-weighted projector needs to look anything up per pair.
    """
    td = cm["td"]
    rp = cm["rho_proj_factor"]
    return (
        td * cm["Ez_const"] + rp * cm["Erho_const"],
        td * cm["Ez_sin"] + rp * cm["Erho_sin"],
        td * cm["Ez_cos"] + rp * cm["Erho_cos"],
    )


class SinusoidalGalerkinSolver(SinusoidalSolver):
    """Piecewise-sinusoidal Galerkin MoM. Same constructor surface as
    `SinusoidalSolver`, plus the test-quadrature controls:

    `n_qp_test`
        Gauss nodes per test segment for the shared FAR-pair rule.
    `n_qp_near`
        Gauss nodes per panel of the graded near-pair rule (M2). The default 8
        is where the G1 residual stops being near-pair quadrature error and
        hits the source-side `n_qp_const` floor (~2e-11) on every validation
        geometry; 4 leaves the bent/junction cases short of the gate.
    `near_factor`
        A pair is "near" when its segments approach within
        `near_factor · (h_m + h_n)/2`. The default 0.5 selects the self and
        node-sharing pairs (both at distance 0) while excluding the
        next-nearest collinear neighbour (at distance h), and additionally
        catches genuinely close non-touching wires.
    `near_correction`
        Set False to fall back to M1 behaviour (one uniform rule everywhere)
        — the contrast the M2 tests use to show the correction is what buys
        the symmetry gate.

    All three ground models are wired (M4): `ground_z` alone gives the PEC
    image, `+ ground_eps` NEC's reflection-coefficient ground, and
    `+ ground_model="sommerfeld"` the Sommerfeld/Norton ground — each by
    reusing that ground's existing source-field evaluator under this test
    quadrature. One inherited caveat, shared with the point-matched solver and
    pinned by `test_finite_ground_at_a_ground_contact_is_an_inherited_defect`:
    a wire END LYING IN the plane is only sound under the PEC ground, because
    #151's ground-connected basis completes the end current with an exact
    mirror image that a finite ground does not provide.

    Wire loading, junction ports (M5), and the C++ accelerator are deliberately
    not wired yet and raise where reached.
    """

    def __init__(
        self,
        *,
        n_qp_test=8,
        n_qp_near=8,
        near_factor=0.5,
        near_correction=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_qp_test = int(n_qp_test)
        self.n_qp_near = int(n_qp_near)
        self.near_factor = float(near_factor)
        self.near_correction = bool(near_correction)

    # ------------------------------------------------------------------
    # Near-pair selection
    # ------------------------------------------------------------------

    def _near_pairs(self, geom, src_c=None, src_t=None, n_samples=5):
        """Ordered (test, source) segment pairs whose test integral needs the
        graded rule, as two index arrays.

        Selection is geometric rather than topological, so it covers the self
        pair, node-sharing neighbours, AND close-approach pairs that share no
        node (closely spaced parallel runs) — all of which put a sub-segment
        length scale into the test integrand.

        `src_c` / `src_t` default to the geometry's own segments (the
        free-space block). The M4 image blocks pass the MIRRORED source
        geometry, which is what makes the correction fire for a wire touching
        the ground plane: there the end segment and its own image share the
        in-plane endpoint, so the image block carries the same width-`a`
        endpoint spike the free-space self pair does. Image half-lengths are
        unchanged by mirroring, so `reach` is still the geometry's own.

        Exact segment-segment distance is only computed for pairs that survive
        a cheap centre-distance prefilter. That prefilter is a strict superset:
        |c_m − c_n| ≤ d_min + h_m + h_n always, so any pair meeting the
        `near_factor` criterion also meets the prefilter.
        """
        c = geom["seg_centers"]
        t = geom["seg_tangents"]
        cs = c if src_c is None else src_c
        ts = t if src_t is None else src_t
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)
        reach = hh[:, None] + hh[None, :]

        # cdist rather than an explicit (N, N, 3) difference: the prefilter
        # should not cost 3× the memory of the matrix it is protecting.
        dc = scipy.spatial.distance.cdist(c, cs)
        cand = np.argwhere(dc <= (1.0 + self.near_factor) * reach)
        m, n = cand[:, 0], cand[:, 1]

        s = np.linspace(-1.0, 1.0, n_samples)

        def _gap(ai, bi, ca, ta, cb, tb):
            """min over sample points of segment `ai` (geometry ca/ta) of the
            distance to the clamped axis of segment `bi` (geometry cb/tb)."""
            pts = (
                ca[ai][:, None, :]
                + (hh[ai][:, None] * s[None, :])[:, :, None] * ta[ai][:, None, :]
            )  # (P, n_samples, 3)
            d = pts - cb[bi][:, None, :]
            u = np.einsum("pgd,pd->pg", d, tb[bi])
            u = np.clip(u, -hh[bi][:, None], hh[bi][:, None])
            perp = d - u[..., None] * tb[bi][:, None, :]
            return np.linalg.norm(perp, axis=-1).min(axis=1)

        gap = np.minimum(
            _gap(m, n, c, t, cs, ts),  # test segment sampled against the source
            _gap(n, m, cs, ts, c, t),  # and the other way round
        )
        keep = gap <= self.near_factor * reach[m, n]
        return m[keep], n[keep]

    # ------------------------------------------------------------------
    # Galerkin matrix assembly
    # ------------------------------------------------------------------

    def _test_context(self, geom, seg_view, k):
        """Everything the TEST side of the Galerkin integral needs, built once
        per assembly and shared by the free-space block and every ground block.

        Holds the Gauss points along each test segment (as a flat (N·nq, 3)
        observer list), the per-observer thin-wire radius, and `w_entry` —
        the (nnz, nq) table of test-function value × arc weight for every
        (segment, basis) support entry. For each such entry the test
        function's value at quad node q is
        f_{i,m}(ξ_q) = σA + B·sin(kξ_q) + σC·cos(kξ_q) (the current-shape
        convention pinned in `_evaluate_basis_at_points`) and the arc weight
        is gw_q·(h_m/2).
        """
        N = geom["n_segs"]
        seg_c = geom["seg_centers"]  # (N, 3)
        seg_t = geom["seg_tangents"]  # (N, 3)
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)  # (N,) half-lengths

        gx, gw = self._leggauss_cached(self.n_qp_test)  # nodes on [-1, 1]
        nq = gx.shape[0]
        xi = hh[:, None] * gx[None, :]  # (N, nq) local arc from each centre
        obs_pts = seg_c[:, None, :] + xi[:, :, None] * seg_t[:, None, :]  # (N,nq,3)
        obs_c = np.ascontiguousarray(obs_pts.reshape(N * nq, 3))
        obs_t = np.ascontiguousarray(
            np.broadcast_to(seg_t[:, None, :], (N, nq, 3)).reshape(N * nq, 3)
        )
        a_seg = None if self._uniform_radius is not None else self._seg_radius(geom)
        # Scalar on the uniform path, (N·nq, 1) per-observer column when mixed
        # — the same shape contract `_field_components_bcast` takes for `a`.
        a_obs = self._uniform_radius if a_seg is None else np.repeat(a_seg, nq)[:, None]

        starts = seg_view["starts"]
        counts = np.diff(starts)
        m_of_entry = np.repeat(np.arange(N, dtype=np.int64), counts)  # (nnz,)
        sig = seg_view["sigma"].astype(np.complex128)
        sigA = sig * seg_view["A"]
        B = seg_view["B"]
        sigC = sig * seg_view["C"]
        fval = (
            sigA[:, None]
            + B[:, None] * np.sin(k * xi)[m_of_entry]
            + sigC[:, None] * np.cos(k * xi)[m_of_entry]
        )  # (nnz, nq)
        w_entry = (gw[None, :] * hh[m_of_entry][:, None]) * fval  # (nnz, nq)

        return {
            "N": N,
            "nq": nq,
            "hh": hh,
            "obs_c": obs_c,
            "obs_t": obs_t,
            # which test segment each flat observer belongs to — the row
            # coordinate a per-segment-pair projector table indexes with.
            "m_of_obs": np.repeat(np.arange(N, dtype=np.int64), nq),
            "a_obs": a_obs,
            "a_seg": a_seg,
            "starts": starts,
            "counts": counts,
            "m_of_entry": m_of_entry,
            "i_of_entry": seg_view["jbasis"],
            "sigA": sigA,
            "B": B,
            "sigC": sigC,
            "w_entry": w_entry,
        }

    def _tested_contrib(self, ctx, Phi):
        """contrib[entry, n] = Σ_q w_entry[entry, q] · Phi[m(entry), q, n].

        Accumulated one quadrature node at a time: the equivalent einsum over
        `Phi[m_of_entry]` would first materialize an (nnz, nq, N) gather, nq×
        the peak memory for the same arithmetic.
        """
        w_entry = ctx["w_entry"]
        m_of_entry = ctx["m_of_entry"]
        out = np.zeros((w_entry.shape[0], ctx["N"]), dtype=np.complex128)
        for q in range(ctx["nq"]):
            out += w_entry[:, q, None] * Phi[m_of_entry, q, :]
        return out

    def _tested_contribs(self, geom, k, ctx, projector, src_c=None, src_t=None):
        """Test-integrate one source block: (contrib_const, sin, cos), each
        (nnz, N).

        `src_c` / `src_t` are the source geometry the field evaluator sees —
        the geometry's own segments for the free-space block, the mirrored
        ones for a ground image block.

        `projector(cm, m_idx, n_idx)` turns the unprojected Eqs 76-79
        component tables into the three tangential field tables. `m_idx` /
        `n_idx` are broadcastable index arrays naming the (test segment,
        source segment) each entry of `cm` belongs to, so a projector can
        look up per-segment-pair tables regardless of how the caller paired
        things: (M, 1) against (1, N) for the full block here, (P, 1) against
        (P, 1) for the near-pair path's one-to-one pairing.

        Two-tier quadrature (M2): a shared uniform rule for the far pairs,
        overwritten per near pair by the endpoint-graded rule.
        """
        N = ctx["N"]
        nq = ctx["nq"]
        src_c = geom["seg_centers"] if src_c is None else src_c
        src_t = geom["seg_tangents"] if src_t is None else src_t

        cm = self._field_components_bcast(
            k,
            obs_c=ctx["obs_c"][:, None, :],  # (M, 1, 3)
            obs_t=ctx["obs_t"][:, None, :],  # (M, 1, 3)
            a=ctx["a_obs"],
            src_c=src_c[None, :, :],
            src_t=src_t[None, :, :],
            src_hh=ctx["hh"][None, :],
        )
        Phi = projector(cm, ctx["m_of_obs"][:, None], np.arange(N)[None, :])
        contribs = tuple(self._tested_contrib(ctx, P.reshape(N, nq, N)) for P in Phi)

        if self.near_correction:
            self._apply_near_correction(geom, k, ctx, contribs, projector, src_c, src_t)
        return contribs

    def _refl_projection(self, geom, eps_t):
        """Projector applying NEC's Fresnel field dyad to the image-source
        field before the tangential projection — the `ground_eps`
        reflection-coefficient ground (IPERF=0).

        Identical algebra to `SinusoidalSolver._field_tensor_image_refl`'s
        numpy path,

            t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂),

        reading the specular tables straight out of `_image_refl_prep` — the
        same per-SEGMENT-PAIR cache the point-matched solver uses. That is
        deliberate and it is what NEC's IPERF=0 model actually says: ρ_v/ρ_h
        are evaluated once at the midpoint-to-image-midpoint specular angle
        and held **constant over the segment pair**. The test quadrature
        integrates the field, not the reflection coefficient.

        The alternative — re-deriving the specular ray at each test
        quadrature point — was implemented and rejected on measurement: it
        makes the weight a function of where on the test segment you are,
        which no longer has a partner term in the transposed entry, and the
        Galerkin matrix loses reciprocity (‖G−Gᵀ‖/‖G‖ = 3.0e-9 on the
        h=0.1λ dipole, 2.3e-9 on the L-shape, and *insensitive to
        quadrature refinement* — 3.00e-9 at n_qp_test 8, 16 and 32, i.e.
        structural). The pair-constant form below sits at the free-space
        floor instead. Since the observers are points ON test segment m,
        `tm_p = t_m·p̂` from the segment tangents is exact for them, not an
        approximation.
        """
        cos_th, px, py, tm_p, tn_p = self._image_refl_prep(geom)
        rho_v_t, rho_h_t = _ground_refl.fresnel_rho(eps_t, cos_th)

        def _project(cm, m_idx, n_idx):
            rho_v = rho_v_t[m_idx, n_idx]
            rvh = rho_v + rho_h_t[m_idx, n_idx]  # → 0 in the PEC limit
            tmp = tm_p[m_idx, n_idx]
            tnp = tn_p[m_idx, n_idx]
            # ρ̂·p̂ from the image-build rho_vec (p̂ is horizontal, so only the
            # x/y components contribute), same radius-regularized denominator
            # as the E_ρ projection rule.
            rho_p = (
                cm["rho_vec"][..., 0] * px[m_idx, n_idx]
                + cm["rho_vec"][..., 1] * py[m_idx, n_idx]
            ) / cm["rho_eval"]
            td = cm["td"]
            rp = cm["rho_proj_factor"]

            def _weighted(Ez, Erho):
                tm_E = td * Ez + rp * Erho  # t_m · E
                E_p = tnp * Ez + rho_p * Erho  # E · p̂
                return rho_v * tm_E - rvh * tmp * E_p

            return (
                _weighted(cm["Ez_const"], cm["Erho_const"]),
                _weighted(cm["Ez_sin"], cm["Erho_sin"]),
                _weighted(cm["Ez_cos"], cm["Erho_cos"]),
            )

        return _project

    def _tested_ground_block(self, geom, k, ctx):
        """The ground sub-assembly, tested exactly like the free-space block
        and SUBTRACTED from it by the caller — the same single global minus
        sign the point-matched `_assemble_Z` uses (the image current + image
        charge sign flips reduce to it).

        Reuse, per ground, of the evaluator the point-matched solver already
        validated against its own references:

        * PEC image — the mirrored-source Eqs 76-79 build, plain projection;
        * `ground_eps` refl-coef — the same build with the Fresnel dyad
          (`_refl_projection`);
        * `ground_model="sommerfeld"` — NEC's decomposition, C2·(PEC image)
          minus the smooth interpolated remainder, so that the caller's
          subtraction reproduces `Phi_free − C2·Phi_img + S`.
        """
        src_c_img, src_t_img = self._image_source_centers_tangents(geom)
        if self.ground_eps is None:
            return self._tested_contribs(
                geom, k, ctx, _plain_projection, src_c_img, src_t_img
            )

        eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)
        if self.ground_model == "sommerfeld":
            c2 = (eps_t - 1.0) / (eps_t + 1.0)
            img = self._tested_contribs(
                geom, k, ctx, _plain_projection, src_c_img, src_t_img
            )
            rem = self._tested_sommerfeld_remainder(geom, k, ctx, eps_t)
            return tuple(c2 * a - b for a, b in zip(img, rem))

        return self._tested_contribs(
            geom, k, ctx, self._refl_projection(geom, eps_t), src_c_img, src_t_img
        )

    def _tested_sommerfeld_remainder(self, geom, k, ctx, eps_t):
        """Test-integrate the smooth Sommerfeld remainder tensor.

        The remainder evaluator is the point-matched solver's, called with the
        test-quadrature points as its observer set. No near-pair correction:
        the remainder kernel lives on the distance to the IMAGE point, which
        stays smooth even where a wire touches the plane (r₁ → 0 has a finite
        limit the grid carries), so the width-`a` endpoint spike the graded
        rule exists for is absent here. The residual that IS left on a
        ground-touching wire is this block's own SOURCE-side rule
        (`n_qp_sommerfeld`), which
        `test_g4_sommerfeld_symmetry_near_the_plane_is_source_quadrature_limited`
        identifies by showing that refining the test rule does not move it
        while refining the source rule drives it to the free-space floor.
        """
        N = ctx["N"]
        nq = ctx["nq"]
        S = self._field_tensor_sommerfeld_remainder(
            geom, k, eps_t, obs_centers=ctx["obs_c"], obs_tangents=ctx["obs_t"]
        )
        return tuple(self._tested_contrib(ctx, s.reshape(N, nq, N)) for s in S)

    def _assemble_Z(self, geom, k):
        """Galerkin system matrix G (basis i tested against source basis j).

        G[i, j] = ∫ f_i(s) · ŝ·E_j(s) ds
                = Σ_shape ( T[shape] @ M[shape] )[i, j]

        where T[shape, i, n] integrates the closed-form field of unit
        `shape` (const/sin/cos) current on source segment n against test
        basis i (quadrature along i's support), and M[shape][n, j] is the
        effective coefficient source basis j puts on segment n — the SAME
        source-side coefficient matrices the collocation path builds.

        With `ground_z` set, the ground's tested sub-assembly is subtracted
        from the free-space one before the scatter — one more source block
        through the same test quadrature, not a second scheme.
        """
        if self._loading_active:
            raise NotImplementedError(
                "SinusoidalGalerkinSolver wire loading (Galerkin overlap form) "
                "is not implemented yet (momwire#182)"
            )

        seg_view = self._basis_coefs(geom, k)
        ctx = self._test_context(geom, seg_view, k)
        N = ctx["N"]

        contribs = self._tested_contribs(geom, k, ctx, _plain_projection)
        if self.ground_z is not None:
            gnd = self._tested_ground_block(geom, k, ctx)
            contribs = tuple(c - g for c, g in zip(contribs, gnd))

        i_of_entry = ctx["i_of_entry"]

        def _scatter(contrib):
            T = np.zeros((N, N), dtype=np.complex128)
            np.add.at(T, i_of_entry, contrib)
            return T

        T_c, T_s, T_co = (_scatter(c) for c in contribs)

        # --- Source-side coefficient matrices (identical to collocation) --
        m_of_entry = ctx["m_of_entry"]
        M_A = np.zeros((N, N), dtype=np.complex128)
        M_B = np.zeros((N, N), dtype=np.complex128)
        M_C = np.zeros((N, N), dtype=np.complex128)
        M_A[m_of_entry, i_of_entry] = ctx["sigA"]
        M_B[m_of_entry, i_of_entry] = ctx["B"]
        M_C[m_of_entry, i_of_entry] = ctx["sigC"]

        G = T_c @ M_A + T_s @ M_B + T_co @ M_C
        return G, seg_view

    def _apply_near_correction(self, geom, k, ctx, contribs, projector, src_c, src_t):
        """Recompute the near-pair test integrals on the endpoint-graded rule,
        overwriting the uniform-rule values in `contribs` (M2).

        Each (entry, source-segment) cell is owned by exactly one near pair —
        the entry fixes the test segment — so the overwrite is an assignment,
        not an accumulation.

        Runs per source block (M4): the free-space block selects its near
        pairs against the real segments, an image block against the MIRRORED
        ones, so a wire touching the plane gets its segment↔own-image spike
        corrected too. The blocks are separate arrays, so the assignments
        never collide.
        """
        mm, nn = self._near_pairs(geom, src_c=src_c, src_t=src_t)
        if mm.size == 0:
            return

        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        hh = ctx["hh"]
        a_seg = ctx["a_seg"]
        starts, counts = ctx["starts"], ctx["counts"]
        sigA, B, sigC = ctx["sigA"], ctx["B"], ctx["sigC"]

        # Feature width relative to the half-length, taken at the tightest
        # segment in the model so one shared template resolves them all.
        a_all = self._seg_radius(geom)
        xg, wg = _graded_endpoint_rule(
            float(np.min(a_all / hh)), self.n_qp_near, self._leggauss_cached
        )

        # Flatten (pair, support-entry-of-its-test-segment) to a run-length
        # layout: entries of pair p occupy cum[p]:cum[p+1].
        cnt = counts[mm]
        cum = np.concatenate(([0], np.cumsum(cnt)))
        pair_of = np.repeat(np.arange(mm.size), cnt)
        entry_of = (
            np.arange(cum[-1]) - np.repeat(cum[:-1], cnt) + np.repeat(starts[mm], cnt)
        )

        contrib_c, contrib_s, contrib_co = contribs
        for p0 in range(0, mm.size, _PAIR_BLOCK):
            p1 = min(p0 + _PAIR_BLOCK, mm.size)
            mi, ni = mm[p0:p1], nn[p0:p1]

            # (P, G, 3) observer points along each test segment.
            xi = hh[mi][:, None] * xg[None, :]  # (P, G)
            obs = seg_c[mi][:, None, :] + xi[:, :, None] * seg_t[mi][:, None, :]
            cm = self._field_components_bcast(
                k,
                obs_c=obs,
                obs_t=seg_t[mi][:, None, :],
                a=self._uniform_radius if a_seg is None else a_seg[mi][:, None],
                src_c=src_c[ni][:, None, :],
                src_t=src_t[ni][:, None, :],
                src_hh=hh[ni][:, None],
            )
            # (P, 1) pair indices broadcast against the (P, G) field tables.
            Phi = projector(cm, mi[:, None], ni[:, None])  # each (P, G)

            e0, e1 = cum[p0], cum[p1]
            ei = entry_of[e0:e1]  # into the flat seg_view arrays
            lp = pair_of[e0:e1] - p0  # into this block's pair axis
            xi_e = xi[lp]  # (E, G)
            fval = (
                sigA[ei][:, None]
                + B[ei][:, None] * np.sin(k * xi_e)
                + sigC[ei][:, None] * np.cos(k * xi_e)
            )
            w = (wg[None, :] * hh[mi][lp][:, None]) * fval  # (E, G)

            col = ni[lp]
            for contrib, Ph in zip((contrib_c, contrib_s, contrib_co), Phi):
                contrib[ei, col] = np.einsum("eg,eg->e", w, Ph[lp])

    # ------------------------------------------------------------------
    # Galerkin-tested source vector + solve
    # ------------------------------------------------------------------

    def _tested_source_vector(self, geom, seg_view, k):
        """Galerkin RHS b_i = -∫ f_i(s) · ŝ·E^app(s) ds. The delta-gap
        applied field is E^app = V/Δ_m along +ŝ_m on each feed segment m and
        zero elsewhere, so only feed-segment support entries contribute:

            ∫_{seg m} f_{i,m}(ξ) dξ = σA·Δ_m + σC·(2/k)·sin(kΔ_m/2)

        (the sin term integrates to zero by parity). Contrast the collocation
        RHS, which point-samples -V/Δ_m at the feed segment centre. This
        integral is exact, so M2's quadrature work does not touch it.
        """
        N = geom["n_segs"]
        h = np.asarray(geom["seg_h"], dtype=float)
        starts = seg_view["starts"]
        b = np.zeros(N, dtype=np.complex128)
        for fseg, (_, _, V) in zip(geom["feed_segs"], self.feeds):
            s, e = starts[fseg], starts[fseg + 1]
            i = seg_view["jbasis"][s:e]
            A = seg_view["A"][s:e]
            C = seg_view["C"][s:e]
            sig = seg_view["sigma"][s:e]
            hm = float(h[fseg])
            int_f = sig * A * hm + sig * C * (2.0 / k) * np.sin(0.5 * k * hm)
            np.add.at(b, i, -(V / hm) * int_f)
        return b

    def compute_impedance(self):
        """Return (Z_drive, alpha). Mirrors `SinusoidalSolver.compute_impedance`
        but assembles the Galerkin matrix and the Galerkin-tested RHS. The
        current readout (I at the feed-segment centre) is a property of the
        solution current, so `_feed_segment_current` is reused unchanged.
        """
        geom = self._build_geometry()
        self._checkpoint()  # after geometry, before the field fill
        G, seg_view = self._assemble_Z(geom, self.k)
        b = self._tested_source_vector(geom, seg_view, self.k)
        self._checkpoint()  # after assembly, before the dense solve

        alpha = scipy.linalg.solve(G, b)

        feed_segs = geom["feed_segs"]
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        feed_currents = np.array(
            [self._feed_segment_current(alpha, seg_view, fi) for fi in feed_segs],
            dtype=np.complex128,
        )
        z_per_feed = voltages / feed_currents
        Z_drive = z_per_feed[0] if len(self.feeds) == 1 else z_per_feed
        return Z_drive, alpha
