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

Still deferred: ground models (M4), wire loading, junction ports (M5), and
the C++ accelerator — all raise rather than return plausible wrong numbers.
"""

import numpy as np
import scipy.linalg
import scipy.spatial.distance

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

    Free space only. Ground models (M4), wire loading, junction ports (M5),
    and the C++ accelerator are deliberately not wired yet and raise where
    reached.
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

    def _near_pairs(self, geom, n_samples=5):
        """Ordered (test, source) segment pairs whose test integral needs the
        graded rule, as two index arrays.

        Selection is geometric rather than topological, so it covers the self
        pair, node-sharing neighbours, AND close-approach pairs that share no
        node (closely spaced parallel runs) — all of which put a sub-segment
        length scale into the test integrand.

        Exact segment-segment distance is only computed for pairs that survive
        a cheap centre-distance prefilter. That prefilter is a strict superset:
        |c_m − c_n| ≤ d_min + h_m + h_n always, so any pair meeting the
        `near_factor` criterion also meets the prefilter.
        """
        c = geom["seg_centers"]
        t = geom["seg_tangents"]
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)
        reach = hh[:, None] + hh[None, :]

        # cdist rather than an explicit (N, N, 3) difference: the prefilter
        # should not cost 3× the memory of the matrix it is protecting.
        dc = scipy.spatial.distance.cdist(c, c)
        cand = np.argwhere(dc <= (1.0 + self.near_factor) * reach)
        m, n = cand[:, 0], cand[:, 1]

        s = np.linspace(-1.0, 1.0, n_samples)

        def _gap(mi, ni):
            """min over sample points of segment mi of the distance to the
            (clamped) axis of segment ni."""
            pts = (
                c[mi][:, None, :]
                + (hh[mi][:, None] * s[None, :])[:, :, None] * t[mi][:, None, :]
            )  # (P, n_samples, 3)
            d = pts - c[ni][:, None, :]
            u = np.einsum("pgd,pd->pg", d, t[ni])
            u = np.clip(u, -hh[ni][:, None], hh[ni][:, None])
            perp = d - u[..., None] * t[ni][:, None, :]
            return np.linalg.norm(perp, axis=-1).min(axis=1)

        keep = np.minimum(_gap(m, n), _gap(n, m)) <= self.near_factor * reach[m, n]
        return m[keep], n[keep]

    # ------------------------------------------------------------------
    # Galerkin matrix assembly
    # ------------------------------------------------------------------

    def _assemble_Z(self, geom, k):
        """Galerkin system matrix G (basis i tested against source basis j).

        G[i, j] = ∫ f_i(s) · ŝ·E_j(s) ds
                = Σ_shape ( T[shape] @ M[shape] )[i, j]

        where T[shape, i, n] integrates the closed-form field of unit
        `shape` (const/sin/cos) current on source segment n against test
        basis i (quadrature along i's support), and M[shape][n, j] is the
        effective coefficient source basis j puts on segment n — the SAME
        source-side coefficient matrices the collocation path builds.

        The test quadrature is two-tier (M2): a shared uniform rule for the
        far pairs, overwritten per near pair by the endpoint-graded rule.
        """
        if self.ground_z is not None:
            raise NotImplementedError(
                "SinusoidalGalerkinSolver ground models arrive in M4 (momwire#182)"
            )
        if self._loading_active:
            raise NotImplementedError(
                "SinusoidalGalerkinSolver wire loading (Galerkin overlap form) "
                "is not implemented yet (momwire#182)"
            )

        seg_view = self._basis_coefs(geom, k)
        N = geom["n_segs"]
        seg_c = geom["seg_centers"]  # (N, 3)
        seg_t = geom["seg_tangents"]  # (N, 3)
        h = np.asarray(geom["seg_h"], dtype=float)  # (N,) full lengths
        hh = 0.5 * h  # (N,) half-lengths

        # --- Observer quadrature points along each test segment ----------
        gx, gw = self._leggauss_cached(self.n_qp_test)  # nodes on [-1, 1]
        nq = gx.shape[0]
        xi = hh[:, None] * gx[None, :]  # (N, nq) local arc from each centre
        obs_pts = seg_c[:, None, :] + xi[:, :, None] * seg_t[:, None, :]  # (N,nq,3)
        obs_c = np.ascontiguousarray(obs_pts.reshape(N * nq, 3))
        obs_t = np.ascontiguousarray(
            np.broadcast_to(seg_t[:, None, :], (N, nq, 3)).reshape(N * nq, 3)
        )
        a_seg = None if self._uniform_radius is not None else self._seg_radius(geom)
        obs_a = None if a_seg is None else np.repeat(a_seg, nq)[:, None]  # (N·nq, 1)

        # --- Field of each source shape at every observer point ----------
        # Reuse the closed-form Eqs 76-79 evaluators; observers are the quad
        # points, sources are the geometry's segments (unchanged).
        cm = self._field_components(
            geom, k, obs_centers=obs_c, obs_tangents=obs_t, obs_radius=obs_a
        )
        td = cm["td"]
        rp = cm["rho_proj_factor"]
        Phi_c = (td * cm["Ez_const"] + rp * cm["Erho_const"]).reshape(N, nq, N)
        Phi_s = (td * cm["Ez_sin"] + rp * cm["Erho_sin"]).reshape(N, nq, N)
        Phi_co = (td * cm["Ez_cos"] + rp * cm["Erho_cos"]).reshape(N, nq, N)

        # --- Test-side integration weights -------------------------------
        # For each (segment m, basis i) support entry, the test function's
        # value at quad node q is f_{i,m}(ξ_q) = σA + B·sin(kξ_q) + σC·cos(kξ_q)
        # (the current-shape convention pinned in _evaluate_basis_at_points),
        # and the arc weight is gw_q·(h_m/2).
        starts = seg_view["starts"]
        counts = np.diff(starts)
        m_of_entry = np.repeat(np.arange(N, dtype=np.int64), counts)  # (nnz,)
        i_of_entry = seg_view["jbasis"]  # (nnz,)
        sig = seg_view["sigma"].astype(np.complex128)
        A = seg_view["A"]
        B = seg_view["B"]
        C = seg_view["C"]
        sigA = sig * A
        sigC = sig * C
        sin_kxi = np.sin(k * xi)  # (N, nq)
        cos_kxi = np.cos(k * xi)  # (N, nq)
        fval = (
            sigA[:, None]
            + B[:, None] * sin_kxi[m_of_entry]
            + sigC[:, None] * cos_kxi[m_of_entry]
        )  # (nnz, nq)
        w_entry = (gw[None, :] * hh[m_of_entry][:, None]) * fval  # (nnz, nq)

        def _tested_contrib(Phi):
            """contrib[entry, n] = Σ_q w_entry[entry,q] · Phi[m(entry), q, n].
            Accumulated one quadrature node at a time: the equivalent einsum
            over `Phi[m_of_entry]` would first materialize an (nnz, nq, N)
            gather, nq× the peak memory for the same arithmetic."""
            out = np.zeros((w_entry.shape[0], N), dtype=np.complex128)
            for q in range(nq):
                out += w_entry[:, q, None] * Phi[m_of_entry, q, :]
            return out

        contrib_c = _tested_contrib(Phi_c)
        contrib_s = _tested_contrib(Phi_s)
        contrib_co = _tested_contrib(Phi_co)

        if self.near_correction:
            self._apply_near_correction(
                geom,
                k,
                seg_view,
                (contrib_c, contrib_s, contrib_co),
                starts=starts,
                counts=counts,
                sigA=sigA,
                B=B,
                sigC=sigC,
                a_seg=a_seg,
            )

        def _scatter(contrib):
            T = np.zeros((N, N), dtype=np.complex128)
            np.add.at(T, i_of_entry, contrib)
            return T

        T_c = _scatter(contrib_c)
        T_s = _scatter(contrib_s)
        T_co = _scatter(contrib_co)

        # --- Source-side coefficient matrices (identical to collocation) --
        M_A = np.zeros((N, N), dtype=np.complex128)
        M_B = np.zeros((N, N), dtype=np.complex128)
        M_C = np.zeros((N, N), dtype=np.complex128)
        M_A[m_of_entry, i_of_entry] = sigA
        M_B[m_of_entry, i_of_entry] = B
        M_C[m_of_entry, i_of_entry] = sigC

        G = T_c @ M_A + T_s @ M_B + T_co @ M_C
        return G, seg_view

    def _apply_near_correction(
        self, geom, k, seg_view, contribs, *, starts, counts, sigA, B, sigC, a_seg
    ):
        """Recompute the near-pair test integrals on the endpoint-graded rule,
        overwriting the uniform-rule values in `contribs` (M2).

        Each (entry, source-segment) cell is owned by exactly one near pair —
        the entry fixes the test segment — so the overwrite is an assignment,
        not an accumulation.
        """
        mm, nn = self._near_pairs(geom)
        if mm.size == 0:
            return

        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)

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
                src_c=seg_c[ni][:, None, :],
                src_t=seg_t[ni][:, None, :],
                src_hh=hh[ni][:, None],
            )
            td = cm["td"]
            rp = cm["rho_proj_factor"]
            Phi = (
                td * cm["Ez_const"] + rp * cm["Erho_const"],
                td * cm["Ez_sin"] + rp * cm["Erho_sin"],
                td * cm["Ez_cos"] + rp * cm["Erho_cos"],
            )  # each (P, G)

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
