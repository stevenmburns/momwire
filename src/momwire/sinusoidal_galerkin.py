"""Richmond-style piecewise-sinusoidal **Galerkin** solver (momwire#182, M1).

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

This is milestone **M1**: free-space only, naive quadrature everywhere
(including the singular self / adjacent test integrals — those become
Richmond closed forms in M2). The load-bearing gate is **G1**: a correct
Galerkin fill is symmetric, so ‖Z − Zᵀ‖/‖Z‖ must collapse to ~1e-10; a
collocation fill cannot fake that and quadrature sloppiness breaks it.

Design fill scheme (issue #182, chosen empirically — "let G1 decide"):
reuse the existing closed-form Eqs 76-79 field evaluators for the SOURCE
integral (analytic over each source segment) and Gauss-quadrature only the
TEST integral. See the module test `tests/test_sinusoidal_galerkin.py` for
the G1 verdict this scheme actually produces.
"""

import numpy as np
import scipy.linalg

from .sinusoidal import SinusoidalSolver


class SinusoidalGalerkinSolver(SinusoidalSolver):
    """Piecewise-sinusoidal Galerkin MoM. Same constructor surface as
    `SinusoidalSolver` plus `n_qp_test` (Gauss nodes per test segment for the
    Galerkin overlap integral).

    M1 scope: free space only. Ground models (M4), wire loading (its Galerkin
    overlap form), junction ports (M5), and the C++ accelerator are deliberately
    not wired yet and raise where reached.
    """

    def __init__(self, *, n_qp_test=8, **kwargs):
        super().__init__(**kwargs)
        self.n_qp_test = int(n_qp_test)

    # ------------------------------------------------------------------
    # Galerkin matrix assembly
    # ------------------------------------------------------------------

    def _assemble_Z(self, geom, k):
        """Galerkin system matrix G (basis i tested against source basis j).

        G[i, j] = ∫ f_i(s) · ŝ·E_j(s) ds
                = Σ_shape ( T[shape] @ M[shape] )[i, j]

        where T[shape, i, n] integrates the closed-form field of unit
        `shape` (const/sin/cos) current on source segment n against test
        basis i (Gauss quadrature along i's support), and M[shape][n, j] is
        the effective coefficient source basis j puts on segment n — the
        SAME source-side coefficient matrices the collocation path builds.
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
        if self._uniform_radius is not None:
            obs_a = None  # scalar radius path
        else:
            obs_a = np.repeat(self._seg_radius(geom), nq)[:, None]  # (N·nq, 1)

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
        # and the arc weight is gw_q·(h_m/2). T sums these over q and scatters
        # into test-basis row i.
        starts = seg_view["starts"]
        counts = np.diff(starts)
        m_of_entry = np.repeat(np.arange(N, dtype=np.int64), counts)  # (nnz,)
        i_of_entry = seg_view["jbasis"]  # (nnz,)
        sig = seg_view["sigma"].astype(np.complex128)
        A = seg_view["A"]
        B = seg_view["B"]
        C = seg_view["C"]
        sin_kxi = np.sin(k * xi)  # (N, nq)
        cos_kxi = np.cos(k * xi)  # (N, nq)
        fval = (
            (sig * A)[:, None]
            + B[:, None] * sin_kxi[m_of_entry]
            + (sig * C)[:, None] * cos_kxi[m_of_entry]
        )  # (nnz, nq)
        aw = gw[None, :] * hh[m_of_entry][:, None]  # (nnz, nq)
        w_entry = aw * fval  # (nnz, nq)

        def _tested(Phi):
            # contrib[entry, n] = Σ_q w_entry[entry, q] · Phi[m_of_entry, q, n]
            contrib = np.einsum("eq,eqn->en", w_entry, Phi[m_of_entry])
            T = np.zeros((N, N), dtype=np.complex128)
            np.add.at(T, i_of_entry, contrib)
            return T

        T_c = _tested(Phi_c)
        T_s = _tested(Phi_s)
        T_co = _tested(Phi_co)

        # --- Source-side coefficient matrices (identical to collocation) --
        n_idx = m_of_entry
        j_idx = i_of_entry
        M_A = np.zeros((N, N), dtype=np.complex128)
        M_B = np.zeros((N, N), dtype=np.complex128)
        M_C = np.zeros((N, N), dtype=np.complex128)
        M_A[n_idx, j_idx] = sig * A
        M_B[n_idx, j_idx] = B
        M_C[n_idx, j_idx] = sig * C

        G = T_c @ M_A + T_s @ M_B + T_co @ M_C
        return G, seg_view

    # ------------------------------------------------------------------
    # Galerkin-tested source vector + solve
    # ------------------------------------------------------------------

    def _tested_source_vector(self, geom, seg_view, k):
        """Galerkin RHS b_i = -∫ f_i(s) · ŝ·E^app(s) ds. The delta-gap
        applied field is E^app = V/Δ_m along +ŝ_m on each feed segment m and
        zero elsewhere, so only feed-segment support entries contribute:

            ∫_{seg m} f_{i,m}(ξ) dξ = σA·Δ_m + σC·(2/k)·sin(kΔ_m/2)

        (the sin term integrates to zero by parity). Contrast the collocation
        RHS, which point-samples -V/Δ_m at the feed segment centre.
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
