"""Higher-order B-spline Galerkin MoM solver.

`TriangularPySim` is the degree-1 B-spline (tent) special case; this
module extends to arbitrary degree d on multi-wire polylines with K-wire
junctions, primarily as an in-codebase arbiter for the hentenna question
(NEXT_STEPS.md items 9, 13, 14): does the tent basis converge to the
correct value, or is it converged-to-the-wrong-place?

Scope:
  * arbitrary number of wires; each wire is a polyline (M ≥ 2 anchors)
  * uniform segments per edge, possibly non-uniform across edges
  * free space, thin-wire kernel with a² wire-radius regularization
  * delta-gap "applied-E" source on one feed wire
  * degree d ∈ {1, 2}  (d=1 reproduces the tent basis up to feed convention)
  * K-wire junctions with KCL constraint (Σ outflow currents = 0)
  * NO ground plane (yet)

Same as TriangularPySim, but with a polynomial-of-degree-d on each segment
instead of just a linear ramp. Each interior basis Φ_m spans up to d+1
contiguous segments within a single wire; on each segment in its support
("wing") the basis equals Σ_p C[m, w, p] · u^p with u local arc length.

J_pq[i, j] = ∫∫ u^p u'^q · exp(-jkR)/(4πR) du' du
with R² = |r_i(u) - r_j(u')|² + a²

Galerkin assembly:
    Z_A[m,n]   = jωμ Σ_{a,b} (t_i · t_j) · Σ_{p,q} C[m,a,p] C[n,b,q]
                 · J_{pq}[supp_seg[m,a], supp_seg[n,b]]
    Z_Φ[m,n]   = (1/jωε) Σ_{a,b} Σ_{p≥1,q≥1} p·q · C[m,a,p] C[n,b,q]
                 · J_{p-1,q-1}[supp_seg[m,a], supp_seg[n,b]]

Junction directional bases: at every junction node with K connected wire-
ends we add K boundary bases (B_0 or B_{N+d-1} of each connected wire,
the ones with value 1 at the junction) and enforce KCL via a Lagrange-
multiplier row, mirroring TriangularPySim's treatment.

Feed: v_m = Φ_m(s_f), Z_drive = 1 / (v^T c).
"""

import numpy as np
import scipy.linalg
from scipy.interpolate import BSpline

from ._bspline_kernels import (
    _seg_seg_full_moments_offedge,
    _seg_seg_reg_moments,
    _seg_seg_static_moments,
)

try:
    from . import _accelerators as _acc

    _HAVE_BSPLINE_ASSEMBLE_ACCEL = hasattr(_acc, "assemble_Z_bspline")
except ImportError:
    _HAVE_BSPLINE_ASSEMBLE_ACCEL = False

_BSPLINE_ASSEMBLE_ACCEL_MAX_D = 2


class BSplinePySim:
    """Degree-d B-spline Galerkin MoM, multi-wire polylines with junctions.

    Parameters
    ----------
    wires : list of (M, 3) polyline arrays, M ≥ 2 anchors each.
    n_per_edge_per_wire : list of (int | sequence | None). Per-wire segment
        counts per edge. None for a wire ⇒ use `nsegs` on every edge; int ⇒
        same count for every edge; sequence ⇒ explicit per-edge count.
    degree : B-spline degree (1 ≤ degree ≤ 2 currently; static-moment file
        only covers max_d=2). d=1 reproduces the tent basis up to the
        feed-convention difference.
    feed_wire_index : index of the wire carrying the delta-gap source.
    feed_arclength : arc length along the feed wire at which to evaluate
        Φ_m(s_f). Default: feed wire midpoint.
    junctions : list of [(wire_idx, "start"|"end"), ...] tuples, each entry
        one junction node where K wire endpoints meet. Same convention as
        TriangularPySim.
    n_qp_pair : Gauss-Legendre nodes per segment per axis for the smooth-
        kernel piece of same-edge pairs and for all cross-edge / cross-wire
        pairs (full kernel with a² regularization).
    wavelength, halfdriver_factor, wire_radius, nsegs : as in
        TriangularPySim.
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        degree=2,
        feed_wire_index=0,
        feed_arclength=None,
        junctions=None,
        n_qp_pair=4,
        wavelength=22,
        halfdriver_factor=0.962,
        wire_radius=0.0005,
        nsegs=101,
    ):
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if degree > 2:
            raise NotImplementedError(
                "degree > 2 needs scripts/derive_bspline_static_moments.py "
                "to be re-run with a larger MAX_D"
            )
        if not wires:
            raise ValueError("wires must be non-empty")

        self.degree = int(degree)
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.wire_radius = wire_radius
        self.nsegs = nsegs

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = self.omega / self.c
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")

        n_w = len(self.wires_polylines)
        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None] * n_w
        if len(n_per_edge_per_wire) != n_w:
            raise ValueError(
                f"n_per_edge_per_wire length {len(n_per_edge_per_wire)} != n_wires {n_w}"
            )

        self.n_per_edge_per_wire = []
        for i, (pl, npe) in enumerate(zip(self.wires_polylines, n_per_edge_per_wire)):
            n_edges_w = pl.shape[0] - 1
            if npe is None:
                npe = self.nsegs
            if np.isscalar(npe):
                npe = [int(npe)] * n_edges_w
            npe = list(npe)
            if len(npe) != n_edges_w:
                raise ValueError(
                    f"wire {i}: n_per_edge length {len(npe)} != n_edges {n_edges_w}"
                )
            self.n_per_edge_per_wire.append(npe)

        if not (0 <= feed_wire_index < n_w):
            raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
        self.feed_wire_index = feed_wire_index
        self.feed_arclength = feed_arclength
        self.n_qp_pair = int(n_qp_pair)

        self.junctions = []
        if junctions is not None:
            for j, jw in enumerate(junctions):
                if len(jw) < 2:
                    raise ValueError(f"junction {j}: need >= 2 wire-ends")
                normalized = []
                for w, end in jw:
                    if not (0 <= w < n_w):
                        raise ValueError(
                            f"junction {j}: wire_idx {w} out of range [0, {n_w})"
                        )
                    if end not in ("start", "end"):
                        raise ValueError(
                            f"junction {j}: end must be 'start' or 'end', got {end!r}"
                        )
                    normalized.append((int(w), end))
                self.junctions.append(normalized)

    # ------------------------------------------------------------------
    # Geometry build
    # ------------------------------------------------------------------

    def _build_geometry(self):
        """Discretize all wires, concatenate to global arrays.

        Per-wire metadata is preserved so the basis-polynomial extraction
        (which operates on each wire's clamped knot vector independently)
        can be done wire-by-wire.

        Returns a `geom` dict with:
          per_wire: list of per-wire dicts (seg_l, seg_r, tangents, h_per_seg,
              edge_offsets, edge_arc_edges, arc_at_knot, n_total)
          seg_offsets: list[n_w+1] of global segment index of wire start
          n_segs_total: total segment count across all wires
          h_per_seg: (N_total,) per-segment edge length
          tangents: (N_total, 3) per-segment tangent unit vector
          seg_l, seg_r: (N_total, 3) per-segment 3D endpoint
        """
        per_wire = []
        seg_offsets = [0]
        h_list = []
        tangents_list = []
        seg_l_list_all = []
        seg_r_list_all = []
        for w_idx, (pl, npe_list) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            seg_l_w = []
            seg_r_w = []
            tan_w = []
            h_w_list = []
            edge_offsets = [0]
            edge_arc_edges = []
            for e_idx in range(pl.shape[0] - 1):
                p0 = pl[e_idx]
                p1 = pl[e_idx + 1]
                edge_vec = p1 - p0
                edge_len = float(np.linalg.norm(edge_vec))
                if edge_len < 1e-15:
                    raise ValueError(f"wire {w_idx} edge {e_idx} has zero length")
                tan = edge_vec / edge_len
                n_e = npe_list[e_idx]
                h_e = edge_len / n_e

                t_node = np.linspace(0.0, 1.0, n_e + 1)
                pts = (1 - t_node[:, None]) * p0[None, :] + t_node[:, None] * p1[
                    None, :
                ]
                seg_l_w.append(pts[:-1])
                seg_r_w.append(pts[1:])
                tan_w.append(np.tile(tan, (n_e, 1)))
                h_w_list.append(np.full(n_e, h_e))
                edge_arc_edges.append(np.linspace(0.0, edge_len, n_e + 1))
                edge_offsets.append(edge_offsets[-1] + n_e)

            seg_l = np.vstack(seg_l_w)
            seg_r = np.vstack(seg_r_w)
            tangents_w = np.vstack(tan_w)
            h_per_seg_w = np.concatenate(h_w_list)
            n_total_w = seg_l.shape[0]
            arc_at_knot = np.concatenate([[0.0], np.cumsum(h_per_seg_w)])

            per_wire.append(
                {
                    "seg_l": seg_l,
                    "seg_r": seg_r,
                    "tangents": tangents_w,
                    "h_per_seg": h_per_seg_w,
                    "edge_offsets": edge_offsets,
                    "edge_arc_edges": edge_arc_edges,
                    "arc_at_knot": arc_at_knot,
                    "n_total": n_total_w,
                }
            )
            seg_offsets.append(seg_offsets[-1] + n_total_w)
            h_list.append(h_per_seg_w)
            tangents_list.append(tangents_w)
            seg_l_list_all.append(seg_l)
            seg_r_list_all.append(seg_r)

        h_per_seg_global = np.concatenate(h_list)
        tangents_global = np.vstack(tangents_list)
        seg_l_global = np.vstack(seg_l_list_all)
        seg_r_global = np.vstack(seg_r_list_all)

        return {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "n_segs_total": seg_offsets[-1],
            "h_per_seg": h_per_seg_global,
            "tangents": tangents_global,
            "seg_l": seg_l_global,
            "seg_r": seg_r_global,
        }

    # ------------------------------------------------------------------
    # Endpoint status (free vs junction)
    # ------------------------------------------------------------------

    def _wire_endpoint_status(self):
        """For each wire, return ("free" | junction_idx, "free" | junction_idx)
        for its (start, end) — the index of the junction connecting it, or
        "free" if the endpoint isn't junctioned.
        """
        n_w = len(self.wires_polylines)
        start_status = ["free"] * n_w
        end_status = ["free"] * n_w
        for j_idx, jw in enumerate(self.junctions):
            for w, end in jw:
                if end == "start":
                    start_status[w] = j_idx
                else:
                    end_status[w] = j_idx
        return start_status, end_status

    # ------------------------------------------------------------------
    # Basis polynomial extraction
    # ------------------------------------------------------------------

    def _build_basis_polynomials(self, geom):
        """Extract polynomial coefficients per (basis, wing).

        For each wire:
          * Build clamped knot vector on the wire's cumulative arc.
          * Determine which of the d+1 boundary bases per end are kept:
              - Free end: drop all d+1 boundary bases (Φ(end) = 0 strictly,
                  AND derivative 0, etc. — for d ≤ 2 this means drop just
                  B_0 because only B_0 has nonzero value, and the higher
                  boundary bases are kept as ordinary interior bases since
                  their value at the end is 0).
              - Junction end: keep the value-1 boundary basis B_0 as a
                  directional basis; keep B_1..B_{d-1} as interior bases.
          * Extract per-segment polynomial coefficients via BSpline +
            Vandermonde (uniform within each segment's local-u range).

        Returns
        -------
        supp_seg, polys : as in the single-wire case, concatenated globally.
        kcl_A : (n_junctions, n_basis_total) Lagrange-multiplier rows
            (+1 / -1 outflow sign per directional basis).
        wire_knots : list of per-wire knot vectors (for the source vector).
        wire_basis_global : list of per-wire (kept_idx, global_basis_idx)
            tuples for the source-vector mapping.
        """
        d = self.degree
        n_wings = d + 1
        n_poly = d + 1

        start_status, end_status = self._wire_endpoint_status()

        all_supp_seg = []
        all_polys = []
        wire_knots = []
        wire_basis_global = []
        # Track per-junction the list of (directional-basis global idx,
        # outflow sign).
        junction_dirs = {j: [] for j in range(len(self.junctions))}

        m_global = 0
        for w_idx, pw in enumerate(geom["per_wire"]):
            arc = pw["arc_at_knot"]
            wire_arc = arc[-1]
            interior_knots = arc.copy()
            knots = np.concatenate(
                [np.full(d, 0.0), interior_knots, np.full(d, wire_arc)]
            )
            wire_knots.append(knots)
            n_basis_w = len(knots) - d - 1  # = N_w + d

            # Determine kept bases. For d ∈ {1, 2}:
            #   B_0 is the value-1 boundary basis at the start
            #   B_{n_basis_w - 1} is the value-1 boundary basis at the end
            #   B_1, ..., B_{n_basis_w - 2} are interior (value 0 at endpoints)
            kept = []  # list of (basis_j, kind, junction_idx-or-None)
            # Start boundary basis (B_0)
            if start_status[w_idx] == "free":
                pass  # drop
            else:
                kept.append((0, "dir", start_status[w_idx], "start"))
            # Truly interior bases
            for j in range(1, n_basis_w - 1):
                kept.append((j, "int", None, None))
            # End boundary basis (B_{n_basis_w - 1})
            if end_status[w_idx] == "free":
                pass  # drop
            else:
                kept.append((n_basis_w - 1, "dir", end_status[w_idx], "end"))

            seg_off = geom["seg_offsets"][w_idx]
            h_per_seg_w = pw["h_per_seg"]
            arc_at_knot_w = pw["arc_at_knot"]
            # build a sample of d+1 uniform points within each segment's
            # local-u range, in GLOBAL arc length (so we can evaluate the
            # BSpline at them)
            # For per-segment extraction, do it inside the loop because h
            # varies across edges.

            per_basis_local_to_global = {}
            for kept_idx, (j, kind, junc_idx, end_pos) in enumerate(kept):
                c_all = np.zeros(n_basis_w, dtype=np.float64)
                c_all[j] = 1.0
                bspl = BSpline(knots, c_all, d, extrapolate=False)

                support_lo = knots[j]
                support_hi = knots[j + d + 1]

                supp_seg_m = np.zeros(n_wings, dtype=np.int64)
                polys_m = np.zeros((n_wings, n_poly), dtype=np.float64)

                # Find segments overlapping the support
                wing = 0
                # iterate over local segments in this wire
                for seg_local in range(pw["n_total"]):
                    seg_l_arc = arc_at_knot_w[seg_local]
                    seg_r_arc = arc_at_knot_w[seg_local + 1]
                    eps_seg = 1e-9 * max(h_per_seg_w[seg_local], 1e-12)
                    if seg_r_arc < support_lo + eps_seg:
                        continue
                    if seg_l_arc > support_hi - eps_seg:
                        break
                    h_seg = h_per_seg_w[seg_local]
                    # uniform sample for Vandermonde
                    u_local = np.linspace(0.0, h_seg, d + 1)
                    u_global = seg_l_arc + u_local
                    vals = bspl(u_global)
                    Vmat = np.vander(u_local, d + 1, increasing=True)
                    coeffs = np.linalg.solve(Vmat, vals)
                    supp_seg_m[wing] = seg_off + seg_local
                    polys_m[wing, :] = coeffs
                    wing += 1
                    if wing >= n_wings:
                        break

                all_supp_seg.append(supp_seg_m)
                all_polys.append(polys_m)
                per_basis_local_to_global[kept_idx] = m_global

                if kind == "dir":
                    sign = +1.0 if end_pos == "start" else -1.0
                    junction_dirs[junc_idx].append((m_global, sign))

                m_global += 1

            wire_basis_global.append((kept, per_basis_local_to_global))

        supp_seg = (
            np.stack(all_supp_seg, axis=0)
            if all_supp_seg
            else (np.zeros((0, n_wings), dtype=np.int64))
        )
        polys = (
            np.stack(all_polys, axis=0)
            if all_polys
            else (np.zeros((0, n_wings, n_poly), dtype=np.float64))
        )
        n_basis_total = supp_seg.shape[0]

        n_junctions = len(self.junctions)
        kcl_A = np.zeros((n_junctions, n_basis_total), dtype=np.float64)
        for j_idx, dirs in junction_dirs.items():
            for m_g, sign in dirs:
                kcl_A[j_idx, m_g] = sign

        return supp_seg, polys, kcl_A, wire_knots, wire_basis_global

    # ------------------------------------------------------------------
    # J moment integrals
    # ------------------------------------------------------------------

    def _build_J_blocks(self, geom, k):
        """All polynomial moment integrals J_pq[i, j] for p, q ∈ {0..d} and
        every (i, j) global segment pair. Returns shape (d+1, d+1, N, N).

        Fused build mirroring TriangularPySim._build_J_blocks: first compute
        every pair by full GL quadrature on the regularized full kernel
        G = exp(-jkR)/(4πR), R² = |Δr|² + a²; then overwrite same-edge
        blocks with the analytic static + GL-regularized split (essential
        for the log-singular diagonal).
        """
        d = self.degree
        a = self.wire_radius
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]

        # All-pairs full kernel (same a² regularization handles touching
        # segments at kink corners and at junctions to within ~1e-5 at
        # antenna scales; off-segment-pair accuracy is what GL is good at).
        J = _seg_seg_full_moments_offedge(
            seg_l, seg_r, seg_l, seg_r, a, k, d, self.n_qp_pair
        )  # (d+1, d+1, N, N) complex

        # Overwrite each same-edge block with analytic static + reg
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        for w in range(len(per_wire)):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            base = seg_off[w]
            for i_e in range(len(ed_off) - 1):
                sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                A_st = _seg_seg_static_moments(ed_arc[i_e], a, max_d=d)
                A_reg = _seg_seg_reg_moments(
                    ed_arc[i_e], a, k, max_d=d, n_qp=self.n_qp_pair
                )
                J[:, :, sl, sl] = A_st + A_reg

        return J

    # ------------------------------------------------------------------
    # Z assembly
    # ------------------------------------------------------------------

    def _assemble_Z(self, J, supp_seg, polys, geom):
        """Assemble the (n_basis, n_basis) complex Z matrix.

        Uses the templated C++ accelerator `assemble_Z_bspline` when
        available and `self.degree` is in its instantiation set; otherwise
        falls back to a numpy-einsum implementation that's a bit-exact
        reference target.
        """
        d = self.degree
        n_basis, n_wings, n_poly = polys.shape
        assert n_wings == d + 1 and n_poly == d + 1

        tangents = geom["tangents"]
        td_all = tangents @ tangents.T

        if _HAVE_BSPLINE_ASSEMBLE_ACCEL and d <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D:
            return _acc.assemble_Z_bspline(
                np.ascontiguousarray(J, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(td_all, dtype=np.float64),
                float(self.omega),
                float(self.eps),
                float(self.mu),
                int(d),
            )

        Z_A = np.zeros((n_basis, n_basis), dtype=np.complex128)
        Z_Phi = np.zeros((n_basis, n_basis), dtype=np.complex128)
        p_vec = np.arange(1, d + 1, dtype=np.float64) if d >= 1 else None

        for a in range(n_wings):
            sm = supp_seg[:, a]
            for b in range(n_wings):
                sn = supp_seg[:, b]
                J_blk = J[:, :, sm[:, None], sn[None, :]]
                td_blk = td_all[sm[:, None], sn[None, :]]

                inner_A = np.einsum(
                    "mp,pPmn,nP->mn", polys[:, a, :], J_blk, polys[:, b, :]
                )
                Z_A += td_blk * inner_A

                if d >= 1:
                    deriv_m = polys[:, a, 1:] * p_vec[None, :]
                    deriv_n = polys[:, b, 1:] * p_vec[None, :]
                    J_blk_lo = J_blk[:d, :d]
                    inner_Phi = np.einsum("mp,pPmn,nP->mn", deriv_m, J_blk_lo, deriv_n)
                    Z_Phi += inner_Phi

        Z_A = 1j * self.omega * self.mu * Z_A
        Z_Phi = Z_Phi / (1j * self.omega * self.eps)
        return Z_A + Z_Phi

    # ------------------------------------------------------------------
    # Source vector
    # ------------------------------------------------------------------

    def _build_source_vector(self, geom, wire_knots, wire_basis_global, n_basis_total):
        """v_m = Φ_m(s_f) on the feed wire, zeros elsewhere."""
        d = self.degree
        w = self.feed_wire_index
        arc = geom["per_wire"][w]["arc_at_knot"]
        wire_arc = arc[-1]
        s_f = self.feed_arclength if self.feed_arclength is not None else wire_arc / 2.0

        # design matrix at s_f, on the full (boundary-kept) basis set for
        # the feed wire
        knots = wire_knots[w]
        DM = BSpline.design_matrix(np.array([s_f]), knots, d).toarray()[0]
        # n_basis_w_full = len(knots) - d - 1 — includes all boundary bases.
        # Map back via wire_basis_global[w] = (kept, local_to_global_dict).
        kept, local_to_global = wire_basis_global[w]

        v = np.zeros(n_basis_total, dtype=np.complex128)
        for kept_idx, (j, kind, junc_idx, end_pos) in enumerate(kept):
            m_global = local_to_global[kept_idx]
            v[m_global] = DM[j]
        return v

    # ------------------------------------------------------------------
    # KCL solve (Schur complement)
    # ------------------------------------------------------------------

    def _solve_with_kcl(self, Z, v, kcl_A):
        """Constrained solve [Z A^T; A 0] [I; λ] = [v; 0] via Schur.

        Identical structure to TriangularPySim._solve_with_kcl. If kcl_A
        is empty (no junctions), do a plain solve.
        """
        if kcl_A.shape[0] == 0:
            return scipy.linalg.solve(Z, v)
        n_b = Z.shape[0]
        n_c = kcl_A.shape[0]
        rhs = np.empty((n_b, 1 + n_c), dtype=np.complex128)
        rhs[:, 0] = v
        rhs[:, 1:] = kcl_A.T
        sol = scipy.linalg.solve(Z, rhs)
        w = sol[:, 0]
        X = sol[:, 1:]
        lam = scipy.linalg.solve(kcl_A @ X, kcl_A @ w)
        return w - X @ lam

    # ------------------------------------------------------------------
    # Driver impedance
    # ------------------------------------------------------------------

    def compute_impedance(self):
        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )
        n_basis_total = supp_seg.shape[0]

        J = self._build_J_blocks(geom, self.k)
        Z = self._assemble_Z(J, supp_seg, polys, geom)
        v = self._build_source_vector(
            geom, wire_knots, wire_basis_global, n_basis_total
        )

        coeffs = self._solve_with_kcl(Z, v, kcl_A)
        I_at_feed = v @ coeffs
        driver_impedance = 1.0 / I_at_feed
        self.z = Z
        return driver_impedance, coeffs
