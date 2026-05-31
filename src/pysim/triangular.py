"""Triangular-basis Galerkin MoM for multiple bent (polyline) wires.

Each wire is its own polyline of M >= 2 anchor points; edges within a wire
may bend at kinks; basis functions are continuous through kinks within a
wire but vanish at every wire's endpoints (no continuity across wires).
A straight dipole is the degenerate case of a single-edge single-wire
polyline; a parallel-element Yagi is multiple single-edge wires; an
inverted-V is one two-edge wire; moxon/hexbeam are multi-edge multi-wire.

Block structure of the segment-pair J integrals:
  * same-edge same-wire: analytic static kernel + GL-quadrature regular remainder
  * different-edge same-wire: full 3D quadrature with wire-radius regularization
    (keeps the kernel finite at kink corners shared by adjacent edges)
  * cross-wire: full 3D quadrature; same a^2 regularization is applied for
    code reuse — at moxon-scale tip-to-tip distances of order cm with
    a ~ 0.5 mm the regularization shifts cross-wire integrals by ~1e-5,
    well below the discretization error.

Feed: at the interior knot of `feed_wire_index` whose arc-length from that
wire's start is closest to `feed_arclength` (default: midpoint).
"""

import numpy as np
import scipy.linalg

from ._triangular_kernels import (
    _seg_seg_static_all,
    _seg_seg_reg_all,
    _seg_seg_reg_all_batch,
    _seg_seg_offedge_quad,
    _seg_seg_offedge_quad_batch,
)

try:
    from . import _accelerators as _acc

    _HAVE_ASSEMBLE_Z = hasattr(_acc, "assemble_Z")
except ImportError:
    _HAVE_ASSEMBLE_Z = False


class TriangularPySim:
    """N-wire triangular Galerkin MoM, each wire a polyline with bends.

    wires: list of (M_w, 3) polyline arrays. M_w >= 2 anchor points per wire.
    n_per_edge_per_wire: list of (int or sequence). Per-wire segments per
        polyline edge. None for a wire means use `nsegs` for each of its
        edges; an int means use that count for each edge; a sequence gives a
        per-edge count. If `n_per_edge_per_wire` itself is None, every wire
        uses `nsegs` on every edge.
    feed_wire_index: index of the wire that carries the delta-gap source.
    feed_arclength: arc length along the feed wire (from its starting
        anchor) at which to place the source. None picks the midpoint.
    n_qp_reg: GL points for same-edge regular-kernel integrals.
    n_qp_off: GL points for off-edge / cross-wire integrals.
    wavelength, halfdriver_factor: set the measurement wavenumber k and the
        default-geometry half-driver length. With explicit `wires` the
        geometry is fully determined by the polylines; `halfdriver_factor`
        is then only informational.
    wire_radius: thin-wire radius used in the kernel regularization.
    nsegs: default segment count when `n_per_edge_per_wire` doesn't specify.
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        feed_wire_index=0,
        feed_arclength=None,
        n_qp_reg=4,
        n_qp_off=4,
        wavelength=22,
        halfdriver_factor=0.962,
        wire_radius=0.0005,
        nsegs=101,
    ):
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.wire_radius = wire_radius
        self.nsegs = nsegs

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = self.omega / self.c
        self.jomega = 1j * self.omega
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        if not wires:
            raise ValueError("wires must be non-empty")
        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")

        n_w = len(self.wires_polylines)
        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None] * n_w
        if len(n_per_edge_per_wire) != n_w:
            raise ValueError(
                f"n_per_edge_per_wire length {len(n_per_edge_per_wire)} "
                f"!= number of wires {n_w}"
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
                    f"wire {i}: n_per_edge length {len(npe)} "
                    f"!= number of edges {n_edges_w}"
                )
            self.n_per_edge_per_wire.append(npe)

        if not (0 <= feed_wire_index < n_w):
            raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
        self.feed_wire_index = feed_wire_index
        self.feed_arclength = feed_arclength
        self.n_qp_reg = n_qp_reg
        self.n_qp_off = n_qp_off

    def _build_geometry(self):
        """Discretize every wire and concatenate into global arrays.

        Per-wire metadata (`edge_offsets`, `edge_arc_edges`) is preserved so
        the same-wire J build can stay edge-local and use the analytic
        static-kernel formula on each edge.
        """
        per_wire = []
        seg_offsets = [0]
        basis_offsets = [0]
        h_list = []
        tangents_list = []
        for w_idx, (pl, npe_list) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            seg_l_list = []
            seg_r_list = []
            tan_list = []
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
                knots = (1 - t_node[:, None]) * p0[None, :] + t_node[:, None] * p1[
                    None, :
                ]
                seg_l_list.append(knots[:-1])
                seg_r_list.append(knots[1:])
                tan_list.append(np.tile(tan, (n_e, 1)))
                h_w_list.append(np.full(n_e, h_e))
                edge_arc_edges.append(np.linspace(0.0, edge_len, n_e + 1))
                edge_offsets.append(edge_offsets[-1] + n_e)

            seg_l = np.vstack(seg_l_list)
            seg_r = np.vstack(seg_r_list)
            tangents = np.vstack(tan_list)
            h_per_seg = np.concatenate(h_w_list)
            n_total = seg_l.shape[0]
            arc_at_knot = np.concatenate([[0.0], np.cumsum(h_per_seg)])

            per_wire.append(
                {
                    "seg_l": seg_l,
                    "seg_r": seg_r,
                    "tangents": tangents,
                    "h_per_seg": h_per_seg,
                    "edge_offsets": edge_offsets,
                    "edge_arc_edges": edge_arc_edges,
                    "arc_at_knot": arc_at_knot,
                    "n_total": n_total,
                }
            )
            seg_offsets.append(seg_offsets[-1] + n_total)
            basis_offsets.append(basis_offsets[-1] + (n_total - 1))
            h_list.append(h_per_seg)
            tangents_list.append(tangents)

        h_per_seg_global = np.concatenate(h_list)
        tangents_global = np.vstack(tangents_list)

        left_segs = []
        right_segs = []
        for w in range(len(per_wire)):
            base = seg_offsets[w]
            N_w = per_wire[w]["n_total"]
            left_segs.append(base + np.arange(N_w - 1, dtype=np.int64))
            right_segs.append(base + np.arange(1, N_w, dtype=np.int64))
        left_seg = np.concatenate(left_segs)
        right_seg = np.concatenate(right_segs)

        return {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "basis_offsets": basis_offsets,
            "n_segs_total": seg_offsets[-1],
            "n_basis_total": basis_offsets[-1],
            "h_per_seg": h_per_seg_global,
            "tangents": tangents_global,
            "left_seg": left_seg,
            "right_seg": right_seg,
        }

    def _feed_basis_index(self, geom):
        """Global basis index of the source: closest interior knot on
        feed_wire_index to feed_arclength (default: wire midpoint).
        """
        w = self.feed_wire_index
        arc_at_knot = geom["per_wire"][w]["arc_at_knot"]
        total_arc = arc_at_knot[-1]
        feed_arc = (
            self.feed_arclength if self.feed_arclength is not None else total_arc / 2.0
        )
        interior_arc = arc_at_knot[1:-1]
        m_local = int(np.argmin(np.abs(interior_arc - feed_arc)))
        return geom["basis_offsets"][w] + m_local

    def _build_J_blocks(self, geom, k):
        """Per-segment-pair J integrals for a single wavenumber.

        Same-wire blocks are filled per-edge using the analytic + regular
        decomposition on same-edge pairs and full 3D quadrature on off-edge
        pairs. Cross-wire blocks use full 3D quadrature on the whole wire.
        """
        a = self.wire_radius
        N = geom["n_segs_total"]
        J00 = np.zeros((N, N), dtype=np.complex128)
        J10 = np.zeros_like(J00)
        J01 = np.zeros_like(J00)
        J11 = np.zeros_like(J00)

        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        n_w = len(per_wire)

        for w in range(n_w):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            n_edges_w = len(ed_off) - 1
            base = seg_off[w]
            for i_e in range(n_edges_w):
                for j_e in range(n_edges_w):
                    sli = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                    slj = slice(base + ed_off[j_e], base + ed_off[j_e + 1])
                    if i_e == j_e:
                        A00, A10, A01, A11 = _seg_seg_static_all(ed_arc[i_e], a)
                        R00, R10, R01, R11 = _seg_seg_reg_all(
                            ed_arc[i_e], a, k, self.n_qp_reg
                        )
                        J00[sli, slj] = A00 + R00
                        J10[sli, slj] = A10 + R10
                        J01[sli, slj] = A01 + R01
                        J11[sli, slj] = A11 + R11
                    else:
                        C00, C10, C01, C11 = _seg_seg_offedge_quad(
                            pw["seg_l"][ed_off[i_e] : ed_off[i_e + 1]],
                            pw["seg_r"][ed_off[i_e] : ed_off[i_e + 1]],
                            pw["seg_l"][ed_off[j_e] : ed_off[j_e + 1]],
                            pw["seg_r"][ed_off[j_e] : ed_off[j_e + 1]],
                            a,
                            k,
                            self.n_qp_off,
                        )
                        J00[sli, slj] = C00
                        J10[sli, slj] = C10
                        J01[sli, slj] = C01
                        J11[sli, slj] = C11

        for i_w in range(n_w):
            for j_w in range(n_w):
                if i_w == j_w:
                    continue
                pwi = per_wire[i_w]
                pwj = per_wire[j_w]
                sli = slice(seg_off[i_w], seg_off[i_w + 1])
                slj = slice(seg_off[j_w], seg_off[j_w + 1])
                C00, C10, C01, C11 = _seg_seg_offedge_quad(
                    pwi["seg_l"],
                    pwi["seg_r"],
                    pwj["seg_l"],
                    pwj["seg_r"],
                    a,
                    k,
                    self.n_qp_off,
                )
                J00[sli, slj] = C00
                J10[sli, slj] = C10
                J01[sli, slj] = C01
                J11[sli, slj] = C11

        return J00, J10, J01, J11

    def _build_J_blocks_batch(self, geom, k_array):
        """Batched (n_k, N, N) J integrals over a vector of wavenumbers."""
        a = self.wire_radius
        N = geom["n_segs_total"]
        n_k = len(k_array)
        J00 = np.zeros((n_k, N, N), dtype=np.complex128)
        J10 = np.zeros_like(J00)
        J01 = np.zeros_like(J00)
        J11 = np.zeros_like(J00)

        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        n_w = len(per_wire)

        for w in range(n_w):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            n_edges_w = len(ed_off) - 1
            base = seg_off[w]
            for i_e in range(n_edges_w):
                for j_e in range(n_edges_w):
                    sli = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                    slj = slice(base + ed_off[j_e], base + ed_off[j_e + 1])
                    if i_e == j_e:
                        A00, A10, A01, A11 = _seg_seg_static_all(ed_arc[i_e], a)
                        R00, R10, R01, R11 = _seg_seg_reg_all_batch(
                            ed_arc[i_e], a, k_array, self.n_qp_reg
                        )
                        J00[:, sli, slj] = A00[None, :, :] + R00
                        J10[:, sli, slj] = A10[None, :, :] + R10
                        J01[:, sli, slj] = A01[None, :, :] + R01
                        J11[:, sli, slj] = A11[None, :, :] + R11
                    else:
                        C00, C10, C01, C11 = _seg_seg_offedge_quad_batch(
                            pw["seg_l"][ed_off[i_e] : ed_off[i_e + 1]],
                            pw["seg_r"][ed_off[i_e] : ed_off[i_e + 1]],
                            pw["seg_l"][ed_off[j_e] : ed_off[j_e + 1]],
                            pw["seg_r"][ed_off[j_e] : ed_off[j_e + 1]],
                            a,
                            k_array,
                            self.n_qp_off,
                        )
                        J00[:, sli, slj] = C00
                        J10[:, sli, slj] = C10
                        J01[:, sli, slj] = C01
                        J11[:, sli, slj] = C11

        for i_w in range(n_w):
            for j_w in range(n_w):
                if i_w == j_w:
                    continue
                pwi = per_wire[i_w]
                pwj = per_wire[j_w]
                sli = slice(seg_off[i_w], seg_off[i_w + 1])
                slj = slice(seg_off[j_w], seg_off[j_w + 1])
                C00, C10, C01, C11 = _seg_seg_offedge_quad_batch(
                    pwi["seg_l"],
                    pwi["seg_r"],
                    pwj["seg_l"],
                    pwj["seg_r"],
                    a,
                    k_array,
                    self.n_qp_off,
                )
                J00[:, sli, slj] = C00
                J10[:, sli, slj] = C10
                J01[:, sli, slj] = C01
                J11[:, sli, slj] = C11

        return J00, J10, J01, J11

    def compute_impedance(self, *, ntrap=None):
        geom = self._build_geometry()
        J00, J10, J01, J11 = self._build_J_blocks(geom, self.k)

        left_seg = geom["left_seg"]
        right_seg = geom["right_seg"]
        h_per_seg = geom["h_per_seg"]
        tangents = geom["tangents"]

        hl_m = h_per_seg[left_seg][:, None]
        hl_n = h_per_seg[left_seg][None, :]
        hr_m = h_per_seg[right_seg][:, None]
        hr_n = h_per_seg[right_seg][None, :]

        S = (
            J00[np.ix_(left_seg, left_seg)] / (hl_m * hl_n)
            - J00[np.ix_(left_seg, right_seg)] / (hl_m * hr_n)
            - J00[np.ix_(right_seg, left_seg)] / (hr_m * hl_n)
            + J00[np.ix_(right_seg, right_seg)] / (hr_m * hr_n)
        )
        Z_Phi = S / (1j * self.omega * self.eps)

        td_all = tangents @ tangents.T
        td_ll = td_all[np.ix_(left_seg, left_seg)]
        td_lr = td_all[np.ix_(left_seg, right_seg)]
        td_rl = td_all[np.ix_(right_seg, left_seg)]
        td_rr = td_all[np.ix_(right_seg, right_seg)]

        I_A = (
            td_ll * (J11[np.ix_(left_seg, left_seg)] / (hl_m * hl_n))
            + td_lr
            * (
                J10[np.ix_(left_seg, right_seg)] / hl_m
                - J11[np.ix_(left_seg, right_seg)] / (hl_m * hr_n)
            )
            + td_rl
            * (
                J01[np.ix_(right_seg, left_seg)] / hl_n
                - J11[np.ix_(right_seg, left_seg)] / (hr_m * hl_n)
            )
            + td_rr
            * (
                J00[np.ix_(right_seg, right_seg)]
                - J10[np.ix_(right_seg, right_seg)] / hr_m
                - J01[np.ix_(right_seg, right_seg)] / hr_n
                + J11[np.ix_(right_seg, right_seg)] / (hr_m * hr_n)
            )
        )
        Z_A = 1j * self.omega * self.mu * I_A

        Z = Z_A + Z_Phi
        self.z = Z

        m_center = self._feed_basis_index(geom)
        v = np.zeros(geom["n_basis_total"], dtype=np.complex128)
        v[m_center] = 1.0
        coeffs = scipy.linalg.solve(Z, v)
        driver_impedance = 1.0 / coeffs[m_center]
        return driver_impedance, coeffs

    def compute_impedance_swept(self, k_array):
        """Driver impedance over a batch of wavenumbers, sharing all
        k-independent work (geometry, static kernel, basis stencil).
        """
        k_array = np.asarray(k_array, dtype=float)
        omega_array = k_array * self.c
        geom = self._build_geometry()
        J00, J10, J01, J11 = self._build_J_blocks_batch(geom, k_array)

        left_seg = geom["left_seg"]
        right_seg = geom["right_seg"]
        h_per_seg = np.ascontiguousarray(geom["h_per_seg"], dtype=np.float64)
        tangents = geom["tangents"]
        td_all = np.ascontiguousarray(tangents @ tangents.T, dtype=np.float64)

        if _HAVE_ASSEMBLE_Z:
            Z = _acc.assemble_Z(
                J00,
                J10,
                J01,
                J11,
                h_per_seg,
                td_all,
                left_seg,
                right_seg,
                np.ascontiguousarray(omega_array, dtype=np.float64),
                float(self.eps),
                float(self.mu),
            )
        else:
            hl_m = h_per_seg[left_seg][:, None]
            hl_n = h_per_seg[left_seg][None, :]
            hr_m = h_per_seg[right_seg][:, None]
            hr_n = h_per_seg[right_seg][None, :]

            ll = (slice(None), left_seg[:, None], left_seg[None, :])
            lr = (slice(None), left_seg[:, None], right_seg[None, :])
            rl = (slice(None), right_seg[:, None], left_seg[None, :])
            rr = (slice(None), right_seg[:, None], right_seg[None, :])

            S = (
                J00[ll] / (hl_m * hl_n)
                - J00[lr] / (hl_m * hr_n)
                - J00[rl] / (hr_m * hl_n)
                + J00[rr] / (hr_m * hr_n)
            )
            Z_Phi = S / (1j * omega_array[:, None, None] * self.eps)

            td_ll = td_all[np.ix_(left_seg, left_seg)][None, ...]
            td_lr = td_all[np.ix_(left_seg, right_seg)][None, ...]
            td_rl = td_all[np.ix_(right_seg, left_seg)][None, ...]
            td_rr = td_all[np.ix_(right_seg, right_seg)][None, ...]

            I_A = (
                td_ll * (J11[ll] / (hl_m * hl_n))
                + td_lr * (J10[lr] / hl_m - J11[lr] / (hl_m * hr_n))
                + td_rl * (J01[rl] / hl_n - J11[rl] / (hr_m * hl_n))
                + td_rr
                * (J00[rr] - J10[rr] / hr_m - J01[rr] / hr_n + J11[rr] / (hr_m * hr_n))
            )
            Z_A = 1j * omega_array[:, None, None] * self.mu * I_A
            Z = Z_A + Z_Phi

        m_center = self._feed_basis_index(geom)
        v = np.zeros(geom["n_basis_total"], dtype=np.complex128)
        v[m_center] = 1.0
        coeffs = np.linalg.solve(Z, v)
        return 1.0 / coeffs[:, m_center]
