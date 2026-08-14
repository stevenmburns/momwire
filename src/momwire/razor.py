"""Tent basis with razor-blade testing — the NEC-5 formulation twin.

NEC-5's Users Manual (§1) states its formulation: a linear (triangular)
current expansion tested by the mixed-potential method of Rao, Wilton and
Glisson — the E-field boundary condition enforced on *path integrals*
between the centroids of connected elements ("razor-blade" testing), NOT
the point matching of NEC-2/4 and NOT the Galerkin testing of momwire's
:class:`~momwire.bspline.BSplineSolver` at ``degree=1``. That single
difference in the testing rule is what produces NEC-5's slow O(1/N)
impedance walk (the momwire#890 / antennaknobs#896 finding); the manual
itself predicts slower convergence than a sinusoidal expansion.

This module is that formulation, written as a momwire solver so the walk is
reproducible without the NEC-5 binary. It is deliberately a *twin*, not an
improvement: the quadratures are converged, but the scheme's discretization
error is NEC-5's, so a fine mesh here walks the same way NEC-5's does.

Formulation
-----------
Unknowns are the interior knots of each wire — one unit tent Λ_n per
interior knot, rising linearly in arc length over the segment before the
knot and falling over the segment after, so the current vanishes at every
wire endpoint. Row m tests along the path P_m that runs from the centroid
of the segment before knot m, through the knot, to the centroid of the
segment after (two straight half-segments, bent at the knot on a kinked
wire):

    Z[m,n] = jωμ₀ · T1[m,n] − T2[m,n] / (jωε₀)

    T1[m,n] = ∫_{P_m} t̂(r) · A_n(r) dl        (A_n carries no μ₀)
    A_n(r)  = ∫ Λ_n(l') g(r,r') t̂(r') dl'
    T2[m,n] = ∫ Λ_n'(l') [ g(c_after, r') − g(c_before, r') ] dl'

with the reduced thin-wire kernel g = exp(−jkR)/(4πR),
R = sqrt(|r−r'|² + a²), the source running along the segment *axis*, and
Λ_n' the tent's ±1/h charge doublet. T1 carries the tangent dot product
(both tangents turn at a bend); T2 does not.

Excitation is NEC-5's EX-at-a-knot: the delta-gap voltage sits inside
exactly one testing path, so it lands entirely in that knot's row and the
solved tent coefficient at the feed knot *is* the drive-point current.

Scope (momwire#309, unit 1)
--------------------------
Free space, reduced kernel, one polyline per wire, no continuity across
wires — separate wires are electrically separate. Cross-wire junctions,
grounds, and the extended kernel are out of scope; each is refused with a
message rather than silently mismodelled. Cross-wire junction support is
unit 2 of momwire#309.
"""

import numpy as np
import scipy.linalg

from ._cancel import _Cancelable
from ._quadrature import leggauss

# Two wire endpoints this close are a junction, not a coincidence. The same
# tolerance the caller-facing geometry helpers use for "same point".
_JUNCTION_TOL = 1e-9

# Working-array budget for the chunked fills, in complex128 elements
# (~32 MB per temporary). The fill's inner tensor is
# (observation points) × (segments) × (source quadrature points), which for
# a 200-segment two-element model at the default orders would otherwise be
# a few hundred MB in one allocation.
_CHUNK_ELEMS = 2_000_000

# Constructor kwargs the sibling solvers accept that this formulation
# deliberately does not, with the reason each is refused. Anything else
# unexpected is a caller typo and stays a TypeError.
_OUT_OF_SCOPE = {
    "ground_z": "ground planes are out of scope for RazorSolver (free space "
    "only): NEC-5's ground is Michalski, which carries its own limit offset "
    "and would contaminate the formulation comparison this class exists for",
    "ground_eps": "finite ground is out of scope for RazorSolver (free space only)",
    "ground_model": "finite ground is out of scope for RazorSolver (free space only)",
    "ground_phi_mode": "finite ground is out of scope for RazorSolver (free "
    "space only)",
    "degree": "RazorSolver has no degree: the razor-blade testing rule is "
    "defined against the tent (degree-1) expansion. Use "
    "BSplineSolver(degree=...) for higher-order bases with Galerkin testing",
    "junctions": "cross-wire junctions are not supported yet — unit 2 of momwire#309",
    "junction_ports": "cross-wire junctions are not supported yet — unit 2 "
    "of momwire#309",
    "extended_kernel": "RazorSolver is reduced-kernel only: NEC-5's "
    "formulation is the comparison target, and its expansion is tested on "
    "the wire axis",
    "node_gaps": "node gaps are not supported yet — the delta-gap feed lands "
    "in a whole testing row here, so a gap is not a local basis edit",
}


def _axis_frame(obs, seg_p0, seg_t, a):
    """Project observation points onto every segment's axis.

    Returns ``(u_r, rho2)`` of shape ``(n_obs, n_seg)``: the signed axial
    coordinate of the projection, measured from the segment's start point,
    and the squared perpendicular distance plus a². In that frame the
    reduced kernel's distance to a source at local arc τ is simply
    R² = (τ − u_r)² + ρ², which is what both the closed-form static
    moments and the quadratured remainder consume.
    """
    d = obs[:, None, :] - seg_p0[None, :, :]
    u_r = np.einsum("psc,sc->ps", d, seg_t)
    # The perpendicular part can go a few ulps negative for a truly
    # collinear observation point; a² dominates it either way.
    rho2 = np.maximum(np.einsum("psc,psc->ps", d, d) - u_r * u_r, 0.0) + a * a
    return u_r, rho2


def _static_axis_moments(u_r, rho2, seg_h):
    """Closed-form static moments of the reduced kernel over each segment.

    Given the axis frame from :func:`_axis_frame`, returns ``(m0, m1)``:

        m0 = ∫₀^h dτ / R,   m1 = ∫₀^h τ dτ / R

    with τ the source's local arc length from the segment start and
    R = sqrt(|r − r'|² + a²). In the axis frame these are the collinear
    forms with u = τ − u_r: ∫du/R = asinh(u/ρ) and ∫u du/R = sqrt(u² + ρ²),
    then m1 = u_r·m0 + [sqrt(u² + ρ²)]. ρ ≥ a > 0 always, which is exactly
    what the reduced kernel's a² buys — the formula never sees a bare axis.
    """
    rho = np.sqrt(rho2)
    u0 = -u_r
    u1 = seg_h[None, :] - u_r
    m0 = np.arcsinh(u1 / rho) - np.arcsinh(u0 / rho)
    m1 = u_r * m0 + (np.sqrt(u1 * u1 + rho2) - np.sqrt(u0 * u0 + rho2))
    return m0, m1


class RazorSolver(_Cancelable):
    """Tent-basis MoM with razor-blade (mixed-potential path) testing.

    The NEC-5 formulation twin — see the module docstring for the physics.
    Free space, reduced kernel, one tent per interior knot, no continuity
    across wires.

    wires: list of (M_w, 3) polyline arrays, M_w >= 2 anchor points per wire.
        A straight dipole is a single two-anchor wire; an inverted-V is one
        three-anchor wire; a Yagi is several two-anchor wires.
    n_per_edge_per_wire: list of (int or sequence). Per-wire segment counts
        per polyline edge. None for a wire means use `nsegs` on each of its
        edges; an int means that count on each edge; a sequence gives a
        per-edge count. None for the whole argument means every wire uses
        `nsegs` on every edge.
    nsegs: default segment count when `n_per_edge_per_wire` doesn't specify.
    wire_radius: scalar thin-wire radius, the a in the reduced kernel's
        R = sqrt(|r−r'|² + a²). Per-wire radii are not supported here.
    wavelength: measurement wavelength in metres; k = 2π / wavelength.
    halfdriver_factor: informational only when `wires` is given explicitly
        (the polylines fully determine the geometry); kept for signature
        parity with the sibling solvers.
    feed_wire_index: wire carrying the delta-gap source on the single-feed
        path (ignored when `feeds` is supplied).
    feed_arclength: arc length along the feed wire, from its first anchor,
        at which to place the source. None picks the wire's midpoint. The
        source always snaps to the NEAREST INTERIOR KNOT: the razor testing
        paths are knot-centred, so a between-knots delta-gap has no row to
        land in. On an even segment count a midpoint feed is already the
        exact centre knot.
    feeds: optional list of (wire_index, arclength_or_None, voltage) tuples
        describing several delta-gap sources with prescribed complex
        voltages. `compute_impedance()` then returns the per-feed
        drive-point impedance vector V_i / I_i.
    n_qp_path: Gauss-Legendre order for the OUTER testing-path integral,
        per half-segment (T1 only; T2's path collapses to two endpoint
        evaluations).
    n_qp_source: Gauss-Legendre order per source segment for the smooth
        remainder (exp(−jkR)−1)/(4πR); the static 1/(4πR) part is analytic.
    cancel: optional :class:`~momwire._cancel.CancelToken`; polled at the
        phase boundaries (after geometry, between the fill chunks, before
        the dense solve).
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        nsegs=101,
        wire_radius=0.0005,
        wavelength=22,
        halfdriver_factor=0.962,
        feed_wire_index=0,
        feed_arclength=None,
        feeds=None,
        n_qp_path=32,
        n_qp_source=12,
        cancel=None,
        **unsupported,
    ):
        for name in unsupported:
            if name in _OUT_OF_SCOPE:
                raise NotImplementedError(f"{name}: {_OUT_OF_SCOPE[name]}")
        if unsupported:
            bad = ", ".join(sorted(unsupported))
            raise TypeError(f"RazorSolver got unexpected keyword argument(s): {bad}")

        self._cancel = cancel
        self._checkpoint()

        self.wavelength = float(wavelength)
        self.halfdriver_factor = float(halfdriver_factor)
        self.nsegs = int(nsegs)
        if not np.isscalar(wire_radius):
            raise NotImplementedError(
                "wire_radius must be a scalar for RazorSolver; per-wire radii "
                "(momwire#147) are not supported by this formulation twin"
            )
        self.wire_radius = float(wire_radius)
        if self.wire_radius <= 0.0:
            raise ValueError("wire_radius must be positive (reduced kernel)")

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = 2 * np.pi / self.wavelength
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        self.n_qp_path = int(n_qp_path)
        self.n_qp_source = int(n_qp_source)
        if self.n_qp_path < 1 or self.n_qp_source < 1:
            raise ValueError("n_qp_path and n_qp_source must be >= 1")

        if not wires:
            raise ValueError("wires must be non-empty")
        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")
        self._refuse_junctions()

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
            npe = [int(v) for v in npe]
            if len(npe) != n_edges_w:
                raise ValueError(
                    f"wire {i}: n_per_edge length {len(npe)} "
                    f"!= number of edges {n_edges_w}"
                )
            if any(v < 1 for v in npe):
                raise ValueError(f"wire {i}: every edge needs >= 1 segment")
            self.n_per_edge_per_wire.append(npe)

        for i, npe in enumerate(self.n_per_edge_per_wire):
            if sum(npe) < 2:
                raise ValueError(
                    f"wire {i}: needs >= 2 segments to have an interior knot "
                    "(a one-segment wire carries no tent)"
                )

        if feeds is None:
            if not (0 <= feed_wire_index < n_w):
                raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
            self.feeds = [(int(feed_wire_index), feed_arclength, 1.0 + 0.0j)]
        else:
            if len(feeds) == 0:
                raise ValueError("feeds must contain at least one entry")
            norm = []
            for i, f in enumerate(feeds):
                if len(f) != 3:
                    raise ValueError(
                        f"feeds[{i}]: expected (wire_index, arclength, voltage), "
                        f"got {f!r}"
                    )
                w_i, arc_i, v_i = f
                if not (0 <= w_i < n_w):
                    raise ValueError(
                        f"feeds[{i}]: wire_index {w_i} out of range [0, {n_w})"
                    )
                norm.append(
                    (int(w_i), None if arc_i is None else float(arc_i), complex(v_i))
                )
            self.feeds = norm
        self.feed_wire_index = self.feeds[0][0]
        self.feed_arclength = self.feeds[0][1]

        self.z = None
        self._cached_geometry = None

    # ------------------------------------------------------------------
    # geometry

    def _refuse_junctions(self):
        """Refuse a geometry where two wires share an endpoint.

        With no cross-wire continuity, a shared endpoint would silently
        model two wires that touch as two wires whose currents both go to
        zero at the contact — physically wrong rather than approximate, so
        it is an error, not a warning.
        """
        ends = []
        for i, pl in enumerate(self.wires_polylines):
            ends.append((i, "start", pl[0]))
            ends.append((i, "end", pl[-1]))
        for p in range(len(ends)):
            for q in range(p + 1, len(ends)):
                wi, ei, pi = ends[p]
                wj, ej, pj = ends[q]
                if wi == wj:
                    continue
                if float(np.linalg.norm(pi - pj)) <= _JUNCTION_TOL:
                    raise NotImplementedError(
                        f"wire {wi} {ei} and wire {wj} {ej} share the point "
                        f"{np.round(pi, 12).tolist()}: cross-wire junctions are "
                        "not supported by RazorSolver yet (unit 2 of "
                        "momwire#309). Model the connected structure as a "
                        "single polyline wire, or use BSplineSolver, which "
                        "carries junction bases with a KCL constraint."
                    )

    def _build_geometry(self):
        """Discretize every wire into segments and concatenate.

        Segments are stored by start point, unit tangent and length, which
        is the form both the static moments and the testing paths want.
        `left_seg` / `right_seg` name the two segments meeting at each
        interior knot: basis n rises over `left_seg[n]` and falls over
        `right_seg[n]`, and row n's testing path runs from `left_seg[n]`'s
        centroid to `right_seg[n]`'s.
        """
        if self._cached_geometry is not None:
            return self._cached_geometry

        per_wire = []
        seg_offsets = [0]
        basis_offsets = [0]
        p0_list, tan_list, h_list = [], [], []
        for w_idx, (pl, npe) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            w_p0, w_tan, w_h = [], [], []
            for e_idx in range(pl.shape[0] - 1):
                edge = pl[e_idx + 1] - pl[e_idx]
                edge_len = float(np.linalg.norm(edge))
                if edge_len < 1e-15:
                    raise ValueError(f"wire {w_idx} edge {e_idx} has zero length")
                tan = edge / edge_len
                n_e = npe[e_idx]
                h_e = edge_len / n_e
                w_p0.append(pl[e_idx][None, :] + np.arange(n_e)[:, None] * h_e * tan)
                w_tan.append(np.tile(tan, (n_e, 1)))
                w_h.append(np.full(n_e, h_e))
            seg_p0 = np.vstack(w_p0)
            seg_t = np.vstack(w_tan)
            seg_h = np.concatenate(w_h)
            n_total = seg_p0.shape[0]
            per_wire.append(
                {
                    "arc_at_knot": np.concatenate([[0.0], np.cumsum(seg_h)]),
                    "n_total": n_total,
                }
            )
            seg_offsets.append(seg_offsets[-1] + n_total)
            basis_offsets.append(basis_offsets[-1] + n_total - 1)
            p0_list.append(seg_p0)
            tan_list.append(seg_t)
            h_list.append(seg_h)

        left = np.concatenate(
            [
                seg_offsets[w] + np.arange(per_wire[w]["n_total"] - 1, dtype=np.int64)
                for w in range(len(per_wire))
            ]
        )
        geom = {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "basis_offsets": basis_offsets,
            "n_segs_total": seg_offsets[-1],
            "n_basis_total": basis_offsets[-1],
            "seg_p0": np.vstack(p0_list),
            "seg_t": np.vstack(tan_list),
            "seg_h": np.concatenate(h_list),
            "left_seg": left,
            "right_seg": left + 1,
        }
        if geom["n_basis_total"] == 0:
            raise ValueError("no unknowns: every wire needs >= 2 segments")
        self._cached_geometry = geom
        return geom

    def _feed_basis_indices(self, geom):
        """Global basis index of each feed's knot.

        Each feed snaps to the interior knot of its wire whose arc length
        from that wire's first anchor is closest to the requested value
        (None → the wire's midpoint).
        """
        idx = []
        for w, arc, _v in self.feeds:
            arc_at_knot = geom["per_wire"][w]["arc_at_knot"]
            target = arc if arc is not None else arc_at_knot[-1] / 2.0
            m_local = int(np.argmin(np.abs(arc_at_knot[1:-1] - target)))
            idx.append(int(geom["basis_offsets"][w] + m_local))
        return idx

    # ------------------------------------------------------------------
    # kernel moments

    def _seg_moments(self, obs, geom, k, *, need_m1=True):
        """Reduced-kernel moments of every segment at every observation point.

        Returns ``(M0, M1)`` of shape ``(n_obs, n_seg)`` with

            M0 = ∫_seg g(r, r') dl',   M1 = ∫_seg τ' g(r, r') dl'

        (τ' the source's local arc from its segment start, g the reduced
        kernel *including* the 1/4π). The static 1/(4πR) part is the
        closed form in :func:`_static_axis_moments`; the remainder
        (exp(−jkR)−1)/(4πR) is smooth everywhere the reduced kernel is
        defined and takes plain Gauss-Legendre. `M1` is None when
        `need_m1` is False — the scalar-potential term only needs M0.

        Chunked over observation points: the remainder's working tensor is
        n_obs × n_seg × n_qp_source, which is the fill's memory high-water
        mark.
        """
        a = self.wire_radius
        seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
        n_seg = seg_h.size
        xg, wg = leggauss(self.n_qp_source)
        # Source quadrature in each segment's own local arc coordinate.
        tau = 0.5 * seg_h[:, None] * (1.0 + xg[None, :])
        wq = 0.5 * seg_h[:, None] * wg[None, :]

        n_obs = obs.shape[0]
        M0 = np.empty((n_obs, n_seg), dtype=np.complex128)
        M1 = np.empty((n_obs, n_seg), dtype=np.complex128) if need_m1 else None
        step = max(1, _CHUNK_ELEMS // max(1, n_seg * self.n_qp_source))
        inv4pi = 1.0 / (4.0 * np.pi)
        for lo in range(0, n_obs, step):
            self._checkpoint()
            hi = min(lo + step, n_obs)
            u_r, rho2 = _axis_frame(obs[lo:hi], seg_p0, seg_t, a)
            m0s, m1s = _static_axis_moments(u_r, rho2, seg_h)
            u = tau[None, :, :] - u_r[:, :, None]
            R = np.sqrt(u * u + rho2[:, :, None])
            rem = (np.exp(-1j * k * R) - 1.0) / R
            M0[lo:hi] = (m0s + np.einsum("psq,sq->ps", rem, wq)) * inv4pi
            if need_m1:
                M1[lo:hi] = (m1s + np.einsum("psq,sq->ps", rem, tau * wq)) * inv4pi
        return M0, M1

    # ------------------------------------------------------------------
    # assembly

    def _testing_paths(self, geom):
        """Quadrature points, tangents and weights of every testing path.

        Path P_m runs centroid(left_seg[m]) → knot m → centroid(right_seg[m]).
        Segment `right_seg[m]` starts at the knot, so the knot is just its
        start point. Each half gets its own `n_qp_path` Gauss-Legendre rule
        and its own segment tangent — on a kinked wire the two halves point
        in different directions, which is exactly what T1's tangent dot
        product has to see.

        Returns ``(pts, tans, wts)`` shaped (n_basis, 2·n_qp_path, 3/3/–).
        """
        seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
        left, right = geom["left_seg"], geom["right_seg"]
        cent = seg_p0 + 0.5 * seg_h[:, None] * seg_t
        knot = seg_p0[right]
        xo, wo = leggauss(self.n_qp_path)

        pts, tans, wts = [], [], []
        for lo_pt, hi_pt, seg in ((cent[left], knot, left), (knot, cent[right], right)):
            mid = 0.5 * (lo_pt + hi_pt)
            half = 0.5 * (hi_pt - lo_pt)
            pts.append(mid[:, None, :] + half[:, None, :] * xo[None, :, None])
            tans.append(np.repeat(seg_t[seg][:, None, :], xo.size, axis=1))
            # Each half-path is h/2 long, so the Gauss-Legendre Jacobian
            # that maps [-1, 1] onto it is h/4, not h/2.
            wts.append(0.25 * seg_h[seg][:, None] * wo[None, :])
        return (
            np.concatenate(pts, axis=1),
            np.concatenate(tans, axis=1),
            np.concatenate(wts, axis=1),
        )

    def _assemble_Z(self, geom, k):
        """Fill the razor-blade impedance matrix.

        Both terms are built from the same two segment moments. For tent n
        with rising segment ``l`` and falling segment ``r`` of length h:

            rise part of A_n = M1[l] / h_l        (Λ_n = τ'/h_l there)
            fall part of A_n = M0[r] − M1[r] / h_r   (Λ_n = 1 − τ'/h_r)

        each carried by its own segment tangent, so the outer path point's
        tangent contracts with them separately. The charge doublet is the
        same moments' M0 differenced between the path's two bounding
        centroids: Λ_n' = +1/h_l on the rising segment, −1/h_r on the
        falling one.
        """
        seg_t, seg_h = geom["seg_t"], geom["seg_h"]
        left, right = geom["left_seg"], geom["right_seg"]
        n_basis = left.size
        h_l, h_r = seg_h[left], seg_h[right]

        # --- scalar potential: M0 at every segment centroid.
        cent = geom["seg_p0"] + 0.5 * seg_h[:, None] * seg_t
        M0c, _ = self._seg_moments(cent, geom, k, need_m1=False)
        dM0 = M0c[right] - M0c[left]  # (row, source segment)
        T2 = dM0[:, left] / h_l[None, :] - dM0[:, right] / h_r[None, :]

        # --- vector potential: the outer path integral, row-chunked.
        pts, tans, wts = self._testing_paths(geom)
        n_path = pts.shape[1]
        td_left = seg_t[left].T  # (3, n_basis)
        td_right = seg_t[right].T
        T1 = np.empty((n_basis, n_basis), dtype=np.complex128)
        rows = max(1, _CHUNK_ELEMS // max(1, n_path * n_basis))
        for lo in range(0, n_basis, rows):
            self._checkpoint()
            hi = min(lo + rows, n_basis)
            obs = pts[lo:hi].reshape(-1, 3)
            M0, M1 = self._seg_moments(obs, geom, k)
            rise = M1[:, left] / h_l[None, :]
            fall = M0[:, right] - M1[:, right] / h_r[None, :]
            t_out = tans[lo:hi].reshape(-1, 3)
            integrand = (t_out @ td_left) * rise + (t_out @ td_right) * fall
            integrand *= wts[lo:hi].reshape(-1)[:, None]
            T1[lo:hi] = integrand.reshape(hi - lo, n_path, n_basis).sum(axis=1)

        return 1j * self.omega * self.mu * T1 - T2 / (1j * self.omega * self.eps)

    # ------------------------------------------------------------------
    # solve

    def compute_impedance(self):
        """Drive-point impedance(s) and the solved tent coefficients.

        Returns ``(z, coeffs)``. With one feed `z` is the scalar
        V / I at the feed knot; with N feeds it is the length-N vector
        V_i / I_i, one entry per feed, from the single solve against the
        superposed right-hand side Σ_i V_i e_{m_i}. `coeffs` is the solved
        coefficient vector — on a tent basis the coefficient at a knot IS
        the current there (amperes), so `coeffs` is the knot-current
        distribution in basis order (per wire, interior knots in arc order).
        """
        geom = self._build_geometry()
        self._checkpoint()
        Z = self._assemble_Z(geom, self.k)
        self.z = Z

        idx = self._feed_basis_indices(geom)
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        # NEC-5's EX at a knot: the delta gap sits inside exactly one
        # testing path, so the whole voltage lands in that one row.
        rhs = np.zeros(geom["n_basis_total"], dtype=np.complex128)
        for m_i, v_i in zip(idx, voltages):
            rhs[m_i] += v_i

        self._checkpoint()
        coeffs = scipy.linalg.solve(Z, rhs)
        feed_currents = np.array([coeffs[m] for m in idx], dtype=np.complex128)
        z_per_feed = voltages / feed_currents
        return (z_per_feed[0] if len(self.feeds) == 1 else z_per_feed), coeffs
