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
free wire endpoint — plus one junction tent per independent through-path
where wire ends meet (see "Junctions" below). Row m tests along the path
P_m that runs from the centroid of the segment before knot m, through the
knot, to the centroid of the segment after (two straight half-segments,
bent at the knot on a kinked wire):

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

Junctions (momwire#309, unit 2)
-------------------------------
A junction is a point where wire ENDS coincide (within `_JUNCTION_TOL`),
including the single wire whose start and end coincide — a closed loop.
They are detected from the geometry; there is no junction spec to write.

K coincident ends carry K−1 independent through-currents, so a junction
gets K−1 extra bases. The first-listed end at the point is the reference
side A; every other end B_i gets one junction tent spanning A's terminal
segment and B_i's, rising linearly to 1 at the junction on both sides.
Its coefficient is the current flowing THROUGH the junction from A into
B_i, in amperes, and its row is the razor path centroid(A) → junction →
centroid(B_i) traversed in the direction of that flow. Because a wire may
join by either of its ends, each half carries a sign relative to the
wire's own arc direction; the charge doublet is then +1/h_A on A's
terminal segment and −1/h_B on B_i's, which is Kirchhoff's law written
into the basis rather than enforced by a constraint row. An interior-knot
tent is the K=2 junction tent of a wire split at that knot, exactly — the
split identities in `tests/test_razor_junctions.py` pin that.

Field readout and sweeps (momwire#309, unit 3)
------------------------------------------------
`currents_at_knots` reads the solved coefficients back as one current per
mesh knot per wire — see its docstring for how a junctioned end's current
is rebuilt from its tents' wing signs, since this formulation's junction
basis is not a per-wire end basis the way the retired triangular-family
solver's was. `element_currents` (the `_ElementCurrents` mixin) rides on
top of it. `compute_impedance_swept` / `compute_y_matrix_swept` solve a
batch of wavenumbers, sharing the wing/path stencils and the closed-form
static segment moments across the sweep (`_assemble_Z_prepare`) so only
the smooth kernel remainder and the ω-dependent prefactors are redone per
k; every solved point is still its own dense LU.

Ground (momwire#398 unit 2)
--------------------------
`ground_z` puts a perfectly conducting plane under the model. The fill
becomes `Z = Z_free − Z_image`: the same rows, the same testing paths and
the same tent basis, evaluated a second time against sources reflected
through the plane, subtracted once. The mirror itself comes from
`_potential_ground.PotentialGround` — this module writes no reflection of
its own, which is the point of the pilot `docs/design/solver-architecture.md`
§6 proposes: a ground method landing on a solver that never implemented
one. `_assemble_Z_from_prepared` carries the sign argument.

Finite grounds are refused: agreement there has its own bar and its own
unit, and the fold hard-codes the image coefficient 1 (momwire#282).

Ground contact (momwire#398 unit 3)
-----------------------------------
A wire END may lie IN the plane. Such an end keeps a degree of freedom
instead of being zeroed the way the tent basis zeroes a free end: its basis
is the junction tent between the wire and its own image — a monopole plus
its image IS a dipole — whose upper wing is the real contact segment and
whose lower wing is that segment mirrored. Only the upper wing is spelled
in the basis tables, because the fold already evaluates every basis against
the mirrored sources: the image wing arrives with the right shape, the
right direction (−M·t̂, parallel to the real current for a vertical
contact) and the opposite charge, so no net charge sits at the contact
point.

The grounded tent's testing path is the REAL half only, from the contact
point to the centroid of the contact segment, in the direction the current
leaves the plane. The image half of that path contributes the identical
number (the total field of a system that is its own image obeys
E(M·r) = −M·E(r), which is exactly the invariance the razor path integral
contracts), so taking the real half alone halves the row — and halving the
row is what makes the feed voltage the voltage of the base GAP rather than
of the equivalent dipole's whole gap. On that spelling a base-fed monopole
returns Z_dipole/2 to LU roundoff, which is the first gate in
`tests/test_razor_ground_contact.py`. The row's other endpoint is the
plane, where the folded scalar potential is identically zero, so the T2
term keeps only the segment-centroid end.

Scope
-----
Free space and the PEC image, reduced kernel, one polyline per wire.
Finite grounds and the extended kernel are out of scope; each is refused
with a message rather than silently mismodelled, as are the contacts that
are not a wire end in the plane (an interior anchor touching down, an edge
lying in the plane, a wire dipping below it). Only wire ENDS junction: a
wire end touching another wire's interior is not a contact here. A wire
with a single segment cannot take part in a junction (its two junction
tents would overlap on one segment) and is refused.
"""

import numpy as np
import scipy.linalg

from . import _potential_ground
from ._cancel import _Cancelable
from ._capabilities import Capabilities
from ._element_currents import _ElementCurrents
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
    "ground_eps": "finite ground is out of scope for RazorSolver (PEC image "
    "only): NEC-5's finite ground is Michalski, which carries its own limit "
    "offset and would contaminate the formulation comparison this class "
    "exists for. The PEC image carries no such offset — it is the same exact "
    "image NEC-5's own `GN 1` uses — which is why it is the ground that "
    "landed first (momwire#398 unit 2). Ground CONTACT over a finite ground "
    "is refused twice over: the fold hard-codes image coefficient 1, i.e. "
    "PEC, so a grounded end over anything else would take spurious contact "
    "charge (momwire#282)",
    "ground_model": "finite ground is out of scope for RazorSolver (PEC image "
    "only): see the `ground_eps` refusal for why the finite-ground bar is "
    "deferred rather than merely unwritten, and why ground contact over a "
    "finite ground stays refused with it (momwire#282)",
    "ground_phi_mode": "finite ground is out of scope for RazorSolver (PEC "
    "image only): the image-charge weighting is a reflection-coefficient "
    "knob, and RazorSolver serves no reflection-coefficient ground",
    "degree": "RazorSolver has no degree: the razor-blade testing rule is "
    "defined against the tent (degree-1) expansion. Use "
    "BSplineSolver(degree=...) for higher-order bases with Galerkin testing",
    "junctions": "RazorSolver takes no junction spec: junctions are detected "
    "from the geometry (coincident wire ends), so listing them is either "
    "redundant or a disagreement with the mesh",
    "junction_ports": "junction ports are not supported: a junction basis is "
    "already a through-current unknown, and a source at a K>=3 junction has no "
    "unambiguous branch pair to drive",
    "extended_kernel": "RazorSolver is reduced-kernel only: NEC-5's "
    "formulation is the comparison target, and its expansion is tested on "
    "the wire axis",
    "node_gaps": "node gaps are not supported yet — the delta-gap feed lands "
    "in a whole testing row here, so a gap is not a local basis edit",
}

# Reused by the wire_radius scalar-only check in __init__ (momwire#147: this
# formulation twin takes no per-wire radii) and by `capabilities.refusals`
# below — one message, not a copy in each.
_PER_WIRE_RADIUS_REFUSAL = (
    "wire_radius must be a scalar for RazorSolver; per-wire radii "
    "(momwire#147) are not supported by this formulation twin"
)


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


class RazorSolver(_ElementCurrents, _Cancelable):
    """Tent-basis MoM with razor-blade (mixed-potential path) testing.

    The NEC-5 formulation twin — see the module docstring for the physics.
    Free space or a PEC ground, reduced kernel, one tent per interior knot
    plus K−1 through-current tents wherever K wire ends meet.

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
    ground_z: height of a perfectly conducting ground plane, or None for
        free space. PEC ONLY, and that is a statement about which bar has
        been met rather than about which physics is hard. Over PEC this
        solver is gated against the licensed NEC-5 binary's own `GN 1`
        printouts on four ladder geometries, at the sharp tolerance the
        formulation twin can actually hold (`tests/test_razor_pec_ground.py`)
        — the exact image is the same object in both codes, so there is
        nothing to disagree about but quadrature. A finite ground is not:
        NEC-5's is Michalski, carrying a limit offset of its own that would
        contaminate the formulation comparison this class exists for, so
        `ground_eps` / `ground_model` / `ground_phi_mode` stay refused until
        that bar is written (momwire#398 §6; `BSplineSolver` serves them
        today). A wire END may lie in the plane — ground CONTACT, the
        grounded-end tent whose lower wing is its own image (momwire#398
        unit 3, and the module docstring for the physics) — which is what
        the vertical/monopole class needs. A wire that dips below the
        plane, has an edge lying in it, or touches it at an interior
        anchor is refused instead.
    wavelength: measurement wavelength in metres; k = 2π / wavelength.
    halfdriver_factor: informational only when `wires` is given explicitly
        (the polylines fully determine the geometry); kept for signature
        parity with the sibling solvers.
    feed_wire_index: wire carrying the delta-gap source on the single-feed
        path (ignored when `feeds` is supplied).
    feed_arclength: arc length along the feed wire, from its first anchor,
        at which to place the source. None picks the wire's midpoint. The
        source always snaps to the NEAREST KNOT THAT CARRIES A BASIS on
        that wire — every interior knot, plus either end that meets other
        wire ends at a junction, plus either end that touches the ground
        plane: the razor testing paths are knot-centred, so a between-knots
        delta-gap has no row to land in. On an even segment count a
        midpoint feed is already the exact centre knot. A feed that snaps
        to a junction of K >= 3 ends is refused (which branch pair the
        source drives is ambiguous); K = 2 — the ordinary split-wire feed —
        drives that junction's through-current basis. A base-fed monopole
        is `feed_arclength=0.0` on a wire whose first anchor lies in the
        plane (or the wire's full arc length, if it is the last anchor that
        touches): the source is then the gap between the plane and that
        end, and V / I is the drive-point impedance measured there.
    feeds: optional list of (wire_index, arclength_or_None, voltage) tuples
        describing several delta-gap sources with prescribed complex
        voltages. `compute_impedance()` then returns the per-feed
        drive-point impedance vector V_i / I_i.
    n_qp_path: Gauss-Legendre order for the OUTER testing-path integral,
        per half-segment (T1 only; T2's path collapses to two endpoint
        evaluations). Ignored under `nec5_quadrature`.
    nec5_quadrature: evaluate ∫A·dl by NEC-5's own rule — the two-point
        trapezoid at the path-end centroids (every potential at element
        centroids, the classic mixed-potential idiom), identified by the
        momwire#316 residue study: with it, the free-space ByDipole1
        ladder matches NEC-5's printouts to a CONSTANT −0.004−0.037j Ω at
        every rung, pair-walk signature identical to the third decimal.
        Default False keeps the converged Gauss-Legendre path integral, so
        the walk you see is the SCHEME's discretization error, cleanly.
        On non-uniform meshes each wing weighs its own half-path length
        (h_wing/2) at its centroid — the element-local reading, identical
        to the whole-path trapezoid on the uniform meshes the rule was
        identified on. Demonstration mode for the census rationale, not an
        NEC-5 substitute (the licensed binary stays the oracle).
    n_qp_source: Gauss-Legendre order per source segment for the smooth
        remainder (exp(−jkR)−1)/(4πR); the static 1/(4πR) part is analytic.
    cancel: optional :class:`~momwire._cancel.CancelToken`; polled at the
        phase boundaries (after geometry, between the fill chunks, before
        the dense solve).
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    # momwire#396: free space plus the PEC image (momwire#398 unit 2), no
    # finite ground / loading / EK / junction_ports / node_gaps / per-wire
    # radii / enrichment — the rest of the row is refused, reusing
    # `_OUT_OF_SCOPE`'s prose (built at __init__ from unsupported kwargs)
    # plus the wire_radius scalar-only check for per_wire_radius.
    capabilities = Capabilities(
        grounds=frozenset({"pec"}),
        wire_loading=False,
        extended_kernel=False,
        junction_ports=False,
        node_gaps=False,
        per_wire_radius=False,
        singular_enrichment=False,
        refusals={
            "refl-coef": _OUT_OF_SCOPE["ground_eps"],
            "sommerfeld": _OUT_OF_SCOPE["ground_model"],
            "junction_ports": _OUT_OF_SCOPE["junction_ports"],
            "node_gaps": _OUT_OF_SCOPE["node_gaps"],
            "extended_kernel": _OUT_OF_SCOPE["extended_kernel"],
            "per_wire_radius": _PER_WIRE_RADIUS_REFUSAL,
        },
    )

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        nsegs=101,
        wire_radius=0.0005,
        ground_z=None,
        wavelength=22,
        halfdriver_factor=0.962,
        feed_wire_index=0,
        feed_arclength=None,
        feeds=None,
        n_qp_path=32,
        n_qp_source=12,
        nec5_quadrature=False,
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
            raise NotImplementedError(_PER_WIRE_RADIUS_REFUSAL)
        self.wire_radius = float(wire_radius)
        if self.wire_radius <= 0.0:
            raise ValueError("wire_radius must be positive (reduced kernel)")

        # The ground, in the four attributes `_potential_ground`'s factory
        # reads. Only the PEC plane is served; the other three are refused
        # above via `_OUT_OF_SCOPE` and are pinned to None here so the
        # factory's own branching lands on the PEC row and nowhere else.
        self.ground_z = None if ground_z is None else float(ground_z)
        if self.ground_z is not None and not np.isfinite(self.ground_z):
            raise ValueError("ground_z must be finite")
        self.ground_eps = None
        self.ground_model = None
        self.ground_phi_mode = None

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = 2 * np.pi / self.wavelength
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        self.n_qp_path = int(n_qp_path)
        self.n_qp_source = int(n_qp_source)
        self.nec5_quadrature = bool(nec5_quadrature)
        if self.n_qp_path < 1 or self.n_qp_source < 1:
            raise ValueError("n_qp_path and n_qp_source must be >= 1")

        if not wires:
            raise ValueError("wires must be non-empty")
        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")
        # Validates the geometry against the plane (and is re-read at
        # basis-build time for the grounded ends themselves).
        self._ground_ends()

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
                    "(a one-segment wire carries no tent, and if it were "
                    "junctioned at both ends its two junction tents would "
                    "overlap on that one segment — split it in two)"
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

    def _ground_ends(self):
        """Which wire ENDS lie in the ground plane; everything else refused.

        Returns the frozen set of ``(wire_index, "start" | "end")`` whose
        anchor is in the plane — empty in free space. A member of that set
        is a grounded end, and gets the grounded tent
        (:meth:`_junction_wings`) whose lower wing is its own image.

        The three geometries that are NOT ground contact, and stay refused:

        * a wire that dips BELOW `ground_z` is a geometry error — there is
          no such antenna — and raises `ValueError`, the same reading and
          the same wording `BSplineSolver._wire_endpoint_status` gives it;
        * an EDGE lying in the plane (both its anchors at `ground_z`) is
          degenerate over a conducting ground: the edge coincides with its
          own image, so the fold cancels it and it carries no independent
          current. `ValueError`, `BSplineSolver`'s wording again;
        * an INTERIOR anchor in the plane is a wire that touches down
          mid-span. That is real physics — the knot there would carry its
          ordinary tent AND a second unknown for the current leaving into
          the plane — but it is a second basis change on top of this one
          and is not written, so `NotImplementedError`.

        A straight edge takes its minimum z at an anchor, so scanning the
        anchors sees every contact there is.

        The touch tolerance is `BSplineSolver._ground_touch_tol`'s: 1e-6 of
        the wire's polyline length, loose enough for deck-import float
        noise at z=0 and far tighter than any deliberate stand-off.
        """
        gz = self.ground_z
        if gz is None:
            return frozenset()
        touching = set()
        for i, pl in enumerate(self.wires_polylines):
            length = float(np.sum(np.linalg.norm(np.diff(pl, axis=0), axis=1)))
            tol = 1e-6 * max(length, 1e-30)
            zmin = float(pl[:, 2].min())
            if zmin < gz - tol:
                raise ValueError(
                    f"wire {i} dips below the ground plane "
                    f"(min z = {zmin:.6g} < ground_z = {gz:g})"
                )
            at = np.abs(pl[:, 2] - gz) <= tol
            if np.any(at[:-1] & at[1:]):
                raise ValueError(
                    f"wire {i} has an edge lying in the ground plane "
                    "(both endpoints at ground_z) — degenerate over a "
                    "conducting ground"
                )
            if np.any(at[1:-1]):
                raise NotImplementedError(
                    f"wire {i} touches the ground plane at an interior "
                    "anchor: RazorSolver serves ground contact at a wire END "
                    "only (momwire#398 unit 3). A mid-span touchdown needs a "
                    "second unknown at a knot that already carries a tent — "
                    "split the wire there, so the contact is two wire ends"
                )
            if at[0]:
                touching.add((i, "start"))
            if at[-1]:
                touching.add((i, "end"))
        return frozenset(touching)

    def _find_junctions(self):
        """Group coincident wire ends into junctions, grounded ones marked.

        Returns a list of ``{"ends": [...], "grounded": bool}``; `ends` is a
        list of ``(wire_index, "start" | "end")`` in the order the ends are
        listed — first wire first, and `start` before `end` within a wire.
        The first entry is the reference side A of every junction tent
        there, so this order is part of the basis definition (not that the
        answer depends on it: picking a different reference re-spells the
        same current space).

        A wire whose own two ends coincide is a closed loop and forms a
        group of two on its own. Grouping is by first match within
        `_JUNCTION_TOL`, the same "same point" tolerance the caller-facing
        geometry helpers use.

        A group survives if it carries a through-path (K >= 2 ends) or if it
        is GROUNDED, because the plane is then one more branch at the point:
        a lone grounded end is a group of one, and its one tent is the
        current leaving into the ground. `grounded` is `any` over the
        group's ends rather than `all` — the two tolerances differ (the
        touch tolerance scales with each wire's own length, the grouping
        tolerance is absolute), so coincident ends could otherwise disagree
        about a plane they share.
        """
        grounded_ends = self._ground_ends()
        ends = []
        for i, pl in enumerate(self.wires_polylines):
            ends.append((i, "start", pl[0]))
            ends.append((i, "end", pl[-1]))

        groups, points = [], []
        for w, kind, p in ends:
            for g, q in enumerate(points):
                if float(np.linalg.norm(p - q)) <= _JUNCTION_TOL:
                    groups[g].append((w, kind))
                    break
            else:
                groups.append([(w, kind)])
                points.append(p)
        out = []
        for g in groups:
            grounded = any(e in grounded_ends for e in g)
            if len(g) >= 2 or grounded:
                out.append({"ends": g, "grounded": grounded})
        return out

    def _junction_wings(self, seg_offsets, group):
        """Wing descriptors for one junction group's tents.

        Yields ``(seg_a, rise_a, sigma_a, seg_b, rise_b, sigma_b)`` per
        tent. `rise` says the junction sits at the segment's far (arc-h)
        end, so the tent rises with the segment's own arc coordinate;
        `sigma` turns +1 A of through-current into a signed multiple of
        that segment's arc direction. Side A carries the current INTO the
        junction (so +1 along arc if the wire joins by its end, −1 if by
        its start) and side B carries it back OUT (the mirror image). A
        free-air group of K ends yields K−1 such tents.

        A GROUNDED group yields K tents instead, one per end (momwire#398
        unit 3): the plane is one more branch, so K real ends meeting there
        carry K independent currents and no through-path is distinguished.
        Each grounded tent is the junction tent between a wire end and its
        OWN IMAGE — the through-current tent of the monopole-plus-image
        dipole, whose upper wing is the real contact segment and whose
        lower wing is that segment mirrored. Only the real wing is spelled
        here: the fold `Z = Z_free − Z_image` already evaluates every basis
        against the mirrored sources, so the image wing arrives with the
        right shape (the mirror preserves each segment's local arc
        coordinate), the right direction (−M·t̂, parallel for a vertical
        contact) and the opposite charge, for free. Side A — the image side
        — is therefore spelled as a wing carrying `sigma = 0`, which zeroes
        its tangent (T1), its charge doublet (T2) and its half of the
        testing path at once; its `(segment, rise)` copy side B's so that
        `_knot_points` still reads the contact point off wing A.
        """
        if group["grounded"]:
            for w, kind in group["ends"]:
                seg = seg_offsets[w + 1] - 1 if kind == "end" else seg_offsets[w]
                rise = kind == "end"
                yield (seg, rise, 0.0, seg, rise, -1.0 if rise else 1.0)
            return
        (w_a, kind_a) = group["ends"][0]
        for w_b, kind_b in group["ends"][1:]:
            seg_a = seg_offsets[w_a + 1] - 1 if kind_a == "end" else seg_offsets[w_a]
            seg_b = seg_offsets[w_b + 1] - 1 if kind_b == "end" else seg_offsets[w_b]
            rise_a = kind_a == "end"
            rise_b = kind_b == "end"
            yield (
                seg_a,
                rise_a,
                1.0 if rise_a else -1.0,
                seg_b,
                rise_b,
                -1.0 if rise_b else 1.0,
            )

    def _build_geometry(self):
        """Discretize every wire into segments and concatenate.

        Segments are stored by start point, unit tangent and length, which
        is the form both the static moments and the testing paths want.

        Every basis — interior tent or junction tent — is described by two
        WINGS, one per side of its knot, in flow order (side A first):

            wing_seg[n, j]    the segment the wing lives on
            wing_rise[n, j]   the knot is at that segment's arc-h end, so
                              the tent shape there is τ/h (else 1 − τ/h)
            wing_sigma[n, j]  ±1: the wing's current direction relative to
                              its segment's own arc direction

        An interior tent at knot n is ``([left, right], [True, False],
        [+1, +1])`` — the unit-1 convention, unchanged. Junction tents are
        appended after all interior bases, junction by junction, and get
        their wings from :meth:`_junction_wings`. A GROUNDED end's tent is
        one of those, with `wing_sigma[n, 0] == 0`: its side-A wing is the
        image's, which the fold supplies, so nothing real lives on it.
        `grounded_bases` lists those tents' indices, which is what the fill
        reads to give their ROW the plane as its potential reference.
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
        n_interior = int(basis_offsets[-1])
        if n_interior == 0:
            raise ValueError("no unknowns: every wire needs >= 2 segments")

        groups = self._find_junctions()
        j_seg, j_rise, j_sigma = [], [], []
        junctions, grounded_bases = [], []
        for group in groups:
            bases = []
            for sa, ra, ga, sb, rb, gb in self._junction_wings(seg_offsets, group):
                bases.append(n_interior + len(j_seg))
                j_seg.append((sa, sb))
                j_rise.append((ra, rb))
                j_sigma.append((ga, gb))
            # A grounded group's bases run parallel to its ends (one tent
            # per end); a free-air group's run parallel to ends[1:].
            junctions.append(
                {"ends": group["ends"], "bases": bases, "grounded": group["grounded"]}
            )
            if group["grounded"]:
                grounded_bases.extend(bases)

        n_junction = len(j_seg)
        n_basis = n_interior + n_junction
        wing_seg = np.empty((n_basis, 2), dtype=np.int64)
        wing_rise = np.empty((n_basis, 2), dtype=bool)
        wing_sigma = np.empty((n_basis, 2), dtype=np.float64)
        wing_seg[:n_interior, 0] = left
        wing_seg[:n_interior, 1] = left + 1
        wing_rise[:n_interior, 0] = True
        wing_rise[:n_interior, 1] = False
        wing_sigma[:n_interior] = 1.0
        if n_junction:
            wing_seg[n_interior:] = np.asarray(j_seg, dtype=np.int64)
            wing_rise[n_interior:] = np.asarray(j_rise, dtype=bool)
            wing_sigma[n_interior:] = np.asarray(j_sigma, dtype=np.float64)

        geom = {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "basis_offsets": basis_offsets,
            "n_segs_total": seg_offsets[-1],
            "n_basis_interior": n_interior,
            "n_basis_total": n_basis,
            "seg_p0": np.vstack(p0_list),
            "seg_t": np.vstack(tan_list),
            "seg_h": np.concatenate(h_list),
            "wing_seg": wing_seg,
            "wing_rise": wing_rise,
            "wing_sigma": wing_sigma,
            "junctions": junctions,
            "grounded_bases": np.asarray(grounded_bases, dtype=np.int64),
        }
        self._cached_geometry = geom
        return geom

    def _feed_knots(self, geom, w):
        """Every knot of wire `w` that carries a basis, as (arc, basis, K).

        The interior knots, plus either wire end that meets other ends at a
        junction — a junction knot's basis is that junction's through-
        current unknown, and driving it is the split-wire feed — plus
        either wire end that TOUCHES the plane, whose basis is that end's
        grounded tent: the source then sits in the gap between the plane
        and that wire end, which is how a monopole is driven.

        `K` is the number of ends at the knot (1 for an interior knot),
        which is what the K >= 3 refusal reads. A grounded end reports 1
        whatever else meets it there, and that is not a fudge: the gap a
        source at a grounded end occupies is between the plane and THIS
        wire's end, so there is no branch pair left to name — each end at a
        grounded point has a tent of its own.
        """
        arc_at_knot = geom["per_wire"][w]["arc_at_knot"]
        base = geom["basis_offsets"][w]
        knots = [
            (float(arc_at_knot[j]), base + j - 1, 1)
            for j in range(1, len(arc_at_knot) - 1)
        ]
        for jn in geom["junctions"]:
            for e_i, (w_e, kind) in enumerate(jn["ends"]):
                if w_e != w:
                    continue
                arc = 0.0 if kind == "start" else float(arc_at_knot[-1])
                if jn["grounded"]:
                    knots.append((arc, jn["bases"][e_i], 1))
                else:
                    knots.append((arc, jn["bases"][0], len(jn["ends"])))
        return knots

    def _feed_basis_indices(self, geom):
        """Global basis index of each feed's knot.

        Each feed snaps to the knot of its wire — interior, junction or
        grounded end — whose arc length from that wire's first anchor is
        closest to the requested value (None → the wire's midpoint).
        """
        idx = []
        for i, (w, arc, _v) in enumerate(self.feeds):
            arc_at_knot = geom["per_wire"][w]["arc_at_knot"]
            target = arc if arc is not None else arc_at_knot[-1] / 2.0
            knots = self._feed_knots(geom, w)
            arcs = np.array([a for a, _b, _k in knots])
            pick = int(np.argmin(np.abs(arcs - target)))
            _a, basis, k_ends = knots[pick]
            if k_ends >= 3:
                raise NotImplementedError(
                    f"feeds[{i}]: the source snaps to a junction where "
                    f"{k_ends} wire ends meet, and a delta-gap voltage there "
                    "is ambiguous — it would have to name which pair of "
                    "branches it drives. Feed an interior knot, or model the "
                    "source on a short bridge wire off the junction."
                )
            idx.append(int(basis))
        return idx

    # ------------------------------------------------------------------
    # kernel moments

    def _seg_moments_prepare(self, obs, geom):
        """K-independent ingredients of every segment's reduced-kernel moments.

        The closed-form static moments ``(m0s, m1s, see
        :func:`_static_axis_moments`)`` and the source-to-observer distance
        ``R`` itself depend only on geometry — R = sqrt(u² + ρ²) with u, ρ²
        from the axis frame (:func:`_axis_frame`) and the source quadrature
        node's local arc τ, none of which a wavenumber sweep changes. Only
        exp(−jkR) is k-dependent, so R is exactly the boundary: caching it
        (rather than recomputing it from the axis frame every k) moves its
        sqrt out of the per-k path along with the asinh/sqrt of the static
        moments. Chunked exactly as :meth:`_seg_moments` used to chunk
        internally, so a k-sweep can replay the same chunk boundaries
        without recomputing them.

        Returns a list of ``(lo, hi, R, m0s, m1s)`` chunks.
        """
        a = self.wire_radius
        seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
        n_seg = seg_h.size
        n_obs = obs.shape[0]
        xg, _wg = leggauss(self.n_qp_source)
        tau = 0.5 * seg_h[:, None] * (1.0 + xg[None, :])
        step = max(1, _CHUNK_ELEMS // max(1, n_seg * self.n_qp_source))
        chunks = []
        for lo in range(0, n_obs, step):
            self._checkpoint()
            hi = min(lo + step, n_obs)
            u_r, rho2 = _axis_frame(obs[lo:hi], seg_p0, seg_t, a)
            m0s, m1s = _static_axis_moments(u_r, rho2, seg_h)
            u = tau[None, :, :] - u_r[:, :, None]
            R = np.sqrt(u * u + rho2[:, :, None])
            chunks.append((lo, hi, R, m0s, m1s))
        return chunks

    def _seg_moments_from_prepared(self, chunks, geom, k, n_obs, *, need_m1=True):
        """Finish :meth:`_seg_moments_prepare`'s chunks at one wavenumber.

        Only the smooth remainder (exp(−jkR)−1)/(4πR) is computed here —
        everything k-independent (R and the static moments) already sits in
        `chunks`. Returns the same ``(M0, M1)`` shape ``(n_obs, n_seg)``
        that :meth:`_seg_moments` did.
        """
        seg_h = geom["seg_h"]
        n_seg = seg_h.size
        xg, wg = leggauss(self.n_qp_source)
        # Source quadrature in each segment's own local arc coordinate.
        tau = 0.5 * seg_h[:, None] * (1.0 + xg[None, :])
        wq = 0.5 * seg_h[:, None] * wg[None, :]

        M0 = np.empty((n_obs, n_seg), dtype=np.complex128)
        M1 = np.empty((n_obs, n_seg), dtype=np.complex128) if need_m1 else None
        inv4pi = 1.0 / (4.0 * np.pi)
        for lo, hi, R, m0s, m1s in chunks:
            self._checkpoint()
            rem = (np.exp(-1j * k * R) - 1.0) / R
            M0[lo:hi] = (m0s + np.einsum("psq,sq->ps", rem, wq)) * inv4pi
            if need_m1:
                M1[lo:hi] = (m1s + np.einsum("psq,sq->ps", rem, tau * wq)) * inv4pi
        return M0, M1

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

        A thin composition of :meth:`_seg_moments_prepare` and
        :meth:`_seg_moments_from_prepared` for a single wavenumber; the
        swept assembly (`compute_impedance_swept`, `compute_y_matrix_swept`)
        calls those two halves directly so the prepare step runs once per
        sweep instead of once per k.
        """
        chunks = self._seg_moments_prepare(obs, geom)
        return self._seg_moments_from_prepared(
            chunks, geom, k, obs.shape[0], need_m1=need_m1
        )

    # ------------------------------------------------------------------
    # assembly

    def _knot_points(self, geom):
        """The knot each basis is centred on, read off its side-A wing."""
        seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
        s_a = geom["wing_seg"][:, 0]
        along = np.where(geom["wing_rise"][:, 0], seg_h[s_a], 0.0)
        return seg_p0[s_a] + along[:, None] * seg_t[s_a]

    def _testing_paths(self, geom):
        """Quadrature points, tangents and weights of every testing path.

        Path P_m runs centroid(wing A) → knot m → centroid(wing B), each
        half traversed in the direction the basis current flows there. For
        an interior tent that is the unit-1 path centroid(left) → knot →
        centroid(right); for a junction tent it is A's terminal segment,
        through the junction, into B's — which may run against either
        wire's own arc direction, hence the ±1 wing signs.

        Each half gets its own `n_qp_path` Gauss-Legendre rule and its own
        tangent: on a kinked wire (or across a junction) the two halves
        point in different directions, which is exactly what T1's tangent
        dot product has to see.

        Returns ``(pts, tans, wts)`` shaped (n_basis, 2·n_qp_path, 3/3/–).
        """
        seg_t, seg_h = geom["seg_t"], geom["seg_h"]
        wing_seg, wing_sigma = geom["wing_seg"], geom["wing_sigma"]
        cent = geom["seg_p0"] + 0.5 * seg_h[:, None] * seg_t
        knot = self._knot_points(geom)

        if self.nec5_quadrature:
            # NEC-5's identified rule (momwire#316): one node per wing, AT
            # the wing's centroid, weighted by that half-path's length —
            # the two-point centroid trapezoid on a uniform mesh. The
            # downstream fill consumes (pts, tans, wts) shape-agnostically.
            s_a, s_b = wing_seg[:, 0], wing_seg[:, 1]
            pts = np.stack([cent[s_a], cent[s_b]], axis=1)
            tans = np.stack(
                [
                    wing_sigma[:, 0][:, None] * seg_t[s_a],
                    wing_sigma[:, 1][:, None] * seg_t[s_b],
                ],
                axis=1,
            )
            wts = 0.5 * np.stack([seg_h[s_a], seg_h[s_b]], axis=1)
            return pts, tans, wts

        xo, wo = leggauss(self.n_qp_path)

        s_a, s_b = wing_seg[:, 0], wing_seg[:, 1]
        pts, tans, wts = [], [], []
        for j, (lo_pt, hi_pt, seg) in enumerate(
            ((cent[s_a], knot, s_a), (knot, cent[s_b], s_b))
        ):
            mid = 0.5 * (lo_pt + hi_pt)
            half = 0.5 * (hi_pt - lo_pt)
            pts.append(mid[:, None, :] + half[:, None, :] * xo[None, :, None])
            # The traversal direction IS the wing's flow direction: σ ±1
            # times the segment's own unit tangent.
            t_dir = wing_sigma[:, j][:, None] * seg_t[seg]
            tans.append(np.repeat(t_dir[:, None, :], xo.size, axis=1))
            # Each half-path is h/2 long, so the Gauss-Legendre Jacobian
            # that maps [-1, 1] onto it is h/4, not h/2.
            wts.append(0.25 * seg_h[seg][:, None] * wo[None, :])
        return (
            np.concatenate(pts, axis=1),
            np.concatenate(tans, axis=1),
            np.concatenate(wts, axis=1),
        )

    def _image_sources(self, geom):
        """The mirrored source geometry, or None in free space.

        The PEC image is a SOURCE-SIDE substitution and nothing else: the
        moment builder is handed the same three arrays it always gets, with
        every segment origin reflected through `z = ground_z` and every
        segment tangent z-flipped. Segment LENGTHS are untouched, and that
        is what makes the substitution total — a mirrored segment's local
        arc coordinate τ maps point-for-point onto the real one's
        (M(p₀ + τ t) = M p₀ + τ M t), so every wing shape, every charge
        doublet ±1/h and every quadrature node index means the same thing on
        both source sets. Observers stay real throughout; only the sources
        move.

        Both operations come from the ground object's `image_geometry()`
        (momwire#398 unit 1, reduced to mirror operations in unit 2), so
        this solver spells no reflection of its own — which is the pilot's
        whole claim: the capability arrives through the shared layer.
        """
        ground = _potential_ground.potential_ground_for(self, geom, self.k, self.omega)
        if ground is None:
            return None
        # PEC only (the constructor refuses every finite ground), so the
        # object is a pure mirror map here: `mode` is "fold", there is no
        # remainder and no weight table, and nothing it hands back depends
        # on k or ω. That is what lets a swept solve build it once in the
        # prepare half and replay it at every wavenumber; a future
        # refl-coef unit CANNOT do that, because its weights are ω-dependent
        # and would have to be rebuilt in `_assemble_Z_from_prepared`.
        img = ground.image_geometry()
        return {
            "seg_p0": img.mirror_positions(geom["seg_p0"]),
            "seg_t": img.mirror_tangents(geom["seg_t"]),
            "seg_h": geom["seg_h"],
            "mirror_tangents": img.mirror_tangents,
        }

    def _assemble_Z_prepare(self, geom):
        """K-independent work for the razor-blade fill: stencils and moments.

        Everything `_assemble_Z_from_prepared` needs that does not depend on
        the wavenumber — the wing/path stencils built once in unit 1/2, and
        the static (:meth:`_seg_moments_prepare`) halves of the segment
        moments at both observation sets the fill uses (segment centroids
        for T2, testing-path quadrature points for T1). A wavenumber sweep
        calls this once and replays it through `_assemble_Z_from_prepared`
        for every k, instead of rebuilding it per k the way a plain loop
        over single solves would.

        Over a PEC ground the same work is done a SECOND time against the
        mirrored sources (:meth:`_image_sources`) and cached beside the
        first, under `prepared["image"]`. **Doubling the cache rather than
        rebuilding the image half per k is a deliberate schedule decision**
        (momwire#398 unit 2), because the mirrored static moments are
        exactly as k-independent as the real ones: R, `asinh` and `sqrt`
        over mirrored sources do not know what wavenumber will be asked
        for. Caching them keeps the prepare/replay contract whole — a swept
        point pays only the smooth remainder, on both source sets — and it
        keeps the two halves of the fold symmetric, which is what lets
        `_assemble_Z_source_block` serve both with one body.

        Priced on the ByDipole1 N=96 deck at h = 5 m over PEC, so the
        trade is on the record rather than asserted:

        | lane               | image cache | rebuild-per-k costs |
        |--------------------|-------------|---------------------|
        | `nec5_quadrature`  | 3.1 MB      | +15.8 % per swept k |
        | default GL path    | 66.4 MB     | +15.6 % per swept k |

        The cache is *exactly* the size of the free-space one it sits
        beside — a grounded razor solve is a 2x-residency solve, in the same
        proportion the grounded fill already doubles the arithmetic — and
        the alternative buys that memory back at about a sixth of every
        swept point. On the 16 GB working assumption the doubling is
        affordable at any mesh this formulation is used at (a 4,000-segment
        GL fill is the first place it would matter, and razor's O(N²) dense
        LU bites first). Should that stop being true, the switch is local:
        drop `prepared["image"]`'s chunk lists and rebuild them inside
        `_assemble_Z_source_block`. `scripts/capture_razor_pec_nec5_lane.py`
        does not measure this — the numbers above come from the unit's
        report, and re-measuring is a ten-line script against
        `_seg_moments_prepare`.
        """
        seg_t, seg_h = geom["seg_t"], geom["seg_h"]
        wing_seg, wing_sigma, wing_rise = (
            geom["wing_seg"],
            geom["wing_sigma"],
            geom["wing_rise"],
        )
        n_basis = wing_seg.shape[0]
        s_a, s_b = wing_seg[:, 0], wing_seg[:, 1]
        h_a, h_b = seg_h[s_a], seg_h[s_b]
        # dΛ/dτ in the segment's own arc coordinate, signed by the flow.
        q_a = wing_sigma[:, 0] * np.where(wing_rise[:, 0], 1.0, -1.0) / h_a
        q_b = wing_sigma[:, 1] * np.where(wing_rise[:, 1], 1.0, -1.0) / h_b

        # --- scalar potential's observation set: segment centroids.
        cent = geom["seg_p0"] + 0.5 * seg_h[:, None] * seg_t
        t2_chunks = self._seg_moments_prepare(cent, geom)

        # --- vector potential's observation set: the outer path, row-chunked.
        pts, tans, wts = self._testing_paths(geom)
        n_path = pts.shape[1]
        # σ folded into the source-side tangent, so the dot product carries
        # the wing's current direction with it.
        tan_a = wing_sigma[:, 0][:, None] * seg_t[s_a]
        tan_b = wing_sigma[:, 1][:, None] * seg_t[s_b]
        td_a = tan_a.T  # (3, n_basis)
        td_b = tan_b.T
        # Columns whose wing falls (knot at the segment's arc-0 end) need
        # M0 − M1/h instead of M1/h. On a junction-free model that is every
        # B wing and no A wing, so patching the exceptions in place keeps
        # the no-junction fill exactly as cheap as it was in unit 1.
        fall_a = np.flatnonzero(~wing_rise[:, 0])
        fall_b = np.flatnonzero(~wing_rise[:, 1])
        # The image, if there is one: the SAME observers and the SAME row
        # windows, against mirrored sources. Built inside the one loop so
        # free space allocates nothing extra and keeps no reference to
        # `pts` past this method — the ground layer is structurally absent
        # when off, not skipped (architecture doc §6, gate (b)).
        img_src = self._image_sources(geom)
        rows = max(1, _CHUNK_ELEMS // max(1, n_path * n_basis))
        t1_row_chunks = []
        t1_row_chunks_img = [] if img_src is not None else None
        for lo in range(0, n_basis, rows):
            hi = min(lo + rows, n_basis)
            obs = pts[lo:hi].reshape(-1, 3)
            t1_row_chunks.append(
                (lo, hi, obs.shape[0], self._seg_moments_prepare(obs, geom))
            )
            if img_src is not None:
                t1_row_chunks_img.append(
                    (lo, hi, obs.shape[0], self._seg_moments_prepare(obs, img_src))
                )

        prepared = {
            "n_basis": n_basis,
            "n_seg": seg_h.size,
            "s_a": s_a,
            "s_b": s_b,
            "h_a": h_a,
            "h_b": h_b,
            "q_a": q_a,
            "q_b": q_b,
            "n_cent": cent.shape[0],
            "t2_chunks": t2_chunks,
            "n_path": n_path,
            "tans": tans,
            "wts": wts,
            "td_a": td_a,
            "td_b": td_b,
            "fall_a": fall_a,
            "fall_b": fall_b,
            "t1_row_chunks": t1_row_chunks,
            "grounded": geom["grounded_bases"],
            "image": None,
        }

        if img_src is not None:
            mirror = img_src["mirror_tangents"]
            prepared["image"] = {
                "t2_chunks": self._seg_moments_prepare(cent, img_src),
                # The image sign, in razor's own (3, n_basis) idiom: M·t on
                # the SOURCE tangent table only. Nothing N² is formed, and
                # nothing on the observer side moves.
                "td_a": mirror(tan_a).T,
                "td_b": mirror(tan_b).T,
                "t1_row_chunks": t1_row_chunks_img,
            }
        return prepared

    def _assemble_Z_from_prepared(self, geom, prepared, k, omega):
        """Fill the razor-blade impedance matrix at one wavenumber.

        Free space is one source block. Over a PEC ground it is that block
        minus the image's:

            Z = Z_free − Z_image

        one global minus at the very end, exactly as the mixed-potential
        trunk's other solvers spell their fold (`PotentialGround.mode ==
        "fold"`, architecture doc §2.2). Both halves of the minus are real
        physics and they arrive together: the horizontal image current runs
        anti-parallel and the image charge is opposite, so *both* the
        vector-potential term and the charge term change sign — which is
        why the sign can be taken once on the assembled block instead of
        term by term. What the mirrored tangent table supplies is the rest:
        M·t restores the +z component the global minus would otherwise have
        flipped the wrong way, so a vertical current's image stays parallel.

        Only `k` and `omega` (=c·k, passed separately so a swept caller can
        reuse one `omega_array` without recomputing it) vary here; every
        other ingredient comes from `prepared` (`_assemble_Z_prepare`).
        """
        Z = self._assemble_Z_source_block(geom, prepared, prepared, k, omega)
        image = prepared["image"]
        if image is not None:
            Z -= self._assemble_Z_source_block(geom, prepared, image, k, omega)
        return Z

    def _assemble_Z_source_block(self, geom, prepared, sources, k, omega):
        """One source set's contribution to the razor-blade matrix.

        `sources` supplies the three source-side pieces — the two moment
        chunk lists and the `(3, n_basis)` tangent tables — and is either
        `prepared` itself (the real sources) or `prepared["image"]` (the
        mirrored ones). Everything else is shared: the observers, the
        testing paths and their weights, the wing stencils, the charge
        doublets. That split IS the PEC image: same rows, same test
        functions, moved sources.

        Both terms are built from the same two segment moments. A wing on
        segment s of length h contributes to the vector potential

            A_n += σ · (M1[s] / h)              when the knot is at arc h
            A_n += σ · (M0[s] − M1[s] / h)      when it is at arc 0

        carried by σ times that segment's tangent, so the outer path
        point's tangent contracts with each wing separately. The charge
        doublet is the constant dσΛ/dτ on each wing — which works out to
        +1/h_A on side A and −1/h_B on side B whichever way round the two
        wires are spelled, i.e. the unit charge that leaves A and lands on
        B — differenced between the path's two bounding centroids.

        A GROUNDED tent (momwire#398 unit 3) is that with side A carried by
        the image instead of by a wire: σ_A = 0 empties its half of both
        terms, and the fold's second pass supplies the mirrored wing with
        the opposite charge, so the through-basis's two doublet halves are
        −1/h on the real segment and +1/h on its image. The unit of current
        that flows into the plane leaves no net charge at the contact point
        — the image takes exactly the charge the real wing deposits, which
        is the same statement as Φ = 0 on the conductor.
        """
        s_a, s_b = prepared["s_a"], prepared["s_b"]
        h_a, h_b = prepared["h_a"], prepared["h_b"]
        q_a, q_b = prepared["q_a"], prepared["q_b"]
        n_basis = prepared["n_basis"]

        M0c, _ = self._seg_moments_from_prepared(
            sources["t2_chunks"], geom, k, prepared["n_cent"], need_m1=False
        )
        dM0 = M0c[s_b] - M0c[s_a]  # (row, source segment)
        grounded = prepared["grounded"]
        if grounded.size:
            # A grounded row's testing path starts AT the plane, where the
            # folded scalar potential is identically zero: a point in the
            # plane is equidistant from every source and its image, so the
            # two blocks' contributions there are the same number and the
            # fold's minus cancels them. Dropping the term in each block is
            # therefore exact rather than approximate — and it is what makes
            # the plane this formulation's potential reference, the discrete
            # form of Φ = 0 on a perfect conductor.
            dM0[grounded] = M0c[s_b[grounded]]
        T2 = dM0[:, s_a] * q_a[None, :] + dM0[:, s_b] * q_b[None, :]

        tans, wts = prepared["tans"], prepared["wts"]
        n_path = prepared["n_path"]
        td_a, td_b = sources["td_a"], sources["td_b"]
        fall_a, fall_b = prepared["fall_a"], prepared["fall_b"]
        T1 = np.empty((n_basis, n_basis), dtype=np.complex128)
        for lo, hi, n_obs_chunk, static in sources["t1_row_chunks"]:
            self._checkpoint()
            M0, M1 = self._seg_moments_from_prepared(static, geom, k, n_obs_chunk)
            mom_a = M1[:, s_a] / h_a[None, :]
            mom_b = M1[:, s_b] / h_b[None, :]
            if fall_a.size:
                mom_a[:, fall_a] = M0[:, s_a[fall_a]] - mom_a[:, fall_a]
            if fall_b.size:
                mom_b[:, fall_b] = M0[:, s_b[fall_b]] - mom_b[:, fall_b]
            t_out = tans[lo:hi].reshape(-1, 3)
            integrand = (t_out @ td_a) * mom_a + (t_out @ td_b) * mom_b
            integrand *= wts[lo:hi].reshape(-1)[:, None]
            T1[lo:hi] = integrand.reshape(hi - lo, n_path, n_basis).sum(axis=1)

        return 1j * omega * self.mu * T1 - T2 / (1j * omega * self.eps)

    def _assemble_Z(self, geom, k):
        """Fill the razor-blade impedance matrix at one wavenumber.

        A thin composition of `_assemble_Z_prepare` and
        `_assemble_Z_from_prepared` for a single solve; `compute_impedance`
        and `compute_y_matrix` call this one k at a time, while
        `compute_impedance_swept` / `compute_y_matrix_swept` call the two
        halves directly so the prepare step runs once per sweep.
        """
        prepared = self._assemble_Z_prepare(geom)
        return self._assemble_Z_from_prepared(geom, prepared, k, self.c * k)

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
        distribution in basis order: per wire, interior knots in arc order,
        then the junction through-currents (junctions in detection order,
        K−1 per junction), each measured from its junction's side A into
        that tent's side B. A GROUNDED point contributes K of them instead,
        one per wire end there, each the current flowing out of the plane
        into that end.
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

    def compute_y_matrix(self):
        """Short-circuit admittance matrix [Y_sc] at the configured feeds.

        Ports are the `feeds` entries; their voltages are ignored here.
        ``Y[i, j]`` is the current at port i when port j is driven with
        1 V and every other port is held at 0 V — which on this delta-gap
        model is simply an all-zero row entry, so the N columns are N
        back-substitutions on one factored Z. Returns an
        ``(n_ports, n_ports)`` complex array; invert it for the
        open-circuit Z matrix of network analysis.

        The result is NOT symmetric, and that is the formulation talking:
        razor-blade testing weights the E-field boundary condition with
        path integrals rather than with the basis itself, so reciprocity
        is only recovered as the mesh refines (momwire#309). A Galerkin
        scheme on the same basis would be symmetric to machine precision.
        `tests/test_razor_junctions.py` pins the decay.
        """
        geom = self._build_geometry()
        self._checkpoint()
        Z = self._assemble_Z(geom, self.k)
        self.z = Z

        idx = self._feed_basis_indices(geom)
        B = np.zeros((geom["n_basis_total"], len(idx)), dtype=np.complex128)
        for j, m_j in enumerate(idx):
            B[m_j, j] = 1.0

        self._checkpoint()
        return scipy.linalg.solve(Z, B)[idx, :]

    # ------------------------------------------------------------------
    # field readout

    def currents_at_knots(self, coeffs, s_array=None):
        """Per-wire complex current at every mesh knot (momwire#309 unit 3).

        Returns a list of 1-D arrays, one per wire in `wires_polylines`
        order, each of length ``n_segments_of_wire + 1`` — one value per
        KNOT, not per basis.

        Interior knot j of wire w reads straight off the tent coefficient:
        interior tent n peaks at value 1 on its own knot and is zero at
        every other knot, so the current there is
        ``coeffs[basis_offsets[w] + j - 1]``.

        A FREE wire end (no junction there, and not in the ground plane) is
        0 — the open-circuit BC the tent basis already builds in.

        A GROUNDED wire end reads its grounded tent the same way a
        junctioned end reads its junction tents: the tent's real wing sits
        on that end with `wing_sigma` ±1, and the ghost wing carrying the
        image contributes 0 by construction (`wing_sigma` 0), so the sum
        below is that one term — the current crossing the plane into the
        wire, in the wire's own arc direction.

        A JUNCTIONED wire end is not itself a basis's home knot the way an
        interior knot is: this formulation's junction tents (unlike the
        retired triangular-family solver's directional end-bases) are
        through-current unknowns shared between two DIFFERENT wires, each
        carrying its own wing sign onto that shared knot. The current at a
        junctioned end knot, expressed in THAT WIRE'S OWN ARC DIRECTION, is
        therefore the signed sum of every junction tent with a wing sitting
        on that (wire, end): each such wing contributes
        ``wing_sigma · coeff`` — `wing_sigma` already carries the sign that
        turns the tent's through-current into a multiple of this wire's own
        arc-length direction (see the module docstring's "Junctions"
        section and `_junction_wings`). At a K=2 junction (an ordinary
        two-wire join, or an interior knot re-spelled as a split) exactly
        one wing sits on each side and the sum is a single term; at a K>=3
        junction, side A carries a wing from every one of the K−1 tents
        there, so its knot sums all of them (Kirchhoff's law falling out of
        the wing bookkeeping — `tests/test_razor_junctions.py`'s
        `_currents_into_junction` helper is the same sum, used there to
        check KCL rather than to read off a knot current).

        With `s_array` given as a list of 1-D per-wire arc-length arrays,
        the tent's own piecewise-linear shape means this is exactly a
        linear interpolation of the knot values along the wire's cumulative
        arc (`per_wire[w]["arc_at_knot"]`), not a re-solve of anything.
        """
        coeffs = np.asarray(coeffs)
        geom = self._build_geometry()
        per_wire = geom["per_wire"]
        seg_offsets = geom["seg_offsets"]
        basis_offsets = geom["basis_offsets"]
        n_interior = geom["n_basis_interior"]
        wing_seg, wing_rise, wing_sigma = (
            geom["wing_seg"],
            geom["wing_rise"],
            geom["wing_sigma"],
        )

        out = []
        for w_idx, pw in enumerate(per_wire):
            n_knots = pw["arc_at_knot"].shape[0]
            I = np.zeros(n_knots, dtype=np.complex128)
            I[1:-1] = coeffs[basis_offsets[w_idx] : basis_offsets[w_idx + 1]]
            out.append(I)

        n_basis_total = wing_seg.shape[0]
        if n_basis_total > n_interior:
            # Flatten the junction tents' two wings into one (seg, rise,
            # sigma, coeff) list per side, so a wire's end knot is just
            # "every wing whose (segment, rise) is that end's".
            j_seg = wing_seg[n_interior:].reshape(-1)
            j_rise = wing_rise[n_interior:].reshape(-1)
            j_sigma = wing_sigma[n_interior:].reshape(-1)
            j_coeff = np.repeat(coeffs[n_interior:], 2)
            for w_idx in range(len(per_wire)):
                start_seg = seg_offsets[w_idx]
                end_seg = seg_offsets[w_idx + 1] - 1
                at_start = (j_seg == start_seg) & ~j_rise
                if at_start.any():
                    out[w_idx][0] = (j_sigma[at_start] * j_coeff[at_start]).sum()
                at_end = (j_seg == end_seg) & j_rise
                if at_end.any():
                    out[w_idx][-1] = (j_sigma[at_end] * j_coeff[at_end]).sum()

        if s_array is None:
            return out

        sampled = []
        for w_idx, sv in enumerate(s_array):
            arc = per_wire[w_idx]["arc_at_knot"]
            knot_I = out[w_idx]
            sv = np.asarray(sv, dtype=np.float64)
            Ire = np.interp(sv, arc, knot_I.real)
            Iim = np.interp(sv, arc, knot_I.imag)
            sampled.append(Ire + 1j * Iim)
        return sampled

    # ------------------------------------------------------------------
    # swept solve

    def compute_impedance_swept(self, k_array):
        """Drive-point impedance(s) over a batch of wavenumbers.

        Same return convention as `compute_impedance` (scalar per k with
        one feed, `(n_k, n_feeds)` with several), stacked over `k_array`
        into shape `(n_k,)` / `(n_k, n_feeds)`. Shares every k-independent
        piece of the fill — geometry, the wing/path stencils, and the
        closed-form static segment moments — across the sweep via
        `_assemble_Z_prepare`, so only the smooth kernel remainder and the
        jωμ / 1/jωε prefactors (ω = c·k) are recomputed per k; each k still
        gets its own dense solve.
        """
        k_array = np.asarray(k_array, dtype=np.float64)
        geom = self._build_geometry()
        self._checkpoint()
        prepared = self._assemble_Z_prepare(geom)

        idx = self._feed_basis_indices(geom)
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        rhs = np.zeros(geom["n_basis_total"], dtype=np.complex128)
        for m_i, v_i in zip(idx, voltages):
            rhs[m_i] += v_i

        feed_currents = np.empty((k_array.shape[0], len(idx)), dtype=np.complex128)
        for i, k in enumerate(k_array):
            self._checkpoint()
            k = float(k)
            Z = self._assemble_Z_from_prepared(geom, prepared, k, self.c * k)
            coeffs = scipy.linalg.solve(Z, rhs)
            feed_currents[i] = coeffs[idx]

        z_per_feed = voltages[None, :] / feed_currents
        return z_per_feed[:, 0] if len(self.feeds) == 1 else z_per_feed

    def compute_y_matrix_swept(self, k_array):
        """Short-circuit admittance matrices over a batch of wavenumbers.

        Returns an `(n_k, n_ports, n_ports)` complex array; `Y[i]` is
        `compute_y_matrix()`'s result at `k_array[i]`, same port/sign
        conventions (including the razor-blade non-reciprocity —
        `tests/test_razor_junctions.py`). Shares the k-independent fill work
        the same way `compute_impedance_swept` does.
        """
        k_array = np.asarray(k_array, dtype=np.float64)
        geom = self._build_geometry()
        self._checkpoint()
        prepared = self._assemble_Z_prepare(geom)

        idx = self._feed_basis_indices(geom)
        n_ports = len(idx)
        B = np.zeros((geom["n_basis_total"], n_ports), dtype=np.complex128)
        for j, m_j in enumerate(idx):
            B[m_j, j] = 1.0

        Y = np.empty((k_array.shape[0], n_ports, n_ports), dtype=np.complex128)
        for i, k in enumerate(k_array):
            self._checkpoint()
            k = float(k)
            Z = self._assemble_Z_from_prepared(geom, prepared, k, self.c * k)
            Y[i] = scipy.linalg.solve(Z, B)[idx, :]
        return Y
