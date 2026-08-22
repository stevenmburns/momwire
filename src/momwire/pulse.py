"""Pulse basis with point matching — the throwaway-tier architecture probe.

The oldest thin-wire MoM scheme there is (Harrington's pulse expansion,
point-matched), written here as momwire's seventh solver row for one
reason: `docs/design/solver-architecture.md` §0.1's metric says a
throwaway formulation must reach green gates in HOURS, and momwire#416
proposes measuring that rather than asserting it. What this module is FOR
is `docs/design/pulse-probe.md` — the line counts, the
consumed-unchanged inventory, and the two shared-layer signatures that
turned out to be shaped for their first consumer. The physics below is
the classic scheme, deliberately: nothing here is new, so everything the
build cost is architecture.

Accuracy is honest and low. Pulse + point matching converges more slowly
than every other row momwire ships — the expansion has no continuity
between segments at all and the testing is a single point per row — so
the gates are exactness of the PEC image twin and MONOTONE approach to
`BSplineSolver`, never beating it.

How low, and whose fault: this row's error is governed by Δ/a and not
Δ/λ (the test below measures it), which is NOT a property of the pulse
basis or of point matching. It is this module's choice to leave the
charge as two POINT charges, where Harrington's own 1967 scheme spreads
it over a half-shifted cell. `HarringtonSolver` is that scheme — same
basis, same testing, same kernel, same feed, one ingredient different —
and it converges O(1/N) with no Δ/a dependence at all, at ~1/64 the mesh
on a thin wire (momwire#557). The two rows are kept side by side
deliberately: the pair is the instrument that attributes the difference
to the charge's SUPPORT, and this row is the one that costs no junction
detection. Reach for `HarringtonSolver` — or the `pulse` roster name,
which is it — for anything but that study.

Formulation
-----------
One pulse per segment: I_n constant on segment n, carried along that
segment's own unit tangent t_n, zero elsewhere. ``n_basis == n_segs``.
There is no continuity constraint anywhere — not between segments of one
wire, not at a junction, not at a free wire end. The CHARGE carries every
jump, which is what makes the scheme's bookkeeping trivial and its
convergence slow.

Continuity gives the charge. Along a wire the pulse's current steps 0 →
I_n → 0, so ``dI/dl = I_n[δ(l_L) − δ(l_R)]`` and the line charge
``q = −(1/jω) dI/dl`` is two POINT charges, one at each end of segment n:

    Q_R = +I_n / (jω)   at e_R (the segment's arc-h end)
    Q_L = −I_n / (jω)   at e_L (its arc-0 end)

— unit charge leaving by the head and arriving at the tail, the discrete
continuity equation written as a basis property rather than enforced by a
constraint row.

The potentials of basis n, with the reduced thin-wire kernel
g(r, r') = exp(−jkR)/(4πR), R = sqrt(|r − r'|² + a²), throughout:

    A_n(r) = μ I_n t_n ∫_{seg n} g(r, r') dl'   =  μ I_n t_n M0[r, n]
    φ_n(r) = (I_n / jωε) [ g(r, e_R^n) − g(r, e_L^n) ]

The reduced kernel is not a refinement here, it is what makes the scheme
exist: a POINT charge's potential on the wire's own axis would diverge,
and R = sqrt(d² + a²) is exactly what keeps ``g(e, e) = e^{−jka}/(4πa)``
finite. A pulse basis cannot be written with the exact kernel.

Testing is point matching on the segment, mixed-potential form. Row m
integrates ``t̂·E`` along segment m — one midpoint evaluation of the
vector term, and the scalar term EXACTLY, because ∫ t̂·∇φ dl telescopes:

    ∫_{seg m} t̂·E dl  =  −jω ∫ t̂·A dl − [φ(e_R^m) − φ(e_L^m)]
                       ≈  −jω h_m t̂_m·A(c_m) − [φ(e_R^m) − φ(e_L^m)]

with c_m the segment's centroid. Setting that against the delta-gap
voltage gives the row, one h_m already in it so the right-hand side is
the applied volts and nothing else:

    Z[m,n] = jωμ h_m (t_m·t_n) M0[c_m, n]  +  ΔΔg[m,n] / (jωε)

    ΔΔg[m,n] = g(e_R^m, e_R^n) − g(e_R^m, e_L^n)
             − g(e_L^m, e_R^n) + g(e_L^m, e_L^n)

The charge term is the four-point stencil of one basis's charge doublet
observed at the other's two endpoints — the whole scalar potential, with
no quadrature in it at all.

Symmetry, since the point-matching literature is vague about it: ΔΔg is
EXACTLY symmetric (g is symmetric and the m↔n swap transposes the
stencil), and the vector term is not — h_m M0[c_m, n] is the midpoint
rule on one side of a pairing that Galerkin testing would integrate on
both. So this scheme's whole reciprocity error lives in the A term, and
`tests/test_pulse.py::test_reciprocity_error_is_the_vector_term_alone`
measures it that way rather than asserting a bound.

Junctions
---------
Nothing to do, and that is a real property of the basis rather than an
omission. Wire ends that coincide put their endpoint charges at the SAME
POINT, so ``g(r, node)`` is one number and the contributions add: the
node's charge comes out as Σ ±I/(jω) over the ends there, signed by
whether each segment's arc direction runs into the node or out of it,
which is the net current flowing in — Kirchhoff's law, superposed by
arithmetic. There is no junction detection in this module and no
`junctions=` spec to write (it is refused as redundant).

What the basis does NOT give is current continuity AT the junction: the
node holds a point charge jωQ = Σ I, which is a genuine O(h) discretization
artefact and vanishes as the mesh refines rather than being enforced.
`tests/test_pulse.py::test_junction_current_balance_decays` measures the
rate instead of pretending the balance is exact. A free wire end is the
K = 1 case of the same statement: the terminal pulse's current does not
vanish, it dumps its charge on the tip, and it is driven small by that
tip's own capacitance rather than by the basis.

Ground
------
Every scrap of ground physics comes from
`_potential_ground.potential_ground_for`; this module contains no
reflection, no Fresnel coefficient and no mirror arithmetic of its own,
which is the pilot's claim and the thing momwire#416 exists to re-test
from a second consumer.

The fill is ``Z = Z_free − Z_image``, one global minus (`mode ==
"fold"`), with the image block assembled from the SAME rows, the same
centroids and the same endpoint stencil against mirrored sources. The
mirrored geometry is `ImageGeometry.mirror_positions` on the segment
endpoints — the only ground operation this file calls by name. The
per-pair weights are `PotentialGround.weight_windows()`, taken as ONE
full-width window ``(0, n_segs)``, which is the N² surface momwire#416
predicted a dense fill would want:

    w_A[m,n]  multiplies the vector term's tangent pairing
    w_Φ[m,n]  multiplies the charge term's four-point stencil

and PEC and refl-coef then share one code path exactly, because PEC's
window IS ``(t_m·M t_n, 1)``. The image sign rides in w_A plus the single
global minus, per architecture doc §2.2 — this module never applies
`image_coefficient` itself, which the object's docstring is explicit is
already inside the tables.

`weight_tables()` is NOT what this consumer uses — the fill is chunked,
so the windows are the right surface — but it is now CONSUMABLE here,
which it was not when this row was written. That was momwire#416's first
interface finding: the refl-coef branch called back into
`BSplineSolver._image_refl_prep` / `._image_refl_weights`, two methods a
non-B-spline solver does not have, so it raised `AttributeError`.
momwire#429 unit 1 moved that chain down into `_potential_ground`, and
`tests/test_pulse_ground.py` now pins the positive statement instead: the
tables are served on this solver and equal the full-width window. See
`docs/design/pulse-probe.md`.

Sommerfeld (momwire#430) is the ground's third row, `PotentialGround.mode
== "compose"`, and it needed no new branch for the exact-image half: C₂ =
(ε̃−1)/(ε̃+1) arrives through `weight_windows`' constant `(w_A, w_Φ)` pair
exactly as PEC's mirror table does, so `_source_block`'s two lines fill
the scaled exact image unchanged. What composing adds is one more term, Q
— the smooth Sommerfeld remainder field — reached through
`Remainder.field_windows(observers, sources, n_moment=1)`, momwire#398
unit 5's generalisation of the Galerkin-only `evaluate(supp_seg, polys)`
that blocked this row when momwire#416's probe first asked
(`docs/design/pulse-probe.md` §4.1). `n_moment = 1` is exactly this
basis's own shape: the zeroth arc moment of a source segment, i.e. weight
1 over the segment, is a pulse. Q is tested by this row's own rule — one
centroid per row, the SAME `h_m` scaling `_seg_M0`'s vector term already
carries — takes no jωμ / 1/jωε prefactor (F is already a field, theory
manual eq 123's unit current moment), integrates over the REAL source
segments even though it is summed into the mirrored block (the plane
enters through the interpolation grid, not through a mirrored source),
and is associated as `C₂·img + Q` inside `_source_block`, BEFORE
`_assemble_Z`'s single global minus — the same `"compose"` contract
`razor.py` states and pins, because `Z_free − (C₂·img + Q) ≠ (Z_free −
C₂·img) − Q` in float64.

Scope
-----
Free space, the PEC image, the reflection-coefficient ground and the
Sommerfeld ground; reduced kernel; one scalar radius; one polyline per
wire; delta-gap sources on segments. Ground CONTACT is refused (the basis
looks structurally ready for it — a contact node's real and image charges
coincide and cancel — but the bar is unwritten, and an unwritten bar is
not a capability). Everything else is refused through `capabilities` and
`_OUT_OF_SCOPE`.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from . import _ground_spec, _potential_ground, _wire_spec
from ._cancel import _Cancelable
from ._capabilities import Capabilities
from ._element_currents import _ElementCurrents
from ._quadrature import leggauss

# The reduced-kernel segment moment ∫ g dl' has no shared home in momwire —
# every solver that needs it has written its own. These two are razor's,
# imported rather than copied so the probe's line count measures what a new
# row actually costs. That the import reaches into a SIBLING SOLVER and not
# into a shared module is itself one of momwire#416's findings.
from ._kernel_moments import _axis_frame, _static_axis_moments

# Working-array budget for the chunked fills, in complex128 elements
# (~32 MB per temporary), razor's constant and razor's reason: the moment
# fill's inner tensor is (observers) x (segments) x (source quadrature
# points) and would otherwise be one huge allocation.
_CHUNK_ELEMS = 2_000_000

# Constructor kwargs the sibling solvers accept that this formulation
# deliberately does not, with the reason each is refused. Anything else
# unexpected is a caller typo and stays a TypeError.
_OUT_OF_SCOPE = {
    "degree": "PulseSolver has no degree: the pulse expansion IS the "
    "degree-0 basis, and point matching is defined against it. Use "
    "BSplineSolver(degree=...) for higher-order bases with Galerkin testing",
    "junctions": "PulseSolver takes no junction spec: coincident wire ends "
    "put their endpoint charges at the same point and superpose by "
    "arithmetic, so there is nothing to declare and nothing to detect",
    "junction_ports": "junction ports are not supported: the pulse basis has "
    "no junction unknown to drive — the current is per SEGMENT, so a source "
    "at a node has no basis of its own. Feed a segment, or model the source "
    "on a short bridge wire off the junction",
    "node_gaps": "node gaps are not supported: this formulation's delta gap "
    "is a whole segment's testing row, so a gap at a node is not a local "
    "basis edit here",
    "extended_kernel": "PulseSolver is reduced-kernel only, and not by "
    "preference: the charge is two POINT charges per basis, whose potential "
    "on the wire axis is finite only because of the reduced kernel's a². "
    "The exact kernel would need a different charge model, not a different "
    "quadrature",
}

# Reused by the wire_radius scalar-only check in __init__ (momwire#147) and by
# `capabilities.refusals` below — one message, not a copy in each.
_PER_WIRE_RADIUS_REFUSAL = (
    "wire_radius must be a scalar for PulseSolver; per-wire radii "
    "(momwire#147) are not supported by this formulation probe"
)


class PulseSolver(_ElementCurrents, _Cancelable):
    """Piecewise-constant (pulse) current basis, point-matched.

    See the module docstring for the formulation. One unknown per segment,
    free space or a PEC / reflection-coefficient / Sommerfeld ground,
    reduced kernel.

    wires: list of (M_w, 3) polyline arrays, M_w >= 2 anchor points per wire.
    n_per_edge_per_wire: list of (int or sequence). Per-wire segment counts
        per polyline edge; None anywhere means use `nsegs`.
    nsegs: default segment count when `n_per_edge_per_wire` doesn't specify.
    wire_radius: scalar thin-wire radius, the a in R = sqrt(|r−r'|² + a²).
    ground_z: height of a ground plane, or None for free space.
    ground_eps: complex relative permittivity of the ground; None (the
        default) with `ground_z` set is the PEC image.
    ground_phi_mode: image-charge weighting for the reflection-coefficient
        ground, passed through to `PotentialGround` untouched. Accepted and
        UNREAD under `ground_model="sommerfeld"`, exactly as `BSplineSolver`
        and `RazorSolver` treat it: the factory sets `phi_mode=None` on a
        composing ground, so there is nothing for the fill to read.
    ground_model: "refl-coef" (the default) or "sommerfeld" — the exact
        NEC `GN 2` style ground (momwire#398 unit 5, momwire#430), which
        requires `ground_eps` and is consumed through
        `Remainder.field_windows(..., n_moment=1)` with zero ground
        arithmetic of its own. See the module docstring's Ground section.
    wavelength: measurement wavelength in metres; k = 2π / wavelength.
    halfdriver_factor: informational only, for signature parity with the
        sibling solvers.
    feed_wire_index / feed_arclength: the single delta-gap source. The feed
        snaps to the SEGMENT whose centroid is nearest the requested arc
        length, because a pulse row is a segment.
    feeds: optional list of (wire_index, arclength_or_None, voltage).
    n_qp_source: Gauss-Legendre order per source segment for the vector
        potential's smooth remainder; its static part is closed-form, and
        the charge term takes no quadrature at all.
    cancel: optional :class:`~momwire._cancel.CancelToken`.
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    # momwire#396 / #416 / #430: free space and all three grounds — PEC, the
    # reflection-coefficient ground and now the Sommerfeld one — all
    # arriving through `PotentialGround` with no ground arithmetic in this
    # file. Everything else refused, reusing `_OUT_OF_SCOPE`'s prose rather
    # than restating it.
    capabilities = Capabilities(
        grounds=frozenset({"pec", "refl-coef", "sommerfeld"}),
        wire_loading=False,
        extended_kernel=False,
        junction_ports=False,
        node_gaps=False,
        per_wire_radius=False,
        singular_enrichment=False,
        refusals={
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
        ground_eps=None,
        ground_phi_mode="normal",
        ground_model="refl-coef",
        wavelength=22,
        halfdriver_factor=0.962,
        feed_wire_index=0,
        feed_arclength=None,
        feeds=None,
        n_qp_source=12,
        cancel=None,
        **unsupported,
    ):
        for name in unsupported:
            if name in _OUT_OF_SCOPE:
                raise NotImplementedError(f"{name}: {_OUT_OF_SCOPE[name]}")
        if unsupported:
            bad = ", ".join(sorted(unsupported))
            raise TypeError(f"PulseSolver got unexpected keyword argument(s): {bad}")

        self._cancel = cancel
        self._checkpoint()

        self.wavelength = float(wavelength)
        self.halfdriver_factor = float(halfdriver_factor)
        self.nsegs = int(nsegs)
        # One spelling of the scalar-only refusal for the whole tree
        # (momwire#147 / #429 rank 3): `per_wire_refusal=` is the argument
        # that exists precisely for this formulation now. It guarantees a
        # scalar, so the wire count is immaterial here (any sequence —
        # even length-1 — is refused, exactly as before).
        _radii, uniform = _wire_spec.normalize_wire_radius(
            wire_radius, 1, per_wire_refusal=_PER_WIRE_RADIUS_REFUSAL
        )
        self.wire_radius = uniform

        # The four attributes `_potential_ground`'s factory reads, and the
        # only place this class touches a ground string at all.
        self.ground_z = None if ground_z is None else float(ground_z)
        if self.ground_z is not None and not np.isfinite(self.ground_z):
            raise ValueError("ground_z must be finite")
        if ground_model not in ("refl-coef", "sommerfeld"):
            raise ValueError(
                "ground_model must be 'refl-coef' or 'sommerfeld', "
                f"got {ground_model!r}"
            )
        if ground_model == "sommerfeld" and ground_eps is None:
            raise ValueError("ground_model='sommerfeld' requires ground_eps")
        self.ground_eps = None if ground_eps is None else complex(ground_eps)
        self.ground_model = ground_model
        self.ground_phi_mode = ground_phi_mode

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = 2 * np.pi / self.wavelength
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        self.n_qp_source = int(n_qp_source)
        if self.n_qp_source < 1:
            raise ValueError("n_qp_source must be >= 1")

        if not wires:
            raise ValueError("wires must be non-empty")
        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")
        self._check_ground_clearance()

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

    def _check_ground_clearance(self):
        """Every wire must stand clear of the plane, or be refused.

        Submerged geometry is a `ValueError` (there is no such antenna);
        a wire TOUCHING the plane is ground CONTACT and a
        `NotImplementedError`. The pulse basis looks structurally ready for
        contact — the terminal segment's endpoint charge and its image
        charge land on the same point with opposite signs and cancel, which
        is exactly the "no charge accumulates where the current continues"
        the contact case needs — but nothing in this probe measures that,
        and an unmeasured claim is not a capability.

        Tolerance is `_ground_spec.ground_touch_tol`'s: 1e-6 of the wire's
        polyline length.
        """
        gz = self.ground_z
        if gz is None:
            return
        for i, pl in enumerate(self.wires_polylines):
            tol = _ground_spec.ground_touch_tol(pl)
            zmin = float(pl[:, 2].min())
            if zmin < gz - tol:
                raise ValueError(
                    f"wire {i} dips below the ground plane "
                    f"(min z = {zmin:.6g} < ground_z = {gz:g})"
                )
            if zmin <= gz + tol:
                raise NotImplementedError(
                    f"wire {i} touches the ground plane (min z = {zmin:.6g}, "
                    f"ground_z = {gz:g}): ground CONTACT is not supported by "
                    "PulseSolver — the fill would run, but no gate in this "
                    "probe measures it. Raise the wire, or use "
                    "BSplineSolver(ground_z=...), which carries the "
                    "grounded-end fold (momwire#151)"
                )

    def _build_geometry(self):
        """Discretize every wire and concatenate, in the shared vocabulary.

        The segment arrays are named `seg_l` / `seg_r` / `tangents` /
        `h_per_seg` on purpose: those are the keys
        `PotentialGround.weight_windows`' producers read, so speaking them
        is what lets this solver consume the ground object with no
        adapter. `per_wire` / `seg_offsets` are this module's own
        bookkeeping for the feed and the readout.
        """
        if self._cached_geometry is not None:
            return self._cached_geometry

        per_wire, seg_offsets = [], [0]
        l_list, r_list, t_list, h_list = [], [], [], []
        for w_idx, (pl, npe) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            w_l, w_r, w_t, w_h = [], [], [], []
            for e_idx in range(pl.shape[0] - 1):
                edge = pl[e_idx + 1] - pl[e_idx]
                edge_len = float(np.linalg.norm(edge))
                if edge_len < 1e-15:
                    raise ValueError(f"wire {w_idx} edge {e_idx} has zero length")
                tan = edge / edge_len
                n_e = npe[e_idx]
                h_e = edge_len / n_e
                start = pl[e_idx][None, :] + np.arange(n_e)[:, None] * h_e * tan
                w_l.append(start)
                w_r.append(start + h_e * tan)
                w_t.append(np.tile(tan, (n_e, 1)))
                w_h.append(np.full(n_e, h_e))
            seg_h = np.concatenate(w_h)
            per_wire.append(
                {
                    "arc_at_knot": np.concatenate([[0.0], np.cumsum(seg_h)]),
                    "n_total": seg_h.size,
                }
            )
            seg_offsets.append(seg_offsets[-1] + seg_h.size)
            l_list.append(np.vstack(w_l))
            r_list.append(np.vstack(w_r))
            t_list.append(np.vstack(w_t))
            h_list.append(seg_h)

        geom = {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "n_basis": seg_offsets[-1],
            "seg_l": np.vstack(l_list),
            "seg_r": np.vstack(r_list),
            "tangents": np.vstack(t_list),
            "h_per_seg": np.concatenate(h_list),
        }
        self._cached_geometry = geom
        return geom

    def _feed_basis_indices(self, geom):
        """Global basis index of each feed: the nearest segment CENTROID.

        A pulse row is a segment, so a delta gap lands on a segment and not
        on a knot — the opposite snap from the tent-basis solvers, and the
        simplest spelling consistent with the house feed conventions.
        """
        idx = []
        for w, arc, _v in self.feeds:
            arc_at_knot = geom["per_wire"][w]["arc_at_knot"]
            target = arc if arc is not None else arc_at_knot[-1] / 2.0
            centres = 0.5 * (arc_at_knot[:-1] + arc_at_knot[1:])
            idx.append(
                int(geom["seg_offsets"][w] + np.argmin(np.abs(centres - target)))
            )
        return idx

    # ------------------------------------------------------------------
    # kernel moments

    def _seg_M0(self, obs, src, k):
        """``M0[p, n] = ∫_{seg n} g(obs_p, r') dl'`` — the vector term's
        only ingredient, reduced kernel, 1/4π included.

        Split exactly as razor's `_seg_moments` splits it: the static
        1/(4πR) part in closed form (`_static_axis_moments`, borrowed
        along with `_axis_frame`) and the smooth remainder
        (exp(−jkR) − 1)/(4πR) by Gauss-Legendre. `src` is a dict of
        `seg_l` / `tangents` / `h_per_seg` — the real geometry, or the
        mirrored one, which is the whole of what the image costs.
        """
        a = self.wire_radius
        seg_p0, seg_t, seg_h = src["seg_l"], src["tangents"], src["h_per_seg"]
        n_seg, n_obs = seg_h.size, obs.shape[0]
        xg, wg = leggauss(self.n_qp_source)
        tau = 0.5 * seg_h[:, None] * (1.0 + xg[None, :])
        wq = 0.5 * seg_h[:, None] * wg[None, :]
        out = np.empty((n_obs, n_seg), dtype=np.complex128)
        step = max(1, _CHUNK_ELEMS // max(1, n_seg * self.n_qp_source))
        for lo in range(0, n_obs, step):
            self._checkpoint()
            hi = min(lo + step, n_obs)
            u_r, rho2 = _axis_frame(obs[lo:hi], seg_p0, seg_t, a)
            # `_static_axis_moments` also returns the tent basis's first
            # moment; a pulse has no use for it. Cheap next to the (obs x
            # seg x qp) remainder below, and noted in the probe record.
            m0s, _m1s = _static_axis_moments(u_r, rho2, seg_h)
            u = tau[None, :, :] - u_r[:, :, None]
            R = np.sqrt(u * u + rho2[:, :, None])
            rem = (np.exp(-1j * k * R) - 1.0) / R
            out[lo:hi] = (m0s + np.einsum("psq,sq->ps", rem, wq)) / (4.0 * np.pi)
        return out

    def _point_g(self, obs, src, k):
        """The reduced kernel between two POINT sets, ``(n_obs, n_src)``.

        R = sqrt(|r − r'|² + a²) never reaches zero, which is what lets the
        charge term be four kernel evaluations rather than an integral.
        """
        a2 = self.wire_radius * self.wire_radius
        n_obs, n_src = obs.shape[0], src.shape[0]
        out = np.empty((n_obs, n_src), dtype=np.complex128)
        step = max(1, _CHUNK_ELEMS // max(1, n_src))
        for lo in range(0, n_obs, step):
            self._checkpoint()
            hi = min(lo + step, n_obs)
            d = obs[lo:hi, None, :] - src[None, :, :]
            R = np.sqrt(np.einsum("psc,psc->ps", d, d) + a2)
            out[lo:hi] = np.exp(-1j * k * R) / (4.0 * np.pi * R)
        return out

    def _charge_stencil(self, geom, src, k):
        """``ΔΔg[m,n]``: the four-point charge stencil of the whole matrix.

        Observers are the REAL segment endpoints throughout (only sources
        move under the image); the two endpoint sets are stacked into one
        kernel block and differenced, so the kernel is evaluated once per
        endpoint pair rather than four times per segment pair.
        """
        n = geom["h_per_seg"].size
        obs = np.vstack([geom["seg_l"], geom["seg_r"]])
        G = self._point_g(obs, np.vstack([src["seg_l"], src["seg_r"]]), k)
        return G[n:, n:] - G[n:, :n] - G[:n, n:] + G[:n, :n]

    # ------------------------------------------------------------------
    # assembly

    def _source_block(self, geom, src, k, omega, w_A, w_Phi, remainder=None):
        """One source set's contribution: ``jωμ h_m w_A M0 + w_Φ ΔΔg / jωε``,
        plus the Sommerfeld remainder Q when `remainder` is given.

        `src` is the real geometry or the mirrored one; `w_A` / `w_Phi` are
        the per-pair weights — the free-space pairing ``t_m·t_n`` and 1 for
        the real sources, and `PotentialGround.weight_windows`' pair for
        the image (PEC's mirror-tangent-dot table, or `mode == "compose"`'s
        C₂-scaled copy of the same table — no branch of its own, see the
        module docstring). Nothing else differs between the two calls,
        which is what makes the ground a source-side substitution here
        exactly as it is on the razor row.

        `remainder` is `PotentialGround.remainder()` — `None` for every
        ground but Sommerfeld's, and passed only on the IMAGE call
        (`_assemble_Z`). Q is the smooth Sommerfeld remainder field, tested
        by THIS row's own rule: one centroid per row, with the SAME `h_m`
        row scaling `T1` already carries above (`field_windows((cent,
        geom["tangents"]), ..., n_moment=1)` is exactly a pulse basis's own
        zeroth arc moment — the shape momwire#398 unit 5 split out of the
        Galerkin-only `Remainder.evaluate` that blocked this row when
        momwire#416's probe first asked). Q takes no jωμ / 1/jωε prefactor
        (F is already a field) and integrates over the REAL source segments
        — `geom`, never `src` — even when this call IS the image one,
        because the plane enters Q through the interpolation grid rather
        than through a mirrored source. `C₂·img + Q` is associated HERE,
        before `_assemble_Z`'s single global minus takes it — the
        `mode == "compose"` contract, unchanged from `razor.py`'s.
        """
        h = geom["h_per_seg"]
        cent = 0.5 * (geom["seg_l"] + geom["seg_r"])
        T1 = (h[:, None] * w_A) * self._seg_M0(cent, src, k)
        T2 = w_Phi * self._charge_stencil(geom, src, k)
        block = 1j * omega * self.mu * T1 + T2 / (1j * omega * self.eps)
        if remainder is None:
            return block

        fw = remainder.field_windows(
            (cent, geom["tangents"]),
            (geom["seg_l"], geom["seg_r"], geom["tangents"]),
            n_moment=1,
        )
        n_obs = h.size
        Q = np.empty((n_obs, n_obs), dtype=np.complex128)
        step = max(1, _CHUNK_ELEMS // max(1, n_obs))
        for lo in range(0, n_obs, step):
            self._checkpoint()
            hi = min(lo + step, n_obs)
            Q[lo:hi] = h[lo:hi, None] * fw(lo, hi)[..., 0]
        return block + Q

    def _assemble_Z(self, geom, k):
        """Fill the point-matched impedance matrix at one wavenumber.

        Free space is one source block. Over a ground it is that block
        minus the image's — one global minus at the end, the `mode ==
        "fold"` contract (architecture doc §2.2), correct for BOTH terms at
        once because the image current runs anti-parallel and the image
        charge is opposite, and the direction bookkeeping that survives
        that minus is already inside `w_A`. Over the composing (Sommerfeld)
        ground it is the same single minus taken on `C₂·img + Q` instead of
        `img` alone — `mode == "compose"`, associated inside
        `_source_block` because `Z_free − (C₂·img + Q) ≠ (Z_free − C₂·img)
        − Q` in float64.

        `weight_windows()(0, n)` is the whole-geometry window — the N²
        weight surface momwire#416 predicted a dense point-matched fill
        would want. It is asked for once per fill, and `image_coefficient`
        is deliberately NOT applied here: the object's docstring is
        explicit that the tables already carry it.

        This solver has no prepare/replay split, so there is no schedule
        question: `ground.remainder()` — `None` for every ground but
        Sommerfeld's — is asked for once per fill, structurally reading the
        ground OBJECT's `mode` rather than `self.ground_model`, exactly as
        the weight windows above already do.
        """
        omega = self.c * k
        ground = _potential_ground.potential_ground_for(self, geom, k, omega)
        t = geom["tangents"]
        Z = self._source_block(geom, geom, k, omega, t @ t.T, 1.0)
        if ground is None:
            return Z

        img = ground.image_geometry()
        n = geom["h_per_seg"].size
        w_A, w_Phi = ground.weight_windows()(0, n)
        mirrored = {
            "seg_l": img.mirror_positions(geom["seg_l"]),
            "seg_r": img.mirror_positions(geom["seg_r"]),
            "tangents": img.mirror_tangents(geom["tangents"]),
            "h_per_seg": geom["h_per_seg"],
        }
        return Z - self._source_block(
            geom, mirrored, k, omega, w_A, w_Phi, remainder=ground.remainder()
        )

    # ------------------------------------------------------------------
    # solve

    def compute_impedance(self):
        """Drive-point impedance(s) and the solved pulse coefficients.

        Returns ``(z, coeffs)``. With one feed `z` is the scalar V / I on
        the fed segment; with N feeds it is the length-N vector V_i / I_i
        from the single solve against Σ_i V_i e_{m_i}. `coeffs` is the
        solved coefficient vector, which on a pulse basis IS the segment
        current in amperes, in segment order (per wire, arc order).
        """
        geom = self._build_geometry()
        self._checkpoint()
        Z = self._assemble_Z(geom, self.k)
        self.z = Z

        idx = self._feed_basis_indices(geom)
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        # The row already carries its own h_m (see the module docstring), so
        # the delta gap's volts land in the row undivided.
        rhs = np.zeros(geom["n_basis"], dtype=np.complex128)
        for m_i, v_i in zip(idx, voltages):
            rhs[m_i] += v_i

        self._checkpoint()
        coeffs = scipy.linalg.solve(Z, rhs)
        z_per_feed = voltages / np.array([coeffs[m] for m in idx])
        return (z_per_feed[0] if len(self.feeds) == 1 else z_per_feed), coeffs

    def compute_y_matrix(self):
        """Short-circuit admittance matrix [Y_sc] at the configured feeds.

        ``Y[i, j]`` is the current at port i when port j is driven with 1 V
        and every other port held at 0 V — N back-substitutions on one
        factored Z. NOT symmetric, and the module docstring says exactly
        where the asymmetry lives: point matching tests the vector term at
        one point per row, so only the charge term is reciprocal by
        construction.
        """
        geom = self._build_geometry()
        self._checkpoint()
        Z = self._assemble_Z(geom, self.k)
        self.z = Z

        idx = self._feed_basis_indices(geom)
        B = np.zeros((geom["n_basis"], len(idx)), dtype=np.complex128)
        for j, m_j in enumerate(idx):
            B[m_j, j] = 1.0

        self._checkpoint()
        return scipy.linalg.solve(Z, B)[idx, :]

    # ------------------------------------------------------------------
    # field readout

    def currents_at_knots(self, coeffs, s_array=None):
        """Per-wire complex current at every mesh KNOT, for `_ElementCurrents`.

        The pulse basis has no knot currents — its current is piecewise
        constant and DISCONTINUOUS at every knot — so this is a stated
        reading rather than a lookup: an interior knot takes the mean of
        its two adjacent pulses, and a wire END takes its single adjacent
        pulse's value.

        The end reading is the honest one and not the physical one. A free
        wire end carries no current in reality, but this basis does not
        know that (the module docstring's "the charge carries the jumps"),
        and reporting 0 there would hide the scheme's largest
        discretization artefact from every field consumer. It shrinks with
        the mesh; it is not zeroed by hand.

        With `s_array` given as per-wire arc-length arrays, the values are
        linearly interpolated between knots, matching how the mean above
        already continuises the staircase.
        """
        coeffs = np.asarray(coeffs)
        geom = self._build_geometry()
        seg_offsets = geom["seg_offsets"]

        out = []
        for w_idx in range(len(geom["per_wire"])):
            seg_I = coeffs[seg_offsets[w_idx] : seg_offsets[w_idx + 1]]
            I = np.empty(seg_I.size + 1, dtype=np.complex128)
            I[1:-1] = 0.5 * (seg_I[:-1] + seg_I[1:])
            I[0], I[-1] = seg_I[0], seg_I[-1]
            out.append(I)

        if s_array is None:
            return out

        sampled = []
        for w_idx, sv in enumerate(s_array):
            arc = geom["per_wire"][w_idx]["arc_at_knot"]
            knot_I = out[w_idx]
            sv = np.asarray(sv, dtype=np.float64)
            sampled.append(
                np.interp(sv, arc, knot_I.real) + 1j * np.interp(sv, arc, knot_I.imag)
            )
        return sampled
