"""The exact ring kernel's solver-independent plumbing (momwire#1433).

`_exact_kernel` is the kernel layer -- K(z), its exact log split, and the
integrals of K against a basis. This module is everything a SOLVER needs
around it that does not depend on how that solver integrates: the
`exact_kernel=False | True | "auto"` option and its resolution, the
construction-time and fill-route refusals, the coaxial groups the correction
covers, and the axial frame of a group. It was `BSplineSolver`'s own code
until momwire#1433 lifted it here so razor-2p (#1433), sin (#1434) and SG
(#1435) take the same option with the same semantics, and so a front end
(the roster in `deck/_solver.py`, the advisories in `_wire_spec`) can ask a
solver class the same questions whichever family it is.

What stays per solver, and why it is not lifted here
----------------------------------------------------
The correction INTEGRAL. bs2 tests by Galerkin, so its correction needs the
2-D moments J_pq = ∫∫ u^p v^q K(c + v - u) du dv of every coaxial pair; the
self and adjacent pairs carry the log singularity across the whole of both
segments, and `_exact_kernel.pair_moments` reduces them EXACTLY to one
dimension (the closed-form piecewise-polynomial weight W_pq) before any
quadrature. The crossing trunk's composition seams (`_crossing_fill`'s
`BasisSampler` on the source side, `path_test_axis` on the test side) sample
a basis at fixed quadrature NODES and contract point kernel tables between
them; a node-sampled outer integral over a self or adjacent pair sees an
(u - e) ln|u - e| endpoint singularity and converges algebraically (~n^-4),
eight orders short of the 1e-12 the 2-D closed form holds -- and a
re-spelling of bs2's correction would change its bits, which this module's
own gate forbids. A point- or path-tested row (razor-2p's centroids, sin's
match points) has no outer integral to lose: it needs only the 1-D moments of
K against the source polynomials at an observer point, which is the kernel
layer's second primitive, 1-D point moments (razor-2p, momwire#1433). So the
kernel and its integration primitives are shared, this plumbing is shared,
and each solver keeps the lines that assemble a primitive into its own rows.
"""

from __future__ import annotations

import numpy as np

from ._bspline_kernels import _ek_axis_groups_coaxial
from ._vec3 import norm3

# What `exact_kernel=` accepts. Public on every serving class (re-exported as
# `EXACT_KERNEL_CHOICES`), so a consumer can gate on the capability ("auto"
# in Solver.EXACT_KERNEL_CHOICES) instead of parsing a version or a signature.
EXACT_KERNEL_CHOICES = (False, True, "auto")
# "auto" engages when some segment's length is below this many radii (the
# kernel-error table: EK 2.1e-3 at |z|/a = 3, 3.1e-2 at 1).
EXACT_KERNEL_AUTO_H_OVER_A = 3.0

# The route refusals, one spelling for every family (momwire#1408).
RESTRICTED_ROUTE_REFUSAL = (
    "exact_kernel=True serves the dense fill only; a restricted-row "
    "fill (sector route) has no exact-kernel correction (momwire#1408)"
)
BURIED_REFUSAL = (
    "exact_kernel=True is refused for buried wires: the exact ring "
    "kernel here is written for a real wavenumber (momwire#1408)"
)
LOSSY_REFUSAL = (
    "exact_kernel=True is refused in a lossy medium (complex k or "
    "permittivity) (momwire#1408)"
)


def parse_option(exact_kernel):
    """``(requested, exact)`` for an `exact_kernel=` argument: what was asked
    for (False, True or "auto"), and the value `exact_kernel` holds until
    "auto" is resolved (False for "auto")."""
    if isinstance(exact_kernel, str):
        if exact_kernel != "auto":
            raise ValueError(
                f"exact_kernel must be True, False or 'auto', got {exact_kernel!r}"
            )
        return "auto", False
    return bool(exact_kernel), bool(exact_kernel)


def not_served_refusal(cls_name):
    """The construction-time refusal of a class whose fills never reach the
    correction."""
    return (
        f"exact_kernel=True is not served by {cls_name}: "
        "its fills never reach the exact-kernel correction "
        "(momwire#1408)"
    )


def point_gap_warning(fixed_gap_feed=True):
    """The RuntimeWarning text for `exact_kernel=True` with a zero-width
    (point or knot) gap feed."""
    head = (
        "exact_kernel=True with a point-gap feed: under the exact ring "
        "kernel a zero-width gap has no limiting reactance (each halving "
        "of the gap adds 4 ln2 w eps0 a of susceptance), so the feed "
        "reactance drifts with the mesh. "
    )
    if fixed_gap_feed:
        return head + (
            "A fixed geometric gap is two "
            "facing wire ends driven through junction_ports (momwire#1408)."
        )
    return head + (
        "This solver has no fixed-width gap; bspline's end-port feed (two "
        "facing wire ends driven through junction_ports) is one "
        "(momwire#1408)."
    )


def coaxial_groups(seg_l, seg_r, tangents, seg_a):
    """``(labels, [(label, segment indices)])`` per coaxial equal-radius
    group, in label order: the segments the one-radius exact ring kernel
    covers. Every segment is in one (its own, at least). The pairs between
    groups on one line are `coaxial_lines`'."""
    labels = _ek_axis_groups_coaxial(seg_l, seg_r, tangents, seg_a)
    return labels, [(int(g), np.flatnonzero(labels == g)) for g in np.unique(labels)]


def coaxial_lines(seg_l, seg_r, tangents, seg_a, labels):
    """``[(segment indices, their radius-group labels)]`` per coaxial LINE
    that carries more than one radius (momwire#1438): the segments whose
    cross-radius pairs take the two-radius kernel (`_exact_kernel.
    ring_kernel2`). `labels` are `coaxial_groups`' one-radius labels.

    Which pairs, and why all of them: every pair on the line whose two
    segments sit in different radius groups, at any separation. The exact
    kernel is exact at every distance, so engagement is a property of the
    geometry (same line) and never of a distance cutoff that Z could jump
    across; the one-radius groups take every pair of the group the same
    way. Far pairs cost one vectorised rule each and move by the EK error
    there (3.6e-5 relative at |z|/a = 10). The pairs that matter are the
    ones touching the step, where the fat side's h/a is small and the
    thin-wire kernel's error is O(1).

    A line is the one-radius rule with the radius test dropped
    (`_ek_axis_groups_coaxial` with every radius set to the largest, so its
    perpendicular tolerance is tol·a_max): same line in NEC's thresholds. A
    line holding one radius group has no cross pairs and is not listed, so
    a deck without a radius step never enters the two-radius path."""
    seg_a = np.asarray(seg_a, dtype=float)
    line = _ek_axis_groups_coaxial(
        seg_l, seg_r, tangents, np.full(seg_a.shape, float(np.max(seg_a)))
    )
    out = []
    for g in np.unique(line):
        S = np.flatnonzero(line == g)
        if np.unique(labels[S]).size > 1:
            out.append((S, labels[S]))
    return out


def group_frame(S, seg_l, seg_r, tangents):
    """``(axis, x0, sgn, h)`` of group `S`: the axis (the first segment's
    tangent), each segment's START along it measured from the first
    segment's start, its direction ±1 along it, and its length."""
    axis = tangents[S[0]]
    x0 = (seg_l[S] - seg_l[S[0]]) @ axis
    sgn = np.where(tangents[S] @ axis >= 0.0, 1.0, -1.0)
    h = norm3(seg_r[S] - seg_l[S])
    return axis, x0, sgn, h


class ExactKernelOption:
    """The `exact_kernel` option, mixed into a serving solver class.

    The class supplies four hooks -- what it refuses besides the shared
    routes, whether the deck is buried or lossy, and its segment lengths and
    radii -- and gets the option's parsing, the "auto" resolution, the
    servable check and the fill-route gate, with the semantics bs2 shipped in
    momwire#1408 (see `BSplineSolver`'s `exact_kernel` docstring, which is
    the long form).
    """

    # Subclasses that replace the dense fill (H-matrix, array blocks) never
    # reach the correction, so they set this False and refuse it.
    _serves_exact_kernel = True
    EXACT_KERNEL_CHOICES = EXACT_KERNEL_CHOICES
    EXACT_KERNEL_AUTO_H_OVER_A = EXACT_KERNEL_AUTO_H_OVER_A
    # Whether this family can drive a gap of FIXED width (bs2's end-port
    # feed: two facing wire ends through `junction_ports`). Under the exact
    # kernel a zero-width gap has no limiting reactance, so the advisories
    # and the point-gap warning name that feed -- or, for a family without
    # one, say where it is (`_wire_spec.advise_gap_mesh_floor`).
    EXACT_KERNEL_FIXED_GAP_FEED = True

    # -- hooks ----------------------------------------------------------------
    def _exact_kernel_family_refusals(self):
        """``[(condition, message)]``: construction-time refusals of
        `exact_kernel=True` specific to this family (each also makes "auto"
        resolve False)."""
        return []

    def _exact_kernel_buried(self):
        """Whether some wire lies below the ground (refused by name)."""
        raise NotImplementedError

    def _exact_kernel_lossy(self):
        """Whether the medium is lossy (complex k or permittivity)."""
        raise NotImplementedError

    def _exact_kernel_h_and_a(self):
        """``(h, a)``: every segment's length and radius, on the final mesh."""
        raise NotImplementedError

    # -- the option -----------------------------------------------------------
    def _exact_kernel_init(self, exact_kernel, refusals=()):
        """Parse `exact_kernel` and raise this class's construction-time
        refusals of True: the shared one, then `refusals` -- the family's
        ``[(condition, message)]`` read off the constructor's arguments,
        since the attributes `_exact_kernel_family_refusals` reads may not
        exist yet. "auto" is resolved later (`_resolve_exact_kernel_auto`),
        once geometry and ground are known."""
        self.exact_kernel_requested, self.exact_kernel = parse_option(exact_kernel)
        if self.exact_kernel:
            if not self._serves_exact_kernel:
                raise NotImplementedError(not_served_refusal(type(self).__name__))
            for cond, message in refusals:
                if cond:
                    raise NotImplementedError(message)

    def _resolve_exact_kernel_auto(self):
        """`exact_kernel="auto"`'s value: True iff this route serves the
        exact kernel and some segment is shorter than
        `EXACT_KERNEL_AUTO_H_OVER_A` radii. Never raises a refusal."""
        if not self._exact_kernel_servable():
            return False
        h, a = self._exact_kernel_h_and_a()
        return bool(np.any(h < self.EXACT_KERNEL_AUTO_H_OVER_A * a))

    def _exact_kernel_servable(self):
        """Whether this solver's construction-time route serves the exact
        ring kernel: condition (i) of "auto"."""
        if not self._serves_exact_kernel:
            return False
        if any(cond for cond, _m in self._exact_kernel_family_refusals()):
            return False
        if self._exact_kernel_buried():
            return False
        return not self._exact_kernel_lossy()

    def _exact_kernel_resolved(self):
        """``(exact_runs, servable)`` after "auto" is resolved, for the
        advisories (momwire#1412). True with a route the fill will refuse
        (buried, lossy) counts as not running: the fill raises before any
        number comes out. A wire-media reading the fill refuses by name (a
        crossing, a wire below a ground with no lower medium) says so there,
        not from an advisory, so it reads as not servable."""
        try:
            servable = self._exact_kernel_servable()
        except ValueError:
            servable = False
        return bool(self.exact_kernel) and servable, servable

    def _exact_kernel_route_refusal(self, rows, compact):
        """Why this fill route does not serve `exact_kernel`, or None."""
        if rows is not None or compact:
            return RESTRICTED_ROUTE_REFUSAL
        if self._exact_kernel_buried():
            return BURIED_REFUSAL
        if self._exact_kernel_lossy():
            return LOSSY_REFUSAL
        return None

    def _refuse_exact_kernel_route(self, rows, compact):
        """The routes `exact_kernel` does not serve, refused by name."""
        why = self._exact_kernel_route_refusal(rows, compact)
        if why is not None:
            raise NotImplementedError(why)

    def _exact_kernel_fill_gate(self, rows=None, compact=False):
        """Whether this fill applies the correction. "auto" never refuses: an
        unserved route skips the correction and the solver says so
        (`exact_kernel` drops to False); True refuses it by name."""
        use_exact = self.exact_kernel
        if use_exact:
            if self.exact_kernel_requested == "auto":
                if self._exact_kernel_route_refusal(rows, compact) is not None:
                    use_exact = self.exact_kernel = False
            else:
                self._refuse_exact_kernel_route(rows, compact)
        return use_exact
