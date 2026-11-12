"""The public field-readout hook (`stevenmburns/momwire#251`).

`currents_at_knots()` already hands a consumer the solved current sampled on a
wire, but every field evaluator that has used it since then has had to bolt the
same two steps onto it: resample the mesh at a finer arc spacing than the solve
used, and difference the resulting element currents into the per-knot
continuity steps a scalar potential needs. Both steps read nothing but the
polylines, the resolved per-edge segment counts, and `currents_at_knots` — none
of which is family-specific — so they belong here once rather than twice, and
they belong in momwire rather than in each consumer's copy of the walk.

The mixin lands on the classes that define their own `currents_at_knots`:
`SinusoidalSolver`, `BSplineSolver`, and `RazorSolver` (momwire#309), each
with its own basis-specific readout. The other three public solvers
(`SinusoidalGalerkinSolver`, `HMatrixSolver`, `ArrayBlockSolver`) inherit
`currents_at_knots` from `SinusoidalSolver` or `BSplineSolver` and pick up
this mixin unchanged along with it.
"""

from __future__ import annotations

import numpy as np
from . import _mesh_walk


class _ElementCurrents:
    """Mixin adding `element_currents()` to a solver family.

    Requires of the host class only `wires_polylines`, the post-`__init__`
    `n_per_edge_per_wire` (already normalised to one int per edge per wire),
    and a `currents_at_knots(coeffs, s_array=None)`.
    """

    def element_currents(self, coeffs, subdiv: int = 1):
        """`(mid, moment, nodes, delta)` — the field evaluator's source terms.

        Walks the mesh as a chain of straight ELEMENTS (knot to knot) and
        returns, for the whole structure with every wire concatenated in
        `wires_polylines` order:

        `mid` : `(n_elem, 3)` float
            Element midpoints.
        `moment` : `(n_elem, 3)` complex
            The current moment `I·dl` of each element: the element's current
            (the mean of its two end-knot currents) times its vector length.
            These are the terms of the far-field sum.
        `nodes` : `(n_knots, 3)` float
            The knots themselves, `n_knots = n_elem + n_wires`.
        `delta` : `(n_knots,)` complex
            The current STEP across each knot — current arriving minus current
            leaving — which is the discrete continuity charge the near-field
            scalar potential is built from.

        Delta convention. A wire's two endpoint knots get the full current of
        their single adjacent element, because nothing carries current past a
        wire end. Two wires meeting at a junction therefore contribute one knot
        EACH at the same point, with the outflow of each; the knots are not
        merged and their steps are not summed here, because the sum is over
        coincident positions and every consumer of `delta` is already summing
        over positions. A wire's own steps telescope, so each wire's slice of
        `delta` sums to zero up to round-off.

        Subdivision. `subdiv=1` (the default) samples exactly the mesh the
        solve used. `subdiv=k` splits every mesh segment into `k` elements and
        resamples the solved current at the intermediate arc positions via
        `currents_at_knots(coeffs, s_array=...)`. The far field never needs it
        — every mesh segment is already electrically small and only the
        radiation-zone limit survives — but a near-field observer a metre from
        a half-metre segment resolves the current variation along it.

        `coeffs` is one solution vector in this family's own basis: a column of
        `compute_port_solution().coeffs` is a valid input, as is anything else
        `currents_at_knots` accepts. It must be 1-D — pass `sol.coeffs[:, j]`,
        not `sol.coeffs`.
        """
        subdiv = int(subdiv)
        if subdiv < 1:
            raise ValueError(f"subdiv must be >= 1, got {subdiv}")
        coeffs = np.asarray(coeffs)
        if coeffs.ndim != 1:
            raise ValueError(
                "element_currents takes ONE solution vector; got coeffs with "
                f"shape {coeffs.shape}. For a multi-port solution pass a single "
                "column, e.g. compute_port_solution().coeffs[:, j]."
            )

        polylines, per_edge = self.wires_polylines, self.n_per_edge_per_wire
        dedup = getattr(self, "_wire_dedup", None)
        if dedup is not None:
            # A wire listed twice was solved once (momwire#1042): walk the
            # CALLER's wires, the dropped copy over its twin's knots (reversed
            # when it runs the other way), so `currents_at_knots` — which
            # answers in the caller's numbering, the copy at zero current —
            # lines up wire for wire.
            polylines = dedup.to_input(
                list(polylines), lambda r, v, rev: v[::-1] if rev else v
            )
            per_edge = dedup.to_input(
                list(per_edge), lambda r, v, rev: list(v)[::-1] if rev else v
            )

        # The knot walk over every edge of every wire at once
        # (`_mesh_walk.wire_knots`, bit for bit the per-edge linspace walk).
        knots, offs = _mesh_walk.wire_knots(polylines, per_edge, subdiv)
        if subdiv == 1:
            # The mesh knots by name, so the sample positions are the
            # solver's own, bit for bit.
            currents = self.currents_at_knots(coeffs, None)
        else:
            # Arc measured along the CHORDS, matching how both families
            # accumulate `arc_at_knot` from the per-segment lengths.
            arc = _mesh_walk.wire_arcs(knots, offs)
            currents = self.currents_at_knots(
                coeffs, [arc[a:b] for a, b in zip(offs[:-1], offs[1:])]
            )
        if len(currents) != offs.size - 1:
            raise ValueError(
                f"currents_at_knots answered {len(currents)} wires for "
                f"{offs.size - 1} walked"
            )
        cur = (
            np.concatenate([np.asarray(c) for c in currents])
            if currents
            else np.zeros(0, dtype=np.complex128)
        )
        if cur.shape[0] != knots.shape[0]:
            raise ValueError(
                f"currents_at_knots answered {cur.shape[0]} knots for "
                f"{knots.shape[0]} walked"
            )
        # Per wire: element = mean of its two end knots, mid the same of the
        # knots, moment = element * chord, and the step at each knot (the
        # current arriving minus the current leaving, 0 past either end).
        lo, hi = _mesh_walk.interior_pairs(offs)
        element = 0.5 * (cur[hi] + cur[lo])
        mids = 0.5 * (knots[hi] + knots[lo])
        moments = element[:, None] * (knots[hi] - knots[lo])
        arriving = np.zeros(knots.shape[0], dtype=np.result_type(element, 0j))
        leaving = np.zeros(knots.shape[0], dtype=arriving.dtype)
        arriving[hi] = element
        leaving[lo] = element
        return mids, moments, knots, arriving - leaving
