"""The public field-readout hook (`stevenmburns/momwire#251`).

`currents_at_knots()` already hands a consumer the solved current sampled on a
wire, but every field evaluator that has used it since then has had to bolt the
same two steps onto it: resample the mesh at a finer arc spacing than the solve
used, and difference the resulting element currents into the per-knot
continuity steps a scalar potential needs. Both steps read nothing but the
polylines, the resolved per-edge segment counts, and `currents_at_knots` — none
of which is family-specific — so they belong here once rather than twice, and
they belong in momwire rather than in each consumer's copy of the walk.

The mixin lands on the two classes that define `currents_at_knots`
(`SinusoidalSolver` and `BSplineSolver`); the other three public solvers
(`SinusoidalGalerkinSolver`, `HMatrixSolver`, `ArrayBlockSolver`) inherit from
one of those two and pick it up unchanged.
"""

from __future__ import annotations

import numpy as np


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

        fine: list[np.ndarray] = []
        arcs: list[np.ndarray] = []
        for polyline, npe in zip(self.wires_polylines, self.n_per_edge_per_wire):
            parts = []
            for i, n_e in enumerate(npe):
                seg = np.linspace(polyline[i], polyline[i + 1], n_e * subdiv + 1)
                parts.append(seg if i == 0 else seg[1:])
            knots = np.vstack(parts)
            fine.append(knots)
            # Arc measured along the CHORDS, matching how both families
            # accumulate `arc_at_knot` from the per-segment lengths. Only the
            # subdiv > 1 path uses it; subdiv == 1 asks for the mesh knots by
            # name so the sample positions are the solver's own, bit for bit.
            step = np.linalg.norm(knots[1:] - knots[:-1], axis=1)
            arcs.append(np.concatenate([[0.0], np.cumsum(step)]))

        currents = self.currents_at_knots(coeffs, None if subdiv == 1 else arcs)

        mids, moments, nodes, deltas = [], [], [], []
        for knots, cur in zip(fine, currents, strict=True):
            cur = np.asarray(cur)
            element = 0.5 * (cur[1:] + cur[:-1])
            mids.append(0.5 * (knots[1:] + knots[:-1]))
            moments.append(element[:, None] * (knots[1:] - knots[:-1]))
            nodes.append(knots)
            zero = np.zeros(1, dtype=np.complex128)
            deltas.append(
                np.concatenate([zero, element]) - np.concatenate([element, zero])
            )
        return (
            np.concatenate(mids, axis=0),
            np.concatenate(moments, axis=0),
            np.concatenate(nodes, axis=0),
            np.concatenate(deltas, axis=0),
        )
