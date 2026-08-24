"""The 1967 pulse row, done to Harrington's own specification.

`PulseSolver` is the pulse basis with the charge left where the basis
literally puts it: two POINT charges per element, whose potential at their
own location is finite only because the reduced kernel floors R at the wire
radius. That is a real scheme and it converges — to the same limit as every
other row in the tree — but its error is governed by Δ/a rather than Δ/λ, so
on a thin wire it needs ~64× the mesh of `BSplineSolver` (momwire#557's
table). Harrington did not do that, and this module is the difference.

Reference
---------
R. F. Harrington, "Matrix Methods for Field Problems," *Proc. IEEE* 55(2),
136-149, Feb 1967, §IX and the Appendix. Three of its statements are the
whole of this file:

  (99)  ψ(n,m) = (1/Δl_n) ∫_{Δl_n} e^{−jkR}/(4πR) dl

        ψ is the source-segment AVERAGED Green's function. No entry of
        Harrington's matrix is ever a bare point-to-point kernel
        evaluation — the averaging is part of the formulation, not a
        quadrature refinement bolted on afterwards.

  (95)/(96)/(100)

        σ(ṅ) = −(1/jω)[I(n+1) − I(n)] / Δl_ṅ, with q = σΔl = ±I(n)/jω.
        The charge is a LINE DENSITY over the half-shifted increment
        Δl^± — a cell of length Δl centred on each node — and not a point
        charge. This is the deviation that matters.

  (103) Z_mn = jωμ Δl_n Δl_m ψ(n,m)
              + (1/jωε)[ψ(ṅ,ṁ) − ψ(n̄,ṁ) − ψ(ṅ,m̄) + ψ(n̄,m̄)]

        the same four-term charge stencil `PulseSolver` fills — except
        every ψ in it is a cell average rather than a point evaluation.

The vector term is already Harrington's on the parent row (`_seg_M0` IS
(99) for the current), so this class inherits it untouched and overrides
exactly one method: `_charge_stencil`. Same basis, same testing, same
kernel, same feed, same grounds — one ingredient differs. That is
deliberate: the pair is an instrument, in the sense
`SinusoidalGalerkinSolver` is one for its own row, and any gap measured
between the two is attributable to the charge's SUPPORT and to nothing
else.

Why the cell has to be centred on the node
------------------------------------------
Two cheaper cells were tried and measured before this one was written
(momwire#557):

**Charge inside its own segment** (each basis smearing its ±q over the
half of segment n adjacent to the node, needing no adjacency information
at all — which would have preserved the parent row's best property)
DIVERGES: the error grows with N, 201 Ω → 382 Ω over N = 25…801. At a
shared node the + cell of segment n and the − cell of segment n+1 must be
the SAME region; side by side instead, they leave a dipole layer of
strength ~qh at every node that refinement does not retire.

**Each wire truncating its own cell at a junction** leaves a colinear
split disagreeing with the unsplit wire by ~10%, and the disagreement
GROWS with refinement rather than decaying — a split would permanently be
a different antenna. So the branched ("star") cell spanning every incident
half-segment is not an optimization, it is the scheme.

Which is why this row detects junctions and the parent does not. That is
not a defect discovered in `PulseSolver`: superposing point charges at a
shared point really is exact bookkeeping (Kirchhoff by arithmetic, the
parent's docstring), and it costs no adjacency machinery. It is only
inaccurate. Buying the accuracy costs the adjacency — and pricing that
trade is what momwire#416's probe exists to do, so the two rows stand
side by side rather than one replacing the other.

Evaluating ψ
------------
Harrington's Appendix is an accuracy ladder for (99): a two-term closed
form (126) for r ≤ 2α, the point approximation (127) beyond it, and for
"better than one percent" a phase-extracted series (129) with closed-form
moments (130)-(133) below r = 10α plus a five-term multipole (135) above.

None of it is implemented here, on purpose. That ladder is a 1967 answer
to "we have no quadrature layer," and momwire has one: `_seg_M0` already
integrates the reduced kernel over an arbitrary segment as a closed-form
static moment plus a Gauss-Legendre remainder, and a half-segment is just
a segment with h/2. So every ψ below is built from the machinery the
vector term already uses — no new numerics, no new special functions, and
an integral rather than a series. The whole charge model is one geometry
construction and one contraction.

Cost: the moment fill runs (2n observers × 2n cell pieces) against the
vector term's (n × n), so a fill here is ~5× the parent's. The cheap row
is still on the roster under its own name.
"""

from __future__ import annotations

import numpy as np

from . import _wire_spec
from ._capabilities import Capabilities
from .pulse import _OUT_OF_SCOPE, _PER_WIRE_RADIUS_REFUSAL, PulseSolver

# Absolute tolerance for calling two wire ends one node — IMPORTED from
# razor rather than re-typed as the same number, because this row walks
# razor's grouping algorithm (see `_node_map`) and two solvers that must
# agree about connectivity should not be able to disagree about the number
# that decides it. There are six disagreeing tolerances across three
# algorithms in this tree already; unifying them is a deliberate future
# decision, and this is one fewer site for it to find.
#
# The caveat travels with the value: the deck front end fuses endpoints onto
# a 1e-6 grid (`deck/_polylines._NODE_EPS`), a thousand times looser and a
# different algorithm, so agreement with the rest of the tree is by
# convention rather than construction. `razor._find_junctions` carries the
# full account; `_node_map` refuses the gap between the two rather than
# answering across it.
from ._junction_rule import JUNCTION_TOL as _JUNCTION_TOL
from ._junction_rule import canonical_groups, coincident_groups

# The deck layer's "same point" grid (`deck/_polylines._NODE_EPS`). Ends
# between the two tolerances are refused rather than silently disconnected —
# see `_node_map`.
_NEAR_COINCIDENT_TOL = 1e-6

# `junctions=` is refused here for a DIFFERENT reason than on the parent
# row, so the sentence cannot be inherited: the parent has nothing to
# detect, this row detects it itself.
_EXTENDED_KERNEL_REFUSAL = (
    "HarringtonSolver is reduced-kernel only. Unlike PulseSolver the charge "
    "is no longer a point, so the a² floor is not what keeps this "
    "formulation finite — but the vector term and every charge-cell moment "
    "are written against the reduced kernel, and the exact kernel would "
    "need both rewritten rather than a different quadrature"
)


def _pieces_of(geom) -> dict:
    """Each segment cut in half, in the `seg_l` / `tangents` / `h_per_seg`
    vocabulary `_seg_M0` reads — piece 2n is segment n's arc-0 half, piece
    2n+1 its arc-h half.

    One spelling, called twice: `_node_map` builds the real geometry's
    pieces and `_mirror_pieces` the image's. The interleave is load-bearing
    (`cell_of_piece` is indexed by it), so writing it out in both places
    would mean a change to one silently desynchronising the other.
    """
    half = 0.5 * geom["h_per_seg"]
    mid = geom["seg_l"] + half[:, None] * geom["tangents"]
    piece_l = np.empty((2 * half.size, 3))
    piece_l[0::2], piece_l[1::2] = geom["seg_l"], mid
    return {
        "seg_l": piece_l,
        "tangents": np.repeat(geom["tangents"], 2, axis=0),
        "h_per_seg": np.repeat(half, 2),
    }


class HarringtonSolver(PulseSolver):
    """Pulse basis, point-matched, with Harrington's dual-cell charge.

    Every constructor argument is `PulseSolver`'s and means the same thing;
    see that class. The formulation differs in one place, the support of
    the charge, described in this module's docstring.

    The convergence this buys, free space, on the ByDipole1 wire
    (10.19 m of #14 at 14 MHz, L/a = 9929) against a converged
    `BSplineSolver` degree 2 (67.86 − 27.89j Ω):

    |    N | Δ/a | this row            | `PulseSolver`        |
    |------|-----|---------------------|----------------------|
    |   25 | 397 |  75.53 +  22.46j    |  45.38 − 14926.38j   |
    |  101 |  98 |  69.58 −  16.38j    |  44.69 −  3632.16j   |
    |  401 |  25 |  68.24 −  25.25j    |  50.14 −   809.91j   |
    |  801 |  12 |  68.03 −  26.63j    |  55.91 −   351.02j   |

    Error halving per doubling or better — the classical O(1/N) rate the
    1967 scheme is supposed to deliver. Harrington's own prescription for
    doing better is on p.145 and it is to leave this basis: "faster
    convergence can be obtained by going from a step approximation to a
    piecewise-linear approximation to the current" — `BSplineSolver`
    degree 1, three rows over.
    """

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
            # NOT inherited: the parent's sentence is false here. It
            # refuses the exact kernel because "the charge is two POINT
            # charges per basis, whose potential on the wire axis is finite
            # only because of the reduced kernel's a²" — which is precisely
            # the defect this class removes. The refusal still stands, for
            # its own reason, and antennaknobs surfaces this prose verbatim
            # in host dialogs, so it has to be true of the class it names.
            "extended_kernel": _EXTENDED_KERNEL_REFUSAL,
            "per_wire_radius": _PER_WIRE_RADIUS_REFUSAL,
        },
    )

    def __init__(self, **kwargs):
        # momwire#590 step 3b. This used to refuse `junctions=` on the grounds
        # that coincident ends are found from the geometry "so there is
        # nothing to declare". True of AGREEING with the geometry, false of
        # disagreeing with it: two coincident ends a caller wants left apart
        # were inexpressible here, and every other solver could say it.
        #
        # Intercepted rather than forwarded: the PARENT (PulseSolver) refuses
        # `junctions=` for a different and still-valid reason -- its basis has
        # no junction unknown to constrain at all.
        junctions = kwargs.pop("junctions", None)
        super().__init__(**kwargs)
        self._declared_junctions = (
            None if junctions is None else [list(g) for g in junctions]
        )
        if self._declared_junctions is not None:
            # momwire#522's guardrail, which BSplineSolver and SinusoidalSolver
            # have run since that issue. Razor and harrington only started
            # accepting a spec in momwire#590 step 3b, so they were the two
            # spellings it did not cover -- a wrong wire index welds ends that
            # sit nowhere near each other and produces a well-posed WRONG model
            # that converges cleanly, which is the #518 postmortem exactly.
            #
            # Calling the existing check rather than writing a second one: its
            # tolerance is scale-aware (1e-3 of the shortest terminal segment,
            # floored at 1e-5 m so the deck front's node grid can never fire
            # it), and a flat threshold picked here would be both a seventh
            # "same point" number and a worse-calibrated one.
            _wire_spec.check_junction_coincidence(
                self.wires_polylines,
                self.n_per_edge_per_wire,
                canonical_groups(self._declared_junctions),
            )
        self._cached_cells = None

    # ------------------------------------------------------------------
    # the charge cells

    def _node_map(self, geom):
        """Global node index per segment end, and each node's cell pieces.

        Returns ``(left_node, right_node, pieces, cell_of_piece)``:

        `left_node[n]` / `right_node[n]` are the node indices of segment
        n's two ends. `pieces` is the per-half-segment geometry — 2·n_seg
        of them, piece 2n being segment n's arc-0 half and piece 2n+1 its
        arc-h half — in the `seg_l` / `tangents` / `h_per_seg` vocabulary
        `_seg_M0` reads, so a cell piece IS a segment as far as the moment
        fill is concerned. `cell_of_piece[p]` is the node each piece's
        charge belongs to.

        Within a wire the knots are structural: segment n's arc-h end and
        segment n+1's arc-0 end are the same knot by construction, so they
        share a node with no comparison at all. Only WIRE ENDS are matched
        geometrically, by first match within `_JUNCTION_TOL` — including a
        wire against itself, so a closed loop joins its own two ends.
        """
        if self._cached_cells is not None:
            return self._cached_cells

        offsets = geom["seg_offsets"]
        n_wires = len(geom["per_wire"])
        n_seg = geom["h_per_seg"].size

        # Per-wire knots numbered 0..n_w; node ids are (knot index + the
        # wire's knot offset) before any cross-wire merging.
        knot_offsets, total = [], 0
        for w in range(n_wires):
            knot_offsets.append(total)
            total += (offsets[w + 1] - offsets[w]) + 1
        node_of = np.arange(total)

        # Merge coincident WIRE ENDS by the shared rule (`_junction_rule`,
        # momwire#590 step 1). This block used to walk razor's algorithm by
        # hand, with a comment explaining that two solvers disagreeing about
        # the same deck's CONNECTIVITY is a worse failure than either answer —
        # so the copy was deliberate, and deleting it in favour of one spelling
        # is that comment taken at its word. The non-transitive first-match
        # semantics it warned about is now stated once, where it is executed.
        ends = []
        for w in range(n_wires):
            n_w = offsets[w + 1] - offsets[w]
            ends.append((knot_offsets[w], geom["seg_l"][offsets[w]]))
            ends.append((knot_offsets[w] + n_w, geom["seg_r"][offsets[w + 1] - 1]))

        if self._declared_junctions is None:
            rep = coincident_groups([p for _node, p in ends], _JUNCTION_TOL)
            for i, (node_i, _p) in enumerate(ends):
                node_of[node_i] = ends[rep[i]][0]
        else:
            # `ends` is indexed 2w / 2w+1 for wire w's start / end, which is
            # the same (wire, end) vocabulary `junctions=` speaks.
            for g in canonical_groups(self._declared_junctions):
                members = [2 * w + (0 if e == "start" else 1) for w, e in g]
                head = ends[members[0]][0]
                for m in members:
                    node_of[ends[m][0]] = head

        # Nearly-coincident ends are REFUSED, not quietly disconnected.
        # Two ends further apart than `_JUNCTION_TOL` are separate nodes, so
        # their charge cells are separate cells and the current that should
        # cross the junction has nowhere to go — this class's own §9.1
        # failure mode, worth a measured 7.9% at N=96 on a dipole split with
        # a 5e-8 m gap, and GROWING with refinement. A caller cannot even
        # work around it, because `junctions=` is refused here.
        #
        # The window is real geometry, not paranoia: the deck front end
        # fuses endpoints onto a 1e-6 grid (`deck/_polylines._NODE_EPS`),
        # a thousand times looser than the grouping tolerance, so a
        # hand-assembled or transformed model can easily land inside it.
        # Ends that ended up as one node are past this question, however they
        # got there -- joined by the tolerance, or joined because the caller
        # said so. Only ends still SEPARATE can fall in the bad window.
        for i, (node_i, p) in enumerate(ends):
            for node_j, q in ends[:i]:
                if node_of[node_i] == node_of[node_j]:
                    continue
                d = float(np.linalg.norm(p - q))
                if _JUNCTION_TOL < d <= _NEAR_COINCIDENT_TOL:
                    raise ValueError(
                        f"two wire ends are {d:.3g} m apart — closer than the "
                        f"{_NEAR_COINCIDENT_TOL:g} m the deck layer calls one "
                        f"node, but further than the {_JUNCTION_TOL:g} m this "
                        f"formulation joins at. They would become SEPARATE "
                        f"charge cells and the junction current would have "
                        f"nowhere to cross, which is a first-order error that "
                        f"grows as the mesh refines. Make the two ends exactly "
                        f"equal, or pass `junctions=` naming them as one node "
                        f"(momwire#590)"
                    )

        _uniq, cell = np.unique(node_of, return_inverse=True)

        left_node = np.empty(n_seg, dtype=np.int64)
        right_node = np.empty(n_seg, dtype=np.int64)
        for w in range(n_wires):
            lo, hi = offsets[w], offsets[w + 1]
            base = knot_offsets[w]
            left_node[lo:hi] = cell[base + np.arange(hi - lo)]
            right_node[lo:hi] = cell[base + np.arange(hi - lo) + 1]

        pieces = _pieces_of(geom)
        cell_of_piece = np.empty(2 * n_seg, dtype=np.int64)
        cell_of_piece[0::2], cell_of_piece[1::2] = left_node, right_node

        self._cached_cells = (left_node, right_node, pieces, cell_of_piece)
        return self._cached_cells

    def _mirror_pieces(self, geom, src, pieces):
        """`pieces` rebuilt on whichever source geometry `src` describes.

        The image call hands `_charge_stencil` a mirrored `src`; the cell
        pieces have to move with it while the NODE grouping — which is
        connectivity, not position — stays as the real geometry defined it.
        Mirroring is an isometry, so coincident ends stay coincident and
        the map is still valid. Rebuilt from `src` rather than reflected
        here so this class contains no mirror arithmetic of its own, which
        is the parent row's ground contract and the reason the three
        ground models need no code in this file.
        """
        if src is geom or src["seg_l"] is geom["seg_l"]:
            return pieces
        return _pieces_of(src)

    def _charge_stencil(self, geom, src, k):
        """Harrington (103)'s charge term: ``(Ψ_right − Ψ_left) · D``.

        The parent fills this as four evaluations of the reduced kernel
        between segment ENDPOINTS. Here the four terms are cell averages:

            Ψ[p, j] = (1/L_j) Σ_{pieces of node j} ∫_{piece} g(p, r') dl'

        one row per observation point p, one column per node j, `L_j` the
        node's total cell length. A free wire end has one piece and a
        half-length cell — the charge stays ON the conductor, which is
        worth a steady ~11% (momwire#557) and reads Harrington's own Fig. 6
        footnote about "the extra 1/2 interval at each wire end".

        The contraction is the ``Dᵀ Ψ D`` form `docs/pulse_basis_d0_nodal
        _charge.md` wrote down in June: `D[j,n] = +1` when node j is
        segment n's arc-h end, `−1` when it is the arc-0 end. Every basis
        sharing a node pours its charge into the SAME cell, which is what
        makes the node's net charge the Kirchhoff sum and what the
        rejected cheaper cells got wrong.

        Observers are the real segment endpoints throughout, exactly as on
        the parent row — only sources move under the image — so the result
        is (n_seg, n_seg) in the parent's own convention and `w_Phi`, the
        image fold and the Sommerfeld remainder path all consume it
        unchanged.
        """
        left_node, right_node, pieces, cell_of_piece = self._node_map(geom)
        pieces = self._mirror_pieces(geom, src, pieces)
        n = geom["h_per_seg"].size
        n_cells = int(cell_of_piece.max()) + 1

        # One moment fill for both endpoint sets, as the parent stacks its
        # two kernel blocks: rows 0..n-1 observe at arc-0 ends, n..2n-1 at
        # arc-h ends.
        obs = np.vstack([geom["seg_l"], geom["seg_r"]])
        m0 = self._seg_M0(obs, pieces, k)

        cell_len = np.bincount(
            cell_of_piece, weights=pieces["h_per_seg"], minlength=n_cells
        )
        psi = np.zeros((2 * n, n_cells), dtype=np.complex128)
        np.add.at(psi.T, cell_of_piece, m0.T)
        psi /= cell_len[None, :]

        # D has exactly two nonzeros per column — +1 at the segment's right
        # node, −1 at its left — so materialising it and calling BLAS is an
        # O(n³) way to spend an O(n²) gather. Bit-identical (each output is
        # still one subtraction of two accumulated columns), measured
        # 49 → 21 ms at N = 800 with ~15 MB of transients not allocated.
        diff = psi[n:] - psi[:n]
        return diff[:, right_node] - diff[:, left_node]
