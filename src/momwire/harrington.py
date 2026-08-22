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

from ._capabilities import Capabilities
from .pulse import _OUT_OF_SCOPE, _PER_WIRE_RADIUS_REFUSAL, PulseSolver

# Absolute tolerance for calling two wire ends one node. Deliberately the
# same 1e-9 `razor._JUNCTION_TOL` uses, and carrying the same caveat: the
# deck front end fuses endpoints on a 1e-6 grid (`deck/_polylines._NODE_EPS`),
# a thousand times looser and a different algorithm, so this agrees with the
# rest of the tree by convention rather than by construction. See razor's
# `_find_junctions` for the full account of the six tolerances in this tree.
_JUNCTION_TOL = 1e-9

# `junctions=` is refused here for a DIFFERENT reason than on the parent
# row, so the sentence cannot be inherited: the parent has nothing to
# detect, this row detects it itself.
_JUNCTIONS_REFUSAL = (
    "HarringtonSolver takes no junction spec: coincident wire ends are "
    "found from the geometry (within 1e-9 m) and become one charge cell "
    "spanning every incident half-segment, so there is nothing to declare"
)


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
            # The extended-kernel refusal is inherited verbatim and stays
            # TRUE for a different reason than the parent states: the
            # charge is no longer a point, so its potential no longer needs
            # the a² floor to exist — but the vector term and every cell
            # moment below are still written against the reduced kernel.
            "extended_kernel": _OUT_OF_SCOPE["extended_kernel"],
            "per_wire_radius": _PER_WIRE_RADIUS_REFUSAL,
        },
    )

    def __init__(self, **kwargs):
        if "junctions" in kwargs:
            raise NotImplementedError(_JUNCTIONS_REFUSAL)
        super().__init__(**kwargs)
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

        def find(i):  # union-find, flattening as it goes
            while node_of[i] != i:
                node_of[i] = node_of[node_of[i]]
                i = node_of[i]
            return i

        # Merge coincident WIRE ENDS. Each wire contributes its two end
        # knots; a group is formed by first match, razor's rule.
        ends = []
        for w in range(n_wires):
            n_w = offsets[w + 1] - offsets[w]
            ends.append((knot_offsets[w], geom["seg_l"][offsets[w]]))
            ends.append((knot_offsets[w] + n_w, geom["seg_r"][offsets[w + 1] - 1]))
        for i, (node_i, p) in enumerate(ends):
            for node_j, q in ends[:i]:
                if float(np.linalg.norm(p - q)) <= _JUNCTION_TOL:
                    ri, rj = find(node_i), find(node_j)
                    if ri != rj:
                        node_of[ri] = rj
                    break

        raw = np.array([find(i) for i in range(total)])
        _uniq, cell = np.unique(raw, return_inverse=True)

        left_node = np.empty(n_seg, dtype=np.int64)
        right_node = np.empty(n_seg, dtype=np.int64)
        for w in range(n_wires):
            lo, hi = offsets[w], offsets[w + 1]
            base = knot_offsets[w]
            left_node[lo:hi] = cell[base + np.arange(hi - lo)]
            right_node[lo:hi] = cell[base + np.arange(hi - lo) + 1]

        # Two half-segment pieces per segment, in the `_seg_M0` vocabulary.
        h = geom["h_per_seg"]
        tan = geom["tangents"]
        half = 0.5 * h
        mid = geom["seg_l"] + half[:, None] * tan
        piece_l = np.empty((2 * n_seg, 3))
        piece_l[0::2], piece_l[1::2] = geom["seg_l"], mid
        pieces = {
            "seg_l": piece_l,
            "tangents": np.repeat(tan, 2, axis=0),
            "h_per_seg": np.repeat(half, 2),
        }
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
        half = 0.5 * src["h_per_seg"]
        mid = src["seg_l"] + half[:, None] * src["tangents"]
        piece_l = np.empty((2 * half.size, 3))
        piece_l[0::2], piece_l[1::2] = src["seg_l"], mid
        return {
            "seg_l": piece_l,
            "tangents": np.repeat(src["tangents"], 2, axis=0),
            "h_per_seg": np.repeat(half, 2),
        }

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

        d = np.zeros((n_cells, n), dtype=np.float64)
        d[right_node, np.arange(n)] += 1.0
        d[left_node, np.arange(n)] -= 1.0
        return (psi[n:] - psi[:n]) @ d
