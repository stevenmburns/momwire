# momwire#1149 U3 (loading on crossing decks) and momwire#1152

2026-09-22. Worktree `feat/razor-buried-u3` off origin/main 999f7b7; fresh
3.12 venv, `make build`; `momwire.__file__` confirmed in the worktree.
[M] measured, [D] derived, [I] inferred.

## 1. U3: the crossing tent's loading [D, then M]

Razor's loading term is `L[m, n] = ∫_{P_m} Z_s Λ_n dl`. The question was
whether the full-geometry stencil on the WHOLE crossing tent is right when
the fill splits that tent into two half tents and chops its path at the knot.

It is, and no special case is needed:

1. The term has no kernel. It is a sum of per-segment closed forms (3h/8,
   h/8) keyed on `wing_seg / wing_rise / wing_sigma / seg_h` alone.
2. A crossing node is a knot, so no segment straddles the plane; each
   segment is wholly in one medium.
3. The crossing row's path is `c_A → knot → c_B`; the fill sums its two
   halves too. On each half the current is the whole tent's, because only
   one half tent is nonzero on each segment. So each half-path term is the
   stencil's per-segment entry of the unsplit tent.
4. A bare conductor's `Z_s` is its internal impedance: the metal and the
   surface field, not the exterior medium. The U2 node charge and the T2
   chop are charge-term facts, and this term has no charge term.

Corollaries, both measured:

- The per-medium sub-geometries' stencils (half tent + sigma=0 ghost),
  mapped back through the basis map, equal the full stencil **identically**
  (0.0 on crossing / rise-2 rod / hub4, probe 2). So "apply the stencil on
  the per-medium sub-geometries" is NOT a red control. The brief listed it as
  one; it cannot fail.
- A lumped load at the crossing knot is one diagonal entry of the crossing
  tent, so it equals razor's own 2-port algebra over a port at that knot to
  rounding (probe 4: ≤ 2.9e-13 Ω).

Implementation: `_assemble_Z_from_prepared` sends crossing and detached decks
through the same branch, `_assemble_Z_crossing` and then `_apply_loading`
once on the full geometry. The detached code path is unchanged.

## 2. Jacketed wires in soil: there was NO existing refusal [M]

The brief said "find the existing refusal and keep it declared". None exists
(probe `u3_probe_jacket.py`, not kept). On origin/main:

| deck | razor | bspline |
|---|---|---|
| wholly-below, jacketed | served | served |
| detached, buried wire jacketed | served | served |
| crossing, jacketed | refused (by the blanket loading refusal) | served |

So the only thing that refused a buried jacket was the blanket crossing
loading refusal U3 removes. What I did: narrowed it to
`_CROSSING_BURIED_JACKET_REFUSAL`, a jacket on a BELOW wire of a crossing
deck, declared as `crossing_junction+insulation`. A jacket on an above wire
is served, and converges onto bspline (probe 6). Nothing is newly refused on
any other route.

Why refuse at all [D, not measured]: `insulation_inductance` is the
thin-sheath term against a free-space exterior, `(1 − 1/ε_r) ln(b/a)`, and
the a′ kernel radius uses the same reading. The same argument with an
exterior medium ε̃ gives `(1 − ε̃/ε_r)`, which reverses sign for ordinary
soil at HF. **Open question for the lead:** razor's below and detached routes
and every bspline buried route serve a buried jacket with the free-space term
today. That looks like a tree-wide physics gap, and it is outside U3's scope.

## 3. U3 gates [M]

Copper-class σ = 3.5e7 S/m, every edge refined, feed on a knot, razor
`nec5_quadrature=True`.

Loaded − unloaded − full stencil (probe 2):

| deck | residual | stencil max | Z max |
|---|---|---|---|
| crossing_deck(1) | 3.4e-12 | 0.076 | 5.3e4 |
| rise/2 rod | 2.6e-11 | 0.64 | 2.1e5 |
| hub_deck(4) | 4.7e-13 | 0.10 | 7.3e3 |

Loading shift, |shift_razor − shift_bspline| in Ω (probe 3):

| deck | x1 | x2 | x4 | x8 | red: buried loading dropped | red: tent entries dropped |
|---|---|---|---|---|---|---|
| crossing | 0.0257 | 0.0145 | 0.0084 | 0.0050 | 0.263 → 0.253 (flat) | 0.042 → 0.0068 (converges) |
| hub4 | 0.0414 | 0.0227 | 0.0128 | 0.0074 | 0.411 → 0.418 (flat) | 0.278 → 0.036 (converges) |
| rod rise/2 | 0.108 | 0.060 | 0.034 | 0.019 | 2.14 → 2.09 (flat) | 0.135 → 0.022 (converges) |

- Dropping the crossing tent's own entries removes O(h_node) of path, so it
  converges as fast as the real thing and is **not a red control for a
  convergence bar**. It is caught by the exact-stencil gate, and by the
  lumped load at the knot, where the tent's share is the whole load:
  84 Ω missing, gated in `test_without_the_crossing_tents_entries_the_node_load_is_lost`.
- Lumped 50+20j Ω at the crossing knot vs bspline's 2-port algebra
  (probe 4): 1.46 → 0.79 → 0.45 → 0.26 Ω on an 84 Ω shift.
- Jacket on the above wire (b = 2 mm, ε_r 3) vs bspline (probe 6):
  0.49 → 0.30 → 0.19 → 0.12 Ω on a 37 Ω shift (ratios 0.62–0.65).
- Loading off is bit-identical to origin/main (probe 5, main's razor.py
  imported beside the branch's): crossing, hub4 and the rod unloaded, and
  detached (unloaded, σ, lumped), wholly-below σ and free-space σ — all
  `array_equal`.
- Reciprocity with loading on decays ≥ 3× per doubling, and a swept solve
  equals single-frequency solves (in the fast lane).

NEC-5 equal-mesh twin: not run for U3. No gate needs it. `LD 5` on a buried
wire is a dialect question that needs its own probe before it could be read.

## Probe index

| file | what |
|---|---|
| `probe1_collapse_1152.py` | #1152: per-block ε̃ = 1 collapse on U1's detached deck and detached_hub |
| `probe2_stencil_identity.py` (+log) | U3 stencil identity, per-medium = full |
| `probe3_loading_convergence.py` (+log) | U3 shift ladder vs bspline, two controls |
| `probe4_lumped_at_node.py` (+log) | lumped load at the crossing knot |
| `probe5_bit_identity.py` (+log) | loading off vs origin/main's razor.py |
| `probe6_above_jacket.py` (+log) | above jacket served and converging; buried jacket pre-flight |
