# momwire#1029 phase 0: feasibility and the cost model — results

- **Registration:** `PLAN.md` at 97e0c13, before any feasibility measurement.
- **Records:**
  - `feasibility.jsonl`, the registered run;
  - `structure.jsonl`, the pre-registration probe (no Z);
  - `cost_model.json`, over the fea1ad0 profiles;
  - `context_entry_eps.json`, the unregistered per-entry context.

## Inputs

- momwire `src` at 227491d. The registration commit changes no source.
- antennaknobs `src` at 97ca2b6b0, read through `PYTHONPATH`.
- `verticals.buried_radial_vertical` with the design's defaults and `n_radials`
  set; bs2 degree 2; soil `("finite", 13.0, 0.005)`; OMP / OpenBLAS / MKL = 4.
- Run under `systemd-run --user --scope -p MemoryMax=24G` with `ulimit -v` at 25 GiB.
- **This box is Haswell-builder, not #1067's Skylake.** Nothing here is a timing
  claim: the extraction still fills dense, and no route exists.

## Predictions

| id | bar | 4 radials | 12 radials | verdict |
|---|---|---|---|---|
| F1a | ε_rr ≤ 1e-9 | 6.8e-19 | 2.2e-15 | **hit** |
| F1b | ε_rr ≤ 1e-5 | 6.8e-19 | 2.2e-15 | **hit** |
| F2 | ε_rM and ε_Mr ≤ 1e-9; feed and port exactly 0 on the radials; KCL columns identical | 2.0e-19 / 2.0e-19; 0 / 0; ε 0 | 5.8e-18 / 7.8e-18; 0 / 0; ε 0 | **hit** |
| F3 | Z_in ≤ 1e-7; currents ≤ 1e-6 (per-harmonic against dense) | 2.2e-13; 2.2e-13 | 5.3e-13; 5.5e-13 | **hit** |
| F4 | max over h ≠ 0 of ‖ĉ_h‖ / ‖ĉ₀‖ ≤ 1e-9 | 0.0 | 0.0 | **hit** |
| F5 | Z_in from the engine against the extracted dense solve ≤ 1e-10 | 0.0 | 0.0 | **hit** |

The stop rule was not triggered. **The premise holds, hub included.**

## The decks

| radials | unknowns | per radial (m) | on the axis (p) | KCL rows | rotation mismatch | dense Z_in Ω | cond K₀ |
|---:|---:|---:|---:|---|---:|---|---:|
| 4 | 255 | 55 | 35 | 1, the hub | 3.9e-16 m | 78.1321 + 46.3377j | 6.8e4 |
| 12 | 695 | 55 | 35 | 1, the hub | 4.4e-15 m | 52.6055 + 38.3242j | 1.2e5 |

- **The 12-radial dense Z_in matches #1067's momwire column** (52.61 + 38.32j), so
  this is the same deck.
- **ε's normalisation.** It is the worst shift difference over the block's largest
  entry. That is 2.22e4 Ω for the radial blocks, and 2.06e4 Ω for the radial–mast
  couplings. The largest absolute shift difference is 1.5e-14 Ω at 4 radials and
  4.8e-11 Ω at 12.

## Context (not registered): graded entry by entry

A whole-block normalisation could hide a small far-coupling entry that differs a
lot relative to itself. `context_entry_eps.py` grades every entry against its own
magnitude, \|X[s] − X[s+1]\| / max(\|X[s]\|, \|X[s+1]\|), over entries above
1e-9 × the block maximum:

| radials | block | worst relative | on an entry of | largest absolute difference Ω | ungraded (below the floor) |
|---:|---|---:|---:|---:|---:|
| 4 | radial–radial | 2.4e-11 | 2.3e-5 Ω | 1.5e-14 | 224 of 48,400 |
| 4 | radial rows × mast columns | 8.2e-13 | 8.2e-4 Ω | 4.1e-15 | 204 of 7,700 |
| 4 | mast rows × radial columns | 8.2e-13 | 8.2e-4 Ω | 4.1e-15 | 204 of 7,700 |
| 12 | radial–radial | 7.6e-10 | 2.8e-5 Ω | 4.8e-11 | 672 of 435,600 |
| 12 | radial rows × mast columns | 1.7e-9 | 2.1e-5 Ω | 1.2e-13 | 612 of 23,100 |
| 12 | mast rows × radial columns | 1.7e-9 | 2.1e-5 Ω | 1.6e-13 | 612 of 23,100 |

- **The worst case is roundoff on the smallest couplings.** The worst per-entry
  non-invariance is 1.7e-9, on an entry about 1e-9 of the block's largest. That is
  roundoff, not a Cartesian-keyed choice in the fill.
- **Not measured here:** whether that still holds at 48 radials and above, where
  size-dependent paths may switch on (`_aca.py:build_cluster_tree` appears in the
  150-radial profile). G1's 48-radial cell is where it would show.

## The hub and the harmonics

- **One KCL row,** the hub. Its entries are identical across sectors, so under the
  DFT it touches harmonic 0 only.
- **What the per-harmonic route does:**
  - it factored K₀ = [[Λ₀, √N B], [√N C_M, Z_MM]], of size m + p = 90;
  - it factored N − 1 blocks of 55;
  - it closed the hub row with `_solve_with_kcl`'s Schur step;
  - the result is equal to the dense solve to about 5e-13.
- **An on-axis feed puts exactly zero into harmonics h ≠ 0.** So for this class,
  Z_in, the currents and the far field need **K₀ alone**, a 90 × 90 solve, whatever
  N is. The other harmonics are needed only for a drive that is not
  rotation-invariant (§2 of `PLAN.md`).
- **The #980 D3 crossing node** sits on the axis. It is carried entirely inside
  Z_MM, B and C_M, and nothing about it had to change.

## Cost model

From `PLAN.md` §7. It is a model over fea1ad0's warm profiles, rescaled to #1067's
Skylake columns.

| radials | sector warm, predicted s | #1067 momwire warm s | #1067 3b75639 s | predicted / 3b75639 |
|---:|---|---:|---:|---|
| 48 | 1.18 – 2.54 | 15.28 | 2.82 | 0.42 – 0.90 |
| 120 | 3.13 – 11.12 | 84.54 | 18.67 | 0.17 – 0.60 |
| 150 | 5.54 – 21.0 | 162.02 | 30.14 | 0.18 – 0.70 |

- **Shrinks by (m + p) / (N m + p)** (1/30 at 48 radials, 1/95 at 150): the
  pair-proportional fill, 83 % of the warm solve. That is the remainder projection
  (56 % at 150), seg-seg moments, field-Galerkin and B-spline assembly, and the
  crossing cross block.
- **Shrinks to about nothing:** the solve, especially for an on-axis drive.
- **Unchanged:** the plan, the basis and the tables, 2.4–4.6 %.
- **Sets the range:** the thread waits, 7–8 %, which are what separate the low and
  high bounds.

**The feasibility result sharpens the model in one direction.** For the on-axis
drive this class always has, the solve term is (m + p)³ alone, not
N m³ + (m + p)³. The fill is unchanged by that: Λ₀ still needs sector 0's rows
against every sector.

## What phase 0 does not show

- **Any speed.** There is no sector fill; this run filled dense and permuted.
- **The far field,** which is in G1.
- **48 radials and above,** or any size-dependent fill path.
- **razor-2p and sinusoidal.**
- **The symmetry check** of `PLAN.md` §1 as code, and its refusal (G2).

---

# Phase 1: the route — results

Phase 0 above shows the premise. Phase 1 built the route on it; **`STATUS.md`
is the current truth** and `PLAN-phase1.md` carries the registration and five
amendments. The headline, so this file is not a dead end:

- **`rotational_symmetry=True` on `BSplineSolver`.** One fill of sector 0's
  rows plus the axial rows against every source, one (m + p) harmonic-0 solve
  whatever N is, coefficients back in the dense ordering. Default off, and off
  is bit-identical to merged main.
- **G1**: Z_in and the currents agree with the dense solve to 2e-13 / 5e-13 /
  9e-13 at 4 / 12 / 48 radials, against 1e-9 and 1e-8 bars. The far field
  agrees to 4.5e-13 of the pattern's scale.
- **G2**: the six registered refusals fire by name, at construction, one deck
  each — and `tests/test_rotational_symmetry_1029.py` pins every sentence.
- **G3**: 2.21x / 3.50x / 5.58x / 7.16x at 12 / 24 / 48 / 96 radials, with peak
  RSS 0.982-0.997 of dense (phase 1 buys time, not memory — registered). The
  150-radial rung raises under this box's 8 GB cap, on BOTH modes, in the
  crossing block that `rows=` cannot narrow.
- **What phase 0 said it did not show**, item by item: the speed is measured;
  the far field is measured; 48 radials and above is measured to 96; the check
  and its refusals are code with a gate. **razor-2p and sinusoidal are still
  out of scope.**
