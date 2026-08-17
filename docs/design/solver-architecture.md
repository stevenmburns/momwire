# Solver architecture: factoring the capability matrix

Design study for momwire#376. Surveyed 2026-08-17 against `main` at
`0acbb7e`/`e261c21`. **No code lands under this document** — it is the
investigation, the scoring, and a recommended staging.

---

## 0. Recommendation, in one page

Do **not** build a single generic assembly engine, and do **not** unify the
two formulations. Both are the naive readings of "factor the matrix" and both
are wrong for reasons this document establishes in §2 and §4.

Do this instead, in this order:

| stage | what | why now | risk |
|---|---|---|---|
| **0** | **Capability registry** — solvers declare what they serve; the refusal surfaces and the consumer allow-lists read the declaration | the matrix has already leaked into antennaknobs as hard-coded string tuples (§1.4); every later stage needs a test surface that says "this cell is claimed, so this gate runs" | trivial |
| **1** | **Ground composition, per formulation** — one `PotentialGround`, one `FieldGround`, each owning mirror map, weight tables, remainder prepare/replay | kills the 4× physics duplication (48 methods, §1.2) along the only seam where the physics genuinely is the same object | low |
| **2** | **Schedule layer** — banding, budgets, blocking, folding, `out=`/`scale`, cancellation, extracted once and shared | the schedule is *already proven not to reach the arithmetic*, per solver, by gates that exist today (§3.1); that proof is exactly the migration acceptance test | low–medium |
| **3** | **Kernel-algebra consolidation** within each formulation, with a small-N operator-algebra implementation as the correctness oracle | only safe once stage 2 has removed the scheduling noise and there is a second, independent implementation to check against | medium |
| **4** | **`zblock(I, J)` on the sinusoidal trunk** (opportunistic) | H-matrix / ACA / array-block / lattice-Floquet compose over *any* basis that offers the contract; today they are B-spline-only | medium |

The pilot that proves the composition claim is **the ground models on
`RazorSolver`** (§6): it is the sparsest row of the matrix, it has no
accelerator path to preserve, it is formulation-B so it consumes stage 1's
object unchanged, and — uniquely among candidate pilots — it has an
**external oracle**, since the reference engine it twins solves the same
grounds.

### 0.1 Decision record (2026-08-17)

The goal metric, stated by the maintainer, is **time-to-prototype**: the
matrix will expand, but how and when is unknown, and right now a new idea
costs *days* of work. The architecture succeeds when a throwaway idea is back
to costing *hours*. That reframes the scoring — composition is not an
aesthetic property, it is what makes an experiment cheap enough to discard.

Accepted acceptance criteria, in the maintainer's terms, mapped onto the
staging:

| # | criterion | maps to | note |
|---|---|---|---|
| 1 | **remove `SinusoidalGalerkinSolver`'s second complete implementation of the ground modes** | stage 1, field trunk | the *within-trunk* half of stage 1 — same formulation, so `FieldGround` serves both solvers with no representation question. The easiest of the set, and the first to attempt |
| 2 | **complete `RazorSolver` to production level** | stage 1 (potential trunk) + pilot | grounds via `PotentialGround` (the pilot as proposed), then the rest of razor's refusal row — wire loading, per-wire radii, node gaps — each a test of whether a capability lands through the shared layers or around them |
| 3 | **remove antennaknobs' separately curated knowledge of momwire capabilities** | stage 0 | the registry, plus an antennaknobs change consuming it — the definition of done is the deletion of `_GROUND_EPS_SOLVERS` / `_WIRE_LOADING_SOLVERS` and the prose in `_extended_kernel_refusal` |
| 4 | **wire loading on `SinusoidalGalerkinSolver`** (momwire#182) | independent | added 2026-08-17. Basis/testing-level, no ground interaction — can land any time. Load-bearing for criterion 3: `_WIRE_LOADING_SOLVERS` exists *solely because* of this hole, so filling it is what lets the tuple be deleted instead of ported into the registry as a special case |
| 5 | *(stretch)* **a new ground model lands once and serves every solver** | stages 1+2 | the extension test rather than a dedup test; not required for the criteria above to count. §6.1 picks the target |

What this decision *defers*: stages 2–4 as programmes of their own. The
schedule layer (stage 2) is built only as far as criteria 1–2 force it;
`zblock` everywhere (stage 4) and the kernel-algebra consolidation (stage 3)
wait until the matrix actually expands. §9's first question is thereby
answered: the slate is not committed, so the correct response is the
front half of the staging, driven by the four criteria.

momwire#388 remains relevant as the *shape* of future expansion — its
plane-wave drive is still a cheap additional test if the stretch (criterion 5) is wanted
— but it is no longer the forcing function.

### 0.2 Step-size review (2026-08-17)

Asked directly — is this too big, too small, and is the end state the right
one — three honest qualifications, recorded so the criteria are judged
against what the architecture can actually deliver.

**Criterion 1 will not empty the file, and should not be judged as if it
could.** SG's ground duplication is entangled with scheduling: the Sommerfeld
fold order (`c2·img − rem` stays associated, §3.4) is *why*
`_fold_ground_block` and `_tested_sommerfeld_remainder` exist as they do. The
realistic end state is: physics and composition move to `FieldGround`, a thin
schedule shim (~100 lines) stays per solver. The measurable acceptance test
is therefore NOT "the file got smaller" but:

> **a new coefficient-level ground requires zero edits to
> `sinusoidal_galerkin.py`** (nor to `sinusoidal.py`) —

with the radial screen (§6.1) as its instrument. The same restated test for
criterion 3: a solver absent from every deleted tuple still runs end-to-end.

**"Lands once" means: coefficient layer once, composition twice.** Under the
two-object ground layer, a structurally new ground (another Sommerfeld-class
model, with its own remainder operator) lands in both `FieldGround` and
`PotentialGround`. Only coefficient-level grounds — those that modify the
reflection coefficients, like the radial screen — land literally once, in
`_ground_refl`. 4× → 1 for the common case, 4× → 2 for the hard case.
Criterion 5 as worded is achievable only for the first class, which the
chosen target is.

**The throwaway tier is the metric, and it is stage 0's real deliverable.**
The historically dominant experiment class here is a new basis or testing
scheme (the seven-basis roster, razor, the Galerkin fork) — and none of
criteria 1–5 directly buys "new basis in hours." But the hours-mode nearly
exists already, and `RazorSolver` is the existence proof: it shipped
refusing junction ports, node gaps, per-wire radii, every ground and EK —
gracefully — and was useful from day one. What breaks the pattern today is
only that antennaknobs' hard-coded tuples punish any solver not on the list.
So stage 0's definition of done is stated as a test:

> **a ~200-line free-space, point-matched prototype solver, declaring only
> what it serves, runs end-to-end in antennaknobs with graceful refusals —
> written and running in hours.**

The registry must stay small enough that this is true: roughly one declared
class attribute and one refusal helper, on the `_ElementCurrents` model
(120 LOC, thin adapter, no ceremony). If stage 0 grows validation machinery
or a plugin system, it has failed even if every criterion passes.

**Deliberately deferred, restated for completeness:** streaming stays
quadruplicated until the matrix expansion forces stage 2, so a *production*
new basis (budgets, EK, grounds) still costs days. The throwaway tier is what
keeps that from blocking experiments; stage 2 is what would fix it for
production, and it waits by decision.

---

## 1. What the matrix actually is

### 1.1 The axes — there are seven, not three

momwire#376 names basis × testing × ground, plus kernel variant. The survey
finds three more, and two of them are the cheapest wins on the board:

| axis | values today | factored? |
|---|---|---|
| **formulation** | direct-field, mixed-potential | *not an axis to factor* — see §2 |
| **basis** | sinusoidal 3-term, B-spline d=1, B-spline d=2, tent, (+singular enrichment DOFs) | no |
| **testing** | point matching, Galerkin, razor-blade path | no |
| **kernel variant** | reduced, extended (EK) | no — 9 of the 27 accel entry points exist only to carry this one flag |
| **environment** | free space, PEC image, refl-coef, Sommerfeld | no — the 4× duplication #376 names |
| **drive** (RHS) | delta-gap, multi-feed, distributed port, junction port | no — three independent implementations |
| **readout** | element currents, port currents, swept ports | **yes** — see §1.3 |
| **acceleration structure** | dense, H-matrix/ACA, array-block, lattice-Floquet | no — bound to one basis |

Adding **drive** and **readout** to the picture matters because momwire#388's
top-priority gap — plane-wave excitation, `EX 1–3` — is a *drive*, not a
kernel. It touches no basis, no testing scheme and no ground. Under a properly
factored architecture it lands once and serves every solver; today it would be
written three times. #388 already nominates it as "the cheapest possible pilot
for the B×T ⇒ zero-work property", and this study agrees with that reading —
though §6 argues the *ground on razor* pilot proves strictly more.

### 1.2 Inventory: the matrix today

Per-file, counted by method-name family on `main` (`^    def` matching the
ground families `image|refl|somm|ground|specular|mirror`, and the streaming
families `band|chunk|block|stream|prepare|replay|_fill|budget|row_bytes`):

| file | LOC | methods | ground | streaming | trunk |
|---|---|---|---|---|---|
| `bspline.py` | 4,635 | 65 | 20 | 11 | mixed-potential |
| `sinusoidal.py` | 4,305 | 61 | 9 | 4 | direct-field |
| `sinusoidal_galerkin.py` | 3,427 | 48 | 9 | 8 | direct-field |
| `hmatrix.py` | 1,919 | 41 | 10 | 10 | mixed-potential |
| `array_block.py` | 1,484 | 35 | 0 | 3 | mixed-potential |
| `razor.py` | 1,101 | 19 | 0 | 4 | mixed-potential |
| **total** | **16,871** | **269** | **48** | **40** | |

**48 methods implementing four ground interactions across four files.** Forty
more implementing the streaming and budget discipline around them. That is the
duplication #376 is about, and the counts are the scoring baseline.

The C++ layer says the same thing more sharply. There are **27 pybind entry
points**, and their names *are* the Cartesian product:

```
seg_seg_full_moments_bspline          × {_ek} × {_swept}        = 4
seg_seg_reg_moments_bspline_swept     × {_ek}                   = 2
seg_seg_static_moments_bspline_uniform× {_ek}                   = 2
bspline_assemble_offedge_block        × {_ek} × {_refl}         = 4
assemble_Z_bspline                    × {_weighted} × {_windowed} × {_swept} = 5
sinusoidal_field_tensor               × {_ek} × {_refl}         = 4
sinusoidal_galerkin_far_fill          × {_ek}                   = 2
assemble_Z_enrich, sommerfeld_remainder_bspline_Q,
remainder_field_proj_batch, somm_six_integrals_batch            = 4
```

Every new axis value multiplies this list. §3.2 explains why that is much less
alarming than it looks.

### 1.3 What is already factored — the in-tree existence proofs

This is not a greenfield argument. momwire has already factored three
capabilities across all three trunks, and they work:

* **`_ElementCurrents`** (120 LOC) — the current readout. Every solver family
  inherits it; antennaknobs' far-field, near-field and pattern code consumes
  one contract regardless of basis. This is a complete, shipped instance of
  the property #376 wants.
* **`_SweptPortSolutions` / `PortSolution`** (155 LOC) — the multi-port solve
  result, shared by the B-spline and sinusoidal trunks. `RazorSolver` carries
  its own `compute_*_swept` instead, which is itself a small instance of the
  duplication this document is about.
* **`_Cancelable`** (79 LOC) — cancellation, on everything.
* **`_ground_refl.py` (287) / `_sommerfeld.py` (1,050)** — the ground
  *coefficient physics* is already one implementation. What is quadruplicated
  is its **assembly**, not its physics. This distinction is the whole of
  stage 1's scope.

The pattern is established. The question is whether it extends to the fill,
where the memory budgets and the bit-frozen oracles live.

### 1.4 The matrix has already leaked out of momwire

`antennaknobs/engines/momwire.py` carries hand-maintained tuples of solver
*class names*:

```python
_GROUND_EPS_SOLVERS   = ("BSplineSolver", "HMatrixSolver", "ArrayBlockSolver",
                         "SinusoidalSolver", "SinusoidalGalerkinSolver")
_WIRE_LOADING_SOLVERS = ("BSplineSolver", "HMatrixSolver", "ArrayBlockSolver",
                         "SinusoidalSolver")   # Galerkin DELIBERATELY absent
```

plus `_extended_kernel_refusal()`, which encodes one solver-kwarg
incompatibility as prose. A consumer is maintaining momwire's capability
matrix by hand, and a hole in that matrix — `SinusoidalGalerkinSolver` has no
wire loading (`NotImplementedError`, momwire#182) — ships today as a
warn-and-drop in the consumer.

Known holes, from the refusal surfaces:

| solver | refuses |
|---|---|
| `SinusoidalGalerkinSolver` | wire loading; junction ports under finite ground / mixed radii |
| `SinusoidalSolver` | junction ports; `feed_model="point"` |
| `RazorSolver` | junction ports, node gaps, per-wire radii, every ground, EK |
| `BSplineSolver` + enrichment | EK, per-wire radii |

Stage 0 exists to make this table *the code* rather than a survey.

---

## 2. The finding that constrains every design: momwire has two formulations

momwire#376's math section says the ground enters `K` alone and depends on
neither basis nor testing scheme. That is true, and it is not sufficient,
because **`K` does not have one representation in this codebase**.

### 2.1 The two trunks compute different objects

**Direct-field** (`sinusoidal.py`, `sinusoidal_galerkin.py`):

```
Z_mn = ⟨t_m, E[b_n]⟩
```

The kernel primitive is the **field** of three canonical source-current shapes
(const, sin, cos) on a segment, in closed form — NEC's Eqs 76–79. The kernel
returns six field tables plus the geometry factors that project them
(`td`, `rho_proj_factor`, `rho_vec`, `rho_eval`).

**Mixed-potential** (`bspline.py`, `hmatrix.py`, `array_block.py`,
`razor.py`):

```
Z_mn = jωμ (t_m·t_n) ⟨f_m, G f_n⟩ + (1/jωε) ⟨f′_m, G f′_n⟩
```

The kernel primitive is a **moment of the scalar Green's function** — B-spline
takes it as an analytic segment-segment double moment `J_pq[i,j]`, razor as a
point-to-segment potential moment at an observation set.

### 2.2 Therefore the ground does not have one type

This is the load-bearing consequence:

| | direct-field | mixed-potential |
|---|---|---|
| image | mirrored **source geometry** into the field evaluator | mirrored geometry into the moment builder, **image sign fused into the `td_all` table** |
| refl-coef weight | a **projector** on `(E_ρ, E_z)` — the Fresnel dyad, applied per pair after the kernel | **two scalars** `(w_A, w_Φ)` — the dyad pre-resolved into the potential decomposition |
| Sommerfeld | `c2·Φ_img − S`, composed on projected field tables | `c2` as constant `(w_A, w_Φ)` tables + an additive Galerkin remainder block |
| approximation knobs | none | `ground_phi_mode` — *no analogue in the field form* |

A single `GroundComposition` interface spanning both would have two disjoint
implementations and no shared code: it would cost a layer and buy nothing. The
`ground_phi_mode` row is the tell — the potential form has a modelling choice
that the field form cannot express, because the field form never separates the
charge term.

**So stage 1 builds two objects, not one.** That is not a failure of the
factorization; it is the factorization being honest about where the seam is.
Within each formulation the ground genuinely *is* one object, and that is where
all 48 methods live.

### 2.3 The primitive each formulation actually wants

Within the mixed-potential trunk there is a further, cleaner unification that
falls out of the survey:

* `razor.py` evaluates **potential moments of a source segment at arbitrary
  observation points** (segment centroids for the scalar term, testing-path
  quadrature points for the vector term).
* `bspline.py` evaluates the **outer quadrature of exactly that**, done
  analytically.

So *testing is a choice of observation-point set and outer weights over one
source-moment primitive*. The exception is the singular same-edge block, where
B-spline must integrate analytically in both variables and cannot be expressed
as an outer quadrature of point moments — which is precisely why
`_seg_seg_static_moments` and `_seg_seg_reg_moments` exist alongside the
off-edge builder. Any generic engine must carry that exception explicitly
rather than discover it.

The direct-field trunk already states the same structure in code:
`_tested_contribs(geom, k, ctx, projector, src_c, src_t, mirror, subtract_into)`
takes source geometry, a per-pair projector, and a fold destination — it is
within one argument of being formulation-A's generic fill.

---

## 3. What makes migration safe — and what makes it dangerous

The issue's central worry is that certified budgets and bit-frozen oracles
constrain migration. Three findings say the constraint is weaker than it looks,
and one says where the real danger is.

### 3.1 Schedule-invariance is already gated, per solver

The property a shared scheduler needs — *the schedule moves no float* — is not
a thing to be established. It is already asserted by tests that exist today:

| trunk | gate | strength |
|---|---|---|
| sinusoidal | `test_sinusoidal_banded_assembly_is_bit_equal` | **bit** |
| sinusoidal | `test_sinusoidal_sommerfeld_remainder_replay_is_bit_equal` | **bit** |
| Galerkin | `test_folded_ground_is_bit_equal_to_the_differenced_spelling` | **bit** |
| Galerkin | `test_streamed_remainder_is_bit_equal_at_every_chunk` | **bit** |
| Galerkin | G-D8a, G-D9a (narrowing/banding/near-block) | **bit** |
| B-spline | `test_bspline_chunked_dense_z_matches_tensor_path` | 1e-10 relative |
| B-spline | `test_bspline_chunked_ground_matches_tensor_path` | 1e-10 relative |

The B-spline trunk has *already accepted* reduction-order tolerance at its
window boundaries, because the C++ windowed assembly reassociates and the gate
pins the algebra rather than the compiler. So the migration bar is known and
differs per trunk: **bit-equality where it exists today, the existing tolerance
where it already does not.** That is the "checkout-dance or explicit
re-certification" #376 asks for, made concrete.

This is the strongest single argument for staging the schedule layer *before*
the kernel layer. Extracting the schedule changes when buffers are touched, not
what arithmetic runs; the gates above are exactly the acceptance test for that
class of change, and they are already green.

### 3.2 The accelerator layer is already templated

The 27 entry points are thin pybind wrappers over a much smaller set of
templated cores — `template<bool EK>`, `template<int D, bool WEIGHTED>`,
`template<bool REFL>`, `template<int D>`. The C++ side has already discovered
the right factorization: **axes as compile-time parameters, one wrapper per
instantiation.**

This substantially de-risks the hardest part of #376's option 2. The accel
story for a shared engine is a *dispatch table* keyed by
(formulation, kernel variant, weighting, layout) resolving to an existing
template instantiation — not a kernel rewrite. The bit-level properties those
kernels are gated on survive by construction, because the same instantiation is
called with the same arrays: G-C2, for instance, pins that an all-ineligible EK
call reproduces the reduced fill to the bit, and a dispatch table that reaches
the same instantiation cannot break it.

One caveat on that optimism. The templates cover the axes *within* a
formulation; there is no template parameter spanning field-tensor and
moment-tensor layouts, and there should not be. §2.2's two-object ground layer
is the Python-side reflection of a split the C++ side also has.

### 3.3 The perf slate is closed

momwire#338, #347, #355, #356, #357, #358 are all **CLOSED**. The streaming and
budget machinery is finished and stable across all four solvers. That changes
the scheduling calculus: sharing it no longer means chasing four moving
implementations, and the four implementations can be read as a *specification*
of what the shared one must do. There will not be a better moment.

### 3.4 The real hazards

* **Evaluation-order discipline does not survive abstraction by convention.**
  momwire#392 (2026-08-17) found the discipline missing inside a function whose
  own docstring claimed it, and #205 had already learned the same lesson. A
  shared engine must make the discipline structural — named intermediates as a
  contract of the kernel layer, gated once — or the refactor will
  re-scatter it.
* **The same-edge singular block is not an outer quadrature** (§2.3). A generic
  engine that assumes it is will be quietly wrong on the diagonal, which is the
  part of the matrix that sets the impedance.
* **`c2·img − rem` cannot be distributed.** `free − (c2·img − rem)` is not
  `(free − c2·img) + rem` in float64. The Sommerfeld composition constrains the
  fold order and therefore the scheduler's freedom; it is why that ground alone
  cannot ride `subtract_into`'s single minus sign today.
* **Feature interactions are not on the axes.** Junction ports × finite ground
  × mixed radii refuses today for formulation reasons, not implementation
  ones. A composition architecture will make it *look* available. The registry
  (stage 0) must be able to express "this cell is refused on physics grounds",
  or the refactor will promise cells it cannot fill.

---

## 4. Candidate architectures

Scored against: **composition** (does B×T become ~zero work?), **migration
cost**, **accel story**, **memory/runtime** (with #376's stated relaxation that
slight regressions are acceptable where they buy composition), and **#388
readiness** (does plane-wave drive land once? does a fifth ground land once?).

### A. Capability registry only

Solvers declare `capabilities()`; refusal surfaces and consumer allow-lists
read it. No math changes.

*Composition*: none. *Migration*: ~1 day. *Accel*: untouched. *Memory*: none.
*#388*: no help with the work, real help with the bookkeeping.

**Verdict: necessary, not sufficient. Fold into every other option as stage 0.**
It is the only change that fixes §1.4, and it gives every later stage its test
surface.

### B. Ground composition, per formulation (#376's option 1, corrected by §2.2)

`PotentialGround` and `FieldGround`, each owning the mirror map, the weight
tables, the remainder prepare/replay, and the composition order.

*Composition*: partial — kills the 4× physics duplication, leaves the 4×
streaming duplication. **But note the sleeper win**: `RazorSolver` is
formulation B and has *no* ground today. It consumes `PotentialGround`
unchanged and acquires PEC + refl-coef + Sommerfeld. That is a genuine
"capability lands on a solver that never implemented it" demonstration.
*Migration*: medium, per-solver, each step bit-checkable against today.
*Accel*: unchanged — weights are already the C++ arguments.
*Memory*: neutral. *#388*: a fifth ground (radial screens) lands twice instead
of four times.

**Verdict: the highest confidence-per-unit-risk item on the board. Stage 1.**

### C. Generic assembly engine (#376's option 2)

`fill(test_functional, source_basis, kernel_composition, schedule)` with
banding, streaming and budgets implemented once.

*Composition*: the goal state. *Migration*: highest — every certified budget
and frozen oracle is in the blast radius simultaneously. *Accel*: better than
feared (§3.2). *#388*: excellent.

**Verdict: right target, wrong unit of work.** Taken whole it puts the safe
half (scheduling) and the dangerous half (arithmetic) in one change, so a
bit-equality failure cannot be attributed. Take it as E.

### D. Operator algebra `Z = Q_test · G · P_src` (#376's option 3)

*As production*: rejected. The explicit intermediate `G` at quadrature
resolution is precisely what momwire#332 spent a release removing — the
Galerkin remainder tensor alone was `(3, N·n_qp, N)`, several times the matrix.
The relaxation permits *slight* regressions; this is not slight.

*As a reference implementation*: **adopt.** A small-N, no-streaming,
no-budget implementation of the algebra is a second, independent expression of
the same physics. Under bit-frozen gates the migration question is always "is
the new fill right, or merely different from the old fill?" — and today the
only answer available is "it matches the old fill". A reference path turns that
into a real check, and it is cheap because it need not be fast.

**Verdict: reject as production, adopt as oracle. It is what makes stage 3
safe.**

### E. Schedule / kernel two-layer split *(not in #376's list)*

Cut C along the seam §3.1 proves is safe:

* **Layer 2, schedule** — banding, blocking, byte budgets, fold destinations
  (`out=`/`scale`), cancellation checkpoints, chunk-boundary alignment.
  Formulation-*agnostic*: it decides *when* buffers are touched, never what
  arithmetic runs.
* **Layer 1, kernel algebra** — pair geometry, kernel variant, mirror map,
  ground weighting, composition order. Formulation-specific (two instances).

*Composition*: same endpoint as C. *Migration*: the risk is now **sequenced**
— layer 2 migrates per solver behind gates that are already green and that
cannot pass if the change touched arithmetic; layer 1 migrates afterwards
against the D oracle. *Accel*: dispatch table (§3.2). *Memory*: layer 2 is
where the budgets live, so it inherits them rather than re-deriving them.

**Verdict: this is how C should be built. Stages 2 and 3.**

### F. Block-evaluator core — `zblock(I, J)` everywhere *(not in #376's list)*

`HMatrixSolver.zblock(I, J)` is already a public block evaluator, and the
entire ACA / H-matrix / array-block / lattice-Floquet stack is built on it. It
exists on exactly one trunk.

*Composition*: a different and very concrete kind — **the acceleration-structure
axis composes over the basis axis.** The sinusoidal trunk would gain
O(N log N) scaling and the array/lattice machinery without any of that code
being written again. This is the one option that turns an *internal* refactor
into a *user-visible capability*.
*Migration*: medium. The contract is small and already specified.
*Risk*: block granularity. The sinusoidal fill's speed comes from whole-row
bands plus a sparse-M matmul; a block evaluator must be allowed to answer at
band granularity (block = row band × all columns) or it will be slow. That is
compatible — `_fill_block` and `_near_block` are already block-size choosers —
but it must be a design constraint, not a discovery.
*Bonus*: a block is exactly layer 2's unit of work, so E and F reinforce.

**Verdict: adopt opportunistically, stage 4. Highest visible payoff; not on the
critical path.**

### G. Formulation unification *(named in order to be rejected)*

Force everything to mixed-potential, or everything to direct-field.

The naive reading of "factor the matrix", and wrong. The direct-field trunk's
entire value is that it computes NEC's own closed-form fields — that is what
makes it the instrument for NEC agreement, and re-deriving it through
potentials would discard the property it exists to have. The mixed-potential
trunk's value is the analytic moment machinery and the singular same-edge
treatment. Unification discards one of the two.

**Verdict: reject. The two formulations are a feature — momwire's cross-check
between independent formulations is a large part of why its numbers are
trusted. §2.2's two-object ground layer is the correct concession to this.**

### Scoring

| | composition | migration | accel | memory | #388 | verdict |
|---|---|---|---|---|---|---|
| A registry | — | trivial | — | — | bookkeeping | **stage 0** |
| B ground/formulation | partial (+razor) | medium | none | neutral | 4×→2× | **stage 1** |
| C generic engine | full | very high | ok | neutral | excellent | build as E |
| D algebra (production) | full | high | poor | **fails** | — | reject |
| D′ algebra (oracle) | — | low | n/a | n/a | — | **adopt** |
| E schedule/kernel split | full | **sequenced** | ok | inherits | excellent | **stages 2–3** |
| F `zblock` everywhere | accel × basis | medium | good | neutral | neutral | **stage 4** |
| G unify formulations | full | extreme | — | — | — | **reject** |

---

## 5. Recommended architecture

```
                    ┌──────────────────────────────────────────┐
   stage 0          │  Capability registry                     │
                    │  declared cells + refusal reasons        │
                    └──────────────────────────────────────────┘
                    ┌──────────────────────────────────────────┐
   stage 2          │  Schedule layer  (formulation-agnostic)  │
                    │  bands · blocks · byte budgets · folds   │
                    │  out=/scale · cancellation · alignment   │
                    └──────────────────────────────────────────┘
        ┌───────────────────────────┬──────────────────────────┐
   1+3  │  FieldGround + field      │  PotentialGround +       │
        │  kernel algebra           │  potential kernel algebra│
        │  (sinusoidal trunk)       │  (bspline/razor trunk)   │
        └───────────────────────────┴──────────────────────────┘
                    ┌──────────────────────────────────────────┐
        (existing)  │  _ground_refl · _sommerfeld · _accel      │
                    │  coefficient physics · templated kernels │
                    └──────────────────────────────────────────┘
   stage 4:  zblock(I, J) contract spanning both trunks
             → H-matrix / ACA / array-block over any basis
```

Two kernel-algebra objects, one schedule, one registry, one oracle. The bottom
layer already exists and does not move. The `zblock` contract is a second entry
point into the same stack, not a parallel one.

---

## 6. Pilot proposal

**The ground models on `RazorSolver`.**

| criterion | why razor |
|---|---|
| proves the claim | razor implements **zero** ground methods today; if `PotentialGround` gives it PEC + refl-coef + Sommerfeld, the "capability lands on a solver that never wrote it" property is demonstrated, not argued |
| lowest blast radius | no accelerator path, no certified memory budget, 1,101 LOC, the sparsest row of the matrix |
| tests the real seam | razor's testing scheme (razor-blade path) is *different* from B-spline's Galerkin, so this is a genuine B×T cell, not a same-trunk copy |
| **external oracle** | razor is momwire's NEC-5 formulation twin, and NEC-5 solves these grounds; agreement is checkable against something that is not momwire (conclusions only — the licensed binary is the oracle, and its internals stay out of the repo). No other candidate pilot has this |
| bounded | if it fails, it fails in a solver that ships no ground today, so nothing regresses |

**Gates.** Tolerance-and-oracle, not bit-equality — there is no existing razor
ground to be bit-equal to. Specifically: (a) ε̃ → ∞ reproduces the PEC image
exactly and the PEC image reproduces the analytic image dipole; (b) ε̃ → 1
reproduces free space **bit-for-bit** (the ground layer must be a true no-op
when off — the same structural collapse `_folded_ek_delta_fields` gets from its
`a²` factor); (c) agreement with the reference engine on the ladder decks
already used for razor's free-space validation; (d) B-spline's grounded results
must be **bit-unchanged** by the extraction, which is the real regression gate.

**Second choice, if razor's ground turns out to need new physics rather than
new plumbing:** plane-wave excitation (`EX 1–3`, momwire#388 priority 1) as a
shared **drive** across all three trunks. It proves a different axis composes,
it is smaller, and it fills a real gap — but it proves less, because the drive
axis never touches the fill machinery that this document is mostly about.

### 6.1 The stretch test's target: which "new ground model"

Two candidates were examined (2026-08-17), and only one is a test.

**The MININEC-type ground is not a new ground model.** It is a name for a
composition this project already shipped and then superseded:
`refl-coef-ground-plan.md` records that before antennaknobs PR #251 the
engine folded every finite ground to *PEC-image solve + Fresnel far field —
"the EZNEC MININEC-style ground"* — and measured why it was replaced (~18–20 Ω
reactance error at 0.2λ heights). The `("finite", …)` far-field path still is
PEC image + Fresnel on the reflected component; only the solve was upgraded.
A named `"mininec"` option is therefore an antennaknobs-level composition of
existing pieces (PEC solve + existing Fresnel far field) — worth having for
EZNEC parity and for grounded verticals with a finite-ground pattern, roughly
an afternoon, and it touches the momwire fill **zero**. It proves nothing
about this architecture and must not be counted as the stretch test.

**The radial-wire screen ground is the real test** (`GN` with `NRADL`,
momwire#388 priority 4): a screen surface impedance in parallel with the
earth, modifying the reflection coefficients. It is common in real
installations (verticals over radials), it is oracle-checkable against NEC-2,
and its physics lands in `_ground_refl.py`'s coefficient layer — so under the
factored architecture it should reach every solver through the weight tables
with **zero per-solver assembly work**. If it needs per-solver work, the
architecture failed its goal metric. Until stage 1 exists it should NOT be
built (it would be paid through today's quadruplicated plumbing — the exact
cost the metric exists to escape); its role now is as a **design constraint**:
`FieldGround` / `PotentialGround` are designed against five grounds, not
four, so the interface cannot overfit to the existing set.

---

## 7. What this subsumes, and what it does not

**Subsumes.** momwire#338/#347/#355/#356/#357/#358 are closed, so nothing is
subsumed in the "replaces open work" sense. What stage 2 does is *absorb their
output*: the byte budgets (`_NEAR_WORKSPACE_BYTES`, `_FILL_WORKSPACE_BYTES`,
`_EK_BRACKET_BAND_BYTES`), the fold destinations, the prepare/replay split and
the chunk-boundary alignment rules become one implementation with one set of
gates instead of four. Their tests become the shared layer's tests.

**Does not subsume.** momwire#392's evaluation-order discipline is a *kernel
layer* property and stays a per-expression obligation; the architecture can
gate it once (§3.4) but cannot make it automatic. momwire#376 does not decide
momwire#375 or #385 (portal parallelism) — those are above this stack.
momwire#332's sinusoidal memory treatment is orthogonal and can land before or
after.

**Explicitly out of scope.** Any change to the formulations themselves (§4G),
any change to the C++ kernel arithmetic, any migration of a solver that is not
being touched for another reason.

---

## 8. Risks that would stop this

1. **Stage 1 finds the two grounds are three.** If the Galerkin ground turns
   out to need a third representation rather than sharing `FieldGround` with
   the point-matched solver, the physics-duplication win shrinks by half and
   stage 1 should be re-scoped to the B-spline trunk alone — where the target
   is sharper anyway, since `hmatrix` inherits `bspline`'s 20 ground methods
   and then adds 10 block-wise re-implementations of the same physics on top.
2. **Stage 2 cannot hold the Sommerfeld fold order** (§3.4). If the shared
   scheduler cannot express `c2·img − rem` without distributing it, Sommerfeld
   keeps a per-trunk path and the schedule layer serves the other three
   environments. Survivable, but it should be discovered in the first migrated
   solver, not the last.
3. **`zblock` at band granularity is not enough for ACA.** ACA wants small
   arbitrary blocks; if the sinusoidal fill cannot answer those at acceptable
   cost, stage 4 delivers H-matrix-over-sinusoidal at a worse constant than
   over B-spline, and the capability may not be worth its complexity.
4. **The maintenance argument cuts both ways.** momwire is ~21k LOC of Python
   and 6.7k of C++ whose value rests on bit-level trust against external
   oracles. A refactor that destabilises that trust costs more than the
   duplication does. This is the reason the staging front-loads changes that
   *cannot* alter arithmetic (stages 0 and 2) and defers the ones that can
   (stage 3) until there is an independent oracle (D′) to check them.

---

## 9. Open questions for the decision

*(1 and 3 were answered by the 2026-08-17 decision record, §0.1; kept here for
the reasoning.)*

1. **Is #388's slate actually going to be worked?** — *Answered: not committed;
   expansion is expected but its shape and timing are unknown. Therefore
   stages 0–1 driven by the acceptance criteria of §0.1, stages 2–4 deferred.*
2. **Does `SinusoidalGalerkinSolver` get wire loading?** — *Answered
   2026-08-17: yes — adopted as acceptance criterion 4. It is also what makes
   criterion 3 clean: `_WIRE_LOADING_SOLVERS` exists solely because of this
   hole.*
3. **Is `RazorSolver` meant to grow?** — *Answered: yes — completing razor to
   production level is acceptance criterion 2, so the pilot stands as
   proposed.*
4. **What is the tolerance policy for re-certified cells?** §3.1 establishes
   the per-trunk bar for *preserved* behaviour. For *new* cells (razor's
   ground) the bar has to be set against the external oracle, and that number
   should be agreed before the pilot rather than after it. Still open.
