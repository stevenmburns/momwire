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

### 6.2 Pilot status (2026-08-17)

**Unit 1** landed `PotentialGround` and the B-spline dense fill's
consumption of it. **Unit 2 landed the pilot's actual claim**:
`RazorSolver` — which implemented zero ground methods — now serves the PEC
image, consuming the object and writing no reflection of its own. What it
took, in full: mirrored sources into `_seg_moments_prepare`, the image sign
on razor's own `(3, n_basis)` tangent table, and one `Z_free − Z_image` at
the seam. **83 executable lines in `razor.py`** — 49 of fill, 12 of
constructor plumbing, and 22 spent *refusing* ground contact. No new
physics, no new kernel, no C++.

The one thing that had to change in the shared layer is the interesting
result. `ImageGeometry` had been shaped entirely by its first consumer
(mirrored segment endpoints plus a fused N² tangent-dot table, because that
is what the B-spline moment builder eats). Razor quadratures in each
segment's own arc frame and must never materialise anything N², so it needs
a mirrored (origin, tangent) pair and an O(N) tangent table from the *same*
mirror. Neither is the mirror, so the class now exposes the two operations
— `mirror_positions` / `mirror_tangents` — with both consumers' shapes as
lazy conveniences over them. **A second consumer forced exactly one
generalisation, one layer deep, and no new concept.** That is the shape of
evidence the goal metric wants.

Gates, as met:

| gate (§6) | result |
|---|---|
| (a) PEC reproduces the analytic image | **exact**, not approximate: on razor's own discretization a dipole over PEC and the explicit mirrored twin driven −1 V are the same linear algebra (`[[A,B],[B,A]]` against `[+e, −e]` reduces to `(A−B)c = e`, and `A−B` *is* the grounded matrix). Measured agreement 6.6e−14 relative |
| (b) free space bit-for-bit | **held**, on 6 deck shapes × 2 quadrature lanes × 3 entry points (`_assemble_Z`, `compute_impedance_swept`, `compute_y_matrix`). The ground layer is structurally absent when off — `prepared["image"] is None`, no mirrored cache built and then multiplied by zero |
| (c) agreement with the reference engine | **held with ≥ 2.7× margin** on the maintainer's sharp-lane bar (below) |
| (d) B-spline grounded results bit-unchanged | **held** across PEC / refl-coef / Sommerfeld / grounded-junction / EK, and on hmatrix and array_block |

The **sharp lane** is the maintainer's decision of 2026-08-17: razor with
`nec5_quadrature=True` over PEC, four ladder geometries × rungs N ≥ 24,
against the licensed binary's printed `GN 1` impedances —
|ΔZ| ≤ max(0.20 Ω, 0.25 %·|Z|) per rung, and the offset constant down each
ladder to 0.05 Ω. Worst measured: 0.078 Ω against a 0.211 Ω bar (fat
dipole, N = 96); worst constancy spread 0.0398 Ω of 0.05 (same deck, in X).
Every other geometry sits at 0.036–0.076 Ω. See
`tests/test_razor_pec_ground.py` and `scripts/capture_razor_pec_nec5_lane.py`.

**Deferred, deliberately, and not attempted here:** the production
(Gauss-Legendre) lane's bar against NEC-5, every finite ground on razor
(`ground_eps` / `ground_model` / `ground_phi_mode` stay refused), and
ground CONTACT — a wire end in the plane needs the image to continue its
current, which is a change to the tent basis rather than to the fill, so
razor refuses it and points at `BSplineSolver`'s momwire#151 fold. The
recorded GL deltas live in `tests/golden_razor_pec_nec5.py` so the deferred
bar has a starting point on file: ΔR stays inside 0.01–0.14 Ω on the three
open geometries but runs 0.39–0.93 Ω on the loop, and ΔX runs 0.12–1.37 Ω
(dipoles), 0.55–1.31 Ω (inverted-V) and 1.7–4.4 Ω (loop). The production
lane is therefore a *different* claim, not a slacker version of this one —
exactly what the twin's free-space story predicts, since converged
Gauss-Legendre is deliberately *not* NEC-5's rule.

**What unit 3 (finite grounds on razor) will hit first** is a schedule
question this unit could dodge. Razor's `_assemble_Z_prepare` /
`_assemble_Z_from_prepared` split is a k-independent / k-dependent split,
and the PEC mirror is entirely k-independent — so the image cache lives in
the prepare half and replays across a sweep for free. A
reflection-coefficient ground's `(w_A, w_Φ)` are ω-dependent: they cannot
live there, and the pairing-agnostic weight question unit 1 raised gets its
sharpest form here. Razor never forms an (N, N) pair table at all — its
weights would have to be applied per (row-chunk × source-segment) window
inside the k-dependent half, which is the `weight_windows` producer shape
and *not* the `weight_tables` one. That is evidence for the windows being
the trunk's primary weight surface and the whole-geometry tables being the
convenience, which is the reverse of how unit 1 ordered them.

### 6.3 Unit 3: ground CONTACT (2026-08-17)

The unit that actually landed in this slot is ground **contact**, not the
finite grounds §6.2 above expected — the paragraph above stands as written
for whichever unit takes the finite-ground bar. Contact was §6.2's own named
deferral ("a wire end in the plane needs the image to continue its current,
which is a change to the tent basis rather than to the fill"), and closing it
is what unlocks the vertical/monopole class.

**It is a basis change and nothing else — the fill is untouched.** A wire end
in the plane keeps a degree of freedom instead of being zeroed: its basis is
the junction tent between the wire and its own image (monopole + image *is* a
dipole), of which only the REAL wing is spelled, because `Z = Z_free −
Z_image` already evaluates every basis against the mirrored sources. The
image wing therefore arrives with the right shape, the right direction
(−M·t̂, parallel for a vertical contact) and the opposite charge for free.
In code that is: side A of the tent carries `sigma = 0` — which empties its
tangent (T1), its charge doublet (T2) and its half of the testing path in one
stroke — plus a two-line patch giving the grounded ROW the plane as its
potential reference (Φ = 0 there, so the T2 endpoint drops), plus the
grounded end becoming feedable. Measured, the same way unit 2's 83 was:
**59 executable lines added and 20 retired in `razor.py`, net +39**, of
which the geometry validation that replaces the old refusal is 34 (it now
has three refusals to spell instead of one) — leaving ~25 lines of actual
basis change, 2 of them in the fill. No new physics, no new kernel, and no
change to `PotentialGround` at all: the shared layer took a second
capability with **zero** edits, which is the stronger half of what unit 2
demonstrated by taking one edit.

The row's halving is the one genuine derivation. The grounded row tests the
REAL half-path only; the image half contributes the identical number
(E(M·r) = −M·E(r) for a system that is its own image), so halving is what
makes the feed voltage the BASE gap's rather than the equivalent dipole's
whole gap — i.e. what makes a monopole return Z_dipole/2 rather than
Z_dipole.

| gate | result |
|---|---|
| exact half-dipole oracle | **exact**, four geometry classes × both quadrature lanes: monopole vs 2L dipole, inverted-L vs the "Z", both-feet-grounded inverted-U vs the closed loop, and a grounded K=2 junction vs its explicit four-wire twin. Worst relative agreement 1.8e−13, best 5.7e−16 |
| NEC-5, sharp lane | **held with 10.5× margin** — against the binary's own MIRROR decks (see below): worst 0.0190 Ω of a 0.20 Ω bar, offset constancy 0.0004 Ω of 0.05 |
| free space bit-for-bit | **held**, 7 deck shapes × 2 lanes × 3 entry points, against the branch point |
| clear-of-plane PEC bit-for-bit | **held**, same shapes × 2 plane heights; `tests/golden_razor_pec_nec5.py` unmoved |

**The finding, and the one place this unit does not meet the bar as
originally stated.** Razor's grounded answer is its own mirror model halved,
exactly. NEC-5's grounded answer is *not* its own mirror model halved: run
the binary twice — once on the grounded deck (`GE 1` + `GN 1`, N segments)
and once on the same radiator plus its image in free space (2N segments,
centre-driven) — and the two disagree by −0.133+0.285j Ω at N=24 on the
monopole ladder, decaying to −0.097+0.074j at N=96 (inverted-L:
+0.102+0.064j → +0.010+0.026j). So against NEC-5's CONTACT deck razor's
residual *decays* instead of holding constant, and misses the constancy half
of the sharp bar (0.21 Ω spread in X on the monopole, 0.09 Ω in R on the
inverted-L) plus the per-rung half at the two coarsest monopole rungs (0.30
and 0.24 Ω against 0.20). Against NEC-5's MIRROR deck the twin property is
sharper than any clearance geometry unit 2 measured: a constant
+0.001..0.003 + 0.019j Ω, spread ≤ 0.0004 Ω down each ladder. The gated lane
is therefore the mirror-deck column, with the contact column recorded and
its decay pinned (`tests/test_razor_ground_contact.py`, tests 6-7). The
reading: this is a grounded-end *discretization* difference between the two
codes that vanishes with the mesh — both ladders are still walking at N=96,
this formulation's O(1/N) walk — and not a formulation error in either. What
it is on NEC-5's side is not knowable from printed output alone, and no
attempt was made to infer it.

**Scope held deliberately narrow.** A wire END may be in the plane, at any
number of ends per point (K real ends at a grounded point get K tents, one
each — the plane is one more branch, so there is no through-path to
distinguish and no KCL row to drop, which is the same physics
`BSplineSolver`'s momwire#151 spells as "grounded junctions keep their
directional bases and lose the closure row"). An interior anchor touching
down is refused (`NotImplementedError`) — it would need a second unknown at a
knot that already carries a tent, a second basis change; an edge lying in the
plane and a wire dipping below it stay `ValueError`s with `BSplineSolver`'s
own wording. Contact over a finite ground is refused twice over and says so:
the fold hard-codes image coefficient 1, so a grounded end over anything but
PEC would take spurious contact charge (momwire#282).

**What the finite-ground unit inherits from this one:** the grounded tent is
pure basis bookkeeping and lives entirely in the k-independent prepare half,
so it costs a reflection-coefficient ground nothing extra in schedule terms —
but momwire#282 is now load-bearing rather than advisory, since a contact
deck over refl-coef must either fix the image-coefficient assumption or keep
refusing.

### 6.4 Unit 4: the reflection-coefficient ground on razor (2026-08-17)

The finite-ground unit §6.2 predicted, landing where §6.3's contact unit sat
in the queue. `RazorSolver` now serves `ground_eps` for wires standing clear
of the plane, consuming `PotentialGround.weight_windows` and writing no
Fresnel coefficient of its own. `ground_model="sommerfeld"` and ground
CONTACT over a finite ground stay refused, the latter citing momwire#282 —
which §6.3 flagged as having become load-bearing, and which this unit
confirms: the grounded tent's lower wing IS the image, so a weighted image
does not repair it, and no amount of `(w_A, w_Φ)` is a fix for a wrong basis
function.

**§6.2's two predictions were both right, and the second one cost the unit
its one shared-layer edit.** The weights are ω-dependent and could not live
in the k-independent prepare half — confirmed, and the split is exactly
where §6.2 said it would be: `prepare` keeps the mirrored geometry and its
static moments plus (new) the observer POINTS the specular rays are drawn
to; `_assemble_Z_from_prepared` builds one ground object per solved
wavenumber and both weight producers from it. And razor's weights had to be
applied per (row-chunk × source-segment) window rather than as an (N, N)
table — confirmed, with nothing N²-scale allocated in the weights at any
point.

What that forced is the same shape of generalisation unit 2 forced on
`ImageGeometry`, one row down the §2.2 table. `weight_windows` had been
shaped by its first consumer too: "observer rows" meant rows of the
geometry's own square segment × segment table, because the B-spline fill's
observers ARE its segments. Razor has **two** observer sets and neither is
the source set — the testing-path quadrature points, each carrying the
path's own tangent (T1), and the segment centroids the charge term
differences between (T2) — and its `geom` dict does not even carry the keys
the old spelling read. So the producer now takes an observer set and a
source set as `(centres, tangents)` pairs, with the square case as a
DEFAULT (`own_segments`) rather than the definition. **A second consumer
forced exactly one generalisation, one layer deep, and no new concept —
twice now, on two different members of the same object.**

The physics detail worth recording is why the windows exist at all. The
whole-geometry `weight_tables` fuse the observer tangent dot INTO w_A
(`a_term_weights` contracts the Fresnel dyad with both tangents to one
number per pair), which is exactly right for a fill whose observers are
segments and exactly unusable for one whose tangent table is `(3, n_basis)`
and whose observers are quadrature points. Razor therefore consumes w_A in
the slot where the PEC fill writes `t_out · M·t_n`, applying the wing's σ
after the gather — legitimate because w_A is linear in the source tangent —
and the PEC limit ρ_v → 1, ρ_h → −1 returns w_A to that number identically.
T2 takes w_Φ on the image kernel at each centroid before the two centroids
are differenced, which is the pilot's finding 4 realised: this formulation
separates the charge term explicitly, so `ground_phi_mode` applies to the
image-side doublets directly rather than through a field dyad that cannot
express it.

Cost, measured the way units 2 and 3 were (executable lines — no blanks,
comments or docstrings): **91 added and 27 retired in `razor.py`, net
+64**, of which 22 are the two refusal messages' prose (`_SOMMERFELD_REFUSAL`
and `_CONTACT_OVER_FINITE_REFUSAL`, 11 lines each) and most of the 27
retired are the three `_OUT_OF_SCOPE` entries they replace — leaving ~42
lines of constructor validation and weighted fill together. In
`_potential_ground.py`: **15 added and 9 retired, net +6**, and even that
overstates it, since the retired nine are the closures' captured arrays
being renamed from `tangents` / `seg_c` to `obs_t` / `src_t` / `obs_c` /
`src_c`. No new physics, no new kernel, no C++.

| gate | result |
|---|---|
| PEC limit | **held and rate-checked.** \|Z(ε̃) − Z_PEC\| on the dipole@0.25λ deck at N = 48: 15.318, 5.876, 1.991, 0.6441, 0.06503 Ω at ε̃ = 10, 10², 10³, 10⁴, 10⁶ — monotone, both lanes, lossless and lossy. The single-decade ratios 2.61 / 2.95 / 3.09 climb toward √10 from below, which is the coefficients' own O(ε̃^{−1/2}) rate rather than a tolerance being met |
| cross-formulation agreement | **held, in the form this formulation can honestly claim.** Razor's disagreement with `BSplineSolver(degree=2)` / `SinusoidalSolver` / `SinusoidalGalerkinSolver` at N = 192 over `ground_eps` = 13 − 0.03j equals its FREE-SPACE disagreement with the same solver to within 0.133 Ω (worst; dipole@0.25λ, GL lane, sin-galerkin). The gap itself is 0.43–1.05 Ω and is razor's own O(1/N) walk — the dipole ladder closes on the Galerkin answer 5.217 → 2.887 → 1.653 → 0.951 Ω, ratios 1.81 / 1.75 / 1.74 |
| swept == per-k | **bit for bit**, over a `(eps_r, sigma)` ground whose ε̃ moves with ω — the gate that would catch weights hoisted into the prepare half |
| free space + PEC + contact bit-identity | **held**: 170 razor arrays (7 deck shapes × up to 3 grounds × 2 quadrature lanes × 5 readouts) unmoved against the branch point. Both golden modules unmoved |
| B-spline grounded bit-identity | **held**: 24 arrays across PEC / refl-coef / two Φ modes / Sommerfeld × degree 1-2 × clear and contact decks, unmoved by the `weight_windows` generalisation; the two spellings pinned `array_equal` |

**The honest negative, recorded rather than tuned.** On the inverted-V deck
`SinusoidalSolver` sits 12.6 Ω from the other four solvers — in FREE SPACE,
unchanged by any ground. That is a free-space property of the sinusoidal
basis on a bent three-anchor wire and has nothing to do with this unit; it
is named here because it is exactly the reason the gate is written on the
difference between the refl-coef and free-space columns rather than on the
refl-coef column alone. A ground unit that gated on absolute cross-solver
agreement would have had to either exclude that solver or invent a
tolerance to cover it.

**What unit 5 (Sommerfeld / `mode == "compose"`) inherits.** The fold side
is now proven twice over on this solver, and the composing side is the part
that has no answer yet. `Remainder.evaluate(supp_seg, polys)` is a B-spline
signature — it takes the basis description that `_Z_sommerfeld_remainder`
consumes, and returns an `(n_basis, n_basis)` Galerkin block. Razor has no
`supp_seg`/`polys`, and its rows are path integrals rather than Galerkin
projections, so unit 5 cannot consume the existing `Remainder` at all: it
needs the remainder FIELD sampled at razor's own testing-path points, which
is a different operation from the block `evaluate` returns, not a different
shape of the same one. The specific ask is in the unit-4 report; the short
version is that `Remainder` must grow an operation ("the remainder field at
these observers, from these source segments") with `evaluate` kept as the
Galerkin convenience over it — which is, for the third time, the same
generalisation this pilot keeps producing.

### 6.5 Unit 5: the Sommerfeld ground on razor (2026-08-17)

The composing ground, and with it the whole ground column for wires
standing clear of the plane: `RazorSolver.capabilities.grounds` is now
`{"pec", "refl-coef", "sommerfeld"}` on a solver that implemented **zero**
ground methods when the pilot began.

**§6.4's prediction was right, and it was right for a reason that made the
unit's shared-layer edit predictable to the line.** Unit 4 recorded that
`Remainder.evaluate(supp_seg, polys)` is a B-spline signature twice over
(basis description in, finished Galerkin block out) and that razor's
path-integral rows need a different OPERATION rather than a different shape
of the same one. What makes this the strongest instance of the pilot's
central claim so far is that a **second, independent consumer said the same
thing first**: the `PulseSolver` probe (momwire#416,
`docs/design/pulse-probe.md` §4.1) was blocked by the identical signature
in the identical way — it needs the plain rectangular field at one centroid
per row — and proposed the identical fix, down to naming
`_sommerfeld.remainder_field_proj` as the primitive that already exists one
level down. Two consumers, arrived at from different formulations on
different days, agreeing on the shape of a generalisation before it was
made, is a different quality of evidence from one consumer forcing one.

So `Remainder` grew `field_windows(observers, sources)`: the remainder
FIELD of each source segment's basis ARC MOMENTS, projected on the observer
tangents, produced in observer-row windows `(i1-i0, n_src, n_moment)`. The
moment count is the whole of what separates the consumers — 2 for razor's
tent (the M0/M1 analogue), 1 for a pulse's rectangular field, `degree + 1`
for the B-spline moments `evaluate`'s own fused kernel forms internally.
`evaluate` stays, unchanged and bit-identical, as the Galerkin convenience
over it. **Third generalisation, third time one layer deep, third time with
no new concept** — after `ImageGeometry` (unit 2) and `weight_windows`
(unit 4), on three different members of the same object.

One honest asymmetry, recorded because units 2 and 4 did not have it. Their
conveniences are literally built out of their operations and are pinned
`array_equal`. `evaluate` is not: its hot path is the fused C++
`sommerfeld_remainder_bspline_Q`, which interpolates, projects,
moment-quadratures on both axes and assembles into `Q` without ever forming
the intermediate `field_windows` returns. That is the same *two kernels,
not one kernel behind two shapes* situation `weight_tables`' `None` already
describes, so the two spellings are pinned at the quadrature's own
agreement instead — 2.9e-15 relative, the same class as the existing
fused-vs-numpy gate.

**The schedule finding is the unit's own, and it is sharper than unit 4's.**
Unit 4's weights are ω-dependent, so they could not live in the
k-independent prepare half. The Sommerfeld grid is k-dependent *in every
part*: its lattice is spaced in wavelengths, its extent is bucketed in
wavelengths, and its surface values carry an ω·μ normalisation. So nothing
of it may cross the prepare/replay boundary in any form — not the grid, not
the source quadrature nodes derived alongside it. The composing fill takes
exactly two more arrays from `prepare`, both pure geometry (the real source
segment endpoints), and builds everything else per solved wavenumber inside
the producer `field_windows` returns. The gate watches
`_sommerfeld.get_grid` itself rather than inferring the schedule from the
answer.

The composition is where the mode contract earns its existence. `C₂·img`
needs no new fill code at all — C₂ arrives as `weight_windows`' constant
`(w_A, w_Φ)` pair, so unit 4's weighted fill assembles the scaled exact
image with the same two lines it uses for the Fresnel one — and `Q` is
summed into that block **inside `_assemble_Z_source_block`, before the
seam's single minus**, because `free − (C₂·img + Q) ≠ (free − C₂·img) − Q`
in float64. Three properties of Q are each a way to get it wrong, and each
is spelled at its seam: it is a FIELD (no jωμ / 1/jωε prefactor), it
integrates over the REAL source segments (the plane is inside the field,
through the grid) even though it rides in the image block, and it is
associated before the minus. §8 risk 2 — "stage 2 cannot hold the
Sommerfeld fold order" — is answered for this trunk: the association is one
line in the solver, and the object names it.

Cost, measured as units 2-4 were (executable lines — no blanks, comments or
docstrings): **44 added and 24 retired in `razor.py`, net +20**, of which
~25 added are the fill (the producer, the in-loop wing algebra, the
association) and the rest is constructor plumbing — the `n_qp_sommerfeld`
knob and the `ground_eps` requirement — while 15 of the 24 retired are
`_SOMMERFELD_REFUSAL`'s prose, deleted because the capability arrived. In
`_potential_ground.py`: **55 added and 4 retired, net +51**, all of it the
new operation. **The shared layer grew more than the consumer did**, which
is the first time in this pilot that has happened and is exactly what one
expects when a unit's content is an interface rather than a capability. No
new physics, no new kernel, no C++.

| gate | result |
|---|---|
| ε̃ → 1 is free space | **bit for bit**, both quadrature lanes — C₂ = 0 and the six λ-integrals short-circuit to exact zeros, so the layer collapses arithmetically rather than approximately |
| PEC limit | **held and rate-checked.** \|Z(ε̃) − Z_PEC\| on dipole@0.25λ, N = 48: 16.625, 6.3687, 2.1477, 0.69348, 0.069959 Ω at ε̃ = 10 … 10⁶, single-decade ratios 2.61 / 2.97 / 3.10 → √10 from below. That is C₂'s rate, not the remainder's, and seeing it means both terms converge rather than one cancelling the other's error |
| the remainder is alive | **22.81 Ω** between the two finite grounds on a dipole 0.04 λ up (48.918 − 21.346j refl-coef vs 71.112 − 16.087j Sommerfeld, N = 48), where `BSplineSolver(degree=2)` splits by 22.837 Ω — the two formulations agree about what the GROUND did to 0.03 Ω. At 0.25 λ the same comparison is 1.211 vs 1.218 Ω, which is why the low deck had to be added |
| cross-formulation agreement | **held**, unit 4's bar and unit 4's form: every Sommerfeld delta is its own free-space delta to within **0.047 Ω** (worst; 3 decks × 2 lanes × 3 reference formulations at N = 192) against 0.25 Ω. Absolute worst 0.968 Ω against the Galerkin siblings; both dipole ladders close monotonically at the O(1/N) rate (0.25 λ: 5.207 → 0.949, ratios 1.81/1.75/1.74; 0.04 λ: 4.692 → 0.864) |
| swept == per-k | **bit for bit**, and `get_grid` observed to fire once per solved wavenumber AT that wavenumber |
| razor bit-identity | **held**: 204 razor arrays (7 deck shapes × up to 3 grounds × 2 lanes × 6 readouts) unmoved against the branch point, free space / PEC clear / PEC contact / refl-coef. Both golden modules unmoved |
| B-spline grounded bit-identity | **held**: 50 arrays unmoved by the `Remainder` generalisation — 48 across PEC / refl-coef / two Φ modes / Sommerfeld × degree 1-2 × three deck shapes (clear, contact, bent), plus the two 41-band grounded swept replays, over refl-coef and over Sommerfeld |

**What is still refused on this row, and why** — the input unit 6's bar
proposal needs. Ground CONTACT over either finite ground (momwire#282: the
composing ground's image coefficient is C₂, not 1, so a grounded tent's
lower wing is no more its own exact image over Sommerfeld than over
Fresnel — the same argument, now with a second ground behind it). A
mid-span touchdown (a second unknown at a knot that already carries a
tent). The extended kernel (NEC-5 tests on the wire axis). Wire loading,
junction ports, node gaps, per-wire radii, singular enrichment. And no C++:
the composing fill is the first place on this solver where that is a
throughput decision rather than a policy one, since the grid interpolation
it leans on is already accelerated while the assembly around it is not.

### 6.6 Unit 6: the production bars (maintainer decision, 2026-08-17)

This closes §9's question 4 for the razor row. The two quadrature lanes
carry two different claims and each is gated on its own terms:

* **The `nec5_quadrature` lane claims the formulation twin** — a fixed,
  N-independent offset from the licensed binary's printed answers — and
  keeps the sharp bar (§6.2 for clearance, §6.3's mirror-deck oracle for
  contact). Nothing new.
* **The Gauss-Legendre lane claims convergence to the continuum.**
  Converged GL is deliberately not NEC-5's rule, so its NEC-5 residual is a
  quadrature difference that must *vanish* with the mesh — and measured, it
  does on five of the six recorded ladders (dipole 1.372 → 0.116 Ω, fat
  dipole 0.957 → 0.130, inverted-V 1.314 → 0.548, monopole 0.195 → 0.029,
  inverted-L 0.520 → 0.140 over N = 24…96). **The decision: gate the
  production lane by DECAY** — residual strictly shrinking down each
  ladder, finest rung pinned at its measured level +25 % — rather than by a
  fixed offset, which is the other lane's claim. The loop is the recorded
  exception (non-monotone, 1.78 → 4.54 → 3.11; four junctions, slowest
  mesh) and carries a 5 Ω envelope pin only.
  `tests/test_razor_production_lane.py`.
* **The finite grounds take no NEC-5 bar at all** (Michalski limit offset,
  §6.4). Their production bars are the cross-formulation
  difference-of-columns gates units 4 and 5 landed — |Δgap − Δgap_free| ≤
  0.25 Ω at N = 192 against three in-house rows — **ratified as final** by
  the same decision, together with the PEC-limit decay pins and the
  low-height formulation-split agreement (22.81 Ω vs 22.837 Ω at 0.04 λ)
  as the standing evidence set.

### 6.7 Unit 7: wire loading on razor (momwire#427, 2026-08-17)

The first CRITERION-2 row completion after the ground units — not a ground
at all, and that is why it is worth recording here: the pilot's claim was
that a capability lands on this solver through a shared layer plus a
testing-idiom term, and loading is the same shape of work with a different
shared layer (`_wire_loading` in place of `PotentialGround`).

**The term.** A loaded wire's surface condition is E_tan = Z_s(l)·I(l), and
razor tests on a path, so `Z = Z_free + L` with

    L[m, n] = ∫_{P_m} Z_s(l) Λ_n(l) dl

In the path/wing idiom that integral is two constants on a shared segment —
`3h/8` when the row's path half and the column's tent ramp rise at the same
end, `h/8` when they rise at opposite ends — times σ_row·σ_col. Junction
tents need no case (a junction tent is two wings on two real segments like
any other), and the grounded-end tent needs no case either: its side-A wing
is its own image and carries σ = 0, which drops both the image half of its
testing path and the image half of its column, leaving the real base
segment. That is the loading-side statement of the halved row that already
makes a base-fed monopole return Z_dipole/2.

**The sign is an oracle result, not a convention.** The siblings assemble G
with the opposite global sign and therefore SUBTRACT their loading term;
this trunk adds. What fixes it is the drive-point identity — a lumped Z_L at
the fed knot must give `Z_driven = Z_unloaded + Z_L` — which is
Sherman-Morrison on a rank-1 diagonal stamp and holds to LU roundoff (1e-14
… 1e-17 relative, both lanes, dipole and grounded monopole).

**A lumped load is razor's own kwarg, and the reason is architectural.** The
other three rows serve a lumped load as deck-level port algebra over a
`node_gaps` port (`momwire.deck._solver`: "stamping a load is port algebra…
and it belongs to whoever owns the answer"). Razor refuses `node_gaps`
because its delta gap lands in a whole testing ROW rather than in a local
basis edit — so the port route is closed here. But the same integral above
with Z_s = Z_L·δ(l − l_p) collapses to ONE diagonal entry, because only P_p
contains knot p and Λ_n(l_p) = δ_np. So the formulation that cannot serve
the house idiom serves the physics directly and more cheaply, and the two
are proved equal on razor itself (a load at a non-driven knot equals the
two-port reduction of razor's own unloaded Y terminated in Z_L, to LU
roundoff, on an asymmetric deck with a non-reciprocal Y).

**Schedule.** The stencil is pure geometry and rides `_assemble_Z_prepare`;
Z_s(ω) is NOT — skin effect goes as √ω and the insulation reactance as ω —
so it is rebuilt per solved wavenumber beside the reflection-coefficient
weights, and the swept gates prove it by sweeping ±20 % in wavelength. The
term is applied OUTSIDE the ground fold, `Z = (Z_free − Z_image) + L`, since
a surface impedance is a property of the conductor and takes no image and no
Fresnel weight; that is why one line serves free space, both folding grounds
and the composing one.

**Bars.** The NEC-5 twin lane is gated on the loading INCREMENT (razor and
NEC-5 never agree pointwise) at 0.05 Ω per rung, measured worst 0.021 Ω;
conventions were verified from printed output first, and `LD` on segment j
turns out to address the same knot `EX` on segment j does. Cross-formulation
is the units-4/5 difference-of-differences at N = 192 with the same 0.25 Ω
bar, measured +0.0025 (copper dipole) to −0.33 (trap-loaded inverted-V).
`tests/test_razor_loading.py`.

**One finding for the next unit, now taken.** The maintainer's observation
that the API-facing half of loading — kwargs, units, the skin-effect model,
resolving a lumped load to a position — is formulation-independent and about
to exist in four solvers was already true here: `RazorSolver._loading_spec`
was exactly that half, isolated behind a plain per-segment/per-knot
description so the extraction had one seam to cut. momwire#428 cut it — the
spec layer is `_wire_loading.configure_loading` / `normalize_lumped_loads` /
`loading_for(solver, ω, geom)`, and every row's `_apply_loading` now consumes
one producer (§6.9).

### 6.8 The shared-layer consolidation (momwire#429, 2026-08-18)

Four **pure moves** on the ground layer, from the sharing audit's ranked
backlog (momwire#429 ranks 2, 5, 6, 7). None of them is a generalisation:
§6.2/§6.4/§6.5 each recorded a second consumer forcing one new operation
one layer down, and this unit is the other half of that pattern — the
places where the operation was already right and only the SPELLING was
duplicated. Every answer is bit-identical by construction and gated as
such: **293 pinned entries** (275 arrays + 18 recorded refusals) across
bspline degree 1-2 × 4 grounds × 3 decks × both image routes × the EK
fill, hmatrix's grounded `zblock` and impedance, array_block, both
sinusoidal solvers, razor on both quadrature lanes, and pulse, compared
with `array_equal` against the branch point.

| unit | rank | the move |
|---|---|---|
| 1 | 2 | the refl-coef **weight chain** (`_image_refl_prep` / `_image_refl_weights`) into `_potential_ground`, and `hmatrix._refl_weight_tables` onto `weight_windows` |
| 2 | 5 | **`_ground_spec.ground_config`** — one decision tree feeding both trunk factories |
| 3 | 6 | **`_ground_mirror`** — one mirror map, retiring 14 spellings across 6 files |
| 4 | 7 | **`_ground_spec.ground_touch_tol`** — one touch tolerance, retiring 5 |

**Unit 1 closes momwire#416's first interface finding.** `weight_tables()`
was documented as the whole geometry's weights and could not produce them
for anyone but `BSplineSolver`, because its refl-coef branch called back
into two solver-private methods. The chain is in `_potential_ground` now;
the CACHE stayed on the solver, because it is schedule, and reaches the
object through `weight_tables(prep=…)` — the same supplier indirection
`FieldGround.projector(tables=…)` uses one trunk over. The audit's
prediction that `hmatrix._refl_weight_tables` maps onto
`weight_windows(observers, sources)` with **no generalisation** held: §6.4
had already reshaped that producer for razor, and the rectangle form was
waiting.

**Unit 2 is what §6.4's precedent implies for the factories themselves.**
They were 82 % literally identical; the shared part reads four solver
strings and answers `(mode, eps_tilde, image_coefficient,
standard_fresnel)`, which is not a piece of either formulation. `weighted`
on the field trunk and "does this ground carry a `phi_mode`" on the
potential trunk turned out to be one predicate. `_ground_spec` imports
only `_ground_refl`, so neither trunk acquires the other's dependencies —
and §6.1's radial screen is now **one row instead of two**, which is the
stretch criterion's instrument getting cheaper before it is used.

**Unit 3's caution earned its keep.** One site of the fourteen is NOT the
mirror and stays: `_ground_refl.specular_ray_tables`' `cos_th -= 2.0 *
ground_z` is the fused in-place `z_m − (2g − z_n)` observer-minus-image
difference, in a tiled scratch buffer whose association momwire#357 item 2
fixed. Two field-trunk sites spelled a mirrored position as `p·M + 2g·ẑ`
where the potential trunk writes `z → 2g − z`; the two agree bit for bit,
and the gate confirms it through the mirrored **EK axis-label scans**,
where a changed label would move the answer rather than merely the
rounding.

**Unit 4 also fixed a false docstring.** `RazorSolver._find_junctions`
claimed `_JUNCTION_TOL` (absolute 1e-9, Euclidean, first-match) was "the
same tolerance the caller-facing geometry helpers use"; the deck layer
that actually writes `junctions=` fuses on a 1e-6 m per-coordinate grid.
What keeps the gap from biting is the deck's own invariant that coincident
ends are already exactly equal — not an agreement between the numbers.
**No tolerance value changed anywhere in this arc.** Six disagreeing
tolerances across three algorithms and two norms remain, and unifying them
is a decision, not a move.

**Cost, measured the way units 2-7 were** (executable lines — no blanks,
comments or docstrings): **5,924 → 5,934 across the eleven touched files,
net +10**, of which 28 are the two new modules (`_ground_spec` 19,
`_ground_mirror` 9) — so the consuming files shed 18 between them, with
`bspline.py` down 13 and `hmatrix.py` down 6. The two modules are 207
lines on disk, which is to say ~86 % prose. What the +10 buys is not size
but arity: 14 mirror spellings → 2 functions, 5 touch tolerances → 1,
2 factory decision trees → 1, 2 weight chains → 1.

**What this says about the goal metric.** Nothing new can be *done* that
could not be done before — which is exactly why these sit below the
constructor normaliser (rank 1) and `loading_for` (rank 4) in the audit's
ranking, both of which unblock capabilities. What they buy is that the
fifth ground, when it comes, is written once.

### 6.9 The radius arc (momwire#428, #425, #147, 2026-08-18)

Three units on one branch, and the shape of the arc is the architecture's
own claim tested end to end: **two extractions that buy nothing on their
own, then the capability they were the price of.** §6.8 closed with the
observation that the ground consolidations could not unblock anything;
this arc is the other case.

| unit | issue | the move |
|---|---|---|
| A | #428 | `loading_for` — one loading spec layer under four testing rules |
| B | #425 | `_kernel_moments.py`, and one radius normaliser (audit rank 3) |
| C | #147 | per-wire radii on `RazorSolver` |

**Unit A is the audit's rank 4, and it went exactly as predicted: a pure
move.** Loading cannot share its MATRIX assembly — SG overlaps, sinusoidal
collocates, razor path-integrates, bspline spline-overlaps, the same
two-object reason the ground layer has — but everything upstream of the
term is formulation-independent and existed four times. It is now
`_wire_loading.configure_loading` (the kwargs, the units, the fail-fast
validation), `normalize_lumped_loads`, and `loading_for(solver, ω, geom)`
returning a `LoadingSpec` of per-wire Z'_w(ω), per-SEGMENT Z_s(ω) and the
lumped loads resolved to sites. What could NOT be shared is exactly one
step — which index a formulation names a site by — so that is a solver
hook (`RazorSolver._lumped_site_index`), carrying its own K≥3 refusal
prose because what is ambiguous at a junction differs from the feed's.
`loading_for` takes ω as an argument and caches nothing, which puts it on
the correct side of every prepare/replay boundary by construction rather
than by discipline.

**Unit B answers a coupling the previous arc left visible on purpose.**
`PulseSolver` imported `razor._axis_frame` / `_static_axis_moments` —
module-level privates of a sibling — and momwire#419 left that import in
place so the finding would surface. The bodies are now `_kernel_moments`,
razor re-exports both names so pulse's import keeps working across the
move (migrating that line is pulse's own one-liner), and a test pins every
consumer naming the same function object. The radius normaliser landed
beside it in a new `_wire_spec.py`, the audit's rank-1 home: three copies
retired, including a weaker one inside `series_impedance_per_wire` that
had the same message and no positivity check, and razor's refusal variant
folded in as a `per_wire_refusal=` argument — which is what made unit C's
constructor change *dropping an argument*.

**Unit C is what the extractions were for.** `_axis_frame`'s `a` enters
only as `a * a` added to ρ², so accepting a `(n_segs,)` column instead of
a float is arithmetically free — the audit called this correctly, and the
scalar path stayed bit-identical through all three units (193 pinned
answers, `array_equal`, every unit). Everything else the radius touches
had already been made per-wire by unit A: the loading spec reads
`_radius_per_wire`, which on razor stopped being a broadcast fiction and
became the real array.

**The convention, and why it is a choice here.** The reduced kernel's a²
is the source segment's — so a fat/thin junction's tent takes each wing's
own segment's radius with no special case, since each wing's moments were
already built against its own source column. The rival is the OBSERVER
radius NEC-2's `EFLD` uses, which momwire's sinusoidal family adopted
after a PyNEC oracle moved 11 Ω. **For this formulation the binary cannot
tell them apart**: they differ only where the perpendicular distance
vanishes (a collinear radius step), and measured there they are 3.0e-6 …
1.1e-5 Ω apart at a 10:1 step, 1.4e-3 … 2.1e-3 Ω at 100:1, against a
0.20 Ω bar — the difference lives in near-diagonal entries worth ~1400 Ω,
2e-5 relative, which the solve absorbs. The source reading is taken
because it is the reduced kernel's own derivation and because it is
chunk-invariant (the fill chunks the OBSERVER axis). **Recording that a
sibling's oracle finding does not transfer is the point of writing it
down**: the two solvers disagree about the convention and agree about the
answer, and nobody should re-litigate it from the sinusoidal doc alone.

**Bars.** NEC-5 twin lane on decks that carry per-`GW` radii natively —
like-for-like, no translation step for a convention to hide in — on a
thin-driven/fat-parasitic pair and a fat/thin junction, free space and
`GN 1`, ladders N = 24…96: worst |ΔZ| **0.0586 Ω** against
max(0.20 Ω, 0.25 %), offsets constant to **0.0211 Ω** against 0.05 Ω.
Cross-formulation difference-of-columns at N = 192 vs `BSplineSolver`:
**0.063 Ω** on the junction deck against a 0.25 Ω bar.

**A finding about the references, and why the parasitic deck is gated
differently.** On the parasitic deck the absolute difference-of-columns is
0.24 Ω — but so is the UNIFORM-radius control's (0.19 thin / 0.26 fat),
because the two Galerkin references disagree with each other by 0.23 Ω
over that ground at any radius. An absolute bar there would be gating the
references, so what is gated is the claim the capability makes: the mixed
model's number is BRACKETED by its own two uniform controls. And on the
fat/thin STEP both sinusoidal-family solvers are the unconverged party and
walk AWAY — at N = 192 `SinusoidalGalerkinSolver` reads 113.8 Ω and
`SinusoidalSolver` 100.8 Ω where `BSplineSolver` reads 69.23, razor 69.06
and the NEC-5 binary 68.88, while the same deck at a uniform radius has
all of them inside 0.1 Ω. That is momwire#435, filed against those
solvers rather than worked around here.

### 6.10 The deck front end (momwire#432, 2026-08-18)

Razor reaches `momwire.deck.build_solver`, closing #432 — the release-blocking
unit PR #431 left as a follow-up. Two roster entries (`"razor"`,
`"razor-nec5"` for the identified quadrature, momwire#316), keyed on class in
a new `_NATIVE_LOADING` set rather than threaded through `basis_kwargs`: LD
0/1/4 translate to `lumped_loads=[(wire, arclength, Z(ω))]`, evaluated at the
build's own frequency (a sweep already calls `build_solver` once per step,
so per-call evaluation already is the swept behaviour — no separate sweep
case to write); LD 5 and `IS` reach `wire_conductivity` /
`insulation_radius` / `insulation_eps_r` unchanged, since razor takes the
siblings' distributed-loading kwargs verbatim (momwire#427). The arclength
is read off the mesh `to_polylines` already built for the port-algebra
route's OWN ports, so a load lands on the identical knot either route would
have used — no separate snapping logic to disagree with the siblings'.
`node_gaps`, `extended_kernel` and contact-over-a-finite-ground refuse with
razor's own constructor messages, unmodified: `build_solver` does not special
case any of them, because razor's `**unsupported` dispatch already produces
exactly the message §6's registry declares. The one thing `build_solver` DOES
special-case is `junctions=`: every sibling takes it, razor takes no such
argument at all (not even `None` — it is not a declared parameter, so any
spelling of it lands in `**unsupported` and refuses), so the roster's two new
entries skip the keyword entirely rather than passing a value razor would
reject.

**Not reached: the portal.** `RazorSolver` has no `compute_port_solution()`
(§6's registry note stands — it exposes `compute_impedance()` only), so
`_y_and_port_coeffs` in `momwire.portal._portal` raises on the first solve
under `--basis razor`. The name is a valid `--basis` choice the moment it
joins `deck.BASES` (the portal's own roster is `deck.BASES` read once, #846
phase III), so the CLI accepts it and then breaks — a real gap, tracked as a
follow-up rather than fixed here, since a `compute_port_solution` for razor
is a unit of its own and #432 is scoped to the library/script front end.

### 6.11 The port solution (the sharing audit's #429 rank-9 item, 2026-08-18)

Closes the gap §6.10 left open. `RazorSolver` gains `compute_port_solution`
and `compute_port_solution_swept` so `_y_and_port_coeffs` in
`momwire.portal._portal` now drives `--basis razor` / `--basis razor-nec5`
exactly like every other roster entry, through the one shared
`_y_and_port_coeffs(solver) -> sol.y, sol.coeffs` call (§6.10's "Not reached:
the portal" paragraph is left in place above, uncorrected, as the record of
what was true at #432's landing; this section is the update).

One narrower portal-side gap surfaced by this unit is tracked as its own
issue rather than folded in here: **#439**, `_port_signs` indexing
`solver.feeds` by `PortPlan.sites` — true for a driven site, and true for a
site that is both fed and loaded, but not for a load-only site razor bakes
straight into `lumped_loads` rather than the port-algebra route every other
family takes (`docs/razor-solver.md`'s "A remaining portal-side gap").

**Which sibling spelling.** Two exist: `BSplineSolver`'s batched-swept
version (a fully batched fast path whose per-k core is NOT
`compute_port_solution`, to preserve the chunking that bounds
`swept_mem_mb`) and `SinusoidalSolver`'s plain one (no batched assembly on
that family, so the per-k core IS the single-k entry point, and the swept Y
cannot drift from the stacked single-k Y by so much as an ulp). Razor has no
batched C++ accelerator (§6's registry note, "No C++ accelerator"), so it
takes `SinusoidalSolver`'s spelling: `_port_solutions_swept` is a bare loop
over `compute_port_solution(prepared=...)`, sharing the fill's k-independent
half exactly the way `compute_impedance_swept` / `compute_y_matrix_swept`
already did.

**The bit gate held by construction, as predicted.** `_assemble_Z(geom, k)`
was already documented as `_assemble_Z_prepare` once + `_assemble_Z_from_prepared`
per k; `compute_port_solution` calls that same pair (with the prepared block
optionally hoisted in), so a swept point's `Z` cannot differ from a
freshly-constructed single-k solve's — replaying a k-independent prepare is
the same arithmetic as rebuilding it, just fewer times. Measured directly:
`compute_port_solution_swept` over a reflection-coefficient ground built as
an `(eps_r, sigma)` tuple — so ε̃ moves with ω exactly as the ground-unit
4/5 swept gates already exercise — matches a per-k `compute_port_solution`
loop `.tobytes()`-equal on both `y` and `coeffs`, at every rung, in both
quadrature lanes and over all three served grounds.

**What changed in the file.** `compute_y_matrix` is now
`compute_port_solution().y` (one line, matching every other family since
#232) instead of its own fill-and-solve; the hand-rolled
`compute_y_matrix_swept` is deleted in favour of `_SweptPortSolutions`'s
generic implementation over the new `_port_solutions_swept` generator.
`compute_impedance` / `compute_impedance_swept` are untouched — they solve a
superposed single right-hand side rather than one column per port, so they
stay their own entry points exactly as they are for every sibling family.
Net line count: `git diff --numstat` on `razor.py` reads +129/−26 — nowhere
near the audit's "net ≈ −6 lines," but almost all of the excess is this
module's own extremely literate docstring convention (`compute_port_solution`
alone carries a ~35-line docstring matching the siblings' own port-solution
docstrings line for line in level of detail); the CODE the two hand-rolled
Y-matrix fills lost against what `_port_count` / `compute_port_solution` /
`_port_solutions_swept` / `_RazorBasis` gained is close to a wash, which is
what the audit's estimate was actually about.

Gates: `tests/test_razor_port_solution.py` — the four gates `_port_solution.py`
promises every family (`y` IS `compute_port_solution().y`, the columns solve
their own port, `coeffs @ V` reproduces `compute_impedance`, one fill and one
factorisation per call), the swept ω-boundary bit gate above, and the
multi-feed port-order check; `tests/test_portal.py` gains the portal
end-to-end gate, `--basis razor` and `--basis razor-nec5` on a deck with a
load and a ground.

### 6.12 The taper agreement floor (momwire#398 unit 1, 2026-08-18)

The taper-readiness study (momwire#398 — the tapered half of the
maintainer's north star, "NEC-5's two irreplaceable use cases are tapered
wires and ground contact; momwire must match or beat it on both") measured
momwire against the licensed NEC-5 binary across four decks spanning a 20:1
range of Δ/a: `ward` (Ward Harriman AE6TY's flagship 20:1 tapered dipole,
antenna-problem-decks issue #1), `step2` (momwire#435's two-wire 10:1 radius
step), and `fat`/`thin` (Ward's fattest/thinnest section, uniform, as Δ/a
floor controls). This unit lands that measurement as a standing gate rather
than a one-time study doc, plus the two maintainer decisions (D2, D3) it
forced onto real numbers.

**The kernel identification, one line.** The study's other half-finding —
which kernel the NEC-5 binary uses, since it carries no `EK` mode card at
all — is that the binary is **extended/tubular-kernel everywhere**,
identified (not assumed) from printed output alone on two independent Δ/a
paths: at Δ/a = 0.5 the binary prints `89.682 + 43.581j` where a matched
`bspline-d1 + EK` solve reads `93.389 + 41.415j` (4.3 Ω away) and the
reduced kernel reads `113.653 + 3.752j` (24 Ω away); the reduced-kernel
quadrature control (`razor-nec5`, the binary's own path rule on a reduced
kernel) sits 32.3 Ω from the binary on the refine-N path and 111.1 Ω on the
fatten-a path; the binary prints zero warnings or clamps at any rung of
either path. This unit's §6.13 sibling (momwire#398 D1, "razor and the
extended kernel") is where that finding's consequences for `RazorSolver`'s
own twin claim are worked out — **razor + EK is a four-axis twin (basis +
testing + quadrature + kernel), `bspline-d1 + EK` a two-axis one (basis +
kernel, Galerkin-tested); the reduced razor lane stays, as the deliberately
different-discretization cross-check, not something this finding retires.**
Full derivation: the study document (kept beside `scripts/
capture_taper_nec5_lane.py` at capture time; not in the standing tree, since
its role — establishing WHICH kernel the binary uses — is done once this
unit's identification gate (below) protects the conclusion going forward).

**Four decks, golden ladders.** `scripts/capture_taper_nec5_lane.py`
regenerates `tests/golden_taper_nec5.py` as pure literals — the
`capture_razor_pec_nec5_lane.py` house pattern, so the suite needs neither
the NEC-5 binary nor antennaknobs. Every momwire row is built DIRECTLY with
`feed_arclength` at the exact knot NEC-5's `EX` field-4 end code addresses,
not through the deck path: NEC-2's `EX` drives a segment CENTRE and NEC-5's
drives a KNOT, a half-element-apart mismatch the study's §1.4 measured
charging momwire up to 2.6 Ω for a feed-placement difference with nothing
to do with physics. `nec2c` (open source) is captured once alongside NEC-5
and pinned as a literal too, the same choice `tests/test_extended_kernel.py`
already made for its own oracle deck, rather than depended on at test time.

**D4 (gate depth).** N=400 on `ward` puts the fat end at Δ/a ≈ 1.05, past
where any reduced-kernel row means anything (momwire#248). The standing
gate stops at Δ/a ≈ 2.1 (`GATE_MAX_N`: N=200 on `ward`/`fat`/`thin`, N=400 —
its full captured ladder, worst Δ/a ≈ 4.4 — on `step2`, which never
approaches the floor); the finer, recorded-not-gated rungs (`ward`/`fat`/
`thin` to N=400) stay in the golden module for §6.13 to build on.

**Bars, exactly as the study's §8 proposed-first-unit list specified, with
one deviation (below):**

| deck | row(s) | bar | measured |
|---|---|---|---|
| `thin` | `razor-nec5` | offset constancy ≤ 0.05 Ω (the twin claim, where it holds) | dR spread 0.0011, dX spread 0.0013 |
| `ward` | `bs1-ek`, `bs2-ek` | decay, finest gated rung +25% | 5.006→0.626 Ω (bar 0.782); 7.551→1.348 (bar 1.685) |
| `step2` | `bs1-ek`, `bs2-ek` | decay, finest gated rung +25% (whole ladder is gated here) | 7.845→0.552 Ω (bar 0.690); 12.394→1.656 (bar 2.070) |
| `fat` | `bs1-ek`, `bs2-ek` | decay, finest gated rung +25% | 5.881→0.705 Ω (bar 0.881); 8.876→1.689 (bar 2.112) |
| `fat` | `razor`, `bs1` | envelope only, non-monotone | max in gated range 1.270 Ω (bar 1.588); 6.134 Ω (bar 7.668) |
| `step2`, `thin` | `sin` vs `nec2c` | ≤ 0.1 Ω at the finest rung (D3) | 0.058 Ω; 0.090 Ω |
| kernel-ID path A, Δ/a ≤ 1.0 | `bs1`/`bs1-ek`/`razor-nec5` vs NEC-5 | α ≥ 0.9 AND \|B\| ≤ 0.35·\|D\| | worst α 1.086, worst \|B\|/\|D\| 0.281 |

The one deviation from the study's proposal: the sin/nec2c gate runs on
`step2` and `thin` (the maintainer's explicit D3 spelling), not the study's
originally suggested `ward`/`fat` — `step2` is momwire#435's own deck class
and `thin` is the uniform control confirming the identification is not an
artifact of the step. `ward`/`fat` are a defensible follow-up, not required
here.

**Why `fat`'s `razor`/`bs1` get an envelope, not a decay bar.** Inside the
gated window (N ≤ 200) `bs1`'s residual happens to still be falling —
gating it by decay there would look correct today and go false the moment a
future gate (or a user) refines further: both rows turn around by N = 280
(`bs1`: 1.816 Ω rising to 3.146 at N=400; `razor`: 1.165 rising to 2.155),
exactly the Δ/a ≈ 2 floor momwire#248 predicts and §5's kernel
identification explains — a reduced kernel is answering the wrong physics
on fat wire past that point, and refining the mesh cannot fix wrong
physics. `tests/test_taper_agreement_floor.py` gates the non-monotonicity
itself as a property (over the FULL recorded ladder, so it cannot pass by
accident of where the gate happens to stop), specifically so a future
change that makes this row decay cleanly is *caught* rather than silently
outgrowing an envelope pin that had quietly become a decay claim.

**D2 — `sinusoidal-galerkin + EK` on a stepped-radius junction: excluded,
gated.** Measured DIVERGENT on `step2` (extrapolated limit
`7.110 − 483.925j` against NEC-5's `132.560 − 11.921j`, residual growing
every rung, 286 Ω dX spread) — worse than momwire#435's reduced-`sg`
finding, not a variant of it. `SinusoidalGalerkinSolver.__init__` now
refuses `extended_kernel=True` at construction whenever any junction's
members carry different radii, following `BSplineSolver`'s
`use_singular_enrichment` combination-refusal idiom (momwire#396):
`_EK_STEPPED_RADIUS_JUNCTION_REFUSAL` is reused by the `__init__` raise and
by `capabilities.refusals["extended_kernel+stepped_radius_junction"]`, and
carries the measured numbers in its text. Uniform-radius EK solves —
including junctioned uniform-radius decks — are untouched (pinned
bit-identical in `tests/test_sinusoidal_galerkin_stepped_ek.py`); the
refusal has no solve-time cost since it never gets that far.

**D3 — `sin` reproducing NEC-2's stepped-radius defect: a feature, gated as
one.** `SinusoidalSolver` (reduced kernel) converges onto nec2c's own
answer on a radius step — including nec2c's defect — because it shares
NEC-2's per-segment end-condition convention; momwire#435 closes as an
IDENTIFICATION rather than a bug: the sinusoidal row exists to be the
NEC-2-closest rung of the basis ladder, and inheriting NEC-2's defect on a
taper is what "closest" means on this deck class.
`tests/test_taper_sin_identification.py` gates `|Z_sin − Z_nec2c| ≤ 0.1 Ω`
at the finest captured rung on `step2` and `thin` as a PROTECTED property:
`sin` is required to agree with nec2c and explicitly allowed to disagree
with NEC-5 (the same file's last test confirms both hold simultaneously on
`step2` — 0.058 Ω from nec2c against >1 Ω from NEC-5 at the same rung).

**Docs.** `site/src/content/docs/reference/portal-usage.md`'s `--basis`
section gains the D3/D2 caveat for `sinusoidal`/`sinusoidal-galerkin` on
tapered or stepped-radius decks, pointing at `bspline-d1` + `EK` as the row
measured closest to NEC-5 there; `deck-grammar-nec2.md`'s `EK` card entry
cross-links it.

**Files.** `scripts/capture_taper_nec5_lane.py`,
`tests/golden_taper_nec5.py` (generated), `tests/test_taper_agreement_
floor.py`, `tests/test_taper_sin_identification.py`,
`tests/test_sinusoidal_galerkin_stepped_ek.py`,
`src/momwire/sinusoidal_galerkin.py` (the D2 refusal),
`tests/test_extended_kernel_galerkin.py` (nine `[radius step]` cases
re-scoped, above), the two doc edits above, this section.

**Gates.** All four measured bars pass at their measured level (table
above; nothing tuned — every bar is the measured value +25% or the study's
own proposed fixed threshold). The capture script is deterministic: run
twice, `tests/golden_taper_nec5.py` is byte-identical (the script now pipes
its own literal writer through `ruff format` before returning, so a
re-capture is a content diff or nothing, never a formatting one). Bit-
identity: every UNIFORM-radius answer is untouched — the D2 check is one
early `if` inside `SinusoidalGalerkinSolver.__init__` that only ever fires
when `extended_kernel=True` AND a declared junction's members disagree
about `wire_radius`, so a solve that never sets both together cannot reach
the new branch at all.

**A finding the D2 exclusion surfaced outside the taper decks entirely.**
`tests/test_extended_kernel_galerkin.py` (momwire#299's own gate file)
carried a "radius step" node kind — a SMALL, 2:1 ratio, at the modest N its
G-D unit runs at — as one of five node kinds exercised under
`extended_kernel=True`, and it was passing. Blindly landing D2's blanket
refusal would have broken nine of those tests. Rather than narrow the
refusal (D2 asks for a simple construction-time, deck-shape check, matching
the house combo-refusal idiom — not one conditioned on mesh size), this unit
re-measured the discrepancy directly: even that mild 2:1 deck, pushed past
the N that file ever ran it at, shows the same non-convergent behaviour
momwire#435's 10:1 step does (chaotic, not merely slow, beyond N ≈ 128 in a
quick probe). #299's bracket-correction fix is not wrong in the regime it
shipped and was gated in — it is real physics at coarse-to-moderate mesh —
but the mesh-refined regime this unit's ladders probe was never checked
before, and there the whole EK-on-a-step combination is unsound regardless
of the fix. The nine tests were updated in place: `_EK_DECK_NAMES` factors
out the four node kinds ("straight", "L", "vee", "T") that stay under EK,
"radius step" is dropped from every EK-on parametrization with a comment
pointing here, and the historical measurements stay in each docstring so
the record is not lost, only reclassified. `test_gd5`'s reduced-kernel call
sites and `_gd_decks()` itself are untouched — only the EK combination
narrows.

**What this does not do.** Path B of the kernel-identification ladder (fix
N, fatten a) is recorded in the study but not captured into the golden
module — path A alone already clears both thresholds this unit gates, and
path B is a defensible next increment rather than a requirement. The `GC`
tapered-wire continuation stays refused (momwire#398 unit 4, undone here).
The `EX` segment-centre-vs-knot translation rule stays undocumented as a
first-class dialect fact outside this file's method section (momwire#398
unit 2). And razor's own twin-claim requalification — giving it (or not)
the extended kernel, and rewriting `docs/razor-solver.md` accordingly — is
§6.13/D1, deliberately not done here: this unit's `fat` envelope pins on
`razor`/`bs1` are the PRE-EK record that unit will revisit.

### 6.13 The extended kernel on razor (momwire#398 D1, 2026-08-18)

**The maintainer's decision, 2026-08-18: give `RazorSolver` the extended
kernel.** This overrides the taper study's own D1 recommendation — the study
proposed keeping the refusal and publishing the boundary, on the argument
that `bspline-d1 + EK` already covers the fat-wire niche and that razor's
value is being a *different* discretization. The decision goes the other way,
and the kill-test is why.

**What the measurement said.** NEC-5 has no `EK` card, which is consistent
with two opposite worlds: reduced-kernel-only, or extended everywhere. The
study settled it from printed output alone, isolating the kernel with
`BSplineSolver(degree=1)` run reduced against EK on an otherwise identical
setup and driving Δ/a from 10 down through 0.5 along two independent paths.
The binary sits on the **EK side of the kernel gap at every rung of both
ladders** — α → 1.09 / 1.02, within 4–9 % of the EK row across a 43–113 Ω
gap — and prints no warning or clamp anywhere, which a reduced-kernel code
returning `113.653 + 3.752j` for a fat dipole would have every reason to do.
The control that removes quadrature as the explanation is razor itself:
`nec5_quadrature` is the reduced kernel running the binary's own identified
quadrature idiom (momwire#316), and at Δ/a = 0.5 it reads **32.3 Ω** from the
binary where the EK row reads 4.3 Ω.

So the refusal's prose — *"RazorSolver is reduced-kernel only: NEC-5's
formulation is the comparison target, and its expansion is tested on the wire
axis"* — was **half right, and its own reasoning argued for EK.** The
expansion is tested on the axis; the SOURCE it is tested against is a tube.
Three of the study's findings collapse into that one fact: razor's 43×
offset-constancy failure on fat wire, its 4.86 Ω limit gap there, and its
inability to open Ward Harriman's flagship deck (which carries an `EK` card)
were never three findings.

**Why this rather than D1(b).** The study's argument for keeping the refusal
was that EK razor would be "a near-duplicate of `bs1-ek`" and cost the
cross-check. The measurement says otherwise on both halves. On the `fat`
control the two rows are not near-duplicates — `razor-EK-n5q` extrapolates
0.047 Ω from the binary and `bs1-ek` 0.405 Ω, at the same rungs — because the
two discretizations remain as different as they ever were; the kernel was
never the thing that made them independent. And the cross-check survives
intact, in a sharper form: the two formulations can now be asked whether they
report the SAME kernel perturbation, which is the units-4/5
difference-of-differences and is gated at 0.25 Ω.

**The kill-test numbers, on the study's uniform `fat` control** (ten colinear
25 mm sections, 14.2 MHz, matched feed at NEC-5's knot, ladder gated to
N ≤ 200 so the fat end stays above the Δ/a ≈ 2 floor of momwire#248):

| row | offset from NEC-5, N = 20 → 200 | dR spread | dX spread | limit gap |
|---|---|---|---|---|
| razor reduced, n5q | +0.02+0.06j → +0.58+0.54j Ω | 0.555 | 0.475 | 1.400 Ω |
| **razor EK, n5q** | +0.005+0.022j → +0.005+0.011j Ω | **0.012** | **0.021** | **0.047 Ω** |
| `bs1-ek` (same rungs) | — | 1.53 | 5.03 | 0.405 Ω |

Every column is the gated ladder: the limit gap is Richardson on rungs
140/200, against NEC-5's own limit from the same pair. Extrapolated instead
from the study's 280/400 — outside the valid Δ/a domain, see the correction
below — the same three read 4.863 / 0.666 / 1.189 Ω. The ratio is what
survives either choice: **EK moves razor an order of magnitude closer to the
reference on fat wire, and past `bs1-ek` while doing it.**

against the `nec5_quadrature` sharp-lane constancy bar of **0.05 Ω** (§6.2).
(The study's own "43×" is the same reduced row measured over its FULL ladder
to N = 400; on the gated rungs above it is 11×. Both are the same finding.)

**One correction to the study, found while gating this.** Its §2.3 `fat`
limits — including NEC-5's own `89.854 + 43.902j` — are Richardson
extrapolations from rungs 280 and 400, i.e. from Δ/a = 1.50 and 1.05, both
BELOW the Δ/a ≈ 2 floor the study itself establishes in §2.4. Redone inside
the valid domain (rungs 140/200), the reference's limit is
`89.326 + 44.114j`, 0.57 Ω away, and every row's published gap moves with
it: `bs1-ek` 1.189 → **0.405 Ω**, `bs1` 6.881 → 1.708, `razor` reduced
4.863 → 1.388. The study's D4 recommendation ("gate to 200, record to 400")
should extend to the LIMITS it publishes and not only to the gates; the
ranking is unchanged, the magnitudes are not.
On Ward's actual 10-step taper the same lane holds the bar in dR (0.020) and
runs 1.6× over it in dX (0.078) — the eligibility rule's own documented
conservatism at a radius step, where momwire extends only coaxial
EQUAL-radius pairs and NEC still extends some cross-arm pairs (`IND = 2`,
#249 §4.3, O(h) in the refinement limit). The uniform control has no step and
is therefore the clean measurement of the kernel.

**What the unit did NOT have to build, which is the pilot's claim again.**
The eligibility rule is `_bspline_kernels._ek_axis_groups`, unchanged and
un-forked — the shared layer's fourth consumer. The mirror policy is the
shared layer's too: over a ground, eligibility is ONE joint scan of the real
segments stacked on the mirrored ones, so the ground supplies mirrored
GEOMETRY and never the kernel's opinion, and ground CONTACT — whose grounded
tent has its own image for a lower wing — needed no line at all. What is new
is one closed form (`_kernel_moments._static_axis_moments_ek`: the extended
kernel's k → 0 limit `1/R − a²/(2R³) + 3a⁴/(4R⁵)` integrated over a segment,
with the two 1/R terms collected as `ρ² − a²` so that the exact cancellation
at ρ = a is exact in IEEE rather than catastrophic) plus the mask threaded
through `_seg_moments_prepare`'s chunk. The EK statics are as k-independent
as the reduced ones, so the prepare/replay split is unchanged in shape.

**Two axes that are orthogonal, said once.** The QUADRATURE rule
(`nec5_quadrature`) and the KERNEL (`extended_kernel`) are independent axes
and all four combinations are served. The lane decides where the testing path
is sampled; the kernel decides what is sampled there. The fat-wire twin is
both of NEC-5's identifications at once.

**Gates.** `tests/test_razor_extended_kernel.py` against
`tests/golden_razor_taper_nec5.py` (captured by
`scripts/capture_razor_taper_nec5_lane.py`): the fat twin bar above, its
negative control on the reduced row, thin-wire kernel agreement
(|EK − reduced| ≤ 0.0014 Ω where the twin's own residual is 0.037), the
cross-formulation difference-of-differences at 0.25 Ω, eligibility and mirror
policy, every capability × both lanes with the kernel on, and the swept
prepare/replay split. Gate (b) of §6 — EK off is structurally absent — is
held by a monkeypatch-counter on razor's five EK entry points, with 84
shadow arrays (21 configurations × Z matrix, impedance, coefficients, a
three-point sweep) measured bit-for-bit identical to the branch point.

**What this leaves open.** The taper study's other units are untouched:
the standing taper golden ladder for the *other* rows (its unit 1), the
`EX` segment-centre/knot translation rule (unit 2), `GC` in the nec2 dialect
(unit 4) and the `sg-ek` divergence refusal (unit 5). The razor doc's
requalification (its unit 3) is subsumed here rather than deferred — the
claim it wanted requalified is now *true* on fat wire rather than merely
bounded — but the routing half of D1(c) ("`--basis razor` on a deck with an
`EK` card names `bspline-d1` instead") is moot: razor opens that deck now.

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
