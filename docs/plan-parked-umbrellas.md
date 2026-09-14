# Parked umbrellas — the open plans of three closed issues

Three umbrella issues outlived the work they organised: most of each shipped,
and what is left is parked rather than scheduled. This file holds what they
still carry so the issues themselves can be closed. **Each section below
replaces one closed GitHub issue**: what shipped under it (with the PR or
commit that shipped it), what stays open (quoted as written, dated by the
comment it came from), and how to pick it up again. The last section records
two caps decided permanent, so the refusals behind them are not re-opened as
if the question were fresh.

Written 2026-09-14 against `main` at `6549550`. Comment dates are GitHub's
UTC timestamps.

## Frontend deck compatibility: serve what EZNEC and 4nec2 emit — [#456](https://github.com/stevenmburns/momwire/issues/456)

Umbrella for momwire as a drop-in text-interface engine behind EZNEC Pro+
(NEC-5 dialect) and 4nec2 (NEC-2 dialect). Opened 2026-08-19; re-cut by
frontend target on 2026-08-20.

### What shipped

- **The statement matrix's silent column went to zero** — #460 (the #458
  hygiene wave: EOF as `EN`, `GD`-with-`GN 1` and `SC` refused by name,
  `PQ -1` served).
- **4nec2's NEC-2 dialect serves its geometry, networks and MININEC ground** —
  #472 (`GX` / `GR`, #415), #474 and #482 (`momwire.networks`; the nec2 portal
  serves `TL` and `NT`), #495 (`GD` with `GN 1`, #487).
- **The EZNEC NEC-5 front exists end to end** — #498–#503 (the #497 seam
  skeleton: process shell, dialect parser, printout renderer, rung-1 physics,
  node-addressed networks with the W7EL gate) and #505–#509 (the #504 ground
  rungs: `GN 0` / `GN 2`, bare `GD`, networks over grounds, phased multi-`EX`).
- **The drop-in is packaged for Windows** — #533 (a Windows exe, byte-gated in
  CI), #711 (Authenticode-signed).
- **Sweep economics at engine parity** — #726 (resident daemon + thin client)
  and #732 (native C client), the #718 arc.

### What stays open

The body's unticked boxes, text as written on 2026-08-19. A box is ticked here
only where a merged PR or closed issue covers it by its own title; each
**Record** line names it. Unticked boxes have no such record.

**§1 Emission census**

- [ ] EZNEC leftovers named by the capture: perfect ground and MININEC-type
      ground unsampled; SOMMPD.NEX statefulness semantics for a drop-in.
  - **Record:** #507 serves the bare-`GD` MININEC mode; neither the capture
    sampling nor SOMMPD.NEX semantics is recorded as done.

**§2 NEC-2 dialect gaps (the 4nec2 half)**

- [ ] #415 — GX / GR geometry cards, now widened by the #413 capture: **GX, GR,
      GC, GH, GM all pass straight through**, as do CP, PQ and the WG/GF
      Green's-function side-file with its NX multi-structure handoff. Geometry
      transforms are the bulk of the serve work (surface patches SP/SM/SC stay
      in the deliberate-exclusion column).
  - **Record:** GX and GR serve (#472; #415 closed). The 2026-08-20 re-cut
    reports the census "88.8 % with the remaining GA/GH/GC geometry waived".
- [x] NT/TL through the portal: the SimNEC dialect refuses network decks by
      spec; 4nec2 emits them. Decide and build the serve path (momwire's NT
      oracle exists — #65's reducer).
  - **Record:** #482; 2026-08-20 comment.
- [x] The two silent manufacturings the capture found, honored exactly: **EX 6
      becomes a voltage source behind an NT network** (drags network solving
      into decks whose authors wrote none) and **GN 3 becomes GN 1 + GD**
      (substituted ground physics). Both arrive at the engine as manufactured
      cards, so serving NT and GD correctly covers them — but the gates should
      pin the manufactured forms specifically.
  - **Record:** 2026-08-20 comment: "both halves are now pinned".
- [ ] GN NRADL / radial screens: decide implement vs loud documented exclusion
      (no oracle on the NEC-5 dialect either — see #444's landmine).
  - **Record:** parked by the 2026-08-20 re-cut (quoted below).
- [ ] Refusals rendered in 4nec2's own error grammar (the #413 census must
      capture what 4nec2 treats as engine failure, the way the EZNEC study
      did).
  - **Record:** parked by the 2026-08-20 re-cut (quoted below).

**§3 NEC-5 dialect front (the EZNEC half)**

- [x] Card reader: GW (knots), GC (intra-wire taper), EX with
      field-4-as-knot, IS, FR, LD, GD/GN 0, TL/NT, RP.
  - **Record:** #500 (the NEC-5 dialect parser, fourteen mnemonics), #514
    (the fifteenth).
- [ ] The virtual-wire feed idiom (the capture: it is the feed system, not a
      passive anchor — and it breaks the TL-anchor virtualization,
      antennaknobs#157).
  - **Record:** none. The reference is mis-numbered: the TL-anchor assembly
    hang is momwire#157 and the TL-anchor translation antennaknobs#427, both
    closed; antennaknobs#157 is an unrelated site PR.
- [x] Printout renderer satisfying the error convention: launch-stamp echo,
      results-block presence semantics, speaking refusals.
  - **Record:** #501.
- [x] Sweep economics: EZNEC launches one process per frequency point with the
      whole deck regenerated — the warm-server/cross-deck-cache pattern (or a
      SOMMPD.NEX-equivalent persistence) is what makes sweeps usable.
  - **Record:** the #718 arc (#726, #732).
- [x] Dialect landmines already mapped: RD/NRADL fields land on NEC-5
      permeability fields (#444).
  - **Record:** #444 closed 2026-08-21.

**§4 Pillar certification, pitch-grade**

- [ ] Taper: Ward AE6TY's fat-wire tapered dipole (antenna-problem-decks#1)
      run as a certification.
- [ ] Taper: GC-card intra-wire taper end-to-end (parse → mesh → gates).
- [ ] Ground contact: ships as-is with the documented lossy-soil envelope
      (#282's named gap; stage 3 tracked there, deliberately off this critical
      path).
  - **Record:** parked by the 2026-08-20 re-cut as "GN 0 ground-contact
    (by-design, #282)".
- [ ] Optional: razor contact over finite grounds (#398 §5.5 experiment) — the
      NEC-5-twin basis serving NEC-5's signature use case.

**§5 Packaging (early-adopters release)**

- [x] Windows drop-in executable (thin frozen client + server, the SimNEC
      portal pattern, or a bundled build).
  - **Record:** #533.
- [x] Windows CI lane.
  - **Record:** #533 ("byte-gated in CI").
- [ ] Install story a non-developer can follow.

**Deliberate exclusions** (body, 2026-08-19):

> Surface patches (SM/SP/SC); anything #388's inventory marks excluded; radial
> screens if workstream 2 decides against implementing.

**Parked by the re-cut** ([2026-08-20](https://github.com/stevenmburns/momwire/issues/456#issuecomment-5357338142),
Target B — 4nec2):

> Deliberately parked, filed rather than forgotten: the long-tail rung
> (~90.6 %), the NRADL/radial-screen decision, GN 0 ground-contact (by-design,
> #282), refusals in 4nec2's own error grammar, and the NEC-4 slot. None of
> these gate Target A.

**Unscheduled by design** ([2026-08-20](https://github.com/stevenmburns/momwire/issues/456#issuecomment-5351826394)):

> Phase D (importer delegating card semantics) stays deliberately unscheduled
> per the design doc.

**The rule that survives the re-cut** ([2026-08-20](https://github.com/stevenmburns/momwire/issues/456#issuecomment-5357338142)):

> Physics decisions are shared across targets and get taken in whichever
> thrust can **measure them cheapest**, then inherited as routing by the
> others.

**The reference list** ([2026-09-02](https://github.com/stevenmburns/momwire/issues/456#issuecomment-5504335909)):

> Absorbs #388 (closed): its body is the surveyed inventory of refusals the
> reference engines solve, and remains the reference list for this umbrella.

The scored ground truth is antennaknobs'
[`docs/status/2026-08-19-frontend-statement-matrix.md`](https://github.com/stevenmburns/antennaknobs/blob/main/docs/status/2026-08-19-frontend-statement-matrix.md),
regenerated by `scripts/census_4nec2_bundle.py` there.

### How to resume

Take one unticked box or parked item at a time as its own issue, scored
against the statement matrix and #388's inventory rather than reopening the
umbrella.

## Site: consolidate the full engine reference on momwire.dev (partition option C) — [#544](https://github.com/stevenmburns/momwire/issues/544)

### What shipped

- **Option A** — #543 (momwire): the selection matrix, runtime, memory and
  razor-lane facts on `site/src/content/docs/reference/choosing-an-engine.md`.
- **The app side slimmed to match** — antennaknobs#977: the solver guide keeps
  app-side guidance and points at momwire.dev for engine facts.

Nothing of option C beyond page 1 exists: `site/src/content/docs/reference/`
holds `choosing-an-engine`, `deck-grammar-nec2`, `eznec-nec5` and
`portal-usage`, and no `cost-and-memory` or `validation-receipts` page.

### What stays open

The issue calls itself not urgent (body, 2026-08-22):

> Not urgent; option A covers the QRZ-era need. This becomes worth doing when
> the validation story starts being cited engine-side (the pitch release, the
> Arie note) rather than app-side.

Option C's unbuilt pages, as written in the body (2026-08-22):

> 2. **`cost-and-memory`** — split the deep material out of page 1 as it grows:
>    full benchmark tables (today only summarized, with GitHub permalinks into
>    antennaknobs `docs/status/`), the certified memory peaks from the
>    memory-release series (0.29.0 certification, #318/#323 reductions), and
>    the profiling story (`docs/profiling.md` is the in-repo source).
> 3. **`validation-receipts`** — the real work item. antennaknobs'
>    `reference/validation.md` is **generated** by
>    `scripts/build_validation_report.py` from antennaknobs-side data (the wild
>    corpus, the per-case verdicts). Retarget or twin the generator to emit a
>    momwire.dev page: data and script stay in antennaknobs, the generator
>    writes across the submodule boundary into `momwire/site/`, and the
>    antennaknobs page becomes the app-flavored view. Design questions:
>    - one generator emitting two pages (engine-voiced vs app-voiced) vs one
>      canonical page + a thin wrapper;
>    - how the cross-repo write interacts with the two release rituals
>      (momwire's tag-deploy vs antennaknobs'); regeneration must not force
>      lockstep releases;
>    - which receipts are engine-side (formulation parity, corpus census,
>      convergence-at-fixed-mesh) vs app-side (catalog case studies, workbench
>      screenshots).
> 4. antennaknobs' `reference/nec5.md` keeps its app-side driving guide; the
>    formulation/cost material it carries gets the same one-home treatment.
>
> **The rule that governs all of it** (release-ritual lesson): every fact lives
> in exactly one place — duplicated tables drift, and the "sweep
> reference/web.md every release" checklist exists because they drifted
> before.

The scope addition
([2026-08-22](https://github.com/stevenmburns/momwire/issues/544#issuecomment-5377606368)),
also unbuilt:

> Addition to the scope (Steve, 2026-08-21): the engine-reference section
> should use **the composition architecture as the frame for describing
> momwire's components**, not a flat roster of nine engines.
>
> Concretely: the #376 design study (`docs/design/solver-architecture.md`)
> factored the engine into axes that compose — and the reference section
> should present it the same way, because that factoring _is_ the accurate
> description of what the components are:
>
> - **Basis** (pulse, tent, B-spline d1/d2, three-term sinusoid) × **testing
>   rule** (point matching, Galerkin, razor-blade paths) — the nine names are
>   cells of this product, and several "engines" are one class under two names
>   (`razor`/`razor-nec5`, `bspline`/`bspline-d1`). Act V already teaches the
>   matrix as pedagogy; the reference should state it as architecture.
> - **Ground** as a shared composed object, not a per-solver
>   reimplementation: `PotentialGround` / `FieldGround` carry the image,
>   reflection-coefficient weights, and the Sommerfeld remainder once, and a
>   solver consumes them (razor's grounds are ~83 executable lines and zero
>   reflection arithmetic of their own — the #398 pilot's own receipt). Two
>   formulations ⇒ two ground objects is the #376 decision record.
> - **Kernel** (reduced / EK) with the closed-form segment moments in their one
>   shared home (`_kernel_moments`, #425).
> - **The capability registry** (#396): solvers _declare_ what they serve, and
>   refusals quote it. This is the machine-readable source of truth — the
>   serve/refuse tables on the reference pages should be generated from (or at
>   minimum tested against) `capabilities`, the same way the validation page
>   is generated from its data, so the site cannot drift from what the code
>   actually declares.
> - **The seam layer** above all of it, governed by the seam rule
>   (`docs/design/seam-rule.md`): the portals are protocol adapters over the
>   composed engine, and the reference section's structure should make that
>   layering visible — axes, composition objects, registry, seams.
>
> Payoff: the "choosing an engine" page answers _which cell_; the composition
> page answers _why the cells are cheap_ — B×T ≈ zero work once B and T exist
> — which is the engineering claim that distinguishes momwire from a pile of
> solvers, and the natural umbrella under which cost/memory/validation each
> hang.

### How to resume

When a release or a pitch starts citing validation engine-side, answer the
three `validation-receipts` design questions first, since the cross-repo
generator is the part that constrains the other pages.

## ACA / ArrayBlock under Sommerfeld ground with C++ kernels — [#981](https://github.com/stevenmburns/momwire/issues/981)

Umbrella opened 2026-09-08: measure the rank first, then the far-block
evaluator, then the below-remainder projection.

### What shipped

- **Step 1's prerequisites** — #979 (#973's sampled residual + dense fallback
  on the global Sommerfeld remainder) and #977's crossover ladder, banked as
  antennaknobs#1300 (206 cells). #977 itself is still open.
- **Step 3a** — #1015 (`a672a57`: stop re-marshalling the ACA's invariant
  side; `f761c04`: take an ACA row as a transposed column by default). Both
  leave the rank unchanged and the sampled residual equal to six digits;
  together they took the n = 1992 build from 20.80 s to 18.68 s (2026-09-09
  comment).
- **The step-2 and step-4 measurements, as records** — #982 (`5e972ba`,
  `7a45981`).
- **The buried-screen half, closed by measurement** — #983 (`de622a7`): the
  #914 levers were already on main (`f537080` unit 1, `130c7ee` unit 2), and
  momwire runs the 48-radial screen in 19.25 s against NEC-5's 21.59 s.
- **Diagnostics** — #1020 (`_last_somm_z_residual`, published, gating nothing)
  and `94a14ad` (`HMatrixSolver.partition_report()`, #1021).

### Step by step

| step | disposition | record |
|---|---|---|
| 1. land #973 + #977 | landed | #979; antennaknobs#1300 |
| 2. measure the below-remainder block structure | **rejected on measurement** — low rank, but the caller already blocks observers into thin `30 × 15,588` calls | [2026-09-08 comment](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5593154423); [`2026-09-08-below-remainder-block-structure.md`](2026-09-08-below-remainder-block-structure.md) |
| 3a. measure and hoist (no C++) | **merged** | #1015 |
| 3b. blocked ACA row sampling | **rejected on measurement** — the ACA term is slower at every block size on every deck | [2026-09-10 comment](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5621318275); branch `perf/981-blocked-aca` at `3fc1f4b`; #1028 closed not planned 2026-09-14; census antennaknobs#1370 |
| 3c. C++ row/column evaluator | not built | [2026-09-09 comment](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5605114794) |
| 4. ACA on the below-remainder projection | **struck** — and its replacement (one square caller) measured no as well | [2026-09-08 plan update](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5593167231); [`2026-09-08-square-below-remainder-caller.md`](2026-09-08-square-below-remainder-caller.md) |
| 5. ArrayBlock under Sommerfeld through the same evaluator | **moot** | the 2026-09-10 decision below |
| 6. serve: hosted caps, advisories, cost column | **moot** | the 2026-09-10 decision below |

"3b" names two things in the thread. The
[2026-09-09 step-3 plan](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5605114794)
used it for a per-far-block ACA partition; the
[2026-09-09 correction](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5609089556)
routed that to #973 ("Rank → #973, regime B"), and "3b" was re-used for
blocked row sampling. `git log` on main shows no per-far-block partition
change for the Sommerfeld remainder; the nearest is `94a14ad`, #1021's report
on the partition as it stands.

Steps 5–6 are moot under the
[2026-09-10 decision](https://github.com/stevenmburns/momwire/issues/1019#issuecomment-5621225297),
recorded on #1019 (the accuracy split of #973; #1028's closing comment cites it
as #973):

> Decision (Steve, 2026-09-10): **the H-matrix lane stays opt-in and no new bar
> is applied.** The residual stays a published diagnostic (#1020), the dense
> fallback keeps catching the gross misses, and the antennaknobs solver page
> now states the census result (32 of 99 catalog decks above 1e-6 vs dense at
> the default tolerance, worst 3e-4; skyloop_lmatch nseg 14 at 4.7e-4 under
> the bar) and that a quoted number comes from the dense lane.

The #973 stagnation record this umbrella rests on is
[`2026-09-08-aca-stagnation-sommerfeld-remainder.md`](2026-09-08-aca-stagnation-sommerfeld-remainder.md).

### What stays open

The unbuilt steps, as written in the body (2026-09-08):

> 5. **ArrayBlock under Sommerfeld through the same evaluator**, checking
>    #978's floor does not become binding.
> 6. **Serve:** hosted `swept_mem_mb` caps, advisories, antennaknobs#1265's
>    cost column once the ladder exists.

Step 3c
([2026-09-09](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5605114794)):

> **3c — the C++ row/column evaluator, scoped by what 3a and 3b leave.** Via
> the #903 column twin, with the #973 sampled check built in. I would rather
> write this kernel against a measured residue than against the 30× headline.

and the verdict on it
([2026-09-09](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5609067110)),
which leaned on the batching that 3b then rejected:

> **The C++ evaluator looks unnecessary either way** — batching gets to 1.1×
> bulk without one.

The crossover question
([2026-09-09](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5609089556)):

> The crossover question stays **open**, not closed on my earlier reasoning:
> it gets re-measured on rhombic and lpda after the speedup lands, and X is
> stated from that measurement rather than from an Amdahl bound computed on the
> wrong class.

The scheduled order
([2026-09-10](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5609592364)),
of which #1011 is closed, the #973 partition change is unbuilt, and the last
step was rejected:

> **SCHEDULED, not parked** (Steve, 2026-09-10), and the order is deliberate:
> **#1011 first, then the #973 partition change for the compact
> ground-coupled class, then this.**

The probe's blind band, for whoever builds a kernel-side check
([2026-09-09](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5595170152)):

> **the sampled probe cannot see a mild degradation.** Measured on the #979
> mesh ladder, a 232-basis rung probes 2.198e-03 while carrying |ΔZ|/|Z|
> 7.71e-04, and the worst healthy catalog deck probes 2.52e-03 with |ΔZ|
> 2.6e-06 — so health and a 7.7e-4 error are not separable by one sample set
> at any threshold. Worth a second independent sample set, or a two-tolerance
> check (factorise at tol and at tol/100 and compare), when the kernel is
> built. The probe's _value_ also drifts with the pivot path across revisions
> (that same deck read 1.66e-03 one revision earlier with its |ΔZ| unchanged),
> so a kernel-side check should not assume the number is stable.

The buried screen's remaining lever
([2026-09-08](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5593595816)):

> The buried screen is kernel-bound, not glue-bound; #982 already ruled out
> evaluating fewer pairs by ACA. Nothing lever-shaped is left; further gain
> needs a new idea about the projection kernel itself.

The one correctness result 3b produced
([2026-09-10](https://github.com/stevenmburns/momwire/issues/981#issuecomment-5621318275)),
not adopted (#1028, 2026-09-14: "the bad tail 100× better, 19 decks worse, 2
new stagnations"):

> At `b = 8`, `loops.skyloop_lmatch` **converges where `b = 1` stagnates** —
> rank 34 and residual 2.1e-05, against rank 94 = N and residual 1.16 with the
> dense fallback carrying it. Sampling several rows at once escapes the trapped
> pivot subspace #973 describes.

### How to resume

Only if the H-matrix lane is proposed for a default path: re-read the
2026-09-10 decision and the three `2026-09-08` records above, re-measure the
rhombic / lpda crossover on today's main, and file each new step as its own
issue.

## Decided caps

Two refusals decided permanent. Neither is a gap waiting for work.

### The transmitted z′ ladder stops at 0.25 λ_m — [#666](https://github.com/stevenmburns/momwire/issues/666)

`_ZPRIME_MAX_LAMBDA_M = 0.25` in `src/momwire/_sommerfeld_transmitted.py`
(line 207 at `6549550`), in in-medium wavelengths λ_m = 2π/|k_m|. `serve_plan`
in `src/momwire/_below_interface.py` enforces it before the fill starts,
raising `BURIED_DEPTH_REFUSAL` with the deck's own depth, the cap in metres and
λ_m. Past a quarter λ_m the two-ray (λ₁/λ₂ saddle) structure of the transmitted
integral returns and the ladder's single phase divide-out no longer flattens
it (#524 phase 0).

[Decided 2026-09-14](https://github.com/stevenmburns/momwire/issues/666#issuecomment-5669638402),
issue closed:

> Decided (Steve, 2026-09-14): 0.25 λ_m stays the permanent depth cap of the
> transmitted z′ ladder, and deep crossing decks keep refusing by name at plan
> time.

The refusal text still offers "or extend the ladder — about 8 extra rungs per
additional quarter lambda_m" as a remedy; under this decision that remedy is
not on offer.

### EK with junction enrichment stays refused — [#271](https://github.com/stevenmburns/momwire/issues/271)

`extended_kernel=True` with `use_singular_enrichment=True` raises
`_ENRICHMENT_EXTENDED_KERNEL_REFUSAL` (`src/momwire/bspline.py`), registered
under `"extended_kernel+singular_enrichment"` in the B-spline capability
declaration's `refusals`. The
argument is NEC's own: the enrichment DOFs exist only at K ≥ 3 junctions,
exactly where NEC's IND=2 gating turns EK off.

[Decided 2026-09-14](https://github.com/stevenmburns/momwire/issues/271#issuecomment-5669638028),
issue closed:

> Decided (Steve, 2026-09-14): the refusal of EK combined with junction
> enrichment is permanent, on NEC's own IND=2 argument, and the tube expansion
> for the enrichment shapes won't be derived. The under-a-medium case in #570
> is related.

The refusal text still opens "not supported yet"; under this decision it is
permanent.
