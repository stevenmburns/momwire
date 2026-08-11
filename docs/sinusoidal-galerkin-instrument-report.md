# The sinusoidal-Galerkin instrument: what the fourth cell reads

**momwire#182, milestone M6.** The deliverable of the whole issue was never a
faster solver — it was an *instrument*. momwire already had a point-matched
sinusoidal solver and a Galerkin B-spline solver, so every disagreement
between them confounded two changes at once (the basis AND the testing) and
none of antennaknobs' open discrepancy issues could be attributed. M1–M5b
built the missing cell: the **same** three-term sinusoidal basis, tested
variationally. This document is what it reads.

Scope: antennaknobs#521 (the no-mutual residue cluster, plus its helix
control) and antennaknobs#478 (the near-open high-Q class), free space, with
the M5/M5b **feed- and port-model** column added.

---

## 0. Reproducing this

Everything here comes out of one harness:

```
# the residue-cluster / near-open sweep (§3-§5)   ~17 min, 11 designs
python scripts/m6_residue_cluster.py

# the feed-model axis on the canonical dipole (§6)   ~9 s
python scripts/m6_residue_cluster.py --dipole-feed-model
```

The geometries are antennaknobs catalog designs *snapshotted* into
`scripts/m6_residue_cluster_geoms.json` — so this reproduces from a momwire
checkout alone, and so a later retune of a catalog design cannot silently
move these numbers. Re-dump (from the antennaknobs superproject, which is
where the designs live) with `--dump`.

Pinned in CI: `tests/test_m6_instrument_report.py`, which re-derives §6's
feed-model claim, §3's headline row, and the snapshot's coverage from the
same harness — so a regression makes the report red rather than merely stale.
The port-model column of §7 is pinned on the antennaknobs side
(`tests/test_balanced_line_physics.py`, section 8).

---

## 1. Three axes, not one

The census that produced antennaknobs#521 compared two columns, `sin` and
`bs2`, and called their difference "the basis gap". Those two columns differ
in **three** things:

| axis | `sin` | `bs2` |
|---|---|---|
| **basis** | NEC's three-term const+sin+cos | quadratic B-spline |
| **testing** | point matching (collocation) | Galerkin |
| **feed model** | NEC's delta gap, `E = V/Δ` over the whole feed segment | a zero-width (point) gap |

M1–M5b make all three separable. This report uses four columns, always on
**one mesh per rung** — so a rung compares four *schemes*, never four
problems:

| column | basis | testing | feed |
|---|---|---|---|
| `coll` | sinusoidal | collocation | segment-wide Δ-gap |
| `gal` | sinusoidal | **Galerkin** | segment-wide Δ-gap |
| `ptgap` | sinusoidal | Galerkin | **point gap** |
| `bs2` | **B-spline d=2** | Galerkin | point gap |

Each adjacent pair isolates exactly one axis: `coll`↔`gal` is the **testing**,
`gal`↔`ptgap` is the **feed model**, `ptgap`↔`bs2` is the **basis**.

`ptgap` is a ten-line research subclass in the harness
(`m6_residue_cluster.PointGapGalerkin`), not a solver option. A point gap's
Galerkin drive column collapses on the delta to `−V·f_i(s0)`, which is the
same basis-evaluation vector the default `feed_readout="centre"` already
reads — so drive and readout are exact duals, its Y is machine-symmetric with
no opt-in, and none of M3's payoff is traded. It is deliberately not adopted
into the solver: its job is to *measure* the feed-model contribution, not to
become a fifth cell (and adopting it would re-litigate every M2–M4 number).

---

## 2. Constraints this report is held to

Carried forward from the milestone record, because each was learned by being
got wrong first:

1. **Per-scheme mesh convergence is confirmed before anything is attributed.**
   M2's 11% junction "basis effect" evaporated under refinement — bspline was
   converged at the rung, the sinusoidal schemes were not. Every table below
   carries each scheme's own last-step drift beside the cross-scheme gap, and
   where a scheme is not converged, this report says so instead of attributing.
2. **Every error names its reference.** M3 established that the delta-gap
   dipole has no mesh limit at all (the gap's stored near-field energy grows
   like ln(1/h)), so "the converged value" is not a number this model has.
   Every figure here is `|ΔZ| / |Z_ref|` with `Z_ref` named — almost always
   "`bs2` at rung N", never an implied limit.
3. **Fixed feed placement across the sweep.** A delta gap snaps to the nearest
   segment centre, so refining the mesh can translate the *source* by up to
   h/2 — an O(h) perturbation of the **problem**, larger than the difference
   between two testings of it. **Measured: the feed-segment centre moves
   0.0000 m across the ladder on every design in this report**, so the
   per-scheme convergence readings are convergence and not feed drift. (The
   cross-scheme gaps are immune regardless — all four columns share one mesh
   and one feed segment at each rung.)
4. **No finite-ground reads on ground-CONTACT geometry, on either sinusoidal
   solver.** M4 pinned that as an inherited #151 defect (the ground-connected
   basis completes the end current with an exact mirror image, which a
   Fresnel/Sommerfeld image is not) and it was not this issue's to fix. It has
   since been fixed by momwire#282, which removes the contact charge that
   mismatch leaves on the plane: both sinusoidal solvers now converge there
   (0.4% over a 7x refinement) and agree with each other to 0.3%, though they
   sit a fixed cross-basis distance from the b-spline reference and the
   EK-on contact fill over a lossy ground is still unstable. This report is
   free space throughout, which sidesteps all of it regardless.
5. **The `_bridged_z` oracle's ~1.4% linear-in-δ extrapolation residue is the
   oracle's**, characterized in M5b and off-limits for re-attribution to any
   solver. It is not used as a reference here.
6. **Multiport comparisons are stated above the visibility floor.** The fill's
   8.3e-12 reciprocity floor amplifies to ~1e-8 through a port solve (M5b), so
   nothing below ~1e-8 is claimed about a port network.

---

## 3. antennaknobs#521's residue cluster: **testing**, decisively

The cluster is T/X-junction-heavy geometry whose `sin`↔`bs2` disagreement is
concentrated in reactance. Swapping only the testing scheme resolves it.

Free space, driving-point Z at feed 0, at the finest rung each design reached
under the harness's 2000-segment cap (§11 — the cap is a memory ceiling, not a
patience one). Gaps are against **`bs2` at that same rung**; "step" is that
rung against the one below it — each scheme's own mesh-convergence check.

| design | finest N | segs | `coll`↔`bs2` | `gal`↔`bs2` | `ptgap`↔`bs2` | `coll` step | `gal` step | `bs2` step |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| specialty.hentenna | 321 | 1835 | **12.89 %** | 0.01 % | 0.008 % | 1.61 % | 0.02 % | 0.04 % |
| specialty.hentenna_slant | 321 | 1869 | 1.20 % | 0.03 % | 0.026 % | 0.54 % | 0.02 % | 0.09 % |
| arrays.hentenna_array | 161 | 1898 | **10.67 %** | 0.04 % | 0.024 % | 1.09 % | 0.07 % | 0.04 % |
| specialty.hourglass | 161 | 1193 | **9.89 %** | 0.04 % | 0.033 % | 0.78 % | 0.04 % | 0.07 % |
| specialty.hourglass_slant | 161 | 1177 | **16.66 %** | 0.09 % | 0.084 % | 1.51 % | 0.04 % | 0.13 % |
| arrays.hourglass_array | 81 | 1134 | **10.16 %** | 0.08 % | 0.045 % | 1.10 % | 0.02 % | 0.02 % |
| broadband.discone | 81 | 1395 | **23.02 %** | 0.33 % | 0.326 % | 11.20 % | 0.10 % | 0.33 % |
| specialty.bowtie | 321 | 1313 | 0.05 % | 0.05 % | 0.017 % | 0.07 % | 0.07 % | 0.02 % |

**The reading.** On every member of the cluster the two Galerkin schemes are
mesh-converged (their own last steps are at most 0.33 %) and agree with each
other to within that same drift, while collocation is *not* converged (steps
0.5–11 %) and sits 1–23 % away.
What #521 filed as a basis gap is **the testing scheme**, full stop.

Two rows deserve their own sentence:

- **`broadband.discone`** was #521's "one genuinely-uniform survivor — the
  cleanest remaining test of the junction-degree mechanism", an apex fan of
  degree 8+. It is the most extreme row here: collocation is 23 % away and its
  own ladder is not merely slow but *non-monotone* (12.96 − 22.30j →
  15.07 − 26.66j → 14.21 − 23.69j over N = 21/41/81), an 11.2 % last step. So
  the apex-fan suspicion is confirmed as a *collocation* effect: point matching
  at a high-degree junction is where it breaks, and the Galerkin testing of the
  same basis has no such trouble. Note the honest limit on the Galerkin claim
  here — `gal`↔`bs2` is 0.33 %, the same size as `bs2`'s own remaining drift,
  so the right statement is *indistinguishable within convergence*, not "agree
  to 0.33 %".
- **`specialty.hentenna_slant`** reads only 1.20 %, an order below its
  siblings. antennaknobs#521's own scope refresh had already scored it clean
  after the #525 conversion and taken it OUT of the cluster; this run agrees,
  and it is kept in the table as a second negative control.

The mechanism is visible directly in the raw ladder — take `specialty.hentenna`:

| N | segs | `coll` | `gal` | `bs2` |
|--:|--:|---|---|---|
| 21 | 119 | 42.44 + 29.23j | 43.01 + **38.90j** | 43.05 + **38.59j** |
| 41 | 237 | 43.21 + 28.35j | 43.07 + 38.91j | 43.09 + 38.79j |
| 81 | 463 | 43.39 + 29.64j | 43.07 + 38.91j | 43.09 + 38.84j |
| 161 | 921 | 43.38 + 30.57j | 43.09 + 38.91j | 43.10 + 38.89j |
| 321 | 1835 | 43.38 + **31.43j** | 43.11 + 38.92j | 43.11 + 38.91j |

The Galerkin testing of the sinusoidal basis has the answer at the **coarsest
rung on the ladder** — 38.90j at 119 segments, against 38.92j at 1835 — while
collocation crawls 29.2 → 31.4 over a 15× refinement and is still 7.5 Ω short.
This is #521's own diagnosis ("sin creeps toward bs2's X but never arrives
inside the affordable ladder") with the cause named.

**Rate gap or limit gap?** Both must be stated or the write-up overstates
(M3). Collocation's sequence is monotone and its steps are shrinking, so
nothing here says it converges to a *different* answer — this is a
convergence-**rate** gap, as every attribution in this tree has been. But the
rate gap is not a footnote on this class: **collocation at N=321 has not
reached what the Galerkin testing has at N=21**, so the mesh advantage is
larger than the 15× the ladder can bound, and inside the affordable mesh the
two schemes give different engineering answers. The limit gap, so far as the
ladder shows one, is ~0.

**bowtie is already resolved** — 0.05 % across the board, consistent with
antennaknobs#521's own scope refresh, which took it off the cluster after the
#537 double-density fix. It is included as a negative control and behaves like
one.

### 3.1 This closes an arbitration momwire has had open since PR #45

`scripts/compare_hentenna_solvers.py` has sat in this repo since 2026-06 with
exactly this geometry and a stated open question: three polynomial-family
columns (tent, B-spline d=2, d=2 + enrichment) **converge** on the hentenna,
while the two three-term-basis columns (this repo's `sin` and PyNEC)
**anti-converge** — fitted X-convergence exponents of −0.46 and −0.50, i.e.
walking away from their own fine-mesh answer. The arbitration argument was
"the polynomial trio agrees, the three-term pair tracks each other's drift, so
believe the polynomial trio" — sound, but it left *why* the three-term family
drifts undetermined, and the natural reading was that the basis was at fault.

The fourth cell answers it. Hold the three-term basis, change only the
testing, and the column joins the polynomial trio: 38.92j against the trio's
value, converged from the coarsest rung. **The drift was the point matching,
not the basis.** (That script solves a slightly different hentenna
parameterization, so its absolute numbers — the trio's 43.05 + 38.84j — are
not directly comparable to the table above; the *behaviour* is what
transfers.) It also explains the PyNEC lockstep the script noticed: PyNEC is
the same basis point-matched with the same delta gap, so it inherits the same
drift, and its agreement with `sin` was never independent evidence about this
geometry.

---

## 4. The helix control: **not resolved, and honestly so**

`specialty.helix` was folded into #521 as the *control* case — 30.6 % apart,
worst row in the census, and mechanically distinct from the rest of the
cluster because a helix has no junctions at all. The suspect there was
curvature discretization.

The first thing the instrument finds is that the census could not have been
reading convergence on this design at all: **`specialty.helix` ignores
`nominal_nsegs`.** Its builder emits one MoM segment per winding chord, so the
mesh knob is `pts_per_turn`, and the census's N=21…641 ladder was re-solving
one identical 53-segment mesh at every rung — which is exactly why #521
recorded it as "byte-identical across both runs". The arithmetic confirms it:
this harness at `pts_per_turn=12` (the default, 53 segments) reads
13.76 + 2.91j vs `bs2`'s 13.51 + 7.66j, i.e. **30.6 % apart** — #521's exact
census figure, reproduced at a rung the census believed was N=641.

Swept on the knob that actually meshes it (`pts_per_turn` 8 → 32, 43 → 107
segments), collocation is 21.1 % from `bs2` and the Galerkin pair agrees to
0.80 %. That *looks* like the same verdict as §3. **It is not reportable**,
because no scheme is mesh-converged: the last-step drifts are 9.25 % (`coll`),
8.20 % (`gal`) and 8.28 % (`bs2`) — two orders larger than anywhere else in
this report, and larger than several of the gaps being compared. Under
constraint (1) that forbids an attribution.

What can be said, and is worth saying, is narrower and more useful:

- The helix's problem is **not** the sinusoidal basis and **not** the testing
  scheme. All four columns move together, in lockstep, by ~8-9 % per rung —
  which is the signature of the *geometry* being under-resolved, not of a
  discretization scheme being wrong. A chorded helix at 8–32 points per turn
  is a coarse polygon approximation of a curve, and the impedance is still
  chasing the polygon.
- So the helix is a **meshing** issue in antennaknobs, not a momwire basis
  issue: it needs a `pts_per_turn` ladder wired to `nominal_nsegs` (or a
  documented per-design mesh knob) before any solver comparison on it means
  anything. Filing that back to #521 is the actionable outcome.
- #521 hoped the helix would be the case where the junction-degree mechanism
  *fails* to explain the residue. It is — but for a reason that disqualifies it
  as evidence either way.

> **Resolved 2026-07-31:** the meshing issue was fixed in antennaknobs (#630,
> #636 — the helix now meshes at the design density like every other design),
> and the re-instrumented control is scored in the full-depth re-run (§13):
> **no basis discrepancy** (`gal↔bs2` 0.01 %, matched-feed 0.005 % at N=321).

---

## 5. antennaknobs#478's near-open class: **the feed model**

`wire.lazy_h` and `wire.vbeam` are the surviving near-open high-Q members.
#478's own finding was that both bases "crawl together" toward the limit —
i.e. basis-independent — and that a higher-order basis "helps modestly,
doesn't solve". The instrument agrees, and then says something new.

Both at N=161 (their N=321 rungs, 2570 and 2563 segments, are over the
memory cap — §11). Gaps against `bs2` at the same rung:

| design | finest N | segs | `coll`↔`bs2` | `gal`↔`bs2` | `ptgap`↔`bs2` | `coll` step | `gal` step | `bs2` step |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| wire.lazy_h | 161 | 1290 | 1.25 % | 1.21 % | **0.022 %** | 4.03 % | 4.00 % | 2.95 % |
| wire.vbeam | 161 | 1287 | 0.78 % | 0.74 % | **0.010 %** | 1.55 % | 1.33 % | 1.01 % |

**The reading.** Swapping the testing buys almost nothing here — `coll`↔`bs2`
and `gal`↔`bs2` are the same number to two digits, which is the exact opposite
of §3 and confirms #478's "no numerical remedy applies" as far as the *testing*
axis goes. But matching the **feed model** collapses the residual by 55× and
74×: `ptgap`↔`bs2` is 0.022 % and 0.010 %. On this class the whole
sinusoidal ↔ B-spline disagreement is the delta gap — not the basis and not
the testing.

That is a real correction to #478's framing. The remaining error is genuinely
*shared* — every scheme's own last step is 1–4 % at the finest affordable rung
(4.03 %/4.00 %/2.95 % on `lazy_h`), i.e. all four columns are equally far from
converged and equally close to each other. #478's "expect the last percent to
be physical, and budget fine mesh for near-open structures" survives intact;
what changes is that the *inter-basis* part of it was a feed-model artifact
and is now accounted for.

Note the interaction with §3: on the residue cluster the feed model is
invisible (`gal` and `ptgap` agree to ≤0.03 % at their finest rungs) because
those designs are
low-Q; on a 4-5 kΩ near-open feed the same small perturbation of the source is
amplified into the dominant term. Same solver, opposite readings — which is
what makes the third axis worth having.

---

## 6. The feed-model axis, isolated

The canonical 0.962 λ/2 dipole, free space, a = 0.5 mm — the M1–M3 validation
geometry, where nothing is confounded by junctions or Q:

| N | `coll` | `gal` | `ptgap` | `bs2` |
|--:|---|---|---|---|
| 161 | 69.631109 − 18.185891j | 69.643869 − 18.097471j | 69.634327 − 18.112452j | 69.634329 − 18.112432j |
| 321 | 69.631876 − 18.107822j | 69.639094 − 18.056294j | 69.633780 − 18.065312j | 69.633780 − 18.065315j |

Against `bs2` at the same N:

| N | `coll`↔`bs2` | `gal`↔`bs2` | `ptgap`↔`bs2` |
|--:|--:|--:|--:|
| 161 | 1.02e-3 | 2.47e-4 | **2.81e-7** |
| 321 | 5.92e-4 | 1.46e-4 | **4.17e-8** |

**Matching the feed removes ~3500× of the sinusoidal↔B-spline difference.**
What is left — 4.2e-8 at N=321 — is the actual basis difference between NEC's
three-term basis and a quadratic B-spline on this geometry, and it is four
orders of magnitude below anything M2/M3 ever filed as a basis effect. Note
also the *direction*: the matched pair tightens with refinement (2.8e-7 →
4.2e-8) while the mismatched pair does not (2.5e-4 → 1.5e-4), because the two
feed models converge to different answers, not to the same one.

This is the concrete form of M5's finding, and it is the sharpest single
result in the report: **on a clean geometry the two bases are, for practical
purposes, the same basis.** Every historical "sin vs bs2" number in this tree
that was not accompanied by a feed-model control was measuring the source
model as much as the basis.

---

## 7. The port-model axis: `wire.sterba_bl` end-to-end

M5b's junction ports make a fourth comparison possible that no other cell can
make: two implementations of the *same port model* sharing no basis, no
testing and no port algebra.

`wire.sterba_bl` is the case that forced it. Its 16 ports are all one-member
junctions at lone conductor ends (the riser metal is deleted by design), so
genuine net inflow is the only mode of use — antennaknobs#608 refuted every
gap-family substitute. Until M5b, `BSplineSolver` was the only implementation,
which made the design's entire physics story rest on a single port
implementation with no independent check.

Free space, catalog default `n_cells=3` (16 junction ports):

| | Z (driving point) | peak gain | azimuth |
|---|---|--:|--:|
| `BSplineSolver` (mixed-potential, Lagrange-multiplier port on a KCL row) | 673.293 + 385.592j | 10.6064 dBi | 180° |
| `SinusoidalGalerkinSolver` (field-based, node charge held outside the reaction integral) | 672.055 + 387.371j | 10.6054 dBi | 180° |
| **difference** | **0.28 %** | **0.001 dB** | — |

At one bay (`n_cells=1`): 315.518 + 138.156j vs 315.338 + 138.840j, 0.20 %.

Both readouts matter. Z is the port network's own output; gain is what the
resulting current distribution *does*. A port-model error that happened to
cancel in one would still show in the other.

The section-5 physics claim of `test_balanced_line_physics.py` re-derives on
the independent formulation too: removing the common-mode path costs
10.605 → 5.518 dBi and swings the beam from 180° to 35°, against the B-spline
solver's own 10.606 → 5.519 on the same pair of builds. That is the file's
most load-bearing claim — that `zcomm` is a *topological* switch, a statement
about the port boundary condition — and it is now checked by a second port
implementation rather than by one.

**Stated above the visibility floor:** the fill's 8.3e-12 reciprocity floor
amplifies to ~1e-8 through a port solve (M5b), so a 0.28 % agreement is five
and a half orders above anything the floor could explain. Backing this up at
the matrix level, M5b measured the two solvers' port networks agreeing
**entrywise** to 3.4e-5 / 3.9e-6 on the two-member and one-member `PortAtEnd`
topologies — tighter than any gap attributed to the basis anywhere in M3's
table.

**Scope, stated openly:** free space only. M5b scoped junction ports over any
ground out (the node charge's *image* is not removed yet, so part of the M5
blocker would survive), and the solver refuses rather than approximating.
(momwire#191 has since lifted that for a **PEC** ground — §14; the finite
grounds this section's catalog check needs still refuse.) So
`test_catalog_curtain_gain_over_average_ground` — the one `sterba_bl` catalog
check that runs over a ground — has **no** sinusoidal-Galerkin column. The
refusal itself is pinned by a test so the omission cannot be read as an
untested path.

---

## 8. Summary of attributions

| class | designs | axis | evidence |
|---|---|---|---|
| #521 residue cluster | hentenna ×2, hentenna_array, hourglass ×2, hourglass_array, discone | **testing** | Galerkin pair converged and agreeing within its own drift (≤0.33 %); collocation 1–23 % away and not converged |
| #521, resolved earlier | bowtie | none left | 0.05 % on every axis |
| #521 control | helix | **unattributable** — geometry under-resolved | all four columns drifting 8–9 %/rung; and the census's mesh knob never meshed it |
| #478 near-open | lazy_h, vbeam | **feed model** | testing buys ~0.04 %; matching the feed collapses 1.2 % → 0.02 % |
| clean-geometry basis | dipole | **basis ≈ 0** | 4.2e-8 with feed matched |
| port model | sterba_bl (16 junction ports) | **agreement**, not a gap | 0.28 % in Z, 0.001 dB in gain, across two formulations |

**The one-line version:** almost nothing this tree called a "basis gap" was a
basis gap. On junction-heavy geometry it is the *testing scheme*; on near-open
geometry it is the *feed model*; on a clean geometry the two bases agree to
4e-8. The B-spline basis' practical advantage over NEC's three-term basis, as
measured here, is essentially not a basis advantage at all — it is that
`BSplineSolver` happens to be Galerkin *and* point-fed.

---

## 9. What this report does not say

- **It does not say collocation is wrong.** The point-matched solver keeps its
  NEC-parity role (it tracks NEC-2 to ~0.08 %, which is the shared-heritage
  artifact this issue named a non-goal). What §3 says is that on junction-heavy
  geometry it needs a mesh the catalog cannot afford. Its numbers on this class
  should be read as under-resolved, not as an alternative answer.
- **It does not say the schemes converge to different limits.** Every
  attribution here is a convergence-*rate* statement, consistent with M2's and
  M3's findings. Where a limit is visible, the schemes share it.
- **It says nothing about grounds.** Free space throughout, deliberately —
  see constraint (4). The M4 measurements stand on their own: the ground adds
  nothing to the sin↔bs2 gap, and the testing payoff is ground-independent.
- **It says nothing about the helix's physics.** §4 is a refusal to attribute,
  not an attribution.

---

## 10. Exclusions

| excluded | why |
|---|---|
| `wire.expanded_lazy_h` (#478) | A **network** design: its two feeds sit at 0 V and the driving-point impedance is produced by antennaknobs' `NetworkReducer` composing a TL network onto the solver's Y matrix. That number is a property of the reduction, not of a bare momwire solve, and the geometry snapshot deliberately carries solver kwargs only — reading it here gives 0/0. It was #478's *control* member anyway (converged at N=21 on both bases). |
| `arrays.delta_looparray_with_tls` (#478) | Same network-design reason, and independently reclassified off #478 as a PyNEC-on-TL-networks artifact: both momwire bases are flat at 55 − 3.5j across the whole ladder while PyNEC spikes at particular segmentations. There is no sin↔bs2 gap here to attribute. |
| `wire.zepp` (#478) | Already resolved before this milestone (PR #485, mesh-stable distributed port; all three engines agree). |
| every ground read | Constraint (4). A finite ground on a ground-contact wire is an inherited defect on both sinusoidal solvers, and junction ports over any ground are scoped out of M5b. Rather than bend either, this report is free space and says so. |
| `specialty.helix`'s attribution | Reported (§4) but explicitly **not attributed** — no scheme is mesh-converged on the affordable ladder. |
| the finest rungs of 6 designs | Over the 2000-segment memory ceiling (§11), recorded as skipped by the harness. Every attribution is made at a rung where the Galerkin columns' own convergence is shown. |

---

## 11. Cost, and the rungs this run reached

The Galerkin fill is pure Python and scales worse than the point-matched one:
on `specialty.hentenna` at 1835 segments a `gal` solve is ~59 s against
`coll`'s ~0.7 s and `bs2`'s ~1.1 s. That is the honest price of the fourth
cell today (no C++ accelerator — a standing rule of the milestone), and it is
why the solver is registered in antennaknobs as an instrument column rather
than an interactive backend.

**The 2000-segment cap is a memory ceiling, not a patience one**, and that is
worth recording as a finding in its own right. The near-pair quadrature
workspace is O(N²·n_qp) complex, so a rung that fills in ~60 s at 1898
segments is OOM-killed at 2379 on a 32 GiB machine — the first attempt at this
sweep raised the cap to 2800 and died there. Waiting longer does not help; the
fill has to be blocked or accelerated, which momwire#182's Python-first
standing rule puts out of scope.

The consequence is stated rather than papered over: **the exact finest rungs
antennaknobs#521 and #478 quoted are partly out of reach here** — `discone` at
N=161 (2767 segs), `hourglass`/`hourglass_slant` at N=321 (2379/2347),
`hourglass_array` at N=161 (2262), `lazy_h`/`vbeam` at N=321 (2570/2563).
Those rungs are *recorded as skipped* by the harness, never silently dropped
or substituted, and every attribution above is made at a rung where the
Galerkin columns' own convergence is demonstrated (§2 constraint 1) — which is
what makes the coarser rung sufficient.

Full sweep, 11 designs, 4 columns per rung: **16m54s** wall clock (measured
2026-07-30 on a 32 GiB / 8-core box; every table above is that run's
verbatim output).

> **Postscript (2026-07-31, #194):** this section describes the run as made.
> Both halves of the "blocked or accelerated" sentence have since landed —
> the numpy fill is blocked over test segments (the O(N²·n_qp) workspace and
> its 2000-segment ceiling are gone; `wire.lazy_h` at N=321 / 2570 segments
> runs in 3.9 GiB) and a fused C++ far fill serves the plain-projected
> blocks (`gal` on that rung: ~70 s numpy → ~7.5 s). The skipped finest
> rungs above are re-runnable; the numbers in this report remain the
> 2026-07-30 run's verbatim output.

---

## 12. Follow-ups this report generates

1. **antennaknobs#521** — the T/X-junction cluster is closed as a *testing*
   effect. The catalog's fine-mesh reference values for hentenna/hourglass are
   now scoreable: solve them on `momwire:sinusoidal-galerkin` or `bspline`,
   and read the `sinusoidal` column as under-resolved rather than as a
   competing answer.
2. **antennaknobs#521, helix** — needs a mesh knob before it can be scored at
   all. `nominal_nsegs` does not mesh it.
3. **antennaknobs#478** — the *inter-basis* part of the near-open residual is
   the feed model and is now explained; the shared whole-structure part stands
   as filed.
4. **momwire** — junction ports over a PEC ground would be exact and cheap
   (the node charge's image is a mirror of a term already removed); that would
   let §7's comparison run over the ground where `sterba_bl` is actually used.
   *(Landed as momwire#191 — §14. The finite grounds still refuse.)*
5. **momwire** — the point-gap feed model is measurably closer to `bspline`
   and self-dual at no cost to the M3 payoff. Whether it should become an
   option on `SinusoidalGalerkinSolver` is a real question, deliberately not
   answered here: it would re-baseline every M2–M4 number, so it needs its own
   issue and its own gate.
   *(Landed as momwire#192 — §16. An option, `feed_model="point"`; the default
   is still the segment-wide gap, and §16 inventories what a flip would move.)*

---

## 13. Addendum (2026-07-31): the full-depth re-run

momwire#194 landed between the runs: the numpy far fill is now blocked over
test segments and the plain-projected blocks run on a fused C++ kernel, so
the §11 memory ceiling is gone. The harness's default cap moved 2000 → 4000
segments, and the helix control was re-instrumented after the antennaknobs
fixes §4 called for (#630 made the helix mesh at the design density; #636
split it into `faceted_helix`/`continuous_helix`): the control is now
`specialty.faceted_helix` on the standard `nominal_nsegs` ladder. The
snapshot was re-dumped for that row alone — **every non-helix rung is
byte-identical to the 2026-07-30 snapshot** (checked mechanically), so the
tables below extend the same experiment rather than re-running a different
one.

Full sweep, 11 designs, 52 rungs — including every rung §11 recorded as
skipped (discone N=161 / 2767 segs, hourglass and hourglass_slant N=321,
hourglass_array N=161, lazy_h and vbeam N=321) — under a 24 GiB address-space
cap, in ~3 minutes wall clock against the previous run's 16m54s at the
smaller cap:

```
design                         finest  coll↔bs2   gal↔bs2  ptgap↔bs2  coll step  gal step  bs2 step
---------------------------------------------------------------------------------------------------
specialty.hentenna                321    12.89%     0.01%     0.008%      1.61%     0.02%     0.04%
specialty.hentenna_slant          321     1.20%     0.03%     0.026%      0.54%     0.02%     0.09%
arrays.hentenna_array             161    10.67%     0.04%     0.024%      1.09%     0.07%     0.04%
specialty.hourglass               321     8.85%     0.01%     0.008%      1.11%     0.02%     0.03%
specialty.hourglass_slant         321    14.88%     0.02%     0.019%      1.98%     0.02%     0.07%
arrays.hourglass_array            161     9.35%     0.04%     0.018%      0.82%     0.07%     0.03%
broadband.discone                 161    14.89%     0.13%     0.122%      9.65%     0.08%     0.13%
specialty.bowtie                  321     0.05%     0.05%     0.017%      0.07%     0.07%     0.02%
specialty.faceted_helix           321     0.59%     0.01%     0.005%      0.21%     0.21%     0.30%
wire.lazy_h                       321     0.84%     0.81%     0.003%      1.85%     1.84%     1.43%
wire.vbeam                        321     0.57%     0.59%     0.001%      0.75%     0.69%     0.54%
```

What the deeper rungs say, attribution by attribution:

- **§3 (T/X-junction cluster → testing) holds and sharpens.** At the rungs
  #521 actually quoted, `coll↔bs2` still reads 1.2–14.9 % while `gal↔bs2`
  sits at 0.01–0.13 % — the point matching, not the basis, at every depth
  the census asked about. `discone`'s 9.65 % collocation step at its finest
  rung is the same story told by convergence rate: the collocation column is
  still far from settled where both Galerkin columns already agree.
- **§4 (helix) is resolved.** The re-instrumented control converges on a
  genuinely refining ladder and scores clean: `gal↔bs2` 0.01 %, matched-feed
  0.005 %. The 2026-07-22 census's 30.6 % helix row measured a frozen mesh;
  there was never a curvature-driven basis effect. Follow-up 2 of §12 is
  closed by antennaknobs#630/#636.
- **§5 (near-open class → feed model) is confirmed at full depth.** At
  N=321, `lazy_h` and `vbeam` still show a `gal↔bs2` gap (0.81 % / 0.59 %)
  that the matched feed collapses to 0.003 % / 0.001 % — three decades. The
  shared whole-structure slow convergence stands as filed (per-scheme steps
  0.5–1.9 % at 2570 segments).

The feed-placement guard reads 0.0000 m on every ladder, as before. Raw
sweep output: reproduce with `python scripts/m6_residue_cluster.py` (the
defaults now reach every rung above).

---

## 14. Addendum (2026-07-31): §12 follow-up 4 landed — junction ports over PEC

momwire#191 closes follow-up 4 for the PEC ground, and it cost what the
follow-up predicted. `_assemble_Z` builds a PEC ground as the free-space
field of *mirrored* sources and subtracts it once, so the image of the
lumped node charge M5b removes is a point charge at the mirrored node — a
mirror of a term already removed. The same `D`/`S` correction therefore runs
on the image block at the mirrored separation and enters with the opposite
sign:

    G' = (A − D − Dᵀ + S) − (B − D_img − D_imgᵀ + S_img)

Measured on M5b's own gate — the entrywise comparison against
`BSplineSolver`'s Lagrange-multiplier port, over a PEC plane:

| topology | free space (§7) | over PEC |
|---|---|---|
| two-member oracle | 3.4e-5 | 8.5e-5 |
| one-member `PortAtEnd` | 3.9e-6 | 4.7e-6 |

The ground rides at the free-space agreement floor while moving the port Y
itself by 33 %, and the port block stays symmetric to 6.3e-13. The sign is
pinned by measurement rather than by reading the code: dropping the image
half misses `BSplineSolver` by 15 %, flipping it by 38 %.

What §7's scope paragraph and §10's "every ground read" row say about
junction ports therefore now applies only to the **finite** grounds — a
Fresnel reflection of a point charge is angle-dependent and Sommerfeld's is
not an image at all, so neither has the closed mirror this argument turns
on, and both still refuse. §7's `sterba_bl` comparison could now be re-run
over a PEC ground; the average-ground catalog check it names still cannot,
which is the remaining half of follow-up 4.

---

## 15. Addendum (2026-07-31): §6's last digits moved — momwire#203

momwire#203 found that every literal spelling of the three-term basis was
throwing away most of the precision it had already computed, and folding it
moved the Galerkin numbers in §6 at the 1e-7 to 1e-5 level. The run records
above stand as measured; this is what they read now.

The basis is normalized to its own segment-centre current, A + C — and that
is O((kΔ)²) while A and C are each O(1). So `σA + B·sin kξ + σC·cos kξ`
computes a small number by cancelling two large ones, at a cost of ε·8/(kΔ)²
relative: 5.5e-14 at N=41, 1.8e-10 at N=2401, growing like N². The same
cancellation runs again in the Galerkin coefficient product, where the
const- and cos-shape source fields nearly coincide. Both are now evaluated on
the well-scaled shape set {1, sin kξ, cos kξ − 1}, which is the same functions
rearranged so that no term is larger than the answer.

Nothing about the physics changed, and the point-matched solver and the
B-spline family are untouched — but cond(G) is ~3e5 at N=321 and ~4e6 at
N=2401, so those lost digits were setting the last digits of Z:

| reading | as filed | folded |
|---|---|---|
| §6 `gal` at N=321 | 69.639094 − 18.056294j | 69.639093 − 18.056307j |
| §6 `ptgap` at N=321 | 69.633780 − 18.065312j | 69.633779 − 18.065325j |
| §6 `ptgap`↔`bs2` at N=321 | 4.17e-8 | 1.30e-7 |
| §6 `ptgap`↔`bs2` at N=161 | 2.81e-7 | 2.81e-7 |
| dipole `gal` at N=2401 | 69.64 − 17.93j | 69.64 − 17.97j |

§6's conclusion is unchanged in kind and slightly smaller in degree: matching
the feed removes ~1100× of the sinusoidal↔B-spline difference rather than
~3500×, the matched pair still tightens with refinement while the mismatched
pair does not, and what is left at N=321 is still three orders below anything
M2/M3 filed as a basis effect. The N=161 rung does not move at all, which is
the expected N² scaling of the defect.

What #203 did **not** fix, contrary to its own second gate: the G1 symmetry
ratio's growth with mesh (3.4e-10 at N=801, 1.3e-9 at N=2401). That is not
the coefficient product's conditioning — the exact 80-bit product of the same
float64 contributions is just as asymmetric — but the same cancellation one
level up, inside the fill: the const- and cos-shape SOURCE fields nearly
coincide, so the ε‖T_const‖ the kernel leaves in them is amplified by
‖T‖/‖G‖, and G1 sits at 1.4 × ε‖T_const‖/‖G‖ at both meshes. Removing it
means giving the field kernel the (cos kξ − 1) source shape itself, which is a
change to the C++ far fill's output contract. Filed as a follow-up;
`test_reciprocity_floor_is_set_by_the_fill_not_the_product` pins the reading
so the follow-up announces itself.

---

## 16. Addendum (2026-08-01): §12 follow-up 5 landed — `feed_model` is an option

momwire#192 closes follow-up 5 the way the follow-up asked: as an **option,
with the default unchanged**.

    SinusoidalGalerkinSolver(..., feed_model="segment")   # default, as before
    SinusoidalGalerkinSolver(..., feed_model="point")     # §6's `ptgap` column

`"point"` is the zero-width gap `BSplineSolver` drives: E_app = V·δ(s − s0) at
the feed segment's centre, so the Galerkin test integral collapses on the
delta and the drive column is −V·f_i(s0) = −σ(A+C). That is one line of RHS
assembly; nothing in the fill, the quadrature or the ports changes, and the
`"segment"` branch is the same expression it always was, so no pinned constant
in the tree moved (837 → 848 tests collected, all green).

`ptgap` in `scripts/m6_residue_cluster.py` is now that option rather than the
ten-line `PointGapGalerkin` research subclass, which is deleted. §1's
description of it as "a ten-line research subclass in the harness, not a
solver option" is superseded here; the run records themselves are untouched
and the column still reads what §15 says it reads, verbatim:

| N | `ptgap` | `ptgap`↔`bs2` | `gal`↔`bs2` |
|--:|---|--:|--:|
| 161 | 69.634327 − 18.112452j | 2.806e-7 | 2.466e-4 |
| 321 | 69.633779 − 18.065325j | 1.303e-7 | 1.454e-4 |

Pinned by `test_feed_model_option_reproduces_section_6`
(`tests/test_m6_instrument_report.py`), which asserts the relative *values*,
not only the ordering — a column that has graduated into the API needs a
before that a future flip has to be measured against.

### The duality, and its interaction with `feed_readout`

M5's finding, now a property of the solver: the point-gap drive column IS the
centre-evaluation functional the default `feed_readout="centre"` reads. So
under `feed_model="point"` the two readouts are the same functional and the
knob stops having consequences at gap feeds. Measured on the two-feed dipole:

| | `segment` (default) | `point` |
|---|--:|--:|
| Y asymmetry, `feed_readout="centre"`, N=21/41/81 | 6.7e-5 / 2.4e-5 / 6.5e-6 | 3.1e-13 / 3.9e-12 / 5.6e-12 |
| Y asymmetry, `feed_readout="variational"` | 4.4e-13 / 3.2e-12 / 5.9e-12 | 3.1e-13 / 3.9e-12 / 5.6e-12 |
| driving-point spread between the two readouts | 2.2e-3 / 1.4e-3 / 8.2e-4 | 0.0 / 0.0 / 1.7e-16 |

The point gap reaches the fill's own reciprocity floor under **either**
readout, so it buys the reciprocal Y that `feed_readout="variational"` buys
without the M3 payoff `variational` costs. What it does **not** do is leave
the payoff alone in general — see the inventory.

### What a default flip WOULD move

Measured, not reasoned: the whole suite re-run with the default flipped to
`"point"` fails 22 tests. That is the price list, and it is why the flip is
its own decision rather than a consequence of this change.

**Pinned impedance constants that move** (both are reference *families* other
gates are scored against, so moving them re-baselines their dependents too):

| file : test | reading | pinned | flipped |
|---|---|---|---|
| `test_sinusoidal_galerkin.py` : `test_m3_reference_constants_are_reproducible` | `dipole.sin_gal_321` | 69.639093 − 18.056307j | 69.633779 − 18.065325j |
| `test_sinusoidal_galerkin.py` : `test_m4_reference_constants_are_reproducible` | `m4_dipole/pec.gal_161` | 20.995250 + 2.666325j | 20.995314 + 2.664821j |

**Payoff gates that fall below their pinned floors** — the M3/M4 headline
"Galerkin testing is worth this much":

| file : test (param) | pinned floor | flipped |
|---|--:|--:|
| `test_sinusoidal_galerkin.py` : `test_g3_verdict_survives_every_candidate_reference` (`dipole`) | 2.0 | 1.948 |
| — (`k2_junction`) | 2.9 | 1.774 |
| `test_sinusoidal_galerkin.py` : `test_g4_variational_payoff_survives_each_ground` (`m4_dipole-pec`) | 1.85 | 1.841 |
| — (`m4_dipole-refl`) | 2.0 | 1.884 |
| — (`m4_dipole-somm`) | 2.15 | 1.960 |

Read these with M3's fixed-feed constraint in mind: the reference family is
built from *this* solver, so flipping the default changes the reference and
the comparand together while the point-matched sibling it is scored against
keeps NEC's segment gap. `errColl/errGal` then compares two feed models, which
is exactly the confound §6 exists to isolate. So these are not simply numbers
to re-record — the comparison itself would have to be re-derived (either give
`SinusoidalSolver` a matching gap, or score the flipped solver against its own
family). That is the substantive argument against a flip, and it is stronger
than the re-baselining cost.

**Qualitative M3 findings that invert** — statements, not constants, so a flip
would need new prose rather than new digits:

| file : test | flipped reading |
|---|---|
| `test_sinusoidal_galerkin.py` : `test_the_variational_payoff_is_in_the_reactance` | collocation converges faster in R on all four geometries (dipole 2.64, vee 1.14, k2_junction 1.74, k3_star 1.39) |
| `test_sinusoidal_galerkin.py` : `test_k2_pre_asymptotic_error_maximum_is_not_quadrature` | the documented pre-asymptotic rise is gone (0.463 % → 0.392 % → 0.330 %) |
| `test_sinusoidal_galerkin.py` : `test_k3_star_gap_is_basis_limited_not_testing_limited` | testing starts mattering (1.42) |
| `test_sinusoidal_galerkin.py` : `test_k2_junction_gap_is_testing_limited` | testing stops mattering (1.84) |

**Contrast tests that become tautologies** — they measure the difference
between the two models, so a flip makes both sides the same thing:
`test_sinusoidal_galerkin.py` : `test_gap_feed_readout_is_not_its_drives_dual`
(3 params), `test_point_gap_readouts_coincide` (3 params),
`test_the_variational_readout_costs_the_m3_payoff_on_k3_star`;
`test_junction_ports.py` :
`test_sg_node_port_full_mixed_y_symmetry_inherits_the_m5_amendment`;
`test_m6_instrument_report.py` :
`test_matched_feed_model_closes_the_dipole_basis_gap`,
`test_feed_model_option_reproduces_section_6`,
`test_point_gap_drive_is_self_dual`.

Outside momwire, the one antennaknobs pin exposed is
`tests/test_momwire_engine.py` :
`test_momwire_sinusoidal_galerkin_reads_the_hentenna_residue`, which quotes a
gap-fed `sin-Galerkin` impedance — §5's reading (`gal`↔`ptgap` ≤ 0.03 % on the
residue cluster) says it would move well inside its tolerance. The
`wire.sterba_bl` comparison in `tests/test_balanced_line_physics.py` is
junction-ported and has no gap feed, so it is not exposed at all.

### Where this leaves the axis

Unchanged as a *reading*: on a clean geometry the two bases are, for practical
purposes, the same basis, and §6 is still the sharpest result in the report.
What is new is that the reading is reproducible from the shipped API rather
than from a subclass in a script, and that anyone with a near-open, high-Q
model (§5, antennaknobs#478) can opt into the feed model `BSplineSolver`
already uses and get a reciprocal Y with it. What is deliberately not new is
the default.

---

## 17. Addendum (2026-08-01): the feed-matched payoff — momwire#212

§16 named the substantive blocker on ever flipping `feed_model`'s default: the
M3/M4 payoff gates score `SinusoidalGalerkinSolver` against `SinusoidalSolver`,
which hard-codes NEC's segment gap, so flipping only the Galerkin default would
make `errColl/errGal` compare two feed models. The remedy §16 proposed was to
give the point-matched solver a matching gap. momwire#212 went to do that and
found it cannot be done — not for want of plumbing, but because the object does
not exist. This addendum records the refutation and the payoff reading that
survives it. **No pinned gate moved; §6, §15 and §16's run records stand as
measured.**

### The zero-width gap has no collocation RHS

Collocation tests with δ(s − s_m), so row *m*'s drive is the pairing
⟨δ_m, E_app⟩ = E_app(s_m) — defined only where E_app is a *function*. The
zero-width gap's applied field is the distribution E_app = V·δ(s − s0), and
δ·δ is not one: there is no number to put in the feed row, and zero in every
other row is the unexcited problem. §16's Galerkin column exists only because
its pairing ⟨f_i, E_app⟩ tests against a continuous f_i, on which the delta
collapses to the drive column −V·f_i(s0). **The feed model is an axis of the
testing scheme, not of the basis** — which is a sharper statement of the axis
than §1 had, and it means the third axis is not orthogonal to the second after
all: one cell of the 2 × 2 is empty.

Any construction has to replace δ by a nascent delta g_w of unit integral and
width *w* and then sample, v_m = −V·g_w(s_m − s0). There are exactly two
regimes, and the dipole ladder was run on both
(`scripts/m6_residue_cluster.py --feed-matched`, canonical 0.962 λ/2 dipole,
a = 0.5 mm, free space):

**w ≪ h.** The only width this model already owns is the wire radius: the
b → a limit of the magnetic frill, i.e. a delta ring of magnetic current at
ρ = a, whose on-axis field

    E_z(d) = (V a²/2)(1 + jkR) e^{−jkR} / R³,   R = √(d² + a²)

is smooth, finite everywhere, integrates to exactly V, and — the reason it is
the right candidate — has **no free parameter left** once b → a. (The textbook
frill's entire content is its b/a; that is precisely the modeling parameter a
zero-width gap may not introduce.) It is evaluated in the solver's own kernel
convention: ring on the surface, observation on the axis, which reproduces the
source-filament / BC-on-surface separation √(d² + a²) the fill already uses.

| N | h | a/h | `coll` (segment gap) | point-sampled | ‖v − c·v_seg‖/‖v‖ | (2a/h)·Z_seg / Z_pt − 1 |
|--:|--:|--:|---|---|--:|--:|
| 41 | 0.25810 | 1.94e-3 | 69.645276 − 18.457609j | 0.269841 − 0.071514j | 7.3e-9 | 2.8e-8 |
| 81 | 0.13064 | 3.83e-3 | 69.633707 − 18.295229j | 0.533012 − 0.140041j | 5.6e-8 | 1.4e-7 |
| 161 | 0.06573 | 7.61e-3 | 69.631109 − 18.185891j | 1.059402 − 0.276689j | 4.4e-7 | 1.1e-6 |
| 321 | 0.03297 | 1.52e-2 | 69.631876 − 18.107822j | 2.112233 − 0.549288j | 3.5e-6 | 8.4e-6 |
| 641 | 0.01651 | 3.03e-2 | 69.634021 − 18.048454j | 4.217769 − 1.093204j | 2.8e-5 | 6.7e-5 |

The sampled drive carries no shape of its own — it is the segment-gap drive
rescaled by h/2a, to parts in 1e5 — because every match point but the feed's
sees ≈ 0 while the feed row sees the spike's *peak*, ≈ V/2a. Z is homogeneous
of degree −1 in the RHS, so Z = (2a/h)·Z_segment and **doubles at every mesh
doubling**. That is the failure in one sentence: *collocation reads the
source's width off the feed row's amplitude and cannot tell a narrow gap from
a small voltage*, so a sub-mesh width comes back as a fraction of a volt. It is
not §16's log walk — it is a linear one, and no refinement fixes it, because a
zero-width source is sub-mesh at every N.

**w ≳ h.** The source is mesh-resolved and collocation is consistent again,
but *w* is then a modeling parameter, and the smallest one the mesh resolves is
w = h — which is NEC's segment gap exactly. Confirmed from the other side: the
cell average of the *same* zero-width field above returns the segment gap's
−V/h to 6.0e-8 / 1.1e-7 / 2.4e-7 / 5.7e-7 / 1.4e-6 of |Z| at N = 41…641, four
decades inside the (a/h)² bound.

So the two regularizations fail in complementary ways — one is mesh-dependent
garbage, the other is the default under a new name — and between them they
prove the reading:

> **Under collocation the segment gap is not a rival source model to the point
> gap. It IS the point gap, rendered at the only resolution collocation has.**

A Hallén-style particular-solution forcing term is the one remaining escape and
was rejected on scope: it needs the infinite-wire kernel's Fourier inversion to
get J_p, J_p satisfies neither this basis's end conditions nor its junction
tables, and moving the unknown changes what "the basis" means — which would
break the one-axis-at-a-time discipline this instrument exists to keep.

`SinusoidalSolver` therefore takes `feed_model=` with the sibling's validation
and message, accepts `"segment"`, and **refuses `"point"`** with the derivation
attached (`_reject_point_feed_model`). A silent fallback to the segment gap
would manufacture exactly the two-feed-model comparison §16 exists to prevent.

### What `errColl/errGal` reads with the feed model held fixed

Given the above, the feed-matched pairing is `segment`/`segment` — which is
what M3 already scores. The one place a feed model still varies inside that
gate is the *reference* family: two of M3's six defensible references are
`BSplineSolver` solves, and those drive a zero-width gap. Dropping them is the
only feed-matching left to do:

| geometry | reference | N=11 | N=15 | N=21 | reference feed |
|---|---|--:|--:|--:|---|
| dipole | `sin_gal_321` | 2.125 | 2.245 | 2.323 | segment |
| | `sin_coll_321` | 2.163 | 2.339 | 2.504 | segment |
| | `bspline2_321` | 2.123 | 2.249 | 2.338 | **point** |
| | `rich_sin_gal` | 2.062 | 2.133 | 2.149 | segment |
| | `rich_sin_coll` | 2.072 | 2.151 | 2.179 | segment |
| | `rich_bspline2` | 2.063 | 2.136 | 2.155 | **point** |
| k2_junction | `sin_gal_321` | 5.712 | 4.341 | 3.628 | segment |
| | `sin_coll_321` | 8.031 | 5.702 | 4.712 | segment |
| | `bspline2_321` | 6.461 | 4.790 | 3.984 | **point** |
| | `rich_sin_gal` | 4.465 | 3.531 | 2.987 | segment |
| | `rich_sin_coll` | 4.440 | 3.515 | 2.974 | segment |
| | `rich_bspline2` | 4.436 | 3.511 | 2.971 | **point** |

| geometry | worst over all six | worst over the four feed-matched | pinned floor | §16's flipped reading |
|---|--:|--:|--:|--:|
| dipole | 2.062 | 2.062 | 2.0 | 1.948 |
| k2_junction | 2.971 | 2.974 | 2.9 | 1.774 |

**The M3 payoff verdict does not rest on the reference's feed model** — matching
it moves the worst case by less than 0.2 %, well inside the ~1e-3 spread the
reference family already spans (§ "G3" in `tests/test_sinusoidal_galerkin.py`).
So §16's confound lives entirely on the *comparand* side, and the last two
columns are the whole story: the pinned 2.06 / 2.97 are a feed-matched
measurement already, and §16's 1.948 / 1.774 are what the same gate reads once
one comparand's feed model is changed and the other's cannot follow.

### What this says about M3's headline claim

M3's headline — *Galerkin testing buys 2.1× on the dipole and 3.0× at a
junction, at coarse mesh* — is **unaffected**, and is now known to be
feed-matched rather than merely assumed to be. What the refutation removes is
the option §16 held open of restoring feed-matching after a default flip: there
is no matching gap to give the point-matched solver, so a flip of
`SinusoidalGalerkinSolver`'s default would confound `errColl/errGal`
*permanently*, not until someone does the plumbing. That converts §16's
"substantive argument against a flip" into a structural one, and it is the
reason the recommendation stands unchanged: keep `"segment"` as both solvers'
default and treat `"point"` as a per-model opt-in on the Galerkin cell, where
the source is admissible.

The constructive repair §16's other half suggested — *score the flipped solver
against its own family* — is untouched by this and remains available; what is
now excluded is the repair by symmetry. A third option this addendum opens:
give `BSplineSolver` a `feed_model="segment"`, which is well defined there (it
is Galerkin) and would let §6's three-way be feed-matched in the other
direction, with the oracle joining the two sinusoidal solvers on NEC's gap
instead of the reverse. That is a new follow-up, not a claim.

Pinned by `test_zero_width_gap_has_no_collocation_rhs` and
`test_feed_matched_payoff_is_the_pinned_m3_ratio`
(`tests/test_m6_instrument_report.py`), plus
`test_coll_point_feed_model_is_refused_not_defaulted`
(`tests/test_sinusoidal_galerkin.py`). Reproduce with
`python scripts/m6_residue_cluster.py --feed-matched`.

---

## 18. Addendum (2026-08-01): the near-open collapse at N=641 — momwire#213

§5 measured antennaknobs#478's near-open class at N=161 and found the
sinusoidal ↔ B-spline residual collapsing **55× / 74×** once the feed model
was matched; §13 re-read it at N=321 through the shipped
`feed_model="point"` (momwire#192) and got 0.003 % / 0.001 %. Both readings
stop short of the question #478's users would actually act on: §5 also filed
a **shared whole-structure** residual that no feed model explains, and the
finest mesh is exactly where a shared term would reassert itself over a
feed-model term. This addendum takes the ladder one rung deeper — N=641,
5130 and 5119 segments, a 24× refinement on the rung #478 quoted — and
records what happens. **No run record is edited; §5 and §13 stand as
measured, and the rungs they quote reproduce here verbatim.**

Harness: `python scripts/m6_residue_cluster.py --near-open` (momwire#213).
The two designs carry N=641 in the snapshot, which was re-dumped for those
two rows alone — **all 52 pre-existing rows are byte-identical to the
2026-07-31 snapshot and in the same order** (checked mechanically), so this
is the same experiment at greater depth. N=641 is above the harness's
`DEFAULT_SEG_CAP`, so the standard sweep records it *skipped* and does not
grow: re-run here it takes **2m31s** against the ~3 minutes §13 quotes and
reproduces §13's eleven-design table row for row. `--near-open` is what
reaches the deep rungs.

### The ladder

Every gap is `|ΔZ| / |Z_bs2|` at the same rung; `collapse` is
`gal↔bs2 / ptgap↔bs2`, i.e. how much of the disagreement the feed model
accounts for at that depth. `step` is each scheme against the rung below it.

`wire.lazy_h`:

| N | segs | `coll`↔`bs2` | `gal`↔`bs2` | `ptgap`↔`bs2` | collapse | `coll` step | `gal` step | `ptgap` step | `bs2` step |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 21 | 170 | 5.41 % | 5.75 % | 0.5728 % | 10× | — | — | — | — |
| 41 | 330 | 2.17 % | 2.31 % | 0.1369 % | 17× | 10.71 % | 11.04 % | 7.98 % | 7.53 % |
| 81 | 646 | 2.32 % | 2.26 % | 0.0655 % | 34× | 0.50 % | 0.07 % | 0.08 % | 0.01 % |
| 161 | 1290 | 1.25 % | 1.21 % | 0.0217 % | 56× | 4.03 % | 4.00 % | 2.99 % | 2.95 % |
| 321 | 2570 | 0.84 % | 0.81 % | 0.0029 % | 278× | 1.85 % | 1.84 % | 1.45 % | 1.43 % |
| **641** | **5130** | **0.60 %** | **0.59 %** | **0.0006 %** | **992×** | 1.44 % | 1.43 % | 1.20 % | 1.20 % |

`wire.vbeam`:

| N | segs | `coll`↔`bs2` | `gal`↔`bs2` | `ptgap`↔`bs2` | collapse | `coll` step | `gal` step | `ptgap` step | `bs2` step |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 21 | 169 | 1.50 % | 2.48 % | 0.3306 % | 8× | — | — | — | — |
| 41 | 327 | 4.19 % | 2.60 % | 0.2203 % | 12× | 3.68 % | 0.17 % | 0.40 % | 0.29 % |
| 81 | 647 | 1.31 % | 1.06 % | 0.0343 % | 31× | 6.34 % | 5.03 % | 3.65 % | 3.46 % |
| 161 | 1287 | 0.78 % | 0.74 % | 0.0098 % | 76× | 1.55 % | 1.33 % | 1.04 % | 1.01 % |
| 321 | 2563 | 0.57 % | 0.59 % | 0.0008 % | 704× | 0.75 % | 0.69 % | 0.55 % | 0.54 % |
| **641** | **5119** | **0.37 %** | **0.41 %** | **0.0011 %** | **374×** | 1.19 % | 1.18 % | 0.99 % | 0.99 % |

The N=161 rows are §5's table to the digit (1.25/1.21/0.022 and
0.78/0.74/0.010) and the N=321 rows are §13's, so the deep rungs extend
those runs rather than replacing them.

### The verdict: it survives, and it is not a coarse-mesh artifact

**The collapse holds at every depth measured and grows over most of the
ladder.** On `lazy_h` it is monotone — 10 → 17 → 34 → 56 → 278 → **992×** —
and the matched column falls monotonically from 0.57 % to 0.0006 % while the
mismatched one is still at 0.59 %. On `vbeam` it peaks: 704× at N=321, then
**374×** at N=641. The peak is not the collapse degrading in the sense §5
warned about — the matched residual only moves 8.4e-6 → 1.1e-5 — it is the
*mismatched* residual continuing to shrink (0.59 % → 0.41 %) while the
matched one has flattened. Both readings are two-and-a-half decades or more,
at 5100 segments, so the answer to #213's question is unambiguous: on this
class essentially the whole sinusoidal ↔ B-spline disagreement is the delta
gap, at every mesh anyone would run.

One new observation the deep rung buys: **the matched pair stops converging
together at ~1e-5 here**, where on §6's clean dipole it keeps tightening past
1e-7. That floor is the first thing in this report that looks like a genuine
basis difference on a near-open geometry rather than a feed-model one — and
it is 2.5 decades below the shared term below, so it is a curiosity, not an
error budget. (Follow-up, not a claim: whether the floor is the basis or the
two Galerkin schemes' own errors ceasing to cancel is unmeasured.)

### §5's caveat is intact — and, at this depth, stronger

The collapse is a statement about the **disagreement between schemes**, never
about the error. At N=641 every scheme's own last step is still **1.2–1.4 %**
on `lazy_h` and **1.0–1.2 %** on `vbeam`, after a 24× refinement of the mesh
#478 quoted; the shared term is two to three orders larger than the matched
inter-scheme gap at the same rung and remains the dominant number on the
page. It also does not fall cleanly: `vbeam`'s own step is *larger* at N=641
(1.19 %) than at N=321 (0.75 %), and `lazy_h`'s N=81 step (0.07 %) sits
between a 11.04 % and a 4.00 % one. So a single rung-to-rung step
under-reports the distance to converged on this class, which is the honest
form of #478's advice:

> **"Expect the last percent to be physical, and budget fine mesh for
> near-open structures" survives this addendum unchanged.** What
> `feed_model="point"` removes is the part of the discrepancy that was an
> artifact of comparing two source models. It buys no accuracy.

### What antennaknobs#478 users should do

For a near-open, high-Q model on `momwire:sinusoidal-galerkin`, set
`feed_model="point"`: it removes ≥278× of the sinusoidal ↔ B-spline
disagreement at every rung from N=321 down (992× / 374× at ~5100 segments,
56× / 76× as coarse as N=161), and per §16 it also delivers a reciprocal Y
under either `feed_readout` at no cost to the M3 payoff. It does **not**
reduce the mesh a near-open model needs — the ~1 % that is left is shared by
every scheme and is what the fine mesh is for. The default stays `"segment"`
for the reasons §16 and §17 give; this is a per-model opt-in.

Free space only, as everywhere in this report: momwire#151 / M4's
ground-contact defect still excludes ground reads on the sinusoidal family.

### Cost

Both ladders, six rungs, four columns: **3m09s** wall clock and **7.0 GiB**
peak RSS under the 24 GiB address-space cap (same box as §13). The N=641
rungs are ~2.5 min of that — per design `coll` 7.3 s, `gal` 27 s, `ptgap`
27 s, `bs2` 11–15 s — which is why they are gated behind `--near-open`
instead of joining the standard sweep.

Pinned by `test_near_open_collapse_survives_at_the_fine_rung`
(`tests/test_m6_instrument_report.py`), at N=321 off the snapshot: the values
above with a ≥100× structural floor, plus the §5 caveat guard that every
scheme's own step still exceeds its cross-scheme gap. N=641 is not in CI —
it is a ~2.5-minute rung, and it lives here and in the harness. Reproduce
with `python scripts/m6_residue_cluster.py --near-open`.

---

## 19. Addendum (2026-08-01): §6 mirrored — the oracle on NEC's gap, momwire#216

§17 closed one direction of feed-matching and opened the other: the point gap
cannot exist on a collocation solver, so if §6's three-way is ever to be
feed-matched from *both* sides, the move has to be the B-spline oracle joining
the sinusoidal solvers on NEC's segment gap rather than the reverse. momwire#216
does that, and this addendum records what the mirror reads. **No pinned gate
moved and no run record is edited; §6, §15, §16 and §17 stand as measured.**

    BSplineSolver(..., feed_model="point")     # default, as before
    BSplineSolver(..., feed_model="segment")   # NEC's Eq 187 gap

`"segment"` is the drive b_m = (1/Δ)∫_cell Φ_m ds over the mesh cell holding
the feed arclength, against `"point"`'s b_m = Φ_m(s_f). Under Galerkin testing
that is well-defined and cheap — every Φ_m is a polynomial of degree *d* on
exactly one knot span, so the RHS assembly's existing 16-node Gauss rule
(exact through degree 31) integrates it exactly and there is no accuracy knob
to add. It is built in `_build_source_vector`, the one place all four solve
paths and every feed of a multi-feed model construct their drive, so the option
is honored everywhere by construction and `HMatrixSolver` inherits it.
Combining it with `feed_smoothing_factor` raises: both replace the point drive
with a spread one, and composing them means nothing.

### The readout is the drive's dual — so the mirror has two pairings

This is the whole subtlety of the mirror, and it is a property of the solver
rather than of the experiment. `BSplineSolver` reads Z = 1/(vᵀc) with the SAME
v it drove — the impedance is automatically its drive's dual. Under the point
gap vᵀc is the current at s_f, which is why §6's `ptgap` pairing worked against
`SinusoidalGalerkinSolver`'s default `feed_readout="centre"`. Under the segment
gap vᵀc is the gap-AVERAGED current, which is `feed_readout="variational"`'s
functional, not the centre one — and §16 measured that those two differ at O(h)
under a segment gap (2.2e-3 / 1.4e-3 / 8.2e-4 at N=21/41/81). So there is no
single "gal(segment)" to mirror against, and both are reported.

Canonical 0.962 λ/2 dipole, free space, a = 0.5 mm — §6's geometry exactly:

| N | `bs2(point)` | `bs2(segment)` | `gal(seg, centre)` | `gal(seg, variational)` |
|--:|---|---|---|---|
| 161 | 69.634329 − 18.112432j | 69.651936 − 18.085105j | 69.643869 − 18.097471j | 69.651933 − 18.085125j |
| 321 | 69.633780 − 18.065315j | 69.643519 − 18.048873j | 69.639093 − 18.056307j | 69.643518 − 18.048882j |

| N | `gal(seg,centre)`↔`bs2(seg)` | `gal(seg,variational)`↔`bs2(seg)` | `gal(point)`↔`bs2(point)` | `gal(seg,centre)`↔`bs2(point)` |
|--:|--:|--:|--:|--:|
| | source matched, readout not | **fully matched** | §6's pair, mirrored | §6's pair |
| 161 | 2.052e-4 | **2.845e-7** | 2.806e-7 | 2.466e-4 |
| 321 | 1.203e-4 | **1.308e-7** | 1.303e-7 | 1.454e-4 |

### The verdict: it collapses, symmetrically, and to the same number

**The fully matched pairing collapses — 867× at N=161 and 1112× at N=321
against §6's own mismatched pairing — and it lands on the same residual §6's
point-gap pairing lands on: 2.845e-7 against 2.806e-7, and 1.308e-7 against
1.303e-7.** That is 1.4 % and 0.4 % agreement between two independent
feed-matched readings of the same quantity, taken under two *different* feed
models. It also tightens with refinement (2.8e-7 → 1.3e-7) exactly as §6's does,
while both mismatched columns stay flat at ~1e-4.

So §6's conclusion is now closed from both directions: **on a clean geometry
the two bases are the same basis, and the ~1e-7 that is left is a basis
difference that does not depend on which gap model the comparison is
feed-matched at.** The mirror found no asymmetry — which is the outcome §6
predicts and the first time it has been checked rather than assumed.

One thing the mirror does add, and it sharpens §16 rather than contradicting
it. Matching only the *source* buys almost nothing: `gal(seg,centre)`↔`bs2(seg)`
is 2.05e-4 / 1.20e-4 against the mismatched 2.47e-4 / 1.45e-4, a factor of
**1.2×**. The other three decades are the readout functional. §6's "matching
the feed removes ~1100× of the difference" is therefore, read precisely, *matching
the feed model and its dual readout together* — under the point gap those are the
same act, because the point gap is self-dual under the centre readout (§16),
and that coincidence is what made §6's single column sufficient. Under the
segment gap they come apart, and the mirror is the measurement that separates
them. The oracle's own sensitivity to the choice is the same size:
`bs2(point)`↔`bs2(segment)` reads 4.52e-4 / 2.66e-4.

### A degeneracy worth naming: the option is a no-op on the tent basis

On `degree=1` with the feed at its cell's centre, `"segment"` and `"point"` are
**bit-identical** drives. Each tent is linear on a knot span and the cell
average of a linear function is its midpoint value, so (1/Δ)∫_cell Φ_m = Φ_m(s_f)
identically. The two models separate at d ≥ 2, or at d = 1 with an off-centre
feed. Nothing depends on this, but it means a d=1 reader must not treat
`feed_model` as a control that is known to be active.

### The M3 reference family, with nothing dropped

§17 could only feed-match M3's reference family by *removing* its two
`BSplineSolver` members, leaving four of six. A segment-gapped oracle lets them
be kept instead, as a seventh reference `bs2_segment_321` =
`BSplineSolver(degree=2, feed_model="segment")` at N=321 — 69.643519 − 18.048873j
on the dipole, 124.492317 + 0.403057j on k2_junction:

| geometry | `bs2_segment_321` at N=11 / 15 / 21 | worst over §17's matched four | worst over the fully matched five | pinned floor |
|---|--:|--:|--:|--:|
| dipole | 2.127 / 2.241 / 2.309 | 2.062 | **2.062** | 2.0 |
| k2_junction | 5.450 / 4.181 / 3.501 | 2.974 | **2.974** | 2.9 |

**The worst case does not move at all.** §17 extrapolated "<0.2 % movement";
measured, it is 0.0 % — the new reference is never the worst member on either
geometry (its coarsest ratio, 2.127 / 5.450, sits mid-family). M3's headline —
*Galerkin testing buys 2.1× on the dipole and 3.0× at a junction, at coarse
mesh* — is now feed-matched with **nothing dropped from the family**, which is
strictly better than §17's position and reaches the same number.

The pinned `M3_REFS` in `tests/test_sinusoidal_galerkin.py` is deliberately
untouched: #216 adds a reading, not a gate, and §17's own test still passes
verbatim. `M3_REFS_QUOTED` in the harness remains a byte-for-byte quote of it;
the seventh reference is a separate object beside it.

### What this does and does not change

It does **not** reopen the default question. §17's structural argument against
flipping `SinusoidalGalerkinSolver`'s default stands untouched — the point
gap still has no collocation RHS, so a flip there would still confound
`errColl/errGal` permanently. And `BSplineSolver`'s own default stays
`"point"`, bit-identical, because every pinned B-spline constant in the tree
is a point-gap reading and the option's purpose is comparison, not physics.

What it does change is that the instrument can now hold the feed model fixed
at **either** value across all three solvers rather than only at the point gap
on two of them, and that §6's sharpest result has been re-derived under a
source it was never measured under. The follow-up §17 opened is closed.

### Cost

`--dipole-feed-mirror`: **3.7 s** and 190 MiB — ten solves, five schemes at
each of two fine meshes. `--feed-matched` is unchanged in cost; its seventh
reference is a constant, not a solve.

Pinned by `test_section_6_mirrors_with_the_oracle_on_the_segment_gap` and
`test_m3_payoff_is_unmoved_by_a_fully_feed_matched_reference_family`
(`tests/test_m6_instrument_report.py`), with the option's own surface —
validation, the bit-identical default, the exactness of the cell integral
against an independent quadrature, multi-feed and the Y path, and the d=1
degeneracy — in `tests/test_momwire.py`. Reproduce with
`python scripts/m6_residue_cluster.py --dipole-feed-mirror` and
`--feed-matched`.

---

## 20. Addendum (2026-08-11): the extended kernel reaches the fourth cell — momwire#246

Until this arc the EK card split the instrument: the point-matched sinusoidal
solver served it (momwire#233) and the Galerkin one refused it, so a fat-wire
disagreement between the two cells confounded the kernel with the testing —
the exact confounding this instrument exists to remove.
`SinusoidalGalerkinSolver(extended_kernel=True)` now solves: the reduced
#205 folded fill untouched, a smooth extended-minus-reduced delta ADDED by
sinh-mapped quadrature on the coaxial equal-radius pairs, eligibility per
PAIR rather than NEC's per-end INDs. The design and the derivation are "The
extended thin-wire kernel (Eqs 84–98)" in `docs/sinusoidal_basis_design.md`;
this addendum records what the wired solver measures.

### The ladder, read as a shift

The oracle is the nec2c EK ladder read DIFFERENTIALLY: δZ = Z(EK on) −
Z(EK off) on one and the same mesh, each solver against nec2c's own δZ
column — so every reduced-kernel disagreement subtracts out and what is
compared is only what the kernel adds. On the thin-rung end nec2c's printed
digits cannot resolve its own shift (δZ = 0.000 ± half a printed digit at
Δ/a ≥ 24), so those rungs are excluded from the sign and ratio gates rather
than scored against noise.

| Δ/a | δZ nec2c | δZ galerkin | δZ collocation |
|--:|--:|--:|--:|
| 2.439 | −2.350 − 2.001j | −2.289 − 2.015j | −3.155 − 3.354j |
| 1.220 | −12.450 + 3.084j | −11.273 + 3.422j | −11.639 + 3.643j |
| 0.813 | −8.220 + 48.470j | −7.607 + 47.350j | −7.265 + 47.684j |
| 0.407 | +37.587 − 49.332j | +36.105 − 48.502j | +37.752 − 48.944j |

Against nec2c: **2.0 %, 9.5 %, 2.6 %, 2.7 %** — where the collocation
solver's own column reads 51 %, 7.7 %, 2.5 %, 0.7 %, so the Galerkin cell is
*stronger* than #233's at the fat end. Cross-basis, on the three rungs where
EK carries the answer: 3.5 %, 1.0 %, 2.8 %, all inside #249 §7's 25 % bar.
The sign structure matches nec2c on every rung the oracle resolves. Over the
PEC image the same differential reads −4.192 − 1.605j against the
point-matched −4.210 − 1.191j (9.5 %) on a fat monopole standing on the
plane — pinned as the image block's shift by forcing its EK payload off (the
answer moves) with the free-space control two orders away.

### The thin rungs: the test integral sees what collocation never samples

Above Δ/a ≈ 6 the Galerkin shift RUNS LARGER than the point-matched one —
0.146 Ω against 0.046 Ω at Δ/a = 24, 0.021 Ω against 0.0014 Ω at Δ/a = 122 —
and that is the fill, not a defect. The extended-minus-reduced field is O(1)
relative to the reduced one within a wire radius of a source-segment END
(0.75 of it at Δ/a = 122, against 4e-4 at mid-segment); a collocation point
sits at a segment CENTRE and never visits that region, while a Galerkin test
integral sweeps straight through it. So the two cells honestly read
different functionals of the same kernel, and the instrument's own axis —
same basis, different testing — is what the deviation measures. Pinned two
ways: refining the delta rule, the near rule and the test rule each moves it
by <1e-7; and rebuilding the whole EK fill from NEC's own EKSCX closed forms
— folded by subtraction, sharing no quadrature with the delta — reproduces
δZ to **eight figures**.

### Symmetry, the fill's own detector, with the kernel on

‖G−Gᵀ‖/‖G‖ on the ladder mesh: 5.8e-13 EK-on against 3.5e-13 reduced at
Δ/a = 2.4 (1.7×), 4.6e-13 against 2.1e-13 at Δ/a = 6.1 (2.2×). On the
Δ/a ≈ 1 monopole decks EK-on sits at 1.9e-12 against the reduced fill's
4.4e-14 — and that is the shared FAR test rule being short, not the pair
rule: `n_qp_test=16` takes it to 5.1e-15, *under* the reduced fill's own.
The falsifying contrast is in the battery: extend only j > i and the ratio
jumps to 1e-2 and stops responding to refinement. ‖G_ek − G_red‖/‖G_red‖
scales as the structural a²: 1.6e-1 (a = 0.05), 2.5e-3 (0.005), 3.7e-5
(5e-4), 2.6e-6 (5e-5), and a zero source radius returns the reduced tables
bit-for-bit.

### The resolution floor

Reduced-plus-delta is a near-cancelling decomposition — the two kernels
agree away from the wire — so an on-segment pair carries ~(H/ρ)² of
cancellation and float64 leaves ~ε·(H/ρ)² of the delta's peak behind. The EK
shift is itself O((ρ/H)²), so the two scale against each other: what reaches
Z stays **~1e-10 relative out to Δ/a ≈ 500**, and past **Δ/a ≈ 1e4** — where
the kernel is a 1e-8 effect anyway — the decomposition is noise-limited.
That floor belongs to reduced-plus-delta, not to any quadrature, and it
bounds how thin a wire this cell can resolve an EK correction for; the far
and on-segment tiers are gated separately so the sharp statement survives in
the tests.

### What still refuses

`ground_model="sommerfeld"` with a plane: the ground-wave remainder is a
separate evaluator, still reduced, and mixing an extended image with a
reduced remainder inside one ground model would be a silent half-answer.
Filed as momwire#287, with `BSplineSolver`'s #269 precedent (remainder left
reduced on a measured O((a/2h)²) argument) as the shape of the fix and the
re-measurement this cell's test integration forces as the open question.
`near_correction=False` also refuses under EK — the near path is where the
on-segment pairs are computed at all, not a refinement.

### Cost, and where it is pinned

With EK on the fused C++ far fill is skipped (it takes no eligibility mask)
behind the `_HAVE_GALERKIN_FAR_FILL_EK` capability flag, so an EK-on fill
costs what the pre-#194 numpy fill did until #246 unit C lands the twin;
EK-off is bit-identical to the pre-arc solver (defaulted vs explicit
`extended_kernel=False` on five decks, plus an entry counter proving the
four EK code objects are never entered). Pinned in
`tests/test_extended_kernel_galerkin.py` — the delta against the 80-bit
reference and against EKSCX, the ladder shift, symmetry, a → 0, the
off-path armor, the PEC image attribution and the pair-mask predicate —
with the acceptance/refusal seam mirrored in
`tests/test_extended_kernel.py`; the derivation replays in
`scripts/derive_galerkin_ek_delta.py`.
