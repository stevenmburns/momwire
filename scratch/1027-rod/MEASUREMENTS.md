# momwire#1027: the wholly buried rod's −1.20 % of R

Haswell (i7-4770K). Scratch only; no `src/momwire` change until the term is
named. AK plan unit U7 (`docs/plan-buried-scope-closure.md`, antennaknobs
0124079e6). Measure-first: every prediction below is committed before the run
that tests it, and every miss is reported.

## Setup

- momwire `1dbd384` (origin/main; `src/` identical to the v0.54.0 tag `260bd91`
  that antennaknobs pins), accelerators rebuilt with `make build`.
- antennaknobs `0124079e6`, engines `MomwireEngine` / `NEC5Engine`, decks from
  `NEC5Engine.deck()`.
- NEC-5: `nec5-linux/nec5cl`, sha256 `52a9489fb51a31d1…`, 1,282,144 B. #1027 used
  `nec5cl-x13` (sha256 `2068ae67a7fb1225…`, 1,265,856 B), which is not on this
  box. Equivalence on the #1027 decks, before anything else: all 9 banked rows of
  `rod_rows.jsonl` at L = 0.15 / 0.60 / 2.40 m, refine 1 / 2 / 4, re-solved here
  with the SAME deck sha256 give the SAME printed impedance.
- **NEC-5 reference (from 2026-09-13): `~/nec5-timing/nec5cl-x13-static`**,
  sha256 `7ebf343d7d01283b83fae8192cd9c6ca434b5befb232065fc12b4d729f06a9f3`,
  29,899,344 B, statically linked: the original x13 source and flags, static
  link. On `nec5_binary_check.py`, which rebuilds each of #1027's 15 banked rows
  (`rod_rows_1027.jsonl`, copied unchanged), **15/15 rows are identical in deck
  sha and printed Z on x13-static, and 15/15 on the local build 52a9489f**. The
  step 1 anchors re-run on x13-static print the same NEC-5 rows as on 52a9489f,
  row for row (`x13/anchor.json`). Results before the reference arrived are
  kept under `local-52a9489f/`; results from it onward are under `x13/`.
- Harness: #1027's `rod_ladder.py`, banked here unchanged as the geometry and
  row definition. Richardson is first order from refine 8 → 16:
  `dR_inf = 2·dR16 − dR8`, fraction = `dR_inf / R_momwire(16)`, dR = NEC-5 −
  momwire.

## Step 1 — reproduce the anchor rows on v0.54.0

Registered before the run:

| id | prediction | verdict |
|---|---|---|
| P1.0 | precondition, not a prediction: refine-16 segment counts equal #1027's `segs16` (36 / 136 / 524 at L = 0.15 / 0.60 / 2.40) | **MET** — 36 / 136 / 524 |
| P1.1 | ΔR/R reproduces #1027 within ±0.01 percentage points: −3.893 % / −1.203 % / −0.274 % | **HIT** — identical to the printed digit on all three |
| P1.2 | the #956 crossing fill is never entered: `cross_complete_block_split` calls = 0 on every rung | **HIT** — 0 on all six solves |
| P1.3 | momwire R(16) within 1e-4 relative of #1027's `R` (550.677 at L = 0.60) — the W-terms fix does not reach a wholly buried deck | **HIT** — bit-identical (relative difference 0.0) on all three |

A miss on P1.1 or P1.3 stops the unit here and is reported before any
localisation.

### Measured (`anchor.py`, `anchor-1dbd384.json`)

| L (m) | segs 8 / 16 | momwire R8 / R16 | NEC-5 R8 / R16 | dR8 | dR16 | dR∞ | **ΔR/R** | #1027 |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 22 / 36 | 1672.6800 / 1670.8663 | 1611.0 / 1607.5 | −61.6800 | −63.3663 | −65.0526 | **−3.893 %** | −3.893 % |
| 0.60 | 74 / 136 | 550.8184 / 550.6771 | 544.47 / 544.19 | −6.3484 | −6.4871 | −6.6258 | **−1.203 %** | −1.203 % |
| 2.40 | 270 / 524 | 176.8146 / 176.8042 | 176.35 / 176.33 | −0.4646 | −0.4742 | −0.4837 | **−0.274 %** | −0.274 % |

Four for four. momwire's R is bit-identical to #1027's run of 2026-09-10, so
nothing merged since — #1040, the 0.53.0 AVX2 work, the W-terms fix — reaches
a wholly buried rod, and the crossing fill is never entered.

**The NEC-5 side has a resolution floor.** The printout carries five
significant figures (`1.6075E+03`, `1.7633E+02`), so each R reading is
quantised to ±0.5 in its last digit: ±0.05 Ω at L = 0.15 m, ±0.005 Ω at
L = 2.40 m. Through `2·dR16 − dR8` that is up to ±3 × 10⁻⁵ of R, about
±0.01 percentage points on the fraction at every length, but ±3 % of the
fraction's own size at L = 2.40 m. Rungs whose fraction falls much below
0.1 % need that floor stated beside them.

## Step 2(a) — take the interface and the loss out

**What each engine runs.** Neither engine's front end, as antennaknobs drives
it, has a whole-space spelling, so each gets one built for this unit:

- **NEC-5**: its infinite-medium card (verified against our licensed
  materials), added to `NEC5Engine.deck(ground="free")`'s free-space deck.
  Segment sizes are unchanged, and are already tiny against λ_m ≈ 10.7 m.
- **momwire**: the wholly buried fill with its two interface terms removed —
  the ±=− image weight `a_m` set to zero and the below/below Sommerfeld
  remainder returning zeros — leaving the direct term at k_m, ε_m.
- **Deep burial** on both engines as the second comparand, to d = 18 m.
  momwire refuses 20 m (below/below pair separation 41.2 m, past its 4 λ_m
  table), and e^{−2αd} at 18 m is 1.8 × 10⁻⁴ (α = 0.2387 Np/m at soil A).

Controls, registered before the run. If one fails, the rung it guards is
withdrawn rather than read.

| id | control | verdict |
|---|---|---|
| C2a.1 | momwire whole-space: Z identical (≤ 1e-9 relative) at d = 0.2 and d = 18 m — only relative geometry can remain | pending |
| C2a.2 | momwire whole-space moves Z against the full fill at d = 0.2 m by ≥ 1e-2 relative (the knob is plumbed) and matches it at d = 18 m to ≤ 1e-3 | pending |
| C2a.3 | NEC-5 infinite-medium card with (ε_r, σ) = (1, 0) prints the same Z as the free-space deck | pending |
| C2a.4 | NEC-5 infinite-medium Z identical at d = 0.2 and d = 18 m, and NEC-5 deep burial at 18 m within 1e-3 relative of it | pending |

Predictions, registered before the run:

| id | prediction | verdict |
|---|---|---|
| P2a.1 | **whole space, both engines, soil A**: ΔR/R = −3.89 % at L = 0.15 m and −1.20 % at L = 0.60 m, each within ±0.05 pp — the interface is not in the path | pending |
| P2a.2 | **deep burial, both engines, L = 0.60 m**, d = 0.2 / 1 / 3 / 10 / 18 m: ΔR/R within ±0.03 pp of −1.203 % at every depth | pending |
| P2a.3 | **whole space, σ → lossless**, ε_r = 13, σ = 5 / 1 / 0.1 / 0 mS/m, L = 0.15 and 0.60 m: ΔR/R unchanged within ±0.05 pp at every σ, while R falls by more than 10× at L = 0.60 m | pending |

The competing outcomes for P2a.3, written down so the result cannot be
re-fitted: |ΔR/R| falling toward zero as σ → 0 makes the missing term
loss-borne; |ΔR/R| growing as R falls makes it an absolute term that does not
scale with R.

### Controls, smoke run on the local build (52a9489f), before any prediction

`step2a_medium.py controls`, L = 0.60 m, refine 8. Kept at
`local-52a9489f/controls.json`. The NEC-5 controls re-run on the reference
binary when it lands; the momwire ones cannot depend on it.

| id | measured | against the registered bar |
|---|---|---|
| C2a.1 | momwire whole-space Z at d = 0.2 vs 18 m: **6.3 × 10⁻¹⁵** relative; the switch was entered on both solves | **MET** |
| C2a.2 | full vs whole-space at d = 0.2 m: **2.9 × 10⁻³**; at d = 18 m: **1.6 × 10⁻¹⁰** | **MISSED** at 0.2 m (bar ≥ 1 × 10⁻²); met at 18 m |
| C2a.3 | NEC-5 infinite-medium card at (1, 0): 0.035755−14117j; free-space deck: 0.035755−14117j | **MET** — identical |
| C2a.4 | NEC-5 infinite medium at d = 0.2 vs 18 m: identical; NEC-5 buried at 18 m vs infinite medium: identical printed Z | **MET** |

**C2a.2 missed its registered bar, and the miss stays on the record.** The
bar was my estimate of how much the interface contributes to Z at 0.2 m depth,
not a test of the switch, and it was 3.4× too high. What the control exists
for, showing that the switch reaches the assembly, holds by twelve orders over
C2a.1's floor, and the 18 m half passes. Amended **before any prediction is
run**:

| id | control | verdict |
|---|---|---|
| C2a.2′ | the switch moves momwire's Z at d = 0.2 m by more than 10⁶ × C2a.1's floor, and matches the full fill at 18 m to ≤ 1e-3 | **MET** — 2.9e-3 vs 6.3e-15; 1.6e-10 |

The miss also sizes the interface. At d = 0.2 m it is worth |ΔZ| ≈ 2.3 Ω of
momwire's 550−552j, which is a third of the 6.6 Ω disagreement in R alone. An
interface-only mechanism would need NEC-5 to get that term wrong by about 3×.
That is an observation, not a verdict: P2a.1 is the test.

### C2a.5 — the two whole-space spellings against a known answer

Registered before its run, and before the P2a results are read. A structure in
a lossless homogeneous medium is the same structure in free space at
f·√ε_r, with impedance scaled by 1/√ε_r. The right side goes through each
engine's ordinary free-space path, so a construction error in either
whole-space spelling cannot reproduce it. ε_r = 13, σ = 0, L = 0.15 and
0.60 m, refine 8, the mesh held fixed and checked.

| id | control | verdict |
|---|---|---|
| C2a.5 | momwire whole-space Z equals free-space Z(f·√13)/√13 to ≤ 1e-6 relative in Z and ≤ 1e-4 in R; NEC-5 infinite-medium Z equals its free-space Z(f·√13)/√13 to ≤ 1e-4 relative in both (five printed figures) | pending |

If C2a.5 fails on either engine, that engine's whole-space rungs (P2a.1 and
P2a.3) are withdrawn rather than read.

### C2a.5 and the P2a ladders, on the local build (52a9489f)

**C2a.5 — MET for NEC-5, MISSED for momwire in Z.**

| L (m) | engine | whole space | free space at f·√13, ÷√13 | rel Z | rel R | bar |
|---|---|---|---|---|---|---|
| 0.15 | momwire | 0.00857146−3343.66j | 0.00857195−3343.86j | 5.9e-5 | 5.7e-5 | Z ≤ 1e-6 **missed**, R ≤ 1e-4 met |
| 0.60 | momwire | 0.133116−1089.23j | 0.133104−1088.90j | 3.0e-4 | 8.8e-5 | Z ≤ 1e-6 **missed**, R ≤ 1e-4 met |
| 0.15 | NEC-5 | 0.0078139−3220.5j | 0.00781378−3220.59j | 2.8e-5 | 1.5e-5 | met |
| 0.60 | NEC-5 | 0.12958−1076.9j | 0.129584−1076.92j | 2.1e-5 | 2.7e-5 | met |

The momwire miss is in the reactance: the two paths agree in R to < 1e-4 but
in X only to 3e-4. A 1e-6 bar assumed two different code paths (the complex-k
buried direct term and the real-k free-space fill) agree to near machine
class. They do not. **By the rule registered with C2a.5, momwire's
whole-space rungs are withdrawn**: P2a.1 and P2a.3 are not read as verdicts.
Their numbers are kept below, labelled, because a withdrawn reading is still
a record, and replacements that do not use the switch are registered at the
end of this section.

This is the second bar in this step set from an estimate rather than a
measurement (C2a.2 was the first). Both errors went the same way: I assumed
more agreement or more effect than the code delivers.

**P2a.2 — HIT.** Deep burial, both engines' ordinary buried paths, L = 0.60 m:

| d (m) | 0.2 | 1 | 3 | 10 | 18 |
|---|---|---|---|---|---|
| e^{−2αd} | 0.91 | 0.62 | 0.24 | 8.5e-3 | 1.8e-4 |
| momwire R16 | 550.677 | 549.454 | 549.461 | 549.465 | 549.465 |
| NEC-5 R16 | 544.19 | 543.00 | 543.01 | 543.01 | 543.01 |
| **ΔR/R** | **−1.2032 %** | **−1.1998 %** | **−1.1991 %** | **−1.1999 %** | **−1.1999 %** |

Within 0.004 pp of −1.203 % at every depth, while the interface's own weight
falls by 5000×. **The interface is not in the path.**

**Withdrawn readings (momwire whole-space switch, NEC-5 infinite-medium card).
Not verdicts.**

| L (m) | σ (mS/m) | momwire R16 | NEC-5 R16 | ΔR/R |
|---|---|---|---|---|
| 0.15 | 5 | 1670.43 | 1607.1 | −3.8923 % |
| 0.15 | 1 | 627.076 | 603.34 | −3.8838 % |
| 0.15 | 0.1 | 65.0688 | 62.606 | −3.8838 % |
| 0.15 | 0 | 0.0085946 | 0.0078621 | −8.2312 % |
| 0.60 | 5 | 549.465 | 543.01 | −1.1999 % |
| 0.60 | 1 | 206.341 | 203.93 | −1.1911 % |
| 0.60 | 0.1 | 21.5279 | 21.274 | −1.2050 % |
| 0.60 | 0 | 0.133201 | 0.12977 | −2.4966 % |

Read as indicative only: the fraction is flat while R falls 26× from 5 to
0.1 mS/m, which is neither of P2a.3's two competing outcomes. At lossless it
roughly doubles, 2.12× at 0.15 m and 2.08× at 0.60 m.

**An observation that needs no switch at all.** C2a.5's free-space solves, the
ordinary free-space path on both engines with no soil, no switch and no card,
already show the disagreement at refine 8: ΔR/R = −8.84 % at L = 0.15 m and
−2.65 % at L = 0.60 m, at 25.6 MHz. **The rod's disagreement survives with no
medium at all.** That moves the question from the medium to the rod and its
feed, which is 2(b) and 2(c).

### Replacements for the withdrawn rungs, registered before they run

Neither uses the momwire switch or the NEC-5 card. Where a withdrawn reading
has already been seen, the prediction says so rather than posing as blind.

| id | prediction | verdict |
|---|---|---|
| P2a.1′ | deep burial d = 18 m, soil A, both engines' buried paths: ΔR/R = −3.89 % ± 0.05 pp at L = 0.15 m (withdrawn reading seen: −3.8923 %) and −1.20 % ± 0.05 pp at L = 0.60 m (P2a.2's 18 m row, −1.1999 %) | pending |
| P2a.3′ | lossless whole space through scaled free space (f·√13, Z ÷ √13), both engines' free-space paths, refine 8 → 16 Richardson: −8.23 % ± 0.15 pp at L = 0.15 m and −2.50 % ± 0.15 pp at L = 0.60 m (not blind; withdrawn readings seen) | pending |
| P2a.3″ | **blind**: the same spelling at L = 2.40 m gives about twice its lossy fraction, −0.55 % ± 0.10 pp, on the 2.1× ratio the two shorter rods show | pending |

### The replacement rungs — on x13-static, and identical on 52a9489f

`step2a_replacements.py`, `x13/p2a1.json`, `x13/p2a3.json` (the local build's
twins under `local-52a9489f/` print the same rows to every digit).

| id | measured | verdict |
|---|---|---|
| P2a.1′ | deep burial 18 m, soil A: **−3.8923 %** at L = 0.15 m, **−1.1999 %** at L = 0.60 m | **HIT** |
| P2a.3′ | lossless medium as scaled free space, Richardson 8 → 16: **−8.2365 %** at L = 0.15 m, **−2.4960 %** at L = 0.60 m | **HIT** |
| P2a.3″ | blind, L = 2.40 m: **−0.5825 %** against −0.55 ± 0.10 | **HIT** |

| L (m) | segs 8 / 16 | momwire R8 / R16 (÷√13) | NEC-5 R16 (÷√13) | dR8 | dR16 | ΔR/R lossless | ΔR/R soil A | ratio |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 22 / 36 | 0.00857195 / 0.00859509 | 0.00786204 | −0.00075817 | −0.00073305 | −8.2365 % | −3.8933 % | 2.12 |
| 0.60 | 74 / 136 | 0.133104 / 0.133189 | 0.129767 | −0.0035209 | −0.0034226 | −2.4960 % | −1.2032 % | 2.07 |
| 2.40 | 270 / 524 | 2.31557 / 2.31598 | 2.30206 | −0.014341 | −0.013915 | −0.5825 % | −0.2736 % | 2.13 |

The withdrawn whole-space readings are reproduced by the switch-free spellings
to within 0.005 pp (−8.2312 against −8.2365, −2.4966 against −2.4960), which
is consistent with C2a.5's R agreement and says nothing more than that.

**Step 2(a) verdict, on the rungs that stand:** the disagreement is not the
interface (P2a.2, P2a.1′) and it does not need the soil at all — a lossless
homogeneous medium, equivalently free space at f·√13, carries it, converged
under the same Richardson as the buried rows (P2a.3′, P2a.3″). Its size in
free space is about twice its size in soil A, at all three lengths.

## Step 2(f) — which side moves: known answers that belong to neither engine

Raised on review: before reading the free-space disagreement as rod or feed,
converge it (done: P2a.3′, P2a.3″) and adjudicate it against answers neither
engine owns.

### P2f.3 — the short-dipole limit, read from numbers already in hand (not blind)

R_tri = 20π²(L/λ)², the radiation resistance of a short dipole carrying a
triangular current (Balanis, *Antenna Theory: Analysis and Design*, 3rd ed.,
2005, §4.3, eq. 4-37). Free space at f = 25.599414 MHz, λ = 11.710911 m,
Richardson R∞ from P2a.3′ / P2a.3″ multiplied back by √13.

| L (m) | kh | R_tri (Ω) | momwire R∞ | vs R_tri | NEC-5 R∞ | vs R_tri |
|---|---|---|---|---|---|---|
| 0.15 | 0.040 | 0.032384 | 0.0310735 | −4.05 % | 0.028521 | −11.93 % |
| 0.60 | 0.161 | 0.518144 | 0.480526 | −7.26 % | 0.46854 | −9.57 % |
| 2.40 | 0.644 | 8.29031 | 8.35184 | (+0.74 %) | 8.3032 | (+0.16 %) |

The L = 2.40 m row is outside the formula's range (kh = 0.64) and is shown only
for completeness.

**This does not name a side, and I am not carrying a correction I cannot
source.** The triangular value is exact only in the double limit kh → 0 and
Ω = 2 ln(L/a) → ∞. Here a = 0.5 mm, so Ω = 11.4 at L = 0.15 m and 14.2 at
0.60 m, and the finite-radius correction to the current's shape is of order
1/Ω, 7–9 %. That is the same size as both engines' misses. Its sign depends on
whether charge crowds the rod's ends, lengthening the effective dipole, or the
feed gap, shortening it. I can neither derive nor cite that sign with
confidence, so no finite-radius formula is claimed. The engine-independent
version of this check is to take the radius toward zero at fixed L and
extrapolate each engine's R / R_tri in 1/Ω, where the known answer is exactly
1. That is registered as 2(c) once the builder's radius knob is confirmed.

### P2f.1 / P2f.2 — power balance, each engine against itself

Registered before the run. In lossless free space every watt in is radiated,
so average power gain = P_rad / P_in = 1 on a self-consistent solve. Free space
at 25.599414 MHz, refine 16, L = 0.15 / 0.60 / 2.40 m.

- **momwire**: P_in from `MomwireEngine.input_power()`; P_rad = η₀k²/(32π²)
  ∫|M⊥|² dΩ over the full sphere at cell centres (180 × 8; the rod is
  φ-symmetric), from the engine's own segment dipoles.
- **NEC-5**: the printed average power gain from an RP run with averaging on,
  `NEC5Engine.average_power_gain(n_theta=90, n_phi=4)`. The rod is symmetric
  about θ = 90°, so the upper-hemisphere average is the sphere's. NEC-5's
  POWER BUDGET is not used: its RADIATED POWER is INPUT − WIRE LOSS by
  construction, so it cannot fail.

| id | prediction | verdict |
|---|---|---|
| P2f.1 | momwire average power gain = 1 ± 1e-3 at all three lengths | pending |
| P2f.2 | NEC-5 average power gain = 1 ± 1e-3 at all three lengths | pending |

Bar: a 1° midpoint rule on a sin³θ integrand is good to ~1e-5, and NEC-5
prints six figures, so 1e-3 is about 20× the numerical floor on both sides. If
both pass, power balance names no side; if one fails, it names that one.

### P2f.1 / P2f.2 measured — both MISSED the 1e-3 bar, at a size that still decides something

`step2f_power_balance.py` on x13-static, `x13/p2f_power_balance.json`.

| L (m) | momwire P_rad / P_in | NEC-5 average power gain | NEC-5 averaging solid angle |
|---|---|---|---|
| 0.15 | 0.992521 | 0.994712 | 1.4869π |
| 0.60 | 1.001870 | 0.994762 | 1.4869π |
| 2.40 | 0.999196 | 0.994780 | 1.4869π |

| id | verdict |
|---|---|
| P2f.1 | **MISSED** at 0.15 m (−7.5e-3) and 0.60 m (+1.9e-3); met at 2.40 m |
| P2f.2 | **MISSED** at all three (−5.2e-3) |

**The NEC-5 miss is at least partly my instrument.** The averaging ran over
1.4869π steradians, not the 2π hemisphere the RP spelling (90 θ × 4 φ,
cell-centred) was meant to cover: NEC-5 averages over the region its sample
points span, so four φ samples cover 270° and the θ samples 0.5–89.5°. A
reading that is the same to 7e-5 at three different lengths is what a fixed
weighting error looks like, not an engine error. I have not pinned NEC-5's
weighting, so 0.9947 is not interpretable at the 1e-3 level.

**The momwire miss has a plausible instrument cause I have not tested.** Its
far field is built from segment-midpoint dipoles, and the midpoint current is
the mean of the two knot currents. That is exact for a linear current but not
across the slope reversal at the feed, which matters most on the shortest rod.
It is non-monotone in L and could also be the solve.

**What the miss sizes, which the bar did not need to get right.** Each engine's
R_in is consistent with its own radiated power to within 0.75 % at every
length. The disagreement being adjudicated is 8.2 % at 0.15 m and 2.5 % at
0.60 m. If either engine's R_in were that far out of line with the currents it
solved for, its average gain would be off by the same 8 % or 2.5 %. Neither
is. So power balance names no side, and it moves the question: **the two
engines agree about how to read R from a current and disagree about the
current.**

### P2f.4 — where along the rod the currents differ (registered before the run)

Free space at 25.599414 MHz, refine 16, L = 0.15 and 0.60 m, one deck mesh for
both engines. NEC-5's printed segment-centre currents; momwire's spline current
evaluated at the same arc positions. Everything is normalised to each engine's
own feed current. m = Σ I·Δ / I_feed, the current moment a short dipole's far
field depends on.

| id | prediction | verdict |
|---|---|---|
| P2f.4a | self-consistency: R_NEC-5 / R_momwire = \|m_NEC-5 / m_momwire\|² to within 2e-2 at L = 0.15 m (moment-only far field, kL = 0.08) | pending |
| P2f.4b | **blind localisation**: more than half of m_momwire − m_NEC-5 comes from within 15 mm of the feed gap, not from along the rod or from its ends | pending |

The competing outcomes for P2f.4b, written down first: difference concentrated
at the rod's ends points to the end treatment or radius (2(c)); difference
spread evenly along the rod points to the kernel; difference at the feed points
to the feed spelling (2(b)). #1027's fraction tracks roughly −0.6 to −0.8 × the
10 mm feed wire's share of L, and AK `scratch/g1b-bs1-bs2/RESULTS.md` found an
electrically short dipole's Z moving several percent with the fed segment's
length in any medium. That is why the prediction points at the feed.

### P2f.4 measured

`step2f_currents.py` on x13-static, `x13/p2f_currents.json`.

| L (m) | R_NEC-5 / R_momwire | \|m_NEC-5 / m_momwire\|² | Δm within 15 mm of the feed | Δm within 15 mm of the ends |
|---|---|---|---|---|
| 0.15 | 0.91471 | 0.91474 | 0.396 | 0.082 |
| 0.60 | 0.97430 | 0.97480 | 0.128 | 0.013 |

| id | verdict |
|---|---|
| P2f.4a | **HIT** — agreement to 3e-5 at 0.15 m (bar 2e-2), and 5e-4 at 0.60 m. The R disagreement IS the current-moment disagreement |
| P2f.4b | **MISSED** — 0.40 and 0.13 of Δm lie within 15 mm of the feed, not more than half |

**What the profile shows instead. Read after the miss, so it is an observation,
not a verdict.** I/I₀, each engine normalised by its own feed current:

| L (m) | distance from feed | momwire | NEC-5 | NEC-5 / momwire |
|---|---|---|---|---|
| 0.15 | 2.50 mm | 0.98287 | 0.93739 | 0.9537 |
| 0.15 | 8.13 mm | 0.85808 | 0.82690 | 0.9637 |
| 0.15 | 14.38 mm | 0.77083 | 0.73702 | 0.9561 |
| 0.15 | 45.63 mm | 0.38387 | 0.36599 | 0.9534 |
| 0.15 | 65.00 mm | 0.15302 | 0.14396 | 0.9408 |
| 0.60 | 2.50 mm | 0.99444 | 0.97907 | 0.9845 |
| 0.60 | 8.12 mm | 0.95393 | 0.94223 | 0.9877 |
| 0.60 | 14.38 mm | 0.92587 | 0.91248 | 0.9855 |
| 0.60 | 45.63 mm | 0.80551 | 0.79413 | 0.9859 |
| 0.60 | 289.88 mm | 0.04509 | 0.04360 | 0.9670 |

From the first segment centre outward the two profiles differ by a roughly
constant factor, about 0.955 at L = 0.15 m and 0.986 at 0.60 m, and that factor
is √(moment ratio) (0.9564, 0.9873) to within 0.3 %. So Δm is spread along the
rod because the whole profile is rescaled, and the rescaling happens between
the source node and the first segment centre 2.5 mm away. NEC-5's current falls
6.3 % across that 2.5 mm at L = 0.15 m, momwire's 1.7 %. My localisation
metric, share of Δm by position, could not see a step that scales everything
after it. That is a flaw in how P2f.4b was posed, recorded as such rather than
re-scored.

The observation points at the source region: the feed wire's two 5 mm
segments and the delta-gap charge they have to represent. That is 2(b)'s axis.

## Step 2(b) — the source region

### A fact about #1027's ladder, from the decks and not a prediction: the source region is never refined

The emitted GW cards within 30 mm of the feed, identical at refine 1, 8 and 16
on both lengths:

| L (m) | refine | total segs | lower graded | lower near-feed | feed wire (EX) | upper near-feed | upper graded |
|---|---|---|---|---|---|---|---|
| 0.15 | 1 / 8 / 16 | 14 / 22 / 36 | 2 × 18.75 mm | 2 × 6.25 mm | **2 × 5.00 mm** | 2 × 6.25 mm | 2 × 18.75 mm |
| 0.60 | 1 / 8 / 16 | 22 / 74 / 136 | 2 × 18.75 mm | 2 × 6.25 mm | **2 × 5.00 mm** | 2 × 6.25 mm | 2 × 18.75 mm |

`refine` divides `rest_h`, the size the graded wires relax to far from the
feed. The grading toward the feed and the feed wire itself, `Wire(lo, hi,
ex=1)` with no `n_seg`, which auto-mesh gives two segments, do not depend on
it. So refine 8 → 16 adds panels only away from the source, and **the
Richardson extrapolation behind every #1027 fraction — and every fraction in
this record so far — is converged in the far mesh with the source region
frozen at 5 mm segments.** Both engines are handed the same frozen region.
NEC-5 excites the knot between the feed wire's two segments (`EX 0 tag 1 2`);
the momwire feed placement is checked below before anything is read from it.

This matches AK `scratch/g1b-bs1-bs2/RESULTS.md`, whose addendum found an
electrically short dipole's Z moving several percent with the fed segment's
length in any medium, approaching a finite limit about as gap^0.5.

### A second fact, from the translation and not a prediction: the two engines are not driven at the same port

Checked on the L = 0.60 m rod at refine 16 (peer review asked for it before any
current is read):

| | feed wire (10 mm) | source |
|---|---|---|
| NEC-5 deck | **2 segments** of 5.00 mm | `EX 0 5 1 2`: the **knot** between them, z = −0.500 m |
| momwire (`MomwireEngine._polylines` / `_feeds`) | **1 segment** of 10.00 mm (edges `[61, 2, 2, 2, 1, 2, 2, 2, 61]`) | delta gap at arclength 0.300 m: the **middle of that segment**, where there is no knot |

momwire's knots near the feed are −0.51125, −0.50500, −0.49500, −0.48875 m, so
nothing sits at −0.500. The same builder deck reaches the two engines as two
meshes that differ by one segment (135 against 136) and as two different
source definitions: a gap at a knot against a gap inside a degree-2 segment.
**The #1027 comparison is between two port spellings, not only two solvers.**
Whether that is the whole −1.20 % is the next measurement, not this entry.
