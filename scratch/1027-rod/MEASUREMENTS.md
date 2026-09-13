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
