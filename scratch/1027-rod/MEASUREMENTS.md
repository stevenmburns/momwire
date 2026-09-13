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
