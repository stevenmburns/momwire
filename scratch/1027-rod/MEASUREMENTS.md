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
