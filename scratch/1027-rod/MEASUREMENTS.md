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
| P1.0 | precondition, not a prediction: refine-16 segment counts equal #1027's `segs16` (36 / 136 / 524 at L = 0.15 / 0.60 / 2.40) | pending |
| P1.1 | ΔR/R reproduces #1027 within ±0.01 percentage points: −3.893 % / −1.203 % / −0.274 % | pending |
| P1.2 | the #956 crossing fill is never entered: `cross_complete_block_split` calls = 0 on every rung | pending |
| P1.3 | momwire R(16) within 1e-4 relative of #1027's `R` (550.677 at L = 0.60) — the W-terms fix does not reach a wholly buried deck | pending |

A miss on P1.1 or P1.3 stops the unit here and is reported before any
localisation.
