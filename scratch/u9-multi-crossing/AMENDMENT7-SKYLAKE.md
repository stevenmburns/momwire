# Amendment 7 on Skylake — the full-deck momwire far × 3 rung

Run 2026-09-14 18:53Z, one process, `systemd-run --user --scope -p MemoryMax=44G`.

| | |
|---|---|
| deck | `lpda_full_far3.nec`, **8,014 segments**, sha256 `f366a08d…c074496` (checked twice, immediately before the run) |
| momwire src | **`6549550`** (momwire main; `git diff 3331ef1 6549550 -- src tests` is empty, per the amendment's own note), `make build` with `MOMWIRE_REQUIRE_ACCEL=1` |
| antennaknobs | main `929407f05`, which carries `fdab21e20` |
| records | `u9-multi-crossing` at `7105826` (`4efa9cd` an ancestor) |
| NEC-5 | not used |

## Result

| | |
|---|---|
| **Z** | **52.0485 − 3.1861j** |
| reference (momwire full refine-1) | 52.0879 − 3.1846j |
| **\|ΔZ\|** | **0.0394 Ω** |
| **peak RSS** | **19.89 GB** (20,369.5 MB) |
| **wall** | **1,073 s = 17.9 min** |
| memory stop | not reached; no kill, no `REFUSED-BY-MEMORY` |

## Verdicts

| id | criterion | measured | |
|---|---|---|---|
| **PS7a** | peak RSS in [20, 40] GB | **19.89 GB** | **MISS** — 0.11 GB below the lower bound |
| **PS7b** | \|ΔZ\| ≤ 1.0 Ω of 52.0879 − 3.1846j | **0.0394 Ω** | **HIT**, by 25× |
| **PS7c** | wall ≤ 6 h | **0.30 h** | **HIT**, by 20× |

**The rung is a HIT on the amendment's own definition**, which is "the run
completes under the stop, with a finite Z and no error row, and PS7b holds" — all
four hold. PS7a is a prediction about memory and is not part of that definition;
it missed, and it missed *low*, which is the harmless direction.

## The projections, scored

PS7a's own projection was 3.9 GB × (8014/2690)² = **34.7 GB**, which is **74 %
high**.

Skylake's note, registered beside PS7a before the run, is the one to keep for next
time. The `full`/`r1` deck PS7a scaled from is not on this box, so the cheap
measurement was the two hash-controlled **one-node** decks: 424 seg → 543 MB,
1,240 seg → 1,615 MB, fitting RSS at **N^1.02** and projecting **10.5 GB** — **47 %
low**. Pure N² from the same point projected 65.9 GB, above the stop.

Rather than pick one, the note applied a **correction measured on this box two days
earlier**: the buried radial-screen ladder fitted RSS at N^1.04 on its lower rungs,
projected 3.9 GB at its top rung and measured 8.6 GB — a 2.2× understatement,
because the exponent steepens at the top of a range. 10.5 GB × 2.2 gave **≈ 23 GB**,
which is **16 % high** — closer than either the raw fit or PS7a's N², and the only
one of the three that put the run inside PS7a's band.

Wall: fitted 25 min, N² 45 min, actual **17.9 min** — the fit was high here, so the
steepening that applies to memory did not apply to time.

## Two notes for the reader

- The amendment's "when it lands, gate (i) is read at far × 3" step is **void** —
  Steve stopped (d) on 2026-09-14 — so this rung stands on its own as momwire's
  mesh step for finding 1. momwire's own step is
  s_momwire = \|Z_mw,full(far × 3) − Z_mw,full(r1)\| = **0.0394 Ω**.
- The deck is comma-delimited, so a segment count taken with
  `awk '$1=="GW"{s+=$3}'` reads **zero** on it. 8,014 is from a split on
  `[ ,\t]+`.

Rows: `d2_lpda_skylake.jsonl` (this rung) and `skylake_projection.jsonl` (the two
one-node points the projection was fitted on).
