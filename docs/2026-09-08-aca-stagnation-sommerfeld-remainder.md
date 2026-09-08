# 2026-09-08 — ACA stagnation on the global Sommerfeld remainder (#973)

**The sampled-residual check shipped with #979 is a catastrophe detector, not
an error bound**: it reliably separates a stagnated factorisation (probe
residual ≥ 1.0, current error ~5.5e-03) from a healthy one, and it does **not**
catch mild degradation — on the 232-basis rung of the ladder below the probe
reads 2.2e-03 and the current error stays at 7.7e-04 with no fallback, because
that probe is only 1.3× the worst healthy catalog deck.

## What fails, and why no tolerance fixes it

`_sommerfeld_global_lowrank` factorises the whole `n × n` Sommerfeld remainder
Q with one partial-pivoted ACA, on the premise that Q is smooth and therefore
globally low rank. On `loops.skyloop_lmatch` at `aca_tol=1e-6`, n = 184, that
block comes back at **rank 13 with a true relative error of 0.782** while the
other 22 admissible blocks sit at a median 5.6e-09.

Rank by rank, the quantity the stopping rule tests beside the residual it is
meant to bound:

| rank | ‖u‖·‖v‖ | tol·‖Ã‖ | true rel err |
|--:|--:|--:|--:|
| 5 | 4.02e-01 | 2.23e-06 | 7.8237e-01 |
| 6 | 4.35e-02 | 2.23e-06 | 7.8229e-01 |
| 9 | 7.62e-05 | 2.23e-06 | 7.8242e-01 |
| 13 | 8.64e-07 | 2.23e-06 | 7.8242e-01 ← stops |

The true error is **frozen** while the tested update falls six orders of
magnitude. At `aca_tol=1e-8` it stays frozen for forty further ranks (6–45)
before a pivot escapes and it collapses (0.64, 0.55, 0.43 … 0.017).

This is partial-pivoting **stagnation**: the updates are small because the
pivots are trapped in a subspace, not because the approximation is good. The
measured quantity is therefore *anti-correlated with progress*, and no
threshold on it — and no choice of `aca_tol` — separates "converged" from
"stuck". Lowering the tolerance only buys iterations until the pivot escapes,
which is exactly why 1e-4 through 1e-7 are bit-flat and 1e-8 works. That
explains #971/#974's default move rather than contradicting it.

## Scope: 98 catalog designs

Probe residual, antennaknobs catalog at the refined mesh, `aca_tol` 1e-6,
`('finite', 13.0, 0.005)`:

| probe residual | design |
|--:|---|
| 1.129e+00 | `loops.skyloop_lmatch` |
| 1.000e+00 | `dipoles.dipole_turnstile` |
| 1.000e+00 | `loops.horizontal_loop` |
| 1.660e-03 | `loops.triangular_skyloop` |
| 2.832e-04 | `loops.diamond_loop_turnstile` |
| 1.857e-04 | `verticals.four_square` |
| 1.629e-04 | `arrays.bowtiearray2x4` |
| 1.593e-04 | `specialty.hourglass` |

95 healthy: median 3.41e-05, max 1.66e-03. Three stagnant. **Nothing lands in
the 602× gap between 1.66e-03 and 1.0**, which is where the shipped 1e-2
threshold sits — 6× above every healthy deck, two decades under every stagnant
one.

**The two designs at exactly 1.0000 are the degenerate case**: `U @ V`
collapsed to ≈ 0, i.e. the Sommerfeld remainder correction was **absent
entirely** on `dipoles.dipole_turnstile` and `loops.horizontal_loop` under the
H-matrix path. Their *impedance* error was nonetheless small — 3.7e-08 and
1.1e-04 against dense — so an accuracy screen keyed on ΔZ alone did not flag
them, and the antennaknobs catalog ladder (antennaknobs#1282) scored them as
merely slow. The correction being missing and the answer being close are
independent facts; a rank/N or probe-residual column shows the first, ΔZ does
not.

Effect of the check on the three:

| design | before | after |
|---|--:|--:|
| `loops.skyloop_lmatch` | 4.803e-03 | 4.674e-07 |
| `loops.horizontal_loop` | 1.063e-04 | 2.622e-06 |
| `dipoles.dipole_turnstile` | 3.745e-08 | 1.235e-10 |

Five healthy designs are unchanged to every printed digit with wall time inside
noise (`wire.rhombic` 11.661 → 11.648 s, `arrays.bowtiearray2x4` 8.132 →
8.226 s).

## Six-rung mesh ladder

Skyloop's loop edges swept, each rung solved dense, with the check disabled
(`somm_residual_tol=inf`, the pre-#973 behaviour), and as shipped:

| n/edge | bases | OFF ΔI | ON ΔI | probe | rank | fallback |
|--:|--:|--:|--:|--:|--:|:--:|
| 15 | 52 | 1.179e-05 | 1.179e-05 | 1.236e-03 | 23 | n |
| 29 | 94 | 5.540e-03 | 1.578e-07 | 1.161e+00 | 94 | **Y** |
| 43 | 136 | 5.573e-03 | 1.541e-06 | 1.182e+00 | 136 | **Y** |
| 59 | 184 | 5.592e-03 | 4.138e-07 | 1.205e+00 | 184 | **Y** |
| 75 | 232 | 7.710e-04 | 7.710e-04 | 2.198e-03 | 22 | n |
| 91 | 280 | 5.613e-03 | 1.383e-06 | 1.196e+00 | 280 | **Y** |

Four rungs of six stagnate, so the failure is endemic to the geometry rather
than an artefact of one mesh. **Note the rank where the fallback fires — 94,
136, 184, 280, i.e. full rank.** Q is not low rank on this geometry at all,
which is `_sommerfeld_global_lowrank`'s own "past ~50" decision point being
blown through with nothing checking it.

Rung 75 is the miss quoted at the top of this page.

## Cost

The check must be batched. A first version probed entries one at a time and
cost **1.76× on the Sommerfeld test lane** (13.1 → 23.1 s on `test_g17` /
`test_g18`), because each 1×1 evaluation re-pays the grid marshalling. One
`k × k` block call returns the lane to baseline (28.44 s against main's
28.38 s) and yields more probes, not fewer.

## Consequences for other work

- **#977's crossover ladder** should carry a rank/N column and an accuracy
  column, not wall and RSS alone: on ground-coupled compact geometry the
  H-matrix route can be a full-rank dense fill wearing a low-rank costume, and
  at the previous default it could be silently wrong rather than slow.
- **antennaknobs#1265** (the cost/coverage column) must not quote the
  `dipoles.dipole_turnstile` or `loops.horizontal_loop` accelerator rows from
  antennaknobs#1282 as evidence of anything but speed; those two were solving
  without the remainder correction.
- **#914 lever 5** — ACA on the below-remainder projection — is the next place
  the same premise could fail in the same way. Noted, not built.

Reproducers: `scratch/973-study/` (`probe_block_error.py`,
`probe_rank_trace.py`, `ladder_mesh.py`, `repro_step.py`).
