# momwire#973 — the global Sommerfeld ACA stagnates, and its own rule cannot see it

## What the scripts are

- `repro_step.py` — replays captured `MomwireEngine` kwargs against
  `HMatrixSolver` / dense `BSplineSolver` to reproduce the `aca_tol` step.
- `probe_block_error.py` — wraps `aca_partial` and densifies every admissible
  block from the same callbacks, so each block's TRUE relative error sits
  beside the rank the stopping rule chose.
- `probe_rank_trace.py` — re-implements the ACA loop with instrumentation:
  the tested quantity and the true residual at every rank.
- `ladder_mesh.py` — six rungs of the skyloop mesh, each solved dense, with the
  check disabled, and as shipped.

## The finding

One block is responsible, and it is not a far block: the single global `n x n`
ACA in `_sommerfeld_global_lowrank`, which factorises the whole Sommerfeld
remainder on the premise that it is globally low rank. On skyloop_lmatch at
`aca_tol=1e-6`, n=184:

    184x184  rank 13  TRUE rel err 7.82e-01
    the other 22 blocks: median 5.6e-09, max 2.2e-07

The rank-by-rank trace shows why no tolerance fixes it:

    rank   ||u||*||v||   tol*||A~||   TRUE rel err
       5     4.02e-01     2.23e-06      7.8237e-01
       6     4.35e-02     2.23e-06      7.8229e-01
       9     7.62e-05     2.23e-06      7.8242e-01
      13     8.64e-07     2.23e-06      7.8242e-01   <- stops

The true error is FROZEN while the tested update falls six orders of
magnitude. At `aca_tol=1e-8` it stays frozen for forty further ranks (6 to 45)
and then a pivot escapes and it collapses (0.64, 0.55, 0.43, ... 0.017). This
is partial-pivoting stagnation: the updates are small because the pivots are
trapped in a subspace, not because the approximation is good. In that regime
the measured quantity is anti-correlated with progress, so no threshold on it
separates "converged" from "stuck", and lowering `aca_tol` only buys
iterations until the pivot escapes. That is exactly why 1e-4 through 1e-7 are
bit-flat and 1e-8 works.

## Scope, measured

Probe residual over 98 catalog designs (refined mesh, `aca_tol` 1e-6,
sommerfeld):

    95 healthy   median 3.4e-05, max 1.66e-03 (loops.triangular_skyloop)
     3 stagnant  1.0000 dipoles.dipole_turnstile
                 1.0000 loops.horizontal_loop
                 1.1287 loops.skyloop_lmatch

The two at exactly 1.0000 are `U @ V` collapsed to ~0 — the remainder
correction absent entirely.

Effect of the fix on those three, against dense:

    loops.skyloop_lmatch      4.803e-03 -> 4.674e-07
    loops.horizontal_loop     1.063e-04 -> 2.622e-06
    dipoles.dipole_turnstile  3.745e-08 -> 1.235e-10

and on five healthy designs the error is unchanged to every printed digit with
wall time inside noise (rhombic 11.661s -> 11.648s, bowtiearray2x4 8.132s ->
8.226s).

## The mesh ladder, six rungs

    n/edge  bases     OFF dI      ON dI      probe  rank  fb
        15     52  1.179e-05  1.179e-05  1.236e-03    23   n
        29     94  5.540e-03  1.578e-07  1.161e+00    94   Y
        43    136  5.573e-03  1.541e-06  1.182e+00   136   Y
        59    184  5.592e-03  4.138e-07  1.205e+00   184   Y
        75    232  7.710e-04  7.710e-04  2.198e-03    22   n
        91    280  5.613e-03  1.383e-06  1.196e+00   280   Y

Four rungs of six stagnate, so the failure is endemic to the geometry rather
than an artefact of one mesh. Note the rank when it fires: 94, 136, 184, 280 —
i.e. FULL rank. Q is simply not low rank on this geometry, which is the
docstring's own "past ~50" decision point being blown through with nothing
checking it.

**Rung 75 is a miss, recorded rather than tuned away.** Probe 2.198e-03, error
7.7e-04, no fallback. That probe is 1.3x the worst healthy catalog deck, so no
threshold separates mild degradation from health. The clean 600x gap is between
catastrophic stagnation and everything else, and the shipped 1e-2 sits in it.
The check is a catastrophe detector, not an error bound.

## Cost

The first version probed entry by entry and cost **1.76x on the sommerfeld test
lane** (13.1s -> 23.1s on `test_g17`/`test_g18`), because each 1x1 evaluation
re-pays the grid marshalling. Batching into one `k x k` block call returns the
lane to baseline (28.44s against main's 28.38s) and gives more probes, not
fewer.
