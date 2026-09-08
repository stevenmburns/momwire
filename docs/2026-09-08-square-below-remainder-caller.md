# 2026-09-08 — would a SQUARE below-remainder evaluation pay? (#981, no)

**Verdict: no, on three independent grounds.** The square matrix is not low
rank, an ACA on it is 3–16× *slower* than the dense fill it would replace, and
the per-call blocking it would remove is the consumer's memory band rather than
an accident. Step 4 of #981 stays struck; #914's plain C++ levers remain the
path.

This tests the open question left by
`2026-09-08-below-remainder-block-structure.md`: restructure the CALLER so the
projection sees the whole observer cloud at once — one `n × n` evaluation
instead of 520 calls of `30 × n`.

Measurement only. No kernel, no production change.

## (d) The premise holds — the restructure was well posed

The per-call observer sets union **exactly** to the source cloud at every rung:
3,924 / 3,924 at 12 radials, 7,812 / 7,812 at 24, and likewise at 36 and 48.
There is no overlap and nothing missing, so "one square evaluation" is a real
alternative and not a category error. It simply does not pay.

## (a) The square matrix is NOT low rank

Global ACA over the square remainder, `tol=1e-6`, row/column callbacks only
(the table is never densified — 3.9 GB at 48):

| radials | n | ACA rank | rank / n | #979 probe residual |
|--:|--:|--:|--:|--:|
| 12 | 3,924 | 230 | 0.059 | 6.35e-02 |
| 24 | 7,812 | 772 | 0.099 | 9.81e-04 |
| 36 | 11,700 | 1,108 | 0.095 | 1.15e-03 |
| 48 | 15,588 | **1,200 (cap hit)** | ≥0.077 | 2.72e-03 |

**The rank grows with n** — roughly 6–10 % of the matrix dimension, not a
constant. At 48 radials the run hit the 1,200 rank cap without converging. And
the probe residual never reaches the requested 1e-6: it lands between 6e-2 and
1e-3, i.e. three to five orders short.

This is the #973 lesson repeating, and the earlier extrapolation in this study
was wrong because of it. The per-call blocks are rank 5–6 *because they are far
blocks between separated clusters*; the square matrix additionally contains the
near-diagonal self-interaction at the hub, which is full rank. "Globally low
rank" fails exactly when near interactions are in the matrix.

## (b) The ACA is slower than the dense fill, by a growing margin

| radials | ACA wall | dense at the measured bulk rate | ratio |
|--:|--:|--:|--:|
| 12 | 1.30 s | 0.50 s | 2.6× slower |
| 24 | 21.83 s | 2.32 s | 9.4× slower |
| 36 | 68.48 s | 4.40 s | 15.6× slower |
| 48 | 98.13 s | 9.04 s | 10.9× slower (rank capped) |

The 48-radial dense estimate of 9.04 s sits beside #914's measured 7.2 s for
the same term — the same order, and slightly conservative, which is the right
direction for a claim that ACA loses.

Against the alternatives: a per-call ACA saves 3.83 s (11 % of wall) and a
per-block partition 4.87 s (14 %). The square route does not save; it costs
between 1 s and 89 s more.

## (c) The callbacks are NOT what kills it

| radials | bulk ns/entry | callback ns/entry | penalty |
|--:|--:|--:|--:|
| 12 | 32.8 | 213.9 | 6.5× |
| 24 | 38.0 | 152.6 | 4.0× |
| 36 | — | — | 4.3× |
| 48 | — | — | 3.4× |

Worth recording because this was the expected failure mode: #972's Python
row/column callbacks run ~30× a bulk fill per entry on the H-matrix path.
**Here they run 3.4–6.5×**, and at rank ~14 (what the thin blocks predicted)
that penalty would have been affordable. The callbacks were never the problem.
The rank is.

## (d) The per-call blocking is the consumer's memory band

`_field_galerkin_block` chunks the observer axis deliberately:

```python
chunk = max(1, (1 << 19) // max(n_src * q * q, 1))
for i0 in range(0, n_obs, chunk):
    proj = proj_fn(obs[i0 * q : i1 * q], ..., src, t_src)
```

with the docstring stating the intent: *"The observer axis is banded exactly as
the ±=+ fill bands it, so nothing bigger than `(d+1, d+1, chunk, n_src)` is
ever live."* The ~30-observer calls measured at 48 radials are that formula, not
an accident.

So the caller change would remove the band and reintroduce the multi-GB tensor
it exists to avoid — and the C++ `assemble_field_galerkin` (#914 lever 2)
consumes the projection chunk by chunk, so the restructure would also cut
across the accelerated assembly path it is supposed to be helping.

## Reproducer

`scratch/981-study/measure_square_caller.py --radials N [--swept-mem-mb M]`,
one process per arm, square table never materialised.
