---
title: "Choosing an engine: cost, memory, accuracy"
description: Which of momwire's nine engines to pick — the selection matrix, runtime and memory behaviour, the ground-cost ladder, and where each formulation earns its place.
---

Nine engines over seven solver families answer through one kernel, and
every one of them is reachable
by name — as a `--basis` argument, as its own `momwire-nec2c-<basis>`
command in [SimNEC's dialog](/reference/portal-usage/), or as the solver
behind [EZNEC's engine slot](/reference/eznec-nec5/) (which serves the
default). This page is the engine-side answer to the question every host
dialog raises: *which name, and what does it cost?*

The numbers here come from three standing benchmark studies — the
[solver-selection benchmark](https://github.com/stevenmburns/antennaknobs/blob/main/docs/status/2026-06-25-solver-selection-benchmark.md)
(10 designs × 7 engines, free space), the
[ground-model benchmark](https://github.com/stevenmburns/antennaknobs/blob/main/docs/status/2026-07-08-ground-model-benchmark.md)
(the same designs × 4 ground models), and the
[basis-convergence census](https://github.com/stevenmburns/antennaknobs/blob/main/docs/status/2026-07-20-basis-convergence-census.md)
(91 designs, meshes N=7–641). Timings quoted at N=81 segments/wire on a
4-core box unless stated.

## The selection matrix

The choice turns on two axes — total problem size, and single-structure
vs. array geometry:

| Antenna class | Use | Why |
| --- | --- | --- |
| Single elements, small loops, beams, multiband dipoles | **`bspline`** (degree 2 — the default everywhere), with **`sinusoidal`** as a cross-check | both solve in milliseconds here; d=2 converges at far coarser meshes (below), the other confirms the answer |
| Large single-wire structures (rhombics, long-wires, big loops) | **`hmatrix`** (ACA) | sub-quadratic scaling — the only engine in the field that wins `rhombic` at high segmentation |
| Arrays of identical / few-shape elements (loop/bowtie arrays, LPDA) | **`arrayblock`** | element-aware block-low-rank; near-linear scaling, 7–12× faster than the NEC-2 lineage on large arrays |
| Cross-checking against NEC-5 behaviour | **`razor-nec5`** | the formulation twin — rides the licensed engine's own convergence path (below) |
| Telling basis effects from testing effects | **`sinusoidal-galerkin`** | same basis as `sinusoidal`, variational testing — the attribution instrument of [Act V](/act-5/the-fourth-cell/) |
| Reading a textbook scheme against the modern ones | **`pulse`** | Harrington's 1967 pulse expansion, point-matched — the oldest thin-wire MoM there is, and the slowest-converging engine here by a wide margin (below) |
| Buried radials, screens, buried fed elements | **`bspline`** (or `bspline-d1`) — the dense B-spline pair carries the below-interface fill | serves impedance/currents/charges over the Sommerfeld ground; every other engine refuses buried decks by name, the compressed pair included — `hmatrix` and `arrayblock` have no per-segment media (see [the serve matrix](/reference/eznec-nec5/#what-refuses-and-why)) |

The same picks hold with a ground in play — the ground model changes what
a solve *costs*, not which engine wins it. One exception is capability,
not cost: wires below the interface are a dense-B-spline capability today,
and a buried deck's first solve pays a table-build of a minute or two
(momwire#568 tracks the accelerated fills).

:::caution
`arrayblock` / `hmatrix` only win at moderate-to-high segment counts on
their target geometries — below ~N=40 they are slower than a dense solve
even there. Don't reach for them on small problems.
:::

## Runtime

- **The ground-cost ladder is consistent everywhere:
  `free ≈ PEC < reflection-coefficient < Sommerfeld`.** PEC is nearly free
  (an image, no material solve); the reflection-coefficient ground runs
  ~1.5–3× a free-space solve on the dense bases; the full Sommerfeld
  ground ~2–5×. The Sommerfeld premium is mostly the *first* solve of a
  session: the interpolation-grid fill grows linearly with antenna size
  (not quadratically) and is reused across a band's frequencies, so warm
  sweep ticks undercut even the NEC-2 lineage's per-tick steady state on
  86/90 benchmark designs
  ([session benchmark](https://github.com/stevenmburns/antennaknobs/blob/main/docs/status/2026-07-19-somm-session-benchmark.md)).
- **ACA earns its place on the rhombic** — fastest engine in free space
  (~4.0 s, scaling ~2×/step where dense engines go ~5×/step), and under
  Sommerfeld the low-rank structure survives (the smooth ground remainder
  rides one compressed term): 9.3 s vs the dense B-spline's 18.7 s.
- **ArrayBlock dominates arrays on every ground** — LPDA free space:
  ~1.2 s vs the NEC-2 lineage's 14 s; Sommerfeld: 2.8 s vs 24 s.
- **At a fixed segment count, `sinusoidal` is the fastest dense basis** on
  small/medium single structures, on every ground. But the fair comparison
  is at fixed *accuracy*, and the census flipped that verdict — next
  section.

## Accuracy per segment: why `bspline` d=2 is the default

The basis-convergence census (91 designs) measured accuracy against mesh
density: **B-spline degree 2 is within 2 % of the converged answer at
N=15–21 segments per quarter-wave on 80 % of scorable designs**, where the
sinusoidal basis needs 3–15× more segments to reach the same value. Single
closed loops converge on d=2 as coarse as N=7. The gap is largest on
port-fed, junction-heavy, and closely-spaced-wire geometry — and the
momwire#182 instrument showed most of it is the *point matching*, not the
basis: rerun under Galerkin testing (`sinusoidal-galerkin`), the 1–23 %
junction-heavy gaps collapse to 0.01–0.33 %.

Dense cost scales as N² in memory and N²–N³ in fill/factor time, so an
engine that converges at one-third the mesh is roughly an order of
magnitude cheaper at equal accuracy. That, not a benchmark sprint, is why
`bspline` d=2 is the default in every portal and every host.

## The razor twins: which lane

### `pulse` is the honest slow one

`pulse` is `HarringtonSolver` — a pulse (piecewise-constant) current basis
with point matching, which is the scheme every other engine on this list was
chosen to improve on. It is here because reading a modern answer against the
classical one is worth being able to do without leaving the roster, not
because it competes: it converges at O(1/N) where degree-2 B-splines converge
far faster, so it wants a mesh several times finer for the same figure.

Measured on a 10 m dipole at 14 MHz against `bspline`'s converged
64.02 − 54.81j Ω:

| Segments | Δ/a | `pulse` |
| --- | --- | --- |
| 11 | 909 | 81.82 + 63.82j |
| 41 | 244 | 68.22 − 25.55j |
| 101 | 99 | 65.62 − 43.52j |
| 401 | 25 | 64.36 − 52.22j |

At the segment counts a host dialog defaults to, that is a *visibly*
different answer, and it is the formulation's own error rather than a defect
— pick it when that is what you want to see, and `bspline` otherwise.

It also serves less: no wire loading (an `LD 5` or `LD 6` is refused by name),
no junction ports, no node gaps, no extended kernel, and one scalar radius.
The EZNEC drop-in does not offer it at all — that dialect drives a *node*, and
this family puts its gap at the nearest segment centre, so it refuses every
deck there for the same reason `sinusoidal` does.

`razor` and `razor-nec5` are one solver class offered as two names, the
way `bspline`/`bspline-d1` are one class on two degrees. Both test the
tent expansion with NEC-5's razor-blade rule; they differ only in the
quadrature bound to the one path integral the public manual leaves
numerically open. **The choice is not a coin flip** (measured 2026-08-18):

| | `razor-nec5` | `razor` (converged GL quadrature) |
| --- | --- | --- |
| Role | Interactive lane | Convergence / certification lane |
| Speed | Sub-second to N≈300–400 free, N≈200–400 grounded; 2–4× behind `bspline` beyond that | 12–80× slower than `bspline`; over a second even at N=100 under any ground |
| Memory | Same order as the other dense bases | Exceeds an 8 GB working set by N≈800 grounded / N≈1600 free |
| Use for | Ordinary solves, A/B checks against NEC-5 behaviour | Convergence ladders, certification against NEC-5 printouts |

On the models where we hold a licensed reference, `razor-nec5` rides the
licensed engine's own convergence path at the 0.01 % level — it converges
*along* NEC-5's trajectory, not merely to its endpoint. The refusal
boundary — down to K≥3 junction ports alone, now that node gaps
(momwire#603), the extended kernel, and contact over finite grounds
(momwire#624) are all served — is documented in
[`docs/razor-solver.md`](https://github.com/stevenmburns/momwire/blob/main/docs/razor-solver.md),
each with a named message at construction.

## Memory

The dense bases form an N×N complex matrix: memory grows as **N²**, and a
runaway segment count costs hundreds of megabytes before it costs minutes.
Practical envelopes on an 8 GB working set: the plain-`razor`
certification lane exceeds it by N≈800 grounded / N≈1600 free (the table
above); the other dense bases reach further at the same N because their
peak was engineered down deliberately — the memory-release series
(momwire 0.29.0's certified peaks, then the #318/#323 B-spline reductions)
trimmed multi-gigabyte transient peaks to near the resident matrix size.
The compressed engines (`hmatrix`, `arrayblock`) skip the dense matrix
entirely, which is exactly why they exist for large problems.

## The extended kernel, in one paragraph

Every engine solves a thin-wire equation; the `EK` card's O(a²) tube
correction matters below Δ/a ≈ 3 and costs about 1.0–1.3× the reduced
solve. **All six families serve it** — the razor twins were the last
holdout, reduced-kernel by design until momwire#603 gave them the tube
correction too. One narrow refusal survives: `sinusoidal-galerkin` declines
`EK` on a deck where two wires of different radii meet at a junction,
measured divergent rather than merely inaccurate there. On ordinary thin
wire, leave it off: it changes the answer by less than the mesh does.

## The workbench view

The [antennaknobs solver guide](https://antennaknobs.dev/reference/solver/)
carries the same engines from the app's side — workbench controls, the
convergence-sweep tool, feed-model choices, and the hosted instance's size
caps — and the
[validation story](https://antennaknobs.dev/reference/validation/) holds
the full three-formulation parity record and the wild-corpus census.
