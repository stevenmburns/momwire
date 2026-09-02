---
title: "Choosing an engine: cost, memory, accuracy"
description: Which of momwire's nine engines to pick — the selection matrix, runtime and memory behaviour, the ground-cost ladder, and where each formulation earns its place.
---

Nine engines over seven solver families answer through one kernel, and
every one of them is reachable
by name — as a `--basis` argument, as its own `momwire-nec2c-<basis>`
command in [SimNEC's dialog](/reference/portal-usage/) (all but the razor
pair: a NEC-2 deck feeds a segment *centre* and razor places its gap at a
knot, so that front door refuses it by name, momwire#821), or as the
solver behind [EZNEC's engine slot](/reference/eznec-nec5/) (which serves
the default, and is where razor lives). This page is the engine-side
answer to the question every host dialog raises: *which name, and what
does it cost?*

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

## The razor lane, and its certification twin

`razor-2p` — `razor-nec5` is its deprecated spelling — tests the tent
expansion with NEC-5's razor-blade rule at NEC-5's own identified two-point
quadrature. It is the orderable member of `RazorSolver`. The class also
takes Gauss-Legendre nodes along the same testing path, which was a second
roster entry, plain `razor`, until momwire#753 retired it (decided
2026-09-02): the two lanes are one class differing only in that sampling
choice, and measured 2026-08-18, the GL lane cost 20x the wall time
(20.2 s vs 0.97 s at N=1600 free space) for a 0.001 Ω difference from
`razor-2p` — not worth ordering. The class stays; the GL lane is reached
by constructing `RazorSolver(nec5_quadrature=False, ...)` directly rather
than through `--basis` or a portal name.

| | `razor-2p` / `razor-nec5` | GL quadrature (`RazorSolver` constructed directly) |
| --- | --- | --- |
| Role | Interactive lane | Convergence / certification lane |
| Speed | Sub-second to N≈300–400 free, N≈200–400 grounded; 2–4× behind `bspline` beyond that | 12–80× slower than `bspline`; over a second even at N=100 under any ground |
| Memory | Same order as the other dense bases | Exceeds an 8 GB working set by N≈800 grounded / N≈1600 free |
| Use for | Ordinary solves, A/B checks against NEC-5 behaviour | Convergence ladders, certification against NEC-5 printouts |

On the models where we hold a licensed reference, `razor-2p` rides the
licensed engine's own convergence path at the 0.01 % level — it converges
*along* NEC-5's trajectory, not merely to its endpoint. Node gaps
(momwire#603), the extended kernel and contact over finite grounds
(momwire#624) are all served now; what the row still refuses — K≥3 junction
ports, buried wires and the crossing (the buried arc, momwire#812/#813, is
lifting these), contact under the reflection-coefficient ground, and a feed
named at a segment *centre*, which is why razor is not a nec2 engine
(momwire#821) — is documented in
[`docs/razor-solver.md`](https://github.com/stevenmburns/momwire/blob/main/docs/razor-solver.md)
and in [the capability
matrix](https://github.com/stevenmburns/momwire/blob/main/docs/capability-matrix.md),
each with a named message.

## `pulse`: the honest slow one

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

## Memory

The dense bases form an N×N complex matrix: memory grows as **N²**, and a
runaway segment count costs hundreds of megabytes before it costs minutes.
Practical envelopes on an 8 GB working set: the GL-quadrature certification
lane (`RazorSolver` constructed directly with `nec5_quadrature=False` —
off the `--basis` roster since momwire#753) exceeds it by N≈800 grounded /
N≈1600 free (the table above); the other dense bases reach further at the same N because their
peak was engineered down deliberately — the memory-release series
(momwire 0.29.0's certified peaks, then the #318/#323 B-spline reductions)
trimmed multi-gigabyte transient peaks to near the resident matrix size.
The compressed engines (`hmatrix`, `arrayblock`) skip the dense matrix
entirely, which is exactly why they exist for large problems.

These envelopes were measured at `RazorSolver`'s outer path order of 32 (so
64 observation points per testing path), which was the default until
momwire#800 made it DERIVE from the mesh on 2026-09-02. At the mesh
densities in this section the derivation returns 16, halving the outer
integral's cost — so the N≈800 / N≈1600 figures are the conservative
reading rather than the current one. An explicit `n_qp_path=32` reproduces
what they were measured at, bit for bit.

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
