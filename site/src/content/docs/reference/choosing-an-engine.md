---
title: "Choosing an engine: cost, memory, accuracy"
description: Which of momwire's eight engines to pick — the selection matrix, runtime and memory behaviour, the ground-cost ladder, and where each formulation earns its place.
---

Eight engines over seven solver families answer through one kernel, and
every one of them is reachable
by name — as a `--basis` argument, as its own `momwire-nec2c-<basis>`
command in [SimNEC's dialog](/reference/portal-usage/) (all but
`razor2p`: a NEC-2 deck feeds a segment *centre* and razor places its gap at a
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
(91 designs, meshes N=7–641), plus a 2026-11-23 segment-count ladder
on five antennas (below). Timings quoted at N=81 segments/wire on a
4-core box unless stated.

## The selection matrix

The choice turns on two axes — total problem size, and single-structure
vs. array geometry:

| Antenna class | Use | Why |
| --- | --- | --- |
| Single elements, small loops, beams, multiband dipoles | **`bs2`** (degree 2 — the default everywhere), with **`sinusoidal`** as a cross-check | both solve in milliseconds here; d=2 converges at coarse meshes to the answer NEC-5's formulation converges to (below), and `sinusoidal` confirms it where it converges |
| Large single-wire structures (rhombics, long-wires, big loops) | **`hmatrix`** (ACA) | sub-quadratic scaling — the only engine in the field that wins `rhombic` at high segmentation |
| Arrays of identical / few-shape elements (loop/bowtie arrays, LPDA) | **`arrayblock`** | element-aware block-low-rank; near-linear scaling, 7–12× faster than the NEC-2 lineage on large arrays |
| Cross-checking against NEC-5 behaviour | **`razor2p`** | the formulation twin — rides the licensed engine's own convergence path (below) |
| Telling basis effects from testing effects | **`sg`** | same basis as `sinusoidal`, variational testing — the attribution instrument of [Act V](/act-5/the-fourth-cell/) |
| Reading a textbook scheme against the modern ones | **`pulse`** | Harrington's 1967 pulse expansion, point-matched — the oldest thin-wire MoM there is, and the slowest-converging engine here by a wide margin (below) |
| Buried radials, screens, buried fed elements | **`bs2`** (or `bs1`), with **`sg`** and **`razor2p`** as independent buried lanes | each carries its own below-interface fill and serves impedance/currents/charges and the radiation pattern over the Sommerfeld ground; `sinusoidal` serves wholly buried and detached decks and a single near-normal crossing (its two members' tilts off vertical summing to 30° or less), such as a ground rod or the buried radial vertical, whose rise meets a buried hub below the plane; it refuses oblique crossings, a hub at the crossing node itself, and several crossings per deck by name, `pulse` refuses buried decks by name, and so does the compressed pair — `hmatrix` and `arrayblock` have no per-segment media (see [the serve matrix](/reference/eznec-nec5/#what-refuses-and-why)) |

The same picks hold with a ground in play — the ground model changes what
a solve *costs*, not which engine wins it. One exception is capability,
not cost: wires below the interface are served by `bs2`, `bs1`,
`sg` and `razor2p`, and by `sinusoidal` when no wire
crosses the interface or one near-normal crossing does (a ground rod, or a
radial screen whose rise meets its hub below the plane; an oblique crossing,
or radials meeting the mast at the plane itself, refuse by name), and a
buried deck's first solve
pays a one-time build of its below-interface Sommerfeld tables.

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

One route differs on purpose: the NEC-2 drop-in's `sinusoidal`
(`momwire-nec2c-sinusoidal`, and `momwire-nec2d-sinusoidal` for 4nec2) takes NEC-2's free-end cap by default, so that
it answers like NEC-2 engines, where every other host and launcher takes the
current to zero at a free end. Set `MOMWIRE_X_NEC2_END_CAP=0` for the
zero-current answer there; [the portal page](/reference/portal-usage/#choosing-the-physics-one-command-per-engine)
has the details. It moves Zin most on fat wires, and it is a match to NEC-2,
not a better model of a real wire end.

## Accuracy per segment: why `bs2` d=2 is the default

A segment-count ladder on five antennas (momwire 0.97.1, 2026-11-23)
measured how many segments each engine needs to settle within 1 % of the
answer it converges to, and whether the engines converge to the same
answer. The antennas: an inverted vee in free space and over average soil,
a hentenna, Ward AE6TY's 20:1 tapered dipole
([antenna-problem-decks](https://github.com/stevenmburns/antenna-problem-decks)),
and L. B. Cebik W4RNL's 3-element 20 m yagi with telescoping, tapered
elements.

| antenna | `bs2` | `razor2p` | `sinusoidal`, NEC-2 rules | `sinusoidal`, NEC-4.2 rules |
| --- | --- | --- | --- | --- |
| inverted vee, free space | 124 | 452 | 123 | 123 |
| inverted vee, over soil | 124 | 780 | 123 | 123 |
| hentenna | 122 | 1,332 | not converged at 12,000 | not converged |
| tapered dipole | 31 | 211 | not converged | not converged |
| tapered yagi | 428 | 1,770 | not converged | not converged |

- **`bs2` and `razor2p` converge to the same answer**, within 0.01 to
  0.23 Ω on every case, and `bs2` gets there with 4 to 11 times fewer
  segments. `razor2p` is NEC-5's formulation, so this is also the answer
  NEC-5 converges to.
- **Point-matched `sinusoidal` converges to that answer on a simple wire**
  like the inverted vee, with as few segments as `bs2`. It does not on
  every antenna. On the hentenna the NEC-2 rules are still 1.9 Ω short at
  12,000 segments and the NEC-4.2 rules settle about 8.7 Ω away. Where the
  wire's radius steps, neither converges: on the tapered yagi the NEC-2
  rules end 55 Ω away (10.6 + j47.7 against 29.8 − j4.3 Ω), and refining
  the mesh makes it worse.
- **Most of that gap is the point matching, not the basis**: rerun under
  Galerkin testing (`sg`), the junction-heavy gaps of an earlier 91-design
  census collapsed from 1–23 % to 0.01–0.33 % (momwire#182).

So the reason `bs2` is the default is that it reaches the same converged
answer as NEC-5's formulation, on every antenna tried, at a fraction of
the segments; not that it needs fewer segments than `sinusoidal` where
`sinusoidal` converges. Dense cost scales as N² in memory and N²–N³ in
fill/factor time, so 4 to 11 times fewer segments is more than an order
of magnitude cheaper at equal accuracy.

## The razor lane, and its certification twin

`razor2p` tests the tent
expansion with NEC-5's razor-blade rule at NEC-5's own identified two-point
quadrature. It is the orderable member of `RazorSolver`. The class also
takes Gauss-Legendre nodes along the same testing path, and that lane is not
on the roster (momwire#753): the two lanes are one class differing only in
that sampling choice, and measured 2026-08-18, the GL lane cost 20x the wall
time (20.2 s vs 0.97 s at N=1600 free space) for a 0.001 Ω difference from
`razor2p` — not worth ordering. The class stays; the GL lane is reached
by constructing `RazorSolver(nec5_quadrature=False, ...)` directly rather
than through `--basis` or a portal name.

| | `razor2p` | GL quadrature (`RazorSolver` constructed directly) |
| --- | --- | --- |
| Role | Interactive lane | Convergence / certification lane |
| Speed | Sub-second to N≈300–400 free, N≈200–400 grounded; 2–4× behind `bs2` beyond that | 12–80× slower than `bs2`; over a second even at N=100 under any ground |
| Memory | Same order as the other dense bases | Exceeds an 8 GB working set by N≈800 grounded / N≈1600 free |
| Use for | Ordinary solves, A/B checks against NEC-5 behaviour | Convergence ladders, certification against NEC-5 printouts |

On the models where we hold a licensed reference, `razor2p` rides the
licensed engine's own convergence path at the 0.01 % level — it converges
*along* NEC-5's trajectory, not merely to its endpoint. Node gaps
(momwire#603), the extended kernel (on a buried deck, on the above-ground
wires, as `bs2` serves it; momwire#1614), contact over the Sommerfeld
ground (momwire#624), buried wires and the crossing junction are all served; what
the row still refuses — a feed named by arclength (`feeds=`) at a junction
of three or more wire ends (a deck's node address serves a source on a
crossing node under one rising wire at any member count), a
wire crossing the interface mid-span rather than at a junction, ground
contact combined with buried wires, contact under the
reflection-coefficient ground, and a feed named at a segment *centre*, which
is why razor is not a nec2 engine (momwire#821) — is documented in
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

Measured on a 10 m dipole at 14 MHz against `bs2`'s converged
64.02 − 54.81j Ω:

| Segments | Δ/a | `pulse` |
| --- | --- | --- |
| 11 | 909 | 81.82 + 63.82j |
| 41 | 244 | 68.22 − 25.55j |
| 101 | 99 | 65.62 − 43.52j |
| 401 | 25 | 64.36 − 52.22j |

At the segment counts a host dialog defaults to, that is a *visibly*
different answer, and it is the formulation's own error rather than a defect
— pick it when that is what you want to see, and `bs2` otherwise.

It also serves less: no wire loading (an `LD 5` or `LD 6` is refused by name),
no junction ports, no node gaps, no extended kernel, and one scalar radius.
The EZNEC drop-in does not offer it at all — that dialect drives a *node*, and
this family puts its gap at the nearest segment centre, so it refuses every
deck there for the same reason `sinusoidal` does.

## Memory

The dense bases form an N×N complex matrix: memory grows as **N²**, and a
runaway segment count costs hundreds of megabytes before it costs minutes.
Practical envelopes on an 8 GB working set: the GL-quadrature certification
lane (`RazorSolver` constructed directly with `nec5_quadrature=False`, which is
not on the `--basis` roster) exceeds it by N≈800 grounded /
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
solve. **Every family but `pulse` serves it** (razor: momwire#603).
One narrow refusal: `sg` declines
`EK` on a deck where two wires of different radii meet at a junction,
measured divergent rather than merely inaccurate there. On a deck with
buried wires `bs2`, `razor2p` and `sg` (momwire#1626) serve it in part: extended on the above-ground wires, reduced on
the buried ones; `sinusoidal` solves such a deck reduced (momwire#1636). On
ordinary thin wire it changes the answer by less than the mesh does. A NEC-2 deck asks for it
with its `EK` card. NEC-4 and NEC-5 decks get it by default (momwire#1326),
because both programs' thin-wire models behave as the extended kernel.
Where a deck or a basis cannot take it, such a deck falls back to the reduced
kernel with an advisory.

## The workbench view

The [antennaknobs solver guide](https://antennaknobs.dev/reference/solver/)
carries the same engines from the app's side — workbench controls, the
convergence-sweep tool, feed-model choices, and the hosted instance's size
caps — and the
[validation story](https://antennaknobs.dev/reference/validation/) holds
the full three-formulation parity record and the wild-corpus census.
