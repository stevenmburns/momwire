---
title: "15 · The fourth cell: an instrument for blame"
description: Chapters 4 and 5 left a confound — every sinusoidal-vs-spline comparison mixed the basis with the testing scheme. Building the missing basis × testing cell turned disagreement into attribution, and most of what this tree had filed as "basis gaps" turned out to be something else.
---

[Chapter 4](/act-2/sinusoids/) gave you NEC's bet — piecewise-sinusoidal
currents, matched at segment centres. [Chapter 5](/act-2/splines/) gave you the
B-spline alternative — a different basis, tested by *integration* (Galerkin)
instead of point-sampling. When the two solvers disagreed on an antenna, which
ingredient did you blame?

You couldn't. The two solvers differed in **two things at once** — basis *and*
testing — so every discrepancy was a confounded experiment. For years this
tree's issues filed such gaps under "basis effect" for lack of a way to know
better. [momwire#182](https://github.com/stevenmburns/momwire/issues/182)
built the missing instrument: a **sinusoidal-Galerkin** solver — the same
three-term basis and junction tables as chapter 4, but with test rows formed
by integrating the field against each basis function, the way chapter 5 does.
The basis × testing matrix got its fourth cell, and any gap could suddenly be
attributed: persists under both testings → basis; vanishes → testing.

|  | point-matched | Galerkin |
|---|---|---|
| **sinusoidal** | `SinusoidalSolver` (ch. 4) | `SinusoidalGalerkinSolver` — **the new cell** |
| **B-spline** | — | `BSplineSolver` (ch. 5) |

## The free oracle

A correct Galerkin fill has a property collocation cannot fake: the matrix is
**symmetric by construction** — reciprocity, to machine precision. That made
`‖Z − Zᵀ‖/‖Z‖ < 1e-10` the gate for the whole build: sloppy quadrature breaks
it, and no amount of plausible-looking impedance can restore it. The hard 20%
was where [chapter 6](/act-2/quadrature/) said it would be — the near-singular
integrals. The kernel's endpoint terms put a spike of width ~`a` (the wire
radius) into every node-sharing test integral; a graded rule that shrinks
panels dyadically toward the endpoints, with the panel count *derived* from
the geometry's own `a/h`, met the gate at default settings at a fraction of
brute-force cost.

Two instrument-grade side-findings from that gate work, both worth keeping:

- **Symmetry is a uniform-radius property.** The thin-wire kernel regularizes
  with the *observer's* radius, so mixed per-wire radii make the kernel itself
  asymmetric — no testing scheme can restore a symmetry the kernel doesn't
  have. Where radii mix, the oracle is brute-force quadrature refinement, not
  reciprocity.
- **A drifting feed fakes non-convergence.** The delta-gap snaps to the
  nearest segment centre, so a feed that isn't a segment centre at every mesh
  density moves by up to h/2 as you refine — an O(h) perturbation of the
  *problem* that can make a perfectly good solver appear to walk away from its
  own converged answer. Every convergence sweep in this act pins the feed to a
  point that is a segment centre at every N.

And one epistemic one: **"the converged impedance" does not exist.** The
delta-gap's width is the segment length, so refining the mesh shrinks the
source, and the gap reactance drifts logarithmically without limit. Every
error figure below is stated against a *named* reference — fine solves plus
Richardson extrapolations — and gated on the verdict that survives the whole
family.

## What the instrument read

The full sweep — the [antennaknobs#521](https://github.com/stevenmburns/antennaknobs/issues/521)
residue cluster, the [#478](https://github.com/stevenmburns/antennaknobs/issues/478)
near-open designs, eleven designs by four solver columns — lives in the
[instrument report](https://github.com/stevenmburns/momwire/blob/main/docs/sinusoidal-galerkin-instrument-report.md),
tables generated verbatim by a checked-in harness. The taxonomy it produced:

**Testing effects** — the big class, and the surprise. The entire T/X-junction
cluster (hentenna, hourglass, their arrays, the discone) is *testing*-limited:
point-matched↔B-spline gaps of 1.2–23% collapse to 0.01–0.33% when only the
testing changes. On the hentenna, Galerkin at 119 segments holds an answer
collocation hasn't reached at 1835. This also closed an arbitration open since
momwire PR #45: the hentenna's anti-convergent impedance columns — the
sinusoidal solver and PyNEC in lockstep, fitted exponent ≈ −0.5 — were the
*point matching*, not the basis.

**Feed-model effects** — the axis nobody knew was there. The near-open,
high-Q designs (#478's lazy-H, the V-beam) barely respond to the testing swap;
what collapses their residual — 55–74×, down to ~0.02% — is matching the
**source model** (a point delta-gap versus a segment-wide one). On a clean
dipole the matched-feed gap between the sinusoidal-Galerkin and B-spline
solvers is 4×10⁻⁸: with basis *and* testing *and* feed matched, there is
essentially nothing left.

**Basis effects** — real, small, and rate-only. The high-Q star and L-shape
geometries are genuinely basis-limited: swapping the testing buys ~1.01×,
swapping the basis buys ~1.4×. But refined, all schemes agree to ≤4×10⁻⁴ —
so a "basis effect" in this tree is a statement about **coarse-mesh cost**,
never about what the solvers converge to.

**Not attributable — and saying so is the result.** The helix's builder emits
one segment per winding chord and ignores the mesh knob entirely, so its
convergence "ladder" re-solved one identical mesh at every rung. Until it has
a real mesh parameter, no attribution is honest, and the report says exactly
that.

**Instrument artifacts — the things you must never re-attribute to a solver.**
The bridged-gap port oracle carries a ~1.4% residue that is its own linear-in-δ
extrapolation tail (halving the gap ladder halves it, for *both* port
implementations); the fill's 8×10⁻¹² reciprocity floor amplifies to ~10⁻⁸
visibility in port networks; and the two experiment-design traps above
(drifting feeds, phantom "converged" references) will cheerfully manufacture
solver defects out of nothing if ignored.

**Inherited defects** — one, pinned: a wire *ending in* a finite (Fresnel or
Sommerfeld) ground is unsound on both sinusoidal solvers, because the
ground-connected basis completes the end current with an exact mirror image
that only a PEC plane actually provides. Recorded with a test, not patched.

## The port detour: refuted, then lifted

The plan's junction-port milestone produced the best kind of failure first.
The obvious construction — a port basis column carrying unit current *into* a
node — was built, measured, and **refused**: a current that terminates at a
point deposits a charge there, and this family's field kernel prices that
charge's self-energy at the wire radius, `Z_pp ≈ 1/(jω·4πε·a)` — matched to
7% with no fitted constant, immovable under mesh refinement. The mechanism
was then dissolved by a measurement on the B-spline side: a Lagrange-style
port's current leaves through an ideal *unmodelled lead*, so the lumped node
charge doesn't belong in the port physics at all. Holding it outside the
reaction integral — plus a second, charge-free through-port formulation for
two-terminal cases — lifted the refusal. The decisive check isn't the oracle:
it's **entrywise agreement between two port implementations sharing no basis,
no testing, and no port algebra** — 3×10⁻⁵, self-terms included. The payoff
downstream: `PortAtEnd` designs in
[antennaknobs](https://antennaknobs.dev/) — the Sterba curtain fed through a
modelled transmission line, chapter for chapter the reason ports had to exist —
now solve on a second basis, agreeing with the reference to 0.28% in impedance
and 0.001 dB in gain.

## What changed, practically

- `SinusoidalGalerkinSolver` sits in the tree beside chapter 4's solver —
  same constructor, all three grounds of [Act III](/act-3/mirror-worlds/) by
  evaluator reuse, junction ports in free space — and is wired into the
  [simulator](https://antennaknobs.dev/) as the `sinusoidal-galerkin` backend.
- The point-matched solver is deliberately untouched: its 0.1–0.3 Ω tracking
  of NEC is collocation heritage — shared basis, shared point matching — and
  it keeps the NEC-parity-probe role. The new cell will *never* track NEC
  that tightly, by design.
- Every claim above is pinned by a test or a checked-in reproduction script;
  the follow-ups the work generated are
  [momwire#191](https://github.com/stevenmburns/momwire/issues/191) (ports
  over a PEC ground) and
  [momwire#192](https://github.com/stevenmburns/momwire/issues/192) (the
  point-gap feed model as an option — it re-baselines pinned numbers, so it
  needs its own gate).

The moral is the same one [chapter 7](/act-2/validation/) ended on, one level
up: a disagreement between two solvers is not information until you can vary
one ingredient at a time. Building the fourth cell cost a solver; it bought a
taxonomy.
