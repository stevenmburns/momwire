# d=0 pulse basis — investigation and negative result

Date: 2026-06-17
Author: smburns47
Status: **Investigated and shelved. Not on main.** d=0 remains unsupported
(`BSplinePySim` still requires `degree >= 1`). This note records why, and what
a future implementation would need.

## What we wanted

A degree-0 (piecewise-constant, "pulse") basis as one more point on the
convergence curve (constant → linear → quadratic), to answer "how much does
basis order actually buy us?" The original plan
(`pulse_basis_d0_extension.md`) assumed this was a small extension — drop the
`degree >= 1` guard and special-case the trivial constant basis, reusing the
rest of `BSplinePySim` unchanged.

## What we found (three formulations, two dead ends)

**1. Drop `Z_Φ` entirely (the original plan, step 7) — degenerate.**
The plan asserted that at d=0 "the entire `Z_Φ` block is zero — only `Z_A`
matters." That is wrong. With the scalar potential dropped, a d=0 dipole's
impedance *collapses toward `Z → 0`* as N grows (reactance 60.9 → 27.1 → 11.9
→ 5.1 j over N = 20 → 160) while d=1/d=2 converge to the physical ~−1031 j.
The capacitive term — dominant off-resonance — was silently lost.

The error: the existing `_assemble_Z` builds `Z_Φ` from the derivative of the
**in-segment** polynomial (the `p·q` coefficient factor), which is identically
zero for a constant. But the charge of a piecewise-constant current is **not**
zero — it concentrates into delta functions at the segment boundaries (the
inter-segment current jumps). The polynomial-moment machinery is structurally
blind to those boundary deltas.

**2. Nodal point charges (2nd difference of point G) — right sign, wrong
magnitude.** Modelling the charge correctly as `Φ'_m = δ(a_m) − δ(b_m)` (point
charges at the segment's left/right nodes) gives

    Z_Φ[m,n] = [ G(a_m,a_n) − G(a_m,b_n) − G(b_m,a_n) + G(b_m,b_n) ] / (jωε),

a second finite difference of `G` over the endpoint nodes. Implemented and
tested: this **fixes the degeneracy** (reactance now capacitive, resistance
converges) but **does not converge usefully** — the reactance carries an
`O(1/N)` error with a huge coefficient (~−700000/N j on the λ=22 dipole), only
crawling toward the true ~−1024 j (N=320 still 3× too capacitive).

Root cause: the point self-term `G(a) = exp(−jka)/(4πa) ≈ 159` (a = wire
radius) is **N-independent** — it scales with `a`, not segment length `h` — so
`1/(jωε)·2·G(a)` dumps ~−4×10⁵ j on every diagonal and never vanishes as the
mesh refines. The true scalar potential of a charge over a *finite* cell is
`h`-dependent and does vanish. (This is why d=1 works: its charge term reuses
the segment-integrated `J_{0,0}` moment weighted by the ±1/h tent slopes — a
proper `h`-dependent self-term, never a point evaluation.)

**3. Staggered dual-mesh charge (the correct fix — not implemented).** The
standard Harrington pulse-basis scheme: model each node's charge as a line
charge over a finite **dual cell** (half of each adjacent segment; half-width
cells at free ends) and integrate `G` over those cells using the same
static-closed-form + GL-reg `J_{0,0}` machinery the vector potential already
uses:

    Z_Φ = Dᵀ Ψ D / (jωε),

with `D` the (n_nodes × n_basis) ±1 current→charge incidence (`+1` left node,
`−1` right node of each pulse) and `Ψ[i,j] = ∫∫_{Di}∫_{Dj} G` the dual-cell
moment (finite, `h`-dependent self-term). This would converge `O(1/N)` to the
d≥1 limit.

## Why we stopped

The staggered dual-mesh charge term is a genuinely **d=0-specific** assembly
(its own dual mesh, incidence operator, and dual-cell moments) — essentially
the charge core a standalone pulse solver would need. That is the opposite of
the original premise ("trivial extension reusing BSpline"). Weighed against
payoff:

- d=0 is the **least** useful degree — it converges `O(1/N)`, the classical
  and already-understood rate, versus `O(1/N²)`/`O(1/N³)` for d=1/d=2.
- It turned out to be the **most** formulation work of the whole
  convergence-curve effort, precisely because the pulse basis crosses the
  continuity boundary (discontinuous current, nodal point charges) that the
  C⁰ B-spline family (d≥1) never does.
- The useful finding for the convergence-study writeup is already in hand:
  *the pulse basis is the odd one out — it needs a staggered charge treatment
  and only reaches `O(1/N)`, so it is not worth separate machinery.*

Decision: keep **d=1 and d=2** (and, if wanted later, d=3) as the
convergence-curve points; leave d=0 unimplemented. If d=0 is ever revisited,
implement the staggered dual-mesh term above — and at that point reconsider
splitting `_BSplineMoMBase` + `BSplinePySim`(d≥1) / `PulsePySim`(d=0), since
d=0 shares infrastructure (geometry, feed, solve, `Z_A`, swept-k) but **not**
the charge assembly.

## Reference: the parts that did work

For a future implementer, these pieces were verified correct and are cheap to
re-derive (they are not on main — reverted with the rest):

- Basis layout: N pulses per wire, one per segment, single-segment support,
  `polys = [[1]]`; the segment touching a junction is its own directional
  basis (B_0 / B_{N−1}), matching the d≥1 "boundary basis is directional"
  convention. None dropped at free ends (every segment's constant is a DOF).
- The feed source is already degree-agnostic: `_build_source_vector`'s
  delta-gap and smoothed-cos² paths both produce the correct d=0 result with
  no change.
- The C++ accelerators (`seg_seg_full_moments_bspline`,
  `seg_seg_static_moments_bspline_uniform`, `assemble_Z_bspline`) only have
  d∈{1,2} template instantiations; d=0 must route to the numpy fallbacks
  (gate them on `1 <= d <= MAX_D`, not `d <= MAX_D`).
- Singular enrichment is incompatible with d=0 (hard-fail), and d=0 + ground
  plane needs an image-charge term the free-space assembly doesn't build.
