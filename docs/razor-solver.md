# RazorSolver — the NEC-5 formulation twin (momwire#309)

`src/momwire/razor.py`. A tent-basis method-of-moments solver whose testing
rule is transcribed from the NEC-5 Users Manual rather than chosen for
numerical convenience, so that its convergence behaviour is NEC-5's own —
reproducible without the (licensed, non-redistributable) NEC-5 binary.

## Why it exists

antennaknobs#890 asked whether the O(1/N) reactance walk visible in NEC-5
printouts (antennaknobs#896) was a property of NEC-5's specific
implementation or of its formulation as described in the public manual. The
manual (§1) states that formulation directly: a linear (triangular) current
expansion tested by the mixed-potential method of Rao, Wilton and Glisson —
the E-field boundary condition enforced on *path integrals* between the
centroids of connected elements ("razor-blade" testing), not the point
matching of NEC-2/4 and not the Galerkin testing of momwire's own
`BSplineSolver(degree=1)` on the same tent basis. The manual itself predicts
slower convergence than a sinusoidal expansion — this solver is that
prediction, made checkable.

`RazorSolver` is deliberately a *twin*, not an improvement: its quadratures
are converged (`tests/test_razor.py::test_quadrature_self_consistency`), so
the O(1/N) walk it produces is the testing rule's discretization error, not
a numerical shortfall. `tests/test_razor_nec5_twin.py` pins the twin
relationship against real NEC-5 printouts (LLNL-CODE-746721) — see
"Twin-gate tests" below for what it does and does not claim.

## Formulation

Unknowns are the interior knots of each wire: one unit tent per interior
knot, rising linearly over the segment before it and falling over the
segment after, so current vanishes at every free wire end. Row `m` tests
along the path `P_m` running from the centroid of the segment before knot
`m`, through the knot, to the centroid of the segment after — two straight
half-segments, bent at the knot on a kinked wire:

```
Z[m,n] = jωμ₀ · T1[m,n] − T2[m,n] / (jωε₀)

T1[m,n] = ∫_{P_m} t̂(r) · A_n(r) dl        (A_n carries no μ₀)
A_n(r)  = ∫ Λ_n(l') g(r,r') t̂(r') dl'
T2[m,n] = ∫ Λ_n'(l') [ g(c_after, r') − g(c_before, r') ] dl'
```

with the reduced thin-wire kernel `g = exp(−jkR)/(4πR)`,
`R = sqrt(|r−r'|² + a²)`, the source running along the segment axis, and
`Λ_n'` the tent's ±1/h charge doublet. `T1` carries the tangent dot product
(both tangents turn at a bend); `T2` does not. Excitation is NEC-5's
EX-at-a-knot: the delta-gap voltage sits inside exactly one testing path, so
it lands entirely in that knot's row and the solved tent coefficient at the
feed knot *is* the drive-point current. Full derivation and the closed-form
static-moment machinery are in the `razor.py` module docstring.

## Junctions

A junction is a point where wire ENDS coincide (including a single wire's
own start and end, i.e. a closed loop), detected from geometry — there is
no junction spec to write. K coincident ends carry K−1 independent
through-currents: the first-listed end is reference side A, and every other
end B gets one junction tent spanning A's terminal segment and B's, rising
to 1 at the junction on both sides. Its coefficient is the through-current
from A into B in amperes; its testing row is the razor path
centroid(A) → junction → centroid(B). Kirchhoff's law falls out of the
basis's own ±1/h charge doublet split across the two sides rather than
being enforced by a constraint row — an interior-knot tent is exactly the
K=2 junction tent of a wire split at that knot
(`tests/test_razor_junctions.py` pins the split identity at 1e-9 relative).

A wire with a single segment cannot take part in a junction (its two
junction tents would overlap on that one segment) and is refused, as is a
delta-gap feed that snaps to a K≥3 junction — the branch pair it would drive
is ambiguous.

## The PEC ground, and ground contact (momwire#398 units 2-3)

`ground_z` puts a perfectly conducting plane under the model. The fill
becomes `Z = Z_free − Z_image`: the same rows, the same testing paths and
the same tent basis, evaluated a second time against sources reflected
through the plane, subtracted once. The mirror comes from
`_potential_ground.PotentialGround`, the shared object
`docs/design/solver-architecture.md` §6 proposes — razor writes no
reflection of its own.

A wire END may lie IN the plane. Such an end keeps a degree of freedom
instead of being zeroed the way the tent basis zeroes a free end: its basis
is the junction tent between the wire and its own image — a monopole plus
its image *is* a dipole — of which only the real wing is spelled, since the
fold already evaluates every basis against the mirrored sources. Its testing
path is the real half only, which is what makes the feed voltage the base
GAP's rather than the equivalent dipole's whole gap, so a base-fed monopole
returns exactly half its mirror model's impedance. A monopole is therefore

```python
RazorSolver(wires=[[[0, 0, 0], [0, 0, 5.35]]], nsegs=48,
            ground_z=0.0, feed_arclength=0.0)   # the base gap is the port
```

K wire ends meeting at one point in the plane get K tents, one each: the
plane is one more branch there, so no through-path is distinguished and
current may leave into the ground. What is refused: a wire dipping below
the plane, an edge lying in it, an interior anchor touching down (split the
wire there instead), and contact over a finite ground — the fold hard-codes
image coefficient 1, i.e. PEC (momwire#282).

## The finite grounds (momwire#398 units 4-5)

`ground_eps` puts a finite ground under the model, for wires standing CLEAR
of the plane, in either of momwire's two flavours:

```python
RazorSolver(..., ground_z=0.0, ground_eps=(13.0, 0.005))            # GN 0
RazorSolver(..., ground_z=0.0, ground_eps=(13.0, 0.005),
            ground_model="sommerfeld")                              # GN 2
```

**`"refl-coef"`** (the default) weights the image block per (observer,
source segment) pair by the Fresnel coefficients at that pair's specular
angle: the A term takes `w_A` where the PEC fill writes `t̂_out · M·t̂_n`,
and the charge term takes `w_Φ`, which the PEC path leaves unweighted and
which `ground_phi_mode` selects. Validity window 0.1–0.5 λ above the plane
(momwire#151/#153).

**`"sommerfeld"`** is the exact ground and has no validity window — it is
the one to use below 0.1 λ, and the difference is not subtle: on a
half-wave dipole 0.04 λ up the two grounds disagree by 22.8 Ω. It is this
solver's one COMPOSING ground:

```
Z = Z_free − (C₂·image + Q),   C₂ = (ε̃−1)/(ε̃+1)
```

The scaled exact image needs no fill code of its own — C₂ arrives as the
constant weight pair, through the same slot the Fresnel weights use — and
`Q`, the smooth remainder field, is tested by this formulation's own rule:
the testing-path integral of the field of each source segment's tent
moments. The sum is associated before the fold's single minus, which is
what "composing" means. `ground_phi_mode` is accepted and unread over it
(the Sommerfeld image coefficient is exact and has no knob), exactly as in
`BSplineSolver`, and `n_qp_sommerfeld` is the remainder's per-segment
quadrature order — converged at its default 3 to 3.6e-6 Ω.

Both weights and the remainder come from
`_potential_ground.PotentialGround`; razor computes no reflection
coefficient and evaluates no Sommerfeld integral of its own.

**NEC-5 is not the oracle for either**, and that is a statement about the
physics: NEC-5's finite ground is Michalski, which carries its own limit
offset, and mixing that in would contaminate the one thing this class
exists to isolate (the testing rule). The PEC image carries no such offset
— it is the same exact image NEC-5's own `GN 1` uses — which is why it is
the ground that could be oracle-gated. The finite ones are gated by
cross-formulation agreement instead, in the form this formulation can
honestly claim: the ground must not widen the razor-vs-Galerkin gap that
razor's own O(1/N) walk already opens in free space. Measured within
0.133 Ω of it for refl-coef and 0.047 Ω for Sommerfeld.

## Scope

Free space and all three grounds — PEC, reflection-coefficient and
Sommerfeld — reduced kernel, one polyline per wire; the remaining gaps
deliberate, not initial-version:

- **Both finite grounds are served** for wires standing CLEAR of the plane
  — see the section above for what each is and how they are gated.
- **Ground contact over a finite ground is out of scope**, over BOTH of
  them, and stays refused citing momwire#282: the fold hard-codes image
  coefficient 1, so the grounded-end tent's lower wing — which IS that
  image — would take spurious contact charge. Over the Sommerfeld ground
  the coefficient is C₂ rather than the Fresnel pair, and the argument is
  unchanged: no weighting of the image block repairs a wrong basis
  function.
- **The extended kernel is out of scope.** NEC-5's formulation tests the
  expansion on the wire axis with the reduced kernel; extending it would
  again be answering a different question than the one this solver was
  built for.
- **No C++ accelerator.** The formulation-comparison role this solver plays
  needs correctness and a checkable derivation far more than it needs
  throughput; every sibling solver's accelerator work also had to survive
  bit-exact regression tests against a pure-numpy reference, which this
  class does not yet have to pay for.

`RazorSolver` refuses `degree`, `junctions`, `junction_ports`,
`extended_kernel`, `node_gaps`, non-scalar `wire_radius` and either finite
ground on a deck that touches the plane, at construction with a message
explaining why, rather than silently mismodelling — a wrong answer
here is worse than no answer.

## Twin-gate tests

- `tests/test_razor.py` — instrument parity against a standalone
  razor-blade solver (`scripts/bench_tri_razor.py` in the antennaknobs
  repo) at 5e-3 Ω, plus the closed-form static moments, bends, multi-wire
  coupling, and the O(1/N) signature itself.
- `tests/test_razor_junctions.py` — the split-wire identity, KCL falling
  out of the wing bookkeeping, and the Y matrix's non-reciprocity (a
  measured property of razor-blade testing at a finite mesh, not a bug).
- `tests/test_razor_currents.py` — field readout (`currents_at_knots`,
  `element_currents`) and the swept-solve work-sharing structure.
- `tests/test_razor_pec_ground.py` — the PEC image: exactness against the
  explicit mirrored twin, and four clearance ladders against NEC-5's
  printed `GN 1` impedances at the sharp lane's bar.
- `tests/test_razor_sommerfeld_ground.py` — the composing ground: both
  limits (ε̃ → 1 free space bit-for-bit, ε̃ → ∞ onto the PEC image at
  C₂'s own O(ε̃^{−1/2}) rate), the 22.8 Ω refl-vs-Sommerfeld split on a
  0.04 λ deck that says the remainder term is alive, cross-formulation
  ladders on three decks, and the schedule — `_sommerfeld.get_grid`
  observed firing once per solved wavenumber, and swept == per-k.
- `tests/test_razor_refl_coef_ground.py` — the finite ground: the ε̃ → ∞
  collapse onto the PEC image at the coefficients' own O(ε̃^{−1/2}) rate,
  the cross-formulation ladders (two decks × two lanes × `BSplineSolver` /
  `SinusoidalSolver` / `SinusoidalGalerkinSolver`, to N = 192), the
  ω-boundary (swept == per-k over a ground whose ε̃ moves with ω), and the
  structural row showing the fill follows the `PotentialGround` object in
  BOTH directions.
- `tests/test_razor_ground_contact.py` — the grounded-end tent: four
  geometry classes proved exactly equal to half their free-space mirror
  models, plus two contact ladders against NEC-5 (and the finding that
  NEC-5's own contact deck is not NEC-5's own mirror deck halved).
- `tests/test_razor_nec5_twin.py` — the only file comparing against real
  NEC-5 printouts (LLNL-CODE-746721) rather than an in-Python instrument:
  pointwise tracking from N=24 up with per-N tolerances that shrink with N,
  the X-pair-diff walk signature (both lanes O(1/N), same ratio band), a
  shared Richardson limit, and `BSplineSolver(degree=1)` as the negative
  control showing the walk comes from razor-blade testing and not from the
  tent basis the two solvers share.

## The identified quadrature (`nec5_quadrature`, momwire#316)

The default solver evaluates the testing-path integral ∫A·dl with a
converged Gauss-Legendre rule, so its O(1/N) walk is the SCHEME's
discretization error and nothing else. NEC-5 itself does less: the #316
residue study identified its rule as the two-point trapezoid at the
path-end centroids — every potential evaluated at element centroids, the
literal reading of the manual's "path integrals between centroids of
connected elements". `RazorSolver(nec5_quadrature=True)` adopts that
rule, and the free-space ByDipole1 ladder then matches NEC-5's printouts
at EVERY rung up to a constant ≈ −0.004−0.037j Ω (an N-independent kernel
nuance; the pair-walk signature agrees to the third decimal). Twin test 5
pins both the size and the CONSTANCY of that residual. The mode exists so
the census pair-recipe rationale is demonstrable rung-for-rung; it is not
a step toward an NEC-5 substitute — the licensed binary remains the only
oracle that can testify.
