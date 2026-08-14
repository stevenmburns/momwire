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

## Scope

Free space, reduced kernel, one polyline per wire — deliberately, not as an
initial-version gap:

- **Grounds are out of scope.** NEC-5's ground is Michalski, which carries
  its own limit offset; mixing that into this comparison would contaminate
  the one thing this class exists to isolate (the testing rule).
- **The extended kernel is out of scope.** NEC-5's formulation tests the
  expansion on the wire axis with the reduced kernel; extending it would
  again be answering a different question than the one this solver was
  built for.
- **No C++ accelerator.** The formulation-comparison role this solver plays
  needs correctness and a checkable derivation far more than it needs
  throughput; every sibling solver's accelerator work also had to survive
  bit-exact regression tests against a pure-numpy reference, which this
  class does not yet have to pay for.

`RazorSolver` refuses `ground_z`/`ground_eps`/`ground_model`/
`ground_phi_mode`, `degree`, `junctions`, `junction_ports`,
`extended_kernel`, and non-scalar `wire_radius` at construction with a
message explaining why, rather than silently mismodelling — a wrong answer
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
