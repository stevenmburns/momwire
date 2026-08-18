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

## Wire loading (momwire#427)

A loaded wire's surface condition is `E_tan = Z_s(l)·I(l)` rather than
`E_tan = 0`, and razor tests the condition on a path, so the loading term is
the testing-path integral of `Z_s` against each tent:

```
L[m, n] = ∫_{P_m} Z_s(l) Λ_n(l) dl,        Z = Z_free + L
```

In the wing idiom that integral is two constants on a shared segment —
`3h/8` when the row's path half and the column's tent ramp rise at the same
end of it, `h/8` when they rise at opposite ends — times the `σ_row·σ_col`
that dots the path's traversal direction with the current's. The resulting
`L` is symmetric even though razor's field matrix is not (a surface
impedance is a local reciprocal object); it is *not* the Galerkin Gram
(`h/3`, `h/6`), which is what `wire_loss_power` uses instead, because
dissipated power is a physical integral and does not care which rule tested
the equation.

Two spellings, one equation:

- **Distributed** — `wire_conductivity`, `insulation_radius`,
  `insulation_eps_r`, the siblings' API verbatim over the same
  `_wire_loading` physics (exact solid-cylinder I₀/I₁ internal impedance,
  King's insulated-antenna jacket inductance, per-wire with `NaN` switching
  a wire off). `wire_loss_power(coeffs)` reads back the dissipated watts.
- **Lumped** — `lumped_loads=[(wire_index, arclength, impedance), …]`, which
  is razor's own kwarg. The other solvers serve a lumped load as deck-level
  port algebra over a `node_gaps` port, and this formulation refuses node
  gaps; but a delta in `Z_s` at a knot collapses the integral above to a
  single diagonal entry, so `Z_driven = Z_unloaded + Z_L` at the fed knot is
  exact here rather than arranged. A load resolves to a knot through the
  same snapping the feeds use, two loads at one knot are in series, and a
  load at a K ≥ 3 junction is refused for the reason a source there is.

Junction tents need no special case. The **grounded-end tent** needs none
either: its side-A wing is its own image and carries `σ = 0`, which drops
both the image half of its testing path and the image half of its column —
so loading it means loading the real base segment, and a lumped load at the
contact knot is the base GAP's load at full value. Unfolded, a loaded
monopole over PEC is exactly half its loaded mirror dipole (with a base load
`Z_L` answering to `2·Z_L` in the dipole's centre gap).

The stencil is pure geometry and rides the k-independent prepare half;
`Z_s(ω)` is not (skin effect goes as `√ω`, insulation reactance as `ω`) and
is rebuilt per solved wavenumber. The term is applied outside the ground
fold — `Z = (Z_free − Z_image) + L` — because a surface impedance takes no
image and no Fresnel weight, which is why one line serves free space and all
three grounds. Gates: `tests/test_razor_loading.py`, including the NEC-5
twin lane on `LD` cards (`tests/golden_razor_loading_nec5.py`, captured by
`scripts/capture_razor_loading_nec5_lane.py`).

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
`extended_kernel`, `node_gaps` and either finite ground on a deck that
touches the plane, at construction with a message explaining why, rather
than silently mismodelling — a wrong answer here is worse than no answer.

## Per-wire radius (momwire#147)

`wire_radius` takes a scalar or one radius per wire, the same spelling
`BSplineSolver` and `SinusoidalSolver` take. A uniform model — however it
was spelled — keeps the scalar code path and is bit-identical to it.

**The convention is the SOURCE segment's radius**: the reduced kernel's a²
is added to the perpendicular distance from the source segment's axis, and
the source column already carries one radius per segment. A junction whose
arms have different radii therefore needs no special case at all — a tent
is two wings on two segments, and each wing's moments were built against
its own source column, so each wing's wired radius is its own segment's.
That is the same statement twice, which is the point.

**The convention is a choice here, and that is itself the finding.** The
two candidates — source radius, and the observer radius NEC-2's `EFLD` uses
and which momwire's sinusoidal family adopted on a PyNEC oracle that moved
11 Ω — differ only where the perpendicular distance vanishes, i.e. on a
COLLINEAR radius step. Measured there against the licensed binary: 3.0e-6 …
1.1e-5 Ω apart on a 10:1 step and 1.4e-3 … 2.1e-3 Ω on a 100:1 one, against
a 0.20 Ω twin-lane bar. The difference lives in near-diagonal matrix entries
worth ~1400 Ω, where 0.1 Ω of it is 2e-5 relative and the solve absorbs it.
The binary cannot separate them, so the source reading is taken on two
grounds that are not fits: it is the reduced kernel's own derivation (the
source current averaged over ITS surface ring onto its axis), and it is
chunk-invariant — the fill chunks the OBSERVER axis, so a per-source column
is seen whole by every chunk and no mesh refinement can move a chunk
boundary into the answer.

Gates: `tests/test_razor_mixed_radius.py`, including the NEC-5 twin lane on
two mixed-radius geometries × free space and `GN 1`
(`tests/golden_razor_mixed_radius_nec5.py`, captured by
`scripts/capture_razor_mixed_radius_nec5_lane.py`) — worst |ΔZ| 0.0586 Ω
against a 0.20 Ω bar, offsets constant to 0.0211 Ω against 0.05 Ω.

## The deck front end (momwire#432)

`momwire.deck.build_solver(model, basis="razor")` — and `"razor-nec5"` for
the identified quadrature below — construct this class from a parsed
`nec2` deck (`momwire.deck.parse`). Both names are the same `RazorSolver`,
in `momwire.deck.BASES` beside the other five families; `"razor-nec5"`
binds `nec5_quadrature=True`, the same "one class, one extra kwarg" shape
`"bspline-d1"` uses for the degree axis.

**Card translation.** `LD 0`/`LD 1`/`LD 4` (series RLC, parallel RLC, a
fixed R+jX — `momwire.deck.model.LoadSpec`'s three `kind`s) become
`lumped_loads=[(wire, arclength, Z)]`, `Z` evaluated at the build's own
frequency via `LoadSpec.impedance`. There is no separate sweep case: a
sweep already calls `build_solver` once per frequency step (the module
docstring's "translate once, fill many times" — the geometry is prepared
once and replayed, but a solver, and with it a load's `Z(ω)`, is an
OPERATING POINT and is not), so evaluating per call already is the swept
behaviour. `LD 5` and `IS` reach `wire_conductivity` /
`insulation_radius` / `insulation_eps_r` exactly as they do for every
sibling — this class takes that part of the loading API verbatim
(momwire#427) — so the deck front end does nothing razor-specific for
them at all.

**The knot a load lands on is the port-algebra route's own.** A load's
`(wire, arclength)` is read off the mesh `to_polylines` already built to
place every feed AND load site on an explicit knot, for every basis —
the SAME walk the siblings' port-algebra route uses to decide where a
`LD` card's zero-volt port goes. Razor's own knot-snapping
(`feed_arclength` "snaps to the NEAREST KNOT THAT CARRIES A BASIS", see
above) never has anything to snap TO here: the position handed to
`lumped_loads` already IS a knot, so the two routes are comparable by
construction rather than by a second, independent placement rule that
could disagree with the first.

**A load-only site is not a port here.** The siblings serve `LD` as a
zero-volt, zero-impedance gap a caller stamps afterward
(`PortPlan.loaded_ports()`); this formulation instead bakes the loaded
impedance directly into the fill (`Z_driven = Z_unloaded + Z_L` at the fed
knot is exact, per the section above), so a load-only site becomes ONLY a
`lumped_loads` entry — no phantom zero-volt `feeds` entry alongside it, since
a port this class does not need would be a fabricated drive point with
nothing to answer for. A site that is BOTH fed and loaded keeps its real
feed AND gets a `lumped_loads` entry at the same knot, which is the
Z_driven identity's own case.

**What still refuses, and how.** `build_solver` does not special-case any
of `node_gaps`, `extended_kernel` or ground CONTACT over a finite ground —
constructing this class with any of them raises with exactly the message
this page and `capabilities.refusals` already carry, because `build_solver`
passes the same kwargs it always does and this class's own `**unsupported`
dispatch does the refusing. The one kwarg `build_solver` DOES withhold is
`junctions`: every sibling takes it (a geometry hint the mesh already
computed), and this class takes no such argument at all — not even
`None`, since it is not a declared parameter and any spelling of it lands
in `**unsupported` — so the roster entry omits the keyword rather than
passing a value that would refuse.

**The portal (the sharing audit's #429 rank-9 item).**
`compute_port_solution()` / `compute_port_solution_swept()` close the gap
#432 left as a follow-up. Ports are the configured `feeds`, in order —
`junction_ports` and `node_gaps` are both refused at
construction, so there is nothing after them the way the B-spline /
sinusoidal-Galerkin families' feeds-then-junction-ports order has. On a
tent basis the coefficient AT a knot IS the current there, so
`port_currents` (== `y`) is read straight off the solved `coeffs` at the
feed rows — no separate per-port readout function the segment-basis
families need. Both entry points assemble the fill through the module's
own `_assemble_Z_prepare` / `_assemble_Z_from_prepared` split — the same
one `compute_impedance_swept` / `compute_y_matrix_swept` already use — so
`compute_y_matrix()` (now `compute_port_solution().y`) and the swept
generator behind `compute_port_solution_swept()` share one code path with
the single-k entry point rather than a second copy of the port algebra:
`momwire.portal`'s Y-matrix-based deck runner (`_y_and_port_coeffs`) drives
`--basis razor` and `--basis razor-nec5` exactly like every other roster
entry now, through the same one-fill-all-ports call.

**A remaining portal-side gap, not in this class.** The portal's
`_port_signs` assumes every `PortPlan` site (every `EX` AND every `LD`) has
a matching `RazorSolver.feeds` entry — true for a driven site, and true for
a site that is both fed and loaded (`_sites()` merges the two into one
`PortSite`, see "A load-only site is not a port here" above) — but a
LOAD-ONLY site on a segment no `EX` drives never reaches `feeds` at all
(it is baked straight into `lumped_loads`), so `_port_signs` indexes past
the end of the list on that one deck shape. Filed as a portal-side
follow-up rather than fixed alongside `compute_port_solution`: repairing it
means teaching `_portal.py`'s load-stamping algebra (built on `plan.n_ports`
== the Y-matrix size) that this family already baked a load into the fill,
not changing anything in this module.

Gates: `tests/test_deck_build_solver_razor.py` — a battery of eight decks
(free dipole, `LD 4` mid-element, `LD 5` copper, a `GN 1` base-fed contact
monopole, a `GN 1` elevated inverted-V, both finite grounds, a mixed-radius
two-wire deck) each checked against an independently-constructed
`RazorSolver` to LU roundoff; a convergence ladder showing the translation's
answer converges to the port-algebra route's own (`BSplineSolver`) with N;
and the `node_gaps` / contact-over-finite-ground refusals surfacing through
`build_solver` unchanged.

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
- `tests/test_razor_loading.py` — wire loading: the drive-point and
  Thevenin identities exact to LU roundoff, the loaded monopole halved onto
  its loaded mirror dipole three ways, the stencil against a direct path
  quadrature, the NEC-5 `LD` lane on the loading INCREMENT, the
  cross-formulation difference-of-differences at N = 192, and the schedule
  (swept == per-k over a skin-effect loss that moves with ω).
- `tests/test_razor_mixed_radius.py` — per-wire radii: the scalar fast
  path bit-frozen on five ground states × two lanes × matrix and swept, the
  NEC-5 twin lane on a fat-parasitic deck and a fat/thin junction deck in
  free space and over `GN 1`, the junction convention read structurally off
  the wing arrays, and the cross-formulation difference-of-columns.
- `tests/test_razor_port_solution.py` — `compute_port_solution` /
  `compute_port_solution_swept` (#429 rank-9): `compute_y_matrix()` bit for
  bit against `compute_port_solution().y` over free space and all three
  grounds in both quadrature lanes, the columns solving their own port
  against an independently reassembled operator, `coeffs @ V`
  superposition, feeds-in-order port ORDER, one fill and one factorisation
  per call, and the swept ω-boundary bit gate over a moving-ε̃ ground.
  `tests/test_portal.py`'s `--basis razor` / `--basis razor-nec5` battery
  carries the portal end-to-end gate — a live deck with a load AND a
  ground, both roster names, finite AIP data.
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
