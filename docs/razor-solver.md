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

That identity is also the answer for a wire with a SINGLE segment, which
this solver refused outright until momwire#608 on the grounds that "its two
junction tents would overlap on that one segment". They do share it, and
that is not a degeneracy: two tents on one segment are that segment's two
Lagrange bases, which is what every interior segment of every wire already
carries. Split a wire at its first knot and the piece that falls out is a
one-segment wire junctioned at one end; split it at two adjacent knots and
the middle piece is one junctioned at both. The split identity applies
unchanged to each, so both reproduce the unsplit wire to solver precision
(`tests/test_razor_one_segment_wire.py`, measured ~1e-14). What is still
refused is a one-segment wire whose ends meet nothing at all — no wire, no
ground plane. That one really does carry no basis, so it holds no current
and a solve including it is bit-identical to one that omits it; the licensed
engine drops it the same way, silently, and this solver says so instead.

A delta-gap feed that snaps to a K≥3 junction is refused too — the branch
pair it would drive is ambiguous, and `node_gaps` is the spelling that names
it (momwire#603 U4).

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
RazorSolver(
    wires=[[[0, 0, 0], [0, 0, 5.35]]], nsegs=48, ground_z=0.0, feed_arclength=0.0
)  # the base gap is the port
```

K wire ends meeting at one point in the plane get K tents, one each: the
plane is one more branch there, so no through-path is distinguished and
current may leave into the ground. What is refused: a wire dipping below
the plane, an edge lying in it, an interior anchor touching down (split the
wire there instead), and contact under `refl-coef` — which is the whole
tree's row, momwire#282 stage 1 having withdrawn it from every solver.
Contact over the SOMMERFELD ground is served (momwire#624); the fold's
hard-coded image coefficient 1 was the argument for refusing both finite
grounds, and #624 measured it — see "Scope" below.

## The finite grounds (momwire#398 units 4-5)

`ground_eps` puts a finite ground under the model, for wires standing CLEAR
of the plane, in either of momwire's two flavours:

```python
RazorSolver(..., ground_z=0.0, ground_eps=(13.0, 0.005))  # GN 0
RazorSolver(
    ..., ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
)  # GN 2
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
  port algebra over a zero-volt `feeds` gap a consumer stamps afterwards,
  which this formulation does not take (it is keyed by class in
  `_NATIVE_LOADING`, not by any refusal — razor has served `node_gaps` since
  momwire#603); but a delta in `Z_s` at a knot collapses the integral above
  to a single diagonal entry, so `Z_driven = Z_unloaded + Z_L` at the fed
  knot is exact here rather than arranged. A load resolves to a knot through
  the same snapping the feeds use, two loads at one knot are in series, and a
  load at a K ≥ 3 junction is refused for the reason a source there is.
  `lumped_load_power(coeffs)` reads back each load's OWN watts,
  `½·Re(Z_L)·|I(knot)|²`, in the same `(total, per_load)` shape — a separate
  readout because `wire_loss_power` counts metal and a load is a component,
  so a power budget wanting both has to add both. Both take ONE solve's
  coefficient vector; neither answers a `compute_port_solution()` block.

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
Sommerfeld — either kernel, one polyline per wire; the remaining gaps
deliberate, not initial-version:

- **Both finite grounds are served** for wires standing CLEAR of the plane
  — see the section above for what each is and how they are gated.
- **Ground contact over the SOMMERFELD ground is served** (momwire#624).
  It was out of scope over both finite grounds, on a refusal that named a
  real asymmetry — over a finite ground the T2 drop discards
  `(1 - w_Phi)*M0(plane)` rather than zero — and offered restoring that
  term as the fix, explicitly as a hypothesis rather than a diagnosis. #624
  ran the experiment and the hypothesis did not survive it: the term makes
  the binary comparison worse at full strength, and on the stubbed ladder
  coefficient 0 is flattest everywhere, so the refusal was not protecting a
  defect the term would repair. What is still refused is contact under
  **`refl-coef`**, and that is the whole tree's row rather than this
  family's: momwire#282 stage 1's D3 withdrew it from every solver because
  the MODEL is wrong at zero clearance. `razor.py` reaches it through the
  shared `_ground_spec.contact_ends` scan and the shared sentence, not a
  copy of its own.
- **The extended kernel is served** (momwire#398 D1) — `extended_kernel=True`,
  the house kwarg. It used to be refused, on the claim that "NEC-5's
  formulation is the comparison target, and its expansion is tested on the
  wire axis"; the 2026-08-18 kernel identification showed that claim was
  half right and the half it got wrong was the kernel. See "The extended
  kernel" below.
- **The fill is accelerated in four flagged pieces.** `_accel_razor.cpp`
  fuses `_seg_moments_prepare` and `_seg_moments_from_prepared` into one
  tiled OpenMP kernel (momwire#742); every branch of the fill funnels
  through that pair, so covering it covers free space, both finite grounds,
  ground contact, loading and both quadrature lanes without any of them
  gaining a branch. Three more kernels landed on top, each behind its own
  capability symbol so one never vouches for another: the complex-k twin of
  the moment fill (momwire#796, which is what lets the below-plane family
  run in a lossy medium), the fused T1 assembly (momwire#780) and its
  WEIGHTED twin (momwire#744), which momwire#806 then extended from the
  refl-coef ground to the composing one by handing the kernel the A-term
  window as a RULE the ground answers with rather than attributes the fill
  reads. The pure-NumPy path stays the reference and stays tested — it is
  what a build without the extension runs, and
  `MOMWIRE_RAZOR_FORCE_NUMPY=1` (or `razor._FORCE_NUMPY`) selects it in a
  build that has one. Agreement is TIGHT rather than bitwise: the kernel's
  per-pair reduction is not `np.einsum`'s, so the two paths differ by about
  two ulps of the matrix scale (measured 4.8e-16 in max|ΔZ|/max|Z|,
  6.3e-14 in the solved impedance), and the repo's standing rule against
  pinning cross-build bit equality applies. The weight windows and the T1
  contraction were co-dominant with the kernel on a finite-ground deck when
  that was measured; they are what momwire#744 and momwire#806 have since
  moved into the fused weighted assembler.

`RazorSolver` refuses `degree`, `junctions`, `junction_ports` and
`refl-coef` on a deck that touches the plane, at construction with a message
explaining why, rather than silently mismodelling — a wrong answer here is
worse than no answer. `node_gaps` is NOT in that list any more (momwire#603
gave this basis the node port off the through-current unknown its junction
tents already carried) and neither is the Sommerfeld half of ground contact
(momwire#624).

## The extended kernel (momwire#398 D1)

`extended_kernel=True` swaps NEC's EXTENDED (tubular) kernel in for the
reduced one on the eligible pairs. The default is `False`, and an EK-off
solve is bit-for-bit what this class computed before the kernel existed, on
every lane it serves.

### Why the refusal fell

The refusal this replaces read: *"RazorSolver is reduced-kernel only: NEC-5's
formulation is the comparison target, and its expansion is tested on the wire
axis."* The 2026-08-18 taper study measured that premise instead of assuming
it, and found it **half right**. NEC-5 has no `EK` card, which is consistent
with two opposite worlds — reduced-only, or extended everywhere. Driving
Δ/a from 10 down through 0.5 along two independent paths (refine N at fixed
radius; fatten the radius at fixed N) and isolating the kernel with
`BSplineSolver(degree=1)` run reduced against EK on an otherwise identical
setup, the binary sits on the **EK side of the kernel gap at every rung of
both ladders**: within 4–9 % of the EK row across a 43–113 Ω gap, and it
prints no warning or clamp anywhere, which a reduced-kernel code returning
`113.653 + 3.752j` for a fat dipole would have every reason to do.

The control that removes quadrature as the explanation is *this class*:
`nec5_quadrature` is the reduced kernel running NEC-5's own identified
quadrature idiom, and at Δ/a = 0.5 it reads 32.3 Ω from the binary where the
EK row reads 4.3 Ω. Basis, testing and quadrature held fixed, the rows still
part company; only the kernel is left. So the expansion *is* tested on the
axis, exactly as the refusal said — but the source it is tested against is a
tube, and the refusal's own reasoning therefore argued for EK rather than
against it.

### The honest new statement

**Which kernel makes razor the twin depends on the wire, and the two claims
partition the domain rather than competing:**

| domain | the twin lane | why |
|---|---|---|
| fat / tapered sections (a/λ ≳ 5e-4) | `extended_kernel=True` + `nec5_quadrature=True` | the reference's kernel, on the reference's basis, testing and quadrature |
| thin wire (a/λ ≲ 5e-4) | either — the reduced one is the cheaper spelling | the two kernels agree there to ~1e-4 Ω; EK buys nothing and costs nothing |

Measured on the study's `fat` control (a uniform 25 mm dipole at 14.2 MHz —
the fattest section of Ward Harriman AE6TY's 20:1 taper — fed at NEC-5's own
knot, ladder N = 20…200 so the fat end stays above the Δ/a ≈ 2 floor
momwire#248 established):

| row | offset from NEC-5, coarsest → finest | dR spread | dX spread | limit gap |
|---|---|---|---|---|
| reduced, `nec5_quadrature` | +0.02+0.06j → +0.58+0.54j Ω | 0.555 | 0.475 | 1.400 Ω |
| **EK, `nec5_quadrature`** | +0.005+0.022j → +0.005+0.011j Ω | **0.012** | **0.021** | **0.047 Ω** |

The limit gap is Richardson on the two finest gated rungs, against NEC-5's
own limit from the same pair. Extrapolated from the study's N = 280/400
instead — outside the Δ/a ≈ 2 domain, which moves the reference's own limit
by 0.57 Ω — the two read 4.863 and 0.666 Ω. Either pair says the same thing
about the ratio.

against the `nec5_quadrature` offset-constancy bar of **0.05 Ω**. The
reduced row misses that bar on this deck by 11× on these same gated rungs
and by 43× on the study's full ladder to N = 400 — the study's headline
finding about razor. The EK row holds it with 4× margin, and its continuum
limit lands 0.047 Ω from the binary's where the reduced row's lands 1.40 Ω
away. That is the twin claim, restored on the reference's home turf.

On Ward's actual 10-step taper the same lane holds the bar to Δ/a ≳ 3
(dR 0.012, dX 0.040 over N = 20…140) and runs 1.6× over it in dX at
Δ/a = 2.1 (dR 0.020, dX 0.078). The residual drift there is not a defect but
the eligibility rule's own documented conservatism: momwire extends only
COAXIAL EQUAL-RADIUS pairs, so it declines to extend ACROSS each of the nine
radius steps, where NEC still extends some cross-arm pairs (`IND = 2`,
#249 §4.3 — O(h) in the refinement limit). The uniform `fat` control, which
has no step, is the clean measurement of the kernel and it is the sharp one.

### How it is spelled

Eligibility is the **shared** rule — `_bspline_kernels._ek_axis_groups`,
already used by `BSplineSolver` and `SinusoidalGalerkinSolver` — and is not
re-derived here: two segments share a label iff they are COAXIAL and of EQUAL
RADIUS on NEC's own thresholds, and a pair is extended iff its labels match.
That is the B-spline trunk's PAIR rule rather than `SinusoidalSolver`'s
per-END `IND1`/`IND2` gating, because this formulation is mixed-potential:
its rows are path integrals over arbitrary (observer point, source segment)
pairs, not per-end brackets. An observer's label is the label of the segment
it lies ON — a testing path's two halves run along the two wing segments, so
each half's quadrature points inherit that wing's label.

The kernel enters the moments in the two halves the prepare/replay split
already has, and the split stays honest because the EK statics are as
k-independent as the reduced ones:

* **static half** (`_kernel_moments._static_axis_moments_ek`, closed form).
  The extended kernel's k → 0 limit is `1/R − a²/(2R³) + 3a⁴/(4R⁵)`, whose
  segment moments `∫dτ` and `∫τ dτ` are elementary in the same axis frame
  the reduced ones use. The two 1/R terms are collected as `ρ² − a²` — the
  observer's squared perpendicular offset — rather than left separate,
  because on an eligible pair the observer is on the source's own axis, so
  `ρ = a` and those two terms cancel *exactly*; written apart the
  cancellation would be catastrophic, written together it is 0.0 in IEEE.
* **replay half** (Gauss-Legendre, `n_qp_source`). The eligible pairs'
  smooth remainder becomes `[(e^{−jkR} − 1)·fac + extra]/(4πR)` with `fac`
  and `extra = fac − fac_static` the shared `_ek_factor` / `_ek_reg_extra` —
  spelled term for term as `_bspline_kernels` spells them, because the two
  formulations must share ONE kernel for a cross-formulation comparison to
  mean anything. Eligible entries are gathered by the chunk's mask rather
  than computed everywhere and selected, so the transient scales with the
  eligible pairs.

### What it composes with

| axis | how |
|---|---|
| the two quadrature lanes | **orthogonal, and both serve it.** `nec5_quadrature` picks where the testing path is sampled; `extended_kernel` picks which kernel is sampled there. Neither reads the other; all four combinations are live |
| PEC / refl-coef / Sommerfeld grounds | the ground supplies mirrored GEOMETRY, never the kernel's opinion. Eligibility over a ground is ONE scan of the shared rule over the real segments stacked on the mirrored ones, so a vertical wire (image coaxial, equal radius) extends — NEC's `IND = 0` perpendicular-ground branch — and a horizontal one (image merely parallel) does not. Two separate scans would call every real/image pair coaxial, the trap `BSplineSolver._ek_axis_labels` records |
| ground CONTACT | no code at all. The grounded tent's lower wing IS its own image, so the mirror policy above already decides it, and for the vertical contact that motivates the basis it decides "extend" |
| per-wire radii | eligibility is equal-radius pairwise, so a taper extends within each section and not across a step. The pair's radius IS the kernel call's `a`, since eligibility requires the two to be equal |
| wire loading | orthogonal, no interaction: `L` is a surface-impedance path integral outside the fold and never sees the kernel |

No combination is refused: every capability this class serves is served with
the extended kernel on, and gated with it on
(`tests/test_razor_extended_kernel.py`).

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

## The deck front ends (momwire#432; the nec2 one retired at momwire#821)

Both dialect front ends read `momwire.deck.BASES`, where `"razor-2p"` — and
`"razor-nec5"`, its deprecated spelling — bind this class at the identified
quadrature below (`nec5_quadrature=True`, the same "one class, one extra
kwarg" shape `"bspline-d1"` uses for the degree axis). The other lane this
class can take — Gauss-Legendre nodes along the testing path,
`nec5_quadrature=False` — was the roster's plain `razor` entry until
momwire#753 retired it as not worth ordering (measured 20x the wall time for
a 0.001 Ω difference from `razor-2p`); it stays reachable by constructing
`RazorSolver(...)` directly. Exactly one of the two front ends serves the
razor names, and which one follows from where this class puts a gap.

**The NEC-5 dialect (`momwire.eznec`) serves razor.** It writes a source at
a NODE, and this class carries its sites on knots, so that dialect's grid is
this family's own: `tests/test_razor_nec5_corpus.py` measures 77 of 80
captures served (exactly the set `bspline` serves), 202 feed/load snaps and
zero ambiguous ones. The EZNEC drop-in ships `momwire-eznec-razor-2p` (and
the deprecated `-razor-nec5` spelling) as signed launchers (momwire#819);
`tests/test_eznec_client_c.py` runs the twin end to end against the
licensed engine's own printed number.

**The nec2 dialect refuses razor by name (momwire#673 → #821).**
`Nec2Structure.resolve` addresses segment CENTRES — every `EX` card names
`(element + 0.5) * length / n_seg` — and `_snap_to_knot` would move that gap
half a cell and answer for a different antenna, silently. momwire#673
declared the cell (`capabilities.centre_feeds` is False on this row, and
`tests/test_capabilities.py` measures the declaration with a symmetry probe
rather than trusting it); momwire#821 landed the gate: `build_solver` raises
whenever `model.feeds` is non-empty and the basis does not place centre
gaps, ending with the sentence `capabilities.refusal("centre_feeds")`
carries. Every parsed nec2 deck carries an `EX` (the parser refuses one
without), so the gate retires razor from that front door outright rather
than case by case: `deck.NEC2_BASES` — the `centre_feeds` column of
`BASES` — is the roster the portal and its thin client read, the
`momwire-nec2c-razor-2p` / `-razor-nec5` console scripts (shipped v0.36.1 to
v0.46.0) are gone, and a razor name arriving by `--basis`, by a stale copy's
filename or from `MOMWIRE_NEC2C_BASIS` fails the `-version` probe with that
same sentence instead of erroring on every deck. The finer gate — refuse
only a feed that does not already sit on a knot, which needs feed provenance
the deck model does not carry — was the option not taken: nobody wants razor
on a nec2 deck, and the dialect that addresses its grid serves it.

**What razor's tenure at the nec2 seam built, and why it stays.** Three
pieces of `momwire.deck._solver` were built for this family and are now the
seam's, exercised through the NEC-5 dialect (`eznec/_serve.py` reads the
same tuple):

- `_NATIVE_LOADING` (momwire#432): `LD 0`/`LD 1`/`LD 4` become
  `lumped_loads=[(wire, arclength, Z)]`, `Z` evaluated at the build's own
  frequency via `LoadSpec.impedance` (a sweep already builds once per step,
  so per-call evaluation is the swept behaviour); `LD 5` and `IS` reach
  `wire_conductivity` / `insulation_radius` / `insulation_eps_r` verbatim
  as for every sibling (momwire#427); the one kwarg withheld is
  `junctions`, which this class does not declare. The knot a load lands on
  is the port-algebra route's own — read off the mesh `to_polylines` built
  to place every site on an explicit knot — so razor's own snapping never
  has anything to snap TO there.
- The load-only rule: a load-only site is ONLY a `lumped_loads` entry, no
  phantom zero-volt `feeds` entry beside it; a site both fed and loaded
  keeps its feed and gets the entry at the same knot, which is the
  `Z_driven = Z_unloaded + Z_L` identity's own case.
- `in_solver_ports` (momwire#588): renumbers the plan onto the rows the
  solver actually built — `BuiltSolver.ports` in the solver's port space,
  `deck_ports` in the deck's, `site_to_solver_port` the bridge — so past
  the seam there is one kind of index in circulation. For every other
  family the two plans are the same object.

The portal's stamp and budget (momwire#433 — `_load_impedances` returning a
structurally zero vector so the port algebra does not stamp Z_L a second
time on top of the fill's copy, and `lumped_load_power` as the third
dissipation term) still read `_NATIVE_LOADING`, because the two facts must
agree wherever a native-loading family is served. The nec2-deck numbers
those PRs measured (`126.60 + 137.29j` on `dipole_load_ld0`, the 57 → 1.2 Ω
route convergence over N = 9…321) were facts about a route that no longer
exists; they live in this section's git history.

Gates: `tests/test_deck_build_solver.py` (the nec2 roster IS the
`centre_feeds` column, and razor refuses a fed deck with the declared
sentence), `tests/test_razor_roster_names.py` (both names in `BASES`, in no
nec2 table, no nec2c script), `tests/test_portal.py`'s momwire#821 block
(probe-time refusal by flag and by a stale copy's filename, a typo still a
typo), and `tests/test_refusals_are_declared.py` (the raise ends with the
row's sentence).

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
  `tests/test_portal.py` no longer drives this class — razor left the nec2
  front door at momwire#821 (the deck front ends, above); the end-to-end
  lane is the NEC-5 dialect's, `tests/test_eznec_client_c.py`.
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

**The constancy is a thin-wire statement (2026-08-17, momwire#405).** On
a fat dipole (a = 1e-2 m, a/λ ≈ 4.7e-4) the residual is NOT constant: it
walks monotonically 0.0293 → 0.0715 Ω from Ntot = 12 to 96 — a 2.4×
drift, forty times the thin-dipole spread (0.0007 Ω) — while the thin
dipole, inverted-V and square loop reproduce their constants exactly.
The radius dependence says the residual is a thin-wire-kernel nuance,
not a pure quadrature-idiom artifact. Consequences: the claim above is
scoped to the thin-radius ladders it was measured on, and any future
constancy gate on a fat-radius deck (the #398 sharp lane's 0.05 Ω bar is
calibrated on thin wires) must budget the drift rather than inherit the
bar. Mechanism analysis — which kernel term carries the a-dependence —
is optional follow-on, relevant if razor ever grows EK.

### The outer path order (`n_qp_path`, momwire#754, then momwire#800)

**The default is DERIVED from the mesh (momwire#800), not a constant.**
`n_qp_path=None` — the default since #800 — returns 32 below about 350
segments per wavelength and 16 above it, on the `k*h_max` rule
`razor.derive_n_qp_path` carries; an explicit integer is taken verbatim and
reproduces the pre-#800 answer bit for bit. Everything below is #754's
derivation of the constant, which stands as the COARSE-mesh half of that
answer and is what #800 built on — read "the rule returns 32" as "32 is
what the derivation returns there", not as "32 is what the solver always
uses".

`n_qp_path=32` entered in the original implementation (a361591) and was
never re-derived. #754 measured three straight dipoles, found Z bit-identical
from q=3-4 up to 48, and proposed 8 — while naming the gap in its own
evidence: straight dipoles may not stress the outer integral, and the honest
move is a sweep across the classes that do.

`scripts/probe_razor_path_754.py` is that sweep — twelve decks over a 90°
corner, a junction with a radius step, close-spaced elements at both catalog
scales (the W8JK's 0.125 λ and the Moxon tail gap's 0.0125 λ), ground CONTACT
over PEC and over Sommerfeld, the extended kernel, and graded/split meshes.
Relative deviation from a converged q=128 reference (itself good to 9e-11):

| deck (N=60) | q=4 | q=8 | q=16 | q=32 |
|---|---|---|---|---|
| `straight` | 4.8e-7 | 3.3e-7 | 2.4e-9 | 7.3e-9 |
| `ek` | 6.2e-9 | 1.7e-11 | 8.9e-15 | 5.8e-16 |
| `contact_pec` | 1.2e-7 | 6.9e-8 | 3.6e-10 | 8.3e-10 |
| `junction_radius_step` | 1.1e-5 | 1.1e-6 | 1.4e-8 | 3.6e-11 |
| `close_spaced` (0.0125 λ) | 3.7e-6 | 2.7e-6 | 4.3e-8 | 7.7e-8 |
| `contact_somm` | 2.0e-5 | 1.7e-6 | 6.7e-8 | 8.9e-10 |
| **`bent`** (90°) | **6.8e-4** | **1.0e-4** | **1.4e-6** | 2.2e-8 |

#754's "converges at 3-4" reproduces exactly on the straight decks. It does
not generalise: the corner at q=8 is four orders of magnitude worse than a
straight dipole at the same rung. Applying #754's own rule — 2x margin over
the largest q at which any deck still moves by more than 1e-6 relative —
the binding cell is `bent` at N=60 (q=16), so the rule returns **32**.

**The required order falls with the mesh, which is the finding worth
keeping.** The same deck, down a mesh ladder:

| `bent` | q=8 | q=16 | q=32 | rule returns |
|---|---|---|---|---|
| N=30 | 3.1e-4 | 2.1e-5 | 8.2e-7 | 32 |
| N=60 | 1.0e-4 | 1.4e-6 | 2.2e-8 | 32 |
| N=120 | 1.8e-5 | 7.2e-7 | 1.7e-9 | 32 |
| N=240 | 1.6e-6 | 3.0e-8 | 3.6e-11 | 16 |
| N=400 | 1.1e-6 | 1.3e-9 | 2.7e-11 | 16 |

A CONSTANT default is applied blindly, including on the coarse meshes where
it is least converged, so 32 was the honest constant. But every rung #754's
own timing table quotes is N ≥ 400, where q=16 clears the bar with 2x margin
— so a mesh-aware order, not a smaller constant, is the shape of a real
saving. **That is what momwire#800 then built**, and it is the default now;
the switch sits at the geometric mean of the corner's own last-coarse and
first-fine rungs (`k*h = 0.018`), so each side keeps a sqrt(2) margin.
Dropping the constant to 8 was never the answer: no hard-class deck is
converged there at any mesh measured, which is why "derive" has 16 as its
floor and 32 as its ceiling rather than a third rung.

**What the order costs** (quiet box, `OMP_NUM_THREADS=1`, median of 3, first
loadavg 0.29; the rise across the table is the benchmark's own load):

| rung | q=32 | q=16 | q=8 | two-point lane |
|---|---|---|---|---|
| free space N=400 | 3.757 s | 1.913 s (0.51x) | 0.997 s (0.27x) | 0.185 s (0.05x) |
| free space N=1600 | 59.996 s | 30.794 s (0.51x) | 16.137 s (0.27x) | 3.221 s (0.05x) |
| refl-coef ground N=800 | 35.720 s | 18.153 s (0.51x) | 9.411 s (0.26x) | 1.726 s (0.05x) |

Cost is linear in the order, as #754 said. On `straight` at N=400 the answer
is bit-identical across q=32/16/8, which is exactly why three straight
dipoles made 8 look free.

**Threading decides the #744 share, so measure it threaded (2026-09-02).**
Profiling the finite-ground fill at N=801 under the timing protocol's
`OMP_NUM_THREADS=1` puts the weight windows at 10.2% against the kernel's
81.8% — but `razor_seg_moments` is `-fopenmp` and the NumPy windows are not,
so pinning to one thread handicaps the kernel alone and understates the
windows. Threaded on 8 cores, the same fill reads:

| component | tottime | share |
|---|---|---|
| `razor_seg_moments` (fused C++ kernel) | 6.53 s | 50.6% |
| weight windows (`_ground_refl`) | 3.67 s | 28.4% |
| T1 `integrand` contraction (`_assemble_Z_source_block`) | 2.41 s | 18.7% |

#744 scopes the windows **and** the contraction together: 47.1% here against
its filed 0.19/(0.19+0.217) = 46.7%. The ratio reproduces; only the
absolutes differ (#744's are per-chunk). The profile above was taken at
`n_qp_path=32`, which was the default when #754 closed; momwire#800 has
since made the default DERIVE, and 801 segments is fine enough that any deck
under about 2.29 lambda of total wire lands on 16 there (the rule is
`k*h_max <= 0.018`). So the observation-point count these shares were
measured at is not necessarily today's on the same deck, and the table
should be read as the q=32 reading it is. What #744 scoped is unchanged,
and #744 and #806 have since moved both of those terms into the fused
weighted assembler.

## Below the plane (momwire#812, unit 1 of the razor buried arc)

A deck whose wires lie **wholly below** the interface is filled in the
lower-medium family — momwire#553's below serve read through this trunk's
own vocabulary:

    Z = Z_direct(k_m, ε_m) − [ A_m·Z_image(k_m, ε_m) + Q_below ] + L

It is the composing ground's fold with three numbers swapped and one block
substituted: the kernel at `k_m = k₂·√ε̃` (the fused moments kernel takes a
complex k since momwire#796), Φ under `ε_m = ε₀·ε̃`, the image weighted by
`A_m = image_coefficient_below(ε̃)` — the negative of C₂, measured not derived
— reaching Z **through the windows** exactly as C₂ does, and the below
remainder (`_sommerfeld_below`) in place of the above one. The whole unit is
one ground object, `_potential_ground.BelowMediumGround` (whose `remainder()`
is `RemainderBelow`), and `_assemble_Z_below_plane`, which calls the same
`_assemble_Z_source_block` twice with `eps` and the ground swapped. Because
`A_m` travels in the ground's `fused_window_rule` (momwire#806), the fused
weighted assembly serves the family with no kernel work.

Measured 2026-09-02 on the phase-0 buried dipole (`tests/test_razor_below_plane_812.py`):

| gate | result |
|---|---|
| ε̃ = 1 collapse to the free-space wire | 2.2e-20 (vertical), 1.1e-19 (horizontal) |
| razor vs bspline at soil A, N = 11 → 81 | 8.83 → 6.10 → 3.93 → 2.58 Ω, monotone (the O(1/N) razor-vs-Galerkin walk) |
| two-point vs Gauss–Legendre lane, N = 41 | 0.0105 Ω |
| fused vs numpy at complex k, both lanes | 4e-16 / 5e-15 |
| above-plane decks (#762 protocol) | bit-identical |

**It is behind the public refusal.** `_SERVE_BELOW_PLANE` is off by default
and razor's `buried` capability cell stays False until unit 3 (momwire#814)
flips it; a mixed above/below deck is refused by name as unit 2's
(momwire#813, the crossing block and the node row on razor rows, which the
test-side probe on momwire#651 designed); the extended kernel is declined
below the plane. The serve-plan refusals a buried grid can hit (past the R₁
cap, below the θ floor) are asked over segment endpoints and centroids before
any grid is filled, the same R₁ and θ `BSplineSolver._buried_serve_plan` asks
over its nodes.
