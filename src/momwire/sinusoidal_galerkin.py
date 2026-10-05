"""Richmond-style piecewise-sinusoidal **Galerkin** solver (momwire#182).

`SinusoidalGalerkinSolver` shares everything geometric and basis-related with
`SinusoidalSolver` — the same three-term sinusoidal basis, the same N⁻/N⁺
neighbour tables that bake junction continuity / KCL into the basis-function
shapes — and differs in ONE place: how the EFIE is tested.

`SinusoidalSolver` collocates: it point-samples ŝ·E at each segment centre
(one test row per segment). This solver instead forms each test row by
**integrating** the field against the corresponding basis function, by Gauss
quadrature along the test segment(s). That makes it the missing
basis × testing cell (sinusoidal basis + Galerkin testing) beside the
point-matched sinusoidal solver and the B-spline Galerkin family.

Fill scheme (issue #182, chosen empirically in M1 — "let G1 decide"): reuse
the existing closed-form Eqs 76-79 field evaluators for the SOURCE integral
(analytic over each source segment) and Gauss-quadrature only the TEST
integral. M1 established that this is a structurally correct Galerkin fill:
‖Z − Zᵀ‖/‖Z‖ is exact on the diagonal and at roundoff for well-separated
pairs.

**M2 — the near-singular test integral.** M1 used one uniform Gauss rule for
every pair, which is the wrong tool for the pairs that touch. When the test
segment shares a node with the source segment (or IS the source segment), the
integrand carries a spike of width ~`a` (the thin-wire radius) located ~`a`
inside the shared endpoint: the Eqs 78-79 endpoint terms go as
Δz·(1+jkr₀)·G₀/r₀² with r₀ = √(a² + Δz²), which peaks at |Δz| ≈ a/√2 with
height ~1/a². A uniform rule cannot see a feature of relative width a/h until
it has ~h/a nodes — which is exactly why M1 needed `n_qp_test` ≈ 128 to reach
the G1 gate, and paid that cost on every one of the N² pairs.

M2 splits the quadrature instead of refining it globally:

* **Far pairs** keep one shared uniform rule (`n_qp_test`, default 8). Their
  integrands vary on the scale of the pair separation, so 8 nodes is already
  converged to roundoff.
* **Near pairs** — self, node-sharing neighbours, and any pair whose segments
  approach within `near_factor` of their combined half-lengths — are re-integrated
  on a composite rule whose panels shrink dyadically toward BOTH endpoints of
  the test segment, down to relative width `a/h` (`_graded_endpoint_rule`).
  Geometric grading plus a fixed per-panel Gauss order is the standard
  exponentially-convergent treatment for an endpoint-localized feature, and
  the panel count is *derived* from the geometry's own a/h — not tuned.

Because the near set is O(N) pairs rather than O(N²), the correction is
evaluated through `_field_components_bcast`, which pairs (P, G) quadrature
points against (P, 1) sources directly instead of forming the N² table those
pairs live in. Net effect: the G1 symmetry gate is met at the DEFAULT
quadrature, at ~1/16 the fill cost of M1's converged setting.

**M3 — the variational payoff, measured.** At 11-21 segments this solver's
driving-point impedance is closer to the fine-mesh answer than the
point-matched solver's, by 1.01× (k3_star) to 2.97× (k2_junction) worst-case
over a family of six references. Two things worth knowing when reading those
numbers, both pinned by tests in `tests/test_sinusoidal_galerkin.py`:

* The win is a **reactance** win. In R the point-matched solver actually
  converges faster on three of the four validation geometries; the net |Z|
  verdict is a reactance gain partly given back on resistance. That is the
  useful direction — the sin↔bs2 questions this solver exists to arbitrate are
  reactance-dominated.
* There is no "converged impedance" to measure against in an absolute sense:
  the delta-gap width IS the segment length, so refining the mesh shrinks the
  source, and X drifts logarithmically without limit while R settles. Every
  error figure above is relative to a *chosen* fine-mesh reference, and the
  gate is the verdict that survives the whole spread of defensible choices.

**M4 — ground models, by reuse.** All three of the point-matched solver's
grounds are wired through the same test quadrature, with NO new field
kernels: each one is the existing per-ground source-field evaluator called
with the Gauss points along the test segment as its observers instead of the
segment centres.

Since momwire#397 unit 3 this file holds none of that ground's *physics*. One
`_field_ground.FieldGround` is built per fill (`_assemble_Z`) and every ground
decision is read off it — the mirror map, the per-pair weight
(`FieldGround.projector`), the image coefficient, and whether the block folds
or must be composed first — so `ground_z` / `ground_eps` / `ground_model` are
read nowhere on this trunk. What stays here is the SCHEDULE, which is the
solver's own and is what the three bullets below actually describe:

* **PEC image** — the mirrored-source build through the same Eqs 76-79
  evaluator, plain tangential projection (`_field_ground.plain_projection`).
* **Reflection-coefficient** (`ground_eps`) — the same mirrored build with
  NEC's Fresnel field dyad applied before the projection. The dyad's specular
  tables (cos θ, p̂) are per (observer, source) *pair* and are built once for
  the whole geometry and cached (`_image_refl_prep`, this solver's schedule
  choice), then read at whatever segment-pair pairing each block names —
  where the point-matched fill builds them per observer band instead. Both go
  through one builder and one projector (`_field_ground.specular_pair_prep`,
  `PairWeights.project`).
* **Sommerfeld** (`ground_model="sommerfeld"`) — NEC's C2·(PEC image) plus the
  smooth interpolated remainder, the latter from the ground's own
  prepare/replay pair (`FieldGround.remainder("cos-1")`, the point-matched
  evaluator underneath) with its observer set overridden to the quadrature
  points. The alignment of its streamed chunks to whole test segments is this
  solver's schedule and stays in `_tested_sommerfeld_remainder`.

A fourth ground — the radial-wire screen — is designed to land as modified
reflection coefficients one level down in `_ground_refl` with **no edit to
this file**, which is criterion 1's acceptance test
(`docs/design/solver-architecture.md` §0.2).

The image blocks get the SAME graded near-pair treatment as the free-space
block, selected against the mirrored source geometry — which matters exactly
when a wire touches or nearly touches the plane, because then a segment and
its own image share the endpoint that carries M2's width-`a` spike.

**M5 — junction ports: built, measured, and REFUSED.** #177 derived why the
point-matched solver cannot have them (every sinusoidal basis function
satisfies KCL at every junction as an algebraic identity, so a current with
nonzero net inflow at a node is outside the span at any mesh density) and
named what a real implementation would need: the port basis COLUMN is cheap,
the port ROW needs the segment-integrated field testing that point
collocation cannot provide. This solver HAS those rows, so the construction
was built here and taken all the way to the oracle:

* `_junction_port_view` appends #177's `g_p = (1/P_J)·Σ_m ext_m` to the CSR
  basis table — unit net inflow at the node, vanishing at every member's far
  end, verified to 1e-15 against the ordinary bases' identically-zero inflow;
* the port row is that column tested against the field, which costs no new
  kernel: G grows from (N, N) to (N+P, N+P) through the same quadrature and
  the same M2 near-pair correction, and stays symmetric to ~1e-11;
* drive and readout are exactly dual (`b_p = -V_p`, readout `α_p`), so the
  port block of Y is symmetric to 1.7e-14.

Structurally everything #177 asked for is there. **The physics is not**, and
the reason is one #177 did not anticipate. A current with unit net inflow at
node A terminates there, so it deposits a point charge q = I/(jω) AT the
node — and the Eqs 78-79 endpoint terms (the same width-`a` feature M2's
graded rule exists to resolve) charge that point charge its own self-energy,
regularized at the thin-wire radius rather than at the mesh scale:

    Z_pp ≈ 1/(jω · 4πε · a)   — measured to 2-8 % over a decade of `a`

That term is set by `a`, not by `h`, so it does not converge away; and it
cannot be cancelled, because every ordinary basis has zero net charge at a
junction by the very KCL identity #177 identified, so no combination of them
carries a matching charge. The whole span removes 2 % of it. The result is a
port that reads as the self-capacitance of the node instead of the antenna:
on `tests/test_junction_ports.py`'s paired-tip oracle the Zdiff misses the
bridged-gap reference by ~2400×, unchanged under mesh refinement.

`BSplineSolver` is not subject to this because its ports are a
MIXED-POTENTIAL construct: the node potential is an explicit Lagrange
multiplier on a KCL constraint row, i.e. the integration-by-parts BOUNDARY
term, held apart from the reaction integral. This solver is deliberately
field-based (Eqs 76-79 give the total E of each source shape, vector and
scalar potential already merged), so the boundary term is inside the kernel
and cannot be separated from it. That is #177's "node-voltage row" bullet,
now with a number attached.

M5 therefore refused every junction-port solve. **M5b formulation (b) lifts
that refusal** (below); `_assemble_Z` still returns the reaction-form matrix
above and the tests still measure it, because the measurement is the only
thing that stops the construction being re-proposed — but the SOLVES now go
through `_assemble_Z_ported`.

**M5b — `node_ports=`, the port M5's mechanism does NOT forbid.** M5's
obstruction is constructive: it forbids exactly one thing, a basis that
TERMINATES current at a node. Formulation (a) terminates nothing. Join the
port's two terminals into one junction, so current flows THROUGH the node
under the ordinary KCL-identical span, and drive it with a ZERO-width
delta-gap EMF sitting exactly at the node, across a declared bipartition of
the junction's members:

    b_i = -V·f_i(node),     I_port = -b·α / V

`f_i(node)` is the cut vector — basis i's current crossing the gap, summed
over the + side's members. Two facts carry the whole formulation. Summed over
ALL the members it is the identically-zero KCL residual (#177's identity,
4e-13 here), so nothing accumulates at the node and M5's
Z_pp ≈ 1/(jω·4πε·a) has nothing to attach to. Summed over half of them it is
O(1), so the source has a well-defined excitation — even though the same
source point-samples to an identically ZERO RHS in the point-matched solver,
because a node is never a segment centre. That was #177's argument for why
NEC has no node port; read forwards it is a statement about the TESTING, and
this is the one capability the fourth basis × testing cell adds that
collocation cannot have.

No basis column is added, drive and readout are the same vector (so the node
ports' block of Y is symmetric to 1.3e-13 under either `feed_readout`, with
none of the gap feed's centre-vs-dual tradeoff), and a 0 V node port leaves
the solve bit-identical. Against the `_bridged_z` oracle — the same structure
re-joined by a real bridge wire of length delta carrying a real delta-gap
feed, extrapolated to delta → 0 — the node port lands 0.13-0.56 % away over
n_half 10-40 at K=2 and K=4, against G5b's 1.5 %, decaying with mesh; the
reference family's own spread is 0.013-0.041 %, so the delta → 0 idealization
is absorbed by the extrapolation rather than assumed away.

**What `node_ports` is not.** It is a two-terminal object. A ONE-terminal
net-inflow port at a lone conductor end — antennaknobs' `PortAtEnd`, which
resolves every one of `wire.sterba_bl`'s 16 ports to a ONE-member junction
whose return path is at a different node — has nothing to bipartition and
genuinely must accept net current at the node. `node_ports` refuses a
one-member junction at construction rather than silently modelling an open.
That topology is `junction_ports`', which formulation (b) below now serves.

**M5b formulation (b) — `junction_ports=`, with the node charge held
OUTSIDE the reaction integral.** M5's mechanism is dissolved, not avoided,
by one measurement: `BSplineSolver`'s own one-terminal port impedance is
−1.87 − 35.05j where the node self-capacitance would be 9.5e4. It carries
NONE of it. That is the physical content of a Lagrange-multiplier port —
the current reaching the node leaves through an ideal UNMODELLED lead, so
nothing accumulates and there is no node charge to price. M5's port basis,
by contrast, terminates its current in vacuum, which is a different physical
model and the wrong one for a lumped port.

So `_assemble_Z_ported` redefines the port basis's CHARGE to be its line
charge only and removes the lumped node charge from the source,
symmetrically:

    G'[i,p] = G'[p,i] = G[i,p] − D[i,p]
    G'[p,q]           = G[p,q] − D[p,q] − D[q,p] + S[p,q]

`D` is every basis tested against the node charge's field under the same
graded test quadrature; `S` is the lumped-lumped term put back after the
double subtraction. Neither constant is fitted — the point-charge field IS
the Eqs 78-79 endpoint term (`pref_rho_const` verbatim), and `S`'s
regularized separation √(d² + a²) is forced by the columns' own integration
by parts. Getting that √ wrong costs an a²/d³ residue: 0.25 % at the oracle's
0.04 gap, 2.1 % at 0.02, a clean a² law across a decade of radius.

Measured: the port diagonal drops from 0.93 × 1/(jω·4πε·a) to a
radius-insensitive ~5e3 (1.63× swing over a decade of `a`, against the raw
form's 11.7×); the oracle passes at 1.429 % against 1.5 %, mesh-independent;
and — the real check — the full port Y agrees ENTRYWISE with `BSplineSolver`
to 3.4e-5 on the two-member oracle and 3.9e-6 on the one-member `PortAtEnd`
topology, self terms included. Two formulations sharing no basis, no testing
and no port algebra, landing on the same numbers.

Scoped out and refused rather than approximated: junction ports under MIXED
per-wire radii (the kernel is not reciprocal there at all, M2) and over a
FINITE ground — see #191 below for what a PEC ground costs instead.

**#191 — the same port over a PEC ground.** M5b removed the lumped node
charge from the free-space source only, so any `ground_z` refused. Under PEC
that refusal costs one repeat of the correction and no new object: the
ground block IS the free-space field of mirrored sources, subtracted once
(`_fold_ground_block`, one global minus sign), and the image of a point
charge is a point charge at the mirrored node. So the removed term's image
is a mirror of a term already removed, and

    G' = (A − D − Dᵀ + S) − (B − D_img − D_imgᵀ + S_img)

with `D_img`/`S_img` the same kernels at the mirrored separation — entering
G with the OPPOSITE sign because the image block is subtracted. The gate is
M5b's own: the full port Y still agrees ENTRYWISE with `BSplineSolver` over
the same PEC plane, 8.5e-5 on the two-member oracle and 4.7e-6 on the
one-member `PortAtEnd` topology, against 5.6e-5 / 3.9e-6 for the same
geometries in free space — the ground rides at the free-space floor, though
it moves the port Y itself by 33 %. The port block stays symmetric at
6.3e-13. Dropping the image half misses by 15 % and flipping its sign by
38 %, which is what pins the derivation rather than the code reading.

The finite grounds stay refused: a Fresnel reflection of a point charge is
angle-dependent and Sommerfeld's is not an image at all, so neither has the
closed mirror this argument turns on. #151's grounded-junction rejection is
untouched — a node IN the plane is pinned by its own image and cannot be a
port at all.

One amendment formulation (b) forces and M5 did not need: the full mixed
gap+port Y is symmetric to 3.7e-8 rather than 1e-10 under
`feed_readout="variational"`. All of it is the fill's own reciprocity
residual (‖G−Gᵀ‖/‖G‖ = 8.3e-12) amplified through the port solve —
symmetrising the matrix by hand drops it to 1.4e-16. M5 did not see it
because its 8.9e4 j port diagonal made the port solve trivially well
conditioned while being physically wrong; removing it takes cond(G) from
9.7e9 to 5.9e8 and lets the fill's honest error through. The port sub-block
itself stays symmetric to 4.2e-12.

One thing M5 did land: `compute_y_matrix` / `compute_y_matrix_swept` /
`compute_impedance_swept` are now honestly Galerkin. They were inherited
verbatim from the point-matched solver, which paired THIS solver's Galerkin
matrix with collocation's point RHS — #182 M4's finding 2 — so they did not
even agree with `compute_impedance`. They now share its drive and readout.
The delta-gap feed's readout is still not its drive's dual (`feed_readout`,
below), which is why a two-gap-feed Y is symmetric only to O(h).

**The C++ far fill (#194).** The Galerkin fill is the point-matched fill's
kernel evaluated at n_qp_test observers per row instead of one, so it is
~n_qp_test× the work by construction, and ~85 % of a solve's wall clock is
that one kernel. `sinusoidal_galerkin_far_fill` fuses the kernel with the
test reduction: per test segment it evaluates the Eqs 76-79 tables at that
segment's quadrature points and contracts each observer's row into the
`(nnz, N)` contributions as it computes it, so the numpy path's
`(rows·nq, N, n_qp_const)` scratch — the thing the blocked fill above exists
to bound — is never formed at all. It serves the PLAIN projection, i.e. the
free-space block and the PEC image block; the reflection-coefficient
projector's per-pair Fresnel tables and the Sommerfeld remainder stay on
numpy, as does the O(N) near-pair correction. The numpy path remains the
reference implementation and the oracle, and is what runs when the extension
is absent.

Since #356 the fill also takes the caller's triple as its `out=`, folding
`out += scale·value` per finished entry instead of allocating and returning
three arrays of its own. That is what lets a grounded accelerated block hold
ONE triple rather than two — the fold weight is −1, and `a + (−b)` is `a − b`
to the bit, so it costs no arithmetic. `scale` multiplies from the LEFT,
`sinusoidal.py`'s C2 convention.

**The folded source shape (#205).** The third shape the fill returns is
cos kξ − 1, not cos kξ, on both paths (`cos_shape="cos-1"`). #203 had folded
that pair everywhere it could be folded from outside the kernel — the basis
value, the coefficient product, the drive — and G1 did not move, because the
same A ≈ −C cancellation runs one level up: the const-shape and cos-shape
SOURCE fields nearly coincide, so subtracting their closed forms leaves
ε·|const-shape field| inside a quantity that is O((kΔ)²) of it, and no
arrangement of the arithmetic downstream can take that back out.
`SinusoidalSolver._folded_cos_fields` computes the folded shape's field
directly instead, with every term at its own size. ‖G−Gᵀ‖/‖G‖ on the
half-wave dipole goes 1.06e-11 → 2.15e-13 at N=41 and 1.45e-9 → 5.5e-14 at
N=2401 — i.e. it now FALLS with mesh refinement, and equals what the same
fill produces with its kernel run at 80 bits, so what is left is the fill's
structural test-vs-source asymmetry rather than its rounding. The fill costs
~1.5× what it did: the folded spelling needs half-angle phases, and the
sweep that produces them is most of the kernel.

Distributed series wire loading is served in the testing scheme's own form
(momwire#395): the Galerkin overlap Σ_w Z'_w·∫f_i f_j of the three-term
sinusoidal shapes, closed-form per segment, in `_apply_loading`. On every
buried route (wholly buried, mixed, crossing) the shapes are written at
their segment's own k and the loading is read at the solve's REAL ω — k_m·c
is not a frequency — and a jacketed buried wire adds #1154's charge-side
term as the derivative overlap zq·∫f_i′ f_j′ (momwire#1156).

## (k, η) are ARGUMENTS of the fill — momwire#995

Until #995 `k` was passed down the fill as an argument and **`η` was not**:
the closed forms read `self.eta`, and so did `_lumped_pair_block`. Each place
that took half the operating point from solver state produced a wrong number
with no failure during the buried serve (#980):

* D1: setting only the fill's `k` left the DRIVE columns in air;
* D1: `_lumped_pair_block` read `self.eta` directly, so the operating
  point had to wrap the whole solve rather than the matrix;
* D2: filling the buried block at `k_m` with AIR's `eta` is wrong by
  |η₀/η_m| = **4.27×** at soil A / 7 MHz — the below quadrant came out
  3.4× wrong until a scoped mutator set η per class, and the deck solved
  and returned a plausible number throughout.

Now every fill, image, near-correction, bracket, contact-charge and
node-charge block takes `eta` beside `k`; the public entries pass the
medium's (`_medium_eta`), a mixed deck passes each class its own, and
`_fill_eta` REFUSES a complex `k` with no `eta` instead of defaulting it to
air's. `_operating_medium` still scopes `self.k` for the solve (the drive and
the readout write the basis at it), but no longer touches `self.eta`.
"""

import collections
import concurrent.futures
import functools
import math

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.spatial.distance

from . import _below_interface, _crossing_fill, _field_ground, _ground_mirror
from . import _medium_spec, _sinusoidal_mp, _sommerfeld, _sommerfeld_below
from . import _wire_loading, _wire_spec
from ._accel import acc as _acc
from ._bspline_kernels import _ek_axis_groups
from ._bspline_kernels import _reg_cplx_pool as _piece_pool
from .bspline import SINGULAR_ENRICHMENT_NEVER
from ._port_solution import PortSolution, port_impedances, refuse_undriven
from .sinusoidal import (
    _COINCIDENT_CROSSING_MEMBERS_REFUSAL,  # noqa: F401 - re-export, hoisted by momwire#1223 U1
    SinusoidalBasisSampler,  # noqa: F401 - re-export, hoisted by momwire#1223 U1
    _DENSE_ASSEMBLY_THRESHOLD,
    _basis_value,
    _entry_k,
    _EKPairs,
    _EULER_GAMMA,
    _N_PANEL_EK_DELTA_NEAR,
    _N_QP_EK_DELTA,
    SinusoidalSolver,
    _SegmentBasis,
    _recip_sin_gap,
    _sin_minus_arg,
)

# D1 serves a FULLY-buried deck (one pair class, one medium) and D2 the
# MIXED deck (above x above at k_p, below x below at k_m, the transmitted
# pair both ways). What a buried deck still refuses is refused BY NAME
# rather than filled with a shape of its own, which is the failure mode that
# produces a plausible wrong number — momwire#1000 is the one that got
# through.

# momwire#1162. `compute_impedance` reads Z = V/I per port, which is 0/0 when
# no port is driven: the solve's RHS is zero, alpha is identically zero, and
# every port current reads exactly 0. That is what returned Z = NaN on the
# junction-port deck of #1159's probe 8 -- `junction_ports=[0]` with no feeds.
# A plain-int junction port means 0 V (the #172 convention every family
# shares), so that deck drove nothing; the junction-port serve itself was
# never involved (driven at 1 V it agrees with BSplineSolver to ~1e-5). The
# drive-independent answer, Y, is served for such a deck, so the refusal
# names it. momwire#1164 moved the sentence and its backstop to
# `_port_solution` (`refuse_undriven` / `port_impedances`), shared by every
# family.

_HAVE_GALERKIN_FAR_FILL = _acc is not None and hasattr(
    _acc, "sinusoidal_galerkin_far_fill"
)

# Reused by `_refuse_junction_port_solve` (the raises) and
# `capabilities.refusals` below — one message per combination, not a copy
# in each.
_JUNCTION_PORTS_FINITE_GROUND_REFUSAL = (
    "junction_ports over a FINITE ground are not implemented on "
    "SinusoidalGalerkinSolver: M5b's node-charge correction removes the "
    "node's own lumped charge, and #191 removes its PEC image too, but the "
    "reflection-coefficient and Sommerfeld images of a point charge are not "
    "point charges, so part of the M5 blocker "
    "(Z_pp ~ 1/(jw*4*pi*eps*a)) would survive. Use BSplineSolver, a PEC "
    "ground (ground_z alone), or free space"
)
_JUNCTION_PORTS_MIXED_RADII_REFUSAL = (
    "junction_ports with mixed per-wire radii are not implemented on "
    "SinusoidalGalerkinSolver: the kernel is not reciprocal under mixed "
    "radii at all (momwire#182 M2) and the node-charge correction's "
    "regularization radius is ambiguous at a node whose members disagree "
    "about `a`. Use one radius, or BSplineSolver"
)
# momwire#398 (taper-readiness study) D2: extended_kernel=True on a wire
# STEPPED at a junction — two members with different radii meeting at one
# node — is refused rather than fixed. Reused by __init__'s raise below and
# by `capabilities.refusals`, same one-message-per-combination idiom as the
# two above and as BSplineSolver's `_ENRICHMENT_*_REFUSAL` trio (#396).
# The kernel x near_correction coupling (momwire#246), hoisted for
# `_couplings.py` on the same grounds as its neighbour below. No behaviour
# change: the raise carries this string verbatim.
_EK_NEAR_CORRECTION_REFUSAL = (
    "extended_kernel=True requires near_correction=True on "
    "SinusoidalGalerkinSolver: the extended kernel's delta is "
    "resolved on the near-pair path, and M1 mode would leave the "
    "self and node-sharing pairs on the far tier's rule "
    "(momwire#246)"
)

_EK_STEPPED_RADIUS_JUNCTION_REFUSAL = (
    "extended_kernel=True with a radius step at a junction is not "
    "implemented on SinusoidalGalerkinSolver: measured DIVERGENT, not "
    "merely inaccurate. On momwire#435's two-wire step deck (10:1 radius "
    "step at the midspan junction, 10.19 m dipole @ 14.2 MHz) the "
    "extrapolated continuum limit is 7.110 - 483.925j against NEC-5's "
    "132.560 - 11.921j, with the residual GROWING every rung refined "
    "(23.2 Ohm -> 285.8 Ohm) and a 286 Ohm dX spread down the ladder — "
    "materially worse than the reduced `sg` row's ~20 Ohm walk-away #435 "
    "already documents (that is a formulation gap; this is a divergence). "
    "The mechanism is under the mixed end-condition constants the extended "
    "delta's end bracket takes at a stepped node — the same node kind "
    "momwire#299 gates for the UNIFORM-radius case, not yet derived here "
    "for a step. Use `extended_kernel=False` (the reduced `sg` row, "
    "correctly documented as NEC-2-identified rather than NEC-5-accurate "
    "on any radius step), or `BSplineSolver(extended_kernel=True)` / "
    "`SinusoidalSolver(extended_kernel=True)`, both of which are served on "
    "a step (taper-readiness study Sec 2-3, maintainer decision D2, "
    "stevenmburns/momwire#398)"
)
# The EXTENDED-kernel twin of that fused far fill (momwire#246 unit C). On a
# build without it — a pure-Python install, or one whose extension predates
# #246 — this is False and `_tested_contribs` routes an EK-on fill through the
# numpy block loop instead, because the reduced far fill takes no eligibility
# payload and would silently drop the delta.
_HAVE_GALERKIN_FAR_FILL_EK = _acc is not None and hasattr(
    _acc, "sinusoidal_galerkin_far_fill_ek"
)

# The complex-wavenumber twin (momwire#980 step E). Flagged separately from
# the real fill because a wheel built before step E carries the real symbol
# and not this one, and such a build must keep taking the numpy path at an
# in-medium k rather than failing to find the entry point.
_HAVE_GALERKIN_FAR_FILL_CPLX = _acc is not None and hasattr(
    _acc, "sinusoidal_galerkin_far_fill_cplx"
)

# The reduced real fill's staged body (perf item 9): bit-identical to the
# per-pair reference body, which `False` selects so the uint64 gate
# (tests/test_sg_real_far_fill_staged_1290.py) can compare whole solves. A
# build without the flag's symbol has only the reference body.
_SG_REAL_STAGED = True
_HAVE_SG_REAL_STAGED = _acc is not None and hasattr(_acc, "sg_real_far_fill_calls")

# The graded near cells' blocks on the shared pool when collected for the
# banded fill (perf item 9); `False` runs them in order on this thread.
_NEAR_THREADS = True

# The Sommerfeld remainder's per-chunk test fold, overlapped with the next
# chunk's projection on one worker thread (`_OverlappedFold`); `False` folds
# inline.
_REMAINDER_OVERLAP = True

# Pairs are corrected in blocks so the (P, G, n_qp_const) source-quadrature
# scratch inside the field kernel stays bounded regardless of model size. It
# is still literally the block for `_ek_bracket_correction_tested`, whose
# per-pair scratch is the closed-form bracket's and carries no third axis;
# for `_apply_near_correction` it is now the CEILING rather than the block,
# `_near_block` sizing that one to the byte budget below (momwire#383). Kept
# as a plain name either way because it is the seam every residency gate in
# the suite already shrinks to see past the pair scratch.
_PAIR_BLOCK = 512

# Byte budget for one pair block of the near correction (momwire#383).
#
# What the block holds, per pair, measured per statement with tracemalloc on
# the #355 bend deck (G = the endpoint-graded rule's node count, nq_c =
# `n_qp_const`, n_d = `_N_QP_EK_DELTA` x `_N_PANEL_EK_DELTA_NEAR` = 128):
#
#   * the REDUCED field kernel's own tables and their source-quadrature
#     scratch — 16·G·(12.5·nq_c + 65) bytes, i.e. ~165 (G,) arrays' worth at
#     the shipped nq_c = 8, of which the (G, nq_c) tensors under `int_G0` and
#     `_folded_cos_fields` are the nq_c half. 254 KB/pair at G = 96;
#   * with the extended kernel on, `_folded_ek_delta_fields`' quadrature in
#     the sinh-mapped variable, which carries a THIRD axis of n_d = 128
#     nodes: ~25 live (G, n_d) complex arrays at its peak — t, cosh_t, R,
#     zeta, xi, w, r2, x, x2, x3, x4, a1…a4, inv2, phase, base, g1…g4, t_c,
#     t_z, l_z, l_r, kxi, s_cos and the sin shape — 4.88 MB/pair at G = 96,
#     i.e. 20x the reduced path's whole per-pair cost and 23x the fill's own
#     per-pair share.
#
# At the shipped `_PAIR_BLOCK` of 512 that second term alone was 2.63 GB of
# fixed working set, independent of N, riding every extended-kernel Galerkin
# assembly with the near correction on — the transient momwire#355 measured
# and could not account for. 8 MB instead, and the block that fits it is one
# pair under the extended kernel and 32 under the reduced one.
#
# 8 MB and not more because the measured wall clock agrees with the budget
# rather than trading against it: the per-pair set is already 5 MB, so a
# bigger block buys no cache locality and no kernel-call amortization it had
# not bought at one pair. Measured on the bend deck at N = 300 (min of 9),
# the correction alone runs 2.24 s at the budgeted block and 6.25 s at 512
# under the extended kernel, 131 ms vs 183 ms reduced. This is the rare
# budget that costs nothing to honour.
_NEAR_WORKSPACE_BYTES = 1 << 23


# Which fill `SinusoidalGalerkinSolver(fill=None)` takes (momwire#1354):
# "direct" is the closed-form field fill, "mixed-potential" the pair-moment
# fill on the B-spline machinery (`_assemble_Z_mp`).
DEFAULT_FILL = "direct"


def _near_block(nq_graded, n_qp_const, extended_kernel):
    """Near pairs per correction block: live kernel scratch ≈ the budget.

    `nq_graded` is `_graded_endpoint_rule`'s node count (the G above), which
    the deck's thinnest Δ/a fixes; the per-pair coefficients are the measured
    ones quoted at `_NEAR_WORKSPACE_BYTES`, rounded up. Capped at
    `_PAIR_BLOCK` so a rule coarse enough to make the budget non-binding
    still cannot ask for an unbounded block, and floored at one pair so a
    rule fine enough to overrun the budget on its own still makes progress —
    the same two ends `_fill_block` has.

    The block size moves no float: each pair's contribution is computed and
    ASSIGNED into its own cells (`_apply_near_correction`), with no
    accumulator crossing pairs, so blockmates never reach each other's
    arithmetic. G-D9a pins it, unconditionally since momwire#392 — until then
    it carried a caveat, that numpy elides a dead temporary on the right of a
    complex product into an in-place multiply above 256 KB and the two loops
    round differently, so `_field_components_bcast`'s tables were a function
    of the block that asked for them. #383 could only CONTAIN that by keeping
    every budgeted block under the boundary (G-D9c, which is why the budget
    once had a correctness reading); #392 removed it at the source by naming
    those operands, and G-D9d pins the kernel's shape independence directly.
    The budget is a residency decision again, which is all it should ever
    have been.
    """
    per_pair = 16 * nq_graded * (13 * n_qp_const + 66)
    if extended_kernel:
        per_pair += 16 * nq_graded * 26 * _N_QP_EK_DELTA * _N_PANEL_EK_DELTA_NEAR
    return max(1, min(_PAIR_BLOCK, _NEAR_WORKSPACE_BYTES // per_pair))


# The far fill is likewise blocked over TEST segments so the field kernel's
# (rows·nq, N, n_qp_const) source-quadrature scratch stays bounded (#194):
# unblocked, the whole-matrix fill peaks at O(N²·n_qp_test·n_qp_const)
# complex — 18.6 GiB at N=1601, OOM at N≈2000 on a 24 GiB budget (the M6
# census ceiling). The block size adapts to N so the live fill workspace
# stays near this budget; the arithmetic per matrix entry is identical, so
# the assembled G is bit-for-bit the unblocked one — a property of the
# arithmetic only since momwire#392. Before it the field kernel's tables
# moved in the last bits across numpy's 256 KB temporary boundary, and this
# claim held only because a block small enough to matter never happens: the
# budget binds only once the tables are hundreds of MB, so blocked and
# unblocked sat on the same side of it. Governs the NUMPY fill only — the
# fused C++ far fill never forms that scratch.
_FILL_WORKSPACE_BYTES = 1 << 30


def _fill_block(n_segs, nq, n_qp_const):
    """Test segments per far-fill block: live kernel scratch ≈ the budget.

    Per test segment the kernel holds ~5 (nq, N, n_qp_const) complex
    quadrature arrays plus ~16 (nq, N) tables at once (measured against
    peak RSS in scripts/profile_sinusoidal_galerkin.py).
    """
    per_seg = nq * n_segs * 16 * (5 * n_qp_const + 16)
    return max(1, _FILL_WORKSPACE_BYTES // per_seg)


# Byte budget for one band of momwire#299's end-bracket correction
# (momwire#355). The correction walks test segments in bands and folds each
# band into its scatter, so this — not the matrix — is what it holds. 32 MB is
# a fifth of the whole (nnz, N) triple at N = 300 and a twenty-fifth of it at
# N = 2401, i.e. the streaming turns itself on exactly as the matrix outgrows
# a fixed working set. The band is capped a second way, by the scatter it
# feeds (see `_ek_bracket_correction_tested`), so on the small decks where
# this budget is not binding the buffer still cannot outweigh the answer.
_EK_BRACKET_BAND_BYTES = 1 << 25

# The fewest test segments an observer band of the fused banded fill holds
# (momwire#1224, `SinusoidalGalerkinSolver._segment_bands`), whatever the
# byte budget says. The fused far-fill kernel threads over a band's test
# segments (`schedule(static)`), so a band of a few segments leaves threads
# idle on its ragged last chunk, once per band: measured on Haswell (4
# threads, hub16 buried), x8 in 141 bands of ~10 segments filled 19 % slower
# in the kernel than in 5 bands, and x16 in 134 bands of ~21 segments 11 %
# slower end to end than base, 66 bands 4 %, 17 bands 1 %. 64 keeps the
# ragged chunk under ~5 % of a band at 4 threads; the price is the band's
# budgeted scratch, ~0.8 Z at N = 2800 and ~0.4 Z at N = 5900 (buried decks).
_BAND_MIN_SEGMENTS = 64


# One source block of momwire#299's end-bracket correction, with everything
# the band loop would otherwise recompute per (band, block) resolved once:
# see `_ek_bracket_plans`. `col_of` is filled in by the caller, which cannot
# know the retained source columns until every block has named its own.
_EKBracketPlan = collections.namedtuple(
    "_EKBracketPlan",
    (
        "projector src_c src_t scale bad_lo bad_hi group_obs group_src "
        "near_key cols col_of xg wg"
    ),
)


# The FUSED far fill's extended-kernel payload (momwire#358). `_EKPairs`, which
# every numpy-side caller of `_field_components_bcast` still takes, carries the
# pair rule already EVALUATED as a mask shaped like that call's field tables;
# this one carries the rule's group labels — one per test segment, one per
# source segment — and lets the C++ sweep evaluate `g_obs[m] == g_src[n] and
# g_obs[m] >= 0` per pair. Same predicate, same eligible set, but nothing of
# the fill's (n_obs, N) shape is built in Python to express it: at N = 1200
# that mask was 11.5 MB of resident input, and it was the whole of what the
# grounded accelerated block still held over its destination triple after
# momwire#356.
#
# `src_a` stays one radius per source segment, as the kernel indexes it, and
# `n_panels` stays the delta quadrature's density so the far tier's single
# panel remains the caller's decision rather than the kernel's.
_EKFarLabels = collections.namedtuple(
    "_EKFarLabels", "src_a group_obs group_src n_panels", defaults=(1,)
)


# The period, in source columns, over which the fused REAL far fill's
# vectorized sincos sweep can be made to put the same phases in its scalar
# tail (momwire#1224). Stage B of `galerkin_far_fill_impl` runs `omp simd`
# over the flat (source, phase) table of N·S entries onto glibc's libmvec
# (`_ZGVdN4v_cos`, 4 lanes), and the last N·S mod 4 entries fall to scalar
# libm, whose last bits differ from the vector routine's. Every entry of the
# table before that tail is one lane of some vector call, and a lane's result
# depends on its own argument alone — measured: restricting the source set
# changes a column's bits exactly when it moves the tail. 4 is also a
# multiple of every narrower width a build might vectorize at (2 for SSE2 or
# NEON), so the rule below holds for those too. The complex twin's sweep
# (`sg_cplx_phase_sweep`, momwire#1224) IS vectorized, but pads its own table
# to the vector width so that no entry takes a scalar tail: it is
# column-independent whatever the padding, and the padding below is inert
# for the complex class rather than load-bearing.
_SIMD_TAIL_PERIOD = 4


# A restricted block of the fill, as the fill sees it (momwire#1224): some of
# the deck's test entries against some of its source segments. Two callers
# build one. A mixed deck's pair class (`_class_view`) is the class's own test
# entries against its own sources, instead of the whole plane masked
# afterwards; an observer BAND of the fused banded fill (`_sub_view`) is the
# entries of a run of basis rows, against the whole deck's sources or, on a
# mixed deck, against its class's.
#
#   segs     the block's TEST segments, ascending (for a class, also its
#            source segments: a class pairs a medium with itself).
#   entries  the block's support entries in the full context, ascending, and
#            grouped by segment because the CSR is segment-major.
#   pad      leading dummy source columns (see `_class_view`); the block's
#            own columns are block[:, pad:].
#   src_idx  the fill's source list as full-deck segment indices, `pad`
#            copies of its first source then the sources themselves.
#   ctx      the sub-context the fused kernel and the remainder read: the
#            block's observers, entries and CSR starts, and `hh` taken at
#            `src_idx` (the kernel reads ctx["hh"] as the SOURCE half-lengths).
#   full_ctx the whole deck's context, which the near correction keeps using
#            so its per-pair arithmetic reads exactly the arrays it always did.
#   row_of_entry / col_of_seg
#            full-deck entry / segment -> row / column of the block, or -1
#            outside it.
#   rem_cols the block columns the Sommerfeld remainder's source axis lands
#            on, in order — None when that axis IS the block's (a whole-deck
#            remainder on a whole-deck source list).
#   rem_obs  which of `segs` the remainder replays at (a mask), or None for
#            all of them: a block whose test side spans both media keeps the
#            remainder to its own medium's observers.
_ClassView = collections.namedtuple(
    "_ClassView",
    "segs entries pad src_idx ctx full_ctx row_of_entry col_of_seg rem_cols rem_obs",
    defaults=(None,),
)


# The whole deck's graded near-pair values for ONE source block, computed once
# per fill so an observer band only has to place its own cells (momwire#1224).
# `entry` / `seg` are each cell's full-deck support entry and source segment;
# `vals` its three folded-shape values, in `_apply_near_correction`'s pair
# order. O(near cells) — a few per support entry — never (nnz, N).
_NearCells = collections.namedtuple("_NearCells", "entry seg vals")


# The fused banded fill's scatter state for one G (momwire#1224; see
# `SinusoidalGalerkinSolver._scatter_band`): G itself, the three
# source-coefficient matrices M[shape], each entry's basis, each basis's last
# entry, and `carry` — basis -> its three partial T rows, for the bases a band
# boundary cuts.
_BandRows = collections.namedtuple("_BandRows", "G Ms i_of_entry last carry")


# Source bytes `_ordered_row_scatter` gathers per fancy-indexed add: its
# scratch stays a few MB rather than a copy of the band it scatters.
_ROW_SCATTER_CHUNK_BYTES = 8 << 20


def _ordered_row_scatter(dest, idx, src):
    """`np.add.at(dest, idx, src)` for a 1-D index along dest's first axis,
    to the bit, without ufunc.at's element-at-a-time loop (momwire#1290:
    1.28 s of a 6.9 s free-space solve at N = 2816).

    `np.add.at` is unbuffered: `dest[idx[e]] += src[e]` for e ascending, so
    each destination row receives its entries' rows one addition at a time,
    in entry order — the order the fused fill's bit-identity rests on
    (`_scatter_band`). Rank each entry among the entries sharing its
    destination (a stable sort, so rank r IS the r-th addition that row
    receives) and add rank 0's rows, then rank 1's, and so on. Within a
    rank the destinations are distinct, so a buffered fancy-indexed `+=`
    loses nothing; across ranks every row meets its entries in ascending
    order. Each addition is an elementwise IEEE add of the same two
    operands, with no reduction to reassociate, so the sums cannot depend
    on the platform's SIMD width or library either. The rank count is a
    basis's entry count within the band: 3 on a plain wire's interior (2
    at a wire end) on the free-space array, more at a junction.

    The fill calls it through `_row_scatter`, which hands it to the
    accelerator when it can; this is that kernel's reference."""
    n = idx.shape[0]
    if n == 0:
        return
    order = np.argsort(idx, kind="stable")
    srt = idx[order]
    pos = np.arange(n, dtype=np.int64)
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(srt[1:], srt[:-1], out=new[1:])
    rank = np.empty(n, dtype=np.int64)
    rank[order] = pos - np.maximum.accumulate(np.where(new, pos, 0))
    by_rank = np.argsort(rank, kind="stable")
    cuts = np.searchsorted(rank[by_rank], np.arange(int(rank.max()) + 2))
    chunk = max(1, _ROW_SCATTER_CHUNK_BYTES // max(1, src.itemsize * (src.size // n)))
    for r0, r1 in zip(cuts[:-1], cuts[1:]):
        for c0 in range(r0, r1, chunk):
            sel = by_rank[c0 : min(c0 + chunk, r1)]
            dest[idx[sel]] += src[sel]


# The band scatter in C++ (`ordered_row_scatter`, momwire#1290) when the
# accelerator carries it. False routes every call through
# `_ordered_row_scatter`, the reference the kernel is gated against bit for
# bit (tests/test_sg_row_scatter_accel_1290.py).
_ROW_SCATTER_ACCEL = True
_HAVE_ROW_SCATTER_ACCEL = _acc is not None and bool(
    getattr(_acc, "row_scatter_1290", False)
)


def _row_scatter(dest, idx, src, fresh=None):
    """`np.add.at(dest, idx, src)` along dest's first axis, to the bit, with
    the rows where `fresh` is true taken as +0 whatever dest holds there
    (so the caller may pass `np.empty` rows).

    `_ordered_row_scatter` reaches `np.add.at`'s order in numpy and pays for
    it in memory traffic: every rank gathers its destination and source
    rows into temporaries and scatters the sums back, 0.65 s of a 6.2 s
    free-space solve at N = 2816 (Skylake). The accelerator's
    `ordered_row_scatter` performs the same adds in the same order — per
    destination row, its entries ascending, one IEEE add each, from +0 on a
    fresh row — reading each source row in place, and threads over rows,
    which are independent. Anything it cannot take (no accelerator, a
    non-complex or 1-D operand, strided rows) goes to the reference."""
    if (
        _ROW_SCATTER_ACCEL
        and _HAVE_ROW_SCATTER_ACCEL
        and dest.ndim == 2
        and src.ndim == 2
        and dest.dtype == np.complex128
        and src.dtype == np.complex128
        and (dest.shape[1] <= 1 or (dest.strides[1] == 16 and src.strides[1] == 16))
    ):
        _acc.ordered_row_scatter(
            dest, idx, src, np.zeros(0, dtype=bool) if fresh is None else fresh
        )
        return
    if fresh is not None:
        dest[fresh] = 0
    _ordered_row_scatter(dest, idx, src)


def _solve_in_place(G, rhs):
    """Solve G·x = rhs, factoring G IN PLACE — G holds its LU factors after.

    `scipy.linalg.solve(G, ·)` leaves its argument alone, so it hands LAPACK
    a Fortran-ordered COPY of G: one extra (n_basis, n_basis) array at the
    solve (momwire#1224). `lu_factor(overwrite_a=True)` factors G's own
    storage whenever G is F-contiguous, which the fill arranges (the sparse
    coefficient product comes out F-ordered, and `_add_crossing_blocks` /
    `_assemble_Z_ported` keep it so); a C-ordered G is still copied, as
    before, and is still solved correctly.

    The same getrf on the same matrix, then getrs: bit-identical to
    `scipy.linalg.solve`'s general solve, measured. Two things `solve` did
    are not repeated. Its reciprocal-condition estimate (gecon) and the
    ill-conditioning `LinAlgWarning` it drives are dropped, as the
    point-matched solver's in-place solve already does. An EXACTLY singular
    G still raises `LinAlgError` rather than returning inf/NaN, which
    `lu_factor` alone would (it only warns).

    Callers must not read G afterwards; neither SG solve does, and this
    family stashes no matrix or factors on the solver.
    """
    lu_piv = scipy.linalg.lu_factor(G, overwrite_a=True)
    if not np.all(np.diagonal(lu_piv[0])):
        raise np.linalg.LinAlgError("singular matrix")
    return scipy.linalg.lu_solve(lu_piv, rhs)


def _solve_constrained(G, rhs, C):
    """Solve G·x = rhs on the subspace C·x = 0 (Galerkin: the constrained
    trial functions are the test functions too), or `_solve_in_place` when C
    is None — the unconstrained path, bit for bit.

    Eliminates one unknown per row of C: pivots p from a column-pivoted QR of
    C, the rest r, and x_p = T·x_r with T = −C_p⁻¹·C_r. The reduced system
    is PᵀGP with P = [I; T] stacked over (r, p), assembled from G's four
    blocks in O(n²·m) rather than as a dense product. The transpose and not
    the adjoint: the bilinear form is the un-conjugated one this family
    assembles G with.
    """
    if C is None:
        return _solve_in_place(G, rhs)
    m, n = C.shape
    _q, _r, perm = scipy.linalg.qr(C, mode="economic", pivoting=True)
    p, r = np.sort(perm[:m]), np.setdiff1d(np.arange(n), perm[:m])
    T = -np.linalg.solve(C[:, p], C[:, r])
    Grp = G[np.ix_(r, p)]
    Gpr = G[np.ix_(p, r)]
    Gc = G[np.ix_(r, r)] + T.T @ Gpr + Grp @ T + T.T @ G[np.ix_(p, p)] @ T
    rhs_c = rhs[r] + T.T @ rhs[p]
    x_r = _solve_in_place(np.asfortranarray(Gc), rhs_c)
    x = np.empty((n,) + x_r.shape[1:], dtype=np.result_type(x_r, T))
    x[r] = x_r
    x[p] = T @ x_r
    return x


def _graded_endpoint_rule(eps, n_per_panel, leggauss):
    """Composite Gauss rule on [-1, 1] with panels graded toward BOTH ends.

    Panel widths double away from each endpoint — eps, 2·eps, 4·eps, … — so a
    feature of relative width `eps` sitting at an endpoint is resolved by a
    panel of its own size, and the smooth interior is covered by a handful of
    large panels. Returns (nodes, weights) with Σw = 2.

    `eps` is the feature width relative to the segment half-length (a/h for
    the thin-wire endpoint spike); `leggauss` is the solver's cached
    Gauss-Legendre factory.
    """
    eps = float(np.clip(eps, 1e-9, 0.5))
    # Distances from the endpoint at which panels break. Stop at the midpoint
    # so the largest panel is [-0.5, 0] and the grading ratio stays ~2.
    d = eps * 2.0 ** np.arange(64)
    d = d[d <= 0.5]
    left = np.concatenate(([-1.0], -1.0 + d, [0.0]))
    edges = np.unique(np.concatenate([left, -left[::-1]]))

    gx, gw = leggauss(n_per_panel)
    lo, hi = edges[:-1], edges[1:]
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    x = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    w = (half[:, None] * gw[None, :]).ravel()
    return x, w


# The unweighted projector, serving the FREE-SPACE block here and returned by
# `FieldGround.projector` for every ground that has no dyad. It is bound, not
# defined: `_tested_contribs` gates its fused C++ far fill on `projector is
# _plain_projection`, so the name and the object the ground hands back have to
# be the same function or the accelerator would quietly stop serving the
# grounded fills (momwire#397 unit 3).
_plain_projection = _field_ground.plain_projection


def _loading_integrals(k, h):
    """``(∫(cos kξ − 1), ∫sin² kξ, ∫(cos kξ − 1)²)`` over ξ ∈ [−h/2, h/2],
    each to full relative precision (momwire#1283).

    The literal forms — (2/k)sin(kh/2) − h, h/2 − sin(kh)/2k and
    3h/2 − (4/k)sin(kh/2) + sin(kh)/2k — subtract terms that agree to
    O((kh)², (kh)², (kh)⁴), so on a short segment they return rounding. With
    x = kh/2 and S(u) = sin u − u (`_sin_minus_arg`, exact at every u):

        ∫(cos kξ − 1)    = (2/k)·S(x)
        ∫sin² kξ         = −S(2x)/(2k)
        ∫(cos kξ − 1)²   = (1/k)·[½·S(2x) − 4·S(x)]

    The last still cancels — both S terms are O(x³) and the answer O(x⁵) —
    so below |x| = 0.25 it is its own Taylor series, Σ_{m≥2} (−1)^m
    (4^m − 4)/(2m+1)!·x^(2m+1), whose first omitted term (m = 8) is 3e-16 of
    the answer there; above it the S form gives up at most 1/(0.15·x²) =
    2 decades. Complex-safe, for the buried shapes' k_m.
    """
    x = 0.5 * np.asarray(k) * np.asarray(h)
    s1 = _sin_minus_arg(x)
    s2 = _sin_minus_arg(2.0 * x)
    i_g = (2.0 / k) * s1
    i_ss = -s2 / (2.0 * k)
    x2 = x * x
    series = 0.0
    for m in range(7, 1, -1):  # Horner over x², highest order first
        coef = (-1) ** m * (4.0**m - 4.0) / math.factorial(2 * m + 1)
        series = series * x2 + coef
    series = series * x2 * x2 * x
    i_gg = np.where(np.abs(x) < 0.25, series, 0.5 * s2 - 4.0 * s1) / k
    return i_g, i_ss, i_gg


@functools.lru_cache(maxsize=1)
def _fold_worker():
    """The one thread the remainder folds run on while the next chunk's
    projection runs (`_OverlappedFold`)."""
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


class _OverlappedFold:
    """Runs a remainder replay's `consume` calls on `_fold_worker`, one chunk
    behind the replay loop, so a chunk's numpy fold overlaps the next chunk's
    C++ projection (perf item 9).

    Exact by construction: each call runs the same expressions on the same
    arrays it would run inline, so numpy takes the same loop for every product
    (the momwire#392 elision note), and two chunks' folds touch disjoint
    destination rows (a chunk is whole test segments, and an entry belongs to
    one segment), so the order they land in is immaterial. One fold is in
    flight at a time — the next call waits for it — which bounds the extra
    residency at one chunk's block. Leaving the context waits for the last
    fold and re-raises a worker's exception; it also waits when the replay
    itself raised, so no fold is still writing when the caller sees the
    error. `_REMAINDER_OVERLAP = False` calls `consume` inline
    (tests/test_sg_remainder_overlap_1290.py)."""

    def __init__(self):
        self._pending = None

    def __enter__(self):
        return self

    def _drain(self):
        if self._pending is not None:
            pending, self._pending = self._pending, None
            pending.result()

    def wrap(self, consume):
        if not _REMAINDER_OVERLAP:
            return consume

        def overlapped(i0, i1, block):
            self._drain()
            self._pending = _fold_worker().submit(consume, i0, i1, block)

        return overlapped

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._drain()
        elif self._pending is not None:
            # The replay's own error wins; wait for the fold in flight and
            # drop what it raised.
            concurrent.futures.wait([self._pending])
            self._pending = None
        return False


class SinusoidalGalerkinSolver(SinusoidalSolver):
    """Piecewise-sinusoidal Galerkin MoM. Same constructor surface as
    `SinusoidalSolver`, plus the test-quadrature controls:

    `n_qp_test`
        Gauss nodes per test segment for the shared FAR-pair rule.
    `n_qp_near`
        Gauss nodes per panel of the graded near-pair rule (M2). The default 8
        is where the G1 residual stops being near-pair quadrature error and
        hits the source-side `n_qp_const` floor (~2e-11) on every validation
        geometry; 4 leaves the bent/junction cases short of the gate. That
        floor is the QUADRATURE one, and it is only the binding one at
        test-scale meshes: past N ≈ 400 the G1 residual is set instead by the
        const/cos cancellation inside the fill (2e-10 at N=801, 1.4e-9 at
        N=2401, i.e. ~ε‖T_const‖/‖G‖), which no test rule can reach —
        stevenmburns/momwire#203 and §15 of the instrument report.
    `near_factor`
        A pair is "near" when its segments approach within
        `near_factor · (h_m + h_n)/2`. The default 0.5 selects the self and
        node-sharing pairs (both at distance 0) while excluding the
        next-nearest collinear neighbour (at distance h), and additionally
        catches genuinely close non-touching wires.
    `near_correction`
        Set False to fall back to M1 behaviour (one uniform rule everywhere)
        — the contrast the M2 tests use to show the correction is what buys
        the symmetry gate.
    `feed_readout`
        Which functional reads the current at a DELTA-GAP feed (junction
        ports are unaffected — theirs is dual either way).

        ``"centre"`` (default) is the point-matched solver's and NEC's: the
        current AT the feed segment's centre. ``"variational"`` is the exact
        dual of the Galerkin drive: the gap-averaged current
        (1/h)∫_gap J ds, i.e. Y = −Uᵀ G⁻¹ U on the drive columns U.

        They differ because the delta-gap source this solver inherits is
        NEC's — E_app = V/Δ spread over the whole feed segment — whose dual
        readout is the gap AVERAGE, not the gap centre. The default keeps
        the point-matched sibling's readout so that the two sinusoidal cells
        differ in exactly one thing (the testing), which is what the whole
        basis × testing instrument is for; the price is that a multiport Y
        with a gap feed in it is symmetric only to the O(h) difference
        between the two functionals (6.7e-5 at N=21, 6.5e-6 at N=81 on the
        two-feed dipole — better than the point-matched solver's own 9.1e-5
        / 7.3e-6 there, and decaying). ``"variational"`` makes that Y
        symmetric to ~1e-12 instead, and costs the M3 payoff gate: on
        k3_star the worst-case errColl/errGal falls from 1.014 to 0.797, so
        it is offered, measured and documented rather than made the default.
    `feed_model`
        Which SOURCE a delta-gap feed applies (junction and node ports are
        unaffected — both are zero-width by construction).

        ``"point"`` (default since momwire#654) is `BSplineSolver`'s
        zero-width gap, E_app = V·δ(s − s0); the Galerkin test integral
        collapses on the delta and the drive column is −V·f_i(s0), i.e. the
        basis-evaluation vector σ(A+C). ``"segment"`` is NEC's and the
        point-matched sibling's: E_app = V/Δ_m spread over the whole feed
        segment, so refining the mesh shrinks the source.

        The source model is a third instrument axis, not a refinement of the
        first two (momwire#182 M5, report §6): on the canonical dipole the
        point gap sits 2.8e-7 / 1.3e-7 from `BSplineSolver` at N=161/321
        against the segment gap's 2.5e-4 / 1.5e-4, so most of what M2/M3
        filed as a sin↔bs2 BASIS gap is a feed-model gap. It is also exactly
        self-dual under the DEFAULT `feed_readout="centre"` — the drive
        column IS the centre-evaluation functional — so a multiport Y with
        gap feeds in it is symmetric to the fill's own reciprocity floor
        under either readout, with none of the M3 payoff traded.

        **Why `"point"` is the default** (momwire#654). It is not a
        refinement of `"segment"` but a better answer to the same question:
        the point gap sits 2.8e-7 / 1.3e-7 from `BSplineSolver` at N=161/321
        against the segment gap's 2.5e-4 / 1.5e-4, it is exactly self-dual
        under the default `feed_readout="centre"` so the readout knob stops
        having consequences at gap feeds, and antennaknobs measures up to
        992× tighter cross-basis agreement with it on the antennaknobs#478
        class (momwire#213).

        `"segment"` was the default until #654, for one reason: flipping it
        re-baselines pinned numbers. It stopped being a good reason once the
        cost was measured rather than estimated — 31 of 6315 tests, every one
        of them a test whose SUBJECT is this axis or the payoff comparison,
        and each fixed by NAMING the source it was silently inheriting. What
        `"segment"` is still for is the matched control: the M3/M4 payoff
        gates score this class against `SinusoidalSolver`, which can carry no
        other source (momwire#212), so those comparisons name it explicitly
        (`_MATCHED_FEED` in `tests/test_sinusoidal_galerkin.py`, gated by
        `test_the_payoff_schemes_carry_a_matched_feed_model`). That is what
        report §17 called "the substantive blocker on ever flipping
        `feed_model`'s default" — real, and answered by making the control
        explicit rather than by leaving a default to imply it.

        NEC-2 REPRODUCTION is not this class's job and never was:
        `SinusoidalSolver` is NEC's formulation — same basis, same point
        matching, same segment gap — and `tests/test_sinusoidal_bend_nec2_twin.py`
        is named for it. Galerkin-with-a-segment-gap is neither NEC nor the
        converged answer; it is a control. See §6, §12 follow-up 5, §16 and
        §17 of `docs/sinusoidal-galerkin-instrument-report.md`.

    `extended_kernel`
        NEC's EK card for this basis (momwire#246). False — the default —
        keeps the reduced ("thin-wire") kernel: the source current is a
        filament on the wire axis and the conductor's girth survives only as
        the a² regularization of ρ. True switches the fill to NEC's EXTENDED
        thin-wire kernel, the O(a²) azimuthal average of the Green's function
        over a source tube of radius a (Eqs 84-98 of the theory manual), which
        is what makes fat conductors — Δ/a below ~3 — answerable at all.

        **What is served.** Every ground model this solver has (momwire#287):
        the free-space block, the PEC image, the reflection-coefficient image
        and the Sommerfeld ground's C2-scaled exact image, plus the graded
        near-pair correction on every one of them. Mechanically the reduced
        fill is untouched: #205's folded closed forms are computed exactly as
        they always were and `SinusoidalSolver._folded_ek_delta_fields` ADDS a
        Gauss-Legendre quadrature of the smooth extended-minus-reduced delta
        on the eligible pairs. Nothing is subtracted anywhere, so the fold's
        cancellation discipline never comes up, and an ineligible pair comes
        back bit-for-bit the reduced fill's.

        **What stays REDUCED** is the Sommerfeld ground-wave remainder
        (`_tested_sommerfeld_remainder` — NEC's eqs 143-147, the smooth
        correction left after the C2-scaled exact image). That is deliberate
        and measured, not an omission, and it is `BSplineSolver`'s answer
        since #269 arrived at independently under this fill's own test
        quadrature. EK is an O((a/R)²) tube correction and the remainder's
        source is the ground reflection, so R is the IMAGE distance
        r₁ ≥ 2h for a wire at height h and the un-applied correction is
        O((a/2h)²). Measured by building the extended remainder outright —
        the same `remainder_field_proj` field azimuthally averaged over a
        ring of radius `a` about the source axis — and re-solving, |ΔZ|/|Z|
        over soil / sea: 9.5e-7 / 9.3e-8 for a monopole clear of the plane,
        4.0e-5 / 4.2e-6 horizontal, 1.0e-5 / 1.3e-6 slanted, i.e. ≤ 4e-5
        anywhere the wire is clear of the plane — three orders below the EK
        shift the image blocks DO carry on those decks (2.2e-2 to 2.4e-2)
        and two below this basis's own cross-basis gap. AT GROUND CONTACT,
        where the (a/2h)² estimate degenerates (r₁ → 0) and only the
        remainder's smoothness bounds it, the cost is 3.5e-3 / 4.5e-3: still
        an order below the cross-basis Z gap on that deck (3.6 %), but 55-66 %
        of the deck's own EK shift, so it is the one place the mixture is
        visible — and refl-coef is invalid at a contact (#153), so Sommerfeld
        is the only model there. The measurement is `test_extended_kernel_
        galerkin.py`'s G-S1, with the O(a²) ladder and the ring-count
        convergence beside it.

        **What refuses.** `near_correction=False`, because under EK the near
        path is not a refinement but where the on-segment pairs are computed:
        the delta's structure is a spike of width `a` around each
        source-segment END, and it is the near path that passes the dense
        quadrature resolving it. The junction/node-port lumped-charge blocks
        stay reduced, and that is not a gap: their source is a point charge
        at a node, which has no tube to average over.

        **What it costs in accuracy at the thin end.** Reduced-plus-delta is
        a near-cancelling decomposition — the two kernels agree away from the
        wire — so an on-segment pair carries ~(H/a)² of cancellation and
        float64 leaves ~ε·(H/a)² of the delta's peak behind. The EK shift is
        itself O((a/H)²), so the two scale against each other and the error
        that reaches Z stays ~1e-10 relative out to Δ/a ≈ 500; past Δ/a ≈ 1e4,
        where the kernel is a 1e-8 effect anyway, the decomposition is
        noise-limited. `_folded_ek_delta_fields` documents the mechanism.

        **Why a PAIR rule, not NEC's per-end gating.** A pair is eligible iff
        its two segments are coaxial and of equal radius (`_ek_pairs`, the
        rule `BSplineSolver` uses — #249 §4). NEC decides per source-segment
        END (IND1/IND2), which in a Galerkin fill would make G(i, j) extended
        while G(j, i) was not and destroy ‖G−Gᵀ‖/‖G‖, the reciprocity residual
        this solver has used as its error detector since M2. The pair rule is
        symmetric by construction, agrees with NEC on straight wires and on
        perpendicular ground contacts (via the mirrored source), and is
        strictly more conservative at bends, radius steps and K ≥ 3 junctions.

        **Where the pair rule is not enough (momwire#299).** One piece of the
        delta is not a pair object: its END BRACKET, the boundary term of the
        integration by parts in ξ, whose contribution to a matrix entry is
        O(1/a) per source end. The two brackets meeting at an interior node
        cancel — current and charge are continuous through it and the two ends
        carry opposite signs — but only if BOTH sides are extended, and the
        pair rule's eligible set stops AT a split node. One uncancelled cap
        made the fill DIVERGE as the wire thinned: δZ on an L ran
        −21.5 − 240.5j at a = 0.02 and −24.1 − 526.7j at a = 0.002 where
        `BSplineSolver` gave −0.035 − 0.657j and −0.002 − 0.041j. So the
        brackets are gated separately, per SOURCE END, by a NODE predicate
        (`_ek_reduced_ends`: extend at node P iff every segment meeting there
        shares one axis line and one radius — NEC's IND = 0 read as a property
        of the node, hence observer-independent) and taken back off by
        `_ek_bracket_correction_tested`, which symmetrizes what it removes so
        that reciprocity does not move. Straight decks, free ends and ground
        contacts are untouched to the bit; the four repaired node kinds — bend,
        shallow vee, collinear radius step, K = 3 junction — now collapse at
        the straight dipole's own rate (worst 0.45 per halving of a).

        The delta is numpy-only for now: with EK on, the fused C++ far fill is
        skipped (it takes no eligibility mask) until momwire#246 unit C lands
        its twin, so an EK-on fill costs what the pre-#194 fill did.

    All three ground models are wired (M4): `ground_z` alone gives the PEC
    image, `+ ground_eps` NEC's reflection-coefficient ground, and
    `+ ground_model="sommerfeld"` the Sommerfeld/Norton ground — each by
    reusing that ground's existing source-field evaluator under this test
    quadrature. A wire END LYING IN the plane used to be sound only under the
    PEC ground — #151's ground-connected basis completes the end current with
    an exact mirror image, which a finite ground does not provide, so the
    leftover contact charge made the answer diverge under refinement.
    momwire#282 subtracts that charge (see
    `SinusoidalSolver._contact_charge_kernel`) and the contact answer settles,
    at one recorded price: the correction is source-side only, so the fill is
    no longer self-adjoint on such a deck
    (`test_the_282_contact_correction_is_not_self_adjoint`). With the EXTENDED
    kernel on the subtraction has to cancel EKSCX's end-charge bracket rather
    than the reduced one's, which momwire#292 does — through
    `_contact_ek_masks`, whose eligibility is this solver's per-PAIR rule
    rather than NEC's per-end IND code.

    `node_ports`
        Two-terminal ports located AT a junction node (M5b formulation (a)):
        a zero-width delta-gap EMF across a declared bipartition of the
        junction's members, so current flows THROUGH the node and nothing
        terminates there. Entries are `(junction_index, side_a)` (voltage 0)
        or `(junction_index, side_a, voltage)`, where `side_a` indexes into
        `junctions[junction_index]` and must be a nonempty PROPER subset —
        both sides of the gap need a conductor. A junction of K >= 3 members
        takes several ports when each cuts ONE member's branch off the node
        (one member on one side) and no branch is cut twice — NEC-5's object
        per named wire (momwire#1300); see `_check_shared_junction`.

        Ports are ordered [gap feeds…, junction ports…, node ports…]. Drive
        and readout are the same vector, so a node port's Y block is
        machine-symmetric under either `feed_readout` and costs none of the
        M3 payoff. Which side is called "+" is a convention the impedance
        does not see (2.5e-16). Grounded junctions are rejected (#151: the
        node's current closes through its image, not on its partners), as is
        a ONE-member junction — that is a one-terminal net-inflow port, i.e.
        `junction_ports`, which this solver refuses. The point-matched
        sibling has no equivalent and takes no such keyword: a node source
        samples to an identically zero collocation RHS.

    `node_gaps`
        `BSplineSolver`'s wire-end spelling of the same series port (#305):
        `(wire_index, "start"|"end", voltage)` entries, each naming the one
        member its gap is in series with — NEC-5's tag/segment/end
        addressing. Normalized here onto the node-port list as the
        COMPLEMENT bipartition, which by the span's KCL identity is the
        single member's exact negation: that orientation flip is what makes
        both families read I_port as the current from the node into the
        named wire (the two σ conventions are mirrored). Entries order
        after any explicit `node_ports`.

    `junction_ports=` — one-terminal net-inflow ports at junction nodes, with
    `BSplineSolver`'s rules (a junction index or an (index, voltage) pair, in
    range, no repeats, no grounded junction) and its Y ordering. M5 built
    #177's port basis here and REFUSED every solve on it; M5b formulation (b)
    lifts that refusal by holding the node's lumped charge outside the
    reaction integral (`_assemble_Z_ported`), which reproduces `BSplineSolver`
    to 3.4e-5 / 3.9e-6 entrywise. `_assemble_Z` still returns M5's refuted
    reaction-form matrix, and `tests/test_junction_ports.py` still measures
    it. Over a PEC ground (`ground_z` alone) the same correction runs on the
    image block at the mirrored separation (#191), holding 8.5e-5 / 4.7e-6
    against `BSplineSolver`. A FINITE ground (`ground_eps`, or
    `ground_model="sommerfeld"`) or mixed per-wire radii still raise.

    `n_qp_node`
        Panels-per-end of the graded rule the node-charge correction is
        integrated on (default 16, converged: 4e-9 from 12 to 16).

    The C++ accelerator serves the far fill of the PLAIN-projected blocks —
    free space and the PEC image — through `_far_fill_accel`, which fuses the
    field kernel with the test reduction (#194) and, with `extended_kernel`
    on, carries #246's delta on the eligible pairs in the same pass.
    Everything else is numpy: the reflection-coefficient and Sommerfeld ground
    blocks, the near-pair correction, and the whole fill when the extension is
    not loaded. Distributed wire loading is numpy too, and sparse: the
    Galerkin overlap term of `_apply_loading`, closed-form on the shapes.
    """

    # momwire#1333: two coincident wires left in place because they are
    # joined to different wires. The Galerkin fill has no answer for them in
    # free space: on the split rise fan it read 61.89 - 0.88j ohm where
    # BSplineSolver read 32.95 - 337.2j. Over a ground the crossing serve already refuses the spelling
    # (`_COINCIDENT_CROSSING_MEMBERS_REFUSAL`).
    _KEPT_COINCIDENT_REFUSAL = (
        "this family cannot solve coincident conductors: its Galerkin fill "
        "answers them from rounding (momwire#1333). Solve the deck with "
        "BSplineSolver, or join both copies to the same wires at each end so "
        "they merge as one"
    )

    # momwire#396: differs from `SinusoidalSolver.capabilities` in exactly
    # the two axes this class's docstring describes — junction_ports and
    # node_gaps are served here (M5b / #305) — plus the three combination
    # refusals __init__'s raises still carry. Wire loading rides the base
    # class's overlap term (#395), on every buried route too since
    # momwire#1156 (the mixed deck's loading refusal was never a declared
    # cell, so no row moves). `extended_kernel+
    # stepped_radius_junction` is momwire#398 D2 (taper-readiness study):
    # unlike the two junction_ports combos above, this one is a measured
    # DIVERGENCE, not an unimplemented feature — see
    # `_EK_STEPPED_RADIUS_JUNCTION_REFUSAL`.
    #
    # `knot_feeds` is True since momwire#648: under `feed_model="point"` —
    # this class's default since momwire#654 — the gap lands at the arclength
    # it was NAMED, with the remainder carried in `feed_xi`, rather than
    # snapping to the nearest segment centre the way the base does. That is
    # what let this family serve the NEC-5 seam, which gates on the cell
    # (`eznec/_serve.py`, the `mesh.feeds and not capabilities.knot_feeds`
    # refusal). §7 of `tests/test_capabilities.py` measures it rather than
    # trusting it: the centre-knot probe reads ~1e-14 here.
    #
    # The cell is true PER INSTANCE, not per class (momwire#686). It holds
    # under `feed_model="point"`; under `feed_model="segment"` this same class
    # snaps to a segment centre and the class attribute still reads True. That
    # gap cannot reach the NEC-5 seam — `serve(deck, *, basis)` takes a basis
    # NAME and no solver kwargs, and momwire#654 collapsed the roster to ONE
    # Galerkin entry binding no `feed_model` — so the declaration is honest
    # for every path that reads it. The invariant is load-bearing rather than
    # obvious, so `test_sg_knot_feeds_describes_the_point_gap_the_roster_can_
    # build` pins it: a roster edit re-adding a `"segment"` spelling fails
    # there rather than silently handing the seam an instance this cell does
    # not describe.
    #
    # A consequence worth stating, because the row does not show it: the
    # base's `_KNOT_FEEDS_REFUSAL` is UNREACHABLE from this class.
    # `Capabilities.refusal` returns a reason only for a cell the solver does
    # not serve, and this one serves it — so `refusal("knot_feeds")` is None
    # here. There is no `_KNOT_FEEDS_REFUSAL` in this module and there should
    # not be one; the prose lives in `sinusoidal.py` for the family that
    # actually snaps.
    #
    # `refusals` REPLACES rather than extends, so the base's entries have to
    # be carried across by hand: the contact/refl-coef withdrawal
    # (momwire#282 stage 1) is inherited behaviour — this class's
    # constructor IS the base's — and the declaration has to say so or the
    # row reads as serving what the constructor refuses. `junction_ports`
    # and `node_gaps` are the two the base refused and this class does not,
    # so they are the two deliberately dropped.
    #
    # `singular_enrichment` is the third thing that has to be carried across,
    # and it is the one the REPLACE semantics actually lost: the cell reads
    # False on both classes for the same reason, and until momwire#792
    # this row answered it with `refusal()`'s generated one-liner while the
    # base answered with the reason. Same sentence, this class's name in it
    # (momwire#564) — the prose reaches antennaknobs' host dialogs verbatim.
    capabilities = SinusoidalSolver.capabilities._replace(
        # Differs from SinusoidalSolver in the TESTING and nothing else,
        # which is the whole point of the pair (antennaknobs#1006). The
        # `feed_model=` kwarg is the axis the bespoke `sin-galerkin` panel
        # hint was standing in for.
        axes={
            **SinusoidalSolver.capabilities.axes,
            "testing": ("galerkin",),
            # Default first (see BSplineSolver's row): this class defaults
            # to feed_model="point", so the point gap leads.
            "feed_model": ("point-gap", "segment-gap"),
        },
        junction_ports=True,
        node_gaps=True,
        knot_feeds=True,
        # momwire#673: under `feed_model="point"` the remainder rides in
        # `feed_xi`, so the gap is placed where it was named on either grid.
        centre_feeds=True,
        # momwire#980 D1: this family serves a FULLY-buried deck over a
        # Sommerfeld ground — direct at k_m through the complex-k far fill
        # (step E), image at k_m weighted A_m, and the below-family
        # remainder. It does NOT serve a MIXED deck (that is D2), which is
        # why the `buried` refusal below is replaced rather than dropped:
        # the capability is now True and the sentence names what is still
        # out of scope, so a caller cannot read "buried" as "any buried".
        buried=True,
        refusals={
            "junction_ports+finite_ground": _JUNCTION_PORTS_FINITE_GROUND_REFUSAL,
            "junction_ports+mixed_radii": _JUNCTION_PORTS_MIXED_RADII_REFUSAL,
            "contact+refl-coef": SinusoidalSolver.capabilities.refusals[
                "contact+refl-coef"
            ],
            "singular_enrichment": SINGULAR_ENRICHMENT_NEVER.format(
                cls="SinusoidalGalerkinSolver"
            ),
            # `contact` is still the base's. There is deliberately NO bare
            # `buried` row: since #980 D1 this class serves the fully-buried
            # deck and since D2 the mixed one, so `buried=True` and
            # `Capabilities.refusal("buried")` never reads a single-cell
            # entry again — but `scripts/capability_matrix.py` renders one
            # verbatim, which is how D1's "no mixed deck" sentence shipped in
            # the published matrix for a release after D2 made it false
            # (found on the D3 branch, which gates it). Everything a buried
            # deck still refuses has its own combination key below.
            # The three decks a buried serve still refuses, each with the
            # sentence `_medium_spec` actually raises — the same four rows
            # bspline declares, for the same reason: since D1 attempts a
            # buried deck, the refusal a caller meets is that module's, not
            # `_build_geometry`'s blanket one.
            "buried+pec": _medium_spec.BURIED_PEC_REFUSAL,
            "buried+refl-coef": _medium_spec.BURIED_REFL_REFUSAL,
            "buried+crossing": _medium_spec.CROSSING_REFUSAL,
            "buried+contact": _medium_spec.CONTACT_WITH_BURIED_REFUSAL,
            # Raised by `_below_interface.refuse_out_of_scope` at the fill, the
            # same sentence every buried family declares; undeclared here
            # until momwire#1326's default-kernel check asked this row first.
            "buried+extended_kernel": _below_interface.BURIED_EXTENDED_KERNEL_REFUSAL,
            "extended_kernel+stepped_radius_junction": (
                _EK_STEPPED_RADIUS_JUNCTION_REFUSAL
            ),
        },
    )

    def __init__(
        self,
        *,
        n_qp_test=8,
        n_qp_near=8,
        n_qp_node=16,
        near_factor=0.5,
        near_correction=True,
        feed_readout="centre",
        feed_model="point",
        node_ports=None,
        node_gaps=None,
        fill=None,
        **kwargs,
    ):
        # Seen by the base's feeds=[] check (its signature never learns the
        # node-port kwargs): a solve driven entirely through node ports/gaps
        # needs no gap feed, same as junction ports (#172/#305).
        self._node_drive_declared = bool(node_ports or node_gaps)
        super().__init__(**kwargs)
        fill = DEFAULT_FILL if fill is None else fill
        if fill not in ("direct", "mixed-potential"):
            raise ValueError(
                f"fill must be 'direct' or 'mixed-potential', got {fill!r}"
            )
        self.fill = fill
        self.n_qp_test = int(n_qp_test)
        self.n_qp_near = int(n_qp_near)
        self.n_qp_node = int(n_qp_node)
        self.near_factor = float(near_factor)
        self.near_correction = bool(near_correction)
        if self.extended_kernel and not self.near_correction:
            # The near path is not an accuracy refinement under EK, it is where
            # the on-segment pairs are computed at all: the delta's spike is a
            # wire radius wide, so the pairs whose observer sits inside the
            # source segment take the dense quadrature the near path passes,
            # and the far fill's cheap rule is only ever correct on those pairs
            # because the near path overwrites them. M1 mode would ship the
            # cheap rule's answer there, which is not an approximation but a
            # wrong number (5× the answer at Δ/a = 6). Refuse the combination.
            raise NotImplementedError(_EK_NEAR_CORRECTION_REFUSAL)
        if self.extended_kernel and self.junctions:
            # momwire#398 D2: a radius step AT a junction under EK is refused
            # at construction, not left to diverge at solve time — see
            # `_EK_STEPPED_RADIUS_JUNCTION_REFUSAL` for the measurement.
            # Uniform-radius junctions (every member sharing one `a`, the
            # overwhelmingly common case) are untouched by this check: the
            # comparison is exact float equality because `self._radius_per_wire`
            # only ever disagrees across wires when the caller asked it to
            # (`wire_radius` given as a per-wire sequence), never from any
            # solver-side rounding.
            radii = self._radius_per_wire
            for jw in self.junctions:
                member_radii = {radii[w] for w, _end in jw}
                if len(member_radii) > 1:
                    raise NotImplementedError(_EK_STEPPED_RADIUS_JUNCTION_REFUSAL)
        if feed_readout not in ("centre", "variational"):
            raise ValueError(
                f"feed_readout must be 'centre' or 'variational', got {feed_readout!r}"
            )
        self.feed_readout = feed_readout
        if feed_model not in ("segment", "point"):
            raise ValueError(
                f"feed_model must be 'segment' or 'point', got {feed_model!r}"
            )
        self.feed_model = feed_model
        # Node gaps (issue #305) are BSplineSolver's wire-end spelling of the
        # series port this family already carries as `node_ports` (M5b): each
        # (wire_index, "start"|"end", voltage) names the single member on
        # side_a, so the two spellings share one validation and one cut
        # vector. Entries append after any explicit node_ports.
        self.node_gaps = []
        node_port_entries = list(node_ports) if node_ports else []
        if node_gaps:
            # The spec's rules are every family's (`normalize_node_gaps`):
            # its messages for a repeated member and for a second gap at a
            # two-wire junction are the ones the other node-gap rows give.
            _wire_spec.normalize_node_gaps(
                node_gaps,
                self.junctions,
                len(self.wires_polylines),
                junction_ports=[j for j, _v in self.junction_ports],
            )
            member_pos = {
                (w, e): (j, m)
                for j, jw in enumerate(self.junctions)
                for m, (w, e) in enumerate(jw)
            }
            for i, g in enumerate(node_gaps):
                if len(g) != 3:
                    raise ValueError(
                        f"node_gaps[{i}]: expected (wire_index, 'start'|'end',"
                        f" voltage), got {g!r}"
                    )
                w_i, end_i, v_i = g
                pos = member_pos.get((int(w_i), end_i))
                if pos is None:
                    raise ValueError(
                        f"node_gaps[{i}]: wire {w_i} {end_i!r} is not a member "
                        "of any junction group — a series node gap lives at a "
                        "junction; for a feed inside a wire use feeds="
                    )
                j_idx, m_idx = pos
                # side_a = every member EXCEPT the named one. By the KCL
                # identity this family's span satisfies at every node, the
                # complement's cut vector is exactly the NEGATION of the
                # single member's — same physical gap, opposite orientation —
                # and the flip is what aligns this port with BSplineSolver's
                # convention (I_port = current from the node into the named
                # wire): the two families' member signs are mirrored (σ here
                # is +1 at a wire's natural end-2, B-spline's is −1 there).
                side_a = tuple(
                    m for m in range(len(self.junctions[j_idx])) if m != m_idx
                )
                node_port_entries.append((j_idx, side_a, complex(v_i)))
                self.node_gaps.append((int(w_i), end_i, complex(v_i)))
        self.node_ports = self._validate_node_ports(node_port_entries)
        # (base seg_view, k) → port-augmented seg_view. Keyed by identity on
        # the inherited view, which `_basis_coefs` already caches per (geom,
        # k, radius), so this rides that cache's validation.
        self._port_basis_cache = None
        # geom → {mirror: (group_obs, group_src)} for the EK pair rule.
        self._cached_ek_groups = None
        # geom → {mirror: (bad_lo, bad_hi)} for the EK end-bracket node rule.
        self._cached_ek_bad_ends = None
        # momwire#959 (a stopgap until #1330): segments below the radius AT A
        # GAP leave the delta-gap model unreliable. Advisory, once per solver.
        _wire_spec.advise_gap_mesh_floor(
            type(self).__name__,
            self.wires_polylines,
            self.n_per_edge_per_wire,
            self._radius_per_wire,
            self.extended_kernel,
            _wire_spec.solver_gaps(self, junctions_are_gaps=self.extended_kernel),
            gap_model="point",
        )

    # ------------------------------------------------------------------
    # Node ports (M5b formulation (a)) — a delta-gap EMF AT a junction node
    # ------------------------------------------------------------------

    def _validate_node_ports(self, node_ports):
        """Normalize `node_ports=` to [(j_idx, (member indices…), voltage), …].

        Accepted entry forms are `(j_idx, side_a)` (voltage 0) and
        `(j_idx, side_a, voltage)`, with `side_a` a sequence of indices into
        `self.junctions[j_idx]` naming the members on the port's + terminal.
        """
        if not node_ports:
            return []
        out = []
        # junction -> the member each earlier port there cuts off the node,
        # or None for a port with several members on both sides.
        seen = {}
        junction_port_idx = {j for j, _v in self.junction_ports}
        for entry in node_ports:
            if not isinstance(entry, (tuple, list)) or len(entry) not in (2, 3):
                raise ValueError(
                    "node_ports entries must be (junction_index, side_a) or "
                    f"(junction_index, side_a, voltage), got {entry!r}"
                )
            j_idx = int(entry[0])
            side_a = tuple(int(m) for m in entry[1])
            volts = complex(entry[2]) if len(entry) == 3 else 0j
            if not 0 <= j_idx < len(self.junctions):
                raise ValueError(
                    f"node_ports: junction index {j_idx} out of range "
                    f"(have {len(self.junctions)} junctions)"
                )
            if j_idx in junction_port_idx:
                raise ValueError(
                    f"junction {j_idx} is declared as both a junction_port and "
                    "a node_port — they are different port objects (net inflow "
                    "at the node vs an EMF across the node) and cannot share a "
                    "junction"
                )
            members = self.junctions[j_idx]
            if len(members) < 2:
                raise ValueError(
                    f"node_ports: junction {j_idx} has {len(members)} member(s); "
                    "a node port is an EMF ACROSS the node, so it needs at "
                    "least two wire-ends to put on opposite sides of the gap. "
                    "A one-terminal net-inflow port at a lone conductor end is "
                    "a junction_port, which this solver refuses (momwire#182 M5)"
                )
            if len(set(side_a)) != len(side_a):
                raise ValueError(f"node_ports: repeated member index in {side_a!r}")
            if any(not 0 <= m < len(members) for m in side_a):
                raise ValueError(
                    f"node_ports: member index out of range in {side_a!r} "
                    f"(junction {j_idx} has {len(members)} members)"
                )
            if not 0 < len(side_a) < len(members):
                raise ValueError(
                    f"node_ports: side_a {side_a!r} must be a nonempty PROPER "
                    f"subset of junction {j_idx}'s {len(members)} members — "
                    "both sides of the gap need at least one conductor"
                )
            self._check_shared_junction(seen, j_idx, side_a, len(members))
            out.append((j_idx, side_a, volts))
        return out

    @staticmethod
    def _check_shared_junction(seen, j_idx, side_a, n_members):
        """Admit a second (third, ...) node port at one junction only where it
        is a different cut (momwire#1300).

        A port with a single member m on one side cuts m's BRANCH off the
        node: its drive and readout are ±f_m, the current flowing into the
        node along m alone (`_node_cut_vectors`). Ports cutting different
        branches are different series EMFs, each in its own wire, and since
        a node port adds no basis column — it is a column of U and nothing
        else — several of them are the superposition of single-port solves
        and exactly as good as one. That is NEC-5's object per named wire at
        a K >= 3 node. With every branch cut the K columns sum to zero
        (#177's KCL identity), so the K-port Y has rank K - 1: one EMF added
        to every branch only moves the node's potential.

        Two shapes stay refused. At K = 2 the two possible cuts are one cut
        (f_0 = -f_1), so a second port there is the first again. And a cut
        with several members on BOTH sides (K >= 4) is not one branch; two
        of those, or one beside a branch cut, are partitions of the node no
        wire-end address can name and nothing measured says how they meet.
        """
        branch = None
        if len(side_a) == 1:
            branch = side_a[0]
        elif len(side_a) == n_members - 1:
            (branch,) = set(range(n_members)) - set(side_a)
        if j_idx not in seen:
            seen[j_idx] = [branch]
            return
        if n_members == 2:
            raise ValueError(
                f"node_ports: junction {j_idx} listed twice — it joins two "
                "wire ends, so it has one cut and one port"
            )
        if branch is None or None in seen[j_idx]:
            raise ValueError(
                f"node_ports: junction {j_idx} listed twice — several ports "
                "share a junction only when each cuts ONE member's branch "
                "off the node (one member on one side of it)"
            )
        if branch in seen[j_idx]:
            raise ValueError(
                f"node_ports: junction {j_idx} listed twice — member {branch}'s "
                "branch is already cut there"
            )
        seen[j_idx].append(branch)

    def _reject_junction_ports(self):
        """No-op at CONSTRUCTION: #177's stated blocker was the missing
        segment-integrated test row and this solver's rows are segment
        integrals, so the port basis is buildable here and gets built. The
        refusal moved to `_refuse_junction_port_solve`, which is where the
        measurement that actually kills it belongs."""

    def _refuse_junction_port_solve(self):
        """Refuse the junction-port solves M5b formulation (b) does NOT cover.

        M5's blanket refusal is superseded — see `_assemble_Z_ported` — and
        #191 narrowed what is left of it to the FINITE grounds:

        * under PEC (`ground_z` with no `ground_eps`) the node charge's image
          is a point charge at the mirrored node, i.e. a mirror of the term
          already removed, so the same correction removes it at the mirrored
          separation and the solve runs;
        * under `ground_eps` (Fresnel) or `ground_model="sommerfeld"` the
          image of a point charge is not a point charge — the reflection is
          angle-dependent and, for Sommerfeld, not an image at all — so the
          removed term has no closed mirror here. Leaving it in would restore
          a fraction of the very term the correction exists to take out, so
          the finite-ground + junction-port combination still raises rather
          than returning a plausible wrong number;
        * with mixed per-wire radii the kernel is not even symmetric (M2's
          finding), and the correction's regularization radius is ambiguous
          at a node whose members disagree about `a`.
        """
        if not self.junction_ports:
            return
        if self.ground_z is not None and self.ground_eps is not None:
            raise NotImplementedError(_JUNCTION_PORTS_FINITE_GROUND_REFUSAL)
        if self._uniform_radius is None:
            raise NotImplementedError(_JUNCTION_PORTS_MIXED_RADII_REFUSAL)

    def _node_cut_vectors(self, geom, seg_view, k):
        """(n_basis, P_node) — per node port, each basis's current THROUGH the
        node from the port's + side to its − side.

        Member m of the junction contributes σ_m·I_{i,m}(node) where σ_m is
        `_junction_members`' sign (+1 when the node is the member segment's
        natural end-2, −1 when it is end-1) and I_{i,m} is basis i's current
        shape σA + B·sin(kξ) + σC·cos(kξ) evaluated at ξ = σ_m·h_m/2. That
        product is basis i's current flowing INTO the node along member m, so
        summing it over the + side gives the current crossing the gap.

        Two facts make this the whole of formulation (a):

        * The sum over ALL members is identically zero — #177's KCL identity,
          the same property that makes a net-inflow junction port impossible
          here. So the cut vector is antisymmetric under swapping the sides
          and the port carries no net charge into the node: nothing terminates
          there, and M5's Z_pp ≈ 1/(jω·4πε·a) node self-energy never appears.
        * It is not identically zero, because current flows THROUGH the node.
          A point EMF sitting exactly at the node therefore has a well-defined
          Galerkin excitation, U_i = f_i(node)·V, even though it point-samples
          to nothing at any collocation point — which is #177's observation
          that a node source has an identically zero RHS in the point-matched
          solver, read as a statement about the TESTING rather than the basis.
        """
        # A crossing junction is grounded by geometry but is not shorted to
        # the plane — one member is in the soil — and its node port is served
        # with the continuity `_crossing_continuity_rows` imposes, which is
        # what restores the identity above there (momwire#1282).
        grounded = geom["grounded_junctions"] - self._node_port_crossing_junctions()
        out = np.zeros(
            (geom["n_segs"] + self._n_extra_cols(), len(self.node_ports)),
            dtype=np.complex128,
        )
        for p, (j_idx, side_a, _v) in enumerate(self.node_ports):
            if j_idx in grounded:
                raise ValueError(
                    f"junction {j_idx} is both grounded and a node port — a "
                    "node in the ground plane carries current into its own "
                    "image (#151), so the members' currents do not close on "
                    "each other and there is no through-current to drive"
                )
            out[:, p] = self._node_inflow(geom, seg_view, k, j_idx, side_a)
        return out

    def _node_inflow(self, geom, seg_view, k, j_idx, member_indices):
        """(n_basis,) — each basis's current flowing INTO junction `j_idx`'s
        node along the members named by `member_indices` (see
        `_node_cut_vectors` for the shape and the sign)."""
        out = np.zeros(geom["n_segs"] + self._n_extra_cols(), dtype=np.complex128)
        starts = seg_view["starts"]
        seg_h = np.asarray(geom["seg_h"], dtype=float)
        members = self._junction_members(geom, j_idx)
        for mi in member_indices:
            m, sgn = members[mi]
            half = 0.5 * float(seg_h[m])
            s, e = starts[m], starts[m + 1]
            sig = seg_view["sigma"][s:e]
            # ξ = σ_m·h_m/2; cos is even in ξ so the folded sin² term
            # needs no sign, and B's sin(kξ) carries it (#203).
            val = _basis_value(
                sig * seg_view["AC"][s:e],
                seg_view["B"][s:e],
                sig * seg_view["C"][s:e],
                _entry_k(seg_view, s, e, k),
                half * sgn,
            )
            np.add.at(out, seg_view["jbasis"][s:e], sgn * val)
        return out

    def _node_port_crossing_junctions(self):
        """Indices of the CROSSING junctions a node port names (momwire#1282).

        Empty without node ports or ground, so no other deck reaches the
        crossing scope from here."""
        if not self.node_ports or self.ground_z is None:
            return frozenset()
        named = {j for j, _side, _v in self.node_ports}
        candidates = named & set(self._grounded_junctions())
        if not candidates:
            return frozenset()
        return frozenset(candidates & set(self._crossing_junction_indices()))

    def _crossing_continuity_rows(self, geom, seg_view, k):
        """(m, n_basis) — the current-continuity row of every crossing
        junction a node port names, or None when there is none (momwire#1282).

        A crossing node is C0 in this family: each member's end basis is
        value 1 there and couples to no partner, and continuity through the
        node emerges from the crossing fill's wings and corner rather than
        being imposed. That is a statement about a node nothing drives. A
        node port's drive is one side's through-current, so with the members
        free the source sits between the node and that side alone and the
        far side's current is free to differ — Dan AC6LA's deck answered
        0.25 + 0.46j ohm apart between its two spellings. The row is the net
        inflow over ALL members, which an ordinary junction satisfies
        identically (#177); imposing it makes the two sides' cut vectors
        minus each other on the solution space, so either member names the
        same port — the series EMF NEC-5 puts on the node."""
        crossing = sorted(self._node_port_crossing_junctions())
        if not crossing:
            return None
        return np.stack(
            [
                self._node_inflow(geom, seg_view, k, j, range(len(self.junctions[j])))
                for j in crossing
            ]
        )

    @property
    def n_ports(self):
        """Network ports, in order: gap feeds, junction ports, node ports."""
        return len(self.feeds) + len(self.junction_ports) + len(self.node_ports)

    # ------------------------------------------------------------------
    # Near-pair selection
    # ------------------------------------------------------------------

    def _near_pairs(self, geom, src_c=None, src_t=None, n_samples=5, chunk_rows=None):
        """Ordered (test, source) segment pairs whose test integral needs the
        graded rule, as two index arrays.

        Selection is geometric rather than topological, so it covers the self
        pair, node-sharing neighbours, AND close-approach pairs that share no
        node (closely spaced parallel runs) — all of which put a sub-segment
        length scale into the test integrand.

        `src_c` / `src_t` default to the geometry's own segments (the
        free-space block). The M4 image blocks pass the MIRRORED source
        geometry, which is what makes the correction fire for a wire touching
        the ground plane: there the end segment and its own image share the
        in-plane endpoint, so the image block carries the same width-`a`
        endpoint spike the free-space self pair does. Image half-lengths are
        unchanged by mirroring, so `reach` is still the geometry's own.

        Exact segment-segment distance is only computed for pairs that survive
        a cheap centre-distance prefilter. That prefilter is a strict superset:
        |c_m − c_n| ≤ d_min + h_m + h_n always, so any pair meeting the
        `near_factor` criterion also meets the prefilter.

        The prefilter itself is row-chunked over the test axis rather than
        run on the full (N, N_src) arrays at once (issue #334): each stage —
        `reach`, `cdist`, the scaled-reach product, and the `<=` boolean —
        is its own (N, N_src) transient (3 float64 + 1 bool = 25 bytes per
        element), alive together for a result that collapses to O(hits)
        index pairs. A row-chunk of `chunk_rows` test segments bounds that
        stack at `chunk_rows * N_src * 25` bytes; `chunk_rows=None` picks
        `max(1, 4_000_000 // (25 * N_src))`, capping one block's transient
        at ~4 MB regardless of N. `np.argwhere` on a C-contiguous array
        already visits rows in order, and each block is processed in
        increasing row order, so concatenating the per-block hits reproduces
        the SAME pairs in the SAME lexicographic order as the unchunked call
        — bit-exact, not an approximation. Pass an explicit `chunk_rows` to
        force a particular chunking (e.g. to exercise a partial tail chunk
        in tests).
        """
        c = geom["seg_centers"]
        t = geom["seg_tangents"]
        cs = c if src_c is None else src_c
        ts = t if src_t is None else src_t
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)

        n_test = c.shape[0]
        n_src = cs.shape[0]
        if chunk_rows is None:
            chunk_rows = max(1, 4_000_000 // (25 * max(1, n_src)))
        thresh_factor = 1.0 + self.near_factor

        m_parts, n_parts = [], []
        for i0 in range(0, n_test, chunk_rows):
            self._checkpoint()  # per test-row chunk of the prefilter
            i1 = min(i0 + chunk_rows, n_test)
            reach_blk = hh[i0:i1, None] + hh[None, :]
            # cdist rather than an explicit (rows, N_src, 3) difference:
            # the prefilter should not cost 3x the memory of the matrix
            # it is protecting.
            dc_blk = scipy.spatial.distance.cdist(c[i0:i1], cs)
            cand_blk = np.argwhere(dc_blk <= thresh_factor * reach_blk)
            m_parts.append(cand_blk[:, 0] + i0)
            n_parts.append(cand_blk[:, 1])
        m = np.concatenate(m_parts) if m_parts else np.empty(0, dtype=np.int64)
        n = np.concatenate(n_parts) if n_parts else np.empty(0, dtype=np.int64)

        s = np.linspace(-1.0, 1.0, n_samples)

        def _gap(ai, bi, ca, ta, cb, tb):
            """min over sample points of segment `ai` (geometry ca/ta) of the
            distance to the clamped axis of segment `bi` (geometry cb/tb)."""
            pts = (
                ca[ai][:, None, :]
                + (hh[ai][:, None] * s[None, :])[:, :, None] * ta[ai][:, None, :]
            )  # (P, n_samples, 3)
            d = pts - cb[bi][:, None, :]
            u = np.einsum("pgd,pd->pg", d, tb[bi])
            u = np.clip(u, -hh[bi][:, None], hh[bi][:, None])
            perp = d - u[..., None] * tb[bi][:, None, :]
            return np.linalg.norm(perp, axis=-1).min(axis=1)

        gap = np.minimum(
            _gap(m, n, c, t, cs, ts),  # test segment sampled against the source
            _gap(n, m, cs, ts, c, t),  # and the other way round
        )
        keep = gap <= self.near_factor * (hh[m] + hh[n])
        return m[keep], n[keep]

    # ------------------------------------------------------------------
    # Junction ports (M5) — the port basis column
    # ------------------------------------------------------------------

    def _junction_port_view(self, geom, k, base_view):
        """Append one basis column per junction port to the CSR seg-view.

        Port p's basis is #177's `g_p = (1/P_J)·Σ_m ext_m`, where `ext_m` is
        the three-term extension a real N⁻ neighbour would carry on member
        segment m at unit Q,

            A_m = a_m/sin(kΔ_m),  B_m = a_m/(2cos(kΔ_m/2)),
            C_m = −a_m/(2sin(kΔ_m/2)),

        with `a_m` the member's own Eq-25 log constant. That shape is zero at
        the member's FAR end and carries current `a_m(1−cos kΔ_m)/sin kΔ_m`
        — the same `P_minus_atom` the basis coefficients are built from —
        into the node. Summing over the members and dividing by
        `P_J = Σ_m P_minus_atom[m]` therefore gives a current distribution
        with **unit net inflow at the node**, KCL-clean everywhere else, and
        vanishing outside the K member segments: precisely the vector #177
        proved the ordinary span cannot contain.

        The entries are merged into the same segment-major CSR the inherited
        `_basis_coefs` returns, with basis indices N…N+P−1. Everything
        downstream — the test quadrature, the M2 near-pair correction, the
        source-side coefficient matrices, `_feed_segment_current`,
        `currents_at_knots` — then treats them as ordinary bases, which is
        why the port row is a Galerkin row and not a bolted-on constraint.
        """
        N = geom["n_segs"]
        grounded = geom["grounded_junctions"]
        for j_idx, _v in self.junction_ports:
            if j_idx in grounded:
                raise ValueError(
                    f"junction {j_idx} is both grounded and a junction port — "
                    "a node in the ground plane is connected to its own image "
                    "instead of to its partners (#151), so its voltage is "
                    "pinned and it cannot also be a driven port"
                )

        seg_h = np.asarray(geom["seg_h"], dtype=float)
        a = (
            self._uniform_radius
            if self._uniform_radius is not None
            else self._seg_radius(geom)
        )
        a_const = 1.0 / (np.log(2.0 / (k * a)) - _EULER_GAMMA)
        a_const = np.broadcast_to(np.asarray(a_const, dtype=float), (N,))
        kd = k * seg_h
        # (1 − cos kΔ)/sin kΔ = tan(kΔ/2), exactly (momwire#799). The literal
        # quotient's numerator is O(kΔ²) computed to an absolute ε — 1.9e-12
        # relative at kΔ = 3.8e-3, growing as 1/kΔ². `sinusoidal.py`'s
        # `P_minus_atom` is the same atom and takes the same spelling.
        atom = np.tan(0.5 * kd) * a_const

        segs, bases, A, B, C, AC, sig = [], [], [], [], [], [], []
        for p, (j_idx, _v) in enumerate(self.junction_ports):
            members = self._junction_members(geom, j_idx)
            m_seg = np.array([m for m, _s in members], dtype=np.int64)
            m_sig = np.array([s for _m, s in members], dtype=np.int8)
            P_J = float(atom[m_seg].sum())
            if P_J == 0.0:
                raise ValueError(
                    f"junction_ports: junction {j_idx} has a degenerate port "
                    "normalization (Σ P⁻ atoms = 0) at this mesh"
                )
            q = a_const[m_seg] / P_J
            segs.append(m_seg)
            bases.append(np.full(m_seg.shape, N + p, dtype=np.int64))
            A.append(q / np.sin(kd[m_seg]))
            B.append(q / (2.0 * np.cos(0.5 * kd[m_seg])))
            C.append(-q / (2.0 * np.sin(0.5 * kd[m_seg])))
            # A + C = q·[1/sin(kΔ) − 1/(2 sin(kΔ/2))], the N⁻ identity exactly
            # (momwire#606). Same closed form, because these ARE that shape.
            AC.append(q * _recip_sin_gap(kd[m_seg]))
            sig.append(m_sig)

        starts = base_view["starts"]
        base_seg = np.repeat(np.arange(N, dtype=np.int64), np.diff(starts))
        all_seg = np.concatenate([base_seg] + segs)
        all_basis = np.concatenate([base_view["jbasis"]] + bases)
        all_A = np.concatenate([base_view["A"]] + A).astype(np.complex128)
        all_B = np.concatenate([base_view["B"]] + B).astype(np.complex128)
        all_C = np.concatenate([base_view["C"]] + C).astype(np.complex128)
        all_AC = np.concatenate([base_view["AC"]] + AC).astype(np.complex128)
        all_sigma = np.concatenate([base_view["sigma"]] + sig)

        order = np.argsort(all_seg, kind="stable")
        new_starts = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(np.bincount(all_seg, minlength=N), out=new_starts[1:])
        A_ord, C_ord = all_A[order], all_C[order]
        # A + C on the port entries cancels exactly as it does on an ordinary
        # N⁻ extension — C_m/A_m = −cos(kΔ_m/2) — so the port columns carry
        # the same closed form, and every consumer can read `AC` without
        # asking whether the entry is a port's.
        #
        # CARRIED, not recomputed (momwire#606). This view used to publish
        # `A_ord + C_ord`, which did two wrong things at once: it gave the
        # port entries the float sum whose relative error is 8ε/(kΔ)², and —
        # because the concatenation covers the base entries too — it threw
        # away the closed-form `AC` the inherited view had already computed
        # for every ORDINARY entry. A junction-port solve therefore lost the
        # whole of #606's coefficient fix, silently, while a portless one
        # kept it. Concatenating `base_view["AC"]` is what keeps the two
        # spellings of this solver the same solver.
        return {
            "starts": new_starts,
            "jbasis": all_basis[order],
            "A": A_ord,
            "B": all_B[order],
            "C": C_ord,
            "AC": all_AC[order],
            "sigma": all_sigma[order],
        }

    # ------------------------------------------------------------------
    # Junction ports (M5b formulation (b)) — the node charge, held outside
    # ------------------------------------------------------------------

    def _junction_node_position(self, geom, j_idx):
        """World coordinates of junction `j_idx`'s node."""
        seg, sgn = self._junction_members(geom, j_idx)[0]
        return (
            geom["seg_centers"][seg]
            + (sgn * 0.5 * float(geom["seg_h"][seg])) * geom["seg_tangents"][seg]
        )

    def _port_node_positions(self, geom, mirror=False):
        """(P, 3) node positions of the junction ports, optionally mirrored
        across z = ground_z — the PEC image's node, where the image of the
        removed lumped charge sits (#191). The mirror is
        `_image_source_centers_tangents`' verbatim, applied to a point."""
        nodes = np.array(
            [self._junction_node_position(geom, j) for j, _v in self.junction_ports]
        )
        if not mirror:
            return nodes
        return _ground_mirror.mirror_positions(nodes, self.ground_z)

    def _node_charge_columns(self, geom, seg_view, k, nodes=None, eta=None):
        """D[i, p] = ∫ f_i(s) ŝ·E_q(s) ds — every basis tested against the
        field of the LUMPED charge port p deposits at its node, (n_basis, P).

        `E_q` is the field of the point charge q = I/(jω) that a unit
        terminating current leaves behind:

            E_q(r) = -jη/(4πk)·(1+jkR)·e^{-jkR}·(r − r_node)/R³,
            R = √(|r − r_node|² + a²)

        Neither constant is asserted. The prefactor -jη/(4πk) is literally
        `pref_rho_const` in `sinusoidal.py`: the Eqs 78-79 ENDPOINT terms are
        the point-charge fields of the charges a constant current deposits at
        a segment's two ends, which is why they are what M5 measured. The
        regularization matches the kernel's own — the kernel's
        r₀ = √(ρ² + a² + Δz²) IS √(|Δ|² + a²), since ρ² + Δz² = |Δ|². And the
        expression is exactly −∇Φ of Φ = q·e^{-jkR}/(4πεR) with the same
        regularized R, so integrating by parts against a basis is exact and
        the boundary term it produces is the one `_node_charge_pair_block`
        adds back.

        Quadrature is graded toward both segment ends (`n_qp_node`), because
        the integrand carries the same width-`a` spike M2's rule exists for —
        now sourced by a genuine point charge instead of a basis endpoint.
        Converged: the port impedance moves 2.8e-5 from 8 to 12 panels-per-end
        and 4e-9 from 12 to 16, which is why the default is 16.

        `nodes` overrides where the charges sit; the PEC-image correction
        (#191) passes the MIRRORED nodes, which is the whole of its cost —
        the image of a point charge is a point charge, so the same kernel at
        the mirrored separation is exact rather than an approximation.
        """
        N = geom["n_segs"]
        n_basis = N + self._n_extra_cols()
        a = float(self._uniform_radius)
        seg_c, seg_t = geom["seg_centers"], geom["seg_tangents"]
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)
        pref = -1j * self._fill_eta(k, eta) / (4.0 * np.pi * k)
        starts = seg_view["starts"]

        if nodes is None:
            nodes = self._port_node_positions(geom)
        D = np.zeros((n_basis, len(nodes)), dtype=np.complex128)
        for m in range(N):
            gx, gw = _graded_endpoint_rule(
                a / hh[m], self.n_qp_node, self._leggauss_cached
            )
            xi = hh[m] * gx
            w = hh[m] * gw
            pts = seg_c[m][None, :] + xi[:, None] * seg_t[m][None, :]
            lo, hi = starts[m], starts[m + 1]
            sig = seg_view["sigma"][lo:hi]
            fval = _basis_value(
                (sig * seg_view["AC"][lo:hi])[:, None],
                seg_view["B"][lo:hi][:, None],
                (sig * seg_view["C"][lo:hi])[:, None],
                k,
                xi[None, :],
            )  # (nnz_m, nq)
            for p, node in enumerate(nodes):
                d = pts - node[None, :]
                R = np.sqrt((d * d).sum(axis=1) + a * a)
                Et = pref * (1.0 + 1j * k * R) * np.exp(-1j * k * R) * (d @ seg_t[m])
                np.add.at(
                    D[:, p],
                    seg_view["jbasis"][lo:hi],
                    (fval * (w * Et / R**3)[None, :]).sum(axis=1),
                )
        return D

    def _node_charge_pair_block(self, geom, k, eta=None):
        """S[p, q] — the lumped-charge × lumped-charge term, (P, P).

        Subtracting `_node_charge_columns` from both the row and the column
        takes this term out TWICE (it is the boundary term the port basis's
        own integration by parts contributes at its node), so it goes back in
        once. Its value is the mixed-potential pairing of the two node
        charges q = 1/(jω) at the same regularized separation the columns
        use:

            S[p, q] = -jω·q_p·Φ_q(node_p) = jη·e^{-jkR}/(4πkR),
            R = √(|node_p − node_q|² + a²)

        Getting that regularization right is not cosmetic. With the bare
        separation instead of √(d² + a²) the residue against `BSplineSolver`
        runs as a²/d³ — 0.25 % at the oracle's 0.04 gap, 2.1 % at 0.02, and a
        clean a² law across a decade of radius. With it, the two formulations
        agree to 2e-5.
        """
        nodes = self._port_node_positions(geom)
        return self._lumped_pair_block(nodes, nodes, k, eta=eta)

    def _node_charge_image_pair_block(self, geom, k, eta=None):
        """S_img[p, q] — the same lumped-lumped term between node p and the
        PEC IMAGE of node q, at the mirrored separation (#191).

        Symmetric because reflection is an isometry:
        |node_p − M·node_q| = |M·node_p − node_q|.
        """
        nodes = self._port_node_positions(geom)
        return self._lumped_pair_block(
            nodes, self._port_node_positions(geom, True), k, eta=eta
        )

    def _lumped_pair_block(self, nodes_row, nodes_col, k, eta=None):
        """Mixed-potential pairing of unit node charges at two point sets,
        regularized at the wire radius exactly as `_node_charge_columns` is."""
        a = float(self._uniform_radius)
        d = nodes_row[:, None, :] - nodes_col[None, :, :]
        R = np.sqrt((d * d).sum(axis=-1) + a * a)
        eta = self._fill_eta(k, eta)
        return 1j * eta * np.exp(-1j * k * R) / (4.0 * np.pi * k * R)

    def _assemble_Z_ported(self, geom, k, eta=None):
        """`_assemble_Z` with M5b formulation (b) applied — the matrix every
        SOLVE uses when junction ports are present.

        `_assemble_Z` itself is left alone, so M5's refuted reaction-form
        construction stays reachable and its measurements keep reproducing;
        the tests that pin the blocker call it directly.

        The correction is one statement: the port basis's charge is its LINE
        charge only. The lumped charge it would otherwise leave at the node
        is removed from the source, symmetrically —

            G'[i, p] = G'[p, i] = G[i, p] − D[i, p]
            G'[p, q] = G[p, q] − D[p, q] − D[q, p] + S[p, q]

        — which is exactly what `BSplineSolver`'s Lagrange-multiplier port
        does implicitly, and what makes it a MIXED-POTENTIAL construct: the
        current that reaches the node leaves through an ideal unmodelled
        lead, so nothing accumulates there. That reading is not an
        interpretation — B-spline's own one-terminal port impedance is
        −1.87 − 35.05j where the node self-capacitance would be 9.5e4,
        i.e. it carries none of it.

        `G − Gᵀ` is untouched by the correction (both halves get the same
        scalars), so the fill's reciprocity is exactly as good as it was.

        Over a PEC ground the same correction runs a second time on the image
        block (#191). `_assemble_Z` builds that block as the free-space field
        of MIRRORED sources and the caller subtracts it once,
        G = A − B, so the removed term's image is a mirror of a term already
        removed and the arithmetic is fixed rather than chosen:

            A' = A − D − Dᵀ + S            (free space, above)
            B' = B − D_img − D_imgᵀ + S_img (mirrored nodes)
            G' = A' − B'

        i.e. the image half enters with the OPPOSITE sign, at the mirrored
        separation. Nothing else changes: the image of a point charge under
        PEC is a point charge, so the correction needs no new kernel and no
        new constant. Fresnel/Sommerfeld images are not point charges and are
        refused upstream (`_refuse_junction_port_solve`).
        """
        G, seg_view = self._assemble_Z(geom, k, eta)
        if not self.junction_ports:
            return G, seg_view
        N = geom["n_segs"]
        D = self._node_charge_columns(geom, seg_view, k, eta=eta)
        # In place on the fill's own G (momwire#1224): `_assemble_Z` returns a
        # fresh matrix nobody else holds, so the copy this used to take was a
        # second (n_basis, n_basis) array for nothing — the same subtractions
        # land on the same values either way, and G stays F-ordered for
        # `_solve_in_place`.
        G[:, N:] -= D
        G[N:, :] -= D.T
        G[N:, N:] += self._node_charge_pair_block(geom, k, eta=eta)
        # The #151/#191 node charge is the Galerkin analogue of the point-
        # matched contact-charge read: a TESTING-SCHEME correction that happens
        # to involve the image, deliberately outside the `FieldGround` sketch
        # (`docs/design/field-ground-interface.md`, "what stays out"), so it
        # still reads `ground_z` here rather than the fill's ground object.
        # Only PEC reaches it — finite grounds are refused upstream.
        if self.ground_z is not None:
            D_img = self._node_charge_columns(
                geom,
                seg_view,
                k,
                nodes=self._port_node_positions(geom, mirror=True),
                eta=eta,
            )
            G[:, N:] += D_img
            G[N:, :] += D_img.T
            G[N:, N:] -= self._node_charge_image_pair_block(geom, k, eta=eta)
        return G, seg_view

    def _basis_coefs(self, geom, k):
        """The inherited CSR basis view, extended with the junction-port
        basis columns. Verbatim passthrough when there are no ports."""
        base_view = super()._basis_coefs(geom, k)
        if not self.junction_ports:
            return base_view
        cached = self._port_basis_cache
        if cached is not None and cached[0] is base_view and cached[1] == k:
            return cached[2]
        view = self._junction_port_view(geom, k, base_view)
        self._port_basis_cache = (base_view, k, view)
        return view

    # ------------------------------------------------------------------
    # Galerkin matrix assembly
    # ------------------------------------------------------------------

    def _crossing_basis(self, geom, k=None, seg_view=None):
        """This solver's basis as the crossing trunk reads it (momwire#980
        step B): a `SinusoidalBasisSampler` over `_basis_coefs`' entries at
        `k`. The segment bases only — a junction-port column (#209) is a
        current distribution on real segments and has no wing of its own on
        the interface axes."""
        k = self.k if k is None else k
        if seg_view is None:
            seg_view = self._basis_coefs(geom, k)
        return SinusoidalBasisSampler(seg_view, k, geom["seg_h"], geom["n_segs"])

    def _test_context(self, geom, seg_view, k):
        """Everything the TEST side of the Galerkin integral needs, built once
        per assembly and shared by the free-space block and every ground block.

        Holds the Gauss points along each test segment (as a flat (N·nq, 3)
        observer list), the per-observer thin-wire radius, and `w_entry` —
        the (nnz, nq) table of test-function value × arc weight for every
        (segment, basis) support entry. For each such entry the test
        function's value at quad node q is
        f_{i,m}(ξ_q) = σA + B·sin(kξ_q) + σC·cos(kξ_q) (the current-shape
        convention pinned in `_evaluate_basis_at_points`) and the arc weight
        is gw_q·(h_m/2).

        That value is evaluated on the FOLDED shape set (#203),

            f = σ(A+C)·1 + B·sin(kξ) + σC·(cos kξ − 1)
              = σ·AC + B·sin(kξ) − 2σC·sin²(kξ/2),

        which is the same function rearranged so that neither term is larger
        than the answer. The literal spelling subtracts two O(1) numbers to
        get an O((kΔ)²/8) one — ε·8/(kΔ)² relative, 1.2e-10 at N=801 — and
        `w_entry` multiplies the whole free-space fill, so that error lands
        on G at full strength. `AC` comes from `_basis_coefs`' per-branch
        closed forms — not a float `A + C`, which carries an absolute ε
        against an O((kΔ)²) answer (momwire#606) — and `cos kξ − 1` is
        spelled as −2sin²(kξ/2), so both cancellations are done where they
        are exact.
        """
        N = geom["n_segs"]
        seg_c = geom["seg_centers"]  # (N, 3)
        seg_t = geom["seg_tangents"]  # (N, 3)
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)  # (N,) half-lengths

        gx, gw = self._leggauss_cached(self.n_qp_test)  # nodes on [-1, 1]
        nq = gx.shape[0]
        xi = hh[:, None] * gx[None, :]  # (N, nq) local arc from each centre
        obs_pts = seg_c[:, None, :] + xi[:, :, None] * seg_t[:, None, :]  # (N,nq,3)
        obs_c = np.ascontiguousarray(obs_pts.reshape(N * nq, 3))
        obs_t = np.ascontiguousarray(
            np.broadcast_to(seg_t[:, None, :], (N, nq, 3)).reshape(N * nq, 3)
        )
        a_seg = None if self._uniform_radius is not None else self._seg_radius(geom)
        # Scalar on the uniform path, (N·nq, 1) per-observer column when mixed
        # — the same shape contract `_field_components_bcast` takes for `a`.
        a_obs = self._uniform_radius if a_seg is None else np.repeat(a_seg, nq)[:, None]

        starts = seg_view["starts"]
        counts = np.diff(starts)
        m_of_entry = np.repeat(np.arange(N, dtype=np.int64), counts)  # (nnz,)
        sig = seg_view["sigma"].astype(np.complex128)
        sigA = sig * seg_view["A"]
        B = seg_view["B"]
        sigC = sig * seg_view["C"]
        sigAC = sig * seg_view["AC"]
        fval = _basis_value(
            sigAC[:, None], B[:, None], sigC[:, None], k, xi[m_of_entry]
        )  # (nnz, nq)
        w_entry = (gw[None, :] * hh[m_of_entry][:, None]) * fval  # (nnz, nq)

        return {
            "N": N,
            "nq": nq,
            "hh": hh,
            "obs_c": obs_c,
            "obs_t": obs_t,
            # which test segment each flat observer belongs to — the row
            # coordinate a per-segment-pair projector table indexes with.
            "m_of_obs": np.repeat(np.arange(N, dtype=np.int64), nq),
            "a_obs": a_obs,
            "a_seg": a_seg,
            "starts": starts,
            "counts": counts,
            "m_of_entry": m_of_entry,
            "i_of_entry": seg_view["jbasis"],
            "sigA": sigA,
            "B": B,
            "sigC": sigC,
            "sigAC": sigAC,
            "w_entry": w_entry,
        }

    @staticmethod
    def _tested_contrib_rows(w_entry, m_local, nq, Phi):
        """contrib[entry, n] = Σ_q w_entry[entry, q] · Phi[m(entry), q, n], on
        caller-sliced rows: `w_entry`/`m_local` cover one contiguous run of
        support entries and `m_local` indexes Phi's first axis directly (the
        caller subtracts its block offset).

        Every caller is blocked (momwire#332): the free-space and image fills
        over test segments, the Sommerfeld remainder over the evaluator's own
        observer chunks. There is no whole-matrix entry point left.

        Accumulated one quadrature node at a time: the equivalent einsum over
        `Phi[m_local]` would first materialize an (nnz, nq, N) gather, nq×
        the peak memory for the same arithmetic.
        """
        out = np.zeros((w_entry.shape[0], Phi.shape[-1]), dtype=np.complex128)
        for q in range(nq):
            out += w_entry[:, q, None] * Phi[m_local, q, :]
        return out

    # ------------------------------------------------------------------
    # The extended kernel's pair rule (momwire#246 / #249 §4)
    # ------------------------------------------------------------------

    def _ek_axis_labels(self, geom, mirror):
        """Coaxial-and-equal-radius group labels for this geometry, as
        `(group_obs, group_src)` — both (n_segs,) int64 — with a pair eligible
        for the extended kernel iff the two labels are equal.

        The rule and the scan are `_bspline_kernels._ek_axis_groups`', shared
        verbatim rather than re-derived: two segments group together iff their
        axes are the same LINE (NEC's |t·t'| ≥ 1 − 1e-6, plus a perpendicular
        offset test that NEC never needs because it only ever asks the question
        of segments already sharing an endpoint) and their radii agree to
        1e-6 relative (NEC's f.2042-2043).

        `mirror=True` is an image block, whose SOURCE segments are the real
        ones reflected through z = ground_z. Its two label arrays come from ONE
        scan over the CONCATENATION of the real and mirrored segments, then
        split — exactly `BSplineSolver._ek_axis_labels`' reason, and it is not
        an optimization: two independent scans would label a horizontal wire
        and its image both 0 and declare every real/image pair coaxial, when
        they are parallel and offset by twice the height. Jointly scanned, a
        vertical monopole standing on the plane maps onto its own axis and IS
        one group (NEC's IND = 0 ground-contact branch), while the horizontal
        wire splits in two and its image block stays reduced.

        Cached per geometry OBJECT (identity) and per mirror flag, so a swept
        solve pays the O(N·G) scan once rather than per k.
        """
        cached = self._cached_ek_groups
        if cached is None or cached[0] is not geom:
            cached = (geom, {})
            self._cached_ek_groups = cached
        hit = cached[1].get(mirror)
        if hit is not None:
            return hit

        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)[:, None]
        seg_l = seg_c - hh * seg_t
        seg_r = seg_c + hh * seg_t
        seg_a = self._seg_radius(geom)
        if mirror:
            n = seg_c.shape[0]
            gz = self.ground_z
            mirror_positions = _ground_mirror.mirror_positions
            joint = _ek_axis_groups(
                np.vstack([seg_l, mirror_positions(seg_l, gz)]),
                np.vstack([seg_r, mirror_positions(seg_r, gz)]),
                np.vstack([seg_t, _ground_mirror.mirror_tangents(seg_t)]),
                np.concatenate([seg_a, seg_a]),
            )
            hit = (joint[:n], joint[n:])
        else:
            labels = _ek_axis_groups(seg_l, seg_r, seg_t, seg_a)
            hit = (labels, labels)
        cached[1][mirror] = hit
        return hit

    def _ek_far_labels(self, geom, mirror, n_panels=1):
        """The `_EKFarLabels` payload for one FUSED far-fill call, or None when
        the extended kernel is off (momwire#358).

        The same pair rule `_ek_pairs` evaluates, handed to the C++ sweep
        UNEVALUATED: the two label arrays and one radius per source segment,
        with `eligible = g_obs[m] == g_src[n] and g_obs[m] >= 0` scored per
        pair inside the kernel.

        Labels per TEST SEGMENT are enough for the fused fill, which the mask
        spelling obscured. Its observer axis is the fill's `obs_c` rows, and
        those rows are `m_of_obs = repeat(arange(N), nq)` — the kernel checks
        that it was given exactly `M*nq` of them and reconstructs
        `o = m*nq + qt` itself. So a mask row depended on its observer only
        through that observer's test segment, and every one of a segment's nq
        rows carried the identical row. `_ek_pairs`' own construction says the
        same thing from the other side: it indexes `group_obs` by the caller's
        `m_idx`, which for this call is `m_of_obs[:, None]`.

        Mirroring is carried by the labels and not by the observers.
        `_ek_axis_labels(geom, True)` returns DIFFERENT obs and src halves out
        of one joint scan — the real segments and their reflections — so an
        image block scores the real test segment against the MIRRORED source,
        which is what makes a vertical monopole eligible against its own image
        and a horizontal wire not. Passing both halves keeps that asymmetry
        where it was; passing one array twice would silently restore #151's
        "every wire is coaxial with its image" bug on the image block.
        """
        if not self.extended_kernel:
            return None
        group_obs, group_src = self._ek_axis_labels(geom, mirror)
        return _EKFarLabels(self._seg_radius(geom), group_obs, group_src, n_panels)

    def _ek_pairs(self, geom, m_idx, n_idx, mirror, n_panels=1):
        """The `_EKPairs` payload for one NUMPY field-kernel call, or None when
        the extended kernel is off.

        `m_idx` / `n_idx` are the (test segment, source segment) index arrays
        of whatever pairing the caller has built — (rows, 1) against (1, N) for
        one block of the numpy far loop, (P, 1) against (P, 1) for the
        near-pair path — so the returned mask and source radius broadcast
        against that call's field tables exactly as the indices do. The FUSED
        far fill no longer comes here: it takes the labels themselves
        (`_ek_far_labels`), because it is the one caller whose pairing is
        big enough for the materialized mask to be a residency item.

        Eligibility is `group_obs[m] == group_src[n]`, i.e. #249 §4's PAIR
        rule, and NOT `SinusoidalSolver._ek_gating`'s per-END IND codes. That
        is the load-bearing choice of this arc. NEC decides per source segment
        end whether the current continues straight through into an identical
        conductor; transplanted into a Galerkin fill that decision depends on
        which segment is the SOURCE, so G(i, j) would be extended while
        G(j, i) was not and ‖G−Gᵀ‖/‖G‖ — the fill's own error detector, and a
        gate this solver has carried since M2 — would stop measuring anything.
        The pair rule is symmetric by construction (label equality is), it
        reproduces NEC's decision on straight wires (IND = 1 free ends and
        IND = 0 collinear junctions are both same-line same-radius) and on
        perpendicular ground contacts via the mirrored source, and it is
        strictly MORE conservative than NEC at bends, radius steps and K ≥ 3
        junctions, where NEC still extends the cross-arm pairs — worth ~1 % of
        Z at Δ/a = 2 and O(h) under refinement (#249 §4.3).

        What this mask must NOT be asked to decide is the delta's END
        BRACKET: that term is O(1/a) per source end and cancels only across a
        whole node, so truncating the eligible set at a node leaves it
        uncancelled and the fill divergent as the wire thins.
        `_ek_reduced_ends` scores that one decision per NODE instead, and
        `_ek_bracket_correction_tested` takes the difference back off
        (momwire#299). Everything this mask still governs is the delta's
        SMOOTH half, which is O(a²) and harmless however it is truncated.

        `n_panels` is the delta quadrature's density, and it is the second
        tier of the same near/far split the test quadrature already runs on.
        An eligible pair whose observer can sit ON the source segment needs the
        dense rule (`_N_PANEL_EK_DELTA_NEAR`); one whose observer is a segment
        length away is converged on a single panel. Coaxial pairs are exactly
        the ones for which "observer inside the source segment's span" means
        "the two segments overlap", i.e. separation zero, i.e. a NEAR pair — so
        the split by near-ness IS the split by whether the spike is inside the
        integration path, and no pair is short-changed.
        """
        if not self.extended_kernel:
            return None
        group_obs, group_src = self._ek_axis_labels(geom, mirror)
        gi = group_obs[m_idx]
        gj = group_src[n_idx]
        # `>= 0` mirrors `_bspline_kernels._ek_pair_mask`: the label convention
        # reserves negatives for a future never-extend marker, and `_ek_axis_
        # groups` emits none today.
        eligible = (gi == gj) & (gi >= 0)
        return _EKPairs(self._seg_radius(geom)[n_idx], eligible, n_panels)

    def _ek_reduced_ends(self, geom, mirror):
        """The NODE predicate of momwire#299, scored per SOURCE segment end:
        `(bad_lo, bad_hi)`, boolean (N,), True where the end sits on a node
        whose extended-kernel cap must be REDUCED away again.

        The predicate itself is the positive one — extend the cap at node P iff
        every segment meeting at P shares one axis line and one radius, which
        is NEC's IND = 0 read as a property of the NODE rather than of the
        (observer, source) pair — so these arrays are its complement:

            bad = (K ≥ 3 at this node) OR (some segment at it carries a
                   different `_ek_axis_labels` group)

        A free end has one segment at its node and is therefore never bad, and
        neither is a ground contact (`ground_minus`/`ground_plus`, no
        neighbour edge): both keep exactly the caps #246 gave them, which is
        what makes a straight deck and every ground deck bit-identical under
        this correction. K ≥ 3 is `count ≥ 2` because momwire emits K(K−1)
        neighbour edges at a K-member junction — the same reading of the same
        tables `SinusoidalSolver._ek_gating` makes of NEC's reciprocal ICON
        test — and it is redundant with the label test on any realizable
        junction (three segments meeting at a point cannot pairwise share one
        line without overlapping); it is spelled anyway because NEC's rule is
        the topological one and this is the place that claims to reproduce it.

        Why a node property and not a pair property. The end bracket
        (`SinusoidalSolver._ek_end_bracket_fields`) is O(1/a) and the two caps
        meeting at an interior node cancel each other; #246's per-PAIR
        eligibility truncates the extended set AT a node, leaving one cap
        uncancelled and the fill divergent as a → 0 (momwire#299). Scored per
        node the decision is observer-independent, hence symmetric, hence
        harmless to ‖G−Gᵀ‖ — which is the one property that made #246 refuse
        to transplant NEC's per-SOURCE-end gating in the first place.

        `mirror=True` scores the MIRRORED source geometry, whose labels are
        `_ek_axis_labels(geom, True)`'s source half. Mirroring is an isometry
        and commutes with the ξ = ±H end convention (the ξ = +H end of the
        mirrored segment is the mirror of the real one's), so the geometry's
        own neighbour tables carry over unchanged and only the labels differ.

        Cached per geometry OBJECT and per mirror flag, like the labels.
        """
        cached = self._cached_ek_bad_ends
        if cached is None or cached[0] is not geom:
            cached = (geom, {})
            self._cached_ek_bad_ends = cached
        hit = cached[1].get(mirror)
        if hit is not None:
            return hit

        _, group_src = self._ek_axis_labels(geom, mirror)
        n = geom["n_segs"]
        out = []
        for count, basis, seg in (
            (geom["nm_count"], geom["nm_basis"], geom["nm_seg"]),
            (geom["np_count"], geom["np_basis"], geom["np_seg"]),
        ):
            bad = np.asarray(count) >= 2  # K ≥ 3 at this node
            bad = bad | (group_src < 0)  # the never-extend marker
            if np.asarray(basis).size:
                split = group_src[basis] != group_src[seg]
                bad = bad.copy()
                bad[np.asarray(basis)[split]] = True
            out.append(np.ascontiguousarray(bad, dtype=bool).reshape(n))
        hit = (out[0], out[1])
        cached[1][mirror] = hit
        return hit

    def _ek_bracket_block(self, geom, k, ctx, corr, plan, m0, m1, eta=None):
        """One source block's share of momwire#299's end-bracket correction
        over the test segments `m0:m1`, ACCUMULATED into `corr` with weight
        `plan.scale` — never applied to the fill's own contributions, which
        `_ek_bracket_correction_tested` explains.

        `corr` is the band's three arrays, shaped (entries of m0:m1,
        `plan.cols`) — the (nnz, N) triple a fill block produces, narrowed to
        one band of test rows and to the source columns that can carry a bad
        end at all (momwire#355). Both narrowings are index bookkeeping: the
        rows are offset by the band's first entry and the columns go through
        `plan.col_of`, and every cell reached is the cell the whole-triple
        spelling reached.

        What is collected, and what is NOT
        ----------------------------------
        Writing e for a source end, the fill capped e with weight
        `eligible(m, n)` (#246's pair rule) and the node rule wants weight
        `pred(node(e))`, so the correction per (test m, source n, end e) is
        (pred − eligible)·bracket. Only the negative half of that is computed
        here, and the positive half is a documented identity rather than an
        omission:

        * pred FALSE, eligible — the divergent case, and the whole defect.
          Taking the cap back off leaves that node with no cap from any
          observer, which is what an EK-off fill has and what makes the caps
          cancel by KCL again.
        * pred TRUE, eligible — nothing to do, the fill already capped it.
        * pred TRUE, NOT eligible — the literal rule would ADD caps here (a
          cross-arm observer looking at the far arm's interior nodes). Every
          segment at a pred-true node carries one label, so a given observer is
          ineligible with ALL of them or none: the added caps come in a
          complete set at the node, share one ρ and one ζ (the node's segments
          share an axis and a radius, so the observer sees one geometry for all
          of them), and are weighted by the source basis's current and charge
          at the node — whose signed sums over the node's segments are zero by
          KCL and by charge continuity. The set therefore sums to zero in G
          before it is computed. At a free end the set has one member and no
          partner, but there the basis current vanishes, which kills the
          O(1/a) s·∂_ξW half outright and leaves the O(a²) s′·W half — the same
          term the fill already omits there for cross-arm observers, and the
          only place this differs from the literal rule at all.
        * pred FALSE, not eligible — nothing was capped, nothing to remove.

        Quadrature consistency is the one thing that has to be exact: the
        bracket is collected on the SAME test rule the cell was filled with —
        the uniform rule for a far cell, `_apply_near_correction`'s graded
        endpoint rule for a near one — because a rule mismatch would leave a
        quadrature-level difference of an O(1/a) quantity behind instead of
        cancelling it.
        """
        if not self.extended_kernel:
            return
        projector, src_c, src_t = plan.projector, plan.src_c, plan.src_t
        group_src = plan.group_src
        N, nq = ctx["N"], ctx["nq"]
        hh = ctx["hh"]
        a_seg = ctx["a_seg"]
        a_all = self._seg_radius(geom)
        seg_c, seg_t = geom["seg_centers"], geom["seg_tangents"]
        starts, counts = ctx["starts"], ctx["counts"]
        sigAC, B, sigC = ctx["sigAC"], ctx["B"], ctx["sigC"]
        corr_c, corr_s, corr_co = corr
        near_key = plan.near_key
        xg, wg = plan.xg, plan.wg
        scale = plan.scale
        # The band's observers, and the entry its first test segment owns —
        # `corr`'s row zero.
        g_obs = plan.group_obs[m0:m1]
        e_base = starts[m0]
        gq = np.arange(nq)

        for sign, bad in ((-1.0, plan.bad_lo), (+1.0, plan.bad_hi)):
            nb = np.flatnonzero(bad)
            if nb.size == 0:
                continue
            # Every pair the fill capped at this end: eligibility is #246's,
            # unchanged, because the cap being taken off is the one it added.
            elig = (g_obs[:, None] == group_src[nb][None, :]) & (g_obs[:, None] >= 0)
            m_sel, j_sel = np.nonzero(elig)
            if m_sel.size == 0:
                continue
            # `np.nonzero` is row-major, so the band's pairs come out in the
            # same (observer, then source) order the whole-matrix scan gave
            # them, and the bands run in ascending observer order — the pair
            # SEQUENCE is unbanded, only its cutting into calls is new.
            m_sel = m_sel + m0
            n_sel = nb[j_sel]
            is_near = np.isin(m_sel * N + n_sel, near_key)
            for graded in (False, True):
                keep = is_near if graded else ~is_near
                mm_g, nn_g = m_sel[keep], n_sel[keep]
                if mm_g.size == 0:
                    continue
                # Flatten (pair, support entry of its test segment) exactly as
                # `_apply_near_correction` does.
                cnt = counts[mm_g]
                cum = np.concatenate(([0], np.cumsum(cnt)))
                pair_of = np.repeat(np.arange(mm_g.size), cnt)
                entry_of = (
                    np.arange(cum[-1])
                    - np.repeat(cum[:-1], cnt)
                    + np.repeat(starts[mm_g], cnt)
                )
                for p0 in range(0, mm_g.size, _PAIR_BLOCK):
                    p1 = min(p0 + _PAIR_BLOCK, mm_g.size)
                    mi, ni = mm_g[p0:p1], nn_g[p0:p1]
                    if graded:
                        xi = hh[mi][:, None] * xg  # (P, G)
                        obs = (
                            seg_c[mi][:, None, :]
                            + xi[:, :, None] * seg_t[mi][:, None, :]
                        )
                    else:
                        # The uniform rule's observers, read back out of the
                        # context the far fill used rather than rebuilt.
                        xi = None
                        obs = ctx["obs_c"][mi[:, None] * nq + gq]
                    obs_t = seg_t[mi][:, None, :]
                    a_obs = (
                        self._uniform_radius if a_seg is None else a_seg[mi][:, None]
                    )
                    z, rho_eval, rho_vec, td, rho_proj = self._pair_geometry(
                        obs, obs_t, a_obs, src_c[ni][:, None, :], src_t[ni][:, None, :]
                    )
                    cm = self._ek_end_bracket_fields(
                        k,
                        np.broadcast_to(hh[ni][:, None], z.shape),
                        z,
                        rho_eval,
                        a_all[ni][:, None],
                        sign,
                        cos_shape="cos-1",
                        eta=eta,
                    )
                    cm["td"] = td
                    cm["rho_proj_factor"] = rho_proj
                    cm["rho_vec"] = rho_vec
                    cm["rho_eval"] = rho_eval
                    Phi = projector(cm, mi[:, None], ni[:, None])  # each (P, G)

                    e0, e1 = cum[p0], cum[p1]
                    ei = entry_of[e0:e1]
                    lp = pair_of[e0:e1] - p0
                    if graded:
                        fval = _basis_value(
                            sigAC[ei][:, None],
                            B[ei][:, None],
                            sigC[ei][:, None],
                            k,
                            xi[lp],
                        )
                        w = (wg[None, :] * hh[mi][lp][:, None]) * fval
                    else:
                        w = ctx["w_entry"][ei]
                    # `ei` stays the fill's own entry index — it reads the
                    # global coefficient and weight tables above — and only
                    # the WRITE moves onto the band's rows and the retained
                    # source columns.
                    row = ei - e_base
                    col = plan.col_of[ni[lp]]
                    # One (entry, source) cell per (pair, entry) and the pairs
                    # of a group are distinct, so the cells are distinct and
                    # this is an assignment-shaped update, not an accumulation
                    # — the two END groups are two separate statements, which
                    # is how a segment bad at BOTH ends pays twice.
                    for c_out, Ph in zip((corr_c, corr_s, corr_co), Phi):
                        c_out[row, col] += scale * np.einsum("eg,eg->e", w, Ph[lp])

    def _contact_ek_masks(self, geom, i, sgn, obs_seg):
        """momwire#292's masks under #246's PAIR rule.

        `SinusoidalSolver._contact_ek_masks` reads NEC's per-END IND code,
        which is the wrong question here for exactly the reason `_ek_pairs`
        gives: this fill extends a pair when observer and source share an
        axis, symmetrically, and never asks whether a current "continues
        through" an end. So the contact node's bracket is extended for
        precisely those observers whose test segment is coaxial-and-equal-
        radius with the contacting segment `i` — the same predicate, scored
        on the same labels, that decided the fill's own delta.

        The two sides come apart here in a way they cannot on the
        point-matched solver. The free-space block scores against the REAL
        sources and the image block against the MIRRORED ones
        (`_ek_axis_labels(mirror=True)`), so a vertical wire standing on the
        plane is eligible on both — it maps onto its own axis — while a
        SLANTED contact is coaxial with itself and not with its image, and
        gets the extended bracket on the real half of the residual only.
        `_contact_charge_ek_delta` takes the two masks separately for that
        case; `sgn` plays no part, both ends of a segment carrying the same
        axis label.

        `None` when EK is off, which keeps #282's arithmetic untouched.
        """
        if not self.extended_kernel:
            return None
        group_obs, group_src = self._ek_axis_labels(geom, False)
        real = (group_obs[obs_seg] == group_src[i]) & (group_obs[obs_seg] >= 0)
        group_obs_m, group_src_m = self._ek_axis_labels(geom, True)
        img = (group_obs_m[obs_seg] == group_src_m[i]) & (group_obs_m[obs_seg] >= 0)
        if not (real.any() or img.any()):
            return None
        return real, img

    def _tested_contribs(
        self,
        geom,
        k,
        ctx,
        projector,
        src_c=None,
        src_t=None,
        mirror=False,
        subtract_into=None,
        eta=None,
        cls=None,
        near=None,
    ):
        """Test-integrate one source block: (contrib_const, sin, cos−1), each
        (nnz, N) — the folded shape set (#203/#205), which is what the field
        kernel is asked for (`cos_shape="cos-1"`).

        `cls` (a `_ClassView`, momwire#1224) restricts the block to one mixed
        deck pair class: `ctx` is then the class's sub-context, `src_c` /
        `src_t` stay the WHOLE deck's (free or mirrored) sources, and the
        block comes back (n_class_entries, pad + n_class_segs) — the class's
        own quadrant and nothing else, where the unrestricted block was the
        whole plane with the quadrant cut out of it afterwards. Every cell is
        the same float64 sequence it was (see `_class_view`). A view with a
        restricted source list is served on the fused C++ path only; the
        caller checks `_class_fill_serves` first. An observer band of the
        fused banded fill (`_sub_view`) is the same object with the whole
        deck's sources, and the numpy loop serves that one too: it walks its
        own test segments and indexes per-pair tables by `m_of_obs`, which a
        view carries in full-deck coordinates.

        `near` (a `_NearCells`, momwire#1224) is this block's graded near
        values, computed once per fill over the whole deck by
        `_near_cells`; the block then only places the cells it holds, by the
        same assignment (or, folding, the same subtraction) the inline
        correction performs. Without it the correction is computed here, as
        it always was.

        `subtract_into` is the ground path's residency lever (momwire#332):
        given a triple, this block is SUBTRACTED into it entry by entry and
        nothing is returned, instead of a parallel triple coming back for the
        caller to difference. Minus one is the only weight any caller needs —
        it is the ground's single global minus sign — and the two spellings
        are the same float64 subtraction per matrix entry, so the fold is
        bit-exact rather than a reassociation. The Sommerfeld ground's C2 is
        NOT a second weight here: `c2·img − rem` has to stay associated as it
        is written to keep that (see `_fold_ground_block`).

        `src_c` / `src_t` are the source geometry the field evaluator sees —
        the geometry's own segments for the free-space block, the mirrored
        ones for a ground image block. `mirror` says which of the two this is,
        and is consulted only by the extended kernel's pair rule
        (`_ek_pairs`), which has to score eligibility against the MIRRORED
        source geometry on an image block.

        `projector(cm, m_idx, n_idx)` turns the unprojected Eqs 76-79
        component tables into the three tangential field tables. `m_idx` /
        `n_idx` are broadcastable index arrays naming the (test segment,
        source segment) each entry of `cm` belongs to, so a projector can
        look up per-segment-pair tables regardless of how the caller paired
        things: (M, 1) against (1, N) for the full block here, (P, 1) against
        (P, 1) for the near-pair path's one-to-one pairing.

        Two-tier quadrature (M2): a shared uniform rule for the far pairs,
        overwritten per near pair by the endpoint-graded rule.

        The far half runs in C++ when the accelerator is present AND the
        projector is the plain tangential one (`_far_fill_accel`) — that is
        the free-space block and the PEC image block. The near correction is
        O(N) pairs and stays here regardless.

        With the extended kernel on (momwire#246) the far half falls back to
        the numpy loop unless the C++ EK twin is present: the reduced far fill
        takes no eligibility payload, so routing an EK-on block through it
        would drop the delta silently rather than fail.

        Both halves honour `subtract_into` as they fill, so a grounded block
        holds ONE triple whichever backend serves it — the destination alone
        (momwire#356). The C++ fill is handed the destination as its `out=`
        with `scale=-1`, which folds each finished entry on as
        `dst += (−1)·value`; the numpy loop accumulates its blocks into the
        destination directly. Neither is a reassociation of the differenced
        spelling: IEEE addition of an exactly-negated operand IS the
        subtraction, and the reduction each entry's value comes out of is the
        same sum in the same order either way.

        The near correction is what makes the accelerated fold non-trivial.
        It OVERWRITES its cells rather than accumulating, so on the numpy path
        it has to run FIRST (`sub=True`) and hand back the cells the far half
        must then skip. The fused kernel cannot skip cells — it is one sweep
        over every (entry, source) pair — so this path instead SAVES those
        cells' values across the fill and puts them back: an exact copy out
        and an exact copy in, leaving `free − graded` where the pre-#356
        spelling left `free − graded` and the far value nowhere.
        """
        N = ctx["N"]
        nq = ctx["nq"]
        src_c = geom["seg_centers"] if src_c is None else src_c
        src_t = geom["seg_tangents"] if src_t is None else src_t

        # An in-medium (complex) k reaches the complex twin when the build
        # carries it (momwire#980 step E) and the numpy path when it does not
        # — `eta` too, since eta_m = sqrt(mu/(eps*eps_tilde)) goes complex
        # with k_m and the REAL kernel's signature takes a double for it.
        # There is no complex EK twin: the extended kernel's Bessel-polynomial
        # split assumes jkR is purely imaginary, so an EK solve in the medium
        # stays on numpy.
        eta = self._fill_eta(k, eta)
        if self._fused_fill_serves(k, eta, projector):
            # Guarded at the call site as well as inside, like every other EK
            # entry point here: G-B4's counter gate is that an EK-off solve
            # does not so much as ENTER this code.
            ek_pairs = (
                self._ek_far_labels(geom, mirror) if self.extended_kernel else None
            )
            # A class block fills against its own source columns only; the
            # near correction below still takes the whole deck's sources and
            # maps its cells into the class block itself.
            if cls is None:
                far_c, far_t = src_c, src_t
            else:
                far_c, far_t = src_c[cls.src_idx], src_t[cls.src_idx]
                if ek_pairs is not None:
                    # The labels at the view's own axes: one per TEST
                    # segment it holds, one per source it fills against.
                    ek_pairs = ek_pairs._replace(
                        src_a=np.asarray(ek_pairs.src_a)[cls.src_idx],
                        group_obs=np.ascontiguousarray(ek_pairs.group_obs[cls.segs]),
                        group_src=np.ascontiguousarray(ek_pairs.group_src[cls.src_idx]),
                    )
            if subtract_into is None:
                contribs = self._far_fill_accel(
                    k, ctx, far_c, far_t, ek=ek_pairs, eta=eta
                )
                if self.near_correction:
                    self._near_correct(
                        geom,
                        k,
                        ctx,
                        contribs,
                        projector,
                        src_c,
                        src_t,
                        mirror,
                        eta=eta,
                        cls=cls,
                        near=near,
                    )
                return contribs
            near_cells = (
                self._near_correct(
                    geom,
                    k,
                    ctx,
                    subtract_into,
                    projector,
                    src_c,
                    src_t,
                    mirror,
                    sub=True,
                    eta=eta,
                    cls=cls,
                    near=near,
                )
                if self.near_correction
                else None
            )
            # O(N) pairs' worth of cells, saved as flat gathers — the whole
            # point of the fold is that nothing (nnz, N) is allocated here.
            held = (
                [dest[near_cells].copy() for dest in subtract_into]
                if near_cells is not None
                else None
            )
            self._far_fill_accel(
                k,
                ctx,
                far_c,
                far_t,
                ek=ek_pairs,
                out=subtract_into,
                scale=-1.0,
                eta=eta,
            )
            if held is not None:
                for dest, saved in zip(subtract_into, held):
                    dest[near_cells] = saved
            return None

        n_src = src_c.shape[0]
        if cls is not None and not (
            cls.pad == 0
            and cls.src_idx.size == n_src
            and np.array_equal(cls.src_idx, np.arange(n_src))
        ):
            # Deliberately not a silent fallback: the numpy loop below indexes
            # sources, projector tables and EK labels in whole-deck
            # coordinates, so a block restricted on its SOURCE side reaching
            # it would be filled over the wrong columns. `_class_fill_serves`
            # is the caller's check. A view that keeps the whole deck's
            # sources (an observer band) is served: its test side is indexed
            # through `m_of_obs`, which the view carries in full-deck terms.
            raise AssertionError(
                "a source-restricted block reached the numpy far fill; "
                "the mixed fill should have taken the whole-plane "
                "class fill (momwire#1224)"
            )
        # Blocked over test segments (#194): identical arithmetic per matrix
        # entry, but the kernel's source-quadrature scratch is (rows·nq, N,
        # n_qp_const) per block instead of (N·nq, N, n_qp_const) once. `N` is
        # the TEST segment count and `n_src` the source count: the same
        # number except on an observer band's sub-context.
        starts = ctx["starts"]
        nnz = ctx["w_entry"].shape[0]
        a_obs = ctx["a_obs"]
        n_idx = np.arange(n_src)[None, :]
        if subtract_into is None:
            contribs = tuple(
                np.zeros((nnz, n_src), dtype=np.complex128) for _ in range(3)
            )
            near_cells = None
        else:
            # Folding as we fill, the near correction can no longer run LAST:
            # it overwrites its cells, and what it would overwrite here is the
            # caller's free-space value rather than this block's own uniform
            # one. So it runs FIRST — nothing it computes depends on the far
            # half — subtracting the graded value straight off the free-space
            # one, and hands back the cells it owns so the far loop can leave
            # them alone. Subtracting zero there is exact, so the far half's
            # arithmetic is unchanged on every other cell and absent on these.
            contribs = subtract_into
            near_cells = (
                self._near_correct(
                    geom,
                    k,
                    ctx,
                    contribs,
                    projector,
                    src_c,
                    src_t,
                    mirror,
                    sub=True,
                    eta=eta,
                    cls=cls,
                    near=near,
                )
                if self.near_correction
                else None
            )
        blk = _fill_block(n_src, nq, self.n_qp_const)
        for m0 in range(0, N, blk):
            m1 = min(m0 + blk, N)
            o0, o1 = m0 * nq, m1 * nq
            cm = self._field_components_bcast(
                k,
                obs_c=ctx["obs_c"][o0:o1, None, :],  # (rows·nq, 1, 3)
                obs_t=ctx["obs_t"][o0:o1, None, :],
                a=a_obs[o0:o1] if isinstance(a_obs, np.ndarray) else a_obs,
                src_c=src_c[None, :, :],
                src_t=src_t[None, :, :],
                src_hh=ctx["hh"][None, :],
                cos_shape="cos-1",
                eta=eta,
                # Per BLOCK, like everything else in this loop: the mask is
                # (rows·nq, N) and would otherwise be the one array in the
                # fill that scales with the whole matrix. Eligibility is a
                # per-pair property, so the blocked masks are slices of the
                # unblocked one and the arithmetic per entry is unchanged.
                ek=(
                    self._ek_pairs(geom, ctx["m_of_obs"][o0:o1, None], n_idx, mirror)
                    if self.extended_kernel
                    else None
                ),
            )
            Phi = projector(cm, ctx["m_of_obs"][o0:o1, None], n_idx)
            del cm  # drop the kernel tables before the reduction allocates
            e0, e1 = starts[m0], nnz if m1 == N else starts[m1]
            w = ctx["w_entry"][e0:e1]
            m_loc = ctx["m_of_entry"][e0:e1] - m0
            if near_cells is not None:
                # This block's share of the cells the near correction already
                # owns, in block-local entry coordinates.
                held = (near_cells[0] >= e0) & (near_cells[0] < e1)
                held_e = near_cells[0][held] - e0
                held_n = near_cells[1][held]
            for c_out, P in zip(contribs, Phi):
                rows = self._tested_contrib_rows(
                    w, m_loc, nq, P.reshape(m1 - m0, nq, n_src)
                )
                if subtract_into is None:
                    c_out[e0:e1] = rows
                    continue
                if near_cells is not None:
                    rows[held_e, held_n] = 0.0
                np.subtract(c_out[e0:e1], rows, out=c_out[e0:e1])

        if subtract_into is not None:
            return None
        if self.near_correction:
            self._near_correct(
                geom,
                k,
                ctx,
                contribs,
                projector,
                src_c,
                src_t,
                mirror,
                eta=eta,
                cls=cls,
                near=near,
            )
        return contribs

    def _fused_fill_serves(self, k, eta, projector):
        """Whether `_tested_contribs` fills a block at (k, eta) through
        `projector` on the fused C++ far fill rather than the numpy loop —
        its own predicate, named so the banded fill (momwire#1224) can ask
        it before it cuts its bands."""
        eta = self._fill_eta(k, eta)
        in_medium = np.iscomplexobj(k) or np.iscomplexobj(eta)
        return bool(
            _HAVE_GALERKIN_FAR_FILL
            and (
                not in_medium
                or (_HAVE_GALERKIN_FAR_FILL_CPLX and not self.extended_kernel)
            )
            and projector is _plain_projection
            and (not self.extended_kernel or _HAVE_GALERKIN_FAR_FILL_EK)
        )

    def _near_correct(
        self,
        geom,
        k,
        ctx,
        contribs,
        projector,
        src_c,
        src_t,
        mirror=False,
        sub=False,
        eta=None,
        cls=None,
        near=None,
    ):
        """`_apply_near_correction`, or — given `near`, the whole deck's
        precomputed `_NearCells` for this block (momwire#1224) — the same
        writes from those values: each cell the block holds is ASSIGNED its
        graded value (or, `sub`, has it SUBTRACTED), and with `sub` the cells
        written come back in block coordinates, exactly as the inline
        correction returns them. A cell's value is the same float64 either
        way: `_near_cells` runs the inline correction's own loop and keeps
        what it would have written. Each cell has one near pair (the entry
        fixes the test segment), so the order the cells are placed in is
        immaterial."""
        if near is None:
            return self._apply_near_correction(
                geom,
                k,
                ctx,
                contribs,
                projector,
                src_c,
                src_t,
                mirror,
                sub=sub,
                eta=eta,
                cls=cls,
            )
        if cls is None:
            row, col, sel = near.entry, near.seg, slice(None)
        else:
            row = cls.row_of_entry[near.entry]
            col = cls.col_of_seg[near.seg]
            sel = (row >= 0) & (col >= 0)
            row, col = row[sel], col[sel]
        for contrib, val in zip(contribs, near.vals):
            if sub:
                contrib[row, col] -= val[sel]
            else:
                contrib[row, col] = val[sel]
        return (row, col) if sub else None

    def _far_fill_accel(
        self, k, ctx, src_c, src_t, ek=None, out=None, scale=1.0, eta=None
    ):
        """C++ far fill for the PLAIN projection: kernel and test reduction
        fused, (contrib_const, sin, cos−1) each (nnz, N) — the same three arrays
        the numpy loop above builds (#194).

        The fusion is what buys the speed AND the memory: the numpy path has
        to materialize the (rows·nq, N) field tables — with the
        (rows·nq, N, n_qp_const) source-quadrature scratch under them, which is
        why it blocks at all — before it can contract the test-quadrature axis
        away, while the kernel contracts each observer's row into `contribs`
        as it computes it. So there is no block loop here and
        `_FILL_WORKSPACE_BYTES` does not apply.

        The reduction is accumulated one test-quadrature node at a time, in
        the same order as `_tested_contrib_rows`; the field arithmetic per
        matrix entry is the point-matched solver's `sinusoidal_field_tensor`
        verbatim. Agreement with the numpy reference is therefore
        reassociation-level, not algorithmic (~1e-15 relative on G).

        Only the plain projector's blocks come here — free space and the PEC
        image, i.e. exactly the grounds whose `FieldGround.projector` is
        `_plain_projection` (that identity is the gate, in `_tested_contribs`).
        A Fresnel-weighted image's per-pair tables and the Sommerfeld
        remainder keep the numpy path, so their callers still see the blocked
        loop above. Nothing here computes a dyad, which is why this solver has
        no analogue of the point-matched fused refl kernels and no use for
        `FieldGround.standard_fresnel`: a coefficient-modified ground reaches
        the same numpy projector the shipped one does.

        `ek` picks the entry point (momwire#246 unit C) and is an
        `_EKFarLabels`, the pair rule's group labels rather than its evaluated
        mask (momwire#358). With no payload the call is the pre-#246 one, byte
        for byte — the reduced symbol is compiled from the shared
        implementation with the delta's code absent, so an EK-off fill cannot
        pay for the option or be perturbed by it. With one,
        `sinusoidal_galerkin_far_fill_ek` scores eligibility per pair from the
        labels and adds the folded delta on the eligible ones inside the same
        fused sweep, which is what keeps the extended kernel on the
        accelerated path at all: the numpy block loop it replaces measured
        25-30x slower on the N=101...401 dipole.
        The toggle itself costs ~6x there, and that is the honest worst case
        — every pair of a straight wire is coaxial, so every pair takes the
        delta's 16-node quadrature.

        `out` is the ground fold's residency lever (momwire#356): a triple the
        caller already holds, which the kernel writes into as
        `out += scale·value` instead of allocating and returning three arrays
        of its own. Without it a grounded accelerated block floors at two
        triples live — the destination plus the kernel's return — where the
        numpy path has always held one.

        `scale` multiplies from the LEFT (`sinusoidal.py`'s C2 convention: a
        complex128 multiply evaluates the imaginary part as
        `x.re*y.im + x.im*y.re`, so the operand order moves the last bit) and
        is applied ONCE per entry, to the finished test-quadrature sum — not
        to each node's contribution, which would reassociate the reduction.
        The ground fold's own weight is −1, a real scale, and there the fold
        is `np.subtract(out, value, out=out)` to the bit: `a + (−b)` is `a − b`
        in IEEE arithmetic, signed zeros included. `scale` without `out` is
        inert — there is no destination for it to weight.
        """
        if ek is not None and not _HAVE_GALERKIN_FAR_FILL_EK:
            # Deliberately not silent: the caller admits an EK payload here
            # only when `_HAVE_GALERKIN_FAR_FILL_EK` says a C++ twin that can
            # consume it exists (momwire#246 unit C). A half-landed twin
            # therefore fails loudly instead of dropping the extended kernel's
            # delta on the floor.
            raise NotImplementedError(
                "the fused C++ far fill has no extended-kernel twin in this "
                "build (momwire#246 unit C); _tested_contribs is supposed to "
                "take the numpy path while _HAVE_GALERKIN_FAR_FILL_EK is False"
            )
        n_obs = ctx["obs_c"].shape[0]
        a_obs = ctx["a_obs"]
        # The kernel takes one radius per OBSERVER (which is per test segment,
        # repeated over its quadrature nodes), so mixed per-wire radii need no
        # run-splitting here — unlike the point-matched scalar-`a` kernel.
        a_flat = (
            np.ascontiguousarray(np.reshape(a_obs, n_obs), dtype=np.float64)
            if isinstance(a_obs, np.ndarray)
            else np.full(n_obs, float(a_obs))
        )
        gx, gw = self._leggauss_cached(self.n_qp_const)
        # Which entry point this call takes. Decided from the VALUES rather
        # than from the solver's ground, because `k` arrives per block.
        eta = self._fill_eta(k, eta)
        _cplx_fill = np.iscomplexobj(k) or np.iscomplexobj(eta)
        if _cplx_fill and not _HAVE_GALERKIN_FAR_FILL_CPLX:
            raise RuntimeError(
                "complex-k far fill requested from a build without "
                "sinusoidal_galerkin_far_fill_cplx; the caller should have "
                "taken the numpy path"
            )
        args = (
            np.ascontiguousarray(ctx["obs_c"], dtype=np.float64),
            np.ascontiguousarray(ctx["obs_t"], dtype=np.float64),
            a_flat,
            np.ascontiguousarray(src_c, dtype=np.float64),
            np.ascontiguousarray(src_t, dtype=np.float64),
            np.ascontiguousarray(ctx["hh"], dtype=np.float64),
            # Complex in the medium, and the SCALAR CAST is the guard: at a
            # real k `float()` would raise on a complex value rather than
            # truncate it, which is the silent-truncation class momwire#980
            # step A exists to keep out.
            complex(k) if _cplx_fill else float(k),
            complex(eta) if _cplx_fill else float(eta),
            np.ascontiguousarray(gx, dtype=np.float64),
            np.ascontiguousarray(gw, dtype=np.float64),
            np.ascontiguousarray(ctx["w_entry"], dtype=np.complex128),
            np.ascontiguousarray(ctx["starts"], dtype=np.int64),
        )
        # `out`/`scale` ride as KEYWORDS, behind the eligibility payload's
        # positional slots. That is what let momwire#358 swap that payload —
        # the (n_obs, N) mask out, the two (N,) label arrays in — as a change
        # to the EK call's positionals alone, never touching the fold.
        fold = {} if out is None else {"out": tuple(out), "scale": complex(scale)}
        if ek is None:
            if _cplx_fill:
                return _acc.sinusoidal_galerkin_far_fill_cplx(
                    *args, self._cancel_flag, **fold
                )
            if _HAVE_SG_REAL_STAGED and not _SG_REAL_STAGED:
                fold["reference"] = True
            return _acc.sinusoidal_galerkin_far_fill(*args, self._cancel_flag, **fold)
        # The EK twin takes the payload at the shapes the kernel indexes: one
        # radius per SOURCE segment, the pair rule's group labels — one per
        # TEST segment and one per source segment — and the delta quadrature's
        # composite rule, built HERE rather than in C++ so the two backends
        # integrate against the same nodes by construction rather than by
        # transcription. `n_panels` rides along from the payload, so the far
        # tier's single panel is the payload's decision and not the kernel's.
        #
        # The labels are what the eligibility argument USED to be an (n_obs,
        # n_src) bool mask of (momwire#358). Nothing of the fill's own shape is
        # built in Python any more: the kernel scores `g_obs[m] == g_src[n] and
        # g_obs[m] >= 0` as it reaches each pair, which is `_ek_pairs`' formula
        # unchanged and the numpy block loop's own per-block derivation moved
        # one level in. What that mask cost was resident input, not a
        # transient — 11.5 MB at N = 1200, and the whole of what a grounded
        # accelerated block still held over its destination triple after #356.
        n_src = args[3].shape[0]
        n_test = args[11].shape[0] - 1
        ek_gx, ek_gw = self._ek_delta_rule(_N_QP_EK_DELTA, ek.n_panels)
        # Producer contract, not a conversion (momwire#332 unit E, #318's
        # "dead wrapper" pattern repeated, now on arrays 1e3 times smaller):
        # `_ek_axis_labels` returns owned, C-contiguous int64 arrays straight
        # out of `_ek_axis_groups`, so the `ascontiguousarray` these would
        # otherwise be wrapped in never copies. The assert pins that contract
        # and vanishes under -O; the pybind11 `c_style | forcecast` array_t on
        # the C++ side stays the final safety net if it is ever wrong, and the
        # kernel checks both lengths itself.
        assert (
            ek.group_obs.shape == (n_test,)
            and ek.group_src.shape == (n_src,)
            and ek.group_obs.dtype == np.int64
            and ek.group_src.dtype == np.int64
        ), (
            f"group labels must be ({n_test},)/({n_src},) int64, got "
            f"{ek.group_obs.shape} {ek.group_obs.dtype} / "
            f"{ek.group_src.shape} {ek.group_src.dtype}"
        )
        return _acc.sinusoidal_galerkin_far_fill_ek(
            *args,
            np.ascontiguousarray(np.reshape(ek.src_a, n_src), dtype=np.float64),
            ek.group_obs,
            ek.group_src,
            np.ascontiguousarray(ek_gx, dtype=np.float64),
            np.ascontiguousarray(ek_gw, dtype=np.float64),
            self._cancel_flag,
            **fold,
        )

    def _image_projector(self, geom, fg):
        """`fg`'s image projector, taken off THIS solver's per-geometry
        specular-table cache.

        The whole of what unit 3 left behind of `_refl_projection`, and it is
        schedule rather than physics: which projector a ground takes is
        `FieldGround.projector`'s ternary, and the dyad itself is
        `_field_ground.PairWeights.project` (unit 1's single spelling).
        What is local is only WHEN the tables are built — once per geometry
        and cached (`_image_refl_prep`), where the point-matched fill builds
        them per observer band and throws them away. The supplier is a
        callable so an unweighted ground never triggers the O(N²) build at
        all.

        The weights are read at the per-SEGMENT-PAIR pairing each block
        names, off tables built once for the whole geometry. That is
        deliberate and it is what NEC's IPERF=0 model actually says: ρ_v/ρ_h
        are evaluated once at the midpoint-to-image-midpoint specular angle
        and held **constant over the segment pair**. The test quadrature
        integrates the field, not the reflection coefficient.

        The alternative — re-deriving the specular ray at each test
        quadrature point — was implemented and rejected on measurement: it
        makes the weight a function of where on the test segment you are,
        which no longer has a partner term in the transposed entry, and the
        Galerkin matrix loses reciprocity (‖G−Gᵀ‖/‖G‖ = 3.0e-9 on the
        h=0.1λ dipole, 2.3e-9 on the L-shape, and *insensitive to
        quadrature refinement* — 3.00e-9 at n_qp_test 8, 16 and 32, i.e.
        structural). The pair-constant form sits at the free-space floor
        instead. Since the observers are points ON test segment m,
        `tm_p = t_m·p̂` from the segment tangents is exact for them, not an
        approximation.
        """
        return fg.projector(lambda: self._image_refl_prep(geom))

    def _fold_ground_block(
        self,
        geom,
        k,
        ctx,
        contribs,
        fg,
        obs_mask=None,
        src_cols=None,
        eta=None,
        cls=None,
        near=None,
    ):
        """The ground sub-assembly, tested exactly like the free-space block
        and SUBTRACTED from it in place — the same single global minus sign
        the point-matched `_assemble_Z` uses (the image current + image charge
        sign flips reduce to it).

        Folded rather than returned (momwire#332). The differenced spelling
        built a whole parallel triple and then a whole third one for
        `free − ground`, so the peak under any ground was 3 × 16·nnz·N with
        the fill's own scratch on top; the fold reaches the same entries with
        the same float64 operations and holds at most the destination plus
        whatever one block materializes.

        Since momwire#356 that holds on the ACCELERATED path too: the fused
        C++ fill takes the destination as its `out=` and folds into it with
        `scale=-1`, where before it allocated and returned a triple of its own
        for `_tested_contribs` to difference off afterwards. Measured on the
        N = 300 bend, extended kernel on, this block's own peak goes 1.18 →
        0.23 triples over a PEC image, and 1.18 → 0.23 at N = 400.

        The SOMMERFELD composition is the one that does not move, and `out=`
        cannot move it: `c2·img − rem` has to stay associated (below), so the
        image block's whole triple must exist before the fold can start —
        1.73 triples at N = 400, the same number unit D left. Folding it needs
        the fill BANDED over observers so that image, remainder and fold meet
        one band at a time, which is momwire#356's option 2 and not this.

        What this method is, since momwire#397 unit 3, is that schedule and
        nothing else. `fg` is the fill's one `_field_ground.FieldGround`, and
        every ground DECISION is read off it: the mirror map
        (`image_sources`), the per-pair weight (`projector`), the image
        coefficient, and whether the block may ride `subtract_into`'s single
        minus (`mode == "fold"`) or has to be composed first
        (`mode == "compose"`). Nothing here reads `ground_z`, `ground_eps` or
        `ground_model`, so a ground this file has never heard of — the
        radial-wire screen, which is a coefficient change one level down in
        `_ground_refl` — folds through the branch it already has.

        Reuse, per ground, of the evaluator the point-matched solver already
        validated against its own references — now read off `fg` rather than
        chosen here:

        * PEC image — the mirrored-source Eqs 76-79 build, plain projection;
        * `ground_eps` refl-coef — the same build with the Fresnel dyad
          (`PairWeights.project`, via `_image_projector`);
        * `ground_model="sommerfeld"` — NEC's decomposition, C2·(PEC image)
          minus the smooth interpolated remainder, so that the subtraction
          reproduces `Phi_free − C2·Phi_img + S`. This is the whole membership
          of `mode == "compose"` today.

        Every image block passes `mirror=True`, which is the extended kernel's
        only ground-specific decision: eligibility is scored between the real
        test segments and the MIRRORED sources, so a wire standing on the plane
        extends against its own image and a wire parallel to it does not. The
        reflection-coefficient block needs nothing further — its Fresnel dyad
        is a per-segment-pair weight applied to the field tables AFTER the
        kernel, so the delta rides through it linearly, exactly as the reduced
        field does. Neither does the Sommerfeld block's C2 scaling, which is
        one complex number times the whole image contribution (momwire#287).
        The Sommerfeld REMAINDER, the second half of that model, stays reduced
        on the measured argument in the class docstring.

        With `cls` (momwire#1224) `ctx` / `contribs` are one mixed-deck pair
        class's sub-context and block, the image is filled over that class's
        mirrored sources only, and the remainder replays at the class's
        observers into the class's own columns — the masked replay below,
        unchanged, on a context that holds nothing else. `obs_mask` /
        `src_cols` are then derived here rather than passed. An observer band
        of the fused banded fill is the same kind of view (`_sub_view`); on a
        single-medium deck it keeps the whole source axis, so its remainder is
        the unmasked replay at the band's observers. `near` is the image
        block's precomputed `_NearCells` (see `_tested_contribs`).
        """
        src_c_img, src_t_img = fg.image_sources()
        projector = self._image_projector(geom, fg)
        if fg.mode == "fold":
            self._tested_contribs(
                geom,
                k,
                ctx,
                projector,
                src_c_img,
                src_t_img,
                mirror=True,
                subtract_into=contribs,
                eta=eta,
                cls=cls,
                near=near,
            )
            return

        img = self._tested_contribs(
            geom,
            k,
            ctx,
            projector,
            src_c_img,
            src_t_img,
            mirror=True,
            eta=eta,
            cls=cls,
            near=near,
        )
        if cls is not None:
            # A restricted block's remainder: at the block's own observers
            # (the whole sub-context), onto the columns its source axis
            # lands on — all of them when the remainder and the block share
            # the whole deck's source axis (an observer band of a
            # single-medium deck), which is the unmasked replay.
            src_cols = cls.rem_cols
            obs_mask = None
            if src_cols is not None:
                obs_mask = (
                    np.ones(ctx["N"], dtype=bool)
                    if cls.rem_obs is None
                    else cls.rem_obs
                )
        # `coef·img − rem` in place, the coefficient on the LEFT — the
        # point-matched band's spelling (`sinusoidal.py`), and the ground
        # object's own interface contract, for its reason: complex multiply
        # evaluates the imaginary part as x.re*y.im + x.im*y.re, so
        # `img *= coef` reorders that sum and moves the last bit. The two
        # terms are NOT distributed over the fold either — `free − (c2·img
        # − rem)` is not `(free − c2·img) + rem` in float64 — which is what
        # `mode == "compose"` declares, and why this block cannot ride
        # `subtract_into`'s single minus sign.
        #
        # The remainder is never a triple of its own (momwire#332 unit D):
        # it is subtracted off the SCALED image as each observer chunk
        # reduces, which is why the scaling runs over the whole image
        # first. Per entry the arithmetic is unchanged — coef·img, minus the
        # remainder, then the one fold — because both steps are elementwise
        # and the chunking only decides when each entry is reached.
        coef = fg.image_coefficient
        for a in img:
            np.multiply(coef, a, out=a)
        self._tested_sommerfeld_remainder(
            ctx, fg, img, obs_mask=obs_mask, src_cols=src_cols
        )
        for dest, a in zip(contribs, img):
            np.subtract(dest, a, out=dest)

    def _tested_sommerfeld_remainder(
        self, ctx, fg, subtract_from, obs_mask=None, src_cols=None
    ):
        """Test-integrate the smooth Sommerfeld remainder tensor, SUBTRACTING
        it from `subtract_from` (the C2-scaled image triple) as it goes.

        Streamed rather than returned (momwire#332 unit D). The evaluator's
        tensor is (3, N·n_qp_test, N) here — `n_qp_test` = 8 times the matrix,
        several times the (nnz, N) triple it reduces to, and on the measured
        fill it was the single largest thing the grounded assembly held. The
        evaluator already walked its observers in chunks, so this reduces each
        chunk to its (nnz_chunk, N) rows and folds them the moment they exist;
        the tensor never has to exist whole.

        Chunk boundaries are aligned to whole test segments (`row_group=nq`),
        which is what makes the streamed result BIT-EQUAL rather than merely
        equivalent: a test entry's reduction is a sum over that entry's nq
        quadrature nodes, and `_tested_contrib_rows` accumulates it one node at
        a time in node order. Split an entry across two chunks and the two
        halves would have to meet as a partial sum — a reassociation of the
        same products (#203/#205 territory). Whole segments per chunk means
        every entry's nodes stay together and each entry's sum is the same
        float64 sequence it always was. The fold that follows is elementwise,
        so the chunking decides only WHEN an entry is reached.

        The remainder evaluator is the ground's — `fg.remainder("cos-1")`,
        whose prepare half is the point-matched solver's and whose replay
        forwards this method's `consume` and `row_group` untouched (momwire
        #397 unit 3). Which is to say the DECISION that this ground has a
        remainder at all, and the source shape it is built on, are the
        object's; the streaming schedule below, and the chunk alignment that
        makes it bit-equal, are this method's and stay here. `"cos-1"` is the
        folded shape set the free-space triple this is subtracted from was
        built on (#205), so both halves are on the same shapes.

        Observers are the test-quadrature points rather than the segment
        centres. No near-pair correction:
        the remainder kernel lives on the distance to the IMAGE point, which
        stays smooth even where a wire touches the plane (r₁ → 0 has a finite
        limit the grid carries), so the width-`a` endpoint spike the graded
        rule exists for is absent here. The residual that IS left on a
        ground-touching wire is this block's own SOURCE-side rule
        (`n_qp_sommerfeld`), which
        `test_g4_sommerfeld_symmetry_near_the_plane_is_source_quadrature_limited`
        identifies by showing that refining the test rule does not move it
        while refining the source rule drives it to the free-space floor.

        No extended kernel either, and that one is a DECISION rather than a
        consequence (momwire#287). With `extended_kernel=True` the C2-scaled
        image half of this ground carries the delta like any other image
        block, while this remainder keeps the reduced-kernel field — the
        source stays a filament on the axis instead of a tube of radius `a`.
        EK is an O((a/R)²) correction and this evaluator's R is the IMAGE
        distance r₁ ≥ 2h, so the term left out is O((a/2h)²) and measures
        ≤ 4e-5 relative on |Z| for any wire clear of the plane, 3.5-4.5e-3 at
        a ground contact where that estimate degenerates. The class docstring
        carries the table and G-S1 in `tests/test_extended_kernel_galerkin.py`
        gates it, by building the extended remainder outright (this same field
        azimuthally averaged over the source tube) and re-solving.
        """
        N = ctx["N"]
        nq = ctx["nq"]
        starts = ctx["starts"]
        w_entry = ctx["w_entry"]
        m_of_entry = ctx["m_of_entry"]
        nnz = w_entry.shape[0]
        starts_pad = np.concatenate((np.asarray(starts), [nnz]))

        def _reduce(i0, i1, block):
            # Observer rows i0:i1 are test segments m0:m1 whole, so the
            # entries they carry are the contiguous run starts[m0]:starts[m1]
            # — the same slicing the blocked fill uses (`_tested_contribs`),
            # and `_tested_contrib_rows`' m-index is block-local there too.
            m0, m1 = i0 // nq, i1 // nq
            e0 = starts[m0]
            e1 = nnz if m1 == N else starts[m1]
            w = w_entry[e0:e1]
            m_loc = m_of_entry[e0:e1] - m0
            for dest, s in zip(subtract_from, block):
                # The source axis is the remainder's own (the whole deck's),
                # which an observer band's sub-context (N = its test
                # segments) does not share — so read it off the block.
                rows = self._tested_contrib_rows(
                    w, m_loc, nq, s.reshape(m1 - m0, nq, s.shape[-1])
                )
                np.subtract(dest[e0:e1], rows, out=dest[e0:e1])

        if obs_mask is None:
            with _OverlappedFold() as overlap:
                fg.remainder("cos-1").replay(
                    obs_centers=ctx["obs_c"],
                    obs_tangents=ctx["obs_t"],
                    consume=overlap.wrap(_reduce),
                    row_group=nq,
                )
            return
        # A CLASS block (momwire#980 D2): this ground models ONE medium, so
        # its remainder is only defined at that medium's observers — handing
        # it the whole deck's makes the below family refuse ("needs BOTH
        # endpoints strictly below ground_z"), correctly. `replay` already
        # takes an explicit observer set; this is a parameter, not a new
        # path. The mask is per TEST SEGMENT, so the rows stay whole and
        # `row_group=nq` still means what it means.
        seg_keep = np.nonzero(obs_mask)[0]
        obs_rows = (seg_keep[:, None] * nq + np.arange(nq)[None, :]).ravel()
        sub_starts = np.concatenate(([0], np.cumsum(np.diff(starts_pad)[seg_keep])))

        def _reduce_masked(i0, i1, block):
            # `i0:i1` index the KEPT observers; map back to whole test
            # segments of the full deck and reduce into their own entries.
            #
            # One reduction per chunk, not one per kept segment (momwire#1224:
            # the per-segment loop was ~4200 small calls and ~0.6 s of the SG
            # buried x8 fill). Each entry's row is still its own nq-term sum
            # in node order -- `_tested_contrib_rows` never mixes entries --
            # and the subtraction is elementwise, so batching moves no bits.
            m0, m1 = i0 // nq, i1 // nq
            segs = seg_keep[m0:m1]
            e_lo = starts_pad[segs]
            counts = starts_pad[segs + 1] - e_lo
            n_ent = int(counts.sum())
            if n_ent == 0:
                return
            # The entries of every kept segment in the chunk, in segment
            # order, and each one's chunk-local segment index.
            m_loc = np.repeat(np.arange(segs.size, dtype=np.int64), counts)
            ent = e_lo[m_loc] + (
                np.arange(n_ent, dtype=np.int64)
                - np.repeat(np.cumsum(counts) - counts, counts)
            )
            w = w_entry[ent]
            for dest, sblk in zip(subtract_from, block):
                rows = self._tested_contrib_rows(
                    w, m_loc, nq, sblk.reshape(segs.size, nq, -1)
                )
                if src_cols is None:
                    dest[ent] -= rows
                else:
                    # The remainder was prepared over ONE medium's geometry,
                    # so its source axis is that class's, while `dest` is
                    # full width. Place it on the class's own columns; every
                    # other column of this block stays as the image left it,
                    # which the caller's quadrant mask then discards.
                    dest[np.ix_(ent, src_cols)] -= rows

        _ = sub_starts
        with _OverlappedFold() as overlap:
            fg.remainder("cos-1").replay(
                obs_centers=np.asarray(ctx["obs_c"])[obs_rows],
                obs_tangents=np.asarray(ctx["obs_t"])[obs_rows],
                consume=overlap.wrap(_reduce_masked),
                row_group=nq,
            )

    # ---------------------------------------------------------------
    # Below-interface plumbing (momwire#980 D1). Thin wrappers over
    # `_below_interface`, which holds the shared bodies; what stays here is
    # this solver's own state, per that module's docstring.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Mixed decks (momwire#980 D2). Every wire wholly above or wholly below,
    # no wire ending in the plane, no junction spanning media.
    #
    # SUBSET-COMPUTE, FULL-WIDTH PRESENT — bspline's shape
    # (`_build_J_blocks_subset`): each class's block is placed into a
    # full-shaped array and the entries belonging to another pair class stay
    # ZERO. Correctness rests on "the untouched entries are zero", which is
    # true by construction, rather than on a renumbering being right. Global
    # indices throughout, so `_scatter_coef_product`, the drive and the
    # loading path are unmodified.
    #
    # The testing-AGNOSTIC half — the class split, the basis stitch, the
    # serve plan, the transmitted field tensor and the medium plumbing — lives
    # on `SinusoidalSolver` since momwire#1221, where the point-matched lane
    # reaches it too (#1220). What stays here is the Galerkin half: the test
    # context's stitch and the tested reductions.
    # ---------------------------------------------------------------

    _CTX_K_FIELDS = ("w_entry", "sigA", "sigAC", "sigC")

    def _stitch_test_context(self, geom, seg_view, below, k_p, k_m):
        """One test context whose TEST functions follow each segment's medium.

        The test side needs the same treatment as the source side: a test
        function on a buried segment is a current in the lower medium and its
        values belong at k_m. Missing this leaves the test side of every
        buried row in air — a wrong number with no failure, which is D1's
        drive-column finding one level up.

        The four fields below are the only k-dependent ones in a context —
        measured — and all are per ENTRY, so they stitch by the entry's own
        segment exactly as the coefficients do.
        """
        ctx_p = self._test_context(geom, seg_view, k_p)
        ctx_m = self._test_context(geom, seg_view, k_m)
        entry_below = np.asarray(below)[np.asarray(ctx_p["m_of_entry"])]
        out = dict(ctx_p)
        for key in self._CTX_K_FIELDS:
            a = np.array(ctx_p[key], copy=True)
            a[entry_below] = np.asarray(ctx_m[key])[entry_below]
            out[key] = a
        return out

    def _class_fill_serves(self, geom, k, eta, fg):
        """Whether a mixed deck's pair class can be filled restricted to its
        own quadrant (momwire#1224): exactly when every block of it — the free
        block and, under a ground, the image block — takes `_tested_contribs`'
        fused C++ branch, whose predicate this repeats. The numpy branch
        indexes sources, projector tables and EK labels in whole-deck
        coordinates, so it keeps the whole-plane class fill."""
        eta = self._fill_eta(k, eta)
        in_medium = np.iscomplexobj(k) or np.iscomplexobj(eta)
        if self.extended_kernel or not _HAVE_GALERKIN_FAR_FILL:
            return False
        if in_medium and not _HAVE_GALERKIN_FAR_FILL_CPLX:
            return False
        return fg is None or self._image_projector(geom, fg) is _plain_projection

    def _class_view(self, ctx, keep, n_segs):
        """The `_ClassView` of one mixed-deck pair class: its test entries
        against its own source segments (momwire#1224).

        Bit-equal to the whole-plane fill's quadrant, cell for cell, and the
        reasons are all per cell:

        * the fused kernel's test side walks test segments independently
          (one OpenMP iteration each, rows `starts[m]:starts[m+1]` owned
          outright), so a sub-context holding only the class's segments —
          their observers, their entries' `w_entry` rows, their CSR runs —
          gives each of those rows the same sums;
        * its source side is a sweep over the source list whose ONLY
          position-dependent step is the real kernel's vectorized sincos
          (`_SIMD_TAIL_PERIOD`): a phase in the scalar tail gets libm's bits,
          one in the vector body libmvec's. Only the LAST source column's
          phases can sit in the tail, so the class's source list is padded
          at the FRONT with copies of its first segment until its length is
          congruent, modulo the period, to what puts each class column where
          the whole-plane fill put it — the deck's own N when the class holds
          the deck's last segment (that column's tail phases stay in the
          tail), and 0 otherwise (no tail at all, so every class column is
          in the vector body, as it was). Measured on hub16 x1 at random
          subsets: bit-equal exactly under this rule, and 7-89 cells differ
          under every other padding. The pad columns are finite (a copy of a
          real segment) and are dropped at the scatter;
        * the near correction and the Sommerfeld remainder keep their own
          per-cell arithmetic (see `_apply_near_correction` and
          `_fold_ground_block`).
        """
        segs = np.nonzero(keep)[0].astype(np.int64)
        starts = np.asarray(ctx["starts"], dtype=np.int64)
        cnt = np.asarray(ctx["counts"])[segs]
        sub_starts = np.concatenate(([0], np.cumsum(cnt))).astype(np.int64)
        entries = np.repeat(starts[segs] - sub_starts[:-1], cnt) + np.arange(
            sub_starts[-1], dtype=np.int64
        )
        return self._sub_view(
            ctx, entries, n_segs, src_keep=keep, segs=segs, rem_cols="sources"
        )

    def _sub_view(
        self,
        ctx,
        entries,
        n_segs,
        src_keep=None,
        segs=None,
        rem_cols=None,
        rem_obs=None,
    ):
        """The `_ClassView` of any block of the fill: the support `entries`
        (full-deck, ascending) against the sources `src_keep` selects —
        every segment of the deck when it is None (momwire#1224).

        `_class_view` is one caller (a mixed deck's pair class); the fused
        banded fill is the other, with `entries` one observer band's. The
        per-cell argument is `_class_view`'s, and it does not care which
        entries a segment carries: the fused kernel reduces each entry's row
        out of its segment's field alone (`w_entry[e] · Φ_m`, one axpy per
        entry), so a sub-context that holds SOME of a segment's entries gives
        those rows the same sums as one that holds all of them. The source
        side is padded exactly as `_class_view` pads it, and a whole-deck
        source list needs no pad (N ≡ N).

        `segs` pins the test segments (it must cover every entry's segment;
        a class keeps its entry-less segments this way), else they are the
        entries' own. `rem_cols` is the view's `rem_cols`: None, an explicit
        column array, or "sources" for the block's own source columns (a
        remainder prepared over exactly the class's geometry); `rem_obs` is
        the view's `rem_obs`.
        """
        entries = np.asarray(entries, dtype=np.int64)
        nq = ctx["nq"]
        m_e = np.asarray(ctx["m_of_entry"])[entries]
        if segs is None:
            segs = np.unique(m_e)
        segs = np.asarray(segs, dtype=np.int64)
        m_local = np.searchsorted(segs, m_e).astype(np.int64)
        cnt = np.bincount(m_local, minlength=segs.size).astype(np.int64)
        sub_starts = np.concatenate(([0], np.cumsum(cnt))).astype(np.int64)
        obs = (segs[:, None] * nq + np.arange(nq)[None, :]).ravel()

        if src_keep is None:
            pad = 0
            src = np.arange(n_segs, dtype=np.int64)
            src_idx = src
        else:
            src = np.nonzero(src_keep)[0].astype(np.int64)
            period = _SIMD_TAIL_PERIOD
            target = n_segs % period if src_keep[n_segs - 1] else 0
            pad = (target - src.size) % period
            src_idx = np.concatenate((np.full(pad, src[0], dtype=np.int64), src))

        a_obs = ctx["a_obs"]
        sub = {
            "N": int(segs.size),
            "nq": nq,
            # The fused kernel reads `hh` as its SOURCE half-lengths, so it
            # follows the padded source list, not the test segments.
            "hh": np.asarray(ctx["hh"])[src_idx],
            "obs_c": np.ascontiguousarray(np.asarray(ctx["obs_c"])[obs]),
            "obs_t": np.ascontiguousarray(np.asarray(ctx["obs_t"])[obs]),
            "a_obs": a_obs[obs] if isinstance(a_obs, np.ndarray) else a_obs,
            "starts": sub_starts,
            "counts": cnt,
            "m_of_entry": m_local,
            "w_entry": np.ascontiguousarray(np.asarray(ctx["w_entry"])[entries]),
            # Each observer's FULL-deck test segment: what a per-pair
            # projector table and the numpy fill's EK mask index with.
            "m_of_obs": np.repeat(segs, nq),
        }
        nnz = np.asarray(ctx["w_entry"]).shape[0]
        row_of_entry = np.full(nnz, -1, dtype=np.int64)
        row_of_entry[entries] = np.arange(entries.size, dtype=np.int64)
        col_of_seg = np.full(n_segs, -1, dtype=np.int64)
        col_of_seg[src] = pad + np.arange(src.size, dtype=np.int64)
        if isinstance(rem_cols, str):
            rem_cols = col_of_seg[src]
        return _ClassView(
            segs,
            entries,
            pad,
            src_idx,
            sub,
            ctx,
            row_of_entry,
            col_of_seg,
            rem_cols,
            rem_obs,
        )

    def _assemble_mixed_contribs(
        self, geom, ctx, below, medium, plan, crossing=False, eta=None
    ):
        """The three pair classes of a mixed deck, into one (nnz, N) triple.

        Quadrants, and the masks are the point. A class's kernel is right
        only between two points of ITS medium: an above-kernel value at a
        below observer is not the free-space direct term, it is the
        transmitted class. Filling only the cells whose test entry and source
        share a medium is what stops each pair being counted twice, once with
        the wrong kernel.

              rows / cols     above sources        below sources
              above entries   free+ground @ k_p    transmitted (a<-b)
              below entries   transmitted (b<-a)   free+ground @ k_m

        Since momwire#1224 a class is filled over its own quadrant only
        (`_class_view`) wherever the fused C++ fill serves it, bit-equal to
        the whole-plane fill it replaced; the whole-plane fill, which
        computed every observer against every source and kept the quadrant,
        remains for the numpy far fill.
        """
        w_entry = np.asarray(ctx["w_entry"])
        nnz = w_entry.shape[0]
        n_segs = int(geom["n_segs"])
        seg_below = np.asarray(below)
        entry_below = seg_below[np.asarray(ctx["m_of_entry"])]
        # Air's eta, the caller's half of the above class's operating point.
        eta_p = self._fill_eta(medium.k_p, eta)
        contribs = None

        def _destination():
            nonlocal contribs
            if contribs is None:
                contribs = tuple(
                    np.zeros((nnz, n_segs), dtype=np.complex128) for _ in range(3)
                )
            return contribs

        classes = [
            (~seg_below, ~entry_below, medium.k_p, None),
            (seg_below, entry_below, medium.k_m, medium),
        ]
        # The larger quadrant first (momwire#1224). The destination triple is
        # allocated at the first class's scatter, so the big class's block and
        # its image exist before it does rather than beside it. Quadrants are
        # disjoint and each is written by ASSIGNMENT, so the order moves no
        # value; the sort is stable, so a tie keeps the above class first.
        classes.sort(key=lambda c: -int(c[0].sum()) * int(c[1].sum()))

        for keep, rows, k_cls, med in classes:
            if not keep.any() or not rows.any():
                continue
            # Each class fills at ITS OWN (k, eta), both as arguments
            # (momwire#995) — the scoped mutator this replaced guarded solver
            # state in a hot path, and a block that forgot it got a
            # plausible number rather than an exception.
            eta_cls = eta_p if med is None else self._medium_eta(med)
            # The ground images the deck's own wires, but its remainder models
            # ONE medium, so it is prepared over that medium's geometry and
            # replayed at that medium's observers.
            fg = _field_ground.field_ground_for(
                self,
                geom,
                k_cls,
                self.omega,
                medium=med,
                r1_below=plan.get("r1_below") if med is not None else None,
                remainder_geom=self._class_geom(geom, keep),
                eta=eta_cls,
            )
            if self._class_fill_serves(geom, k_cls, eta_cls, fg):
                # The class's own test entries against its own sources, free
                # block, image and remainder alike (momwire#1224). The
                # whole-plane fill below computed every observer against every
                # source and kept the quadrant: on hub16 x4, 60 of 702
                # segments are above, so the above class discarded ~99 % of
                # what it filled, and the (nnz, N) block it built was the
                # mixed peak's largest owner.
                cv = self._class_view(ctx, keep, n_segs)
                block = self._tested_contribs(
                    geom, k_cls, cv.ctx, _plain_projection, eta=eta_cls, cls=cv
                )
                if fg is not None:
                    self._fold_ground_block(
                        geom, k_cls, cv.ctx, block, fg, eta=eta_cls, cls=cv
                    )
                for dest, b in zip(_destination(), block):
                    dest[np.ix_(cv.entries, cv.segs)] = b[:, cv.pad :]
            else:
                # The numpy far fill (no accelerator, or a weighted image
                # projector) indexes in whole-deck coordinates, so it keeps the
                # whole-plane fill: every observer against every source, the
                # quadrant kept.
                block = self._tested_contribs(
                    geom, k_cls, ctx, _plain_projection, eta=eta_cls
                )
                if fg is not None:
                    self._fold_ground_block(
                        geom,
                        k_cls,
                        ctx,
                        block,
                        fg,
                        obs_mask=keep,
                        src_cols=np.nonzero(keep)[0],
                        eta=eta_cls,
                    )
                quadrant = np.ix_(np.nonzero(rows)[0], np.nonzero(keep)[0])
                for dest, b in zip(_destination(), block):
                    dest[quadrant] = b[quadrant]
            # The quadrant is copied, so the class block is dead — and the
            # loop variable `b` is one of its three arrays. Left bound, `b`
            # kept the FIRST class's array alive through the second class's
            # fill and fold, which is where the peak is (momwire#1224).
            del block, b
        contribs = _destination()

        # The two transmitted directions, each into the quadrant whose
        # observers are in the OTHER medium from its sources — ADDED, not
        # subtracted (momwire#1159). The projected tables `t_m . D . t_n` are
        # one family with one sign, and on THIS trunk that family enters G
        # with a plus: the within-class remainder reaches G through
        # `_fold_ground_block`'s `free - (c2*img - rem)`, i.e. `+rem`, and
        # the crossing block is `G + t_ab + t_ab.T`. bspline SUBTRACTS its
        # field blocks (`_sub_field_galerkin`) because its Z carries the
        # opposite global sign — it subtracts its remainder directly, with no
        # second minus — and copying its spelling here negated both cross
        # quadrants: a similarity D*G*D with D = diag(I_above, -I_below),
        # which leaves every single-port Z exactly invariant and flips Y12,
        # the unfed medium's current and so the detached far field. At
        # eps~ = 1 the transmitted table IS the direct field, and the minus
        # put -1.000008x the free-space block in both cross quadrants
        # (scratch/1159-sg-mixed-sign/probe1).
        if crossing:
            # A crossing deck's cross pair is `_crossing_fill`'s, added to
            # the assembled G afterwards. No transmitted grid is built for
            # it — `serve_plan(crossing=True)` does not even size one.
            return contribs
        for src_keep, rows, obs_below in (
            (np.asarray(below), ~entry_below, False),
            (~np.asarray(below), entry_below, True),
        ):
            if not src_keep.any() or not rows.any():
                continue
            obs_keep = np.asarray(below) if obs_below else ~np.asarray(below)
            tensor = self._transmitted_tensor(
                geom,
                medium,
                plan,
                src_keep,
                obs_keep,
                obs_below,
                ctx["obs_c"],
                ctx["obs_t"],
                row_group=ctx["nq"],
                cos_shape="cos-1",
            )
            reduced = self._reduce_field_tensor(ctx, tensor)
            r_idx = np.nonzero(rows)[0]
            c_idx = np.nonzero(src_keep)[0]
            for dest, r in zip(contribs, reduced):
                dest[np.ix_(r_idx, c_idx)] += r[np.ix_(r_idx, c_idx)]
        return contribs

    def _reduce_field_tensor(self, ctx, tensor):
        """Test-integrate a (3, M, N) field tensor into three (nnz, N) blocks —
        `_tested_sommerfeld_remainder`'s reduction, lifted so the transmitted
        block reuses it rather than growing a second one."""
        nq = ctx["nq"]
        w_entry = np.asarray(ctx["w_entry"])
        m_of_entry = np.asarray(ctx["m_of_entry"])
        n_test = int(np.asarray(ctx["obs_c"]).shape[0] // nq)
        return [
            self._tested_contrib_rows(
                w_entry, m_of_entry, nq, sb.reshape(n_test, nq, sb.shape[-1])
            )
            for sb in tensor
        ]

    def _add_crossing_blocks(self, geom, seg_view, medium, below, G, rows=None):
        """The crossing junction's blocks, onto the assembled G.

        `rows` is `_crossing_rows`' answer when the caller computed it
        before the fill (momwire#1224: the crossing fill's own transients —
        its kernel tables, memo and far-block weights, ~1 Z at hub16 x16 —
        then never coexist with G); otherwise it is computed here.

        NOT bspline's spelling, and the two differences were measured on the
        node block rather than argued. bspline writes `Z -= t_ab; Z -= t_ab.T;
        Z += self_completions`; this trunk adds the cross block and takes no
        self completions at all.

        **No self completions.** They exist because bspline's polynomial fill
        evaluates the by-parts integrand pointwise and so drops the boundary
        content at a value-1 end. This family's source fields are the exact
        per-shape ones, and the constant shape's own charge INCLUDES the
        endpoint deltas, so the base fill already carries that content: on
        `crossing_deck(1)` at ε̃ = 1 the base fill puts −0.948·Q on the node
        wing's diagonal (Q = 1/(jωε·4πa), the point-charge scale) where
        `self_completions` would add +0.862·Q. Adding them CANCELS the term
        instead of completing it — the assembled diagonal came out at
        −0.085·Q, the node choked to 1.5e-6 against an ordinary junction's
        1.9e-4, and the collapse gate read 4.6e-01.

        **Plus, not minus.** With the completions gone the cross block's sign
        is fixed by the node's KCL, which the answer alone cannot see: both
        signs give the same Z to seven digits, and Σ inflow at the node is
        4.6e-08 with `+` against 3.8e-04 with `−` — the latter larger than
        the 1.9e-04 node current itself. Same deck, same ε̃ = 1.

        Nothing here imposes a node condition. Continuity through the node
        and the AGARD slope EMERGE from these terms — the by-parts ends and
        the corner — which is why the segment bases were left as free ends
        and the node was given its own dofs (`_crossing_wing_view`).
        """
        rows, t_rows = (
            self._crossing_rows(geom, seg_view, medium, below) if rows is None else rows
        )
        n = G.shape[0]
        if t_rows.shape[1] != n:
            # The port columns `_junction_port_view` appends are not part of
            # the crossing fill's basis axis; a crossing deck with junction
            # ports is out of scope rather than silently mis-scattered.
            raise NotImplementedError(
                "junction ports on a crossing deck are not served "
                f"(crossing block is {t_rows.shape[1]} wide, G is {n} x {n})"
            )
        # `(G + t_ab) + t_ab.T`, the same two elementwise sums, in place on
        # G: first t_ab's rows, then its transpose's columns. Outside `rows`
        # t_ab is zero, so every cell sees its two terms in the order the
        # whole spelling added them, and G keeps its F order for the
        # in-place solve.
        G[rows, :] += t_rows
        G[:, rows] += t_rows.T
        return G

    def _crossing_rows(self, geom, seg_view, medium, below):
        """`(rows, t_ab[rows, :])`: the crossing block's live rows
        (momwire#1224), for `_add_crossing_blocks`.

        t_ab is (n_basis, n_basis) but lives in the rows of the bases with
        support ABOVE: the block is (above axis A x below axis B), and A's
        samples are those bases' only. So it is asked for those rows alone
        (`rows=`, momwire#1029 phase 2) rather than built whole beside G.
        """
        ctx_x = self._crossing_context(geom, seg_view, medium)
        below = np.asarray(below)
        a_idx = np.nonzero(~below)[0]
        b_idx = np.nonzero(below)[0]
        ax_a = _crossing_fill.axis_data(ctx_x, a_idx)
        ax_b = _crossing_fill.axis_data(ctx_x, b_idx)
        starts = np.asarray(seg_view["starts"])
        seg_of_entry = np.repeat(np.arange(starts.size - 1), np.diff(starts))
        rows = np.unique(np.asarray(seg_view["jbasis"])[~below[seg_of_entry]])
        t_rows, _t_cols = _crossing_fill.cross_complete_block_split(
            ctx_x, a_idx, b_idx, ax_a, ax_b, rows=rows
        )
        return rows, t_rows

    def _n_crossing_wings(self):
        """How many NODE-WING columns the crossing junctions add: one per
        member wire-end.

        That count is `BSplineSolver`'s, not a new convention. Its
        `_build_basis_polynomials` keeps every member's value-1 directional
        basis at a junction and, at a GROUNDED one — which every crossing
        junction is — drops the KCL row that would otherwise tie them
        together. So a crossing node carries K free node dofs. The
        KCL-closed through-tent is one combination of those K and lives in
        their span, so this widens the space without picking between the
        two readings; which one the physics wants is then a MEASUREMENT on
        the solved coefficients rather than a choice made in the basis.

        Geometry-free, so every `n_basis` site can ask it without a geom.
        """
        cj = self._crossing_junction_indices()
        return sum(len(self.junctions[j]) for j in sorted(cj))

    def _refuse_crossing_scope(self, crossing):
        """SG serves every crossing geometry the shared rules admit
        (momwire#980): the point-matched lane's stage-2 limits are not its."""

    def _crossing_node_in_basis(self):
        """SG's crossing node is C0 with no condition in the basis: continuity
        and the slope emerge from the wings, the by-parts ends and the corner
        (momwire#980 D3), so its members stay grounded-junction ends."""
        return False

    def _n_extra_cols(self):
        """Basis columns beyond the one-per-segment expansion: the junction
        ports' (#209) and the crossing nodes' wings (#980 D3)."""
        return len(self.junction_ports) + self._n_crossing_wings()

    def _crossing_wing_view(self, geom, base_view, below, medium):
        """Append one node-wing column per crossing-junction member.

        The wing is the three-term extension a real neighbour would carry
        across the node — `_junction_port_view`'s `ext_m`, which is also the
        shape `_basis_coefs` puts on an N⁻ entry — normalised so the
        member's current INTO the node is 1 and its value at the member's
        FAR end is 0. It lives on ONE segment, in ONE medium, at that
        segment's own k, so D2's per-entry stitch is untouched and no basis
        spans the interface.

        Why a column at all. With one dof per segment, a segment's own basis
        cannot carry a free value AND a free slope at the node: its end
        condition fixes the ratio, and all three conditions available are
        wrong here — I = 0 (a free end), dI/ds = 0 (the contact/image shape,
        which the C1-kink lesson rules out, since the charge must be free to
        jump), and a neighbour extension (there is no neighbour in this
        medium). The measured collapse said as much: the C0 end came out at
        3.4e-21 against an ordinary junction's 9.9e-5, i.e. formulation (b)
        was a free end in disguise, `axis_data` then kept no end for it
        (nnz = 0), and the by-parts and corner terms it was supposed to feed
        vanished with it. So the node gets its OWN dof, which is what
        bspline's C0 knot does, and the segment bases keep the free end they
        already have.

        Normalisation is free, and deliberately NOT "value 1 against
        bspline's tents": every term in the trunk is linear in the basis —
        the main sandwich in F/Fd, the by-parts terms rank-1 in fv, the
        corner's outer(fv_a, fv_b) — so scaling a column scales its row, its
        column and its solved coefficient inversely and the solve is
        invariant. What the fill requires is only that `ends`, `F` and `Fd`
        describe the SAME function, which one sampler guarantees. Unit
        INFLOW is chosen because it makes the KCL combination a row of ones,
        which is what the tent-vs-wings measurement needs to be cheap.

        The `a_m` of the extension's own Eq-25 log constant cancels in the
        normalisation — it scales A, B and C alike — so the wing is a pure
        shape and carries no medium in its coefficients beyond its k.
        """
        cj = sorted(self._crossing_junction_indices())
        if not cj:
            return base_view
        N = int(geom["n_segs"])
        seg_h = np.asarray(geom["seg_h"], dtype=float)
        below = np.asarray(below, dtype=bool)
        col = N + len(self.junction_ports)
        segs, bases, A, B, C, AC, sig, kent = [], [], [], [], [], [], [], []
        for j_idx in cj:
            for m, sgn in self._junction_members(geom, j_idx):
                k = medium.k_m if below[m] else medium.k_p
                kent.append(k)
                kd = k * float(seg_h[m])
                # q = 1/tan(kΔ/2): the extension carries a·tan(kΔ/2) into the
                # node, so this is the unit-inflow scale with `a` divided out.
                q = np.cos(0.5 * kd) / np.sin(0.5 * kd)
                segs.append(m)
                bases.append(col)
                A.append(q / np.sin(kd))
                B.append(q / (2.0 * np.cos(0.5 * kd)))
                C.append(-q / (2.0 * np.sin(0.5 * kd)))
                # A + C = q·[1/sin(kΔ) − 1/(2 sin(kΔ/2))] in closed form
                # (momwire#606) — the same identity the N⁻ entry uses,
                # because this IS that shape.
                AC.append(q * _recip_sin_gap(kd))
                sig.append(sgn)
                col += 1

        starts = np.asarray(base_view["starts"], dtype=np.int64)
        base_seg = np.repeat(np.arange(N, dtype=np.int64), np.diff(starts))
        all_seg = np.concatenate([base_seg, np.asarray(segs, dtype=np.int64)])
        order = np.argsort(all_seg, kind="stable")
        new_starts = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(np.bincount(all_seg, minlength=N), out=new_starts[1:])

        def _cat(key, extra, dtype=np.complex128):
            return np.concatenate(
                [
                    np.asarray(base_view[key], dtype=dtype),
                    np.asarray(extra, dtype=dtype),
                ]
            )[order]

        out = {
            "starts": new_starts,
            "jbasis": np.concatenate(
                [np.asarray(base_view["jbasis"]), np.asarray(bases, dtype=np.int64)]
            )[order],
            "A": _cat("A", A),
            "B": _cat("B", B),
            "C": _cat("C", C),
            "AC": _cat("AC", AC),
            "sigma": _cat("sigma", sig, dtype=np.int8),
        }
        if "k_entry" in base_view:
            out["k_entry"] = _cat("k_entry", kent)
        return out

    def _serves_buried(self):
        """momwire#980 D1: this family serves a FULLY-buried deck over a
        Sommerfeld ground, and nothing else below the interface.

        True unconditionally, and NOT `self._lower_medium()`, which was the
        first spelling and gave the wrong sentence: over a PEC or
        reflection-coefficient ground it left `_build_geometry`'s blanket
        refusal to fire, and that sentence says this family "has no in-medium
        kernel at all", which stopped being true at D1. What is actually
        wrong with those decks is that the ground has no half-space below it,
        and `_medium_spec.wire_media` says exactly that.

        So the geometry refusal is lifted for this family entirely and every
        narrower question is answered by name, one level down, where the
        reason is known: `wire_media` for a buried deck over a ground with no
        lower medium and for a crossing junction, and `_fill_medium` for a
        MIXED deck. This widens what is *attempted*, never what is silently
        answered — `_fill_medium` asks `_below_segments` (and so `wire_media`)
        BEFORE it can return `None`, so a buried deck cannot reach the fill
        labelled as air.
        """
        return True

    def _assemble_Z(self, geom, k, eta=None):
        """Galerkin system matrix G (basis i tested against source basis j).

        G[i, j] = ∫ f_i(s) · ŝ·E_j(s) ds
                = Σ_shape ( T[shape] @ M[shape] )[i, j]

        where T[shape, i, n] integrates the closed-form field of unit
        `shape` current on source segment n against test basis i (quadrature
        along i's support), and M[shape][n, j] is the effective coefficient
        source basis j puts on segment n — the SAME source-side coefficient
        matrices the collocation path builds.

        The three shapes are the FOLDED set {1, sin kξ, cos kξ − 1} rather
        than the literal {1, sin kξ, cos kξ} (stevenmburns/momwire#203). The
        reassociation is exact,

            T_c @ M_A + T_co @ M_C = T_c @ M_{A+C} + (T_co − T_c) @ M_C,

        and it is what makes the product well-scaled: A ≈ −C makes M_{A+C}
        small, and cos kξ ≈ 1 over a segment makes the cos-shape source
        radiate almost the const-shape field, so T_co − T_c is small too.
        Literally spelled, the two terms have Frobenius norm 8.8 against G's
        2.7e-6 at N=2401 (3.3e6 amplification, growing like N²); folded they
        are 2.8e-6 and 4.2e-7, i.e. the size of the answer. Measured against
        an 80-bit reference product of the same float64 contributions, the
        float64 result moves from 3.5e-10 to 9.5e-14 relative there.

        `A + C` is formed by a single float64 subtraction of quantities the
        caller already has, so it is correctly rounded *to its own result*.
        `T_co − T_c` is NOT formed here at all: since #205 the fill returns
        the (cos kξ − 1)-shape field directly (`cos_shape="cos-1"`), because
        subtracting the two contribution arrays — however exactly — cannot
        recover the ε·‖T_const‖ the kernel had already left inside them. That
        residual was the fill's reciprocity floor, and it is what
        `SinusoidalSolver._folded_cos_fields` removes.

        With `ground_z` set, the ground's tested sub-assembly is subtracted
        from the free-space one, in place, before the scatter — one more
        source block through the same test quadrature, not a second scheme.

        With junction ports the index `i`/`j` runs over N+P bases rather than
        N (the port columns appended by `_junction_port_view`), so G grows to
        (N+P, N+P). The source index `n` still runs over the N segments —
        a port basis is a current distribution on real segments like any
        other, so it needs no new field kernel and gets none.

        Distributed wire loading enters here and nowhere else (see
        `_apply_loading`): every solve on this family reaches its matrix
        through this method, so one call covers `compute_impedance`,
        `compute_port_solution`/`compute_y_matrix`, both swept loops, and
        `_assemble_Z_ported`, which wraps this one.
        """
        # momwire#980 D1: a fully-buried deck fills in the LOWER MEDIUM —
        # k_m, eta_m and eps_m throughout — and reaches its ground through
        # the below family rather than the above one. Resolved here because
        # this is the scope that fixes k, and returned as one object because
        # a fully-buried deck has exactly one pair class (a mixed deck has
        # three, and is refused above until D2).
        #
        # `None` keeps the shipped path structurally untouched, not merely
        # skipped: at free space or an above-only deck nothing below this
        # line differs, which is the same standard `field_ground_for`'s
        # `None` is held to.
        # A MIXED deck (momwire#980 D2) stitches its basis and its test
        # context per medium and fills three pair classes. Everything below
        # is the single-medium path, structurally unchanged.
        if self._is_mixed(geom):
            below = self._below_segments(geom)
            medium = self._fill_medium(geom)
            crossing = self._is_crossing(geom)
            # The scope refusals, which a single-medium deck raises inside
            # `_operating_medium` — the mixed route does not enter it (two k
            # are live), so they are raised here rather than skipped.
            _below_interface.refuse_out_of_scope(
                use_singular_enrichment=getattr(self, "use_singular_enrichment", False),
                extended_kernel=self.extended_kernel,
                n=int(geom["n_segs"]),
                degree=1,
                dense_fits=True,
                chunked_serves=True,
                swept_mem_mb=getattr(self, "swept_mem_mb", None),
            )
            seg_view = self._stitch_basis_coefs(geom, below, medium.k_p, medium.k_m)
            if crossing:
                # The node's own dofs, appended AFTER the stitch: each wing is
                # already built at its segment's own k, and the stitch selects
                # by entry segment, so it would only re-select what is right.
                seg_view = self._crossing_wing_view(geom, seg_view, below, medium)
            ctx = self._stitch_test_context(
                geom, seg_view, below, medium.k_p, medium.k_m
            )
            # After `ctx`: the plan's extents must cover the TEST observers
            # too, not only the field nodes (see `_mixed_serve_plan`).
            plan = self._mixed_serve_plan(
                geom,
                below,
                medium,
                crossing,
                test_obs=ctx["obs_c"],
                row_group=ctx["nq"],
            )
            if self.fill == "mixed-potential" and crossing:
                return self._assemble_Z_mp_mixed(
                    geom, eta, below, medium, seg_view, ctx, plan
                )
            cross_rows = None
            if self._band_fill_serves(geom["n_segs"]):
                if crossing:
                    # The cross pair FIRST (momwire#1224): its value does not
                    # depend on G, and its fill's transients then never sit
                    # beside G — the banded fill has nothing else that size.
                    cross_rows = self._crossing_rows(geom, seg_view, medium, below)
                # The fill fused with its scatter, one observer band at a
                # time (momwire#1224): no (nnz, N) triple, G F-ordered.
                G = self._assemble_mixed_G_banded(
                    geom, ctx, below, medium, plan, crossing, eta=eta
                )
            else:
                contribs = self._assemble_mixed_contribs(
                    geom, ctx, below, medium, plan, crossing, eta=eta
                )
                G = self._scatter_coef_product(ctx, contribs)
                del contribs
            if crossing:
                # The cross pair, as the designed DIRECT evaluation rather
                # than a transmitted grid: the complete mixed-potential
                # spelling with all by-parts ends and the corner, which is
                # what supplies continuity and the AGARD slope at the C0
                # node the basis leaves free.
                G = self._add_crossing_blocks(
                    geom, seg_view, medium, below, G, rows=cross_rows
                )
            # Loading on a mixed deck (momwire#1156): each shared-segment
            # overlap is written at its segment's own k (`k_entry`, which the
            # wing columns carry too) and read at the REAL ω. It was refused
            # while the only ω on hand was the fill's one k.
            self._apply_loading(G, geom, seg_view, None, medium=medium)
            return G, seg_view

        if self.fill == "mixed-potential" and self._mp_serves(geom):
            return self._assemble_Z_mp(geom, k, eta)

        # (k, eta) is ONE operating point and both halves are arguments
        # (momwire#995); a wholly-buried solve hands in k_m with eta_m.
        eta = self._fill_eta(k, eta)
        seg_view = self._basis_coefs(geom, k)
        ctx = self._test_context(geom, seg_view, k)

        # The ground, as ONE object (momwire#397 unit 3), built here because
        # this is the scope that fixes k: which per-pair weight, which image
        # coefficient, which composition. `None` is free space and is not a
        # null object — the two ground blocks below are then structurally
        # absent rather than skipped, so not one float operation differs
        # (the `extended_kernel=False` standard). Nothing downstream of this
        # line reads `ground_z`, `ground_eps` or `ground_model` on the fill
        # path; what they branch on is `fg.mode`.
        medium = self._active_medium
        fg = _field_ground.field_ground_for(
            self,
            geom,
            k,
            self.omega,
            medium=medium,
            r1_below=self._active_r1_below,
            eta=eta,
        )

        if self._band_fill_serves(geom["n_segs"]):
            # The fill fused with its scatter, one observer band at a time
            # (momwire#1224): G plus one band's scratch, never the (nnz, N)
            # triple, and G F-ordered for the in-place solve.
            G = self._assemble_G_banded(geom, k, ctx, fg, eta=eta)
        else:
            contribs = self._tested_contribs(geom, k, ctx, _plain_projection, eta=eta)
            if fg is not None:
                self._fold_ground_block(geom, k, ctx, contribs, fg, eta=eta)

            G = self._scatter_coef_product(ctx, contribs)
            # The fill's triple is dead the moment its product exists, and
            # what runs next used to be measured on top of it (momwire#355):
            # the end-bracket correction is a second sub-assembly of the same
            # shape, so holding this one across it doubled the fill's own
            # footprint for nothing. Dropping the name here is not an
            # optimization of the arithmetic — it changes no value — it just
            # stops the peak from counting a triple nobody reads again.
            del contribs
        self._ek_bracket_correction_tested(G, geom, k, ctx, fg, eta=eta)
        self._contact_charge_correction_tested(G, geom, k, seg_view, ctx, eta=eta)
        self._apply_loading(G, geom, seg_view, k)
        return G, seg_view

    # ------------------------------------------------------------------
    # The mixed-potential fill on the B-spline pair machinery (momwire#1354)
    # ------------------------------------------------------------------

    def _mp_serves(self, geom):
        """Whether `fill="mixed-potential"` fills THIS deck; the rest take
        the direct-field fill unchanged.

        Served: free space, the PEC image and the Sommerfeld composition, on
        a single-medium deck with the reduced kernel. Not yet served: the
        extended kernel (the moment kernels carry no EK factor on this
        basis), a wire end in the plane (the direct form's #282 contact
        correction has no mixed-form twin yet), and the reflection-coefficient
        ground (NEC's dyad on the image FIELD has no exact potential-form
        spelling — the potential trunk's `(w_A, w_Φ)` is a different
        approximation, and this fill must not change the model under a
        quadrature change).
        """
        if self.extended_kernel or self.junction_ports:
            # A junction port's column has a net inflow at its node by
            # design, so the by-parts boundary terms do not telescope for it.
            return False
        if geom["ground_minus"].any() or geom["ground_plus"].any():
            return False
        if self.ground_z is not None and self.ground_eps is not None:
            if self.ground_model != "sommerfeld":
                return False
        return True

    def _assemble_Z_mp(self, geom, k, eta=None):
        """The Galerkin matrix in mixed-potential form,

            G[i, j] = −Σ_{m,n} [ jkη (t_m·t_n) ⟨f_im, G f_jn⟩ + (η/jk) ⟨f'_im, G f'_jn⟩ ],

        free space minus the image (weighted by the ground's coefficient),
        plus the Sommerfeld remainder — the same three blocks `_assemble_Z`
        tests directly, each written as pair moments against the folded
        shape set (`_sinusoidal_mp`). Block by block the two forms are equal:
        the integration by parts that takes one to the other leaves boundary
        terms that telescope over a continuous, KCL-satisfying basis, so what
        differs is the quadrature and nothing else. The leading minus is the
        direct form's own: it tests E = −jωA − ∇Φ.

        Pair moments come from the B-spline fill's distance-tiered kernel,
        its ladder and orders (`DEFAULT_PAIR_ORDER_LADDER`); the near pairs
        that machinery cannot serve on this basis — a segment against itself
        and its collinear neighbours, a bend's touching pair — are
        overwritten by `_sinusoidal_mp.near_pair_moments`. One observer
        window at a time, so the moment transient is bounded, never N².

        The remainder is the direct form's own `_tested_sommerfeld_remainder`
        evaluator, streamed through the test-context reduction and scattered
        straight into G's rows: a smooth kernel the field form already
        serves at every test node, and whose potential-form twin on this
        basis is the next stage, not this one.
        """
        eta = self._fill_eta(k, eta)
        seg_view = self._basis_coefs(geom, k)
        N = int(geom["n_segs"])
        n_basis = N + self._n_extra_cols()
        c = np.asarray(geom["seg_centers"], dtype=float)
        t = np.asarray(geom["seg_tangents"], dtype=float)
        h = np.asarray(geom["seg_h"], dtype=float)
        a_row = (
            np.full(N, float(self._uniform_radius))
            if self._uniform_radius is not None
            else np.asarray(self._seg_radius(geom), dtype=float)
        )
        starts, jbasis, coef, dcoef = _sinusoidal_mp.basis_csr(seg_view, k)
        G = np.zeros((n_basis, n_basis), dtype=np.complex128, order="F")
        fg = _field_ground.field_ground_for(
            self,
            geom,
            k,
            self.omega,
            medium=self._active_medium,
            r1_below=self._active_r1_below,
            eta=eta,
        )
        from .bspline import DEFAULT_N_QP_PAIR, DEFAULT_PAIR_ORDER_LADDER

        fill = _sinusoidal_mp.WindowFill(
            G,
            starts,
            jbasis,
            coef,
            dcoef,
            k,
            eta,
            n_qp=DEFAULT_N_QP_PAIR,
            ladder=DEFAULT_PAIR_ORDER_LADDER,
            checkpoint=self._checkpoint,
        )
        fill.accumulate(c, t, h, a_row, c, t, h, scale=_sinusoidal_mp.DIRECT_SIGN)
        if fg is not None:
            src_c, src_t = fg.image_sources()
            fill.accumulate(
                c,
                t,
                h,
                a_row,
                src_c,
                src_t,
                h,
                scale=-_sinusoidal_mp.DIRECT_SIGN * fg.image_coefficient,
            )
            if fg.mode == "compose":
                self._mp_remainder(G, geom, seg_view, k, fg)
        self._apply_loading(G, geom, seg_view, k)
        return G, seg_view

    def _mp_remainder(self, G, geom, seg_view, k, fg):
        """The Sommerfeld remainder onto G, with the sign `_fold_ground_block`
        gives it (`free − (c2·img − rem)` puts `+rem` on G).

        At a real k — the above-ground class — it is the B-spline fill's
        route, `_sinusoidal_mp.remainder_Q_above`: the fused C++ kernel at
        `n_qp_sommerfeld` nodes a side with the shapes in place of the
        monomials, and the grazing pairs on graded panels. In the medium (a
        complex k, whose shapes the kernel's real tables cannot carry) it is
        the direct form's evaluator at the test nodes, reduced per entry
        (`_tested_contrib_rows`) and scattered through the source
        coefficients one observer chunk at a time.
        """
        if self._active_medium is None:
            c = np.asarray(geom["seg_centers"], dtype=float)
            t = np.asarray(geom["seg_tangents"], dtype=float)
            h = np.asarray(geom["seg_h"], dtype=float)
            seg_l, seg_r = _sinusoidal_mp.segment_ends(c, t, h)
            gz = float(self.ground_z)
            r1_max = _sommerfeld.max_image_distance(seg_l, seg_r, gz)
            grid = _below_interface.somm_grid(
                fg.eps_tilde, float(k), r1_max, self.omega, self.mu, self._cancel_flag
            )
            starts, jbasis, coef, _dcoef = _sinusoidal_mp.basis_csr(seg_view, k)
            _sinusoidal_mp.remainder_Q_above(
                G,
                starts,
                jbasis,
                coef,
                seg_l,
                seg_r,
                t,
                h,
                gz,
                float(k),
                grid,
                base_q=self.n_qp_sommerfeld,
                scale=_sinusoidal_mp.REMAINDER_SIGN,
                cancel_flag=self._cancel_flag,
                checkpoint=self._checkpoint,
            )
            return
        ctx = self._test_context(geom, seg_view, k)
        self._mp_remainder_below(
            G,
            geom,
            seg_view,
            ctx,
            fg,
            np.ones(ctx["N"], dtype=bool),
            self._active_medium,
            self._active_r1_below,
        )

    def _assemble_Z_mp_mixed(self, geom, eta, below, medium, seg_view, ctx, plan):
        """`_assemble_Z_mp` for a CROSSING deck: the two within-medium
        classes in mixed-potential form, the crossing pair by `_crossing_fill`
        exactly as the direct fill adds it, and bspline's `self_completions`
        taken OFF.

        The completions are the by-parts content a mixed-potential class block
        drops at a value-1 end: the node's bases do not vanish there, so the
        test-side boundary term and the source's end charge no longer
        telescope within the class, and bspline — whose class blocks are the
        same mixed-potential spelling — adds them back as `self_completions`.
        The direct fill carries that content in its closed-form fields and
        `_add_crossing_blocks` documents why it must NOT add them; this fill
        is on the other side of that line, and takes them with G's sign
        (G = −Z on this trunk).

        The above class runs the free-space route at k_p with the Sommerfeld
        composition; the below class at k_m with the medium's image weight
        and the below remainder through the direct form's evaluator at the
        class's own test nodes. The pair ladders are bspline's per class
        (`DEFAULT_*` above, `BURIED_*` in the medium).
        """
        from .bspline import (
            BURIED_N_QP_PAIR,
            BURIED_PAIR_ORDER_LADDER,
            DEFAULT_N_QP_PAIR,
            DEFAULT_PAIR_ORDER_LADDER,
        )

        N = int(geom["n_segs"])
        n_basis = N + self._n_extra_cols()
        c = np.asarray(geom["seg_centers"], dtype=float)
        t = np.asarray(geom["seg_tangents"], dtype=float)
        h = np.asarray(geom["seg_h"], dtype=float)
        a_row = (
            np.full(N, float(self._uniform_radius))
            if self._uniform_radius is not None
            else np.asarray(self._seg_radius(geom), dtype=float)
        )
        seg_below = np.asarray(below, dtype=bool)
        starts, jbasis, coef, dcoef = _sinusoidal_mp.basis_csr(seg_view, medium.k_p)
        # The crossing pair first (momwire#1224): its transients never sit
        # beside G.
        rows_x, t_rows = self._crossing_rows(geom, seg_view, medium, below)
        G = np.zeros((n_basis, n_basis), dtype=np.complex128, order="F")
        eta_p = self._fill_eta(medium.k_p, eta)
        for keep, k_cls, med in (
            (~seg_below, medium.k_p, None),
            (seg_below, medium.k_m, medium),
        ):
            if not keep.any():
                continue
            idx = np.nonzero(keep)[0].astype(np.int64)
            eta_cls = eta_p if med is None else self._medium_eta(med)
            fg = _field_ground.field_ground_for(
                self,
                geom,
                k_cls,
                self.omega,
                medium=med,
                r1_below=plan.get("r1_below") if med is not None else None,
                remainder_geom=self._class_geom(geom, keep),
                eta=eta_cls,
            )
            fill = _sinusoidal_mp.WindowFill(
                G,
                starts,
                jbasis,
                coef,
                dcoef,
                k_cls,
                eta_cls,
                n_qp=DEFAULT_N_QP_PAIR if med is None else BURIED_N_QP_PAIR,
                ladder=DEFAULT_PAIR_ORDER_LADDER
                if med is None
                else BURIED_PAIR_ORDER_LADDER,
                checkpoint=self._checkpoint,
            )
            fill.accumulate(
                c[idx],
                t[idx],
                h[idx],
                a_row[idx],
                c[idx],
                t[idx],
                h[idx],
                scale=_sinusoidal_mp.DIRECT_SIGN,
                obs_idx=idx,
                src_idx=idx,
            )
            if fg is None:
                continue
            src_c, src_t = fg.image_sources()
            fill.accumulate(
                c[idx],
                t[idx],
                h[idx],
                a_row[idx],
                src_c[idx],
                src_t[idx],
                h[idx],
                scale=-_sinusoidal_mp.DIRECT_SIGN * fg.image_coefficient,
                obs_idx=idx,
                src_idx=idx,
            )
            if fg.mode != "compose":
                continue
            if med is None:
                sub = _sinusoidal_mp.csr_subset(starts, jbasis, coef, dcoef, idx)
                seg_l, seg_r = _sinusoidal_mp.segment_ends(c[idx], t[idx], h[idx])
                gz = float(self.ground_z)
                r1_max = _sommerfeld.max_image_distance(seg_l, seg_r, gz)
                grid = _below_interface.somm_grid(
                    fg.eps_tilde,
                    float(k_cls),
                    r1_max,
                    self.omega,
                    self.mu,
                    self._cancel_flag,
                )
                _sinusoidal_mp.remainder_Q_above(
                    G,
                    sub[0],
                    sub[1],
                    sub[2],
                    seg_l,
                    seg_r,
                    t[idx],
                    h[idx],
                    gz,
                    float(k_cls),
                    grid,
                    base_q=self.n_qp_sommerfeld,
                    scale=_sinusoidal_mp.REMAINDER_SIGN,
                    cancel_flag=self._cancel_flag,
                    checkpoint=self._checkpoint,
                )
            else:
                self._mp_remainder_below(
                    G, geom, seg_view, ctx, fg, keep, med, plan["r1_below"]
                )
        # The crossing pair, as `_add_crossing_blocks` adds it, then the
        # completions off.
        G[rows_x, :] += t_rows
        G[:, rows_x] += t_rows.T
        del t_rows
        ctx_x = self._crossing_context(geom, seg_view, medium)
        a_idx = np.nonzero(~seg_below)[0]
        b_idx = np.nonzero(seg_below)[0]
        density = dict(
            q=_sinusoidal_mp.COMPLETION_Q,
            panel_order=_sinusoidal_mp.COMPLETION_PANEL_ORDER,
            growth=_sinusoidal_mp.COMPLETION_GROWTH,
        )
        ax_a = _crossing_fill.axis_data(ctx_x, a_idx, **density)
        ax_b = _crossing_fill.axis_data(ctx_x, b_idx, **density)
        comp = np.zeros((n_basis, n_basis), dtype=np.complex128)
        _crossing_fill.self_completions(ctx_x, ax_b, ax_a, out=comp)
        G += _sinusoidal_mp.COMPLETION_SIGN * comp
        del comp
        self._apply_loading(G, geom, seg_view, None, medium=medium)
        return G, seg_view

    def _mp_remainder_below(self, G, geom, seg_view, ctx, fg, keep, medium, r1_below):
        """The below-class remainder onto G through bspline's field-form
        assembler (`_sinusoidal_mp.remainder_Q_field`), the direct form's
        masked replay (`_mp_remainder_masked`) when the accelerator is
        absent. `fg` is the class's ground; `medium` its `BuriedMedium`."""
        if not _sinusoidal_mp._HAVE_FIELD_MOMENTS:
            self._mp_remainder_masked(G, ctx, fg, keep)
            return
        idx = np.nonzero(np.asarray(keep))[0].astype(np.int64)
        c = np.asarray(geom["seg_centers"], dtype=float)[idx]
        t = np.asarray(geom["seg_tangents"], dtype=float)[idx]
        h = np.asarray(geom["seg_h"], dtype=float)[idx]
        seg_l, seg_r = _sinusoidal_mp.segment_ends(c, t, h)
        gz = float(self.ground_z)
        grid = _sommerfeld_below.get_grid_below(
            medium.eps_t, medium.k_p, r1_below, self.omega, mu=self.mu
        )
        k_p, k_m = medium.k_p, medium.k_m

        def proj(o, to, s_, ts):
            return _sommerfeld_below.remainder_field_proj_below(
                o, to, s_, ts, gz, k_p, k_m, grid
            )

        starts, jbasis, coef, _dcoef = _sinusoidal_mp.basis_csr(seg_view, k_m)
        _sinusoidal_mp.remainder_Q_field(
            G,
            starts,
            jbasis,
            coef,
            seg_l,
            seg_r,
            t,
            h,
            idx,
            k_m,
            self.n_qp_sommerfeld,
            proj,
            gz=gz,
            scale=_sinusoidal_mp.REMAINDER_SIGN,
            cancel_flag=self._cancel_flag,
            checkpoint=self._checkpoint,
        )

    def _mp_remainder_masked(self, G, ctx, fg, keep):
        """The below-class remainder onto G: `fg`'s evaluator (prepared over
        the class geometry) replayed at the class's own test nodes, reduced
        per entry and scattered through the source coefficients onto the
        class's columns — `_tested_sommerfeld_remainder`'s masked replay
        with G, not a triple, as the destination."""
        nq = ctx["nq"]
        starts = np.asarray(ctx["starts"])
        w_entry = ctx["w_entry"]
        m_of_entry, i_of_entry = ctx["m_of_entry"], ctx["i_of_entry"]
        nnz = w_entry.shape[0]
        starts_pad = np.concatenate((starts, [nnz]))
        seg_keep = np.nonzero(keep)[0]
        n_basis = G.shape[0]
        obs_rows = (seg_keep[:, None] * nq + np.arange(nq)[None, :]).ravel()
        # The class's source coefficient matrices: class segments (the
        # remainder's own source axis) x bases.
        ent_src = np.concatenate(
            [np.arange(starts_pad[s], starts_pad[s + 1]) for s in seg_keep]
        )
        row_of_src = np.searchsorted(seg_keep, m_of_entry[ent_src])
        Ms = [
            scipy.sparse.csc_matrix(
                (coef[ent_src], (row_of_src, i_of_entry[ent_src])),
                shape=(seg_keep.size, n_basis),
            )
            for coef in (ctx["sigAC"], ctx["B"], ctx["sigC"])
        ]

        def _consume(i0, i1, block):
            m0, m1 = i0 // nq, i1 // nq
            segs = seg_keep[m0:m1]
            e_lo = starts_pad[segs]
            counts = starts_pad[segs + 1] - e_lo
            n_ent = int(counts.sum())
            if n_ent == 0:
                return
            m_loc = np.repeat(np.arange(segs.size, dtype=np.int64), counts)
            ent = e_lo[m_loc] + (
                np.arange(n_ent, dtype=np.int64)
                - np.repeat(np.cumsum(counts) - counts, counts)
            )
            w = w_entry[ent]
            rows_i = i_of_entry[ent]
            touched, local = np.unique(rows_i, return_inverse=True)
            R = scipy.sparse.csr_matrix(
                (np.ones(local.size), (local, np.arange(local.size))),
                shape=(touched.size, local.size),
            )
            for s_blk, M in zip(block, Ms):
                rows = self._tested_contrib_rows(
                    w, m_loc, nq, s_blk.reshape(segs.size, nq, -1)
                )
                G[touched] += _sinusoidal_mp.REMAINDER_SIGN * np.asarray(
                    R @ np.asarray(rows @ M)
                )

        fg.remainder("cos-1").replay(
            obs_centers=np.asarray(ctx["obs_c"])[obs_rows],
            obs_tangents=np.asarray(ctx["obs_t"])[obs_rows],
            consume=_consume,
            row_group=nq,
        )

    # ------------------------------------------------------------------
    # Distributed series wire loading, Galerkin form (momwire#131, #395)
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_segment_pairs(starts):
        """(left, right, m_of_left) — every ORDERED pair of support entries
        that shares a segment, from the segment-major CSR's `starts`.

        The CSR holds one entry per (segment, basis) pair, so segment s's
        entries name distinct bases and the pairs of run `starts[s]:starts[s+1]`
        are exactly the (test, source) basis pairs whose supports meet on s.
        A given (i, j) pair can recur across segments (adjacent bases overlap
        on one segment, a basis with itself on two), so consumers must
        accumulate unbuffered.

        Built by ragged expansion rather than a per-segment loop: `left`
        repeats each entry once per entry of its own segment and `right`
        walks that segment's run for each repeat.
        """
        counts = np.diff(starts)
        nnz = int(starts[-1])
        m_of_entry = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
        reps = counts[m_of_entry]
        left = np.repeat(np.arange(nnz, dtype=np.int64), reps)
        # ramp = 0,1,…,reps[e]-1 per source entry e, i.e. arange minus the
        # exclusive prefix sum broadcast back over each run.
        ramp = np.arange(int(reps.sum()), dtype=np.int64) - np.repeat(
            np.cumsum(reps) - reps, reps
        )
        right = np.repeat(starts[m_of_entry], reps) + ramp
        return left, right, m_of_entry[left]

    def _apply_loading(self, G, geom, seg_view, k, medium=None):
        """The Galerkin form of NEC's impedance boundary condition, in place;
        no-op when loading is off.

        A distributed series impedance changes the wire surface condition to
        E_scat − Z'_w·I = −E_app. Testing that with the same basis set the
        expansion uses turns the point-matched subtraction of the inherited
        `SinusoidalSolver._apply_loading` (one match point per row) into an
        overlap over the shared support:

            G[i, j] −= Σ_w Z'_w(ω) · Σ_{s ∈ w} ∫_s f_i(ξ)·f_j(ξ) dξ

        — the same global sign, because it is the same equation, only tested
        differently.

        On segment s each basis carries ONE support entry with the three-term
        shape f_e(ξ) = P_e + Q_e·sin kξ + R_e·cos kξ, (P, Q, R) = (σA, B, σC),
        so the segment's block is the outer product of its entries under the
        closed-form bilinear form on ξ ∈ [−h/2, h/2]. Two of the six shape
        products integrate to zero because their integrands are odd:

            ∫1·1   = h                      ∫1·sin   = 0
            ∫1·cos = (2/k)·sin(kh/2)        ∫sin·cos = 0
            ∫sin·sin = h/2 − sin(kh)/2k     ∫cos·cos = h/2 + sin(kh)/2k

        giving

            L_s[e, f] = P_eP_f·h + (P_eR_f + R_eP_f)·(2/k)sin(kh/2)
                      + Q_eQ_f·(h/2 − sin(kh)/2k)
                      + R_eR_f·(h/2 + sin(kh)/2k).

        EVALUATED in a cancellation-free spelling of that same integral
        (momwire#1283). On a short segment A and C grow like 1/(kh)² with
        opposite signs while A + C stays O(1) — #606's mechanism — so the
        literal sum above adds four terms of size (1/kh)⁴·h to return one of
        size h. A crossing node's wing on a 9.46 µm sliver (kh = 2.8e-6) has
        A = −C = 2.6e11 with A + C = 0.25: the literal L_s read −256 against
        an exact 1.89e-6, and Dan's through deck answered 460 Ω instead of
        71. Rewriting the shape as U + Q·sin kξ + R·(cos kξ − 1), U = σ·AC
        (the closed-form `AC`, never σA + σC), leaves no term larger than
        the answer:

            L_s[e, f] = U_eU_f·h + (U_eR_f + R_eU_f)·∫(cos kξ − 1)
                      + Q_eQ_f·∫sin² kξ + R_eR_f·∫(cos kξ − 1)²

        with each integral itself spelled without subtraction
        (`_loading_integrals`).

        That is the bilinear (unconjugated) sibling of the |I|² family
        `SinusoidalSolver.wire_loss_power` integrates for the power readout,
        which this family inherits unchanged — a physical integral does not
        care which testing scheme produced the coefficients.

        Unlike `BSplineSolver._loading_gram`'s polynomial moments these VALUES
        move with k, because {1, sin kξ, cos kξ} carries k; only the sparsity
        STRUCTURE is geometry-only. Nothing is cached: the structure is a
        handful of integer ops per support entry and the values a handful of
        flops, against a fill that is O(N²) kernel evaluations.

        Junction-port columns need no special case. A port basis is a current
        distribution on real segments like any other (`_junction_port_view`),
        so its entries carry the same three-term shape on their member
        segments and enter the same overlap; M5b's node-charge correction
        acts on the port's CHARGE, so it cannot interact with a term built
        from current alone.
        """
        if not self._loading_active:
            return G
        if medium is None:
            medium = self._active_medium
        starts = seg_view["starts"]
        left, right, m_of_pair = self._shared_segment_pairs(starts)
        if "k_entry" in seg_view:
            # A mixed deck's stitched view (momwire#980 D2/D3): every entry
            # carries the k it was built at, and the two entries of a pair
            # share a segment, so the pair's shapes are written in that
            # segment's own medium (momwire#1156).
            k = np.asarray(seg_view["k_entry"])[left]

        sig = seg_view["sigma"].astype(np.complex128)
        U = sig * seg_view["AC"]
        Q = seg_view["B"]
        R = sig * seg_view["C"]
        h = np.asarray(geom["seg_h"], dtype=float)[m_of_pair]
        i_g, i_ss, i_gg = _loading_integrals(k, h)
        vals = (
            U[left] * U[right] * h
            + (U[left] * R[right] + R[left] * U[right]) * i_g
            + Q[left] * Q[right] * i_ss
            + R[left] * R[right] * i_gg
        )
        # The loading is read at the solve's REAL angular frequency. `k` is
        # the SHAPES' wavenumber — k_m, complex, on a buried segment — and
        # k_m·c is not a frequency: a bare conductor's surface impedance and
        # a lumped/distributed RLC are local, medium-independent functions
        # of ω (momwire#1156; razor's #1149 U3 derivation). `medium.k_p` is
        # the upper medium's wavenumber, float(self.k) before the swap, so
        # an above-ground or free-space deck (medium None) reads the same
        # `k * self.c` it always did, bit for bit.
        omega = (k if medium is None else medium.k_p) * self.c
        # (n_segs,), zeros where switched off — the shared spec layer
        # (momwire#428); the overlap VALUES above are this rule's own share.
        spec = _wire_loading.loading_for(self, omega, geom)
        if spec.zq_seg is not None:
            # A jacketed BURIED wire's charge-side term (momwire#1154): the
            # local potential dS'·q, q = −(1/jω)·dI/dl, tested by parts the
            # way bspline's `_charge_gram` tests it —
            #
            #     G[i, j] −= zq_s · ∫_s f_i′(ξ)·f_j′(ξ) dξ,  zq = dS'/(jω)
            #
            # — the same global sign as the series term above, because both
            # are impedances the wire adds to Z = −G. The boundary terms the
            # by-parts step drops vanish at a free end (every basis is exactly
            # 0 there, and continuous across every interior segment boundary:
            # scratch/1156-sg-buried-loading/probe1.out), and at a junction
            # are dropped as bspline drops them. On the three-term shape
            # f′ = k·(Q·cos kξ − R·sin kξ), and ∫sin·cos = 0 by parity:
            #
            #     D_s[e, f] = k²·[ Q_eQ_f·(h/2 + sin(kh)/2k)
            #                    + R_eR_f·(h/2 − sin(kh)/2k) ].
            #
            # `zq_seg` is None on every deck without a jacketed wire below
            # the interface, so every other fill is structurally untouched.
            #
            # The R·R term's h/2 − sin(kh)/2k is `i_ss`, spelled without the
            # subtraction for the same reason as the series term (#1283).
            dvals = (k * k) * (
                Q[left] * Q[right] * (0.5 * h + np.sin(k * h) / (2.0 * k))
                + R[left] * R[right] * i_ss
            )
            vals = vals * spec.z_seg[m_of_pair] + dvals * spec.zq_seg[m_of_pair]
        else:
            vals *= spec.z_seg[m_of_pair]
        # Unbuffered: a basis pair recurs once per shared segment.
        np.subtract.at(G, (seg_view["jbasis"][left], seg_view["jbasis"][right]), vals)
        return G

    # ------------------------------------------------------------------
    # The fused banded fill (momwire#1224): G one observer band at a time
    # ------------------------------------------------------------------

    def _band_fill_serves(self, n_segs):
        """Whether G is assembled band by band (momwire#1224). Below the
        dense-assembly threshold the scatter is a zgemm whose blocking
        follows the row count, so there the fill stays whole (tiny: N < 60)
        and is the old fill verbatim — the point-matched fused fill's rule
        (`SinusoidalSolver._mixed_band_rows`). A method so a test can refuse
        it and recompute the whole-triple fill in-process."""
        return int(n_segs) >= _DENSE_ASSEMBLY_THRESHOLD

    def _band_budget_bytes(self, n_basis):
        """Scratch budget for one observer band of the fused fill
        (momwire#1224): a quarter of G, capped at the solver's
        `swept_mem_mb` — the budget the point-matched fused fill sizes its
        bands by.

        Relative to G because what the fill is FOR is to hold G and little
        else: a fixed budget is a fixed multiple of a small G (256 MB is 2 Z
        at N = 2800) and a vanishing one of a large G. `_segment_bands` puts
        a floor in segments under it. A method so a test can force many
        bands."""
        z_bytes = 16 * int(n_basis) * int(n_basis)
        return min(int(self.swept_mem_mb) * 1024 * 1024, z_bytes // 4)

    def _segment_bands(self, ctx, n_basis, n_cols, n_triples, align=None):
        """The fused fill's observer bands: `(m0, m1, e0, e1)` for runs of
        TEST segments `m0:m1` and their support entries `e0:e1` — contiguous,
        since the CSR is segment-major (momwire#1224).

        Every test segment is filled exactly once, in ascending order, which
        is what lets a basis whose entries straddle two bands still meet them
        in the whole product's order (`_scatter_band`). Rows per band come
        from the byte budget (`_band_budget_bytes`) over what one segment
        holds: its entries' rows of `n_triples` (entries, n_cols) triples —
        the band's own and, under a Sommerfeld ground, its image (a mixed
        deck's class block too) — plus its share of the scatter: three T rows
        and the product's copies and results, ~4 n_basis. Never fewer than
        `_BAND_MIN_SEGMENTS` a band, though: the fused kernel threads over a
        band's test segments, so a thin band idles threads.

        `align` (the numpy fill's block, `_fill_block`) rounds the band up to
        whole blocks of that loop, so a band that takes the numpy far fill
        walks exactly the blocks the whole fill walked. That loop's per-entry
        values are NOT independent of its block height on every path: on a
        complex-k block they move in the last bits across numpy's 256 KB
        temporary-elision boundary (the #392 mechanism, still live there —
        measured, mixed(4) with the accelerator hidden: 116 cells move in
        bands of 3 segments, none in bands holding the whole block)."""
        N = int(ctx["N"])
        starts = np.asarray(ctx["starts"], dtype=np.int64)
        nnz = int(np.asarray(ctx["w_entry"]).shape[0])
        entries_per_seg = nnz / max(1, N)
        per_seg = 16 * (
            3 * n_triples * entries_per_seg * n_cols + 3 * n_cols + 4 * n_basis
        )
        step = max(
            min(_BAND_MIN_SEGMENTS, N),
            int(self._band_budget_bytes(n_basis) // per_seg),
        )
        if align:
            step = -(-step // int(align)) * int(align)
        for m0 in range(0, N, step):
            m1 = min(m0 + step, N)
            yield m0, m1, int(starts[m0]), nnz if m1 == N else int(starts[m1])

    def _band_rows(self, ctx, G):
        """The fused fill's scatter state (`_BandRows`) for one G: the three
        source-coefficient matrices M[shape] of `_scatter_coef_product`'s
        sparse product — the same csc construction, so the same stored order
        per column — each basis's LAST support entry, and the carry of T
        rows a basis has begun but not finished."""
        N = int(ctx["N"])
        n_basis = N + self._n_extra_cols()
        mi = (ctx["m_of_entry"], ctx["i_of_entry"])
        Ms = tuple(
            scipy.sparse.csc_matrix((coef, mi), shape=(N, n_basis))
            for coef in (ctx["sigAC"], ctx["B"], ctx["sigC"])
        )
        i_of_entry = np.asarray(ctx["i_of_entry"], dtype=np.int64)
        last = np.full(n_basis, -1, dtype=np.int64)
        np.maximum.at(last, i_of_entry, np.arange(i_of_entry.size, dtype=np.int64))
        return _BandRows(G, Ms, i_of_entry, last, {})

    def _scatter_band(self, rows, e0, e1, band):
        """Entries `e0:e1`' band triple into G (momwire#1224): the whole
        `_scatter_coef_product`, a band at a time, to the bit.

        That product is `G[i, j] = Σ_shape Σ_{e′ ∈ col j} coef·T[i, m(e′)]`
        with `T[i] = R[i] @ contrib`: scipy's csr product starts each row at
        zero and adds basis i's entries' rows in ascending entry order (the
        unit weights multiply exactly). So T rows are accumulated here the
        same way — the basis's carried partial row, or zeros, then this
        band's entries added one at a time in ascending order
        (`_row_scatter`, `np.add.at`'s order without its per-element loop;
        a row with no carry starts from +0 in the scatter itself rather than
        from a zeroed buffer) — and since every band
        follows every earlier one, a basis that straddles bands meets its
        entries in the sequence the whole product did. A basis is finished at
        the band holding its LAST entry; its rows of the three products
        `T[rows] @ M[shape]` (each cell's sum is over column j's stored
        entries whatever the row count) are summed as Python's `sum`,
        `(0 + P0) + P1 + P2`, as they were, and ASSIGNED into G. Unfinished
        rows ride the carry, which holds only the bases open across a band
        boundary — O(N) each, a handful at a time."""
        i_of = rows.i_of_entry[e0:e1]
        touched, local = np.unique(i_of, return_inverse=True)
        n_src = band[0].shape[1]
        T = [np.empty((touched.size, n_src), dtype=np.complex128) for _ in range(3)]
        fresh = np.ones(touched.size, dtype=bool)
        for r, b in enumerate(touched):
            got = rows.carry.pop(int(b), None)
            if got is not None:
                fresh[r] = False
                for t, g in zip(T, got):
                    t[r] = g
        for t, c in zip(T, band):
            _row_scatter(t, local, c, fresh)
        done = rows.last[touched] < e1
        if done.any():
            fin = np.flatnonzero(done)
            rows.G[touched[fin]] = sum(t[fin] @ M for t, M in zip(T, rows.Ms))
        for r in np.flatnonzero(~done):
            rows.carry[int(touched[r])] = tuple(t[r].copy() for t in T)

    def _assemble_G_banded(self, geom, k, ctx, fg, eta=None):
        """G of a single-medium deck, one observer band at a time
        (momwire#1224): per band of test segments the free block, the
        ground's image block and Sommerfeld remainder with the fold, and then
        the band's scatter straight into a preallocated, FORTRAN-ordered G
        (`_scatter_band`). What this holds is G plus one band's triple(s) and
        the scatter's carried rows: the whole (nnz, N) triple never exists.

        Bit-identical to the whole-triple fill, cell for cell (gated by
        `tests/test_sg_fused_banded_fill_1224.py`):

        * a band is a `_sub_view` of its segments' entries against the deck's
          whole source list, so every entry's row is the fused kernel's own
          per-entry reduction (see `_sub_view`), and the SIMD-tail position of
          every source phase is unchanged (the source list IS the deck's);
        * the near correction's values are the whole deck's, computed once
          (`_near_cells`) and placed per band by the same assignment or
          subtraction;
        * the remainder replays at the band's observers — per observer it
          does not see the observer set (`_replay_remainder`), and the
          reduction is per entry;
        * the fold is elementwise, and the scatter is `_scatter_band`.

        G comes out F-ordered, so `_solve_in_place` factors it with no copy.
        """
        N = int(ctx["N"])
        n_basis = N + self._n_extra_cols()
        near_free = near_img = None
        if self.near_correction:
            near_free = self._near_cells(geom, k, ctx, _plain_projection, eta=eta)
            if fg is not None:
                src_c_img, src_t_img = fg.image_sources()
                near_img = self._near_cells(
                    geom,
                    k,
                    ctx,
                    self._image_projector(geom, fg),
                    src_c_img,
                    src_t_img,
                    mirror=True,
                    eta=eta,
                )
        n_triples = 2 if (fg is not None and fg.mode != "fold") else 1
        # A block on the numpy far fill bands on that loop's own blocks (see
        # `_segment_bands`): the refl-coef image's weighted projector, or no
        # accelerator for this (k, eta).
        numpy_fill = not self._fused_fill_serves(k, eta, _plain_projection) or (
            fg is not None
            and not self._fused_fill_serves(k, eta, self._image_projector(geom, fg))
        )
        align = _fill_block(N, ctx["nq"], self.n_qp_const) if numpy_fill else None
        G = np.zeros((n_basis, n_basis), dtype=np.complex128, order="F")
        rows = self._band_rows(ctx, G)
        for m0, m1, e0, e1 in self._segment_bands(
            ctx, n_basis, N, n_triples, align=align
        ):
            if e1 == e0:
                continue
            self._checkpoint()
            view = self._sub_view(ctx, np.arange(e0, e1), N, segs=np.arange(m0, m1))
            band = self._tested_contribs(
                geom, k, view.ctx, _plain_projection, eta=eta, cls=view, near=near_free
            )
            if fg is not None:
                self._fold_ground_block(
                    geom, k, view.ctx, band, fg, eta=eta, cls=view, near=near_img
                )
            self._scatter_band(rows, e0, e1, band)
            band = view = None
        assert not rows.carry, "a basis row was left unfinished (momwire#1224)"
        return G

    def _assemble_mixed_G_banded(
        self, geom, ctx, below, medium, plan, crossing=False, eta=None
    ):
        """G of a mixed deck, one observer band at a time (momwire#1224):
        `_assemble_mixed_contribs` and `_scatter_coef_product` fused, so the
        (nnz, N) triple — per class or whole — never exists, nor does the
        transmitted directions' field tensor.

        A band (a run of test segments, `_segment_bands`) has a (entries, N)
        triple of zeros, which receives exactly what those rows of the whole
        triple received, in the same way:

        * each pair class's quadrant, ASSIGNED: the class's entries in the
          band against the class's sources (`_sub_view` with the class's
          padded source list, so each class column keeps its SIMD-tail
          position — `_class_view`'s rule), its image, its remainder at its
          band observers into its own columns, its precomputed near cells.
          Where the fused C++ fill does not serve a class the band fills
          every entry it holds against the whole source list and the
          quadrant is cut out of it, which is the whole-plane fallback on the
          band's rows (on the numpy loop's own blocks, `_segment_bands`);
        * each transmitted direction, ADDED onto the zeros of the cross
          quadrant (non-crossing decks): `_transmitted_block` at the band's
          observers only, reduced per entry. The whole fill reduced a
          full-width (3, n_obs, N) tensor holding the same floats;
        * the band is scattered into G (`_scatter_band`).

        Quadrants partition the matrix, so every cell has one writer, and
        each write is the operation the whole fill performed. The crossing
        deck's cross pair is added to G afterwards, as before."""
        n_segs = int(geom["n_segs"])
        n_basis = n_segs + self._n_extra_cols()
        nq = ctx["nq"]
        w_entry = np.asarray(ctx["w_entry"])
        m_of_entry = np.asarray(ctx["m_of_entry"])
        seg_below = np.asarray(below)
        entry_below = seg_below[m_of_entry]
        eta_p = self._fill_eta(medium.k_p, eta)

        classes = []
        for keep, rows, k_cls, med in (
            (~seg_below, ~entry_below, medium.k_p, None),
            (seg_below, entry_below, medium.k_m, medium),
        ):
            if not keep.any() or not rows.any():
                continue
            eta_cls = eta_p if med is None else self._medium_eta(med)
            fg = _field_ground.field_ground_for(
                self,
                geom,
                k_cls,
                self.omega,
                medium=med,
                r1_below=plan.get("r1_below") if med is not None else None,
                remainder_geom=self._class_geom(geom, keep),
                eta=eta_cls,
            )
            near_free = near_img = None
            if self.near_correction:
                near_free = self._near_cells(
                    geom, k_cls, ctx, _plain_projection, eta=eta_cls, keep=keep
                )
                if fg is not None:
                    src_c_img, src_t_img = fg.image_sources()
                    near_img = self._near_cells(
                        geom,
                        k_cls,
                        ctx,
                        self._image_projector(geom, fg),
                        src_c_img,
                        src_t_img,
                        mirror=True,
                        eta=eta_cls,
                        keep=keep,
                    )
            classes.append(
                (
                    keep,
                    rows,
                    k_cls,
                    eta_cls,
                    fg,
                    self._class_fill_serves(geom, k_cls, eta_cls, fg),
                    near_free,
                    near_img,
                )
            )
        transmitted = []
        if not crossing:
            # See `_assemble_mixed_contribs` for the sign: ADDED (momwire#1159).
            for src_keep, rows, obs_below in (
                (seg_below, ~entry_below, False),
                (~seg_below, entry_below, True),
            ):
                if src_keep.any() and rows.any():
                    transmitted.append((src_keep, rows, obs_below))

        # A class the fused fill does not serve is filled the whole-plane way
        # on the band — every entry of it against every source, the class's
        # quadrant cut out after — on the numpy loop's own blocks (see
        # `_segment_bands`), which is the whole fill's fallback row for row.
        align = None
        if not all(c[5] for c in classes):
            align = _fill_block(n_segs, nq, self.n_qp_const)
        G = np.zeros((n_basis, n_basis), dtype=np.complex128, order="F")
        acc = self._band_rows(ctx, G)
        for m0, m1, e0, e1 in self._segment_bands(ctx, n_basis, n_segs, 3, align=align):
            if e1 == e0:
                continue
            self._checkpoint()
            entries = np.arange(e0, e1, dtype=np.int64)
            band = tuple(
                np.zeros((e1 - e0, n_segs), dtype=np.complex128) for _ in range(3)
            )
            for keep, rows, k_cls, eta_cls, fg, serves, near_free, near_img in classes:
                ent = entries[rows[e0:e1]]
                if ent.size == 0:
                    continue
                cols = np.nonzero(keep)[0]
                if serves:
                    view = self._sub_view(
                        ctx, ent, n_segs, src_keep=keep, rem_cols="sources"
                    )
                    brow = np.arange(ent.size)
                else:
                    view = self._sub_view(
                        ctx,
                        entries,
                        n_segs,
                        segs=np.arange(m0, m1),
                        rem_cols=cols,
                        rem_obs=keep[m0:m1],
                    )
                    brow = ent - e0
                block = self._tested_contribs(
                    geom,
                    k_cls,
                    view.ctx,
                    _plain_projection,
                    eta=eta_cls,
                    cls=view,
                    near=near_free,
                )
                if fg is not None:
                    self._fold_ground_block(
                        geom,
                        k_cls,
                        view.ctx,
                        block,
                        fg,
                        eta=eta_cls,
                        cls=view,
                        near=near_img,
                    )
                quadrant = np.ix_(ent - e0, cols)
                own = np.ix_(brow, view.col_of_seg[cols])
                for dest, b in zip(band, block):
                    dest[quadrant] = b[own]
                block = view = b = None
            for src_keep, rows, obs_below in transmitted:
                ent = entries[rows[e0:e1]]
                if ent.size == 0:
                    continue
                segs = np.unique(m_of_entry[ent])
                obs_keep = np.zeros(n_segs, dtype=bool)
                obs_keep[segs] = True
                small, _obs_rows, idx = self._transmitted_block(
                    geom,
                    medium,
                    plan,
                    src_keep,
                    obs_keep,
                    obs_below,
                    ctx["obs_c"],
                    ctx["obs_t"],
                    row_group=nq,
                    cos_shape="cos-1",
                )
                m_loc = np.searchsorted(segs, m_of_entry[ent])
                quadrant = np.ix_(ent - e0, idx)
                for dest, sb in zip(band, small):
                    dest[quadrant] += self._tested_contrib_rows(
                        w_entry[ent], m_loc, nq, sb.reshape(segs.size, nq, idx.size)
                    )
                small = sb = None
            self._scatter_band(acc, e0, e1, band)
            band = dest = None
        assert not acc.carry, "a basis row was left unfinished (momwire#1224)"
        return G

    def _scatter_coef_product(self, ctx, contribs):
        """Σ_shape T[shape] @ M[shape] — the (n_basis, n_basis) matrix a triple
        of (nnz, N) tested contributions assembles to.

        Factored out of `_assemble_Z` so momwire#299's end-bracket correction,
        which is a triple of exactly that shape, reaches G through the same
        product rather than a second spelling of it.
        """
        N = ctx["N"]
        i_of_entry = ctx["i_of_entry"]
        m_of_entry = ctx["m_of_entry"]
        n_basis = N + self._n_extra_cols()
        # Source-side coefficient values: the SAME per-entry coefficients the
        # collocation path builds, re-paired to the folded shapes.
        coefs = (ctx["sigAC"], ctx["B"], ctx["sigC"])

        # Both factors of the product carry exactly one nonzero per support
        # entry e: the scatter T[shape] = R @ contrib[shape] with R[i(e),e]=1,
        # and M[shape][m(e), i(e)] = the entry's own coefficient. The CSR is
        # segment-major with one entry per (segment, basis) pair, so a basis's
        # entries name DISTINCT segments — a junction-port column too, whose
        # entries are its K distinct member segments — and no (m, i) cell is
        # written twice. Dense pays 3·n_basis·N² in the matmuls alone, plus an
        # nnz-way `np.add.at`; sparse pays 3·nnz·(N + n_basis) with nnz ≈ 3N.
        # Measured at N=2401 (M1-M3 dipole): 2.75 s → 0.79 s, and the tail of
        # `_assemble_Z` 3.10 s → 0.87 s. Threshold is the point-matched
        # `_assemble_Z`'s (`sinusoidal.py`), re-measured for this product —
        # below N≈57 the scipy constructors cost more than the BLAS call they
        # replace. The two paths sum the same terms in a different order, so
        # they agree to reassociation level rather than bit-exactly; on the
        # folded shapes above that sum is well-scaled, so the two spellings
        # now land ~1e-16 apart instead of the ~1e-13 the literal const/cos
        # pair cost (#203).
        if N < _DENSE_ASSEMBLY_THRESHOLD:

            def _scatter(contrib):
                T = np.zeros((n_basis, N), dtype=np.complex128)
                np.add.at(T, i_of_entry, contrib)
                return T

            def _coef_matrix(coef):
                M = np.zeros((N, n_basis), dtype=np.complex128)
                M[m_of_entry, i_of_entry] = coef
                return M

            G = sum(
                _scatter(contrib) @ _coef_matrix(coef)
                for contrib, coef in zip(contribs, coefs)
            )
        else:
            nnz = i_of_entry.shape[0]
            R = scipy.sparse.csr_matrix(
                (np.ones(nnz), (i_of_entry, np.arange(nnz))), shape=(n_basis, nnz)
            )
            mi = (m_of_entry, i_of_entry)
            G = sum(
                (R @ contrib) @ scipy.sparse.csc_matrix((coef, mi), shape=(N, n_basis))
                for contrib, coef in zip(contribs, coefs)
            )
        return G

    def _ek_bracket_correction_tested(self, G, geom, k, ctx, fg, eta=None):
        """momwire#299's end-bracket correction, assembled and SYMMETRIZED
        into G: `G −= ½(C + Cᵀ)`.

        The blocks below mirror `_fold_ground_block`'s dispatch exactly —
        free space, then the image block with the projector and the signed
        coefficient the fold gives it — because the bracket rides every one of
        them for the same reason the delta does. The Sommerfeld REMAINDER
        carries no delta (#287) and so has no bracket to take off.

        Since momwire#397 unit 3 "mirrors exactly" is structural rather than
        maintained: the image row reads the same `fg` the fold read, so its
        three-way string branch collapses to `−fg.image_coefficient` on the
        one image block (1 for PEC and refl-coef, C2 for sommerfeld — the
        three scales it used to spell out) with `fg.projector`'s choice of
        weight. A ground added to the object appears here with no edit, which
        is the half of criterion 1's acceptance test that is NOT the fill.

        Why it is applied here and not to the fill's contributions
        ----------------------------------------------------------
        The cap is a boundary term of an integration by parts in the SOURCE
        variable, so it is a source-sided spelling of a two-sided object: for
        the pair (m, n) it removes n's cap and for (n, m) it removes m's, and
        those are different quantities. Left as spelled it puts an asymmetry
        into G the size of the whole EK correction — measured
        ‖G−Gᵀ‖/‖G‖ = 1.6e-3 on the L at a = 0.02 against the reduced fill's
        6.2e-11 — and reciprocity is this solver's own error detector (G-B2),
        the very property #246 refused NEC's per-source-end gating to protect.
        Halving C with its transpose costs nothing that matters: the DIVERGENT
        content of C is the node cap, which is κ_P·f_i(P)·f_j(P) — a rank-one
        outer product in the two sides' current at the node, hence symmetric —
        so ½(C + Cᵀ) removes exactly the same O(1/a) term and averages only
        C's O(a²) asymmetric remainder. G-D4's
        `test_gd4_the_bracket_correction_diverges_symmetrically` measures
        ‖C−Cᵀ‖/‖C‖ collapsing like a² while ‖C‖ grows exactly like 1/a, which
        is that claim.

        No-op with the extended kernel off, and on any geometry whose nodes all
        pass the predicate (every straight deck, and every deck in G-B4/G-C/G-S
        — which is what keeps their numbers bit-identical).

        Residency: neither C nor its triple is ever whole (momwire#355)
        ---------------------------------------------------------------
        Spelled literally this built its own (nnz, N) triple — a second copy of
        the fill's largest object, live while the fill's own was still named in
        `_assemble_Z` — and then scattered it. Two facts take that away without
        moving a single float:

        * The bracket can only ever write the source columns that HAVE a bad
          end (`_ek_reduced_ends`), so the triple is allocated over those
          columns alone. Every column it drops was identically zero, and
          dropping exact zeros out of the scatter's sums and the coefficient
          product's is exact in float64 — `x + 0` is `x` under any
          association, so this is bit-equality rather than agreement. On the
          decks the correction actually fires for that is a handful of columns
          out of N: the bad nodes are the bends, and a mesh is mostly straight.
        * What survives the columns is streamed into the SCATTER rather than
          accumulated first. The blocks band over test segments on the fill's
          own block size (`_fill_block`), each band's rows are folded into
          T[shape] = R @ corr[shape] the moment every block has written them,
          and the band buffer dies. Banding over TEST segments (not source
          columns, and not per block) is what keeps it bit-exact: a matrix
          cell's writers are its own test segment's entries, so a cell is
          finished inside one band and the scatter (`_row_scatter`)
          reaches it in the same ascending-entry order the whole-triple
          scatter did. Folding per
          BLOCK instead — the other shape momwire#355 floated — would have
          re-associated the free-space and image writes into G, which is not
          the same float64 sum.

        The symmetrization still needs all of C, and gets it: C is (n_basis,
        n_basis), the size of the answer, not of a fill.
        """
        if not self.extended_kernel:
            return
        blocks = [(_plain_projection, None, None, False, 1.0)]
        if fg is not None:
            src_c_img, src_t_img = fg.image_sources()
            blocks.append(
                (
                    self._image_projector(geom, fg),
                    src_c_img,
                    src_t_img,
                    True,
                    -fg.image_coefficient,
                )
            )
        nnz, N = ctx["w_entry"].shape[0], ctx["N"]
        plans = self._ek_bracket_plans(geom, ctx, blocks)
        if not plans:
            return
        cols = np.unique(np.concatenate([p.cols for p in plans]))
        col_of = np.full(N, -1, dtype=np.int64)
        col_of[cols] = np.arange(cols.size)
        plans = [p._replace(col_of=col_of) for p in plans]

        starts = ctx["starts"]
        i_of_entry = ctx["i_of_entry"]
        n_basis = N + self._n_extra_cols()
        T = tuple(np.zeros((n_basis, cols.size), dtype=np.complex128) for _ in range(3))
        # Test segments per band, capped twice: by the byte budget, and by the
        # scatter the band folds into — a band buffer bigger than T would be
        # streaming into something it dwarfs, which is how a deck whose nodes
        # are ALL bends (cols = N) would otherwise slab the whole triple again
        # and pay for T on top of it.
        per_seg = 48 * cols.size * max(1, nnz // N)
        blk = min(
            max(1, _EK_BRACKET_BAND_BYTES // per_seg),
            max(1, (n_basis * N) // nnz),
        )
        for m0 in range(0, N, blk):
            m1 = min(m0 + blk, N)
            e0, e1 = starts[m0], nnz if m1 == N else starts[m1]
            corr = tuple(
                np.zeros((e1 - e0, cols.size), dtype=np.complex128) for _ in range(3)
            )
            for plan in plans:
                self._ek_bracket_block(geom, k, ctx, corr, plan, m0, m1, eta=eta)
            if not any(np.any(c) for c in corr):
                continue
            rows = i_of_entry[e0:e1]
            for dest, c in zip(T, corr):
                # The scatter's own accumulation, reached one band early.
                # `_row_scatter` adds the band's rows one at a time in
                # ascending entry order, which is the order `R @ corr` sums a basis
                # row's entries in — and the bands are ascending too, so every
                # T cell sees exactly the sequence of additions the
                # whole-triple product performed.
                _row_scatter(dest, rows, c)
        if not any(np.any(t) for t in T):
            return
        if not self._band_fill_serves(N):
            C = self._bracket_coef_product(ctx, T, cols, col_of)
            G -= 0.5 * (C + C.T)
            return
        # C is (n_basis, n_basis), the size of the answer, and so is C + Cᵀ
        # and its half: three matrices beside G (momwire#1224). Taken by row
        # bands instead — rows i0:i1 of C and the same COLUMNS of C, whose
        # transpose is those rows of Cᵀ — each cell is the product's own sum
        # (a row or column slice of the dense factor or of the csc factor
        # leaves every cell's terms and their order alone), and the update
        # `G − ½(C + Cᵀ)` is elementwise, so the bands are the same bits.
        mats = self._bracket_coef_mats(ctx, cols, col_of)
        step = max(1, self._band_budget_bytes(n_basis) // (16 * 6 * n_basis))
        for i0 in range(0, n_basis, step):
            i1 = min(i0 + step, n_basis)
            c_rows = sum(t[i0:i1] @ M for t, M in zip(T, mats))
            c_cols = sum(t @ M[:, i0:i1] for t, M in zip(T, mats))
            G[i0:i1] -= 0.5 * (c_rows + c_cols.T)

    def _ek_bracket_plans(self, geom, ctx, blocks):
        """Per-block prep for `_ek_bracket_block` that does NOT depend on the
        test band: the bad-end masks, the axis labels, the near set and the
        graded rule.

        Hoisted out of the block itself because the band loop calls it once
        per (band, block) and every one of these is a whole-geometry query —
        `_near_pairs` in particular is the near set of the entire fill, which
        no band narrows. Blocks with no bad end at all are dropped here rather
        than returned and skipped, so a straight deck leaves the caller with
        nothing to allocate.
        """
        N = ctx["N"]
        a_all = self._seg_radius(geom)
        xg, wg = _graded_endpoint_rule(
            float(np.min(a_all / ctx["hh"])), self.n_qp_near, self._leggauss_cached
        )
        plans = []
        for projector, src_c, src_t, mirror, scale in blocks:
            bad_lo, bad_hi = self._ek_reduced_ends(geom, mirror)
            if not (bad_lo.any() or bad_hi.any()):
                continue
            group_obs, group_src = self._ek_axis_labels(geom, mirror)
            src_c = geom["seg_centers"] if src_c is None else src_c
            src_t = geom["seg_tangents"] if src_t is None else src_t
            # The near set is the fill's own, selected against the same source
            # geometry the caller filled with, so "which rule did this cell
            # get" is answered by the same predicate that decided it.
            if self.near_correction:
                mm, nn = self._near_pairs(geom, src_c=src_c, src_t=src_t)
                near_key = mm * N + nn
            else:
                near_key = np.empty(0, dtype=np.int64)
            plans.append(
                _EKBracketPlan(
                    projector=projector,
                    src_c=src_c,
                    src_t=src_t,
                    scale=scale,
                    bad_lo=bad_lo,
                    bad_hi=bad_hi,
                    group_obs=group_obs,
                    group_src=group_src,
                    near_key=near_key,
                    cols=np.flatnonzero(bad_lo | bad_hi),
                    col_of=None,
                    xg=xg,
                    wg=wg,
                )
            )
        return plans

    def _bracket_coef_product(self, ctx, T, cols, col_of):
        """Σ_shape T[shape] @ M[shape] for the end-bracket's COLUMN-RESTRICTED,
        already-scattered triple — `_scatter_coef_product`'s second half, on a
        source axis that runs over `cols` instead of all N segments.

        The scatter half is not repeated here because the caller already did
        it band by band (see `_ek_bracket_correction_tested`). What is left is
        the same product against the same per-entry source coefficients, with
        the entries whose segment is not a retained column struck out: their
        T column is identically zero, so the terms they contributed to every
        C cell were exact zeros and removing them is exact.
        """
        N = ctx["N"]
        n_basis = N + self._n_extra_cols()
        coefs = (ctx["sigAC"], ctx["B"], ctx["sigC"])
        row = col_of[ctx["m_of_entry"]]
        sel = row >= 0
        mi = (row[sel], ctx["i_of_entry"][sel])

        if N < _DENSE_ASSEMBLY_THRESHOLD:

            def _coef_matrix(coef):
                M = np.zeros((cols.size, n_basis), dtype=np.complex128)
                M[mi] = coef[sel]
                return M

            return sum(t @ _coef_matrix(coef) for t, coef in zip(T, coefs))
        return sum(t @ M for t, M in zip(T, self._bracket_coef_mats(ctx, cols, col_of)))

    def _bracket_coef_mats(self, ctx, cols, col_of):
        """`_bracket_coef_product`'s three sparse factors, as (cols.size,
        n_basis) csc matrices — the one construction both the whole product
        and the banded update (momwire#1224) multiply by."""
        N = ctx["N"]
        n_basis = N + self._n_extra_cols()
        row = col_of[ctx["m_of_entry"]]
        sel = row >= 0
        mi = (row[sel], ctx["i_of_entry"][sel])
        return tuple(
            scipy.sparse.csc_matrix((coef[sel], mi), shape=(cols.size, n_basis))
            for coef in (ctx["sigAC"], ctx["B"], ctx["sigC"])
        )

    def _contact_charge_correction_tested(self, G, geom, k, seg_view, ctx, eta=None):
        """#282's ground-contact charge correction, test-integrated.

        Identical physics to the point-matched
        `SinusoidalSolver._contact_charge_correction` — the residual charge
        a wire end lying in a FINITE ground plane leaves behind, which is
        double-counting and diverges under refinement — through this
        solver's own testing scheme: the same per-observer kernel evaluated
        at the test quadrature points and reduced against the test
        functions, exactly like every other source block here.

        The correction is a source-side term (it belongs to the basis whose
        current reaches the plane), so it lands on that basis's COLUMN;
        the test side is the ordinary Galerkin integral over the real wire.
        No-op without a finite ground or without a contact.

        momwire#292's extended-kernel amendment rides here too, and rides
        through the test integration unchanged — it is one more per-observer
        kernel value, reduced against the same test functions. What is
        specific to this solver is WHERE it applies: `_contact_ek_masks`
        scores #246's per-pair eligibility at every quadrature point rather
        than reading NEC's per-end IND code.
        """
        if self.ground_eps is None:
            return G
        nodes = self._contact_nodes(geom)
        if not nodes:
            return G
        # The charge kernel's whole variation lives within a wire radius of
        # the node — the same width-`a` endpoint spike M2's near correction
        # exists for — so the uniform test rule cannot see it and the
        # ENDPOINT-GRADED rule is used instead, over every test segment
        # (the correction is one rank-one update per contact, so paying the
        # richer rule everywhere costs nothing measurable).
        N = ctx["N"]
        seg_c, seg_t = geom["seg_centers"], geom["seg_tangents"]
        hh = ctx["hh"]
        a_all = self._seg_radius(geom)
        m_of_entry, i_of_entry = ctx["m_of_entry"], ctx["i_of_entry"]
        xg, wg = _graded_endpoint_rule(
            float(np.min(a_all / hh)), self.n_qp_near, self._leggauss_cached
        )
        nq = xg.shape[0]
        xi = hh[:, None] * xg[None, :]  # (N, nq)
        obs_c = (seg_c[:, None, :] + xi[:, :, None] * seg_t[:, None, :]).reshape(-1, 3)
        obs_t = np.broadcast_to(seg_t[:, None, :], (N, nq, 3)).reshape(-1, 3)
        a_obs = np.repeat(a_all, nq)
        fval = _basis_value(
            ctx["sigAC"][:, None],
            ctx["B"][:, None],
            ctx["sigC"][:, None],
            k,
            xi[m_of_entry],
        )
        w_entry = (wg[None, :] * hh[m_of_entry][:, None]) * fval
        obs_seg = np.repeat(np.arange(N, dtype=np.int64), nq)
        for i, sgn, node in nodes:
            R = self._contact_charge_kernel(geom, k, node, obs_c, obs_t, a_obs, eta=eta)
            masks = self._contact_ek_masks(geom, i, sgn, obs_seg)
            if masks is not None:
                R = R + self._contact_charge_ek_delta(
                    geom, k, i, sgn, node, obs_c, obs_t, a_obs, *masks, eta=eta
                )
            R_t = self._tested_contrib_rows(
                w_entry, m_of_entry, nq, R.reshape(N, nq, 1)
            )[:, 0]
            T = np.zeros(G.shape[0], dtype=np.complex128)
            np.add.at(T, i_of_entry, R_t)
            jb, val = self._contact_node_values(geom, k, seg_view, i, sgn)
            G[:, jb] -= sgn * T[:, None] * val[None, :]
        return G

    def _near_cells(
        self,
        geom,
        k,
        ctx,
        projector,
        src_c=None,
        src_t=None,
        mirror=False,
        eta=None,
        keep=None,
    ):
        """One source block's graded near values over the WHOLE deck, as a
        `_NearCells` (momwire#1224) — empty when it has no near pair.

        The banded fill's hoist of `_apply_near_correction`: that method
        selects its pairs over the whole deck on every call, and an observer
        band that called it would pay the whole deck's selection per band.
        So the values are computed ONCE, by that method's own loop (its
        `collect` mode — same pairs, same blocks, same einsum per cell), and
        each band places the cells it holds (`_near_correct`). `keep` is a
        mixed deck's pair class, filtering the pairs exactly as a class view
        does."""
        src_c = geom["seg_centers"] if src_c is None else src_c
        src_t = geom["seg_tangents"] if src_t is None else src_t
        got = []
        self._apply_near_correction(
            geom,
            k,
            ctx,
            None,
            projector,
            src_c,
            src_t,
            mirror,
            eta=eta,
            keep=keep,
            collect=got,
        )
        if not got:
            # Empty rather than None: None means "compute it here" to
            # `_near_correct`, which would re-select over the whole deck.
            none = np.zeros(0, dtype=np.int64)
            return _NearCells(
                none, none, tuple(np.zeros(0, dtype=np.complex128) for _ in range(3))
            )
        return _NearCells(
            np.concatenate([g[0] for g in got]),
            np.concatenate([g[1] for g in got]),
            tuple(np.concatenate([g[2][t] for g in got]) for t in range(3)),
        )

    def _apply_near_correction(
        self,
        geom,
        k,
        ctx,
        contribs,
        projector,
        src_c,
        src_t,
        mirror=False,
        sub=False,
        eta=None,
        cls=None,
        keep=None,
        collect=None,
    ):
        """Recompute the near-pair test integrals on the endpoint-graded rule,
        overwriting the uniform-rule values in `contribs` (M2).

        `collect` (a list, momwire#1224) writes nothing: each block's cells
        are appended to it as (entry, source segment, (c, s, co) values) in
        full-deck coordinates, which is how `_near_cells` precomputes them
        once for the banded fill. `keep` restricts the pairs to one mixed
        deck's pair class, by the same test that `cls` applies.

        With `cls` (momwire#1224) `contribs` is one pair class's block: the
        pairs are still selected over the WHOLE deck and computed from the
        whole deck's context — the graded rule's width is the whole model's
        tightest segment, and it must stay that — and only the pairs whose
        test and source segments are both in the class are kept, their cells
        written at the class block's own (row, column).

        Each (entry, source-segment) cell is owned by exactly one near pair —
        the entry fixes the test segment — so the overwrite is an assignment,
        not an accumulation.

        `sub` is `_tested_contribs`' fold mode (momwire#332): `contribs` is
        then the caller's free-space triple rather than this block's own, so
        the graded value is SUBTRACTED off what is already in the cell instead
        of replacing it, and the cells written come back as a flat
        `(entry, source segment)` index pair so the far half can skip them.
        Same float64 subtraction per entry as differencing two whole triples.

        Runs per source block (M4): the free-space block selects its near
        pairs against the real segments, an image block against the MIRRORED
        ones, so a wire touching the plane gets its segment↔own-image spike
        corrected too. The blocks are separate arrays, so the assignments
        never collide.

        The extended kernel reaches here too, and it matters more here than
        anywhere: the near set is the self and node-sharing pairs, which are
        both the coaxial ones the pair rule admits and the ones at the smallest
        R, where the O(a²/R²) delta is largest. Because this path OVERWRITES
        rather than accumulates, an EK-on far fill with a reduced near
        correction would quietly throw the delta away again on exactly those
        pairs.
        """
        mm, nn = self._near_pairs(geom, src_c=src_c, src_t=src_t)
        if cls is not None:
            # A subsequence of the whole plane's pairs, in the same order; a
            # pair's cells depend on nothing but the pair (the block loop
            # below is elementwise across pairs).
            in_cls = (cls.col_of_seg[mm] >= 0) & (cls.col_of_seg[nn] >= 0)
            mm, nn = mm[in_cls], nn[in_cls]
            ctx = cls.full_ctx
        elif keep is not None:
            in_cls = keep[mm] & keep[nn]
            mm, nn = mm[in_cls], nn[in_cls]
        if mm.size == 0:
            return None

        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        hh = ctx["hh"]
        a_seg = ctx["a_seg"]
        starts, counts = ctx["starts"], ctx["counts"]
        sigAC, B, sigC = ctx["sigAC"], ctx["B"], ctx["sigC"]

        # Feature width relative to the half-length, taken at the tightest
        # segment in the model so one shared template resolves them all.
        a_all = self._seg_radius(geom)
        xg, wg = _graded_endpoint_rule(
            float(np.min(a_all / hh)), self.n_qp_near, self._leggauss_cached
        )

        # Flatten (pair, support-entry-of-its-test-segment) to a run-length
        # layout: entries of pair p occupy cum[p]:cum[p+1].
        cnt = counts[mm]
        cum = np.concatenate(([0], np.cumsum(cnt)))
        pair_of = np.repeat(np.arange(mm.size), cnt)
        entry_of = (
            np.arange(cum[-1]) - np.repeat(cum[:-1], cnt) + np.repeat(starts[mm], cnt)
        )

        if collect is None:
            contrib_c, contrib_s, contrib_co = contribs
        # Pairs per block from the byte budget rather than a flat 512
        # (momwire#383): under the extended kernel the block holds
        # `_folded_ek_delta_fields`' (P, G, n_d) quadrature, 4.9 MB per pair
        # at this rule, and 512 of those was 2.6 GB of fixed working set.
        blk = _near_block(xg.shape[0], self.n_qp_const, self.extended_kernel)
        # Pooled (collecting for the banded fill, perf item 9), up to `live`
        # blocks run at once. Their scratch together has to fit where the
        # fill has room for it: the near cells are collected before any
        # observer band exists, so a band's budget (`_band_budget_bytes`) is
        # free then, and `live` blocks of `_NEAR_WORKSPACE_BYTES` each stay
        # inside it — the fill's peak does not move. A deck whose band budget
        # is under two blocks collects serially.
        live = 1
        if collect is not None and _NEAR_THREADS:
            band = self._band_budget_bytes(int(ctx["N"]) + self._n_extra_cols())
            live = max(
                1, min(_piece_pool()._max_workers, band // _NEAR_WORKSPACE_BYTES)
            )

        def block(p0):
            p1 = min(p0 + blk, mm.size)
            mi, ni = mm[p0:p1], nn[p0:p1]

            # (P, G, 3) observer points along each test segment.
            xi = hh[mi][:, None] * xg[None, :]  # (P, G)
            obs = seg_c[mi][:, None, :] + xi[:, :, None] * seg_t[mi][:, None, :]
            cm = self._field_components_bcast(
                k,
                obs_c=obs,
                obs_t=seg_t[mi][:, None, :],
                a=self._uniform_radius if a_seg is None else a_seg[mi][:, None],
                src_c=src_c[ni][:, None, :],
                src_t=src_t[ni][:, None, :],
                src_hh=hh[ni][:, None],
                cos_shape="cos-1",
                eta=eta,
                # (P, 1) against the (P, G) field tables: one eligibility
                # decision per PAIR, shared by that pair's graded observers.
                # These are the pairs whose observers sit on the source
                # segment, so they take the DENSE delta rule (see `_ek_pairs`).
                ek=(
                    self._ek_pairs(
                        geom,
                        mi[:, None],
                        ni[:, None],
                        mirror,
                        n_panels=_N_PANEL_EK_DELTA_NEAR,
                    )
                    if self.extended_kernel
                    else None
                ),
            )
            # (P, 1) pair indices broadcast against the (P, G) field tables.
            Phi = projector(cm, mi[:, None], ni[:, None])  # each (P, G)

            e0, e1 = cum[p0], cum[p1]
            ei = entry_of[e0:e1]  # into the flat seg_view arrays
            lp = pair_of[e0:e1] - p0  # into this block's pair axis
            xi_e = xi[lp]  # (E, G)
            # Same folded evaluation `_test_context` uses for the uniform
            # rule — the near pairs' weights have to be the SAME function of
            # ξ, or the overwrite would leave the row inconsistent (#203).
            fval = _basis_value(
                sigAC[ei][:, None], B[ei][:, None], sigC[ei][:, None], k, xi_e
            )
            w = (wg[None, :] * hh[mi][lp][:, None]) * fval  # (E, G)

            col = ni[lp]
            # The block's tables ride along to the caller, which holds them
            # until the next block exists, as the serial loop's own locals
            # always lived: freed at the block's end instead, they leave the
            # top of the heap free, glibc trims it, and the next block
            # page-faults it back (measured: 5x the minor faults and +30 %
            # on the near cells of free x32).
            return ei, col, w, lp, Phi, (xi, obs, cm, xi_e, fval)

        starts_p = range(0, mm.size, blk)
        if collect is not None:
            # Collecting, a block's cells are a function of its pairs alone
            # and nothing is written, so the blocks run on the shared pool
            # (perf item 9) and are appended in block order. Each block runs
            # the serial loop's expressions on the same arrays — the same
            # blocks, so numpy takes the same loop for every product
            # (`_field_components_bcast`'s momwire#392 note).
            # `_NEAR_THREADS = False` is the serial loop
            # (tests/test_sg_near_threads_1290.py).
            def cells(p0):
                ei, col, w, lp, Phi, _tables = block(p0)
                return ei, col, tuple(np.einsum("eg,eg->e", w, Ph[lp]) for Ph in Phi)

            # The cancel poll stays on this thread, once per block before
            # it is handed out; at most `live` blocks are in flight, and the
            # cells come back in block order.
            pending = collections.deque()
            held = None
            for p0 in starts_p:
                self._checkpoint()  # per block of near pairs
                if live < 2:
                    held = block(p0)
                    ei, col, w, lp, Phi, _tables = held
                    collect.append(
                        (ei, col, tuple(np.einsum("eg,eg->e", w, Ph[lp]) for Ph in Phi))
                    )
                    continue
                if len(pending) >= live:
                    collect.append(pending.popleft().result())
                pending.append(_piece_pool().submit(cells, p0))
            while pending:
                collect.append(pending.popleft().result())
            return None
        for p0 in starts_p:
            self._checkpoint()  # per block of near pairs
            ei, col, w, lp, Phi, _tables = block(p0)
            row = ei
            if cls is not None:
                row, col = cls.row_of_entry[ei], cls.col_of_seg[col]
            for contrib, Ph in zip((contrib_c, contrib_s, contrib_co), Phi):
                val = np.einsum("eg,eg->e", w, Ph[lp])
                if sub:
                    contrib[row, col] -= val
                else:
                    contrib[row, col] = val

        if not sub:
            return None
        if cls is None:
            return entry_of, nn[pair_of]
        return cls.row_of_entry[entry_of], cls.col_of_seg[nn[pair_of]]

    # ------------------------------------------------------------------
    # Galerkin-tested source vector + solve
    # ------------------------------------------------------------------

    def _feed_placed_at_request(self):
        """Under ``feed_model="point"`` the gap sits where it was named, with
        the snap's remainder carried in ``feed_xi`` (momwire#648); under
        ``"segment"`` it is the snapped segment, as in the parent."""
        return self.feed_model == "point"

    def _drive_columns(self, geom, seg_view, k):
        """Unit-voltage Galerkin excitation column per port, (N+P, n_ports),
        ordered [gap feeds…, junction ports…, node ports…].

        Gap feed j, `feed_model="segment"`: b_i = -∫ f_i(s)·ŝ·E^app(s)
        ds with the delta-gap applied field E^app = 1/Δ_m along +ŝ_m on feed
        segment m and zero elsewhere, so only that segment's support entries
        contribute:

            ∫_{seg m} f_{i,m}(ξ) dξ = σA·Δ_m + σC·(2/k)·sin(kΔ_m/2)

        (the sin term integrates to zero by parity). Contrast the collocation
        RHS, which point-samples -1/Δ_m at the feed segment centre. This
        integral is exact, so M2's quadrature work does not touch it.

        It carries the #203 cancellation too — both terms are O(Δ) and the
        integral is O(Δ·(kΔ)²) — so it is evaluated on the folded shapes as
        well:

            ∫ f dξ = σ(A+C)·Δ_m + σC·(2/k)·(sin u − u),  u = kΔ_m/2,

        with `sin u − u` taken from its series rather than from a float64
        subtraction (`_sin_minus_arg`). That is the RHS, not G, so no gate
        below moves on it; it is folded because leaving one consumer of the
        pair unfolded is how the defect comes back.

        Gap feed j, `feed_model="point"` (the default since momwire#654):
        E^app = δ(s − s0) at the feed segment's CENTRE, so the test integral
        collapses on the delta and the
        column is just -f_i(s0) = -σ(A+C) — sin(k·0) = 0 kills the B shape and
        cos(k·0) − 1 kills the third, leaving the one folded coefficient
        `AC` the centre readout already reads. Drive and readout are then the
        same functional, which is the duality the class docstring states; the
        1/Δ_m of the segment model is absent because the source no longer
        spreads. `AC` rather than `A + C` for the #203 discipline — and since
        momwire#606 the two are no longer even equal: `AC` is a per-branch
        closed form, while the float sum loses 8ε/(kΔ)² relative and has no
        correct digits below kΔ ≈ 1e-5. What was a stylistic preference for
        one spelling of the pair is now the difference between a right and a
        wrong drive vector.

        Junction port p: the source is an EMF in the infinitesimal lead
        between the port terminal and the node, so testing it against basis
        w gives -1 × (w's net current THROUGH that lead) = -1 × (w's net
        inflow at the node). Every ordinary basis has zero net inflow there
        — that is #177's KCL identity, the very property that makes the port
        impossible in the point-matched solver — and `g_q` has δ_pq by
        construction. So the whole column is a single -1 at row N+p: the
        port's excitation touches exactly the port's own row, exactly.

        Node port p (M5b formulation (a)): the source is a ZERO-width delta
        gap sitting exactly at the junction node, E^app = V·δ(s − s_node)
        along the arc that runs from the port's + side to its − side. The
        Galerkin test integral collapses on the delta, so

            b_i = -∫ f_i(s)·ŝ·E^app(s) ds = -V·f_i(node)

        with f_i(node) the through-current `_node_cut_vectors` builds. No
        basis column, no node charge — the whole port is one RHS column plus
        its exact dual readout.
        """
        N = geom["n_segs"]
        h = np.asarray(geom["seg_h"], dtype=float)
        starts = seg_view["starts"]
        n_basis = N + self._n_extra_cols()
        U = np.zeros((n_basis, self.n_ports), dtype=np.complex128)
        for j, fseg in enumerate(geom["feed_segs"]):
            s, e = starts[fseg], starts[fseg + 1]
            sig = seg_view["sigma"][s:e]
            if self.feed_model == "point":
                # -f_i(s0), evaluated at the gap's own position rather than at
                # the segment's centre (momwire#648). A caller that names a
                # KNOT gets `feed_xi` = ±h/2 and the other two shapes carry
                # the difference; a caller that names a segment CENTRE gets 0
                # and this is bit-identical to the expression the branch used
                # to be, σ·AC, since sin(0) = 0 and cos(0) − 1 = 0.
                #
                # Bit-identical for the NEC-2 front end, which builds that
                # arclength the way `_build_geometry` does. NOT for a caller
                # that spells the same centre by another route: antennaknobs
                # sums whole edge lengths and halves the last one, which
                # differs from `cumsum(h) − h/2` in the last bits, so `feed_xi`
                # comes out at ULP scale rather than at zero — measured
                # −2.7e-15 m on a 10 m wire at N=41, ξ/h ≈ 1e-14. That is
                # accurate, not identical: the shapes are evaluated at the
                # position asked for, and the O(kξ) the other two terms pick
                # up is 1e-14 of the column. See tests/test_feed_snap_623.py,
                # which measures what tracking a gap that finely is worth.
                col = -_basis_value(
                    sig * seg_view["AC"][s:e],
                    seg_view["B"][s:e],
                    sig * seg_view["C"][s:e],
                    _entry_k(seg_view, s, e, k),
                    float(geom["feed_xi"][j]),
                )
            else:
                hm = float(h[fseg])
                ke = _entry_k(seg_view, s, e, k)
                int_f = sig * seg_view["AC"][s:e] * hm + sig * seg_view["C"][s:e] * (
                    2.0 / ke
                ) * _sin_minus_arg(0.5 * ke * hm)
                col = -int_f / hm
            np.add.at(U[:, j], seg_view["jbasis"][s:e], col)
        for p in range(len(self.junction_ports)):
            U[N + p, len(self.feeds) + p] = -1.0
        if self.node_ports:
            n0 = len(self.feeds) + len(self.junction_ports)
            U[:, n0:] = -self._node_cut_vectors(geom, seg_view, k)
        return U

    def _tested_source_vector(self, geom, seg_view, k):
        """Galerkin RHS for the configured port voltages: Σ_ports V·U[:, port].
        See `_drive_columns` for what each column is."""
        U = self._drive_columns(geom, seg_view, k)
        return U @ self._port_voltages()

    def _port_voltages(self):
        return np.array(
            [v for _, _, v in self.feeds]
            + [v for _j, v in self.junction_ports]
            + [v for _j, _s, v in self.node_ports],
            dtype=np.complex128,
        )

    def _with_crossing_wings(self, geom, view, below, medium):
        """A crossing deck's readout view carries the node wings
        `_crossing_wing_view` appends — the air view omits them outright, so
        their amplitudes would never be read (momwire#1159). Every other deck
        is the base's view."""
        if self._is_crossing(geom):
            return self._crossing_wing_view(geom, view, below, medium)
        return view

    def _serves_crossing(self):
        """This family's crossing serve (momwire#980 D3)."""
        return True

    def _port_currents(self, alpha, geom, seg_view, U):
        """Per-port current readout, ordered [gap feeds…, junction ports…,
        node ports…].

        A junction port's readout is `α_{N+p}` — its own basis amplitude,
        which by construction IS the current injected at the node — and that
        equals -U[:, port]·α exactly, so the port block of Y is symmetric to
        machine precision whichever branch below runs.

        A NODE port's readout is the current through the node, -U[:, port]·α,
        which is likewise the exact dual of its drive: the drive column IS the
        through-current functional, so there is no centre-vs-average choice to
        make here and no payoff to trade (contrast the gap feed below). Its
        block of Y is symmetric to machine precision in both branches.

        A gap feed's readout depends on `feed_readout`: the segment-CENTRE
        current (default, the point-matched solver's and NEC's) or the
        gap-averaged current -U[:, j]·α, which is the exact dual of the
        Galerkin drive. See the class docstring for why the non-dual one is
        the default and what it costs.

        Under `feed_model="point"` the two branches COINCIDE: that drive
        column is the centre-evaluation functional itself, so -U[:, j]·α and
        `_feed_segment_current` differ only in summation order (~1e-16
        relative, `test_point_gap_readouts_coincide`) and `feed_readout` stops
        being a choice with consequences at gap feeds.
        """
        N = geom["n_segs"]
        if self.feed_readout == "variational":
            return -(U.T @ alpha)
        n0 = len(self.feeds) + len(self.junction_ports)
        return np.array(
            [
                # `feed_xi` under the point gap only: the drive column is the
                # evaluation functional AT the gap, so the readout has to be
                # the same point or Y stops being symmetric (momwire#648). The
                # segment gap spreads over the whole segment and its centre
                # readout is the NEC convention, unmoved.
                self._feed_segment_current(
                    alpha,
                    seg_view,
                    fi,
                    float(geom["feed_xi"][j]) if self.feed_model == "point" else 0.0,
                )
                for j, fi in enumerate(geom["feed_segs"])
            ]
            + [alpha[N + p] for p in range(len(self.junction_ports))]
            + [-(U[:, n0 + p] @ alpha) for p in range(len(self.node_ports))],
            dtype=np.complex128,
        )

    def compute_impedance(self):
        """Return (Z_drive, alpha). Mirrors `SinusoidalSolver.compute_impedance`
        but assembles the Galerkin matrix and the Galerkin-tested RHS.

        With junction ports, `alpha` is length N+P and `Z_drive` covers
        [gap feeds…, junction ports…]; it stays a bare scalar only when the
        model has exactly one port of any kind.
        """
        self._refuse_junction_port_solve()
        refuse_undriven(self._port_voltages())
        geom = self._build_geometry()
        self._checkpoint()  # after geometry, before the field fill
        # The medium wraps the WHOLE solve, not the matrix alone: the drive
        # columns are built at `k` too (momwire#980 D1).
        with self._operating_medium(geom) as medium:
            G, seg_view = self._assemble_Z_ported(
                geom, self.k, self._medium_eta(medium)
            )
            U = self._drive_columns(geom, seg_view, self.k)
            voltages = self._port_voltages()
            self._checkpoint()  # after assembly, before the dense solve

            alpha = _solve_constrained(
                G,
                U @ voltages,
                self._crossing_continuity_rows(geom, seg_view, self.k),
            )

            # Inside the medium too (momwire#1159): an off-centre point gap's
            # readout writes the shapes at `self.k`, which is k_m only here.
            currents = self._port_currents(alpha, geom, seg_view, U)
        # The backstop behind the up-front refusal (momwire#1162): a port
        # whose current reads exactly zero has no V/I, and is refused by name.
        z_per_port = port_impedances(voltages, currents)
        Z_drive = z_per_port[0] if self.n_ports == 1 else z_per_port
        return Z_drive, alpha

    def compute_y_matrix(self) -> np.ndarray:
        """Short-circuit admittance matrix over [gap feeds…, junction ports…].

        Honestly Galerkin throughout, which the inherited implementation was
        not: it paired this solver's Galerkin matrix with the point-matched
        solver's RHS (a bare -1/h at the feed segment's row), so its Y did
        not even agree with `compute_impedance`. Here the columns are the
        same `_drive_columns` the impedance solve uses and the readout is the
        same `_port_currents`, so 1/Y⁻¹ reproduces `compute_impedance` to
        solver precision by construction.

        This is the `y` field of `compute_port_solution()` and nothing else —
        see there for the per-port solution columns this throws away (#232).
        """
        return self.compute_port_solution().y

    def compute_port_solution(self) -> PortSolution:
        """Solve every port from ONE fill and ONE factorisation.

        Returns a `PortSolution` whose `y` is identical to
        `compute_y_matrix()` and whose `coeffs` column j is the solution for a
        1 V drive at port j with the others shorted. Ports run
        [gap feeds…, junction ports…, node ports…] — `coeffs` therefore has
        N + len(junction_ports) rows, one per segment basis plus one per
        junction-port basis, and the port rows are part of the solution, not
        Lagrange multipliers bolted on.

        Everything basis-specific stays inside: the Galerkin-tested drive
        columns (`feed_model`), the ported assembly with M5b's node-charge
        correction, and the `feed_readout` choice of centre-vs-variational
        current. Junction ports ride the same #191 rules as
        `compute_y_matrix`, and refuse at the same moment with the same
        `NotImplementedError` where the formulation does not cover them
        (finite ground, mixed radii).

        `basis` is stable across the ports of this one solution and NOT
        across solves.
        """
        self._refuse_junction_port_solve()
        geom = self._build_geometry()
        # The WHOLE solve inside the medium, as `compute_impedance` does
        # (momwire#1159): a wholly-buried deck's fill, drive and readout all
        # run at k_m. Outside it this route filled a buried deck through the
        # ABOVE ground family, which refuses it by name; `compute_y_matrix`
        # and both swept loops reach the matrix only through here.
        with self._operating_medium(geom) as medium:
            G, seg_view = self._assemble_Z_ported(
                geom, self.k, self._medium_eta(medium)
            )
            U = self._drive_columns(geom, seg_view, self.k)
            alphas = _solve_constrained(
                G, U, self._crossing_continuity_rows(geom, seg_view, self.k)
            )
            Y = np.stack(
                [
                    self._port_currents(alphas[:, j], geom, seg_view, U)
                    for j in range(self.n_ports)
                ],
                axis=1,
            )
            basis = _SegmentBasis(geom=geom, seg_view=seg_view, k=self.k)
        return PortSolution(
            y=Y,
            coeffs=alphas,
            port_currents=Y,  # the same object: the readout IS the Y matrix
            basis=basis,
        )

    def _port_count(self):
        """Ports `compute_port_solution` returns: [gap feeds…, junction
        ports…, node ports…], from the configuration alone."""
        return self.n_ports

    def _port_solutions_swept(self, k_array):
        """Per-k `PortSolution` generator behind `compute_y_matrix_swept` and
        `compute_port_solution_swept` (#252).

        A per-k loop over `compute_port_solution` — no batched assembly on
        this family, so the swept Y is the stacked single-k Y bit for bit.
        Nothing is hoisted out of the loop because nothing can be: the drive
        columns are k-DEPENDENT here (the gap column integrates the basis
        shapes, which move with k). The refusal is raised up front so an
        unserved configuration is rejected before any solve, even at an empty
        sweep.
        """
        self._refuse_junction_port_solve()
        with self._k_restored():
            for kk in np.asarray(k_array, dtype=float):
                self._checkpoint()
                self._set_k(kk)
                yield self.compute_port_solution()

    def compute_impedance_swept(self, k_array):
        """Driving-point impedance over a batch of wavenumbers — a per-k loop
        over `compute_impedance` itself (the inherited version used the
        collocation RHS, so it disagreed with `compute_impedance`; #252 then
        replaced the local copy of its algebra with the call).

        This family's `compute_impedance` stashes nothing and hoists nothing,
        so driving it per k is the same arithmetic in the same order.
        """
        self._refuse_junction_port_solve()
        # Up front, so an undriven deck refuses even at an empty sweep.
        refuse_undriven(self._port_voltages())
        k_array = np.asarray(k_array, dtype=float)
        n_p = self.n_ports
        z_out = np.zeros(
            k_array.shape[0] if n_p == 1 else (k_array.shape[0], n_p),
            dtype=np.complex128,
        )
        with self._k_restored():
            for i, kk in enumerate(k_array):
                self._checkpoint()
                self._set_k(kk)
                z_out[i], _alpha = self.compute_impedance()
        return z_out
