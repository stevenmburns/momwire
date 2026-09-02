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
sinusoidal shapes, closed-form per segment, in `_apply_loading`.
"""

import collections

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.spatial.distance

from . import _field_ground, _ground_mirror, _wire_loading
from ._accel import acc as _acc
from ._bspline_kernels import _ek_axis_groups
from .bspline import SINGULAR_ENRICHMENT_NOT_YET
from ._port_solution import PortSolution
from .sinusoidal import (
    _BURIED_REFUSAL,
    _DENSE_ASSEMBLY_THRESHOLD,
    _EKPairs,
    _EULER_GAMMA,
    _N_PANEL_EK_DELTA_NEAR,
    _N_QP_EK_DELTA,
    SinusoidalSolver,
    _SegmentBasis,
    _recip_sin_gap,
    _sin_minus_arg,
)

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


def _basis_value(sigAC, B, sigC, k, xi):
    """The three-term basis current at local arc ξ, on the well-scaled shape
    set {1, sin kξ, cos kξ − 1} (stevenmburns/momwire#203):

        f(ξ) = σ(A+C) + B·sin(kξ) − 2σC·sin²(kξ/2)

    — identically the NEC form σA + B·sin(kξ) + σC·cos(kξ), rearranged so that
    no term is larger than the result. In the literal spelling σA and σC·cos
    are O(1) and cancel to O((kΔ)²/8), which costs ε·8/(kΔ)² relative — 3e-13
    at N=41 but 1.2e-10 at N=801 and rising like N², because the basis is
    normalized to its own segment-centre current A+C. Here σ(A+C) is supplied
    by `_basis_coefs` and cos kξ − 1 is spelled −2sin²(kξ/2), so both
    cancellations happen where they are exact and f comes out to full relative
    precision.

    `AC` is a per-branch CLOSED FORM, not the float sum `A + C`
    (stevenmburns/momwire#606). It used to be that sum, described here as
    "correctly rounded to the sum" — true, and not enough: the sum of two
    rounded values is not the rounded value of the sum, and with A and C each
    O(1) carrying an absolute ε against an O((kΔ)²) answer the summed spelling
    is 1 % wrong at kΔ = 2.1e-4 and has no correct digits at all by 1e-5. This
    function was always the accurate SPELLING; #606 is what made the
    coefficient it is handed accurate too.

    Every argument broadcasts: callers supply coefficient columns and an arc
    array in whatever pairing they already hold.
    """
    half = np.sin(0.5 * k * xi)
    return sigAC + B * np.sin(k * xi) - 2.0 * sigC * (half * half)


# The unweighted projector, serving the FREE-SPACE block here and returned by
# `FieldGround.projector` for every ground that has no dyad. It is bound, not
# defined: `_tested_contribs` gates its fused C++ far fill on `projector is
# _plain_projection`, so the name and the object the ground hands back have to
# be the same function or the accelerator would quietly stop serving the
# grounded fills (momwire#397 unit 3).
_plain_projection = _field_ground.plain_projection


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
        both sides of the gap need a conductor.

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

    # momwire#396: differs from `SinusoidalSolver.capabilities` in exactly
    # the two axes this class's docstring describes — junction_ports and
    # node_gaps are served here (M5b / #305) — plus the three combination
    # refusals __init__'s raises still carry. Wire loading rides the base
    # class's overlap term (#395) unchanged. `extended_kernel+
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
    # False on both classes for the same reason, and until momwire#396 goal 3
    # this row answered it with `refusal()`'s generated one-liner while the
    # base answered with the reason. Same sentence, this class's name in it
    # (momwire#564) — the prose reaches antennaknobs' host dialogs verbatim.
    capabilities = SinusoidalSolver.capabilities._replace(
        junction_ports=True,
        node_gaps=True,
        knot_feeds=True,
        refusals={
            "junction_ports+finite_ground": _JUNCTION_PORTS_FINITE_GROUND_REFUSAL,
            "junction_ports+mixed_radii": _JUNCTION_PORTS_MIXED_RADII_REFUSAL,
            "contact+refl-coef": SinusoidalSolver.capabilities.refusals[
                "contact+refl-coef"
            ],
            "singular_enrichment": SINGULAR_ENRICHMENT_NOT_YET.format(
                cls="SinusoidalGalerkinSolver"
            ),
            # `buried` and `contact` are the base's, unchanged: this class
            # inherits `_build_geometry`'s scan whole, so the deck it refuses
            # and the sentence it refuses with are the base's too — only the
            # class name in the prose differs (momwire#564). Carried across
            # by hand for the same REPLACE reason as `contact+refl-coef`
            # above.
            "buried": _BURIED_REFUSAL.format(cls="SinusoidalGalerkinSolver"),
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
        **kwargs,
    ):
        # Seen by the base's feeds=[] check (its signature never learns the
        # node-port kwargs): a solve driven entirely through node ports/gaps
        # needs no gap feed, same as junction ports (#172/#305).
        self._node_drive_declared = bool(node_ports or node_gaps)
        super().__init__(**kwargs)
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
            raise NotImplementedError(
                "extended_kernel=True requires near_correction=True on "
                "SinusoidalGalerkinSolver: the extended kernel's delta is "
                "resolved on the near-pair path, and M1 mode would leave the "
                "self and node-sharing pairs on the far tier's rule "
                "(momwire#246)"
            )
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
        seen = set()
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
            if j_idx in seen:
                raise ValueError(f"node_ports: junction {j_idx} listed twice")
            if j_idx in junction_port_idx:
                raise ValueError(
                    f"junction {j_idx} is declared as both a junction_port and "
                    "a node_port — they are different port objects (net inflow "
                    "at the node vs an EMF across the node) and cannot share a "
                    "junction"
                )
            seen.add(j_idx)
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
            out.append((j_idx, side_a, volts))
        return out

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
        N = geom["n_segs"]
        n_basis = N + len(self.junction_ports)
        grounded = geom["grounded_junctions"]
        out = np.zeros((n_basis, len(self.node_ports)), dtype=np.complex128)
        starts = seg_view["starts"]
        seg_h = np.asarray(geom["seg_h"], dtype=float)
        for p, (j_idx, side_a, _v) in enumerate(self.node_ports):
            if j_idx in grounded:
                raise ValueError(
                    f"junction {j_idx} is both grounded and a node port — a "
                    "node in the ground plane carries current into its own "
                    "image (#151), so the members' currents do not close on "
                    "each other and there is no through-current to drive"
                )
            members = self._junction_members(geom, j_idx)
            for mi in side_a:
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
                    k,
                    half * sgn,
                )
                np.add.at(out[:, p], seg_view["jbasis"][s:e], sgn * val)
        return out

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

    def _junction_members(self, geom, j_idx):
        """(segment index, σ) for every wire-end at junction `j_idx`.

        σ is the sign the extension shape below is stamped with: +1 when the
        junction node is at the member segment's natural end-2 (`seg_r`), −1
        when it is at end-1 (`seg_l`) — the same L/R rule `_build_geometry`
        applies to a real N⁻ neighbour. Under the current-shape convention
        I(s) = σA + B·sin(ks) + σC·cos(ks) that sign is exactly what makes
        the shape's current flow INTO the node in both orientations.
        """
        members = []
        for w, end in self.junctions[j_idx]:
            if end == "start":
                members.append((geom["wire_first"][w], -1))
            else:
                members.append((geom["wire_last"][w], +1))
        return members

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
        atom = (1.0 - np.cos(kd)) / np.sin(kd) * a_const

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

    def _node_charge_columns(self, geom, seg_view, k, nodes=None):
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
        n_basis = N + len(self.junction_ports)
        a = float(self._uniform_radius)
        seg_c, seg_t = geom["seg_centers"], geom["seg_tangents"]
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)
        pref = -1j * self.eta / (4.0 * np.pi * k)
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

    def _node_charge_pair_block(self, geom, k):
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
        return self._lumped_pair_block(nodes, nodes, k)

    def _node_charge_image_pair_block(self, geom, k):
        """S_img[p, q] — the same lumped-lumped term between node p and the
        PEC IMAGE of node q, at the mirrored separation (#191).

        Symmetric because reflection is an isometry:
        |node_p − M·node_q| = |M·node_p − node_q|.
        """
        nodes = self._port_node_positions(geom)
        return self._lumped_pair_block(nodes, self._port_node_positions(geom, True), k)

    def _lumped_pair_block(self, nodes_row, nodes_col, k):
        """Mixed-potential pairing of unit node charges at two point sets,
        regularized at the wire radius exactly as `_node_charge_columns` is."""
        a = float(self._uniform_radius)
        d = nodes_row[:, None, :] - nodes_col[None, :, :]
        R = np.sqrt((d * d).sum(axis=-1) + a * a)
        return 1j * self.eta * np.exp(-1j * k * R) / (4.0 * np.pi * k * R)

    def _assemble_Z_ported(self, geom, k):
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
        G, seg_view = self._assemble_Z(geom, k)
        if not self.junction_ports:
            return G, seg_view
        N = geom["n_segs"]
        D = self._node_charge_columns(geom, seg_view, k)
        G = G.copy()
        G[:, N:] -= D
        G[N:, :] -= D.T
        G[N:, N:] += self._node_charge_pair_block(geom, k)
        # The #151/#191 node charge is the Galerkin analogue of the point-
        # matched contact-charge read: a TESTING-SCHEME correction that happens
        # to involve the image, deliberately outside the `FieldGround` sketch
        # (`docs/design/field-ground-interface.md`, "what stays out"), so it
        # still reads `ground_z` here rather than the fill's ground object.
        # Only PEC reaches it — finite grounds are refused upstream.
        if self.ground_z is not None:
            D_img = self._node_charge_columns(
                geom, seg_view, k, nodes=self._port_node_positions(geom, mirror=True)
            )
            G[:, N:] += D_img
            G[N:, :] += D_img.T
            G[N:, N:] -= self._node_charge_image_pair_block(geom, k)
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

    def _ek_bracket_block(self, geom, k, ctx, corr, plan, m0, m1):
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
    ):
        """Test-integrate one source block: (contrib_const, sin, cos−1), each
        (nnz, N) — the folded shape set (#203/#205), which is what the field
        kernel is asked for (`cos_shape="cos-1"`).

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

        if (
            _HAVE_GALERKIN_FAR_FILL
            and projector is _plain_projection
            and (not self.extended_kernel or _HAVE_GALERKIN_FAR_FILL_EK)
        ):
            # Guarded at the call site as well as inside, like every other EK
            # entry point here: G-B4's counter gate is that an EK-off solve
            # does not so much as ENTER this code.
            ek_pairs = (
                self._ek_far_labels(geom, mirror) if self.extended_kernel else None
            )
            if subtract_into is None:
                contribs = self._far_fill_accel(k, ctx, src_c, src_t, ek=ek_pairs)
                if self.near_correction:
                    self._apply_near_correction(
                        geom, k, ctx, contribs, projector, src_c, src_t, mirror
                    )
                return contribs
            near_cells = (
                self._apply_near_correction(
                    geom,
                    k,
                    ctx,
                    subtract_into,
                    projector,
                    src_c,
                    src_t,
                    mirror,
                    sub=True,
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
                k, ctx, src_c, src_t, ek=ek_pairs, out=subtract_into, scale=-1.0
            )
            if held is not None:
                for dest, saved in zip(subtract_into, held):
                    dest[near_cells] = saved
            return None

        # Blocked over test segments (#194): identical arithmetic per matrix
        # entry, but the kernel's source-quadrature scratch is (rows·nq, N,
        # n_qp_const) per block instead of (N·nq, N, n_qp_const) once.
        starts = ctx["starts"]
        nnz = ctx["w_entry"].shape[0]
        a_obs = ctx["a_obs"]
        n_idx = np.arange(N)[None, :]
        if subtract_into is None:
            contribs = tuple(np.zeros((nnz, N), dtype=np.complex128) for _ in range(3))
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
                self._apply_near_correction(
                    geom, k, ctx, contribs, projector, src_c, src_t, mirror, sub=True
                )
                if self.near_correction
                else None
            )
        blk = _fill_block(N, nq, self.n_qp_const)
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
                    w, m_loc, nq, P.reshape(m1 - m0, nq, N)
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
            self._apply_near_correction(
                geom, k, ctx, contribs, projector, src_c, src_t, mirror
            )
        return contribs

    def _far_fill_accel(self, k, ctx, src_c, src_t, ek=None, out=None, scale=1.0):
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
        args = (
            np.ascontiguousarray(ctx["obs_c"], dtype=np.float64),
            np.ascontiguousarray(ctx["obs_t"], dtype=np.float64),
            a_flat,
            np.ascontiguousarray(src_c, dtype=np.float64),
            np.ascontiguousarray(src_t, dtype=np.float64),
            np.ascontiguousarray(ctx["hh"], dtype=np.float64),
            float(k),
            float(self.eta),
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

    def _fold_ground_block(self, geom, k, ctx, contribs, fg):
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
            )
            return

        img = self._tested_contribs(
            geom, k, ctx, projector, src_c_img, src_t_img, mirror=True
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
        self._tested_sommerfeld_remainder(ctx, fg, img)
        for dest, a in zip(contribs, img):
            np.subtract(dest, a, out=dest)

    def _tested_sommerfeld_remainder(self, ctx, fg, subtract_from):
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
                rows = self._tested_contrib_rows(
                    w, m_loc, nq, s.reshape(m1 - m0, nq, N)
                )
                np.subtract(dest[e0:e1], rows, out=dest[e0:e1])

        fg.remainder("cos-1").replay(
            obs_centers=ctx["obs_c"],
            obs_tangents=ctx["obs_t"],
            consume=_reduce,
            row_group=nq,
        )

    def _assemble_Z(self, geom, k):
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
        fg = _field_ground.field_ground_for(self, geom, k, self.omega)

        contribs = self._tested_contribs(geom, k, ctx, _plain_projection)
        if fg is not None:
            self._fold_ground_block(geom, k, ctx, contribs, fg)

        G = self._scatter_coef_product(ctx, contribs)
        # The fill's triple is dead the moment its product exists, and what
        # runs next used to be measured on top of it (momwire#355): the
        # end-bracket correction is a second sub-assembly of the same shape,
        # so holding this one across it doubled the fill's own footprint for
        # nothing. Dropping the name here is not an optimization of the
        # arithmetic — it changes no value — it just stops the peak from
        # counting a triple nobody reads again.
        del contribs
        self._ek_bracket_correction_tested(G, geom, k, ctx, fg)
        self._contact_charge_correction_tested(G, geom, k, seg_view, ctx)
        self._apply_loading(G, geom, seg_view, k)
        return G, seg_view

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

    def _apply_loading(self, G, geom, seg_view, k):
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
        starts = seg_view["starts"]
        left, right, m_of_pair = self._shared_segment_pairs(starts)

        sig = seg_view["sigma"].astype(np.complex128)
        P = sig * seg_view["A"]
        Q = seg_view["B"]
        R = sig * seg_view["C"]
        h = np.asarray(geom["seg_h"], dtype=float)[m_of_pair]
        w_pr = (2.0 / k) * np.sin(0.5 * k * h)
        half_sin = np.sin(k * h) / (2.0 * k)
        vals = (
            P[left] * P[right] * h
            + (P[left] * R[right] + R[left] * P[right]) * w_pr
            + Q[left] * Q[right] * (0.5 * h - half_sin)
            + R[left] * R[right] * (0.5 * h + half_sin)
        )
        # (n_segs,), zeros where switched off — the shared spec layer
        # (momwire#428); the overlap VALUES above are this rule's own share.
        z_seg = _wire_loading.loading_for(self, k * self.c, geom).z_seg
        vals *= z_seg[m_of_pair]
        # Unbuffered: a basis pair recurs once per shared segment.
        np.subtract.at(G, (seg_view["jbasis"][left], seg_view["jbasis"][right]), vals)
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
        n_basis = N + len(self.junction_ports)
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

    def _ek_bracket_correction_tested(self, G, geom, k, ctx, fg):
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
          finished inside one band and `np.add.at` reaches it in the same
          ascending-entry order the whole-triple scatter did. Folding per
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
        n_basis = N + len(self.junction_ports)
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
                self._ek_bracket_block(geom, k, ctx, corr, plan, m0, m1)
            if not any(np.any(c) for c in corr):
                continue
            rows = i_of_entry[e0:e1]
            for dest, c in zip(T, corr):
                # The scatter's own accumulation, reached one band early.
                # `np.add.at` is unbuffered and walks the band in ascending
                # entry order, which is the order `R @ corr` sums a basis
                # row's entries in — and the bands are ascending too, so every
                # T cell sees exactly the sequence of additions the
                # whole-triple product performed.
                np.add.at(dest, rows, c)
        if not any(np.any(t) for t in T):
            return
        C = self._bracket_coef_product(ctx, T, cols, col_of)
        G -= 0.5 * (C + C.T)

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
        n_basis = N + len(self.junction_ports)
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
        return sum(
            t @ scipy.sparse.csc_matrix((coef[sel], mi), shape=(cols.size, n_basis))
            for t, coef in zip(T, coefs)
        )

    def _contact_charge_correction_tested(self, G, geom, k, seg_view, ctx):
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
            R = self._contact_charge_kernel(geom, k, node, obs_c, obs_t, a_obs)
            masks = self._contact_ek_masks(geom, i, sgn, obs_seg)
            if masks is not None:
                R = R + self._contact_charge_ek_delta(
                    geom, k, i, sgn, node, obs_c, obs_t, a_obs, *masks
                )
            R_t = self._tested_contrib_rows(
                w_entry, m_of_entry, nq, R.reshape(N, nq, 1)
            )[:, 0]
            T = np.zeros(G.shape[0], dtype=np.complex128)
            np.add.at(T, i_of_entry, R_t)
            jb, val = self._contact_node_values(geom, k, seg_view, i, sgn)
            G[:, jb] -= sgn * T[:, None] * val[None, :]
        return G

    def _apply_near_correction(
        self, geom, k, ctx, contribs, projector, src_c, src_t, mirror=False, sub=False
    ):
        """Recompute the near-pair test integrals on the endpoint-graded rule,
        overwriting the uniform-rule values in `contribs` (M2).

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

        contrib_c, contrib_s, contrib_co = contribs
        # Pairs per block from the byte budget rather than a flat 512
        # (momwire#383): under the extended kernel the block holds
        # `_folded_ek_delta_fields`' (P, G, n_d) quadrature, 4.9 MB per pair
        # at this rule, and 512 of those was 2.6 GB of fixed working set.
        blk = _near_block(xg.shape[0], self.n_qp_const, self.extended_kernel)
        for p0 in range(0, mm.size, blk):
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
            for contrib, Ph in zip((contrib_c, contrib_s, contrib_co), Phi):
                val = np.einsum("eg,eg->e", w, Ph[lp])
                if sub:
                    contrib[ei, col] -= val
                else:
                    contrib[ei, col] = val

        return (entry_of, nn[pair_of]) if sub else None

    # ------------------------------------------------------------------
    # Galerkin-tested source vector + solve
    # ------------------------------------------------------------------

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
        n_basis = N + len(self.junction_ports)
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
                    k,
                    float(geom["feed_xi"][j]),
                )
            else:
                hm = float(h[fseg])
                int_f = sig * seg_view["AC"][s:e] * hm + sig * seg_view["C"][s:e] * (
                    2.0 / k
                ) * _sin_minus_arg(0.5 * k * hm)
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
        geom = self._build_geometry()
        self._checkpoint()  # after geometry, before the field fill
        G, seg_view = self._assemble_Z_ported(geom, self.k)
        U = self._drive_columns(geom, seg_view, self.k)
        voltages = self._port_voltages()
        self._checkpoint()  # after assembly, before the dense solve

        alpha = scipy.linalg.solve(G, U @ voltages)

        currents = self._port_currents(alpha, geom, seg_view, U)
        z_per_port = voltages / currents
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
        G, seg_view = self._assemble_Z_ported(geom, self.k)
        U = self._drive_columns(geom, seg_view, self.k)
        alphas = scipy.linalg.solve(G, U)
        Y = np.stack(
            [
                self._port_currents(alphas[:, j], geom, seg_view, U)
                for j in range(self.n_ports)
            ],
            axis=1,
        )
        return PortSolution(
            y=Y,
            coeffs=alphas,
            port_currents=Y,  # the same object: the readout IS the Y matrix
            basis=_SegmentBasis(geom=geom, seg_view=seg_view, k=self.k),
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
