"""Tent basis with razor-blade testing — the NEC-5 formulation twin.

NEC-5's Users Manual (§1) states its formulation: a linear (triangular)
current expansion tested by the mixed-potential method of Rao, Wilton and
Glisson — the E-field boundary condition enforced on *path integrals*
between the centroids of connected elements ("razor-blade" testing), NOT
the point matching of NEC-2/4 and NOT the Galerkin testing of momwire's
:class:`~momwire.bspline.BSplineSolver` at ``degree=1``. That single
difference in the testing rule is what produces NEC-5's slow O(1/N)
impedance walk (the momwire#890 / antennaknobs#896 finding); the manual
itself predicts slower convergence than a sinusoidal expansion.

This module is that formulation, written as a momwire solver so the walk is
reproducible without the NEC-5 binary. It is deliberately a *twin*, not an
improvement: the quadratures are converged, but the scheme's discretization
error is NEC-5's, so a fine mesh here walks the same way NEC-5's does.

Formulation
-----------
Unknowns are the interior knots of each wire — one unit tent Λ_n per
interior knot, rising linearly in arc length over the segment before the
knot and falling over the segment after, so the current vanishes at every
free wire endpoint — plus one junction tent per independent through-path
where wire ends meet (see "Junctions" below). Row m tests along the path
P_m that runs from the centroid of the segment before knot m, through the
knot, to the centroid of the segment after (two straight half-segments,
bent at the knot on a kinked wire):

    Z[m,n] = jωμ₀ · T1[m,n] − T2[m,n] / (jωε₀)

    T1[m,n] = ∫_{P_m} t̂(r) · A_n(r) dl        (A_n carries no μ₀)
    A_n(r)  = ∫ Λ_n(l') g(r,r') t̂(r') dl'
    T2[m,n] = ∫ Λ_n'(l') [ g(c_after, r') − g(c_before, r') ] dl'

with the thin-wire kernel g = exp(−jkR)/(4πR), R = sqrt(|r−r'|² + a²), the
source running along the segment *axis*, and Λ_n' the tent's ±1/h charge
doublet. T1 carries the tangent dot product (both tangents turn at a bend);
T2 does not. `g` is the REDUCED kernel by default and the EXTENDED one under
`extended_kernel=True` — see "The extended kernel" below, and note that the
kernel is an axis of its own: it does not change a single line of the
formulation above.

Excitation is NEC-5's EX-at-a-knot: the delta-gap voltage sits inside
exactly one testing path, so it lands entirely in that knot's row and the
solved tent coefficient at the feed knot *is* the drive-point current.

Junctions (momwire#309, unit 2)
-------------------------------
A junction is a point where wire ENDS coincide (within `_JUNCTION_TOL`),
including the single wire whose start and end coincide — a closed loop.
They are detected from the geometry; there is no junction spec to write.

K coincident ends carry K−1 independent through-currents, so a junction
gets K−1 extra bases. The first-listed end at the point is the reference
side A; every other end B_i gets one junction tent spanning A's terminal
segment and B_i's, rising linearly to 1 at the junction on both sides.
Its coefficient is the current flowing THROUGH the junction from A into
B_i, in amperes, and its row is the razor path centroid(A) → junction →
centroid(B_i) traversed in the direction of that flow. Because a wire may
join by either of its ends, each half carries a sign relative to the
wire's own arc direction; the charge doublet is then +1/h_A on A's
terminal segment and −1/h_B on B_i's, which is Kirchhoff's law written
into the basis rather than enforced by a constraint row. An interior-knot
tent is the K=2 junction tent of a wire split at that knot, exactly — the
split identities in `tests/test_razor_junctions.py` pin that.

Field readout and sweeps (momwire#309, unit 3)
------------------------------------------------
`currents_at_knots` reads the solved coefficients back as one current per
mesh knot per wire — see its docstring for how a junctioned end's current
is rebuilt from its tents' wing signs, since this formulation's junction
basis is not a per-wire end basis the way the retired triangular-family
solver's was. `element_currents` (the `_ElementCurrents` mixin) rides on
top of it. `compute_impedance_swept` / `compute_y_matrix_swept` solve a
batch of wavenumbers, sharing the wing/path stencils and the closed-form
static segment moments across the sweep (`_assemble_Z_prepare`) so only
the smooth kernel remainder and the ω-dependent prefactors are redone per
k; every solved point is still its own dense LU.

Ground (momwire#398 unit 2)
--------------------------
`ground_z` puts a perfectly conducting plane under the model. The fill
becomes `Z = Z_free − Z_image`: the same rows, the same testing paths and
the same tent basis, evaluated a second time against sources reflected
through the plane, subtracted once. The mirror itself comes from
`_potential_ground.PotentialGround` — this module writes no reflection of
its own, which is the point of the pilot `docs/design/solver-architecture.md`
§6 proposes: a ground method landing on a solver that never implemented
one. `_assemble_Z_from_prepared` carries the sign argument.

Finite ground (momwire#398 unit 4)
---------------------------------
`ground_eps` puts a reflection-coefficient (NEC `GN 0` style) ground under
the model instead, for wires standing CLEAR of the plane. It changes the
image block and nothing else: where the PEC fold contracts the mirrored
tangent table, the weighted one contracts a per-pair Fresnel weight w_A
that already carries that tangent dot inside it, and where the PEC fold
leaves the charge term unweighted, the weighted one scales the image
kernel at each testing-path endpoint by w_Φ. Both weights come from
`PotentialGround.weight_windows` — this module still writes no reflection
of its own — and the seams' single `Z = Z_free − Z_image` is untouched.

The weights are ω-dependent, which is the one schedule consequence: they
are built per solved wavenumber in `_assemble_Z_from_prepared`, while the
mirrored geometry and its static moments stay in the k-independent
`_assemble_Z_prepare` where unit 2 put them.

NEC-5 is not the oracle for this ground — its finite ground is Michalski
and carries a limit offset of its own — so the bar is cross-formulation
agreement with momwire's B-spline and sinusoidal solvers, in the shape
this formulation can honestly claim: the ground must not widen the
razor-vs-Galerkin gap that its own O(1/N) walk already opens in free
space. `tests/test_razor_refl_coef_ground.py` measures it.

Sommerfeld ground (momwire#398 unit 5)
--------------------------------------
`ground_model="sommerfeld"` puts the exact (NEC `GN 2` style) finite
ground under the model, again for wires standing CLEAR of the plane. It
is this solver's first COMPOSING ground — `PotentialGround.mode ==
"compose"` — and composing is the whole of what is new:

    Z = Z_free − (C₂·image + Q)

The first term needs no new code at all. C₂ = (ε̃−1)/(ε̃+1) arrives as
`weight_windows`' constant `(w_A, w_Φ)` pair, so unit 4's weighted fill
assembles the scaled exact image with the same two lines it uses for the
Fresnel one — the coefficient enters THROUGH the windows and is never
applied to a block. The second is the smooth remainder field Q, and it is
where the composing ground differs from the folding ones twice over:

* it is a FIELD, not a potential pair, so it takes no jωμ / 1/jωε
  prefactor and is summed into the image block AFTER those — inside
  `_assemble_Z_source_block`, before the seam's single minus, because
  `free − (C₂·img + Q) ≠ (free − C₂·img) − Q` in float64;
* its interpolation grid is measured in WAVELENGTHS, so unlike unit 4's
  weights (ω-dependent) it is k-dependent, and it may not cross the
  prepare/replay boundary in any form. Grid and source nodes are built per
  solved wavenumber, inside the producer `Remainder.field_windows` returns.

Razor's rows are path integrals, not Galerkin projections, so it cannot
consume `Remainder.evaluate`'s finished block; unit 5's shared-layer edit
is the operation underneath it, which hands back the remainder field of
each source segment's tent moments at an arbitrary observer set. The
`PulseSolver` probe (momwire#416) had independently asked for the same
thing. `tests/test_razor_sommerfeld_ground.py` carries the physics, and
the bar is again cross-formulation rather than NEC-5.

Ground CONTACT over a finite ground stays refused, over BOTH of them, and
momwire#282 is the reason: the fold hard-codes image coefficient 1, so the
grounded-end tent's lower wing — which IS that image — would take spurious
contact charge.

Ground contact (momwire#398 unit 3)
-----------------------------------
A wire END may lie IN the plane. Such an end keeps a degree of freedom
instead of being zeroed the way the tent basis zeroes a free end: its basis
is the junction tent between the wire and its own image — a monopole plus
its image IS a dipole — whose upper wing is the real contact segment and
whose lower wing is that segment mirrored. Only the upper wing is spelled
in the basis tables, because the fold already evaluates every basis against
the mirrored sources: the image wing arrives with the right shape, the
right direction (−M·t̂, parallel to the real current for a vertical
contact) and the opposite charge, so no net charge sits at the contact
point.

The grounded tent's testing path is the REAL half only, from the contact
point to the centroid of the contact segment, in the direction the current
leaves the plane. The image half of that path contributes the identical
number (the total field of a system that is its own image obeys
E(M·r) = −M·E(r), which is exactly the invariance the razor path integral
contracts), so taking the real half alone halves the row — and halving the
row is what makes the feed voltage the voltage of the base GAP rather than
of the equivalent dipole's whole gap. On that spelling a base-fed monopole
returns Z_dipole/2 to LU roundoff, which is the first gate in
`tests/test_razor_ground_contact.py`. The row's other endpoint is the
plane, where the folded scalar potential is identically zero, so the T2
term keeps only the segment-centroid end.

Wire loading (momwire#427)
--------------------------
A loaded wire's surface condition is E_tan = Z_s(l)·I(l) rather than
E_tan = 0, and razor tests the condition on a path, so the loading term is
the testing-path integral of Z_s against each tent:

    L[m, n] = ∫_{P_m} Z_s(l) Λ_n(l) dl,        Z = Z_free + L

In the wing idiom that is two numbers per shared segment — 3h/8 when the
path half and the tent ramp rise at the same end of the segment, h/8 when
they rise at opposite ends — times the σ·σ that dots the path's direction
with the current's. `_loading_stencil` carries the full derivation, the
sign's oracle, the junction and grounded-end readings, and why the
resulting L is symmetric while the field matrix is not.

DISTRIBUTED loading (`wire_conductivity`, `insulation_radius`,
`insulation_eps_r`) is the siblings' API verbatim, over the same
`_wire_loading` physics. LUMPED loads (`lumped_loads`, a sequence of
``(wire_index, arclength, impedance)``) are razor's own kwarg: the other
rows serve a lumped load as port algebra over a zero-volt gap a consumer
stamps afterwards, which this formulation does not take, but a delta in
Z_s at a knot collapses the integral above to a single diagonal entry — so
`Z_driven = Z_unloaded + Z_L` at the fed knot is exact here rather than
arranged. `wire_loss_power`
reads back the DISTRIBUTED dissipated watts (like the siblings');
`lumped_load_power` (momwire#433) reads back each load's own share.

The stencil is pure geometry and rides `_assemble_Z_prepare`; Z_s(ω) is
not (skin effect and insulation reactance both move with ω) and is built
per solved wavenumber beside the reflection-coefficient weights.

Per-wire radius (momwire#147)
-----------------------------
`wire_radius` takes a scalar or one radius per wire, the siblings' spelling.
The reduced kernel's a² is regularised with the SOURCE segment's radius, so
a fat/thin junction's tent takes each wing's own segment's radius and no
special case is written for it: the tent is two wings on two segments, and
each wing's moments were already computed against its own source column.
A uniform model keeps the scalar path and is bit-identical to it.

The convention is a choice, not an oracle finding — which is itself the
finding. On the only geometry where the two candidates diverge (a collinear
fat/thin STEP, where the perpendicular distance vanishes and a² is all
there is), source and observer conventions differ by ~1e-5 Ω at a 10:1
radius step against a 0.20 Ω twin-lane bar, so the licensed binary cannot
separate them. `_seg_moments_prepare` records the numbers and the reason
the source reading is taken.

The extended kernel (momwire#398 D1)
--------------------------------
`extended_kernel=True` swaps the reduced kernel for NEC's EXTENDED (tubular)
one on the pairs that are eligible for it. The default is False and the
reduced kernel, and an EK-off solve is bit-for-bit the answer this class
gave before the kernel existed, on every lane.

**Why the twin needs it.** The taper study of 2026-08-18 identified the
NEC-5 binary as extended-kernel EVERYWHERE — it has no `EK` card because its
formulation does not offer the choice — by driving Δ/a from 10 down through
0.5 along two independent paths and watching which side of the kernel gap
the binary's printed impedance sits on (α → 1.09 / 1.02, the binary within
4–9 % of the EK row across a 43–113 Ω gap). The control that removes
quadrature as the explanation is this class: `nec5_quadrature` running the
REDUCED kernel — NEC-5's own testing, basis and quadrature idiom — sits
32.3 Ω from the binary at Δ/a = 0.5 where the EK row sits 4.3 Ω away. So the
older refusal here ("NEC-5's formulation is the comparison target, and its
expansion is tested on the wire axis") had the basis, the testing and the
quadrature right and the KERNEL wrong: the expansion is indeed tested on the
axis, but the source it is tested against is a tube. The measurement
overturned the premise and the refusal fell with it.

**What it buys, measured** (the study's `fat` control — a uniform 25 mm
dipole at 14.2 MHz, the fattest section of Ward Harriman's 20:1 taper, fed
at NEC-5's own knot; the ladder gated to Δ/a ≥ 2 per momwire#248):

    row                    offset from NEC-5, N = 20 … 200      limit gap
    razor (reduced, n5q)   +0.02+0.06j … +0.58+0.54j Ω          1.400 Ω
    razor (EK, n5q)        +0.005+0.022j … +0.005+0.011j Ω      0.047 Ω

(the limit gap Richardson on the two finest gated rungs; extrapolated from
the study's N = 280/400 instead, outside the valid Δ/a domain, the two read
4.863 and 0.666 Ω — see `docs/design/solver-architecture.md` §6.13)

i.e. the EK row is a CONSTANT offset at the sharp `nec5_quadrature` bar
(dR spread 0.012 Ω, dX 0.021 Ω against 0.05 Ω) on the very deck where the
reduced row misses that bar — by 11× on these same gated rungs, and by 43×
on the study's full ladder to N = 400. **This is the twin claim, restored
on fat wire.** On THIN wire the two kernels agree and the reduced row remains
the twin it always was; the two statements do not compete, they partition
the domain by a/λ.

**The eligibility rule is the shared one and is not re-derived here.**
`_bspline_kernels._ek_axis_groups` labels two segments alike iff they are
COAXIAL and of EQUAL RADIUS, on NEC's own thresholds, and a PAIR is extended
iff its two labels match. That is the B-spline trunk's rule rather than
`SinusoidalSolver`'s per-END IND1/IND2 gating, and the reason is that this
formulation is mixed-potential: its rows are path integrals over arbitrary
(observer point, source segment) pairs, not per-end brackets, so the pair
rule is the precedent that fits. An observation point's label is the label
of the segment it LIES ON — the testing path's two halves run along the two
wing segments, so each half's quadrature points inherit that wing's label.

**Five things it composes with, and how:**

* **the two quadrature lanes.** The quadrature rule and the kernel are
  ORTHOGONAL axes and both lanes serve the kernel. `nec5_quadrature` decides
  where the testing path is sampled; `extended_kernel` decides what kernel is
  sampled there. Neither reads the other: the lane only changes how many
  observation points a path has (2 versus 2·`n_qp_path`), and the EK labels
  are attached per point either way. The four combinations are all live, and
  the one that is the FAT-WIRE TWIN is `extended_kernel=True,
  nec5_quadrature=True` — both of NEC-5's identifications at once.
* **the PEC and finite grounds.** The ground supplies mirrored GEOMETRY and
  never the kernel's opinion. Eligibility over a ground is one scan of the
  shared rule over the real segments STACKED ON the mirrored ones, so a
  vertical wire — whose image is coaxial with it and of equal radius — is one
  group and extends, which is NEC's own IND = 0 perpendicular-ground branch,
  while a horizontal wire and its image are merely parallel and do not.
  Scanning the two sets separately would declare every real/image pair
  coaxial and is the trap `BSplineSolver._ek_axis_labels` records.
* **ground CONTACT.** Nothing is written for it. The grounded tent's lower
  wing IS its own image, so it is extended exactly when the joint scan says
  the mirrored source is coaxial with the observer — which for the vertical
  contact that motivates the basis it is. The contact case needed no branch
  because the mirror policy already covers it.
* **per-wire radii.** EK eligibility is EQUAL-RADIUS pairwise, so the shared
  rule handles a taper by refusing to extend ACROSS a radius step while
  extending within each section. That is strictly more conservative than
  NEC, which still extends some cross-arm pairs at an `IND = 2` step
  (#249 §4.3, O(h) in the refinement limit) — and it is visible as the one
  place the fat-wire constancy is not quite as sharp: on Ward's 10-step
  taper the same EK twin lane holds the 0.05 Ω bar to Δ/a ≳ 3 and runs
  1.6× over it (dX) at Δ/a = 2.1, against the uniform `fat` control's 0.021.
  The pair's radius IS the kernel call's own `a`, because eligibility
  requires the two to be equal.
* **loading.** Orthogonal, with no interaction at all: `L` is the testing-
  path integral of the conductor's surface impedance and lives outside the
  fold, so it neither sees the kernel nor is seen by it.

Scope
-----
Free space and all three grounds — PEC, reflection-coefficient and
Sommerfeld — either kernel, one polyline per wire. Ground contact is served
over PEC and over the SOMMERFELD ground (momwire#624); what stays refused
inside that column is contact under `refl-coef`, which is the whole tree's
refusal rather than this family's (momwire#282 stage 1, `_ground_spec`).
Refused too, with a message rather than silently
mismodelled, are the contacts that are not a wire end in the plane (an
interior anchor touching down, an edge lying in the plane, a wire dipping
below it). Only wire ENDS junction: a wire end touching another wire's
interior is not a contact here. A wire with a single segment is served
wherever either of its ends meets something — another wire, or the ground
plane — because such an end carries a tent, and two of them sharing one
segment are the two Lagrange bases an interior segment already carries.
Splitting a wire so that a one-segment piece falls out therefore reproduces
the unsplit wire exactly (momwire#608). One whose ends meet NOTHING is
refused: it carries no basis at all and would scatter nothing. The extended
kernel adds no combination hole of its own:
every capability this class serves — all three grounds, contact, junctions,
loading, mixed radii, both quadrature lanes, swept solves — is served with
it on, and each is gated with it on.
"""

from dataclasses import dataclass
import os

import math

import numpy as np
import scipy.linalg

from . import (
    _feed_snap,
    _ground_refl,
    _ground_spec,
    _medium_spec,
    _crossing_fill,
    _potential_ground,
    _sommerfeld_below,
    _wire_loading,
    _wire_spec,
)
from .bspline import SINGULAR_ENRICHMENT_NEVER
from ._bspline_kernels import (
    _EK,
    _complex_k,
    _ek_axis_groups,
    _ek_factor,
    _ek_pair_mask,
    _ek_radius,
    _ek_reg_extra,
)
from ._accel import acc as _acc
from ._cancel import _Cancelable
from ._capabilities import Capabilities
from ._element_currents import _ElementCurrents

# Re-exported under their own names, deliberately: `pulse.py` imports them
# FROM this module (momwire#419, where the coupling was left visible on
# purpose), and momwire#425 moved the bodies to `_kernel_moments` without
# moving that import. Migrating pulse's import line is a one-liner, on
# pulse's own branch.
from ._kernel_moments import (
    _axis_frame,
    _static_axis_moments,
    _static_axis_moments_ek,
)
from ._stable import expm1_neg_jkR as _expm1_neg_jkR
from ._junction_rule import JUNCTION_TOL, canonical_groups, grouped
from ._port_solution import PortSolution, _SweptPortSolutions
from . import _quadrature
from ._quadrature import leggauss

# Two wire endpoints this close are a junction, not a coincidence. The rule
# and the value both live in `_junction_rule` now (momwire#590 step 1); this
# name is kept because it is what the docstrings in this module cite.
#
# The comment that used to sit here claimed this was "the same tolerance the
# caller-facing geometry helpers use for 'same point'". It is not — momwire#429
# correction 2 caught that, and `_junction_rule`'s docstring says what is true.
_JUNCTION_TOL = JUNCTION_TOL

# Working-array budget for the chunked fills, in complex128 elements
# (~32 MB per temporary). The fill's inner tensor is
# (observation points) × (segments) × (source quadrature points), which for
# a 200-segment two-element model at the default orders would otherwise be
# a few hundred MB in one allocation.
_CHUNK_ELEMS = 2_000_000

# The same budget for a fill whose image block is WEIGHTED (momwire#398
# unit 4). A reflection-coefficient window costs far more per (observer,
# source segment) pair than the one complex moment the unweighted fill
# holds there: `BSplineSolver._image_weight_row_bytes` prices the same
# closure at 14 float64 + 1 bool + 5 complex128 per pair — 193 B, i.e. ~12
# complex128 — because CPython keeps every local of `specular_ray_tables`
# / `specular_pair_tables` / `fresnel_rho` alive for those functions' whole
# bodies. Dividing the element budget by that factor keeps a weighted
# chunk's transient in the same ~32 MB class as an unweighted one's, at the
# cost of more (smaller) chunks. Purely a schedule number: the weighted
# fill is elementwise on the observer axis, so the answer does not depend
# on it, which `tests/test_razor_refl_coef_ground.py` pins directly.
_WEIGHTED_CHUNK_ELEMS = _CHUNK_ELEMS // 12

# The C++ segment-moment fill (momwire#742), and the two ways to turn it off.
# The capability flag is the kernel's OWN symbol, never a shared one: a .so
# built before this arc exports every other section's entries and not this,
# and a shared flag would claim a contract it cannot serve.
_HAVE_RAZOR_FILL_ACCEL = _acc is not None and bool(
    getattr(_acc, "razor_fill_742", False)
)

# The tests' handle on the dispatch — the agreement gates drive BOTH machines
# inside one process. `MOMWIRE_RAZOR_FORCE_NUMPY` is the whole-run switch (a
# timing comparison, a bisect); `monkeypatch.setattr(razor, "_FORCE_NUMPY",
# True)` is the per-test one. `_use_razor_fill_accel` reads both at CALL time,
# so a fixture that flips either mid-solve is honoured by the next chunk.
_FORCE_NUMPY = bool(os.environ.get("MOMWIRE_RAZOR_FORCE_NUMPY"))


# The fused T1 assembly (momwire#780), flagged on its OWN symbol for the same
# reason `razor_fill_742` is: a .so built before this arc exports the moments
# kernel and not this one.
_HAVE_RAZOR_ASSEMBLE_ACCEL = _acc is not None and bool(
    getattr(_acc, "razor_assemble_780", False)
)


# The in-medium (complex k) twin of the moment fill (momwire#796), on its OWN
# symbol for the same reason the other two are: a .so built before #796 landed
# exports `razor_seg_moments` and not `razor_seg_moments_cplx`, and the gate
# below has to be able to tell rather than assume one implies the other.
_HAVE_RAZOR_CPLX_ACCEL = _acc is not None and bool(
    getattr(_acc, "razor_cplx_796", False)
)


# The fused WEIGHTED T1 assembly (momwire#744), on its OWN symbol for the same
# reason: a .so built before #744 landed exports `razor_assemble_t1` and not
# its weighted twin, and one symbol must never be read as vouching for the
# other.
# momwire#814 (razor buried, unit 3): THE ONE CONSTANT THE FLIP MOVES.
#
# Three things have to change together for razor to serve buried decks — the
# wholly-below family (momwire#812), the crossing family (momwire#813) and the
# declared `buried` capability cell — and while they were three independent
# `False`s the flip was three edits that could be made separately and land
# half-done. A deck served by the fill while the row still declares a refusal
# is the worst of the three states: consumers read the row, so the deck would
# be refused by the roster and served by the solver at the same time.
#
# They are derived from one name instead. `tests/test_814_prep.py` holds the
# derivation, so flipping this line is the whole flip and nothing else has to
# be remembered.
_SERVE_BURIED = False

# momwire#812 (razor buried, unit 1): serve a WHOLLY-below deck through the
# lower-medium family. Kept as its own NAME because the unit's gates
# monkeypatch it by name and read it at call time; its VALUE is not its own.
_SERVE_BELOW_PLANE = _SERVE_BURIED

# momwire#813 unit 2: serve a deck whose wires span the interface through a
# crossing junction. Same rule: its own name for the gates, its value from
# `_SERVE_BURIED`.
_SERVE_CROSSING = _SERVE_BURIED

# The sentence a crossing deck gets while `_SERVE_CROSSING` is off. It is
# ALSO the refusal razor's capability row declares for the
# `buried+crossing_junction` cell below, and it must stay one object: the row promises the sentence a
# refusal ends with, and antennaknobs' catalog gate
# (`test_razor_2p_on_the_buried_decks_follows_its_capability_cell`) holds
# razor to that promise on the bonded screen. Before antennaknobs#1109 the
# catalog's node was never a declared junction, so razor refused at the
# `buried` cell first and the declared crossing sentence (bspline's) was
# never emitted; with the node declared, this is the sentence that fires.
# momwire#846: N geometrically COINCIDENT segments give the tent basis N
# identical columns, so the matrix is singular and the solve dies in LAPACK
# with no idea what the deck was. Declared here and raised from the solve
# entry points, not from the constructor: the FILL on such a deck is well
# defined and momwire#813's collapse gates measure it (they compare matrices
# and never solve, which is exactly why the singularity was invisible to them
# until step 4 ran one). It is the SOLVE that has no answer.
_BUNDLE_REFUSAL = (
    "this deck spells a conductor as N geometrically COINCIDENT segments (a "
    "bundle). Razor's tent basis puts one column per segment, so N coincident "
    "segments are N identical columns and the matrix is singular by "
    "construction -- at any mesh, in free space and in soil alike, and "
    "whatever the quadrature. Razor has no bundle rule; BSplineSolver does "
    "(momwire#524 phase 2's fan widening). Respell the bundle as ONE "
    "conductor: a screen whose N radials meet at a buried HUB and rise to the "
    "node on a single rise is the same antenna without the coincidence, and "
    "razor serves it (antennaknobs' `buried_radial_vertical` is spelled that "
    "way since antennaknobs#1108; its `bundle` variant is this deck). Or "
    "solve the bundle with BSplineSolver, remembering that a bundle of N "
    "coincident thin wires and one wire of the same radius are two "
    "structures, never two meshes of one"
)

_CROSSING_NOT_SERVED_REFUSAL = (
    "this deck's wires cross the interface at a junction. Razor's crossing "
    "fill is momwire#813 and is not served yet; the below-plane family it "
    "stands on is momwire#812. Solve it with BSplineSolver, which serves the "
    "crossing junction since momwire#524 phase 2, or leave the buried part "
    "DETACHED"
)

# The crossing blocks' axis density (momwire#813). Razor's cross rows are
# PATH-tested and one of them ends AT the node, on the below wire's last
# segment, whose by-parts integrand ~ 1/sqrt(a^2 + s^2) from s = 0 is carried
# by `_crossing_fill._graded_u`'s a-scale panels. The trunk's shipped
# defaults (growth 4.0, Gauss-4 per panel, `_NEAR_Q` = 4) are a GALERKIN
# axis's setting and leave that row at 5.3e-5.
#
# The two error plateaus are separate, which is what makes this easy to
# mis-measure: sweeping `q` alone leaves 5.3e-5 untouched at any order to 32,
# and sweeping `panel_order` alone stops at 2.2e-6 at any growth (momwire#836
# records both, one test each). Measured on the whole-matrix eps~ = 1
# collapse, both quadrature lanes identical:
#
#   growth  panel   q  | crossing_deck(1)  fan_rise_deck()  (axis nodes)
#      4.0      4   4  |      2.643e-05        2.649e-05     144 /  320
#      2.0      8   8  |      1.909e-11        3.202e-09     320 /  768
#      2.0      8  12  |      6.829e-13        7.254e-11     432 / 1000
#      2.0     16  12  |      8.532e-13        7.171e-11     528 / 1304
#
# So (2.0, 8, 12) is the floor for both decks at 3x the shipped node count,
# and 16-per-panel buys nothing over 8. The cost is the crossing blocks'
# only, and only on a deck that HAS a crossing junction.
_CROSSING_GROWTH = 2.0
_CROSSING_PANEL_ORDER = 8
_CROSSING_Q = 12

_HAVE_RAZOR_WEIGHTED_ACCEL = _acc is not None and bool(
    getattr(_acc, "razor_weighted_744", False)
)


# ---------------------------------------------------------------------------
# The outer testing-path order, derived from the mesh (momwire#800)
# ---------------------------------------------------------------------------
# What #754 measured (PR #795) is that 32 is the COARSE-mesh answer: on the
# binding deck -- a 90 degree corner -- the order the outer integral needs
# falls as the mesh refines, and every rung #754's own timing table quotes
# sits where half of it would do.
#
#   bent   N=30   N=60   N=120  |  N=240  N=400
#   need    32     32     32    |   16     16
#   k*h    .102   .051   .0254  |  .0127  .0076
#
# WHAT THE SCALAR HAS TO BE, AND WHY IT IS NOT A DISTANCE RATIO
# -------------------------------------------------------------
# #800 proposed `h_max / d_min`, the longest half-segment over the shortest
# inter-arm distance. That cannot work, and the reason is structural rather
# than a matter of calibration: these decks refine SELF-SIMILARLY, so every
# inter-segment distance scales with h and any ratio of two mesh-set lengths
# is mesh-INVARIANT. Measured on `bent`, h_max/d_min is 0.6325 at N=30 and
# 0.6325 at N=400, to four figures, for every neighbour exclusion tried.
#
# The two lengths a mesh does not set are the wavelength and the wire
# radius, so the only candidates are `k*h_max` and `h_max/a`. That maps onto
# the physics as two separate terms, which #800 conflated into one:
#
#   * the DECK's floor -- how fast a neighbouring conductor's kernel varies
#     across a testing path -- is the h/d_min term, and it is exactly the
#     part that does NOT move with the mesh. It is why `bent` (0.63) needs
#     more than `straight` (0.50) at every mesh, and it is why a blind
#     default has to serve the corner.
#   * the MESH's saving is the phase variation across the path, k*h/2, which
#     is what falls under refinement and what this rule reads.
#
# `k*h_max` over `h_max/a`, on the evidence of the `ek` deck: its radius
# scales with the mesh (a = L/N/4), so h/a is pinned at 4.0 at every N and a
# rule keyed on it could never grant that deck a fine-mesh saving, while
# k*h_max falls 0.102 -> 0.0076 across the same ladder.
#
# WHERE THE SWITCH SITS
# ---------------------
# Between the corner's last coarse row (N=120, k*h = 0.0254, needs 32) and
# its first fine one (N=240, k*h = 0.0127, needs 16). Those are a factor of
# two apart, and the switch is their GEOMETRIC mean -- so the margin is
# sqrt(2) on each side rather than tight against either. In segments per
# wavelength that is about 350: coarser than that takes 32, finer takes 16.
#
# 32 remains the documented ceiling of what this returns, 16 the floor: no
# deck in the bank is converged at q=8 on the corner at ANY mesh measured
# (1.1e-6 relative at N=400, still over #754's 1e-6 bar), so there is no
# evidence for a third rung and the rule does not invent one.
PATH_ORDER_COARSE = 32
PATH_ORDER_FINE = 16
PATH_ORDER_KH_SWITCH = 0.018  # sqrt(0.0254 * 0.0127), the corner's own gap


def derive_n_qp_path(k, wires_polylines, n_per_edge_per_wire):
    """The outer testing-path order for this mesh — momwire#800.

    A pure function of the geometry and the wavenumber, not of the segment
    COUNT: two decks with the same electrical segment length get the same
    order whatever their N, which is what lets `contact_pec` at N=240 (a
    half-length arm, so k*h = 0.0063) sort with the fine meshes rather than
    with its own segment count.

    Returns `PATH_ORDER_FINE` or `PATH_ORDER_COARSE`; see the block comment
    above for the calibration and for why the scalar is `k*h_max`.
    """
    h_max = 0.0
    for pl, npe in zip(wires_polylines, n_per_edge_per_wire):
        for e_idx in range(pl.shape[0] - 1):
            edge_len = float(np.linalg.norm(pl[e_idx + 1] - pl[e_idx]))
            h_max = max(h_max, edge_len / npe[e_idx])
    if k * h_max <= PATH_ORDER_KH_SWITCH:
        return PATH_ORDER_FINE
    return PATH_ORDER_COARSE


def _use_razor_fill_accel():
    """The fused C++ moment fill serves when built and not forced off."""
    return _HAVE_RAZOR_FILL_ACCEL and not _FORCE_NUMPY


def _use_razor_cplx_accel():
    """The complex-k kernel serves when built and not forced off.

    Read separately from `_use_razor_fill_accel`: a build can have the real-k
    fill without this one, and that combination must take the numpy lane on a
    complex k rather than the fused lane (momwire#796).
    """
    return _HAVE_RAZOR_CPLX_ACCEL and not _FORCE_NUMPY


def _use_razor_assemble_accel():
    """The fused C++ T1 assembly, under the same two off-switches.

    Both quadrature lanes go through it: `n_path` is the only thing that
    differs between them (2 under `nec5_quadrature`, 2*`n_qp_path` under
    Gauss-Legendre) and it is a loop bound inside the kernel, not a branch.
    """
    return _HAVE_RAZOR_ASSEMBLE_ACCEL and not _FORCE_NUMPY


def _use_razor_weighted_accel():
    """The fused C++ WEIGHTED T1 assembly (momwire#744), same two off-switches.

    Separate from `_use_razor_assemble_accel` because the two kernels are
    separate symbols: a .so carrying the unweighted assembler need not carry
    this one, and the weighted branch must fall back on its own evidence.
    """
    return _HAVE_RAZOR_WEIGHTED_ACCEL and not _FORCE_NUMPY


def _weighted_window_rule(ground):
    """The fused weighted assembler's window, or None to keep numpy.

    A question the GROUND answers about itself (momwire#806). The fill used to
    decide this by reading `ground.mode`, `ground.eps_tilde`,
    `ground.standard_fresnel` and -- in #804's dropped draft --
    `ground.image_coefficient`, which is exactly the reading that doubled the
    exact-image half and that
    `test_the_consumer_never_applies_the_image_coefficient_itself` exists to
    catch. `PotentialGround.fused_window_rule` now hands over the window as a
    rule, coefficient included, from the same `self` its closure reads; this
    fill forwards it opaquely and names none of those attributes.

    `None` covers free space (no ground object at all), PEC (no weighted
    branch to fuse), and any ground that declines to be a rule -- today the
    radial screen's `standard_fresnel = False` row, whose screen-modified
    rho_v / rho_h the stock chain would silently get wrong.
    """
    if ground is None:
        return None
    return ground.fused_window_rule()


# momwire#510: the ceiling the grazing-height keying below may raise the
# Sommerfeld remainder's source order to, and the constant in the rule.
#
# The rule is one Gauss point per closest-approach distance along the source
# segment — `n_qp ≈ len / R_min`, with `R_min` the nearest an observer comes
# to that segment's MIRROR. Measured against the licensed binary on capture
# 0033 (a 39.624 m radial 1.778 cm over average soil, so len/R_min ≈ 223):
# order 3 is 171.86 % out, 48 is 26.05 %, 96 is 9.87 % and 192 is 1.44 %.
# The two rungs either side agree on the constant — len/R_min = 48.4 needs
# ~48 and 121 needs ~96-192 — so C = 1.0 is what the measurement says and
# not a fitted number.
_REMAINDER_QP_C = 1.0
_REMAINDER_QP_CAP = 192

# Constructor kwargs the sibling solvers accept that this formulation
# deliberately does not, with the reason each is refused. Anything else
# unexpected is a caller typo and stays a TypeError.
_OUT_OF_SCOPE = {
    "degree": "RazorSolver has no degree: the razor-blade testing rule is "
    "defined against the tent (degree-1) expansion. Use "
    "BSplineSolver(degree=...) for higher-order bases with Galerkin testing",
    "junction_ports": "junction ports are not supported: a junction basis is "
    "already a through-current unknown, so a port that adds one would be a "
    "second unknown for one current",
}

# momwire#651's first half. Razor's buried refusal was raised inline in
# `_refuse_buried_geometry` and appeared nowhere in the declared row, so
# `capabilities.refusal("buried", "sommerfeld")` said None on the one deck
# class this solver is certain to raise on. Named here so the row and the
# raise are one sentence; the raised message is unchanged.
#
# This is the ONE buried reading that is razor's own. The other three — a
# wire crossing the interface mid-span, a wire below a ground with no lower
# medium, and contact plus buried — come out of `_medium_spec` with the same
# sentences `BSplineSolver` raises, which is the point of routing both trunks
# through it, so the row quotes those from there.
# momwire#813 unit 2: a CHOPPED testing row — one whose T2 takes the knot as
# an endpoint instead of a centroid — is the crossing arc's row, and that arc
# declines the extended kernel for the same reason the below-plane fill does
# (NEC's O(a^2) tube expansion was derived in free space, not in a medium, and
# a chopped row's knot observer sits ON the interface where the eligibility
# scan has no answer). Refused rather than labelled by guess.
_CHOP_EK_REFUSAL = (
    "razor's chopped testing rows (momwire#813) do not take the extended "
    "kernel: the knot observer sits at the interface, where the "
    "coaxial-and-equal-radius eligibility scan the pair rule reads has no "
    "answer, and the crossing arc declines EK in a medium anyway "
    "(momwire#812). Build the crossing fill with extended_kernel=False"
)

_BURIED_FILL_REFUSAL = (
    "RazorSolver has no "
    "buried fill: the momwire#553 buried serve (direct + image "
    "+ Sommerfeld-remainder blocks in the lower medium) is "
    "written for BSplineSolver's testing side only. A detached "
    "buried wire is a LEGAL deck - solve it with BSplineSolver, "
    "which serves buried ground since momwire#553, or raise the "
    "wire clear of the plane. Razor's own below-plane fill exists since "
    "momwire#812 (the lower-medium family behind `_SERVE_BELOW_PLANE`, "
    "wholly-below decks only); the crossing block on razor rows "
    "(momwire#813) and the roster flip (momwire#814) are what turn this "
    "cell True; the arc is momwire#651"
)


# momwire#624 removed `_CONTACT_OVER_FINITE_REFUSAL` from this module. A
# grounded end over the SOMMERFELD ground is served here now, and what is
# left refused is the row `_ground_spec` owns for every solver that has one:
# contact under `refl-coef`, withdrawn from the whole tree by momwire#282
# stage 1's D3 because the MODEL is wrong at zero clearance (stock nec2c
# prints 175 - 779j Ω on the same deck). Razor reaches that refusal through
# the same `_ground_spec.contact_ends` scan and the same prose as
# `BSplineSolver` and `SinusoidalSolver`, so there is one sentence for it in
# the tree rather than a fourth copy.
#
# What the old refusal claimed, and what measuring it found. Its text named
# a real asymmetry — over a finite ground the T2 drop discards
# (1 - w_Phi)*M0(plane) rather than zero (§4.3) — and offered restoring that
# term as the fix, explicitly as "a hypothesis, not a diagnosis". §5.5's
# experiment ran under momwire#624 and the hypothesis did not survive it:
# the term makes the binary comparison WORSE at full strength, and on the
# stubbed ladder — the instrument that needs no reference — coefficient 0 is
# flattest everywhere, so no scale for it is self-consistent. `_fill_T2`'s
# grounded branch carries the measurement.
#
# The refusal was therefore not protecting a defect the term would repair.
# What it was costing is five of the EZNEC captures — 0021, 0047, 0048, 0110
# and 0111, five of the 62 the corpus held then and five of 80 now — and the most common
# HF model there is, a base-fed vertical over real ground, on an engine a
# user points EZNEC at. Serving it puts razor on bspline's bar (D1): a
# residual that is BOUNDED and saturating rather than diverging, pinned by
# an envelope with the saturation checked. Measured at N = 61 against the
# binary's own printed shift, razor is 0.005 Ω on sea water where bspline is
# 0.201, and 3.384 Ω on poor soil where bspline is 3.309 — the same row,
# not a worse one.


def _remainder_qp(obs_pts, src_l, src_r, ground_z, base, cap=None):
    """The Sommerfeld remainder's source order, keyed to grazing height.

    momwire#510.  ``field_windows`` lays a single Gauss rule of one order over
    every source segment, and the constructor's default of 3 rests on the
    remainder field being "smooth on the scale of a segment" — true wherever
    this unit was gated (its hardest deck is a dipole 0.04 λ up) and false at
    grazing.  When an observer sits almost directly over a source segment's
    IMAGE, the projected remainder has a spike of width ~R_min in a segment of
    length ``len``, and three points cannot see a feature of relative width
    R_min/len.  On capture 0033 that ratio is 1/223 and the served impedance
    was 171.86 % from the binary; at order 192 it is 1.44 %.

    So the order is keyed to the geometry the same way momwire#443 keyed the
    interpolation grid to its boundary layer: ``ceil(C · len / R_min)``, with
    ``R_min`` the nearest any observer comes to the segment's mirror, clipped
    below by ``base`` and above by ``cap``.

    Two properties matter as much as the rule.

    **A deck with nothing grazing is bit-identical.**  Every ratio comes out
    below 1 and the clip returns ``base`` exactly, so the order is the number
    it always was and no shipped gate moves.  That is why the keying is a
    max-with-base rather than a replacement.

    **The cap is a real limit, not a formality.**  The order is one scalar for
    the whole fill, so a single grazing pair raises it for every source
    segment; the cap is what stops one wire in a large model multiplying the
    remainder's cost without bound.  A deck grazing enough to need more than
    ``cap`` is served MORE accurately than before but not to the binary — see
    ``docs/`` and the arc's record.  Per-segment orders would remove that
    coupling and are the follow-up, not this change.

    The rule itself is ``_quadrature.remainder_qp`` — a statement about
    GEOMETRY, not about razor's formulation, and momwire#631 found bspline
    needs the identical one.  This wrapper is what keeps razor's own cap and
    constant the things that decide razor's order.
    """
    # Read at CALL time, not bound as a default: a default argument captures
    # the module constant when this function is defined, which silently makes
    # the cap unpatchable — and the gate that pins the pre-#510 behaviour has
    # to be able to move it.
    cap = _REMAINDER_QP_CAP if cap is None else cap
    return _quadrature.remainder_qp(
        obs_pts, src_l, src_r, ground_z, base, cap, _REMAINDER_QP_C
    )


class _PreparedChunks:
    """The numpy fill's chunk list, BOUND to the source set it was built from.

    :class:`_FusedMoments` binds the geometry prepare was handed; this is the
    same property for the other lane, and it exists so the two lanes agree for
    a REASON rather than by luck (momwire#745). The numpy replay used to read
    `seg_h` back off a `geom` argument its callers passed as the REAL geometry
    even for the image block, whose chunks were prepared from the MIRRORED
    source set. That agreed only because `_image_sources` mirrors — and a
    mirror preserves segment lengths. Bind them here and the coincidence is
    not load-bearing any more.

    Iterable, because a chunk list is what every consumer wants and the
    binding is the only thing being added.
    """

    __slots__ = ("chunks", "seg_h")

    def __init__(self, chunks, seg_h):
        self.chunks = chunks
        self.seg_h = seg_h

    def __iter__(self):
        return iter(self.chunks)


class _FusedMoments:
    """The C++ fill's stand-in for a prepared moment chunk list (momwire#742).

    `RazorSolver._seg_moments_prepare` returns one of these instead of the
    ``(lo, hi, R, m0s, m1s, ekc)`` list when the accelerator is built, and
    `_seg_moments_from_prepared` recognises it and calls the kernel. Every
    caller in between — the T2 centroid block, the T1 row chunks, both source
    sets, both quadrature lanes — only ever passes the token along, so this is
    the whole dispatch surface.

    What it holds is the ARGUMENTS of the fill, not its tables: the observers,
    this source set's segment origins/tangents/lengths, the regularising
    radius column, and the extended kernel's eligibility labels. All of that
    is O(n_obs + n_seg); the tables it replaces are O(n_obs · n_seg · n_qp).
    `a` is normalised to a ``(n_seg,)`` column here rather than in the kernel
    because that is where the scalar-or-column convention is already written
    down (`_seg_moments_prepare`'s docstring), and because a uniform column is
    bit-for-bit the scalar in `a * a` (momwire#425).

    The `geom` captured is the one PREPARE was handed, which over a ground is
    the MIRRORED source set. :class:`_PreparedChunks` binds the same thing for
    the numpy lane (momwire#745), so neither path can read segment lengths off
    a geometry that is not the source set it integrated — the agreement is a
    property of both spellings now, not of `_image_sources` happening to
    preserve lengths.
    """

    __slots__ = (
        "obs",
        "seg_p0",
        "seg_t",
        "seg_h",
        "a",
        "group_i",
        "group_j",
        "a_ek",
        "xg",
        "wg",
        # The complex-k fallback's memo (momwire#796). Deliberately the ONLY
        # thing added to the token: the numpy lane it may have to build needs
        # nothing this object is not already holding, so the fill's arguments
        # stay the whole of its state and the #742 residency gate keeps its
        # meaning. Empty unless a build without the complex kernel is asked
        # for a complex k.
        "_numpy",
    )

    _EMPTY_I64 = np.empty(0, dtype=np.int64)
    _EMPTY_F64 = np.empty(0, dtype=np.float64)

    def __init__(self, obs, geom, a, ek, n_qp_source):
        self.obs = obs
        self.seg_p0 = geom["seg_p0"]
        self.seg_t = geom["seg_t"]
        self.seg_h = geom["seg_h"]
        n_seg = self.seg_h.size
        self.a = np.broadcast_to(np.asarray(a, dtype=float), (n_seg,)).copy()
        if ek is None or ek.group_i is None or ek.group_j is None:
            # Two empty label arrays are how the kernel spells "reduced
            # kernel everywhere", which is `_ek_pair_mask`'s all-True case
            # read from the other side.
            self.group_i = self._EMPTY_I64
            self.group_j = self._EMPTY_I64
            self.a_ek = self._EMPTY_F64
        else:
            self.group_i = np.ascontiguousarray(ek.group_i, dtype=np.int64)
            self.group_j = np.ascontiguousarray(ek.group_j, dtype=np.int64)
            self.a_ek = np.broadcast_to(
                np.asarray(_ek_radius(ek, self.a), dtype=float), (n_seg,)
            ).copy()
        self.xg, self.wg = leggauss(n_qp_source)
        self._numpy = None

    def _numpy_lane(self, solver):
        """The numpy chunk list for this same source set, built once.

        REBUILT from this token's own arrays rather than from retained
        `geom`/`ek` references: everything the numpy prepare reads is already
        here — the three geometry columns it takes off `geom`, and the EK
        labels plus `a_ek`, which IS `_ek_radius(ek, a)` and so reconstructs
        the `_EK` faithfully. Keeping it that way is what lets the token stay
        the fill's arguments and nothing else (the #742 residency gate).

        Only a complex k on a build WITHOUT the `_cplx` kernel reaches here,
        so the O(n_obs·n_seg·n_qp) tables the fused lane exists to avoid are
        formed only when there is no fused lane to take.
        """
        if self._numpy is None:
            ek = None
            if self.group_i.size and self.group_j.size:
                ek = _EK(a=self.a_ek, group_i=self.group_i, group_j=self.group_j)
            geom = {
                "seg_p0": self.seg_p0,
                "seg_t": self.seg_t,
                "seg_h": self.seg_h,
            }
            self._numpy = solver._seg_moments_prepare_numpy(
                self.obs, geom, self.a, ek=ek
            )
        return self._numpy

    def evaluate(self, solver, k, *, need_m1, n_obs):
        """Both halves of the split, at one wavenumber, in one kernel call.

        **The k-type gate (momwire#796).** A real k goes to the real-k kernel,
        as it always has. A complex (in-medium) k goes to `_cplx` when the
        build has it and to the numpy lane when it does not — never to
        `float(k)`, which is the TypeError #796 was filed for. `_complex_k`
        is the shared predicate and carries the Im k > 0 refusal, so the
        growing-exponential branch raises `ValueError` here as well as at the
        entry point.
        """
        if n_obs != self.obs.shape[0]:
            raise AssertionError(
                f"prepared for {self.obs.shape[0]} observers, replayed at {n_obs}"
            )
        if _complex_k(k):
            if not _use_razor_cplx_accel():
                return solver._seg_moments_from_prepared(
                    self._numpy_lane(solver), k, n_obs, need_m1=need_m1
                )
            kernel, kval = _acc.razor_seg_moments_cplx, complex(k)
        else:
            kernel, kval = _acc.razor_seg_moments, float(k)
        M0, M1 = kernel(
            self.obs,
            self.seg_p0,
            self.seg_t,
            self.seg_h,
            self.a,
            self.xg,
            self.wg,
            kval,
            bool(need_m1),
            self.group_i,
            self.group_j,
            self.a_ek,
            solver._cancel_flag,
        )
        return M0, (M1 if need_m1 else None)


@dataclass(frozen=True)
class _RazorBasis:
    """Opaque `PortSolution.basis` payload for `RazorSolver` (#429 rank-9).

    The per-solve context: the geometry — wing/path stencils, knot layout,
    junction table — that a solved `coeffs` column is expressed against, and
    the wavenumber it was solved at. `currents_at_knots` does not actually
    read this handle: it rebuilds geometry off `self._build_geometry()`'s own
    cache, which is stable for the solver's whole lifetime (one `wires`
    config, one mesh), so nothing here is load-bearing for TODAY's readout.
    It exists so the `basis` field has a concrete, private-typed home rather
    than `None` or the bare geometry dict — #232's contract is an opaque
    handle a consumer never introspects, and a future readout that DOES need
    solve-scoped context (a batched accelerator, say) has somewhere to add
    it without changing the field's meaning. Private on purpose. Not stable
    across solves.
    """

    geom: dict
    k: float


# The `centre_feeds` refusal (momwire#673), one sentence so the raise in
# `momwire.deck.build_solver` and the matrix row are the same words --
# `tests/test_refusals_are_declared.py` checks exactly that.
_CENTRE_FEEDS_REFUSAL = (
    "RazorSolver places a gap at the nearest basis-carrying KNOT "
    "(`_snap_to_knot`), so a feed named as a segment CENTRE -- which is the "
    "grid the nec2 dialect addresses -- lands half a cell from where it was "
    "named; build the solver directly with a parity-correct mesh if that is "
    "what you want."
)


class RazorSolver(_ElementCurrents, _SweptPortSolutions, _Cancelable):
    """Tent-basis MoM with razor-blade (mixed-potential path) testing.

    The NEC-5 formulation twin — see the module docstring for the physics.
    Free space or any of the three grounds, either kernel, one tent per
    interior knot plus K−1 through-current tents wherever K wire ends meet.

    wires: list of (M_w, 3) polyline arrays, M_w >= 2 anchor points per wire.
        A straight dipole is a single two-anchor wire; an inverted-V is one
        three-anchor wire; a Yagi is several two-anchor wires.
    n_per_edge_per_wire: list of (int or sequence). Per-wire segment counts
        per polyline edge. None for a wire means use `nsegs` on each of its
        edges; an int means that count on each edge; a sequence gives a
        per-edge count. None for the whole argument means every wire uses
        `nsegs` on every edge.
    nsegs: default segment count when `n_per_edge_per_wire` doesn't specify.
    wire_radius: thin-wire radius, the a in the reduced kernel's
        R = sqrt(|r−r'|² + a²). A scalar applies to every wire; a
        length-n_wires sequence gives each wire (polyline) its own conductor
        radius (momwire#147), the same spelling `BSplineSolver` and
        `SinusoidalSolver` take. Every entry must be positive and finite.
        A uniform model — however it was spelled — keeps the scalar code
        path and is bit-identical to the scalar. **Mixed radii use the
        SOURCE segment's radius** in the a²-regularised kernel; the tents of
        a fat/thin JUNCTION therefore take each wing's own segment's radius,
        with no special case, which is the same statement. See
        `_seg_moments_prepare` for the convention's measurement against the
        binary (which cannot separate it from the observer convention the
        siblings use), and `tests/test_razor_mixed_radius.py` for the gates.
        Under `extended_kernel` the same number is also the EK radius, since
        eligibility requires the pair's two radii to be equal.
    extended_kernel: use NEC's EXTENDED (tubular) kernel on the eligible
        pairs instead of the reduced one (momwire#398 D1). False — the default —
        is the reduced kernel and is bit-for-bit what this class always
        computed. True is the row the taper study's kernel identification
        calls for on fat wire: the NEC-5 binary is extended-kernel
        everywhere, so `extended_kernel=True, nec5_quadrature=True` is the
        twin on fat and tapered sections (measured: a constant offset to
        0.012/0.021 Ω down the study's `fat` ladder against a 0.05 Ω bar,
        and a continuum limit 0.047 Ω from the binary's, where the reduced
        row sits 1.40 Ω away), while the reduced kernel stays the twin on
        thin wire, where the two kernels agree to 1e-4 Ω and the extra
        arithmetic buys nothing. Eligibility is the shared coaxial-and-
        equal-radius pair rule (`_bspline_kernels._ek_axis_groups`) — see the
        module docstring for how it composes with the grounds, ground
        contact, per-wire radii, loading and the two quadrature lanes, all of
        which are served with it on. The house kwarg name and semantics,
        identical to `BSplineSolver`'s and `SinusoidalSolver`'s, so a deck's
        `EK` card reaches this class through the same `build_solver` line
        that reaches them.
    ground_z: height of the ground plane, or None for free space. With
        `ground_eps` unset the plane is a perfect conductor. Over PEC this
        solver is gated against the licensed NEC-5 binary's own `GN 1`
        printouts on four ladder geometries, at the sharp tolerance the
        formulation twin can actually hold (`tests/test_razor_pec_ground.py`)
        — the exact image is the same object in both codes, so there is
        nothing to disagree about but quadrature. A wire END may lie in the
        plane — ground CONTACT, the grounded-end tent whose lower wing is
        its own image (momwire#398 unit 3, and the module docstring for the
        physics) — which is what the vertical/monopole class needs. A wire
        with points below the plane is refused through the shared
        `_medium_spec` sentences (momwire#651) — crossing, no-lower-medium,
        contact+buried — or with razor's own buried-fill gap sentence; an
        edge lying in the plane or an interior-anchor touchdown is refused
        by name here.
    ground_eps: complex relative permittivity ε̃ (Im ≤ 0 for a passive
        ground in momwire's e^{+jωt} convention) or an `(eps_r, sigma)`
        tuple with sigma in S/m, for the NEC `GN 0` style
        reflection-coefficient ground (momwire#398 unit 4). None — the
        default — keeps the PEC image. Requires `ground_z`. The image block
        is weighted per (observer, source segment) pair by the Fresnel
        coefficients at that pair's specular angle: the A term takes the
        field dyad's w_A in place of the PEC mirror tangent dot, and the
        charge term takes w_Φ, which the PEC path leaves unweighted. Both
        weights come from `_potential_ground.PotentialGround.weight_windows`
        — this solver computes no reflection coefficient of its own.
        Validity window (momwire#151/#153): 0.1–0.5 λ above the plane, for
        `ground_model="refl-coef"`. Below that the Φ term's approximate
        weighting degrades — use `ground_model="sommerfeld"`, which is exact
        at every height. NO ground CONTACT is served over either finite
        ground (momwire#282 — the fold hard-codes image coefficient 1, so a
        grounded end would take spurious contact charge); a wire end in the
        plane with `ground_eps` set is refused. NEC-5 is NOT the oracle for
        either: its finite ground is Michalski, carrying a limit offset of
        its own, so both are gated by cross-formulation agreement against
        momwire's own B-spline and sinusoidal solvers instead
        (`tests/test_razor_refl_coef_ground.py`,
        `tests/test_razor_sommerfeld_ground.py`).
    ground_phi_mode: which image-charge (Φ-term) weighting the
        reflection-coefficient ground uses — one of
        `_ground_refl.PHI_MODES` ("rho_v", "image", "normal", "blend"),
        default "normal", exactly `BSplineSolver`'s set, semantics and
        default. It is the one knob the mixed-potential form has and the
        field form does not: NEC weights fields, and this trunk separates
        the charge term, so the image charge's weight is a modelling
        choice. Ignored without `ground_eps`.
    ground_model: "refl-coef" (the default, and what `ground_eps` selects)
        or "sommerfeld" — the exact NEC `GN 2` style ground (momwire#398
        unit 5), which requires `ground_eps` to be the permittivity of, and
        is served at ANY height rather than in a validity window. It is
        this solver's one COMPOSING ground: the image block is the exact
        image scaled by the constant C₂ = (ε̃−1)/(ε̃+1) (which arrives
        through the same weight windows the Fresnel ground uses) PLUS the
        smooth remainder field, summed before the fold's single minus. See
        the module docstring for what composing costs the schedule.
        `ground_phi_mode` is unread over it: the Sommerfeld image
        coefficient is exact and has no knob, exactly as in `BSplineSolver`.
    wavelength: measurement wavelength in metres; k = 2π / wavelength.
    halfdriver_factor: informational only when `wires` is given explicitly
        (the polylines fully determine the geometry); kept for signature
        parity with the sibling solvers.
    feed_wire_index: wire carrying the delta-gap source on the single-feed
        path (ignored when `feeds` is supplied).
    feed_arclength: arc length along the feed wire, from its first anchor,
        at which to place the source. None picks the wire's midpoint. The
        source always snaps to the NEAREST KNOT THAT CARRIES A BASIS on
        that wire — every interior knot, plus either end that meets other
        wire ends at a junction, plus either end that touches the ground
        plane: the razor testing paths are knot-centred, so a between-knots
        delta-gap has no row to land in. On an even segment count a
        midpoint feed is already the exact centre knot. A feed that snaps
        to a junction of K >= 3 ends is refused (which branch pair the
        source drives is ambiguous); K = 2 — the ordinary split-wire feed —
        drives that junction's through-current basis. A base-fed monopole
        is `feed_arclength=0.0` on a wire whose first anchor lies in the
        plane (or the wire's full arc length, if it is the last anchor that
        touches): the source is then the gap between the plane and that
        end, and V / I is the drive-point impedance measured there.
    feeds: optional list of (wire_index, arclength_or_None, voltage) tuples
        describing several delta-gap sources with prescribed complex
        voltages. `compute_impedance()` then returns the per-feed
        drive-point impedance vector V_i / I_i.
    n_qp_path: Gauss-Legendre order for the OUTER testing-path integral,
        per half-segment (T1 only; T2's path collapses to two endpoint
        evaluations). Ignored under `nec5_quadrature`.

        **`None` (the default) means DERIVE it from the mesh
        (momwire#800)**, the way `auto_mesh=None` derives a density; an
        explicit integer is taken verbatim and reproduces the pre-#800
        answer bit for bit. The derivation is `derive_n_qp_path`, whose
        block comment carries the calibration: 32 below about 350 segments
        per wavelength and 16 above it, because what #754 measured is that
        32 is the COARSE-mesh answer and every rung its own timing table
        quotes sits where 16 would do. Half the outer integral's cost on
        exactly the meshes where razor's fill is felt.

        The rest of this note is #754's derivation of the constant, which
        stands as the coarse-mesh half of the answer.

        **32 is derived, not a first cut (2026-09-02, momwire#754).**
        `scripts/probe_razor_path_754.py` swept twelve decks across the
        geometry classes that stress the outer integral hardest -- a 90
        degree corner, a junction with a radius step, close-spaced elements
        at both catalog scales, ground CONTACT over PEC and Sommerfeld, the
        extended kernel, graded and split meshes -- scored against a
        converged q=128 reference. #754 proposed 8, on the strength of three
        straight dipoles; straight dipoles are simply the easy class. The
        binding deck is the corner, which at N=60 is still 1.0e-4 relative
        at q=8 and 1.4e-6 at q=16, four orders of magnitude worse than a
        straight dipole at the same rung. Applying #754's own rule -- 2x
        margin over the last q moving by more than 1e-6 relative -- returns
        32, the value already here.

        The order needed FALLS as the mesh refines (q=32 at N<=120, q=16 at
        N>=240), so 32 is the coarse-mesh answer and a default is applied
        blindly. A mesh-aware order, not a smaller constant, is the shape of
        any real saving; the cost is linear in the order, so q=16 would
        halve the fill on meshes that can take it.
    nec5_quadrature: evaluate ∫A·dl by NEC-5's own rule — the two-point
        trapezoid at the path-end centroids (every potential at element
        centroids, the classic mixed-potential idiom), identified by the
        momwire#316 residue study: with it, the free-space ByDipole1
        ladder matches NEC-5's printouts to a CONSTANT −0.004−0.037j Ω at
        every rung, pair-walk signature identical to the third decimal.
        Default False keeps the converged Gauss-Legendre path integral, so
        the walk you see is the SCHEME's discretization error, cleanly.
        On non-uniform meshes each wing weighs its own half-path length
        (h_wing/2) at its centroid — the element-local reading, identical
        to the whole-path trapezoid on the uniform meshes the rule was
        identified on. Demonstration mode for the census rationale, not an
        NEC-5 substitute (the licensed binary stays the oracle).
        **Orthogonal to `extended_kernel`**: this chooses where the testing
        path is sampled, that chooses which kernel is sampled there, and
        neither reads the other. All four combinations are served; the
        fat-wire twin is both of them on at once, which is the two
        independent NEC-5 identifications (momwire#316's quadrature idiom and
        momwire#398 D1's kernel) applied together.
    n_qp_source: Gauss-Legendre order per source segment for the smooth
        remainder (exp(−jkR)−1)/(4πR); the static 1/(4πR) part is analytic.
    n_qp_sommerfeld: Gauss-Legendre order per source segment for the
        SOMMERFELD remainder field's tent moments, unread unless
        `ground_model="sommerfeld"`. Same name, same default and same
        meaning as `BSplineSolver`'s: that field is smooth on the scale of
        a segment (its singular C₂-image part has been removed by
        construction), so a low order converges — measured at 3.6e-6 Ω
        between orders 3 and 8 on the hardest deck this unit gates (a
        dipole 0.04 λ up, where the remainder is worth 22 Ω), in
        `tests/test_razor_sommerfeld_ground.py`.

        **Since momwire#510 this is a FLOOR, not the order.** The
        smoothness claim above holds down to about 1e-2 λ and fails below
        it: when an observer sits nearly over a source segment's image the
        remainder carries a spike of width ~R_min, and at capture 0033's
        1.09e-4 λ three points put the served impedance 171.86 % from the
        binary. `_remainder_qp` keys the actual order to `len / R_min` and
        clips it below by this kwarg, so a deck with nothing near the plane
        is bit-identical and a grazing one is not.
    cancel: optional :class:`~momwire._cancel.CancelToken`; polled at the
        phase boundaries (after geometry, between the fill chunks, before
        the dense solve).
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    # momwire#396: free space and all three grounds — the PEC image
    # (momwire#398 unit 2), the reflection-coefficient ground (unit 4) and
    # the Sommerfeld ground (unit 5) — for wires standing clear of the
    # plane, plus ground contact over PEC (unit 3), plus wire loading
    # (momwire#427 — distributed loss and insulation through the house
    # kwargs, and lumped loads at knots, which the siblings serve as
    # deck-level port algebra instead), plus PER-WIRE RADII (momwire#147,
    # gated against the binary's own mixed-radius `GW` decks), plus the
    # EXTENDED KERNEL (momwire#398 D1 — the taper study identified the
    # reference as extended-kernel everywhere, so the twin needs it on fat
    # wire; module docstring, "The extended kernel"), plus SERIES NODE GAPS
    # (momwire#603 U4 — the K−1 through-current tents were always built here,
    # only the port that drives one was missing), plus GROUND CONTACT over
    # the Sommerfeld ground (momwire#624 — §5.5's experiment measured the
    # residual bounded and on bspline's own bar, so the refusal that stood
    # here went). No junction_ports / enrichment: the rest of the row is
    # refused, reusing `_OUT_OF_SCOPE`'s prose (built at __init__ from
    # unsupported kwargs).
    #
    # `contact+finite_ground` and `contact+sommerfeld` are GONE from this
    # roster rather than answered None, which is the honest spelling: a key
    # that is absent says "not a refusal here", and a key mapping to prose
    # says "refused, and here is why". What remains is the one combination
    # the whole tree refuses, under the one spelling a caller holding a deck
    # actually has — it knows which ground it asked for, not the abstraction
    # "finite_ground".
    capabilities = Capabilities(
        grounds=frozenset({"pec", "refl-coef", "sommerfeld"}),
        wire_loading=True,
        extended_kernel=True,
        junction_ports=False,
        node_gaps=True,
        knot_feeds=True,
        # The mirror, and the one False in the roster (momwire#673).
        # `_snap_to_knot` moves a gap to the nearest basis-carrying knot, so a
        # site named as a segment CENTRE -- which is what the `nec2` dialect
        # addresses -- lands half a cell from where it was named. That is not
        # a defect: it is this family answering the node question well, and
        # this cell is where it says it answers the centre question by moving
        # the port instead. `bspline-d1` is a tent basis too and is True here,
        # because the axis is where the port ENDS UP, not the accuracy once
        # there.
        centre_feeds=False,
        per_wire_radius=True,
        singular_enrichment=False,
        # Contact at a wire END is served over PEC (unit 3) and over the
        # Sommerfeld ground (momwire#624); what is refused inside that column
        # is the refl-coef row below, which is a combination and not this
        # cell. BURIED is momwire#651's first half: razor refuses it, has
        # always refused it, and now says so — all four sentences the
        # geometry scan can reach, three of them `_medium_spec`'s shared ones
        # and reached through combination keys because they are the ground
        # column and the mid-span crossing rather than this family's own gap.
        buried=_SERVE_BURIED,
        contact=True,
        refusals={
            "contact+refl-coef": _ground_spec.CONTACT_UNDER_REFL_COEF_REFUSAL,
            "centre_feeds": _CENTRE_FEEDS_REFUSAL,
            "junction_ports": _OUT_OF_SCOPE["junction_ports"],
            "singular_enrichment": SINGULAR_ENRICHMENT_NEVER.format(cls="RazorSolver"),
            # The two cells the flip RETIRES, and only these two. Both say
            # "not served YET"; the other four below are real refusals that
            # outlive momwire#814 — a PEC or refl-coef ground has no lower
            # medium whatever razor can fill, a mid-span crossing is still
            # momwire's guess where the model must speak, and contact+buried
            # is momwire#567's measured scope decision for both trunks.
            **(
                {}
                if _SERVE_BURIED
                else {
                    "buried": _BURIED_FILL_REFUSAL,
                    "buried+crossing_junction": _CROSSING_NOT_SERVED_REFUSAL,
                }
            ),
            # Not reachable through `refusal()`'s cell algebra: a bare
            # condition token is SERVED unless it pairs into a combination
            # key, and "does this family have a bundle rule" is an axis
            # question rather than a combination (the promotion rule in
            # `_capabilities._served`). Promoting it means declaring the axis
            # on all eight rows against decks nobody has measured, which is
            # the omission that rule exists to prevent, so it is declared here
            # as prose and left as a question on momwire#846.
            "bundle": _BUNDLE_REFUSAL,
            "buried+pec": _medium_spec.BURIED_PEC_REFUSAL,
            "buried+refl-coef": _medium_spec.BURIED_REFL_REFUSAL,
            "buried+contact": _medium_spec.CONTACT_WITH_BURIED_REFUSAL,
            "buried+crossing": _medium_spec.CROSSING_REFUSAL,
            # `buried+crossing_junction` is declared ABOVE, with the other
            # cell the flip retires. It is its own cell (momwire#850) because
            # the mid-span probe and a DECLARED crossing junction are two
            # refusals under one geometry word, and a row declares one
            # sentence per cell; antennaknobs' catalog gate names this cell
            # for its bonded screen.
        },
    )

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        junctions=None,
        node_gaps=None,
        nsegs=101,
        wire_radius=0.0005,
        extended_kernel=False,
        wire_conductivity=None,
        insulation_radius=None,
        insulation_eps_r=None,
        lumped_loads=None,
        ground_z=None,
        ground_eps=None,
        ground_phi_mode="normal",
        ground_model="refl-coef",
        wavelength=22,
        halfdriver_factor=0.962,
        feed_wire_index=0,
        feed_arclength=None,
        feeds=None,
        n_qp_path=None,
        n_qp_source=12,
        n_qp_sommerfeld=3,
        nec5_quadrature=False,
        cancel=None,
        **unsupported,
    ):
        for name in unsupported:
            if name in _OUT_OF_SCOPE:
                raise NotImplementedError(f"{name}: {_OUT_OF_SCOPE[name]}")
        if unsupported:
            bad = ", ".join(sorted(unsupported))
            raise TypeError(f"RazorSolver got unexpected keyword argument(s): {bad}")

        self._cancel = cancel
        self._checkpoint()

        self.wavelength = float(wavelength)
        self.halfdriver_factor = float(halfdriver_factor)
        self.nsegs = int(nsegs)
        # The ground, in the four attributes `_potential_ground`'s factory
        # reads. All three of its rows are served now — the PEC plane (unit
        # 2), the reflection-coefficient ground (unit 4) and the Sommerfeld
        # one (unit 5) — so these four are the solver's own state, validated
        # here exactly as `BSplineSolver` validates them (same values, same
        # wording, same ValueErrors, including sommerfeld's requirement of a
        # ground_eps to be the permittivity OF), and the factory's branching
        # then lands where the caller asked.
        self.ground_z = None if ground_z is None else float(ground_z)
        if self.ground_z is not None and not np.isfinite(self.ground_z):
            raise ValueError("ground_z must be finite")
        if ground_model not in ("refl-coef", "sommerfeld"):
            raise ValueError(
                "ground_model must be 'refl-coef' or 'sommerfeld', "
                f"got {ground_model!r}"
            )
        if ground_model == "sommerfeld" and ground_eps is None:
            raise ValueError("ground_model='sommerfeld' requires ground_eps")
        if ground_eps is not None and self.ground_z is None:
            raise ValueError("ground_eps requires ground_z to be set")
        if ground_phi_mode not in _ground_refl.PHI_MODES:
            raise ValueError(
                f"ground_phi_mode must be one of {_ground_refl.PHI_MODES}, "
                f"got {ground_phi_mode!r}"
            )
        self.ground_eps = ground_eps
        self.ground_model = ground_model
        self.ground_phi_mode = ground_phi_mode

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = 2 * np.pi / self.wavelength
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        # `None` means DERIVE, resolved below once the mesh is known
        # (momwire#800); an explicit integer is taken verbatim and reproduces
        # the pre-#800 answer bit for bit. `n_qp_path` itself is assigned
        # after the mesh resolution, not here.
        self._n_qp_path_arg = None if n_qp_path is None else int(n_qp_path)
        self.n_qp_source = int(n_qp_source)
        self.n_qp_sommerfeld = int(n_qp_sommerfeld)
        self.nec5_quadrature = bool(nec5_quadrature)
        if self._n_qp_path_arg is not None and self._n_qp_path_arg < 1:
            raise ValueError("n_qp_path, n_qp_source and n_qp_sommerfeld must be >= 1")
        if self.n_qp_source < 1 or self.n_qp_sommerfeld < 1:
            raise ValueError("n_qp_path, n_qp_source and n_qp_sommerfeld must be >= 1")

        if not wires:
            raise ValueError("wires must be non-empty")
        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        # None means infer from the geometry (momwire#590 step 3b). A list
        # overrides it -- most usefully by declaring FEWER junctions than
        # the geometry has, which is how a caller says two coincident ends
        # are deliberately apart. That case was previously inexpressible
        # here, and it is the whole reason the old refusal was wrong to
        # call a spec "either redundant or a disagreement with the mesh":
        # a deliberate disagreement is a legitimate model.
        self._declared_junctions = (
            None if junctions is None else [list(g) for g in junctions]
        )
        # `_wire_media`'s memo (momwire#813). Geometry and the three ground
        # kwargs are frozen after `__init__`, so the labels are computed once
        # — `BSplineSolver._cached_wire_media` is the same store under the
        # same rule.
        self._cached_wire_media = None
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")
        # Validates the geometry against the plane (and is re-read at
        # basis-build time for the grounded ends themselves).
        self._ground_ends()
        # momwire#282 stage 1's D3, the row every solver with a refl-coef
        # ground refuses: contact under `refl-coef` is a MODEL failure at zero
        # clearance and not an implementation one, so it is refused here on
        # exactly the scan `BSplineSolver` and `SinusoidalSolver` use and with
        # exactly their sentence. momwire#624 lifted the SOMMERFELD half of
        # what this block used to refuse — see the module note above
        # `_remainder_qp` for what measuring §4.3's hypothesis found.
        if self.ground_eps is not None and self.ground_model == "refl-coef":
            touching = _ground_spec.contact_ends(self.wires_polylines, self.ground_z)
            if touching:
                where = ", ".join(f"wire {w} {kind}" for w, kind in touching)
                raise NotImplementedError(
                    f"{where} lies in the ground plane: "
                    f"{_ground_spec.CONTACT_UNDER_REFL_COEF_REFUSAL}"
                )

        n_w = len(self.wires_polylines)
        # Per-wire conductor radius (stevenmburns/momwire#147), spelled
        # exactly as the siblings spell it: `wire_radius` keeps the caller's
        # value, `_radius_per_wire` is the normalized (n_wires,) array and
        # `_uniform_radius` is the scalar fast path — the common radius when
        # every wire shares one (however it was spelled), else None. The
        # reduced kernel takes the SOURCE segment's radius; see
        # `_seg_moments_prepare` for the convention, and for what the NEC-5
        # lane can and cannot say about it.
        self.wire_radius = wire_radius
        self._radius_per_wire, self._uniform_radius = _wire_spec.normalize_wire_radius(
            wire_radius, n_w
        )

        # The buried readings, AFTER the radius normalisation: the crossing
        # arm's scope check (`_crossing_junctions`) reads `_radius_per_wire`,
        # because ONE wire radius across the deck is part of the crossing
        # serve's validated scope (momwire#524 phase 2). They still need the
        # ground attributes set above, which bare `__new__` probes of the
        # scan itself never set.
        self._refuse_buried_geometry()
        # The extended kernel (momwire#398 D1). Off is the default and is
        # structurally absent, not skipped: `_ek_labels` is never called, no
        # EK spec is built, and `_seg_moments_prepare` takes `ek=None` — the
        # EK-off answer is bit-for-bit the answer this class gave before the
        # kernel existed, on every lane. The eligibility scan is O(N·G) and
        # is cached per geometry object and per mirror flag, exactly as
        # `BSplineSolver._ek_axis_labels` caches it.
        self.extended_kernel = bool(extended_kernel)
        if getattr(self, "_below_plane", False) and self.extended_kernel:
            # momwire#812: NEC's O(a²) tube expansion was derived in free
            # space, not in a lossy medium; the below-plane fill declines it.
            raise ValueError(
                "razor's below-plane fill does not take the extended kernel: "
                "NEC's O(a²) tube expansion was derived in free space, not in "
                "a lossy medium (momwire#812)"
            )
        self._cached_ek_groups = None

        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None] * n_w
        if len(n_per_edge_per_wire) != n_w:
            raise ValueError(
                f"n_per_edge_per_wire length {len(n_per_edge_per_wire)} "
                f"!= number of wires {n_w}"
            )
        self.n_per_edge_per_wire = []
        for i, (pl, npe) in enumerate(zip(self.wires_polylines, n_per_edge_per_wire)):
            n_edges_w = pl.shape[0] - 1
            if npe is None:
                npe = self.nsegs
            if np.isscalar(npe):
                npe = [int(npe)] * n_edges_w
            npe = [int(v) for v in npe]
            if len(npe) != n_edges_w:
                raise ValueError(
                    f"wire {i}: n_per_edge length {len(npe)} "
                    f"!= number of edges {n_edges_w}"
                )
            if any(v < 1 for v in npe):
                raise ValueError(f"wire {i}: every edge needs >= 1 segment")
            self.n_per_edge_per_wire.append(npe)

        # ---- the outer path order (momwire#800) --------------------------
        # Resolved HERE rather than beside the other quadrature knobs,
        # because deriving it needs the mesh and the mesh is only settled on
        # the line above. An explicit integer never reaches the derivation.
        self.n_qp_path = (
            self._n_qp_path_arg
            if self._n_qp_path_arg is not None
            else derive_n_qp_path(
                self.k, self.wires_polylines, self.n_per_edge_per_wire
            )
        )

        if self._declared_junctions is not None:
            # momwire#522's guardrail, which BSplineSolver and SinusoidalSolver
            # have run since that issue. Razor and harrington only started
            # accepting a spec in momwire#590 step 3b, so they were the two
            # spellings it did not cover -- a wrong wire index welds ends that
            # sit nowhere near each other and produces a well-posed WRONG model
            # that converges cleanly, which is the #518 postmortem exactly.
            #
            # Calling the existing check rather than writing a second one: its
            # tolerance is scale-aware (1e-3 of the shortest terminal segment,
            # floored at 1e-5 m so the deck front's node grid can never fire
            # it), and a flat threshold picked here would be both a seventh
            # "same point" number and a worse-calibrated one.
            _wire_spec.check_junction_coincidence(
                self.wires_polylines,
                self.n_per_edge_per_wire,
                canonical_groups(self._declared_junctions),
            )

        # ---- series node gaps (momwire#603 U4) ---------------------------
        # The SPEC is `_wire_spec.normalize_node_gaps`, shared with the
        # B-spline row: every rule in it is about the spec and the topology,
        # not about a basis.  Validated here rather than at first solve so a
        # malformed gap is a constructor error like every sibling's.
        groups = self._find_junctions()

        # ---- the one-segment wire (momwire#608) --------------------------
        # A one-segment wire used to be refused outright, on the grounds that
        # it "carries no tent" and that two junction tents "would overlap on
        # that one segment".  The second half was simply wrong: a wire
        # junctioned at both ends carries one tent per junction, and two tents
        # sharing a segment is what every INTERIOR segment of every wire
        # already is — they are that segment's two Lagrange bases.  The
        # refusal cost the EZNEC corpus five decks (74 one-segment polylines
        # across them, every one junctioned at one end or both).
        #
        # What is true is the FIRST half, and only for a wire junctioned at
        # NEITHER end: it carries no basis at all, so it holds no current, and
        # a solve including it is bit-identical to one that omits it.  That is
        # not razor's quirk — the licensed engine counts the same wire as an
        # element and gives it no unknown either, and prints the same
        # impedance with and without it.  Reproducing that silently is the one
        # thing this class will not do: the caller declared a scatterer and
        # would get a scatterer-free answer with nothing said.  So the wire is
        # refused, and the message says which wire and what to do about it.
        #
        # This check is also the ONLY one the empty model needs, which is why
        # `_build_geometry` no longer carries a second. It used to raise "no
        # unknowns: every wire needs >= 2 segments" on `n_interior == 0`,
        # counting the interior knots and not the junction tents — so it
        # refused a closed triangle of three one-segment wires, which has no
        # interior knot anywhere and three perfectly good tents. A model with
        # no basis at all is exactly a model whose every wire is one segment
        # and unjoined, so the loop below is that check, per wire and with a
        # wire index in the message.
        joined = {end for g in groups for end in g["ends"]}
        for i, npe in enumerate(self.n_per_edge_per_wire):
            if sum(npe) > 1:
                continue
            if (i, "start") in joined or (i, "end") in joined:
                continue
            raise ValueError(
                f"wire {i}: a one-segment wire junctioned at neither end "
                "carries no basis — it would hold no current and scatter "
                "nothing, and the solve would be identical to one without it "
                "(split it in two, or drop it). A one-segment wire whose end "
                "meets something is fine, whether that is another wire or the "
                "ground plane: the tent such an end carries is the basis an "
                "interior segment carries."
            )

        self.node_gaps = _wire_spec.normalize_node_gaps(
            node_gaps, [list(g["ends"]) for g in groups], len(self.wires_polylines)
        )
        grounded_members = {end for g in groups if g["grounded"] for end in g["ends"]}
        for i, (w_i, end_i, _v) in enumerate(self.node_gaps):
            if (w_i, end_i) in grounded_members:
                raise ValueError(
                    f"node_gaps[{i}]: wire {w_i} {end_i!r} stands in the "
                    "ground plane, and a grounded end's tent is already the "
                    "series path between that wire and the plane (momwire#151"
                    " grounds the node through the image) — drive it with "
                    "feeds= at that end instead"
                )

        # ---- wire loading, the house API (momwire#427) -------------------
        # Two kinds, one equation. DISTRIBUTED series impedance Z'_w(ω)
        # [Ω/m] is spelled exactly as `BSplineSolver` / `SinusoidalSolver` /
        # `SinusoidalGalerkinSolver` spell it — same three kwargs, same
        # normalize/validate contract, same `_wire_loading` physics — so a
        # consumer swapping formulations passes the same dict. LUMPED loads
        # are razor's own kwarg, because the siblings serve a lumped load as
        # deck-level port algebra over a zero-volt `feeds` gap a consumer
        # stamps afterwards (see `momwire.deck._solver`), which this
        # formulation does not take: it is keyed by CLASS in
        # `_NATIVE_LOADING` there, not by any refusal — razor has served
        # `node_gaps` since momwire#603. What razor does instead is exact
        # and cheaper —
        # a load at a knot is a delta in Z_s(l) sitting inside exactly one
        # testing path, i.e. one diagonal entry (`_loading_stencil`).
        #
        # Only NORMALISATION happens here, and since momwire#428 not even
        # that is written here: `configure_loading` is the one normaliser
        # all four formulations run, and `_wire_loading.loading_for(self, ω,
        # geom)` is the one producer the fill reads (per-segment Z_s and the
        # resolved lumped sites). What stays razor's is the term —
        # `_loading_stencil` and `_apply_loading`.
        _wire_loading.configure_loading(
            self, n_w, wire_conductivity, insulation_radius, insulation_eps_r
        )
        self.lumped_loads = _wire_loading.normalize_lumped_loads(lumped_loads, n_w)

        if feeds is None:
            if not (0 <= feed_wire_index < n_w):
                raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
            self.feeds = [(int(feed_wire_index), feed_arclength, 1.0 + 0.0j)]
        else:
            # An EMPTY list is legal once there is a second kind of port
            # (momwire#603 U4): a deck whose only source is a series EMF at
            # an apex -- 0013's `EX 4,5,-1` -- has no gap feed at all, and
            # inventing one would be a spurious second port. What must not
            # happen is a solve with NO port, and that is the check below,
            # which sees both kinds.
            norm = []
            for i, f in enumerate(feeds):
                if len(f) != 3:
                    raise ValueError(
                        f"feeds[{i}]: expected (wire_index, arclength, voltage), "
                        f"got {f!r}"
                    )
                w_i, arc_i, v_i = f
                if not (0 <= w_i < n_w):
                    raise ValueError(
                        f"feeds[{i}]: wire_index {w_i} out of range [0, {n_w})"
                    )
                norm.append(
                    (int(w_i), None if arc_i is None else float(arc_i), complex(v_i))
                )
            self.feeds = norm
        if not self.feeds and not self.node_gaps:
            raise ValueError(
                "no ports: give at least one feeds= gap or one node_gaps= series EMF"
            )
        # The single-feed convenience attributes, and None when there is no
        # gap feed to describe — a node-gap-only model is a legal shape since
        # momwire#603 U4 and these two were never more than a shorthand for
        # `feeds[0]`.
        self.feed_wire_index = self.feeds[0][0] if self.feeds else None
        self.feed_arclength = self.feeds[0][1] if self.feeds else None

        self.z = None
        self._cached_geometry = None

    # ------------------------------------------------------------------
    # geometry

    def _grounded_junction_ends(self):
        """The ``(wire, "start"|"end")`` pairs in a junction whose shared
        point lies IN the ground plane — the crossing-junction exemption
        `_medium_spec.wire_media` keys on (momwire#524 phase 2), and
        `BSplineSolver._grounded_junction_ends` under the same name.

        The one difference from that twin is where the groups come from:
        this formulation has no `junctions=` spec to read (it detects them,
        `_find_junctions`), so the scan runs over the DETECTED groups —
        which is also what makes the answer right for a deck that declared
        nothing, the ordinary case here.

        Groups with fewer than two members are skipped, momwire#698's rule
        and for its reason: a lone grounded end is a legal group of one in
        this formulation (it carries the contact tent), but one wire end
        cannot join two media, so such a group can never be the crossing
        junction the exemption is granted for. Admitting it would hand a
        contact+buried deck a silent escape from the refusal. A group no
        member of which reaches above the plane is skipped for the same
        shape of reason (momwire#700).

        Both conditions live in `_medium_spec.grounded_crossing_exemption`,
        which is also what the B-spline twin calls: the DETECTED-vs-declared
        group source is the only thing the two trunks may differ by here,
        and momwire#700 is what happened when they differed by more.
        """
        if self.ground_z is None:
            return frozenset()
        return _medium_spec.grounded_crossing_exemption(
            self.wires_polylines,
            self.ground_z,
            (g["ends"] for g in self._find_junctions()),
        )

    def _wire_media(self):
        """One `_medium_spec` label per wire, cached per instance.

        `BSplineSolver._wire_media`'s twin under its own name (momwire#813),
        so a consumer that wants "which side of the interface is each wire
        on" asks both trunks the same question — antennaknobs' buried
        catalog gate is the first (antennaknobs#1103), and this module's own
        crossing assembly is the second.

        Raises the crossing / no-lower-medium / contact+buried refusals by
        their shared names, exactly as it did inline before this method
        existed.
        """
        if self._cached_wire_media is None:
            self._cached_wire_media = _medium_spec.wire_media(
                self.wires_polylines,
                self.ground_z,
                lower_medium=(
                    self.ground_eps is not None and self.ground_model == "sommerfeld"
                ),
                pec=self.ground_eps is None,
                crossing_ends=self._grounded_junction_ends(),
            )
        return self._cached_wire_media

    def _crossing_junctions(self):
        """Indices of the DETECTED junctions that cross the interface —
        grounded junctions joining an ABOVE wire to a BELOW wire — after
        checking the deck against the crossing serve's scope.

        `BSplineSolver._crossing_junctions`' twin, scope for scope
        (momwire#524 phase 2): exactly ONE above member per crossing
        junction with N >= 1 below members, one wire radius across the deck,
        and other junctions only wholly BELOW and off the plane. The
        sentences are that method's, because the scope is the adjudication's
        rather than either formulation's.

        Reading it does NOT mean razor can fill such a deck: the constructor
        refuses a mixed above/below deck by name until momwire#813's
        assembly lands. This is the LABEL, which the assembly and
        antennaknobs#1103 both need to exist before then.
        """
        media = self._wire_media()
        if _medium_spec.BELOW not in media:
            return ()
        groups = self._find_junctions()
        crossing = [
            j
            for j, g in enumerate(groups)
            if g["grounded"] and len({media[w] for w, _e in g["ends"]}) == 2
        ]

        # momwire#698's exemption audit, and here for the reason the B-spline
        # twin gives: `_grounded_junction_ends` grants its exemption on
        # GEOMETRY, before the labels exist, and only a junction that
        # actually CROSSES earns the silence it buys from the
        # contact+buried refusal.
        earned = {tuple(m) for j in crossing for m in groups[j]["ends"]}
        stranded = [
            c
            for c in _ground_spec.contact_ends(self.wires_polylines, self.ground_z)
            if c not in earned
        ]
        if stranded:
            raise ValueError(
                _medium_spec.contact_with_buried_refusal(
                    stranded[0][0], media.index(_medium_spec.BELOW)
                )
            )
        if not crossing:
            return ()
        for j in crossing:
            n_above = sum(
                1 for w, _e in groups[j]["ends"] if media[w] == _medium_spec.ABOVE
            )
            if n_above != 1:
                raise NotImplementedError(
                    "crossing junction with more than one above member: the "
                    "crossing serve joins ONE above wire to N below wires "
                    "at the interface (momwire#524 fan widening); the "
                    "above-tent x above-tent interface corner has no "
                    "measured convention"
                )
        for j, g in enumerate(groups):
            if j in crossing:
                continue
            if g["grounded"] or any(
                media[w] != _medium_spec.BELOW for w, _e in g["ends"]
            ):
                raise NotImplementedError(
                    "a deck with a crossing junction and an above-side or "
                    "in-plane OTHER junction is not served: the complete "
                    "crossing spelling completes every value-1 end on its "
                    "axes, and only the below axis's completions (the "
                    "crossing node and the buried hub) are measured "
                    "(momwire#524 phase 2)"
                )
        radii = np.asarray(self._radius_per_wire, dtype=float)
        if float(radii.max()) - float(radii.min()) > 0.0:
            raise NotImplementedError(
                "crossing serve with per-wire radii: the radius rule "
                "rho_eff = sqrt(rho^2 + a^2) regularizes the corner with "
                "ONE wire radius, and a mixed-radius convention is not "
                "pinned (momwire#524 phase 2)"
            )
        return tuple(crossing)

    def _refuse_buried_geometry(self):
        """The construction-time buried readings, through `_medium_spec`.

        Routed through the shared `_medium_spec.wire_media` so BOTH trunks
        refuse buried geometry with the SAME sentences (momwire#651): it
        raises the crossing, the no-lower-medium and the contact+buried
        refusals by their shared names. Whatever it labels BELOW — a wholly
        buried DETACHED wire over a Sommerfeld ground, which `BSplineSolver`
        serves — is a legal deck razor cannot fill yet, and that gap gets
        razor's own sentence here.

        Called from ``__init__`` only: this is validation of the frozen
        geometry-plus-ground state, not part of the `_ground_ends` scan the
        basis build re-asks (and which bare ``__new__`` probes call without
        the ground attributes this reading needs).

        **The crossing exemption is passed now** (momwire#813). It used to
        be withheld, on `wire_media`'s own rule that "a caller with no
        crossing basis (razor) passes nothing and keeps the refusals
        verbatim" — and the cost of that was a refusal that told the reader
        to do what they had already done: a deck SPLIT at the interface into
        a below wire and an above wire with the junction declared, which is
        the shape the crossing refusal asks for, was answered with the
        crossing refusal, because without the exemption the below wire's
        plane-touching anchor reads as a mid-span crossing. It refuses
        still — the mixed above/below sentence below is razor's own and is
        the accurate one — but it now refuses for the reason that is true.
        """
        gz = self.ground_z
        if gz is None:
            return
        media = self._wire_media()
        self._below_plane = False
        self._crossing = False
        if _medium_spec.BELOW in media:
            # The CROSSING deck first, and keyed on the declared junction
            # rather than on the media pair. BELOW + ABOVE is not enough:
            # a DETACHED buried radial under an elevated monopole is also
            # both, and it is not a crossing deck — it is a buried deck razor
            # cannot fill, and it must keep `_BURIED_FILL_REFUSAL`'s sentence
            # naming the trunk that does (momwire#651,
            # `test_651_razors_own_gap_sentence_names_the_serving_trunk`).
            # `_crossing_junctions` is the question, and past the crossing
            # serve's scope it raises the adjudication's own sentences.
            if self._crossing_junctions():
                if not _SERVE_CROSSING:
                    raise ValueError(_CROSSING_NOT_SERVED_REFUSAL)
                self._crossing = True
                return
            w = media.index(_medium_spec.BELOW)
            zmin = float(np.asarray(self.wires_polylines[w])[:, 2].min())
            if not _SERVE_BELOW_PLANE:
                raise ValueError(
                    f"wire {w} lies wholly below the ground plane (min z = "
                    f"{zmin:.6g} < ground_z = {gz:g}), and "
                    f"{_BURIED_FILL_REFUSAL}"
                )
            # momwire#812, unit 1: the lower-medium family serves a deck that
            # is wholly below the plane. A deck that also has an ABOVE wire
            # without a crossing junction is a DETACHED pair, and neither
            # #812's fill nor #813's crossing block covers it.
            if _medium_spec.ABOVE in media:
                raise ValueError(
                    "razor serves a wholly-below deck (momwire#812) but not "
                    "one that also carries an above wire with no junction "
                    "crossing the interface: solve it with BSplineSolver, "
                    "which serves a detached buried wire since momwire#553"
                )
            self._below_plane = True

    def _ground_ends(self):
        """Which wire ENDS lie in the ground plane; everything else refused.

        Returns the frozen set of ``(wire_index, "start" | "end")`` whose
        anchor is in the plane — empty in free space. A member of that set
        is a grounded end, and gets the grounded tent
        (:meth:`_junction_wings`) whose lower wing is its own image.

        The geometries that are NOT ground contact, and stay refused:

        * a wire with points BELOW `ground_z` is refused at construction by
          `_refuse_buried_geometry` (momwire#651) — the shared
          `_medium_spec` sentences, or razor's own buried-fill gap — so no
          such wire reaches this scan;
        * an EDGE lying in the plane (both its anchors at `ground_z`) is
          degenerate over a conducting ground: the edge coincides with its
          own image, so the fold cancels it and it carries no independent
          current. `ValueError`, `BSplineSolver`'s wording again;
        * an INTERIOR anchor in the plane is a wire that touches down
          mid-span. That is real physics — the knot there would carry its
          ordinary tent AND a second unknown for the current leaving into
          the plane — but it is a second basis change on top of this one
          and is not written, so `NotImplementedError`.

        A straight edge takes its minimum z at an anchor, so scanning the
        anchors sees every contact there is.

        The touch tolerance is `_ground_spec.ground_touch_tol`'s: 1e-6 of
        the wire's polyline length, loose enough for deck-import float
        noise at z=0 and far tighter than any deliberate stand-off.
        """
        gz = self.ground_z
        if gz is None:
            return frozenset()
        touching = set()
        for i, pl in enumerate(self.wires_polylines):
            tol = _ground_spec.ground_touch_tol(pl)
            at = np.abs(pl[:, 2] - gz) <= tol
            if np.any(at[:-1] & at[1:]):
                raise ValueError(
                    f"wire {i} has an edge lying in the ground plane "
                    "(both endpoints at ground_z) — degenerate over a "
                    "conducting ground"
                )
            if np.any(at[1:-1]):
                raise NotImplementedError(
                    f"wire {i} touches the ground plane at an interior "
                    "anchor: RazorSolver serves ground contact at a wire END "
                    "only (momwire#398 unit 3). A mid-span touchdown needs a "
                    "second unknown at a knot that already carries a tent — "
                    "split the wire there, so the contact is two wire ends"
                )
            if at[0]:
                touching.add((i, "start"))
            if at[-1]:
                touching.add((i, "end"))
        return frozenset(touching)

    def _find_junctions(self):
        """Group coincident wire ends into junctions, grounded ones marked.

        Returns a list of ``{"ends": [...], "grounded": bool}``; `ends` is a
        list of ``(wire_index, "start" | "end")`` in the order the ends are
        listed — first wire first, and `start` before `end` within a wire.
        The first entry is the reference side A of every junction tent
        there, so this order is part of the basis definition (not that the
        answer depends on it: picking a different reference re-spells the
        same current space).

        A wire whose own two ends coincide is a closed loop and forms a
        group of two on its own. Grouping is by first match within
        `_JUNCTION_TOL`, which is **this module's own** absolute 1e-9 and
        agrees with nothing else in the tree. This paragraph used to claim
        it was "the same 'same point' tolerance the caller-facing geometry
        helpers use"; that was false, and momwire#429 correction 2 caught
        it. The layer that actually produces `junctions=` is the deck
        front end, which fuses span endpoints onto a
        `deck/_polylines._NODE_EPS` = 1e-6 m grid — a THOUSAND times
        looser, and a different algorithm (grid quantization, not
        first-match) under a different norm (per-coordinate rounding, not
        the Euclidean distance used here). Two ends the deck calls one
        node can therefore reach this grouping as two.

        What keeps that from biting today is the deck's own invariant, not
        an agreement between the numbers: `_polylines` documents that "by
        the time a model exists its coincident ends are already exactly
        equal", so the coarse grid is absorbing transform ulps rather than
        deciding connectivity. A caller assembling `junctions=` by hand,
        or a future front end that relaxes that invariant, would see the
        gap.

        Six disagreeing tolerances live across three algorithms and two
        norms in this tree, and unifying them is a deliberate future
        decision, NOT part of momwire#429's pure moves — that unit shared
        the TOUCH tolerance (`_ground_spec.ground_touch_tol`, five
        byte-identical spellings) and changed no value anywhere.

        A group survives if it carries a through-path (K >= 2 ends) or if it
        is GROUNDED, because the plane is then one more branch at the point:
        a lone grounded end is a group of one, and its one tent is the
        current leaving into the ground. `grounded` is `any` over the
        group's ends rather than `all` — the two tolerances differ (the
        touch tolerance scales with each wire's own length, the grouping
        tolerance is absolute), so coincident ends could otherwise disagree
        about a plane they share.
        """
        grounded_ends = self._ground_ends()
        labels, points = [], []
        for i, pl in enumerate(self.wires_polylines):
            labels.append((i, "start"))
            points.append(pl[0])
            labels.append((i, "end"))
            points.append(pl[-1])

        # The rule itself is `_junction_rule.grouped` (momwire#590 step 1) —
        # first match against group REPRESENTATIVES, non-transitively. It used
        # to be spelled here and a second time by hand in `harrington.py`.
        # Label and point order are unchanged, so the grouping and the
        # reference side A of every junction tent are what they always were.
        if self._declared_junctions is None:
            groups = grouped(labels, points, _JUNCTION_TOL)
        else:
            groups = canonical_groups(self._declared_junctions)
        out = []
        for g in groups:
            grounded = any(e in grounded_ends for e in g)
            if len(g) >= 2 or grounded:
                out.append({"ends": g, "grounded": grounded})
        return out

    def _junction_wings(self, seg_offsets, group):
        """Wing descriptors for one junction group's tents.

        Yields ``(seg_a, rise_a, sigma_a, seg_b, rise_b, sigma_b)`` per
        tent. `rise` says the junction sits at the segment's far (arc-h)
        end, so the tent rises with the segment's own arc coordinate;
        `sigma` turns +1 A of through-current into a signed multiple of
        that segment's arc direction. Side A carries the current INTO the
        junction (so +1 along arc if the wire joins by its end, −1 if by
        its start) and side B carries it back OUT (the mirror image). A
        free-air group of K ends yields K−1 such tents.

        A GROUNDED group yields K tents instead, one per end (momwire#398
        unit 3): the plane is one more branch, so K real ends meeting there
        carry K independent currents and no through-path is distinguished.
        Each grounded tent is the junction tent between a wire end and its
        OWN IMAGE — the through-current tent of the monopole-plus-image
        dipole, whose upper wing is the real contact segment and whose
        lower wing is that segment mirrored. Only the real wing is spelled
        here: the fold `Z = Z_free − Z_image` already evaluates every basis
        against the mirrored sources, so the image wing arrives with the
        right shape (the mirror preserves each segment's local arc
        coordinate), the right direction (−M·t̂, parallel for a vertical
        contact) and the opposite charge, for free. Side A — the image side
        — is therefore spelled as a wing carrying `sigma = 0`, which zeroes
        its tangent (T1), its charge doublet (T2) and its half of the
        testing path at once; its `(segment, rise)` copy side B's so that
        `_knot_points` still reads the contact point off wing A.
        """
        if group["grounded"]:
            for w, kind in group["ends"]:
                seg = seg_offsets[w + 1] - 1 if kind == "end" else seg_offsets[w]
                rise = kind == "end"
                yield (seg, rise, 0.0, seg, rise, -1.0 if rise else 1.0)
            return
        (w_a, kind_a) = group["ends"][0]
        for w_b, kind_b in group["ends"][1:]:
            seg_a = seg_offsets[w_a + 1] - 1 if kind_a == "end" else seg_offsets[w_a]
            seg_b = seg_offsets[w_b + 1] - 1 if kind_b == "end" else seg_offsets[w_b]
            rise_a = kind_a == "end"
            rise_b = kind_b == "end"
            yield (
                seg_a,
                rise_a,
                1.0 if rise_a else -1.0,
                seg_b,
                rise_b,
                -1.0 if rise_b else 1.0,
            )

    def _build_geometry(self):
        """Discretize every wire into segments and concatenate.

        Segments are stored by start point, unit tangent and length, which
        is the form both the static moments and the testing paths want.

        Every basis — interior tent or junction tent — is described by two
        WINGS, one per side of its knot, in flow order (side A first):

            wing_seg[n, j]    the segment the wing lives on
            wing_rise[n, j]   the knot is at that segment's arc-h end, so
                              the tent shape there is τ/h (else 1 − τ/h)
            wing_sigma[n, j]  ±1: the wing's current direction relative to
                              its segment's own arc direction

        An interior tent at knot n is ``([left, right], [True, False],
        [+1, +1])`` — the unit-1 convention, unchanged. Junction tents are
        appended after all interior bases, junction by junction, and get
        their wings from :meth:`_junction_wings`. A GROUNDED end's tent is
        one of those, with `wing_sigma[n, 0] == 0`: its side-A wing is the
        image's, which the fold supplies, so nothing real lives on it.
        `grounded_bases` lists those tents' indices, which is what the fill
        reads to give their ROW the plane as its potential reference.
        """
        if self._cached_geometry is not None:
            return self._cached_geometry

        per_wire = []
        seg_offsets = [0]
        basis_offsets = [0]
        p0_list, tan_list, h_list = [], [], []
        for w_idx, (pl, npe) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            w_p0, w_tan, w_h = [], [], []
            for e_idx in range(pl.shape[0] - 1):
                edge = pl[e_idx + 1] - pl[e_idx]
                edge_len = float(np.linalg.norm(edge))
                if edge_len < 1e-15:
                    raise ValueError(f"wire {w_idx} edge {e_idx} has zero length")
                tan = edge / edge_len
                n_e = npe[e_idx]
                h_e = edge_len / n_e
                w_p0.append(pl[e_idx][None, :] + np.arange(n_e)[:, None] * h_e * tan)
                w_tan.append(np.tile(tan, (n_e, 1)))
                w_h.append(np.full(n_e, h_e))
            seg_p0 = np.vstack(w_p0)
            seg_t = np.vstack(w_tan)
            seg_h = np.concatenate(w_h)
            n_total = seg_p0.shape[0]
            per_wire.append(
                {
                    "arc_at_knot": np.concatenate([[0.0], np.cumsum(seg_h)]),
                    "n_total": n_total,
                }
            )
            seg_offsets.append(seg_offsets[-1] + n_total)
            basis_offsets.append(basis_offsets[-1] + n_total - 1)
            p0_list.append(seg_p0)
            tan_list.append(seg_t)
            h_list.append(seg_h)

        left = np.concatenate(
            [
                seg_offsets[w] + np.arange(per_wire[w]["n_total"] - 1, dtype=np.int64)
                for w in range(len(per_wire))
            ]
        )
        n_interior = int(basis_offsets[-1])

        groups = self._find_junctions()
        # A DECLARED crossing junction is not a grounded one (momwire#813).
        #
        # `_find_junctions` marks a group grounded when its shared point lies
        # in the plane, and `_junction_wings` then emits K contact tents
        # rather than K-1 through tents — the right basis where the plane is
        # a CONDUCTOR's boundary, because current genuinely leaves there and
        # the image completes it. At a crossing node the plane is the
        # boundary between two MEDIA and not a sink: a buried wire sheds
        # current along its whole buried length through `k_m`, which
        # momwire#812's fill already carries, and the interface point itself
        # has zero measure. A contact tent there injects the base current
        # into the soil AT A POINT — NEC-5's interface-node treatment, whose
        # signature momwire#838 measured: the radial-count law goes flat and
        # the engine reads the same connected or detached.
        #
        # So a crossing group takes the free-space topology: K ends, K-1
        # through-current tents, no contact tent and no ghost wing. Demoting
        # the flag HERE rather than in `_junction_wings` is what makes that
        # one edit: `geom["junctions"]` is what `_feed_knots` and
        # `grounded_bases` (the T2 plane-reference drop) both read, and all
        # three have to agree that this node is not a potential reference.
        # Demoting it in `_find_junctions` instead would recurse, since
        # `_crossing_junctions` reaches `_wire_media` reaches
        # `_grounded_junction_ends` reaches `_find_junctions`.
        #
        # Every grounded end that is NOT a crossing member keeps its contact
        # tent exactly as before, which is what `test_a_grounded_end_that_is_
        # not_a_crossing_member_is_untouched` holds.
        #
        # ONE CONSEQUENCE, NAMED SO IT IS NOT "FIXED" BACK. Dropping the flag
        # also takes this node out of `grounded_bases`, so
        # `_assemble_Z_source_block` no longer applies the plane-reference T2
        # drop there — and that is intended, not fallout. The drop is the
        # CONDUCTOR identity: the folded potential is identically zero on a
        # PEC plane, so a grounded row's path may start there and lose that
        # endpoint exactly. At a medium interface Φ is neither zero nor even
        # single-valued across the two families — each family's (A, Φ) is its
        # own gauge, and the trunk's transmitted V at z = 0 is NOT Φ(node⁻)
        # (momwire#813 derivation (a)) — so no potential reference may be
        # taken at a crossing node at all. That row's T2 comes from
        # momwire#813 unit 2's evaluation AT the node instead, one family per
        # chopped half.
        crossing = (
            set(self._crossing_junctions()) if self.ground_z is not None else set()
        )
        if crossing:
            groups = [
                dict(g, grounded=False) if j in crossing else g
                for j, g in enumerate(groups)
            ]
        j_seg, j_rise, j_sigma = [], [], []
        junctions, grounded_bases = [], []
        for group in groups:
            bases = []
            for sa, ra, ga, sb, rb, gb in self._junction_wings(seg_offsets, group):
                bases.append(n_interior + len(j_seg))
                j_seg.append((sa, sb))
                j_rise.append((ra, rb))
                j_sigma.append((ga, gb))
            # A grounded group's bases run parallel to its ends (one tent
            # per end); a free-air group's run parallel to ends[1:].
            junctions.append(
                {"ends": group["ends"], "bases": bases, "grounded": group["grounded"]}
            )
            if group["grounded"]:
                grounded_bases.extend(bases)

        n_junction = len(j_seg)
        n_basis = n_interior + n_junction
        wing_seg = np.empty((n_basis, 2), dtype=np.int64)
        wing_rise = np.empty((n_basis, 2), dtype=bool)
        wing_sigma = np.empty((n_basis, 2), dtype=np.float64)
        wing_seg[:n_interior, 0] = left
        wing_seg[:n_interior, 1] = left + 1
        wing_rise[:n_interior, 0] = True
        wing_rise[:n_interior, 1] = False
        wing_sigma[:n_interior] = 1.0
        if n_junction:
            wing_seg[n_interior:] = np.asarray(j_seg, dtype=np.int64)
            wing_rise[n_interior:] = np.asarray(j_rise, dtype=bool)
            wing_sigma[n_interior:] = np.asarray(j_sigma, dtype=np.float64)

        geom = {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "basis_offsets": basis_offsets,
            "n_segs_total": seg_offsets[-1],
            "n_basis_interior": n_interior,
            "n_basis_total": n_basis,
            "seg_p0": np.vstack(p0_list),
            "seg_t": np.vstack(tan_list),
            "seg_h": np.concatenate(h_list),
            "wing_seg": wing_seg,
            "wing_rise": wing_rise,
            "wing_sigma": wing_sigma,
            "junctions": junctions,
            "grounded_bases": np.asarray(grounded_bases, dtype=np.int64),
        }
        self._cached_geometry = geom
        return geom

    def _feed_knots(self, geom, w):
        """Every knot of wire `w` that carries a basis, as (arc, basis, K).

        The interior knots, plus either wire end that meets other ends at a
        junction — a junction knot's basis is that junction's through-
        current unknown, and driving it is the split-wire feed — plus
        either wire end that TOUCHES the plane, whose basis is that end's
        grounded tent: the source then sits in the gap between the plane
        and that wire end, which is how a monopole is driven.

        `K` is the number of ends at the knot (1 for an interior knot),
        which is what the K >= 3 refusal reads. A grounded end reports 1
        whatever else meets it there, and that is not a fudge: the gap a
        source at a grounded end occupies is between the plane and THIS
        wire's end, so there is no branch pair left to name — each end at a
        grounded point has a tent of its own.
        """
        arc_at_knot = geom["per_wire"][w]["arc_at_knot"]
        base = geom["basis_offsets"][w]
        knots = [
            (float(arc_at_knot[j]), base + j - 1, 1)
            for j in range(1, len(arc_at_knot) - 1)
        ]
        for jn in geom["junctions"]:
            for e_i, (w_e, kind) in enumerate(jn["ends"]):
                if w_e != w:
                    continue
                arc = 0.0 if kind == "start" else float(arc_at_knot[-1])
                if jn["grounded"]:
                    knots.append((arc, jn["bases"][e_i], 1))
                else:
                    knots.append((arc, jn["bases"][0], len(jn["ends"])))
        # In ARC ORDER (momwire#672). The interior knots are built in it and
        # the junction and grounded ends are appended, so a wire whose START
        # is a junction reported that knot last. `_snap_to_knot` runs argmin
        # over this list and uses the INDEX, which is invisible wherever one
        # knot is plainly nearest and decisive at a tie — and left the index
        # a property of how the list was built rather than of the geometry.
        # The same two physical knots then read as (9, 10) when a bent wire
        # is one polyline and (9, 0) when it is two, so any rule phrased on
        # the index named different sites in two spellings of one antenna.
        knots.sort(key=lambda knot: knot[0])
        return knots

    def _feed_basis_indices(self, geom):
        """Global basis index of each feed's knot.

        Each feed snaps to the knot of its wire — interior, junction or
        grounded end — whose arc length from that wire's first anchor is
        closest to the requested value (None → the wire's midpoint).
        """
        idx = []
        for i, (w, arc, _v) in enumerate(self.feeds):
            basis, k_ends = self._snap_to_knot(geom, w, arc)
            if k_ends >= 3:
                # The ambiguity is real and it is THIS spelling's: `feeds`
                # names an arclength on a wire, and an arclength cannot say
                # which of the K-1 through-current tents at the node the EMF
                # sits in. It is not a statement that the source has no
                # meaning at a K >= 3 node — NEC-5 serves one (0013's
                # `EX 4,5,-1` at a five-wire apex) and the dialect that
                # writes it names the branch, because "NEC-5 attaches an
                # inserted source not AT a junction but on the wire next to
                # it, named by the tag" (`momwire.deck._nec5.Nec5Node`). A
                # kwarg that carries the branch is what momwire#603 U4 is
                # about; the tents themselves are already built.
                raise NotImplementedError(
                    f"feeds[{i}]: the source snaps to a junction where "
                    f"{k_ends} wire ends meet, and an ARCLENGTH cannot name "
                    "which of that node's branches the gap sits in. Feed an "
                    "interior knot, or model the source on a short bridge "
                    "wire off the junction."
                )
            idx.append(basis)
        return idx

    def _port_columns(self, geom):
        """``(n_basis_total, n_ports)`` — every port's drive AND readout vector.

        Ports run ``[gap feeds..., node gaps...]``, the order
        :class:`~momwire._port_solution.PortSolution` documents (this row
        declares no junction ports, so the middle block is empty).

        A GAP FEED is a one-hot: its delta gap sits inside exactly one
        testing path, so the whole voltage lands in that one row and the
        current read back is that one coefficient.

        A NODE GAP is not, and that is the whole of what momwire#603 U4 had
        to work out.  This formulation's junction unknowns are K−1 PAIR
        tents — tent ``t`` carries +1 A in along the group's first end and
        out along end ``t+1`` (:meth:`_junction_wings`) — so the current
        from the node into member ``ends[j]`` is::

            j >= 1   +c[j-1]                 (one-hot)
            j == 0   -(c[0] + ... + c[K-2])  (dense, over every tent)

        which is KCL: the first end carries whatever the others do not.  No
        sigma appears — the wings already carry it — and both rows were
        measured against :meth:`currents_at_knots` on a three-wire star with
        one wire joining by its ``end`` and two by their ``start``.

        The B-spline row's column is a sigma-signed one-hot on the named
        member's OWN directional basis, which is why this cannot be shared
        and the spec around it can (``_wire_spec.normalize_node_gaps``).
        """
        idx = self._feed_basis_indices(geom)
        n_ports = len(idx) + len(self.node_gaps)
        cols = np.zeros((geom["n_basis_total"], n_ports), dtype=np.float64)
        for j, m_j in enumerate(idx):
            cols[m_j, j] = 1.0
        if not self.node_gaps:
            return cols
        where = {
            member: (jn, pos)
            for jn in geom["junctions"]
            for pos, member in enumerate(jn["ends"])
        }
        for p, (w_i, end_i, _v) in enumerate(self.node_gaps):
            jn, pos = where[(int(w_i), end_i)]
            column = len(idx) + p
            if pos == 0:
                cols[jn["bases"], column] = -1.0
            else:
                cols[jn["bases"][pos - 1], column] = 1.0
        return cols

    def _port_voltages(self):
        """The configured drive of every port, in :meth:`_port_columns` order."""
        return np.array(
            [v for _, _, v in self.feeds] + [v for _, _, v in self.node_gaps],
            dtype=np.complex128,
        )

    def _snap_to_knot(self, geom, w, arc):
        """``(basis index, ends at that knot)`` for one site on wire `w`.

        The site is an arc length from that wire's first anchor, or None for
        the wire's midpoint; it snaps to the nearest knot of wire `w` that
        carries a basis — interior, junction or grounded end
        (:meth:`_feed_knots`). Shared by the feeds and by the lumped loads
        (momwire#427), because a load and a source name a site the same way
        and must land on the same knot when they name the same one: gate 2
        of #427 is precisely `Z_driven == Z_unloaded + Z_L`, and it holds
        only if `feeds` and `lumped_loads` resolve identically. Each caller
        writes its own K >= 3 refusal, since what is ambiguous there differs
        (which branch pair a source drives, which branch pair a load is in).
        """
        arc_at_knot = geom["per_wire"][w]["arc_at_knot"]
        target = arc if arc is not None else arc_at_knot[-1] / 2.0
        knots = self._feed_knots(geom, w)
        arcs = np.array([a for a, _b, _k in knots])
        pick, _margin = _feed_snap.snap(
            arcs,
            target,
            total_arc=float(arc_at_knot[-1]),
            family=type(self).__name__,
            what="site",
            wire=w,
        )
        _a, basis, k_ends = knots[pick]
        return int(basis), int(k_ends)

    # ------------------------------------------------------------------
    # kernel moments

    def _seg_radius(self, geom):
        """``(n_segs,)`` per-segment radius — each segment inherits its
        wire's (stevenmburns/momwire#147), the same spelling and the same
        `seg_offsets` reading `BSplineSolver._seg_radius` uses."""
        seg_off = np.asarray(geom["seg_offsets"], dtype=np.int64)
        return np.repeat(self._radius_per_wire, np.diff(seg_off))

    def _kernel_radius(self, geom):
        """The `a` the reduced kernel is regularised with, for every source
        segment: the scalar when the model is uniform (the historical fast
        path, bit-identical), else the ``(n_segs,)`` per-source-segment
        column. See :meth:`_seg_moments_prepare` for why it is the SOURCE's.

        Read off the REAL geometry and handed to the mirrored source set
        unchanged: an image segment is its own segment's reflection, so it
        carries that segment's radius, exactly as its length and its local
        arc coordinate are carried across (`_image_sources`).
        """
        if self._uniform_radius is not None:
            return self._uniform_radius
        return self._seg_radius(geom)

    def _ek_labels(self, geom, mirror=False):
        """Per-segment EK eligibility labels, as ``(source, mirrored source)``.

        The shared rule and nothing else: `_ek_axis_groups` labels two
        segments alike iff they are COAXIAL and of EQUAL RADIUS, on NEC's own
        thresholds. This solver does not re-derive it and does not soften it
        — in particular it does NOT reuse `SinusoidalSolver`'s per-END
        IND1/IND2 gating, which is the sinusoidal family's spelling of the
        same NEC rule against per-segment neighbour tables. Razor is
        mixed-potential like the B-spline trunk, its rows are path integrals
        over arbitrary (observer point, source segment) pairs rather than
        per-end brackets, so the PAIR rule is the precedent that fits
        (`_bspline_kernels`, "NEC's per-END gating ... is deliberately NOT
        reused").

        **The mirror policy.** Over a ground the eligibility scan runs ONCE
        over the real segments stacked on the mirrored ones, and the two
        halves of the answer are handed back separately. That is the whole
        of the ground's involvement: the ground object supplies mirrored
        GEOMETRY (`_image_sources`) and the shared rule then reads it exactly
        as it reads any other geometry — the ground has no opinion about the
        kernel, and no ground branch appears in the EK code at all. Scanning
        the two sets JOINTLY rather than separately is what makes the answer
        right, and `BSplineSolver._ek_axis_labels` records the same trap: two
        independent scans would label a wire and its image 0 and 0 and
        declare every real/image pair coaxial, when a HORIZONTAL wire and its
        image are merely parallel, offset by twice the height. Jointly, a
        vertical wire mirrors onto its own axis and IS one group — NEC's
        IND = 0 perpendicular-ground branch, and the case that matters, since
        it is the grounded-contact tent's own lower wing.

        Cached per geometry OBJECT (identity) and per mirror flag, so a
        grounded swept solve pays the O(N·G) scan once rather than per k.
        """
        cached = self._cached_ek_groups
        if cached is None or cached[0] is not geom:
            cached = (geom, {})
            self._cached_ek_groups = cached
        hit = cached[1].get(mirror)
        if hit is not None:
            return hit

        seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
        seg_l = seg_p0
        seg_r = seg_p0 + seg_h[:, None] * seg_t
        seg_a = self._seg_radius(geom)
        img = self._image_sources(geom) if mirror else None
        if img is not None:
            img_l = img["seg_p0"]
            img_r = img["seg_p0"] + seg_h[:, None] * img["seg_t"]
            n = seg_l.shape[0]
            joint = _ek_axis_groups(
                np.vstack([seg_l, img_l]),
                np.vstack([seg_r, img_r]),
                np.vstack([seg_t, img["seg_t"]]),
                np.concatenate([seg_a, seg_a]),
            )
            hit = (joint[:n], joint[n:])
        else:
            labels = _ek_axis_groups(seg_l, seg_r, seg_t, seg_a)
            hit = (labels, labels)
        cached[1][mirror] = hit
        return hit

    def _ek_obs_labels_path(self, geom, labels):
        """The EK label of every TESTING-PATH quadrature point, ``(n_basis·n_path,)``.

        A path point is an observer, and the pair rule wants the label of the
        segment that observer lies ON. Path P_m is two straight halves, the
        first running along wing A's segment and the second along wing B's
        (`_testing_paths`), so each half's points inherit that wing segment's
        label — which is also the honest reading of what the razor row is:
        the field of the source segment tested along the OBSERVER wire's
        axis, and "same axis, same radius" is a statement about those two
        wires.

        Shaped and flattened exactly as `_assemble_Z_prepare` flattens
        `pts[lo:hi]`, so a row chunk's labels are that chunk's slice of this
        array. The two quadrature lanes need no branch: `n_path` is
        2·`n_qp_path` on the Gauss-Legendre path and 2 under
        `nec5_quadrature`, and either way the first half of the columns is
        wing A and the second is wing B.
        """
        s_a, s_b = geom["wing_seg"][:, 0], geom["wing_seg"][:, 1]
        half = self._path_nodes_per_wing()
        return np.concatenate(
            [
                np.repeat(labels[s_a][:, None], half, axis=1),
                np.repeat(labels[s_b][:, None], half, axis=1),
            ],
            axis=1,
        ).reshape(-1)

    def _path_nodes_per_wing(self):
        """Quadrature nodes per testing-path HALF — the one place the two
        lanes differ, and the only thing the EK observer labelling needs to
        know about them."""
        return 1 if self.nec5_quadrature else self.n_qp_path

    def _seg_moments_prepare(self, obs, geom, a, *, ek=None):
        """K-independent ingredients of every segment's kernel moments.

        The closed-form static moments ``(m0s, m1s, see
        :func:`_static_axis_moments`)`` and the source-to-observer distance
        ``R`` itself depend only on geometry — R = sqrt(u² + ρ²) with u, ρ²
        from the axis frame (:func:`_axis_frame`) and the source quadrature
        node's local arc τ, none of which a wavenumber sweep changes. Only
        exp(−jkR) is k-dependent, so R is exactly the boundary: caching it
        (rather than recomputing it from the axis frame every k) moves its
        sqrt out of the per-k path along with the asinh/sqrt of the static
        moments. Chunked exactly as :meth:`_seg_moments` used to chunk
        internally, so a k-sweep can replay the same chunk boundaries
        without recomputing them.

        `a` is the reduced kernel's regularising radius
        (:meth:`_kernel_radius`): the scalar for a uniform model, or a
        ``(n_segs,)`` column indexed by SOURCE segment. **The convention is
        the source's, and — unlike the sinusoidal family's — it is a choice
        rather than an oracle finding, because this formulation's oracle
        cannot see the difference.** Measured against the licensed binary on
        a fat/thin collinear STEP (the geometry where the two candidates
        diverge at all, since a² only matters where the perpendicular
        distance vanishes): source and observer conventions differ by
        3.0e-6 … 1.1e-5 Ω on a 10:1 step and 1.4e-3 … 2.1e-3 Ω on a 100:1
        one, against a twin-lane bar of 0.20 Ω — the difference lives in
        near-diagonal entries worth ~1400 Ω, where 0.1 Ω of it is 2e-5
        relative and the solve absorbs it. The source reading is taken
        because it is the reduced kernel's own derivation (the source
        current averaged over ITS surface ring onto its axis, observed on
        the axis) and because it is chunk-invariant: the chunks split the
        OBSERVER axis, so every chunk sees the whole column and a mesh
        refinement cannot move a chunk boundary into the answer. The
        siblings' observer convention (NEC-2's `EFLD`, PyNEC-oracled for
        `SinusoidalSolver`, where it was worth 11 Ω) is recorded here as
        immaterial for THIS formulation rather than as wrong. See
        `docs/design/solver-architecture.md` §6.9.

        `ek` is the `_EK` spec of the extended kernel (momwire#398 D1) or None,
        which is the default and the reduced kernel. Its `group_i` labels
        THIS call's observers and its `group_j` labels `geom`'s source
        segments; `_ek_pair_mask` turns them into the (observer, segment)
        eligibility mask, and every eligible entry takes
        :func:`_static_axis_moments_ek` in place of
        :func:`_static_axis_moments`. **The EK statics are as k-independent
        as the reduced ones** — the extended kernel's k → 0 limit is a
        function of R and a alone — so they belong on this side of the
        prepare/replay boundary exactly as the reduced ones do, and the mask
        rides along in the chunk so the replay half can weight the smooth
        remainder by the same pairs without re-deriving eligibility per k.

        Returns a list of ``(lo, hi, R, m0s, m1s, ekc)`` chunks. `ekc` is None
        off the extended kernel, and ``(mask, a_ek)`` on it — with `mask`
        itself None when EVERY pair of the chunk is eligible, the common case
        and the one a straight uniform wire is.

        **With the C++ fill built** (momwire#742) this returns a
        :class:`_FusedMoments` instead — the same opaque token to every caller,
        since the only thing anyone does with it is hand it back to
        :meth:`_seg_moments_from_prepared`. It carries the geometry rather
        than the tables built from it, because the tables ARE the problem:
        `R` alone is `n_obs · n_seg · n_qp_source` doubles and the chunk list
        retains every chunk's at once, which is where razor's 52×-the-matrix
        peak comes from. The kernel forms R one scalar at a time instead, so
        the k-independent half has nothing left to cache and the split
        degenerates — deliberately — to a geometry reference. What that costs
        is the statics recomputed at every swept k (the same ~16 % this
        method's caller prices for dropping the image cache); what it buys is
        the whole O(N²·n_qp) transient, on both source sets.
        """
        if _use_razor_fill_accel():
            return _FusedMoments(obs, geom, a, ek, self.n_qp_source)
        return self._seg_moments_prepare_numpy(obs, geom, a, ek=ek)

    def _seg_moments_prepare_numpy(self, obs, geom, a, *, ek=None):
        """The numpy lane's half of :meth:`_seg_moments_prepare`.

        Split out (momwire#796) because it has a SECOND caller now: a
        `_FusedMoments` handed a complex k by a build whose kernel is real-k
        only falls back here, and that fallback cannot go through
        `_seg_moments_prepare` itself — the accelerator branch at its head
        would hand back another `_FusedMoments` and the fallback would be a
        loop.
        """
        seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
        n_seg = seg_h.size
        n_obs = obs.shape[0]
        xg, _wg = leggauss(self.n_qp_source)
        tau = 0.5 * seg_h[:, None] * (1.0 + xg[None, :])
        step = max(1, _CHUNK_ELEMS // max(1, n_seg * self.n_qp_source))
        chunks = []
        for lo in range(0, n_obs, step):
            self._checkpoint()
            hi = min(lo + step, n_obs)
            u_r, rho2 = _axis_frame(obs[lo:hi], seg_p0, seg_t, a)
            m0s, m1s = _static_axis_moments(u_r, rho2, seg_h)
            ekc = None
            if ek is not None:
                # The shared pair rule, restricted to this chunk's observer
                # rows. `a_ek` is the pair's common radius: eligibility
                # REQUIRES equal radii, so the source column `a` already is
                # it, which is what `_EK(a=None, ...)` means here exactly as
                # it means in `BSplineSolver._ek_spec`.
                mask = _ek_pair_mask(
                    _EK(a=ek.a, group_i=ek.group_i[lo:hi], group_j=ek.group_j),
                    hi - lo,
                    n_seg,
                )
                a_ek = _ek_radius(ek, a)
                m0e, m1e = _static_axis_moments_ek(u_r, rho2, seg_h, a_ek)
                if mask.all():
                    # Eligibility is GEOMETRY, so "every pair of this chunk
                    # extends" — a straight uniform wire, and the fat twin
                    # lane's own case — is settled here rather than rescanned
                    # at every solved wavenumber. `None` in the chunk means
                    # exactly that, and the replay half then skips the gather
                    # instead of copying the whole array to reach all of it.
                    m0s, m1s, mask = m0e, m1e, None
                else:
                    m0s = np.where(mask, m0e, m0s)
                    m1s = np.where(mask, m1e, m1s)
                ekc = (mask, a_ek)
            u = tau[None, :, :] - u_r[:, :, None]
            R = np.sqrt(u * u + rho2[:, :, None])
            chunks.append((lo, hi, R, m0s, m1s, ekc))
        return _PreparedChunks(chunks, seg_h)

    def _seg_moments_from_prepared(self, chunks, k, n_obs, *, need_m1=True):
        """Finish :meth:`_seg_moments_prepare`'s chunks at one wavenumber.

        Only the smooth remainder (exp(−jkR)−1)/(4πR) is computed here —
        everything k-independent (R and the static moments) already sits in
        `chunks`. Returns the same ``(M0, M1)`` shape ``(n_obs, n_seg)``
        that :meth:`_seg_moments` did.

        Under the extended kernel (momwire#398 D1) the eligible pairs' remainder
        is the same object with NEC Eq 89's coaxial factor in it,

            [ (e^{−jkR} − 1)·fac + extra ] / (4πR)

        with `fac = _ek_factor(R, a, k)` and `extra = fac − fac_static =
        _ek_reg_extra(R, a, k)` — the k → 0 part of the factor having already
        been taken analytically by :func:`_static_axis_moments_ek` on the
        other side of the prepare boundary. It is spelled exactly as
        `_bspline_kernels` spells it, term for term, because the two
        formulations must share one kernel for a cross-formulation
        difference-of-columns to mean anything. The eligible entries are
        gathered by the chunk's own mask rather than computed everywhere and
        selected, so an EK fill's transient stays proportional to the
        eligible pairs and a mostly-ineligible model costs almost nothing.

        **The C++ fill (momwire#742) takes the whole method**, statics
        included: what arrives then is a :class:`_FusedMoments`, and the two
        halves of the split run together inside one kernel call. Nothing after
        this line changes shape or meaning — the callers still get two
        ``(n_obs, n_seg)`` complex planes — so the dispatch is exactly this
        `isinstance` and nothing downstream has a branch.
        """
        if isinstance(chunks, _FusedMoments):
            return chunks.evaluate(self, k, need_m1=need_m1, n_obs=n_obs)

        # The SOURCE set prepare was handed, off the token that carries it --
        # never a geometry passed in beside it (momwire#745).
        seg_h = chunks.seg_h
        n_seg = seg_h.size
        xg, wg = leggauss(self.n_qp_source)
        # Source quadrature in each segment's own local arc coordinate.
        tau = 0.5 * seg_h[:, None] * (1.0 + xg[None, :])
        wq = 0.5 * seg_h[:, None] * wg[None, :]

        M0 = np.empty((n_obs, n_seg), dtype=np.complex128)
        M1 = np.empty((n_obs, n_seg), dtype=np.complex128) if need_m1 else None
        inv4pi = 1.0 / (4.0 * np.pi)
        for lo, hi, R, m0s, m1s, ekc in chunks:
            self._checkpoint()
            if ekc is None:
                rem = _expm1_neg_jkR(k, R) / R
            else:
                mask, a_ek = ekc
                num = _expm1_neg_jkR(k, R)
                scalar_a = np.ndim(a_ek) == 0
                if mask is None:
                    # `None` = every pair of this chunk is eligible, settled
                    # on the geometry side. Elementwise-identical to the
                    # gathered branch — same operands in the same order, only
                    # the broadcast shape of `a` differs.
                    a_m = a_ek if scalar_a else np.asarray(a_ek)[None, :, None]
                    num = num * _ek_factor(R, a_m, k) + _ek_reg_extra(R, a_m, k)
                else:
                    Rm = R[mask]
                    a_m = (
                        a_ek
                        if scalar_a
                        else np.broadcast_to(np.asarray(a_ek)[None, :], mask.shape)[
                            mask
                        ][:, None]
                    )
                    num[mask] = num[mask] * _ek_factor(Rm, a_m, k) + _ek_reg_extra(
                        Rm, a_m, k
                    )
                rem = num / R
            M0[lo:hi] = (m0s + np.einsum("psq,sq->ps", rem, wq)) * inv4pi
            if need_m1:
                M1[lo:hi] = (m1s + np.einsum("psq,sq->ps", rem, tau * wq)) * inv4pi
        return M0, M1

    def _seg_moments(self, obs, geom, k, *, need_m1=True, ek=None):
        """Kernel moments of every segment at every observation point.

        Returns ``(M0, M1)`` of shape ``(n_obs, n_seg)`` with

            M0 = ∫_seg g(r, r') dl',   M1 = ∫_seg τ' g(r, r') dl'

        (τ' the source's local arc from its segment start, g the reduced
        kernel *including* the 1/4π). The static 1/(4πR) part is the
        closed form in :func:`_static_axis_moments`; the remainder
        (exp(−jkR)−1)/(4πR) is smooth everywhere the reduced kernel is
        defined and takes plain Gauss-Legendre. `M1` is None when
        `need_m1` is False — the scalar-potential term only needs M0.

        A thin composition of :meth:`_seg_moments_prepare` and
        :meth:`_seg_moments_from_prepared` for a single wavenumber; the
        swept assembly (`compute_impedance_swept`, `compute_y_matrix_swept`)
        calls those two halves directly so the prepare step runs once per
        sweep instead of once per k.

        `ek` is passed straight through. It is not defaulted from
        `self.extended_kernel`, because eligibility is a property of the
        (observer, source) PAIR and `obs` here is an arbitrary point set this
        method cannot label — the fill knows which segment each of its
        observers lies on (`_ek_obs_labels_path`) and supplies the spec.
        """
        chunks = self._seg_moments_prepare(obs, geom, self._kernel_radius(geom), ek=ek)
        return self._seg_moments_from_prepared(chunks, k, obs.shape[0], need_m1=need_m1)

    # ------------------------------------------------------------------
    # assembly

    def _knot_points(self, geom):
        """The knot each basis is centred on, read off its side-A wing."""
        seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
        s_a = geom["wing_seg"][:, 0]
        along = np.where(geom["wing_rise"][:, 0], seg_h[s_a], 0.0)
        return seg_p0[s_a] + along[:, None] * seg_t[s_a]

    def _testing_paths(self, geom):
        """Quadrature points, tangents and weights of every testing path.

        Path P_m runs centroid(wing A) → knot m → centroid(wing B), each
        half traversed in the direction the basis current flows there. For
        an interior tent that is the unit-1 path centroid(left) → knot →
        centroid(right); for a junction tent it is A's terminal segment,
        through the junction, into B's — which may run against either
        wire's own arc direction, hence the ±1 wing signs.

        Each half gets its own `n_qp_path` Gauss-Legendre rule and its own
        tangent: on a kinked wire (or across a junction) the two halves
        point in different directions, which is exactly what T1's tangent
        dot product has to see.

        Returns ``(pts, tans, wts)`` shaped (n_basis, 2·n_qp_path, 3/3/–).
        """
        seg_t, seg_h = geom["seg_t"], geom["seg_h"]
        wing_seg, wing_sigma = geom["wing_seg"], geom["wing_sigma"]
        cent = geom["seg_p0"] + 0.5 * seg_h[:, None] * seg_t
        knot = self._knot_points(geom)

        if self.nec5_quadrature:
            # NEC-5's identified rule (momwire#316): one node per wing, AT
            # the wing's centroid, weighted by that half-path's length —
            # the two-point centroid trapezoid on a uniform mesh. The
            # downstream fill consumes (pts, tans, wts) shape-agnostically.
            s_a, s_b = wing_seg[:, 0], wing_seg[:, 1]
            pts = np.stack([cent[s_a], cent[s_b]], axis=1)
            tans = np.stack(
                [
                    wing_sigma[:, 0][:, None] * seg_t[s_a],
                    wing_sigma[:, 1][:, None] * seg_t[s_b],
                ],
                axis=1,
            )
            wts = 0.5 * np.stack([seg_h[s_a], seg_h[s_b]], axis=1)
            return pts, tans, wts

        xo, wo = leggauss(self.n_qp_path)

        s_a, s_b = wing_seg[:, 0], wing_seg[:, 1]
        pts, tans, wts = [], [], []
        for j, (lo_pt, hi_pt, seg) in enumerate(
            ((cent[s_a], knot, s_a), (knot, cent[s_b], s_b))
        ):
            mid = 0.5 * (lo_pt + hi_pt)
            half = 0.5 * (hi_pt - lo_pt)
            pts.append(mid[:, None, :] + half[:, None, :] * xo[None, :, None])
            # The traversal direction IS the wing's flow direction: σ ±1
            # times the segment's own unit tangent.
            t_dir = wing_sigma[:, j][:, None] * seg_t[seg]
            tans.append(np.repeat(t_dir[:, None, :], xo.size, axis=1))
            # Each half-path is h/2 long, so the Gauss-Legendre Jacobian
            # that maps [-1, 1] onto it is h/4, not h/2.
            wts.append(0.25 * seg_h[seg][:, None] * wo[None, :])
        return (
            np.concatenate(pts, axis=1),
            np.concatenate(tans, axis=1),
            np.concatenate(wts, axis=1),
        )

    def _image_sources(self, geom):
        """The mirrored source geometry, or None in free space.

        The PEC image is a SOURCE-SIDE substitution and nothing else: the
        moment builder is handed the same three arrays it always gets, with
        every segment origin reflected through `z = ground_z` and every
        segment tangent z-flipped. Segment LENGTHS are untouched, and that
        is what makes the substitution total — a mirrored segment's local
        arc coordinate τ maps point-for-point onto the real one's
        (M(p₀ + τ t) = M p₀ + τ M t), so every wing shape, every charge
        doublet ±1/h and every quadrature node index means the same thing on
        both source sets. Observers stay real throughout; only the sources
        move.

        Both operations come from the ground object's `image_geometry()`
        (momwire#398 unit 1, reduced to mirror operations in unit 2), so
        this solver spells no reflection of its own — which is the pilot's
        whole claim: the capability arrives through the shared layer.
        """
        ground = _potential_ground.potential_ground_for(self, geom, self.k, self.omega)
        if ground is None:
            return None
        # The object is a pure mirror map HERE whichever ground it is — even
        # the composing one, whose exact-image half is the same mirrored
        # geometry scaled by C2. The mirrored geometry and its static
        # moments do not know what wavenumber will be asked for, which is
        # what lets a swept solve build them once in the prepare half and
        # replay them at every k.
        #
        # What DOES depend on ω is everything else the finite grounds need:
        # the reflection-coefficient `(w_A, w_Φ)` through ε̃(ω), and — one
        # step further — the Sommerfeld remainder's interpolation grid,
        # whose whole lattice is measured in wavelengths. So this method
        # carries forward only two BITS of the ground, `weighted` and
        # `compose`, both of which are properties of the ground MODEL and
        # not of the frequency; the weights and the grid are built per
        # solved wavenumber in `_assemble_Z_from_prepared`, and all the
        # k-independent half owes them is geometry (`_assemble_Z_prepare`).
        img = ground.image_geometry()
        return {
            "seg_p0": img.mirror_positions(geom["seg_p0"]),
            "seg_t": img.mirror_tangents(geom["seg_t"]),
            "seg_h": geom["seg_h"],
            "mirror_tangents": img.mirror_tangents,
            "weighted": ground.eps_tilde is not None,
            "compose": ground.mode == "compose",
        }

    # ------------------------------------------------------------------
    # wire loading (momwire#427)

    def _wire_of_seg(self, geom):
        """``(n_segs,)`` int array mapping segment index → wire index."""
        off = np.asarray(geom["seg_offsets"], dtype=np.int64)
        return np.repeat(np.arange(off.shape[0] - 1, dtype=np.int64), np.diff(off))

    def _lumped_site_index(self, geom, i, wire, arclength):
        """Basis index a lumped load's site names — the `loading_for` hook.

        The site is resolved through the same `_snap_to_knot` the feeds use,
        because a load and a source name a site the same way and must land
        on the same knot when they name the same one (momwire#427 gate 2 is
        exactly `Z_driven == Z_unloaded + Z_L`). A K >= 3 junction is
        refused here rather than in the shared layer: what is ambiguous
        there is a basis-layer fact, and the sentence differs from the
        feed's (which branch pair a load sits BETWEEN, not which pair a
        source drives).
        """
        basis, k_ends = self._snap_to_knot(geom, wire, arclength)
        if k_ends >= 3:
            raise NotImplementedError(
                f"lumped_loads[{i}]: the load snaps to a junction "
                f"where {k_ends} wire ends meet, and a series "
                "impedance there is ambiguous — it would have to "
                "name which pair of branches it sits between. Load "
                "an interior knot, or model the load on a short "
                "bridge wire off the junction."
            )
        return basis

    def _loading_stencil(self, geom):
        """The k-independent half of the loading term: which (row, column)
        pairs it touches, on which segment, with what geometric weight.

        **The derivation.** A loaded wire's surface condition is not
        E_tan = 0 but E_tan = Z_s(l)·I(l): the tangential field no longer
        vanishes on the conductor, it equals the series impedance per metre
        times the current there. Razor-blade testing enforces the condition
        as a PATH INTEGRAL along P_m (module docstring), so the loaded
        equation tested by row m is

            ∫_{P_m} E_tan dl  =  ∫_{P_m} Z_s(l)·I(l) dl

        and with I(l) = Σ_n I_n Λ_n(l) the extra term is a matrix, not a
        vector:

            L[m, n] = ∫_{P_m} Z_s(l) Λ_n(l) dl,      Z = Z_free + L

        — the testing-path integral of Z_s times each tent. The sign is
        `+`: the fill's Z is the voltage a unit current needs, and a series
        impedance in the path needs Z_s·I more of it. (`SinusoidalSolver` /
        `SinusoidalGalerkinSolver` spell the same physics with a MINUS
        because their G is assembled with the opposite global sign; the
        oracle that fixes it here is #427 gate 2, `Z_driven = Z_unloaded +
        Z_L` — the sign is not a convention this module gets to choose.)

        **In the path/wing idiom it is four numbers.** Both P_m and Λ_n are
        built out of wings. Row m's wings are HALF-segments: the half of
        wing i's segment adjacent to knot m. Column n's wings are the tent's
        two linear ramps. So a (row wing, column wing) pair contributes only
        when they live on the SAME segment, and then the integral is one of
        two numbers on a segment of length h:

            ∫ over the knot-side half of the ramp that peaks at that knot
                = ∫_{h/2}^{h} (τ/h) dτ = 3h/8
            ∫ over the same half of the ramp that peaks at the OTHER end
                = ∫_{h/2}^{h} (1 − τ/h) dτ = h/8

        i.e. `3h/8` when the two wings rise at the same end of the segment
        and `h/8` when they rise at opposite ends — `wing_rise` is exactly
        that bit. The path integral is over |dl| with the traversal
        direction carried in the tangent (`_testing_paths`), and Z_s·I is a
        vector along the wire, so the pair also carries σ_row·σ_col: the dot
        product of the path's direction with the current's. That is the
        whole term:

            L[m, n] = Σ_{wings i of m, j of n on a shared segment s}
                      σ_{m,i} σ_{n,j} · Z_s(segment s) ·
                      (3h_s/8 if rise_i == rise_j else h_s/8)

        Two sanity readings. On a uniform wire the row sums to
        Z'·(3h/4 + h/8 + h/8) = Z'·h, the exact ∫Z' dl over a path of length
        h, which is the statement that a CONSTANT current sees the whole
        path's impedance and no more. And L is SYMMETRIC — swapping row and
        column swaps `rise_i`/`rise_j`, which the comparison above does not
        see — even though razor's field matrix is not: a surface impedance
        is a local, reciprocal object and the testing rule does not spoil
        that. It is not, however, the Galerkin Gram (h/3, h/6 per segment);
        `wire_loss_power` uses that one, because dissipated power is a
        physical integral and does not care which rule tested the equation.

        **Junction tents need no special case.** A junction tent is two
        wings on two real segments exactly like an interior tent's, with σ
        telling each half which way the through-current runs there; a load
        on either arm of a junction is picked up by whichever wings share
        that arm's terminal segment.

        **The grounded-end tent takes the real half, and nothing else.**
        Its side-A wing is its own IMAGE (`_junction_wings`), spelled with
        σ_A = 0 — so entries with σ = 0 are dropped here, and both the
        grounded ROW (its testing path is the real half only) and the
        grounded COLUMN (only the real wing is a conductor) reduce to the
        contact segment. That is the physically right answer and it needs no
        branch: loading ON a grounded tent means loading the real base
        segment, at half the equivalent dipole's tented length, which is the
        same halving that makes a base-fed monopole return Z_dipole/2. A
        lumped load at the contact knot is likewise the base GAP's load, in
        series with the base gap's source, and lands on that tent's diagonal
        at full value — the same convention as the feed voltage there.

        **Lumped loads are the delta case of the same integral.** A load at
        knot p is Z_s(l) = Z_L·δ(l − l_p). Only P_p contains knot p (its
        neighbours' paths stop at the bounding centroids), and Λ_n(l_p) =
        δ_np because a tent is 1 at its own knot and 0 at every other, so
        the integral collapses to L[p, p] += Z_L — one diagonal entry, no
        stencil, no geometry. `_apply_loading` adds it directly.

        **The closed form is lane-independent, deliberately.**
        `nec5_quadrature` swaps the path rule for NEC-5's identified
        two-point centroid trapezoid (momwire#316), but that is a
        quadrature choice about a KERNEL integrand; this integrand is a
        product of two linear ramps, so there is no accuracy to trade and
        the closed form above IS the path integral in both lanes. Measured
        against the binary before it was decided: applying the trapezoid to
        the loading integral too moves the `LD 5` copper increment by
        ~0.002 Ω (0.8180 → 0.8162 + …j at N=24), which is below the
        printed resolution the twin lane is gated at — so the binary cannot
        discriminate, and the exact integral is chosen on principle rather
        than on a fit.

        Returns ``(rows, cols, seg, vals)`` arrays for the distributed term;
        `vals` is the real geometric weight above, so the per-solve work is
        one gather of Z_s and one unbuffered scatter-add. Pure geometry, so
        this lives on the k-independent side of `_assemble_Z_prepare` even
        though the VALUES it is multiplied by are not (skin effect moves
        with ω, and the swept gate in `tests/test_razor_loading.py` is what
        proves the split is on the right side).
        """
        wing_seg, wing_rise = geom["wing_seg"], geom["wing_rise"]
        wing_sigma = geom["wing_sigma"]
        n_basis, n_seg = wing_seg.shape[0], geom["n_segs_total"]

        ent_basis = np.repeat(np.arange(n_basis, dtype=np.int64), 2)
        ent_seg = wing_seg.reshape(-1)
        ent_rise = wing_rise.reshape(-1)
        ent_sigma = wing_sigma.reshape(-1)
        # σ == 0 is the grounded tent's image wing: no conductor lives there,
        # so it is dropped rather than multiplied by zero. Structural, not
        # arithmetic — the same standard the ground blocks are held to.
        keep = ent_sigma != 0.0
        ent_basis, ent_seg = ent_basis[keep], ent_seg[keep]
        ent_rise, ent_sigma = ent_rise[keep], ent_sigma[keep]

        order = np.argsort(ent_seg, kind="stable")
        ent_basis, ent_seg = ent_basis[order], ent_seg[order]
        ent_rise, ent_sigma = ent_rise[order], ent_sigma[order]
        starts = np.searchsorted(ent_seg, np.arange(n_seg + 1))

        # Every ORDERED pair of entries sharing a segment, by ragged
        # expansion of the segment-major runs — the same construction
        # `SinusoidalGalerkinSolver._shared_segment_pairs` uses on its own
        # CSR, written out here because the two solvers share no view type.
        counts = np.diff(starts)
        nnz = int(starts[-1])
        seg_of_entry = np.repeat(np.arange(n_seg, dtype=np.int64), counts)
        reps = counts[seg_of_entry]
        left = np.repeat(np.arange(nnz, dtype=np.int64), reps)
        ramp = np.arange(int(reps.sum()), dtype=np.int64) - np.repeat(
            np.cumsum(reps) - reps, reps
        )
        right = np.repeat(starts[seg_of_entry], reps) + ramp

        seg = seg_of_entry[left]
        h = np.asarray(geom["seg_h"], dtype=np.float64)[seg]
        same_end = ent_rise[left] == ent_rise[right]
        vals = np.where(same_end, 0.375, 0.125) * h
        vals *= ent_sigma[left] * ent_sigma[right]
        return {
            "rows": ent_basis[left],
            "cols": ent_basis[right],
            "seg": seg,
            "vals": vals,
        }

    @staticmethod
    def _apply_loading(Z, stencil, spec):
        """`Z += L` in place: the loading term at one ω (`_loading_stencil`).

        `spec` is the shared `_wire_loading.LoadingSpec` (momwire#428) — the
        per-segment Z_s(ω) and the resolved lumped sites, neither of which
        names a testing rule; the stencil is this formulation's whole share
        of the term.

        Unbuffered on both halves — a basis pair recurs once per shared
        segment, and two lumped loads may name the same knot (they are in
        series there, so they add).
        """
        z_seg = spec.z_seg
        if z_seg is not None:
            np.add.at(
                Z,
                (stencil["rows"], stencil["cols"]),
                z_seg[stencil["seg"]] * stencil["vals"],
            )
        if spec.lumped is not None:
            idx, z_l = spec.lumped
            np.add.at(Z, (idx, idx), z_l)
        return Z

    def _assemble_Z_prepare(self, geom, *, chop=None):
        """K-independent work for the razor-blade fill: stencils and moments.

        Everything `_assemble_Z_from_prepared` needs that does not depend on
        the wavenumber — the wing/path stencils built once in unit 1/2, and
        the static (:meth:`_seg_moments_prepare`) halves of the segment
        moments at both observation sets the fill uses (segment centroids
        for T2, testing-path quadrature points for T1). A wavenumber sweep
        calls this once and replays it through `_assemble_Z_from_prepared`
        for every k, instead of rebuilding it per k the way a plain loop
        over single solves would.

        Over a PEC ground the same work is done a SECOND time against the
        mirrored sources (:meth:`_image_sources`) and cached beside the
        first, under `prepared["image"]`. **Doubling the cache rather than
        rebuilding the image half per k is a deliberate schedule decision**
        (momwire#398 unit 2), because the mirrored static moments are
        exactly as k-independent as the real ones: R, `asinh` and `sqrt`
        over mirrored sources do not know what wavenumber will be asked
        for. Caching them keeps the prepare/replay contract whole — a swept
        point pays only the smooth remainder, on both source sets — and it
        keeps the two halves of the fold symmetric, which is what lets
        `_assemble_Z_source_block` serve both with one body.

        **What is NOT cached here, deliberately: the reflection-coefficient
        weights** (momwire#398 unit 4). `(w_A, w_Φ)` are functions of ε̃(ω),
        so caching them would be caching a wavenumber — the one thing this
        half must not hold. What the weighted fill takes from here instead
        is pure geometry: the testing-path quadrature POINTS (`obs_pts`,
        which free space and PEC never keep past this method) with their
        tangents, and the segment centroids and tangents the source side's
        specular rays are reflected from. The weights themselves are built
        per solved wavenumber in `_assemble_Z_from_prepared`, which is what
        `tests/test_razor_refl_coef_ground.py`'s swept gate proves by
        sweeping a ground whose ε̃ moves with ω. A weighted fill also takes
        a smaller row chunk (`_WEIGHTED_CHUNK_ELEMS`), because a weight
        window costs an order of magnitude more per (observer, source) pair
        than a moment window does; that is a memory decision and the answer
        does not depend on it.

        **And nothing at all of the Sommerfeld remainder** (momwire#398
        unit 5), which is the sharper form of the same rule. Its weights
        are constant (C₂ does not vary over the geometry), but its
        interpolation grid is a k-dependent object from top to bottom — the
        lattice is spaced in wavelengths, `max_image_distance` is bucketed
        in wavelengths, and the surfaces carry an ωμ normalisation — so a
        grid on this side of the boundary would be a wavenumber cached in
        the one place that must hold none. The composing fill therefore
        takes exactly two more arrays from here, both of them geometry: the
        REAL source segment endpoints (`src_l` / `src_r` — unmirrored, since
        the plane enters the remainder through the grid rather than through
        a mirrored source). Grid, source nodes and moment weights are built
        per solved wavenumber inside the producer
        `Remainder.field_windows` returns, and the swept gate in
        `tests/test_razor_sommerfeld_ground.py` is what proves it.

        Priced on the ByDipole1 N=96 deck at h = 5 m over PEC, so the
        trade is on the record rather than asserted:

        | lane               | image cache | rebuild-per-k costs |
        |--------------------|-------------|---------------------|
        | `nec5_quadrature`  | 3.1 MB      | +15.8 % per swept k |
        | default GL path    | 66.4 MB     | +15.6 % per swept k |

        The cache is *exactly* the size of the free-space one it sits
        beside — a grounded razor solve is a 2x-residency solve, in the same
        proportion the grounded fill already doubles the arithmetic — and
        the alternative buys that memory back at about a sixth of every
        swept point. On the 16 GB working assumption the doubling is
        affordable at any mesh this formulation is used at (a 4,000-segment
        GL fill is the first place it would matter, and razor's O(N²) dense
        LU bites first). Should that stop being true, the switch is local:
        drop `prepared["image"]`'s chunk lists and rebuild them inside
        `_assemble_Z_source_block`. `scripts/capture_razor_pec_nec5_lane.py`
        does not measure this — the numbers above come from the unit's
        report, and re-measuring is a ten-line script against
        `_seg_moments_prepare`.

        **`chop` — T2 evaluated AT the node** (momwire#813 unit 2). Razor's
        charge term differences the scalar potential between a testing
        path's two CENTROIDS, which is every row this formulation has ever
        filled. A row whose path is chopped at the interface does not have
        two centroids: one of its endpoints IS the knot, and the crossing
        assembly needs Φ(node) there. `chop` is the mapping
        ``{row: "A" | "B"}`` naming those rows and which HALF survives, in
        `_path_test_rows`' own vocabulary — ``"A"`` is centroid(A) → knot,
        ``"B"`` is knot → centroid(B) — and it adds ONE observer set here:
        the knot of each named row, prepared exactly as the centroids are,
        against both source sets when there is an image.

        `None` (the default) is every fill that exists today and builds
        nothing: `prepared["t2_chop"]` is absent, `_assemble_Z_source_block`
        takes no branch, and an unchopped row is bit-identical to what it
        was before this argument existed. That is the gate
        `tests/test_razor_t2_at_node_813.py` gets, on three anchors, rather
        than an argument that it must be.

        Two things it refuses rather than guesses. The extended kernel
        (`_CHOP_EK_REFUSAL`): a knot observer sits ON the interface, where
        the coaxial-and-equal-radius eligibility scan has no answer. And a
        row that is already GROUNDED, whose T2 the fill rewrites for the
        plane-is-the-reference reason below — two rules for one row's
        endpoints is a silent contradiction, so it raises.
        """
        seg_t, seg_h = geom["seg_t"], geom["seg_h"]
        wing_seg, wing_sigma, wing_rise = (
            geom["wing_seg"],
            geom["wing_sigma"],
            geom["wing_rise"],
        )
        n_basis = wing_seg.shape[0]
        s_a, s_b = wing_seg[:, 0], wing_seg[:, 1]
        h_a, h_b = seg_h[s_a], seg_h[s_b]
        # dΛ/dτ in the segment's own arc coordinate, signed by the flow.
        q_a = wing_sigma[:, 0] * np.where(wing_rise[:, 0], 1.0, -1.0) / h_a
        q_b = wing_sigma[:, 1] * np.where(wing_rise[:, 1], 1.0, -1.0) / h_b

        # The kernel's `a`, per SOURCE segment (or the scalar, when the model
        # is uniform). Read once off the real geometry and reused for the
        # mirrored source set: an image segment carries its own segment's
        # radius, the same way it carries its length. Under the extended
        # kernel it is also the EK radius, because eligibility requires equal
        # radii and the pair's common radius is therefore the source column's
        # (`_EK(a=None, ...)`).
        a_src = self._kernel_radius(geom)

        # The extended kernel's eligibility labels (momwire#398 D1), or None
        # when it is off — in which case nothing below builds a spec and no
        # EK code is entered at all. `src_lab` labels the REAL source
        # segments and `img_lab` the mirrored ones; the observers are real in
        # both blocks, so the observer labels are read off `src_lab` either
        # way and only the source half of the spec moves. That is the mirror
        # policy in one line: the ground supplies mirrored geometry, the
        # shared rule reads it, and the kernel has no ground branch.
        src_lab = img_lab = None
        if self.extended_kernel:
            src_lab, img_lab = self._ek_labels(geom, mirror=self.ground_z is not None)

        # --- scalar potential's observation set: segment centroids.
        # Centroid i lies on segment i, so the centroid observers' labels ARE
        # the segment labels.
        cent = geom["seg_p0"] + 0.5 * seg_h[:, None] * seg_t
        ek_cent = None if src_lab is None else _EK(None, src_lab, src_lab)
        ek_cent_img = None if img_lab is None else _EK(None, src_lab, img_lab)
        t2_chunks = self._seg_moments_prepare(cent, geom, a_src, ek=ek_cent)

        # --- the CHOPPED rows' second observation set: the knot itself
        # (momwire#813 unit 2). Same producer, same source column, same
        # k-independence — a knot is a point like a centroid — so this is one
        # more `_seg_moments_prepare` and no new machinery. Absent entirely
        # when nothing is chopped, which is what keeps every existing fill
        # bit-identical rather than merely equal.
        t2_chop = None
        if chop:
            if self.extended_kernel:
                raise ValueError(_CHOP_EK_REFUSAL)
            chop_rows = np.array(sorted(int(m) for m in chop), dtype=np.int64)
            bad = np.intersect1d(chop_rows, geom["grounded_bases"])
            if bad.size:
                raise ValueError(
                    f"row(s) {bad.tolist()} are both GROUNDED and chopped "
                    "(momwire#813): a grounded row's T2 already drops its "
                    "plane endpoint because the folded potential is zero "
                    "there, and a chopped row replaces an endpoint with the "
                    "knot -- two rules for one row's endpoints. Chop the "
                    "crossing rows only"
                )
            sides = [str(chop[int(m)]) for m in chop_rows]
            if any(s not in ("A", "B") for s in sides):
                raise ValueError(
                    f"chop sides must be 'A' or 'B' (momwire#813), got {sides}"
                )
            keep_a = np.array([s == "A" for s in sides])
            knot_pts = self._knot_points(geom)[chop_rows]
            t2_chop = {
                "rows": chop_rows,
                # True when the surviving half is centroid(A) -> knot, so the
                # knot is the path's AFTER endpoint; False when it is
                # knot -> centroid(B) and the knot is the BEFORE endpoint.
                "keep_a": keep_a,
                "n_obs": chop_rows.size,
                "pts": knot_pts,
                # Only ever handed to a w_Phi producer, which never reads a
                # tangent (`PotentialGround.weight_windows`: "Both axes'
                # tangents enter only through the A-term dyad"). The
                # surviving wing's is the honest one to carry anyway.
                "tans": np.where(
                    keep_a[:, None],
                    seg_t[s_a[chop_rows]],
                    seg_t[s_b[chop_rows]],
                ),
            }

        # --- vector potential's observation set: the outer path, row-chunked.
        pts, tans, wts = self._testing_paths(geom)
        n_path = pts.shape[1]
        # σ folded into the source-side tangent, so the dot product carries
        # the wing's current direction with it.
        tan_a = wing_sigma[:, 0][:, None] * seg_t[s_a]
        tan_b = wing_sigma[:, 1][:, None] * seg_t[s_b]
        td_a = tan_a.T  # (3, n_basis)
        td_b = tan_b.T
        # Columns whose wing falls (knot at the segment's arc-0 end) need
        # M0 − M1/h instead of M1/h. On a junction-free model that is every
        # B wing and no A wing, so patching the exceptions in place keeps
        # the no-junction fill exactly as cheap as it was in unit 1.
        fall_a = np.flatnonzero(~wing_rise[:, 0])
        fall_b = np.flatnonzero(~wing_rise[:, 1])
        # The image, if there is one: the SAME observers and the SAME row
        # windows, against mirrored sources. Built inside the one loop so
        # free space allocates nothing extra and keeps no reference to
        # `pts` past this method — the ground layer is structurally absent
        # when off, not skipped (architecture doc §6, gate (b)).
        img_src = self._image_sources(geom)
        weighted = img_src is not None and img_src["weighted"]
        budget = _WEIGHTED_CHUNK_ELEMS if weighted else _CHUNK_ELEMS
        rows = max(1, budget // max(1, n_path * n_basis))
        # Each path point's EK label is the label of the wing segment it lies
        # on, in the same flattening the observers themselves take.
        path_lab = None if src_lab is None else self._ek_obs_labels_path(geom, src_lab)
        t1_row_chunks = []
        t1_row_chunks_img = [] if img_src is not None else None
        for lo in range(0, n_basis, rows):
            hi = min(lo + rows, n_basis)
            obs = pts[lo:hi].reshape(-1, 3)
            o_lab = None if path_lab is None else path_lab[lo * n_path : hi * n_path]
            ek_path = None if o_lab is None else _EK(None, o_lab, src_lab)
            t1_row_chunks.append(
                (
                    lo,
                    hi,
                    obs.shape[0],
                    self._seg_moments_prepare(obs, geom, a_src, ek=ek_path),
                )
            )
            if img_src is not None:
                ek_path_img = None if img_lab is None else _EK(None, o_lab, img_lab)
                t1_row_chunks_img.append(
                    (
                        lo,
                        hi,
                        obs.shape[0],
                        self._seg_moments_prepare(obs, img_src, a_src, ek=ek_path_img),
                    )
                )

        prepared = {
            "n_basis": n_basis,
            "n_seg": seg_h.size,
            "s_a": s_a,
            "s_b": s_b,
            "h_a": h_a,
            "h_b": h_b,
            "q_a": q_a,
            "q_b": q_b,
            "n_cent": cent.shape[0],
            "t2_chunks": t2_chunks,
            # `None` unless a caller named chopped rows; the block branches on
            # it and on nothing else.
            "t2_chop": t2_chop,
            "n_path": n_path,
            "tans": tans,
            "wts": wts,
            "td_a": td_a,
            "td_b": td_b,
            "fall_a": fall_a,
            "fall_b": fall_b,
            "t1_row_chunks": t1_row_chunks,
            "grounded": geom["grounded_bases"],
            "image": None,
            # Wire loading's k-independent half (momwire#427). The stencil
            # is pure geometry — which path half meets which tent ramp on
            # which segment, and the 3h/8 / h/8 that is — so it belongs
            # here; the Z_s(ω) it is scaled by does NOT, because the
            # skin-effect internal impedance and the insulation reactance
            # both move with ω, so they are built per solved wavenumber in
            # `_assemble_Z_from_prepared` exactly as the refl-coef weights
            # are. `None` when nothing is loaded, so an unloaded fill is
            # structurally unchanged rather than adding a zero.
            "loading": (
                self._loading_stencil(geom)
                if (self._loading_active or self.lumped_loads)
                else None
            ),
        }
        if t2_chop is not None:
            # Keyed off `sources` in the block, exactly as `t2_chunks` is, so
            # the real and mirrored source sets are served by one body.
            prepared["t2_chop_chunks"] = self._seg_moments_prepare(
                t2_chop["pts"], geom, a_src, ek=None
            )

        if img_src is not None:
            mirror = img_src["mirror_tangents"]
            prepared["image"] = {
                "t2_chunks": self._seg_moments_prepare(
                    cent, img_src, a_src, ek=ek_cent_img
                ),
                # The image sign, in razor's own (3, n_basis) idiom: M·t on
                # the SOURCE tangent table only. Nothing N² is formed, and
                # nothing on the observer side moves.
                "td_a": mirror(tan_a).T,
                "td_b": mirror(tan_b).T,
                "t1_row_chunks": t1_row_chunks_img,
                "weighted": weighted,
            }
            if t2_chop is not None:
                prepared["image"]["t2_chop_chunks"] = self._seg_moments_prepare(
                    t2_chop["pts"], img_src, a_src, ek=None
                )
            if weighted:
                # What the ω-dependent half needs from the k-independent
                # one: the two OBSERVER SETS the specular rays are drawn to
                # — the testing-path quadrature points with the path's own
                # tangent there (T1), and the segment centroids the charge
                # term differences between (T2) — and the SOURCE set, the
                # real segment centroids and tangents the pair's image
                # midpoint is reflected from. All geometry, all views or
                # arrays already built above; the weights themselves are
                # not here, because ε̃(ω) is not.
                prepared["image"]["obs_pts"] = pts.reshape(-1, 3)
                prepared["image"]["obs_tans"] = tans.reshape(-1, 3)
                prepared["image"]["src_c"] = cent
                prepared["image"]["src_t"] = seg_t
            if img_src["compose"]:
                # The composing ground's remainder integrates over the REAL
                # source segments — the plane enters its field through the
                # interpolation grid, not through a mirrored source — so
                # these endpoints are unmirrored even though they ride in
                # the image block's dict, which is where the term is summed
                # (`_assemble_Z_source_block`). Endpoints rather than
                # centroids because that axis is integrated, not sampled.
                prepared["image"]["src_l"] = geom["seg_p0"]
                prepared["image"]["src_r"] = geom["seg_p0"] + seg_h[:, None] * seg_t
        return prepared

    def _assemble_Z_from_prepared(self, geom, prepared, k, omega):
        """Fill the razor-blade impedance matrix at one wavenumber.

        Free space is one source block. Over a ground it is that block
        minus the image's:

            Z = Z_free − Z_image

        one global minus at the very end, exactly as the mixed-potential
        trunk's other solvers spell their fold (`PotentialGround.mode ==
        "fold"`, architecture doc §2.2) — and the same one minus whether
        the image is the PEC one or the Fresnel-weighted one, because a
        weight is a per-pair scale on the image block and cannot move a
        sign the fold takes once. It is the same one minus over the
        COMPOSING ground too (unit 5), and that is the point of composing
        rather than a coincidence: `C₂·img + Q` is associated inside
        `_assemble_Z_source_block`, so what arrives here is one block
        again. Both halves of the minus are real
        physics and they arrive together: the horizontal image current runs
        anti-parallel and the image charge is opposite, so *both* the
        vector-potential term and the charge term change sign — which is
        why the sign can be taken once on the assembled block instead of
        term by term. What the mirrored tangent table supplies is the rest:
        M·t restores the +z component the global minus would otherwise have
        flipped the wrong way, so a vertical current's image stays parallel.

        Only `k` and `omega` (=c·k, passed separately so a swept caller can
        reuse one `omega_array` without recomputing it) vary here; every
        other ingredient comes from `prepared` (`_assemble_Z_prepare`).
        """
        if getattr(self, "_crossing", False):
            return self._assemble_Z_crossing(geom, k, omega)
        if getattr(self, "_below_plane", False):
            return self._assemble_Z_below_plane(geom, prepared, k, omega)
        Z = self._assemble_Z_source_block(geom, prepared, prepared, k, omega)
        image = prepared["image"]
        if image is not None:
            # The ground object is built HERE, not carried from the prepare
            # half, because this is the ω-dependent side of the split and a
            # reflection-coefficient ground's weights are ω-dependent: ε̃(ω)
            # is hoisted into the factory, so one object per solved
            # wavenumber is exactly one ε̃ per wavenumber. Over PEC it is a
            # handful of scalar stores and the block ignores it.
            ground = _potential_ground.potential_ground_for(self, geom, k, omega)
            Z -= self._assemble_Z_source_block(
                geom, prepared, image, k, omega, ground=ground
            )
        # Loading last, and OUTSIDE the fold: `Z = (Z_free − Z_image) + L`.
        # The loading term is a property of the conductor's surface, not of
        # the field, so it takes no image and no weight — the plane's
        # presence changes which fields the wire sees, never the impedance
        # per metre of the wire itself. That is also why it needs no branch
        # per ground: this one line serves free space, both folding grounds
        # and the composing one.
        if prepared["loading"] is not None:
            self._apply_loading(
                Z, prepared["loading"], _wire_loading.loading_for(self, omega, geom)
            )
        return Z

    # ------------------------------------------------------------------
    # the crossing trunk's view of this solver (momwire#813)
    # ------------------------------------------------------------------

    def _crossing_context(self, geom, k=None, omega=None, *, ground_eps=None):
        """What `_crossing_fill` reads off this solver, as data (momwire#801):
        razor's tents as per-segment polynomials, the five geometry
        columns, the buried medium, and the four scalars.

        The tent basis in `BasisPolynomials`' language is one line per wing:
        `σ·u/h` on a rising wing (value 1 at the knot end), `σ·(1 − u/h)` on
        a falling one, read straight off `wing_seg` / `wing_rise` /
        `wing_sigma`. A wing with σ = 0 (a grounded tent's image side) is a
        zero polynomial, which is the padding case `axis_data` guards.
        Measured through the trunk against razor's own free-space fill at
        ε̃ = 1 (momwire#651's probe): the interior cross block to 6.6e-6.

        `ground_eps` overrides the solver's, so a gate can ask for the ε̃ = 1
        collapse on a free-space solver whose geometry is the same.
        """
        k = self.k if k is None else k
        omega = self.omega if omega is None else omega
        eps_spec = self.ground_eps if ground_eps is None else ground_eps
        n_basis = geom["n_basis_total"]
        seg_h, seg_t, seg_p0 = geom["seg_h"], geom["seg_t"], geom["seg_p0"]
        wing_seg, wing_rise = geom["wing_seg"], geom["wing_rise"]
        wing_sigma = geom["wing_sigma"]
        supp = np.zeros((n_basis, 2), dtype=np.int64)
        polys = np.zeros((n_basis, 2, 2))
        for n in range(n_basis):
            for j in range(2):
                seg = int(wing_seg[n, j])
                h = seg_h[seg]
                sig = float(wing_sigma[n, j])
                supp[n, j] = seg
                if wing_rise[n, j]:
                    polys[n, j] = (0.0, sig / h)
                else:
                    polys[n, j] = (sig, -sig / h)
        return _crossing_fill.CrossingContext(
            basis=_crossing_fill.BasisPolynomials(supp, polys, 1),
            geom=_crossing_fill.AxisGeometry(
                seg_p0,
                seg_p0 + seg_h[:, None] * seg_t,
                seg_h,
                seg_t,
                np.asarray(geom["seg_offsets"], dtype=np.int64),
            ),
            medium=_crossing_fill.buried_medium(eps_spec, omega, self.eps, k),
            ground_z=0.0 if self.ground_z is None else float(self.ground_z),
            a_wire=float(self._radius_per_wire[0]),
            omega=omega,
            mu=self.mu,
            eps=self.eps,
        )

    def _path_test_rows(self, geom, rows, *, halves="both"):
        """`_crossing_fill.path_test_axis` records for razor's testing paths.

        One record per row in `rows` — the path's quadrature points, tangents
        and weights from `_testing_paths`, the segment each point lies on,
        and the two centroids the T2 term differences between. `halves`
        selects the whole path (`"both"`), or one half of it with the KNOT
        as the shared endpoint (`"A"`: centroid(A) → knot, `"B"`: knot →
        centroid(B)) — how a row whose path crosses the plane is chopped at
        it (momwire#813), since the trunk's tables take an observer on one
        side only.
        """
        pts, tans, wts = self._testing_paths(geom)
        q = pts.shape[1] // 2
        seg_h, seg_t, seg_p0 = geom["seg_h"], geom["seg_t"], geom["seg_p0"]
        wing_seg = geom["wing_seg"]
        cent = seg_p0 + 0.5 * seg_h[:, None] * seg_t
        knot = self._knot_points(geom)
        out = []
        for m in rows:
            s_a, s_b = int(wing_seg[m, 0]), int(wing_seg[m, 1])
            if halves == "both":
                sl = slice(0, 2 * q)
                seg = np.concatenate([np.full(q, s_a), np.full(q, s_b)])
                before, after = cent[s_a], cent[s_b]
            elif halves == "A":
                sl, seg, before, after = (
                    slice(0, q),
                    np.full(q, s_a),
                    cent[s_a],
                    knot[m],
                )
            elif halves == "B":
                sl, seg, before, after = (
                    slice(q, 2 * q),
                    np.full(q, s_b),
                    knot[m],
                    cent[s_b],
                )
            else:
                raise ValueError(f"halves must be 'both', 'A' or 'B', got {halves!r}")
            out.append((m, pts[m, sl], tans[m, sl], wts[m, sl], seg, before, after))
        return out

    def _assemble_Z_below_plane(self, geom, prepared, k, omega, *, plan_skip=None):
        """The razor-blade matrix of a WHOLLY-below deck (momwire#812, unit 1
        of the razor buried arc), in the lower-medium family:

            Z = Z_direct(k_m, ε_m) − [ A_m·Z_image(k_m, ε_m) + Q_below ] + L

        It is the composing ground's fold with the medium's numbers in it —
        the kernel at `k_m = k₂·√ε̃` (the fused moments kernel takes a complex
        k since momwire#796), Φ under `ε_m = ε₀·ε̃`, the image weighted by
        `A_m = image_coefficient_below(ε̃)` THROUGH the windows (never applied
        here: `BelowMediumGround` hands it over as C₂ is handed over), and the
        below remainder in place of the above one. Nothing else in the fill
        changes, which is the point: `_assemble_Z_source_block` is called
        twice exactly as it is over a Sommerfeld ground, with `eps` and the
        ground object swapped.

        The two serve-plan refusals a buried grid can hit
        (`_BURIED_PAST_CAP_REFUSAL`, `_BURIED_GRAZING_REFUSAL`) are asked
        here, before any grid is filled, over segment endpoints and
        centroids — the same R₁ = hypot(ρ, d + d′) and θ = atan2(d + d′, ρ)
        `BSplineSolver._buried_serve_plan` asks over its nodes.
        """
        from .bspline import _BURIED_GRAZING_REFUSAL, _BURIED_PAST_CAP_REFUSAL

        gz = float(self.ground_z)
        eps_t = _ground_refl.eps_tilde(self.ground_eps, omega, self.eps)
        ground = _potential_ground.BelowMediumGround(
            self, geom, k, omega, eps_tilde=eps_t
        )
        k_m, eps_m = ground.k_m, ground.eps_m

        # The plan's two refusals, over endpoints + centroids.
        seg_h, seg_t, seg_p0 = geom["seg_h"], geom["seg_t"], geom["seg_p0"]
        pts = np.concatenate(
            [
                seg_p0,
                seg_p0 + seg_h[:, None] * seg_t,
                seg_p0 + 0.5 * seg_h[:, None] * seg_t,
            ]
        )
        if plan_skip is not None and len(plan_skip):
            # `plan_skip` is the DECLARED crossing nodes (momwire#813), and
            # only those. A point sitting exactly in the interface has depth
            # 0, so its pair with ITSELF has rho = 0 and d + d' = 0 and
            # `atan2(0, 0)` reads 0 deg — below any floor, from a pair that
            # is not a physical pair at all. That is a sampling artefact of a
            # point the CROSSING block owns rather than a grazing geometry,
            # and one nanometre lower the same deck fills without complaint
            # (measured, `scratch/813-crossing-assembly-a/probe1_plane_end.py`).
            #
            # Keyed on the declared node and NEVER on "any point at depth 0":
            # a wholly-below wire that merely touches the plane at an end
            # which is not a crossing member has no crossing block to carry
            # its current, and must keep refusing exactly as it does today.
            # `test_a_plane_touching_end_that_is_not_a_crossing_member_still_
            # refuses` is what holds that line.
            skip = np.asarray(plan_skip, dtype=float).reshape(-1, 3)
            keep = np.ones(pts.shape[0], dtype=bool)
            for node in skip:
                keep &= np.linalg.norm(pts - node[None, :], axis=1) > _JUNCTION_TOL
            pts = pts[keep]
        d = gz - pts[:, 2]
        rho = np.hypot(
            pts[:, 0][:, None] - pts[:, 0][None, :],
            pts[:, 1][:, None] - pts[:, 1][None, :],
        )
        hh = d[:, None] + d[None, :]
        r1_max = float(np.max(np.hypot(rho, hh)))
        th_min = float(np.min(np.arctan2(hh, rho)))
        lam_m = 2.0 * np.pi / abs(k_m)
        cap = _sommerfeld_below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
        if r1_max > cap:
            raise ValueError(
                _BURIED_PAST_CAP_REFUSAL.format(
                    r1=r1_max,
                    wl=r1_max / lam_m,
                    cap=cap,
                    capwl=_sommerfeld_below._SOMM_BELOW_R1_CAP_LAMBDA_M,
                    lam_m=lam_m,
                )
            )
        floor = math.radians(_sommerfeld_below._SOMM_BELOW_TH_MIN_DEG)
        if th_min < floor:
            raise ValueError(
                _BURIED_GRAZING_REFUSAL.format(
                    th=math.degrees(th_min),
                    floor=_sommerfeld_below._SOMM_BELOW_TH_MIN_DEG,
                    depth=2.0 * float(np.min(d)),
                )
            )

        Z = self._assemble_Z_source_block(
            geom, prepared, prepared, k_m, omega, eps=eps_m
        )
        Z -= self._assemble_Z_source_block(
            geom, prepared, prepared["image"], k_m, omega, ground=ground, eps=eps_m
        )
        # Loading last and outside the fold, exactly as above.
        if prepared["loading"] is not None:
            self._apply_loading(
                Z, prepared["loading"], _wire_loading.loading_for(self, omega, geom)
            )
        return Z

    # ------------------------------------------------------------------
    # the crossing fill (momwire#813 unit 2)
    # ------------------------------------------------------------------

    def _crossing_tents(self, geom):
        """``[(basis, below_wing)]`` for every tent whose wings span the plane.

        `below_wing` is 0 or 1 — which wing sits on a BELOW segment. A tent
        with both wings in one medium is not here; it is an ordinary row of
        that medium's fill. Which tents are crossing ones depends on the
        deck's WIRE ORDER, because `_junction_wings` pairs every member
        against `ends[0]`: on `fan_rise_deck()` as written the above riser is
        the last wire, so one tent crosses and three are below-family, while
        with the riser listed first all four cross. Both spell the same
        current space, and the assembly serves either.
        """
        media = self._wire_media()
        seg_off = np.asarray(geom["seg_offsets"])
        below = set()
        for w, label in enumerate(media):
            if label == _medium_spec.BELOW:
                below.update(range(int(seg_off[w]), int(seg_off[w + 1])))
        out = []
        for m in range(geom["n_basis_total"]):
            sides = [int(geom["wing_seg"][m, j]) in below for j in (0, 1)]
            if sides[0] != sides[1]:
                out.append((m, 0 if sides[0] else 1))
        return out

    def _medium_geometry(self, geom, side):
        """``(geom, rows, chop)`` for ONE medium's own fill.

        The medium's segments, the bases it owns, and — for each crossing
        tent — that tent's wing on THIS side as a half tent, the other wing
        becoming a ghost at ``sigma = 0``, which is razor's own contact-tent
        shape. `rows` is each sub-row's index in the FULL basis, in order,
        and is therefore the basis map; `chop` moves each crossing row's T2
        endpoint to the node (momwire#813 unit 2), which the ghost alone does
        NOT do — zeroing a wing chops T1 and the doublet and leaves T2
        spanning both centroids, the trap
        `test_the_sigma_trick_does_not_chop_the_path` pins.

        Seven keys, because seven is all the fill reads: `_kernel_radius`
        returns the scalar here (the crossing serve refuses per-wire radii),
        so no `seg_offsets`, no `per_wire`, no junction table. `grounded_bases`
        is EMPTY on purpose — see the demotion note in `_build_geometry` for
        why no potential reference may be taken at a medium interface.
        """
        media = self._wire_media()
        seg_off = np.asarray(geom["seg_offsets"])
        bas_off = np.asarray(geom["basis_offsets"])
        seg_i, bas_i = [], []
        for w, label in enumerate(media):
            if label == side:
                seg_i.append(np.arange(int(seg_off[w]), int(seg_off[w + 1])))
                bas_i.append(np.arange(int(bas_off[w]), int(bas_off[w + 1])))
        seg_i = np.concatenate(seg_i) if seg_i else np.zeros(0, dtype=np.int64)
        mine = set(seg_i.tolist())
        remap = {int(s): i for i, s in enumerate(seg_i)}
        ws, wr, wg = geom["wing_seg"], geom["wing_rise"], geom["wing_sigma"]

        rows = list(np.concatenate(bas_i) if bas_i else np.zeros(0, dtype=np.int64))
        for m in range(int(bas_off[-1]), geom["n_basis_total"]):
            if any(int(ws[m, j]) in mine for j in (0, 1)):
                rows.append(m)
        n = len(rows)
        g_ws = np.empty((n, 2), dtype=np.int64)
        g_wr = np.empty((n, 2), dtype=bool)
        g_wg = np.empty((n, 2), dtype=np.float64)
        chop = {}
        for i, m in enumerate(rows):
            sides = [int(ws[m, j]) in mine for j in (0, 1)]
            for j in (0, 1):
                src = j if sides[j] else (1 - j)
                g_ws[i, j] = remap[int(ws[m, src])]
                g_wr[i, j] = bool(wr[m, src])
                g_wg[i, j] = float(wg[m, j]) if sides[j] else 0.0
            if not all(sides):
                # the surviving wing names the surviving half, in
                # `_path_test_rows`' vocabulary
                chop[i] = "A" if sides[0] else "B"
        return (
            {
                "seg_p0": geom["seg_p0"][seg_i],
                "seg_t": geom["seg_t"][seg_i],
                "seg_h": geom["seg_h"][seg_i],
                "wing_seg": g_ws,
                "wing_rise": g_wr,
                "wing_sigma": g_wg,
                "grounded_bases": np.zeros(0, dtype=np.int64),
                "n_basis_total": n,
            },
            np.asarray(rows, dtype=np.int64),
            chop,
        )

    def _crossing_path_axis(self, geom, tents, side):
        """The path-test axis of one medium's rows over the FULL geometry:
        its own tents whole, plus each crossing tent's `side` half."""
        media = self._wire_media()
        seg_off = np.asarray(geom["seg_offsets"])
        mine_seg = set()
        for w, label in enumerate(media):
            if label == side:
                mine_seg.update(range(int(seg_off[w]), int(seg_off[w + 1])))
        crossing = {m for m, _ in tents}
        recs = []
        for m in range(geom["n_basis_total"]):
            if m in crossing:
                alive = [j for j in (0, 1) if int(geom["wing_seg"][m, j]) in mine_seg][
                    0
                ]
                recs += self._path_test_rows(
                    geom, [m], halves="A" if alive == 0 else "B"
                )
            elif int(geom["wing_seg"][m, 0]) in mine_seg:
                recs += self._path_test_rows(geom, [m])
        return _crossing_fill.path_test_axis(geom["n_basis_total"], recs)

    def _assemble_Z_crossing(self, geom, k, omega):
        """The razor-blade matrix of a deck that CROSSES the interface, as
        four masked terms indexed by (row HALF) x (column WING):

            Z[R_a, C_a] += the above fill at k_p, on the above geometry
            Z[R_b, C_b] += the below fill at k_m, on the below geometry
            Z[R_a, C_b] -= the trunk's cross block          (corner=False)
            Z[R_b, C_a] -= the trunk's REVERSED cross block (momwire#832)

        Every crossing tent is in all four, so its (jn, jn) entry is the sum
        of its four half-x-wing pieces and every other entry takes exactly
        one term.

        Each same-medium block is filled on its OWN geometry rather than
        sliced out of a whole-deck fill, and that is not tidiness: the below
        family's remainder refuses any observer or source that is not
        strictly below the plane, so a whole-deck fill cannot be computed and
        then restricted. `_medium_geometry` builds the seven keys the fill
        reads and hands back the basis map.

        `corner=False` on both trunk blocks: the corner is a Galerkin
        by-parts term and a path-tested row has no by-parts to do. The SW end
        term STAYS, which is momwire#813 derivation (b) — and the eps~ = 1
        collapse this method is gated on cannot see that choice, because W
        vanishes there. Soil is what reads it.
        """
        if self._loading_active or self.lumped_loads:
            # Before any fill: the stencil is per-medium here and its cross
            # terms are not derived, and `_medium_geometry` does not carry
            # the keys `_loading_stencil` reads anyway.
            raise NotImplementedError(
                "wire loading on a crossing deck is not served "
                "(momwire#813): the loading stencil is per-medium here and "
                "its cross terms are not derived"
            )
        tents = self._crossing_tents(geom)
        nodes = self._knot_points(geom)[[m for m, _ in tents]]

        geom_a, rows_a, chop_a = self._medium_geometry(geom, _medium_spec.ABOVE)
        prep_a = self._assemble_Z_prepare(geom_a, chop=chop_a)
        Z_a = self._assemble_Z_source_block(geom_a, prep_a, prep_a, k, omega)
        if prep_a["image"] is not None:
            ground_a = _potential_ground.potential_ground_for(self, geom_a, k, omega)
            Z_a -= self._assemble_Z_source_block(
                geom_a, prep_a, prep_a["image"], k, omega, ground=ground_a
            )

        geom_b, rows_b, chop_b = self._medium_geometry(geom, _medium_spec.BELOW)
        prep_b = self._assemble_Z_prepare(geom_b, chop=chop_b)
        Z_b = self._assemble_Z_below_plane(geom_b, prep_b, k, omega, plan_skip=nodes)

        ctx = self._crossing_context(geom, k=k, omega=omega)
        axis_kw = dict(
            growth=_CROSSING_GROWTH,
            panel_order=_CROSSING_PANEL_ORDER,
            q=_CROSSING_Q,
        )
        A = self._crossing_path_axis(geom, tents, _medium_spec.ABOVE)
        P = self._crossing_path_axis(geom, tents, _medium_spec.BELOW)
        seg_off = np.asarray(geom["seg_offsets"])
        media = self._wire_media()
        seg_of = {
            side: np.concatenate(
                [
                    np.arange(int(seg_off[w]), int(seg_off[w + 1]))
                    for w, m in enumerate(media)
                    if m == side
                ]
            )
            for side in (_medium_spec.ABOVE, _medium_spec.BELOW)
        }
        ax_a = _crossing_fill.axis_data(ctx, seg_of[_medium_spec.ABOVE], **axis_kw)
        ax_b = _crossing_fill.axis_data(ctx, seg_of[_medium_spec.BELOW], **axis_kw)

        n = geom["n_basis_total"]
        Z = np.zeros((n, n), dtype=np.complex128)
        Z[np.ix_(rows_a, rows_a)] += Z_a
        Z[np.ix_(rows_b, rows_b)] += Z_b
        Z[np.ix_(rows_a, rows_b)] -= _crossing_fill.cross_complete_block(
            ctx, A, ax_b, corner=False
        )[np.ix_(rows_a, rows_b)]
        Z[np.ix_(rows_b, rows_a)] -= _crossing_fill.cross_complete_block_reversed(
            ctx, P, ax_a, corner=False
        )[np.ix_(rows_b, rows_a)]

        return Z

    def _assemble_Z_source_block(
        self, geom, prepared, sources, k, omega, *, ground=None, eps=None
    ):
        """One source set's contribution to the razor-blade matrix.

        `sources` supplies the three source-side pieces — the two moment
        chunk lists and the `(3, n_basis)` tangent tables — and is either
        `prepared` itself (the real sources) or `prepared["image"]` (the
        mirrored ones). Everything else is shared: the observers, the
        testing paths and their weights, the wing stencils, the charge
        doublets. That split IS the PEC image: same rows, same test
        functions, moved sources.

        Both terms are built from the same two segment moments. A wing on
        segment s of length h contributes to the vector potential

            A_n += σ · (M1[s] / h)              when the knot is at arc h
            A_n += σ · (M0[s] − M1[s] / h)      when it is at arc 0

        carried by σ times that segment's tangent, so the outer path
        point's tangent contracts with each wing separately. The charge
        doublet is the constant dσΛ/dτ on each wing — which works out to
        +1/h_A on side A and −1/h_B on side B whichever way round the two
        wires are spelled, i.e. the unit charge that leaves A and lands on
        B — differenced between the path's two bounding centroids.

        A GROUNDED tent (momwire#398 unit 3) is that with side A carried by
        the image instead of by a wire: σ_A = 0 empties its half of both
        terms, and the fold's second pass supplies the mirrored wing with
        the opposite charge, so the through-basis's two doublet halves are
        −1/h on the real segment and +1/h on its image. The unit of current
        that flows into the plane leaves no net charge at the contact point
        — the image takes exactly the charge the real wing deposits, which
        is the same statement as Φ = 0 on the conductor.

        `ground` is the caller's `PotentialGround` when this block is the
        IMAGE one, and `None` for the real sources. Over PEC nothing here
        reads it. Over the reflection-coefficient ground (momwire#398 unit
        4) it supplies the two `(w_A, w_Φ)` window producers this block's
        two terms consume, and the branch is on the OBJECT — `eps_tilde is
        None` is `PotentialGround`'s own tell for "PEC" — not on the
        solver's `ground_eps` string. Their physics, per term:

        * **T1 takes w_A.** The Fresnel field dyad is not a scalar on the
          image current: it scales the in-plane components by ρ_v and the
          out-of-plane horizontal one by −ρ_h, and `a_term_weights` has
          already resolved that into a single number per (observer, source
          segment) pair by contracting it with BOTH tangents. So w_A is
          exactly what the PEC fill spells as `t_out · M·t_n` — the same
          slot, generalised — and the PEC limit ρ_v → 1, ρ_h → −1 returns
          it to that number identically. Razor's `(3, n_basis)` tangent
          table cannot carry it (a table of that shape has no room for the
          pair's specular angle), which is why the weights arrive as
          windows over the path points instead, and why the wing's σ — the
          only part of that table that is not a tangent — is applied here,
          after the gather. w_A is linear in the source tangent, so σ
          multiplying it is the same physics as σ multiplying the tangent.
        * **T2 takes w_Φ,** applied to the image kernel at each centroid
          BEFORE the two centroids are differenced, because each end of a
          testing path has its own specular geometry. The PEC path leaves
          this term unweighted (w_Φ ≡ 1) and this is the term with no
          field-trunk analogue at all: `ground_phi_mode` is a modelling
          choice about the image CHARGE, which only a formulation that
          separates the charge term can express. This one separates it
          explicitly, so the choice applies to T2's image-side doublets
          directly.

        Neither term multiplies `ground.image_coefficient`: on this trunk
        the coefficient enters THROUGH the weights (`PotentialGround`'s
        class docstring is explicit that applying it again is the obvious
        bug), and for a folding ground it is 1 in any case. Over the
        composing ground it is C₂ and the same sentence holds — the
        constant `(w_A, w_Φ)` pair `weight_windows` returns for
        `mode == "compose"` IS C₂ times the PEC mirror table, so the two
        terms above assemble the scaled exact image with no branch of their
        own. The seam's single global minus stays exactly where it was.

        **The composing ground's third term** (momwire#398 unit 5) is the
        one thing here that is not a weighting of the fold. Q is the smooth
        Sommerfeld remainder FIELD, tested by this formulation's own rule:
        the testing-path integral of the field of each source segment's
        tent moments, projected on the path's tangent. It rides T1's
        observer windows — same chunks, same path weights — and reaches
        them through `Remainder.field_windows`, the operation unit 5 split
        out of the Galerkin block `Remainder.evaluate` returns (razor's
        rows are not Galerkin projections, so the block was unusable at any
        shape). Three properties of Q are worth naming, because each is a
        way to get it wrong:

        * **no prefactor.** F is already an E-field (theory manual eq 123's
          unit current moment), so Q takes neither jωμ nor 1/jωε; it is
          added to the assembled block, not to T1 or T2;
        * **real sources.** Q integrates over the REAL segments — the plane
          is inside F, through the interpolation grid — even though it is
          summed into the IMAGE block, which is simply where the
          composition happens;
        * **before the minus.** `C₂·img + Q` is associated here so the
          caller's `Z -=` lands both terms; distributing it would be a
          different float64 answer, which is the whole content of the
          `"compose"` mode contract.
        """
        s_a, s_b = prepared["s_a"], prepared["s_b"]
        h_a, h_b = prepared["h_a"], prepared["h_b"]
        q_a, q_b = prepared["q_a"], prepared["q_b"]
        n_basis = prepared["n_basis"]

        w_A_fn = w_Phi_fn = rem_fn = w_Phi_knot_fn = None
        if ground is not None and ground.eps_tilde is not None:
            # Two observer sets, one source set. T2's observers ARE the
            # source set, which is what the producer's square default
            # means, so it is spelled by omitting them.
            src = (sources["src_c"], sources["src_t"])
            w_A_fn = ground.weight_windows(
                observers=(sources["obs_pts"], sources["obs_tans"]), sources=src
            )
            w_Phi_fn = ground.weight_windows(sources=src)
            if prepared["t2_chop"] is not None:
                # The chopped rows' knot observers have their own specular
                # geometry -- a knot is not a centroid -- so they get their
                # own window over the same source set. Only the Phi half is
                # read; the tangents carried in are the surviving wing's and
                # do not enter it.
                w_Phi_knot_fn = ground.weight_windows(
                    observers=(
                        prepared["t2_chop"]["pts"],
                        prepared["t2_chop"]["tans"],
                    ),
                    sources=src,
                )
            remainder = ground.remainder()
            if remainder is not None:
                # The composing ground's second term, over the SAME observer
                # windows T1 already walks and the REAL source segments.
                # `n_moment=2` is the tent: ∫Λ and ∫τΛ, from which the same
                # two combinations the reduced-kernel moments take (M1/h on
                # a rising wing, M0 − M1/h on a falling one) fall out below.
                # momwire#510: the order is keyed to grazing height rather
                # than taken flat from the kwarg. A deck with nothing near the
                # plane gets `self.n_qp_sommerfeld` exactly and is unchanged.
                rem_fn = remainder.field_windows(
                    (sources["obs_pts"], sources["obs_tans"]),
                    (sources["src_l"], sources["src_r"], sources["src_t"]),
                    n_moment=2,
                    n_qp=self.n_qp_sommerfeld
                    if ground.below
                    else _remainder_qp(
                        sources["obs_pts"],
                        sources["src_l"],
                        sources["src_r"],
                        self.ground_z,
                        self.n_qp_sommerfeld,
                    ),
                )

        M0c, _ = self._seg_moments_from_prepared(
            sources["t2_chunks"], k, prepared["n_cent"], need_m1=False
        )
        if w_Phi_fn is not None:
            n_cent = prepared["n_cent"]
            step = max(1, _WEIGHTED_CHUNK_ELEMS // max(1, prepared["n_seg"]))
            for c0 in range(0, n_cent, step):
                self._checkpoint()
                c1 = min(c0 + step, n_cent)
                _w_A_unused, w_Phi = w_Phi_fn(c0, c1)
                M0c[c0:c1] *= w_Phi
        dM0 = M0c[s_b] - M0c[s_a]  # (row, source segment)
        grounded = prepared["grounded"]
        if grounded.size:
            # A grounded row's testing path starts AT the plane, where the
            # folded scalar potential is identically zero: a point in the
            # plane is equidistant from every source and its image, so the
            # two blocks' contributions there are the same number and the
            # fold's minus cancels them. Dropping the term in each block is
            # therefore exact rather than approximate — and it is what makes
            # the plane this formulation's potential reference, the discrete
            # form of Φ = 0 on a perfect conductor.
            #
            # **Over a FINITE ground the drop is NOT exact, and it stays.**
            # This block's plane term has been scaled by w_Φ above and the
            # real block's has not, so the pair of drops discards
            # (1 − w_Φ)·M0(plane) rather than zero. The study's §4.3 read
            # that as the defect behind razor's contact refusal and §5.5
            # named the experiment; momwire#624 ran it, and the term does
            # not survive its own instrument:
            #
            #   * on the STUBBED LADDER — momwire against momwire, no binary,
            #     a self-consistent contact node must give an h-independent
            #     answer — coefficient 0 is flattest on every row by an order
            #     of magnitude, on both soils and BOTH ground models. At 0.4
            #     the ladder slides 42.18+25.82j → 33.58+16.50j as the stub
            #     shrinks, converging back onto the coefficient-0 answer: the
            #     term's contribution evaporates with the contacting element,
            #     so no SCALE makes it self-consistent;
            #   * against the binary it is worse at full strength (poor soil
            #     3.384 → 3.906 Ω at N = 61). One coefficient ≈ 0.4 is the
            #     argmin at a fixed mesh on every lossy ground, but that is a
            #     fit at one mesh, not a derivation, and the ladder above is
            #     the instrument that needs no reference.
            #
            # So the reference the plane gives this row is kept as it stands.
            # What the ladder DOES leave is a residual with a target: the
            # finite-ground ladders spread 0.21-0.55 Ω where PEC holds 0.002,
            # so the contact node is internally inconsistent over a finite
            # ground by about half an ohm. `test_razor_contact_finite_ground`
            # pins that, and it is the thing to attack next — not this term.
            dM0[grounded] = M0c[s_b[grounded]]
        chop = prepared["t2_chop"]
        if chop is not None:
            # momwire#813 unit 2: a chopped row's path ends at the KNOT, so
            # one of the two potentials differenced above is Phi(node) rather
            # than Phi(centroid). Same moments, same source column, one
            # different observer -- and the difference keeps its orientation,
            # `after - before`, because that is what carries T2's sign.
            M0k, _ = self._seg_moments_from_prepared(
                sources["t2_chop_chunks"], k, chop["n_obs"], need_m1=False
            )
            if w_Phi_knot_fn is not None:
                _w_A_unused, w_Phi_k = w_Phi_knot_fn(0, chop["n_obs"])
                M0k = M0k * w_Phi_k
            rows, keep_a = chop["rows"], chop["keep_a"]
            # "A": centroid(A) -> knot, the knot is AFTER.
            # "B": knot -> centroid(B), the knot is BEFORE.
            dM0[rows] = np.where(
                keep_a[:, None],
                M0k - M0c[s_a[rows]],
                M0c[s_b[rows]] - M0k,
            )
        T2 = dM0[:, s_a] * q_a[None, :] + dM0[:, s_b] * q_b[None, :]

        tans, wts = prepared["tans"], prepared["wts"]
        n_path = prepared["n_path"]
        td_a, td_b = sources["td_a"], sources["td_b"]
        fall_a, fall_b = prepared["fall_a"], prepared["fall_b"]
        # The kernel wants per-basis flags rather than the index arrays numpy
        # fancy-indexes with; the branch then hoists out of its observer loop.
        fall_mask_a = np.zeros(n_basis, dtype=bool)
        fall_mask_b = np.zeros(n_basis, dtype=bool)
        if fall_a.size:
            fall_mask_a[fall_a] = True
        if fall_b.size:
            fall_mask_b[fall_b] = True
        # The wings' current directions, needed on their own only by the
        # weighted branch: the unweighted one has them fused into `td_a` /
        # `td_b`, which the weight window replaces wholesale.
        sig_a, sig_b = geom["wing_sigma"][:, 0], geom["wing_sigma"][:, 1]
        # momwire#744, routed by #806: the window the fused weighted assembler
        # will apply, asked ONCE per block rather than per chunk -- it is a
        # property of the ground, not of the row window. None keeps the numpy
        # closure. Every parameter the kernel needs travels inside the rule,
        # so nothing below reads the ground's attributes.
        w_rule = (
            _weighted_window_rule(ground)
            if (w_A_fn is not None and _use_razor_weighted_accel())
            else None
        )

        T1 = np.empty((n_basis, n_basis), dtype=np.complex128)
        Q = None if rem_fn is None else np.empty_like(T1)
        for lo, hi, n_obs_chunk, static in sources["t1_row_chunks"]:
            self._checkpoint()
            M0, M1 = self._seg_moments_from_prepared(static, k, n_obs_chunk)
            if w_A_fn is None and _use_razor_assemble_accel():
                # momwire#780: the gather, the falling-wing correction, the
                # tangent contraction, the weighting and the path-point sum in
                # ONE pass, so the (n_obs, n_basis) complex intermediate the
                # numpy branch builds and immediately reduces is never formed.
                # At n_path = 64 on a 200-segment deck that temporary is ~32 MB
                # per chunk, and the assembly around it measured 52-76% of
                # razor's wall time (both lanes, both grounds).
                #
                # Both lanes come through here. `n_path` is a loop bound in the
                # kernel, not a branch: 2 under `nec5_quadrature`, 2*n_qp_path
                # under Gauss-Legendre.
                T1[lo:hi] = _acc.razor_assemble_t1(
                    M0,
                    M1,
                    s_a,
                    s_b,
                    h_a,
                    h_b,
                    fall_mask_a,
                    fall_mask_b,
                    tans[lo:hi].reshape(-1, 3),
                    td_a,
                    td_b,
                    wts[lo:hi].reshape(-1),
                    n_path,
                )
            elif w_rule is not None:
                # momwire#744, extended to the composing ground by #806: the
                # weighted twin of the branch above. The per-pair A-term
                # window is formed INSIDE the tile from the same geometry the
                # numpy closure reads, so the (n_obs_chunk, n_seg) window
                # plane is never materialised -- at N=801 over a refl-coef
                # ground that plane and the contraction around it were 40%
                # and 28% of the fill.
                #
                # WHICH window is the ground's own answer, arriving as a rule
                # rather than as attributes this fill reads. That is the whole
                # of #806: C2 reaches Z through the windows, and #804's draft
                # doubled the exact-image half by reading the coefficient
                # here.
                #
                # The observer rows are this chunk's path points, sliced out
                # of the SAME arrays `w_A_fn` closes over, so the kernel and
                # the closure cannot disagree about which observers these are.
                T1[lo:hi] = _acc.razor_assemble_t1_weighted(
                    M0,
                    M1,
                    s_a,
                    s_b,
                    h_a,
                    h_b,
                    fall_mask_a,
                    fall_mask_b,
                    sig_a,
                    sig_b,
                    sources["obs_pts"][lo * n_path : hi * n_path],
                    sources["obs_tans"][lo * n_path : hi * n_path],
                    sources["src_c"],
                    sources["src_t"],
                    wts[lo:hi].reshape(-1),
                    n_path,
                    w_rule.kind,
                    w_rule.eps_t,
                    w_rule.ground_z,
                    w_rule.coefficient,
                )
            else:
                mom_a = M1[:, s_a] / h_a[None, :]
                mom_b = M1[:, s_b] / h_b[None, :]
                if fall_a.size:
                    mom_a[:, fall_a] = M0[:, s_a[fall_a]] - mom_a[:, fall_a]
                if fall_b.size:
                    mom_b[:, fall_b] = M0[:, s_b[fall_b]] - mom_b[:, fall_b]
                if w_A_fn is None:
                    t_out = tans[lo:hi].reshape(-1, 3)
                    integrand = (t_out @ td_a) * mom_a + (t_out @ td_b) * mom_b
                else:
                    # The window's observer rows are this chunk's path points,
                    # which are rows [lo, hi) of the path table flattened by
                    # `n_path` — the same reshape the unweighted branch takes
                    # on `tans`.
                    w_A, _w_Phi_unused = w_A_fn(lo * n_path, hi * n_path)
                    wA_a = w_A[:, s_a] * sig_a[None, :]
                    wA_b = w_A[:, s_b] * sig_b[None, :]
                    integrand = wA_a * mom_a + wA_b * mom_b
                integrand *= wts[lo:hi].reshape(-1)[:, None]
                T1[lo:hi] = integrand.reshape(hi - lo, n_path, n_basis).sum(axis=1)
            if rem_fn is not None:
                # The remainder rides the same window, and the same wing
                # algebra one axis over: the moment axis carries ∫Λ and ∫τΛ of
                # the projected field where `mom_a` / `mom_b` carry M0 and M1.
                f_mom = rem_fn(lo * n_path, hi * n_path)
                rem_a = f_mom[:, s_a, 1] / h_a[None, :]
                rem_b = f_mom[:, s_b, 1] / h_b[None, :]
                if fall_a.size:
                    rem_a[:, fall_a] = f_mom[:, s_a[fall_a], 0] - rem_a[:, fall_a]
                if fall_b.size:
                    rem_b[:, fall_b] = f_mom[:, s_b[fall_b], 0] - rem_b[:, fall_b]
                rem_int = rem_a * sig_a[None, :] + rem_b * sig_b[None, :]
                rem_int *= wts[lo:hi].reshape(-1)[:, None]
                Q[lo:hi] = rem_int.reshape(hi - lo, n_path, n_basis).sum(axis=1)
        # `eps` is the medium's (momwire#812's lower medium hands ε_m); the
        # default is the free-space ε₀ every other call passes implicitly.
        eps_here = self.eps if eps is None else eps
        block = 1j * omega * self.mu * T1 - T2 / (1j * omega * eps_here)
        if Q is None:
            return block
        # `C2·img + Q`, associated BEFORE the seam's single minus — the
        # whole content of `mode == "compose"`, since
        # `free − (C2·img + Q) ≠ (free − C2·img) − Q` in float64. The two
        # halves meet HERE and the caller's `Z -=` is untouched.
        return block + Q

    def _assemble_Z(self, geom, k):
        """Fill the razor-blade impedance matrix at one wavenumber.

        A thin composition of `_assemble_Z_prepare` and
        `_assemble_Z_from_prepared` for a single solve; `compute_impedance`
        and `compute_y_matrix` call this one k at a time, while
        `compute_impedance_swept` / `compute_y_matrix_swept` call the two
        halves directly so the prepare step runs once per sweep.
        """
        prepared = self._assemble_Z_prepare(geom)
        return self._assemble_Z_from_prepared(geom, prepared, k, self.c * k)

    # ------------------------------------------------------------------
    # solve

    def _refuse_coincident_segments(self, geom):
        """Refuse a bundle BY NAME before the solve dies in LAPACK
        (momwire#846).

        Called from the solve entry points and NOT from `_build_geometry` or
        the assembly: the fill on such a deck is well defined and
        momwire#813's collapse gates measure it against bspline's to 7.3e-11.
        It is the solve that has no answer, so this is where the sentence
        belongs.

        The test is exact rather than tolerant — two segments with the same
        pair of endpoints, rounded to nanometres. A bundle is authored as
        coincident geometry, not arrived at by drift, and a tolerant test
        would start refusing merely CLOSE conductors, which are a different
        (and served) thing.
        """
        p0 = np.asarray(geom["seg_p0"], dtype=float)
        p1 = p0 + np.asarray(geom["seg_h"], dtype=float)[:, None] * np.asarray(
            geom["seg_t"], dtype=float
        )
        seen = {}
        for i in range(p0.shape[0]):
            a = tuple(np.round(p0[i], 9))
            b = tuple(np.round(p1[i], 9))
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                raise ValueError(
                    f"segments {seen[key]} and {i} run between the same two "
                    f"points: {_BUNDLE_REFUSAL}"
                )
            seen[key] = i

    def compute_impedance(self):
        """Drive-point impedance(s) and the solved tent coefficients.

        Returns ``(z, coeffs)``. With one feed `z` is the scalar
        V / I at the feed knot; with N feeds it is the length-N vector
        V_i / I_i, one entry per feed, from the single solve against the
        superposed right-hand side Σ_i V_i e_{m_i}. `coeffs` is the solved
        coefficient vector — on a tent basis the coefficient at a knot IS
        the current there (amperes), so `coeffs` is the knot-current
        distribution in basis order: per wire, interior knots in arc order,
        then the junction through-currents (junctions in detection order,
        K−1 per junction), each measured from its junction's side A into
        that tent's side B. A GROUNDED point contributes K of them instead,
        one per wire end there, each the current flowing out of the plane
        into that end.
        """
        geom = self._build_geometry()
        self._checkpoint()
        Z = self._assemble_Z(geom, self.k)
        self.z = Z

        self._refuse_coincident_segments(geom)
        cols = self._port_columns(geom)
        # NEC-5's EX at a knot: the delta gap sits inside exactly one
        # testing path, so the whole voltage lands in that one row.  A node
        # gap spreads over its junction's tents instead (`_port_columns`).
        rhs = cols @ self._port_voltages()

        self._checkpoint()
        coeffs = scipy.linalg.solve(Z, rhs)
        voltages = self._port_voltages()
        port_currents = cols.T @ coeffs
        z_per_port = voltages / port_currents
        n_ports = cols.shape[1]
        return (z_per_port[0] if n_ports == 1 else z_per_port), coeffs

    def compute_y_matrix(self):
        """Short-circuit admittance matrix [Y_sc] at the configured feeds.

        Ports are the `feeds` entries; their voltages are ignored here.
        ``Y[i, j]`` is the current at port i when port j is driven with
        1 V and every other port is held at 0 V — which on this delta-gap
        model is simply an all-zero row entry, so the N columns are N
        back-substitutions on one factored Z. Returns an
        ``(n_ports, n_ports)`` complex array; invert it for the
        open-circuit Z matrix of network analysis.

        The result is NOT symmetric, and that is the formulation talking:
        razor-blade testing weights the E-field boundary condition with
        path integrals rather than with the basis itself, so reciprocity
        is only recovered as the mesh refines (momwire#309). A Galerkin
        scheme on the same basis would be symmetric to machine precision.
        `tests/test_razor_junctions.py` pins the decay.

        This is the `y` field of `compute_port_solution()` and nothing else
        — see there for the per-port solution columns this throws away
        (momwire#232, #429 rank-9).
        """
        return self.compute_port_solution().y

    def _port_count(self):
        """Ports `compute_port_solution` returns — the configured `feeds`
        and nothing else. Junction ports and node gaps are both refused at
        construction (`_OUT_OF_SCOPE`), so there is nothing after them."""
        return len(self.feeds)

    def compute_port_solution(self, prepared=None) -> PortSolution:
        """Solve every port from ONE fill and ONE factorisation (#429 rank-9).

        Returns a `PortSolution` whose `y` is identical to
        `compute_y_matrix()` — that method is implemented as
        `compute_port_solution().y`, so the two cannot drift apart — and
        whose `coeffs` column j is the tent/junction coefficient vector for
        a 1 V drive at port j with every other port shorted. On this basis
        the coefficient AT a knot IS the current there, so `port_currents`
        (== `y`) is read straight off `coeffs` at the feed rows, with no
        separate readout function the way the segment-basis families need
        one. Ports run ``[gap feeds..., node gaps...]``, the order
        :meth:`_port_columns` builds and :class:`PortSolution` documents —
        this formulation refuses `junction_ports` at construction, so it is
        the MIDDLE block that is empty here, not the tail (unlike the
        B-spline / sinusoidal-Galerkin families, whose ports run
        feeds-then-junction-ports-then-node-gaps). A node gap has been a
        razor port since momwire#603 gave this basis the apex port it always
        had the through-current unknown for.

        The fill is exactly `_assemble_Z`'s own composition — one
        k-independent `_assemble_Z_prepare` and one `_assemble_Z_from_prepared`
        at `(self.k, self.c * self.k)`, the same omega expression `_assemble_Z`
        itself uses rather than the separately-stashed `self.omega` — so this
        method's `Z` cannot diverge from `compute_impedance` /
        `compute_y_matrix`'s, not even by the 1-ULP omega drift a bare
        `self.omega` read would cost it. Each call factors and
        solves fresh (`scipy.linalg.solve`, not a stashed LU): no
        factorisation is reused ACROSS `compute_impedance` /
        `compute_y_matrix` / `compute_port_solution` calls on one instance,
        but WITHIN this call every port shares the one fill and the one
        `scipy.linalg.solve` over all `n_ports` right-hand-side columns at
        once — the "one fill, one factorisation" the swept generator below
        relies on.

        `prepared` is the sweep's hoisted `_assemble_Z_prepare` result (see
        `_port_solutions_swept`); it changes nothing about the answer, it
        just spares the per-k rebuild of the wing/path stencils and the
        closed-form static segment moments when the sweep drives this
        method frequency by frequency — the same schedule
        `compute_impedance_swept` / `compute_y_matrix_swept` already use.

        `basis` is an opaque `_RazorBasis` handle, stable across the ports
        of this one solution and NOT across solves.
        """
        geom = self._build_geometry()
        self._checkpoint()
        if prepared is None:
            prepared = self._assemble_Z_prepare(geom)
        # `self.c * self.k`, not `self.omega`: `_assemble_Z` (and every
        # existing entry point built on it — `compute_impedance`, the old
        # `compute_y_matrix`, both swept loops) computes omega this way,
        # never from the stashed `self.omega`. The two are mathematically
        # identical but not bit-identical (`2·π·(c/λ)` at `__init__` time vs
        # `c·(2·π/λ)` here), so reading `self.omega` would cost this method
        # the branch point's bit-for-bit answer over a 1-ULP omega drift.
        Z = self._assemble_Z_from_prepared(geom, prepared, self.k, self.c * self.k)
        self.z = Z

        self._refuse_coincident_segments(geom)
        cols = self._port_columns(geom)

        self._checkpoint()
        X = scipy.linalg.solve(Z, cols.astype(np.complex128))
        Y = cols.T @ X
        return PortSolution(
            y=Y,
            coeffs=X,
            port_currents=Y,  # the same object: the readout IS the Y matrix
            basis=_RazorBasis(geom=geom, k=self.k),
        )

    # ------------------------------------------------------------------
    # field readout

    def currents_at_knots(self, coeffs, s_array=None):
        """Per-wire complex current at every mesh knot (momwire#309 unit 3).

        Returns a list of 1-D arrays, one per wire in `wires_polylines`
        order, each of length ``n_segments_of_wire + 1`` — one value per
        KNOT, not per basis.

        Interior knot j of wire w reads straight off the tent coefficient:
        interior tent n peaks at value 1 on its own knot and is zero at
        every other knot, so the current there is
        ``coeffs[basis_offsets[w] + j - 1]``.

        A FREE wire end (no junction there, and not in the ground plane) is
        0 — the open-circuit BC the tent basis already builds in.

        A GROUNDED wire end reads its grounded tent the same way a
        junctioned end reads its junction tents: the tent's real wing sits
        on that end with `wing_sigma` ±1, and the ghost wing carrying the
        image contributes 0 by construction (`wing_sigma` 0), so the sum
        below is that one term — the current crossing the plane into the
        wire, in the wire's own arc direction.

        A JUNCTIONED wire end is not itself a basis's home knot the way an
        interior knot is: this formulation's junction tents (unlike the
        retired triangular-family solver's directional end-bases) are
        through-current unknowns shared between two DIFFERENT wires, each
        carrying its own wing sign onto that shared knot. The current at a
        junctioned end knot, expressed in THAT WIRE'S OWN ARC DIRECTION, is
        therefore the signed sum of every junction tent with a wing sitting
        on that (wire, end): each such wing contributes
        ``wing_sigma · coeff`` — `wing_sigma` already carries the sign that
        turns the tent's through-current into a multiple of this wire's own
        arc-length direction (see the module docstring's "Junctions"
        section and `_junction_wings`). At a K=2 junction (an ordinary
        two-wire join, or an interior knot re-spelled as a split) exactly
        one wing sits on each side and the sum is a single term; at a K>=3
        junction, side A carries a wing from every one of the K−1 tents
        there, so its knot sums all of them (Kirchhoff's law falling out of
        the wing bookkeeping — `tests/test_razor_junctions.py`'s
        `_currents_into_junction` helper is the same sum, used there to
        check KCL rather than to read off a knot current).

        With `s_array` given as a list of 1-D per-wire arc-length arrays,
        the tent's own piecewise-linear shape means this is exactly a
        linear interpolation of the knot values along the wire's cumulative
        arc (`per_wire[w]["arc_at_knot"]`), not a re-solve of anything.
        """
        coeffs = np.asarray(coeffs)
        geom = self._build_geometry()
        per_wire = geom["per_wire"]
        seg_offsets = geom["seg_offsets"]
        basis_offsets = geom["basis_offsets"]
        n_interior = geom["n_basis_interior"]
        wing_seg, wing_rise, wing_sigma = (
            geom["wing_seg"],
            geom["wing_rise"],
            geom["wing_sigma"],
        )

        out = []
        for w_idx, pw in enumerate(per_wire):
            n_knots = pw["arc_at_knot"].shape[0]
            I = np.zeros(n_knots, dtype=np.complex128)
            I[1:-1] = coeffs[basis_offsets[w_idx] : basis_offsets[w_idx + 1]]
            out.append(I)

        n_basis_total = wing_seg.shape[0]
        if n_basis_total > n_interior:
            # Flatten the junction tents' two wings into one (seg, rise,
            # sigma, coeff) list per side, so a wire's end knot is just
            # "every wing whose (segment, rise) is that end's".
            j_seg = wing_seg[n_interior:].reshape(-1)
            j_rise = wing_rise[n_interior:].reshape(-1)
            j_sigma = wing_sigma[n_interior:].reshape(-1)
            j_coeff = np.repeat(coeffs[n_interior:], 2)
            for w_idx in range(len(per_wire)):
                start_seg = seg_offsets[w_idx]
                end_seg = seg_offsets[w_idx + 1] - 1
                at_start = (j_seg == start_seg) & ~j_rise
                if at_start.any():
                    out[w_idx][0] = (j_sigma[at_start] * j_coeff[at_start]).sum()
                at_end = (j_seg == end_seg) & j_rise
                if at_end.any():
                    out[w_idx][-1] = (j_sigma[at_end] * j_coeff[at_end]).sum()

        if s_array is None:
            return out

        sampled = []
        for w_idx, sv in enumerate(s_array):
            arc = per_wire[w_idx]["arc_at_knot"]
            knot_I = out[w_idx]
            sv = np.asarray(sv, dtype=np.float64)
            Ire = np.interp(sv, arc, knot_I.real)
            Iim = np.interp(sv, arc, knot_I.imag)
            sampled.append(Ire + 1j * Iim)
        return sampled

    def current_slopes(self, coeffs, s_array=None):
        """Per-wire ``dI/ds`` — the solved current's arc-length derivative.

        The twin of :meth:`currents_at_knots`, with the same signature and
        the same two calling conventions as
        :meth:`~momwire.bspline.BSplineSolver.current_slopes`: a list of 1-D
        complex arrays, one per wire in ``wires_polylines`` order, at the mesh
        knots (``s_array=None``) or at the per-wire arc positions given,
        clipped into the wire's own arc range.

        **Why it exists** (momwire#497, and momwire#603 for this family): the
        linear charge density a NEC printout reports is ``q = -(1/jw)·dI/ds``
        at each element's centre.  The EZNEC seam reads it through this
        method, and until it existed here that seam could not run on this
        family at all — every deck that reached the readout died on a missing
        attribute rather than on anything about the physics.

        Here it is not merely exact, it is trivial.  A tent expansion IS the
        piecewise-linear interpolant of its own knot currents, so on the
        segment between two knots the current is a straight line and ``dI/ds``
        is the CONSTANT rise over run.  Nothing is differenced that was not
        already a difference: this is the same "differentiated in the basis
        rather than around it" argument the B-spline twin makes, at the one
        degree where the two readings coincide.

        The derivative therefore JUMPS at a knot, and a sample taken exactly
        on one has to pick a side.  This picks the span to the RIGHT, and the
        left span at the final knot — which is what scipy's ``BSpline``
        derivative does at ``degree == 1``, so a caller cannot tell the two
        implementations apart by their tie-break either.  As over there, the
        honest thing to ask for is element CENTRES, and that is what the seam
        asks for.
        """
        coeffs = np.asarray(coeffs)
        per_wire = self._build_geometry()["per_wire"]
        knot_currents = self.currents_at_knots(coeffs)

        out = []
        for w_idx, pw in enumerate(per_wire):
            arc = pw["arc_at_knot"]
            # One constant per segment, and the whole of the derivative.
            slope = np.diff(knot_currents[w_idx]) / np.diff(arc)
            s_eval = (
                arc if s_array is None else np.asarray(s_array[w_idx], dtype=np.float64)
            )
            if s_eval.shape[0] == 0:
                out.append(np.zeros(0, dtype=np.complex128))
                continue
            s_eval = np.clip(s_eval, arc[0], arc[-1])
            # `side="right"` is the right-hand span; the clip puts the final
            # knot back on the last one.
            span = np.clip(
                np.searchsorted(arc, s_eval, side="right") - 1, 0, slope.shape[0] - 1
            )
            out.append(np.asarray(slope[span], dtype=np.complex128))
        return out

    # ------------------------------------------------------------------
    # swept solve

    def wire_loss_power(self, coeffs, omega=None):
        """Ohmic power dissipated in the wire metal, from a solve's coeffs.

        P_wire = ½ Σ_w Re[Z'_w(ω)] · ∫_w |I(l)|² dl — the readout the
        downstream power budget reports, with the same signature and the
        same ``(total_watts, per_wire_watts)`` return as
        `BSplineSolver.wire_loss_power` and `SinusoidalSolver`'s. Insulation
        loading is purely reactive and contributes nothing here.

        This is a PHYSICAL integral over the conductor, so it is the
        Galerkin overlap of the tent basis with itself, not the testing-path
        stencil `_loading_stencil` builds: on a segment of length h carrying
        end currents a (at arc 0) and b (at arc h) the linear current
        interpolates between them and

            ∫_0^h |a(1−τ/h) + b τ/h|² dτ = h(|a|² + Re[a b̄] + |b|²)/3.

        A grounded end's tent contributes its REAL wing only (σ = 0 empties
        the image wing), which is right: the image is not metal. Lumped
        loads are not counted — they are components, not wire; their own
        watts are `lumped_load_power`, right below.
        """
        n_w = len(self.wires_polylines)
        per_wire = np.zeros(n_w, dtype=np.float64)
        if not self._loading_active:
            return 0.0, per_wire
        if omega is None:
            omega = self.omega
        geom = self._build_geometry()
        alpha = np.asarray(coeffs, dtype=np.complex128)[: geom["n_basis_total"]]

        n_seg = geom["n_segs_total"]
        ent_seg = geom["wing_seg"].reshape(-1)
        ent_rise = geom["wing_rise"].reshape(-1)
        ent_val = np.repeat(alpha, 2) * geom["wing_sigma"].reshape(-1)
        # The knot current each wing deposits at its segment's two ends: a
        # rising wing peaks at arc h, a falling one at arc 0.
        i_lo = np.zeros(n_seg, dtype=np.complex128)
        i_hi = np.zeros(n_seg, dtype=np.complex128)
        np.add.at(i_hi, ent_seg[ent_rise], ent_val[ent_rise])
        np.add.at(i_lo, ent_seg[~ent_rise], ent_val[~ent_rise])

        h = np.asarray(geom["seg_h"], dtype=np.float64)
        int_abs_i2 = (
            h
            * (np.abs(i_lo) ** 2 + np.real(i_lo * np.conj(i_hi)) + np.abs(i_hi) ** 2)
            / 3.0
        )
        # zeros where switched off. `geom` goes in even though only `z_wire`
        # is read from the spec: `loading_for` ALSO resolves the lumped sites,
        # and `_lumped_site_index` needs the geometry to snap them — omitting
        # it raised `TypeError` on any solver carrying BOTH a conductivity and
        # a `lumped_loads` entry, which is every lossy loaded deck.
        r_w = np.real(_wire_loading.loading_for(self, omega, geom).z_wire)
        wire_of = self._wire_of_seg(geom)
        np.add.at(per_wire, wire_of, 0.5 * r_w[wire_of] * int_abs_i2)
        return float(per_wire.sum()), per_wire

    def lumped_load_power(self, coeffs):
        """Ohmic power dissipated in each LUMPED load, from a solve's coeffs
        (momwire#433, following on #427/#431).

        P_load[i] = ½·Re(Z_Li)·|I(knot_i)|² — the same ``(total_watts,
        per_load_watts)`` convention as `wire_loss_power`, but this is a
        SUM over the loads' own delta terms, not a physical integral over
        the wire: `_loading_stencil`'s "Lumped loads are the delta case of
        the same integral" reading says a load at knot p is Z_s(l) =
        Z_L·δ(l − l_p), so its readout is one term per load, not a Gram.

        **Why `coeffs[idx]` is already I(knot) in amps — no wing_sigma.**
        `wire_loss_power` reconstructs current as `repeat(alpha, 2) *
        wing_sigma` because it needs each wire SEGMENT's own local-arc
        current for a per-segment integral. A knot current is a simpler
        question the tent basis answers directly: `_loading_stencil`
        collapses the loading integral at a lumped site because "Λ_n(l_p)
        = δ_np — a tent is 1 at its own knot and 0 at every other", i.e.
        I(l_p) = Σ_n I_n Λ_n(l_p) = I_p. The coefficient IS the nodal
        current; `wing_sigma` only re-expresses that same current in a
        segment's own arc direction for a different (segment-length)
        integral, and does not change here because |σ| = 1 on every wing a
        load can reach: an interior knot's two wings are both σ = 1; a K=2
        junction's real wing is ±1; a grounded end's side-A wing is its
        image (σ = 0, unused) and side-B — the real base segment — is ±1.
        (`_lumped_site_index` refuses K >= 3, so no other shape reaches
        here.) ±1 flips a sign that |I|² erases, so `coeffs[idx]` —
        NOT `coeffs[idx] * wing_sigma` — is exactly right, and a grounded
        load lands on its tent's diagonal at full value, the same
        convention as the feed voltage there (`_loading_stencil`).

        Two loads named at the same knot (`_apply_loading` sums both Z_L
        onto one diagonal entry) share that knot's current, so their
        shares ½·Re(Z_L1)|I|² and ½·Re(Z_L2)|I|² sum to
        ½·Re(Z_L1+Z_L2)|I|² — the series-equivalent load's own reading,
        because Re is linear. No special case is needed for it here.

        No loads configured ⇒ the structurally absent shape `(0.0, empty
        array)`, matching `wire_loss_power`'s "zeros, not a crash": there
        is no `omega` parameter because Z_L is a fixed complex constant
        from construction (`_wire_loading.normalize_lumped_loads`), unlike
        Z'_w(ω), so there is nothing to rebuild per solved wavenumber.
        That is also why the `(indices, Z_L)` resolution is taken straight
        off `_wire_loading.loading_for` — the SAME read the fill's own
        stamp used (momwire#428's "one shared resolution, four rows"), so
        a site this readout charges can never be a different knot from the
        site the matrix was bumped at, whatever `_lumped_site_index` later
        decides about snapping. `loading_for`'s ω-dependent half is the
        DISTRIBUTED one; the lumped branch it hands back does not read
        `omega` at all, so passing `self.omega` here is a formality, not a
        claim about when this readout is valid.

        ONE solve's coefficients only: `coeffs` must be a 1-D vector, and
        a 2-D multi-port block raises rather than broadcasting. `alpha[idx]`
        on an `(n_basis, n_ports)` block is `(n_loads, n_ports)`, and the
        power expression would silently sum ONE load's watts across EVERY
        port excitation whenever `n_loads` is 1 or equals `n_ports` — a
        wrong number with a right-looking shape. `wire_loss_power` refuses
        the same input already (its `repeat`/`wing_sigma` product cannot
        line up), so this only spells that shared contract out loud rather
        than narrowing it. The check runs BEFORE the no-loads early return:
        a contract that holds only when the solver happens to be loaded is
        not a contract a caller can rely on.

        **The power-budget closure, and where razor's own gap lives.**
        Razor's rows are PATH integrals, not Galerkin projections (module
        docstring), so `Re(Z)` does not exactly satisfy the power/reaction
        theorem a Galerkin `Z` would — #427's gate notes measured this as a
        gap between ΔR_in from RE-SOLVING a loaded system and ΔR predicted
        by `wire_loss_power`/|I_feed|², ~5.4 % on all four solvers; this
        readout's own test module reproduces that comparison at 5.36 % on
        the copper (`wire_conductivity=5.8e7`) 24-segment dipole gate
        `tests/test_razor_loading.py` already runs. That gap belongs to
        `wire_loss_power` and a RE-SOLVE, not to this readout: because a
        lumped stamp is Sherman-Morrison-exact on one diagonal entry,
        `P_in − P_wire − P_lumped` equals `½·Re(I^H · Z_rad · I)` — the
        bilinear reaction power of the UNLOADED radiation matrix at the
        LOADED current — to 1.5e-15 relative, an algebraic identity
        (`Z = Z_rad + diag(Z_L)` exactly) that contributes zero error of
        its own. Whether that
        number equals the true Poynting-flux radiated power cannot be
        checked directly — momwire keeps no far-field power reader — but
        it inherits whatever razor-vs-Galerkin gap `Z_rad` itself carries,
        at roughly the distributed term's few-percent scale, not this
        readout's.
        """
        alpha = np.asarray(coeffs, dtype=np.complex128)
        if alpha.ndim != 1:
            raise ValueError(
                "lumped_load_power: coeffs must be ONE solve's coefficient "
                f"vector, got shape {alpha.shape} — a multi-port block would "
                "sum each load's watts across every excitation. Read one "
                "column at a time."
            )
        if not self.lumped_loads:
            return 0.0, np.zeros(0, dtype=np.float64)
        geom = self._build_geometry()
        idx, z_l = _wire_loading.loading_for(self, self.omega, geom).lumped
        i_knot = alpha[: geom["n_basis_total"]][idx]
        per_load = 0.5 * np.real(z_l) * np.abs(i_knot) ** 2
        return float(per_load.sum()), per_load

    def compute_impedance_swept(self, k_array):
        """Drive-point impedance(s) over a batch of wavenumbers.

        Same return convention as `compute_impedance` (scalar per k with
        one feed, `(n_k, n_feeds)` with several), stacked over `k_array`
        into shape `(n_k,)` / `(n_k, n_feeds)`. Shares every k-independent
        piece of the fill — geometry, the wing/path stencils, and the
        closed-form static segment moments — across the sweep via
        `_assemble_Z_prepare`, so only the smooth kernel remainder and the
        jωμ / 1/jωε prefactors (ω = c·k) are recomputed per k; each k still
        gets its own dense solve.
        """
        k_array = np.asarray(k_array, dtype=np.float64)
        geom = self._build_geometry()
        self._refuse_coincident_segments(geom)
        self._checkpoint()
        prepared = self._assemble_Z_prepare(geom)

        cols = self._port_columns(geom)
        rhs = cols @ self._port_voltages()

        voltages = self._port_voltages()
        feed_currents = np.empty((k_array.shape[0], cols.shape[1]), dtype=np.complex128)
        for i, k in enumerate(k_array):
            self._checkpoint()
            k = float(k)
            Z = self._assemble_Z_from_prepared(geom, prepared, k, self.c * k)
            coeffs = scipy.linalg.solve(Z, rhs)
            feed_currents[i] = cols.T @ coeffs

        z_per_feed = voltages[None, :] / feed_currents
        return z_per_feed[:, 0] if len(self.feeds) == 1 else z_per_feed

    def _port_solutions_swept(self, k_array):
        """Per-k `PortSolution` generator behind `compute_y_matrix_swept` and
        `compute_port_solution_swept` (momwire#252, #429 rank-9).

        Shares the k-independent fill work the same way
        `compute_impedance_swept` does: geometry, the wing/path stencils and
        the closed-form static segment moments are built ONCE via
        `_assemble_Z_prepare` and replayed at every k through
        `compute_port_solution(prepared=...)`, which is exactly the
        composition `_assemble_Z` documents — prepare once, finish per k —
        so this loop cannot diverge from calling `compute_port_solution()`
        fresh at every point. The old hand-rolled `compute_y_matrix_swept`
        this replaces did the identical fill-and-solve inline; routing it
        through the single-k entry point instead means the swept Y and the
        stacked single-k Y are the SAME code path evaluated in a loop, not
        two copies of the port algebra that could drift apart (the
        sharing-audit #429 rank-9 finding this closes).

        `self.k` is the mutated frequency `compute_port_solution` reads
        implicitly (matching `compute_impedance` / `compute_y_matrix`'s own
        convention — it derives omega as `self.c * self.k`, so `self.omega`
        never has to be read); `_set_k` / `_k_restored`
        (`_SweptPortSolutions`) rebind the whole frequency triple per k and
        put it back — including on an exception or an abandoned generator —
        so a caller who only consumes part of the sweep leaves the instance
        exactly as it found it.
        """
        k_array = np.asarray(k_array, dtype=np.float64)
        geom = self._build_geometry()
        self._refuse_coincident_segments(geom)
        self._checkpoint()
        prepared = self._assemble_Z_prepare(geom)
        with self._k_restored():
            for kk in k_array:
                self._checkpoint()
                self._set_k(float(kk))
                yield self.compute_port_solution(prepared=prepared)
