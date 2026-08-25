# Ground CONTACT over a FINITE ground — a design study

**Status:** design study, for maintainer review before any implementation.
Nothing here is a plan of record. No code beyond probe scripts was written.
Written 2026-08-18 against momwire v0.32.0, in the shape of
`solver-architecture.md` (the momwire#376 study): survey, measure, propose,
and be explicit about what is not known.

**Scope.** A wire END lying in the ground plane, over a ground that is not a
perfect conductor. The ground-mounted vertical over real earth — the single
most-requested antenna class no momwire row serves with a gate behind it.

**On the licensed binary.** Every NEC-5 number in this document is a number
the binary PRINTED. No NEC-5 source was read, quoted, or reasoned from.
Where this document says what NEC-5 "does", it means what NEC-5 printed, and
says so.

**On the probe scripts.** The suites cited as `sN_*.py` are study artifacts,
not repo code; they live in the session scratch directory
(`.../scratchpad/study282/`, with every deck and printout under `decks/`) and
are inventoried in the appendix. If any of this becomes a plan of record, the
lane script Stage 1 proposes is what lands in `scripts/`.

---

## 0. Recommendation, in one page

**The study's first finding is that the question in the issue title is not
the question the code is asking.** Ground contact over a finite ground is
not uniformly refused: it is *served, ungated, on three solvers* and refused
on two, and the two refusals cite each other's mechanism.

| solver | contact over PEC | contact over refl-coef | contact over sommerfeld |
|---|---|---|---|
| `BSplineSolver` (+ `HMatrixSolver`, `ArrayBlockSolver`) | served | **served** | **served** |
| `SinusoidalSolver` / `SinusoidalGalerkinSolver` | served | **served**, with the #282 charge correction | **served**, with the #282 charge correction |
| `RazorSolver` | served | refused (`_CONTACT_OVER_FINITE_REFUSAL`) | refused (same) |
| `PulseSolver` | refused | refused | refused |

> **The table above is the state this study OPENED with, 2026-08-18, and is
> kept as written.** Two decisions have since moved it: D3 refused the whole
> refl-coef column on every trunk that served it (stage 1), and D6 served
> razor's sommerfeld cell (stage 3, momwire#624). The current state is that
> every solver but `PulseSolver` serves contact over sommerfeld, no solver
> serves it over refl-coef, and `RazorSolver`'s row no longer differs from
> `BSplineSolver`'s.

So the honest framing of momwire#282 today is **not** "add a capability". It
is:

1. **A served capability with no oracle behind it.** `BSplineSolver` has
   solved a base-fed vertical standing in lossy earth since momwire#151, and
   nothing in the tree has ever compared that answer to a reference engine.
   The gates that exist (`test_282_*`, `test_g16c`) are *self-consistency*
   gates: they check that the answer stops walking and that the two trunks
   agree with each other. Both trunks agreeing on a wrong answer would pass
   every one of them.
2. **A measurable error inside that served capability.** Measured here
   against the binary's printed numbers (§3.5): the ground-induced impedance
   shift at contact agrees to 0.005 Ω over "very good" ground and to
   0.31 Ω over sea water, but misses by **1.2 Ω over average soil and
   3.3 Ω over poor soil**, and the miss *grows with mesh refinement* — it is
   a limit difference, not a discretization one.
3. **A refusal whose stated mechanism does not survive contact with the
   record.** Razor's refusal says the fold "hard-codes image coefficient 1,
   i.e. PEC". But `_ground_spec.ground_config` assigns `image_coefficient=1`
   to the *reflection-coefficient* ground too (§4.1), the binary's own
   printed environment banner says it continues the contact current to the
   image *regardless of the ground constants* (§3.1), and `BSplineSolver`
   converges at contact over both finite grounds with coefficient 1 in
   place. Coefficient 1 is not the defect. §4.3 identifies what razor's
   actual defect is, and it is a different thing wearing the same name.

**The recommendation** — model (b), the lumped base termination, is **not**
the first unit, and neither is (c). The recommended ordering is:

> **Stage 1 — gate what is already shipped.** Build the NEC-5 contact lane
> (difference-of-columns against printed impedances) and let it decide, per
> ground, whether the served row keeps serving. On today's measurements that
> means: **refuse `refl-coef` at contact** (it sits ~27 Ω from momwire's own
> Sommerfeld answer on the same deck, §3.6), and pin `sommerfeld` at contact
> at its measured residual, which is honest for high-conductivity grounds
> and openly loose for low-permittivity ones.
>
> **Stage 2 — close, or explain, the low-ε_r gap.** The 2.6–3.3 Ω under-
> prediction of ground-loss resistance over poor soil is the study's one
> unexplained number. §5.4 names three candidate causes and the experiment
> that separates them. This is where model (b) becomes relevant — but as a
> *diagnosis* of the missing ohms, not as a shipping model.
>
> **Stage 3 — razor's grounded tent over a finite ground.** The actual
> standing refusal. §4.3 argues the fix is *not* to weight the image wing
> (that would make it wrong, not right) but to replace razor's T2 plane-
> reference drop, which is where the PEC assumption really lives.
>
> **Model (b) — the lumped base termination — is recommended as a
> documented antennaknobs-level composition, not as a momwire ground.**
> §2.3 explains why: it is not a MoM model at all, it has no place to live
> in either trunk's basis, and its one real virtue (matching the ARRL
> ground-system tables that hams actually use) is a station-modelling
> concern that antennaknobs already has the vocabulary for.

**Maintainer decision points** are collected in §7. The three that gate
everything else: (D1) does an ungated served capability get gated or
withdrawn; (D2) what bar shape the contact lane carries; (D3) whether the
refl-coef contact row is refused, warned, or left alone.

---

## 1. What momwire#282 actually is, in the record

### 1.1 The issue, and what closed it

momwire#282 was filed against `SinusoidalSolver`: a base-fed vertical
contacting the plane over lossy earth *diverged under mesh refinement*, with
the extended kernel off. The recorded table:

| NS | bspline refl soil | sinusoidal refl soil | sinusoidal somm soil |
|---|---|---|---|
| 11 | 28.23+15.44j | 66.84−392.22j | 80.66−107.05j |
| 21 | 28.74+15.82j | 91.04−702.06j | 101.10−207.36j |
| 41 | 29.22+16.26j | 112.75−1019.03j | 121.23−314.69j |
| 61 | 29.48+16.54j | 109.14−1054.33j | 122.34−335.55j |

Read that table again with the trunks in mind: **the bspline column is
already converged.** #282 was never a statement that momwire could not solve
a grounded vertical over earth; it was a statement that one of two
formulations could not.

What closed it is a **rank-one column correction** on the sinusoidal trunk
(`sinusoidal.py:3572`, `_contact_charge_correction`; the Galerkin twin at
`sinusoidal_galerkin.py:3106`), subtracting the field of the spurious point
charge that the direct-field formulation leaves at the contact node. Its own
docstring names bspline as the reference it restores agreement with
(`sinusoidal.py:3341-3345`):

> The mixed-potential solvers never had it: `BSplineSolver` builds its charge
> term from the basis DERIVATIVE over the support, so a ground-contact basis
> simply has no end charge, which is why it converges here and is the
> reference this correction restores agreement with.

### 1.2 What the recorded mechanism note gets wrong

The maintainer's mechanism note for #282 reads: *"#151 fold hard-codes image
coeff 1 (PEC); spurious contact charge; fix = unmerge fold, keep free/image
tensors separate."* Three corrections, in increasing order of consequence:

1. **There is no per-basis fold to unmerge in `bspline.py`.** The image
   enters once, globally, after the free-space operator is complete
   (`bspline.py:3699-3739`): one `Z -= (image assembly)` against the *same*
   `supp_seg` and `polys` the free assembly used. "Image coefficient 1" is
   therefore a **structural identification** — the mirrored half of every
   basis is driven by the same unknown as its real half — not a literal
   multiply that could be un-hard-coded.
2. **Free and image tensors are already separate**, and the weighting hook
   already exists. `_image_Z_weighted` (`bspline.py:1691`) applies per-pair
   `(w_A, w_Φ)` from `PotentialGround.weight_windows` to the image block
   alone. The reflection-coefficient ground *is* "the fold with a weighted
   image", shipped since momwire#151/#153.
3. **The fix that shipped was not this fix.** It was a rank-one post-
   assembly subtraction on the other trunk, and the mixed-potential trunk
   needed nothing.

The note is best read as a *razor-era* note: razor's refusal prose was
written in that language (`razor.py:305-320`) and inherits the same framing.
§4.3 argues that even for razor the framing is off by one layer.

### 1.3 The standing limitation the record does record honestly

`tests/test_ground_junction.py:607`,
`test_291_contact_over_finite_ground_still_diverges_slowly`, asserts that the
*sinusoidal* contact answer still walks over NS 81→641 while pinning bspline
as the converged reference (`drift_b < 0.5` between NS=81 and NS=321). So the
tree already knows one trunk is not finished at contact. What it does not
know is whether the trunk it treats as finished is *right*.

---

## 2. The physics of a wire–ground contact over lossy earth

### 2.1 What actually happens at the junction

A conductor ends at `z = 0` on a half-space of complex permittivity
ε̃ = ε_r − jσ/(ωε₀). The current I₀ arriving at the end does not stop; it
crosses the interface as conduction-plus-displacement current and spreads
into the earth. Three length scales govern the near field, and a thin-wire
MoM code resolves none of them:

* **The electrode scale, ~a.** Immediately around the contact the problem is
  the quasi-static spreading-current problem of a small electrode in a
  conducting medium. For a hemispherical electrode of radius a the spreading
  admittance is `Y = 2π(σ + jωε)a`. At 14 MHz over average soil
  (13, 0.005 S/m), σ + jωε = 0.005 + j0.0101 S/m, so a 5 mm rod tip alone
  presents |Z| ≈ 2.8 kΩ. This is not the impedance a NEC-class code returns,
  and it should not be: a real ground connection is a stake or a radial
  system, not a wire tip, and the electrode geometry lives entirely below any
  mesh a wire code will ever build.
* **The mesh scale, Δ.** What a MoM code actually models is the current
  leaving the *lowest segment*, spread over a region of order Δ. As Δ→0 that
  region shrinks toward the electrode scale but the model does not follow it
  there — the basis has no radial spreading shape, only an axial one.
* **The skin/wavelength scale in the earth**, |1/k₁| = λ/(2π√|ε̃|). At
  14 MHz over average soil that is 0.94 m; over poor soil (5, 0.001), 1.5 m.
  This is the scale over which the earth's own field structure — including
  the lateral wave along the interface — varies.

**Consequence, and it is the single most important physical statement in this
study:** *the contact impedance of a wire ending on earth is not determined by
the wire model.* Every code that solves this problem is making a modelling
choice about what happens below the mesh scale, and the choices differ. A
gate that demands two codes agree at contact is demanding they made the same
choice, not that either is right. §5 is written around that constraint.

### 2.2 Where "image continuation" breaks, and what replaces it

Over a perfect conductor, the boundary condition at the plane makes a
vertical wire's grounded end *exactly* the mid-point of the wire and its
mirror image: the monopole IS half a dipole, the tangential E vanishes on the
plane, and the current is continuous with coefficient exactly 1. No
approximation anywhere.

Over a finite ground, three separate things that coincide at PEC come apart:

| at PEC | over a finite ground |
|---|---|
| the ground's field = the exact image's field | the ground's field ≠ any image's field; the exact image is the leading term of a decomposition with a remainder |
| the plane is an equipotential (Φ = 0) | the plane is **not** an equipotential |
| the physical current continues into the image with coefficient 1 | the physical current continues into the earth — as a spreading current, not as a filament — but it **still continues**; nothing about a lossy ground makes current stop at the interface |

The third row is where the standing refusal and the record both slip. **The
image coefficient in the GROUND MODEL (how the ground's field is
represented) and the continuation coefficient in the BASIS (where the
physical current goes) are different numbers**, and only the first one
depends on ε̃. Charge conservation fixes the second at 1: whatever current
arrives at the node leaves it. `_ground_spec.py`'s own table says the same
thing in code — `image_coefficient` is **1** for the reflection-coefficient
ground, which is not PEC, and the Fresnel physics arrives separately as
per-pair `(w_A, w_Φ)` weights.

The binary agrees, in printed output. §3.1.

### 2.3 The four modelling traditions

#### (a) Exact-image continuation scaled by a coefficient c ≠ 1

*What naive un-hard-coding would give.* Let the contact basis continue into
the image with amplitude c (the Fresnel ρ_v, or the Sommerfeld C₂). Then the
current arriving at the node is I₀ and the current leaving is c·I₀, so
`(1−c)·I₀` has nowhere to go and appears as a **point charge at the contact
node**:

```
    Q = (1 − c) · I₀ / (jω)
```

Its potential at the nearest collocation point, a distance ≈ Δ/2 away, gives
an impedance contribution of order

```
    |ΔZ| ≈ |1 − c| / (ω · 4πε₀ · Δ/2)         [ohms, Δ in metres]
```

which **diverges like 1/Δ under mesh refinement**. This is exactly momwire#282's
recorded pathology, and it is exactly what `sinusoidal.py:3327-3345`
describes. The estimate was checked here by switching the #282 correction off
(`scratchpad/study282/s17_scaling.py`; λ/4 vertical, 14 MHz, a = 5 mm,
refl-coef over (13, 0.005), ρ_v(normal) = 0.5891 − 0.0756j, |1−ρ_v| = 0.4178):

| N | Δ (m) | Z with correction | Z with correction OFF | \|off − on\| | 1/Δ estimate |
|---|---|---|---|---|---|
| 11 | 0.4867 | 32.852+23.152j | 105.913−392.662j | 422.2 | 175.4 |
| 21 | 0.2549 | 32.132+22.387j | 174.243−775.454j | 810.4 | 334.9 |
| 41 | 0.1306 | 31.195+21.445j | 310.069−1532.388j | 1578.7 | 653.9 |
| 81 | 0.0661 | 29.950+20.242j | 571.199−2986.020j | 3054.6 | 1291.8 |
| 161 | 0.0333 | 28.279+18.667j | 1017.456−5486.047j | 5592.9 | 2567.6 |

The measured term tracks 1/Δ to within 8 % per rung and sits a near-constant
2.2–2.4× above the crude single-collocation-point estimate — the factor being
the rest of the structure the point charge also illuminates. The N = 11
"off" value (105.9 − 392.7j) reproduces momwire#282's recorded 66.84 −
392.22j in its reactance to 0.44 Ω. The two decks are not identical (the
recorded one uses a = 0.02), so the closeness of that particular match is
partly luck — the 1/Δ scaling law, not the coincidence, is the point.

**Verdict on (a): not a candidate. It is the defect, quantified.** Model (a)
is what you get by taking the ground model's image coefficient and applying
it to the basis, and it violates charge conservation by construction. This
study's clearest single conclusion is that **the right image coefficient for
the contact BASIS is 1, over every ground.**

* *models:* nothing. It is a bookkeeping error with a physical-looking name.
* *ignores:* charge conservation.
* *cost:* negative — it is the thing to avoid.
* *oracle:* the 1/Δ growth above is itself the detector; any contact
  formulation whose answer walks like 1/Δ has this bug.

#### (b) A lumped ground-loss termination at the base

*The classic engineering model.* Model the antenna over a perfect ground,
then add a series resistance R_g at the base representing the ground system
(the ARRL/Sevick tables: ~2 Ω for 120 buried radials, 10–20 Ω for four
radials, 30–100 Ω for a single stake in poor soil). Feed impedance becomes
`Z_pec + R_g`, and efficiency `R_rad/(R_rad + R_g)`.

* *models:* the aggregate near-field loss of a real ground system, which is
  precisely the quantity a builder can change (add radials) and precisely the
  quantity full-wave codes model worst, because it lives below the mesh
  (§2.1).
* *ignores:* everything else — the reactance shift, the pattern, the ground's
  effect on the current distribution, and any frequency dependence beyond
  what the tabulated R_g carries.
* *where it would sit in a MoM basis:* **nowhere.** This is the finding that
  decides its place. A series R at the base is a load on a *port*, and the
  grounded end is not a port — `bspline.py:1167-1171` already refuses a
  grounded junction as a junction port ("a grounded node's voltage is pinned
  by the ground image, so it cannot also be a driven port") and
  `bspline.py:3771-3775` refuses a node gap there ("a series gap between a
  wire and the ground stake is not supported"). To make (b) a momwire ground
  you would have to unpin the grounded node's voltage, i.e. introduce the
  very unknown those two refusals exist to deny. The cheap alternative — a
  `LD 4`-style series load on the lowest segment — is not the same thing and
  is measurably not the same thing on a refined mesh, because the lowest
  segment shrinks.
* *cost per trunk:* as a momwire ground, high and structural (a new unknown
  class at grounded nodes, both trunks). As an antennaknobs composition
  (PEC solve + `Z += R_g` on the port + efficiency bookkeeping), **an
  afternoon and zero momwire edits** — the same shape as
  `solver-architecture.md` §6.1's verdict on the MININEC ground.
* *oracle:* EZNEC/4nec2 parity for the composed answer; the ARRL tables for
  R_g itself. No NEC oracle, because no NEC has this model.

**Verdict on (b): recommended, as an antennaknobs composition, and
explicitly not as a momwire ground.** It is the model most users actually
want and it proves nothing about momwire's fill. It also has genuine
diagnostic value here: §3.5's missing 2.6 Ω of resistance over poor soil is
exactly the magnitude of a base-loss term, which is a lead, not a
coincidence, and §5.4 turns it into an experiment.

#### (c) The rigorous route: the half-space formulation's own contact

*What the theory requires.* The Sommerfeld half-space Green's function is
exact for sources above the interface, and momwire composes it as
`C₂·(exact image) + Q`, with `Q` the smooth remainder
(`_potential_ground.Remainder`, theory manual eqs 143–147). Two questions
matter at contact:

**Does the remainder machinery extend to a basis touching the plane?** In
code, yes, and deliberately (`bspline.py:1784-1794`):

> Touching (`zmin == 0`) is allowed since #151: the ground-junction basis
> handles contact, and the remainder quadrature samples Gauss nodes strictly
> interior to segments, so `z+z' > 0` holds even for a wire ending in the
> plane.

That is true and it is not the whole story. The remainder's interpolation
surfaces are tabulated in `(R₁, θ)` with `R₁ = |r − r'_image|` and
`θ = atan2(z+z', ρ)`. **Ground contact is the only geometry that drives
`R₁ → 0`**: for a clear deck the lowest source and the highest image are
separated by twice the clearance, so `R₁` has a floor; at contact the near-
diagonal blocks query `R₁ ~ Δ`, which shrinks with the mesh. §3.7 measures
what the grid does down there.

**What singular behaviour appears?** The Sommerfeld integrands converge
because of an `exp(−λ(z+z'))` factor. At contact that factor → 1 for the
nearest pairs, so the spectral integrals lose their decay exactly where the
mesh is finest. This is the classical difficulty of a source *on* an
interface, and it is a property of the formulation, not of momwire's
implementation of it. Everything else about contact — the basis, the fold,
the coefficient — is easy by comparison.

* *models:* the earth exactly, as a homogeneous half-space, given a
  filamentary current continuation across the interface.
* *ignores:* the sub-mesh electrode geometry (§2.1); the radial spreading of
  the current, which it replaces with an axial continuation; earth
  inhomogeneity and stratification.
* *cost per trunk:* **on the mixed-potential trunk, already paid** — bspline
  serves it today. On razor, §4.3. On the direct-field trunk, paid via the
  #282 correction, with a residual walk (`test_291_*`).
* *oracle:* NEC-5's printed impedances, which are a rigorous half-space
  formulation applied to the same class of problem. §3 and §5.

**Verdict on (c): this is the shipping model, and it already ships. The work
is gating it, not building it.**

#### (d) Radial-screen approximations (GN NRADL-style)

*Coefficient-level.* NEC-2's `NRADL` models a buried radial screen as a
surface impedance in parallel with the earth, modifying the reflection
coefficients — `solver-architecture.md` §6.1 nominates it as the
architecture's stretch test precisely because its physics lands in
`_ground_refl.py`'s coefficient layer and should reach every solver for free.

Contact interacts with screens in a specific and awkward way, and
`field-ground-interface.md:104-111` already records it:

> the **#282 contact-charge correction** calls `_ground_refl.fresnel_rho`
> directly on both solvers … A screen deck with a wire END IN THE PLANE
> would take the bare earth's ρ_v/ρ_h there … it is a wrong answer rather
> than a missing feature, and whoever lands the screen has to route those
> coefficients through the ground or refuse ground contacts under it.

That prescription is written for the direct-field trunk. Note what it means
for the mixed-potential trunk: **nothing**, because bspline has no contact-
specific coefficient call to route — its contact basis is an ordinary basis
and the screen would reach it through the ordinary weights. The screen is
therefore *easier* at contact on the trunk that serves contact, which is the
opposite of the note's implication.

The genuinely hard part is physical, not architectural: a surface-impedance
screen is a **far-field/reflection** device. It is derived from a plane wave
striking the interface. A wire in contact with the screen is in its near
field, where a surface impedance is not the right object at all — the real
answer is the radials as *wires*, which momwire can already model
explicitly, at N-radial cost.

* *models:* the effect of a screen on reflected fields.
* *ignores:* the near-field current sharing between the vertical and the
  radials — i.e. the entire mechanism by which radials help a ground-mounted
  vertical.
* *cost per trunk:* per `solver-architecture.md` §6.1, zero per-solver work
  if the architecture holds; the contact interaction is one direct
  `fresnel_rho` call to reroute on the direct-field trunk, and nothing on
  the mixed-potential trunk.
* *oracle:* **not NEC-5, on this dialect.** §3.8 records the probe: NEC-5's
  `GN` card has no `NRADL` field, and NEC-2's radial parameters land silently
  on the permeability fields.

**Verdict on (d): out of scope for #282, and worth saying so in the
refusal.** A user who wants radials modelled correctly at contact should be
pointed at modelling the radials as wires. If (d) ships for reflected fields,
it should refuse or warn at contact rather than silently apply a plane-wave
surface impedance to a near-field junction.

---

## 3. What the licensed binary prints

All decks and printouts are kept beside the probe scripts (see the header
note). Unless noted: base-fed λ/4 vertical, 14 MHz
(λ = 21.414 m), height 5.3535 m, radius 5 mm, `GE 1 0`, source at the
grounded base knot. NEC-5's ground cards, as our own adapter records them:
`GN 1` is PEC, `GN 0 … NOFILE` is its native Sommerfeld solution (NEC-5 has
no reflection-coefficient ground at all).

### 3.1 It accepts contact, and it says what it does there

The environment banner the binary prints for **every** grounded deck,
before any `GN` card is read — identically for `GN 1` and for `GN 0` with
lossy constants:

```
   GROUND PLANE SPECIFIED.

   WHERE WIRE ENDS TOUCH GROUND, CURRENT WILL BE INTERPOLATED TO IMAGE IN GROUND PLANE.
```

Two things follow from that printed line, and only from it:

1. **Contact is a first-class, documented case**, not a tolerated one.
2. **The continuation rule is announced independently of the ground
   constants.** This is the printed-output corroboration of §2.2: the
   reference engine's contact basis does not carry ε̃, and the finite-ground
   physics reaches it through the kernel. It is direct evidence against the
   premise of razor's refusal.

A second printed fact pins the same point from the other side. A voltage
source at a **free** wire end, in free space, is rejected:

```
 SORVT1: ERROR - Voltage source specified where there is no basis function.  IELEM, INODE =        1   2
```

while the identical source at a **grounded** end solves. The grounded end has
a basis function; the free end does not. That is exactly momwire#151's
`"gnd"`-vs-`"free"` distinction (`bspline.py:1261-1283`), independently
observed.

### 3.2 Ground constants × mesh, at contact

`scratchpad/study282/s2_grounds.py`. Printed impedance, Ω:

| ground | N=11 | N=21 | N=41 | N=61 |
|---|---|---|---|---|
| PEC (`GN 1`) | 39.882+19.260j | 40.379+21.087j | 40.643+22.018j | 40.745+22.361j |
| sea 81 / 5.0 | 41.843+20.315j | 42.505+22.422j | 42.858+23.573j | 42.988+24.010j |
| v.good 20 / 0.0303 | 51.936+22.261j | 52.662+24.677j | 53.031+25.926j | 53.170+26.376j |
| average 13 / 0.005 | 51.286+19.207j | 52.006+21.505j | 52.372+22.685j | 52.508+23.110j |
| poor 5 / 0.001 | 43.571+15.113j | 44.244+17.169j | 44.587+18.222j | 44.713+18.607j |
| v.poor 3 / 0.0001 | 33.396+14.159j | 33.942+16.017j | 34.221+16.967j | 34.325+17.320j |
| ε_r=2, σ=1e−5 | 23.730+11.044j | 24.198+12.702j | 24.438+13.547j | 24.526+13.866j |
| ε_r=1.05, σ=1e−9 | 4.678+2.187j | 5.024+3.455j | 5.204+4.096j | 5.271+4.347j |
| ε_r=1, σ=0 | 3.091+1.348j | 3.427+2.584j | 3.602+3.209j | 3.668+3.455j |

**Findings.**

* Every row converges cleanly, at the usual O(1/N) walk. There is no analogue
  of momwire#282's divergence anywhere in the reference engine's contact
  behaviour, over any ground.
* The resistance is **non-monotone in ground quality**: PEC 40.4 → sea 42.5 →
  very good 52.7 → average 52.0 → poor 44.2 → very poor 33.9. The classic
  ground-loss hump, peaking around ε_r 13–20, and worth knowing before
  designing any gate that assumes monotonicity.
* **The printed POWER BUDGET carries no ground loss.** At every finite
  ground, `RADIATED POWER = INPUT POWER`, `WIRE LOSS = 0`, `EFFICIENCY =
  100.00 PERCENT`. An efficiency-based gate would have to come from an `RP`
  run's `AVERAGE POWER GAIN`, not from the plain `XQ` budget.

A second geometry, a grounded inverted-L (3 m vertical + 6 m top wire),
converges the same way (`s5_limits.py`, S6): PEC 264.2+910.8j → 294.4+956.9j
→ 313.6+983.9j over N = 12/24/48; average soil 358.6+796.1j → 391.9+823.4j →
412.0+837.7j; poor soil 337.4+754.8j → 363.8+776.0j → 378.6+786.2j.

### 3.3 The ε̃ → ∞ limit: contact recovers PEC, at C₂'s rate

`s5_limits.py` S5(i). ε_r = 13 fixed, σ swept, N = 41 contact; PEC reference
`GN 1` = 40.643 + 22.018j.

| σ (S/m) | printed Z | \|Z − Z_PEC\| |
|---|---|---|
| 1e−2 | 53.807+23.269j | 13.223 |
| 1e−1 | 49.847+27.114j | 10.521 |
| 1e0 | 44.750+24.916j | 5.027 |
| 1e1 | 42.308+23.179j | 2.030 |
| 1e2 | 41.249+22.411j | 0.722 |
| 1e3 | 40.846+22.142j | 0.238 |
| 1e4 | 40.708+22.055j | 0.0748 |
| 1e6 | 40.649+22.019j | 0.0061 |
| 1e8 | 40.643+22.016j | 0.0020 |

Single-decade ratios from σ = 1 upward: 2.48, 2.81, 3.03, 3.18 → √10 from
below, then a floor at ≈ 0.002 Ω which is the printed precision. **That is
C₂'s rate** — the same rate `solver-architecture.md` §6.5 measured for the
clearance case on razor. So the analytic PEC limit is available at contact,
it is clean in the reference engine, and its shape is known: √10 per decade
of σ down to a 0.002 Ω floor. §5.2 makes a gate of it, and §3.7 records that
momwire fails it.

### 3.4 The ε̃ → 1 limit: contact does not become free space, and both codes agree it doesn't

`s5_limits.py` S5(ii), σ = 0, N = 41 contact:

| ε_r | printed Z |
|---|---|
| 10 | 49.428+23.552j |
| 4 | 39.172+19.964j |
| 2 | 24.360+13.714j |
| 1.5 | 16.345+9.904j |
| 1.2 | 9.503+6.411j |
| 1.05 | 5.204+4.096j |
| 1.01 | 3.930+3.391j |
| 1.001 | 3.635+3.227j |
| **1.0** | **3.602+3.209j** |

The sequence is smooth and converges to a definite finite value. It is **not**
free space: the same wire in free space, fed at the first interior knot,
prints 11.247 − 4917.8j.

The reason is physical and it answers the study's brief directly. Asking
"what should ε̃ → 1 recover at contact?" presumes there is a well-posed
free-space problem to recover. There is not: **the ground is the current's
return path.** Remove the ground and the base feed has nothing to push
against; the wire end reverts to a free end, which (per §3.1's printed error)
has no basis function at all. The ε̃ → 1 contact limit is a limit of a family
whose endpoint is a different problem, and what the codes print there is the
numerical residue of that degeneracy.

The surprise is that **momwire prints almost the same residue.**
`BSplineSolver(degree=2)`, N = 41, contact, sommerfeld, ε_r → 1 at σ = 0
(`s10_mw_limits.py`): 47.574+24.213j, 35.790+19.376j, 20.979+12.590j,
13.872+9.132j, 8.273+6.326j, 4.964+4.636j, 4.015+4.147j, **3.775+4.024j** at
ε_r = 1.0001 — against the binary's 3.602+3.209j. Two independent
formulations converge on the same degenerate object to 0.83 Ω. That is worth
recording, but §5.2 declines to make a gate of it: agreeing about a
degenerate limit is not evidence of agreeing about physics.

### 3.5 The headline measurement: momwire vs the binary at contact

The comparison that matters is the **ground-induced shift**
`δ = Z(soil) − Z(PEC)` at matched N — the difference-of-columns pattern,
which cancels the formulation's own discretization offset (at PEC contact,
N = 41: binary 40.643+22.018j, bspline 40.662+23.278j — 1.26 Ω apart in X,
all of it basis difference). `s15_columns.py`, `BSplineSolver(degree=2,
feed_model="segment")`, `ground_model="sommerfeld"`:

| soil | N | δ momwire | δ binary (printed) | \|difference\| |
|---|---|---|---|---|
| sea 81/5.0 | 11 | 2.119+1.741j | 1.961+1.055j | 0.704 |
| | 21 | 2.131+1.860j | 2.126+1.335j | 0.525 |
| | 41 | 2.146+1.921j | 2.215+1.555j | 0.372 |
| | 61 | 2.154+1.941j | 2.243+1.649j | **0.306** |
| v.good 20/0.0303 | 11 | 12.243+3.960j | 12.054+3.001j | 0.977 |
| | 21 | 12.310+3.939j | 12.283+3.590j | 0.350 |
| | 41 | 12.387+3.913j | 12.388+3.908j | **0.005** |
| | 61 | 12.424+3.898j | 12.425+4.015j | 0.117 |
| average 13/0.005 | 11 | 10.734−0.007j | 11.404−0.053j | 0.672 |
| | 21 | 10.782−0.044j | 11.627+0.418j | 0.963 |
| | 41 | 10.840−0.082j | 11.729+0.667j | 1.163 |
| | 61 | 10.867−0.101j | 11.763+0.749j | **1.236** |
| poor 5/0.001 | 11 | 1.238−5.725j | 3.689−4.147j | 2.915 |
| | 21 | 1.268−5.703j | 3.865−3.918j | 3.151 |
| | 41 | 1.288−5.701j | 3.944−3.796j | 3.269 |
| | 61 | 1.294−5.702j | 3.968−3.754j | **3.309** |

The sinusoidal trunk on the same decks: sea 0.013 → 0.136 (degrading),
very good 1.174 → 0.643, average 2.808 → 1.859, poor 6.640 → 4.376. Both
trunks are wrong in the same direction on poor soil, by different amounts.

**Findings.**

1. **Two rows converge onto the binary and two walk away from it.** Sea and
   very-good ground close with mesh (0.70→0.31, 0.98→0.005/0.12); average and
   poor *open* with mesh (0.67→1.24, 2.92→3.31, both saturating). A gap that
   grows with refinement and then flattens is a **difference of limits**, not
   a discretization artifact.
2. **The error tracks low ε_r, not low conductivity.** Sea (ε̃ = 81 − 6424j)
   and very good (20 − 38.9j) are fine; average (13 − 6.42j) misses by
   1.2 Ω; poor (5 − 1.284j) by 3.3 Ω. The ordering is by ε_r, and poor soil
   is the outlier at both frequencies tested.
3. **Nearly all the poor-soil miss is in RESISTANCE**: momwire's ground adds
   1.29 Ω of R where the binary adds 3.94 Ω. **momwire under-predicts the
   ground-loss resistance of a grounded vertical over poor soil by ~2.7 Ω** —
   ≈6 points of efficiency on a full-size 40 Ω monopole (90.1 % against
   96.5 %), and a much larger one on the
   short loaded verticals this class of user actually builds.
4. It is not a 14 MHz artifact. At **3.5 MHz** (`s16_freq.py`, quarter-wave
   = 21.414 m): average soil closes to 0.13 Ω (N=21) / 0.32 Ω (N=41), very
   good to 0.61/0.29 — but poor soil stays at **2.38/2.64 Ω**, still almost
   all in R (momwire δ = 10.599−7.272j vs printed 11.686−4.865j at N=41).

### 3.6 The refl-coef contact row is served and is ~27 Ω off

`s9_momwire.py`/`s10_mw_limits.py`, `BSplineSolver(degree=2)`, contact,
average soil (13, 0.005), N = 41:

| model | Z |
|---|---|
| PEC | 40.662+23.278j |
| **sommerfeld** | **51.502+23.196j** |
| **refl-coef** | **26.997+12.619j** |
| binary (`GN 0`, printed) | 52.372+22.685j |

The reflection-coefficient ground at contact is **26.7 Ω from momwire's own
Sommerfeld answer on the same deck, and 27.3 Ω from the binary's** — and it
is on the wrong side of PEC. The sinusoidal trunk's refl-coef contact answer
(31.195+21.445j at N=41, still walking down: 32.85 → 32.13 → 31.19 → 30.51
over N = 11…61) is differently wrong.

This is not new physics; momwire#153's validity window is stated in
`bspline.py:568-575` and names contact explicitly:

> Below ~0.1λ or for ground-touching wires, prefer `ground_model="sommerfeld"`
> (exact everywhere, contact-capable since #151)

But it is a docstring, not a refusal. **A user who writes
`ground_model="refl-coef"` — the default — on a ground-mounted vertical gets
a silently wrong answer today.** §6 Stage 1 proposes refusing it.

### 3.7 momwire fails the PEC limit at contact, and only at contact

`s11_floor.py`. `|Z(ε_r=13, σ) − Z_PEC|` for `BSplineSolver(degree=2)`; the
"clear" deck is the same wire with its base 2 m (0.093 λ) up, fed at
mid-height.

**`ground_model="sommerfeld"`:**

| deck | N | σ=1e2 | σ=1e4 | σ=1e6 | σ=1e8 |
|---|---|---|---|---|---|
| contact | 11 | 0.8646 | 0.3706 | 0.3355 | **0.3324** |
| contact | 21 | 0.9768 | 0.5129 | 0.4796 | **0.4766** |
| contact | 41 | 1.0376 | 0.5855 | 0.5529 | **0.5499** |
| contact | 81 | 1.0709 | 0.6242 | 0.5918 | **0.5888** |
| clear | 11 | 0.0280 | 0.0028 | 0.00028 | 0.00003 |
| clear | 21 | 0.0264 | 0.0026 | 0.00026 | 0.00003 |
| clear | 41 | 0.0254 | 0.0025 | 0.00025 | 0.00003 |
| clear | 81 | 0.0248 | 0.0025 | 0.00025 | 0.00002 |

**`ground_model="refl-coef"` (control — the folding ground):**

| deck | N | σ=1e2 | σ=1e4 | σ=1e6 | σ=1e8 |
|---|---|---|---|---|---|
| contact | 11 | 0.2315 | 0.0232 | 0.00232 | 0.00023 |
| contact | 41 | 0.2306 | 0.0231 | 0.00231 | 0.00023 |
| clear | 11 | 0.0369 | 0.0037 | 0.00037 | 0.00004 |
| clear | 41 | 0.0334 | 0.0033 | 0.00033 | 0.00003 |

**Findings.**

* The clear deck converges to PEC at the textbook rate (10× per decade of σ)
  and reaches 3e−5 Ω. So does refl-coef, at contact and clear alike.
* **The Sommerfeld ground at contact floors** at 0.33 / 0.48 / 0.55 / 0.59 Ω,
  and the floor *rises with mesh refinement*. It does not go away.
* It is not under-quadrature (`s12_knobs.py`): raising `n_qp_sommerfeld` 3 →
  5 → 8 → 12 moves the residual 0.553 → 0.600 → 0.618 → 0.625, i.e. the
  quadrature **converges to a wrong number**. It is not a radius effect
  either (a = 5/20/50 mm: 0.5529/0.5522/0.5512).
* **The direct-field trunk shows the same signature**: `SinusoidalSolver` at
  contact floors at 0.817/0.320/0.285 Ω over σ = 1e2/1e4/1e6 under
  sommerfeld, while its refl-coef control converges to 0.0014. The defect is
  in the shared composing-ground machinery, not in either basis.

**Mechanism, measured.** `s13_grid.py`/`s14_smallR.py` compare
`SommerfeldGrid.eval` against `iv_surfaces_direct` at θ = 45°, sweeping R₁:

| ε̃ | R₁ = 0.5λ | 0.1λ | 0.02λ | 0.006λ | 0.002λ | 2e−4 λ |
|---|---|---|---|---|---|---|
| 13 − 6.42j (average) | 1.8e−4 | 7.5e−5 | 7.3e−4 | 9.0e−4 | 3.9e−4 | 5.6e−5 |
| 5 − 1.284j (poor) | 1.5e−4 | 1.5e−4 | 5.3e−4 | 6.3e−4 | 3.9e−4 | 1.9e−4 |
| 13 − 6.42e9j (σ=1e6) | 2.3e−3 | 7.2e−3 | **4.9e+2** | **4.4e+2** | **1.9e+2** | **2.0e+1** |

(relative error on `IzV`; the other three surfaces behave the same.)

Two conclusions, and they point in opposite directions:

1. **The PEC-limit failure at contact is an INSTRUMENT defect, not a physics
   defect.** In the near-PEC regime the true surfaces are ~1/√ε̃ small
   (|IzV| = 4.2e−3 at R₁ = 0.02λ against 7.1 for average soil), and the
   grid's absolute interpolation error does not shrink with them, so the
   remainder Q stops vanishing. Contact is the only geometry that queries
   R₁ ≲ 0.02λ, which is why only contact sees it. **The ε̃ → ∞ gate at
   contact cannot be run through the Sommerfeld path as currently
   instrumented**, and a stage-1 plan that assumes it can will burn a week.
2. **It does NOT explain §3.5.** At real soils the grid holds 1e−4–1e−3
   relative accuracy all the way down to R₁ = 1e−5 λ. The 1.2 Ω and 3.3 Ω
   misses against the binary are somewhere else, and this study did not find
   them. §5.4 lists the candidates.

### 3.8 The lift-off ladder: contact is not the limit of clearance

`s4_liftoff.py`. Two families, both fed at the same knot (end 2 of segment 1)
so the only thing that changes is whether the lower end is grounded or free.

**(A) "clear"** — one wire from z = h to z = h + 5.3535, N = 21:

| h (m) | over PEC | over average soil |
|---|---|---|
| 0 (contact) | 40.113+21.523j | 51.594+22.210j |
| 1e−4 | 30.657−2687.1j | 58.220−2732.3j |
| 1e−3 | 30.634−2690.7j | 57.986−2735.5j |
| 1e−2 | 30.409−2723.2j | 55.845−2764.1j |
| 1e−1 | 28.726−2883.9j | 44.161−2906.3j |
| 5e−1 | 25.449−3011.4j | 30.681−3019.9j |

**(B) "stubbed"** — a one-segment contacting stub z = 0…h plus the radiator,
fed at the stub's grounded base:

| h (m) | over PEC | over average soil |
|---|---|---|
| 1e−4 | 40.539+20.988j | 52.138+21.367j |
| 1e−3 | 40.556+21.098j | 52.185+21.478j |
| 1e−2 | 40.751+22.166j | 52.447+22.632j |
| 1e−1 | 43.071+32.625j | 55.040+33.016j |
| 5e−1 | 55.034+80.251j | 67.875+78.746j |

**Findings.**

* **Ladder A answers the brief's question with a flat no.** Lifting the base
  by 0.1 mm — 5 millionths of a wavelength — moves the reactance by
  **−2750 Ω** and then holds it essentially constant over four decades of h.
  The limit `lim_{h→0} Z_clear(h)` exists and is perfectly well behaved; it
  is simply a *different antenna*. A free end carries no current, a grounded
  end carries maximum current, and no amount of closing the gap interpolates
  between them, because a thin-wire kernel has no tip capacitance to
  interpolate with. **"Contact = the lift-off limit" is dead as a serving
  strategy and dead as an oracle.**
* **Ladder B is the useful one.** With contact preserved, an arbitrarily
  short contacting segment reproduces the direct contact deck: at h = 0.1 mm
  (a 0.1 mm first segment!) PEC gives 40.539+20.988j against the plain
  contact deck's 40.379+21.087j at the same segment count elsewhere, and
  average soil 52.138+21.367j against 52.006+21.505j. **The reference
  engine's contact treatment is numerically stable down to degenerate
  contacting segments**, which makes it a usable own-code gate: §5.3.

### 3.9 Radial screens: not probeable on this dialect

`s5_limits.py` S7 / `s8_confirm.py`. NEC-2's `GN 2 NRADL … FRATI FRATIS`
spelling was tried four ways. Findings, from printed output alone:

* `GN 0`, `GN 2` and `GN 1` with I2 = 16 (NEC-2's `NRADL` slot) print
  **exactly** the same environment banner and the same impedance as with
  I2 = 0 — 52.006+21.505j for the finite ground, 40.379+21.087j for PEC.
  I2/I3/I4 are inert (checked to I2=16, I3=99, I4=7).
* No line containing "RADIAL" or "SCREEN" is printed by any variant.
* The apparent effect seen in the first pass was a **dialect trap**: NEC-2's
  radial-length/radius fields F3/F4 are NEC-5's permeability FMUR/FMUI.
  Writing NEC-2's `NRADL` parameters produces a silently *magnetic* ground —
  `GN 0 … 13.0 0.005 5.3535 0.001` prints 57.416+19.224j against the
  correct 52.006+21.505j, with the printed banner still reading
  `RELATIVE DIELECTRIC CONST.= 13.000 / CONDUCTIVITY= 5.000E-03`, i.e. the
  banner does not reveal the substitution.

**Consequence:** the radial screen has **no NEC-5 oracle through this
spelling**, and any future NEC-5 deck front end must guard the GN F3/F4
fields. If (d) is ever built, its oracle is NEC-2 (nec2c/PyNEC), and it
should refuse contact rather than compose with it (§2.3(d)).

---

## 4. The two trunks' mechanics

### 4.1 `BSplineSolver` — the #151 fold, concretely

**What exists.** A wire end within `1e-6 × (polyline length)` of `ground_z`
is tagged `"ground"` by `_wire_endpoint_status` (`bspline.py:1075`), and the
basis assembly (`bspline.py:1261-1283`) then **keeps** the value-1 boundary
basis it would otherwise drop:

```python
elif start_status[w_idx] == "ground":
    # Ground junction: keep the value-1 end basis so the end
    # current is a real dof — its image (integrated by the
    # ground blocks like every basis's) is the continuation
    # through the plane. No KCL partner: the image IS the
    # return path.
    kept.append((0, "gnd", None, "start"))
```

A grounded end is therefore *a junction end minus the constraint*: same
bases, but `"gnd"` never enters `junction_dirs`, so no KCL row references it.
`_grounded_junctions` (`bspline.py:1119`) drops the closure row for a K-way
junction lying in the plane, for the same reason.

**Where "coefficient 1" lives.** Not in a fold, and not in a multiply. The
image enters once, after the free operator is complete
(`bspline.py:3699-3739`), as one subtraction of an assembly built from the
*mirrored* segments contracted against **the same `supp_seg` and `polys`**:

```python
Z -= self._ground_finite_Z(J_img, supp_seg, polys, geom, ground=ground)
```

Because the image block is contracted against the same coefficient vector,
the mirrored half of every basis is driven by the same unknown as its real
half — amplitude ratio exactly 1, structurally, with nothing to un-hard-code.
The image *sign* (mirrored direction + image charge flip, "one minus
combined") is the single `-=` plus the mirror tangent-dot table.

**The charge bookkeeping at the contact point, exactly.** bspline's Φ term is
built from the basis **derivative** over its support
(`bspline.py:2306-2311`), and only from it — there is no endpoint term
anywhere in the assembly. So:

* the grounded basis *does* have non-zero divergence right up to the contact
  point (it rises 0→1 across its support), and it *does* carry net charge
  over that support;
* but that charge is a **bounded line density on the contact segment**, not a
  point charge at the node;
* the image half carries the mirrored density with the opposite sign, so at
  PEC the pair is charge-neutral at the contact;
* over a finite ground the Φ-image is scaled per-pair by `w_Φ`, so the
  cancellation is *imperfect* — and the residual, being a difference of two
  bounded distributions, stays bounded. That is precisely why bspline
  converges where the direct-field trunk diverged, and it is why **no
  compensating term is needed on this trunk.**

**What "unmerge the fold" would actually mean here**, for the record, since
the note asked: it would mean giving the image block its own coefficient
vector — a second unknown per basis. That is model (a) of §2.3, it violates
charge conservation, and it would introduce the 1/Δ point charge bspline
currently does not have. **It should not be built.**

### 4.2 The shared layers, and what they would carry

`_ground_spec.GroundConfig` — `(mode, eps_tilde, image_coefficient,
standard_fresnel)`, from four solver attributes. Its table:

| ground | mode | eps_tilde | image_coefficient | weighted |
|---|---|---|---|---|
| PEC | fold | None | **1** | False |
| refl-coef | fold | ε̃(ω) | **1** | True |
| sommerfeld | compose | ε̃(ω) | C₂ = (ε̃−1)/(ε̃+1) | False |

Note row 2: a genuinely finite ground with `image_coefficient == 1`. The
refusal's "coefficient 1, i.e. PEC" equation is not the code's equation.

`_potential_ground.PotentialGround` exposes `image_geometry()`,
`weight_tables()`, `weight_windows(observers, sources)` and `remainder()`;
`_field_ground.FieldGround` the direct-field analogues. **Neither carries any
concept of contact.** An exhaustive grep of the five ground modules for
`contact|clearance|touch` returns only `_ground_spec.ground_touch_tol` — a
tolerance, which answers "does this end touch the plane" and nothing else.

Two structural facts a contact-aware ground layer would have to face:

1. **The weight APIs are defined per (observer, source segment) pair at a
   specular angle** (`_potential_ground.py:744-750`): ρ_v/ρ_h are evaluated
   once per pair on the ray from the source segment's *image midpoint* to
   the observer. **A contact node is its own image**, so that ray
   degenerates and the API has no spelling for it. The sinusoidal trunk
   works around this by calling `_ground_refl.fresnel_rho` directly
   (`sinusoidal.py:3294`, "The node IS its own mirror image, so the specular
   ray is simply (node → observer)") — which is exactly the bypass
   `field-ground-interface.md:104-111` flags as the screen's future problem.
2. **The one thing a contact-aware layer would genuinely need to add** is not
   a coefficient. It is an answer to *"what is the folded scalar potential at
   a point IN the plane?"* — which is `(1 − w_Φ)·M0(plane)`, zero at PEC and
   non-zero otherwise, and which §4.3 shows is the load-bearing quantity for
   razor. That is one new operation, on the potential trunk only, with a
   degenerate-specular convention to pin.

### 4.3 `RazorSolver` — what its refusal actually protects

**The basis.** A grounded end gets a junction tent between the wire end and
its own image, with only the real wing spelled: the image wing is a σ = 0
ghost that the fold's second pass supplies (`razor.py:880-885`, six lines).
The row is the real half of the testing path only, halved by the self-image
invariance `E(M·r) = −M·E(r)`, and the fill states the charge story
(`razor.py:1866-1873`):

> σ_A = 0 empties its half of both terms, and the fold's second pass supplies
> the mirrored wing with the opposite charge, so the through-basis's two
> doublet halves are −1/h on the real segment and **+1/h on its image**. The
> unit of current that flows into the plane leaves no net charge at the
> contact point.

**Read that carefully: razor's contact charge is a doublet on the two
segments, not a point charge at the node.** Razor is structurally on
bspline's side of §2.3(a), not on the sinusoidal trunk's side. The refusal's
phrase "would take spurious contact charge (momwire#282)" imports language
from a formulation that has a 1/Δ node charge; razor does not have one.

**What razor genuinely has that bspline does not** is the T2 plane-reference
drop (`razor.py:1981-1993`):

```python
dM0 = M0c[s_b] - M0c[s_a]
if grounded.size:
    # A grounded row's testing path starts AT the plane, where the
    # folded scalar potential is identically zero: a point in the
    # plane is equidistant from every source and its image, so the
    # two blocks' contributions there are the same number and the
    # fold's minus cancels them. ...
    dM0[grounded] = M0c[s_b[grounded]]
```

This routine runs **twice** — once over real sources, once over mirrored —
and dropping the plane endpoint in both passes is exact *only because the two
passes produce the identical number there*. Over a finite ground the image
kernel is multiplied by `w_Φ` **before** the difference (`razor.py:1973-1980`,
`M0c[c0:c1] *= w_Phi`), so the two plane-endpoint terms become `M0(plane)` and
`w_Φ·M0(plane)`, and the drop silently discards `(1 − w_Φ)·M0(plane)` instead
of zero.

**So the defect is a missing term in the row's potential reference, not a
wrong basis function.** The physics is the second row of §2.2's table: *the
plane is not an equipotential over a finite ground*, so a formulation that
uses "Φ = 0 on the plane" as its reference must earn that reference back.
Three consequences:

* **A weighted image wing is the wrong fix and the refusal is right to reject
  it** — for the reason §2.3(a) gives, not the reason the refusal gives.
  Keep the wing at coefficient 1.
* **The right fix is bounded and local**: restore the dropped term as
  `(1 − w_Φ)·M0(plane)` (identically zero at PEC, so PEC stays bit-for-bit),
  which needs one new operation on `PotentialGround` — a weight at a
  degenerate specular geometry (§4.2, point 2). Under `mode == "compose"` the
  same statement is `(1 − C₂)·M0(plane) − Q(plane)`, with the association
  rule unchanged.
* **This study did not build it, so this is a hypothesis, not a finding.**
  §5.5 names the experiment that settles it in an afternoon: implement the
  restored term behind a flag and check razor against bspline on the S15
  decks with the difference-of-columns bar. If razor lands on bspline, the
  hypothesis holds; if it does not, the refusal was protecting something this
  study has not identified, and that is worth knowing before Stage 3 is
  scheduled.

**What razor's grounded row also assumes**, and which should be checked in
the same experiment: the row halving rests on `E(M·r) = −M·E(r)`, the
self-image invariance, which is a PEC statement. Over a finite ground the
image half of the testing path is **not** worth the identical number, so the
factor of 2 is also in question. That is a second, independent term, and it
is not obvious that it is small.

### 4.4 What the direct-field trunk does, for contrast

`SinusoidalSolver`'s contact machinery is a rank-one column subtraction
(`sinusoidal.py:3572`) removing the field of the `(1−ρ)I₀/jω` point charge
its formulation genuinely creates, per §2.3(a). It is *not self-adjoint*
(`sinusoidal_galerkin.py`'s
`test_the_282_contact_correction_is_not_self_adjoint`), it bypasses both
ground objects to call `_ground_refl.fresnel_rho` directly, and it leaves a
residual walk (`test_291_*`). It is the one place in the tree that already
handles a non-unit image coefficient at a contact node — and it handles it by
*cancelling* it, which is the same conclusion §2.3(a) reaches from the
physics.

---

## 5. Oracle strategy and acceptance bars

### 5.1 Is the CONTACT case gateable against printed output at all?

**Yes, and better than the standing refusal prose assumes** — but on a
difference-of-columns bar, not an absolute one, and with an explicit
per-ground shape.

The evidence, all from §3: the binary accepts contact over its Sommerfeld
ground; announces its contact rule in printed output; converges cleanly on
two geometries and nine grounds; recovers PEC at C₂'s rate; and is stable
under a degenerate contacting stub. That is a well-behaved oracle. What it is
*not* is an absolute reference for `Z` — at PEC contact momwire and the
binary already sit 1.26 Ω apart in X, all of it formulation.

So the bar is on `δ = Z(ground) − Z(PEC)` at matched N and matched geometry,
which is the pattern `solver-architecture.md` §6.5 used for the clearance
case ("every Sommerfeld delta is its own free-space delta to within 0.047 Ω").

**Bar shape, per §3.5's measurements.** The two behaviours seen are
different claims and should not share a number:

* **Convergent rows** (sea, very good — the high-|ε̃| grounds): the residual
  *shrinks* down the ladder. Gate by **decay**, finest rung pinned at its
  measured level + 25 %. That is §6.6's production-lane rule, unchanged.
* **Divergent-then-flat rows** (average, poor — the low-ε_r grounds): the
  residual *grows and saturates*. There is no honest tight bar here. Gate by
  an **envelope pin** at the measured saturation (1.5 Ω average, 4.0 Ω poor),
  recorded as a known gap with the issue number on it, exactly as §6.6
  handled the loop.

That asymmetry is the honest outcome and it should be written into the gate,
not averaged away.

### 5.2 The analytic limits

* **ε̃ → ∞ must recover PEC contact.** Available in principle and *broken in
  practice* through the Sommerfeld path (§3.7): the residual floors at
  0.33–0.59 Ω and rises with mesh, because the interpolation grid's error
  stops scaling with the ~1/√ε̃ surfaces at the small R₁ only contact
  queries. **Recommended: run this gate on the refl-coef path (where it
  passes to 2e−4 Ω at contact) and on the Sommerfeld path only at moderate
  σ ≤ 1e2, with the near-PEC floor recorded as a known instrument limit.**
  A stage that promises "bit or near-bit PEC recovery at contact under
  Sommerfeld" is promising something the current grid cannot deliver, and
  fixing the grid is its own unit.
* **ε̃ → 1 recovers nothing** (§3.4). The ground is the current's return
  path; remove it and the base feed has nothing to push against. The limit is
  degenerate, both codes print a finite residue, and the residues agree to
  0.83 Ω. **Recommended: record the value as a regression pin against
  momwire's own history; do not gate against the binary.** Gating agreement
  about a degenerate object is a false comfort.

### 5.3 Own-code gates that do not need the binary

* **The stubbed-limit gate** (§3.8 ladder B, and it is the study's most
  useful accidental find): a contact deck must equal the same deck with its
  contact replaced by a vanishing grounded stub. The binary holds this to
  ~0.19 Ω down to a 0.1 mm stub. It is cheap, formulation-agnostic, needs no
  licensed binary, and catches every class of contact-node bookkeeping error
  including model (a)'s.
* **The 1/Δ detector**: any contact formulation whose answer walks like 1/Δ
  under refinement has the §2.3(a) point charge. `test_291_*` already encodes
  the negative of this for the sinusoidal trunk.
* **Cross-formulation agreement, difference-of-columns.** Both trunks
  implementing the same model must agree on δ. Today they do not at contact —
  average soil N = 61: bspline 10.867−0.101j, sinusoidal 10.491+2.105j, a
  2.2 Ω gap, both of them wrong against the binary in the same direction.
  A cross-formulation gate at contact should be set at the *measured* gap and
  tightened as Stage 2 closes it — not set at the clearance bar and expected
  to hold.
* **PEC bit-identity.** Any contact work must leave the PEC contact fill
  bit-for-bit unmoved. `test_282_leaves_the_pec_contact_fill_bit_identical`
  is the pattern; razor's Stage-3 term is identically zero at PEC by
  construction, so this is achievable rather than aspirational.

### 5.4 The one unexplained number, and how to separate its causes

momwire under-predicts the ground-induced resistance at contact by 2.6–3.3 Ω
over poor soil and 0.9 Ω over average soil, growing with mesh and present on
both trunks at two frequencies (§3.5). It is **not** the interpolation grid
(§3.7: 1e−4 relative at real soils down to R₁ = 1e−5 λ). Three candidates,
and an experiment for each:

1. **The remainder's near-interface behaviour.** The `exp(−λ(z+z'))` decay
   that makes the spectral integrals converge is absent for the contact
   segment's near-diagonal pairs. *Experiment:* recompute the near-diagonal
   remainder blocks by direct evaluation at very high `rtol`, bypassing the
   grid entirely, at N = 41 poor soil, and see whether the 3.3 Ω moves. If it
   does, this is it and it is a quadrature/asymptotics problem in `Q`.
2. **A genuine model difference at the contact node.** The binary and
   momwire may be continuing the current differently below the mesh (§2.1).
   *Experiment:* the ladder-B stub test (§5.3) run in momwire — if momwire's
   own stubbed limit disagrees with its own contact deck by ~3 Ω over poor
   soil while the binary's agree to 0.19 Ω, the disagreement is in momwire's
   contact node, not in its ground.
3. **The missing base-loss resistance is real and momwire is right to lack
   it.** §2.3(b)'s tables give 2–100 Ω for exactly this quantity. *This is
   the least likely* — the binary is a full-wave code with no lumped base
   term either — but it would be settled by (1) and (2) coming back negative.

**This is the study's honest gap.** It is a served capability with a
measurable error of unknown cause, and §6 puts it in Stage 2 rather than
pretending Stage 1 closes it.

### 5.5 The razor experiment

Per §4.3: implement the restored plane-reference term `(1 − w_Φ)·M0(plane)`
behind a flag, keep the image wing at coefficient 1, and measure razor
against bspline on the §3.5 decks under the difference-of-columns bar,
plus PEC bit-identity. Also measure the row-halving assumption separately
(compare a grounded-row solve against the explicitly mirrored twin over a
finite ground, which no longer reduces analytically). **Until that experiment
runs, Stage 3 has no schedule.**

---

## 6. A staged plan

House style: each unit names its gates, and each stage names what stays
refused.

### Stage 1 — gate what is already shipped *(the smallest honest first unit)*

**Content.** Build `scripts/capture_contact_nec5_lane.py` on the pattern of
`capture_razor_pec_nec5_lane.py`: the §3.5 decks (monopole + inverted-L ×
PEC/sea/very-good/average/poor × N = 11/21/41/61), capturing the binary's
printed impedances into a `tests/golden_contact_nec5.py` of pure literals.
Then gate `BSplineSolver` against it on the difference-of-columns bar of
§5.1, with the two bar shapes it earns. Add the stubbed-limit gate (§5.3).

**And make the two decisions the measurements force:**

* **Refuse `ground_model="refl-coef"` at contact** (§3.6) — a new refusal
  string on the mixed-potential trunk and the direct-field trunk, replacing a
  docstring warning with an error, pointing at `sommerfeld`. This *removes* a
  served capability, which is why it is D3.
* **Record the near-PEC grid floor** (§3.7) as a known instrument limit, with
  the ε̃ → ∞ gate run on the refl-coef path and on σ ≤ 1e2 Sommerfeld.

**Gates:** difference-of-columns vs the golden binary numbers, per ground,
per §5.1's two shapes; stubbed-limit ≤ 0.3 Ω; PEC contact bit-identity across
the whole change; `test_282_*` / `test_291_*` / `test_g16c` unmoved.

**Still refused after Stage 1:** contact on razor (unchanged prose, but
corrected — see §6.4); contact on pulse; contact under refl-coef (new);
mid-span touchdown; radial screens.

**Why this is first.** It is the only unit that reduces risk rather than
adding surface. Everything downstream needs the lane, and the lane's first
run is what tells the maintainer whether §3.5's gap is a one-geometry
artifact.

### Stage 2 — close, or name, the low-ε_r gap

**Content.** Run §5.4's three experiments in order and act on whichever
fires. If (1), the fix is in the remainder's near-interface evaluation and
lands in `_sommerfeld.py` / `_potential_ground.Remainder` — shared, both
trunks, no basis change. If (2), the fix is in the contact node's
bookkeeping and is per-trunk. If neither, the gap is documented as a
formulation difference and the envelope pins from Stage 1 become permanent.

**Gates:** the Stage-1 bars, tightened to the newly measured level; no
regression on the clearance ladders (`solver-architecture.md` §6.5's 0.047 Ω);
no change to any PEC result.

**Still refused after Stage 2:** as Stage 1, minus nothing. Stage 2 improves
an answer; it does not open a row.

### Stage 3 — razor's grounded tent over a finite ground *(DONE, momwire#624)*

**Content.** §5.5's experiment first, as a spike, with a written go/no-go.
If go: restore the plane-reference term, keep the image wing at coefficient
1, add the degenerate-specular weight operation to `PotentialGround`, and
check the row-halving factor separately.

**Gates:** razor vs bspline on the §3.5 decks, difference-of-columns, at the
gap Stage 2 leaves; razor vs the binary on the same bar; PEC contact
bit-identity (the new term is identically zero at PEC); every existing razor
golden unmoved.

**Still refused after Stage 3:** contact under refl-coef on razor, unless the
same term repairs it (it should, by construction — but measure); pulse;
mid-span touchdown; screens at contact.

> **What actually happened — see the Stage 3 record at the end of this
> document.** The go/no-go came back GO, and the plan above is wrong in its
> conditional: the term was *not* restored, because measuring it is what
> killed it. No `PotentialGround` operation was needed and the row-halving
> was not settled, only narrowed. The gates listed here were all met, by the
> solver as it already stood. The refl-coef caveat is also wrong — that row
> stays refused by D3 on every trunk, and the term could not have repaired
> it, since coefficient 0 is flattest under refl-coef too.
>
> Kept unedited above because the shape of the miss is the useful part: this
> plan was written from a code reading that was correct about the arithmetic
> and wrong about its consequence, and it looked specific enough to schedule.

### Stage 4 *(optional, antennaknobs-side)* — the lumped base termination

**Content.** §2.3(b) as an antennaknobs composition: PEC solve + `R_g` on the
port + efficiency bookkeeping + the ARRL radial tables as data. Zero momwire
edits.

**Gates:** EZNEC/4nec2 parity on the composed number; a documentation gate
that the model's limits are stated where the user meets it.

**This stage is independent of 1–3 and can run at any time.** It is also the
stage most users would notice.

### Refusal prose evolution

Today, razor's `_CONTACT_OVER_FINITE_REFUSAL` says the fold "hard-codes image
coefficient 1, i.e. PEC, so a grounded end over anything else would take
spurious contact charge (momwire#282)". Three problems: `image_coefficient`
is 1 for the refl-coef ground too; the "spurious contact charge" is the
direct-field trunk's 1/Δ node charge, which razor's doublet does not have;
and the binary announces coefficient-1 continuation over every ground.

That replacement was written and shipped in stage 1, stating what razor was
argued to lack — the `(1 − w_Φ)·M0` term. **It is gone entirely as of
momwire#624**, and the reason is worth keeping here rather than only in the
git history: the sentence was accurate about the arithmetic and wrong about
what followed from it. The term the fill drops is real; restoring it does not
improve the answer, and no scale for it is self-consistent (Stage 3 record,
below). A refusal that names a mechanism is more persuasive than one that
does not, and this one was persuasive for eight months on the strength of an
argument nobody had measured.

The whole class is worth naming, because the tree has now produced two:
**a refusal whose prose is a diagnosis rather than an observation is a
hypothesis wearing a uniform.** Stage 1 already corrected this refusal's
prose once, on exactly that ground (the old wording was wrong three ways
about its own mechanism). The corrected wording was more careful and still
wrong — it said "Restoring the dropped term is a hypothesis, not a
diagnosis", which was honest, and then went on refusing as though it were a
diagnosis. The lesson is not "write better refusals": it is that a refusal
resting on an unmeasured mechanism should carry an expiry — an issue number
and an experiment — rather than a mechanism.

What remains is Stage 1's refl-coef withdrawal, on every trunk:

> ground CONTACT under `ground_model="refl-coef"` is refused: the
> reflection-coefficient ground's Φ-term weight is a specular-angle
> approximation with no validity at zero clearance (momwire#153), and at
> contact it lands ~27 Ω from the Sommerfeld answer on the same deck rather
> than approximating it. Use `ground_model="sommerfeld"`, which is
> contact-capable and gated there.

---

## 7. Maintainer decision points

| # | decision | what hangs on it |
|---|---|---|
| **D1** | **Does an ungated served capability get gated, or withdrawn?** momwire has solved grounded verticals over earth since #151 with no reference comparison. Stage 1 says gate it. The alternative — refuse contact over finite grounds entirely until gated — is defensible and much more disruptive. | the whole plan's shape |
| **D2** | **The contact lane's bar shape.** §5.1 proposes two shapes (decay for high-\|ε̃\|, envelope pin for low-ε_r) rather than one number. One bar would be simpler and would be a lie on one half of the table. | Stage 1's gate |
| **D3** | **Refuse refl-coef at contact?** It is a silently wrong default answer (§3.6) and refusing it removes a capability users may be relying on. Options: refuse / warn / leave and document. | Stage 1; user-visible |
| **D4** | **Is the near-PEC grid floor (§3.7) worth its own issue?** It breaks the ε̃ → ∞ gate at contact and is a shared-layer accuracy defect, not a contact defect. It may matter for high-σ clearance work too. | Stage 1's limit gates; possibly a separate unit |
| **D5** | **How much is 2.6 Ω of missing ground-loss resistance worth?** On a full-size 40 Ω monopole it is ~6 points of efficiency (90.1 % against 96.5 %). On the short loaded verticals this audience builds it is much larger. If the answer is "a lot", Stage 2 moves ahead of Stage 1's polish. | Stage 2's priority |
| **D6** | **Does razor need contact over finite grounds at all?** ~~bspline serves it. Razor's value is the NEC-5 twin claim, which at contact is measured (§3.5) to be the thing in question. Stage 3 is a consistency argument, not a capability argument.~~ **ANSWERED YES, momwire#624 (2026-08-25)** — and the question went stale before it was answered: when it was written, razor was one formulation inside momwire and the EZNEC seam had one basis, so serving contact bought a second internal opinion. momwire#593/#603 made `razor-nec5` an executable a user points EZNEC at, at which point a base-fed vertical over real ground refusing is a capability question and not a consistency one. See the Stage 3 record. | Stage 3's existence |
| **D7** | **Is model (b) an antennaknobs deliverable?** §2.3(b) says it is the model most users want and proves nothing about momwire. It is also the only one of the four that a user can act on. | Stage 4 |

---

## 8. What this study does not know

* **Why momwire's contact answer misses the binary's by 2.6–3.3 Ω over poor
  soil.** §5.4 narrows it to three candidates and rules out the
  interpolation grid, but does not close it.
* ~~**Whether §4.3's diagnosis of razor is right.** It is a reading of the
  code plus a physical argument, not a measurement. §5.5 is the
  measurement.~~ **ANSWERED NO** (momwire#624). §5.5 ran. The dropped term
  is real; restoring it makes the binary comparison worse at full strength
  and fails the stubbed ladder at every scale. The arithmetic in §4.3 is
  correct and the conclusion drawn from it is not.
* **Whether razor's grounded-row halving survives a finite ground.** The
  identity it rests on is a PEC identity. Still not measured directly —
  momwire#624 reached it only by elimination. The restored term's accuracy
  argmin sat near 0.4 rather than the ~0.5 a pure halving error predicts,
  and the stubbed ladder then showed the term is not mis-scaled at all, so
  the halving is neither confirmed nor cleared. What IS measured is the
  residual it would have to explain: 0.21–0.55 Ω of contact-node
  self-inconsistency over a finite ground, against 0.002 Ω at PEC.
* **What NEC-5 does at the contact node.** Only what it printed is recorded:
  that it continues the current to the image, over every ground, and that the
  result converges and recovers PEC at C₂'s rate. No inference beyond that
  was attempted, and none should be.
* **Whether any of this generalizes past a vertical.** Two geometries were
  probed (monopole, inverted-L). A grounded K-way junction, a slant wire into
  the plane, and a tower with guys are all untested at finite ground.

---

## Appendix: probe inventory

All in the session scratch directory's `study282/` (decks and printouts in
`decks/`); none of them are repo code.

| script | what it establishes |
|---|---|
| `probe.py` | deck builder + runner + printed-output parsers |
| `s1_address.py` | base-feed address; the free-end "no basis function" printed error |
| `s2_grounds.py` | §3.2 ground × mesh table; the power-budget finding |
| `s4_liftoff.py` | §3.8 both lift-off ladders |
| `s5_limits.py` | §3.3 PEC limit, §3.4 free-space limit, S6 inverted-L, S7 screens |
| `s8_confirm.py` | §3.9 GN field semantics (the permeability trap) |
| `s9_momwire.py` | momwire's own contact ladders, all trunks × grounds |
| `s10_mw_limits.py` | momwire's analytic limits at contact; razor's refusal |
| `s11_floor.py` | §3.7 the PEC-limit floor, contact vs clear, both ground models |
| `s12_knobs.py` | §3.7 quadrature/radius controls; the sinusoidal twin |
| `s13_grid.py`, `s14_smallR.py` | §3.7 grid-vs-direct accuracy, the small-R₁ corner |
| `s15_columns.py` | §3.5 the headline difference-of-columns table |
| `s16_freq.py` | §3.5 finding 4, the 3.5 MHz repeat |
| `s17_scaling.py` | §2.3(a) the measured 1/Δ spurious-charge law |

---

## Decision record — 2026-08-18

*Appended after maintainer review. The study above is unchanged and stays
the record of what was measured before any decision was taken; this section
records what was decided and what stage 1 then measured while implementing
it. Where the two disagree, this section is later and wins, and says so.*

### The decisions

**D1 — gate, don't withdraw wholesale.** The served-but-ungated capability
is gated rather than refused across the board. Stage 1 builds the lane and
lets it decide per ground, which is §6's recommended ordering.

**D2 — two bar shapes, as §5.1 proposed.** A DECAY bar on the high-|ε̃|
grounds (sea water, very good — the residual shrinks with mesh, gated at the
finest rung's measured level + 25 % with net decay required) and an ENVELOPE
PIN on the low-ε_r grounds (average 1.5 Ω, poor 4.0 Ω), each carrying a
docstring that names stage 2 as the investigation that may tighten it. The
envelope rows additionally carry a SATURATION check — the last rung's
increment under a quarter of the first's — because an envelope on a residual
that is still growing is worthless.

**D3 — refl-coef at contact is REFUSED**, on every solver that served it:
`BSplineSolver` and its `HMatrixSolver` / `ArrayBlockSolver` subclasses, and
`SinusoidalSolver` with its `SinusoidalGalerkinSolver` subclass. At
construction, on the same anchor scan the grounded basis itself is built
from (`_ground_spec.contact_ends`), with the measurement in the message.
This WITHDRAWS a served capability and every test that exercised it was
changed with a comment naming this decision.

**D4 — the near-PEC grid floor (§3.7) gets its own issue: momwire#443.** It
is a shared-layer accuracy defect in the interpolation grid, not a contact
defect, and it may matter for high-σ clearance work too. Stage 1 pins the
floor where it is (0.477 / 0.550 / 0.589 Ω at N = 21/41/81 on the
mixed-potential trunk) with the issue number on the pin, and does not try to
fix it.

**The dialect landmine (§3.9) is filed as momwire#444.** NEC-2's
`GN 2 NRADL … FRATI FRATIS` spelling writes NEC-2's radial-length/radius
fields into NEC-5's permeability fields, producing a silently MAGNETIC
ground whose printed banner does not reveal the substitution. Any NEC-5 deck
front end must guard `GN` F3/F4.

**Model (b), the lumped base termination, is filed as antennaknobs#951** —
an antennaknobs composition (PEC solve + `R_g` on the port + efficiency
bookkeeping + the ARRL radial tables as data), with zero momwire edits, per
§2.3(b) and D7.

**A post-study cross-check strengthens D1 and D3.** The maintainer asked
what NEC-2's own reflection-coefficient ground does at contact. Measured on
stock nec2c, the same 5.35 m contact monopole at 14 MHz: `GN 1` prints
39.4 + 22.1j Ω (sane); `GN 0` over average soil (13, 0.005) prints
175 − 779j Ω and over poor soil (4, 0.001) prints 155 − 1248j Ω; the same
wire lifted 0.5 m (0.023 λ) over `GN 0` prints 45 − 10472j Ω. Hundreds to
thousands of ohms of spurious reactance, in the reference implementation of
the model, below its own validity floor. So the refl-coef row's failure at
contact is a property of the MODEL and not of momwire's implementation of
it, and the refusal message says so — which matters, because the wrong
lesson for a user to take from a refusal is "momwire cannot do what other
codes can".

### What stage 1 measured while implementing this

Four findings, in descending order of consequence for stage 2.

**1. The low-ε_r gap belongs to the GROUND, not to the geometry.** The
study measured §3.5 on one geometry and §8 lists "whether any of this
generalizes past a vertical" as something it did not know. The lane carries a
second deck — the same quarter wave bent into an inverted-L, with an interior
bend and a second edge between the contact node and the far end — and the
saturated residuals land on top of each other:

| ground | monopole (N = 81) | inverted-L (N = 96) |
|---|---|---|
| sea | 0.2703 | 0.2484 |
| very good | 0.1766 | 0.2398 |
| average | **1.2712** | **1.2646** |
| poor | **3.3274** | **3.3244** |

Under 1 % apart on both envelope rows. Stage 2 should look at the shared
half-space machinery (§5.4 candidate 1), not at the vertical's contact node.

**2. §5.4 candidate 2 is dead.** The study proposed the ladder-B stub test
in momwire as the experiment that would separate a contact-node fault from a
ground fault: *"if momwire's own stubbed limit disagrees with its own
contact deck by ~3 Ω over poor soil … the disagreement is in momwire's
contact node."* Measured: **0.011 Ω** over poor soil, at every mesh
(N = 21/41/81), the same size as the PEC and average-soil figures, which
carry no gap to explain at all. Whatever momwire is missing over poor soil,
it is not the contact node's bookkeeping. That leaves candidates 1 and 3,
and finding 1 above points at the same place. *(Stage 2's record restates
this verdict at its true width — the stub ladder is a self-consistency
instrument, so what it excludes is an internally **inconsistent** contact
node, not a formulation difference shared with the stubbed limit. Read the
candidate-2 paragraph there before quoting this one.)*

**3. A correction to §3.8's ladder B, which makes it a much sharper
instrument.** The study's stubbed ladder is fed AT the stub's grounded base,
which is what NEC's `EX` on segment 1 does. But then the feed segment
shrinks with the stub, so the ladder measures the delta-gap source model as
much as it measures the contact node: fed that way momwire's own PEC ladder
is **53 Ω** out at a 0.1 mm stub, which is a statement about a delta-gap
over a 0.1 mm gap and nothing at all about the ground. Move the feed onto the
radiator, where the segment length is fixed, and the same ladder holds to
**3.2e-4 Ω** at the same stub — three orders sharper than the ~0.19 Ω §3.8
records for the binary, whose own wobble down that ladder is very likely the
same effect seen through its own source model. Anyone re-running ladder B in
any code should move the feed first.

**4. §6's proposed second deck is not usable, and the reason is the deck.**
Stage 1 was specified with the study's 3 m + 6 m inverted-L (§3.2 / S6) as
the lane's second geometry. It was captured and rejected: at 0.42 λ it sits
near the grounded half-wave antiresonance, where |Z| ≈ 1050 Ω and neither
code is converged — over N = 12 → 96 the binary's own PEC answer walks
264+911j → 328+1004j (+24 % in R) and `BSplineSolver`'s walks 133+659j →
269+917j (+103 %). The difference of columns cancels a formulation offset
but cannot cancel two offsets that are both still moving, and the residuals
came out at 30–95 Ω on every ground, including the high-|ε̃| ones the
monopole closes to 0.27 Ω. That is a statement about an antiresonant deck at
coarse mesh, not about ground contact. The lane takes the bent quarter wave
instead.

### Two smaller corrections to the study's text

* §5.1's envelope numbers were proposed as "1.5 Ω average, 4.0 Ω poor" and
  are shipped at exactly those values, but note the monopole's average-soil
  residual is 1.271 Ω at N = 81 and still creeping — the 1.5 Ω envelope has
  about 18 % of headroom, not much. A finer rung would want re-derivation.
* §4.3's razor diagnosis is a hypothesis and stage 1 did not test it. Razor's
  `_CONTACT_OVER_FINITE_REFUSAL` prose was corrected — the three things §4.3
  showed the old wording got wrong about its own mechanism are gone, and the
  doublet is stated correctly — but the replacement says the plane-reference
  term is what stage 3 will TEST, not what is known. No razor behaviour
  changed; only prose, and two alias keys in the capability declaration so a
  consumer holding a concrete ground reads the truth on every row.

### What is still refused after stage 1

Contact on razor (both finite grounds, unchanged); contact on pulse;
**contact under refl-coef, on every trunk (new)**; mid-span touchdown;
radial screens.

---

## Stage 2 record — 2026-08-19

*Same convention as the stage-1 record above: the study body is unchanged
and stays what was measured before any decision was taken. This section is
later and wins where the two disagree, and says so.*

Stage 2's brief (§6) was "run §5.4's three experiments in order and act on
whichever fires". All three have now been run. **None of them fired**, and
no fourth candidate replaced them, so §6's own fallback applies: the gap is
a formulation difference, and stage 1's envelope pins are permanent.

That is a smaller headline than "closed", so this record is mostly about
what is now *known* rather than merely suspected — because the difference
between a shrug and a result is whether the next person can tell which doors
are already shut.

### The three candidates, and how each died

**Candidate 2 — an *internally inconsistent* contact node — died in
stage 1** (the stub ladder: momwire's own stubbed limit agrees with its own
contact deck to 0.011 Ω over poor soil, where the study predicted ~3 Ω if
the contact node's bookkeeping were at fault). Recorded above; repeated here
so the three verdicts sit together — and restated more carefully than
stage 1 wrote it, because this record's own conclusion (below) puts the gap
at the contact node and the two must not read as a contradiction. The stub
ladder is momwire against momwire: a **self-consistency** instrument. It
excludes any bookkeeping error that would make the contact deck part
company with its own stubbed limit — the 1/Δ double-counting pathology
included — but it is blind, by construction, to a formulation difference
the contact deck *shares* with the stubbed geometry. And stage 2 measured
that the suspect quantity is shared: `w_Φ` is the ground's charge response
for every low segment, not a property of the junction (patching it moves
the *clearance* deck by 27–117 Ω — see the hypothesis section below). So
what died is §5.4's literal candidate — the two codes continuing the
current differently below the mesh in a way momwire's own limit would
expose — not the possibility of a shared, self-consistent formulation
difference at that node, which is where the verdict below lands.

**Candidate 1 — the remainder's near-interface behaviour — is dead, and
here is the exact width of the kill.** The study specified the experiment:
*"recompute the near-diagonal remainder blocks by direct evaluation at very
high rtol, bypassing the grid entirely, at N = 41 poor soil, and see whether
the 3.3 Ω moves."* `scripts/probe_contact_direct_remainder.py` is that
experiment. It swaps a `DirectGrid` — same `eval(R1, θ)` contract, answering
with `iv_surfaces_direct` instead of a cubic Lagrange stencil — into the
numpy remainder path, and masks the two grid-consuming accelerator kernels
so that path runs.

| ground | N | residual, shipped | residual, direct (interpolation bypassed) |
|---|---|---|---|
| poor | 41 | 3.2691 | 3.2722 |
| poor | 81 | 3.3274 | 3.3305 |
| average | 81 | 1.2712 | 1.2715 |

Be precise about what that bypass bypasses. `iv_surfaces_direct` is the
shipped grid's **own fill function**, so the swap removes the
*interpolation* and sweeps the integration *tolerance* while reusing the
`_six_integrals` contour machinery the candidate accuses. And the tolerance
axis is saturated: rtol 1e−7 through 1e−13 all give residual 3.272166 at
N = 41 poor, so the production default of 1e−9 was already converged and
the whole 0.003 Ω in the table is interpolation. What closes the
correctness half is the contour machinery's own exactness check:
`greens_free_space_check` integrates the Sommerfeld identity over the same
contours and lands 2.6e−12 relative (rtol 1e−11, both the fig-13 and fig-14
paths) at h = 0.0294 m — twice the lowest Gauss node on the N = 41 contact
segment, the smallest argument this deck actually queries.

The other half of "quadrature/asymptotics in Q" — the remainder's *spatial*
order — was swept too: `n_qp_sommerfeld` 3 → 5 → 8 → 12 at N = 41 poor moves
the residual 3.2691 → 3.2506 → 3.2436 → 3.2409, i.e. 0.03 Ω converging.
Neither knob is worth 3 Ω.

**Candidate 3 — "the missing base-loss resistance is real and momwire is
right to lack it" — is dead, and this is the measurement stage 2 did not
expect to be able to make.** `scripts/probe_contact_halfspace_sweep.py`
sweeps the half-space against the binary at a conductivity whose loss
tangent stays under 1e−2 across the whole sweep — a ground that cannot
dissipate. |discrepancy| at N = 41, monopole:

| ε_r | 1.5 | 2.0 | **2.5** | 3.0 | 4.0 | 5.0 | 6.5 | 8.0 | 10 | 13 | 16 | 20 | 30 | 50 | 81 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ω | 3.21 | 4.15 | **4.36** | 4.28 | 3.87 | 3.42 | 2.84 | 2.40 | 1.97 | 1.53 | 1.24 | 0.98 | 0.67 | 0.47 | 0.41 |

A missing LOSS term cannot be at its maximum over a ground with no loss in
it. The gap is not a loss term, and §2.3(b)'s 2–100 Ω tables are not what
momwire is missing.

### What the half-space map says

The sweep above is the first picture of the gap as a function of the ground
rather than as four soils, and it reorganises the stage-1 table:

* The discrepancy is a **smooth single-peaked curve vanishing at both
  physical limits** — ε̃ → 1 (no ground) and ε̃ → ∞ (PEC) — peaking near
  ε_r = 2.5. Both zeros are structural, so the curve had to look roughly
  like this; what is informative is *where* the peak is and how large. The
  low-side zero is measured, not asserted: extending the sweep below
  ε_r = 1.5 gives 1.35 → 2.628, 1.2 → 1.816, 1.1 → 1.105, 1.05 → 0.701,
  1.02 → 0.464 Ω — monotone to zero.
* A second, independent path through the ε̃ plane agrees. `--mode sigma` at
  ε_r = 5 walks σ from 0 to 1: the discrepancy is 3.42 Ω at σ = 0 and falls
  monotonically to a 0.2–0.35 Ω floor by σ = 0.03.
* **Stage 1's four-soil table was reading a sign crossing as agreement.**
  Very good ground's 0.005 Ω residual at N = 41 is not the two codes meeting;
  it is the imaginary part of the discrepancy passing through zero between
  sea water (+0.37∠101°) and average soil (−0.89 − 0.75j). At σ → 0 the same
  ε_r = 20 ground shows a 0.98 Ω discrepancy. The `vgood` row's non-monotone
  ladder, which stage 1 handled by demanding *net* decay rather than
  rung-by-rung descent, is that crossing.
* Along both paths the **real part tracks the uncancelled image-charge
  fraction** `1 − C₂ = 2/(ε̃ + 1)`: `K = Re(disc / (1 − C₂))` is −10.3 ± 0.3
  over ε_r ∈ [10, 30] and −9.2 ± 0.2 along the whole σ path from 0 to 3e−3.
  The spelling matters with complex ε̃: `disc.real / (1 − C₂)`, which an
  earlier draft of this record wrote, is a different quantity and walks
  −9.19 → −6.88 along the σ path; `K` is what reproduces. **This is a good
  description and not a law.** It drifts ~40 % below ε_r = 6, the constant
  is −8.6 on the bent quarter wave, and across the wire radii whose decks
  pass the 1.5 Ω PEC-conditioning bar (0.5–5 mm, a 10× range) it is −9.51
  to −10.41. (An earlier draft claimed "−10 to −12 across a 100× radius
  range"; the −12 end came from the a = 50 mm deck, whose PEC columns are
  2.668 Ω apart and which `probe_contact_deck_conditioning.py` itself
  rejects.) It is quoted here because it is the sharpest quantitative
  handle anyone has on the gap, not because it should be gated.

### The hypothesis the evidence now points at, stated as a hypothesis

Everything measured is consistent with the gap living in **§2.2's third row
— what continues the current into the earth at the contact node** — and
nothing else is left standing. The charge bookkeeping is where to look, and
it means **correcting a piece of this study's own reasoning**.

§4.1 records the one structural asymmetry a grounded basis has, and dismisses
it:

> the image half carries the mirrored density with the opposite sign, so at
> PEC the pair is charge-neutral at the contact; over a finite ground the
> Φ-image is scaled per-pair by `w_Φ`, so the cancellation is *imperfect* —
> and the residual, being a difference of two bounded distributions, stays
> bounded. That is precisely why bspline converges where the direct-field
> trunk diverged, and it is why **no compensating term is needed on this
> trunk.**

The first three clauses are right and the conclusion does not follow.
*Bounded* explains why bspline converges where sinusoidal diverges; it says
nothing about *what it converges to*. The grounded basis integrates to 1 at
the node and its image to −w_Φ, so the composite carries a net contact
charge proportional to `1 − w_Φ = 1 − C₂` that does **not** shrink under
refinement — which is the shape of a limit difference that saturates with
mesh, vanishes at PEC, and scales the way the measured one scales.

Two things stop this being a finding rather than a hypothesis, and both were
tested:

* **A symmetry test cannot fire.** The obvious sharp check — a dropped
  testing-side bracket `[f_m Φ_n]` would break `Z = Zᵀ` — is a null by
  construction: both sides of the Φ term carry a basis derivative, so the
  weak form is symmetric whatever the bracket does. Measured
  `max|Z − Zᵀ|/max|Z|` is 6.8e−11 at PEC and the same order on every finite
  ground, contact and clearance alike (diel contact, the one that differs
  at all, is 6.4e−11). `scripts/probe_contact_node_structure.py --mode
  symmetry` prints the full table.
* **Forcing `w_Φ = 1` near the contact does not close the gap and is not a
  fix.** Setting the Φ-image weight to its charge-conserving value on the
  bottom 1/2/3/5 segments moves the poor-soil residual 3.269 → 3.318 →
  3.375 → 3.386 → 3.315 — the wrong direction, and small. It is also not the
  local operation it looks like: the same patch moves the *clearance* deck by
  27–117 Ω, because `w_Φ` is the ground's charge response for those segments
  and not a property of the junction. Whatever the right compensating term
  is, it is not a re-weighting of the existing table.

So: the contact node is where to look, a naive weight patch is not the
answer, and nobody should start building until there is a derivation. That
is a stage-3-sized piece of work and it is not scheduled here.

### The instrument's validity condition, which cost stage 2 real time

Stage 1 found one deck the difference-of-columns cannot measure (finding 4:
the 3 m + 6 m inverted-L, antiresonant, both codes' PEC columns still
walking). Stage 2 found three more while looking for the law behind the gap,
and they all fail the same way:

| deck | PEC column offset | what the residual did |
|---|---|---|
| quarter wave @ 14 MHz (the lane's) | 0.02 Ω in R, 1.26 Ω in X | clean, reproducible |
| bent quarter wave (the lane's) | 0.056 Ω in R, 0.66 Ω in X | clean, tracks the monopole to 1 % |
| same wire @ 21 MHz (0.375 λ) | 41 Ω in R | `K` walks 17 → 14, sign-inconsistent |
| grounded half wave (0.5 λ) | 349 Ω in R (43 %) | `K` walks 151 → 276 |
| grounded inverted-U | 20.8 Ω | `K` walks −25 → +11 |

A difference of columns cancels a *constant* formulation offset; it cannot
cancel two offsets that are both still moving. `test_the_pec_columns_agree_
well_enough_to_difference` is now the lane's first test so that a third deck
fails there rather than publishing a residual that means nothing.

### What stage 2 hands to momwire#443

The direct-grid probe measures what the interpolation grid is worth on a
shipped answer, per ground, which the ε̃ → ∞ gate could not. Two measures,
kept apart because an earlier draft mixed them under one header:

| ground | grid's cost on the answer, \|z(grid) − z(direct)\| | residual's movement when the grid goes |
|---|---|---|
| sea water | **0.13 Ω at every mesh** | 0.0803 / 0.0965 / 0.1092 (N = 21/41/81: residuals 0.5248/0.3721/0.2703 with the grid, 0.4445/0.2756/0.1611 without) |
| very good | 0.0037 Ω | — |
| average | 0.0008 Ω | — |
| poor | 0.0032 Ω | — |

That is exactly momwire#443's shape — a near-PEC error at the small R₁ only
contact geometries query — and on sea water it is **about 40 % of stage 1's
decay bar on the answer measure, 32 % on the residual measure**. So the
high-|ε̃| DECAY rows are partly gating the instrument. If #443 is fixed,
sea water's finest rung should fall to ~0.16 Ω and that bar wants
re-deriving; the numbers to beat are in the table above.

*(Closed 2026-08-19: momwire#443 landed — the grid's steep-band R₁ spacing
is now keyed to the near-interface boundary layer 1/|k₁|. Sea water's
finest rung measured **0.1614** against the 0.1611 predicted above, the
decay pins are re-derived at the new levels, the answer-measure share fell
0.13 → 0.0013, and the ε̃ → ∞ contact recovery went from a rising
0.48–0.59 Ω floor to ~0.001 Ω — the same class as the binary's own §3.3
ladder. The lossy soils' grids are untouched and their envelope rows are
unchanged, so stage 2's verdict and the #282 pins are unaffected.)*

### What stage 2 shipped

* `diel` (ε_r = 2.5, σ = 1e−5) joins the lane on both geometries — the row
  that kills candidate 3, kept as a gate rather than as a paragraph. It is
  also the best-behaved row in the table: the **largest** residual anywhere
  (4.341 monopole / 4.314 inverted-L at the finest rung), and **flat** —
  4.4905 → 4.3410 from N = 11 to N = 81, a 3 % walk where average soil nearly
  doubles down its ladder. A residual already at its limit on an 11-segment
  mesh needs no saturation argument at all, which is why this row is gated on
  its level *and* its flatness and the soil rows still need theirs. One
  caveat the gate's docstring also carries: the flatness claim spans rungs
  of **unequal conditioning** — at N = 11 the two codes' PEC columns are
  3.98 Ω apart, well outside the 1.5 Ω bar the guard holds the finest rung
  to, and max(res) (which sets the 4.7 Ω envelope) is attained exactly
  there. The flatness is still evidence of a limit difference; its left
  edge is read through a poorer instrument than its right.
* The PEC-column conditioning guard described above, covering the whole
  ladder the gates use: the columns must converge toward each other
  monotonically rung by rung, and the finest rung must be inside 1.5 Ω.
  (Measured, coarse to fine: monopole 3.98/2.13/1.26/0.96/0.80, inverted-L
  1.67/0.96/0.66/0.55/0.49 — the 1.5 Ω standard holds from mid-ladder
  down, not at the coarse end.)
* `test_the_gap_is_not_the_remainder_quadrature` and
  `test_the_gap_is_not_the_interpolation_grid` — the binary-free halves of
  the candidate-1 kill, pinning that the answer stays insensitive to the
  remainder's quadrature order (0.0373 Ω under n_qp 3 → 8) and to the
  interpolation stencil (0.0032 Ω against a DirectGrid swap), so candidate 1
  reopens loudly if either changes.
* The two probes, promoted out of scratch:
  `scripts/probe_contact_direct_remainder.py` and
  `scripts/probe_contact_halfspace_sweep.py`.

### What is still refused after stage 2

Unchanged from stage 1 — stage 2 improved understanding, not capability.
Contact on razor (both finite grounds); contact on pulse; contact under
refl-coef, on every trunk; mid-span touchdown; radial screens.

---

## Stage 3 record — 2026-08-25

**momwire#624. D6 answered YES; §4.3's diagnosis answered NO.** Razor serves
ground contact over the Sommerfeld ground. The plane-reference term this
study proposed as the fix is *not* part of it, and that is the finding.

### What §5.5 asked for, and what it returned

§5.5 said: implement `(1 − w_Φ)·M0(plane)` behind a flag, keep the image wing
at coefficient 1, measure against bspline on the §3.5 decks plus PEC
bit-identity, and measure the row-halving separately. Two spikes ran it
(`b16d67e`, `e38c77b`); both scripts are kept, at
`scripts/spike_contact_plane_reference.py` and
`scripts/spike_contact_stub_ladder.py`.

**PEC bit-identity: passed**, at every rung, as `==` rather than `approx` —
at PEC there is no `w_Φ` table, so the term's branch never runs and the
arithmetic is untouched rather than augmented by a zero.

**The parity gate passed, and it did not need the term.** With the term OFF —
i.e. what lifting the refusal alone would give — razor's residual against the
binary's printed shift is bounded and saturating on all five grounds, and is
already competitive with the row that ships:

| ground | razor OFF, N = 61 | bspline, N = 61 |
|---|---|---|
| sea | **0.005** | 0.201 |
| very good | 0.405 | 0.116 |
| average | 1.397 | 1.236 |
| poor | 3.384 | 3.309 |
| dielectric | 4.332 | 4.346 |

**The term makes it worse.** At full strength, poor soil goes 3.384 → 3.906 Ω.
A swept coefficient has an argmin near 0.4 on every lossy ground at every
mesh (poor 3.375 → 2.058), which is a real and repeatable shape — one
coefficient, soil-independent — but a fitted coefficient is not a derivation.

### The discriminator, and why the fit died

The obvious next move was to ask *why* 0.4. Two candidates: the row halving
(a PEC identity a weighted image does not satisfy, predicting ≈ 2×) and the
Sommerfeld remainder (under `mode == "compose"` the term reconstructs the
plane potential from the weighted exact-image half only, leaving out `Q`).

Half of that needed no experiment. `T2`'s `M0c` is the reduced-kernel moments
times `w_Φ` and nothing else, and `rem_fn` is used only in the `Q` FIELD term
*after* `T2` is assembled — so under refl-coef the term IS the whole folded
plane potential and under sommerfeld it is missing the remainder's, by
construction. But that predicts a **soil-dependent** deficit and the measured
argmin did not vary with soil, pointing instead at a model-independent scale.

The experiment §5.5 implied — sweep the coefficient under refl-coef against
the binary — **cannot be run as written**: refl-coef at contact sits ~26 Ω
from the binary (52.006+21.505j against 26.643+10.767j, average soil,
N = 21), which is exactly the model error D3 withdrew that row for, and it
dwarfs the ~3 Ω the term is worth. Fitting a 5 Ω knob to close a 26 Ω model
gap measures the gap.

So the reference had to go and the instrument became **self-consistency**:
§3.8's ladder B, momwire against momwire, replacing the contacting element
with a vanishing grounded stub. Shrinking the stub does not change the
antenna, so a self-consistent contact node must give an h-independent answer.
Two stage-2 corrections are built in and neither is optional — the feed goes
on the RADIATOR (fed at the shrinking base, the ladder measures the delta-gap
source model instead: 53 Ω out at a 0.1 mm stub), and the mesh above the stub
is held FIXED on the original knots (re-meshing every rung drifts the PEC
control 2.5 Ω, a mesh artefact with no contact node in it). Held fixed, the
PEC control is flat to 2.15e-3 Ω, which is the harness certifying itself.

**Coefficient 0 is flattest on every row, by an order of magnitude, on both
soils and both ground models.** At 0.4 the ladder slides 42.18+25.82j →
33.58+16.50j as the stub shrinks, converging back onto the coefficient-0
answer: the term's contribution evaporates with the contacting element. The
term is not mis-scaled — no scale makes it self-consistent — and the deficit
is not `Q` either, since refl-coef and sommerfeld behave alike.

### What shipped

The refusal, deleted. The bar is **D1's, literally**:
`tests/test_razor_contact_finite_ground.py` imports `_ENVELOPE`,
`_DIEL_ENVELOPE` and `_DIEL_FLATNESS` from `test_contact_nec5_lane` rather
than restating them, so razor passes the constants bspline ships under and
re-deriving one moves both. Measured maxima 1.3996 (avg) and 3.3931 (poor)
against pins of 1.5 and 4.0.

Where the two trunks differ in SHAPE, that is stated rather than forged.
razor's residual grows-then-flattens on every ground, including the two where
bspline's decays, so razor's high-|ε̃| rows do not take bspline's decay bar:
`sea` is pinned on level only and says out loud that it is still growing;
`vgood` gets an envelope with the saturation check. This is D2's own
reasoning applied across trunks instead of across grounds — one bar shape
would be a lie on half the table.

Two rows are worth reading as physics rather than as gates. On **sea water**
razor is twenty times closer to the binary than bspline (0.008 against
0.161 Ω), which is what being NEC-5's own formulation is for. On the
**lossless dielectric** the two land within 0.01 Ω of each other (4.3376
against 4.3410) — two different bases with two different testing schemes
agreeing that closely on the size of a limit difference is the strongest
evidence yet that §5.4's gap belongs to neither trunk.

### What Stage 3 leaves, and it is a better thing than it removes

At coefficient 0 the finite-ground stub ladders spread **0.21–0.55 Ω** where
the PEC control holds 0.002. So the contact node *is* internally inconsistent
over a finite ground, by about half an ohm — and that is a residual with a
**target**, on an instrument needing no licensed binary and no capture, which
is precisely what Stage 3 did not have going in. It is pinned by
`test_the_contact_node_is_self_consistent_to_under_an_ohm`.

Note carefully that this is **not** §5.4's 2.6–3.3 Ω. That gap is shared with
bspline, is largest over a ground that cannot dissipate, and survives every
candidate stage 2 tested. The half-ohm here is razor's own and is the smaller,
sharper question.

### What is still refused after stage 3

Contact under **refl-coef**, on every trunk — D3, unchanged, and razor now
reaches it through `_ground_spec`'s own scan and sentence rather than a copy.
Contact on **pulse**. **Mid-span touchdown**. **Radial screens** at contact.
Razor's line is gone from this list.
