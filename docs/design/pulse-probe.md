# The pulse probe: what a new solver row cost

**momwire#416.** A deliberate test of `docs/design/solver-architecture.md`
§0.1's metric — *a throwaway formulation must reach green gates in hours,
not days* — run by standing up a seventh solver row, `PulseSolver`
(piecewise-constant current, point matching, mixed potential), on the
shared layers the razor-grounds pilot (#398) built, and MEASURING what
they gave.

The physics is the oldest thin-wire scheme there is. That is the design of
the experiment, not a shortcoming of it: nothing in `src/momwire/pulse.py`
is novel, so everything the build cost is architecture.

Launched 2026-08-18T00:07:00Z.

---

## 1. Wall clock

| milestone | UTC | elapsed |
|---|---|---|
| brief | 00:07 | — |
| shared layers read (`_potential_ground`, `_capabilities`, `_element_currents`, `_cancel`, `razor.py`, the four test idioms) | 00:14 | 7 min |
| `pulse.py` written, first dipole solved | 00:18 | 11 min |
| formulation diagnosed and the Δ/a floor found | 00:21 | 14 min |
| solver committed | 00:22 | 15 min |
| all 33 gates green, committed | 00:27 | 20 min |
| full default lane (3438 passed / 5 skipped / 61 deselected — baseline + exactly the 33 new tests) | 00:36 | 29 min |
| `-m memgate` 47 green, ruff clean | 00:38 | **31 min** |

**Hours, not days — 31 minutes, of which 9 were the two certification
lanes running.** The honest
qualifier: the elapsed time above is one agent working from a brief that
had already named the formulation, the four layers to consume, and the
gate list. A cold start would add the design conversation, not the
implementation.

## 2. Line counts

`src/momwire/pulse.py`, 729 physical lines: **363 code**, 284 docstring,
35 comment, 47 blank. Split by what the lines are *for*:

| block | code lines | share |
|---|---|---|
| **the fill — all of the physics** (`_seg_M0`, `_point_g`, `_charge_stencil`, `_source_block`, `_assemble_Z`, including BOTH grounds) | **59** | 16 % |
| **refusals + the capability declaration** (`_OUT_OF_SCOPE`, `_PER_WIRE_RADIUS_REFUSAL`, `capabilities`, `_check_ground_clearance`) | **77** | 21 % |
| **plumbing** (`__init__`, `_build_geometry`, `_feed_basis_indices`, `compute_impedance`, `compute_y_matrix`, `currents_at_knots`) | **214** | 59 % |
| imports, module constants, class scaffolding | 13 | 4 % |

Gates: 867 physical / 427 code lines across three new test modules
(`test_pulse.py`, `test_pulse_ground.py`, `test_pulse_capabilities.py`).

**The headline is the ratio.** A complete new formulation — free space,
the PEC image and a reflection-coefficient ground — is **59 lines of
physics**. Saying no to the seven axes it does not serve costs **more than
the physics does** (77 lines). And the plumbing, which is neither, is
almost four times the physics.

## 3. Consumed UNCHANGED

Nine things, and not one edit to any of them. The whole diff outside
`pulse.py` and the three test files is a **two-line** `__init__.py` export.

| layer | what it gave | what it cost |
|---|---|---|
| `_capabilities.Capabilities` | the declared row, `refusal()` for every cell | 16 lines, one class attribute |
| `_cancel._Cancelable` | `CancelToken` support at the fill seams | 1 assignment + 4 `_checkpoint()` calls |
| `_element_currents._ElementCurrents` | `element_currents()` — midpoints, moments, nodes, continuity steps, `subdiv` resampling | one `currents_at_knots` (22 lines) |
| `_potential_ground.potential_ground_for` | the factory; free space is `None`, structurally absent | 1 call |
| `ImageGeometry.mirror_positions` / `.mirror_tangents` | the mirror map — **unit 2's generalisation held**: a THIRD consumer needed no further move | 3 lines |
| `PotentialGround.weight_windows()` | `(w_A, w_Φ)` per pair, for **both** grounds | 1 call, one full-width window |
| `PotentialGround.mode` / `.image_coefficient` / `.remainder()` | the fold/compose contract, read as vocabulary | asserted, not branched on |
| `_quadrature.leggauss` | memoized GL nodes | 1 call |
| `_ground_refl` | *never imported* — the Fresnel arithmetic stayed entirely behind `weight_windows` | 0 |

### 3.1 The predicted finding landed: `weight_windows` is the N² surface

momwire#416 predicted a dense pulse fill would want "the N² `weight_tables`,
the surface razor could NOT use, completing the trunk's coverage story".
Half right, and the correction is the interesting half.

A dense point-matched fill does want the whole-geometry pair — but it gets
it from `weight_windows()(0, n_segs)`, a full-width WINDOW, not from
`weight_tables()` (see §4.2). And because PEC's window is literally
`(t_m·M t_n, 1)`, **PEC and refl-coef reach the identical eleven lines of
`_assemble_Z` with no branch between them**. The `weight_tables() is None`
tell that `bspline._ground_finite_Z` branches on to reach a different
kernel has no analogue here, because this fill has only one kernel.

That is the trunk's coverage story completed, with one amendment: the
surface that completes it is `weight_windows`, and `weight_tables`'
distinct existence is a B-spline kernel-selection detail rather than a
consumer-facing choice.

## 4. What would have needed generalisation — recorded, not made

Per the pilot's rule and the brief, the need is recorded here rather than
taken, because razor.py's owner was mid-flight.

### 4.1 Sommerfeld: `Remainder.evaluate(supp_seg, polys)` — the primary finding

This is the signature that blocked the third ground, and it blocked it
**twice over**:

```python
def evaluate(self, supp_seg, polys):
    return self._solver._Z_sommerfeld_remainder(
        self._geom, supp_seg, polys, self._eps_tilde
    )
```

1. **The arguments are a B-spline basis description.** `supp_seg` is
   `(n_basis, degree+1)` — the segment under each wing — and `polys` is
   `(n_basis, degree+1, degree+1)` per-wing polynomial coefficients in the
   segment-local coordinate. A pulse basis *can* spell this: degree 0,
   `supp_seg = arange(N)[:, None]`, `polys = ones((N, 1, 1))`. So argument
   shape alone is **not** the blocker.
2. **The return value is a finished GALERKIN block.** `Q[m, n]` is the
   remainder field already integrated against the basis on both sides.
   A point-matched consumer needs the remainder FIELD at chosen
   observation points — one centroid per row here — and would have to
   *undo* a Galerkin integration to get it. There is no way to spell that
   request through this signature at any degree.

There is also a third, structural observation: `Remainder` holds no shared
code. Its one method delegates to `BSplineSolver._Z_sommerfeld_remainder`,
which reads `self.degree`, `self.n_qp_sommerfeld` and `geom["h_per_seg"]`.
A second consumer inherits the *vocabulary* (`mode`, "compose", the
`C₂·img + Q` association) and none of the implementation.

**The signature that would have served**, and it is the shape the sibling
trunk already uses (`_field_ground`, momwire#392):

```python
def evaluate(
    self, obs_points, obs_tangents, src_seg_l, src_seg_r, src_tangents
) -> "(n_obs, n_src) complex":
    """The smooth remainder FIELD, projected on the observer tangents."""
```

— i.e. hand back the tested field and leave the TESTING RULE to the
consumer. `bspline` would then apply its own Galerkin quadrature (it
already builds the nodes and moment weights it would pass), `razor` its
path integral, and `pulse` a single centroid evaluation. Note that
`_sommerfeld.remainder_field_proj(obs, t_obs, src, t_src, gz, k, grid)` —
which `_Z_sommerfeld_remainder`'s own numpy fallback calls — is *already*
that function. The generalisation is not new physics; it is exposing the
layer that exists one level down. **This is the input momwire#398 unit 5
asked a second consumer for.**

Recorded in the refusal message itself (`pulse._OUT_OF_SCOPE["ground_model"]`)
and pinned by
`tests/test_pulse_ground.py::test_sommerfeld_refusal_names_the_signature_that_blocked_it`,
so it cannot decay into a comment.

### 4.2 `weight_tables()` is not consumable by a non-B-spline solver

Its refl-coef branch is:

```python
solver = self._solver
return solver._image_refl_weights(solver._image_refl_prep(self._geom), self._omega)
```

Both are **`BSplineSolver` methods**, so a second consumer gets
`AttributeError: 'PulseSolver' object has no attribute '_image_refl_weights'`.
This is a leak in the direction the module's own docstring warns about for
budgets: `_image_refl_prep` exists to CACHE the k-independent specular
tables, which is schedule — but the weight computation rode out of the
shared layer with it.

The fix is one honest move and no new concept: `_image_refl_prep` /
`_image_refl_weights` read only `geom["seg_l"] / ["seg_r"] / ["tangents"]`
and `solver.ground_z / .ground_eps / .ground_phi_mode / .eps`, all of
which `PotentialGround` already holds or already reads. Move the bodies
into `_potential_ground` and leave `BSplineSolver`'s pair as the cache
wrapper the sweep wants. Not taken here: `weight_windows()` served, and
the brief's rule was to record.

Pinned by `test_weight_tables_is_not_consumable_by_a_non_bspline_solver`,
which should be **deleted** when that move lands, not loosened.

> **Landed 2026-08-18 (momwire#429 unit 1, the sharing audit's rank 2).**
> The move is exactly the one sketched above: `_potential_ground.specular_prep`
> and `_potential_ground.refl_weight_tables` are module functions there,
> `BSplineSolver._image_refl_prep` is the cache wrapper the sweep wants and
> is handed to `weight_tables(prep=…)` by its own fill, and
> `weight_tables()` called bare is self-contained. The test above was
> deleted, not loosened, and replaced by
> `test_weight_tables_are_served_on_a_non_bspline_solver`, which pins the
> tables equal to the full-width `weight_windows` on this solver.

### 4.3 The reduced-kernel segment moment has no home

`_seg_M0` needs `∫_seg exp(−jkR)/(4πR) dl'` split into a closed-form
static part and a smooth remainder. That is not this formulation's idea —
it is every mixed-potential solver's — but it has no shared module, so
this row imports `razor._axis_frame` and `razor._static_axis_moments`:
**module-level privates of a SIBLING SOLVER**, reached across a layer
boundary that does not exist.

Copying them instead would have hidden the finding and added ~25 lines to
§2's "physics" column dishonestly. `_static_axis_moments` also returns the
tent basis's first moment `m1`, which a pulse discards — the helper is
shaped for its first consumer too, mildly.

Suggested home: a `_kernel_moments.py` beside `_quadrature.py`. Filed as a
follow-up rather than taken.

### 4.4 The largest single cost is constructor plumbing, and it is duplicated

`PulseSolver.__init__` is **112 code lines — 31 % of the whole module** —
and `difflib` says **88 % of them (99/112) are line-identical to
`RazorSolver.__init__`**: the `wires` / `n_per_edge_per_wire` / `nsegs`
normalisation, the `feeds` tuple validation, the scalar-radius check, the
c/freq/ω/k derivation, the `**unsupported` → `_OUT_OF_SCOPE` dance.

There is a shared layer for grounds, for readout, for cancellation and for
the capability declaration. There is none for *"what is a wire, and where
is the feed"* — and that, not the physics and not the ground, is where a
new row's lines actually go. `_build_geometry` repeats the story more
mildly (28 of 45 lines shared with razor's, the rest genuinely
basis-specific).

This is the sharpest architectural recommendation the probe produced.

## 5. Measured results

### 5.1 PEC image — the exact mirrored-twin oracle

A dipole at h over PEC against that dipole plus its explicit mirror at −h
driven −1 V, N = 24, on this row's own discretization:

| quantity | value |
|---|---|
| grounded | 58.885290 − 570.092451j Ω |
| twin | 58.885290 − 570.092451j Ω |
| **relative agreement** | **1.85 × 10⁻¹⁴** |
| free space, same wire | 52.887 − 594.252j Ω (24.9 Ω away) |

Roundoff-exact, as the `[[A, B], [B, A]] ⇒ (A − B)c = e` argument
requires. For this scheme the block identities are exact rather than
approximate, because every ingredient of a row — `h_m`, `t_m·t_n`, every
kernel distance — is invariant under the reflection.

### 5.2 Free-space convergence to `BSplineSolver`

14 MHz half-wave dipole (L = 10.1715 m, a = 20 mm), reference
`BSplineSolver(degree=2)` at 400 segments = 72.445 + 1.125j Ω:

| N | Δ/a | Z (Ω) | \|Z − Z_ref\| | ratio |
|---|---|---|---|---|
| 32 | 15.9 | 54.202 − 416.352j | 417.88 | — |
| 64 | 7.9 | 60.202 − 155.668j | 157.27 | 2.66 |
| 128 | 4.0 | 67.682 − 40.672j | 42.07 | 3.74 |
| 256 | 2.0 | 71.919 − 3.575j | 4.73 | 8.89 |
| 512 | 1.0 | 72.734 + 1.101j | **0.29** | 16.3 |

Monotone at every rung, with the improvement RATIO growing. Pinned at
"better than 2.5× per rung" plus 0.6 Ω absolute at N = 512.

### 5.3 The scheme's own floor: Δ/a, not Δ/λ

The finding the probe did not expect. This basis's charge is two POINT
charges observed at their own location, regularised only by the reduced
kernel's `a²`, so the concentration error is a function of **Δ/a**.
Refining past Δ/a ≈ 1 does not stop helping — it **reverses**. On a 50 mm
conductor, reference 79.064 + 5.491j Ω:

| N | Δ/a | Z (Ω) | \|Z − Z_ref\| |
|---|---|---|---|
| 64 | 3.2 | 73.092 − 10.251j | 16.84 |
| 128 | 1.6 | 77.229 + 10.258j | 5.11 |
| 512 | 0.4 | 37.630 − 39.948j | **61.49** |

momwire#248's "never validate below Δ/a ≈ 1" is a comparison convention
elsewhere; here it is a property of the formulation. Every ladder above
therefore stops AT Δ/a ≈ 1, which is also where this scheme is sharpest.

### 5.4 Reflection-coefficient ground

Same dipole at 0.25 λ over ε = 13 − 0.03j, reference
`BSplineSolver(degree=2)` at 200 segments = 81.535 + 16.195j Ω:

| N | Δ/a | Z (Ω) | \|Z − Z_ref\| |
|---|---|---|---|
| 256 | 2.0 | 81.252 + 11.657j | 4.55 |
| 512 | 1.0 | 82.643 + 16.266j | **1.11** |

Part of the residue is the bar's: the reference itself moves 0.45 Ω
between its own 200- and 400-segment answers over this ground. Pinned at
2 Ω plus a 3× improvement across the rung.

Structural: `test_the_fill_follows_the_ground_object_not_the_strings`
hands a refl-coef-configured solver a PEC `PotentialGround` and gets the
PEC matrix **bit for bit** — the fill reads the object, and it reads
nothing else.

### 5.5 Reciprocity — measured and attributed, not forced

Point matching does not give a symmetric Z, and this row does not
symmetrise it. Asymmetric bent-plus-parasitic deck,
`max|X − Xᵀ| / max|X|`:

| segments/wire | charge term ΔΔg | vector term h·M0 | assembled Z |
|---|---|---|---|
| 12 | 5.9 × 10⁻¹⁷ | 2.19 × 10⁻⁵ | 5.14 × 10⁻⁸ |
| 24 | 1.2 × 10⁻¹⁶ | 4.19 × 10⁻⁶ | 4.02 × 10⁻⁹ |
| 48 | 6.8 × 10⁻¹⁷ | 7.36 × 10⁻⁷ | 2.81 × 10⁻¹⁰ |

The charge term is symmetric **to roundoff, exactly** — `g` is symmetric
and the m↔n swap transposes the four-point endpoint stencil. The whole
reciprocity error is the vector term: `h_m M0[c_m, n]` is the midpoint
rule on one side of a pairing Galerkin testing would integrate on both.
It decays faster than O(h²). The assembled Z looks far better than the
vector term because the norm is dominated by the (exactly symmetric)
charge diagonal — which is why the table reports all three.

### 5.6 Junctions cost nothing, and what they do not enforce

Coincident wire ends put their endpoint charges on the **same point**, so
they superpose by arithmetic. There is no junction detection in this
module and no `junctions=` spec.

* **Colinear split identity**: one wire of 96 segments vs two wires of 48
  meeting at the midpoint — 3.0 × 10⁻¹⁵ relative in Z, 5 × 10⁻¹⁵ in the
  coefficients. The same matrix, not an agreement.
* **What is NOT enforced**: current continuity at the node. The basis
  leaves a point charge there, so `Σ I` out of a 3-wire junction is
  jω·Q_node — a real O(h) artefact. Measured, relative to the largest
  branch current: 5.18 × 10⁻², 2.33 × 10⁻², 9.81 × 10⁻³, 4.25 × 10⁻³ at
  N = 8, 16, 32, 64. Clean first order; not claimed to be zero.

A free wire end is the K = 1 case: the terminal pulse's current does not
vanish (5.4 % of peak at N = 64) and `currents_at_knots` reports it rather
than zeroing it.

## 6. The probe answer

**Does this architecture put a new basis within "hours not days"? Yes —
measured at 20 minutes to green gates, 45 to a green full lane.** With
three qualifications that are the actual content of the finding:

1. **The layers that were designed to be shared, were.** Two grounds, the
   field readout, cancellation and the capability row arrived for 20-odd
   lines and zero edits to any shared file. `ImageGeometry`'s unit-2
   generalisation held for a third consumer with no further move — the
   pilot's central claim survived its first independent re-test.
2. **The layers that were NOT designed are where the time went.** 59 lines
   of physics against 214 of plumbing, 88 % of the constructor duplicated
   from a sibling, and a reduced-kernel moment imported across a layer
   boundary that does not exist. If the next probe is to be faster, that
   is the only place left to take the time from.
3. **Two shared surfaces are shaped for their first consumer**, and a
   second consumer is how you find that out: `weight_tables()` raises
   `AttributeError` outside `BSplineSolver`, and `Remainder.evaluate`
   returns a Galerkin block to a point-matching caller. Neither was
   generalised here — both are recorded above with the exact signature
   that would serve, and both are pinned by tests that should be deleted
   when the moves land.

## 7. Follow-ups worth filing

1. ~~Move `_image_refl_prep` / `_image_refl_weights` bodies into
   `_potential_ground` so `weight_tables()` serves any consumer (§4.2).~~
   **Done 2026-08-18**, momwire#429 unit 1 (filed as momwire#424).
2. Re-shape `Remainder.evaluate` to the field-trunk signature — the
   remainder FIELD at given observers — feeding momwire#398 unit 5 (§4.1).
3. A `_kernel_moments.py` for the reduced-kernel segment moment, retiring
   `pulse.py`'s import from `razor.py` (§4.3).
4. A shared constructor helper for `wires` / `n_per_edge_per_wire` /
   `feeds` normalisation — the largest duplicated block in momwire's
   solver rows (§4.4).
5. Merge `tests/test_pulse_capabilities.py` into `tests/test_capabilities.py`
   as its § 2.5; it is separate only because #416 was built alongside an
   in-flight razor unit.
6. Ground CONTACT for `PulseSolver`: the basis looks structurally ready
   (a contact node's real and image charges coincide and cancel), but
   nothing here measures it, so it is refused.

## 8. Correction (momwire#430, 2026-08-18): §4.1's blocker is resolved

Section 4.1 above is left exactly as it was written — this is an appended
correction, not a rewrite, because a refusal message is a physics finding
on the record and the record should show what was true and when.

§4.1 was right about the mechanism and wrong about its shelf life. #398
unit 5 landed one week after this probe, and it landed the EXACT signature
§4.1 asked for: `Remainder.field_windows(observers, sources, n_moment=…)`,
hanging the remainder FIELD back at chosen observation points instead of a
finished Galerkin block. `n_moment = 1` — the zeroth arc moment of a
source segment, i.e. weight 1 over the segment — IS this basis's own
shape, exactly as §4.1 predicted a generalised signature would serve. So
the refusal `_OUT_OF_SCOPE["ground_model"]` pinned, and the test that pinned
it (`test_sommerfeld_refusal_names_the_signature_that_blocked_it`), were
asserting a blocker that had already been removed from underneath them —
which is what momwire#430 filed against.

**momwire#430's resolution was (a): grant the capability**, not (b):
reword the refusal. `pulse.py` now consumes `field_windows` the same way
`razor.py` does — Q, the smooth Sommerfeld remainder field, added to the
image block as `C₂·img + Q` before `_assemble_Z`'s single global minus,
`mode == "compose"`'s contract, unchanged — with zero new branch for the
C₂-scaled exact-image half (it rides the same `weight_windows` pair PEC's
mirror table already uses). `PulseSolver.capabilities.grounds` gained
`"sommerfeld"`; the refusal and its test are gone, not reworded.

Measured (2026-08-18), on this row's own Δ/a ≈ 1 floor (§5.3):

* **ε̃ → 1 is free space, bit for bit** (`array_equal`, not `allclose`):
  the exact-image half and Q both vanish structurally at C₂ = 0.
* **ε̃ → PEC decays at C₂'s own O(ε̃^{-1/2}) rate**, independent of N:
  |Z(ε̃) − Z_PEC| at ε̃ = 10, 10², 10³, 10⁴, 10⁶ on the dipole@0.25λ deck,
  N = 64: 14.677, 5.626, 1.898, 0.613, 0.0618 Ω — the same shape
  `tests/test_razor_sommerfeld_ground.py` measured on a different
  formulation entirely.
* **Q is alive at low height.** Refl-vs-Sommerfeld split on the
  dipole@0.04λ deck, this row's own N ladder up to its Δ/a ≈ 1 floor:
  20.366, 21.982, 22.932, 23.167 Ω at N = 64, 128, 256, 512 — tens of ohms,
  as unit 5 found — agreeing with `BSplineSolver(degree=2)`'s 23.126 Ω
  split at N = 512 to 0.042 Ω, well inside the 1 Ω this basis's slower
  convergence would excuse (razor agreed with `BSplineSolver` to 0.03 Ω).
* **The ground adds no cross-formulation gap at 0.25λ beyond this row's
  own free-space one**: |Z_pulse − Z_bspline(400)| widens by 0.089 / 0.422
  Ω at N = 256 / 512 when the ground switches from free space to
  Sommerfeld — the same order as this row's own free-space residual at
  that N, not a new source of error.
* **Structural**: the fill reads the ground OBJECT's `mode`, not
  `self.ground_model` — a composing `PotentialGround` handed to a
  refl-coef-configured solver (and the reverse) fills the composed (folded)
  matrix bit for bit; a ground that LIES about `image_coefficient` while
  its `weight_windows` / `remainder` stay honest moves nothing, because
  this consumer never applies the coefficient itself; `ground_phi_mode` is
  accepted and unread under Sommerfeld exactly as `BSplineSolver` and
  `RazorSolver` treat it.
* **Bit-identical to the pre-#430 branch point** on every path this unit
  did not touch: free space, PEC and refl-coef fills reproduce the branch
  point's matrices exactly (`array_equal`, 3/3 measured), and the full
  existing `test_pulse.py` / `test_pulse_ground.py` /
  `test_pulse_capabilities.py` suites stayed green except the two tests
  this correction replaced.

Gates in `tests/test_pulse_ground.py` §5 and `tests/test_pulse_capabilities.py`.
Follow-up 2 above (re-shaping `Remainder.evaluate`, the reason this section
exists) is CLOSED by #398 unit 5; the rest of §7's list is unaffected.

---

## 9. The sequel (momwire#557, 2026-08-22): what ACCURACY cost

§6 answered "what does a throwaway formulation cost when it reuses the
shared layers." It did not ask what the same formulation costs when it also
has to be *usable*, and the row it produced is not: `PulseSolver`'s error
is governed by Δ/a rather than Δ/λ, so on a thin wire it needs ~64× the
mesh of `BSplineSolver` to reach the same answer. `HarringtonSolver`
(`src/momwire/harrington.py`) is the second build, and the delta between
the two is the price of the accuracy.

### 9.1 The defect was a charge model, not the basis or the testing

Harrington (*Proc. IEEE* 55(2), Feb 1967) spreads the charge over a
half-shifted **dual cell** — his (95)/(96)/(100) — and averages the
Green's function over the source segment everywhere, his (99). The probe
row put the whole charge at a POINT and observed it at another point,
where only the reduced kernel's `a` keeps the self term finite. That single
substitution is the whole Δ/a dependence, and `docs/pulse_basis_d0_nodal
_charge.md` had predicted it two months earlier (variant 2, "~−700000/N j")
and named the dual cell as "the correct fix — not implemented".

Worth stating plainly because it cuts against the probe's own framing:
**§6's 31 minutes bought a row whose physics was already documented as
insufficient in this repo.** The gates it passed were real and remain
green — the row converges, monotonically, to the right limit. They simply
did not include an accuracy floor, because none of them compared the row
to a sibling at a mesh a user would choose.

### 9.2 What the second build reused

The dual cell needed **zero new numerics**. `_seg_M0` already integrates
the reduced kernel over an arbitrary segment (closed-form static moment +
GL remainder), and a half-segment is a segment with `h/2`, so every cell
average is that same call on a rebuilt piece geometry. Harrington's own
Appendix ladder — the (126) closed form, the (129) series, the (135)
multipole — is *not* implemented, and did not need to be: it is a 1967
answer to "we have no quadrature layer," and its purpose here is served
better by the layer that already exists. The resulting row converges
faster than the ladder does (the test docstring's ratio improves down the
ladder rather than sitting at 2×, because ψ is exact rather than two-term).

That is the sharpest form of the §3 claim the first build could make:
a formulation change deep enough to move the answer by three orders of
magnitude touched **one method**, `_charge_stencil`, and no shared layer at
all. The ground models came along untouched — the charge model is a
source-side substitution, so PEC / refl-coef / Sommerfeld needed no code in
the new file.

### 9.3 What it did NOT get for free

**Junction detection**, which §6 recorded as costing nothing. That finding
was true of the nodal row and is not transferable: coincident endpoint
POINT charges superpose by arithmetic, but dual cells have to be the same
region at a shared node or they leave a spurious dipole layer there. Two
cheaper cells were built and measured before the star cell was accepted:

- charge spread inside its own segment (no adjacency needed): **diverges**,
  error growing 201 Ω → 382 Ω over N = 25…801;
- each wire truncating its own cell at a junction: a colinear split
  disagrees with the unsplit wire by ~10%, and the disagreement **grows**
  with refinement.

So this row carries its own node map (union-find over structural knots,
geometric matching only at wire ends) and `deck/_solver.py` grew a
`_NO_JUNCTION_SPEC` tuple, because the junction fact had been welded to
`_NATIVE_LOADING` while RazorSolver was the only class with either
property. One shared-layer generalisation, forced by the second consumer —
which is exactly the shape §4 predicted such generalisations would take.

### 9.4 The two rows as an instrument

They differ in one ingredient. Same basis, same testing, same kernel, same
feed, same mesh — only the charge's support. On the thin deck at N = 64
that ingredient is worth 172×, and the pair now measures a third axis
beside the basis × testing matrix of Act V: **the support of the charge**.
Neither row replaces the other, and `PulseSolver` keeps its name, its
thesis and its gates.
