# FieldGround: the interface sketch (momwire#397)

The reviewed-before-code gate for criterion 1 (`solver-architecture.md`
§0.1). One page of interface, then the checks that page must survive. The
design constraint from §6.1 applies throughout: **designed against five
grounds, implemented for four** — the radial-wire screen must land later by
changing coefficient functions only.

## What the survey says the object is

Both sinusoidal solvers compose every ground from the same four ingredients,
each currently spelled twice:

| ingredient | point-matched spelling | Galerkin spelling |
|---|---|---|
| mirror map | `_image_source_centers_tangents` | *(inherited — already shared)* |
| per-pair weight | `_image_refl_prep` / `_image_refl_band` → dyad inside `_field_tensor_image_refl` | `_refl_projection` → projector arg of `_tested_contribs` |
| image coefficient | `np.multiply(c2, Φᵢ, out=Φᵢ)` in the band loop | the same line in `_fold_ground_block` |
| remainder | `_sommerfeld_remainder_prepare` / `_replay_...` | `_tested_sommerfeld_remainder` (streams the same evaluator) |

The **schedule** around these (bands vs folds, `subtract_into`, chunk
alignment) is *not* in the table and does not move — it is stage 2's
business, and a ~100-line shim legitimately stays per solver (§0.2).

## The interface

New module `src/momwire/_field_ground.py`. One small class per ground
model, one shared factory; a solver holds at most one instance per solve.

```python
def field_ground_for(solver, geom, k, omega) -> FieldGround | None
    # None ⇔ free space. The off path stays a structural no-op —
    # not one float op, same standard as extended_kernel=False.

class FieldGround:                    # frozen per (geom, k, omega)
    mode: Literal["fold", "compose"]  # the ONE scheduling fact declared here

    def image_sources(self) -> tuple[src_c, src_t]
        # The mirror map. EK eligibility still scores against these
        # mirrored sources via the solver's own mirror=True plumbing —
        # the ground provides geometry, never EK policy.

    def pair_weights(self, m_idx, n_idx) -> PairWeights | None
        # None ⇔ the plain tangential projection (PEC, and the sommerfeld
        # IMAGE part). Otherwise the per-(test, source)-pair Fresnel dyad
        # tables, at whatever pairing the caller names — the same
        # broadcast-agnostic index contract `_field_components_bcast`'s
        # projector already takes ((M,1)×(1,N) for a tensor build,
        # (P,1)×(P,1) for near pairs). ONE implementation of the
        # cosθ → ρv/ρh → dyad chain, replacing both spellings above.
        # k-independent prep cached inside; per-ω weights on demand
        # (the existing prep/weights split, kept).

    image_coefficient: complex        # 1.0, or C2(ε̃) for sommerfeld.
        # Contract, not convention: consumers MUST apply it as
        # np.multiply(coef, block, out=block) — coefficient on the LEFT —
        # per the measured one-ULP operand-order effect both solvers
        # already document. The docstring carries the #392-class rule:
        # evaluation order is part of this interface's value.

    def remainder(self) -> Remainder | None
        # None for PEC / refl-coef. For sommerfeld: the prepare/replay
        # pair. prepare() is observer-independent (built once per fill —
        # #357 item 1 stays); replay(obs_c, obs_t) evaluates a band. The
        # GALERKIN caller streams replay chunks through its own
        # test-integration fold exactly as today — chunk alignment
        # (row_group = nq) is the caller's schedule, not the ground's.
```

`mode` is the load-bearing declaration. `"fold"` (PEC, refl-coef, screen):
the solver may take the block through its single global minus —
`subtract_into`, band-wise `Φ -= Φᵢ` — entry by entry, in any order.
`"compose"` (sommerfeld): `coef·img − remainder` must be associated before
the fold; the solver's existing compose-first paths key off this instead of
off `ground_model` strings. `free − (c2·img − rem) ≠ (free − c2·img) + rem`
in float64 is the whole reason this field exists.

## The five-ground check

| ground | mode | pair_weights | coef | remainder |
|---|---|---|---|---|
| free | *(None — no object)* | | | |
| PEC | fold | None | 1 | None |
| refl-coef | fold | Fresnel dyad | 1 | None |
| sommerfeld | compose | None | C2(ε̃) | prepare/replay |
| **radial screen** | fold | **dyad with screen-modified ρv/ρh** | 1 | None |

The screen lands by giving `_ground_refl` a modified reflection-coefficient
function (screen surface impedance in parallel with the earth, per angle)
and passing it through `pair_weights` — **zero edits to either sinusoidal
file**, which is criterion 1's acceptance test (§0.2). A future
Sommerfeld-class ground (own remainder operator) lands as a fourth mode-
"compose" row — new physics in the object, still zero solver edits.

Measured at the end of unit 3, that claim holds of the **fill** and needs
two qualifications, both outside the object by design and neither of them
a ground DECISION:

* the **configuration surface** is still `SinusoidalSolver.__init__`'s —
  `ground_model`'s whitelist (`"refl-coef"`/`"sommerfeld"`) and whatever
  kwarg carries the screen's geometry. Accepting a new ground is an edit
  to `sinusoidal.py`; *filling* it is not. (`_capabilities`' ground
  enumeration is the same kind of edit, in its own file.)
* the **#282 contact-charge correction** calls `_ground_refl.fresnel_rho`
  directly on both solvers (`_contact_charge_fresnel`, and the
  `ground_model == "sommerfeld"` branches in the two charge kernels). A
  screen deck with a wire END IN THE PLANE would take the bare earth's
  ρ_v/ρ_h there. That is the testing-scheme correction this sketch puts
  out of scope, so it is not a broken promise — but it is a wrong answer
  rather than a missing feature, and whoever lands the screen has to route
  those coefficients through the ground or refuse ground contacts under it.

## Amendments from the build (units 1-3)

The page above is the approved sketch; these are the four places the
implemented object differs from it, each recorded here rather than
discovered later. 1-3 came out of units 1-2 and were binding on unit 3; 4
is unit 3's own.

1. **The dyad has a third spelling, so the fused kernels need a key.**
   `sinusoidal_field_tensor_refl` and its EK twin compute ρ_v/ρ_h
   *in-kernel* from ε̃ and cos θ — the survey's two spellings were three.
   The fused kernels therefore stay an optimization keyed to
   standard-Fresnel grounds: `FieldGround.standard_fresnel` (True for all
   four grounds momwire ships) gates the backend dispatch inside
   `_field_tensor_image_refl`, and a coefficient-modified ground — the
   radial-wire screen — sets it False and is served by the numpy
   `PairWeights.project` path with zero solver edits. `pair_weights` is
   therefore DATA (unit 1's `PairTables` / `PairWeights`) and not a
   sealed operator.
2. **`pair_weights` takes an observer window, not a pairing.** Unit 1
   split the sketch's one method in two along the existing prep/weights
   line: `FieldGround.pair_weights(obs_rows)` hands back the k-independent
   `PairTables` for a window (`None` = the whole geometry), and
   `PairTables.weights(ε̃).project(cm, m_idx, n_idx)` is the sketch's
   pairing-agnostic projector. The point-matched fill asks per band and
   caches nothing; a consumer that wants the whole-geometry table cached
   per geometry — the Galerkin one — owns that as part of its schedule,
   as `_image_refl_prep` does today.
3. **`image_field` is on the object.** "Which image tensor does this
   ground build" is the decision the band loop used to make from strings,
   so it belongs with the other three; it delegates to the solver's own
   `_field_tensor_image` / `_field_tensor_image_refl`, which keep the
   band, the mixed-radius runs and the backend choice. `remainder` also
   takes `cos_shape` (`"cos"` point-matched, `"cos-1"` Galerkin) and
   `Remainder.replay` forwards `consume` / `row_group` untouched, so the
   streaming schedule stays the caller's.

4. **The weight row has a second spelling, so the object serves a
   projector too — and `plain_projection` moved with it.** The Galerkin
   fill does not build an image tensor; it hands a *projector* to its test
   integration. `FieldGround.projector(tables)` is therefore the same
   ternary as `image_field`, returning `_field_ground.plain_projection`
   (moved here from `sinusoidal_galerkin`, because "this ground has no
   dyad" is what it *returns*) or `PairTables.weights(ε̃).project`. Two
   consequences worth having written down: the identity of that plain
   projector is load-bearing — `_tested_contribs` gates its fused C++ far
   fill on `projector is _plain_projection`, so the solver binds the name
   rather than defining a second function — and `tables` is a zero-argument
   SUPPLIER, not a table, so the ground can decline to call it and an
   unweighted fill never builds the O(N²) specular quintet. That is what
   let the Galerkin solver keep its whole-geometry `_image_refl_prep` cache
   as schedule (option one of the unit-2 finding) without the object
   growing a `cached` flag. Unit 3 also took `_ek_bracket_correction_
   tested`, which was not in its brief but mirrored the fold's string
   dispatch exactly; its three scales collapse to `−fg.image_coefficient`.

## What deliberately stays out

- **Scheduling** — bands, folds, budgets, `subtract_into`: stage 2.
- **EK policy** — `_ek_pairs`' mirror-eligibility stays solver-side; the
  ground only supplies the mirrored geometry it scores against.
- **`_node_charge_image_pair_block`** — SG's #151 ground-connected node
  charge is a *testing-scheme* correction that happens to involve the
  image; it stays in SG for now and is named here so its later migration
  is a decision, not a discovery.
- **`PotentialGround`** — the bspline/razor twin follows in #398 with its
  own `(w_A, w_Φ)` weight shape; only the *factory* and the mode contract
  are shared vocabulary. §2.2 of the architecture doc says why they never
  merge.

  > **Amended 2026-08-18 (momwire#429 unit 2, the sharing audit's rank 5
  > and correction 3).** "Only the factory is shared vocabulary"
  > understated it: the two factories turned out to be **82 % literally
  > identical**, which is an extraction and not a vocabulary. The shared
  > part — free / PEC / refl-coef fold / Sommerfeld compose, and the
  > `(mode, eps_tilde, image_coefficient, standard_fresnel)` each implies
  > — now lives in `_ground_spec.ground_config`, which both factories
  > consume; each keeps only its own object construction, plus
  > `ground_phi_mode` on the potential side. The classes still never
  > merge, for §2.2's reasons, unchanged.

## Migration units and gates (the stacked arc, once this is approved)

1. **Unify the Fresnel pair tables** — one builder behind `pair_weights`,
   consumed by both existing call sites. Gate: both solvers' grounded fills
   **bit-identical** on the existing suites (the refl-coef gates, G-D8a,
   the banded-assembly and folded-ground bit tests).
2. **`FieldGround` + point-matched consumption** — `_assemble_Z`'s band
   loop reads the object; `ground_model` string dispatch collapses to
   `mode`. Gate: bit-identity, all grounds, plus the Sommerfeld replay
   bit test.
3. **Galerkin consumption** — `_fold_ground_block` keeps its fold order
   and `subtract_into` schedule but loses its physics to the object.
   Gate: bit-identity (`test_folded_ground_is_bit_equal...`,
   `test_streamed_remainder_is_bit_equal...`), and the schedule shim that
   remains is measured in lines and reported in the PR. **Landed**: 61
   lines of ground code left in `sinusoidal_galerkin.py`, all schedule —
   the fold (23), the streamed remainder (23), the EK bracket's image row
   (11), the fill's entry (3) and the cache supplier (1), against 84
   before — plus the 7-line plain projector, which moved to
   `_field_ground` and is a one-line binding here now. Bit-identical over
   34 arrays (three decks × four grounds × two kernels, the streamed
   remainder at two forced chunk sizes, the ported PEC path).

Each unit is delegable once this sketch is approved; the arc lands as one
integration PR per the house stacked-arc pattern.

## The risks this sketch is exposed to

- §8 risk 1 (the two grounds are three): if unit 3 finds the Galerkin fill
  needs a *different* `pair_weights` contract than the tensor build — not
  just a different pairing — the shared-builder claim fails and this page
  gets amended before unit 3 lands, not after. **Did not fire.** The
  Galerkin fill needed a different *shape* of answer (a projector, not a
  tensor) but the same contract underneath: `PairWeights.project` serves
  both, and unit 3 added no table, no ρ and no pairing rule. What it did
  add is amendment 4's supplier callable, which is a scheduling
  concession, not a contract change.
- The `compose` mode currently has exactly one member; if the screen turns
  out to need remainder-like structure (it should not — it is coefficient-
  level by construction), `mode` was the wrong axis and should become a
  capability of the remainder instead.
