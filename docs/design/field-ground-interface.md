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
   remains is measured in lines and reported in the PR.

Each unit is delegable once this sketch is approved; the arc lands as one
integration PR per the house stacked-arc pattern.

## The risks this sketch is exposed to

- §8 risk 1 (the two grounds are three): if unit 3 finds the Galerkin fill
  needs a *different* `pair_weights` contract than the tensor build — not
  just a different pairing — the shared-builder claim fails and this page
  gets amended before unit 3 lands, not after.
- The `compose` mode currently has exactly one member; if the screen turns
  out to need remainder-like structure (it should not — it is coefficient-
  level by construction), `mode` was the wrong axis and should become a
  capability of the remainder instead.
