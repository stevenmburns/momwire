# momwire#1029 phase 1: where this branch stands

Written 2026-09-16 at Steve's request to bring the work to a stopping point.

**Phase 1 has no code.** The scope question was raised and answered, option A was
approved, and the design below is read off the source. Nothing is implemented,
and **no gate has been run**. Phase 0 is complete and unaffected.

## The branch

- `feat/1029-rotational-symmetry`, based on momwire 227491d.
- **97e0c13** — phase 0's registration, before any feasibility measurement.
- **8d42ea7** — phase 0's records and `README.md`.
- This commit adds only this file.
- The antennaknobs submodule pointer has **not** been moved by me.

**The base is current.** momwire origin/main is a925d37, which is the v0.56.0
tag commit, and 227491d is an ancestor of it. The only `src/` change between the
two is the eznec printout work (`eznec/_printout.py`, `eznec/_shell.py`); nothing
in the buried, crossing or bspline fill code. So phase 0 measured on a base that
already carries U9's several crossing nodes and #1074, and a rebase onto
a925d37 would change nothing this work reads.

**One baseline note for the records:** antennaknobs v0.79.0 moved the pin triple
to momwire 0.56.0 (that same a925d37). "momwire ahead of the pointer" now means
ahead of 0.56.0, not 0.55.0.

**Baseline moved again, later the same day.** momwire origin/main is now
**a6a67f93a** — U8's far field (momwire#1078) and LD 5 in the nec5 seam (#1086).
227491d is still an ancestor of it. What moved under `src/` since this branch's
base: `_far_readout.py` (+252), `deck/_nec5.py` (+216), `eznec/_printout.py`,
`eznec/_serve.py`, `eznec/_shell.py`, `portal/_portal.py`, and `_medium_spec.py`
(−37). **The files this phase-1 design reads — `bspline.py`,
`_below_interface.py`, `_crossing_fill.py` — are unchanged.**

**On resuming:** rebase on a6a67f93a and rebuild (`make build`), then re-check two
things before trusting the notes above, because both moved underneath them:

- `_medium_spec.py`'s delta, against the media labelling the buried fill's pair
  classes use;
- the far-field path note at the end of this file, since U8 rewrote
  `_far_readout.py`.

## Phase 0, for context (complete, measured)

At 4 and 12 radials on `verticals.buried_radial_vertical`, soil 13/0.005, bs2
degree 2: the radial blocks are block-circulant to 6.8e-19 and 2.2e-15; the mast
couplings to 2e-19 and 8e-18; a per-harmonic solve reproduces the dense one to
2.2e-13 and 5.3e-13 on Z_in and on the currents; an on-axis feed puts exactly zero
into harmonics h ≠ 0. Graded entry by entry, the worst non-invariance is 1.7e-9 on
entries of about 2e-5 Ω. The cost model predicts 1.18–2.54 s at 48 radials,
3.13–11.12 s at 120 and 5.54–21.0 s at 150, against #1067's momwire warm column
of 15.28 / 84.54 / 162.02 s and 3b75639's 2.82 / 18.67 / 30.14 s.

## The approved scope (option A), and the design as read

`rows=None` on the shared buried entry, meaning exactly today's path.

**Where a row restriction has to go** (file:line at 227491d):

| what | where | state |
|---|---|---|
| the shared routing, which has no observer restriction today | `_below_interface.compute_Z_operator_buried` — `_below_interface.py:949` | needs the new optional `rows=None` |
| the field-form blocks, including the below-remainder projection (56 % of warm self time at 150 radials) | `_field_galerkin_block(..., obs_idx, src_idx, obs, t_obs, ...)` — `bspline.py:5023`, with nodes from `_buried_nodes(geom, seg_idx, *, q_factor=1)` — `bspline.py:5003` | **already observer-parameterised**; the routing passes a narrowed `obs_idx` |
| the mixed-potential direct and image blocks | `_build_J_blocks_subset(geom, k, seg_idx, mirror_sources=False)` — `bspline.py:5144`; `_accumulate_Z_subset_chunked(Z, geom, k, seg_idx, supp_seg, polys, *, mirror_sources, eps, scale, weight=None)` — `bspline.py:5243` | one optional observer subset each, bspline-side |
| the assembly itself | `_assemble_Z` — `bspline.py:3348`; `_image_Z_weighted` — `bspline.py:2654` | **no change**: zero observer rows in the moment tensor give zero Z rows |

**So the answer to "did it stay inside one optional argument?"** — by design, one
optional argument on the shared entry plus one on each of two bspline-side fills,
and no change to the assemblers or to `BuriedFills`' shape beyond those two
callback signatures. That is not a three-layer thread. **It is a reading of the
code, not an implemented result.**

**A constraint the implementation must respect:** `rows` may change only which
observer rows are computed. It must not reach `plan_buried`'s decisions — the
Sommerfeld grids, the quadrature orders, `pair_extents`, `near_q_factor` — which
stay computed from the full observer sets. Otherwise S-B (the restricted fill
equals the corresponding rows of the full fill) fails by construction rather than
by a bug.

## Gate table

| gate | what it checks | result |
|---|---|---|
| **S-A** | `rows=None` is bit-identical on non-symmetric buried decks (the default `buried_radial_vertical` convention, a multi-node crossing deck) | **not run** — the parameter does not exist |
| **S-B** | the restricted fill equals the corresponding rows of the full fill, entry by entry | **not run** |
| **S-C** | sinusoidal-Galerkin's buried path is bit-identical and never passes `rows` | **not run** |
| **S-D** | `make test` green plus momwire CI | **not run** |
| **G1** | Z, currents and far field against dense at 4 / 12 / 48 radials, plus the entry-by-entry check at 48 | **not run.** The nearest existing evidence is phase 0's solve-level agreement at 4 and 12 (2.2e-13 and 5.3e-13), which went through a dense fill and a permutation, not through a route, and covered no far field and no 48-radial cell |
| **G2** | six refusals by name: a radial 1 % longer; a radial off its sector angle; an off-axis feed; mixed materials between radials; a non-axisymmetric ground; a tilted axis | **not run**, and the sentences are not drafted |
| **G3** | the cost ladder at 12 / 24 / 48 / 96 / 150 radials with RSS, beside dense and #1067's NEC-5 columns | **not run.** The registered bars stand: ≤ 30.14 s at 150, ≥ 5× over dense, ≤ 3.0 s at 48 |
| **G4** | bit-identity on non-symmetric decks and on the default path | **not run** |

## What is left, in order

1. **Register phase 1** before any measurement: the route's placement, the `rows`
   contract (what None means, what a restriction means, and that a caller
   restricting rows owns the consequences downstream), the six G2 refusal
   sentences, the far-field path, and predictions with bars for G1 and G3.
2. **Implement**, smallest surface first: the `rows=` parameter and its S-A to S-D
   gates, run **before** the sector route is wired up; then the opt-in
   `rotational_symmetry=True`, the symmetry check with by-name refusals, the
   sector fill and the harmonic-0 solve.
3. **Gate** in order: S-A to S-D, then G1, G2, G3, G4 plus `make test`.
4. **PR to momwire** when the gates are green. Steve merges.

## Notes carried forward

- **The far field needs no second path.** antennaknobs' `far_field` takes only
  `coeffs`, through `currents_at_knots`, so reassembled harmonic coefficients feed
  it directly — which is what G1 asks for.
- **An on-axis feed needs only the harmonic-0 block** K₀ (90 × 90 on this deck),
  whatever N is. Phase 0 measured harmonics h ≠ 0 at exactly zero.
- **`elt_whip` is not ladderable** (4392 segments, a mesh that barely responds), so
  it will not appear in any convergence claim here.
