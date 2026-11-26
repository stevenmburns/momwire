# Stage 0 survey, part B: the ground terms (momwire#2063 stage B; absorbs #2159, #2074)

Read-only survey. Tree: `reuse-survey-stage0` = main at v0.98.0 (`ed479892`).
The failed attempt is `bs2-image-reuse`. Its own three commits are
`7e100fbf..534b4b6e`: `48d2151e` (the copy), `0d375d16` (the report), `534b4b6e` (the tests).
Line numbers are main's unless marked `br:` (that is `git show 534b4b6e:<path>`).
Nothing was run. No timings are new: every number below is quoted from #2159,
#2063, the branch's commit messages, or AK `scratch/yagi-uniform-cores-1124/skylake/raw.jsonl`.

Notation: P is an observer point or segment and Q a source. M is the mirror in the
plane z = gz, and h = z − gz. d is the horizontal separation, and hs = h_P + h_Q.
"Plane group" means horizontal translations, rotations about a vertical
axis and reflections in vertical planes.

---

## 1. Per basis and per ground kind: today's state

### 1a. bs2 (`bspline.py`, `_bspline_fill_core.cpp`)

| | PEC image | refl-coef image | Sommerfeld: exact image (C2) | Sommerfeld: remainder | buried / transmitted |
|---|---|---|---|---|---|
| **fill** | `_accumulate_Z_image_chunked` bspline.py:7044. Off-edge moments of the real segment against the MIRRORED segment, `_image_moments` 7117 → `_seg_seg_full_moments_offedge` with `seg_l_img/seg_r_img` (7077-7078). Then the weighted windowed assembly at scale −1, `_assemble_window` 7133 / `_image_block` 7203. Near-image analytic fixup 7275+ (`_near_image_edge_blocks` 3996). | same sweep and moments | same sweep and moments | `sommerfeld_remainder_bspline_Q` symmetric route, bspline.py:4631. Core `_somm_remainder_core.cpp`; per node pair `proj_core(rho, hh)` in `_accel_somm_proj_inline.h:340` | outside this path (`!fr.buried`, `_nec5_bspline_printout_core.cpp:117`) |
| **reused today** | only `offset_reuse`'s spread within one edge (`offset` 7122-7131, `_image_offset_block` 7005). Nothing between edges. | same | same | edge-block copy map `_remainder_reuse_args` 5662 / `_remainder_reuse_map` 5687 → `bspline_reuse_remainder_blocks`. Offset fold where offset edges hold ≥ 50 % of pairs (5832-5837) | none (no "reuse" in `_bspline_transmitted_core.cpp`) |
| **key** | none | none | none | edge pair (x, y), `remainder_blocks` `_bspline_pair_reuse_core.cpp:312`. Key 367-392: segment counts, segmentation ids (inner arcs over length plus radii), the SIX DIRECT distances among (x0, x1, y0, y1), and the FOUR heights EACH (377), quantised at 2^-40 of the structure size. Orientation by min of (x,y)/(y,x) with a 1-bit code | — |
| **eval unit** | segment pair, (d+1)² complex moments | same | same | segment pair, (d+1)² moments `Jf` (node block q×q, then `contract`, `_somm_remainder_core.cpp:446-451`), then `finish_block` assembles the bases | — |
| **weights / signs** | `pec_weights` `_potential_ground.py:1109`: w_A = t_P·(t_Qx, t_Qy, −t_Qz) (`mirror_tangents`, `_ground_mirror.py:47`), w_Φ = 1, scale −1. C++ `weights` lambda `_bspline_fill_core.cpp:2858` | `refl_weights` `_potential_ground.py:1133`: cos θ and p̂ from the SEGMENT CENTRES (`specular_pair_tables` `_ground_refl.py:187`), w_A = ρ_v(td − P) − ρ_h P (225), w_Φ by `phi_mode` (239). Python only: the exe declines GN 0 refl (`_bspline_nec4_run_core.cpp:33`) | `sommerfeld_weights` 1122: C2·td, w_Φ = C2 (C2 constant) | in the kernel: the tangent projections relative to ρ̂ (`proj_project` inline.h:365) | — |
| **fill report** | family "image", `_note_ground_fill("image", n², n² − offset_saved, …)` 7268-7273. `_ground_families` 5465. Headline "partial", "the image term copied no block between edges" (5437-5451) | same | same | family "remainder", `_note_remainder_kernel` 5344: reused = Σ target-block sizes when the cache was used. `remainder_map_unused` 5364, `pair_reuse_remainder_cut` 5720-5727 (cache cut to `_REMAINDER_REUSE_CACHE_BYTES` = 64 MiB, bspline.py:873) | — |

The exe side (`_bspline_fill_core.cpp`):
- `accumulate_image` 2821 is a line-for-line twin. It covers PEC and compose only: `ImageGround` `_bspline_fill_core.h:163` has no Fresnel.
- `subtract_sommerfeld_remainder` 4672 reads `remainder_copy_map` 4186, which calls the same `reuse::remainder_blocks`.
- It is called from `_nec5_bspline_printout_core.cpp:116-133`.

### 1b. razor2p (`_razor_pair_reuse_core.{h,cpp}`, `razor.py`)

| | PEC image | refl-coef image | Sommerfeld: C2 image | Sommerfeld: remainder | buried |
|---|---|---|---|---|---|
| **fill** | the image is a second SOURCE SET (mirrored p0, t), `Geometry::p0[1]` h:227-231, built in `razor.py` 4846-4886 | same moments | same moments | `remainder.field_windows` razor.py:6837 | complex k declines (`_pair_reuse_refusal` razor.py:4970) |
| **reused** | yes | yes (the moments) | yes | yes, when every path row is its wing centroid (`q`, razor.py:5009-5025; `_razor_solve_core.cpp:539-554`) | no |
| **key** | THE DIRECT KEY, ONE class space for both blocks: `fam[DIRECT]` with `n_blocks = 2` (cpp:54-66). `direct_key(b, o, s)` cpp:121 reads (u_r, perp, h, a, EK type), which is rigid-motion invariant with reflections included (h:8-25). So a pair (o, M(s)) can share a class with a direct pair. | same | same | `q_key` cpp:166-210: W=9 words (exponent, **hs**, \|d\|, h, then d·a, d·b, a·b, t_sz, t_oz in the best of 4 orientations). The plane group's complete invariants, height SUM (h:75-93) | — |
| **eval unit** | (observer centroid, source segment): M0, M1 | same | same | (centroid, source segment): Q0, Q1 | — |
| **weights / signs** | outside the class value: in the contraction from the pair's own tables, or fused in `assemble_rows` (momwire#2144, h:333-366; `rows_direct` razor.py:7096-7111 for unweighted or `WINDOW_RULE_CONSTANT_MIRROR`) | `weight_windows` per (observer point, source segment) razor.py:6797-6806. The gather route, not `assemble_rows` | constant-mirror rule, C2 in `RowTables::c2` | inside the value (`eval_remainder`) | — |
| **report** | families "direct", "image", "remainder". Per block, counted by the block of each class's REPRESENTATIVE (`dcls_block[b]`, razor.py:4909-4917; `cls_block` `_pair_reuse_core.h:192-194`) | | | `q_classes` | — |

The exe uses the same core: `_razor_solve_core.cpp:520-560`, `reuse::plan_for`.

### 1c. sinusoidal (`_sin_pair_reuse_core.{h,cpp}`, `_sin_pair_reuse.py`, `sinusoidal.py`)

| | PEC image | refl-coef | Sommerfeld (image + remainder together) | buried |
|---|---|---|---|---|
| **fill** | block "image": the free-space kernel against the mirrored source centres and tangents (`_image_source_centers_tangents`, sinusoidal.py:7630-7637) | block "reflected", REAL sources (7630-7631). Value from `sinusoidal_field_tensor_refl` (7783) | block "reflected", value `c2·Φ_img − S` (7786-7805) | complex k declines (sinusoidal.py:7446-7447) |
| **key** | `DirectKey` cpp:203 (3-D: z, rho, rdt, td + types) on the mirrored sources | `GroundKey` cpp:245-300: **hs** (`(oc_z − gz) + (sc_z − gz)`, 258), \|d\|, d·a, d·b, a·b, with t_nz, t_mz in the types (`reversible_ground_types`), quantised against the distance to the IMAGE segment, 2-bit orientation code | same GroundKey | — |
| **eval unit** | (observer centroid, source segment): 3 shapes Φ_c, Φ_s, Φ_co (`VALUE_BYTES` 48, h:73) | same | same | — |
| **weights / signs** | the subtraction in the fused reduce (`reduce_band`, h:285-288, "Φ gathered from d less image") | INSIDE the class value (Fresnel) | INSIDE the class value (C2, S) | — |
| **report** | families "direct" plus "image" or "reflected" (sinusoidal.py:7914-7920). Over Sommerfeld, image and remainder are ONE family | | | — |

On the exe side, the C++ fill declines refl-coef reuse:
`_sinusoidal_fill_core.cpp:2291` reads "pair_reuse over a reflection-coefficient ground (its reflected block is not copied)".
Python serves it, so exe and Python cover different grounds.

---

## 2. The height-sum identity against each key

**The identity.** For points, |P − M(Q)|² = ρ² + (h_P + h_Q)² and |P − Q|² = ρ² + (h_P − h_Q)².
Their difference is 4·h_P·h_Q. For segments P(s) = P₀ + sA and Q(t) = Q₀ + tB, the identity holds node by node:
|P(s) − M Q(t)|² = |P(s) − Q(t)|² + 4·h_P(s)·h_Q(t), where h_P(s) = h_P₀ + sA_z and h_Q(t) = h_Q₀ + tB_z.

**What "direct key + equal h_P·h_Q" buys.** Take two pairs that agree under a rigid direct key, so |P−Q|²(s,t) is the same polynomial.
- Suppose the product polynomial h_P(s)·h_Q(t) is equal too. That means four coefficients: h_P₀h_Q₀, h_P₀B_z, A_zh_Q₀ and A_zB_z.
- Then |P−M(Q)|(s,t) is equal node for node. That is enough for any term that reads only the mirrored distance: the PEC image and the C2 image.
- The PEC weight t_P·Mt_Q = t_P·t_Q − 2t_Pz·t_Qz is fixed as well. The direct key fixes t_P·t_Q, and A_zB_z fixes the rest.
- It is NOT enough for refl-coef or the Sommerfeld remainder. Both read ρ and hs SEPARATELY: cos θ = hs/R, and the grid is at (R1, θ) from (ρ, hh), inline.h:311-316.
- Those two need the difference h_P − h_Q as well, and a rigid direct key does not fix it.
- Counterexample: a horizontal pair at heights (h, h) separated by D horizontally, against a vertical pair separated by D vertically with h₁h₂ = h². The direct distance is D in both. |P−M(Q)|² = D² + 4h² in both. But ρ = D, hs = 2h in the first, and ρ = 0, hs = √(D²+4h²) in the second. The Fresnel and Sommerfeld values differ.
- So the product form is too strict for PEC and too weak for a finite ground.
- It also MISSES the stack. Take pairs (i, j) and (i+1, j−1) of an evenly spaced vertical stack. They have equal hs and equal ρ. But their direct distances and products both differ.

**The key that is right for every ground term is the plane-group key on hs directly.**
- What it allows: the plane group, plus the vertical shift that moves P up by τ and Q down by τ (M T_τ = T_−τ M, so (P, M(Q)) moves rigidly).
- The complete invariants of a segment-segment pair under it:
  - h_P and h_Q (the lengths);
  - A_z and B_z;
  - hs (taken at the centres);
  - |d|, d·A_h, d·B_h and A_h·B_h (the O(2) Gram of the horizontal parts);
  - the radii and types.
- With hs, A_z and B_z fixed, every node pair's hs(s,t) = hs + sA_z + tB_z is fixed, and so is ρ(s,t).
- Every image model's weights depend on the pair only through these:
  - PEC: t_P·Mt_Q = A_h·B_h − A_zB_z over the lengths;
  - refl-coef: cos θ and the dyad P = (t_P·p̂)(t_Q·p̂). It is quadratic in p̂, so it is invariant under vertical-plane reflections, which flip p̂;
  - Sommerfeld: (ρ, hh) and the azimuth factors cos φ and sin φ, which enter only as products. Check this per term when it is built.

**Orientation and sign terms under the mirror.**
- M flips the vertical components: Mt = (t_x, t_y, −t_z).
- PEC: the image current is −Mt. The code carries it as the mirrored tangent with scale −1 (bs2 assembly `complex(-1.0)` bspline.py:7161; sinusoidal "d less image"). w_Φ = 1 is the image charge, also through scale −1.
- Reversing the source tangent negates d·a, a·b and t_sz. Reversing the observer's negates d·b, a·b and t_oz.
- razor and sinusoidal fold these into a 2-bit code (razor cpp:186-201, sin `canonical`).
- bs2 does NOT fold: a reversed segment keys apart, because folding moments needs a binomial transform (`_bspline_pair_reuse_core.h:54-58`).
- At the BASIS level a reversal is an exact sign flip of the entry. So a post-assembly copy could fold orientation with no arithmetic. Proposal only; the drift is unmeasured.

**Each key in use today:**

| key | what it is | correct for | finds stacks? |
|---|---|---|---|
| razor image through direct key (`_razor_pair_reuse_core.cpp:121`, n_blocks=2) | rigid invariants of (centroid, M(segment)) | **every geometry and every ground**. The class value M0/M1 is blind to the plane, and every plane-dependent factor (C2, Fresnel windows, tangents) is applied per pair outside it (h:20-25; razor.py:6797-6806) | yes: (o, M(s)) are congruent for equal hs. #2159 quotes 69x on a 6-element stack |
| razor remainder `q_key` cpp:166 | plane-group key with hs | remainder | yes |
| sinusoidal PEC "image" DirectKey on mirrored sources | rigid | PEC (value = free-space field of M(Q)) | yes |
| sinusoidal `GroundKey` cpp:245 | **the plane-group hs key** (point observer form) | refl-coef and Sommerfeld, weights INSIDE the value | yes |
| bs2 remainder `remainder_blocks` cpp:367 | edge pairs: six DIRECT end-point distances plus FOUR individual heights | remainder, image (any model) | **no**: it allows only horizontal motions at kept heights (the branch's own `test_a_vertical_stack_copies_no_image_block`, br: tests/test_bs2_image_reuse_2074.py:169) |
| bs2 direct key (`_bspline_pair_reuse_core.h:7-33`) | rigid: Gram of (m, A, B), h_i, h_j, a_i, EK | would be exact for bs2's IMAGE MOMENTS applied to (P, M(Q)) as razor does. bs2's image moment is the off-edge kernel on the mirrored segment, with the same ladder by default (`_image_fill_ladder` bspline.py:3112-3140). EK: one joint label, `b_j` = radii (`_bspline_fill_core.cpp:2841-2852`). It is NOT valid for weighted entries over refl-coef | yes |

Answers to the brief's two questions:
- Sinusoidal's reflected key IS the height-sum key: hs is taken directly, not through the product identity.
- razor's "image through the direct key" is correct for ALL geometries. It is correct BECAUSE it copies plane-blind moments and applies the weights per pair. The limit is what it saves, not correctness.
- A bs2 copy of WEIGHTED or assembled image entries through a rigid key would be wrong over refl-coef: a tilted pair would merge with a horizontal one. Over PEC and C2 it would be right.

---

## 3. The Sommerfeld remainder

- **What it depends on.** Per node pair, only on ρ = hypot(dx, dy) and hh = (z_o − gz) + (z_s − gz). Code: `pair_geometry` inline.h:383-391 and `proj_core` 340-362. The table is read at R1 = √(ρ² + hh²) and θ = atan2(hh, ρ), clamped to [0, π/2].
- **Grid regions.** Up to 6 regions above the ground, chosen by r_break, r_near and th_split (`proj_address` 323-338). Past r1_max there is a continuation (`far_continue` 293).
- **Projection.** Onto the tangents through ρ̂, φ̂ and ẑ (`proj_project` 365-381).
- **Sufficiency of the height-sum key.** In exact arithmetic it is sufficient: segment endpoints in the plane-group invariants, with hs, A_z and B_z, fix (ρ, hh) at every node pair, and the azimuth factors.
- **A (ρ, z+z') key is not needed separately.** It is what that key fixes node by node.
- **Caveats for a key (risks, not blockers):**
  - Region breaks (r_break, r_near, th_split, r1_max) are table switches. A member and its representative on either side of one read different tables. That calls for a TIE rule like bs2's ladder `TIE_REL` (`_bspline_pair_reuse_core.h:62-64`). Lagrange stencil shifts inside a region are continuous, because both stencils hit the node.
  - The remainder ORDER is per pair and raised near grazing (`_remainder_qp` 3582, `_remainder_qp_pairs` 3612). The raised set is symmetrised and DILATED, so it is not a function of one pair alone (bspline.py:3612-3630). Today the raised pairs are a correction added after the copy (bspline.py:4648-4652, `n_raised`). A new key must keep that order: copy the base-order entry, then correct.
  - Bit identity: ρ and hh are recomputed per pair, so a copy differs from the member's own value by key rounding. That drift is the same class as the direct family's.
- **What bs2's edge-block map keys on now:** whole edge pairs, keyed on the six direct end-point distances and the four individual heights (cpp:367-392).
  - It is stricter than needed: no stacks, and no tilted-but-plane-congruent pairs.
  - It works at edge granularity: about 2E − 1 distinct blocks on E equal elements (#2159 point 2). There are no j−i repeats inside a block beyond the offset fold.
  - The kernel copies MOMENTS (`_somm_remainder_core.cpp:443-466`), and `finish_block` still assembles every basis pair.
  - The copy pays here, unlike the image, because the per-pair cost is the q×q node block of 4-surface 4×4 interpolations (`node_block`), not the assembly. That is inferred from the code structure; there is no profile in the tree.

---

## 4. Why `bs2-image-reuse` saved no time

**What it did** (`48d2151e`):
- `_image_copy_map` (br: bspline.py:5726) reuses the remainder's edge-block map, minus the offset and near-image edges' own blocks.
- It fills a cache of the source blocks' MOMENTS (br: 7194).
- `_pieces` (br: 7204) splits each run of observer rows into per-edge column pieces. A cached piece is passed to `_assemble_window` as J (br: 7278-7288).
- The C++ twin is `pieces` (br: _bspline_fill_core.cpp:3027). It copies a sub-block when the row run is not the whole edge (br: 3054).

**Measured** (#2159; 10-element Yagi, uniform cores, Skylake): pair-terms 5.15M → 1.25M, fill 0.937 → 0.921 s, image kernel 0.26 → 0.12 s.

**Where the cost sits**, from the code (the profile itself is not in the tree):
1. **Weights for every pair.** `weights_fn(r0, r1)` is still called for the whole row band against every column (br: bspline.py:7275; br: _bspline_fill_core.cpp:3096). That is PEC's gemm, or over refl-coef `specular_pair_tables` plus `fresnel_rho` per pair. Copied pairs pay it too.
2. **Weighted assembly for every pair.** `_assemble_window` runs over every target pair with the cached J. The per-pair work there is (d+1)² products into iA, the Φ products, and two complex weight multiplies, for each of up to (d+1)² basis pairs a segment pair feeds (`_bspline_core.cpp:3722-3766` reference; lane kernel 3040-3070). Copying J removes only the kernel.
3. **New overhead from the split.** Inferred: the kernel fell 0.14 s but the fill only 0.016 s, so most of the saving went somewhere. In Python each piece costs:
   - a fresh `_touching(a, b)` O(n_basis) scan (br: 7288; main 7167-7170);
   - `np.ascontiguousarray` copies of the strided cache slice and of `w_A_win[:, a:b]` (main 7134-7145);
   - smaller lane-kernel windows.
   The C++ route has the same scans (`touching` `_bspline_fill_core.cpp:2880-2891`), `cols_of` copies (br: ~3062) and `Block J` copies. The sources' cache fill is also computed separately from the sweep.

   Whether item 3 or items 1-2 dominate is not settled by any number in the tree. See question Q6.
4. **Not new, but the same shape:** the bs2 DIRECT family also copies moments and assembles every pair (`plan.gather` then `_accumulate`, bspline.py:6544-6560; C++ `reuse::gather` 1172). Its free-space floor (AK raw.jsonl, warm medians):

   | engine | computed | fill, reuse on | fill, reuse off |
   |---|---|---|---|
   | bs2 | 92 k | 0.195 s | 0.371 s |
   | razor2p | 95 k | 0.043 s | 0.171 s |

   razor avoids this floor with the fused `assemble_rows` (#2144). It is the same problem one family over (stage A's or the direct family's concern; flagged here because the fix is the same).

**What has to change for a copy after assembly to pay:**
- The unit copied is the ASSEMBLED image contribution of a basis pair (or a block of basis pairs). For a target, the kernel, the weights window and the assembly are all skipped.
- The weights producer must be asked only for the computed columns. Today it is row-band × all columns.
- The source contributions must be held apart from Z, because Z already holds direct (+ remainder). Use a side buffer, budgeted like the remainder cache (64 MiB, bspline.py:873), and add it once per target with scale −1.
- Sources must be filled in large windows: no per-edge fragmentation of the lane kernel, and no per-piece O(n_basis) scans. Targets are filled with strided copies.
- The key is the plane-group hs key at the basis level. Junction and bend bases, whose supports leave one straight edge, are computed.
- Over Sommerfeld, image and remainder could be ONE copied value, as sinusoidal does (sinusoidal.py:7786-7805). That is one census and one copy. See Q2.

---

## 5. Proposal: the ground family in `_pair_reuse_core` (separate from the evidence)

**Key policies, in the core rather than three adapters.**
- `KeyPolicy::Rigid(mirrored)`. The direct adapter's own key applied to (P, M(Q)), sharing the direct class space (razor's n_blocks = 2 is the model).
  - Valid for any value that is blind to the plane: moments, the PEC image and the C2 image.
- `KeyPolicy::Plane`. One shared implementation of the plane-group invariants:
  - hs, |d|, d·a, d·b and a·b, plus a_z, b_z and the lengths;
  - quantised against the image distance;
  - a 2-bit orientation code;
  - a near test against the image segment.
  - Today it exists twice: razor `q_key` (cpp:166) and sinusoidal `GroundKey` (cpp:245). Lift them into one.
  - Valid for any value, weights included.
- Tie hooks the adapter supplies: ladder tiers on the mirrored distance (bs2), and Sommerfeld region breaks (every basis). A pair within the tie tolerance is its own class.

**The adapter supplies:**
- its observer and source units: a centroid point (razor, sin), a segment (bs2 moments) or a basis chain (bs2 entries);
- its evaluation of one class's value, with a flag for whether the weights are inside it or applied per pair (razor's refl-coef stays outside);
- its image weights, as a per-pair producer or as part of the value;
- its copy and transform at gather time (code signs);
- its budget constants.

**Per basis:**

| basis | PEC / C2 image | refl-coef image | remainder | where the copy goes |
|---|---|---|---|---|
| razor | unchanged: Rigid, shared with direct | unchanged (moments copied, weights per pair) | Plane (its q_key, moved into the core) | as today; refl-coef `rows_direct` stays off |
| sinusoidal | unchanged: Rigid on mirrored | Plane, value inside | Plane, joint with the image ("reflected") | as today. The exe must serve refl-coef (`_sinusoidal_fill_core.cpp:2291`) for exe == Python reach |
| bs2 | Plane on basis-pair entries (Rigid would also be exact here, and finds direct⇄image sharing, but a copy that includes refl weights cannot use it) | Plane on basis-pair entries | Plane at segment-pair moments in the kernel (replacing `remainder_blocks`), or joint with the image as one entry | Python: in `_accumulate_Z_image_chunked`, sweep only the computed columns, then after `_schedule.sweep` (7253) and the offset blocks (7262-7268), and BEFORE the near-image fixup (7275), add the side buffer into the targets. C++: the same place in `accumulate_image` (2821), and `remainder_bspline_Q`'s symmetric route for the remainder |

**Risks to byte identity and to exe == Python:**
1. **Reuse off must stay bitwise.** The census, the side buffer and the column skipping all have to sit behind the flag. The branch held this (`48d2151e` message); keep that gate.
2. **Accumulation order.** Off, a basis pair whose support spans two column pieces is added to Z in parts. On, a copied entry is added once. So source entries may differ from off by the rounding of the split. Any split that is not on the off path's own windows moves bits. exe and Python must split identically.
3. **The near-image fixup** (#631) accumulates a DIFFERENCE against what the sweep added (7275-7290). A copied target of a near-image edge would break that invariant. Keep those edges Own, as the branch did (br: 5752-5754).
4. **`offset_reuse`'s own-edge blocks** (7122-7131) and their cuts must stay outside the family, or the family must subsume them. Both cannot run on the same pairs.
5. **The remainder's raised pairs** (dilated, not per pair) are corrected after the base fill. A copy before the correction is safe. A copy of a corrected entry is not.
6. **Sommerfeld region and tier ties**: see §3. The quantum can straddle a table switch.
7. **`restrict` (`_ObserverRows`), the sector route (`_rotational_map`) and the tensor route** (`dense_tensor_fits`, `_nec5_bspline_printout_core.cpp:104`; `_image_Z_refl` 4317): the family serves whole-mesh chunked fills only, as the image note does (7269-7273). Each other route must decline in both languages with the same reason.
8. **Merging image and remainder into one value** changes the rounding (one add of the sum instead of two adds) and the report's families. It is a deliberate drift, to be agreed (Q2).
9. **refl-coef reach differs by route:** the bs2 exe has no refl-coef, and the sinusoidal exe declines it. exe == Python can only be gated on PEC and Sommerfeld for bs2.
10. **Budget.** Basis-pair class maps are 4 B per basis pair per family, on top of the direct map, which is inside the 4/3 cap's allowance. The side buffer is a new transient that #2123's allowance does not price yet.

---

## 6. Contradictions with the issues, and open questions

**Contradictions:**
1. **#2063 stage B, "One key policy built on the height sum: |P−M(Q)|² = |P−Q|² + 4·h_P·h_Q".** The formula is the PRODUCT identity (§2). A key built from it (direct key plus equal h_P·h_Q):
   - is unnecessary for PEC and C2, where the rigid key on (P, M(Q)) is exact and coarser;
   - is insufficient for refl-coef and Sommerfeld, which read ρ and hs separately;
   - misses the vertical stack.
   #2159 point 3 makes the same slip ("depends on h_P + h_Q (identity …)"), and so does the branch's commit message. The working key is sinusoidal's and razor's remainder key: hs taken directly.
2. **#2063, "razor2p reuses its image through the direct key"** is true, and it is correct for every geometry (§2), not a shortcut that holds only for some. Its value carries nothing plane-dependent.
3. **#2063, "bs2 … no image reuse at all".** There is the offset spread within an edge (7122-7131, 7268-7273), which #2074 itself notes.
4. **#2159 point 1, "bound by weighted assembly, not the kernel".** Weights and assembly do run for every pair (§4, items 1-2). But the branch's own numbers (kernel −0.14 s, fill −0.016 s) also implicate the copy machinery's own overhead (§4, item 3). The split is unmeasured in the tree, and stage B's design depends on which dominates.
5. **#2074, "the remainder's copy map … finds exactly these pairs".** It finds horizontal translations at kept heights only: four heights each, plus direct distances (cpp:367-392). That is a subset of the hs classes.
6. **The cross-engine gate (`tests/test_bs2_ground_reuse_report_2159.py:104-127`) compares different quantities:**
   - bs2 "image" (image alone);
   - sinusoidal "reflected" (image plus remainder, one key);
   - razor "image", whose classes are shared with the direct block. Image pairs reusing a direct-block representative count as reused in "image" (razor.py:4909-4917; `cls_block`, `_pair_reuse_core.h:192`).
   razor's and sinusoidal's image compression are therefore not "image-only key" figures. The FACTOR = 4 bound may be unreachable for a bs2 image-only key on some decks, or easy on others.
7. **#2063, "Each basis supplies only its evaluation and its image weights".** For bs2 with a copy after assembly, the weights are inside the copied value. For razor they are outside, per pair. The interface has to allow both (§5).

**Open questions for Steve:**
- **Q1.** bs2's image: copy ASSEMBLED entries on the plane key (covers refl-coef, skips weights and assembly), or the Rigid key on moments as razor does (exact, finds direct⇄image sharing, but saves only the kernel, which #2159 says is not the cost)? Or both?
- **Q2.** Over Sommerfeld, one "ground" family (image + remainder as one value, as sinusoidal does), changing the report's families and the rounding? Or keep two?
- **Q3.** bs2's copy unit: a census of basis pairs (finds the j−i repeats, costs a basis-pair map), or basis-level edge blocks (cheap, about 2E−1 blocks, #2159 point 2)?
- **Q4.** Allow orientation folding at the basis level, where it is an exact sign flip, for bs2 only?
- **Q5.** A tie rule at the Sommerfeld grid's region breaks: what tolerance, and does it apply to razor's and sinusoidal's remainder keys too (they have none today)?
- **Q6.** Before stage B is briefed, should the branch's image fill be profiled into kernel / weights / assembly / piece overhead on Skylake? Only the kernel figure is recorded, and only in #2159's text.
- **Q7.** Does the same post-assembly copy belong to the DIRECT family on bs2 (its 0.195 s floor against razor's 0.043 s)? That is, stage A, or B, or a stage of its own?
- **Q8.** The cross-engine gate: compare like with like? For example, image-only on PEC and refl, and "ground total" over Sommerfeld; and count razor's image against an image-only census.
- **Q9.** refl-coef reach: should the sinusoidal exe serve the reflected block (it declines today), and is bs2 refl-coef exe-out-of-scope for exe == Python gating?
