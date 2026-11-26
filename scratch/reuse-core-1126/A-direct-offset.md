# momwire#2063 stage 0, part A: the direct and offset families

2026-11-26. A read-only survey of `reuse-survey-stage0` (= main at v0.98.0,
`ed479892`). No source file was changed, nothing was built and nothing was
run. Paths are relative to `src/momwire/` unless they start with
`standalone/`, `tests/` or `AK:` (antennaknobs). `file:N` is a line and
`file:N-M` a range. Sections 1–4 and 6 are evidence. Section 5 is a proposal
and is labelled as one.

Ground terms (image weights, the reflected term, the Sommerfeld remainder
copy map) belong to stage B. They appear here only where the offset path
touches them.

---

## 0. What the prior survey got wrong, or what has moved since

The prior survey is `origin/reuse-common-survey:scratch/reuse-common-1120/DESIGN.md`, at `d5feb5c7`.

| prior claim | now |
|---|---|
| bspline has no census in momwire; AK's hint finds the classes | False. bs2 runs its own census on the core: `_bspline_pair_reuse_core.cpp:73-181`, called from `bspline.py:5566-5596` and `_bspline_fill_core.cpp:677-710`. A hint is accepted and read as `True` (`bspline.py:1947-1954`). |
| `offset_reuse` has no C++ (blocks the bs2 exe) | False. `_bspline_fill_core.cpp` carries all four offset mechanisms (§3.1), with `fp.offset_reuse` read at `_nec5_bspline_printout_core.cpp:62-63`. |
| razor declines at `MAX_CLASS_FRACTION` 0.75 under a 256 MB plan budget | Replaced. The class cap is 4/3x compression (`_pair_reuse_core.h:122,136-139`) and an allowance of 4 B/pair plus per-class bytes applies (`_pair_reuse_core.h:100-121,139`; razor `_razor_pair_reuse_core.h:161-206`). |
| sinusoidal C++ census runs only in the certification build | False. It ships in the nec42- and nec2c-sinusoidal exes at reach 11 (`_features.py:270-282`; `_sinusoidal_fill_core.cpp:2265-2400`). |
| sinusoidal's census is numpy + `IntGroups` in Python | False. Python calls the C++ census (`_sin_pair_reuse.py:671-695` → `_acc.sin_pair_reuse_plan`). Only the plan views, the evaluation and verify stay in Python, as the issue says. |
| sinusoidal has no compression cap | False. It has `pol.capped = true` (`_sin_pair_reuse_core.cpp:388`). |
| "No implementation declines early" | Still true in substance. The cap is checked after every chunk (`_pair_reuse_core.h:627-641`), but `so_far > class_cap(pairs)` cannot fire before 3/4 of the pairs have been keyed, so #2061 is still needed. |
| `offset_reuse` "shares exactly those [near] pairs" that the razor census excludes (addendum) | Wrong in scope. bs2's offset covers a uniform straight edge's **whole own block** (n² pairs, near and far), and bs2's census excludes that block as its sentinel (`_bspline_pair_reuse_core.cpp:113-117`). razor's and sinusoidal's censuses already share the far pairs on a run. An offset stage there would add only the near band, about 7 pairs per row (§3.3). |
| offset reuse is "one structured special case" | It is four mechanisms with different keys (§3.1). |
| line refs (`razor.py:4942-4977`, `sinusoidal.py:6970-7004`, …) | Moved: now `razor.py:4956-4991` and `sinusoidal.py:7430-7507`. Every ref below is re-taken at `ed479892`. |

Stale comments in the tree itself:
- `_sin_pair_reuse_core.h:25-28` says "keeps at most M N / 2 classes". That was the old 2x cap. The core caps at 3/4 (`_pair_reuse_core.h:103-107,122`).
- `bspline.py:5796` `_offset_rows` has no callers (`git grep _offset_rows`). It is dead code.

---

## 1. What the core owns now: `_pair_reuse_core.{h,cpp}`

| concern | owner | evidence |
|---|---|---|
| census (hash, parts, chunked rows, class numbering) | core, as a template instantiated in the adapter's TU | `_pair_reuse_core.h:479-689`; parts `:350-363`; numbering `:24-31,61,436-438` |
| class cap 4/3 | core | `:122,136-139`; checked in the census `:488-489,628-641,651-654`; test hook `:128-131` |
| allowance arithmetic | core helpers; each adapter sums them | `class_bytes` `:96-98`, `plan_allowance` `:139`, `maps_bytes`/`held_bytes` `:221-223` |
| decline sentences + `decline()` | core | `cap_reason`/`allowance_reason` `:146-147`, `_pair_reuse_core.cpp:251-264`; `decline` `:266-274` |
| plan cache (LRU, 256 MB of old plans, `keep`) | core | `_pair_reuse_core.h:240-254`, `.cpp:276-330` |
| counters built / found / declined | core, per basis tag | `.h:258-263`, `.cpp:332-341` |
| evaluation groups (≤ `rep_rows` observer rows × the source union) | core, opt-in (`rep_rows > 0`) | `Policy::rep_rows` `.h:155-157`; `build_groups` `.cpp:25-101`, called from `finish` `.cpp:128-129` |
| report split (cls/near/evals per block) + near bits | core | `.h:191-194,179-182`; `finish` `.cpp:117-143` |
| wire map | core | `.h:302-303`, `.cpp:357-387` |
| gather decode (entry → class, rel code) | core | `.h:696-717` |
| band budget (sinusoidal's census accounting) | core | `.h:282-295`, `.cpp:199-216` |
| key quantisation helpers | core | `bits`/`quantum`/`qword`/`Ids` `.h:308-336` |
| **small-deck decline** | **not core**: each adapter has its own constants and its own sites | razor `_razor_pair_reuse_core.h:153-160` (400 / 100), sites `razor.py:4988-4990`, `_razor_solve_core.cpp:510-516`; sinusoidal `_sin_pair_reuse_core.h:73-74` = `_sin_pair_reuse.py:250-251` (400 / 200), sites `sinusoidal.py:7496-7501`, `_sinusoidal_fill_core.cpp:2281-2282`; bs2 `_bspline_pair_reuse_core.h:116` (200), sites `bspline.py:5561-5563`, `_bspline_fill_core.cpp:683` |
| fingerprint (cache key) | adapter | razor `_razor_pair_reuse_core.cpp:228-262`; sinusoidal `plan_key` `_sin_pair_reuse_core.cpp:804-826`; bs2 `_bspline_pair_reuse_core.cpp:188-212` |

### The core's interface, as a basis calls it

```cpp
// _pair_reuse_core.h
struct FamilySpec { name; key_words /*W*/; code_bits; n_blocks /*<=2*/; rows, cols;
                    value_bytes; near_singles; class_limit; numbering; keep_reps; list_near; };   // :64-89
struct Policy { capped=true; rep_rows=16; census_rows=0; measured=true; };                        // :150-164
enum class Key : uint8_t { Shared, Own, Near };                                                   // :226
template <int W, class Tab = detail::Table<W>, class KeyFn, class Hook>
Outcome census(FamilyPlan&, const Policy&, int64_t room, int64_t& peak, KeyFn key, Hook& hook);  // :479-683
//   key(int block, int64_t o, int64_t s, uint64_t* w /*W words*/, uint8_t& code) -> Key
//   hook.start(chunk) / hook.done(chunk, classes, near, tail_new); Hook::tail
template <class Visit> void gather(const FamilyPlan&, int block, const int64_t* rows, ptrdiff_t n,
                                   Visit visit /*(i, s, cls, rel)*/);                             // :696-717
std::shared_ptr<const Plan> plan_for(basis, fingerprint, build /*void(Plan&)*/, bool* built, int keep); // :252-254
```

| adapter | W / code_bits / blocks | Policy | Tab / Hook | numbering | evaluation the core does not see |
|---|---|---|---|---|---|
| razor direct | 4 / 1 / 1–2 (direct, PEC image) `_razor_pair_reuse_core.cpp:55-67` | capped, rep_rows 16, measured `:43-48` | `Table<4>` / NoHook `:158` | PartOffsets (default) | `eval_direct` over the core's groups via `seg_moments` `:328-388` |
| razor remainder | 9 / 2 / 1 `:68-79` | same | `Table<9>` / NoHook `:213` | PartOffsets | `eval_remainder` / `store_remainder` `:390-447` (stage B) |
| sinusoidal direct / image | 5 / 2 / 1 per plan `_sin_pair_reuse_core.cpp:374-381,699` | capped, **rep_rows 0**, census_rows = band, **measured off** `:382-388` | `GroupTable<5>` / `FigureHook` (BandBudget) `:337-347,389-401` | FirstAppearance + `keep_reps` `:380-381` | its own: `evaluate` (canonical, EVAL_CHUNK) `:912-947`, `evaluate_near` `:954-981`, `near_blocks` `:431-471` |
| sinusoidal reflected | 5 or 6 / 0 or 2 / 1 `:778-782` | same | same | same | `evaluate_classes` via `rep_blocks` `:949-952` (stage B) |
| bs2 direct | 6 / 1 / 1 `_bspline_pair_reuse_core.cpp:48-61` | capped, rep_rows 16, measured `:37-42` | `Table<6>` / NoHook `:175` | PartOffsets | Python `_reuse_values` `bspline.py:5598`; exe `reuse_values` `_bspline_fill_core.cpp:529-571` (shipped off-edge kernel over the core's groups) |

---

## 2. What each basis still owns: the five adapter items

### 2.1 Pair key

| | what is hashed | quantum / tolerance | orientation |
|---|---|---|---|
| razor direct | `|v|` (v = u_r − h/2) and ρ = √perp, quantised against D = max(|v| − h/2, ρ, a) floored to 2^e; h **exact** by type id; a, EK radius and tube b by type; EK eligibility bit (`_razor_pair_reuse_core.cpp:121-156`) | KEY_BITS 40 (`_razor_pair_reuse_core.h:119`): step 2^-40 · 2^⌊log2 D⌋ | folded: v<0 → code 1 (`.cpp:150`); a pair with \|v\| quantising to 0 keeps 0 |
| sinusoidal direct | k[0] = (e + 2^11) << 40 + src_type·n_obs_types + obs_type; z, ρ, rdt on the D step; td on the absolute 2^-48 step (`_sin_pair_reuse_core.cpp:203-238`) | KEY_BITS 48 (`_sin_pair_reuse_core.h:68`); D = max(\|z\| − H, ρ, a_obs) (`.cpp:218-221`); exponent from the bits (`:134-137`) | canonical of 4, 2-bit code (source flip, observer flip) (`:146-192`); source type moves by `ds` under reversal (`:550-559`) |
| bs2 direct | (h_i, h_j) on a 2^-44 step of their own floor; m·m, m·A, m·B, A·B on one step of D², D = max(\|m\| − (h_i + h_j)/2, a_i); a_i by type; EK eligibility bit and b_j by type (`_bspline_pair_reuse_core.cpp:111-173`, `length_type` `:63-71`) | KEY_BITS 40, LENGTH_BITS 44 (`_bspline_pair_reuse_core.h:90,113`) | **not folded** (`.h:52-55`). One code bit for **reciprocity**: the transpose form when a_i = a_j (and b, eligibility agree), the smaller key kept (`.cpp:146-170`) |

All three use exact equality of the quantised words; nothing is hashed into fewer bits (`_pair_reuse_core.h:407-422` memcmp).

### 2.2 Near rule (pairs left out of the census)

| | rule | constant | representation |
|---|---|---|---|
| razor | max(\|v\|, ρ) < 4·max(h, a) (`_razor_pair_reuse_core.cpp:136-138`) | `NEAR_LENGTHS = 4.0` (`.h:121`); probe `PROBE_NEAR_H` drops the a (`.h:295-306`) | `Key::Own`: its own class (a "single"), still evaluated in the core's groups by `seg_moments` (`near_singles = true`, `.cpp:63`) |
| sinusoidal | max(\|z\|, ρ) < 4·max(2H, a_obs) (`_sin_pair_reuse_core.cpp:195-198,216`; reflected `:277`) | `NEAR_LENGTHS = 4.0` (`.h:69`, `_sin_pair_reuse.py:154`) | `Key::Near`: one sentinel class (`kNearSentinel = -1`, `.cpp:46,222-224`), listed in `near_flat`, evaluated directly in the original frame in `NEAR_ROWS = 16` blocks (`near_blocks` `.cpp:431-471`, `evaluate_near` `:954`; Python twin `_sin_pair_reuse.py:806-825`), written over the gather (`band_block` `.cpp:1119-1135`) |
| bs2 | (a) same edge: every pair of an edge's own block; (b) cross-edge: \|m\| < 4·max(h_i, h_j, a_i); (c) ladder ties: \|ratio − t\| ≤ 1e-9·t (`_bspline_pair_reuse_core.cpp:113-127`) | `NEAR_LENGTHS = 4.0`, `TIE_REL = 1e-9` (`.h:92,95`) | (a) `Key::Near`, **unlisted** (`list_near = false`, `.cpp:56`): its gathered value is overwritten by the same-edge fill or the offset route; (b), (c) `Key::Own` |

### 2.3 Pair evaluation (function, unit)

| | function | unit | batching contract |
|---|---|---|---|
| razor | `seg_moments` (`_razor_core.cpp:581`; impl `:215`), called by `eval_direct` (`_razor_pair_reuse_core.cpp:328-388`) | (centroid observer, source segment): M0, M1, 2 complex = 32 B (`.cpp:40`) | each group's observers × the sorted source union; values do not depend on numbering (`_pair_reuse_core.h:26-29`) |
| sinusoidal | caller-supplied `CanonFn` / `RowsFn` (`_sin_pair_reuse_core.h:244-253`). The exe's is `sinusoidal_field_tensor_ek` (`_sinusoidal_fill_core.cpp:2184-2208`) | (observer centre, source segment): Φ0, Φ1, Φ2, 3 complex = 48 B (`.h:77,81`) | class order and EVAL_CHUNK pieces are part of the plan: batched sincos moves an ulp (`.h:38-42`, `_sin_pair_reuse.py:168-176`) |
| bs2 | the shipped off-edge moments `seg_seg_full_moments_bspline*`: Python `_reuse_values` (`bspline.py:5598`), exe `reuse_values` (`_bspline_fill_core.cpp:529-571`) | (observer segment, source segment): J[p, P], (d+1)² complex = 144 B at d = 2 (`_bspline_pair_reuse_core.cpp:44-47`) | a pair's moments do not depend on the call's other pairs (`.h:31-35`) |

### 2.4 The transform a mirrored copy takes (rel ≠ 0)

| | transform | where |
|---|---|---|
| razor direct | source reversed: M1 → h·M0 − M1 (M0 unchanged) | `gather_direct` `_razor_pair_reuse_core.cpp:449-465`; fused in `assemble_rows` `:540-735` (flip at `:584ff`) |
| razor remainder | source: (Q0, Q1) → (−Q0, Q1 − h·Q0) and sign; observer: both negated | `gather_remainder` `:467-527`; key `.h:86-93` |
| sinusoidal | `kCodeSigns`: source (−, +, −), observer (−, −, −), both (+, −, +) on (Φ0, Φ1, Φ2) | `_sin_pair_reuse_core.cpp:500-505`; `to_canonical` `:508-514`; reduce `set_code_signs` `:1034-1038` |
| bs2 | reciprocity transpose J[P, p] (a copy, no arithmetic) | `gather` `_bspline_pair_reuse_core.cpp:292-310` |

### 2.5 Budget constants

| | class cap | small deck | max N | per-class kept bytes | other |
|---|---|---|---|---|---|
| razor | core 4/3 | 400 free/PEC, 100 finite (`.h:153-154`) | 46340 (`.h:273`) | `kept_bytes_per_class` = class_bytes + 8 = 49 B (`.cpp:306-312`) | REP_ROWS 16 (`.h:123`); VERIFY_TOL 1e-9 (`.h:126`) |
| sinusoidal | core 4/3 | 400 / 200 (`.h:73-74`) | `CODED_PAIRS` 2^29 (`.h:76`) | direct 105 B, reflected 73 B, near pair 72 B (`.h:91-98`) | BAND_BYTES_PER_PAIR 136/168, BAND_SHARE 0.25 (`.h:78-80`); EVAL_CHUNK 2048, NEAR_ROWS 16 (`.h:70-72`); VERIFY_TOL 1e-7 (`_sin_pair_reuse.py:204`) |
| bs2 | core 4/3 | 200 (`.h:116`) | 46340 (`.h:100`) | `kept_bytes_per_class(d)` = (d+1)²·16 + 9 + 8 = 161 B at d = 2 (`.h:120-124`, `.cpp:216-222`) | REP_ROWS 16 (`.h:97`) |
| bs2 **offset** | none (no map) | an edge of `≥ 12` segments (`_bspline_kernels.py:962-964`, `_bspline_fill_core.cpp:349`) | – | table (d+1)²·(2n − 1) complex per edge | `_PAIR_REUSE_RTOL = 1e-12` × structure scale (`bspline.py:860`, `_bspline_fill_core.cpp:350`); `_REMAINDER_REUSE_CACHE_BYTES` 64 MB (`_bspline_fill_core.cpp:351`); chunk ≤ 8 MB (`bspline.py:6898`) |

---

## 3. The offset path

### 3.1 bs2's `offset_reuse` is four mechanisms

| # | mechanism | what repeats | table built from | Python | exe (`_bspline_fill_core.cpp`) |
|---|---|---|---|---|---|
| R | **reg Toeplitz**, inside a kernel: the same-edge smooth (reg) moments of a uniform edge, served from one row | j − i on the 1-D arc | an **exactly synthesized** edge of 2n − 1 segments: `arange(-(n-1), n+1) * h`, its centre row (`_bspline_kernels.py:970-991`); gathered as a strided view (`:994-1004`) | ambient: a `ContextVar` `_REG_TOEPLITZ` (`_bspline_kernels.py:949-958`) set by the decorator `_offset_reuse_fill` (`bspline.py:1338-1369`) on 7 fill methods (`bspline.py:4065,5144,5196,8909,9044,9574,11070`); read at one site, `_seg_seg_reg_geometry` (`_bspline_kernels.py:866-877`) | explicit, at **6** sites: `reg_rows_toeplitz` `:404-427`, used at `:853-871` (tensor J), `:1227,1262-1266` (chunked fixup), `:2230-2236` (windowed reg), `:3071-3075` (near-image), `:5261-5265` + `reg_moments_cplx_toeplitz` `:4958` (complex k) |
| F | **free offset block**: a uniform straight edge's own free-space block (static + reg), assembled at basis level | j − i, any orientation | the edge's **own arc**: row 0 and row n − 1 (`reciprocal=False`) (`bspline.py:6953-6977`) | `_free_offset_edges` `bspline.py:6923-6951`; chosen `:6503-6508`; sweep skips own columns `:6510ff`; blocks `:6775-6812`; `_free_offset_block` `:6979-7003` | `free_offset_edges` `:712-717`; skip `:1122-1144,1185-1203,1226`; tables + blocks `:1305-1375` |
| I | **image offset block** (PEC / refl / Sommerfeld image weights) | horizontal: j − i ("toeplitz"); vertical: i + j ("hankel") | the image kernel's own row 0 (+ row n − 1 for hankel), reciprocity for toeplitz (`bspline.py:7012-7018`) | `_offset_edges` `:5740-5769`; selection, near-image edges excluded `:7133-7143`; `_image_offset_block` `:7005-7040`, applied `:7262-7270` | `offset_edges_of(g, false)` `:474-526`; selection `:2940-2953`; split sweep `:2978-3013`; block `:3017-3050` |
| S | **Sommerfeld remainder offset fold** | same kinds as I | `remainder_field_proj` row 0 (+ n − 1), einsum on W (`bspline.py:5839-5866`) | `_remainder_offset_edges` `:5804-5837` (refused when a `pair_reuse` copy map applies and offset edges hold < ½ the pairs, `:5831-5836`); `_remainder_offset_block` `:5868`; `_remainder_offset_fold` `:5981`; callers `:4618-4630,4735-4745` | `remainder_fold_edges` `:4269-4300` (same ½ rule `:4285-4289`); `remainder_offset_block` `:4394`; `remainder_offset_fold` `:4475`; caller `:4780-4790` |

Shared helpers, one copy per language:

| helper | Python | C++ |
|---|---|---|
| uniform test `allclose(h, h0, rtol 1e-12, atol 1e-15)` | `_static_is_toeplitz` `_bspline_kernels.py:691-700` | `is_toeplitz` `:330-339` |
| straight + uniform + contiguous edge, then kind by end heights | `_offset_edges` `bspline.py:5740-5769`, `_free_offset_edges` `:6923-6951` (**the same predicate written twice**) | `offset_edges_of(g, free_)` `:467-526` (one copy) |
| table layout (toeplitz reciprocal / toeplitz two-row / hankel) | `_offset_table` `bspline.py:5772-5793` | `offset_table` `:649-675` |
| strided window + basis-level assembly of the edge's own bases (wings off the edge zeroed and parked) | `_offset_basis_block` `bspline.py:6866-6921` | `offset_basis_block` `:573-647` (chunk = Python's `mb_rows` emulated, `:618-625`) |
| table sharing between edges with byte-equal arcs and radius (**only under `pair_reuse`**) | `_offset_table_sources` `bspline.py:6853-6864`, gated `:6786-6788` | inline loop `:1335-1347`, gated `fp.pair_reuse` `:1337` |
| pairs saved, n² − (2n − 1) | `_offset_saved` `bspline.py:5340-5342` | – (report is Python's) |

### 3.2 What it shares, and how it meets the census

- **Same uniform straight edge.** F, I and S need a straight, contiguous edge of equal segment lengths, within `_PAIR_REUSE_RTOL = 1e-12` of the structure's scale (`bspline.py:5749-5763`; `_bspline_fill_core.cpp:477-509`). R needs only the arc to be uniform (`_static_is_toeplitz`) and the edge to have ≥ 12 segments. A bs2 same-edge block is 1-D (arc length) whatever the edge's shape (`_bspline_kernels.py:840-877`).
- **Exact offset, or not?** Only R synthesizes exact offsets (`arange·h`). F, I and S read the edge's own rows (0 and n − 1) and spread them by index, so every member takes row 0's bits. The result is not byte-identical to the off fill: the registry declares `offset_reuse` drift 5e-11, measured worst on WA7ARK-OCF over the 821-deck exe corpus (`_features.py:369-373`).
- **Key and storage.** There is no hash and no class map. The key is the index alone: toeplitz `j − i + n − 1`, hankel `i + j` (`bspline.py:6899-6915`; `_bspline_fill_core.cpp:633`). Storage is a (d+1, d+1, 2n − 1) complex table per edge, or per shared edge group for F (`bspline.py:6790-6800`; `_bspline_fill_core.cpp:1334,1371`).
- **The census.**
  - bs2's census marks every same-edge pair as the unlisted sentinel (`_bspline_pair_reuse_core.cpp:113-117`), so the census and F/R never serve the same pair.
  - In the chunked fill, a gathered window's own-edge columns are set to zero on an offset edge and to the sweep's own moments otherwise (`_bspline_fill_core.cpp:1157-1182`; Python `_reuse_block` `bspline.py:6544`).
  - The two switches couple in two places: F's cross-edge table sharing needs `pair_reuse` (`bspline.py:6786`, `:1337`), and S steps aside for the remainder copy map unless the offset edges hold at least half the pairs (`bspline.py:5831-5836`, `:4285-4289`).
- **A fifth, ungated exact copy.** The dense J copies one same-edge block to any edge whose arcs and radius repeat to the bit (momwire#1772, `_bspline_fill_core.cpp:794-829`; buried route `:5243-5255`). No switch gates it.

### 3.3 razor2p and sinusoidal: near fill today, and what an offset stage would need

| | razor2p | sinusoidal |
|---|---|---|
| near pairs today | `Key::Own` singles, evaluated as group representatives by the same `seg_moments` as far pairs (`_razor_pair_reuse_core.cpp:138`, `:328-388`); no near-specific quadrature branch in `seg_moments_impl` (`_razor_core.cpp:215-300`) | `Key::Near`, evaluated in the original frame by `RowsFn` in 16-row blocks (`_sin_pair_reuse_core.cpp:431-471,954-981`; exe `direct_rows` / `image_rows`, `_sinusoidal_fill_core.cpp:2371-2376`) |
| on-run near band per row | \|v\| < 4·max(h, a) → offsets \|k\| ≤ 3 when h ≥ a: 7 pairs a row; ~4a/h when h < a | \|z\| < 4·max(h, a) → the same 7 when h ≥ a |
| far on-run pairs | already shared by the census (exact equality of quantised keys) | same |
| what an offset stage must supply | (1) runs: consecutive segments, collinear and contiguous, equal h, a and EK group (razor's `Geometry` has no wire or edge ids: `_razor_pair_reuse_core.h:222-237`); (2) a key, (run class, k, EK eligibility, tube b type); (3) evaluation, its own `seg_moments` on the representative (no new kernel); (4) blocks, direct (toeplitz) and PEC image (toeplitz horizontal / hankel vertical), with the remainder left to stage B | (1) runs, as razor, plus equal end codes `ind1/ind2` for interior segments (source types, `reversible_segment_types` `_sin_pair_reuse_core.cpp:615-644`); (2) key (run class, k, src/obs types); (3) evaluation in the **original** frame via `RowsFn`, never `CanonFn` (canonical-frame near pairs moved a bowtie's Zin 7.7e-11 against 6e-14 in the original frame, `_sin_pair_reuse.py:146-153`); (4) direct and image; reflected is stage B |
| payoff bound | ~7N of N² pairs, at the same per-pair cost as a far pair, so the census already holds the larger share. It pays only where h < a (wide near band) or on decks with little else to share. | the same bound |

**Razor2p and the NEC-5 twin rule.** An offset stage that copies razor's own `seg_moments` value from a representative pair at the same run offset solves the same integral, so it is a speed-up and allowed. Three things would cross the line or need care:

1. **Do not import bs2's same-edge formulation.** bs2's tables come from an analytic static part plus a regularised part on the 1-D arc (`_free_offset_table`, `bspline.py:6953-6977`). razor's offset values must be its own `seg_moments`. Using bs2's static/reg split for razor's near pairs would be a different near-term model.
2. **This is the hairpin hazard, measured.** razor excludes near pairs because sharing bridge-on-bridge pairs a few h apart, whose v differed only by coordinate rounding, moved Zin 1.5e-5 (reduced kernel, exact_kernel off) while every pair passed `verify` at 2e-13 (`_razor_pair_reuse_core.h:57-73`; probe `PROBE_NEAR_H`, `.h:295-306`). That bridge is 21 straight segments of 95 µm on a 1 mm radius, a uniform run of ≥ 12, so an offset stage would share exactly those pairs again. This is an accuracy risk, not a modelling one, and the stage-A drift gate must run this deck.
3. **The exact-kernel correction subtracts "the base moments through the fill's own kernel"** (`_razor_exact_core.h:1-12`). If the fill's near values become copies, the correction's base no longer matches them bit for bit, and the residual is the copy drift. The same question applies to bs2's correction under offset (`offset_reuse` already runs there, drift 5e-11).

---

## 4. The switches

### 4.1 Resolution

| | keyword | env | pragma | registry | exe |
|---|---|---|---|---|---|
| `pair_reuse` (whole) | `pair_reuse=` on all three solvers: razor `razor.py:2073,2173`; sinusoidal `sinusoidal.py:1329,1443`; bs2 `bspline.py:1894,1954-1960` | `MOMWIRE_X_PAIR_REUSE` | `CM momwire: pair_reuse` (`pragma=True`, `_features.py:343`) | reach 11, python (`_features.py:284-344`) | `EXE_FEATURES` reach 11 (`standalone/momwire_razor2p.cpp:449-450`); mapped per basis by `EXE_PARTS` (`:497-501`) |
| `offset_reuse` (part, bs2) | `offset_reuse=` (`bspline.py:1895,1961,1998`) | `MOMWIRE_X_OFFSET_REUSE` | through the whole only | reach 9, python, `part_of="pair_reuse"` (`_features.py:358-380`) | reach 9 (`momwire_razor2p.cpp:451`), bs2 only (`EXE_FEATURE_BASES` `:467-471`) |
| `razor_pair_reuse` (part, razor) | `razor_pair_reuse=` incl. `"verify"` (`razor.py:2072,2153-2157,2174-2176`) | `MOMWIRE_X_RAZOR_PAIR_REUSE` | through the whole only | reach 8, **native**, `part_of="pair_reuse"` (`_features.py:736-795`) | reach 8 (`:456`), razor2p only (`:468`) |

The order (`_features._resolve`, `:1340-1358`): the part's keyword, then the whole's keyword, then the part's env, then the whole's env (a pragma reads as the env set on, `:1311-1312`), then the default.
- A native part that is asked for only through the whole, outside its reach, is **withheld** rather than refused (`enabled` `:1361-1385`, `withheld` `:1394-1405`, `bind` `:1414-1488`, kept on `features_withheld` `:1485`).
- Line 2 names a whole only where one of its parts runs (`_runs_on` `:1662-1679`). Certification checks each part that runs (`_certified_as_run` `:1712-1726`).

Exe side:
- `features_requested` reads `MOMWIRE_X_*` (`momwire_razor2p.cpp:554-574`).
- `configure_asked` maps a whole to its part (`:672-712`). When the part's own variable is set off, the whole stays unstamped, and the bs2 census slot still runs (`:697-707`).
- Pragmas re-configure each deck (`DeckPragmas` `:840-880`; `deck_pragmas` `:783`).
- The slots live in `ni_eval::Features` (`_near_interface_eval.h:222,236-243`; `feature_slot` `_near_interface_eval.cpp:1035-1050`; the line-2 stamp `:1074-1091`).
- Readers: razor `_nec5_razor_seam_core.cpp:184` (`razor_pair_reuse`); bs2 `_nec5_bspline_printout_core.cpp:62-63` (`pair_reuse`, `offset_reuse`); sinusoidal `_sinusoidal_fill_core.cpp:2269` (`pair_reuse`).

### 4.2 Every reader site stage D will touch

Python, momwire:
- `_features.py`: the two rows (`:358-380`, `:736-795`); `part_of` logic (`:228-232,1340-1358,1361-1405,1414-1488,1662-1726`); the module docstring (`:24-41`).
- `razor.py:2072-2232` (keyword, validation, `bound["razor_pair_reuse"]`, `features_withheld` check `:2226-2230`, `_cxx_kwargs["pair_reuse"]` `:2232`); `:4900-4904`, `:7696-7697`, `:7872`.
- `bspline.py:1338-1369` (decorator), `:1895,1947-2000` (keyword, bind, `_offset_reuse`, `_fill_reuse_asked`); every `getattr(self, "_offset_reuse")` (`:5747,6930`); `_bspline_kernels.py:945-958` (scope).
- `momwire_serve_client.py:427` (`FEATURE_PARTS`, part-off key identity, `:450`).

C++ / exe:
- `standalone/momwire_razor2p.cpp:440-501` (EXE_FEATURES rows, EXE_FEATURE_BASES, EXE_PARTS), `:682-712` (part mapping), `:176` (comment).
- `_near_interface_eval.h:222,236-243`; `_near_interface_eval.cpp:1045,1049,1077,1087`.
- `_nec5_razor_seam_core.cpp:183-184`; `_nec5_bspline_printout_core.cpp:60-63`.
- `scripts/eznec_client_c/momwire_eznec_client.c:315-316` (the feature-name list).

Docs: `site/src/content/docs/reference/portal-usage.md:459-463`; `site/src/content/docs/reference/speed-vs-nec5.md:119`; `.gitattributes:6` (comment).

Tests, by `git grep -c`: `test_pair_reuse_contract_1982.py` (60), `test_standalone_nopython.py` (26), `test_features_registry.py` (17), `test_feature_solvers_1980.py` (16), `test_pair_reuse_chunked_2020.py` (15), `test_razor_pair_reuse_1971.py` (14), `test_fill_report_ground_2068.py` (9), `test_offset_reuse_1809.py` (8), `test_stable_path_tripwire.py` (7), `test_razor_reuse_rows_2144.py` (5), `test_deck_pragma.py` (4), `test_pair_reuse_1809.py` (4), `test_deck_pragma_dropin.py` (3), `test_fill_report_2016.py` (3), `test_standalone_nec42.py` (2), `test_dropin_feature_refusal_1971.py` (1), `test_razor_core_standalone.py` (1).

**Outside momwire (Steve's call):**
- AK passes `offset_reuse=True` behind a probe (`AK:src/antennaknobs/engines/momwire.py:1615-1616,2424-2425`; doc `:797,1463`).
- AK offers the checkbox by each part's reach (`AK:src/antennaknobs/web/adapter.py:928-931`; frontend test `panelEquivalence.test.tsx:403`).

A deprecation warning on `offset_reuse` would therefore fire on every AK bs2 fill until AK drops the keyword.

---

## 5. PROPOSAL: the stage-A interface (not evidence)

### 5.1 Principle

Every value-bearing operation stays in the basis: rows, the assembly, the `seg_moments` / `RowsFn` calls. The core takes over the decisions and the index arithmetic: which runs qualify, which runs share a table, where (i, j) sits in a table, the chunking, and the counts. This is the rule `_pair_reuse_core.h:12-15` already applies, with integer-only code in the core's TU and float code in templates instantiated in the adapter's TU. Under it, bs2 can move byte-identically because the core gains no arithmetic that reaches Z.

### 5.2 A0: bs2's F / I / S onto the core, byte-identical

```cpp
// _pair_reuse_core.h, namespace mw::pair_reuse::offset (new)
enum class Kind : uint8_t { Toeplitz = 1, Hankel = 2 };   // j - i ; i + j
struct Edge { int64_t s0, s1; };                          // the basis's same-edge slices, in order
struct Run  { int64_t s0, s1; Kind kind; int64_t table_of = -1; };  // -1: its own table
struct RunRule {
    int64_t min_n = 12;     // _REG_TOEPLITZ_MIN_N
    double  rtol  = 1e-12;  // _PAIR_REUSE_RTOL x structure scale; h by allclose(rtol, atol 0)
    bool    any_orientation;// F: every straight uniform edge is Toeplitz; I/S: horizontal/vertical only
};
// `_offset_edges` / `_free_offset_edges` / `offset_edges_of`: ONE copy, header-inline
// (instantiated in the adapter TU, -ffp-contract=off), same arithmetic order as today.
inline std::vector<Run> uniform_runs(const double *seg_l, const double *seg_r, int64_t n_seg,
                                     const std::vector<Edge> &edges, const RunRule &);
// `_offset_table_sources`: first run in order whose input bytes match (arcs + radius bits).
void share_tables(std::vector<Run> &, const std::function<std::string(const Run &)> &input_bytes);
// `_offset_table`: (width, 2n-1) from row(0) [and row(n-1)]; `transpose` maps the
// width axis under reciprocity ((p,P)->(P,p) for bs2; null: no reciprocal form).
template <class T>
std::vector<T> table(const std::function<std::vector<T>(int64_t)> &row, int64_t n, int width,
                     Kind, const int32_t *transpose);
inline int64_t at(Kind k, int64_t n, int64_t i, int64_t j) {          // the only "key"
    return k == Kind::Toeplitz ? j - i + n - 1 : i + j; }
template <class T>  // rows [c0, c1) of the run's own block, (width, c1-c0, n), a copy
void window(const T *tab, int64_t n, int width, Kind, int64_t c0, int64_t c1, T *out);
int64_t chunk_rows(int64_t n, int width, double mem_mb);  // `max(1, min(n, mb_rows(min(mb,8), w*n*16)))`
int64_t saved_pairs(const std::vector<Run> &);            // `_offset_saved`
```

What bs2 supplies:
- edges (`_same_edge_slices` / `edge_ingredients`);
- the row callbacks: F static + reg, I image moments, S remainder projection plus einsum;
- `transpose`;
- the assembler (`assemble_Z_bspline_windowed` / `_weighted_windowed`), which stays in `offset_basis_block`'s caller.

Python binds `uniform_runs`, `share_tables`, `table` and `window` through `_accel_bspline_pair_reuse.cpp`. Python can keep its numpy copies, which are exact; only the run list and the chunking must match. R (the reg-Toeplitz `ContextVar` and its 6 C++ sites) is a property of bs2's 1-D same-edge kernel geometry, not of a pair map, and it **stays bs2's**. Only the switch reading it changes in stage D.

### 5.3 A1: an offset key in the census, for razor and sinusoidal

```cpp
enum class Key : uint8_t { Shared, Own, Near, Offset };   // Offset: hashed like Shared,
                                                           // Rep::single = 3, counted apart in the report
struct RunIndex { std::vector<int32_t> run_of_seg; std::vector<int64_t> pos;
                  std::vector<int64_t> table_of; std::vector<Kind> kind; };   // from uniform_runs
// in the adapter's near branch: on success w[0..2] = {TAG | table_of(run), kind | at(...), 0}
// and the adapter fills its own type words (razor: EK eligibility, tube b; sin: src/obs types)
inline bool offset_key(const RunIndex &, int block_kind, int64_t o, int64_t s, uint64_t *w);
```

- **razor:**
  - at `_razor_pair_reuse_core.cpp:138` replace `return Key::Own` with `if (offset_key(...)) { ...; return Key::Offset; } return Key::Own;`;
  - the near pairs become ordinary classes, so `eval_direct`, `gather_direct` and `assemble_rows` are unchanged;
  - W stays 4 if the tag is packed in w[3], which is non-zero for every quantised key (`e + 1100`, `:152`);
  - razor needs `RunIndex` built from `cent`, `p0`, `t`, `seg_h` and `a`, because its `Geometry` has no edges.
- **sinusoidal:**
  - its direct classes are evaluated in the canonical frame (`evaluate` `_sin_pair_reuse_core.cpp:912-947`), which near pairs must not be;
  - keep the on-run near pairs as `Key::Near` (listed), and have the core return a second-level class per listed near pair (`FamilyPlan::near_cls`, numbered by first appearance in `near_flat`), so `near_blocks` evaluates one representative per near class and `evaluate_near` fans the values out;
  - the class map, EVAL_CHUNK slicing and canonical evaluation are then untouched.

### 5.4 Risks to byte-identity

| # | risk | A0 | A1 | mitigation |
|---|---|---|---|---|
| 1 | the run predicate's float order (`np.linalg.norm` = `sqrt((x²+y²)+z²)`, `allclose`) | yes (a decision) | yes | keep `offset_edges_of`'s spelling (`_bspline_fill_core.cpp:493-507`); header-inline, `-ffp-contract=off`; gate the run list Python == C++ |
| 2 | row evaluation differs per route (numpy kernels vs C++) | already, and gated exe == Python | – | rows stay route-owned callbacks |
| 3 | chunk size sets the assembler's `+=` order into B | yes | – | `chunk_rows` must reproduce `mb_rows`, including the int/float branch (`_bspline_fill_core.cpp:618-625`) |
| 4 | table sharing: Python (arc bytes, `float(a_w)`) vs C++ (memcmp, `==`); first match in edge order | –0.0 / NaN corner cases | – | `share_tables` on bytes in both; one copy |
| 5 | Z accumulation order (F after the sweep and fixup, I after the image sweep, S per fold) | yes | – | the core supplies tables and windows only and never adds to Z |
| 6 | Windows MSVC `/fp:fast` (`standalone/CMakeLists.txt:383`, `setup.py:208`) | no bit-compat required there | | — |
| 7 | the census's class set changes (Offset classes), so group unions change | – | razor: values are per-pair and numbering-independent (`_pair_reuse_core.h:26-29`), so only near values should move; must be shown | gate far pairs bit-equal with A1 on, near pairs within drift |
| 8 | sinusoidal EVAL_CHUNK / class order moves ulps (`_sin_pair_reuse_core.h:38-42`) | – | avoided by the `near_cls` route | keep offset out of the canonical classes |
| 9 | near-pair copy drift (hairpin 1.5e-5, bowtie 7.7e-11), and the exact-kernel base mismatch | – | **by design not byte-identical** | gate on the #1971 hairpin (exact_kernel False / auto), close-spaced lines, and the bowtie array; flag-off unchanged |
| 10 | the core TU must stay integer-only (momwire#1187) | yes | yes | float helpers inline in the header only |

---

## 6. Open questions for Steve, and contradictions with the issue

1. **Will an offset stage pay for razor2p and sinusoidal?** Their censuses already share the far on-run pairs. What is left is ~7 near pairs a row at ordinary per-pair cost (§3.3). bs2's offset pays because (a) its census excludes the whole same-edge block and (b) same-edge moments are dearer (analytic static + reg). Neither holds for the other two. The larger near band (h < a) is exactly the hairpin hazard. Proposal: in stage A, ship A0 and only *measure* A1 (the near-pair share of fill time from the fill report). Enable A1 only if Skylake fill time moves.
2. **The hairpin gate.** Is the #1971 hairpin (exact_kernel False) the acceptance deck for razor's A1, at the registry's 2e-9 drift? It moved 1.5e-5 when near pairs were shared by rounded keys.
3. **Exact-kernel consistency.** Should A1 decline under `exact_kernel` (razor, `_razor_exact_core.h:1-12`) until the base moments are taken from the same copies?
4. **Stage D needs a registry shape.** Per-basis reach on one feature is not expressible today: `offset_reuse` is reach 9 python and `razor_pair_reuse` is reach 8 native under a reach-11 whole, and `withheld`, `_runs_on` and `_certified_as_run` all read `part_of`. Also lost: the combinations "census on, offset off" (`MOMWIRE_X_OFFSET_REUSE=0`, honoured at `momwire_razor2p.cpp:697-707`) and "offset alone". Is losing them intended?
5. **AK.** AK passes `offset_reuse=True` (`AK:engines/momwire.py:2424-2425`) and offers the checkbox by part reach (`AK:web/adapter.py:928-931`). The two-release deprecation warning would show on every AK bs2 fill until AK changes. Should that be an AK PR alongside the pin, or a silent alias?
6. **R (reg Toeplitz).** Keep it as bs2's kernel concern (this survey's view), or count it as part of the core's offset stage? It has a different key (exact synthesized offsets) and 6 explicit C++ sites against 1 ambient Python one.
7. **Contradictions with the issue text:**
   - "offset_reuse … the generic offset stage": it is four mechanisms, and only F/I/S fit the table form above.
   - The addendum's "shares exactly those [near] pairs": no; it shares whole same-edge blocks (§0).
   - "the core holds the census, the plan cache, the 4/3 class cap and the report counters": correct. The small-deck decline, the evaluation schedule (sinusoidal: `rep_rows = 0`) and the fingerprint are still per basis.
   - Not raised by the issue: `_sin_pair_reuse_core.h:25-28` still documents the old M·N/2 cap.
   - A fifth exact reuse, the momwire#1772 identical-edge block copy, runs ungated (`_bspline_fill_core.cpp:794-829`). Stage D's "one switch" does not cover it, and it should not need to (it is bit-exact).
