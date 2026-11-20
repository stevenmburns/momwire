# One fill-pair-reuse core for every basis: survey and design

2026-11-20. Survey of momwire at `d5feb5c7` (origin/main). This is read-only:
no `src/` or `tests/` file was changed, and nothing was built or measured.
Every number below is quoted from a code comment or an issue, and its source is
cited. Paths are relative to the momwire root unless they start with `AK:`
(antennaknobs).

## 0. Summary

The brief counts three implementations. In fact there are four. Razor's is
already the common shape, and the other two differ from it more in how they
are written than in substance.

| | where the census runs | where plan, values and copy run | who runs it |
|---|---|---|---|
| razor-2p | C++, `_razor_pair_reuse_core.cpp` | the same C++ | Python route (binding) and exe, **one copy** |
| sinusoidal, Python route | numpy + `IntGroups`, `_sin_pair_reuse.py` | numpy + the fused C++ reduction | Python route, the drop-in |
| sinusoidal, C++ route | C++, `_sin_pair_reuse_core.cpp`, "Python's plan to the byte" | the same C++ | nec42-sinusoidal exe (certification build only) |
| bspline | **none in momwire**: AK's `structure.pair_reuse_hint` finds the classes | Python validates and copies J blocks; C++ fused remainder kernel copies by a map | Python route only |

What the three bases have in common is larger than the brief assumed. All of
them copy segment-level moments: values per (observer, source segment) pair,
taken before the basis assembly. None copies basis-function entries of Z. So
the basis footprint (the tent's two segments, the sinusoid's own segment, the
bspline's neighbours) never enters a reuse plan. What is genuinely per basis is
short:

1. what a pair's value reads of the geometry (the key);
2. how the value transforms when a tangent is reversed;
3. the near rule;
4. how many values a pair carries;
5. how the kernel is batched;
6. one structured special case (bspline's `offset_reuse`).

Everything else is the same idea written two or three times: hash census,
first-appearance classes, budgets, declines, the cache, evaluation groups, the
gather, the counters and the wire map. That shared part is the proposed core:
`src/momwire/_pair_reuse_core.{h,cpp}`, extracted from razor's core.

---

## 1. Side by side: what each implementation does at each stage

### 1a. Census

| | razor-2p | sinusoidal (Py ref = C++ copy) | bspline |
|---|---|---|---|
| unit | (centroid observer o, source segment s), point-matched; `_razor_pair_reuse_core.h:8-25` | (observer centre m, source segment n), point-matched; `_sin_pair_reuse.py:3-19` | **(edge A, edge B) block** of the Galerkin J tensor `(d+1,d+1,N,N)`; `bspline.py:5716-5734` |
| who finds classes | the core itself | the fill itself | **AK**: `AK:structure.py:265` (`pair_reuse_hint`); momwire only checks it |
| what the value reads | u_r, perp, h, a, EK eligibility, EK radius, observer tube b; tangents enter later (`.h:14-21`) | z, rho, td, rdt in the source frame + half-length, observer radius, EK radius and end codes (`.py:8-19`) | all segment-pair distances (isometry), per-segment radii, EK pattern (`bspline.py:5720-5726`) |
| key (direct) | 4 words: \|v\| = \|u_r - h/2\| and rho quantised against D = max(\|v\|-h/2, rho, a), floored to a power of 2; h exact by type id; exponent + EK/type/b ids packed (`.cpp:475-509`) | 5 columns: exponent+type, z, rho, rdt quantised against max(\|z\|-H, rho, a_obs); td at absolute 2^-48; H quantised at 48 mantissa bits (`.py:537-541, 654-694`) | AK: 6 end-point distances on a 1e-9 m grid + same-edge flag + segment counts + materials, 64-bit FNV hash (`AK:structure.py:232, 330-345`) |
| KEY_BITS | 40 (`.h:118`) | 48 (`.py:141`) | grid 1e-9 m (AK); check 1e-12 × structure size (`bspline.py:867`) |
| hash table | `Table<W>` open addressing, sharded over P = min(32, omp threads) parts by hash bits (`.cpp:62-103, 254-386`) | `IntGroups` (`_int_groups_core.h`), bands folded in order into one running table (`.py:915`, `_sin_pair_reuse_core.cpp:331`) | numpy argsort on hash (AK) |
| class numbering | per-part local numbers, then offset by part (`.cpp:363-378`); rep = first pair in row-major order | first appearance in flat order, then a stable sort on the rep's observer type (direct) or flat index (ground) (`_sin_pair_reuse_core.h:10-17`) | AK's hash order; rep = class head |
| orientation | v folded to \|v\|, 1 flip bit (`.h:27-32`) | canonical of 4 orientations, 2 code bits (`.py:99-123, 696`) | **none**: end order only, k-to-k (`AK:structure.py:270-283`); #1809's reversal never built |
| mirrors | free: the key is reflection-invariant | free: complete invariants | via distances (any isometry) |
| PEC image | the same direct key on mirrored sources; block 1 of one class space (`.h:22-25`) | its own plan on the image sources ("image" block) | **not copied**: the image J is filled in full (`tests/test_pair_reuse_1809.py:158`) |
| finite ground | Sommerfeld: image = C2 × the PEC key; remainder its own 9-word height-aware family (`.cpp:537-600`); refl-coef also (`.h:131`) | "reflected" ground plan: height sum, \|d\|, d.a, d.b, a.b, t_z (`.py:67-97`, `build_ground_plan:1279`) | remainder only, a block copied when both edges keep their heights (`bspline.py:5897-5952`) |
| near rule | max(\|v\|, rho) < 4·max(h, a): own class, "single" (`.h:57-73`, `.cpp:491-492`) | max(\|z\|, rho) < 4·max(2H, a): sentinel key, direct eval (`.py:151, 647-668`) | the same-edge blocks are never copied off-edge (the analytic static + reg overwrite them, `bspline.py:6410`) |

### 1b. Plan and validation

| | razor-2p | sinusoidal | bspline |
|---|---|---|---|
| map | int32 per pair per block, class<<1\|flip; remainder class<<2\|code (`.h:255-264`) | int32 (M,N) class<<2\|code (`_sin_pair_reuse_core.h:118`) | `_PairReusePlan` arrays ta,tb ← sa,sb (`bspline.py:997`) |
| reps evaluated | `Group`: ≤16 observer rows × the union of their reps' sources, joined while the call evaluates ≤2× what it needs (`.cpp:156-235`) | direct: the canonical observer (origin, +z) against EVAL_CHUNK=2048 sources at one observer radius (`.py:167-175, 1453`); ground: original frame, REP_ROWS=16 (`.py:157-163`) | representative block filled by the off-edge kernel on its own columns (`bspline.py:6363-6390`) |
| validation against geometry | none needed: exact-equality classes of quantised keys + `verify` mode (VERIFY_TOL 1e-9, `.h:125`) | the same; `verify` VERIFY_TOL 1e-7, 512 samples a band (`.py:183-212`) | **every member checked**: segment counts, 6 distances within 1e-12·scale, radii and EK pattern elementwise; failures dropped, named, `PairReuseDropped` (`bspline.py:5746-5873`) |
| cache | process LRU by geometry fingerprint bytes, SHARED_PLAN_BYTES = 256 MB (`.h:218`, `.cpp:612-706`) | process LRU `shared_plan(key)`, 256 MB (`.py:311, 418`; C++ `_sin_pair_reuse_core.h:158`) | per solver (`_pair_reuse_plan_cache`) + the last hint conversion (`bspline.py:966`); AK caches the hint per mesh (`AK:engines/momwire.py:2306`) |

### 1c. Budget and decline

| | razor-2p | sinusoidal | bspline |
|---|---|---|---|
| small deck | <400 segments free/PEC, <100 finite (`.h:152-153`) | <400 free/PEC, <200 finite (`.py:255-256`) | <8 copied pairs per member per fill thread, from a 2048-member sample of the hint (`bspline.py:877-880, 5649-5704`) |
| compression | classes > 0.75 × pairs → "too few shared pairs to repay the plan" (`.h:187`, `.cpp:390-393`) | **none** (#2047 asks for one) | the per-member floor above |
| bytes | PLAN_BUDGET 256 MB on the census peak + plan: real C++ transients at 1.5× capacity, CLASS_BYTES 41 (`.h:188-209`, `.cpp:280-304`) | 256 MB under **Python's** accounting: BAND_BYTES_PER_PAIR 136, even in C++ (`_sin_pair_reuse_core.h:19-28`) | remainder map cut to the kernel's 64 MB slab (`bspline.py:881-885`); chunked route holds source blocks in 0.75 × `swept_mem_mb` (`bspline.py:1088`) |
| early exit | checked after every 64-row chunk (`.cpp:349-360`), but the poor bound is 0.75 of ALL pairs, so it cannot fire before 75 % of the census has run | per band, bytes only | sampling 2048 members, then unread |
| route refusals | `_pair_reuse_refusal` (`razor.py:4942-4977`): path rule, chop, complex k, tokens | `_sin_pair_reuse_route` (`sinusoidal.py:6970-7004`): NEC-4 junction kernel, folded cos, ground medium, sector route | `_decline_unread_hint` (buried, batched sweep, chosen rows) (`bspline.py:7133`) |

### 1d. Copy

| | razor-2p | sinusoidal | bspline |
|---|---|---|---|
| where | `gather_direct` into the window's (n_need, n_seg) M0/M1 (`.cpp:864-892`), then the shipped contraction | fused into the Φ→Z reduction, `reduce_band` (`_sin_pair_reuse_core.h:205-228`) | `copy_blocks`: J[...,target] = J[...,source] (`bspline.py:1041`); chunked: held source blocks (`_ChunkedReuse`, `:1118`); remainder: C++ `reuse_src` map (`_accel_somm.cpp:276-473`) |
| orientation transform | flip: M1 → h·M0 − M1 (`.cpp:884-886`); remainder: (Q0,Q1) → (−Q0, Q1 − h·Q0), observer flip negates both (`.cpp:913-927`) | `CODE_SIGNS`: source reversed (−,+,−) on (const, sin, cos); observer reversed: all negated (`.py:271-281`) | none: copies are the source's bits |
| value per pair | 2 complex (`.h:286-291`) | 3 complex (`.h:196`) | (d+1)² complex per segment pair (9 at d=2) |

### 1e. Reporting (#2016)

| | razor-2p | sinusoidal | bspline |
|---|---|---|---|
| counts | `Plan` fields: pairs, near_pairs, q_pairs, rep_evals, `dcls_block`/`dnear_block`/`devals_block`, census and budget bytes (`.h:265-281`) | `pair_reuse_census` (pairs, classes, near) per block (`.h:175-182`) | `pair_reuse_served` dict + `_fill_served` (`bspline.py:6413-6430, 7106`) |
| wire map | C++ `wire_map` over the class maps, own TU (`_razor_fill_map_core.cpp`) | C++ `fill_report::wire_map`, own TU (`_accel_fill_report.cpp`) | numpy over edge blocks (`bspline.py:5611-5647`) |
| families | direct, image, remainder | direct, image or reflected | direct only; image and remainder "uncounted" (`bspline.py:5593-5597`) |

---

## 2. Incidental against genuine differences

### The same idea written more than once (the core absorbs these)

- **Exact-equality hashing of quantised keys.** `Table<W>` and `IntGroups` are
  two open-addressing tables over int64 words. Both are integer-only. AK's FNV
  hash is a third, and the only lossy one: it relies on momwire's check to
  catch collisions (`AK:structure.py:241-245`).
- **The quantum.** Each key quantises against a power-of-two floor of a
  per-pair singular distance. The KEY_BITS values differ (40 against 48) and
  so do the D formulae. #1982 calls those per-engine tuning, and they stay
  adapter constants. The mechanism (`quantum`, `qword`; numpy's exponent
  bit-trick) is the same.
- **The orientation fold.** Key each pair in the canonical orientation of its
  2 or 4 and carry the flips in the low bits of the map entry. Razor's direct
  key folds 1 bit, its remainder 2 bits; sinusoidal folds 2 bits. This is one
  mechanism, and only the transform it implies is per basis.
- **The near carve-out.** "Pairs within 4 kernel scales are not shared." The
  rule max(h, a) was found on razor and ported to sinusoidal (#1982,
  `1ae61296`). The representations differ (razor: one class per near pair,
  "single"; sinusoidal: a sentinel key and a `near_flat` list). That is
  incidental.
- **Evaluation groups.** Rows × the union of their reps' sources. Razor
  (`build_groups`) and sinusoidal ground (`REP_ROWS`, `near_blocks`) use the
  same 16-row heuristic.
- **The budget.** 256 MB plan + census, one dense Z, headroom (Steve 11-17).
  The number is the same everywhere; the ACCOUNTING differs (real C++
  transients against Python's numpy figures).
- **Decline outcomes.** Small deck, poor compression, over budget, uncodeable.
  The strings are already shared between razor and sinusoidal ("too few
  segments to repay the plan (N < x)").
- **The process cache.** An LRU by geometry key under SHARED_PLAN_BYTES. There
  are two copies, so a long-lived process holds up to 2 × 256 MB of plans
  today (3 × once bspline has one).
- **Counters and the report.** Built, found and declined; per-family
  pairs/classes/near/evals; the wire map. That is two C++ wire maps and one
  numpy one.
- **Threads.** Razor and the sinusoidal C++ route use OpenMP. The Python
  sinusoidal route uses its own `ThreadPoolExecutor` (`.py:830-858`), which is
  #2047's second finding (25 against 31 threads on Windows).

### Genuine per-basis needs (each adapter supplies these)

1. **The key: what the kernel reads.** Each basis has its own key.
   - Razor's moments read no tangent of the observer (`.h:14-21`).
   - Sinusoidal's Φ is projected on the observer tangent, so td and rdt enter
     (`.py:8-19`).
   - bspline's J is Galerkin, segment against segment. It reads the full
     relative pose of two segments: six distances of four end points, or an
     equivalent complete invariant, plus both lengths, the observer-row radius
     (`_bspline_kernels.py:1580-1586`) and the EK pattern. This is a 2-segment
     key, not a point-segment one.
2. **The orientation transform.** A reversed tangent maps local s → h − s.
   - Razor (degree-1 source moments): M1 → h·M0 − M1.
   - Sinusoidal: parity signs.
   - bspline: J[p,P] holds monomial moments u^p, u'^P on both segments
     (`_bspline_kernels.py:143-153, 1862`). So a flip is a binomial
     (d+1)×(d+1) triangular transform on each side, applied as B_i · J · B_jᵀ.
     It is the d-degree generalisation of razor's rule, and it is arithmetic,
     so a flipped copy is not the representative's bits (§5).
3. **Values per pair and their bytes**: 2, 3 or (d+1)² complex. This sets
   `value_bytes` in the budget.
4. **The near rule's meaning.**
   - Razor and sinusoidal: pairs whose near-singular part the basis sums
     cancel, so any re-rounding is amplified (`.h:57-73`).
   - bspline: the same-edge block is the analytic static + reg overwrite, so it
     is never a census candidate. Off-edge touching segments at kinks are full
     kernel values with a² regularisation (`_bspline_kernels.py:1574-1578`),
     and they need the same near carve-out.

   The scale rule max(h, a) is shared; what is excluded differs.
5. **Kernel batching and its bits.**
   - Sinusoidal's batched sincos makes a value depend on its position in the
     call (`.py:172-175`), so the evaluation call shapes are part of the plan's
     contract, and class numbering matters.
   - bspline says a pair's moments do not depend on the call's other pairs
     (`bspline.py:6349-6352`, the per-pair phase guard).
   - Razor's group composition is numbering-independent (the union is sorted,
     `.cpp:186`).

   So "which calls evaluate the reps" is the adapter's, and the core offers
   razor's grouping as the default.
6. **Ground terms per basis.**
   - The *invariance group* is common: horizontal isometries for anything read
     through the ground plane.
   - What is keyed differs. Razor keys the PEC image like the direct block and
     the remainder by height sum. Sinusoidal keys "reflected" = C2·Φ_img − S as
     one term. bspline copies the remainder through a C++ kernel's slab cache.

   The core gives families ("direct", "image", "reflected", "remainder") and
   each basis declares which it has and how each is keyed.
7. **Junctions: not a reuse concern in any basis.**
   - Razor's T1/T2 contraction reads the pair's own tangents and wing signs
     after the gather (`.h:19-21`, `gather_remainder`'s row sign).
   - Sinusoidal folds junction coefficients in the reduction, unchanged
     (`.py:60-62`).
   - bspline assembles by `supp`/`polys` after J.

   The junction-specific refusal is sinusoidal's NEC-4 junction kernel
   (`sinusoidal.py:6976`). That is a route refusal, the adapter's.
8. **offset_reuse's within-edge case is genuine and stays bspline's.**
   - It serves exactly the pairs every census excludes: the same-edge reg
     moments, which are near pairs.
   - It works from analytic structure: a uniform straight edge depends on
     j − i (Toeplitz), or on i + j for the image and remainder of a vertical
     edge (Hankel) (`bspline.py:5954-5994`, `_bspline_kernels.py:953-983`).
   - It needs O(N) values with no map, and it composes with block copies
     (#1982 comment 11-17).
   - The core can host it as an optional "structured runs" pre-pass. Razor and
     sinusoidal could later use the same pass on their near pairs along
     uniform straight wires. That would be an improvement, not a port.

---

## 3. AK's hint

The hint carries nothing the core cannot compute itself:

- It is built from the engine's own edges (`AK:structure.py:197`,
  `edges_of_engine`), which the solver also has.
- Its 64-bit hash classes are a subset of what a segment-level census finds.
  Congruent edge blocks give congruent segment pairs. The census also finds
  repeats inside blocks and flipped pairs: the hint drops a member whose edge
  runs against its representative (`AK:structure.py:281-283`). #1809's
  comment of 11-05 counted that at 1.18-3.79× (blocks) against 3.3-10.6×
  (segments) on Yagis.

**The hint stops being necessary at stage 3a**, when bspline runs the core's
census. It does not survive as a performance input either:

- Checking it cost more than the fill on a wire grid (#1854).
- The executables have no front end to supply it, so a hint-only design can
  never reach the nec5-bspline exe. That is the argument that settles it.

**The kwarg stays.** `pair_reuse=<list>` is still accepted, and read as True,
for at least one pin cycle:

- AK passes it today (`AK:engines/momwire.py:2291-2297`).
- AK imports `pair_reuse_min_pairs_per_member` from momwire's private surface
  (`AK:mw/_private.py:73`).
- AK then drops the hint behind a live probe, not a version check (the
  `_takes_pair_reuse` pattern). The probe should test a capability attribute
  such as `BSplineSolver.PAIR_REUSE_CENSUS = True`.

That last change is AK work, and so Steve's call.

There is one honest cost. The hint route copies exact bits (Z moves ≤1e-13
against the off fill, `tests/test_pair_reuse_1809.py:91-96`). A quantised
census moves Z by rounding at the key's resolution (2^-41 of the pair scale at
40 bits). So the bspline drift goes from ~1e-13 toward razor's measured 1e-13
to 4e-10 (`_features.py:714-742`). That is still under the whole switch's
declared 5e-10 only if measured. See §5.

---

## 4. The proposed core

### Where it lives

- `src/momwire/_pair_reuse_core.h` and `_pair_reuse_core.cpp`, Python-free,
  namespace `mw::pair_reuse`. It is built in its own TU, as every reuse file is
  today (`setup.py:589-628`'s comments, momwire#1187).
- `standalone/CMakeLists.txt` adds `${MW_SRC}/_pair_reuse_core.cpp` for
  **every** basis. Today `_razor_pair_reuse_core.cpp` is listed for razor at
  `:267` and `_sin_pair_reuse_core.cpp` for sin42 at `:331`.
- `setup.py`'s `_accelerators` source list gains it. The adapters stay where
  they are:
  - `_razor_pair_reuse_core.cpp`: razor's adapter, its key lambdas and
    `seg_moments` calls;
  - `_sin_pair_reuse_core.cpp`: sinusoidal's;
  - a new `_bspline_pair_reuse_core.cpp`.

### Types and functions

```cpp
namespace mw::pair_reuse {

// A family: one class space over n_blocks blocks of (rows x cols) pairs.
// razor: {direct, W=4, code_bits=1, blocks=1|2 (direct, PEC image)}, {remainder, W=9, code_bits=2}
// sin:   {direct, W=5, 2}, {image, W=5, 2} | {reflected, W=6, 2}
// bspl:  {direct, W=?, 0|2}, later {remainder, ...}
struct FamilySpec {
    const char *name;
    int key_words, code_bits, n_blocks;
    int64_t rows, cols;          // observers x sources (sector route: rows < cols)
    int64_t value_bytes;         // per class per k: razor 32, sin 48, bspline 16*(d+1)^2
};

enum class Numbering : uint8_t { FirstAppearance, PartOffsets /* razor today */ };
enum class NearRepr  : uint8_t { OwnClass /* razor */, SentinelList /* sin */ };

struct Policy {                 // the adapter's constants, by name
    int64_t budget_bytes;       // PLAN_BUDGET_BYTES, test hook
    double  max_class_fraction; // razor 0.75; sin: 1.0 (none) until decided
    int64_t small_deck_free, small_deck_ground;
    Numbering numbering; NearRepr near;
    Accounting accounting;      // Measured (razor) | PerPairFigure{136,168} (sin, Python's)
    Sampling sampling;          // stage 4, off until then
};

struct Rep { uint8_t block; int32_t obs, src; uint8_t code, single; };

struct FamilyPlan {
    FamilySpec spec;
    std::vector<int32_t> map[2];          // class << code_bits | code, per block
    std::vector<Rep> reps;                // one per class
    std::vector<int64_t> near_flat;       // NearRepr::SentinelList
    std::vector<uint64_t> near_bits;      // NearRepr::OwnClass (the report)
    int64_t pairs, classes, near, rep_evals;
    int64_t cls_block[2], near_block[2], evals_block[2];
    size_t nbytes() const;
};

struct Plan {
    std::string fingerprint, declined;    // declined: the one reason sentence
    std::vector<FamilyPlan> fam;
    double census_seconds; int64_t budget_bytes, census_peak_bytes;
    size_t nbytes() const;
};

// THE CENSUS: razor's census<W, KeyFn> (`_razor_pair_reuse_core.cpp:254`),
// lifted as is. Instantiated in the ADAPTER's TU, so a key's float
// arithmetic stays where it is compiled today (§7).
//   key(block, o, s, uint64_t *w, uint8_t &code) -> Key::{Shared, Own}
template <int W, class KeyFn>
Outcome census(const FamilySpec &, const Policy &, int64_t room, KeyFn key,
               FamilyPlan &out, int64_t &peak);

// Optional renumbering after the census (sin: stable sort of classes by
// the rep's observer type; `_sin_pair_reuse_core.h:15`).
void renumber(FamilyPlan &, const int64_t *rank_of_class);

// Evaluation schedule: rows x union of their reps' sources (razor's
// build_groups, `.cpp:159`), REP_ROWS from the adapter.
struct Group { uint8_t block; std::vector<int32_t> obs, src, cls, at; };
std::vector<Group> groups(const FamilyPlan &, int rep_rows);

// One cache for every basis: fingerprint -> plan, LRU, one SHARED budget.
std::shared_ptr<const Plan> plan_for(const std::string &fingerprint,
                                     const std::function<std::shared_ptr<Plan>()> &build,
                                     bool *built = nullptr);

// THE COPY. `Reorient` is the adapter's transform, inlined:
//   void operator()(int rel_code, int64_t src, const cd *in, cd *out) const;
// razor: M1 -> h M0 - M1; sin: CODE_SIGNS; bspline: B_i J B_j^T or identity.
template <int NV, class Reorient>
void gather(const FamilyPlan &, int block, const int64_t *rows, ptrdiff_t n,
            const cd *values /* (classes, NV) */, Reorient, cd *out);

// THE REPORT: one shape for `_fill_report.py`.
struct FamilyReport { const char *name; int64_t pairs, computed, reused, near, kernel_evals; };
std::vector<FamilyReport> report(const Plan &);
void wire_map(const Plan &, const int32_t *wire_of_row, const int32_t *wire_of_col,
              int64_t n_wires, int64_t *out);   // (F, W, W, 3)

struct Counters { uint64_t built, found, declined; };   // one set, per basis tag
Counters counters(const char *basis); void clear_cache();
}  // namespace mw::pair_reuse
```

### What each basis adapter implements

| | razor-2p | sinusoidal | bspline |
|---|---|---|---|
| `families()` | direct (+ image block), remainder | direct, image or reflected | direct (off-edge segment pairs), later remainder and image |
| `key` | `direct_key`, `q_key` as today (`.cpp:475, 540`) | `DirectKey`, `GroundKey` (`_sin_pair_reuse_core.cpp:188, 223`) | new: a two-segment invariant (§2.1), h_i, h_j and a_row exact by type, EK eligibility |
| near rule | max(\|v\|, rho) < 4·max(h, a) | max(\|z\|, rho) < 4·max(2H, a) | same-edge pairs always, plus touching / 4·max(h, a) |
| `evaluate` | `seg_moments` over groups (`.cpp:742`); remainder `field_proj_batch` | canonical observer call (`evaluate`), `evaluate_near`, `evaluate_classes` | `offedge(si.take(rows), sj.take(src))` (`_bspline_fill_core.cpp:153-190`) per group |
| `Reorient` | flip bit / 2-bit remainder code | `CODE_SIGNS` | identity (code_bits 0) first; binomial later |
| consumer | `gather` into the window M0/M1 | `reduce_band` (fused into the reduction; its own TU stays, `setup.py:586-591`) | `gather` into the J window; the chunked route reads class values instead of held blocks |
| route refusals | `_pair_reuse_refusal` | `_sin_pair_reuse_route` | `_decline_unread_hint` reasons |

### How Python calls it

Use the existing accelerator pattern: per-basis bindings in the files that
bind those bases today, plus one shared binding file.

- `_accel_razor_solve.cpp` keeps `razor_reuse_plan` and friends (`:1836-1869`),
  re-pointed at the core.
- `_accel_sin_pair_reuse.cpp` gains `sin_reuse_plan`, which **the Python
  route calls** in place of `_sin_pair_reuse.build_plan`/`build_ground_plan`.
  Its C++ twin already exists and is gated equal to Python
  (`tests/test_sin_pair_reuse_cxx_1911.py:247`, `test_the_plan_is_pythons`).
- A new `_accel_pair_reuse.cpp` binds the basis-neutral pieces:
  - a `PlanHandle` (py::class_ over `shared_ptr<const Plan>`) with
    `report()`, `wire_map()` and `nbytes`;
  - the test hooks (budget, small-deck thresholds, class limit, probe);
  - `counters(basis)`.

  `_fill_report.py` then reads one shape from three solvers, and
  `_accel_fill_report.cpp` and `_razor_fill_map_core.cpp` retire into the
  core's `wire_map`.

So Python and the executables run one planner, as razor does today. The numpy
census in `_sin_pair_reuse.py` (`build_plan`, `build_ground_plan`, the band
keys, `_Fold`, the pool: most of its 1,629 lines) becomes the reference only in
tests, or is deleted.

---

## 5. Bit identity, per basis

"Byte-identical" here means flag-on Z and printouts, same machine, base
against change. Flag-off bytes are held by `tests/test_stable_path_tripwire.py`
throughout and cannot move: the core runs only when reuse is asked
(`_fill_reuse_asked`, `bspline.py:2297`; #2016's "no census work runs" gate,
`tests/test_fill_report_2016.py:206`).

| basis | onto the core byte-identical? | why / condition |
|---|---|---|
| razor-2p | **yes** on Linux and macOS | The census, keys and gather are the same code moved. The key lambdas stay in razor's TU, and `-ffp-contract=off` (`setup.py:243, 287`) means an edit elsewhere cannot fuse their a·b+c. Numbering: keep `PartOffsets` in stage 1. A later switch to `FirstAppearance` changes class indices only. The groups' calls are numbering-independent (the union is sorted, `.cpp:186`), so values should not move, and the stage's byte gate must show it. |
| sinusoidal | **yes**, if the numbering, the evaluation call shapes and the key's float order are kept | The C++ census already reproduces Python's plan to the byte (`test_the_plan_is_pythons`). The census's hash table can be the core's, because exact equality gives the same classes. Numbering must be `FirstAppearance` + the observer-type renumber, because EVAL_CHUNK slicing depends on class order and sincos position moves ulps (`.py:172-175`). Python's budget accounting must be kept as a `PerPairFigure` policy, or which decks serve changes. That is not served bytes, but it is a different answer for a deck at the boundary. |
| bspline vs today's hint route | **no** | The plan changes granularity: segment pairs, not edge blocks. The representatives are chosen by first appearance, not by AK's hash order. Quantised keys replace an exact distance check. |
| bspline, given an identical plan | yes | A stage-3 sub-gate: feed the core a plan equal to the hint's (`ExternalClasses`) and compare bytes. That proves the copy and consumer wiring before the census changes the plan. |

**Allowed tolerance for bspline under the repo's rules.**

- The registry's `drift` is |ΔZin|/|Zin| against the flag-off fill, measured
  and declared with margin (`_features.py:283-299`).
- `offset_reuse` declares 1e-12 and the whole `pair_reuse` 5e-10, both far
  under DEFAULT_DRIFT_MAX 1e-6 (antennaknobs' LAST_DIGITS_DRIFT).
- With no flips (code_bits 0) and KEY_BITS 40, the reference point is razor's
  measured 1.3e-13 to 4.3e-10 at the same key resolution (`_features.py:719-735`).
  Whether bspline lands there must be measured on #1971's deck set plus the
  #1854 ELT whip.
- If it exceeds 5e-10, the whole's `drift` is raised in its own registry PR
  with the measurements. Its kept bucket ("(faster)": seedable under 1e-6) is
  unaffected, so this is not a default-behaviour change. Moving
  LAST_DIGITS_DRIFT is Steve's call (AK CLAUDE.md).
- Flips (the binomial transform) add arithmetic per copy. They come later,
  with their own measured drift.

---

## 6. Staged plan

Each stage is its own PR, run with every gate in §7.

### Stage 1: extract the core under razor-2p
- **Files:**
  - new `_pair_reuse_core.{h,cpp}` (Table, census template, build_groups,
    budget and decline, cache, counters, `wire_map`, report), lifted from
    `_razor_pair_reuse_core.cpp:27-420, 607-740` and `_razor_fill_map_core.cpp`;
  - `_razor_pair_reuse_core.{h,cpp}` shrinks to razor's adapter (keys, eval,
    Reorient, policy constants);
  - `_accel_razor_solve.cpp` and `_nec5_razor_seam_core.cpp` re-pointed;
  - `setup.py`, `standalone/CMakeLists.txt`.
- **Gate:**
  - flag-on bytes identical, base against change, Python route and the
    nec5-razor-2p exe, on #1971's decks (fd_h27, fd_h51, gfd_h51, sky_m24,
    p_m24, gsky_m24, sfd/gsfd ladders), the hairpin, the close-spaced lines,
    and 3000-segment random wires (the decline path);
  - `tests/test_razor_pair_reuse_1971.py` (its probes, `verify`, cxx==python),
    `test_fill_report_2016.py`, `test_standalone_nopython.py:1018-1100`;
  - reach is 8, so levels 6-8's jobs: `standalone.yml:windows`,
    `windows-nopython`, `windows-single`.
- **Size:** M. About 700 lines moved, about 150 new, net negative.
- **Blocks:** nothing. Choose `PartOffsets` numbering here and leave the
  switch to stage 2b.

### Stage 2: sinusoidal onto the core
- **2a: the Python route calls the C++ plan.**
  - Files: `sinusoidal.py:6934-7360`, `_sin_pair_reuse.py` (the census goes;
    the reduction and `verify` stay), `_accel_sin_pair_reuse.cpp`.
  - Gate: byte-identical on the #1916 / #1969 / #1970 decks, the ELT whip, the
    LogYag, and the 3220-segment tapered Yagi (the decline,
    `tests/test_sin_pair_reuse_budget_2019.py`), at several worker counts. All
    15 pair_reuse test files.
  - Reach 4 means the drop-in jobs: `eznec-dropin.yml:freeze` and
    `resident-smoke` (level 4), and `wheels.yml` Windows cells (level 3).
  - Size: M. The census now runs under OpenMP, not the Python pool. That
    should address #2047 item 2 without changing a served value: classes are
    numbered by first appearance in flat order whatever the bands
    (`_sin_pair_reuse.py`'s `_Fold` note, `0c8bc751`), and
    `test_the_plan_is_pythons` is parametrised over workers. The budget's
    decline boundary does depend on the worker count by design
    (bands in flight), so keep the count the same on both sides of the gate.
- **2b: the sinusoidal C++ census on the core's census.**
  - The core gains `FirstAppearance` + `renumber` + `NearRepr::SentinelList`
    + `PerPairFigure` accounting. `_sin_pair_reuse_core.cpp` keeps its keys
    and evaluation.
  - Razor moves to `FirstAppearance` here as well, under its own byte gate.
  - Size: M-L.
- **2c (separate, NOT byte-preserving by design):**
  - give sinusoidal razor's `MAX_CLASS_FRACTION` and measured accounting;
  - unify the two SHARED_PLAN_BYTES caches into one 256 MB (#2026).

  This changes which decks serve. Gate: drift unchanged where served, and
  declined decks equal to flag-off bytes. It needs Steve's agreement on the
  thresholds.
- **Blocks:**
  - the sector route's `obs_rows` plans (SECTOR_ROUTE is False today,
    `.py:231`) must go through the C++ plan too;
  - the pure-Python `_PyGroups` fallback goes away, which is harmless, since
    reuse already declines with "no C++ field-tensor kernel".

### Stage 3: bspline onto the core (the executable port)
- **3a: the Python tensor route reads a core plan.**
  - New `_bspline_pair_reuse_core.cpp` (key, near rule, groups → `offedge`,
    identity Reorient). `_build_J_blocks` (`bspline.py:6297-6430`) gathers
    instead of `copy_blocks`.
  - First the `ExternalClasses` sub-gate: today's hint-derived plan, fed to
    the core, gives identical bytes. Then the core's own census, drift-gated
    (§5).
  - Size: L.
- **3b: the chunked and swept routes, and the remainder.**
  - `_ChunkedReuse` (#2020) becomes class values under the budget.
  - The fused remainder kernel's `reuse_src` map (`bspline.py:5897-5952`) is
    derived from the core's remainder family, or replaced by a core gather on
    Jf.
  - Image J reuse (never done today) is a separate opt-in.
  - Size: L.
- **3c: the exe.**
  - `_bspline_fill_core.cpp` (the nec5-bspline and nec42-bspline C++ route)
    calls the adapter.
  - `standalone/momwire_razor2p.cpp`: an `EXE_FEATURES`/certify row for
    bspline.
  - Gate: the exe equal to the Python route to the bit, the same shape as
    `test_the_cxx_front_route_is_the_python_route_to_the_bit`
    (`tests/test_razor_pair_reuse_1971.py:312`).
  - Size: L.
- **Blocks:**
  1. **offset_reuse has no C++.** Neither `_bspline_fill_core.cpp` nor the
     front core has it (no `REG_TOEPLITZ`; `is_toeplitz` there is the shipped
     static path). `pair_reuse` turns `offset_reuse` on for bspline
     (`_features.py:310-334`), so an exe carrying only the core's part would
     answer differently from Python under the same switch. Port it in 3c
     (M), or give the exe the part split explicitly.
  2. The AK hint consumer contract (§3).
  3. bspline's many fill routes (tensor, chunked, swept, exact-kernel,
     buried, batched sweep, sector). Each either reads the plan or declines by
     name, as `_decline_unread_hint` does today.
  4. The census cost against bspline's per-pair J cost has never been
     measured. It is expected to be favourable, because J is several times a
     point kernel, but that is not established.

### Stage 4: the sampling early exit (#2061), once, in the core
- **Files:** `_pair_reuse_core.{h,cpp}` (`Sampling`), plus per-adapter cost
  constants: c_pair_fill, c_key, c_class_eval, measured.
- **Shape:**
  - Before the census, key a fixed, deterministic sample: about 20,000 pairs
    as whole observer rows, evenly spaced. Count classes at two sample sizes
    to extrapolate the class fraction. A small sample over-counts classes,
    because repeats across rows are seen less, so a raw fraction would wrongly
    decline.
  - Predict the saving: pairs × c_fill − census × c_key − classes × c_eval.
  - Decline by name below the threshold.
- **Gate (#2061's own):**
  - byte-identical wherever reuse still serves (the decision is the only thing
    that changes);
  - a decline equals flag-off bytes;
  - the whip still serves (19.3 M pairs, about 1 in 21 computed) and the
    LogYag declines, on razor and sinusoidal (and bspline once it exists).
- **Size:** S-M.
- **Blocks:** the measurement #2061 asks first: is #2047's loss the census, or
  thread width?

### Stage 5: certify reach for the executables
- One PR per raise in `_features.py`:
  - sinusoidal on nec42-sinusoidal (level 10; today certification build only,
    `momwire_razor2p.cpp:498-508`);
  - bspline on nec5-bspline (levels 6-8; remove `EXE_FEATURE_BASES`'
    razor-only refusal, `:424-427`).
- Gate: exe parity with Python; the exe never slower than Python on the
  Windows box; the slow lane run on base and change (`momwire CLAUDE.md:37`);
  the level's jobs.
- Size: S each, plus measurement time.

---

## 7. Risks

**Codegen drift.** #1187 moved Z through inlining with no source change to the
affected functions. #1209 (memory: slow-lane-gate) and #1194 show the same.

- Mitigations already in the tree:
  - every reuse file is its own TU;
  - Linux and macOS build with `-ffp-contract=off`;
  - the core's shared half is integer-only (hashing, numbering, maps, budget
    arithmetic), as `_int_groups_core.h` and `_razor_fill_map_core.cpp` are
    today.
- The census template is instantiated in each adapter's TU, so the float keys
  are compiled beside the code they are compiled with now.
- **The residual risk is a moved key bit.** Under libmvec/loop vectorisation,
  which `-ffp-contract=off` does not cover (`setup.py:279-282`), a moved key
  bit would move a pair across a quantum into a neighbouring class. Its Z then
  moves by about 1e-13 with no failing test, because the drift gates have
  margin.
- **So every stage's merge gate is same-machine base against change, flag ON,
  byte comparison of Z and printouts.** The slow lane runs on base and on
  change for each C++ stage (numeric C++ rule).
- **Instrument which branch ran.** Assert `counters(...).built - declined ≥ 1`
  on both sides, per the "a gate can be green and measure nothing" rule
  (momwire CLAUDE.md). Identical bytes from two runs that both declined prove
  nothing.

**Windows.** MSVC builds `/fp:fast`, which reassociates TU-wide (`setup.py:191-198`).

- Moving code between TUs can move Windows bits in the keys themselves.
- The repo accepts no bit-compat on Windows (a derived win32 tolerance; memory
  windows-no-bit-compat). So the Windows gate is drift, not bytes.
- The drop-in (level 4, a frozen PyInstaller bundle of the release wheel) and
  the exes (levels 6-8) see a change only at the next release.
- The pre-tag sweep is the last net. Stage 2a touches a reach-4 certified use
  (`MOMWIRE_X_PAIR_REUSE=1` with the Windows NEC-2 drop-in,
  `_features.py:244-251`), so it runs the level-3 and level-4 jobs.

**Memory budgets.**

- The core must preserve each basis's accounting until stage 2c deliberately
  changes it. Otherwise "which decks decline" moves silently.
- Today a long-lived process can hold 256 MB of razor plans + 256 MB of
  sinusoidal plans (#2026 is still open for measurement). Unifying is a
  reduction.
- bspline brings the largest per-class values ((d+1)² × 16 B = 144 B at d=2).
  It also has two existing budgets: the tensor J itself, and `swept_mem_mb`'s
  0.75 reuse share for the chunked route. The core's budget must count class
  values in place of held blocks, not on top of them.
- #1988's census-table shrink (compact keys, load factor) lands once in the
  core and helps razor and sinusoidal alike. It is not byte-affecting: the
  table layout is not in any value.
- Do the reuse runs on the Skylake box. This laptop holds 16 GB with another
  job running (memory: OOM kills the session), and one census can reach
  1-2 GB unbounded (`.h:170-176`).

**The frozen executables.**

- `EXE_FEATURES` (`momwire_razor2p.cpp:409`) and `EXE_FEATURE_BASES` gate
  what each exe runs.
- Stage 1 must not change nec5-razor-2p's reach-8 behaviour.
- Stage 3c adds the bspline route behind a certification build before any
  reach row, as sinusoidal did (`MW_EXE_CERTIFY`, `CMakeLists.txt:571`).

**Contract surface AK reads.**

- `pair_reuse_declined` strings, `pair_reuse_census`, `fill_report()`'s shape
  and the hint kwarg. AK's checkbox and seed derive from the registry
  (AK#2233).
- Keep the reason sentences verbatim through every stage. A changed sentence
  is a change AK displays.
