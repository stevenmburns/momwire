# momwire#1029 phase 2 — the crossing family: columns, memory, and where the 8.4 GB is

Registered 2026-09-17 on momwire main 5bfc731 (branch `fix/1109-crossing-columns`),
BEFORE any source edit. Phase 1 (`PLAN-phase1.md`, PR #1106) bought time and not
memory, and said so (G3c, Amendments 2 and 5). This phase is the memory, and the
one term of the route's cost that still scales with N.

## 0. What the memory actually is — measured first (`p2_memory_probe.py`)

Amendment 5 and momwire#1109 name `_crossing_fill._row_weights` — a 124 MiB
`(2035, 7992)` float64 — as the allocation that raised under the laptop's 8 GB
cap. It is the allocation that TIPPED, not the floor. Attributed phase by phase
on the Skylake box (46 GB, 40 GB cap, one cold solve, `tracemalloc` peak reset at
every phase entry, `p2mem_150_{route,dense}_skylake.jsonl`), 150 radials, 8285
unknowns, route mode:

| phase | adds at its peak | s (traced) | note |
|---|---:|---:|---|
| the four direct/image fills into Z | 1.1 GB | 0.5 | Z itself, (8285)² complex |
| `_field_galerkin_block` (each of two) | 1.05 GB | 0.8 | a full-size `Q` returned and subtracted |
| `axis_data`, the below axis | **4.13 GB** | 1.3 | `F` and `Fd`, (8285 × 31 k) float64 **dense**, ≤ 3 nonzeros per column |
| `_main_split` | 2.60 GB | 12.9 | `t_main` 1.1 GB + `_row_weights` on the far-direct blocks: 528 MB × 4, 292 calls, 7.3 GB of churn |
| `_ends_and_corner` | 1.06 GB | 1.5 | its own full-size `t_ab` |
| `self_completions` | 1.99 GB | 2.8 | full-size `total` + the `total[live, :] +=` temporaries |
| **peak** | **8.55 GB** | 22.6 | dense mode: 8.51 GB, 134 s (89.8 s of it the below/below remainder) |

Three readings that decide the design:

1. **The floor is `axis_data`'s dense basis samples**, 4.1 GB live from the
   moment the below axis is built until the crossing fill returns. A B-spline
   basis has local support, so `F[:, node]` has at most `degree + 1 = 3`
   nonzeros; the exact structure is already on the axis (`seg_rows`,
   `seg_runs`) and `_fdw_sparse` (momwire#914) already builds `Fd·w` as a CSR
   from it. The dense arrays exist only to be gathered from.
2. **The crossing family is filled in full by contract** (Amendment 1), so the
   route's `rows=` narrows nothing there: `t_ab`, the ends' block and
   `self_completions` are each (n, n), and the route pays the full 17 traced
   seconds that the dense path pays. It is the only term left in the route that
   scales with N.
3. **The block tree is fine.** At 48 radials the partition is 2 near blocks
   (2 × 3 and 2 × 2 segments) and 144 far ones, and the biggest far blocks
   (7 mast segments × 708 radial segments) fall UNDER `_ACA_COST_GUARD`, so they
   are evaluated direct — correctly: sampling a (56 × 2840)-node block costs
   more than its product. What makes them expensive is not the kernel table
   (56 × 8240 entries at 150) but `_row_weights` materialising the DENSE
   `(2098, 8240)` weight matrices, four of them, for a product whose Q side
   has three nonzeros per column.

## 1. The units, in order

**A. Sparse basis samples on the axis** (memory, both paths). `axis_data`
carries `F_csr` / `Fd_csr` (scipy `csr_array`, (n_basis, n_nodes)), built from
the segment structure by `_basis_samples` without ever forming the dense pair;
the keys `F` / `Fd` go away so every consumer is explicit (a `csr_matrix` would
have turned `ax["F"] * w` into a silent matmul). The sampler protocol admits a
sparse return; the sinusoidal sampler keeps returning dense and `axis_data`
converts. Consumers:

- `_row_weights(ax, ii, rows)` returns the four weight matrices as CSR
  sub-blocks (rows × ii) — no dense (2098 × 8240) anywhere;
- `_sandwich_dense` forms `P @ K @ Q.T` as sparse @ dense @ sparse.T, cost
  O(nnz(P)·|iB| + nnz(Q)·|rA|) instead of O(|rA|·|iA|·|iB| + |rA|·|iB|·|rB|);
- the far-ACA `_lr` products restrict to `_support_rows` on both sides and
  scatter (the same exact restriction #688 G6 licensed for `_sandwich_dense`)
  instead of building (n_basis × |iA|) weights;
- `_ends_and_corner` / `_reversed`: `_real_matvec_c` takes the CSR;
- `_main_sandwich` (dense, razor and `MOMWIRE_CROSSING_FORCE_DENSE`) densifies
  on the spot — that path's decks are small and it keeps today's bytes;
- `_fdw_sparse` builds from `Fd_csr` directly; `_support_rows`' fallback scan
  reads the CSR's row pattern.

**B. `rows=` reaches the crossing family** (time and memory, the route).
`cross_complete_block_split(..., rows=R)`, `_ends_and_corner(..., rows=R)` and
`self_completions(..., rows=R)` take a BASIS-row subset `R` and return the pair
`(t[R, :], t[:, R])` — shapes (|R|, n) and (n, |R|) — from ONE evaluation of
the kernel tables: in `_sandwich_dense` both `P[R∩rA] @ K @ Q[rB].T` and
`P[rA] @ (K @ Q[R∩rB].T)` read the same `K`; in the far-ACA branch both read
the same `(Uf, Vf)`. `_below_interface.compute_Z_operator_buried` derives `R`
from its segment `rows` through `supp_seg` (all-or-nothing per basis — asserted,
since no basis straddles two wires, S-0's S0a) and composes
`Z[R, :] -= t_r; Z[R, :] -= t_c.T; Z[R, :] += s_r`. **The contract changes:**
with `rows` given, rows outside the request are now EXACTLY zero (Amendment 1's
"except the crossing block" clause is gone), and S-B's item 2 reads "the
non-requested rows are zero" instead of "equal a `rows=[]` control". `rows=None`
takes today's path.

**C. The full-size transients go** (memory, both paths; bit-identical).
`_field_galerkin_block` accumulates into the caller's `Z` with a `scale`
(`out=` and `scale=-1`, the #914 `_sandwich_dense(out=)` pattern — the entries it
skips were exact zeros, so the bytes cannot move); `_ends_and_corner`
accumulates into `_main_split`'s `t_main` instead of allocating its own block;
`self_completions` writes its three scattered terms straight into `Z` under the
route's row restriction and, on the default path, still returns (n, n).

**D. The route's algebra moves out of `bspline.py`** (factoring; behaviour-
preserving; its own commit, last). `_rotational_rows`, `_rotational_dof_groups`,
`_rotational_K0`, `_rotational_solve` and `_compute_impedance_rotational` become
functions in `_rotational_symmetry.py` that take the solver; the four
`BSplineSolver` methods stay as one-line delegations so every phase-1 test and
scratch harness keeps its call. Phase 2 grows exactly this code (the general
drive is phase 3's), which is the reason to give it its own module now rather
than after.

**Not this phase, filed instead:** the compact `(|R|, n)` Z for the route (the
four narrowed fills and the chunked accumulator all write into a full Z by
observer index; after units A–C the full Z is 1.1 GB at 150 radials and fits the
laptop, so the remaining 1.1 GB is a later lever); the unburied deck (the
free-space/grounded operator has no `rows=`, which is the only reason the route
refuses it); the general drive (all N harmonics; off-axis or per-sector ports).

## 2. What bit-identity phase 2 can and cannot promise

Phase 1's S-A pinned `rows=None` bit-identical to main. Unit A cannot: a sparse
product sums the same nonzero terms as the dense GEMM in a different order, and
the matvecs in the ends term reassociate the same way. momwire#914
(`_bnd_and_corner`) and #919 (the weight fold) made exactly this kind of change
to exactly this fill and gated it "to scale, never to the bit", at 1e-12
relative. `_sandwich_dense`'s own docstring already declines to pin bits
("BLAS chooses kernels by shape and by build"). So:

- **units C and D are bit-identical** on the default path and gated as such;
- **unit A is gated at 1e-12 relative** on Z, the coefficients and Z_in
  (P2-7), with the sha recorded beside it so a bit-identical result is seen
  as one; and the crossing suites and the `crossgate` lane are the gate on
  every deck class the fill serves.

## 3. Gates, registered before the build

Bars are on the boxes named; predictions are what the design implies and are
recorded so a miss is a finding. Phase 1's Skylake ladder (`g3_skylake.jsonl`,
Amendment 6) is the reference row: route 18.148 s / 8394 MB, dense 130.933 s /
8424 MB at 150 radials; route 2.627 s / 1044 MB at 48.

| gate | what | bar | prediction |
|---|---|---|---|
| **P2-1** | route peak RSS at 150 radials, Skylake, warm pair as G3 | **≤ 2.5 GB** (from 8394 MB) | 1.3–2.0 GB: Z 1.1 GB + the narrowed fills' windows |
| **P2-2** | dense peak RSS at 150, Skylake | **≤ 4.0 GB** (from 8424 MB) | 2.0–3.2 GB |
| **P2-3** | route warm seconds at 150 / at 48, Skylake | **≤ 18.15 s / ≤ 2.63 s** (no regression) | 8–12 s / 1.4–2.2 s (the crossing family's ~8.7 clean seconds mostly go; the tables stay) |
| **P2-4** | dense warm seconds at 150, Skylake | **≤ 131 s** (no regression) | 118–128 s |
| **P2-5** | the 150-radial rung on the LAPTOP under `prlimit --as=8G`, both modes | **runs** (phase 1: REFUSED on both) | route ≈ 25 s, dense ≈ 160 s at the laptop's 1.2× |
| **P2-6** | route = dense, `g1_route.py` at 4 / 12 / 48 | G1a ≤ 1e-9, G1b ≤ 1e-8, G1c ≤ 1e-8 (phase 1's) | ≤ 1e-12, as phase 1 measured |
| **P2-7** | default path vs main, `p2_default_path.py` on the 4- and 12-radial decks (`p2_pre_xps13.npz`, sha_Z `4a15d55c…` / `a28d9ef6…`, the 0.58.0 bytes) | **≤ 1e-12** relative on Z, coefficients, Z_in | ~1e-15, the reassociation floor; units C and D alone: bit-identical |
| **P2-8** | the `rows=` contract, `s_ab_gates.py --sb` rewritten | non-requested rows exactly 0; requested rows ≤ 1e-12 of the full fill; plan fields equal | passes by construction |
| **P2-9** | `make lint`; default lane; the crossing suites (`test_crossing_*`, `test_razor_crossing_*`, `test_sg_crossing_*`, `test_tilted_crossing_936`, `test_rotational_symmetry_1029`); `make crossgate` on Skylake | green | green |

**Stop rule.** If P2-7 exceeds 1e-12 anywhere, stop and report the deck and the
entry — do not loosen the bar and do not tune the summation order to chase it.
If P2-3 regresses (the route slower than 18.15 s at 150), stop and report before
adding any further lever.

## 4. Order

1. Commit this registration with the probe and its three records; push.
2. Unit A, then B, then C, each with its tests, each gated on the laptop by
   P2-6 (4 / 12), P2-7 and the crossing suites before the next starts. Unit D
   last, alone.
3. P2-1 … P2-5 and `make crossgate` on the Skylake box (rsync the worktree,
   `make build` there, run under ssh, bring the records back); P2-5 on the
   laptop.
4. STATUS.md rewritten for phase 2; PR to momwire. Steve merges. Nothing on
   the issue until then.

## Amendment 1 (2026-09-18, after the build, before the Skylake ladder was read): unit C landed half

`_field_galerkin_block` did NOT get `out=Z, scale=-1`. Two reasons, both
measured by the builder and both pinned by tests (`test_crossing_sparse_1109.py`,
`test_p2c_5` and `test_p2c_6`):

1. `_acc.assemble_field_galerkin` declares its target `py::array_t<...,
   c_style>` without `forcecast`, so a column-major array is handed to the C++
   as a C-contiguous COPY and every accumulation is lost — the call returns
   cleanly and writes nothing (12 nonzeros into a C-ordered target, 0 into an
   F-ordered one, same call). The buried Z is column-major by momwire#136's
   decision, so `out=Z` would have silently zeroed the below/below remainder
   and both transmitted blocks. That is a latent hazard for any caller of the
   binding today and is filed as its own follow-up.
2. Even on a C-contiguous target the observer loop chunks, and a basis row
   whose support straddles a chunk boundary accumulates across chunks —
   `Z − (c₁ + c₂)` becomes `(Z − c₁) − c₂`. At 150 radials `chunk` is 1, so
   every row straddles; a fix is a reassociation and needs the 1e-12 gate, not
   the bit gate unit C was registered under.

**Consequence, registered before reading the ladder:** ONE `(n, n)` transient
stays on BOTH paths, the field block's `Q` (1.05 GB at 150 radials). P2-1's
prediction band (1.3–2.0 GB) was written without it and should read
2.3–3.0 GB; the BAR (2.5 GB) stands. P2-2's band moves the same way; its bar
stands. The rest of unit C landed as registered: the ends and the self
completions accumulate their support in place, bit-identically.

## Amendment 2 (2026-09-18, momwire#1115 parts 1 and 2)

Reason 1 above is no longer true of the code: the binding takes its target as
a bare `py::array` and REFUSES one it cannot accumulate into (not
C-contiguous, not complex128, not writeable), and it takes a `scale` applied
to each contribution, so a caller can accumulate `Z -= Q` in one pass.
`test_p2c_5` pins the refusal instead of the silent copy.

Reason 2 stands unchanged, and is now the whole of what blocks
`_field_galerkin_block(out=Z, scale=-1)`: the reassociation across chunks
needs the 1e-12 gate `p2_default_path.py` provides and a 150-radial memory
measurement on the fleet. Follow-up 1 in STATUS.md is that remainder.
