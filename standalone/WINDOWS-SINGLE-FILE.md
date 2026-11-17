# Single-file Windows executables (MinGW-w64): what shipping them takes

Status: experiment on branch `windows-static-exe-v2` (no PR). It restarts the
parked `windows-static-exe` (tip d35e80c1). It was built on `fat-msvc-exe`
(PR #1975), which has since landed on main, and main is merged in (e57f724c).
Workflow: `.github/workflows/windows-static.yml`. It runs on pushes to this
branch and on dispatch. The 4 x 70 min exit-hang soak runs on dispatch only.
Measured on GitHub `windows-latest` runners, 2026-11-17:

| run | commit | what it measured |
|---|---|---|
| [44057069874](https://github.com/stevenmburns/momwire/actions/runs/44057069874) | 80ae9fba | MINGW64 full gates, libm shims, the libm microbench, the first stress A/B (stdout to DEVNULL) |
| [44081988222](https://github.com/stevenmburns/momwire/actions/runs/44081988222) | 734612ca | UCRT64 avx2 + sse2 + fat: SDE, fat == one-variant, two hang backtraces |
| [44092464961](https://github.com/stevenmburns/momwire/actions/runs/44092464961) | b554dae3 | UCRT64 contract tests and corpus with the `_cexit` exit, a three-exit stress run, one more backtrace |
| [44112554494](https://github.com/stevenmburns/momwire/actions/runs/44112554494) | 46caa047 | **the numbers this report rests on:** MSVC built from the same commit, `/fp:precise`, MSVC's assembly listings, gcc FMA / fast-math / column-kernel experiments, the exit-hang soak with its control (4 shards), the gates with the `-version` toolchain tag |
| [44132258736](https://github.com/stevenmburns/momwire/actions/runs/44132258736) | 7d4d7024 | the exit fix (068af7d0): its soak against the positive control, the gates and the timing again (another runner, an EPYC 7763: the same ratios within a few per cent) |

Runs 1-3 timed MinGW against the MSVC zip of the latest green
`standalone.yml` run on main (101fcabb in run 1, 005fc5dc in runs 2-3).
Neither commit is an ancestor of the branch's code at the time. 005fc5dc
carries #1956's perf commits, which touch the razor crossing core, the razor
solve and the bs2 route (c211623b changes buried bs2's thread policy). So
those comparisons shared a runner but not a source tree. **Every timing in
this report is from run 4 or later, where the `msvc` job builds the MSVC
reference from the same commit as the MinGW builds.** The run 1-3 figures
in the earlier version of this report are superseded.

Every timing is a median of runs, with the builds interleaved on one runner
(run 4: a 2-core / 4-thread AMD EPYC 9V74 VM). It is indicative: the point
is the pairing. "1 thread" is MOMWIRE_THREADS=1; "2 threads" is the size
policy's count on that VM.

## Bottom line

- **The scalar libm explained the original 3-8x gap.** mingw-w64's own
  `exp`/`sin`/`cos`/`log`/`pow`/`atan2` (libmingwex's, which win gcc's link)
  take 39-81 ns a call; ucrtbase.dll's take 5-17 ns. **MSYS2 UCRT64 binds
  ucrtbase's, with no code change.**
- **What is left is MSVC's vector math, now measured, not inferred.**
  - MSVC's own fill listings call `__vdecl_sin4`, `__vdecl_cos4` and
    `__vdecl_exp4` (42 / 38 / 11 call sites in `_bspline_core`, 4 in
    `_razor_core`).
  - Built at `/fp:precise`, MSVC's razor-2p free-space fill is **400.6 ms,
    the same as UCRT64's 401.1 ms**, against 180.5 ms at the shipped
    `/fp:fast`.
  - On Linux, gcc with libmvec switched off loses a similar factor
    (1.85-2.33x): razor-2p free-space fill 222 -> 517 ms, bs2 buried
    447 -> 917 ms.
  - FMA contraction (`-ffp-contract=fast`) buys 0-8 %. `-ffast-math` broke
    the buried route and bought at most 4 % in free space.
- **UCRT64 against MSVC, total wall, one thread:** razor-2p free space 1.73x
  slower; bs2 free space 0.94x (faster); razor-2p buried 0.88x (faster); bs2
  buried 1.32x slower. The OpenBLAS factor is 8-44 % faster under MinGW.
- **The exit hang is diagnosed, and b554dae3 did not fix it.**
  - The control run the earlier version lacked shows hangs in every exit
    arm, b554dae3's `_cexit` exit included (1 in 44,736).
  - The cause is OpenBLAS's Win32 thread server. At exit it waits 50 ms per
    pool thread, then `TerminateThread()`s it, which can orphan the loader
    lock or the heap lock.
  - The fix (068af7d0) ends the process before OpenBLAS's exit-time
    shutdown runs. Soak: 0 hangs in 89,472 runs, against 48 in 44,736 for
    the positive control, interleaved on one runner.
- **Recommendation: ship-beside, as a preview,** a UCRT64 fat single-file
  zip next to the MSVC zip, which stays the default. Closing the last gap
  is a libmvec-named shim over the in-repo vector math (`_vmath_inline.h`,
  momwire#1441), costed at about 1 day below. It is not built, as briefed.

## 1. What works today (gates)

Every row is measured, not inferred. UCRT64 is the candidate product build
and is gated on every run. MINGW64 was gated in run 1 (`MINGW64_GATES=1`
runs it again).

| gate | MINGW64 (msvcrt.dll) | UCRT64 (ucrtbase via api-ms-win-crt-*) |
|---|---|---|
| import table names system DLLs only | pass: `kernel32.dll msvcrt.dll`, nothing else (run 1) | pass: `kernel32.dll` + 13 `api-ms-win-crt-*` sets (runs 1-4) |
| Windows 7 import scan (`scan_win7_imports.py`) | pass (runs 1-2) | pass. api-ms-win-crt-* counts as the UCRT prerequisite (KB2999226), as for the MSVC zip |
| rehearsal signature (`sign_windows.py`) | pass, 6 exes (4 one-variant + 2 fat) | pass, 6 exes |
| run from a System32-only PATH (deck + SimNEC stream) | pass, all builds | pass |
| `-version` names the toolchain (new, 9afe2612) | `...-standalone-avx2-mingw64 [sb] 0.93.0`, all 6 exes (run 4) | `...-standalone-avx2-ucrt64 [sb] 0.93.0`, all 6 exes, and the fat pair under SDE names the variant it picked (runs 4-5) |
| launcher contract tests (`test_standalone_nopython.py`) | pass: razor-2p 49 + 4 skipped, bs2 52 + 1 skipped, fat likewise (run 1) | pass, one-variant and fat, with `MW_EXE_EXPECT_TOOLCHAIN=ucrt64`: razor-2p 50 + 4 skipped, bs2 53 + 1 skipped (runs 4-5; main's merge added one test each). Run 5 is the 068af7d0 exit: the test that reads the exit-time atexit report still passes |
| corpus vs the launcher, `--rtol win32` | 0 mismatches, 783 razor-2p / 786 bs2 decks, avx2 and sse2 (run 1) | 0 mismatches, 783 razor-2p / 786 bs2 decks, avx2 and sse2; worst ratio 5.66e-15 / 4.54e-16 (razor-2p), 4.71e-16 / 5.54e-15 (bs2) (runs 3-5) |
| fat exe under Intel SDE (Westmere, Sandy Bridge -> sse2; Haswell -> avx2), printouts == one-variant | pass, whole-process checking (run 1) | pass, whole-process checking: ucrtbase's own dispatch did not fault (runs 2-4). The MSVC fat build needs `-chip_check_exe_only` for VCRUNTIME140's memcpy |
| fat == one-variant byte for byte, every fixture deck x 2 copies x both paths | razor-2p avx2: 1402/1404 (both differences were exit hangs, not bytes); others 1404/1404 (run 1) | 1404/1404 in all four (runs 2-5) |
| the avx2 initializers run only on the avx2 path | 10 (razor-2p) and 12 (bs2) deferred `.ctors` entries; none on sse2 | same |
| no libquadmath / libgfortran linked (licensing) | 0 symbols in each exe | 0 |
| CI registry (`test_every_ci_job_is_placed_once_or_unassigned`) | the seven `windows-static.yml` jobs are placed as unassigned (60732c4b); it failed on this branch before | |

Unlike the MSVC fat port, the MinGW fat build does not rename its shared
template and inline bodies (`fat_private_comdats.py`). The avx2 copies may
therefore call an sse2-compiled `std::` body. Under gcc, fat == one-variant
still holds for every deck (1404/1404 x 4, UCRT64). The MSVC port saw three
decks differ before it added the renaming. The fat-vs-one-variant gate is
what holds the MinGW build to it.

## 2. Performance vs MSVC

### The libm, per call (`scripts/windows_static/libm_bench.c`)

ns per call, best of 5, run 4's perf runner (EPYC 9V74). The first table is
a MINGW64 build of the bench, the second a UCRT64 build; each also calls
msvcrt.dll's and ucrtbase.dll's own functions through `GetProcAddress`.

| function | MINGW64 link (libmingwex) | msvcrt.dll | ucrtbase.dll | UCRT64 link |
|---|---|---|---|---|
| exp | 38.7 | 8.8 | 4.7 | 4.7 |
| log | 24.6 | 8.4 | 4.5 | 5.7 |
| sin | 41.4 | 10.5 | 7.2 | 7.7 |
| cos | 42.2 | 10.0 | 7.3 | 7.6 |
| pow | 81.3 | 41.5 | 17.5 | 17.3 |
| atan2 | 43.8 | 15.2 | 12.4 | 12.3 |
| sincos | 43.6 | - | - | 15.5 |
| cexp | 109.4 | - | - | 27.2 |

Which symbols resolve where (nm on the exe, objdump on its imports):

- **MINGW64.** `exp log sin cos pow atan2 hypot sincos cexp __muldc3
  __divdc3` are libmingwex's own code in the exe. libmingwex is ahead of
  `-lmsvcrt` on gcc's link line, so msvcrt.dll's faster ones are never
  reached. msvcrt supplies only `_hypot asin atan ceil cosh log10 sinh tan`.
- **UCRT64 and CLANG64.** `exp log sin cos pow atan2 hypot` are imported
  from `api-ms-win-crt-math-l1-1-0.dll` (ucrtbase). The exe's `T` symbols
  for them are 8-byte import thunks. `sincos` and `cexp` stay libmingwex's,
  written over the imported `sin`/`cos`/`exp`. So no link-order trick or
  `__imp_` call is needed: the UCRT toolchain already takes ucrtbase's libm.
- **The values move by 0-2 ulp** between libmingwex and the DLLs, and
  ucrtbase matches UCRT64's link exactly (0 ulp). All stay within
  PRINTOUT_RTOL_WIN32 on the whole corpus.

### Optimisations tried

| # | what | result |
|---|---|---|
| 1 | MINGW64 with msvcrt's / ucrtbase's libm forwarded (`MW_EXE_LIBM_SHIM`, run 1) | closes most of the libm gap; an experiment device (`GetProcAddress`), superseded by 2 |
| 2 | **MSYS2 UCRT64** | **the candidate**: ucrtbase's libm with no code change |
| 2b | CLANG64 | libm only (ucrtbase too, ~1.2x UCRT64's per call). A CLANG64 exe would swap libgomp for LLVM's libomp and build OpenBLAS with clang: a second OpenMP runtime and OpenBLAS toolchain, for no libm gain. Not built |
| 3 | `-fno-math-errno` | already in every gcc build's `MW_CORE_FLAGS`; the gaps were measured with it on |
| 3b | `-ffp-contract=fast` (FMA contraction, as `/fp:fast` allows) | razor-2p free-space fill -7.5 %, bs2 free -2 %, bs2 buried -5 %, razor-2p buried +1 %. Small, and it gives up the IEEE contract the gcc branches keep. Not adopted |
| 3c | `-ffast-math -fno-finite-math-only` | **broken**: the buried decks ran with no factor stage (Linux and Windows alike), and free space gained 0.5-4 %. Not adopted, and not safe without the corpus gate |
| 3d | the near-interface column kernel on `mw_vmath` for MinGW avx2 (8819db99) | **no gain**: bs2 buried fill 683 ms against 635 without at one thread, 515 against 523 at two (Linux: 905 against 917). Those decks do not spend their fill there. Reverted (edf19053) |
| 4 | vector math for gcc on Windows | costed below; not built, as briefed |

### Where the gap is: MSVC's vector math, measured

Three measurements, all run 4 unless marked, fill stage in ms at one thread:

| deck | MSVC `/fp:fast` (shipped) | MSVC `/fp:precise` | UCRT64 | UCRT64 FMA | Linux gcc, libmvec on | Linux gcc, libmvec off |
|---|---|---|---|---|---|---|
| razor-2p fs_yagi 6x201 | **180.5** | 400.6 | 401.1 | 371.0 | 222.3 | 517.2 |
| bs2 fs_yagi 6x201 | 502.4 | 660.3 | **476.6** | 465.8 | 318.3 | 601.5 |
| razor-2p buried hub 32x20 | 368.4 | 431.4 | **325.4** | 329.1 | 251.3 | 466.1 |
| bs2 buried hub 32x20 | **503.3** | 736.4 | 682.6 | 648.9 | 447.1 | 917.3 |

(The Linux columns are Skylake, a different machine: compare within the
pair, not across.)

1. **MSVC at `/fp:precise` loses exactly the lead.** razor-2p free space
   becomes UCRT64's time to the millisecond, and bs2 buried goes past it.
   `/fp:fast` is what licenses MSVC's vectoriser to call its vector math.
2. **The listings show the calls.** The `msvc` job compiles the avx2 cores
   with `/FAs` and counts `call __vdecl_*` in them. razor-2p: 4 x
   `__vdecl_sin4` in `_razor_core`; `__vdecl_cos4` / `__vdecl_exp4` /
   `__vdecl_tan4` in the Sommerfeld cores. bs2: 42 x `__vdecl_sin4`, 38 x
   `__vdecl_cos4` and 11 x `__vdecl_exp4` in `_bspline_core`, plus the
   razor and Sommerfeld ones. (An exe's strings could not show this: the
   functions are linked statically and the exe is stripped.)
3. **gcc's equivalent is libmvec, and it is worth a similar factor.** The
   `#pragma omp declare simd` declarations in six files bind gcc's
   vectorised `sin`/`cos`/`exp` loops to glibc's `_ZGVdN4v_*` on Linux. A
   Linux build with them compiled out (the `_WIN32` guard's effect, tried
   on Skylake with a scratch define) has no `_ZGVdN4v_` import and a fill
   1.85-2.33x slower. MSVC's `/fp:precise`-to-`/fp:fast` ratio on the same
   decks is 1.17-2.22x: largest on razor-2p free space for both compilers,
   and smaller for MSVC elsewhere.
   Under gcc on Windows, sse2 and avx2 fills are therefore about equal
   (razor-2p free: 405.8 vs 401.1 ms). MSVC's sse2 fill is 316 ms against
   its avx2's 180 (its sse2 listings were not read).

FMA contraction is not it (3b), and neither is reassociation (3c's free
space). The remaining razor-2p free-space and bs2 buried gaps are vector
`sin`/`cos`/`exp`, and nothing else measured.

### Wall time and stages vs MSVC (run 4, avx2, ms; ratio to MSVC in brackets)

Medians of 3 per stage (`stage_times.py`); `total` is the exe's own clock.
Run 5, on another runner (EPYC 7763, about 20 % slower in absolute terms),
repeats the ratios. UCRT64 against MSVC, total at one thread: razor-2p free
space 1.65x, bs2 free 0.96x, razor-2p buried 0.88x, bs2 buried 1.31x.
MSVC `/fp:precise`'s razor-2p free-space fill is again UCRT64's (479.8
against 469.6 ms; `/fp:fast` 229.7).

| deck | threads | exe | start | seam | ground | fill | factor | rest | total |
|---|---|---|---|---|---|---|---|---|---|
| razor-2p fs_yagi 6x201 | 1 | MSVC | 4.0 | 1.0 | 0 | 180.5 | 98.5 | 4.5 | 291.0 |
| | | UCRT64 | 4.3 | 1.0 | 0 | 401.1 | 90.7 | 3.5 | 502.9 (1.73) |
| | 2 | MSVC | 3.2 | 1.9 | 0 | 99.8 | 93.6 | 4.4 | 207.2 |
| | | UCRT64 | 4.5 | 1.9 | 0 | 211.3 | 52.1 | 3.5 | 275.6 (1.33) |
| bs2 fs_yagi 6x201 | 1 | MSVC | 3.9 | 1.3 | 0 | 502.4 | 102.4 | 5.1 | 616.2 |
| | | UCRT64 | 4.9 | 1.1 | 0 | 476.6 | 94.6 | 4.1 | 582.0 (0.94) |
| | 2 | MSVC | 3.9 | 2.5 | 0 | 304.4 | 87.3 | 5.2 | 417.2 |
| | | UCRT64 | 3.8 | 2.3 | 0 | 313.6 | 56.9 | 4.1 | 382.6 (0.92) |
| razor-2p buried hub 32x20 | 1 | MSVC | 3.8 | 1.4 | 0.5 | 368.4 | 17.3 | 3.1 | 394.9 |
| | | UCRT64 | 4.0 | 1.4 | 0.8 | 325.4 | 15.2 | 2.5 | 350.1 (0.89) |
| | 2 | MSVC | 4.1 | 2.7 | 0.5 | 199.1 | 17.5 | 3.4 | 230.0 |
| | | UCRT64 | 4.3 | 2.6 | 0.8 | 172.3 | 10.6 | 2.4 | 193.9 (0.84) |
| bs2 buried hub 32x20 | 1 | MSVC | 4.3 | 1.3 | 0.9 | 503.3 | 21.0 | 3.5 | 538.4 |
| | | UCRT64 | 3.9 | 1.2 | 1.2 | 682.6 | 18.6 | 2.8 | 710.8 (1.32) |
| | 2 | MSVC | 3.0 | 2.4 | 0.8 | 320.9 | 21.5 | 3.3 | 354.1 |
| | | UCRT64 | 4.1 | 2.4 | 1.1 | 515.1 | 13.3 | 2.7 | 539.4 (1.52) |

sse2, one thread (the Windows 7 / pre-AVX2 path): UCRT64 against MSVC's sse2
is 1.19x on razor-2p free space (507.6 vs 427.5 ms total), and faster
elsewhere: bs2 free 0.77x (750.4 vs 976.0), razor-2p buried 0.88x (431.0 vs
489.6), bs2 buried 0.87x (841.3 vs 968.4).

The fat UCRT64 exe times as its one-variant exe (razor-2p 6x201, one thread:
0.505 vs 0.510 s, `time_exes.py`, run 4).

- **The whole gap is the fill.** Start-up, the size-policy seam, the
  printout and the write match.
- **The factor is faster under MinGW:** 8-12 % at one thread and 35-44 %
  at two (fs_yagi 6x201: 52-57 ms against 87-94). That build links the two
  OpenBLAS archives Linux links, built by gcc with pthreads; MSVC loads
  scipy's one DLL.
- **bs2 buried scales worse at two threads under libgomp:** 683 -> 515 ms
  (1.33x) against MSVC's 503 -> 321 (1.57x), now from one source tree with
  one thread policy. razor-2p buried scales normally (325 -> 172). Not
  investigated: a 2-core VM is the wrong place to measure thread scaling.

### Optimisation 4 (costed, not built): vector math for gcc on Windows

The cheapest route is now **in-repo**, not SLEEF:

- `src/momwire/_vmath_inline.h` (momwire#1441) already has AVX2 and SSE2
  `exp_lanes` and `sincos_lanes`: exp within 1 ulp and sin/cos within 2 ulp
  of glibc, `tests/test_vmath_1441.py`. MSVC and gcc-sse2 already run it in
  the column kernels.
- The work:
  1. drop `&& !defined(_WIN32)` from the declare-simd guards in the six
     files, for MinGW avx2 only;
  2. add one MinGW-only TU that defines `_ZGVdN4v_sin`, `_ZGVdN4v_cos` and
     `_ZGVdN4v_exp` (the names gcc emits) on `mw_vmath::avx2`, with a
     whole-vector fallback to ucrtbase's scalar functions for lanes beyond
     `MW_VMATH_TRIG_MAX`, non-finite input and exp overflow (x > 709);
  3. compile it with the same gcc, so gcc's Win64 vector-call convention
     matches by construction (a mismatch fails the corpus at once);
  4. the corpus gate and the paired timing.
- Expected, if the Linux libmvec A/B carries over to Windows: fill
  1.85-2.3x faster, which would put UCRT64 at or under MSVC on every deck
  above. That is an extrapolation across OSes, to be measured.
- Estimate: 1 day, mostly CI turns. Risk: the ABI of gcc's simd clones
  under the MS ABI is untested.
- The alternatives cost more. SLEEF's GNU-ABI library (`libsleefgnuabi`)
  exports the same names, but it is a second static library to build and
  cache in each MSYS2 environment, and its MinGW build is less trodden. openlibm is scalar
  and no faster than ucrtbase.
- Steve declined bundled-libm work before, so this is listed only.

## 3. Remaining blockers to shipping, with estimates

### 1. Performance

- **UCRT64, one thread:** razor-2p free space 1.73x MSVC, bs2 buried 1.32x
  (1.52x at two threads); bs2 free space and razor-2p buried faster than
  MSVC.
- **Closing it:** optimisation 4, about 1 day.
- **Shipping without it:** the zip's README says which program to prefer
  (a decision for Steve).

### 2. The exit hang

- **Symptom.** A run that has finished its work (every printout written
  and closed) sits at exit and never returns. 0 odd exit statuses in every
  arm of every run: only hangs.
- **The earlier analysis was confounded, and its fix was not one.** The
  earlier version of this report credited b554dae3 (`_cexit` then
  `TerminateProcess`) with "0 hangs in 50,376". Every hang in its piped
  normal-exit row also ran with d35e80c1's pool shutdown at exit, so it
  could not tell "the `_cexit` exit fixes it" from "not shutting the pools
  down fixes it". Its p ~ 6e-4 treated a measured 5 in 33,576 as a known
  rate. The conditional test, which asks how likely it is that all 5 hangs
  land in the normal-exit runs, gives 0.4^5 ~ 0.010 pooled and
  (33,576/55,968)^5 ~ 0.08 for the `_cexit` arm alone. The pooled row also
  mixed MINGW64 runs with UCRT64 runs.
- **The control (run 4).** UCRT64 fat razor-2p, stdout piped, 44,736 runs
  per arm, interleaved:

  | exit | runs | hangs | where the main thread waits |
  |---|---|---|---|
  | C runtime's exit, pools shut down at exit (d35e80c1) | 44,736 | 9 | 9 x `LoadLibraryExW` (loader lock) |
  | C runtime's exit, pools left alone (the control) | 44,736 | 4 | 4 x `LoadLibraryExW` |
  | `_cexit` + `TerminateProcess` (b554dae3, then the default) | 44,736 | **1** | inside `_cexit`'s atexit table, in `free()` (heap lock) |

  - The pool shutdown is not necessary for the hang: the control hangs too.
    9 against 4 does not show it raising the rate either (P(>= 9 of 13 |
    equal rates) = 0.13).
  - `_cexit` lowers the rate (P(<= 1 of 14 hangs in the `_cexit` third |
    equal rates) = 0.027), but **b554dae3's default hangs**.
- **The three backtrace shapes, and the cause.** They are not one stack,
  as the earlier version said.
  - **Loader lock.** The main thread is in the UCRT's `exit`, inside
    `LoadLibraryExW`, waiting on the loader lock. Another thread is in
    `ZwTerminateThread`. This is run 2's Y_elem_x1.nec and run 3's a1.nec,
    and all 13 normal-exit hangs in run 4.
  - **Heap lock.** The main thread is in `_execute_onexit_table` -> a
    handler in the exe -> `_free_base` -> `RtlFreeHeap` ->
    `RtlEnterCriticalSection`. Two other threads also wait on the heap
    lock, and one is in `ZwTerminateThread`. This is run 3's
    comment_ff_infinite.nec (normal exit) and run 4's `_cexit` hang. The
    handler is unsymbolised (the exes are stripped).
  - **The cause, inferred from OpenBLAS's source**
    (`driver/others/blas_server_win32.c` at both commits `build_openblas.sh`
    builds, 1ad90fb1 and 5ffbf38b). `blas_thread_shutdown_` signals the pool, waits
    **50 ms** for each thread, then calls **`TerminateThread`** on any thread
    still running. OpenBLAS's own destructor, `gotoblas_quit`, calls it on
    every C-runtime exit (`gotoblas_quit` -> `blas_shutdown` ->
    `blas_thread_shutdown_`, `memory.c`), and d35e80c1 called it once more.
  - A thread killed by `TerminateThread` while it holds a lock never
    releases it. A pool thread already on its way out can hold the loader
    lock (thread detach), and one freeing memory holds the heap lock. The exit
    sequence's next `LoadLibraryExW` or `free()` then waits for ever. Each
    backtrace shows one of exactly those two locks, with a thread in
    `ZwTerminateThread` beside it.
  - That also fits the counts. Every C-runtime exit runs `gotoblas_quit`,
    `_cexit` runs it too (inside the table), and run 2-3's
    `TerminateProcess`-alone arm, which runs no table, hung 0 times in
    27,984. No run instrumented the `TerminateThread` calls themselves, so
    this is an inference, though it fits every backtrace and count.
- **The fix (068af7d0).** At the top of `main`, the MinGW wrapper registers
  `end_before_static_teardown` with `atexit`.
  - The table runs last-in first-out. The handler therefore runs **after**
    every handler the work registered, so the exit-time diagnostics
    (`MOMWIRE_SOMM_LAZY_CHECK`'s report) still print. It runs **before**
    everything registered before `main`: the static destructors, `.dtors`,
    and `gotoblas_quit`.
  - It ends the process there, with main's status, so OpenBLAS's
    `TerminateThread` never runs.
  - `MW_EXE_EXIT=cexit` keeps b554dae3's exit for the A/B.
- **The soak of the fix (run 5).** The default ran as two labels, for twice
  the runs, against the run-4 arm that hung most:

  | exit | runs | hangs |
  |---|---|---|
  | the default (068af7d0) | 89,472 | **0** |
  | C runtime's exit, pools shut down at exit (positive control) | 44,736 | 48 |

  - The control's rate is 5x run 4's on the same executable and corpus
    (runners differ). The pairing is what counts. With 48 hangs and two
    thirds of the runs in the default arm, P(0 in the default | equal
    rates) = (1/3)^48 ~ 1e-23.
  - **What the soak cannot show.** It cannot show the fix beating
    b554dae3's rate of 1 in 44,736: that rate predicts 2 hangs in 89,472,
    and P(0) = e^-2 ~ 0.14. That claim rests on the mechanism, because
    the default never runs the code that kills the threads.
- **What is left.**
  - **A 100k-run soak of the shipped default** with stdout piped, as a
    requirement before release, not a confirmation. Run 5 already covers
    89,472. One more dispatch is about 4 x 70 min of runner time.
    Estimate: 0.5 day including reading it.
  - **Upstream.** OpenBLAS's `TerminateThread` in a statically linked
    Windows build is a real bug for any program that exits through the C
    runtime. An issue there is worth filing (Steve's call). The MSVC zip
    has not shown the hang. Its OpenBLAS is a DLL, whose shutdown runs at
    process detach, after `ExitProcess` has already ended the other
    threads. That is the expected reason; it was not measured here.

### 3. CI cost and time

- **MSVC:** `windows-nopython` takes 31 min.
- **UCRT64 build** (run 4):
  - The two OpenBLAS archives take 55-56 min cold, once per
    `build_openblas.sh` change. **They are cached per branch.** The cache
    key does not change between branches, but GitHub only lets a PR branch
    restore caches from its own branch and from main, so main must hold a
    warm entry or every PR pays the cold build.
  - With the cache warm: 4 one-variant + 2 fat exes in about 23 min
    (razor-2p one-variant 2 min, fat 4 min; bs2 one-variant 4 min, fat 7 min).
- **Gates:** contract tests ~3 min, corpus ~4 min per build x variant,
  `fat_compare` ~27 min for the four combinations, SDE ~1 min, venv 10-14
  min. About 60-80 min of Windows runner time per run in all.
- **Soak:** 4 x 70 min, a diagnostic, dispatch only.
- **Per-PR cost if it joins `standalone.yml`.** `standalone.yml` runs on
  every PR touching `src/**`, `standalone/**`, `tests/fixtures/**`,
  `setup.py` and the rest, which is almost every momwire PR. A
  `windows-single` job there adds 60-80 min of Windows runner time to each
  of those runs (more on a cold cache), in parallel with
  `windows-nopython`. The repo is public, so the minutes are free, but the
  runners are shared with the rest of the PR's lanes. **Proposal:** run it
  on tags, on pushes to main (which also keeps main's OpenBLAS cache warm
  for every branch) and on dispatch, not on `pull_request`. A PR that
  breaks MinGW is then caught on main, not in the PR. That trade-off is
  Steve's call.
- **Estimate:** 0.5 day to add the job (warm cache from main, gate set,
  artifact naming).

### 4. Signing

- `sign_windows.py` works on MinGW PEs unchanged (rehearsal-signed 6 exes
  per run).
- The MSVC zip signs 4 exes (its win7 folder reuses the sse2 bytes).
- A single-file zip signs 2 (fat) or 4 (one-variant), and nothing else:
  there are no DLLs beside them. The MSVC zip's DLLs are not signed by this
  workflow either.
- eznec-dropin's signing is unaffected (a different bundle).
- **Change needed:** the file list in the signing step, and one
  `azure_sign` dispatch with the real certificate on the new exes.
- **Estimate:** 1-2 h plus one dispatch.

### 5. Release zip layout and naming

- **Today:** `momwire-nec5-windows-x64.zip`. It holds 4 exes,
  `libscipy_openblas.dll`, `msvcp140.dll`, `vcruntime140.dll`,
  `vcruntime140_1.dll`, `libomp140.x86_64.dll`, `README.txt`, and `win7\`
  with the sse2 pair, the same DLLs and Microsoft's app-local UCRT.
- **Single file:** `momwire-nec5-razor-2p.exe` and
  `momwire-nec5-bspline.exe`, the launcher names (the fat pair picks avx2
  or sse2 by CPU), and `README.txt`. Each fat exe is 27 MB (razor-2p) and
  39 MB (bs2) stripped, against MSVC's 4.6-6.8 MB per exe plus the DLLs.
  The OpenBLAS archives are most of that (two DYNAMIC_ARCH builds linked
  in).
- **Recommendation:** ship beside, as a separate asset,
  `momwire-nec5-windows-x64-single.zip`, labelled preview. Keep the MSVC
  zip canonical until the perf decision. Switching later is a rename.
- **Estimate:** 0.5 day (staging step, zip, release attach, guard test).

### 6. The `-version` line says which build answered (done)

- README-windows.txt says the `-version` line is there "so SimNEC (and the
  CM version card it puts on each deck) shows which program ran".
- Before 9afe2612 every Windows build printed
  `NEC5 momwire-nec5-razor-2p-standalone-avx2 [sb] 0.93.0`, so with a
  preview beside the MSVC zip, a deck or a bug report could not say which
  build answered.
- Now a MinGW build adds its toolchain after the variant:
  `...-standalone-avx2-ucrt64`, `-mingw64` or `-clang64`. The MSVC line is
  unchanged.
- `tests/_standalone_version.py` accepts the suffix and, under
  `MW_EXE_EXPECT_TOOLCHAIN`, requires it. The gates check it in the
  System32 step and the contract tests; the SDE step accepts it.
- **Left:** README-windows.txt's sentence about the line, and SimNEC's
  engine list if it matches on the name (it matches `nec5` in the file name,
  momwire#1239, not this line). Estimate: 0.5 h, part of task 6.

### 7. Docs and tests that name files

Each moves with the asset:

- `standalone/release/README-windows.txt`: which program to use, and the
  `-version` line. For the fat pair it collapses to one program per engine.
- README.md, "The NEC-5 engines on Windows, without Python".
- `site/src/content/docs/reference/standalone.md`, the asset table and the
  Windows paragraph.
- `tests/test_release_notes.py::test_the_windows_and_macos_executables_attach_to_the_release`,
  which asserts the `files:` line and the README mention.
- **Estimate:** 0.5 day.

### 8. Support level and reach (a decision, not work)

- `support/hierarchy.toml` now lists the seven jobs as unassigned
  (60732c4b), as `windows-fat.yml` is. That only makes the registry test
  pass.
- A shipped MinGW exe needs a place in the hierarchy: the hosted app is
  level 1 (`ak-linux-avx2`), Linux sse2 level 2, Windows level 3. Either the
  MinGW exe shares Windows' level 3 (a second build of one platform, held
  by the same gates), or it gets its own context (say `win-mingw-avx2` /
  `-sse2`) so a feature's reach can say "certified on MSVC, not MinGW".
- The second is more honest, because the two builds differ in libm and vector math, but every feature's reach then
  has to say which. It is Steve's call; the registry change is 1-2 h once
  it is made.

### 9. Windows 7

- **MINGW64 (msvcrt.dll)** is the only single-file build that runs on a
  stock Windows 7 SP1. msvcrt.dll is on every Windows, and it passes the
  Win7 import scan with no prerequisite. It needs the libm forwarded to be
  usable (plain MINGW64 is 3-8x slower, run 1). The forwarding would need
  a proper import library (dlltool) in place of `GetProcAddress`. Win7's own
  msvcrt.dll speed is unmeasured: no CI runner runs Windows 7.
- **UCRT64** needs the UCRT on Windows 7: KB2999226 (or any VC++ 2015+
  redistributable), or Microsoft's app-local UCRT DLLs beside the exe, as
  the MSVC zip's win7 folder does today. With the app-local DLLs it is not
  a single file on Windows 7.
- **Options:**
  - (a) UCRT64 for Windows 10/11, plus a win7 folder with the fat exe and
    app-local UCRT (the MSVC zip's pattern).
  - (b) The README requires KB2999226 on Windows 7.
  - (c) A third, MINGW64 + msvcrt-forwarded, build for Windows 7.
- **Recommendation:** (a). It is no new mechanism, and the fat exe picks
  sse2 on old CPUs by itself.
- **Estimate:** 2 h for (a).

### 10. Licensing of the statically linked runtimes

No GPL or LGPL obligation arises. Nothing GPL-without-exception or LGPL is
linked: the build counts 0 libquadmath/libgfortran symbols in every exe,
and OpenBLAS is built `NOFORTRAN=1`.

| component | licence | what it requires |
|---|---|---|
| libgcc, libstdc++, libgomp | GPLv3 with the GCC Runtime Library Exception 3.1 | none for a binary compiled by GCC without a GPL-incompatible plugin (an "Eligible Compilation Process"). Static linking is allowed, and no source offer or notice is required |
| winpthreads (libwinpthread) | MIT-style (mingw-w64) | keep its copyright and permission notice in the distribution |
| mingw-w64 CRT startup and libmingwex | mostly ZPL 2.1 / public domain, some BSD/MIT parts (`COPYING.MinGW-w64-runtime.txt`) | ship that file's notices |
| OpenBLAS (and its C LAPACK translation) | BSD-3-Clause | ship the copyright notice and licence text |
| ucrtbase / msvcrt | system DLLs, not redistributed | none |

**Action:** add a `THIRD-PARTY-NOTICES.txt` to the zip with the OpenBLAS,
LAPACK, winpthreads and mingw-w64 runtime notices. The MSVC zip arguably
owes the OpenBLAS one today too. Estimate: 2 h.

### 11. Support burden of two toolchains

- **MinGW-specific source** (all guarded `__MINGW32__`/`_WIN32`):
  - the locale_t workaround (three sites in two files);
  - the declare-simd guards in six files;
  - the pool-thread and spin settings constructor;
  - the exit wrapper;
  - the `-version` toolchain tag;
  - `pe_comdat_prefix.sh` for the archives;
  - `defer_ctors.cmake` for the fat build.
- **Two Windows numerical baselines** to keep inside PRINTOUT_RTOL_WIN32.
  They already differ from each other, and from Linux, by round-off.
- **Two toolchains to bump:** MSYS2's gcc rolls (16.2 today) and MSVC 14.44.
- If the single file replaces the MSVC zip, the burden drops back to one
  toolchain for the executables. The MSVC toolchain stays for the Python
  wheels.
- **Estimate:** ongoing, about one CI-break-fix a quarter from MSYS2
  package rolls. Pin the MSYS2 package versions to make it deliberate.

## 4. Recommendation

**Ship beside, as a preview: a UCRT64 fat single-file zip next to the MSVC
zip.**

- It is the build that is a single file in Ward's sense: system DLLs only
  on Windows 10/11.
- It passes every gate the MSVC zip passes, and SDE without the exe-only
  waiver.
- It is faster than MSVC on bs2 free space and razor-2p buried, faster on
  sse2 almost everywhere, and 1.3-1.7x slower on razor-2p free space and
  bs2 buried.
- Making it the default is a performance call to take after optimisation
  4, not before.
- The exit hang is no longer a blocker once the release soak (task 2)
  confirms 068af7d0 at 100k runs.

Ordered tasks:

1. ~~Land PR #1975 and bring this branch up to main~~: done (merge e57f724c).
2. Soak the shipped exit (068af7d0) to about 100k runs, stdout piped: run
   5 already has 89,472 with 0 hangs; one more dispatch (0.5 day with the
   reading). Optionally file the OpenBLAS `TerminateThread` issue upstream.
3. Decide the CI trigger for a `windows-single` job (proposal: tags, main
   pushes and dispatch, not PRs), then add it to `standalone.yml`: OpenBLAS
   cache warmed from main, 2 fat exes, gates as here (one-variant builds
   only for the fat == one-variant gate), signing (0.5-1 day).
4. Decide the support level / reach context of a MinGW exe and register it
   in `support/hierarchy.toml` (1-2 h after the decision).
5. Add the win7 folder (fat exe + app-local UCRT) and THIRD-PARTY-NOTICES
   (0.5 day).
6. Release plumbing: the asset name, the release attach, README-windows
   (which program, the `-version` tag), README.md, the site table, and the
   test_release_notes assertion (0.5 day).
7. Optional, Steve's call: optimisation 4 (libmvec names over `mw_vmath`)
   to close the razor-2p free-space and bs2 buried fill gaps (1 day). Then
   decide whether the single file replaces the MSVC zip.

Total to a shipped preview: about 2.5-3 days of work, plus CI waits.
