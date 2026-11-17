# Single-file Windows executables (MinGW-w64): what shipping them takes

Status: experiment on branch `windows-static-exe-v2` (no PR). It restarts the
parked `windows-static-exe` (tip d35e80c1) on top of `fat-msvc-exe`
(PR #1975, open). Workflow: `.github/workflows/windows-static.yml`, which runs
on pushes to this branch and on dispatch only. Measured on GitHub
`windows-latest` runners, 2026-11-17:

| run | commit | what it measured |
|---|---|---|
| [44057069874](https://github.com/stevenmburns/momwire/actions/runs/44057069874) | 80ae9fba | MINGW64 full gates, libm shims, UCRT64 avx2, CLANG64 libm, the first paired timing, stress A/B (pool shutdown) |
| [44081988222](https://github.com/stevenmburns/momwire/actions/runs/44081988222) | 734612ca | UCRT64 avx2 + sse2 + fat: SDE, fat == one-variant, the second paired timing, the hang's backtrace |
| [44092464961](https://github.com/stevenmburns/momwire/actions/runs/44092464961) | b554dae3 | UCRT64 contract tests and corpus with the `_cexit` exit, the three-exit stress A/B |

Every timing below is the median of 5 runs, with the builds interleaved on
one runner (a 2-core / 4-thread AMD EPYC 7763 VM). It is indicative: the
point is the pairing. The MSVC column is the shipped zip from the latest
green `standalone.yml` run on main, timed on the same runner.

## Bottom line

- **The libm explains the 3-8x gap.** mingw-w64's own `exp`/`sin`/`cos`/
  `log`/`pow`/`atan2` (libmingwex's, which win gcc's link) take 39-79 ns a
  call. ucrtbase.dll's take 4-12 ns.
- **MSYS2 UCRT64 closes most of the gap with no code change.** It binds
  those functions to ucrtbase. bs2 runs at MSVC's speed in free space, and
  razor-2p beats MSVC over buried radials.
- **Two gaps remain, both in the fill.** razor-2p in free space is 1.67x
  slower, and bs2 over buried radials 1.32x, at one thread. Both come from
  MSVC's vectorised math (`/fp:fast` + AVX2), which gcc on Windows lacks.
  On Linux, glibc's libmvec gives gcc vector math. A libmvec-compatible
  SLEEF build is the costed next step.
- **The rare exit hang is diagnosed and fixed.** The UCRT's own `exit`
  calls `LoadLibraryExW` and waits on the loader lock, which a worker thread
  holds while it exits (gdb, runs 2-3). With `_cexit` then
  `TerminateProcess`: 0 hangs in 50,376 runs. The C runtime's normal exit
  hung 5 times in 33,576 (p ~ 6e-4 that this is chance).
- **Recommendation: ship-beside.** Ship a UCRT64 fat single-file zip as a
  preview next to the MSVC zip. Keep the MSVC zip as the default until the
  razor-2p free-space gap closes. The task list is at the end.

## 1. What works today (gates)

Every row is measured, not inferred.

| gate | MINGW64 (msvcrt.dll) | UCRT64 (ucrtbase via api-ms-win-crt-*) |
|---|---|---|
| import table names system DLLs only | pass: `kernel32.dll msvcrt.dll`, nothing else (run 1) | pass: `kernel32.dll` + 13 `api-ms-win-crt-*` sets (runs 1-2) |
| Windows 7 import scan (`scan_win7_imports.py`) | pass (runs 1-2) | pass; api-ms-win-crt-* counts as the UCRT prerequisite (KB2999226), as for the MSVC zip |
| rehearsal signature (`sign_windows.py`) | pass, 6 exes (4 one-variant + 2 fat) | pass, 6 exes |
| run from a System32-only PATH (deck + SimNEC stream) | pass, all 12 builds | pass |
| launcher contract tests (`test_standalone_nopython.py`) | pass: razor-2p 49 + 4 skipped, bs2 52 + 1 skipped, fat likewise (run 1) | pass, one-variant and fat: 49 + 4 skipped, 52 + 1 skipped (run 3). Run 2's bs2 failure (the `TerminateProcess`-only exit skipped an atexit report) is what b554dae3's `_cexit` fixes |
| corpus vs the launcher, `--rtol win32` | 0 mismatches, 783 razor-2p / 786 bs2 decks, avx2 and sse2; worst ratio 5.78e-15 (run 1) | 0 mismatches, avx2 and sse2: worst 5.66e-15 / 4.54e-16 (razor-2p), 4.71e-16 / 5.54e-15 (bs2) (run 3) |
| fat exe under Intel SDE (Westmere, Sandy Bridge -> sse2; Haswell -> avx2), printouts == one-variant | pass, whole-process checking (run 1) | pass, whole-process checking: ucrtbase's own dispatch did not fault (run 2). The MSVC fat build needs `-chip_check_exe_only` for VCRUNTIME140's memcpy |
| fat == one-variant byte for byte, 702 decks x 2 copies x both paths | razor-2p avx2: 1402/1404 (both differences were exit hangs, not bytes); all others 1404/1404 (run 1) | **1404/1404 in all four**, runs 2 and 3 |
| the avx2 initializers run only on the avx2 path | 10 (razor-2p) and 12 (bs2) deferred `.ctors` entries; none on sse2 | same |
| no libquadmath / libgfortran linked (licensing) | 0 symbols in each exe | 0 |

Unlike the MSVC fat port, the MinGW fat build does not rename its shared
template and inline bodies (`fat_private_comdats.py`). The avx2 copies may
therefore call an sse2-compiled `std::` body. Under gcc, fat == one-variant
still holds for every deck (1404/1404 x 4, UCRT64). The MSVC port saw three
decks differ before it added the renaming. The fat-vs-one-variant gate is
what holds the MinGW build to it.

## 2. Performance vs the MSVC zip

### The libm, per call (`scripts/windows_static/libm_bench.c`)

ns per call, best of 5, run 1. Each environment's columns come from its own
job (its own runner); run 2's MINGW64 link column read 49-84 ns.

| function | MINGW64 link (libmingwex) | msvcrt.dll | ucrtbase.dll | UCRT64 link | CLANG64 link |
|---|---|---|---|---|---|
| exp | 38.7 | 5.5 | 3.9 | 3.5 | 4.5 |
| log | 23.7 | 4.9 | 4.0 | 4.1 | 5.0 |
| sin | 41.0 | 7.7 | 5.9 | 4.9 | 7.0 |
| cos | 41.2 | 7.0 | 5.6 | 5.3 | 6.9 |
| pow | 79.2 | 30.4 | 12.0 | 12.4 | 16.2 |
| atan2 | 41.7 | 10.1 | 8.4 | 9.7 | 14.4 |
| sincos | 42.0 | - | - | 11.1 | 15.9 |
| cexp | 106.7 | - | - | 21.5 | 33.9 |

Which symbols resolve where (nm on the exe, objdump on its imports):

- **MINGW64.** `exp log sin cos pow atan2 hypot sincos cexp __muldc3
  __divdc3` are libmingwex's own code in the exe. libmingwex is ahead of
  `-lmsvcrt` on gcc's link line, so msvcrt.dll's faster ones are never
  reached. msvcrt supplies only `_hypot asin atan ceil cosh log10 sinh tan`.
- **UCRT64 and CLANG64.** `exp log sin cos pow atan2 hypot` are imported
  from `api-ms-win-crt-math-l1-1-0.dll` (ucrtbase). The exe's `T` symbols
  for them are 8-byte import thunks. `sincos` and `cexp` stay libmingwex's,
  written over the imported `sin`/`cos`/`exp` (11 and 22 ns). So no
  link-order trick or `__imp_` call is needed: the UCRT toolchain already
  takes ucrtbase's libm.
- **The values move by 0-2 ulp** between libmingwex and the DLLs, and
  ucrtbase matches UCRT64's link exactly. All stay within
  PRINTOUT_RTOL_WIN32 on the whole corpus.

### Optimisation 1: msvcrt/ucrtbase libm in a MINGW64 build (`MW_EXE_LIBM_SHIM`)

`standalone/mingw_libm_shim.c` defines the nine names in the exe, so
libmingwex's members are never pulled in, and forwards them to
msvcrt.dll's or ucrtbase.dll's through `GetProcAddress`. The corpus passes
for both (0 mismatches, run 1). The lookup is an experiment device: a
product build would import the functions so the gates can see the DLL.

### Optimisation 2: UCRT64

No code change. See the tables below. Its fat exe times the same as its
one-variant exe.

### Optimisation 2b: CLANG64

Not built as an exe. Its libm is ucrtbase's too (the table above, about
1.2x UCRT64's per call). A CLANG64 build would swap libgomp for LLVM's
libomp (CLANG64 ships `libomp.a`) and build OpenBLAS with clang. That is a
second OpenMP runtime and a second OpenBLAS toolchain, against Linux's
libgomp + gcc, for no libm gain.

### Optimisation 3: `-fno-math-errno`

Already in `MW_CORE_FLAGS` for every gcc build: `standalone/CMakeLists.txt`
has `-O3 -fopenmp-simd -fno-math-errno -ffp-contract=off`. The 3-8x gap was
measured with it on, so there was nothing left to try there. `-ffast-math`
was not tried. gcc on Windows has no vector libm to vectorise into, so it
would buy reassociation only, at the cost of the IEEE contract the gcc
branches keep.

### Wall time, seconds (run 1, avx2; ratio to MSVC in brackets)

razor-2p:

| deck | threads | MSVC | MINGW64 | shim msvcrt | shim ucrtbase | UCRT64 |
|---|---|---|---|---|---|---|
| fs_yagi 6x41 | 1 | 0.024 | 0.120 (5.0) | 0.038 (1.58) | 0.035 (1.46) | 0.034 (1.42) |
| fs_yagi 6x201 | 1 | 0.358 | 2.673 (7.5) | 0.749 (2.09) | 0.657 (1.84) | 0.598 (1.67) |
| fs_yagi 6x201 | auto (2) | 0.241 | 1.372 (5.7) | 0.426 (1.77) | 0.373 (1.55) | 0.342 (1.42) |
| buried hub 16x20 | 1 | 0.140 | 0.479 (3.4) | 0.168 (1.20) | 0.132 (0.94) | 0.123 (0.88) |
| buried hub 32x20 | 1 | 0.526 | 1.899 (3.6) | 0.655 (1.25) | 0.508 (0.97) | 0.469 (0.89) |
| buried hub 32x20 | auto (2) | 0.312 | 0.981 (3.1) | 0.355 (1.14) | 0.286 (0.92) | 0.258 (0.83) |

bs2:

| deck | threads | MSVC | MINGW64 | shim msvcrt | shim ucrtbase | UCRT64 |
|---|---|---|---|---|---|---|
| fs_yagi 6x41 | 1 | 0.051 | 0.244 (4.8) | 0.058 (1.14) | 0.049 (0.96) | 0.051 (1.00) |
| fs_yagi 6x201 | 1 | 0.754 | 3.720 (4.9) | 0.878 (1.16) | 0.738 (0.98) | 0.744 (0.99) |
| fs_yagi 6x201 | auto (2) | 0.524 | 1.934 (3.7) | 0.542 (1.03) | 0.471 (0.90) | 0.476 (0.91) |
| buried hub 16x20 | 1 | 0.174 | 1.315 (7.6) | 0.281 (1.61) | 0.223 (1.28) | 0.224 (1.29) |
| buried hub 32x20 | 1 | 0.677 | 5.441 (8.0) | 1.130 (1.67) | 0.886 (1.31) | 0.893 (1.32) |
| buried hub 32x20 | auto (2) | 0.469 | 3.008 (6.4) | 0.851 (1.81) | 0.750 (1.60) | 0.749 (1.60) |

Run 2 (another runner) repeats these within a few per cent. It adds the
UCRT64 fat exe, which equals its one-variant (razor-2p 6x201: 0.587 vs
0.586; bs2 hub32: 0.844 vs 0.845). It also adds sse2, where UCRT64 is
mostly faster than MSVC's sse2 (bs2 6x201 0.942 vs 1.155; razor-2p hub32
0.551 vs 0.616; razor-2p 6x201 0.595 vs 0.516 is the exception). That
matters for Windows 7 and pre-AVX2 CPUs.

### Per stage, ms (`stage_times.py`, run 2, one thread)

| deck / exe | start | seam | ground | fill | factor | rest |
|---|---|---|---|---|---|---|
| fs_yagi 6x201 razor-2p MSVC | 4.5 | 1.3 | 0 | 219.2 | 113.8 | 5.2 |
| ... MINGW64 | 4.4 | 1.1 | 0 | 2541.7 | 109.1 | 4.5 |
| ... UCRT64 | 4.9 | 1.1 | 0 | 462.8 | 107.2 | 4.6 |
| fs_yagi 6x201 bs2 MSVC | 3.8 | 1.5 | 0 | 592.8 | 117.2 | 5.9 |
| ... UCRT64 | 4.8 | 1.4 | 0 | 609.6 | 113.0 | 5.1 |
| buried hub32 razor-2p MSVC | 5.1 | 1.7 | 0.6 | 481.9 | 20.4 | 3.7 |
| ... UCRT64 | 5.8 | 1.6 | 1.0 | 414.9 | 18.1 | 3.3 |
| buried hub32 bs2 MSVC | 4.2 | 1.6 | 0.9 | 602.6 | 24.2 | 4.0 |
| ... UCRT64 | 5.8 | 1.5 | 1.3 | 809.0 | 21.9 | 3.5 |

- **The whole gap is the fill.** Start-up, the size-policy seam, the
  printout and the write match. The factor is 5-10 % faster in the MinGW
  build at one thread, and 10-30 % faster at two threads (fs_yagi 6x201:
  71-91 ms against MSVC's 89-108 ms, both engines, both runs). That build links the two OpenBLAS archives Linux links,
  built by gcc with pthreads, where MSVC loads scipy's one DLL.
- **The remaining fill gap is vector math, not scalar libm.**
  - Under gcc, the razor-2p free-space fill is the same in sse2 and avx2
    (UCRT64 471.8 vs 462.8 ms). MSVC's avx2 fill is 1.75x faster than its
    own sse2 (219 vs 383 ms).
  - MSVC `/fp:fast /arch:AVX2` turns the kernels' `sin`/`cos`/`exp` loops
    into calls to its vector math library. gcc does the same on Linux
    through glibc's libmvec: the `#pragma omp declare simd` declarations in
    `_mw_core.h` and its siblings, compiled out on `_WIN32` because Windows
    has no libmvec.
  - The bs2 buried fill likewise: UCRT64 avx2 809 ms, sse2 1068 ms;
    MSVC 603 ms.
- **bs2 buried also scales worse at two threads under libgomp:** 809 -> 667
  ms (1.21x), against MSVC's 603 -> 389 (1.55x). The razor-2p buried fill
  scales normally (415 -> 221). Not investigated: a 2-core VM is the wrong
  place to measure thread scaling.

### Optimisation 4 (costed, not built): a vector libm

1-3 do not close the razor-2p free-space gap, and the only lever left is
vector math. The cheapest route is **SLEEF's GNU-ABI library**
(`libsleefgnuabi`). It exports the libmvec names (`_ZGVdN4v_sin`,
`_ZGVdN4v_cos`, `_ZGVdN4v_exp`, ...) that gcc emits once
`#pragma omp declare simd` is in force. The work is to:

1. drop `&& !defined(_WIN32)` from the declare-simd guards in six files
   (the same six guards this branch had to add);
2. build SLEEF (Boost Software License 1.0) statically in each MSYS2
   environment, cached like OpenBLAS, and link it;
3. re-run the corpus gate and the paired timing. SLEEF's u10 functions
   are within 1.0 ulp. libmvec's are within 4, and the Linux gates already
   pass with those, so PRINTOUT_RTOL_WIN32 should hold.

Estimate: 1-2 days, mostly CI turns. The risks: SLEEF's MinGW build is
less trodden than its Linux one, and gcc's Win64 vector ABI passes `__m256d`
by reference (SLEEF compiled by the same gcc matches it, MSVC-built SLEEF
would not). The alternatives cost more: openlibm is scalar only and no
faster than ucrtbase, and a small in-repo vector `sincos`/`exp` for the hot
kernels means owning numerical code. Steve declined this work before, so it
is listed only.

## 3. Remaining blockers to shipping, with estimates

### 1. Performance (above)

- **UCRT64:** razor-2p free space 1.4-1.7x MSVC; bs2 buried 1.3-1.6x;
  elsewhere at or better than MSVC.
- **Closing the gap:** optimisation 4, 1-2 days.
- **Shipping without it:** the zip's README says which program to prefer
  (a decision for Steve).

### 2. The exit hang

- **Symptom.** About 1 run in 6,700 (stdout piped) sits at exit after
  writing a complete printout. Run 1 had 2 in 5,616 `fat_compare` runs, with the
  d35e80c1 pool shutdown on. Run 2's normal-exit arm had 1 in 16,776.
- **Mechanism** (gdb, the same in all three backtraces, runs 2 and 3). The main thread is in ucrtbase's `exit`,
  inside `LoadLibraryExW`, waiting on the loader lock. A second thread is
  in `ZwTerminateThread`, exiting. d35e80c1's shutdown of the OpenBLAS pools
  at exit does not prevent it, and can cause it: it makes pool threads exit
  at that moment.
- **Fix** (b554dae3, `momwire_razor2p.cpp` MinGW `main`). `fflush`, then
  `_cexit()` (the atexit table, so exit-time diagnostics still print), then
  `TerminateProcess(rc)`. The pool shutdown is now only on the old path,
  `MW_EXE_EXIT=normal`.
- **Measured.** Hangs per run, all with the printout complete:

  | exit | runs | hangs |
  |---|---|---|
  | C runtime's normal exit, stdout piped (run 1 `fat_compare`, runs 2-3 stress) | 33,576 | 5 |
  | the same, stdout to DEVNULL (run 1 stress: the pool shutdown on and off) | 33,552 | 0 |
  | `TerminateProcess` alone (run 2 stress + `fat_compare`, run 3 stress) | 27,984 | 0 |
  | `_cexit` + `TerminateProcess`, the default (run 3 stress + `fat_compare`) | 22,392 | 0 |

  With stdout to DEVNULL the hang never showed, which is why run 1's stress
  measured nothing; `stress.py` now pipes stdout as `fat_compare.py` does.
  At the piped normal exit's rate, 0 in 50,376 has probability about 6e-4.
- **What is left.** No exit status changed (0 odd exits in every arm).
  Before release, one longer soak on a real Windows machine, about 100k
  runs, which is half a day of CI or one night locally. Estimate: 0.5 day.

### 3. CI cost and time

- **MSVC:** `windows-nopython` takes 31 min.
- **UCRT64 build:**
  - The two OpenBLAS archives take 55-56 min, once per
    `build_openblas.sh` change. They are cached per branch, so the first
    run on any branch pays (cache from main to share it).
  - With the cache warm: 4 one-variant + 2 fat exes in about 20 min
    (90-320 s each; fat bs2 is the longest).
- **Gates:**
  - Contract tests, ~3 min.
  - Corpus, ~4 min per build x variant.
  - `fat_compare`, ~27 min for the 4 combinations.
  - SDE, ~1 min.
  - Venv, 10-14 min.
- **Stress:** 60-70 min. It is a diagnostic, not a gate.
- **Two lanes as a product:** about +60 min of Windows runner time per
  standalone run, in parallel with the MSVC job, so little added wall time
  if they run side by side.
- **Estimate:** 0.5 day to fold the UCRT64 job into `standalone.yml` (warm
  cache from main, gate set, artifact naming).

### 4. Signing

- `sign_windows.py` works on MinGW PEs unchanged (rehearsal-signed 12 exes).
- The MSVC zip signs 4 exes (its win7 folder reuses the sse2 bytes).
- A single-file zip signs 2 (fat) or 4 (one-variant), and nothing else:
  there are no DLLs beside them (the MSVC zip's DLLs are not signed by
  this workflow either).
- eznec-dropin's signing is unaffected (a different bundle).
- **Change needed:** the file list in the signing step, and the
  azure_sign rehearsal run once with the real certificate on the new exes.
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
  `momwire-nec5-windows-x64-single.zip`, labelled preview. Keep the MSVC zip
  canonical until the perf decision. Switching later is a rename.
- **Estimate:** 0.5 day (staging step, zip, release attach, guard test).

### 6. Docs and tests that name files

Each moves with the asset:

- `standalone/release/README-windows.txt`: which program to use. For the
  fat pair it collapses to one program per engine.
- README.md, "The NEC-5 engines on Windows, without Python".
- `site/src/content/docs/reference/standalone.md`, the asset table and the
  Windows paragraph.
- `tests/test_release_notes.py::test_the_windows_and_macos_executables_attach_to_the_release`,
  which asserts the `files:` line and the README mention.
- **Estimate:** 0.5 day.

### 7. Windows 7

- **MINGW64 (msvcrt.dll)** is the only single-file build that runs on a
  stock Windows 7 SP1. msvcrt.dll is on every Windows, and it passes the
  Win7 import scan with no prerequisite. Its speed is "shim msvcrt" above,
  1.0-2.1x MSVC, but only with the libm forwarded.
  - The plain MINGW64 build is 3-8x slower.
  - The forwarding would need a proper import library (dlltool) in place of
    `GetProcAddress`.
  - Win7's own msvcrt.dll speed is unmeasured: no CI runner runs Windows 7.
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

### 8. Licensing of the statically linked runtimes

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

### 9. Support burden of two toolchains

- **MinGW-specific source** (all guarded `__MINGW32__`/`_WIN32`):
  - the locale_t workaround (three sites in two files);
  - the declare-simd guards in six files;
  - the pool-thread and spin settings constructor;
  - the exit wrapper;
  - `pe_comdat_prefix.sh` for the archives;
  - `defer_ctors.cmake` for the fat build.
- **Two Windows numerical baselines** to keep inside PRINTOUT_RTOL_WIN32.
  They already differ from each other, and from Linux, by round-off.
- **Two toolchains to bump:** MSYS2's gcc rolls (16.2 today) and MSVC 14.44.
- If the single file replaces the MSVC zip, the burden drops back to one
  toolchain. The MSVC build would remain only if the Python extension needs
  it, and it does: the wheels are MSVC.
- **Estimate:** ongoing, about one CI-break-fix a quarter from MSYS2
  package rolls. Pin the MSYS2 package versions to make it deliberate.

## 4. Recommendation

**Ship beside, as a preview: a UCRT64 fat single-file zip next to the MSVC
zip.**

- It is the build that is a single file in Ward's sense: system DLLs only
  on Windows 10/11.
- It passes every gate the MSVC zip passes, and SDE without the
  exe-only waiver.
- It is at or better than MSVC speed on bs2 free space and on razor-2p
  buried, and 1.3-1.7x slower on razor-2p free space and bs2 buried.
- Making it the default is a performance call to take after optimisation 4,
  not before.

Ordered tasks:

1. Land PR #1975 (the shared fat source), then rebase this branch on main.
   The MinGW pieces sit beside it under `__MINGW32__` (0.5 h).
2. Soak the exit fix (already 0 in 50,376) for about 100k runs on a real
   Windows machine (0.5 day).
3. Move UCRT64 into `standalone.yml` as a `windows-single` job: OpenBLAS
   cache warmed from main, 2 fat exes, gates as here (one-variant builds
   only for the fat == one-variant gate), signing (0.5-1 day).
4. Add the win7 folder (fat exe + app-local UCRT) and THIRD-PARTY-NOTICES
   (0.5 day).
5. Release plumbing: the asset name, the release attach, the README-windows
   section, README.md, the site table, and the test_release_notes assertion
   (0.5 day).
6. Optional, Steve's call: optimisation 4 (SLEEF GNU-ABI) to close the
   razor-2p free-space fill gap (1-2 days). Then decide whether the single
   file replaces the MSVC zip.

Total to a shipped preview: about 2.5-3 days of work, plus CI waits.
