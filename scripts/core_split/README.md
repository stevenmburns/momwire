# The core/binding split's drift gate

A kernel TU split into a Python-free core (`_<name>_core.cpp`, taking
`mw::view`s) and a thin pybind11 binding must not move a bit or a nanosecond.
The test suite cannot show that: its exactness gates compare C++ against numpy
inside one build, never a new binary against the previous one. These tools
do, and every split runs them, base vs change, on one quiet machine.

| tool | what it answers |
|---|---|
| `mwtrace_plugin.py` + `cmp_traces.py` | Did any bound function's output move? The plugin hashes every native call's inputs and outputs over a pytest lane; the comparator matches two runs by (function, input hash). This is the bit-identity gate. |
| `objdiff.py` | Is the machine code the same? Every TU the split did not touch must be instruction-identical; for the split kernels, compare FP-op sequences or multisets (`--pair`). |
| `bench_calls.py` (+ `capture_window.py`) | Does a call cost the same? Per-call medians on tiny inputs (binding and OpenMP-wake cost) and on captured real windows. |
| `profile_razor_solve.py` | Where a warm razor-2p solve's time goes: construction, geometry, prepare, fill (and the native time in it), LU, readout, per N and ground. `MOMWIRE_RAZOR_CXX=1` times the C++ solve prototype instead. |
| `profile_server_call.py` | One `momwire-nec5-razor-2p` call on a tiny deck: client wall, the launcher's interpreter start, and the server's render split into parse, serve, solve and printout. |
| `cxx_solve_corpus.py` | The C++ solve prototype (`_razor_solve_core.cpp`, and its front half `_razor_front_core.cpp`) against the Python route on every deck given: Z, X, Y and the printout byte for byte, every front-half array against the Python front half's (`front_compare.py`), with the route each solve took and why the front half declined where it did. |
| `front_compare.py` | The Python front half's arrays under the C++ front half's names (`python_front`), and a bytes/dtype/shape/type comparison (`differences`). Shared by the corpus script and `tests/test_razor_cxx_front.py`. |
| `razor_front_driver.cpp` | geometry -> currents with no Python: a plain-text deck through the front half, the fill and the LU, with BLAS/LAPACK dlopen'ed from an OpenBLAS (SciPy's bundled one makes it the Python route's routines). `tests/test_razor_core_standalone.py` builds it from the core TUs alone and requires Z and X byte-equal to the Python route's on `tests/fixtures/razor_front_deck.txt`; with a REPEATS argument it times the chain warm. |
| `deck_compare.py` | The C++ NEC-5 deck reader (`_nec5_deck_core.cpp`, unit 2a) against the Python one: `deck_compare.py DIR ...` reads every `.nec`/`.deck` under each DIR (latin-1, as the seam does) and compares `parse_nec5`'s outcome and the seam's text-level front (NX structures, XQ runs, which refusal prints) line by line, one line per field; `--texts FILE.jsonl` reads harvested texts instead. Reports parsed / refused / raised identically and mismatches (exit 1 on any). |
| `nec5_harvest_plugin.py` | A pytest plugin (`-p nec5_harvest_plugin`, `NEC5_HARVEST_OUT=...`) that records every deck text the suite hands the Python reader, so `deck_compare.py --texts` can replay the suite's hand-written edge cases through both readers. |
| `nec5_deck_driver.cpp` | The deck reader with no Python: prints the C++ dumps for deck files. `tests/test_razor_core_standalone.py` builds it from `_nec5_deck_core.cpp` alone and requires every fixture deck's dumps to equal the Python reader's. |
| `seam_compare.py` | The C++ NEC-5 seam's structure half for razor-2p (`_nec5_seam_core.cpp` with razor-2p's engine, `_nec5_razor_seam_core.cpp`; unit 2b) against the Python seam: `seam_compare.py OUT.jsonl DIR ...` serves every deck through both and, for every deck the C++ seam answers (served, refused, raised), requires the same refusal sentence or exception, the same preparations, and for each the same `RazorSolver` keyword arguments (bytes and Python types), mesh, structure counts, medium and sites, and the same X and Y bytes. Declines are counted by reason, never compared. Exit 1 on any mismatch. |
| `nec5_razor_driver.cpp` | deck text -> currents with no Python: the reader, the seam's structure half, the front half, the fill and the LU, LAPACK `dlopen`ed from an OpenBLAS. `tests/test_razor_core_standalone.py` builds it from the core TUs alone and requires every deck it answers in `tests/fixtures` to be the Python seam's answer, coefficients byte for byte. |
| `printout_compare.py` | The C++ printout for razor-2p (`_nec5_printout_core.cpp` with razor-2p's engine, `_nec5_razor_printout_core.cpp`; unit 3a) against the file `momwire-nec5-razor-2p` writes: `printout_compare.py OUT.jsonl DIR ...` (or `--texts HARVEST.jsonl`, or `--golden DIR` for `scripts/golden_printouts.py`'s stored razor-2p printouts) builds `nec5_razor_printout_driver` from the core TUs, runs every deck through it, and requires every printed deck's bytes (results, refusals, the INTERNAL ERROR frame; CRLF and latin-1) to be Python's. Declines are counted by reason, never compared. Exit 1 on any mismatch. |
| `exe_compare.py` | The standalone executable (`standalone/`, unit 4) against `momwire-nec5-razor-2p`: `OPENBLAS_NUM_THREADS=T OMP_NUM_THREADS=T exe_compare.py OUT.jsonl --exe EXE --launcher LAUNCHER DIR ...` answers every deck four ways (the executable as shipped; the executable with every deck forced through its embedded Python; the real launcher, on a fresh private resident server; `python_printout` in-process) and requires all four printouts to be one file, byte for byte. For a deck the C++ route answered, every finished run's arrays must equal the embedded Python's on the executable's own library (`exe_arrays_hook.py`, which also records `_lapack.HOST_CALLS` to prove the fallback ran there). Counts each deck by the route that answered it (`cxx:serial`, `cxx:threads`, `fallback:<why>`). Exit 1 on any mismatch. |
| `stream_compare.py` | SimNEC's streaming call (`- -`, momwire#1512): the no-Python executable against `momwire-nec5-razor-2p`'s streaming mode through real pipes. `stream_compare.py OUT.jsonl --exe EXE --launcher LAUNCHER STREAM ...` writes each input structure by structure (SimNEC's lockstep: each printout read to its `NX` echo before the next write) and all at once; every segment must be the launcher's, byte for byte, and the launcher's stream its own file form less the truncated empty structure. Exit 1 on any mismatch. |
| `nec5_razor_printout_driver.cpp` | deck file in -> printout file out, no Python: the reader, the seam's structure half, the front half, the fill and the LU (SciPy's OpenBLAS), the run half (numpy's own ILP64 OpenBLAS for every product and solve numpy makes) and the writer. `tests/test_razor_core_standalone.py` builds it from the core TUs alone and requires every fixture deck it prints to be the launcher's file, byte for byte; its `--transliterate` and `--format` probes pin the codec and the number cells. |
| `exact_kernel_coaxial_driver.cpp` | The B-spline basis' exact kernel with no Python (`_exact_kernel_coaxial_core.h`: the coaxial pair moments, their row windows and the ring kernel, momwire#1517 step 0). `tests/test_razor_core_standalone.py` builds it from the core TUs alone and requires a graded, mixed-direction group's windows, pair moments and ring kernel to be the extension's bytes. |
| `bspline_core_driver.cpp` | The B-spline kernels with no Python (`_bspline_core.h`, momwire#1517 step 0) on the default host: a bent segment set's off-edge moments (lane path and per-pair walk) and a uniform edge's static table. `tests/test_razor_core_standalone.py` builds it from the core TUs alone and requires the extension's bytes. The core is the extension TU's kernel text over `_mw_nd.h` (pybind11's array vocabulary over a host's memory); `_accel_bspline.cpp` binds it through `_mw_nd_binding.h`, so the trace / objdiff recipe below applies with `--pair base/_accel_bspline.o change/_bspline_core.o`. |
| `sinusoidal_core_driver.cpp` | The sinusoidal basis' point-matched field tensors with no Python (`_sinusoidal_core.h`, momwire#1607 B1) on the default host. `tests/test_razor_core_standalone.py` records every call real solves make to the extension (free space, PEC image, reflection-coefficient ground, extended kernel, complex-k buried, literal and folded) and requires the core to return the same bytes. The core is the extension TU's point-matched text over `_mw_nd.h`; `_accel_sinusoidal.cpp` keeps the Galerkin far fill and binds the core, so the recipe below applies with `--pair base/_accel_sinusoidal.o change/_sinusoidal_core.o`. |

`capture_window.py` also regenerates the fixture of
`tests/test_razor_core_standalone.py`, which builds the razor core TUs and
`razor_chain_driver.cpp` with no Python include path and checks the C++-only
chain against the Python one byte for byte. That test runs in the default
lane; the tools here are run by hand for each split.

## Recipe

Two fresh clones (base and change) with fresh venvs, each built with
`make build`. With `B` and `C` their venvs' pythons and `S` this directory:

```sh
# 1. outputs, per lane and per ISA variant (repeat with
#    MOMWIRE_FORCE_VARIANT=sse2, and with the slow lane)
MWTRACE_OUT=trace-base PYTHONPATH=$S $B -m pytest tests/ \
    -m "not slow and not memgate and not integration" -p mwtrace_plugin -q
MWTRACE_OUT=trace-change PYTHONPATH=$S $C -m pytest tests/ \
    -m "not slow and not memgate and not integration" -p mwtrace_plugin -q
python $S/cmp_traces.py trace-base trace-change --per-function '^razor_'
# Control: the same comparison between an avx2 and an sse2 run must go RED.

# 2. object code (temp dirs of the two builds)
T=build/temp.linux-x86_64-cpython-312/_accelerators_avx2/src/momwire
python $S/objdiff.py base/$T change/$T
python $S/objdiff.py --pair base/$T/_accel_razor.o change/$T/_razor_core.o \
    --rename '^razor_=' --report '_omp_fn|seg_moments_impl|assemble_t1|cplx'

# 3. speed: capture windows once, then interleave base and change runs
$C $S/capture_window.py w108.npz --nsegs 108
$C $S/capture_window.py w801.npz --nsegs 801
for r in 1 2 3; do for t in base change; do
  OMP_NUM_THREADS=8 $t/.venv/bin/python $S/bench_calls.py $t-8t-r$r.json \
      --window real108=w108.npz --window real801=w801.npz
done; done
```

Pass: zero moved outputs in every lane and variant, the control red;
untouched TUs instruction-identical and the split kernels' FP-op multisets
unchanged; per-call medians within run-to-run noise (about +-2 %).

## What a split keeps

- Kernel bodies verbatim. `mw::view` answers `x(i, j)` and `x.shape(k)` as
  pybind11's unchecked proxy did; its offset spelling is part of the codegen
  contract (see its comment in `src/momwire/_mw_core.h`).
- Outputs allocated with numpy in the binding, so the tracemalloc memgates
  still see them. A C++ caller passes its own buffers.
- Names, keyword arguments, defaults, docstrings and capability flags of the
  bound functions, byte for byte.
- Validation in the core, in the pre-split order and wording, except the
  rank checks a view cannot express, which stay in the binding.
