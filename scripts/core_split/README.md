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
| `cxx_solve_corpus.py` | The C++ solve prototype (`_razor_solve_core.cpp`) against the Python route on every deck given: Z, X, Y and the printout byte for byte, with the route each solve took. |

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
