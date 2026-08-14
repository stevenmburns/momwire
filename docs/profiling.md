# Profiling pysim

The N=21 hentenna sin sweep is our standard interactive-UI workload
(`scripts/vtune_hentenna_width_sweep.py`). When tuning it, pick the
right tool for the question you're asking — none of them is "always
the best one."

## Quick guide

| Question | Tool |
|---|---|
| Where in the C kernel is time going? | VTune `hotspots`, source-line view |
| Which Python function is called the most? | cProfile `--sort=ncalls` |
| Which Python frame dominates wall-clock? | py-spy `record` |
| Does change X actually save N ms wall-clock? | direct `time.perf_counter()` micro-bench |
| Of the time inside Python function F, how much is in C function G? | VTune + pyitt annotations (see below) |

cProfile inflates Python-frame-heavy code (its callback adds ~30 µs
per Python frame entry). py-spy doesn't have that distortion. For
before/after comparisons that swap a Python loop for vectorized numpy,
trust py-spy or direct wall-clock, not cProfile.

## py-spy

No setup beyond `pip install py-spy`. Sampled at low overhead, gives
real Python frame attribution. Run on the harness:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
  .venv/bin/py-spy record -r 200 -F -f raw -o /tmp/sin.raw \
    -- .venv/bin/python -m scripts.vtune_hentenna_width_sweep --solver sin --warmup --reps 300
```

Aggregate by leaf frame:

```bash
awk '{n=NF;c=$n;$n="";split($0,p,";");l=p[length(p)];cnt[l]+=c}
     END {for(l in cnt) print cnt[l]"\t"l}' /tmp/sin.raw \
  | sort -k1 -rn | head -15
```

## VTune (hotspots) — C-side

VTune resolves to C-level symbols (libm, `_accelerators.cpp`, BLAS
kernels). It can NOT name Python functions on its own — Python code
all bubbles up to `_PyEval_EvalFrameDefault`.

```bash
. /opt/intel/oneapi/vtune/latest/env/vars.sh
vtune -collect hotspots -knob enable-stack-collection=true -knob sampling-mode=sw \
  -result-dir /tmp/vtune_sin \
  -- .venv/bin/python -m scripts.vtune_hentenna_width_sweep --solver sin --warmup --reps 300

vtune -report hotspots -r /tmp/vtune_sin -group-by source-line \
  -format=csv -csv-delimiter='|' | head -20
```

Needs `ptrace_scope=0` and `perf_event_paranoid<=1`:

```bash
sudo sysctl kernel.yama.ptrace_scope=0 kernel.perf_event_paranoid=1
```

These revert at reboot.

## VTune + pyitt — Python-frame attribution

Intel's ITT API lets Python code emit named task ranges that VTune
picks up alongside its C-level sampling. The result: you can group
the same VTune run by **task** (Python function name) AND by
**function** (C function), and cross-reference them in the GUI.

### One-off profiling without touching library code

Use `pyitt` as a context manager in the harness or a throwaway script:

```python
import pyitt
import scripts.vtune_hentenna_width_sweep as s

for w in width_factors:
    with pyitt.task("step"):
        s._step_sin(21, w)
    with pyitt.task("post-step-cleanup"):
        do_other_stuff()
```

For decorator-style annotation of specific functions in an ad-hoc
profiling script:

```python
import pyitt

@pyitt.task
def my_workload():
    ...
```

See `scripts/pyitt_smoke.py` for a minimal end-to-end example that
runs three artificial workloads and confirms VTune picks up the task
names.

### Running it

```bash
pip install pyitt

. /opt/intel/oneapi/vtune/latest/env/vars.sh
vtune -collect hotspots -knob enable-stack-collection=true -knob sampling-mode=sw \
  -result-dir /tmp/vtune_annotated \
  -- .venv/bin/python -m scripts.pyitt_smoke

vtune -report hotspots -r /tmp/vtune_annotated -group-by task \
  -format=csv -csv-delimiter='|'
```

### Cost & workflow

Each `@pyitt.task` decorator call adds ~ 2 µs of ITT-API overhead per
function invocation. On the sinusoidal hot path (1× `compute_impedance`
+ 2× `_build_geometry` + 2× `_basis_coefs` + … per step), annotating
the whole module raised wall-clock from 5.5 → 6.7 ms/step (+21%).

For *profiling sessions* this is fine — you accept ~ 20% slowdown for
the duration of the run to get the attribution. For *production / CI*
it's not fine, and there's a real foot-gun risk: someone adds `pyitt`
to a requirements file and silently slows every prod solve by 20%.

**The workflow we settled on**: don't keep library annotations on a
branch. When you need them:

1. `pip install pyitt` into your dev venv.
2. Add `@pyitt.task` decorators (or `with pyitt.task("name"):` blocks)
   to the specific functions / regions you want attributed. Edit them
   into `src/pysim/*.py` or wrap them at the harness level — whichever
   gives you the granularity you need.
3. Run VTune as above and read the report.
4. Revert your edits with `git checkout HEAD -- src/pysim/...` before
   committing anything else. Library code stays clean; `pyitt` stays
   out of any committed requirements file.

If you only need a couple of named regions inside the harness loop
(not deep in the library), prefer the context-manager form in a
throwaway script — no library edits required, no revert needed.

## Chunked dense fill: memory model (as of #318)

The bspline chunked dense-fill path (`_compute_Z_dense_chunked`) builds
Z in observer-row chunks bounded by `swept_mem_mb`, rather than
materializing the full `(d+1, d+1, N, N)` moment tensor (#136). As of
#318 its peak RSS decomposes into three buckets:

- **Z itself** — `16 N²` bytes (complex128 `(N, N)`), factored in
  place at solve time (`overwrite_a=True`, F-order — #136), so it's
  never copied.
- **Chunk transients** — bounded by the `swept_mem_mb` budget (default
  256 MB): one observer-row window of the all-pairs moment tensor plus
  its same-edge correction blocks, dead again as soon as the window's
  contribution has been accumulated into Z.
- **O(N) tables** — the `(N, 3)` tangent table, per-segment radii, and
  the support-segment / polynomial arrays. Before #318 this bucket
  also carried an `(N, N)` tangent-dot matrix (`td_all`, 8 bytes/entry
  — exactly half the size of Z) that stayed alive for the whole fill.
  The windowed C++ assembler (`assemble_Z_bspline_windowed`) now takes
  the `(N, 3)` tangent table directly and forms each pair's dot
  product inline, so no N²-scale table survives the fill besides Z.

Also audited in the same arc: the `_accumulate` closure used to wrap
every incoming moment window in `np.ascontiguousarray(J_win)` "just in
case." Every production path (C++ reduced kernel, C++ EK twin, numpy
einsum fallback, the mixed-radius `concatenate`, and the same-edge
`(A_st + A_reg) - J_edge` difference) already hands it a C-contiguous
complex128 array, so the copy was a no-op on every call. It's been
replaced by a contract assert that vanishes under `-O`; the pybind
`c_style | forcecast` cast on the accelerator call stays as the real
safety net.

### Certified numbers

Single dense solve, `arrays.bowtiearray2x4`, bs2, free space, 28.57 MHz,
8,320 segments — fresh-subprocess peak RSS via
`scripts/bench_converge.py::run_engine` (xps13, 16 GB):

| | peak RSS | solve time |
|---|---|---|
| before (momwire main @ 9c41f8d) | 2,382 MB | 34.9 s |
| after (#318, run 1) | 1,827.9 MB | 25.9 s |
| after (#318, run 2) | 1,827.8 MB | 27.6 s |

`Z[0]` agrees bit-for-bit with the before-value
(`212.34285313452253 + 36.66494828784576j`) — moving the dot product
into the windowed assembler introduced no numerical drift at this
scale. The ~554 MB drop matches the predicted `td_all` footprint
(`8 · 8320² ≈ 554 MB`) almost exactly. External ladder: antennaknobs
`scratch/bs2-memory-ladder.json`, which swept peak RSS vs N across
this same design and first identified `td_all` as the largest
avoidable term.
