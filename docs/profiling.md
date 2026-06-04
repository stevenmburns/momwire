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
