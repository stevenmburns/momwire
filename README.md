# momwire

A pure-Python method-of-moments antenna simulator with optional C++ accelerators (pybind11).

Extracted from [antenna_designer](https://github.com/stevenmburns/antenna_designer).

## Solvers

`BSplineSolver` (degree-d Galerkin, default d=1/2) is the default solver;
`HMatrixSolver` and `ArrayBlockSolver` are structural accelerators built on
top of it. `SinusoidalSolver` and `SinusoidalGalerkinSolver` reproduce NEC2's
three-term basis (collocation and Galerkin testing respectively) as
in-codebase NEC comparators. `RazorSolver` is the NEC-5 formulation twin: a
tent basis with razor-blade (RWG mixed-potential path) testing, transcribed
from the NEC-5 Users Manual rather than NEC2's, free space only. Because its
testing rule — not its basis, which it shares with `BSplineSolver(degree=1)`
— is NEC-5's own, it reproduces NEC-5's characteristic slow O(1/N) impedance
walk without needing the (licensed) NEC-5 binary; see
[docs/razor-solver.md](docs/razor-solver.md) (momwire#309).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

**macOS:** the C++ accelerator uses Homebrew's OpenMP runtime, so `brew install
libomp` is required — both to compile from source and to run the prebuilt
Apple-Silicon wheel. The wheel deliberately does **not** bundle `libomp` (it links
Homebrew's by absolute path) so that it shares a single OpenMP runtime with
pynec-accel; two private copies in one process abort with `OMP: Error #15` (or
deadlock). Without `libomp` installed, the accelerator can't load and momwire
warns and falls back to the slower pure-Python path. On Linux the system
`libgomp` covers this, so no extra step is needed.

## Test

```bash
pip install -e ".[test]"   # core + test deps (pytest, matplotlib, scikit-rf)
pytest tests/
```

The cross-validation against NEC2 (`tests/test_pynec_backend.py`) additionally
needs PyNEC — a test-only dependency installed separately from a wheel (see
below). Those tests skip cleanly when it isn't present; everything else runs
without it.

## Optional: PyNEC backend (test-only)

momwire can be cross-validated against NEC2 via [PyNEC](https://github.com/tmolteno/python-necpp) (the `tests/test_pynec_backend.py` suite); NEC2 also delivers ~5–10× faster single-frequency solves. PyNEC is a **test-flow-only** dependency — momwire's own solver never imports it.

### Install the PyNEC wheel

Install PyNEC from the [python-necpp fork](https://github.com/stevenmburns/python-necpp)'s release. The distribution is named `pynec-accel` (the import name stays `PyNEC`); the wheels are self-contained — OpenBLAS is vendored (via scipy-openblas32), so no system BLAS, SWIG, or build toolchain is needed — and cover Linux, Windows, and macOS (arm64) on CPython 3.10–3.14:

```bash
pip install pynec-accel --no-index \
    --find-links https://github.com/stevenmburns/python-necpp/releases/expanded_assets/v1.7.6
```

`--no-index` ensures pip takes the fork's wheel rather than upstream PyNEC on PyPI (which is broken on current Python and lacks the OpenBLAS/OpenMP work). On macOS the wheel shares Homebrew's libomp (`brew install libomp`) rather than vendoring its own, so it can coexist with momwire's accelerator in one process. After install, `from PyNEC import nec_context` works and the cross-validation tests run; without it they're skipped (momwire itself needs no PyNEC).

### Runtime thread pinning

The wheel links OpenBLAS and parallelises the NEC2 matrix fill with OpenMP. Pick thread counts up front:

```bash
export OMP_NUM_THREADS=$(nproc --all)   # PyNEC matrix fill
export OPENBLAS_NUM_THREADS=1           # muzzle numpy/scipy's idle pool
```

Pinning `OPENBLAS_NUM_THREADS=1` stops numpy/scipy from spinning up their own OpenBLAS thread pool that contends with PyNEC's threads on the same cores. On a 100-director Yagi (2142 segs) this is worth ~8% wall time at NP=4.
