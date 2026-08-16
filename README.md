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

## Ports

Every solver family exposes the same port surface, and it is the surface to
build on: `compute_port_solution()` runs **one** fill and **one**
factorisation over every port at once, so a multi-port structure costs what a
single-port one costs.

Three kinds of port, declared at construction, and they can be mixed:

| kwarg | what it is | entry |
| --- | --- | --- |
| `feeds=` | a delta gap at a point along a wire — NEC's `EX 0` | `(wire_index, arclength, voltage)` |
| `junction_ports=` | a shunt port on a junction NODE's KCL row: the node drives net inflow (the node's row leaves the constraint set) | `(junction_index, voltage)`, or a bare index for 0 V |
| `node_gaps=` | a SERIES EMF at a junction node, in series with one named wire end — the apex feed | `(wire_index, "start"\|"end", voltage)` |

Ports are numbered `[feeds…, junction_ports…, node_gaps…]`, and that is the
order every port readout is in.

```python
sol = solver.compute_port_solution()      # one fill, one factorisation
sol.y                                     # (n_ports, n_ports) short-circuit Y
sol.coeffs                                # (n_dof, n_ports) — column j is the
                                          #   solution for 1 V at port j
sol.port_currents                         # the same matrix as `y`, asserted
```

`compute_y_matrix()` is `compute_port_solution().y`, so the two cannot drift.
Any other excitation is `coeffs @ V` with no second fill — that is the point
of the class. To turn a column into currents on the structure, use the
solver's `currents_at_knots(coeffs[:, j])`, or `element_currents(coeffs[:, j],
subdiv=…)` for the `(mid, moment, nodes, delta)` source terms a field
evaluator wants. `PortSolution.basis` is an opaque per-solve handle — do not
introspect it.

## Decks

`momwire.deck` reads a NEC-2 deck and puts it on a solver. The dialect —
which cards run, which are refused and in exactly what words — is specified
at
**[momwire.dev/reference/deck-grammar-nec2/](https://momwire.dev/reference/deck-grammar-nec2/)**;
that page is normative, and the code is tested against its anchors.

```python
from momwire.deck import build_solver, parse

deck = """CM 20 m dipole, 10 m up over average ground
CE
GW 1 21 -5.05 0. 10. 5.05 0. 10. 1.E-3
GE 1
GN 2 0 0 0 13. 0.005
EX 0 1 11 0 1. 0.
FR 0 1 0 0 14.1
XQ
EN
"""
built = build_solver(parse(deck), basis="bspline")
y = built.solver.compute_port_solution().y
port = built.ports.feed_ports[0]          # the solver row this EX card drives
print(f"Z = {1.0 / y[port, port]:.1f} ohm")   # Z = 67.0-41.1j ohm
```

`parse()` returns a dialect-neutral `DeckModel`; `build_solver()` maps it onto
one of the seven `BASES` names (five solver families — `"bspline"` is the
default, the degree-2 B-spline) and returns the solver together with a
`PortPlan`. The plan is what makes the solver's ports readable: which row is
which `EX` or `LD` card, each load's `LoadSpec`, and one drive vector per
execute group over a port set that never changes. Stamping a load impedance
is port algebra and stays with the consumer — `momwire.deck` puts the gap in
the matrix and hands over the spec.

A deck's **execute groups** are what the model says about running it: one per
execute card, each carrying its own frequency list, kernel flag and
`Environment` (the ground, its plane and a cliff's second medium). A `GN` card
arms, so a deck may run once in free space and once over ground and each group
says which; `build_solver(model, group=k)` builds over group `k`'s, and
`frequency_mhz=`, `extended_kernel=` and `environment=` override it.

Only the operating point moves between those calls, never the geometry, so a
swept caller translates once:

```python
from momwire.deck import prepare_mesh

mesh = prepare_mesh(model)                       # the polylines, the port plan
solvers = [build_solver(model, mesh=mesh, frequency_mhz=f) for f in sweep]
```

Every solver built from one handle is given the same coordinate arrays, so a
prepared solve is bit-equal to an unprepared one.

## SimNEC portal

Installing momwire puts **`momwire-nec2c`** on your path — a resident NEC
engine speaking the protocol [SimNEC](https://ae6ty.com/smith_charts/) uses
to drive `nec2c`, with momwire's solver behind it. Point SimNEC's NEC portal
dialog at that command and its Smith chart, tuner and sweeps run on momwire.
`python -m momwire.portal` is the long spelling, and `--selftest` is the
deployment smoke.

```bash
momwire-nec2c -version                            # NEC2momwire.<major>.<minor>
momwire-nec2c --selftest                          # PASS / FAIL, no checkout needed
momwire-nec2c --basis sinusoidal < dipole.nec     # or run a deck by hand
```

Setup, the two filename rules SimNEC enforces, `--basis`, the caching flags
and what refusals look like:
**[momwire.dev/reference/portal-usage/](https://momwire.dev/reference/portal-usage/)**.

`momwire.portal` may use the solver API and `momwire.deck`; nothing else in
momwire may import from it. The SimNEC protocol is the portal's business
alone, and a test enforces that.

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
