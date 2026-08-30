# momwire — working notes for Claude Code

Scope: things that are easy to get wrong here and are not obvious from the
code. Deliberately not a codebase tour.

## Lanes: run CI's commands, not your own

`Makefile` mirrors `.github/workflows/ci.yml` lane for lane — `make lint`,
`make test`, `make pynec`, `make integration`, `make slow`, `make crossgate`,
`make memgate`, and `make gates` for all of them. `tests/test_makefile_lanes.py`
asserts the two cannot drift apart, so prefer `make <lane>` over hand-rolling a
pytest invocation.

That tripwire gates the lane **commands**. It does not read version strings in
comments or `echo` lines, which is how the `lint` guard came to advertise a
ruff version CI had already moved off (fixed in momwire#720).

## Linting: the contract

CI runs **two** gates: `ruff check` **and** `ruff format --check`. A green
`check` does not mean the lint job passes.

**`select` is pinned explicitly** in `[tool.ruff.lint]` (momwire#720). Ruff's
implicit default is not a constant — between 0.15.21 and 0.16.5 it grew from
59 enabled rules to 414, which is why an unpinned bump used to turn CI red on
unchanged code. With `select` written down, a bump changes how a rule
*behaves* but never *which* rules run.

So bumping ruff is now cheap: the pin in `ci.yml`, plus the version the
Makefile's `lint` guard advertises. It is **not** the repo-wide cleanup the
old comments implied.

**Formatting is not governed by `select`.** Re-run `ruff format` after a bump
and take the diff — 0.16 added Markdown to the formatter's scope, so a bump
reformats `.md` files no earlier ruff touched (324 files at 0.16.5 here vs
267 at 0.15.21).

The rule set here is smaller than antennaknobs': `extend-select = ["B023"]`
only, per-file-ignored under `scripts/` and `site/figures/`. antennaknobs also
carries `BLE001`, `TID252` and `S310`; do not assume a directive style that
works there passes here, or vice versa.

### `unfixable = ["F401", "RUF100"]` — leave both in

**RUF100's fix deletes the whole comment, prose included**, so
`# noqa: B023 — consumed in this iteration` loses its justification along
with its suppression. Worse, it reads a directive as dead whenever the rule
it names is not *currently* selected — so a `--fix` run taken while auditing
which rules to adopt deletes exactly the annotations that are the argument
for adopting them.

Not hypothetical: **22** directives in this tree are reported unused today
(they name rules this repo has not selected). Before RUF100 was listed here
they were all marked `[*] fixable`, i.e. one `--fix` from gone.

When auditing directives, measure with **`--extend-select RUF100`**, never
`--select RUF100` — the latter *replaces* the rule set, so every directive
naming a now-unselected rule reads as unused.

## Building the C++ extension

`make build`. setuptools has no per-object staleness check — it recompiles
*every* source in an extension when one is newer, which after the
`_accelerators.cpp` split (momwire#687) is five compiles per edit. `make build`
wires `ccache` through `BUILD_ENV`, which is what makes the split pay: a real
edit is ~12-28 s instead of ~48 s.

After moving the submodule pointer, assume the built extension is stale and
rebuild before trusting any local solve.

## Quadrature: `n_qp_pair` is not a converged default

`n_qp_pair=4` entered in the first bspline commit ("straight-wire first cut")
and has never been re-derived. It is exact for same-edge pairs and
under-integrates cross-edge ones, which makes results depend on how a
*straight* wire was subdivided. See momwire#743 for measurements.

To run `n_qp_pair > 8` (the C++ kernel refuses, on a compile-time stack-array
size), block the extension before importing momwire:

```python
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.endswith("momwire._accelerators"):
            raise ImportError("blocked")
        return None


sys.meta_path.insert(0, _Block())
```

Then assert `momwire._accel.LOADED is False`. The two paths agree to all
printed digits at `n_qp_pair=8`, so pure Python is a sound oracle — at ~4x the
memory and time per doubling of the order.
