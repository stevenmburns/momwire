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

### `# noqa: E402` on the `sys.path` idiom — dead only in the INLINED form

Ruff's E402 exempts an import only when *every* statement above it is itself
exempt: imports, the docstring, comments, dunder assignments, and `sys.path`
mutations. The mutation does **not** confer the exemption on what follows — it
is simply one more exempt statement. One ordinary assignment anywhere above the
import ends it, and its position relative to the mutation is irrelevant.
Measured 2026-09-01, identical under `ruff@0.15.21` and `ruff@0.16.5`:

| shape | E402 |
|---|---|
| `sys.path.insert(0, str(Path(__file__)...))` then the import | exempt |
| `HERE = Path(__file__).resolve().parent` **before** the mutation | **fires** |
| the same assignment **after** the mutation, above the import | **fires** |
| `__version__ = "1"` (dunder), then the mutation | exempt |
| `sys.path.append(...)` | exempt |
| ordinary `X = 1` then the import | fires (the case the rule is for) |

So only the fully inlined head-of-file shape needs no directive:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from momwire import BSplineSolver  # exempt — no E402, no noqa needed
```

**The two repos sit on opposite sides of that line, so an audit does not
transfer.** antennaknobs has 320 directives and *zero* dead, because its probes
bind `HERE = Path(__file__).resolve().parent` first and that ends the exemption.
Here the scripts inline the expression, so the directive really is a no-op — but
only in those files. See antennaknobs' CLAUDE.md for that side of it.

**Do not sweep `noqa: E402` by pattern.** Most are load-bearing: of the 35 lines
carrying one (measured 2026-09-01), only 14 are dead, and the two sets do not
overlap at all — deleting the other 21 turns the gate red. Those 21 are live for
a third reason again: they are deep mid-file imports
(`tests/test_extended_kernel_bspline.py:657`,
`tests/test_sinusoidal_galerkin.py:1241`) sitting after hundreds of lines of
ordinary code, which is precisely what E402 exists to flag.

**Any inventory of dead directives goes stale faster than it gets cleaned.**
momwire#761 was filed against 22 (16 removable, 6 `non-enabled` keepers). While
*this* note sat in review, #760/#762/#769 landed and took the tree to **24/17/7**
— `tests/test_sign_timestamp_755.py:36` is a 14th dead E402 and
`scripts/probe_quadrature_defaults.py:327` a 7th keeper. Hours, not weeks. Read
the issue's file:line lists as of-a-date and re-measure before acting on one.

A dead directive is not always this idiom, either. `tests/test_harrington.py`
has no `sys.path` call anywhere; its import is preceded only by a docstring and
comments, so nothing triggers E402 there in the first place. Decide each site
from the lines *above* the import, not from the directive on it.

The **file-level** form is legitimate where it is earned:
`validation/momwire_backend.py:99` carries `# ruff: noqa: E402` for six imports
that must follow env-var setup, and RUF100 correctly never reports it.

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
and has never been re-derived. It under-integrates *cross-edge* pairs, which
makes results depend on how a *straight* wire was subdivided. See momwire#743
for measurements. (It is near-exact, not exact, for same-edge pairs — see the
accuracy note below.)

**There are two knobs and only one of them still has a ceiling.** Getting that
backwards is how the workaround this section used to document outlived the
refusal it was for.

| knob | pairs | ceiling | over it |
|---|---|---|---|
| `n_qp_pair` | cross-edge | **none** — `BSPLINE_MAX_N_QP` is literally `static_cast<size_t>(-1)`, so the C++ guard is always true (momwire#762 tiled the six off-edge kernels) | runs *accelerated*, silent |
| `n_qp_pair_same_edge` | same-edge reg kernel, which #762 did **not** tile | `SAME_EDGE_MAX_N_QP` = **8** | `serves_n_qp()` **routes to numpy** with a `RuntimeWarning` naming the ceiling — correct, slower, cost grows as n_qp² |

So **nothing refuses any more, on either knob.** The old `sys.meta_path`
extension-blocker documented here was a workaround for a `RuntimeError: n_qp
too large`, and momwire#769 replaced that raise with the fallback above. Its
tests are `tests/test_accel_fallback.py`. Re-measured on `origin/main` @
`9eda56f`, the accelerator loaded, on that file's own 12-segment decks — every
order from 4 to 32 solves on both knobs and both decks:

```
n_qp_pair            straight: 4..32  all Z = 70.19390 -18.58923j  (silent)
                     bent:        4       Z = 31.33852-134.21518j
                                  8..32   Z = 31.33854-134.21523j

n_qp_pair_same_edge  straight:    8       Z = 70.19398 -18.58811j  accelerated
                                  9       Z = 70.19399 -18.58802j  numpy, warns
                                 32       Z = 70.19401 -18.58771j  numpy, warns
```

**The same-edge drift above is not accuracy to be bought.** That knob is the
*memory* cost — one edge's `(N_e*n_qp, N_e*n_qp)` R table, quadratic in both —
and it buys nothing: every pair on an unsplit edge is same-edge, so the answer
must not depend on it. It moves in the 6th digit here only because the same-edge
*smooth-kernel* piece does read it, which on a coarse mesh shows; it is
bit-identical by N=400. `test_n_qp_pair_same_edge_is_the_memory_knob_and_moves_nothing`
holds that at a 1e-2 span. Raise the **cross-edge** knob for accuracy.

**Measure the cross-edge knob on a BENT deck.** Since momwire#759 a straight
wire has one edge and never enters the off-edge kernel at all, so `n_qp_pair`
moves nothing there and a straight-wire sweep converges *vacuously* — the
identical straight column above is that artefact, not evidence. The bent deck
is where this section's first paragraph is visible: q=4 differs from the
converged value in the 7th digit.

To take the pure-Python path deliberately, do what `test_accel_fallback.py`
does — monkeypatch the `_HAVE_*_ACCEL` flags in `momwire._bspline_kernels` —
or just ask for a same-edge order above 8. The two paths agree to ~1e-9 at the
ceiling, so numpy is a sound oracle. On a deck this small the fallback is cheap
(single-shot: 0.005 s at q=8, 0.020 s at q=32), so **do not quote those
figures as the cost** — the cliff `serves_n_qp` warns about needs a deck big
enough for the same-edge fill to matter.
