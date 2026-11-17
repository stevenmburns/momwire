# PD2-measure: SG-mp's buried pair orders against bspline's #1752 ladder (11-17)

Measurement only, on the scratch branch `measure-sg-buried-orders-1117`. Not a
merge candidate. The branch's src commits are clearly marked MEASUREMENT
PATCHes. They add env overrides that Python and the C++ fill core read the
same way. With the env unset, the shipped constants hold, and an unpatched
build of the merge base gives bit-identical Z (see "Identity").

## Verdict in one paragraph

**bspline's #1752 far-pair ladder is inert for SG** (it moves Z by at most
about 1e-8 ohm on every deck), so SG need not follow it. **But SG is not free
of a bs2-class quadrature error.** On fx_slant_base_source, the deck where bs2
at q = 32 was 1.18 ohm off, SG's own touching-pair GRADED rule
(`_sinusoidal_mp.GRADED_N_PER_PANEL = 8`, the C++ twin's constant of the same
name) leaves 0.43 ohm at the shipped mesh. That rule is the one SG uses for
the pairs that bs2's base order carried. It is a separate question from the
#1752 ladder, and it is listed for Steve below. The first version of this
README recommended a docstring that said "these orders stay, within 1.4e-8
ohm on every deck". **That claim was false and is withdrawn.** It came from
varying only the far-pair ladder.

## Question

SG-mp pins its crossing deck's below-class pair order at 32, with ladder
`((2, 8), (16, 4))`. These are bspline's values from before #1752
(`_sinusoidal_galerkin_fill_core.cpp` `buried_ladder` / `BURIED_N_QP_PAIR`;
`sinusoidal_galerkin.py` `_assemble_Z_mp_mixed` locals). bspline moved to 128
with `((0.1, 32), (1.5, 16), (2, 8), (4, 4))` after measuring a 1.18 ohm error
at q = 32 on fx_slant_base_source. Does SG have a bs2-class error, about 1 ohm
or at least well above the 1e-9 drift class?

A **wholly-buried** deck goes through SG's single-medium `_assemble_Z_mp` at
k_m. That path uses `DEFAULT_N_QP_PAIR` = 8 with `((16, 4),)`, not the buried
constants. The C++ twin is `assemble_Z_buried`, which uses the default
`WinOpts`. Ten of the 15 decks take this route, so the patch covers it too.

## SG has two quadrature regimes. Only one of them was varied at first

- **Far pairs (ratio >= 2.0): the WindowFill ladder.** These pairs are what
  the #1752 constants govern.
- **Near pairs (ratio < `NEAR_RATIO` = 2.0): `near_pair_moments`.**
  `WindowFill.accumulate` overwrites every one of them
  (`J[:, :, ii, jj] = Jn`). They take three rules of their own:
  - the graded composite rule (`GRADED_N_PER_PANEL` = 8) on a touching
    non-parallel pair;
  - the product rule (`NEAR_N_QP` = 32) on a near non-touching pair;
  - the parallel reduction (`PARALLEL_N_T` / `_N_XI` = 24 / 8).

`NEAR_RATIO` equals the ladder's first threshold (2.0). So **every pair that
would take the ladder's BASE order is a near pair, and gets overwritten**. The
base-order knob (q32 / q64 / q128 / the base of `ref`) is therefore
structurally dead. The results agree: those settings move Z by exactly 0.0.
The first run's q4 probe moved Z only because `4|` also deleted the far
tiers. That proved the env reached the kernel, but it did not exercise the
regime that matters for near-contact pairs. In bs2 that regime is the base
order. In SG it is `near_pair_moments`.

## Method

- **Decks:** the #1752 15-deck set, copied from the perf-1028 harness
  (`decks/`). Each is a repo fixture (fx_*), an antennaknobs catalog export
  (ak_*, MIT), or a perf/S-D synthetic deck.
- **Python:** the production route. `momwire.dropin.nec5._shell.render(text,
  basis="sinusoidal-galerkin")` runs in-process, which is what
  `momwire-nec5-sinusoidal-galerkin` serves. The solver class is
  instrumented to:
  - capture Y;
  - time `_assemble_Z`;
  - record the fill route;
  - record the near rules in force;
  - count the graded-rule calls (`graded calls` in the summary, 0 = the deck
    has no touching non-parallel pair).

  Z = inv(Y)[0, 0], cross-checked against the printout's five figures.
- **C++:** `_accelerators.sinusoidal_galerkin_fill_dump`, the
  `scripts/core_split` comparator entry, which runs the front, the fill and
  the port solve. It serves 10 of the 15 decks. It records its operating
  points through the NEC-4.2 build, so its Z is a different operating point
  from the NEC-5 launcher's, but its MOVES under each knob are comparable.
- **bs2:** `render(text, basis="bspline")` at the shipped settings. It
  reproduces #1752's converged slant value, 86.622-18.923j.
- **Env knobs** (the measurement patches):
  - `MOMWIRE_MEASURE_SG_CROSSING_BELOW` / `_WHOLLY_BURIED` =
    `<base>|<ladder>`: the far regime.
  - `MOMWIRE_MEASURE_SG_NEAR` = `<graded>|<near_n_qp>|<n_t>|<n_xi>`: the
    near regime. Python binds it at import, before `parallel_pair_moments`
    takes its defaults; C++ reads it at load.
- **Settings:**
  - `cur`: shipped, plus `rep2` / `rep3`, which are repeats of it.
  - Far regime: `q4` (liveness), `q32`, `q64`, `q128`, `bsl` (bspline's
    ladder), and `ref` (256 | 2:32,16:16, near rules shipped).
  - Near regime: `g2` (liveness), `g16`, `g24` and `g48` (graded), `nq96`
    (product), and `par3` (parallel 72/24; Python only, because the C++ dump
    has no route that takes it separately).
  - `qref`: the quadrature reference, with the far tiers at `ref` and every
    near rule at 6x (48|192|144|48). C++ uses `qref_cxx` instead, with far
    192 and near 3x (24|96|72|24). The C++ two-rule batch declines a rule
    wider than 256 nodes, which graded 48 exceeds, so `g48` declines there
    (recorded in the table).
- **Mesh:**
  - every deck at m = 3 (every GW segment count tripled), SG at
    `cur` / `bsl` / `qref`, and bs2 at m = 1 and m = 3;
  - slant also at m = 5 and m = 9.
- **Box:** Skylake, one subprocess per run at 4 threads, with the box shared
  (load 5-10). **Timings are single runs and indicative only.** Decks with
  zero graded calls still show g16 "costs" from 0.9x to 1.7x, which is the
  noise on those numbers.

## Result

`summary.md` has one row per deck. `table.md` has every deck × setting ×
side. `ladder.md` is slant against the mesh.

| regime | largest move over the 15 decks (py, vs `cur`) | where |
|---|---|---|
| far: base order (q32/q64/q128) | 0.0 exactly | dead by construction |
| far: bspline's ladder (`bsl`) | 9.2e-9 ohm | drift class |
| far: every tier raised (`ref`) | 1.4e-8 ohm | drift class |
| near: product rule (`nq96`) | 8.8e-4 ohm | slant |
| near: parallel (`par3`) | 2e-13 ohm | — |
| **near: graded (`g48`)** | **0.42 ohm** | **slant** |
| near: graded (`g48`), next largest | 8.6e-5 ohm | buried_hub32_crossing |
| near: graded (`g48`), every other deck | <= 1.3e-8 ohm | — |

- **Total quadrature error at the shipped rules, |cur − qref|:**
  - slant: 0.43 ohm in Python, 0.42 ohm in C++ (against its 3x reference);
  - buried_hub32_crossing: 8.6e-5 ohm in Python, 1.0e-4 ohm in C++;
  - every other deck: <= 1e-8 ohm.

  Python and C++ agree on every move they both serve. On slant, g24 moves
  Z by 0.4216 ohm on both sides.
- **The graded rule converges.** On slant at m = 1, |Z − qref| is:

  | rule | \|Z − qref\| (ohm) |
  |---|---|
  | g8 (shipped) | 0.43 |
  | g16 | 3.0e-2 |
  | g24 | 3.5e-3 |
  | g48 | 8.7e-4 |

  The near product rule contributes the remaining 8.8e-4. That is
  consistent with an independent reviewer probe that monkeypatched the same
  constants (0.4216 ohm at graded 24).
- **The liveness probes both fire.** `q4` moves Z by 7e-7 to 7e-6 ohm on the
  13 decks that take a patched route. `g2` moves Z by up to 6.8 ohm (slant)
  on the decks with graded calls, and by 0.0 on the decks without them.
  - ak_elevated_buried_cpoise and mixed_vert_over_buried8 take neither
    patched route and make no graded call: every setting is bit-identical
    there. That is expected, and it is not evidence either way.
- **Slant, the near error against the mesh** (`ladder.md`):

  | m | \|cur − qref\| | SG qref − bs2 (same m) |
  |---|---|---|
  | 1 | 0.43 | 13.0 |
  | 3 | 0.12 | 4.8 |
  | 5 | 2.2e-2 | 2.9 |
  | 9 | 1.5e-3 | 1.6 |

  The graded error shrinks faster than the mesh error. At the shipped mesh
  it is about 3 % of SG's distance to bs2, not "7 to 10 orders" below it,
  as the first version said. It is not the cause of the 13 ohm reactance gap
  at m = 1: that is mesh error, and SG and bs2 approach each other as m
  grows (1.6 ohm at m = 9). But it is a real 0.4 ohm on a deck at its
  shipped segmentation.
- **Reproducibility.** On this build, `cur`, `rep2` and `rep3` are
  bit-identical on every deck, on both sides (py noise / cxx noise = 0 in
  `summary.md`). An unpatched base build also matches bit-for-bit (see
  "Identity"). The reviewer's rerun of ak_buried_dipole landed 1.5e-8 ohm
  from the recorded row, but it came from another clone, and that
  difference is not explained. Until it is, a 1e-8 delta between two
  setups is not resolvable, and it is the same size as the far-regime
  deltas above. Those deltas say "drift class", and nothing finer. **Do
  not put their digits in a docstring as measured bounds.**
- **Cost, indicative** (fill time against warm q32, single runs, shared box):

  | rule | py | C++ |
  |---|---|---|
  | g24 | 1.4x (slant), 2.0x (r4), 2.2x (hub16), 2.8x (hub32) | 1.6-2.2x |
  | g48 | 2.0-8.5x | declined (256-node cap) |
  | `bsl` | 1.0-1.5x | 1.0-1.5x |
  | `qref` | 8-30x | 4-9x |

## Identity (env unset, against an unpatched build)

`identity` runs the worker on a fresh clone and build of the merge base
(4ec7912d, origin/main at branch time) on the same box. Every override is
unset. It compares each deck's Z on both sides with this branch's `cur`
row. The result is in `identity.md`.

**The result: bit-identical (`==` on the complex Z) on all 15 decks in
Python and on all 10 the C++ dump serves.** So the patches are inert when
the env is unset, and two independent builds on this box agree to the last
bit. The reviewer's 1.5e-8 rerun therefore did not come from run-to-run noise
within a build. Its cause (the other clone's build, or its thread count) was
not traced.

The targeted SG tests pass on both base and branch, with the same (empty)
FAILED set: `test_sinusoidal_galerkin_fill_core.py`,
`test_sinusoidal_mp_1354.py` and `test_sinusoidal_mp_ek_1362.py`, 172
passed each.

## Recommendation

1. **Far regime: keep SG's constants.** Do not follow bspline's #1752
   ladder (no PB1/PD2 on that account). It moves Z only in the drift class.
   A docstring note may say that bspline's #1752 ladder was measured inert
   for SG (11-17, this branch). It should not quote 1e-8 digits as bounds.
2. **Near regime: SG has a bs2-class error, and it is in the graded
   touching-pair rule, not the ladder.** 0.43 ohm on slant at the shipped
   mesh, 8.6e-5 on hub32, drift class elsewhere. That is PD2-proper
   redirected. Whether to raise `GRADED_N_PER_PANEL` (Python and the C++
   twin together), or to make it adaptive, moves default-path numbers, so it
   is Steve's call. g24 brings slant to 3.5e-3 of the reference at
   indicatively 1.4-2.8x fill time.
   - Why slant needs it was not traced pair by pair. It has 2 graded
     groups. A plausible reading is the acute angle between the slant wire
     and the wire it touches, which the doubling panels toward the shared
     node resolve poorly at 8 nodes. That is a hypothesis.

## Side findings (not acted on)

- The C++ dump's operating point is not the shipped launcher's. For example,
  buried_hub32_crossing gives 82.10+31.23j through the dump against
  84.83+49.95j through the launcher. The dump checks the C++ core against
  Python on the NEC-4.2 front. Worth knowing when the orphaned SG C++ core is
  moved.
- Python against the C++ dump on the same front: 5 of the 10 served decks
  differ at 4e-11 to 6e-11 relative in Z, under every setting tried.
- The C++ two-rule batch caps a near rule at 256 nodes
  (`_sinusoidal_galerkin_fill_core.cpp`: "a near-pair rule wider than the
  einsum replay is measured for"). A graded rule above about 24 per panel
  would hit it on these decks.

## Files

- `sg_buried_orders.py`: `worker`, `drive`, `table`, `summary`, `ladder`,
  `identity`.
- `decks/`: the 15 decks.
- `results.jsonl`: one row per run. The first run's rows are kept (the env
  was unset; same values, as `rep2` / `rep3` show).
- `table.md`, `summary.md`, `ladder.md`, `identity.md`: rendered from the
  results.
- `drive.log`, `drive2.log`, `drive3.log`: the run logs.
