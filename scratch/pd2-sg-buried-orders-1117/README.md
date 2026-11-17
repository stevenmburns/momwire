# PD2-measure: SG-mp's buried pair orders against bspline's #1752 ladder (11-17)

Measurement only, on the scratch branch `measure-sg-buried-orders-1117`. Not a
merge candidate. The branch's one src commit is a clearly marked MEASUREMENT
PATCH. It adds env overrides that Python and the C++ fill core read the same
way. With the env unset, the shipped constants hold, so no default moves.

## Question

SG-mp pins its crossing deck's below-class pair order at 32, with ladder
`((2, 8), (16, 4))`. These are bspline's values from before #1752
(`_sinusoidal_galerkin_fill_core.cpp` `buried_ladder` / `BURIED_N_QP_PAIR`;
`sinusoidal_galerkin.py` `_assemble_Z_mp_mixed` locals). bspline moved to 128
with `((0.1, 32), (1.5, 16), (2, 8), (4, 4))` after measuring a 1.18 ohm error
at q = 32 on fx_slant_base_source. Does SG have a bs2-class error, about 1 ohm
or at least well above the 1e-9 drift class?

The run turned up a second fact. A **wholly-buried** deck goes through SG's
single-medium `_assemble_Z_mp` at k_m, and that path uses `DEFAULT_N_QP_PAIR`
= 8 with `((16, 4),)`, not the buried constants. The C++ twin is
`assemble_Z_buried`, which uses the default `WinOpts`. Ten of the 15 decks take
this route. So the patch covers it too.

## Method

- **Decks:** the #1752 15-deck set, copied from the perf-1028 harness on
  Rocketlake (`decks/`). Each is a repo fixture (fx_*), an antennaknobs catalog
  export (ak_*, MIT), or a perf/S-D synthetic deck.
- **Python:** the production route. `momwire.dropin.nec5._shell.render(text,
  basis="sinusoidal-galerkin")` runs in-process, which is what
  `momwire-nec5-sinusoidal-galerkin` serves. It runs in-process because a
  resident `--serve` daemon would not see a per-run env. The solver class is
  instrumented to capture Y, time `_assemble_Z`, and record which fill route
  ran. Z = inv(Y)[0, 0], cross-checked against the printout's five figures.
- **C++:** `_accelerators.sinusoidal_galerkin_fill_dump`, the
  `scripts/core_split` comparator entry, which does the front, fill and port
  solve. It records its operating points through the NEC-4.2 build, so its Z
  is a different operating point from the NEC-5 launcher's (see "Side
  findings"). It serves 10 of the 15 decks. The 4 NEC-5-syntax decks record no
  point, and mixed_vert_over_buried8 is declined: the transmitted blocks are
  not copied (momwire#1718 S12).
- **bs2:** `render(text, basis="bspline")` at the shipped settings. It
  reproduces #1752's converged slant value, 86.622-18.923j.
- **Settings**, as `<base>|<ladder>` on both routes:
  - `cur`: shipped. Crossing 32 | 2:8,16:4; wholly-buried 8 | 16:4.
  - `q4`: `4|`, a liveness probe. It must move Z.
  - `q32`: wholly-buried 32 | 2:8,16:4.
  - `q64` / `q128`: the base raised, SG's ladder kept.
  - `bsl`: bspline's 128 | 0.1:32,1.5:16,2:8,4:4.
  - `ref`: 256 | 2:32,16:16. The C++ version is `ref_cxx`, 192 | 2:32,16:16,
    because its Gauss table stops at 192.
- **Mesh reference:** every GW segment count tripled (m = 3), SG at `bsl`, and
  bs2 at m = 1 and m = 3.
- **Box:** one subprocess per run on Skylake, at 4 threads, while other jobs
  were also running. **Timings are single runs and indicative only.**

## Result

`summary.md` (all in ohms) holds:

- the largest |dZ| against the quadrature reference over cur/q32/q64/q128/bsl;
- the cur→bsl move;
- the q4 liveness move;
- SG's mesh error |Z(m1) − Z(m3)|;
- |SG(m3) − bs2(m3)|.

`table.md` has every deck × setting × side, with dZ against the reference,
dZ against cur, and the time ratio.

- **Quadrature error at SG's shipped orders: at most 1.4e-8 ohm on every deck,
  on both routes, in Python and C++.** On fx_slant_base_source, the deck where
  bs2 was 1.18 ohm off at q = 32, SG at 32 sits 7e-13 ohm from its q = 256
  reference. q64 and q128 do not change it at 1e-12. bspline's ladder moves it
  5.7e-9, which is the H2 order-4 rung at [4, 16) lengths, i.e. drift class.
- **The knob is live.** Starving the base order to 4 moves Z by 7e-7 to 7e-6
  ohm on all 13 decks that take a patched route. Python and C++ move by the
  same amount: on slant, cur and bsl are bit-identical between the two, and
  q4 moves 1.87e-6 against 1.93e-6.
- **Two decks take neither patched route.** These are
  ak_elevated_buried_cpoise and mixed_vert_over_buried8 (the direct-form /
  mixed fill, with route instrumentation empty). Every setting gives them
  bit-identical Z. That is expected, and it is not evidence either way.
- **Discretisation dominates by 7 to 10 orders of magnitude.** SG's own mesh
  error at m = 1 runs from 0.07 ohm (ak r4) to 8.6 ohm (slant), 15 to 22 ohm
  on the hubs, and 1e2 to 2e3 ohm on the high-reactance decks. On slant,
  SG(m3) and bs2(m3) still differ by 4.7 ohm.
- **Cost, indicative, C++ total time against warm q32:**
  - `bsl` 1.0 to 1.5x;
  - `q128` 1.2 to 1.9x;
  - `ref` 2 to 5x.

  Python's first run per deck (`cur`) is cold, so compare its ratios to `q32`,
  not to `cur`.

## Recommendation

**SG shows no bs2-class error. Do not follow (no PD2 proper, no PB1 on this
account).** Keep SG's constants, on both routes, and record the measurement in
the docstring. Suggested text for the `_assemble_Z_mp_mixed` locals and the C++
`buried_ladder` comment:

> Measured 11-17 (PD2-measure, scratch branch measure-sg-buried-orders-1117)
> on the #1752 15-deck set: against q = 256 (every tier raised), this fill's
> 32 / ((2, 8), (16, 4)) is within 1.4e-8 ohm on every deck, fx_slant_base_source
> 7e-13, where bs2 at 32 was 1.18 ohm off; bspline's #1752 ladder moves Z
> <= 9e-9 ohm at 1.0-1.5x the time. SG's mesh error (0.07-22 ohm on the same
> decks) dominates, so these orders stay.

The wholly-buried path (8 / ((16, 4),)) is in the same state, with at most
7.3e-9 ohm against the reference. A one-line note there would stop the next
audit from reading "8, not 128" as a gap.

*Why SG is immune* is not established. A plausible reading is that SG's own
near-pair override (`_sinusoidal_mp.near_pair_moments`: self, collinear
neighbours, touching pairs) serves the near-contact pairs that bs2's base
order had to carry. That fits a q4 move of only ~1e-6. It was not traced
pair by pair.

## Side findings (not acted on)

- The C++ dump's operating point is not the shipped launcher's. Examples:
  - buried_hub32_crossing: 82.10+31.23j (dump) against 84.83+49.95j
    (launcher);
  - fx_ab_long_wire: 780-160j against 467-70j.

  The dump records its kwargs through the NEC-4.2 build, so it checks the C++
  core against Python on that front, not on the NEC-5 launcher's. Worth
  knowing if the orphaned SG C++ core is moved.
- Python against the C++ dump on the same front: 5 of the 10 served decks
  (buried_dipole401, the hubs, buried_hub8_crossing) are not bit-identical,
  at 4e-11 to 6e-11 relative in Z. All three settings tried (cur/q4/bsl) give
  the same 5. At `cur` the patch is value-identical to the shipped constants
  by construction. This was not re-checked against an unpatched base build.

## Files

- `sg_buried_orders.py`: `worker` / `drive` / `table` / `summary`.
- `decks/`: the 15 decks.
- `results.jsonl`: one row per run.
- `table.md`, `summary.md`: rendered from results.
- `drive.log`: the run log.
