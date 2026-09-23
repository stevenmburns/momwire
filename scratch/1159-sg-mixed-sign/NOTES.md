# momwire#1159 — SG on decks with wires on both sides of the interface

Three defects on `SinusoidalGalerkinSolver`'s mixed / buried route, plus the
wholly-buried port solution. All three share one shape: a spelling that is
right somewhere else, copied to a place where it is not.

## 1. The transmitted pair class had the wrong sign

**Where it lived.** `SG._assemble_mixed_contribs` placed the two transmitted
directions with `dest[...] -= reduced`.

**Which block (probe1).** At ε̃ = 1 the mixed deck is free space, so its
assembled G must equal the free-space G on the same wires quadrant by
quadrant. Pre-fix:

| quadrant | ‖G₁ − G_f‖/‖G_f‖ | ‖G₁ + G_f‖/‖G_f‖ |
|---|---|---|
| above × above | 0 | 2 |
| above × below | 2.000 | **8.05e-6** |
| below × above | 2.000 | **8.05e-6** |
| below × below | 4.6e-17 | 2 |

Both cross quadrants are exactly the negated free-space block, to the
transmitted grid's own accuracy (8e-6). Post-fix they read 8.05e-6 on the
other column.

**The algebra.** The projected tables `t̂_m · D · t̂_n` (above remainder,
below remainder, transmitted) are one family with one sign convention — bspline
passes all three through the same `_sub_field_galerkin`. What differs between
the trunks is how that family enters the matrix:

* bspline: `Z -= field_block` for every one of them, remainder included, with
  no second minus anywhere. Its Z carries the opposite global sign.
* SG: `G[i,j] = +∫ f_i ŝ·E_j`. The within-class remainder reaches G through
  `_fold_ground_block`, which computes `img ← c2·img − rem` and then
  `G ← G − img`, i.e. `G = free − c2·img + rem`: the field table enters with
  a **plus**. The D3 crossing block independently landed on the same sign
  (`G + t_ab + t_ab.T`, adjudicated by the node's KCL in #980 D3).

At ε̃ = 1 the transmitted table IS the direct field (the "whole field per unit
source moment"), so its correct contribution equals SG's own free-space block
for those pairs — the plus. The D2 fill copied bspline's minus.

**Why nothing saw it.** Negating both off-diagonal class blocks is
G → D·G·D with D = diag(I_above, −I_below). Driving one port in medium A,
v' = D v = ±v, the solution is D x, and the port current is unchanged, so every
single-port Z (every D2 gate, every one-feed deck, the ε̃ = 1 Z collapse) is
exactly invariant. Multi-port Y, both-ports-driven Z, the unfed medium's
current and so the far field all flip.

**Crossing (D3) decks were not affected**: their cross pair is
`_crossing_fill`'s, with no transmitted grid (the red-control seam is never
called on a crossing deck — counted in the test), and its Y collapses at ε̃ = 1
identically on main and branch (probe3: 3.9e-6 per entry).

## 2. A mixed deck's drive and port readout wrote buried shapes at air's k

Found while measuring item 1's consumers, not in the issue. A mixed deck is
solved outside `_operating_medium` (two k live), so `_drive_columns`,
`_node_cut_vectors` and `_feed_segment_current` are handed `self.k` = k_p and
wrote a buried entry's `{sin kξ, cos kξ − 1}` at it. Invisible at a segment
centre gap (both shapes vanish at ξ = 0), live at:

* a knot gap (`feed_xi` = ±h/2) — which is where probe5's mixed deck, razor
  #1149's `detached` deck and `detached_hub` all feed;
* the segment gap's `(2/k)(sin u − u)`;
* a node port's member-end values.

`_entry_k(seg_view, s, e, k)` returns the view's `k_entry` slice when present,
else `k` itself, so single-medium views keep their arithmetic.

Effect, probe2 on probe5's mixed deck, SG vs bspline Y11 at m = 1/2/4:
3.7e-2 / 2.0e-2 / 1.1e-2 before, 4.9e-4 / 7.2e-5 / 1.1e-5 after. This is the
residual Y11 gap #1156's notes recorded ("Y11 to 0.8 %" at m = 4 there).

## 3. The current readout rebuilt the basis in air

`currents_at_knots` / `current_slopes` (inherited from `SinusoidalSolver`)
rebuilt `_basis_coefs(geom, self.k)`: on a wholly-buried deck that is k_p (the
solve ran at k_m inside `_operating_medium`), on a mixed deck it is the
unstitched air view, and on a crossing deck it has no node-wing columns, so
the wing amplitudes were never read. Fixed with a `_readout_view(geom)` hook
in `SinusoidalSolver` (returns the same `_basis_coefs(geom, self.k)`, so the
point-matched family is untouched) overridden in SG, plus SG overrides of the
two point evaluators that honour `k_entry`.

Pre-fix readout cost, against bspline:

| deck | pre-fix | post-fix |
|---|---|---|
| wholly buried (probe6), m = 1/2 | 0.964 / 0.962, flat | 1.1e-4 / 1.7e-5 |
| mixed buried wire (probe2) | 0.964 / 0.963 / 0.962 | 2.6e-4 / 3.0e-5 / 4.0e-6 |
| crossing at ε̃ = 1 vs free (probe3) | 1.2e-2 | 1.1e-4 (the node knot) |

## 4. Wholly-buried port solution

`compute_port_solution` now runs fill, drive, solve, readout and the
`_SegmentBasis` inside `_operating_medium`, as `compute_impedance` does;
`compute_y_matrix` and both swept loops reach the matrix only through it.
`compute_impedance`'s port readout moved inside the medium too (an off-centre
point gap reads at `self.k`). probe5: 1/ΣY reproduces `compute_impedance` to
4e-16; Y12 vs bspline 1.4e-4 / 2.6e-5 / 5.1e-6 at m = 1/2/4.

## Far field of a detached deck (probe4, `_far_moments` over `element_currents`)

probe5's mixed deck, SG vs bspline, ‖M_sg − M_bs‖/‖M_bs‖:

| drive | main | sign alone: readout fixed, sign reverted, drive at air (m = 2) | branch, m = 1/2/4 |
|---|---|---|---|
| both ports | 1.04 | 3.6e-2 | 1.4e-4 / 1.9e-5 / 3.1e-6 |
| above port only | 0.10 | 0.20 | 8.1e-6 / 1.4e-6 / 2.0e-7 |
| buried port only | 0.97 | 2.1e-2 | 1.3e-4 / 1.8e-5 / 2.9e-6 |

The above-port row is the headline: an elevated wire over a buried parasitic
had a 20 % pattern error from the sign alone.

## Open

* **`detached_hub` with its three-radius `HUB_RADII`**: after the fix the
  SG-bspline Y12 gap is FLAT at 1.9e-3 (m = 1/2/4), buried radials flat at
  2-5 %, the short rise at 36-45 %. The same deck at one radius converges
  (Y12 3.7e-4 / 2.0e-4 / 9.3e-5, the rise 8.5e-2 / 4.3e-2 / 2.0e-2). Razor
  (probe7) sits 2.2e-2 from bspline and 2.4e-2 from SG at m = 4, so it does not
  adjudicate. SG's mixed-radius junction is the suspect (its class docstring
  already records the kernel as non-reciprocal under mixed radii); not this
  issue.
* The junction-port deck's `compute_impedance` Z is NaN on main and branch
  alike (probe8 compares with `equal_nan`); unrelated, not looked into.

## Gates

`tests/test_sg_mixed_sign_1159.py`. Each red control reproduces one pre-fix
seam bit for bit: `_transmitted_tensor` negated (= the old `-=`),
`_readout_view` reverted to `SinusoidalSolver`'s, `_entry_k` returning `k`.

Above ground and in free space SG is bit-identical to main: probe8 dumps Z,
alpha, Y, coeffs, knot currents, slopes, sampled currents and element moments
for 14 decks (free / PEC / refl-coef / Sommerfeld × knot point gap, segment
gap, node port, junction port) — 112 arrays, 0 differ, with main's source
imported from a detached worktree (`momwire from` printed by each run).

## Probe outputs and the state they reflect

* `probe1_pre/post`, `probe3_main`, `probe4_*_main`, `probe5_main`,
  `probe6_main`: main vs the sign + readout fix. ε̃ = 1 and wholly-buried rows
  do not depend on item 2 (k_m = k_p, or no stitched view).
* `probe2_mixed_signfix_only`: sign fix only (readout and drive at air).
* `probe2_mixed_readout_fix_only`, `probe4_mixed_readout_fix_only`: readout
  fixed, sign reverted, before item 2.
* `probe2_mixed`, `probe2_hub`, `probe2_detached`, `probe4_*_branch`,
  `probe3_branch`, `probe5_branch`, `probe6_branch`, `probe7`, `probe8_*`: the
  final branch.
