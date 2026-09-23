# momwire#1149 U2b — razor-2p serves bspline's buried deck shapes

Labels: **[M]** measured here (log named), **[R]** read from the code,
**[I]** inferred. Every probe asserts `momwire.__file__` is in this worktree
(`common.py`); accelerators built with `make build` into this worktree's venv.
All numbers were measured on main after U2 (`ececcf2`), not differenced.

## 1. The one convention: every term at its SOURCE wire's radius [R]

Razor's reduced kernel regularises every pair with the source segment's
radius (`_seg_moments_prepare`, `_kernel_radius`). U2b extends that to the
three places on the buried routes that took ONE radius:

| place | before | after |
|---|---|---|
| per-medium sub-geometry (`_medium_geometry`) | no radius table; `_kernel_radius` died on `seg_offsets` for mixed radii | carries `seg_a` (its segments' radii) when the deck has more than one radius; uniform decks unchanged (no key) |
| cross blocks (`_assemble_Z_crossing`) | one `ctx.a_wire` = wire 0's radius | each block at its source wires' radius; the source axis partitioned by radius (whole wires per partition), one call per radius |
| node term (`_crossing_node_charges`) | one `ctx.a_wire` | per crossing tent: the above rows see the above half's charge at the above wing's radius, the below rows the below half's at the below wing's |

A partition is a set of WHOLE wires, and `axis_data`'s end table is per wire,
so a junction between two radii on one side keeps both wires' by-parts end
terms, each at its own radius — what the direct form over the two wires
leaves. At ε̃ = 1 that makes the crossing/detached assembly razor's own
free-space fill of the same deck, entry for entry, which is why the collapse
DISCRIMINATES the rule here (§3).

## 2. The node-term radius: derived, not fitted [R + M]

The risk named in the brief: U2's term used one `a` in R = √(d² + a²).

What the term removes is the families' Sommerfeld remainders' potential of the
half tents' node charges. The remainders are radius-free
(`RemainderBelow.field_windows` / the above `Remainder` read no radius) [R].
`fam` and `c1·V` each diverge as 1/R at the node with the same coefficient,
1 − C₂ = (1 − A_m)/ε̃ = 2/(ε̃ + 1) (U2 NOTES §3), so `fam − c1·V` is finite
there, and the radius only regularises the ONE endpoint that sits AT the
node. So the choice is a regulariser of a radius-free quantity, and the rule
taken is the one every other razor term follows (source: each half's charge at
its own wire's radius). Measured on the two-radius rod at x4 [M, probe2c.log]:

| change to every node radius | rise/4 |dZ| | top/4 |dZ| |
|---|---|---|
| all at the above wing's | 1.08e-3 Ω | 1.10e-3 |
| all at the below wing's | 5.9e-5 | 6.0e-5 |
| source / 10 | 2.8e-4 | 1.31e-3 |
| source / 100 | 3.1e-4 | 1.44e-3 |
| source × 10 | 2.8e-3 | 1.31e-2 |

/10 → /100 moves 3e-5 / 1.3e-4: the a → 0 limit is reached at O(a), and every
reasonable rule sits within ~1.4 mΩ of it. The soil radius-response
differential (§3) cannot tell the node rules apart (they move it ≤ 5e-4 Ω,
probe2d.log controls), which is the same statement measured the other way.
What that differential DOES discriminate is the cross-block rule.

## 3. Gates [M]

### U2b-1, detached mixed radii (probe1*.log)
Decks: U1's `detached()` at radii 0.25/1 mm and inverted; `detached_hub`,
`hub_deck(2)` pulled 0.1 m apart with radii 0.5/1/0.25/2 mm (three buried
radii, so the forward block's source axis is partitioned three ways).
- Route counted: fwd at 0.25 mm / rev at 1 mm; inverted; hub fwd ×3 radii,
  rev at 2 mm.
- ε̃ = 1 collapse per block vs razor's free-space fill, source rule: 7e-14
  (two radii), 2.4e-11 (hub). Observer / wire-0 / min / max: 1.5e-6 (two
  radii) .. 1.8e-5 (hub) on the block each gets wrong.
- Reciprocity: 4.08 / 4.09 / 4.38, 4.04 / 4.08 / 4.62, 3.23 / 3.67 / 3.89.
- Gap to bspline shrinking per doubling to x8: ratios 0.52–0.60 (two radii),
  0.25–0.57 (hub).

### U2b-2, the two-radius node (probe2*.log)
Deck: the U5 rod `crossing_deck(2)`, A = 0.25 mm, fed at 4.5 (a knot at every
rung), every edge refined.
- Route counted: rise/2 fwd a/2, rev a; #1140 deck rev partitioned (1 mm,
  4 mm); node term once.
- ε̃ = 1 collapse, whole matrix: 1.8e-12 / 1.7e-12 (rise/2, top/4, both
  lanes); observer / min / max / wire-0: 4.3e-2 / 7.3e-2 (node rows).
- Reciprocity (ports above 4.5 / buried 1.0): 4.11 / 4.17 / 4.19 and
  4.12 / 4.22 / 4.25 to x8. Observer rule: FLAT at 8.3e-2 / 1.4e-1 (probe2g).
- **Radius response vs bspline**, R(mixed) − R(equal), razor minus bspline,
  feed on a knot (probe2h.log):

  | case | x1 | x2 | x4 | x8 | response |
  |---|---|---|---|---|---|
  | rise/2 | −0.192 | −0.079 | −0.037 | −0.020 | 8.0 Ω |
  | rise/4 | −0.388 | −0.161 | −0.076 | −0.039 | 15.9 Ω |
  | top/2 | +0.103 | +0.056 | +0.033 | +0.021 | 0.62 Ω |
  | top/4 | +0.191 | +0.102 | +0.060 | +0.038 | 1.13 Ω |

  Within 0.1 Ω at x4 and halving per doubling. Observer cross-block rule:
  8–16 Ω off (probe2d). NOTE: with the feed OFF a knot (4.3333, probe2d/2g)
  the same differential is not monotone in mesh (rise/4 −0.30, +0.046,
  −0.073 at x2/x4/x8), because razor's feed snap moves with the mesh; the
  test uses the knot feed.
- Driving point onto bspline: 6.85 → 3.03 → 1.50 → 0.81 (rise/2),
  6.55 → 2.84 → 1.38 → 0.73 (top/4), 7.19 → 3.59 → 2.00 → 1.20 (#1140).

### U2b-3, several nodes (probe3*.log)
- Route: two tents (two_node_deck), four (hub/fan), one node-term call.
- The own-node block of a two-node deck equals the one-rod deck's to 1e-12;
  the two cross-node blocks equal each other (translation symmetry). They are
  NOT small (peak 1.77 vs own 2.38 at 2 m): the term is the remainder's
  potential, whose 1/R parts cancel at every distance.
- ε̃ = 1 collapse 6.8e-13 (2 m and 8 m, both lanes).
- Reciprocity, asymmetric ports (above rod 1 / buried rod 2): 4.05 / 3.95;
  hub/fan 6 m: 3.87 / 3.90. Node term zeroed: flat at 4.3e-2 / 0.34.
- Z onto bspline's U9 route, every entry: 0.48–0.58 per doubling.

## 4. The grazing floor, and the pre-flight [M]

The brief expected parity ("8 m served, 12 m refused"). Measured on
`two_node_deck`, ports as above (probe3f.log, probe3g*.log):

| rung | bspline refuses from | razor refuses from |
|---|---|---|
| x1 | 12 m | 128 m |
| x2 | 6 m | 64 m |
| x4 | 3 m | 48 m |

Not parity, and not a defect: each fill's floor is a property of the points
IT evaluates. bspline's crossing axes are graded on the a-scale into the
plane; razor's remainder pairs are testing-path points × Gauss nodes and stay
deeper. At 12 m razor's reciprocity decays at every rung (slow test).

What the pre-flight work DID find is a razor inconsistency [M, probe3g.log]:
razor's plan asked only segment endpoints + centroids, and a path point on a
node segment sits shallower than its centroid, so at 128 m (x1) and 64 m (x2)
the plan passed and the fill then died inside the grid, in the grid's words.
`_below_plane_grazing_refusal` now also asks the pairs the remainder actually
evaluates (`_below_remainder_th_min`); every probed deck then gives the same
verdict and the same sentence from the pre-flight and the fill
(probe3g2.log). Nothing the old plan served is refused except decks the grid
refused anyway.

`RazorSolver.buried_serve_refusal()`: loading on a crossing deck, and the
grazing plan (full geometry on a wholly-buried deck, the buried sub-geometry
with the declared nodes skipped otherwise). Everything else razor refuses on a
buried deck, it refuses at construction.

## 5. The advisory [M, probe3a.log]

Razor raises `CoarseCrossingNode` at the shared bar with its own sentences
(`warn_coarse_node(worth=, levers=)`; bspline's text is the default and is
unchanged). What an unresolved node costs razor, far mesh refined with node
edges held coarse vs refined with them:

| deck, node h | x2 | x4 | x8 | razor − bspline at x8 |
|---|---|---|---|---|
| crossing_deck, 50 mm | 0.045 | 0.069 | 0.082 Ω | 1.03 Ω |
| hub_deck(4), 75 mm rise | 0.14 | 0.22 | 0.26 Ω | 1.04 Ω |

So bspline's "~4.5 ohm ... raise n_qp_pair" text would have been false on
razor.

## 6. Instrument only: the licensed reference [M, probe4.log]

Black box, verified against our licensed materials, equal mesh, the
two-radius rod, every edge refined: |razor − ref| 0.41 / 0.078 / 0.013 /
0.038 Ω at x1/x2/x4/x8 across equal, rise/2, rise/4, top/4 alike; the radius
response razor − ref within 0.015 Ω at every rung (the reference prints two
decimals).

## 7. Surprises / open

- **Pre-existing U1 quadrature (not U2b):** on a COAXIAL detached deck with a
  coarse above segment close to the plane (`detached_hub`, 0.67 m mast
  segment 20 mm above a buried rise end), the reversed cross block's ε̃ = 1
  collapse is 6e-6 at gap 0.01 m and 1e-8 at 0.05 m, uniform radii included
  (probe1e.log). `axis_data` grades only segments TOUCHING the plane, so a
  long source segment just off it is plain Gauss-12 against a near observer.
  U1's collapse gate used dx = 0.5 (7e-14) and never saw it; its dx = 0
  stand-off gates are reciprocity and a bspline bar only. The U2b gates use
  gap 0.1 (2.4e-11).
- Razor's grazing floor is not bspline's (§4). If parity is wanted it would
  have to be imposed, refusing decks razor serves with decaying reciprocity.
- Consumer: antennaknobs' `_below_reach_refusal` builds the solver through
  `solver_factory` once the class has `buried_serve_refusal`, OUTSIDE its try;
  razor's constructor refusals would then surface at engine construction.
