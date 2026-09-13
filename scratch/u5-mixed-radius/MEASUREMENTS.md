# U5 (b) — the two-radius crossing rod against NEC-5

Haswell (i7-4770K). Scratch only; no `src/` change. The candidate rule of
`DERIVATION-MIXED-RADIUS.md` (observer-side, pinned at the row level by
check 2) is applied harness-side. Every prediction below is committed before
the run that tests it, and every miss is reported.

## Setup (registered before the run)

- momwire `src/` at `42112b9` (the AK venv's editable checkout, `1dbd384`, is
  src-identical); antennaknobs `1b3a6a920`; NEC-5 reference
  `~/nec5-timing/nec5cl-x13-static`, sha256 `7ebf343d…`.
- **Deck:** #931's crossing rod (`scratch/931-study/probe_rod_ladder.py`
  `RodBuilder`) at soil A, 7.1 MHz: a graded rise from the hub at z = −L up to
  the node, a 50 mm fed wire from the node up with its source at the centre,
  and a graded radiator to λ/4. Radii per wire through `WireSpec`: the rise
  a_B; the fed wire and radiator a_A. The above member at the node is the fed
  wire.
- **Ladder:** uniform refinement r = 1, 2, 4. Every segment is divided by r
  with the graded panel boundaries held: `per_panel` = 2r on both graded wires,
  the fed wire at 2r segments, the rise's far panel at 25 mm / r and the
  radiator's at its design segment / r. The source region refines with
  everything else. First-order Richardson from r = 2 → 4 on both engines.
- **Configurations (a_A, a_B)**: (0.25, 0.25) mm control; (0.25, 0.125) ratio 2;
  (0.25, 0.0625) ratio 4; (0.125, 0.25) ratio 0.5. The largest radius, 0.25 mm,
  keeps Δ/a ≥ 6.25 at r = 4.
- **Lengths:** L = 0.30, 0.60, 1.20 m (the #956 rod lengths; 0.15 m is left out
  because NEC-5's ladder there was not first order).
- **Ports matched:** momwire's feed parity patched to even, so both engines
  have the same mesh and the same knot source, asserted at every rung. U7 found
  the port split immaterial on this class; matching removes it as a variable.
- **momwire spellings:** observer-side at every rung (check 2's harness
  patches); the naive one-radius spelling `single_B` (the whole fill at the
  rise's radius, which is `_radius_per_wire[0]`) at r = 2 only, as a reading.
- **Resolved settings asserted at every momwire rung:** `extended_kernel` False
  on the solver; momwire's segment count equal to NEC-5's; the feed on a knot;
  the crossing fill entered exactly once. The equal-radius control through the
  patched path must reproduce the unpatched solver bit for bit at each L, or
  the run stops.
- **Read at every rung, beside Z:** momwire's node KCL deficit and its slope
  ratio against 1/ε̃.

## Predictions (registered before the run)

| id | prediction | verdict |
|---|---|---|
| Pb.1 | observer-side momwire keeps the node KCL deficit below 1e-5 at every rung of every configuration and length | pending |
| Pb.2 | **the U5 gate**: at each L, the mixed-radius Richardson residual ΔZ∞ = NEC-5 − momwire lies within the equal-radius control's residual band, \|ΔZ∞(mixed) − ΔZ∞(control)\| ≤ b(L), where b(L) is the control's own last step \|ΔZ(r = 4) − ΔZ(r = 2)\| at that L, floored at 0.5 Ω | pending |
| Pb.3 | **informed by check 2**: the observer-side slope ratio is within 1e-3 of 1/ε̃ at every rung | pending |
| Pb.4 | **blind**: the naive `single_B` residual at r = 2 lies outside that band at ratios 2 and 4, at every L | pending |

Competing outcome, registered with them: if NEC-5's own ladder is not first
order at a radius step (its successive steps not shrinking by about 2×), then
Pb.2 cannot be read against NEC-5 in that configuration. That is reported, and
momwire's self-consistency (Pb.1, Pb.3) stands alone there. momwire's docs
record NEC-2 failing to converge at an in-line radius step; NEC-5's behaviour
at the node's radius step is measured here, not assumed.
