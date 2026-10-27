# momwire#1668: the two open cases of `junction_rule="nec42"`

Rocketlake, 2026-10-26. The reference is licensed NEC-4.2, release build
`nec42-release` (sha256 `5fc0d508…`), on Haswell, run as a black box: deck in,
printout out (`run_nec42.sh`). Every probe is in free space.

## What the two cases were

#1644 left (1) a feed on the segment beside a radius step, whose X drifted
from NEC-4.2 as the mesh refined (−0.2 to −4.1 Ω), and (2) an equal-radius T
fed beside its node (+3.0 to +10.6 Ω). Both are the members' charge **ratio**
alone. `ratio_fit.py` scales one member's junction factor by λ and fits λ and
a floor common to two feed positions. Under the #1644 solve:

- one λ serves every feed position on a deck. On the 1:10 step at 40 segments
  per half, eight feeds on both sides of the step give λ = 1.01448 to 1.01463.
  On the 20-segment T, six feeds give 1.02211 to 1.02242;
- the floor comes out as the uniform-wire floor of that mesh: +0.03 Ω at 20
  to 40 segments, −0.59 Ω at 160 on the step, matching the forced-NEC-4
  uniform twins.

The feed beside the node only shows the ratio error best. The feed-away
ladder carried the same error, inside #1644's 2 % tolerance.

So the question became which junction solve gives NEC-4.2's ratios. The λ
tables in `results/ratio_fits_1644_solve.txt` are the targets. The variants
were run against them, scaled to the #1644 solve:

| reading | steps (equal Δ) | steps (unequal Δ) | T family | bends |
|---|---|---|---|---|
| #1644: n = 2, own Δ, at node | off | off | off | off |
| n = 3, own Δ, at node | to 2e-4 | off by 1.5 % | off | to 5e-5 |
| n = 3, mean Δ, at node | to 2e-4 | to 1e-4 | off | to 5e-5 |
| n = 3, mean Δ, at own centre Δ_l/2 for ≥ 3 members | to 2e-4 | to 1e-4 | to 9e-4 | to 5e-5 |

These were tried and rejected (the T family needs about 2 % and none gives
it):

- n = 4, 5, 6;
- other match points: (Δ/2, 3Δ/2), (Δ/4, Δ), (Δ/2, 2Δ);
- a complex kernel;
- reading only q₁ or only qₙ;
- a finite constant charge (truncated at 0.5 to 3 m);
- a scaled self-term radius;
- a scaled cross-member coupling;
- a common Δ of max, min or "the others' sum";
- each member's own Δ doubled.

The T is not orientation-dependent: reversing or reordering its wires leaves
NEC-4.2's printout unchanged (`e6`).

## Result

`results/before_after_fixtures_1644.txt` covers every #1644 fixture.
`results/before_after_probes.txt` covers every probe here, the holdouts
(`e7`, built after the fit) included. Coarse meshes now sit at the
uniform-wire floor (+0.02 to +0.05 Ω of X). At 80 and 160 segments they sit
at the short-segment floor the uniform decks carry too.

## Reproduce

    python scratch/1668-junction/probe_decks.py OUT     # e1..e7
    scratch/1668-junction/run_nec42.sh OUT/e2           # NEC-4.2, Haswell
    python scratch/1668-junction/ratio_fit.py OUT/e2 Td_20_40_w3s1 Td_20_40_w1s20 perp --1644
