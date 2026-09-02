"""Re-derive RazorSolver's `n_qp_path` default across the hard geometry
classes (momwire#754). Measured 2026-09-02 on an i7-4770K / Ubuntu 22.04.

WHAT #754 CLAIMED, AND WHAT THIS ADDS
-------------------------------------
#754 measured three straight dipoles, found Z bit-identical from q=3-4 up to
48, and proposed dropping the default from 32 to 8. Its own text named the
gap in that evidence: straight dipoles may simply not stress the outer
testing-path integral, and "the honest move is a convergence sweep across
those classes, not a swap based on three decks".

This is that sweep. It reuses `probe_quadrature_defaults`' deck bank (which
this study extended with the three classes #754 asked for and it lacked:
close-spaced elements at both catalog scales, and ground CONTACT over PEC and
over Sommerfeld) and adds the two things that decide the question.

TWO METHOD FIXES, BOTH OF WHICH MOVE THE ANSWER
-----------------------------------------------
1. **A reference ABOVE the ladder.** `probe_quadrature_defaults` scored each
   rung against the ladder's own top, which forces the top rung's error to
   read exactly zero and every rung below it to read too small. Scoring
   against a dedicated q=128 solve changes the shape of the tail: on `bent`
   at N=18 the old reading put q=32 at 1.8e-6 relative, the honest one puts
   it at 2.2e-08. The reference is itself converged -- |Z(128)-Z(192)|/|Z| is
   9.3e-11 (N=60) and 3.5e-12 (N=120), i.e. far below anything scored here.
2. **More than one mesh.** The convergence order in q is not a property of
   the deck alone; it improves as the mesh refines, and the required default
   falls with it. Reporting one mesh would have produced a confident answer
   that the next mesh contradicts.

THE ANSWER, AND WHY IT IS "KEEP 32"
------------------------------------
Applying #754's own rule -- the smallest q with 2x margin over the largest q
at which ANY deck still moves by more than 1e-6 relative -- the binding deck
is `bent` (a 90-degree corner), and the answer depends on the mesh:

    N=30, 60  ->  still moving at q=16  ->  default 32
    N=120     ->  still moving at q=12  ->  default 32 (the ladder's next rung)
    N=240,400 ->  still moving at q=8   ->  default 16

A default is applied blindly, including on the coarse meshes where it is
LEAST converged, so the worst case governs and the derived answer is the
value already in the tree: **32**. #754's "converges at 3-4" reproduces
exactly on the straight decks; it simply does not generalise -- `bent` at
N=60 is still 1.0e-4 relative at q=8, four orders of magnitude worse than
`straight` at the same rung.

WHAT THIS DOES NOT SAY
----------------------
It does not say 32 is optimal. On meshes at or above N=240 -- which includes
every rung #754's own timing table quotes -- q=16 clears the bar with 2x
margin and costs half as much. The finding this study actually supports is
that the default is mesh-dependent, and a mesh-aware default (or a documented
"q=16 is enough above N~240") is the shape of a real fix. Dropping the
constant to 8 is not: no deck in the hard classes is converged there at any
mesh measured.

    python scripts/probe_razor_path_754.py            # the decision table
    python scripts/probe_razor_path_754.py --mesh-trend
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_quadrature_defaults as P

LADDER = (1, 2, 3, 4, 6, 8, 12, 16, 32)
REF_Q = 128
THRESHOLD = 1e-6  # relative, #754's own bar

# One mesh per deck, chosen to sit where each class is affordable: the two
# Sommerfeld decks carry a per-solve cost the free-space ones do not.
MESHES = {"near_ground_somm": (60,), "contact_somm": (60,)}
DEFAULT_MESHES = (60, 120)

DECKS = (
    "straight",
    "split",
    "graded",
    "bent",
    "junction_radius_step",
    "close_spaced",
    "close_spaced_wide",
    "near_ground_refl",
    "near_ground_somm",
    "contact_pec",
    "contact_somm",
    "ek",
)


def _rel(deck, nsegs, q, ref):
    z, _ = P.solve("razor", deck, nsegs, "n_qp_path", q)
    return abs(z - ref) / abs(ref)


def decision_table():
    print(
        f"{'deck':22s}{'N':>5s} " + "".join(f"{q:>10d}" for q in LADDER) + "   q_last"
    )
    worst, worst_cell = 0, None
    for deck in DECKS:
        for nsegs in MESHES.get(deck, DEFAULT_MESHES):
            ref, _ = P.solve("razor", deck, nsegs, "n_qp_path", REF_Q)
            devs = {q: _rel(deck, nsegs, q, ref) for q in LADDER}
            last = max([q for q in LADDER if devs[q] > THRESHOLD], default=0)
            if last > worst:
                worst, worst_cell = last, (deck, nsegs)
            print(
                f"{deck:22s}{nsegs:>5d} "
                + "".join(f"{devs[q]:10.1e}" for q in LADDER)
                + f"   {last:>5d}"
            )
    need = 2 * worst
    rung = next((q for q in LADDER if q >= need), None)
    print(
        f"\nlargest q still moving > {THRESHOLD:.0e} relative on ANY deck: "
        f"q={worst} ({worst_cell[0]} N={worst_cell[1]})"
    )
    print(f"2x margin requires q >= {need}  ->  chosen default: {rung}")
    return rung


def mesh_trend(deck="bent"):
    """The binding deck alone, down a mesh ladder -- the finding that the
    required default falls as the mesh refines."""
    print(f"=== {deck}: required default vs mesh ===")
    print(f"{'N':>5s} " + "".join(f"{q:>10d}" for q in LADDER) + "   q_last  rule")
    for nsegs in (30, 60, 120, 240, 400):
        ref, _ = P.solve("razor", deck, nsegs, "n_qp_path", REF_Q)
        devs = {q: _rel(deck, nsegs, q, ref) for q in LADDER}
        last = max([q for q in LADDER if devs[q] > THRESHOLD], default=0)
        rung = next((q for q in LADDER if q >= 2 * last), ">32")
        print(
            f"{nsegs:>5d} "
            + "".join(f"{devs[q]:10.1e}" for q in LADDER)
            + f"   {last:>5d}  {rung}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mesh-trend", action="store_true")
    p.add_argument("--deck", default="bent")
    args = p.parse_args()
    if args.mesh_trend:
        mesh_trend(args.deck)
    else:
        decision_table()


if __name__ == "__main__":
    sys.exit(main())
