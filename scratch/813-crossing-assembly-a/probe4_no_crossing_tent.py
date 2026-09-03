"""#813 step 3 shape (a), probe 4: there is no crossing TENT on a grounded deck.

Shape (a) — and the design note's whole account of the junction tent, "one
basis in both media", its column the sum of two wing pieces and its row the
path chopped at the node — assumes razor builds a single through-current tent
at the crossing node. On the deck that actually has a ground it does not.

razor's own rule, from its module docstring: "K wire ends meeting at one
point in the plane get K tents, one each: the plane is one more branch there,
so no through-path is distinguished and current may leave into the ground."
The crossing node IS a point in the plane, so that rule fires there.
"""

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402

GROUND = ("ground_z", "ground_eps", "ground_model")


def bases(deck, **kw):
    orig = RazorSolver._refuse_buried_geometry
    RazorSolver._refuse_buried_geometry = lambda self: None
    try:
        rs = RazorSolver(**deck, nec5_quadrature=False, n_qp_path=8, **kw)
        g = rs._build_geometry()
    finally:
        RazorSolver._refuse_buried_geometry = orig
    bo = int(np.asarray(g["basis_offsets"])[-1])
    return (
        g,
        bo,
        [
            (m, [int(g["wing_seg"][m, j]) for j in (0, 1)], g["wing_sigma"][m].tolist())
            for m in range(bo, g["n_basis_total"])
        ],
    )


def main():
    warnings.filterwarnings("ignore")
    deck = crossing_deck(1)
    print("the deck declares junctions:", deck.get("junctions"))
    for label, d in (
        (
            "free space, junction detected",
            {k: v for k, v in deck.items() if k not in ("junctions", *GROUND)},
        ),
        (
            "grounded, junction detected  ",
            {k: v for k, v in deck.items() if k != "junctions"},
        ),
        ("grounded, junction DECLARED  ", dict(deck)),
    ):
        g, bo, extra = bases(d)
        print(
            f"  {label}: n_basis = {g['n_basis_total']:>3}  wire bases {bo}  extra {extra}"
        )
    print()
    print("Reading:")
    print("  free space  -> ONE tent, wings on segments [6, 7], sigma [1, 1]:")
    print("                 a through-current unknown spanning both wires.")
    print("  grounded    -> TWO tents, [6, 6] sigma [0, -1] and [7, 7] sigma [0, +1]:")
    print("                 a grounded contact tent per wire, ghost wing at sigma 0,")
    print("                 and NO through-current unknown. Declaring the junction")
    print(
        "                 changes nothing -- the rule is about the PLANE, not the spec."
    )


if __name__ == "__main__":
    main()
