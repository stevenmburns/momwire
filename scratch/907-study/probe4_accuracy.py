"""#907 probe 4: what the free-space ladder does to Z, before touching defaults.

The ladder is passed EXPLICITLY here (`pair_order_ladder=((16,4),)`) so the
comparison is against the shipped default, with nothing edited.
"""

from momwire.bspline import BSplineSolver

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe2_decks import square_loop, vee, yagi  # noqa: E402

LADDER = ((16.0, 4),)


def compare(name, deck):
    z0, _ = BSplineSolver(**deck).compute_impedance()
    z1, _ = BSplineSolver(**deck, pair_order_ladder=LADDER).compute_impedance()
    # order 32 flat as the truth arm
    z32, _ = BSplineSolver(**deck, n_qp_pair=32).compute_impedance()
    d = abs(z1 - z0)
    rel = d / abs(z0)
    print(f"{name}")
    print(f"   base8 no ladder  {z0!r}")
    print(f"   base8 + ladder   {z1!r}")
    print(f"   flat 32 (truth)  {z32!r}")
    print(f"   |ladder-base|    {d:.3e}   rel {rel:.3e}")
    print(f"   |base-32|        {abs(z0 - z32):.3e}   |ladder-32| {abs(z1 - z32):.3e}")


compare("square loop 1 lam, 400 seg", square_loop(100))
compare("square loop 1 lam, 100 seg", square_loop(25))
compare("vee dipole 200 seg", vee(100))
compare("yagi 5x40 (200 seg)", yagi())
