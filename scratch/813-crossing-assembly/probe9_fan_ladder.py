"""#813 step 3, probe 9: is the fan's 3.2e-9 a floor or still quadrature?

`crossing_deck(1)` collapses to 1.9e-11 at growth 2 / panel 8 / q 8;
`fan_rise_deck()` reaches 3.2e-9 at the same setting. Two orders apart, and
the fan is the deck that matters more (horizontal below members separate t_z
from F'). Push the grading further: if it keeps falling it is quadrature and
the bar follows the setting; if it plateaus, that is the fan's own floor and
the bar has to say so.
"""

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import probe8_collapse_fine as P8  # noqa: E402
from test_crossing_serve_524 import crossing_deck, fan_rise_deck  # noqa: E402


def main():
    warnings.filterwarnings("ignore")
    for name, mk in (
        ("crossing_deck(1)", lambda: crossing_deck(1)),
        ("fan_rise_deck()", fan_rise_deck),
    ):
        f = P8.setup_deck(mk, False)
        print(f"\n=== {name} ===")
        print(f"{'growth':>7} {'panel':>6} {'q':>3}  {'rel':>10}  {'nodes':>7}")
        for g, pn, q in (
            (2.0, 8, 8),
            (2.0, 16, 8),
            (2.0, 16, 12),
            (1.5, 16, 12),
            (1.5, 24, 16),
            (1.25, 24, 16),
        ):
            P8.FINE = dict(growth=g, panel_order=pn, q=q)
            M, nodes = P8.assemble(f, True)
            d = np.abs(M - f["Z"]) / np.abs(f["Z"]).max()
            print(f"{g:>7} {pn:>6} {q:>3}  {d.max():>10.3e}  {nodes:>7}")


if __name__ == "__main__":
    main()
