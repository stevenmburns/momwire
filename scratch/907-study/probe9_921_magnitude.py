"""#921: how far off is the uncancelled subtraction?

Compares Z on the straddling buried deck under the fixed per-fill resolution
against the raw per-window one main ships, by monkeypatching `_fill_ladder`
back to the old behaviour rather than editing the source.
"""

import numpy as np

from momwire.bspline import BSplineSolver

FINE = np.array([[0.0, 0.0, -0.05], [0.5, 0.0, -0.05]])
COARSE = np.array([[0.0, 0.3, -0.05], [0.3, 0.3, -0.05]])
DECK = dict(
    wires=[FINE, COARSE],
    n_per_edge_per_wire=[[60], [1]],
    feeds=[(0, 0.25, 1 + 0j)],
    wavelength=1.0,
    wire_radius=1e-4,
    ground_z=0.0,
    ground_eps=(13.0, 0.005),
    ground_model="sommerfeld",
)

z_fixed, _ = BSplineSolver(**DECK).compute_impedance()

# main's behaviour: hand the kernel the raw wish and let each block re-resolve
_real = BSplineSolver._fill_ladder
BSplineSolver._fill_ladder = lambda self, k, sl, sr, ek: (
    () if ek is not None else self.pair_order_ladder
)
z_raw, _ = BSplineSolver(**DECK).compute_impedance()
BSplineSolver._fill_ladder = _real

# and the untiered truth arm
z_flat, _ = BSplineSolver(**DECK, pair_order_ladder=()).compute_impedance()

print(f"per-fill (this PR)      {z_fixed!r}")
print(f"per-window (main)       {z_raw!r}")
print(f"no ladder at all        {z_flat!r}")
print(f"|per-fill  - no ladder| {abs(z_fixed - z_flat):.3e}")
print(f"|per-window- no ladder| {abs(z_raw - z_flat):.3e}   <- the uncancelled part")
print(f"|per-fill  - per-window| {abs(z_fixed - z_raw):.3e}")
