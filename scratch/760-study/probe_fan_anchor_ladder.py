"""momwire#760: re-derive FAN_SOIL_A_N2 under the quadrature as it is NOW.

#760's ladder was measured when buried decks defaulted to n_qp_pair=8 and
there was no pair-order ladder. Both have moved: #906/#907 gave buried decks
`BURIED_N_QP_PAIR = 32` with `BURIED_PAIR_ORDER_LADDER = ((2.0, 8), (16.0, 4))`,
so the SHIPPED default now runs order 32 on the near pairs that matter and
drops to 8 and 4 further out.

Three questions:

  1. where the shipped default lands against the banked anchor, which is the
     thing a user actually gets;
  2. whether the flat-order limit reproduces from here (the anchor was banked
     at q=64 off a numpy-fallback ladder, and the accelerated path has since
     been tiled past its n_qp <= 8 cap by #762 -- so this is also a check that
     the two paths still agree);
  3. what the converged value's own error bar is, from the top rungs rather
     than from an assumption.

The ladder is a QUADRATURE axis at fixed mesh: the deck is
`fan_rise_deck_graded("n2")` unchanged on every row, so anything that moves is
quadrature. That is the axis #674 held fixed at 4 and #760 is about.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from momwire.bspline import (  # noqa: E402
    BURIED_N_QP_PAIR,
    BURIED_PAIR_ORDER_LADDER,
    BSplineSolver,
)
from test_crossing_serve_524 import (  # noqa: E402
    FAN_SOIL_A_N2,
    fan_rise_deck_graded,
)


def z_at(**kw):
    t0 = time.time()
    z, _ = BSplineSolver(**fan_rise_deck_graded("n2"), **kw).compute_impedance()
    return complex(z), time.time() - t0


print(f"banked FAN_SOIL_A_N2 = {FAN_SOIL_A_N2:.6f}")
print(
    f"buried defaults now: n_qp_pair={BURIED_N_QP_PAIR} "
    f"ladder={BURIED_PAIR_ORDER_LADDER}\n"
)

print("--- the SHIPPED default (what a user gets) ---")
z, t = z_at()
print(f"  default            {z:.6f}   |dZ| {abs(z - FAN_SOIL_A_N2):8.5f}  [{t:.1f}s]")

print("\n--- flat order, ladder OFF: re-derive the limit ---")
prev = None
for q in (4, 8, 16, 32, 64, 96, 128):
    z, t = z_at(n_qp_pair=q, pair_order_ladder=())
    step = f"  step {abs(z - prev):9.6f}" if prev is not None else ""
    print(
        f"  q={q:4d} flat      {z:.6f}   "
        f"|dZ| {abs(z - FAN_SOIL_A_N2):8.5f}{step}  [{t:.1f}s]"
    )
    prev = z

print("\n--- WITH the shipped ladder, at raised bases ---")
for q in (32, 64, 96):
    z, t = z_at(n_qp_pair=q)
    print(
        f"  q={q:4d} laddered  {z:.6f}   |dZ| {abs(z - FAN_SOIL_A_N2):8.5f}  [{t:.1f}s]"
    )


# ===========================================================================
# MEASURED 2026-09-06, this box, on the re-pin of FAN_SOIL_A_N2.
#
# --- flat order, ladder off ------------------------------------------------
#   q=  4  142.190245-36.483630j            (#674's fixed order)
#   q=  8  141.397015-40.661184j
#   q= 16  141.058008-42.488288j
#   q= 32  140.952269-43.071729j
#   q= 64  140.934548-43.170569j            (the OLD bank's order)
#   q= 96  140.933786-43.174773j
#   q=128  140.933745-43.175002j   step 0.0002329
#   q=160  140.933742-43.175015j   step 0.0000139
#   q=192  140.933742-43.175016j   step 0.0000009
#   q=256  140.933742-43.175016j   step 0.0000001
#
# LIMIT 140.93374 - 43.17502j, bar ~1e-6. The old bank (140.9358 - 43.1622j)
# sat 0.0129 ohm from it; q=64 sits 0.0045 ohm from it.
#
# --- the ladder is LIVE but numerically irrelevant here --------------------
# Deck ladder wish at q=32 and q=64: ((2.0, 8), (16.0, 4)) -- non-empty, so
# not dropped. Laddered vs flat:
#   q=32  140.95226875157832-43.07172946721259j  (laddered)
#         140.95226875160066-43.071729467083586j (flat)   differ at 1e-11
#   q=64  140.93454761139472-43.17056903420367j  (laddered)
#         140.93454761140896-43.17056903415315j  (flat)   differ at 1e-11
# Every pair that matters on this deck is inside the near tier, so the base
# order alone sets the answer. Recorded because "changed nothing" and "never
# ran" look the same in a table (momwire#920).
#
# --- both fill paths agree, so the limit is not an artefact of either ------
#   q= 64  numpy 140.93454761139645-43.170569034115545j [5.9s]
#          cpp   140.93454761140896-43.17056903415315j  [0.4s]  diff 4.0e-11
#   q=128  numpy 140.93374463659327-43.175001673015664j [16.8s]
#          cpp   140.93374463656818-43.17500167312671j  [1.1s]  diff 1.1e-10
#
# --- what a USER gets ------------------------------------------------------
# The shipped buried default (n_qp_pair=32 + ladder) answers
# 140.952269-43.071729j, which is 0.092 ohm from the anchor -- outside its
# 0.05 gate. That is why the gate names its q explicitly.
