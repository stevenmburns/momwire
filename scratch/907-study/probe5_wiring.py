"""#907 probe 5: does the free-space fill PASS the ladder to the kernel?

probe 4 found |ladder - base| bit-identically zero on four decks with the
ladder passed EXPLICITLY, which a real quadrature-order change cannot be.
This probe intercepts the kernel itself and reports the `ladder=` keyword as
actually received, per call site.
"""

import traceback

import momwire.bspline as _bs
from momwire.bspline import BSplineSolver

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe2_decks import square_loop, yagi  # noqa: E402

LADDER = ((16.0, 4),)


def wiring(name, deck, **extra):
    s = BSplineSolver(**deck, **extra)
    print(f"\n{name}")
    print(f"   solver.pair_order_ladder = {s.pair_order_ladder}")
    seen = []
    real = _bs._seg_seg_full_moments_offedge

    def spy(*args, **kw):
        # the caller's line, so each site is identified
        st = traceback.extract_stack()[-2]
        seen.append((st.lineno, st.name, kw.get("ladder", "<NOT PASSED>")))
        return real(*args, **kw)

    _bs._seg_seg_full_moments_offedge = spy
    try:
        s.compute_impedance()
    finally:
        _bs._seg_seg_full_moments_offedge = real

    for lineno, fn, lad in dict.fromkeys(seen):
        n = sum(1 for x in seen if x[0] == lineno)
        print(f"   bspline.py:{lineno} {fn}()  x{n}  ladder={lad}")
    if not seen:
        print("   (no off-edge kernel calls)")


wiring(
    "square loop 400 seg, ladder passed EXPLICITLY",
    square_loop(100),
    pair_order_ladder=LADDER,
)
wiring("yagi 5x40, ladder passed EXPLICITLY", yagi(), pair_order_ladder=LADDER)
wiring(
    "square loop 400 seg, swept/chunked (swept_mem_mb=1)",
    square_loop(100),
    pair_order_ladder=LADDER,
    swept_mem_mb=1,
)
