"""#907 probe 8: after the wiring -- does the ladder reach the kernel, and
what does it do to Z? The "before" arm is `pair_order_ladder=()`, which is
exactly the shipped behaviour.
"""

import traceback

import momwire.bspline as _bs
from momwire.bspline import DEFAULT_PAIR_ORDER_LADDER, BSplineSolver

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe2_decks import square_loop, vee, yagi  # noqa: E402

print(f"DEFAULT_PAIR_ORDER_LADDER = {DEFAULT_PAIR_ORDER_LADDER}\n")


def reaches(deck, **extra):
    s = BSplineSolver(**deck, **extra)
    got = []
    real = _bs._seg_seg_full_moments_offedge

    def spy(*a, **kw):
        st = traceback.extract_stack()[-2]
        got.append((st.lineno, kw.get("ladder", "<NOT PASSED>")))
        return real(*a, **kw)

    _bs._seg_seg_full_moments_offedge = spy
    try:
        z, _ = s.compute_impedance()
    finally:
        _bs._seg_seg_full_moments_offedge = real
    return z, got


def compare(name, deck, **extra):
    z1, got = reaches(deck, **extra)
    z0, _ = BSplineSolver(**deck, **extra, pair_order_ladder=()).compute_impedance()
    z32, _ = BSplineSolver(
        **deck, **extra, n_qp_pair=32, pair_order_ladder=()
    ).compute_impedance()
    sites = sorted({(l, str(x)) for l, x in got})
    print(f"{name}")
    for l, x in sites:
        print(f"   bspline.py:{l}  ladder={x}")
    print(f"   ladder  {z1!r}")
    print(f"   none    {z0!r}")
    print(f"   flat32  {z32!r}")
    print(
        f"   |ladder-none| {abs(z1 - z0):.3e} (rel {abs(z1 - z0) / abs(z0):.3e})   "
        f"|ladder-32| {abs(z1 - z32):.3e}   |none-32| {abs(z0 - z32):.3e}"
    )


compare("square loop 400 seg (dense)", square_loop(100))
compare(
    "square loop 400 seg (chunked, swept_mem_mb=1)", square_loop(100), swept_mem_mb=1
)
compare("vee 200 seg", vee(100))
compare("yagi 5x40", yagi())
print("\n=== extended kernel must NOT raise and must NOT tier ===")
compare(
    "square loop 100 seg, extended_kernel=True", square_loop(25), extended_kernel=True
)
