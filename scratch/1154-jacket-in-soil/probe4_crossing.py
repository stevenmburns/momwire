"""momwire#1154 probe 4: the charge term on CROSSING and screen decks.

Razor's `_crossing_jacket_refusal` is switched off here (monkeypatched to
None) to ask whether it can be retired.

(a) exact pair, per wire: lossless soil eps~=4, jacket eps_r=10 on EVERY
    wire. The reference serves each wire bare at its own medium's
    equivalent radius (a'_fs above, a'(eps~) below) with its own series L
    (DistributedRLC). fixed = jacket kwargs; old = charge term dropped.
(b) soil A, PVC jacket on every wire: bspline-vs-razor gap in the jacket
    shift under refinement, fixed and old.
"""

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import _medium_spec, _wire_loading  # noqa: E402
from momwire._wire_loading import MU0, DistributedRLC  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_razor_crossing_loading_1149 import crossing, hub  # noqa: E402
from test_razor_detached_1149 import detached_hub  # noqa: E402

warnings.simplefilter("ignore")
RazorSolver._crossing_jacket_refusal = lambda self: None

DECKS = {"crossing": crossing, "hub4": hub, "detached_hub": detached_hub}


def z(cls, d, old=False):
    d = dict(d)
    if cls is RazorSolver:
        d.pop("junctions", None)
        extra = dict(nec5_quadrature=True)
    else:
        extra = {}
    orig = _wire_loading.buried_jacket_charge
    if old:
        _wire_loading.buried_jacket_charge = lambda s, w: None
    try:
        return complex(np.ravel(cls(**d, **extra).compute_impedance()[0])[0])
    finally:
        _wire_loading.buried_jacket_charge = orig


def media_of(d):
    return BSplineSolver(**d)._wire_media()


def part_a(names, ms):
    A, B, ER, ET = 0.25e-3, 0.75e-3, 10.0, 4.0
    eps = complex(ET, -1e-9)
    print(f"(a) exact pair per wire, eps~={ET} lossless, jacket eps_r={ER}")
    for name in names:
        mk = DECKS[name]
        for m in ms:
            d0 = mk(m, eps=eps) if name == "detached_hub" else mk(m, ground_eps=eps)
            media = media_of(d0)
            nw = len(media)
            rad, dl = [], []
            for med in media:
                ext = ET if med == _medium_spec.BELOW else 1.0
                rad.append(A * (B / A) ** (1 - ext / ER))
                L = MU0 / (2 * np.pi) * (1 - ext / ER) * np.log(B / A)
                dl.append(DistributedRLC("series", l=L))
            dex = dict(d0, wire_radius=rad, distributed_rlc=dl)
            dj = dict(
                d0,
                wire_radius=A,
                insulation_radius=[B] * nw,
                insulation_eps_r=[ER] * nw,
            )
            for cls in (BSplineSolver, RazorSolver):
                ze, zf, zo = z(cls, dex), z(cls, dj), z(cls, dj, old=True)
                print(
                    f"  {name:12s} m={m} {cls.__name__:14s} exact {ze:.4f}  "
                    f"fixed-exact {zf - ze:.4f}  old-exact {zo - ze:.4f}",
                    flush=True,
                )


def part_b(names, ms):
    A, B, ER = 0.25e-3, 0.45e-3, 3.5
    print(f"(b) soil A, PVC jacket b={B * 1e3} mm eps_r={ER} on every wire")
    for name in names:
        mk = DECKS[name]
        res = {}
        for m in ms:
            d0 = dict(mk(m), wire_radius=A)
            nw = len(d0["wires"])
            dj = dict(d0, insulation_radius=[B] * nw, insulation_eps_r=[ER] * nw)
            for cls in (BSplineSolver, RazorSolver):
                zb, zf, zo = z(cls, d0), z(cls, dj), z(cls, dj, old=True)
                res[(cls, m)] = (zb, zf, zo)
                print(
                    f"  {name:12s} m={m} {cls.__name__:14s} bare {zb:.3f} "
                    f"fixed {zf:.3f} old {zo:.3f} (fixed-old {zf - zo:.3f})",
                    flush=True,
                )
        for m in ms:
            b, r = res[(BSplineSolver, m)], res[(RazorSolver, m)]
            gf = abs((b[1] - b[0]) - (r[1] - r[0]))
            go = abs((b[2] - b[0]) - (r[2] - r[0]))
            print(f"   {name} m={m} gap fixed {gf:.4f}  old {go:.4f}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1].split(",")
    ms = [int(x) for x in sys.argv[2].split(",")]
    parts = sys.argv[3] if len(sys.argv) > 3 else "ab"
    if "a" in parts:
        part_a(names, ms)
    if "b" in parts:
        part_b(names, ms)
