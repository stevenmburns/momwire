"""momwire#1156 probe 4 (every port printed; a detached deck's port 1 is
the BURIED one): #1154's exact in-medium-pair reference on SG.

Lossless soil eps~ = 4, jacket eps_r = 10: the exact quasi-static pair is
real, each wire served BARE at its own medium's equivalent radius with its
own series L. The jacket kwargs (with the charge term) against it, and the
red control (charge term dropped, `buried_jacket_charge` -> None)."""

import sys
import warnings

import numpy as np

sys.path.insert(0, "tests")

from momwire import _wire_loading  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_jacket_in_soil_1154 import LOSSLESS, _pair_decks, dipole  # noqa: E402
from test_razor_crossing_loading_1149 import crossing, hub  # noqa: E402
from test_razor_detached_1149 import detached, detached_hub  # noqa: E402

DECKS = {
    "dipole": lambda: dipole(eps=LOSSLESS),
    "crossing": lambda: crossing(1, ground_eps=LOSSLESS),
    "hub4": lambda: hub(1, ground_eps=LOSSLESS),
    "detached": lambda: detached(1, eps=LOSSLESS),
    "detached_hub": lambda: detached_hub(1, eps=LOSSLESS),
}


def z(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.ravel(SinusoidalGalerkinSolver(**d).compute_impedance()[0])


if __name__ == "__main__":
    real = _wire_loading.buried_jacket_charge
    for name in sys.argv[1:] or DECKS:
        exact, jacket = _pair_decks(DECKS[name]())
        try:
            ze = z(exact)
        except NotImplementedError as e:
            print(f"{name:12s} exact pair not servable on SG: {str(e)[:90]}")
            continue
        zj = z(jacket)
        _wire_loading.buried_jacket_charge = lambda s, w: None
        try:
            zf = z(jacket)
        finally:
            _wire_loading.buried_jacket_charge = real
        for p in range(ze.size):
            print(
                f"{name:12s} port {p} exact {ze[p]:.3f}  "
                f"|fix-exact| {abs(zj[p] - ze[p]):.4f}  "
                f"|dropped-exact| {abs(zf[p] - ze[p]):.3f}",
                flush=True,
            )
