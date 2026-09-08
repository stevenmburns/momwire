"""NEC-5 on the same four rungs, on THIS box.

#914's NEC-5 column (1.44 / 5.0 / 11.6 / 21.5 s) was measured on the laptop.
Quoting it against a haswell momwire number would be a cross-box ratio, so the
reference is re-measured here.
"""

import argparse
import json
import sys
import time

sys.path.insert(0, "/home/smburns/stevenmburns/antennaknobs/scripts")

import bench_converge as bnc  # noqa: E402

from antennaknobs.engines.nec5 import NEC5Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, required=True)
    a = ap.parse_args()
    cls = bnc.load_design("verticals.buried_radial_vertical")
    b = cls()
    b.n_radials = a.radials
    eng = NEC5Engine(b, ground=("finite", 13.0, 0.005))
    t = time.perf_counter()
    z = eng.impedance()[0]
    cold = time.perf_counter() - t
    t = time.perf_counter()
    z2 = eng.impedance()[0]
    warm = time.perf_counter() - t
    print(
        json.dumps(
            {
                "radials": a.radials,
                "cold_s": cold,
                "warm_s": warm,
                "z": [z.real, z.imag],
                "z2": [z2.real, z2.imag],
            }
        )
    )


main()
