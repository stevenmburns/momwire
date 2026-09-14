"""x13 at 150 radials, with the engine's timeout raised — a SUPPLEMENT, not the ladder.

All four x13 arms at 150 radials failed in the verbatim ladder, and the reason is
ours rather than NEC-5's: `NEC5Engine`'s default timeout is 120 s and x13 needs
more than that at 8,132 segments. It is not a refusal and not a size limit — the
binary was still working when the wrapper gave up, and 3b75639 finished the same
deck in 60.7 s.

So the ladder's 150 row records the timeout as measured, and this script supplies
the number the ratio needs, labelled separately. It builds the deck through the
same path the harness uses (`bench_converge.load_design` -> `NEC5Engine` with the
same ground) and changes exactly one thing: `timeout=900`.
"""

import json
import sys
import time

sys.path.insert(0, "/home/smburns/stevenmburns/antennaknobs/scripts")

import bench_converge as bnc  # noqa: E402

from antennaknobs.engines.nec5 import NEC5Engine  # noqa: E402

cls = bnc.load_design("verticals.buried_radial_vertical")
b = cls()
b.n_radials = int(sys.argv[1]) if len(sys.argv) > 1 else 150
eng = NEC5Engine(b, ground=("finite", 13.0, 0.005), timeout=900.0)
t = time.perf_counter()
z = eng.impedance()[0]
cold = time.perf_counter() - t
t = time.perf_counter()
z2 = eng.impedance()[0]
warm = time.perf_counter() - t
print(
    json.dumps(
        {
            "radials": b.n_radials,
            "cold_s": cold,
            "warm_s": warm,
            "z": [z.real, z.imag],
            "z2": [z2.real, z2.imag],
            "timeout_s": 900.0,
        }
    )
)
