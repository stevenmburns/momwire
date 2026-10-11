"""Dan's tapered-dipole family (momwire#1422) through momwire's NEC-5 seam,
razor-2p with and without the exact ring kernel (momwire#1433).

The decks and the NEC-5 / bs2 columns are antennaknobs
`scratch/dan-tapered-1422` (decks/, tables.md); this adds the razor-2p exact
lanes. Configs:
  razor-2p        production razor-2p on the seam (extended kernel)
  razor-2p-exact  exact_kernel=True injected at construction
  razor-2p-auto   exact_kernel="auto" injected

Usage: python -I run_dan.py CONFIG deck.nec [...] >> results.jsonl
"""

import json
import re
import sys
import time
import warnings

from momwire.eznec import _shell
from momwire.razor import RazorSolver

BUILT = []
FORCE = {
    "razor-2p": {},
    "razor-2p-exact": {"exact_kernel": True},
    "razor-2p-auto": {"exact_kernel": "auto"},
}
ROW = re.compile(r"ANTENNA INPUT PARAMETERS.*?\n\s*1\s+\d+\s+\d+\s+(.*?)\n", re.S)


def main():
    config = sys.argv[1]
    force = FORCE[config]
    orig = RazorSolver.__init__

    def init(self, *a, **k):
        k.update(force)
        orig(self, *a, **k)
        BUILT.append(self)

    RazorSolver.__init__ = init
    for path in sys.argv[2:]:
        BUILT.clear()
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _shell.render(open(path).read(), basis="razor-2p")
        dt = time.perf_counter() - t0
        f = ROW.search(out).group(1).split()
        s = BUILT[-1]
        print(
            json.dumps(
                {
                    "deck": path.rsplit("/", 1)[-1],
                    "config": config,
                    "R": float(f[4]),
                    "X": float(f[5]),
                    "sec": round(dt, 2),
                    "extended_kernel": s.extended_kernel,
                    "exact_kernel": s.exact_kernel,
                    "warnings": sorted({type(x.message).__name__ for x in w}),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
