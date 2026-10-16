"""One cold render; prints: wall_s fill_s Z(feed line).
argv: deck out basis"""

import sys
import time

t0 = time.perf_counter()
from momwire import _sommerfeld_below as sb  # noqa: E402
from momwire.eznec import _shell  # noqa: E402

fill = [0.0]
real = sb.iv_surfaces_direct_below


def timed(*a, **k):
    s = time.perf_counter()
    try:
        return real(*a, **k)
    finally:
        fill[0] += time.perf_counter() - s


sb.iv_surfaces_direct_below = timed
_shell.main(sys.argv[1:3], basis=sys.argv[3])
wall = time.perf_counter() - t0
z = "?"
lines = open(sys.argv[2], encoding="latin-1").read().splitlines()
for i, ln in enumerate(lines):
    if "ANTENNA INPUT" in ln:
        z = " ".join(lines[i + 4].split()[7:9])
        break
print(f"{wall:.3f} {fill[0]:.3f} {z}")
