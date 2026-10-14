"""Time the three-kernel razor window chain from Python (raw bound functions),
median of 15 repeats; compare with the standalone driver's best-of."""

import sys
import timeit

sys.argv = [sys.argv[0]] + sys.argv[1:]
from bench_overhead import load, raw, real  # noqa: E402

A = load(sys.argv[1])
sm, t1a, orow, t2a = real(A)
out = t2a[0]


def chain():
    M0, M1 = raw["razor_seg_moments"](*sm)
    T1 = raw["razor_assemble_t1"](M0, M1, *t1a[2:], obs_row=orow)
    raw["razor_t2_rows"](out, T1, *t2a[2:])


n = 1
while timeit.timeit(chain, number=n) < 0.2:
    n *= 2
ts = sorted(t / n for t in timeit.repeat(chain, number=n, repeat=15))
print(f"python chain: median {ts[7] * 1e6:.1f} us, min {ts[0] * 1e6:.1f} us")
