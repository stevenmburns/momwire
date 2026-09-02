"""Per-entry diagnostic for momwire#824.

The far static-moment table is S_pq[i, j] over an edge's N segments. This
prints, for every entry, the two lanes' values and the pair's `series_ratio`
r, so the disagreement can be LOCATED against the regime switch rather than
guessed at. Summarised per |i-j| band, with the worst entries in full.
"""

import sys
import numpy as np

sys.path.insert(0, "tests")
from test_bspline_static_far_808 import A_WIRE, EDGE, LADDER, _numpy_table  # noqa: E402

from momwire._accel import acc as _acc  # noqa: E402
from momwire._bspline_static_far import FAR_RATIO, series_ratio  # noqa: E402


def ratio_table(N, h, a):
    """r for every (i, j) pair — the same quantity both lanes dispatch on."""
    x = np.linspace(0.0, N * h, N + 1)
    i = np.arange(N)
    a0, b0 = x[i][:, None], x[i + 1][:, None]
    A0, B0 = x[i][None, :], x[i + 1][None, :]
    return np.asarray(series_ratio(a0, b0, A0, B0, a))


def main():
    print(f"FAR_RATIO = {FAR_RATIO}   N_TERMS = 64   a = {A_WIRE}   edge = {EDGE} m")
    for N in LADDER:
        h = EDGE / N
        cxx = _acc.seg_seg_static_moments_bspline_uniform(float(h), float(A_WIRE), N, 2)
        npy = _numpy_table(N, h, A_WIRE)
        d = np.abs(cxx - npy)
        scale = np.abs(npy).max()
        bar = 1e-14 * scale
        r = ratio_table(N, h, A_WIRE)
        far = r <= FAR_RATIO
        print(f"\n===== N={N}  h={h!r} =====")
        print(
            f"  max|npy| {scale:.6e}   bar {bar:.6e}   max|cxx-npy| {d.max():.6e}"
            f"   -> {d.max() / bar:.4f} x bar   [{'PASS' if d.max() <= bar else 'FAIL'}]"
        )
        print(
            f"  far-branch pairs: {far.sum()} / {far.size}"
            f"   r in [{r.min():.15f}, {r.max():.15f}]"
            f"   within 1e-12 of the switch: {int((np.abs(r - FAR_RATIO) < 1e-12).sum())}"
        )

        # The band question: is the disagreement concentrated near the switch?
        dmax_ij = d.max(axis=(0, 1))  # worst over the nine moments
        print(
            f"  {'|i-j|':>6} {'r':>18} {'branch':>7} {'max|cxx-npy|':>14} {'/bar':>9}"
        )
        deltas = sorted({0, 1, 2, 3, 4, 5, N // 8, N // 4, N // 2, N - 2, N - 1})
        for dd in deltas:
            if dd >= N:
                continue
            m = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) == dd
            rr = r[m]
            br = "far" if far[m].all() else ("near" if not far[m].any() else "mixed")
            print(
                f"  {dd:>6} {rr.max():>18.15f} {br:>7} {dmax_ij[m].max():>14.6e}"
                f" {dmax_ij[m].max() / bar:>9.4f}"
            )

        # Worst entries in full, most-severe first, deduplicated by value.
        order = np.argsort(d, axis=None)[::-1]
        print("  worst entries:")
        print(
            f"  {'p':>2} {'q':>2} {'i':>4} {'j':>4} {'|i-j|':>5} {'r':>18} {'branch':>6} "
            f"{'cxx':>25} {'npy':>25} {'|diff|':>11} {'rel':>9}"
        )
        seen, shown = set(), 0
        for f in order:
            p, q, i, j = (int(v) for v in np.unravel_index(f, d.shape))
            if d[p, q, i, j] == 0.0 or shown >= 8:
                break
            key = (p, q, abs(i - j))
            if key in seen:
                continue
            seen.add(key)
            shown += 1
            rel = d[p, q, i, j] / abs(npy[p, q, i, j]) if npy[p, q, i, j] else np.inf
            print(
                f"  {p:>2} {q:>2} {i:>4} {j:>4} {abs(i - j):>5} {r[i, j]:>18.15f} "
                f"{'far' if far[i, j] else 'near':>6} "
                f"{cxx[p, q, i, j]:>25.17e} {npy[p, q, i, j]:>25.17e} "
                f"{d[p, q, i, j]:>11.4e} {rel:>9.2e}"
            )


main()
