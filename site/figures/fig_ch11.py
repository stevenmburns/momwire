"""Figures for Act IV chapter 11 — "N² is the enemy".

Run from the repo root:  .venv/bin/python site/figures/fig_ch11.py
(needs momwire importable — the repo venv.)
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, GREEN, MUTED, RED, save

from momwire import BSplineSolver

WAVELENGTH = 22.0
HD = 0.962 * WAVELENGTH / 4
A_THIN = 0.0005


def _solver(N):
    wire = np.array([[-HD, 0, 0], [HD, 0, 0]])
    return BSplineSolver(wires=[wire], nsegs=N, wavelength=WAVELENGTH,
                         wire_radius=A_THIN, degree=2, feed_wire_index=0,
                         feed_arclength=HD)


def fig_scaling():
    """Dense solve wall-time vs N: fill is O(N²), factor O(N³). Either way the
    curve climbs far faster than the antenna gets bigger."""
    Ns = np.array([64, 128, 256, 512, 1024, 2048])
    ts = []
    for N in Ns:
        s = _solver(N)
        s.compute_impedance()  # warm

        def _one():
            t0 = time.perf_counter()
            s.compute_impedance()
            return time.perf_counter() - t0

        ts.append(min(_one() for _ in range(2)))
    ts = np.array(ts)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.loglog(Ns, ts * 1e3, "o-", color=CYAN, lw=2, ms=6, label="measured dense solve")
    # reference slopes anchored at N=512
    i0 = 3
    ax.loglog(Ns, ts[i0] * 1e3 * (Ns / Ns[i0]) ** 2, "--", color=MUTED, lw=1,
              label=r"$\propto N^2$ (fill)")
    ax.loglog(Ns, ts[i0] * 1e3 * (Ns / Ns[i0]) ** 3, ":", color=RED, lw=1,
              label=r"$\propto N^3$ (factor)")
    ax.set_xlabel("number of unknowns N")
    ax.set_ylabel("wall time per solve  (ms)")
    ax.set_title("The dense wall: cost climbs between N² and N³")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    save(fig, "ch11-scaling")


def fig_sweep():
    """A frequency sweep is many solves — but the frequency-independent work
    (geometry, the ch6 static moments) is done once, so the swept solver costs
    far less than resolving from scratch at every point."""
    s = _solver(41)
    s.compute_impedance()  # warm
    t = time.perf_counter()
    s.compute_impedance()
    single = time.perf_counter() - t

    lams = np.linspace(18, 27, 46)
    ks = 2 * np.pi / lams
    t = time.perf_counter()
    s.compute_impedance_swept(ks)
    swept = time.perf_counter() - t

    naive = single * 46  # 46 cold solves
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(["46 separate\nsolves", "swept solver\n(amortized)"],
                  [naive * 1e3, swept * 1e3], color=[RED, GREEN], width=0.55)
    for b, v, note in zip(bars, [naive * 1e3, swept * 1e3],
                          [f"{single*1e3:.1f} ms × 46", f"{swept/46*1e3:.2f} ms/point"]):
        ax.text(b.get_x() + b.get_width() / 2, v + naive * 1e3 * 0.02, f"{v:.0f} ms\n{note}",
                ha="center", fontsize=9, color=AMBER)
    ax.set_ylabel("wall time for a 46-point sweep  (ms)")
    ax.set_ylim(0, naive * 1e3 * 1.25)
    ax.set_title("Sweep once, not 46 times: the frequency-free work is shared")
    save(fig, "ch11-sweep")


if __name__ == "__main__":
    fig_scaling()
    fig_sweep()
