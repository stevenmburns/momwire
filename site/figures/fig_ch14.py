"""Figure for Act IV chapter 14 — "Epilogue: the same math, twice".

Run from the repo root:  .venv/bin/python site/figures/fig_ch14.py
(needs momwire importable — the repo venv.)
"""

import threading
import time

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, GREEN, INK, RED, save

from momwire import BSplineSolver
from momwire._cancel import CancelToken, SolveAborted

WAVELENGTH = 22.0
HD = 0.962 * WAVELENGTH / 4
A_THIN = 0.0005


def _mk(tok=None):
    wire = np.array([[-HD, 0, 0], [HD, 0, 0]])
    return BSplineSolver(wires=[wire], nsegs=500, wavelength=WAVELENGTH,
                         wire_radius=A_THIN, degree=2, feed_wire_index=0,
                         feed_arclength=HD, cancel=tok)


def fig_cancel():
    """A production solver can be interrupted. A long sweep, cancelled mid-flight
    when the knob moves, aborts at the next checkpoint — one solve-point later,
    not at the end."""
    lams = np.linspace(16, 30, 120)
    ks = 2 * np.pi / lams
    s = _mk()
    s.compute_impedance()  # warm
    t = time.perf_counter()
    s.compute_impedance_swept(ks)
    full = time.perf_counter() - t

    tok = CancelToken()
    res = {}

    def run():
        try:
            _mk(tok).compute_impedance_swept(ks)
        except SolveAborted:
            res["abort"] = time.perf_counter() - res["c"]

    th = threading.Thread(target=run)
    th.start()
    time.sleep(0.4)
    res["c"] = time.perf_counter()
    tok.cancel()
    th.join(timeout=10)
    abort = res.get("abort", 0.0)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.barh(["let it finish", "cancel on knob-move"],
                   [full * 1e3, abort * 1e3], color=[RED, GREEN], height=0.55)
    for b, v in zip(bars, [full * 1e3, abort * 1e3]):
        ax.text(v + full * 1e3 * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.0f} ms", va="center", fontsize=10, color=INK)
    ax.set_xlabel("wall time before the solver is free again  (ms)")
    ax.set_title("Cooperative cancellation: a stale solve dies in one checkpoint")
    ax.text(full * 1e3 * 0.5, -0.75,
            f"120-point sweep, N=500 — aborted {abort / full * 100:.0f}% of the way through",
            fontsize=8.5, color=AMBER, ha="center")
    save(fig, "ch14-cancel")


if __name__ == "__main__":
    fig_cancel()
