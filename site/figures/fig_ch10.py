"""Figures for Act III chapter 10 — "Sommerfeld, or paying full price".

Run from the repo root:  .venv/bin/python site/figures/fig_ch10.py
(needs momwire importable — the repo venv.)
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from style import CYAN, GREEN, INK, MUTED, RED, save

from momwire import BSplineSolver
import momwire._sommerfeld as sf

WAVELENGTH = 22.0
HD = 0.962 * WAVELENGTH / 4
A_THIN = 0.0005
C0 = 299792458.0
EPS0 = 8.8541878188e-12


def _solver(**kw):
    h = 0.15 * WAVELENGTH
    wire = np.array([[-HD, 0, h], [HD, 0, h]])
    return BSplineSolver(wires=[wire], nsegs=21, wavelength=WAVELENGTH,
                         wire_radius=A_THIN, degree=2, feed_wire_index=0,
                         feed_arclength=HD, **kw)


def _time(make, warm_first):
    if warm_first:
        make().compute_impedance()
    best = min(_one(make) for _ in range(3))
    return best


def _one(make):
    t = time.perf_counter()
    make().compute_impedance()
    return time.perf_counter() - t


def fig_cost():
    """The cost ladder (this machine): the exact ground is orders slower on its
    first solve — the Sommerfeld grid fill — but the grid is cached, so every
    solve after (a sweep, a knob drag) is back to milliseconds."""
    somm = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
    t_free = _time(lambda: _solver(), True) * 1e3
    t_pec = _time(lambda: _solver(ground_z=0.0), True) * 1e3
    t_fres = _time(lambda: _solver(ground_z=0.0, ground_eps=(13.0, 0.005)), True) * 1e3
    t_cold = _one(lambda: _solver(**somm)) * 1e3          # first ever = grid fill
    t_warm = _time(lambda: _solver(**somm), True) * 1e3   # cached grid

    labels = ["free\nspace", "PEC\nimage", "Fresnel\nrefl-coef",
              "Sommerfeld\n(1st: grid fill)", "Sommerfeld\n(cached)"]
    vals = [t_free, t_pec, t_fres, t_cold, t_warm]
    colors = [MUTED, CYAN, CYAN, RED, GREEN]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("wall time per solve  (ms, log)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2g}", ha="center",
                fontsize=9, color=INK)
    ax.annotate("fill the grid once,\nreuse it across the sweep",
                xy=(4, t_warm * 2), xytext=(2.4, t_cold * 0.35), color=GREEN, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax.set_title("Exact ground costs a grid fill up front — then it's milliseconds again")
    ax.set_ylim(0.3, t_cold * 4)
    save(fig, "ch10-cost")


def fig_remainder():
    """What the grid stores: the Sommerfeld remainder field (image already
    subtracted) is smooth in (R₁, θ), so a coarse grid + interpolation nails
    it — the trick that makes the exact solve affordable at all."""
    k2 = 2 * np.pi / WAVELENGTH
    omega = k2 * C0
    eps_t = 13.0 - 1j * 0.005 / (omega * EPS0)
    grid = sf.get_grid(eps_t, k2, 3.0 * WAVELENGTH, omega)

    R1 = np.linspace(0.03 * WAVELENGTH, 2.8 * WAVELENGTH, 200)
    th = np.linspace(0.01, np.pi / 2, 200)
    RR, TT = np.meshgrid(R1 / WAVELENGTH, th)
    field = np.abs(grid.eval(RR * WAVELENGTH, TT)["IzV"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.pcolormesh(RR, np.degrees(TT), field, cmap="magma", shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|I_z^V|$  remainder (image removed)", color=INK)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)
    # sketch a coarse interpolation grid over the smooth surface
    for r in np.linspace(0.1, 2.7, 9):
        ax.axvline(r, color="white", lw=0.4, alpha=0.25)
    for a in np.linspace(2, 88, 9):
        ax.axhline(a, color="white", lw=0.4, alpha=0.25)
    ax.set_xlabel("distance from the image point  R₁ / λ")
    ax.set_ylabel("angle θ  (degrees)")
    ax.set_title("The Sommerfeld remainder is smooth — coarse grid, 4×4 interpolation")
    save(fig, "ch10-remainder")


if __name__ == "__main__":
    fig_cost()
    fig_remainder()
