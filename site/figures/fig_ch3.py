"""Figures for Act I chapter 3 — "The feed and the answer".

Run from the repo root:  .venv/bin/python site/figures/fig_ch3.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

import style
from style import AMBER, CYAN, GREEN, GRID, MUTED, RED, save
from toy_solver import toy_dipole

from momwire import BSplineSolver, SinusoidalSolver

WAVELENGTH = 22.0
L = 2 * 0.962 * WAVELENGTH / 4
A_THIN = 0.0005
A_FAT = 0.05


def fig_convergence():
    """The toy has no convergence — only a window. Fat dipole (a = 5 cm):
    toy impedance vs N against momwire's settled answer. The plausible
    window is dz between ~a and ~2a; on either side the toy collapses."""
    Ns = np.array([21, 31, 41, 61, 81, 107, 131, 161, 201, 301, 401, 601, 801])
    z_toy = np.array([toy_dipole(L, A_FAT, WAVELENGTH, int(n))[0] for n in Ns])

    wire = np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])
    z_ref, _ = BSplineSolver(
        wires=[wire], nsegs=81, wavelength=WAVELENGTH, wire_radius=A_FAT
    ).compute_impedance()
    z_sin, _ = SinusoidalSolver(
        wires=[wire], nsegs=81, wavelength=WAVELENGTH, wire_radius=A_FAT
    ).compute_impedance()

    n_lo = L / (2 * A_FAT)  # dz = 2a
    n_hi = L / A_FAT  # dz = a

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, part, label in [(axes[0], np.real, "R  (Ω)"), (axes[1], np.imag, "X  (Ω)")]:
        ax.axvspan(n_lo, n_hi, color=GREEN, alpha=0.10, lw=0)
        ax.plot(Ns, part(z_toy), "o-", color=CYAN, lw=1.6, ms=4, label="toy pulses")
        ax.axhline(part(z_ref), color=AMBER, lw=1.6, ls="--",
                   label=f"momwire B-spline N=81  ({part(z_ref):.1f})")
        ax.axhline(part(z_sin), color=RED, lw=1.2, ls=":",
                   label=f"momwire sinusoidal N=81  ({part(z_sin):.1f})")
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 200, 400, 800])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_ylabel(label)
    axes[0].set_ylim(-10, 160)
    axes[1].set_ylim(-60, 120)
    axes[0].set_title(f"Fat dipole (a = {A_FAT*100:.0f} cm): the toy's window of plausibility")
    axes[0].text(np.sqrt(n_lo * n_hi), 145, "dz ≈ a…2a", color=GREEN, ha="center", fontsize=10)
    axes[1].set_xlabel("number of segments N")
    axes[0].legend(fontsize=9, loc="upper left")
    save(fig, "ch3-convergence")


def fig_sweep():
    """The payoff plot every ham knows: R and X of the real specimen
    (a = 0.5 mm) across frequency, momwire sinusoidal basis, swept solve."""
    wire = np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])
    solver = SinusoidalSolver(
        wires=[wire], nsegs=81, wavelength=WAVELENGTH, wire_radius=A_THIN
    )
    lams = np.linspace(18.0, 27.0, 46)
    z = np.asarray(solver.compute_impedance_swept(2 * np.pi / lams)).ravel()
    l_over_lam = L / lams

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(l_over_lam, z.real, color=AMBER, lw=2, label="R")
    ax.plot(l_over_lam, z.imag, color=CYAN, lw=2, label="X")
    ax.axhline(0, color=MUTED, lw=0.8)

    # Mark resonance (X = 0) by linear interpolation on the sweep.
    i = np.where(np.diff(np.sign(z.imag)))[0][0]
    x0, x1 = l_over_lam[i], l_over_lam[i + 1]
    y0, y1 = z.imag[i], z.imag[i + 1]
    xr = x0 - y0 * (x1 - x0) / (y1 - y0)
    ax.axvline(xr, color=GREEN, lw=1, ls=":")
    ax.annotate(f"resonance\nL ≈ {xr:.3f} λ", xy=(xr, 0), xytext=(xr - 0.055, 150),
                color=GREEN, fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))

    # Mark the design point (lambda = 22 m).
    j = np.argmin(np.abs(lams - WAVELENGTH))
    ax.plot([l_over_lam[j]], [z.real[j]], "o", color=AMBER, ms=6)
    ax.plot([l_over_lam[j]], [z.imag[j]], "o", color=CYAN, ms=6)
    ax.annotate(f"the specimen at λ = 22 m:\nZ = {z.real[j]:.1f} {z.imag[j]:+.1f}j Ω",
                xy=(l_over_lam[j], z.imag[j]), xytext=(0.487, -230),
                color=MUTED, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=GRID))

    ax.set_xlabel("dipole length in wavelengths  L/λ")
    ax.set_ylabel("impedance  (Ω)")
    ax.set_title("Drive-point impedance through resonance (a = 0.5 mm)")
    ax.legend(loc="upper left")
    save(fig, "ch3-sweep")


if __name__ == "__main__":
    fig_convergence()
    fig_sweep()
