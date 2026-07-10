"""Figures for Act I chapter 3 — "The feed and the answer".

Run from the repo root:  .venv/bin/python site/figures/fig_ch3.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, GREEN, GRID, MUTED, RED, save
from toy_solver import toy_dipole

from momwire import BSplineSolver, SinusoidalSolver

WAVELENGTH = 22.0
L = 2 * 0.962 * WAVELENGTH / 4
A_THIN = 0.0005


def fig_convergence():
    """The toy converges — but slowly, and unevenly. Thin specimen (a = 0.5 mm):
    R settles by a few hundred segments and Richardson-extrapolates dead-on; X
    crawls on a log-corrected trend X∞ + (b + c·lnN)/N. Fitting that trend to a
    fistful of coarse solves recovers the converged reactance — the payoff, and
    the price: many solves plus a curve fit, all to still land ~1 Ω short of a
    good basis because the delta-gap feed is crude."""
    Ns = np.array([11, 21, 41, 81, 121, 161, 241, 321, 481])
    z_toy = np.array([toy_dipole(L, A_THIN, WAVELENGTH, int(n))[0] for n in Ns])
    R, X = z_toy.real, z_toy.imag

    wire = np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])
    z_ref, _ = BSplineSolver(
        wires=[wire], nsegs=21, wavelength=WAVELENGTH, wire_radius=A_THIN
    ).compute_impedance()
    n_wall = L / (8 * A_THIN)  # dz = 8a: the thin-wire kernel's own 1% limit

    # Fit the trends (from these very runs): R ~ Rinf + b/N ; X ~ Xinf + (b + c lnN)/N
    fit = np.arange(len(Ns))[Ns >= 21]  # drop the coarsest point from the fit
    one = np.ones(fit.size)
    R_inf = np.linalg.lstsq(np.column_stack([one, 1 / Ns[fit]]), R[fit], rcond=None)[0][0]
    cX = np.linalg.lstsq(
        np.column_stack([one, 1 / Ns[fit], np.log(Ns[fit]) / Ns[fit]]), X[fit], rcond=None
    )[0]
    nn = np.geomspace(21, n_wall * 1.6, 200)
    X_curve = cX[0] + cX[1] / nn + cX[2] * np.log(nn) / nn

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    axes[0].plot(Ns, R, "o", color=CYAN, ms=5, label="toy pulses")
    axes[0].axhline(R_inf, color=GREEN, lw=1.3, ls="-", alpha=0.8,
                    label=f"extrapolated R → {R_inf:.1f}")
    axes[0].axhline(z_ref.real, color=AMBER, lw=1.6, ls="--",
                    label=f"momwire (N=21):  {z_ref.real:.1f}")

    axes[1].plot(Ns, X, "o", color=CYAN, ms=5, label="toy pulses")
    axes[1].plot(nn, X_curve, color=CYAN, lw=1.3, ls=":", alpha=0.9,
                 label=r"fitted trend  $X_\infty + (b + c\,\ln N)/N$")
    axes[1].axhline(cX[0], color=GREEN, lw=1.3, ls="-", alpha=0.8,
                    label=f"extrapolated X → {cX[0]:.1f}")
    axes[1].axhline(z_ref.imag, color=AMBER, lw=1.6, ls="--",
                    label=f"momwire (N=21):  {z_ref.imag:.1f}")

    for ax in axes:
        ax.axvline(n_wall, color=RED, lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlim(9, n_wall * 1.6)
        ax.legend(fontsize=8.5, loc="upper right")
    axes[0].set_ylabel("R  (Ω)")
    axes[1].set_ylabel("X  (Ω)")
    axes[0].set_ylim(64, 94)
    axes[1].set_ylim(-30, 60)
    axes[0].set_title(f"Thin specimen (a = {A_THIN * 1e3:.1f} mm): converges, but only by extrapolation")
    axes[1].text(n_wall, 44, "  dz = 8a\n  (kernel's own\n  1% limit)",
                 color=RED, ha="left", va="top", fontsize=8.5)
    axes[1].set_xlabel("number of segments N")
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
    ax.annotate(
        f"resonance\nL ≈ {xr:.3f} λ",
        xy=(xr, 0),
        xytext=(xr - 0.055, 150),
        color=GREEN,
        fontsize=10,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1),
    )

    # Mark the design point (lambda = 22 m).
    j = np.argmin(np.abs(lams - WAVELENGTH))
    ax.plot([l_over_lam[j]], [z.real[j]], "o", color=AMBER, ms=6)
    ax.plot([l_over_lam[j]], [z.imag[j]], "o", color=CYAN, ms=6)
    ax.annotate(
        f"the specimen at λ = 22 m:\nZ = {z.real[j]:.1f} {z.imag[j]:+.1f}j Ω",
        xy=(l_over_lam[j], z.imag[j]),
        xytext=(0.487, -230),
        color=MUTED,
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=GRID),
    )

    ax.set_xlabel("dipole length in wavelengths  L/λ")
    ax.set_ylabel("impedance  (Ω)")
    ax.set_title("Drive-point impedance through resonance (a = 0.5 mm)")
    ax.legend(loc="upper left")
    save(fig, "ch3-sweep")


if __name__ == "__main__":
    fig_convergence()
    fig_sweep()
