"""Figures for Act I chapter 2 — "Solve for coefficients".

Run from the repo root:  .venv/bin/python site/figures/fig_ch2.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

import style
from style import AMBER, CYAN, GRID, INK, MUTED, save
from toy_solver import toy_dipole

from momwire import BSplineSolver

WAVELENGTH = 22.0
L = 2 * 0.962 * WAVELENGTH / 4
A_THIN = 0.0005  # the real specimen
A_FAT = 0.05  # the fattened pipe the toy can handle (chapter 3 explains)


def fig_pulses():
    """What "expand in pulses" means: boxes on segments, and their weighted
    sum is a staircase pretending to be a smooth current."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    N = 9
    edges = np.linspace(-L / 2, L / 2, N + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    coeff = np.cos(np.pi * mids / L)  # a plausible current, as weights

    ax = axes[0]
    for i, (e0, e1) in enumerate(zip(edges[:-1], edges[1:])):
        color = CYAN if i == N // 2 else MUTED
        ax.fill_between([e0, e1], 0, 1, step="pre", alpha=0.25 if i == N // 2 else 0.12, color=color, lw=0)
        ax.plot([e0, e0, e1, e1], [0, 1, 1, 0], color=color, lw=1.4)
    ax.text(mids[N // 2], 1.08, "one pulse = one unknown $I_n$", ha="center", color=CYAN, fontsize=10)
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("basis $P_n(z)$")
    ax.grid(False)

    ax = axes[1]
    ax.step(np.repeat(edges, 2)[1:-1], np.repeat(coeff, 2), color=CYAN, lw=2, label="Σ IₙPₙ(z) — the staircase guess")
    zf = np.linspace(-L / 2, L / 2, 300)
    ax.plot(zf, np.cos(np.pi * zf / L), color=AMBER, lw=1.6, ls="--", label="the current it is imitating")
    ax.set_xlabel("position along the dipole z  (m)")
    ax.set_ylabel("current  (arb.)")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "ch2-pulses")


def fig_currents():
    """The staircase does work — inside its window. Toy (pulses, N=107) vs
    momwire's smooth B-spline current on the same fat dipole."""
    z_in, I_all, z_mid = toy_dipole(L, A_FAT, WAVELENGTH, 107)

    wire = np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])
    solver = BSplineSolver(wires=[wire], nsegs=81, wavelength=WAVELENGTH, wire_radius=A_FAT)
    z_mom, coeffs = solver.compute_impedance()
    s_dense = np.linspace(0, L, 600)
    I_mom = solver.currents_at_knots(coeffs, s_array=[s_dense])[0]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(z_mid, np.abs(I_all) * 1e3, drawstyle="steps-mid", color=CYAN, lw=1.4,
            label=f"toy pulses, N=107   (Z = {z_in.real:.1f} {z_in.imag:+.1f}j Ω)")
    ax.plot(s_dense - L / 2, np.abs(I_mom) * 1e3, color=AMBER, lw=2,
            label=f"momwire B-spline, N=81   (Z = {z_mom.real:.1f} {z_mom.imag:+.1f}j Ω)")
    ax.set_xlabel("position along the dipole z  (m)")
    ax.set_ylabel("|I(z)|   (mA, 1 V drive)")
    ax.set_title(f"Same fat dipole (a = {A_FAT*100:.0f} cm), two bases")
    ax.legend(fontsize=10, loc="lower right", bbox_to_anchor=(0.99, 0.12))

    # Zoom inset near the feed, where the staircase is visible.
    axins = ax.inset_axes([0.06, 0.08, 0.34, 0.42])
    m = np.abs(z_mid) < 0.8
    axins.plot(z_mid[m], np.abs(I_all[m]) * 1e3, drawstyle="steps-mid", color=CYAN, lw=1.4)
    md = np.abs(s_dense - L / 2) < 0.8
    axins.plot(s_dense[md] - L / 2, np.abs(I_mom[md]) * 1e3, color=AMBER, lw=2)
    axins.set_xlim(-0.8, 0.8)
    axins.tick_params(labelsize=7, colors=MUTED)
    for s in axins.spines.values():
        s.set_color(GRID)
    ax.indicate_inset_zoom(axins, edgecolor=MUTED)
    save(fig, "ch2-currents")


def fig_zmatrix():
    """log10 |Z_mn| of the toy matrix on the real thin specimen: a screaming
    diagonal, smooth rapid decay away from it — the structure Act IV mines."""
    # Rebuild the toy matrix inline (toy_dipole returns only the solution).
    N = 81
    k = 2 * np.pi / WAVELENGTH
    dz = L / N
    z_mid = np.linspace(-L / 2 + dz / 2, L / 2 - dz / 2, N)
    xg, wg = np.polynomial.legendre.leggauss(8)
    z_src = z_mid[None, :, None] + (dz / 2) * xg[None, None, :]
    R = np.sqrt((z_mid[:, None, None] - z_src) ** 2 + A_THIN**2)
    kern = np.exp(-1j * k * R) / (4 * np.pi * R**5) * (
        (1 + 1j * k * R) * (2 * R**2 - 3 * A_THIN**2) + (k * A_THIN * R) ** 2
    )
    Z = (kern * (dz / 2) * wg).sum(axis=2)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(np.log10(np.abs(Z)), cmap="cividis", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r"$\log_{10}|Z_{mn}|$", color=INK)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)
    ax.set_xlabel("source segment n")
    ax.set_ylabel("match segment m")
    ax.set_title("The moment matrix: huge diagonal, smooth far field")
    ax.grid(False)
    save(fig, "ch2-zmatrix")


if __name__ == "__main__":
    fig_pulses()
    fig_currents()
    fig_zmatrix()
