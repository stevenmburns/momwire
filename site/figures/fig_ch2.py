"""Figures for Act I chapter 2 — "Solve for coefficients".

Run from the repo root:  .venv/bin/python site/figures/fig_ch2.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, GREEN, INK, MUTED, save
from toy_solver import toy_dipole, C0, EPS0, MU0

from momwire import BSplineSolver

WAVELENGTH = 22.0
L = 2 * 0.962 * WAVELENGTH / 4
A_THIN = 0.0005  # the real specimen — and the toy now handles it


def fig_pulses():
    """What the pulse basis is, on the 2n+1-point grid: current pulses at the
    n midpoints, charge nodes at the n+1 endpoints; the weighted sum is a
    staircase imitating a smooth current."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    N = 9
    edges = np.linspace(-L / 2, L / 2, N + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    coeff = np.cos(np.pi * mids / L)  # a plausible current, as weights

    ax = axes[0]
    for i, (e0, e1) in enumerate(zip(edges[:-1], edges[1:])):
        color = CYAN if i == N // 2 else MUTED
        ax.fill_between([e0, e1], 0, 1, step="pre",
                        alpha=0.22 if i == N // 2 else 0.10, color=color, lw=0)
        ax.plot([e0, e0, e1, e1], [0, 1, 1, 0], color=color, lw=1.3)
    # the 2n+1 points: midpoints (current) and endpoints (charge)
    ax.plot(mids, np.full_like(mids, 0.5), "x", color=CYAN, ms=7, mew=1.6,
            label="n midpoints — current $I_n$")
    ax.plot(edges, np.zeros_like(edges), "o", color=AMBER, ms=6,
            label="n+1 endpoints — charge nodes")
    ax.set_ylim(-0.18, 1.35)
    ax.set_ylabel("basis $P_n(z)$")
    ax.legend(fontsize=8.5, loc="upper right", ncol=1)
    ax.grid(False)

    ax = axes[1]
    ax.step(np.repeat(edges, 2)[1:-1], np.repeat(coeff, 2), color=CYAN, lw=2,
            label="Σ IₙPₙ(z) — the staircase guess")
    zf = np.linspace(-L / 2, L / 2, 300)
    ax.plot(zf, np.cos(np.pi * zf / L), color=AMBER, lw=1.6, ls="--",
            label="the current it is imitating")
    ax.set_xlabel("position along the dipole z  (m)")
    ax.set_ylabel("current  (arb.)")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "ch2-pulses")


def fig_currents():
    """The staircase works — on the real thin wire. Toy (pulses) vs momwire's
    smooth B-spline current on the a = 0.5 mm specimen."""
    N = 161
    z_in, I_all, z_mid = toy_dipole(L, A_THIN, WAVELENGTH, N)

    wire = np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])
    solver = BSplineSolver(wires=[wire], nsegs=81, wavelength=WAVELENGTH,
                           wire_radius=A_THIN)
    z_mom, coeffs = solver.compute_impedance()
    s_dense = np.linspace(0, L, 600)
    I_mom = solver.currents_at_knots(coeffs, s_array=[s_dense])[0]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(z_mid, np.abs(I_all) * 1e3, drawstyle="steps-mid", color=CYAN, lw=1.2,
            label=f"toy pulses, N={N}   (Z = {z_in.real:.1f} {z_in.imag:+.1f}j Ω)")
    ax.plot(s_dense - L / 2, np.abs(I_mom) * 1e3, color=AMBER, lw=2,
            label=f"momwire B-spline, N=81   (Z = {z_mom.real:.1f} {z_mom.imag:+.1f}j Ω)")
    # Mark the feed: the 1 V delta gap at the center, where the current peaks.
    i_feed = N // 2
    ax.axvline(0, color=GREEN, ls=":", lw=1.0)
    ax.plot([0], [np.abs(I_all[i_feed]) * 1e3], "o", color=GREEN, ms=6, zorder=5)
    ax.annotate("feed: 1 V gap\n(current max)", xy=(0, np.abs(I_all[i_feed]) * 1e3),
                xytext=(1.6, np.abs(I_all[i_feed]) * 1e3 * 0.62), color=GREEN, fontsize=9,
                ha="left", arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax.set_xlabel("position along the dipole z  (m)")
    ax.set_ylabel("|I(z)|   (mA, 1 V drive)")
    ax.set_title(f"The real specimen (a = {A_THIN * 1e3:.1f} mm), two bases")
    ax.legend(fontsize=10, loc="lower center")
    save(fig, "ch2-currents")


def fig_zmatrix():
    """log10 |Z_mn| of the toy's mixed-potential matrix on the thin specimen:
    a screaming diagonal, smooth rapid decay away from it — Act IV's quarry."""
    N = 81
    k = 2 * np.pi / WAVELENGTH
    omega = 2 * np.pi * C0 / WAVELENGTH
    dz = L / N
    pts = np.linspace(-L / 2, L / 2, 2 * N + 1)
    mid, lo, hi = pts[1::2], pts[0:-1:2], pts[2::2]

    def psi(A, B):
        R = np.abs(A[:, None] - B[None, :])
        same = R < dz / 1e6
        R[same] = 1.0
        out = np.exp(-1j * k * R) / (4 * np.pi * R)
        out[same] = np.log(dz / A_THIN) / (2 * np.pi * dz) - 1j * k / (4 * np.pi)
        return out

    Z = 1j * omega * MU0 * dz**2 * psi(mid, mid)
    Z += (psi(hi, hi) - psi(lo, hi) - psi(hi, lo) + psi(lo, lo)) / (1j * omega * EPS0)

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
